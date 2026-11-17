import json
import re
from decimal import Decimal
from test.isolated_asyncio_wrapper_test_case import IsolatedAsyncioWrapperTestCase
from unittest.mock import patch

from aioresponses import aioresponses
from bidict import bidict

import hummingbot.connector.derivative.mexc_perpetual.mexc_perpetual_constants as CONSTANTS
import hummingbot.connector.derivative.mexc_perpetual.mexc_perpetual_web_utils as web_utils
from hummingbot.connector.derivative.mexc_perpetual.mexc_perpetual_derivative import MexcPerpetualDerivative
from hummingbot.core.data_type.common import OrderType, PositionAction, PositionMode, TradeType


class MexcPerpetualAdvancedOrderTests(IsolatedAsyncioWrapperTestCase):
    def setUp(self):
        super().setUp()
        self.exchange = MexcPerpetualDerivative(
            mexc_perpetual_api_key="key",
            mexc_perpetual_api_secret="secret",
            trading_pairs=["AIOT-USDT"],
            trading_required=False,
        )
        self.exchange._set_trading_pair_symbol_map(bidict({"AIOT_USDT": "AIOT-USDT"}))
        self.exchange._contract_sizes["AIOT-USDT"] = Decimal("10")
        self.exchange._set_current_timestamp(1_700_000_000)

    def _regex(self, path: str):
        return re.compile(f"^{re.escape(web_utils.public_rest_url(path))}.*")

    def _body(self, mock_api, path: str, response, method: str = "post"):
        getattr(mock_api, method)(self._regex(path), body=json.dumps(response))

    def _payload(self, mock_api, path: str):
        calls = []
        for key, value in mock_api.requests.items():
            if self._regex(path).search(key[1].human_repr()):
                calls.extend(value)
        return json.loads(calls[0].kwargs["data"])

    @aioresponses()
    async def test_limit_order_can_carry_stops_fok_leverage_and_position_mode(self, mock_api):
        self._body(mock_api, CONSTANTS.PLACE_ORDER_PATH, {"success": True, "code": 0, "data": {"orderId": "55", "ts": 1}})
        await self.exchange._place_order(
            order_id="client-1",
            trading_pair="AIOT-USDT",
            amount=Decimal("100"),
            trade_type=TradeType.BUY,
            order_type=OrderType.LIMIT,
            price=Decimal("0.085"),
            position_action=PositionAction.OPEN,
            time_in_force="FOK",
            leverage=7,
            position_mode=PositionMode.HEDGE,
            reduce_only=True,
            stop_loss_price=Decimal("0.07"),
            take_profit_price=Decimal("0.10"),
            loss_trend=2,
            profit_trend=3,
        )
        payload = self._payload(mock_api, CONSTANTS.PLACE_ORDER_PATH)
        self.assertEqual(CONSTANTS.ORDER_TYPE_FOK, payload["type"])
        self.assertEqual(7, payload["leverage"])
        self.assertEqual(CONSTANTS.POSITION_MODE_HEDGE, payload["positionMode"])
        self.assertTrue(payload["reduceOnly"])
        self.assertEqual("10", payload["vol"])
        self.assertEqual("0.07", payload["stopLossPrice"])
        self.assertEqual("0.1", payload["takeProfitPrice"])
        self.assertEqual(2, payload["lossTrend"])
        self.assertEqual(3, payload["profitTrend"])

    @aioresponses()
    async def test_plan_order_place_cancel_list_and_price_change(self, mock_api):
        self._body(mock_api, CONSTANTS.PLAN_ORDER_PLACE_PATH, {"success": True, "code": 0, "data": "900"})
        self._body(mock_api, CONSTANTS.PLAN_ORDER_CANCEL_PATH, {"success": True, "code": 0})
        self._body(mock_api, CONSTANTS.PLAN_ORDER_CANCEL_ALL_PATH, {"success": True, "code": 0})
        self._body(mock_api, CONSTANTS.PLAN_ORDER_CHANGE_PRICE_PATH, {"success": True, "code": 0})
        self._body(
            mock_api,
            CONSTANTS.PLAN_ORDER_LIST_PATH,
            {"success": True, "code": 0, "data": [{"id": "900", "state": 1}]},
            method="get",
        )
        order_id = await self.exchange.place_plan_order(
            trading_pair="AIOT-USDT",
            amount=Decimal("20"),
            trade_type=TradeType.SELL,
            trigger_price=Decimal("0.1"),
            trigger_type=CONSTANTS.TRIGGER_TYPE_GTE,
            price=Decimal("0.11"),
            position_action=PositionAction.CLOSE,
            time_in_force="IOC",
            reduce_only=True,
        )
        self.assertEqual("900", order_id)
        placed = self._payload(mock_api, CONSTANTS.PLAN_ORDER_PLACE_PATH)
        self.assertEqual(CONSTANTS.ORDER_TYPE_IOC, placed["orderType"])
        self.assertEqual(CONSTANTS.SIDE_CLOSE_LONG, placed["side"])
        self.assertEqual("2", placed["vol"])
        self.assertTrue(placed["reduceOnly"])

        await self.exchange.cancel_plan_orders([{"symbol": "AIOT_USDT", "order_id": "900"}])
        await self.exchange.cancel_all_plan_orders("AIOT-USDT")
        await self.exchange.change_plan_order_price(
            "AIOT-USDT", "900", Decimal("0.12"), Decimal("0.13"), order_type=1, trigger_type=1,
        )
        with patch("time.time", return_value=1_700_000_000):
            listed = await self.exchange.list_plan_orders("AIOT-USDT", states="1")
        self.assertEqual("900", listed[0]["id"])
        changed = self._payload(mock_api, CONSTANTS.PLAN_ORDER_CHANGE_PRICE_PATH)
        self.assertEqual(2, changed["from"])
        self.assertEqual("0.12", changed["triggerPrice"])

    @aioresponses()
    async def test_stop_orders_place_list_change_and_cancel(self, mock_api):
        self._body(mock_api, CONSTANTS.STOP_ORDER_PLACE_PATH, {"success": True, "code": 0, "data": "77"})
        self._body(mock_api, CONSTANTS.STOP_ORDER_CHANGE_PRICE_PATH, {"success": True, "code": 0})
        self._body(mock_api, CONSTANTS.STOP_ORDER_CHANGE_PLAN_PRICE_PATH, {"success": True, "code": 0})
        self._body(mock_api, CONSTANTS.STOP_ORDER_CANCEL_PATH, {"success": True, "code": 0})
        self._body(mock_api, CONSTANTS.STOP_ORDER_CANCEL_ALL_PATH, {"success": True, "code": 0})
        self._body(
            mock_api,
            CONSTANTS.STOP_ORDER_OPEN_PATH,
            {"success": True, "code": 0, "data": [{"id": 77, "state": 1}]},
            method="get",
        )
        stop_id = await self.exchange.place_position_stop_order(
            trading_pair="AIOT-USDT",
            position_id="15",
            amount=Decimal("50"),
            stop_loss_price=Decimal("0.05"),
            take_profit_price=Decimal("0.2"),
            loss_trend=2,
            profit_trend=3,
        )
        self.assertEqual("77", stop_id)
        placed = self._payload(mock_api, CONSTANTS.STOP_ORDER_PLACE_PATH)
        self.assertEqual(15, placed["positionId"])
        self.assertEqual("5", placed["vol"])
        self.assertEqual("0.05", placed["stopLossPrice"])
        self.assertEqual(2, placed["lossTrend"])

        await self.exchange.change_stop_order_price("55", stop_loss_price=Decimal("0.04"), loss_trend=1)
        await self.exchange.change_stop_plan_price("77", take_profit_price=Decimal("0.21"), profit_trend=1)
        await self.exchange.cancel_stop_orders(["77"])
        await self.exchange.cancel_all_stop_orders(trading_pair="AIOT-USDT")
        listed = await self.exchange.list_open_stop_orders("AIOT-USDT")
        self.assertEqual(77, listed[0]["id"])
        limit_change = self._payload(mock_api, CONSTANTS.STOP_ORDER_CHANGE_PRICE_PATH)
        self.assertEqual(55, limit_change["orderId"])
        plan_change = self._payload(mock_api, CONSTANTS.STOP_ORDER_CHANGE_PLAN_PRICE_PATH)
        self.assertEqual(77, plan_change["stopPlanOrderId"])
        self.assertEqual("0.21", plan_change["takeProfitPrice"])

    @aioresponses()
    async def test_batch_orders_send_a_list_and_reject_more_than_fifty(self, mock_api):
        self._body(
            mock_api,
            CONSTANTS.BATCH_ORDER_PATH,
            {"success": True, "code": 0, "data": [{"orderId": 1, "errorCode": 0}, {"orderId": 2, "errorCode": 0}]},
        )
        result = await self.exchange.place_batch_orders([
            {
                "trading_pair": "AIOT-USDT",
                "amount": Decimal("10"),
                "price": Decimal("0.08"),
                "trade_type": TradeType.BUY,
                "order_type": OrderType.LIMIT,
                "client_order_id": "b1",
                "leverage": 3,
                "position_mode": 1,
            },
            {
                "trading_pair": "AIOT-USDT",
                "amount": Decimal("20"),
                "price": Decimal("0.09"),
                "trade_type": TradeType.SELL,
                "order_type": OrderType.LIMIT_MAKER,
                "position_action": PositionAction.CLOSE,
                "time_in_force": "FOK",
            },
        ])
        self.assertEqual(2, len(result))
        payload = self._payload(mock_api, CONSTANTS.BATCH_ORDER_PATH)
        self.assertEqual(2, len(payload))
        self.assertEqual("1", payload[0]["vol"])
        self.assertEqual(3, payload[0]["leverage"])
        self.assertEqual(CONSTANTS.STP_MODE_NONE, payload[0]["stpMode"])
        self.assertEqual(CONSTANTS.ORDER_TYPE_FOK, payload[1]["type"])
        with self.assertRaises(ValueError):
            await self.exchange.place_batch_orders([
                {
                    "trading_pair": "AIOT-USDT",
                    "amount": Decimal("10"),
                    "price": Decimal("0.08"),
                    "trade_type": TradeType.BUY,
                    "order_type": OrderType.LIMIT,
                }
                for _ in range(51)
            ])
