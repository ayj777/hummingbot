import asyncio
import json
from decimal import Decimal
from test.isolated_asyncio_wrapper_test_case import IsolatedAsyncioWrapperTestCase
from unittest.mock import AsyncMock

from aioresponses import aioresponses
from bidict import bidict

import hummingbot.connector.derivative.mexc_perpetual.mexc_perpetual_constants as CONSTANTS
from hummingbot.connector.derivative.mexc_perpetual.mexc_perpetual_derivative import (
    MarginMode,
    MexcPerpetualDerivative,
)
from hummingbot.connector.trading_rule import TradingRule
from hummingbot.core.data_type.common import OrderType, PositionAction, PositionSide, TradeType
from hummingbot.core.data_type.in_flight_order import OrderState


class MexcPerpetualReviewFixTests(IsolatedAsyncioWrapperTestCase):
    def setUp(self):
        super().setUp()
        self.trading_pair = "AIOT-USDT"
        self.exchange = MexcPerpetualDerivative(
            mexc_perpetual_api_key="key",
            mexc_perpetual_api_secret="secret",
            trading_pairs=[self.trading_pair],
            trading_required=True,
        )
        self.exchange._set_trading_pair_symbol_map(bidict({"AIOT_USDT": self.trading_pair}))
        self.exchange._set_current_timestamp(1_700_000_000)
        self.exchange._trading_rules[self.trading_pair] = TradingRule(
            trading_pair=self.trading_pair,
            min_order_size=Decimal("10"),
            min_price_increment=Decimal("0.00001"),
            min_base_amount_increment=Decimal("10"),
            buy_order_collateral_token="USDT",
            sell_order_collateral_token="USDT",
        )

    def _track(self, amount: Decimal = Decimal("1000")):
        self.exchange._contract_sizes[self.trading_pair] = Decimal("10")
        self.exchange.start_tracking_order(
            order_id="oid-1",
            exchange_order_id="ex-1",
            trading_pair=self.trading_pair,
            trade_type=TradeType.BUY,
            price=Decimal("0.07816"),
            amount=amount,
            order_type=OrderType.LIMIT,
            position_action=PositionAction.OPEN,
        )
        return self.exchange.in_flight_orders["oid-1"]

    async def test_cancel_push_records_deal_volume_before_the_order_is_done(self):
        order = self._track(Decimal("1000"))
        self.exchange._order_tracker.TRADE_FILLS_WAIT_TIMEOUT = 0.05
        self.exchange._process_order_event_message({
            "orderId": "ex-1",
            "externalOid": "oid-1",
            "state": 4,
            "dealVol": "34",
            "dealAvgPrice": "0.07816",
            "price": "0.07816",
            "vol": "100",
        })
        await asyncio.sleep(0.2)
        self.assertEqual(Decimal("340"), order.executed_amount_base)
        self.assertEqual(Decimal("0.07816"), order.average_executed_price)
        self.assertEqual(OrderState.CANCELED, order.current_state)

    async def test_filled_push_synthesizes_the_missing_fill_after_the_wait(self):
        order = self._track(Decimal("410"))
        self.exchange._order_tracker.TRADE_FILLS_WAIT_TIMEOUT = 0.05
        self.exchange._process_order_event_message({
            "orderId": "ex-1",
            "externalOid": "oid-1",
            "state": 3,
            "dealVol": "41",
            "dealAvgPrice": "0.07816",
        })
        await asyncio.sleep(0.3)
        self.assertEqual(Decimal("410"), order.executed_amount_base)
        self.assertEqual(OrderState.FILLED, order.current_state)

    async def test_maintenance_and_timeout_do_not_count_as_order_not_found(self):
        order = self._track()
        for code in (604, 801):
            await self.exchange._handle_update_error_for_active_order(
                order,
                IOError(f"Error: {code} - Futures API is under maintenance (retry)"),
            )
        await self.exchange._handle_update_error_for_active_order(order, asyncio.TimeoutError())
        self.assertEqual(0, self.exchange._order_tracker._order_not_found_records.get(order.client_order_id, 0))
        self.assertNotEqual(OrderState.FAILED, order.current_state)

    async def test_create_timeout_keeps_the_same_client_id_in_flight(self):
        self.exchange._position_mode_confirmed = True
        self.exchange._contract_sizes[self.trading_pair] = Decimal("10")

        async def post(*args, **kwargs):
            raise asyncio.TimeoutError()

        async def get(*args, **kwargs):
            raise asyncio.TimeoutError()

        self.exchange._api_post = post
        self.exchange._api_get = get
        order_id = self.exchange.buy(
            trading_pair=self.trading_pair,
            amount=Decimal("410"),
            order_type=OrderType.LIMIT,
            price=Decimal("0.07816"),
            position_action=PositionAction.OPEN,
        )
        await asyncio.sleep(0.3)
        self.assertIn(order_id, self.exchange.in_flight_orders)
        self.assertNotEqual(OrderState.FAILED, self.exchange.in_flight_orders[order_id].current_state)

    async def test_orders_are_refused_until_position_mode_is_read(self):
        self.exchange._contract_sizes[self.trading_pair] = Decimal("10")
        sent = []

        async def post(*args, **kwargs):
            sent.append(kwargs.get("data"))
            return {"success": True, "code": 0, "data": {"orderId": "1", "ts": 1}}

        self.exchange._api_post = post
        with self.assertRaises(IOError):
            await self.exchange._place_order(
                order_id="client-1",
                trading_pair=self.trading_pair,
                amount=Decimal("100"),
                trade_type=TradeType.BUY,
                order_type=OrderType.LIMIT,
                price=Decimal("0.08"),
                position_action=PositionAction.OPEN,
            )
        self.assertEqual([], sent)

    async def test_missing_contract_size_refuses_the_order(self):
        self.exchange._position_mode_confirmed = True
        sent = []

        async def post(*args, **kwargs):
            sent.append(kwargs.get("data"))
            return {"success": True, "code": 0, "data": {"orderId": "1", "ts": 1}}

        self.exchange._api_post = post
        with self.assertRaises(ValueError):
            await self.exchange._place_order(
                order_id="client-1",
                trading_pair=self.trading_pair,
                amount=Decimal("410"),
                trade_type=TradeType.BUY,
                order_type=OrderType.LIMIT,
                price=Decimal("0.08"),
                position_action=PositionAction.OPEN,
            )
        self.assertEqual([], sent)
        self.assertNotIn(Decimal("1"), self.exchange._contract_sizes.values())

    async def test_cancel_2041_reconciles_the_fill_and_does_not_stay_open(self):
        order = self._track(Decimal("100"))

        async def post(*args, **kwargs):
            return {
                "success": True,
                "code": 0,
                "data": [{"orderId": "ex-1", "errorCode": 2041, "errorMsg": "order state cannot be cancelled"}],
            }

        async def get(path_url, **kwargs):
            if "deal_details" in path_url:
                return {
                    "success": True,
                    "code": 0,
                    "data": [{
                        "id": "9",
                        "price": "2",
                        "vol": "4",
                        "fee": "0.1",
                        "feeCurrency": "USDT",
                        "timestamp": 1_700_000_000_000,
                    }],
                }
            return {
                "success": True,
                "code": 0,
                "data": {
                    "orderId": "ex-1",
                    "externalOid": "oid-1",
                    "state": 3,
                    "dealVol": "10",
                    "dealAvgPrice": "2",
                    "price": "2",
                    "vol": "10",
                },
            }

        self.exchange._api_post = post
        self.exchange._api_get = get
        await self.exchange._place_cancel("oid-1", order)
        await asyncio.sleep(0.2)
        self.assertEqual(Decimal("100"), order.executed_amount_base)
        self.assertEqual(OrderState.FILLED, order.current_state)

    async def test_stop_cancels_an_order_that_is_still_being_created(self):
        self.exchange._position_mode_confirmed = True
        self.exchange._contract_sizes[self.trading_pair] = Decimal("10")
        release = asyncio.Event()
        started = asyncio.Event()
        calls = []

        async def post(path_url, data=None, **kwargs):
            calls.append((path_url, data))
            if path_url == CONSTANTS.PLACE_ORDER_PATH:
                started.set()
                await release.wait()
                return {"success": True, "code": 0, "data": {"orderId": "55", "ts": 1}}
            return {
                "success": True,
                "code": 0,
                "data": [{"orderId": 55, "errorCode": 0, "errorMsg": "success"}],
            }

        async def get(path_url, **kwargs):
            return {
                "success": True,
                "code": 0,
                "data": {
                    "orderId": "55",
                    "externalOid": "ignored",
                    "state": 4,
                    "dealVol": "0",
                    "dealAvgPrice": "0",
                    "price": "0.08",
                    "vol": "10",
                },
            }

        self.exchange._api_post = post
        self.exchange._api_get = get
        order_id = self.exchange.buy(
            trading_pair=self.trading_pair,
            amount=Decimal("100"),
            order_type=OrderType.LIMIT,
            price=Decimal("0.08"),
            position_action=PositionAction.OPEN,
        )
        await asyncio.wait_for(started.wait(), 1)
        self.exchange.cancel(self.trading_pair, order_id)
        release.set()
        for _ in range(50):
            order = self.exchange._order_tracker.fetch_order(order_id)
            if order is not None and order.current_state in (OrderState.CANCELED, OrderState.FAILED):
                break
            await asyncio.sleep(0.05)
        order = self.exchange._order_tracker.fetch_order(order_id)
        self.assertIsNotNone(order)
        self.assertIn(order.current_state, (OrderState.CANCELED, OrderState.FAILED))
        self.assertTrue(any(path == CONSTANTS.CANCEL_BY_EXTERNAL_PATH or path == CONSTANTS.CANCEL_ORDER_PATH for path, _data in calls))

    def test_fee_rebate_is_stored_as_a_negative_cost(self):
        order = self._track(Decimal("100"))
        rebate = self.exchange._parse_trade_update(
            {"id": "1", "price": "1", "vol": "1", "fee": "-0.01", "feeCurrency": "USDT", "timestamp": 1},
            order,
        )
        paid = self.exchange._parse_trade_update(
            {"id": "2", "price": "1", "vol": "1", "fee": "0.01", "feeCurrency": "USDT", "timestamp": 1},
            order,
        )
        self.assertEqual(Decimal("-0.01"), rebate.fee.flat_fees[0].amount)
        self.assertEqual(Decimal("0.01"), paid.fee.flat_fees[0].amount)

    async def test_websocket_balance_keeps_the_equity_gap(self):
        async def get(*args, **kwargs):
            return {
                "success": True,
                "code": 0,
                "data": [{
                    "currency": "USDT",
                    "availableBalance": "10",
                    "frozenBalance": "0",
                    "positionMargin": "0",
                    "equity": "15",
                }],
            }

        self.exchange._api_get = get
        await self.exchange._update_balances()
        self.assertEqual(Decimal("15"), self.exchange.get_balance("USDT"))
        self.exchange._process_wallet_event_message({
            "currency": "USDT",
            "availableBalance": "10",
            "frozenBalance": "0",
            "positionMargin": "0",
        })
        self.assertEqual(Decimal("15"), self.exchange.get_balance("USDT"))
        self.assertEqual(Decimal("10"), self.exchange.available_balances["USDT"])

    async def test_price_and_volume_are_json_numbers_and_use_contract_size_ten(self):
        self.exchange._position_mode_confirmed = True
        self.exchange._contract_sizes[self.trading_pair] = Decimal("10")
        self.exchange._perpetual_trading.set_leverage(self.trading_pair, 2)
        await self.exchange.set_margin_mode(MarginMode.ISOLATED)
        captured = {}

        async def post(path_url, data=None, **kwargs):
            captured["data"] = data
            return {"success": True, "code": 0, "data": {"orderId": "9", "ts": 1}}

        self.exchange._api_post = post
        await self.exchange._place_order(
            order_id="client-1",
            trading_pair=self.trading_pair,
            amount=Decimal("100"),
            trade_type=TradeType.BUY,
            order_type=OrderType.LIMIT,
            price=Decimal("0.085"),
            position_action=PositionAction.OPEN,
        )
        payload = captured["data"]
        self.assertIsInstance(payload["price"], (int, float))
        self.assertIsInstance(payload["vol"], (int, float))
        self.assertNotIsInstance(payload["price"], str)
        self.assertEqual(10, payload["vol"])
        self.assertEqual(Decimal("0.085"), Decimal(str(payload["price"])))
        self.assertEqual(2, payload["leverage"])
        self.assertEqual(CONSTANTS.OPEN_TYPE_ISOLATED, payload["openType"])

    async def test_leverage_change_sends_position_id_when_a_position_exists(self):
        self.exchange._position_ids[(self.trading_pair, PositionSide.LONG)] = 42
        await self.exchange.set_margin_mode(MarginMode.ISOLATED)
        captured = []

        async def post(path_url, data=None, **kwargs):
            captured.append(data)
            return {"success": True, "code": 0}

        self.exchange._api_post = post
        success, _message = await self.exchange._set_trading_pair_leverage(self.trading_pair, 2)
        self.assertTrue(success)
        self.assertEqual(42, captured[0]["positionId"])
        self.assertEqual(2, captured[0]["leverage"])
        self.assertEqual(CONSTANTS.OPEN_TYPE_ISOLATED, captured[0]["openType"])

    async def test_confirm_leverage_and_margin_returns_false_when_the_exchange_rejects_it(self):
        self.exchange._position_mode_confirmed = True

        async def post(*args, **kwargs):
            raise IOError("Error: 2006 - Leverage multiplier must be within the upper limit")

        self.exchange._api_post = post
        confirmed = await self.exchange.confirm_leverage_and_margin(self.trading_pair, 2, MarginMode.ISOLATED)
        self.assertFalse(confirmed)
        self.assertNotEqual(2, self.exchange.get_leverage(self.trading_pair))

    async def test_plan_and_stop_cancels_use_the_documented_array_body(self):
        captured = {}

        async def post(path_url, data=None, **kwargs):
            captured[path_url] = data
            return {"success": True, "code": 0}

        self.exchange._api_post = post
        await self.exchange.cancel_plan_orders([{"symbol": "AIOT_USDT", "order_id": "900"}])
        await self.exchange.cancel_stop_orders(["77"])
        plan_body = captured[CONSTANTS.PLAN_ORDER_CANCEL_PATH]
        stop_body = captured[CONSTANTS.STOP_ORDER_CANCEL_PATH]
        self.assertIsInstance(plan_body, list)
        self.assertEqual({"symbol": "AIOT_USDT", "orderId": 900}, plan_body[0])
        self.assertIsInstance(stop_body, list)
        self.assertEqual({"stopPlanOrderId": 77}, stop_body[0])

    def test_funding_parser_handles_the_result_list_and_the_published_rate_snapshot(self):
        listed = {
            "success": True,
            "data": {
                "resultList": [{
                    "symbol": "AIOT_USDT",
                    "funding": "-1.5",
                    "rate": "0.0001",
                    "settleTime": 1_700_000_000_000,
                }]
            },
        }
        records = self.exchange._funding_records(listed)
        self.assertEqual("-1.5", str(records[0]["funding"]))
        sample = {
            "success": True,
            "code": 0,
            "data": {
                "symbol": "BTC_USDT",
                "fundingRate": -0.000019,
                "maxFundingRate": 0.0018,
                "minFundingRate": -0.0018,
                "collectCycle": 8,
                "nextSettleTime": 1761897600000,
                "timestamp": 1761887166142,
            },
        }
        parsed = self.exchange.parse_funding_response(sample)
        self.assertEqual("rate_snapshot", parsed["kind"])
        self.assertIsNone(parsed["payment"])
        self.assertEqual(Decimal("-0.000019"), parsed["rate"])
        self.assertEqual([], self.exchange._funding_records(sample))

    async def test_close_order_sends_position_id(self):
        self.exchange._position_mode_confirmed = True
        self.exchange._contract_sizes[self.trading_pair] = Decimal("10")
        self.exchange._position_ids[(self.trading_pair, PositionSide.LONG)] = 99
        captured = {}

        async def post(path_url, data=None, **kwargs):
            captured["data"] = data
            return {"success": True, "code": 0, "data": {"orderId": "9", "ts": 1}}

        self.exchange._api_post = post
        await self.exchange._place_order(
            order_id="client-1",
            trading_pair=self.trading_pair,
            amount=Decimal("100"),
            trade_type=TradeType.SELL,
            order_type=OrderType.LIMIT,
            price=Decimal("0.09"),
            position_action=PositionAction.CLOSE,
        )
        self.assertEqual(99, captured["data"]["positionId"])
        self.assertEqual(CONSTANTS.SIDE_CLOSE_LONG, captured["data"]["side"])
