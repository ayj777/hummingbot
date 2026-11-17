import asyncio
import json
import re
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock

from aioresponses import aioresponses
from aioresponses.core import RequestCall

import hummingbot.connector.derivative.mexc_perpetual.mexc_perpetual_constants as CONSTANTS
import hummingbot.connector.derivative.mexc_perpetual.mexc_perpetual_web_utils as web_utils
from hummingbot.connector.derivative.mexc_perpetual.mexc_perpetual_derivative import (
    MarginMode,
    MexcPerpetualDerivative,
)
from hummingbot.connector.test_support.perpetual_derivative_test import AbstractPerpetualDerivativeTests
from hummingbot.connector.trading_rule import TradingRule
from hummingbot.connector.utils import combine_to_hb_trading_pair
from hummingbot.core.data_type.common import OrderType, PositionAction, PositionMode, TradeType
from hummingbot.core.data_type.funding_info import FundingInfo
from hummingbot.core.data_type.in_flight_order import InFlightOrder, OrderState
from hummingbot.core.data_type.trade_fee import AddedToCostTradeFee, TokenAmount, TradeFeeBase


class MexcPerpetualDerivativeTests(AbstractPerpetualDerivativeTests.PerpetualDerivativeTests):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.api_key = "test_api_key"
        cls.api_secret = "test_secret_key"
        cls.base_asset = "BTC"
        cls.quote_asset = "USDT"
        cls.trading_pair = combine_to_hb_trading_pair(cls.base_asset, cls.quote_asset)

    def _url_regex(self, path_url: str):
        url = web_utils.public_rest_url(path_url)
        return re.compile(f"^{re.escape(url)}.*")

    @property
    def all_symbols_url(self):
        return self._url_regex(CONSTANTS.CONTRACT_INFO_PATH)

    @property
    def latest_prices_url(self):
        return self._url_regex(CONSTANTS.TICKER_PATH)

    @property
    def network_status_url(self):
        return self._url_regex(CONSTANTS.PING_PATH)

    @property
    def trading_rules_url(self):
        return self.all_symbols_url

    @property
    def order_creation_url(self):
        return self._url_regex(CONSTANTS.PLACE_ORDER_PATH)

    @property
    def balance_url(self):
        return self._url_regex(CONSTANTS.ASSETS_PATH)

    @property
    def funding_info_url(self):
        path = CONSTANTS.FUNDING_RATE_PATH.format(symbol=self.exchange_trading_pair)
        return self._url_regex(path)

    @property
    def funding_payment_url(self):
        return self._url_regex(CONSTANTS.FUNDING_RECORDS_PATH)

    @property
    def cancel_url(self):
        return self._url_regex(CONSTANTS.CANCEL_ORDER_PATH)

    def _status_url(self, order: InFlightOrder):
        path = CONSTANTS.ORDER_BY_EXTERNAL_PATH.format(
            symbol=self.exchange_trading_pair,
            external_oid=order.client_order_id,
        )
        return self._url_regex(path)

    def _fills_url(self, order: InFlightOrder):
        path = CONSTANTS.ORDER_DEALS_PATH.format(order_id=order.exchange_order_id)
        return self._url_regex(path)

    def _contract(self, **overrides) -> Dict[str, Any]:
        contract = {
            "symbol": self.exchange_trading_pair,
            "baseCoin": self.base_asset,
            "quoteCoin": self.quote_asset,
            "settleCoin": self.quote_asset,
            "futureType": 1,
            "state": 0,
            "apiAllowed": True,
            "contractSize": "1",
            "priceUnit": "0.0001",
            "volUnit": "0.01",
            "minVol": "1",
            "maxVol": "100000",
            "limitMaxVol": "50000",
            "makerFeeRate": "0",
            "takerFeeRate": "0.0002",
            "maxLeverage": 100,
            "marketOrderMaxLevel": 20,
        }
        contract.update(overrides)
        return contract

    def _wrapped(self, data: Any) -> Dict[str, Any]:
        return {"success": True, "code": 0, "data": data}

    @property
    def all_symbols_request_mock_response(self):
        return self._wrapped([self._contract()])

    @property
    def all_symbols_including_invalid_pair_mock_response(self) -> Tuple[str, Any]:
        return "INVALID-PAIR", self._wrapped([
            self._contract(),
            self._contract(symbol="INVALID_PAIR", baseCoin="INVALID", quoteCoin="PAIR", settleCoin="PAIR", state=3),
        ])

    @property
    def latest_prices_request_mock_response(self):
        return self._wrapped({
            "symbol": self.exchange_trading_pair,
            "lastPrice": "100",
        })

    @property
    def network_status_request_successful_mock_response(self):
        return self._wrapped(1761875313209)

    @property
    def trading_rules_request_mock_response(self):
        return self.all_symbols_request_mock_response

    @property
    def trading_rules_request_erroneous_mock_response(self):
        contract = self._contract()
        del contract["priceUnit"]
        return json.loads(json.dumps(self._wrapped([contract])))

    @property
    def order_creation_request_successful_mock_response(self):
        return self._wrapped({"orderId": "1627293504612", "ts": 1761888808839})

    def _balance_entry(self, currency: str, available: str, equity: str) -> Dict[str, Any]:
        return {
            "currency": currency,
            "availableBalance": available,
            "equity": equity,
            "frozenBalance": "0",
            "positionMargin": "0",
        }

    @property
    def balance_request_mock_response_for_base_and_quote(self):
        return self._wrapped([
            self._balance_entry(self.base_asset, "10", "15"),
            self._balance_entry(self.quote_asset, "2000", "2000"),
        ])

    @property
    def balance_request_mock_response_only_base(self):
        return self._wrapped([self._balance_entry(self.base_asset, "10", "15")])

    @property
    def balance_event_websocket_update(self):
        return {
            "channel": CONSTANTS.WS_ASSET_CHANNEL,
            "data": {
                "currency": self.base_asset,
                "availableBalance": "10",
                "frozenBalance": "5",
                "positionMargin": "0",
            },
            "ts": 1640780000000,
        }

    @property
    def expected_latest_price(self):
        return 100.0

    @property
    def expected_supported_position_modes(self) -> List[PositionMode]:
        return [PositionMode.ONEWAY, PositionMode.HEDGE]

    @property
    def expected_supported_order_types(self):
        return [OrderType.LIMIT, OrderType.LIMIT_MAKER, OrderType.MARKET]

    @property
    def expected_trading_rule(self):
        return TradingRule(
            trading_pair=self.trading_pair,
            min_order_size=Decimal("1"),
            max_order_size=Decimal("50000"),
            min_price_increment=Decimal("0.0001"),
            min_base_amount_increment=Decimal("0.01"),
            buy_order_collateral_token=self.quote_asset,
            sell_order_collateral_token=self.quote_asset,
        )

    @property
    def expected_logged_error_for_erroneous_trading_rule(self):
        rule = self.trading_rules_request_erroneous_mock_response["data"][0]
        return f"Error parsing the trading pair rule: {rule}. Skipping."

    @property
    def expected_exchange_order_id(self):
        return "1627293504612"

    @property
    def is_order_fill_http_update_included_in_status_update(self) -> bool:
        return True

    @property
    def is_order_fill_http_update_executed_during_websocket_order_event_processing(self) -> bool:
        return False

    @property
    def expected_partial_fill_price(self) -> Decimal:
        return Decimal("10500")

    @property
    def expected_partial_fill_amount(self) -> Decimal:
        return Decimal("0.5")

    @property
    def expected_fill_fee(self) -> TradeFeeBase:
        return AddedToCostTradeFee(
            percent=Decimal("0"),
            percent_token=self.quote_asset,
            flat_fees=[TokenAmount(token=self.quote_asset, amount=Decimal("0.1"))],
        )

    @property
    def expected_partial_fill_fee(self) -> TradeFeeBase:
        return self.expected_fill_fee

    @property
    def expected_fill_trade_id(self) -> str:
        return "159274416"

    @property
    def funding_info_mock_response(self):
        return self._wrapped({
            "symbol": self.exchange_trading_pair,
            "fundingRate": "3",
            "nextSettleTime": self.target_funding_info_next_funding_utc_timestamp * 1000,
            "idxPrice": "1",
            "fairPrice": "2",
        })

    @property
    def empty_funding_payment_mock_response(self):
        return self._wrapped({"resultList": []})

    @property
    def funding_payment_mock_response(self):
        return self._wrapped({
            "resultList": [{
                "symbol": self.exchange_trading_pair,
                "funding": str(self.target_funding_payment_payment_amount),
                "rate": str(self.target_funding_payment_funding_rate),
                "settleTime": self.target_funding_payment_timestamp * 1000,
            }]
        })

    def exchange_symbol_for_tokens(self, base_token: str, quote_token: str) -> str:
        return f"{base_token}_{quote_token}"

    def create_exchange_instance(self):
        exchange = MexcPerpetualDerivative(
            mexc_perpetual_api_key=self.api_key,
            mexc_perpetual_api_secret=self.api_secret,
            trading_pairs=[self.trading_pair],
        )
        funding_info = FundingInfo(
            trading_pair=self.trading_pair,
            index_price=Decimal("-1"),
            mark_price=Decimal("-1"),
            next_funding_utc_timestamp=1640001119,
            rate=Decimal(str(self.target_funding_payment_funding_rate)),
        )
        exchange._perpetual_trading._funding_info[self.trading_pair] = funding_info
        exchange._contract_sizes[self.trading_pair] = Decimal("10")
        exchange._position_mode_confirmed = True
        return exchange

    def validate_auth_credentials_present(self, request_call: RequestCall):
        headers = request_call.kwargs["headers"]
        self.assertEqual(self.api_key, headers["ApiKey"])
        self.assertIn("Request-Time", headers)
        self.assertIn("Signature", headers)
        self.assertEqual("10", headers["Recv-Window"])

    def validate_order_creation_request(self, order: InFlightOrder, request_call: RequestCall):
        request_data = json.loads(request_call.kwargs["data"])
        position_action = order.position if order.position is not PositionAction.NIL else PositionAction.OPEN
        self.assertEqual(self.exchange_trading_pair, request_data["symbol"])
        self.assertEqual(order.client_order_id, request_data["externalOid"])
        self.assertEqual(order.amount / self.exchange.get_contract_size(self.trading_pair), Decimal(str(request_data["vol"])))
        self.assertIsInstance(request_data["vol"], (int, float))
        self.assertEqual(self.exchange._exchange_side(order.trade_type, position_action), request_data["side"])
        self.assertEqual(CONSTANTS.ORDER_TYPE_MAP[order.order_type], request_data["type"])
        self.assertEqual(CONSTANTS.OPEN_TYPE_ISOLATED, request_data["openType"])
        self.assertEqual(int(self.exchange.get_leverage(self.trading_pair)), request_data["leverage"])
        if order.order_type.is_limit_type():
            self.assertEqual(order.price, Decimal(request_data["price"]))

    def validate_order_cancelation_request(self, order: InFlightOrder, request_call: RequestCall):
        request_data = json.loads(request_call.kwargs["data"])
        self.assertEqual([int(order.exchange_order_id)], request_data["orderIds"])

    def validate_order_status_request(self, order: InFlightOrder, request_call: RequestCall):
        self.assertFalse(request_call.kwargs.get("params"))

    def validate_trades_request(self, order: InFlightOrder, request_call: RequestCall):
        self.assertFalse(request_call.kwargs.get("params"))

    def _contracts_text(self, base_amount: Any) -> str:
        size = self.exchange._contract_sizes.get(self.trading_pair, Decimal("1"))
        value = Decimal(str(base_amount)) / size
        text = format(value, "f")
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text or "0"

    def _mock_response(
        self,
        mock_api,
        url,
        response: Dict[str, Any],
        method: str = "get",
        callback: Optional[Callable] = None,
        repeat: bool = False,
    ):
        getattr(mock_api, method)(url, body=json.dumps(response), callback=callback, repeat=repeat)
        return url

    def configure_successful_cancelation_response(
        self,
        order: InFlightOrder,
        mock_api: aioresponses,
        callback: Optional[Callable] = lambda *args, **kwargs: None,
    ) -> str:
        self._mock_response(
            mock_api,
            self._status_url(order),
            self._order_status_body(order, state=4, deal_vol="0"),
            repeat=True,
        )
        return self._mock_response(
            mock_api,
            self.cancel_url,
            self._wrapped([{"orderId": order.exchange_order_id, "errorCode": 0, "errorMsg": "success"}]),
            method="post",
            callback=callback,
        )

    def configure_erroneous_cancelation_response(
        self,
        order: InFlightOrder,
        mock_api: aioresponses,
        callback: Optional[Callable] = lambda *args, **kwargs: None,
    ) -> str:
        return self._mock_response(
            mock_api,
            self.cancel_url,
            {"success": False, "code": 500, "message": "Internal error"},
            method="post",
            callback=callback,
        )

    def configure_one_successful_one_erroneous_cancel_all_response(
        self,
        successful_order: InFlightOrder,
        erroneous_order: InFlightOrder,
        mock_api: aioresponses,
    ) -> List[str]:
        return [
            self.configure_successful_cancelation_response(order=successful_order, mock_api=mock_api),
            self.configure_erroneous_cancelation_response(order=erroneous_order, mock_api=mock_api),
        ]

    def configure_order_not_found_error_cancelation_response(
        self,
        order: InFlightOrder,
        mock_api: aioresponses,
        callback: Optional[Callable] = lambda *args, **kwargs: None,
    ) -> str:
        return self._mock_response(
            mock_api,
            self.cancel_url,
            self._wrapped([{"orderId": order.exchange_order_id, "errorCode": 2040, "errorMsg": "order not exist"}]),
            method="post",
            callback=callback,
        )

    def _order_status_body(self, order: InFlightOrder, state: int, deal_vol: str = "0") -> Dict[str, Any]:
        return self._wrapped({
            "orderId": order.exchange_order_id,
            "symbol": self.exchange_trading_pair,
            "externalOid": order.client_order_id,
            "price": str(order.price),
            "vol": self._contracts_text(order.amount),
            "dealVol": self._contracts_text(deal_vol),
            "dealAvgPrice": str(order.price),
            "state": state,
            "side": 1,
        })

    def configure_completely_filled_order_status_response(
        self,
        order: InFlightOrder,
        mock_api: aioresponses,
        callback: Optional[Callable] = lambda *args, **kwargs: None,
    ) -> str:
        return self._mock_response(
            mock_api,
            self._status_url(order),
            self._order_status_body(order, state=3, deal_vol=str(order.amount)),
            callback=callback,
        )

    def configure_canceled_order_status_response(
        self,
        order: InFlightOrder,
        mock_api: aioresponses,
        callback: Optional[Callable] = lambda *args, **kwargs: None,
    ) -> str:
        return self._mock_response(
            mock_api,
            self._status_url(order),
            self._order_status_body(order, state=4),
            callback=callback,
        )

    def configure_open_order_status_response(
        self,
        order: InFlightOrder,
        mock_api: aioresponses,
        callback: Optional[Callable] = lambda *args, **kwargs: None,
    ) -> str:
        return self._mock_response(
            mock_api,
            self._status_url(order),
            self._order_status_body(order, state=2),
            callback=callback,
        )

    def configure_http_error_order_status_response(
        self,
        order: InFlightOrder,
        mock_api: aioresponses,
        callback: Optional[Callable] = lambda *args, **kwargs: None,
    ) -> str:
        url = self._status_url(order)
        mock_api.get(url, status=500, callback=callback)
        return url

    def configure_partially_filled_order_status_response(
        self,
        order: InFlightOrder,
        mock_api: aioresponses,
        callback: Optional[Callable] = lambda *args, **kwargs: None,
    ) -> str:
        return self._mock_response(
            mock_api,
            self._status_url(order),
            self._order_status_body(order, state=2, deal_vol=str(self.expected_partial_fill_amount)),
            callback=callback,
        )

    def configure_order_not_found_error_order_status_response(
        self,
        order: InFlightOrder,
        mock_api: aioresponses,
        callback: Optional[Callable] = lambda *args, **kwargs: None,
    ) -> str:
        return self._mock_response(
            mock_api,
            self._status_url(order),
            {"success": False, "code": 2040, "message": "order not exist"},
            callback=callback,
        )

    def _fill_body(self, order: InFlightOrder, price: Decimal, amount: Decimal) -> Dict[str, Any]:
        return self._wrapped([{
            "id": self.expected_fill_trade_id,
            "symbol": self.exchange_trading_pair,
            "orderId": order.exchange_order_id,
            "price": str(price),
            "vol": self._contracts_text(amount),
            "fee": "0.1",
            "feeCurrency": self.quote_asset,
            "timestamp": 1640780000000,
            "taker": True,
        }])

    def configure_partial_fill_trade_response(
        self,
        order: InFlightOrder,
        mock_api: aioresponses,
        callback: Optional[Callable] = lambda *args, **kwargs: None,
    ) -> str:
        return self._mock_response(
            mock_api,
            self._fills_url(order),
            self._fill_body(order, self.expected_partial_fill_price, self.expected_partial_fill_amount),
            callback=callback,
        )

    def configure_full_fill_trade_response(
        self,
        order: InFlightOrder,
        mock_api: aioresponses,
        callback: Optional[Callable] = lambda *args, **kwargs: None,
    ) -> str:
        return self._mock_response(
            mock_api,
            self._fills_url(order),
            self._fill_body(order, order.price, order.amount),
            callback=callback,
        )

    def configure_erroneous_http_fill_trade_response(
        self,
        order: InFlightOrder,
        mock_api: aioresponses,
        callback: Optional[Callable] = lambda *args, **kwargs: None,
    ) -> str:
        url = self._fills_url(order)
        mock_api.get(url, status=500, callback=callback)
        return url

    def _order_ws_event(self, order: InFlightOrder, state: int, deal_vol: str = "0") -> Dict[str, Any]:
        return {
            "channel": CONSTANTS.WS_ORDER_CHANNEL,
            "data": {
                "orderId": order.exchange_order_id,
                "symbol": self.exchange_trading_pair,
                "externalOid": order.client_order_id,
                "price": str(order.price),
                "vol": self._contracts_text(order.amount),
                "dealVol": self._contracts_text(deal_vol),
                "state": state,
                "side": 1,
            },
            "ts": 1640780000000,
        }

    def order_event_for_new_order_websocket_update(self, order: InFlightOrder):
        return self._order_ws_event(order, state=2)

    def order_event_for_canceled_order_websocket_update(self, order: InFlightOrder):
        return self._order_ws_event(order, state=4)

    def order_event_for_full_fill_websocket_update(self, order: InFlightOrder):
        return self._order_ws_event(order, state=3, deal_vol=str(order.amount))

    def trade_event_for_full_fill_websocket_update(self, order: InFlightOrder):
        return {
            "channel": CONSTANTS.WS_ORDER_DEAL_CHANNEL,
            "data": {
                "id": self.expected_fill_trade_id,
                "symbol": self.exchange_trading_pair,
                "orderId": order.exchange_order_id,
                "externalOid": order.client_order_id,
                "price": str(order.price),
                "vol": self._contracts_text(order.amount),
                "fee": "0.1",
                "feeCurrency": self.quote_asset,
                "timestamp": 1640780000000,
                "isTaker": True,
                "side": 3,
            },
            "ts": 1640780000000,
        }

    def position_event_for_full_fill_websocket_update(self, order: InFlightOrder, unrealized_pnl: float):
        return {
            "channel": CONSTANTS.WS_POSITION_CHANNEL,
            "data": {
                "symbol": self.exchange_trading_pair,
                "positionType": 2,
                "holdVol": self._contracts_text(order.amount),
                "holdAvgPrice": str(order.price),
                "leverage": 2,
                "pnl": unrealized_pnl,
                "state": 1,
                "openType": 2,
            },
            "ts": 1640780000000,
        }

    def funding_info_event_for_websocket_update(self):
        return {
            "channel": CONSTANTS.WS_FUNDING_CHANNEL,
            "symbol": self.exchange_trading_pair,
            "data": {
                "symbol": self.exchange_trading_pair,
                "rate": str(self.target_funding_info_rate_ws_updated),
            },
            "ts": self.target_funding_info_next_funding_utc_timestamp_ws_updated * 1000,
        }

    def configure_successful_set_position_mode(
        self,
        position_mode: PositionMode,
        mock_api: aioresponses,
        callback: Optional[Callable] = lambda *args, **kwargs: None,
    ):
        self._mock_response(
            mock_api,
            self._url_regex(CONSTANTS.CHANGE_POSITION_MODE_PATH),
            self._wrapped(None),
            method="post",
            callback=callback,
        )

    def configure_failed_set_position_mode(
        self,
        position_mode: PositionMode,
        mock_api: aioresponses,
        callback: Optional[Callable] = lambda *args, **kwargs: None,
    ) -> Tuple[str, str]:
        message = "Please cancel active orders or close positions before modifying position mode"
        url = self._mock_response(
            mock_api,
            self._url_regex(CONSTANTS.CHANGE_POSITION_MODE_PATH),
            {"success": False, "code": 7001, "message": message},
            method="post",
            callback=callback,
        )
        return url, f"Error: 7001 - {message}"

    def configure_failed_set_leverage(
        self,
        leverage: int,
        mock_api: aioresponses,
        callback: Optional[Callable] = lambda *args, **kwargs: None,
    ) -> Tuple[str, str]:
        message = "Leverage multiplier must be within the upper limit"
        url = self._mock_response(
            mock_api,
            self._url_regex(CONSTANTS.CHANGE_LEVERAGE_PATH),
            {"success": False, "code": 2006, "message": message},
            method="post",
            callback=callback,
        )
        return url, f"Error: 2006 - {message}"

    def configure_successful_set_leverage(
        self,
        leverage: int,
        mock_api: aioresponses,
        callback: Optional[Callable] = lambda *args, **kwargs: None,
    ):
        self._mock_response(
            mock_api,
            self._url_regex(CONSTANTS.CHANGE_LEVERAGE_PATH),
            {"success": True, "code": 0},
            method="post",
            callback=callback,
        )

    def _configure_balance_response(
        self,
        response: Dict[str, Any],
        mock_api: aioresponses,
        callback: Optional[Callable] = lambda *args, **kwargs: None,
    ) -> str:
        mock_api.get(self.balance_url, body=json.dumps(response), callback=callback)
        return self.balance_url

    def test_get_buy_and_sell_collateral_tokens(self):
        self.assertEqual(self.quote_asset, self.exchange.get_buy_collateral_token(self.trading_pair))
        self.assertEqual(
            self.exchange.get_buy_collateral_token(self.trading_pair),
            self.exchange.get_sell_collateral_token(self.trading_pair),
        )

    def test_supported_position_modes(self):
        self.assertEqual(self.expected_supported_position_modes, self.exchange.supported_position_modes())

    @aioresponses()
    def test_create_limit_maker_order_sends_post_only(self, mock_api):
        self._simulate_trading_rules_initialized()
        self.exchange._set_current_timestamp(1640780000)
        request_sent_event = asyncio.Event()
        mock_api.post(
            self.order_creation_url,
            body=json.dumps(self.order_creation_request_successful_mock_response),
            callback=lambda *args, **kwargs: request_sent_event.set(),
        )
        order_id = self.place_buy_order(order_type=OrderType.LIMIT_MAKER)
        self.async_run_with_timeout(request_sent_event.wait())
        request = self._all_executed_requests(mock_api, self.order_creation_url)[0]
        payload = json.loads(request.kwargs["data"])
        self.assertEqual(CONSTANTS.ORDER_TYPE_POST_ONLY, payload["type"])
        self.assertEqual(order_id, payload["externalOid"])

    @aioresponses()
    def test_order_volume_is_converted_from_base_size_using_contract_size(self, mock_api):
        self._simulate_trading_rules_initialized()
        self.exchange._contract_sizes[self.trading_pair] = Decimal("10")
        self.exchange._trading_rules[self.trading_pair] = TradingRule(
            trading_pair=self.trading_pair,
            min_order_size=Decimal("10"),
            min_price_increment=Decimal("0.00001"),
            min_base_amount_increment=Decimal("10"),
        )
        self.exchange._set_current_timestamp(1640780000)
        request_sent_event = asyncio.Event()
        mock_api.post(
            self.order_creation_url,
            body=json.dumps(self.order_creation_request_successful_mock_response),
            callback=lambda *args, **kwargs: request_sent_event.set(),
        )
        self.place_buy_order(amount=Decimal("100"), price=Decimal("1"))
        self.async_run_with_timeout(request_sent_event.wait())
        payload = json.loads(self._all_executed_requests(mock_api, self.order_creation_url)[0].kwargs["data"])
        self.assertEqual(10, payload["vol"])
        self.assertIsInstance(payload["vol"], int)

    @aioresponses()
    def test_create_failure_looks_up_the_client_order_id_before_giving_up(self, mock_api):
        self._simulate_trading_rules_initialized()
        self.exchange._set_current_timestamp(1640780000)
        request_sent_event = asyncio.Event()
        mock_api.post(
            self.order_creation_url,
            body=json.dumps({"success": False, "code": 2042, "message": "Duplicate order ID"}),
            callback=lambda *args, **kwargs: request_sent_event.set(),
        )
        order_id = self.place_buy_order()
        status_path = CONSTANTS.ORDER_BY_EXTERNAL_PATH.format(
            symbol=self.exchange_trading_pair,
            external_oid=order_id,
        )
        mock_api.get(
            self._url_regex(status_path),
            body=json.dumps(self._wrapped({
                "orderId": self.expected_exchange_order_id,
                "symbol": self.exchange_trading_pair,
                "externalOid": order_id,
                "price": "10000",
                "vol": "100",
                "dealVol": "0",
                "state": 2,
            })),
        )
        self.async_run_with_timeout(request_sent_event.wait())
        self.async_run_with_timeout(asyncio.sleep(0.3))
        tracked = self.exchange.in_flight_orders[order_id]
        self.assertEqual(self.expected_exchange_order_id, tracked.exchange_order_id)
        self.assertEqual(OrderState.OPEN, tracked.current_state)

    @aioresponses()
    async def test_update_order_status_when_request_fails_marks_order_as_not_found(self, mock_api):
        self.exchange._set_current_timestamp(1640780000)
        self.exchange.start_tracking_order(
            order_id=self.client_order_id_prefix + "1",
            exchange_order_id=str(self.expected_exchange_order_id),
            trading_pair=self.trading_pair,
            order_type=OrderType.LIMIT,
            trade_type=TradeType.BUY,
            price=Decimal("10000"),
            amount=Decimal("1"),
        )
        order: InFlightOrder = self.exchange.in_flight_orders[self.client_order_id_prefix + "1"]
        url = self.configure_http_error_order_status_response(order=order, mock_api=mock_api)
        await self.exchange._update_order_status()
        if url:
            order_status_request = self._all_executed_requests(mock_api, url)[0]
            self.validate_auth_credentials_present(order_status_request)
            self.validate_order_status_request(order=order, request_call=order_status_request)
        self.assertTrue(order.is_open)
        self.assertFalse(order.is_filled)
        self.assertFalse(order.is_done)
        self.assertEqual(0, self.exchange._order_tracker._order_not_found_records.get(order.client_order_id, 0))

    def test_maintenance_errors_are_reported_as_maintenance(self):
        for code in (604, 801):
            with self.assertRaises(IOError) as raised:
                self.exchange.ensure_successful_response(
                    {"success": False, "code": code, "message": "please retry"}
                )
            self.assertIn("under maintenance", str(raised.exception))
            self.assertIn(str(code), str(raised.exception))

    def test_margin_mode_changes_the_open_type_sent_with_orders(self):
        self.assertEqual(CONSTANTS.OPEN_TYPE_ISOLATED, self.exchange._open_type())
        self.async_run_with_timeout(self.exchange.set_margin_mode(MarginMode.CROSS))
        self.assertEqual(CONSTANTS.OPEN_TYPE_CROSS, self.exchange._open_type())

    @aioresponses()
    def test_user_stream_position_snapshot_converts_contract_size(self, mock_api):
        del mock_api
        self.exchange._contract_sizes[self.trading_pair] = Decimal("10")
        event = {
            "channel": CONSTANTS.WS_POSITION_CHANNEL,
            "data": {
                "symbol": self.exchange_trading_pair,
                "positionType": 1,
                "holdVol": "5",
                "holdAvgPrice": "100",
                "leverage": 3,
                "unRealizedPnl": "1.5",
                "state": 1,
            },
        }
        queue = AsyncMock()
        queue.get.side_effect = [event, asyncio.CancelledError]
        self.exchange._user_stream_tracker._user_stream = queue
        try:
            self.async_run_with_timeout(self.exchange._user_stream_event_listener())
        except asyncio.CancelledError:
            pass
        position = self.exchange.account_positions[self.trading_pair]
        self.assertEqual(Decimal("50"), position.amount)
        self.assertEqual(Decimal("100"), position.entry_price)
        self.assertEqual(Decimal("1.5"), position.unrealized_pnl)
