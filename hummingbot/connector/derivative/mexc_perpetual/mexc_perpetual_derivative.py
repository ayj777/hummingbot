import asyncio
import time
from decimal import ROUND_DOWN, Decimal
from typing import Any, Dict, List, Optional, Set, Tuple

from bidict import bidict

from hummingbot.connector.derivative.mexc_perpetual import (
    mexc_perpetual_constants as CONSTANTS,
    mexc_perpetual_utils,
    mexc_perpetual_web_utils as web_utils,
)
from hummingbot.connector.derivative.mexc_perpetual.mexc_perpetual_api_order_book_data_source import (
    MexcPerpetualAPIOrderBookDataSource,
)
from hummingbot.connector.derivative.mexc_perpetual.mexc_perpetual_auth import MexcPerpetualAuth
from hummingbot.connector.derivative.mexc_perpetual.mexc_perpetual_user_stream_data_source import (
    MexcPerpetualUserStreamDataSource,
)
from hummingbot.connector.derivative.position import Position
from hummingbot.connector.perpetual_derivative_py_base import PerpetualDerivativePyBase
from hummingbot.connector.trading_rule import TradingRule
from hummingbot.connector.utils import combine_to_hb_trading_pair
from hummingbot.core.api_throttler.data_types import RateLimit
from hummingbot.core.data_type.common import OrderType, PositionAction, PositionMode, PositionSide, TradeType
from hummingbot.connector.client_order_tracker import ClientOrderTracker
from hummingbot.core.data_type.in_flight_order import InFlightOrder, OrderState, OrderUpdate, TradeUpdate
from hummingbot.core.data_type.order_book_tracker_data_source import OrderBookTrackerDataSource
from hummingbot.core.data_type.trade_fee import TokenAmount, TradeFeeBase, TradeFeeSchema
from hummingbot.core.data_type.user_stream_tracker_data_source import UserStreamTrackerDataSource
from hummingbot.core.utils.estimate_fee import build_trade_fee
from hummingbot.core.web_assistant.web_assistants_factory import WebAssistantsFactory

s_decimal_NaN = Decimal("nan")
s_decimal_0 = Decimal("0")


class MarginMode:
    CROSS = "CROSS"
    ISOLATED = "ISOLATED"


class UncertainOrderCreate(Exception):
    """The exchange did not prove that this client id is absent. Keep the order in flight."""


class MexcResponseError(IOError):
    """A MEXC JSON body whose code field is not success. The code is the parsed field, not the message text."""

    def __init__(self, code: Optional[int], message: str):
        super().__init__(f"Error: {code} - {message}")
        self.code = code


class MexcPerpetualOrderTracker(ClientOrderTracker):
    """
    Wait for deal messages on a full fill, then synthesize any quantity the order
    payload already reported. A cancel records that quantity before the order is done.
    """

    async def _process_order_update(self, order_update: OrderUpdate):
        # Safe: this only applies an OrderUpdate that record_order_observation already
        # recorded, or that the base status poller applies after that same call.
        # It does not decide that an order is absent.
        tracked = self.fetch_order(order_update.client_order_id, order_update.exchange_order_id)
        if tracked is not None and order_update.new_state == OrderState.FILLED and not tracked.is_done:
            try:
                await asyncio.wait_for(
                    tracked.wait_until_completely_filled(),
                    timeout=self.TRADE_FILLS_WAIT_TIMEOUT,
                )
            except asyncio.TimeoutError:
                self._connector.synthesize_reported_fill(tracked)
        elif tracked is not None and order_update.new_state == OrderState.CANCELED:
            self._connector.synthesize_reported_fill(tracked)
        await super()._process_order_update(order_update)


class MexcPerpetualDerivative(PerpetualDerivativePyBase):
    web_utils = web_utils

    def __init__(
        self,
        balance_asset_limit: Optional[Dict[str, Dict[str, Decimal]]] = None,
        rate_limits_share_pct: Decimal = Decimal("100"),
        mexc_perpetual_api_key: str = None,
        mexc_perpetual_api_secret: str = None,
        trading_pairs: Optional[List[str]] = None,
        trading_required: bool = True,
    ):
        self.mexc_perpetual_api_key = mexc_perpetual_api_key
        self.mexc_perpetual_api_secret = mexc_perpetual_api_secret
        self._trading_required = trading_required
        self._trading_pairs = trading_pairs
        self._contract_sizes: Dict[str, Decimal] = {}
        self._position_ids: Dict[Tuple[str, PositionSide], int] = {}
        # Margin mode is sent as openType on each order. Isolated is the default.
        self._margin_mode = MarginMode.ISOLATED
        # Stay false until GET position_mode succeeds. Orders are refused before that.
        self._position_mode_confirmed = False
        self._equity_gap: Dict[str, Decimal] = {}
        self._cancel_requests = set()
        self._confirmed_absent = set()
        self._creates_in_flight = set()
        self._creates_finished = set()
        # Highest fill ever reported for a client id, recorded before any state decision.
        self._max_fill_base: Dict[str, Decimal] = {}
        self._max_fill_quote: Dict[str, Decimal] = {}
        self._seen_live_client_ids = set()
        self._submitted_client_ids = set()
        self._acked_client_ids = set()
        self._status_unknown_client_ids = set()
        super().__init__(balance_asset_limit, rate_limits_share_pct)

    @property
    def name(self) -> str:
        return CONSTANTS.EXCHANGE_NAME

    @property
    def authenticator(self) -> MexcPerpetualAuth:
        return MexcPerpetualAuth(
            api_key=self.mexc_perpetual_api_key,
            secret_key=self.mexc_perpetual_api_secret,
            time_provider=self._time_synchronizer,
        )

    @property
    def rate_limits_rules(self) -> List[RateLimit]:
        return CONSTANTS.RATE_LIMITS

    @property
    def domain(self) -> str:
        return "mexc.com"

    @property
    def client_order_id_max_length(self) -> int:
        return CONSTANTS.ORDER_ID_MAX_LEN

    @property
    def client_order_id_prefix(self) -> str:
        return CONSTANTS.HBOT_ORDER_ID_PREFIX

    @property
    def trading_rules_request_path(self) -> str:
        return CONSTANTS.CONTRACT_INFO_PATH

    @property
    def trading_pairs_request_path(self) -> str:
        return CONSTANTS.CONTRACT_INFO_PATH

    @property
    def check_network_request_path(self) -> str:
        return CONSTANTS.PING_PATH

    @property
    def trading_pairs(self) -> Optional[List[str]]:
        return self._trading_pairs

    @property
    def is_cancel_request_in_exchange_synchronous(self) -> bool:
        return True

    @property
    def is_trading_required(self) -> bool:
        return self._trading_required

    @property
    def funding_fee_poll_interval(self) -> int:
        return 120

    def supported_order_types(self) -> List[OrderType]:
        return [OrderType.LIMIT, OrderType.LIMIT_MAKER, OrderType.MARKET]

    def supported_position_modes(self) -> List[PositionMode]:
        return [PositionMode.ONEWAY, PositionMode.HEDGE]

    def get_contract_size(self, trading_pair: str) -> Decimal:
        try:
            size = self._contract_sizes[trading_pair]
        except KeyError as exc:
            raise ValueError(f"contractSize for {trading_pair} is not loaded") from exc
        if size <= s_decimal_0:
            raise ValueError(f"contractSize for {trading_pair} is not loaded")
        return size

    def _create_order_tracker(self) -> MexcPerpetualOrderTracker:
        return MexcPerpetualOrderTracker(connector=self)

    def get_buy_collateral_token(self, trading_pair: str) -> str:
        trading_rule = self._trading_rules.get(trading_pair)
        if trading_rule is not None:
            return trading_rule.buy_order_collateral_token
        return "USDT"

    def get_sell_collateral_token(self, trading_pair: str) -> str:
        return self.get_buy_collateral_token(trading_pair)

    async def start_network(self):
        await self._initialize_trading_pair_symbol_map()
        await super().start_network()

    def _is_request_exception_related_to_time_synchronizer(self, request_exception: Exception) -> bool:
        text = str(request_exception).lower()
        return any(token in text for token in ("timestamp", "request time", "request-time", "recv-window", "recv_window"))

    def _is_order_not_found_during_status_update_error(self, status_update_exception: Exception) -> bool:
        return self._response_error_code(status_update_exception) in CONSTANTS.ORDER_NOT_FOUND_ERROR_CODES

    def _is_maintenance_error(self, exception: Exception) -> bool:
        return self._exception_contains_code(exception, CONSTANTS.MAINTENANCE_ERROR_CODES)

    def _is_unconfirmed_status_error(self, error: Exception) -> bool:
        if isinstance(error, (asyncio.TimeoutError, TimeoutError)) or self._is_maintenance_error(error):
            return True
        return "http status is 500" in str(error).lower()

    async def _handle_update_error_for_active_order(self, order: InFlightOrder, error: Exception):
        # Timeouts, maintenance, and HTTP 500 do not prove the order is gone.
        # The base not-found counter marks FAILED and drops the order. That path
        # stays only for an order that was never submitted, acked, or seen.
        if self._is_unconfirmed_status_error(error) or not self._absence_is_allowed(order.client_order_id):
            self.record_order_observation(
                order.client_order_id,
                order.trading_pair,
                status_unknown=True,
                apply_state=False,
            )
            self.logger().warning(
                "Status update for %s failed without proving the order is gone: %s",
                order.client_order_id,
                error,
            )
            return
        await super()._handle_update_error_for_active_order(order, error)

    def _is_order_not_found_during_cancelation_error(self, cancelation_exception: Exception) -> bool:
        return self._response_error_code(cancelation_exception) in CONSTANTS.ORDER_NOT_FOUND_ERROR_CODES

    @staticmethod
    def _response_error_code(exception: Exception) -> Optional[int]:
        code = getattr(exception, "code", None)
        try:
            return int(code) if code is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _exception_contains_code(exception: Exception, codes: set) -> bool:
        text = str(exception)
        return any(str(code) in text for code in codes)

    @staticmethod
    def ensure_successful_response(response: Any) -> Any:
        if not isinstance(response, dict):
            return response
        code = response.get("code")
        success = response.get("success", True)
        code_ok = code in (None, 0, "0")
        if success is False or not code_ok:
            message = response.get("message") or response.get("msg") or ""
            try:
                numeric_code = int(code)
            except (TypeError, ValueError):
                numeric_code = None
            if numeric_code in CONSTANTS.MAINTENANCE_ERROR_CODES:
                message = f"Futures API is under maintenance ({message})"
            raise MexcResponseError(numeric_code, message)
        return response

    async def _api_request(self, *args, **kwargs) -> Dict[str, Any]:
        response = await super()._api_request(*args, **kwargs)
        if kwargs.get("return_err"):
            return response
        return self.ensure_successful_response(response)

    def _create_web_assistants_factory(self) -> WebAssistantsFactory:
        return web_utils.build_api_factory(
            throttler=self._throttler,
            time_synchronizer=self._time_synchronizer,
            auth=self._auth,
        )

    def _create_order_book_data_source(self) -> OrderBookTrackerDataSource:
        return MexcPerpetualAPIOrderBookDataSource(
            trading_pairs=self.trading_pairs,
            connector=self,
            api_factory=self._web_assistants_factory,
        )

    def _create_user_stream_data_source(self) -> UserStreamTrackerDataSource:
        return MexcPerpetualUserStreamDataSource(
            auth=self._auth,
            trading_pairs=self._trading_pairs,
            connector=self,
            api_factory=self._web_assistants_factory,
        )

    def _get_fee(
        self,
        base_currency: str,
        quote_currency: str,
        order_type: OrderType,
        order_side: TradeType,
        position_action: PositionAction,
        amount: Decimal,
        price: Decimal = s_decimal_NaN,
        is_maker: Optional[bool] = None,
    ) -> TradeFeeBase:
        is_maker = is_maker or (order_type is OrderType.LIMIT_MAKER)
        trading_pair = combine_to_hb_trading_pair(base=base_currency, quote=quote_currency)
        if trading_pair in self._trading_fees:
            fee_schema: TradeFeeSchema = self._trading_fees[trading_pair]
            fee_rate = fee_schema.maker_percent_fee_decimal if is_maker else fee_schema.taker_percent_fee_decimal
            return TradeFeeBase.new_spot_fee(
                fee_schema=fee_schema,
                trade_type=order_side,
                percent=fee_rate,
            )
        return build_trade_fee(
            self.name,
            is_maker,
            base_currency=base_currency,
            quote_currency=quote_currency,
            order_type=order_type,
            order_side=order_side,
            amount=amount,
            price=price,
        )

    async def _update_trading_fees(self):
        contracts = await self._make_trading_pairs_request()
        for contract in contracts:
            if not mexc_perpetual_utils.is_exchange_information_valid(contract):
                continue
            trading_pair = await self.trading_pair_associated_to_exchange_symbol(symbol=contract["symbol"])
            self._trading_fees[trading_pair] = TradeFeeSchema(
                maker_percent_fee_decimal=Decimal(str(contract.get("makerFeeRate") or "0")),
                taker_percent_fee_decimal=Decimal(str(contract.get("takerFeeRate") or "0")),
                percent_fee_token=contract.get("settleCoin") or "USDT",
            )

    async def _place_cancel(self, order_id: str, tracked_order: InFlightOrder):
        if tracked_order.exchange_order_id:
            response = await self._api_post(
                path_url=CONSTANTS.CANCEL_ORDER_PATH,
                data={"orderIds": [self._normalize_order_id(tracked_order.exchange_order_id)]},
                is_auth_required=True,
            )
        else:
            symbol = await self.exchange_symbol_associated_to_pair(tracked_order.trading_pair)
            response = await self._api_post(
                path_url=CONSTANTS.CANCEL_BY_EXTERNAL_PATH,
                data=[{"symbol": symbol, "externalOid": tracked_order.client_order_id}],
                is_auth_required=True,
            )
        results = response.get("data") or []
        if isinstance(results, dict):
            results = [results]
        for item in results:
            if not isinstance(item, dict):
                continue
            error_code = int(item.get("errorCode") or 0)
            if error_code == CONSTANTS.ORDER_CANNOT_CANCEL_ERROR_CODE:
                await self._reconcile_uncancelable_order(tracked_order)
                return False
            if error_code != 0:
                self.record_order_observation(
                    tracked_order.client_order_id,
                    tracked_order.trading_pair,
                    payload=item,
                    apply_state=False,
                )
                raise MexcResponseError(error_code, str(item.get("errorMsg", "")))
            self.record_order_observation(
                tracked_order.client_order_id,
                tracked_order.trading_pair,
                payload=item,
                apply_state=False,
            )
        accounted = await self._account_for_cancel_fills(tracked_order)
        return accounted

    async def _place_order(
        self,
        order_id: str,
        trading_pair: str,
        amount: Decimal,
        trade_type: TradeType,
        order_type: OrderType,
        price: Decimal,
        position_action: PositionAction = PositionAction.NIL,
        **kwargs,
    ) -> Tuple[str, float]:
        data = await self._build_order_payload(
            trading_pair=trading_pair,
            amount=amount,
            trade_type=trade_type,
            order_type=order_type,
            price=price,
            position_action=position_action,
            client_order_id=order_id,
            **kwargs,
        )
        self.record_order_observation(order_id, trading_pair, submitted=True, apply_state=False)
        if not self._position_mode_confirmed:
            raise IOError("Refusing to trade until the account position mode has been read.")
        if order_id in self._cancel_requests:
            raise RuntimeError(f"Order {order_id} was cancelled before it was sent")
        try:
            response = await self._api_post(
                path_url=CONSTANTS.PLACE_ORDER_PATH,
                data=data,
                is_auth_required=True,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self._is_ambiguous_create_error(exc):
                recovered = await self._recover_uncertain_order(trading_pair, order_id)
                if recovered == "absent":
                    raise
                if recovered is not None:
                    return recovered
                raise UncertainOrderCreate(str(exc)) from exc
            recovered = await self._recover_order_by_client_id(trading_pair, order_id)
            if recovered is not None:
                return recovered
            raise
        exchange_order_id = str(response["data"]["orderId"])
        self.record_order_observation(
            order_id,
            trading_pair,
            payload=response.get("data") if isinstance(response.get("data"), dict) else None,
            exchange_order_id=exchange_order_id,
            acked=True,
            apply_state=False,
        )
        return exchange_order_id, self.current_timestamp

    async def _build_order_payload(
        self,
        trading_pair: str,
        amount: Decimal,
        trade_type: TradeType,
        order_type: OrderType,
        price: Decimal,
        position_action: PositionAction,
        client_order_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        symbol = await self.exchange_symbol_associated_to_pair(trading_pair)
        leverage = kwargs.get("leverage")
        if leverage is None:
            leverage = self.get_leverage(trading_pair)
        data: Dict[str, Any] = {
            "symbol": symbol,
            "price": self._json_number(self._price_string(price, order_type)),
            "vol": self._json_number(self._base_amount_to_contracts(trading_pair, amount)),
            "leverage": int(leverage),
            "side": self._exchange_side(trade_type, position_action),
            "type": self._execution_type_code(order_type, kwargs.get("time_in_force")),
            "openType": self._open_type(),
            "positionMode": self._position_mode_code(kwargs.get("position_mode")),
        }
        position_id = self._position_id_for_order(trading_pair, trade_type, position_action, kwargs.get("reduce_only"))
        if position_id is not None:
            data["positionId"] = position_id
        if client_order_id:
            data["externalOid"] = client_order_id
        reduce_only = kwargs.get("reduce_only")
        if reduce_only is None:
            reduce_only = self.position_mode is PositionMode.ONEWAY and position_action is PositionAction.CLOSE
        if reduce_only:
            data["reduceOnly"] = True
        self._attach_stop_prices(data, kwargs)
        return data

    async def place_batch_orders(self, orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        POST /api/v1/private/order/submit_batch. MEXC allows this only for market maker accounts,
        at most 50 orders, and counts it as 4 requests per 2 seconds.
        """
        if len(orders) > CONSTANTS.BATCH_ORDER_MAX:
            raise ValueError(f"Batch place accepts at most {CONSTANTS.BATCH_ORDER_MAX} orders.")
        payload = []
        for order in orders:
            body = await self._build_order_payload(
                trading_pair=order["trading_pair"],
                amount=Decimal(str(order["amount"])),
                trade_type=order["trade_type"],
                order_type=order["order_type"],
                price=Decimal(str(order["price"])),
                position_action=order.get("position_action", PositionAction.OPEN),
                client_order_id=order.get("client_order_id"),
                leverage=order.get("leverage"),
                time_in_force=order.get("time_in_force"),
                position_mode=order.get("position_mode"),
                reduce_only=order.get("reduce_only"),
            )
            body["stpMode"] = int(order.get("stp_mode", CONSTANTS.STP_MODE_NONE))
            payload.append(body)
        response = await self._api_post(
            path_url=CONSTANTS.BATCH_ORDER_PATH,
            data=payload,
            is_auth_required=True,
        )
        return response.get("data") or []

    async def place_plan_order(
        self,
        trading_pair: str,
        amount: Decimal,
        trade_type: TradeType,
        trigger_price: Decimal,
        trigger_type: int,
        order_type: OrderType = OrderType.LIMIT,
        price: Optional[Decimal] = None,
        position_action: PositionAction = PositionAction.OPEN,
        trend: int = CONSTANTS.TRIGGER_PRICE_LATEST,
        execute_cycle: int = CONSTANTS.EXECUTE_CYCLE_7D,
        **kwargs,
    ) -> str:
        """
        Standalone trigger order. POST /api/v1/private/planorder/place/v2.
        trigger_type 1 fires at or above the trigger, 2 at or below.
        trend and the optional loss_trend / profit_trend are 1 latest, 2 fair, 3 index.
        """
        symbol = await self.exchange_symbol_associated_to_pair(trading_pair)
        data: Dict[str, Any] = {
            "symbol": symbol,
            "vol": self._json_number(self._base_amount_to_contracts(trading_pair, amount)),
            "leverage": int(kwargs.get("leverage", self.get_leverage(trading_pair))),
            "side": self._exchange_side(trade_type, position_action),
            "openType": self._open_type(),
            "triggerPrice": self._json_number(trigger_price),
            "triggerType": int(trigger_type),
            "executeCycle": int(execute_cycle),
            "orderType": self._execution_type_code(order_type, kwargs.get("time_in_force")),
            "trend": int(trend),
            "positionMode": self._position_mode_code(kwargs.get("position_mode")),
        }
        if price is not None and order_type is not OrderType.MARKET:
            data["price"] = self._json_number(price)
        if kwargs.get("reduce_only"):
            data["reduceOnly"] = True
        if kwargs.get("price_protect") is not None:
            data["priceProtect"] = int(kwargs["price_protect"])
        self._attach_stop_prices(data, kwargs)
        response = await self._api_post(
            path_url=CONSTANTS.PLAN_ORDER_PLACE_PATH,
            data=data,
            is_auth_required=True,
        )
        return str(response.get("data"))

    async def cancel_plan_orders(self, orders: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Cancel up to 50 plan orders. Each item needs symbol and order_id.
        """
        if len(orders) > CONSTANTS.BATCH_ORDER_MAX:
            raise ValueError(f"Cancel plan orders accepts at most {CONSTANTS.BATCH_ORDER_MAX} orders.")
        # The parameter table names this list `orders`. The published example is the bare array.
        payload = [
            {"symbol": item["symbol"], "orderId": self._normalize_order_id(item["order_id"])}
            for item in orders
        ]
        return await self._api_post(
            path_url=CONSTANTS.PLAN_ORDER_CANCEL_PATH,
            data=payload,
            is_auth_required=True,
        )

    async def cancel_all_plan_orders(self, trading_pair: Optional[str] = None) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        if trading_pair:
            data["symbol"] = await self.exchange_symbol_associated_to_pair(trading_pair)
        return await self._api_post(
            path_url=CONSTANTS.PLAN_ORDER_CANCEL_ALL_PATH,
            data=data or {},
            is_auth_required=True,
        )

    async def list_plan_orders(
        self,
        trading_pair: Optional[str] = None,
        states: str = "1",
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        page_num: int = 1,
        page_size: int = 20,
    ) -> Any:
        """
        Plan order list. states=1 is untriggered, which is the open set.
        The endpoint requires a start and end time. The default window is the last 7 days.
        """
        now_ms = int(time.time() * 1000)
        params: Dict[str, Any] = {
            "states": states,
            "start_time": int(start_time if start_time is not None else now_ms - 7 * 24 * 60 * 60 * 1000),
            "end_time": int(end_time if end_time is not None else now_ms),
            "page_num": page_num,
            "page_size": page_size,
        }
        if trading_pair:
            params["symbol"] = await self.exchange_symbol_associated_to_pair(trading_pair)
        response = await self._api_get(
            path_url=CONSTANTS.PLAN_ORDER_LIST_PATH,
            params=params,
            is_auth_required=True,
        )
        return response.get("data")

    async def change_plan_order_price(
        self,
        trading_pair: str,
        order_id: str,
        trigger_price: Decimal,
        price: Decimal,
        order_type: int,
        trigger_type: int,
        trend: int = CONSTANTS.TRIGGER_PRICE_LATEST,
        source: int = 2,
    ) -> Dict[str, Any]:
        """
        POST /api/v1/private/planorder/change_price.
        source 1 is the trade list and 2 is the order list.
        """
        return await self._api_post(
            path_url=CONSTANTS.PLAN_ORDER_CHANGE_PRICE_PATH,
            data={
                "symbol": await self.exchange_symbol_associated_to_pair(trading_pair),
                "orderId": self._normalize_order_id(order_id),
                "triggerPrice": self._json_number(trigger_price),
                "price": self._json_number(price),
                "orderType": int(order_type),
                "triggerType": int(trigger_type),
                "trend": int(trend),
                "from": int(source),
            },
            is_auth_required=True,
        )

    async def change_plan_order_stop(
        self,
        trading_pair: str,
        order_id: str,
        stop_loss_price: Optional[Decimal] = None,
        take_profit_price: Optional[Decimal] = None,
        loss_trend: Optional[int] = None,
        profit_trend: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Change or clear the stop and take profit attached to a plan order.
        Sending 0 for a price clears that side.
        """
        data: Dict[str, Any] = {
            "symbol": await self.exchange_symbol_associated_to_pair(trading_pair),
            "orderId": self._normalize_order_id(order_id),
        }
        self._put_optional_price(data, "stopLossPrice", stop_loss_price)
        self._put_optional_price(data, "takeProfitPrice", take_profit_price)
        if loss_trend is not None:
            data["lossTrend"] = int(loss_trend)
        if profit_trend is not None:
            data["profitTrend"] = int(profit_trend)
        return await self._api_post(
            path_url=CONSTANTS.PLAN_ORDER_CHANGE_STOP_PATH,
            data=data,
            is_auth_required=True,
        )

    async def place_position_stop_order(
        self,
        trading_pair: str,
        position_id: str,
        amount: Decimal,
        stop_loss_price: Optional[Decimal] = None,
        take_profit_price: Optional[Decimal] = None,
        loss_trend: int = CONSTANTS.TRIGGER_PRICE_LATEST,
        profit_trend: int = CONSTANTS.TRIGGER_PRICE_LATEST,
        **kwargs,
    ) -> str:
        """
        Standalone TP/SL for an open position. POST /api/v1/private/stoporder/place.
        Amount is base size and is converted to contracts.
        """
        if (stop_loss_price is None or stop_loss_price <= 0) and (take_profit_price is None or take_profit_price <= 0):
            raise ValueError("A position stop order needs a stop loss price or a take profit price.")
        data: Dict[str, Any] = {
            "positionId": self._normalize_order_id(position_id),
            "vol": self._json_number(self._base_amount_to_contracts(trading_pair, amount)),
            "lossTrend": int(loss_trend),
            "profitTrend": int(profit_trend),
        }
        self._put_optional_price(data, "stopLossPrice", stop_loss_price)
        self._put_optional_price(data, "takeProfitPrice", take_profit_price)
        for key, field in (
            ("price_protect", "priceProtect"),
            ("vol_type", "volType"),
            ("take_profit_type", "takeProfitType"),
            ("stop_loss_type", "stopLossType"),
            ("take_profit_reverse", "takeProfitReverse"),
            ("stop_loss_reverse", "stopLossReverse"),
        ):
            if kwargs.get(key) is not None:
                data[field] = int(kwargs[key])
        if kwargs.get("profit_loss_vol_type"):
            data["profitLossVolType"] = kwargs["profit_loss_vol_type"]
        if kwargs.get("take_profit_order_price") is not None:
            data["takeProfitOrderPrice"] = self._json_number(kwargs["take_profit_order_price"])
        if kwargs.get("stop_loss_order_price") is not None:
            data["stopLossOrderPrice"] = self._json_number(kwargs["stop_loss_order_price"])
        response = await self._api_post(
            path_url=CONSTANTS.STOP_ORDER_PLACE_PATH,
            data=data,
            is_auth_required=True,
        )
        return str(response.get("data"))

    async def cancel_stop_orders(self, stop_plan_order_ids: List[str]) -> Dict[str, Any]:
        if len(stop_plan_order_ids) > CONSTANTS.BATCH_ORDER_MAX:
            raise ValueError(f"Cancel stop orders accepts at most {CONSTANTS.BATCH_ORDER_MAX} orders.")
        # The parameter table names this list `orders`. The published example is the bare array.
        return await self._api_post(
            path_url=CONSTANTS.STOP_ORDER_CANCEL_PATH,
            data=[{"stopPlanOrderId": self._normalize_order_id(order_id)} for order_id in stop_plan_order_ids],
            is_auth_required=True,
        )

    async def cancel_all_stop_orders(
        self,
        trading_pair: Optional[str] = None,
        position_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        if position_id is not None:
            data["positionId"] = self._normalize_order_id(position_id)
        elif trading_pair:
            data["symbol"] = await self.exchange_symbol_associated_to_pair(trading_pair)
        return await self._api_post(
            path_url=CONSTANTS.STOP_ORDER_CANCEL_ALL_PATH,
            data=data or {},
            is_auth_required=True,
        )

    async def list_open_stop_orders(self, trading_pair: Optional[str] = None) -> Any:
        params: Dict[str, Any] = {}
        if trading_pair:
            params["symbol"] = await self.exchange_symbol_associated_to_pair(trading_pair)
        response = await self._api_get(
            path_url=CONSTANTS.STOP_ORDER_OPEN_PATH,
            params=params or None,
            is_auth_required=True,
        )
        return response.get("data")

    async def change_stop_order_price(
        self,
        order_id: str,
        stop_loss_price: Optional[Decimal] = None,
        take_profit_price: Optional[Decimal] = None,
        loss_trend: Optional[int] = None,
        profit_trend: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        POST /api/v1/private/stoporder/change_price. order_id is the limit order id.
        Empty or zero prices clear that side.
        """
        data: Dict[str, Any] = {"orderId": self._normalize_order_id(order_id)}
        self._put_stop_change_fields(data, stop_loss_price, take_profit_price, loss_trend, profit_trend, kwargs)
        return await self._api_post(
            path_url=CONSTANTS.STOP_ORDER_CHANGE_PRICE_PATH,
            data=data,
            is_auth_required=True,
        )

    async def change_stop_plan_price(
        self,
        stop_plan_order_id: str,
        stop_loss_price: Optional[Decimal] = None,
        take_profit_price: Optional[Decimal] = None,
        loss_trend: Optional[int] = None,
        profit_trend: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        POST /api/v1/private/stoporder/change_plan_price. At least one price must be above zero.
        """
        data: Dict[str, Any] = {"stopPlanOrderId": self._normalize_order_id(stop_plan_order_id)}
        self._put_stop_change_fields(data, stop_loss_price, take_profit_price, loss_trend, profit_trend, kwargs)
        return await self._api_post(
            path_url=CONSTANTS.STOP_ORDER_CHANGE_PLAN_PRICE_PATH,
            data=data,
            is_auth_required=True,
        )

    async def _recover_order_by_client_id(self, trading_pair: str, client_order_id: str) -> Optional[Tuple[str, float]]:
        """
        A create call can time out after the order was accepted. Look the client id up before
        the caller treats the order as failed, so a retry does not submit a second order.
        """
        try:
            order_data = await self._request_order_by_client_id(trading_pair, client_order_id)
        except Exception:
            return None
        if not order_data or order_data.get("orderId") in (None, ""):
            return None
        return str(order_data["orderId"]), self.current_timestamp

    def cancel(self, trading_pair: str, client_order_id: str):
        self._cancel_requests.add(client_order_id)
        return super().cancel(trading_pair, client_order_id)

    async def _create_order(
        self,
        trade_type: TradeType,
        order_id: str,
        trading_pair: str,
        amount: Decimal,
        order_type: OrderType,
        price: Optional[Decimal] = None,
        position_action: PositionAction = PositionAction.NIL,
        **kwargs,
    ):
        self._creates_in_flight.add(order_id)
        try:
            await super()._create_order(
                trade_type,
                order_id,
                trading_pair,
                amount,
                order_type,
                price,
                position_action=position_action,
                **kwargs,
            )
        finally:
            self._creates_in_flight.discard(order_id)
            self._creates_finished.add(order_id)

    def _find_order(self, order_id: str) -> Optional[InFlightOrder]:
        return (
            self._order_tracker.fetch_tracked_order(order_id)
            or self._order_tracker.fetch_order(order_id)
            or self._order_tracker.fetch_lost_order(order_id)
        )

    @staticmethod
    def _order_is_terminal(order: Optional[InFlightOrder]) -> bool:
        if order is None:
            return False
        return order.current_state in (OrderState.CANCELED, OrderState.FAILED) or (
            order.is_done and order.is_filled
        )

    def is_cancel_confirmed(self, order_id: str) -> bool:
        """
        True only after the cancel is done, the order is filled, or the exchange has
        confirmed that this client id does not exist. An in-flight cancel is not confirmed.
        """
        if order_id in self._cancel_requests:
            return False
        if order_id in self._confirmed_absent:
            return True
        return self._order_is_terminal(self._find_order(order_id))

    def is_position_mode_ready(self) -> bool:
        return bool(self._position_mode_confirmed)

    def _settle_cancel(self, order_id: str, order: Optional[InFlightOrder]) -> str:
        self._cancel_requests.discard(order_id)
        return order_id

    async def _execute_cancel(self, trading_pair: str, order_id: str):
        self._cancel_requests.add(order_id)
        spins = 0
        while order_id in self._cancel_requests and spins < 200:
            tracked = self._find_order(order_id)
            if self._order_is_terminal(tracked):
                return self._settle_cancel(order_id, tracked)
            if tracked is None:
                if order_id in self._creates_in_flight or order_id not in self._creates_finished:
                    await self._sleep(0.01)
                    spins += 1
                    continue
                return self._settle_cancel(order_id, None)
            await self._execute_order_cancel(tracked)
            await self._sleep(0)
            tracked = self._find_order(order_id)
            if self._order_is_terminal(tracked):
                return self._settle_cancel(order_id, tracked)
            # Not canceled and not confirmed absent. Leave the request open so the
            # executor retries instead of placing a replacement.
            return None
        return None

    async def _place_order_and_process_update(self, order: InFlightOrder, **kwargs) -> str:
        kwargs = dict(kwargs)
        kwargs.pop("position_action", None)
        exchange_order_id, _update_timestamp = await self._place_order(
            order_id=order.client_order_id,
            trading_pair=order.trading_pair,
            amount=order.amount,
            trade_type=order.trade_type,
            order_type=order.order_type,
            price=order.price,
            position_action=order.position,
            **kwargs,
        )
        if order.client_order_id in self._cancel_requests:
            if order.exchange_order_id is None:
                order.update_exchange_order_id(str(exchange_order_id))
            await self._execute_order_cancel(order)
            return str(exchange_order_id)
        self.record_order_observation(
            order.client_order_id,
            order.trading_pair,
            exchange_order_id=str(exchange_order_id),
            new_state=OrderState.OPEN,
            acked=True,
            submitted=True,
        )
        return str(exchange_order_id)

    def _on_order_failure(self, order_id: str, trading_pair: str, amount: Decimal, trade_type: TradeType,
                          order_type: OrderType, price: Optional[Decimal], exception: Exception, **kwargs):
        if isinstance(exception, UncertainOrderCreate):
            self.logger().warning(
                "Order %s was not confirmed after the create call. Keeping client id %s in flight.",
                trading_pair,
                order_id,
            )
            return
        super()._on_order_failure(
            order_id=order_id,
            trading_pair=trading_pair,
            amount=amount,
            trade_type=trade_type,
            order_type=order_type,
            price=price,
            exception=exception,
            **kwargs,
        )

    def _is_ambiguous_create_error(self, exception: Exception) -> bool:
        if isinstance(exception, (asyncio.TimeoutError, TimeoutError)):
            return True
        if self._is_maintenance_error(exception):
            return True
        text = str(exception).lower()
        if "http status is" in text:
            return False
        return any(token in text for token in ("timeout", "connection", "network", "server disconnected"))

    async def _recover_uncertain_order(self, trading_pair: str, client_order_id: str):
        absent = 0
        for _attempt in range(3):
            try:
                order_data = await self._request_order_by_client_id(trading_pair, client_order_id)
            except Exception as exc:
                if self._is_order_not_found_during_status_update_error(exc):
                    absent += 1
                    if absent >= 3:
                        return "absent"
                    continue
                if self._is_ambiguous_create_error(exc) or self._is_maintenance_error(exc):
                    continue
                return None
            else:
                if order_data and order_data.get("orderId") not in (None, ""):
                    return str(order_data["orderId"]), self.current_timestamp
        return None

    async def _account_for_cancel_fills(self, order: InFlightOrder) -> bool:
        try:
            order_data = await self._request_order_by_client_id(order.trading_pair, order.client_order_id)
        except Exception as exc:
            self.logger().warning(
                "Cancel of %s succeeded but the order fill could not be read: %s",
                order.client_order_id,
                exc,
            )
            return False
        self.record_order_observation(
            order.client_order_id,
            order.trading_pair,
            payload=order_data,
            apply_state=False,
            materialize_fill=True,
        )
        await self._apply_deal_details(order)
        self.synthesize_reported_fill(order)
        return True

    async def _reconcile_uncancelable_order(self, order: InFlightOrder):
        last_error: Optional[Exception] = None
        for _attempt in range(5):
            try:
                order_data = await self._request_order_by_client_id(order.trading_pair, order.client_order_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_error = exc
                await self._sleep(0.02)
                continue
            state = self._order_state_from_payload(order_data)
            self.record_order_observation(
                order.client_order_id,
                order.trading_pair,
                payload=order_data,
                new_state=state if state in (OrderState.FILLED, OrderState.CANCELED, OrderState.FAILED) else None,
                apply_state=state in (OrderState.FILLED, OrderState.CANCELED, OrderState.FAILED),
                materialize_fill=True,
            )
            await self._apply_deal_details(order)
            self.synthesize_reported_fill(order)
            return
        self.logger().warning(
            "Cancel error 2041 for %s and the order could not be read after retries: %s",
            order.client_order_id,
            last_error,
        )

    async def _apply_deal_details(self, order: InFlightOrder):
        if order.exchange_order_id in (None, ""):
            return
        reported = getattr(order, "reported_deal_base", None)
        if reported is not None and Decimal(str(reported)) <= s_decimal_0:
            return
        try:
            updates = await self._all_trade_updates_for_order(order)
        except Exception as exc:
            self.logger().debug("Deal details for %s were not available: %s", order.client_order_id, exc)
            return
        self._drop_synthetic_fill(order)
        for update in updates:
            self._order_tracker.process_trade_update(update)

    def _remember_reported_deal(self, order: InFlightOrder, payload: Dict[str, Any]):
        if not isinstance(payload, dict) or payload.get("dealVol") in (None, ""):
            return
        try:
            contracts = Decimal(str(payload.get("dealVol") or "0"))
            base = contracts * self.get_contract_size(order.trading_pair)
        except Exception:
            return
        previous = getattr(order, "reported_deal_base", None)
        if previous is None or base > Decimal(str(previous)):
            order.reported_deal_base = base
        average = payload.get("dealAvgPrice")
        if average not in (None, "", 0, "0", "0.0"):
            order.reported_deal_price = Decimal(str(average))

    def synthesize_reported_fill(self, order: InFlightOrder):
        # Safe: this only books the gap up to reported_deal_base, which
        # record_order_observation already stored in the counted-fill ledger.
        reported = getattr(order, "reported_deal_base", None)
        if reported is None:
            return
        self._drop_synthetic_fill(order)
        gap = Decimal(str(reported)) - order.executed_amount_base
        if gap <= Decimal("0.00000001"):
            return
        price = getattr(order, "reported_deal_price", None) or order.price
        if price is None:
            return
        price = Decimal(str(price))
        if price <= s_decimal_0:
            price = order.price
        fee = TradeFeeBase.new_perpetual_fee(
            fee_schema=self.trade_fee_schema(),
            position_action=getattr(order, "position", PositionAction.NIL),
            percent_token=self.get_buy_collateral_token(order.trading_pair),
            flat_fees=[],
        )
        self._order_tracker.process_trade_update(
            TradeUpdate(
                trade_id=f"mexc-reported-deal-{order.client_order_id}",
                client_order_id=order.client_order_id,
                exchange_order_id=str(order.exchange_order_id or ""),
                trading_pair=order.trading_pair,
                fill_timestamp=self.current_timestamp or 0,
                fill_price=price,
                fill_base_amount=gap,
                fill_quote_amount=price * gap,
                fee=fee,
            )
        )

    def _drop_synthetic_fill(self, order: InFlightOrder):
        # Safe: this removes only the temporary synthetic trade so a real deal can
        # replace it. The counted-fill ledger is not reduced, and synthesize_reported_fill
        # restores any gap up to that ledger.
        trade_id = f"mexc-reported-deal-{order.client_order_id}"
        fill = order.order_fills.pop(trade_id, None)
        if fill is None:
            return
        order.executed_amount_base -= fill.fill_base_amount
        order.executed_amount_quote -= fill.fill_quote_amount
        if order.executed_amount_base < s_decimal_0:
            order.executed_amount_base = s_decimal_0
        if order.executed_amount_quote < s_decimal_0:
            order.executed_amount_quote = s_decimal_0

    def _apply_balance_row(self, balance_data: Dict[str, Any], from_rest: bool):
        asset = balance_data.get("currency")
        if not asset:
            return
        available = Decimal(str(balance_data.get("availableBalance") or "0"))
        frozen = Decimal(str(balance_data.get("frozenBalance") or "0"))
        margin = Decimal(str(balance_data.get("positionMargin") or "0"))
        cash = available + frozen + margin
        self._account_available_balances[asset] = available
        explicit_unrealized = None
        for field in ("unrealized", "unrealizedPnl", "unRealizedPnl", "unProfit"):
            if balance_data.get(field) not in (None, ""):
                explicit_unrealized = Decimal(str(balance_data[field]))
                break
        if from_rest and balance_data.get("equity") is not None:
            equity = Decimal(str(balance_data["equity"]))
            self._equity_gap[asset] = equity - cash
            self._account_balances[asset] = equity
            return
        if explicit_unrealized is not None:
            self._equity_gap[asset] = explicit_unrealized
        gap = self._equity_gap.get(asset, s_decimal_0)
        self._account_balances[asset] = cash + gap

    def _position_id_for_order(
        self,
        trading_pair: str,
        trade_type: TradeType,
        position_action: PositionAction,
        reduce_only: Optional[bool],
    ) -> Optional[int]:
        closing = position_action is PositionAction.CLOSE or bool(reduce_only)
        if not closing:
            return None
        if trade_type is TradeType.SELL:
            side = PositionSide.LONG
        else:
            side = PositionSide.SHORT
        return self._position_ids.get((trading_pair, side))

    async def _update_balances(self):
        response = await self._api_get(path_url=CONSTANTS.ASSETS_PATH, is_auth_required=True)
        self._account_available_balances.clear()
        self._account_balances.clear()
        for balance_data in response.get("data") or []:
            self._apply_balance_row(balance_data, from_rest=True)

    async def _update_positions(self):
        response = await self._api_get(path_url=CONSTANTS.OPEN_POSITIONS_PATH, is_auth_required=True)
        seen_keys = set()
        for raw_position in response.get("data") or []:
            parsed = await self._parse_position(raw_position)
            if parsed is None:
                continue
            pos_key, position = parsed
            seen_keys.add(pos_key)
            self._perpetual_trading.set_position(pos_key, position)
        for pos_key in list(self.account_positions.keys()):
            if pos_key not in seen_keys:
                self._perpetual_trading.remove_position(pos_key)

    async def _update_orders_fills(self, orders: List[InFlightOrder]):
        for order in orders:
            try:
                await self._reconcile_exchange_deals(order)
            except asyncio.CancelledError:
                raise
            except Exception as request_error:
                self.logger().warning(
                    f"Failed to fetch trade updates for order {order.client_order_id}. Error: {request_error}",
                    exc_info=request_error,
                )

    async def _reconcile_exchange_deals(self, order: InFlightOrder):
        """
        Replace a synthesized dealVol fill with the exchange's real deals.
        Executed size stays at the reported deal when the real deals are short,
        and it does not grow past that by keeping the synthetic fill as well.
        """
        # Dropping the synthetic trade only adjusts the in-memory order so the real
        # deals can replace it. The counted-fill ledger is not reduced, and the
        # following record_order_observation calls put each real deal back.
        self._drop_synthetic_fill(order)
        try:
            response = await self._request_order_fills(order)
        except Exception:
            self.synthesize_reported_fill(order)
            raise
        for fill in response.get("data") or []:
            if not isinstance(fill, dict):
                continue
            self.record_order_observation(
                order.client_order_id,
                order.trading_pair,
                trade_fill=fill,
                exchange_order_id=str(fill.get("orderId") or order.exchange_order_id or ""),
                apply_state=False,
            )
        self.synthesize_reported_fill(order)

    async def _all_trade_updates_for_order(self, order: InFlightOrder) -> List[TradeUpdate]:
        if order.exchange_order_id is None:
            return []
        response = await self._request_order_fills(order)
        updates = []
        for fill in response.get("data") or []:
            updates.append(self._parse_trade_update(fill, order))
        return updates

    async def _request_order_fills(self, order: InFlightOrder) -> Dict[str, Any]:
        path = CONSTANTS.ORDER_DEALS_PATH.format(order_id=order.exchange_order_id)
        return await self._api_get(
            path_url=path,
            is_auth_required=True,
            limit_id=CONSTANTS.ORDER_DEALS_LIMIT_ID,
        )

    async def _request_order_status(self, tracked_order: InFlightOrder) -> OrderUpdate:
        try:
            order_data = await self._request_order_status_data(tracked_order)
            self.record_order_observation(
                tracked_order.client_order_id,
                tracked_order.trading_pair,
                payload=order_data,
                apply_state=False,
            )
            if self._order_state_from_payload(order_data) == OrderState.CANCELED:
                await self._apply_deal_details(tracked_order)
                self.synthesize_reported_fill(tracked_order)
            return self._order_update_from_rest(order_data, tracked_order)
        except IOError as exception:
            if self._is_request_exception_related_to_time_synchronizer(exception):
                return OrderUpdate(
                    client_order_id=tracked_order.client_order_id,
                    trading_pair=tracked_order.trading_pair,
                    update_timestamp=self.current_timestamp,
                    new_state=tracked_order.current_state,
                )
            raise

    async def _request_order_status_data(self, tracked_order: InFlightOrder) -> Dict[str, Any]:
        return await self._request_order_by_client_id(tracked_order.trading_pair, tracked_order.client_order_id)

    async def _request_order_by_client_id(self, trading_pair: str, client_order_id: str) -> Dict[str, Any]:
        symbol = await self.exchange_symbol_associated_to_pair(trading_pair)
        path = CONSTANTS.ORDER_BY_EXTERNAL_PATH.format(symbol=symbol, external_oid=client_order_id)
        response = await self._api_get(
            path_url=path,
            is_auth_required=True,
            limit_id=CONSTANTS.ORDER_BY_EXTERNAL_LIMIT_ID,
        )
        data = response.get("data")
        if not data:
            raise ValueError(f"Order {client_order_id} not found")
        return data

    async def _request_order_by_exchange_id(self, exchange_order_id: str) -> Dict[str, Any]:
        path = CONSTANTS.ORDER_GET_PATH.format(order_id=exchange_order_id)
        response = await self._api_get(
            path_url=path,
            is_auth_required=True,
            limit_id=CONSTANTS.ORDER_GET_LIMIT_ID,
        )
        data = response.get("data")
        if not data:
            raise ValueError(f"Order {exchange_order_id} not found")
        return data

    def record_order_observation(
        self,
        client_order_id: str,
        trading_pair: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        *,
        exchange_order_id: Optional[str] = None,
        new_state: Optional[OrderState] = None,
        submitted: bool = False,
        acked: bool = False,
        trade_fill: Optional[Dict[str, Any]] = None,
        apply_state: bool = True,
        materialize_fill: bool = False,
        status_unknown: bool = False,
    ) -> None:
        """
        The only writer of order memory: submission, acknowledgement, seen-live, and counted fill.
        Every websocket, REST, create, cancel, and lookup path records through here before a
        later absence decision.
        """
        if not client_order_id:
            return
        if submitted:
            self._submitted_client_ids.add(client_order_id)
        if status_unknown:
            self._status_unknown_client_ids.add(client_order_id)
        if acked or exchange_order_id not in (None, ""):
            self._acked_client_ids.add(client_order_id)
            self._seen_live_client_ids.add(client_order_id)
        order = self._find_order(client_order_id)
        pair = trading_pair or (order.trading_pair if order is not None else None)
        if isinstance(payload, dict) and pair is not None:
            if payload.get("orderId") not in (None, "") and exchange_order_id in (None, ""):
                exchange_order_id = str(payload.get("orderId"))
                self._acked_client_ids.add(client_order_id)
                self._seen_live_client_ids.add(client_order_id)
            self._remember_counted_fill(client_order_id, pair, [payload])
        if trade_fill is not None and order is not None:
            try:
                base = Decimal(str(trade_fill["vol"])) * self.get_contract_size(order.trading_pair)
                price = Decimal(str(trade_fill["price"]))
            except Exception:
                base, price = s_decimal_0, s_decimal_0
            self._remember_counted_fill(client_order_id, order.trading_pair, [{
                "dealVol": str(base / self.get_contract_size(order.trading_pair)) if base else "0",
                "dealAvgPrice": str(price),
                "state": CONSTANTS.ORDER_STATE_FILLED,
            }])
            if trade_fill.get("orderId") not in (None, ""):
                self._acked_client_ids.add(client_order_id)
                self._seen_live_client_ids.add(client_order_id)
            self._drop_synthetic_fill(order)
            self._order_tracker.process_trade_update(self._parse_trade_update(trade_fill, order))
            self._apply_counted_fill(order, client_order_id)
        elif materialize_fill and order is not None:
            self._apply_counted_fill(order, client_order_id)
        if order is not None and exchange_order_id not in (None, "") and order.exchange_order_id in (None, ""):
            order.update_exchange_order_id(str(exchange_order_id))
        if new_state in (OrderState.FILLED, OrderState.CANCELED) or (
            isinstance(payload, dict) and self._payload_fill(pair, payload)[0] > s_decimal_0
            if pair is not None else False
        ):
            self._status_unknown_client_ids.discard(client_order_id)
        if not apply_state or order is None:
            return
        state = new_state
        if state is None and isinstance(payload, dict) and payload.get("state") not in (None, ""):
            state = self._order_state_from_payload(payload)
        if state is None:
            return
        self._order_tracker.process_order_update(
            OrderUpdate(
                trading_pair=order.trading_pair,
                update_timestamp=self.current_timestamp,
                new_state=state,
                client_order_id=order.client_order_id,
                exchange_order_id=str(exchange_order_id or order.exchange_order_id or ""),
            )
        )

    def _absence_is_allowed(self, client_order_id: str) -> bool:
        if client_order_id in self._submitted_client_ids or client_order_id in self._acked_client_ids:
            return False
        if client_order_id in self._seen_live_client_ids:
            return False
        if self._max_fill_base.get(client_order_id, s_decimal_0) > s_decimal_0:
            return False
        order = self._find_order(client_order_id)
        if order is None:
            return True
        if order.exchange_order_id not in (None, ""):
            return False
        return order.current_state not in (OrderState.OPEN, OrderState.PARTIALLY_FILLED, OrderState.FILLED)

    def _apply_counted_fill(self, order: InFlightOrder, client_order_id: str):
        counted = self._max_fill_base.get(client_order_id, s_decimal_0)
        if counted <= s_decimal_0:
            return
        reported = getattr(order, "reported_deal_base", None)
        if reported is None or counted > Decimal(str(reported)):
            order.reported_deal_base = counted
            order.reported_deal_price = self._max_fill_quote.get(client_order_id, s_decimal_0) / counted
        self.synthesize_reported_fill(order)

    def counted_fill(self, client_order_id: str) -> Tuple[Decimal, Decimal]:
        return (
            self._max_fill_base.get(client_order_id, s_decimal_0),
            self._max_fill_quote.get(client_order_id, s_decimal_0),
        )

    def order_was_seen_live(self, client_order_id: str) -> bool:
        return client_order_id in self._seen_live_client_ids

    def orders_awaiting_resolution(self) -> Set[str]:
        return set(self._status_unknown_client_ids)

    async def ground_truth_exposure(self, trading_pair: str, side: TradeType) -> Tuple[Decimal, Decimal]:
        """
        Read the live position and open orders for this pair.
        Returns position base in the order's direction, and the remaining size of open
        orders on that side. Raises when either read fails.
        """
        symbol = await self.exchange_symbol_associated_to_pair(trading_pair)
        contract_size = self.get_contract_size(trading_pair)
        positions = await self._api_get(
            path_url=CONSTANTS.OPEN_POSITIONS_PATH,
            params={"symbol": symbol},
            is_auth_required=True,
        )
        orders = await self._api_get(
            path_url=CONSTANTS.OPEN_ORDERS_PATH.format(symbol=symbol),
            is_auth_required=True,
            limit_id=CONSTANTS.OPEN_ORDERS_LIMIT_ID,
        )
        if not isinstance(positions, dict) or not isinstance(orders, dict):
            raise IOError("Position or open-order response was not JSON")
        if positions.get("success") is False or orders.get("success") is False:
            raise IOError("Position or open-order response was rejected")
        wanted_position = 1 if side is TradeType.BUY else 2
        buy_sides = {1, 2}
        sell_sides = {3, 4}
        wanted_sides = buy_sides if side is TradeType.BUY else sell_sides
        position_base = s_decimal_0
        for row in positions.get("data") or []:
            if not isinstance(row, dict):
                continue
            if int(row.get("positionType") or 0) != wanted_position:
                continue
            position_base += Decimal(str(row.get("holdVol") or "0")) * contract_size
        open_base = s_decimal_0
        order_rows = orders.get("data") or []
        if isinstance(order_rows, dict):
            order_rows = order_rows.get("resultList") or order_rows.get("list") or []
        for row in order_rows:
            if not isinstance(row, dict):
                continue
            if int(row.get("side") or 0) not in wanted_sides:
                continue
            remaining = Decimal(str(row.get("vol") or "0")) - Decimal(str(row.get("dealVol") or "0"))
            if remaining < s_decimal_0:
                remaining = s_decimal_0
            open_base += remaining * contract_size
        return position_base, open_base

    async def _read_order_response(self, path: str, limit_id: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Classify one order read from the response body's code field.
        A raised error, including HTTP 500, is unknown. Digits in the URL are not a code.
        """
        try:
            response = await self._api_get(
                path_url=path,
                is_auth_required=True,
                limit_id=limit_id,
                return_err=True,
            )
        except Exception as exc:
            self.logger().warning("Order lookup failed: %s", exc)
            return "failed", None
        return self._classify_order_response(response)

    @staticmethod
    def _classify_order_response(response: Any) -> Tuple[str, Optional[Dict[str, Any]]]:
        if not isinstance(response, dict):
            return "failed", None
        code = response.get("code")
        if code in (None, ""):
            numeric = 0
        else:
            try:
                numeric = int(code)
            except (TypeError, ValueError):
                return "failed", None
        if numeric in CONSTANTS.ORDER_NOT_FOUND_ERROR_CODES:
            return "not_found", None
        if response.get("success") is False or numeric != 0:
            return "failed", None
        data = response.get("data")
        if not isinstance(data, dict) or not data:
            return "empty", None
        return "ok", data

    def _remember_counted_fill(self, client_order_id: str, trading_pair: str, payloads: List[Dict[str, Any]]):
        for payload in payloads:
            if not isinstance(payload, dict):
                continue
            base, quote = self._payload_fill(trading_pair, payload)
            previous = self._max_fill_base.get(client_order_id, s_decimal_0)
            previous_quote = self._max_fill_quote.get(client_order_id, s_decimal_0)
            if base > previous:
                self._max_fill_base[client_order_id] = base
                self._max_fill_quote[client_order_id] = max(previous_quote, quote)
            elif base == previous and quote > previous_quote:
                self._max_fill_quote[client_order_id] = quote
            if self._payload_was_seen_live(payload):
                self._seen_live_client_ids.add(client_order_id)
        counted = self._max_fill_base.get(client_order_id, s_decimal_0)
        order = self._find_order(client_order_id)
        if order is None:
            for payload in payloads:
                external = payload.get("externalOid") if isinstance(payload, dict) else None
                if external:
                    order = self._find_order(str(external))
                    if order is not None:
                        break
        if order is None or counted <= s_decimal_0:
            return
        reported = getattr(order, "reported_deal_base", None)
        if reported is None or counted > Decimal(str(reported)):
            order.reported_deal_base = counted
            order.reported_deal_price = self._max_fill_quote.get(client_order_id, s_decimal_0) / counted

    def _payload_fill(self, trading_pair: str, payload: Dict[str, Any]) -> Tuple[Decimal, Decimal]:
        if payload.get("dealVol") in (None, ""):
            return s_decimal_0, s_decimal_0
        try:
            base = Decimal(str(payload.get("dealVol") or "0")) * self.get_contract_size(trading_pair)
        except Exception:
            return s_decimal_0, s_decimal_0
        price_raw = payload.get("dealAvgPrice")
        if price_raw in (None, "", 0, "0", "0.0"):
            price_raw = payload.get("price") or "0"
        try:
            price = Decimal(str(price_raw or "0"))
        except Exception:
            price = s_decimal_0
        if price < s_decimal_0:
            price = s_decimal_0
        return base, base * price

    @staticmethod
    def _payload_was_seen_live(payload: Dict[str, Any]) -> bool:
        try:
            state = int(payload.get("state") or 0)
            deal = Decimal(str(payload.get("dealVol") or "0"))
        except Exception:
            return False
        return state in (
            CONSTANTS.ORDER_STATE_PENDING,
            CONSTANTS.ORDER_STATE_OPEN,
            CONSTANTS.ORDER_STATE_FILLED,
        ) or deal > s_decimal_0

    async def resolve_unknown_order(
        self,
        trading_pair: str,
        client_order_id: str,
        exchange_order_id: Optional[str] = None,
    ):
        """
        Look the order up by client id and by exchange order id.
        Fills are recorded before the state is chosen. A conflict or a failed read stays
        unknown. confirmed_absent is only for an order no read has ever shown as live.
        """
        symbol = await self.exchange_symbol_associated_to_pair(trading_pair)
        client_path = CONSTANTS.ORDER_BY_EXTERNAL_PATH.format(symbol=symbol, external_oid=client_order_id)
        client_kind, client_data = await self._read_order_response(client_path, CONSTANTS.ORDER_BY_EXTERNAL_LIMIT_ID)
        kinds = [client_kind]
        payloads = [client_data] if client_kind == "ok" and client_data else []
        order_data = None
        if exchange_order_id:
            order_path = CONSTANTS.ORDER_GET_PATH.format(order_id=exchange_order_id)
            order_kind, order_data = await self._read_order_response(order_path, CONSTANTS.ORDER_GET_LIMIT_ID)
            kinds.append(order_kind)
            if order_kind == "ok" and order_data:
                payloads.append(order_data)
        for payload in payloads:
            self.record_order_observation(
                client_order_id,
                trading_pair,
                payload=payload,
                apply_state=False,
                materialize_fill=True,
            )
        if any(kind == "failed" for kind in kinds):
            return None
        if self._payloads_conflict(client_data, order_data):
            self.logger().warning("Order %s lookups conflict. Leaving the state unknown.", client_order_id)
            return None
        if not payloads:
            if kinds and all(kind == "not_found" for kind in kinds) and self._absence_is_allowed(client_order_id):
                self._cancel_requests.discard(client_order_id)
                # Safe: this is the only absence mark, and only after a structured
                # 2040 on an order that was never submitted, acked, or seen.
                self._confirmed_absent.add(client_order_id)
                return CONSTANTS.CONFIRMED_ABSENT
            return None
        data = self._prefer_order_payload(client_data, order_data)
        if data is None or not self._reads_allow_terminal(payloads, kinds, data):
            return None
        order = self._find_order(client_order_id)
        if order is None and data.get("externalOid"):
            order = self._find_order(str(data["externalOid"]))
        state = self._order_state_from_payload(data)
        if order is not None:
            self.record_order_observation(
                order.client_order_id,
                order.trading_pair,
                payload=data,
                new_state=state,
                exchange_order_id=str(data.get("orderId") or "") or None,
                apply_state=True,
            )
        if state in (OrderState.CANCELED, OrderState.FILLED):
            self._cancel_requests.discard(client_order_id)
        return state

    async def _get_last_traded_price(self, trading_pair: str) -> float:
        symbol = await self.exchange_symbol_associated_to_pair(trading_pair)
        response = await self._api_get(path_url=CONSTANTS.TICKER_PATH, params={"symbol": symbol})
        data = response["data"]
        if isinstance(data, list):
            data = data[0]
        return float(data["lastPrice"])

    async def set_margin_mode(self, mode: str):
        """
        Remember isolated or cross margin and send it as openType on later orders and leverage changes.
        MEXC applies the mode per order. Switching it while a position is open is rejected by the exchange.
        """
        if mode not in (MarginMode.CROSS, MarginMode.ISOLATED):
            raise ValueError(f"Unsupported margin mode {mode}")
        self._margin_mode = mode
        self.logger().info(f"Margin mode set to {mode}. New orders will use this open type.")

    async def _fetch_account_position_mode(self) -> Optional[PositionMode]:
        response = await self._api_get(path_url=CONSTANTS.POSITION_MODE_PATH, is_auth_required=True)
        mode = CONSTANTS.POSITION_MODE_FROM_EXCHANGE.get(int(response["data"]))
        if mode is None:
            self.logger().error(f"Unknown position mode value: {response['data']}")
        return mode

    async def _trading_pair_position_mode_set(self, mode: PositionMode, trading_pair: str) -> Tuple[bool, str]:
        del trading_pair
        try:
            await self._api_post(
                path_url=CONSTANTS.CHANGE_POSITION_MODE_PATH,
                data={"positionMode": CONSTANTS.POSITION_MODE_MAP[mode]},
                is_auth_required=True,
            )
        except Exception as exception:
            return False, str(exception)
        return True, ""

    async def _set_trading_pair_leverage(self, trading_pair: str, leverage: int) -> Tuple[bool, str]:
        symbol = await self.exchange_symbol_associated_to_pair(trading_pair)
        position_ids = [
            (side, position_id)
            for (pair, side), position_id in self._position_ids.items()
            if pair == trading_pair
        ]
        try:
            if position_ids:
                for _side, position_id in position_ids:
                    await self._api_post(
                        path_url=CONSTANTS.CHANGE_LEVERAGE_PATH,
                        data={
                            "positionId": position_id,
                            "leverage": int(leverage),
                            "openType": self._open_type(),
                        },
                        is_auth_required=True,
                    )
            else:
                position_types = [CONSTANTS.POSITION_TYPE_LONG]
                if self.position_mode is PositionMode.HEDGE:
                    position_types.append(CONSTANTS.POSITION_TYPE_SHORT)
                for position_type in position_types:
                    await self._api_post(
                        path_url=CONSTANTS.CHANGE_LEVERAGE_PATH,
                        data={
                            "symbol": symbol,
                            "leverage": int(leverage),
                            "openType": self._open_type(),
                            "positionType": position_type,
                        },
                        is_auth_required=True,
                    )
        except Exception as exception:
            return False, str(exception)
        return True, ""

    async def confirm_leverage_and_margin(self, trading_pair: str, leverage: int, margin_mode: str) -> bool:
        """
        Apply leverage and isolated or cross margin and return whether the exchange accepted them.
        Local state changes only after the exchange call succeeds.
        """
        if not self._position_mode_confirmed:
            self.logger().warning("Cannot confirm leverage before the account position mode is read.")
            return False
        mode = str(getattr(margin_mode, "value", margin_mode))
        if mode not in (MarginMode.CROSS, MarginMode.ISOLATED):
            self.logger().error(f"Unsupported margin mode {margin_mode}")
            return False
        previous_mode = self._margin_mode
        self._margin_mode = mode
        success, message = await self._set_trading_pair_leverage(trading_pair, int(leverage))
        if not success:
            self._margin_mode = previous_mode
            self.logger().warning(f"Exchange rejected leverage {leverage} {mode} for {trading_pair}: {message}")
            return False
        self._perpetual_trading.set_leverage(trading_pair, int(leverage))
        self.logger().info(f"Confirmed {mode} leverage {leverage} for {trading_pair}.")
        return True

    async def _initialize_position_mode(self):
        async with self._set_position_mode_lock:
            try:
                mode = await self._fetch_account_position_mode()
            except asyncio.CancelledError:
                raise
            except Exception:
                self._position_mode_confirmed = False
                self.logger().warning(
                    "Could not read the account position mode. Refusing to trade until it is read.",
                    exc_info=True,
                )
                return
            if mode is None:
                self._position_mode_confirmed = False
                self.logger().warning("The account position mode was empty. Refusing to trade.")
                return
            self._perpetual_trading.set_position_mode(mode)
            self._position_mode_confirmed = True
            self.logger().info(f"Position mode initialized to {mode} from the exchange.")

    async def _fetch_last_fee_payment(self, trading_pair: str) -> Tuple[float, Decimal, Decimal]:
        symbol = await self.exchange_symbol_associated_to_pair(trading_pair)
        response = await self._api_get(
            path_url=CONSTANTS.FUNDING_RECORDS_PATH,
            params={"symbol": symbol, "page_num": 1, "page_size": 20},
            is_auth_required=True,
        )
        records = self._funding_records(response)
        matching = [record for record in records if record.get("symbol") in (None, symbol)]
        if not matching:
            return 0, Decimal("-1"), Decimal("-1")
        latest = max(matching, key=lambda record: int(record.get("settleTime") or 0))
        timestamp = int(latest.get("settleTime") or 0)
        if timestamp > 10**12:
            timestamp = timestamp * 1e-3
        rate = Decimal(str(latest.get("rate") if latest.get("rate") is not None else latest.get("fundingRate") or "0"))
        payment = Decimal(str(latest.get("funding") or "0"))
        return float(timestamp), rate, payment

    async def _user_stream_event_listener(self):
        async for event_message in self._iter_user_event_queue():
            try:
                channel = event_message.get("channel")
                data = event_message.get("data")
                if channel == CONSTANTS.WS_ORDER_CHANNEL and isinstance(data, dict):
                    self._process_order_event_message(data)
                elif channel == CONSTANTS.WS_ORDER_DEAL_CHANNEL and isinstance(data, dict):
                    self._process_trade_event_message(data)
                elif channel == CONSTANTS.WS_POSITION_CHANNEL and data is not None:
                    await self._process_account_position_event(data)
                elif channel == CONSTANTS.WS_ASSET_CHANNEL and isinstance(data, dict):
                    self._process_wallet_event_message(data)
            except asyncio.CancelledError:
                raise
            except Exception:
                self.logger().exception("Unexpected error in user stream listener loop.")

    def _process_order_event_message(self, order_msg: Dict[str, Any]):
        client_order_id = str(order_msg.get("externalOid") or "")
        tracked_order = self._order_tracker.all_updatable_orders.get(client_order_id)
        if tracked_order is None or not client_order_id:
            return
        self.record_order_observation(
            client_order_id,
            tracked_order.trading_pair,
            payload=order_msg,
            exchange_order_id=str(order_msg.get("orderId") or ""),
            acked=order_msg.get("orderId") not in (None, ""),
        )

    def _process_trade_event_message(self, trade_msg: Dict[str, Any]):
        client_order_id = str(trade_msg.get("externalOid") or "")
        tracked_order = self._order_tracker.all_fillable_orders.get(client_order_id)
        if tracked_order is None or not client_order_id:
            return
        self.record_order_observation(
            client_order_id,
            tracked_order.trading_pair,
            trade_fill=trade_msg,
            exchange_order_id=str(trade_msg.get("orderId") or ""),
            acked=trade_msg.get("orderId") not in (None, ""),
            apply_state=False,
        )

    async def _process_account_position_event(self, position_msg: Any):
        entries = position_msg if isinstance(position_msg, list) else [position_msg]
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            parsed = await self._parse_position(entry)
            if parsed is None:
                symbol = entry.get("symbol")
                if symbol is None:
                    continue
                try:
                    trading_pair = await self.trading_pair_associated_to_exchange_symbol(symbol)
                except Exception:
                    continue
                side = self._position_side(entry.get("positionType"))
                self._perpetual_trading.remove_position(self._perpetual_trading.position_key(trading_pair, side))
                continue
            pos_key, position = parsed
            self._perpetual_trading.set_position(pos_key, position)

    def _process_wallet_event_message(self, wallet_msg: Dict[str, Any]):
        self._apply_balance_row(wallet_msg, from_rest=False)

    async def _make_trading_pairs_request(self) -> Any:
        response = await self._api_get(path_url=self.trading_pairs_request_path)
        data = response.get("data") or []
        if isinstance(data, dict):
            return [data]
        return data

    async def _make_trading_rules_request(self) -> Any:
        return await self._make_trading_pairs_request()

    def _initialize_trading_pair_symbols_from_exchange_info(self, exchange_info: List[Dict[str, Any]]):
        mapping = bidict()
        for contract in exchange_info:
            if not mexc_perpetual_utils.is_exchange_information_valid(contract):
                continue
            symbol = contract["symbol"]
            try:
                trading_pair = combine_to_hb_trading_pair(contract["baseCoin"], contract["quoteCoin"])
                mapping[symbol] = trading_pair
            except Exception:
                self.logger().exception(f"Could not parse trading pair from contract {symbol}.")
        self._set_trading_pair_symbol_map(mapping)

    async def _format_trading_rules(self, exchange_info_dict: List[Dict[str, Any]]) -> List[TradingRule]:
        trading_rules = []
        for contract in exchange_info_dict:
            if not mexc_perpetual_utils.is_exchange_information_valid(contract):
                continue
            try:
                trading_pair = await self.trading_pair_associated_to_exchange_symbol(symbol=contract["symbol"])
                contract_size = Decimal(str(contract["contractSize"]))
                self._contract_sizes[trading_pair] = contract_size
                min_vol = Decimal(str(contract["minVol"]))
                max_candidates = [Decimal(str(contract["maxVol"]))]
                if contract.get("limitMaxVol") not in (None, ""):
                    max_candidates.append(Decimal(str(contract["limitMaxVol"])))
                max_vol = min(max_candidates)
                vol_step = Decimal(str(contract["volUnit"]))
                collateral = contract.get("settleCoin") or "USDT"
                trading_rules.append(
                    TradingRule(
                        trading_pair=trading_pair,
                        min_order_size=min_vol * contract_size,
                        max_order_size=max_vol * contract_size,
                        min_price_increment=Decimal(str(contract["priceUnit"])),
                        min_base_amount_increment=vol_step * contract_size,
                        buy_order_collateral_token=collateral,
                        sell_order_collateral_token=collateral,
                    )
                )
            except Exception:
                self.logger().exception(f"Error parsing the trading pair rule: {contract}. Skipping.")
        return trading_rules

    async def _parse_position(self, raw_position: Dict[str, Any]) -> Optional[Tuple[str, Position]]:
        symbol = raw_position.get("symbol")
        if not symbol:
            return None
        try:
            trading_pair = await self.trading_pair_associated_to_exchange_symbol(symbol)
        except Exception:
            self.logger().debug(f"Skipping position for unknown symbol {symbol}.")
            return None
        contracts = Decimal(str(raw_position.get("holdVol") or "0"))
        state = int(raw_position.get("state") or 1)
        side = self._position_side(raw_position.get("positionType"))
        pos_key = self._perpetual_trading.position_key(trading_pair, side)
        position_id = raw_position.get("positionId")
        if position_id not in (None, "", 0, "0"):
            self._position_ids[(trading_pair, side)] = self._normalize_order_id(position_id)
        if contracts == s_decimal_0 or state == 3:
            self._position_ids.pop((trading_pair, side), None)
            return None
        amount = contracts * self.get_contract_size(trading_pair)
        if side is PositionSide.SHORT:
            amount *= Decimal("-1")
        unrealized = raw_position.get("unRealizedPnl", raw_position.get("pnl", raw_position.get("unrealized", "0")))
        entry_price = raw_position.get("holdAvgPrice", raw_position.get("openAvgPrice", "0"))
        position = Position(
            trading_pair=trading_pair,
            position_side=side,
            unrealized_pnl=Decimal(str(unrealized or "0")),
            entry_price=Decimal(str(entry_price or "0")),
            amount=amount,
            leverage=Decimal(str(raw_position.get("leverage") or "1")),
        )
        return pos_key, position

    def _parse_trade_update(self, trade_msg: Dict[str, Any], tracked_order: InFlightOrder) -> TradeUpdate:
        contract_size = self.get_contract_size(tracked_order.trading_pair)
        fill_price = Decimal(str(trade_msg["price"]))
        fill_base_amount = Decimal(str(trade_msg["vol"])) * contract_size
        fee_amount = Decimal(str(trade_msg.get("fee") or "0"))
        fee_asset = trade_msg.get("feeCurrency") or self.get_buy_collateral_token(tracked_order.trading_pair)
        flat_fees = [] if fee_amount == s_decimal_0 else [TokenAmount(amount=fee_amount, token=fee_asset)]
        position_action = getattr(tracked_order, "position", PositionAction.NIL)
        fee = TradeFeeBase.new_perpetual_fee(
            fee_schema=self.trade_fee_schema(),
            position_action=position_action,
            percent_token=fee_asset,
            flat_fees=flat_fees,
        )
        timestamp = trade_msg.get("timestamp") or trade_msg.get("t") or 0
        fill_timestamp = int(timestamp) * 1e-3 if int(timestamp) > 10**12 else float(timestamp or self.current_timestamp)
        return TradeUpdate(
            trade_id=str(trade_msg.get("id") or trade_msg.get("tradeId")),
            client_order_id=tracked_order.client_order_id,
            exchange_order_id=str(trade_msg.get("orderId") or tracked_order.exchange_order_id),
            trading_pair=tracked_order.trading_pair,
            fill_timestamp=fill_timestamp,
            fill_price=fill_price,
            fill_base_amount=fill_base_amount,
            fill_quote_amount=fill_price * fill_base_amount,
            fee=fee,
        )

    def _order_update_from_rest(self, order_data: Dict[str, Any], tracked_order: InFlightOrder) -> OrderUpdate:
        return OrderUpdate(
            trading_pair=tracked_order.trading_pair,
            update_timestamp=self.current_timestamp,
            new_state=self._order_state_from_payload(order_data),
            client_order_id=str(order_data.get("externalOid") or tracked_order.client_order_id),
            exchange_order_id=str(order_data.get("orderId") or tracked_order.exchange_order_id or ""),
        )

    @staticmethod
    def _payload_bucket(payload: Dict[str, Any]) -> str:
        state = int(payload.get("state") or 0)
        if state in (CONSTANTS.ORDER_STATE_PENDING, CONSTANTS.ORDER_STATE_OPEN):
            return "live"
        if state == CONSTANTS.ORDER_STATE_FILLED:
            return "filled"
        if state == CONSTANTS.ORDER_STATE_CANCELED:
            return "canceled"
        if state == CONSTANTS.ORDER_STATE_INVALID:
            return "invalid"
        return "other"

    @staticmethod
    def _payload_is_live(payload: Dict[str, Any]) -> bool:
        return MexcPerpetualDerivative._payload_bucket(payload) == "live"

    @classmethod
    def _payloads_conflict(cls, left: Optional[Dict[str, Any]], right: Optional[Dict[str, Any]]) -> bool:
        if not left or not right:
            return False
        return cls._payload_bucket(left) != cls._payload_bucket(right)

    @classmethod
    def _reads_allow_terminal(cls, payloads: List[Dict[str, Any]], kinds: List[str], chosen: Dict[str, Any]) -> bool:
        if any(cls._payload_is_live(item) for item in payloads):
            return cls._order_state_from_payload(chosen) not in (
                OrderState.CANCELED,
                OrderState.FAILED,
                OrderState.FILLED,
            )
        state = cls._order_state_from_payload(chosen)
        deal = Decimal(str(chosen.get("dealVol") or "0"))
        if state == OrderState.CANCELED and deal <= 0:
            if any(kind == "failed" for kind in kinds):
                return False
            return all(
                cls._order_state_from_payload(item) == OrderState.CANCELED
                and Decimal(str(item.get("dealVol") or "0")) <= 0
                for item in payloads
            )
        return True

    @staticmethod
    def _prefer_order_payload(left: Optional[Dict[str, Any]], right: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Prefer the order-id payload (right). A larger reported fill wins so a deal is not dropped.
        Disagreeing states are a conflict and stay unresolved.
        """
        if not left:
            return right
        if not right:
            return left
        if MexcPerpetualDerivative._payload_bucket(left) != MexcPerpetualDerivative._payload_bucket(right):
            return None
        left_deal = Decimal(str(left.get("dealVol") or "0"))
        right_deal = Decimal(str(right.get("dealVol") or "0"))
        if left_deal != right_deal:
            return right if right_deal > left_deal else left
        return right

    @staticmethod
    def _order_state_from_payload(order_data: Dict[str, Any]) -> OrderState:
        state = int(order_data.get("state") or CONSTANTS.ORDER_STATE_OPEN)
        deal_vol = Decimal(str(order_data.get("dealVol") or "0"))
        if state == CONSTANTS.ORDER_STATE_FILLED:
            return OrderState.FILLED
        if state == CONSTANTS.ORDER_STATE_CANCELED:
            return OrderState.CANCELED
        if state == CONSTANTS.ORDER_STATE_INVALID:
            return OrderState.FAILED
        if deal_vol > s_decimal_0:
            return OrderState.PARTIALLY_FILLED
        return OrderState.OPEN

    @staticmethod
    def _execution_type_code(order_type: OrderType, time_in_force: Optional[str]) -> int:
        normalized = (time_in_force or "").upper()
        if normalized == "FOK":
            return CONSTANTS.ORDER_TYPE_FOK
        if normalized == "IOC":
            return CONSTANTS.ORDER_TYPE_IOC
        if normalized == "POST_ONLY":
            return CONSTANTS.ORDER_TYPE_POST_ONLY
        return CONSTANTS.ORDER_TYPE_MAP[order_type]

    def _position_mode_code(self, position_mode: Any) -> int:
        if position_mode is None:
            return CONSTANTS.POSITION_MODE_MAP[self.position_mode]
        if isinstance(position_mode, PositionMode):
            return CONSTANTS.POSITION_MODE_MAP[position_mode]
        return int(position_mode)

    @staticmethod
    def _attach_stop_prices(data: Dict[str, Any], kwargs: Dict[str, Any]):
        stop_loss = kwargs.get("stop_loss_price")
        take_profit = kwargs.get("take_profit_price")
        if stop_loss is not None:
            data["stopLossPrice"] = MexcPerpetualDerivative._json_number(stop_loss)
            data["lossTrend"] = int(kwargs.get("loss_trend") or CONSTANTS.TRIGGER_PRICE_LATEST)
        if take_profit is not None:
            data["takeProfitPrice"] = MexcPerpetualDerivative._json_number(take_profit)
            data["profitTrend"] = int(kwargs.get("profit_trend") or CONSTANTS.TRIGGER_PRICE_LATEST)

    @staticmethod
    def _put_optional_price(data: Dict[str, Any], field: str, price: Optional[Decimal]):
        if price is not None:
            data[field] = MexcPerpetualDerivative._json_number(price)

    @staticmethod
    def _put_stop_change_fields(
        data: Dict[str, Any],
        stop_loss_price: Optional[Decimal],
        take_profit_price: Optional[Decimal],
        loss_trend: Optional[int],
        profit_trend: Optional[int],
        kwargs: Dict[str, Any],
    ):
        MexcPerpetualDerivative._put_optional_price(data, "stopLossPrice", stop_loss_price)
        MexcPerpetualDerivative._put_optional_price(data, "takeProfitPrice", take_profit_price)
        if loss_trend is not None:
            data["lossTrend"] = int(loss_trend)
        if profit_trend is not None:
            data["profitTrend"] = int(profit_trend)
        if kwargs.get("take_profit_reverse") is not None:
            data["takeProfitReverse"] = int(kwargs["take_profit_reverse"])
        if kwargs.get("stop_loss_reverse") is not None:
            data["stopLossReverse"] = int(kwargs["stop_loss_reverse"])

    def _exchange_side(self, trade_type: TradeType, position_action: PositionAction) -> int:
        if trade_type is TradeType.BUY and position_action is PositionAction.OPEN:
            return CONSTANTS.SIDE_OPEN_LONG
        if trade_type is TradeType.BUY and position_action is PositionAction.CLOSE:
            return CONSTANTS.SIDE_CLOSE_SHORT
        if trade_type is TradeType.SELL and position_action is PositionAction.OPEN:
            return CONSTANTS.SIDE_OPEN_SHORT
        if trade_type is TradeType.SELL and position_action is PositionAction.CLOSE:
            return CONSTANTS.SIDE_CLOSE_LONG
        raise ValueError(f"Unsupported order side {trade_type} {position_action}")

    def _open_type(self) -> int:
        if self._margin_mode == MarginMode.ISOLATED:
            return CONSTANTS.OPEN_TYPE_ISOLATED
        return CONSTANTS.OPEN_TYPE_CROSS

    def _base_amount_to_contracts(self, trading_pair: str, amount: Decimal) -> Decimal:
        contract_size = self.get_contract_size(trading_pair)
        contracts = amount / contract_size
        trading_rule = self._trading_rules.get(trading_pair)
        if trading_rule is not None and trading_rule.min_base_amount_increment > s_decimal_0:
            step = trading_rule.min_base_amount_increment / contract_size
            if step > s_decimal_0:
                contracts = (contracts / step).to_integral_value(rounding=ROUND_DOWN) * step
        return contracts

    def _price_string(self, price: Decimal, order_type: OrderType) -> str:
        if order_type is OrderType.MARKET and (price is None or price.is_nan()):
            return "0"
        return self._decimal_str(price)

    @staticmethod
    def _json_number(value: Any):
        """
        JSON number, matching the futures docs examples (vol 5, price 109777.5).
        Whole values stay integers so 10 is not encoded as 10.0.
        """
        number = Decimal(str(value))
        if number == number.to_integral_value():
            return int(number)
        return float(number)

    @staticmethod
    def _decimal_str(value: Decimal) -> str:
        text = format(value, "f")
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text or "0"

    @staticmethod
    def _normalize_order_id(order_id: str):
        text = str(order_id)
        return int(text) if text.isdigit() else text

    @staticmethod
    def _position_side(position_type: Any) -> PositionSide:
        return PositionSide.LONG if int(position_type or 1) == CONSTANTS.POSITION_TYPE_LONG else PositionSide.SHORT

    @staticmethod
    def parse_funding_response(response: Dict[str, Any]) -> Dict[str, Any]:
        """
        The field table describes resultList rows. The only sample body on that page is a
        funding-rate object. Both shapes are recognized. A rate snapshot is not a payment.
        """
        data = response.get("data") if isinstance(response, dict) else None
        records: List[Dict[str, Any]] = []
        if isinstance(data, dict):
            listed = data.get("resultList")
            if listed is None:
                listed = data.get("result")
            if isinstance(listed, list):
                records = [item for item in listed if isinstance(item, dict)]
            elif MexcPerpetualDerivative._is_funding_payment(data):
                records = [data]
            elif data.get("fundingRate") is not None or data.get("nextSettleTime") is not None:
                rate = data.get("fundingRate", data.get("rate"))
                return {
                    "kind": "rate_snapshot",
                    "records": [],
                    "payment": None,
                    "rate": Decimal(str(rate)) if rate is not None else None,
                    "next_settle_time": data.get("nextSettleTime"),
                    "symbol": data.get("symbol"),
                }
        elif isinstance(data, list):
            records = [item for item in data if isinstance(item, dict)]
        return {"kind": "records", "records": records, "payment": None, "rate": None}

    @staticmethod
    def _is_funding_payment(data: Dict[str, Any]) -> bool:
        return any(key in data for key in ("funding", "fundingFee", "settleTime"))

    @staticmethod
    def _funding_records(response: Dict[str, Any]) -> List[Dict[str, Any]]:
        return MexcPerpetualDerivative.parse_funding_response(response)["records"]
