import asyncio
from decimal import Decimal, ROUND_DOWN
from typing import Any, Dict, List, Optional, Tuple

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
        # Margin mode is sent as openType on each order. MEXC has no separate account-wide switch.
        self._margin_mode = MarginMode.CROSS
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
        return self._contract_sizes.get(trading_pair, Decimal("1"))

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
        if isinstance(status_update_exception, ValueError):
            return True
        return self._exception_contains_code(status_update_exception, CONSTANTS.ORDER_NOT_FOUND_ERROR_CODES)

    def _is_order_not_found_during_cancelation_error(self, cancelation_exception: Exception) -> bool:
        return self._exception_contains_code(cancelation_exception, CONSTANTS.ORDER_NOT_FOUND_ERROR_CODES)

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
            raise IOError(f"Error: {code} - {message}")
        return response

    async def _api_request(self, *args, **kwargs) -> Dict[str, Any]:
        response = await super()._api_request(*args, **kwargs)
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
            error_code = int(item.get("errorCode") or 0)
            if error_code != 0:
                raise IOError(f"Error: {error_code} - {item.get('errorMsg', '')}")
        return True

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
        symbol = await self.exchange_symbol_associated_to_pair(trading_pair)
        data = {
            "symbol": symbol,
            "price": self._price_string(price, order_type),
            "vol": self._decimal_str(self._base_amount_to_contracts(trading_pair, amount)),
            "leverage": int(self.get_leverage(trading_pair)),
            "side": self._exchange_side(trade_type, position_action),
            "type": CONSTANTS.ORDER_TYPE_MAP[order_type],
            "openType": self._open_type(),
            "externalOid": order_id,
            "positionMode": CONSTANTS.POSITION_MODE_MAP[self.position_mode],
        }
        if self.position_mode is PositionMode.ONEWAY and position_action is PositionAction.CLOSE:
            data["reduceOnly"] = True
        try:
            response = await self._api_post(
                path_url=CONSTANTS.PLACE_ORDER_PATH,
                data=data,
                is_auth_required=True,
            )
        except Exception:
            recovered = await self._recover_order_by_client_id(trading_pair, order_id)
            if recovered is not None:
                return recovered
            raise
        return str(response["data"]["orderId"]), self.current_timestamp

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

    async def _update_balances(self):
        response = await self._api_get(path_url=CONSTANTS.ASSETS_PATH, is_auth_required=True)
        self._account_available_balances.clear()
        self._account_balances.clear()
        for balance_data in response.get("data") or []:
            asset = balance_data["currency"]
            available = Decimal(str(balance_data.get("availableBalance") or "0"))
            if balance_data.get("equity") is not None:
                total = Decimal(str(balance_data["equity"]))
            else:
                frozen = Decimal(str(balance_data.get("frozenBalance") or "0"))
                margin = Decimal(str(balance_data.get("positionMargin") or "0"))
                total = available + frozen + margin
            self._account_available_balances[asset] = available
            self._account_balances[asset] = total

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
        position_types = [CONSTANTS.POSITION_TYPE_LONG]
        if self.position_mode is PositionMode.HEDGE:
            position_types.append(CONSTANTS.POSITION_TYPE_SHORT)
        try:
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
        if tracked_order is None:
            return
        order_update = OrderUpdate(
            trading_pair=tracked_order.trading_pair,
            update_timestamp=self.current_timestamp,
            new_state=self._order_state_from_payload(order_msg),
            client_order_id=client_order_id,
            exchange_order_id=str(order_msg.get("orderId") or tracked_order.exchange_order_id or ""),
        )
        self._order_tracker.process_order_update(order_update)

    def _process_trade_event_message(self, trade_msg: Dict[str, Any]):
        client_order_id = str(trade_msg.get("externalOid") or "")
        tracked_order = self._order_tracker.all_fillable_orders.get(client_order_id)
        if tracked_order is None:
            return
        self._order_tracker.process_trade_update(self._parse_trade_update(trade_msg, tracked_order))

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
        asset = wallet_msg.get("currency")
        if not asset:
            return
        available = Decimal(str(wallet_msg.get("availableBalance") or "0"))
        frozen = Decimal(str(wallet_msg.get("frozenBalance") or "0"))
        margin = Decimal(str(wallet_msg.get("positionMargin") or "0"))
        self._account_available_balances[asset] = available
        self._account_balances[asset] = available + frozen + margin

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
        if contracts == s_decimal_0 or state == 3:
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
        fee_amount = abs(Decimal(str(trade_msg.get("fee") or "0")))
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
        if contract_size <= s_decimal_0:
            contract_size = Decimal("1")
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
    def _funding_records(response: Dict[str, Any]) -> List[Dict[str, Any]]:
        data = response.get("data")
        if isinstance(data, dict):
            records = data.get("resultList") or data.get("result") or []
        elif isinstance(data, list):
            records = data
        else:
            records = []
        return records
