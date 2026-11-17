import asyncio
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from hummingbot.connector.derivative.mexc_perpetual import (
    mexc_perpetual_constants as CONSTANTS,
    mexc_perpetual_web_utils as web_utils,
)
from hummingbot.core.data_type.common import TradeType
from hummingbot.core.data_type.funding_info import FundingInfo, FundingInfoUpdate
from hummingbot.core.data_type.order_book_message import OrderBookMessage, OrderBookMessageType
from hummingbot.core.data_type.perpetual_api_order_book_data_source import PerpetualAPIOrderBookDataSource
from hummingbot.core.web_assistant.connections.data_types import RESTMethod, WSJSONRequest
from hummingbot.core.web_assistant.web_assistants_factory import WebAssistantsFactory
from hummingbot.core.web_assistant.ws_assistant import WSAssistant

if TYPE_CHECKING:
    from hummingbot.connector.derivative.mexc_perpetual.mexc_perpetual_derivative import MexcPerpetualDerivative


class MexcPerpetualAPIOrderBookDataSource(PerpetualAPIOrderBookDataSource):
    """
    Public REST snapshots and the incremental websocket depth, trade, and funding streams.

    Depth quantities are contracts. They are converted to base-asset size with the contract size
    before they enter the local order book. A quantity of 0 deletes that price level.
    """

    def __init__(
        self,
        trading_pairs: List[str],
        connector: "MexcPerpetualDerivative",
        api_factory: WebAssistantsFactory,
    ):
        super().__init__(trading_pairs)
        self._connector = connector
        self._api_factory = api_factory
        self._ping_task: Optional[asyncio.Task] = None
        self._last_depth_version: Dict[str, int] = {}
        self._snapshot_output: Optional[asyncio.Queue] = None

    async def get_last_traded_prices(self, trading_pairs: List[str], domain: Optional[str] = None) -> Dict[str, float]:
        return await self._connector.get_last_traded_prices(trading_pairs=trading_pairs)

    def _channel_originating_message(self, event_message: Dict[str, Any]) -> Optional[str]:
        if not isinstance(event_message, dict):
            return None
        channel = event_message.get("channel")
        if channel == CONSTANTS.WS_DEPTH_CHANNEL:
            return self._diff_messages_queue_key
        if channel == CONSTANTS.WS_DEAL_CHANNEL:
            return self._trade_messages_queue_key
        if channel == CONSTANTS.WS_FUNDING_CHANNEL:
            return self._funding_info_messages_queue_key
        return None

    async def _process_message_for_unknown_channel(
        self,
        event_message: Dict[str, Any],
        websocket_assistant: WSAssistant,
    ):
        if isinstance(event_message, dict) and event_message.get("channel") == CONSTANTS.WS_PONG_CHANNEL:
            self.logger().debug("Received public websocket pong.")
            return
        if isinstance(event_message, dict) and event_message.get("channel") == CONSTANTS.WS_ERROR_CHANNEL:
            raise IOError(f"Public websocket error: {event_message}")

    async def get_funding_info(self, trading_pair: str) -> FundingInfo:
        symbol = await self._connector.exchange_symbol_associated_to_pair(trading_pair)
        payload = await self._public_get(
            path_url=CONSTANTS.FUNDING_RATE_PATH.format(symbol=symbol),
            limit_id=CONSTANTS.FUNDING_RATE_LIMIT_ID,
        )
        data = payload["data"]
        return FundingInfo(
            trading_pair=trading_pair,
            index_price=Decimal(str(data["idxPrice"])),
            mark_price=Decimal(str(data["fairPrice"])),
            next_funding_utc_timestamp=int(int(data["nextSettleTime"]) * 1e-3),
            rate=Decimal(str(data["fundingRate"])),
        )

    async def _parse_funding_info_message(self, raw_message: Dict[str, Any], message_queue: asyncio.Queue):
        data = raw_message.get("data") or {}
        symbol = data.get("symbol") or raw_message.get("symbol")
        trading_pair = await self._connector.trading_pair_associated_to_exchange_symbol(symbol)
        rate = data.get("rate", data.get("fundingRate"))
        next_settle = data.get("nextSettleTime")
        update = FundingInfoUpdate(
            trading_pair=trading_pair,
            rate=Decimal(str(rate)) if rate is not None else None,
            next_funding_utc_timestamp=int(int(next_settle) * 1e-3) if next_settle is not None else None,
        )
        message_queue.put_nowait(update)

    async def _connected_websocket_assistant(self) -> WSAssistant:
        websocket_assistant = await self._api_factory.get_ws_assistant()
        await websocket_assistant.connect(
            ws_url=web_utils.wss_url(),
            message_timeout=CONSTANTS.SECONDS_TO_WAIT_TO_RECEIVE_MESSAGE,
        )
        return websocket_assistant

    async def _subscribe_channels(self, ws: WSAssistant):
        try:
            for trading_pair in self._trading_pairs:
                await self._send_pair_subscription(ws, trading_pair, subscribe=True)
            self.logger().info("Subscribed to public depth, trade, and funding channels...")
        except asyncio.CancelledError:
            raise
        except Exception:
            self.logger().exception("Unexpected error occurred subscribing to public channels...")
            raise

    async def _send_pair_subscription(self, ws: WSAssistant, trading_pair: str, subscribe: bool):
        symbol = await self._connector.exchange_symbol_associated_to_pair(trading_pair)
        prefix = "sub" if subscribe else "unsub"
        messages = [
            {"method": f"{prefix}.depth", "param": {"symbol": symbol, "compress": False}},
            {"method": f"{prefix}.deal", "param": {"symbol": symbol, "compress": False, "reduce": False}},
            {"method": f"{prefix}.funding.rate", "param": {"symbol": symbol}},
        ]
        for message in messages:
            await ws.send(WSJSONRequest(message))

    async def subscribe_to_trading_pair(self, trading_pair: str) -> bool:
        if self._ws_assistant is None:
            self.logger().warning(f"Cannot subscribe to {trading_pair}: WebSocket connection not established.")
            return False
        try:
            await self._send_pair_subscription(self._ws_assistant, trading_pair, subscribe=True)
            self.add_trading_pair(trading_pair)
            return True
        except asyncio.CancelledError:
            raise
        except Exception:
            self.logger().exception(f"Error subscribing to {trading_pair}")
            return False

    async def unsubscribe_from_trading_pair(self, trading_pair: str) -> bool:
        if self._ws_assistant is None:
            self.logger().warning(f"Cannot unsubscribe from {trading_pair}: WebSocket connection not established.")
            return False
        try:
            await self._send_pair_subscription(self._ws_assistant, trading_pair, subscribe=False)
            self.remove_trading_pair(trading_pair)
            return True
        except asyncio.CancelledError:
            raise
        except Exception:
            self.logger().exception(f"Error unsubscribing from {trading_pair}")
            return False

    async def _request_order_book_snapshot(self, trading_pair: str) -> Dict[str, Any]:
        symbol = await self._connector.exchange_symbol_associated_to_pair(trading_pair)
        return await self._public_get(
            path_url=CONSTANTS.DEPTH_PATH.format(symbol=symbol),
            limit_id=CONSTANTS.DEPTH_LIMIT_ID,
            params={"limit": CONSTANTS.DEPTH_SNAPSHOT_LIMIT},
        )

    async def _order_book_snapshot(self, trading_pair: str) -> OrderBookMessage:
        snapshot = await self._request_order_book_snapshot(trading_pair)
        data = snapshot["data"]
        symbol = await self._connector.exchange_symbol_associated_to_pair(trading_pair)
        update_id = int(data["version"])
        self._last_depth_version[symbol] = update_id
        timestamp = int(data.get("timestamp") or 0) * 1e-3
        contract_size = self._connector.get_contract_size(trading_pair)
        return OrderBookMessage(
            OrderBookMessageType.SNAPSHOT,
            {
                "trading_pair": trading_pair,
                "update_id": update_id,
                "bids": self._depth_rows(data.get("bids") or [], contract_size),
                "asks": self._depth_rows(data.get("asks") or [], contract_size),
            },
            timestamp,
        )

    async def _parse_order_book_diff_message(self, raw_message: Dict[str, Any], message_queue: asyncio.Queue):
        data = raw_message.get("data") or {}
        symbol = raw_message.get("symbol") or data.get("symbol")
        trading_pair = await self._connector.trading_pair_associated_to_exchange_symbol(symbol)
        begin = self._optional_int(data.get("begin"))
        end = self._optional_int(data.get("end", data.get("version")))
        last_version = self._last_depth_version.get(symbol)
        if last_version is not None and begin is not None and begin > last_version + 1:
            self.logger().warning(
                f"Order book depth gap for {symbol}: local version {last_version}, update begins at {begin}."
            )
            self._last_depth_version.pop(symbol, None)
            await self._refresh_snapshot(trading_pair)
            return
        if end is not None:
            self._last_depth_version[symbol] = end
        contract_size = self._connector.get_contract_size(trading_pair)
        timestamp = int(raw_message.get("ts") or data.get("ts") or 0) * 1e-3
        message_queue.put_nowait(
            OrderBookMessage(
                OrderBookMessageType.DIFF,
                {
                    "trading_pair": trading_pair,
                    "update_id": end or 0,
                    "bids": self._depth_rows(data.get("bids") or [], contract_size),
                    "asks": self._depth_rows(data.get("asks") or [], contract_size),
                },
                timestamp,
            )
        )

    async def _parse_order_book_snapshot_message(self, raw_message: Dict[str, Any], message_queue: asyncio.Queue):
        # Public depth pushes are incremental. REST snapshots are built in _order_book_snapshot.
        await self._parse_order_book_diff_message(raw_message, message_queue)

    async def _parse_trade_message(self, raw_message: Dict[str, Any], message_queue: asyncio.Queue):
        symbol = raw_message.get("symbol")
        trading_pair = await self._connector.trading_pair_associated_to_exchange_symbol(symbol)
        contract_size = self._connector.get_contract_size(trading_pair)
        trades = raw_message.get("data") or []
        if isinstance(trades, dict):
            trades = [trades]
        for trade_data in trades:
            trade_type = float(TradeType.BUY.value) if int(trade_data.get("T", 0)) == 1 else float(TradeType.SELL.value)
            trade_id = trade_data.get("i") or trade_data.get("t")
            amount = Decimal(str(trade_data.get("v", "0"))) * contract_size
            message_queue.put_nowait(
                OrderBookMessage(
                    OrderBookMessageType.TRADE,
                    {
                        "trade_id": int(trade_id),
                        "trading_pair": trading_pair,
                        "trade_type": trade_type,
                        "amount": self._decimal_str(amount),
                        "price": str(trade_data.get("p")),
                    },
                    int(trade_data.get("t") or raw_message.get("ts") or 0) * 1e-3,
                )
            )

    async def listen_for_order_book_snapshots(self, ev_loop: asyncio.AbstractEventLoop, output: asyncio.Queue):
        self._snapshot_output = output
        await super().listen_for_order_book_snapshots(ev_loop, output)

    async def listen_for_subscriptions(self):
        ws: Optional[WSAssistant] = None
        while True:
            try:
                ws = await self._connected_websocket_assistant()
                self._ws_assistant = ws
                await self._subscribe_channels(ws)
                self._ping_task = asyncio.create_task(self._send_interval_ping(ws))
                await self._process_websocket_messages(websocket_assistant=ws)
            except asyncio.CancelledError:
                raise
            except ConnectionError as connection_exception:
                self.logger().warning(f"The websocket connection was closed ({connection_exception})")
            except Exception:
                self.logger().exception(
                    "Unexpected error occurred when listening to order book streams. Retrying in 5 seconds..."
                )
                await self._sleep(5.0)
            finally:
                await self._cancel_ping_task()
                self._last_depth_version.clear()
                self._ws_assistant = None
                await self._on_order_stream_interruption(websocket_assistant=ws)

    async def _send_interval_ping(self, websocket_assistant: WSAssistant):
        try:
            while True:
                await websocket_assistant.send(WSJSONRequest({"method": CONSTANTS.WS_PING_METHOD}))
                await asyncio.sleep(CONSTANTS.WS_HEARTBEAT_TIME_INTERVAL)
        except asyncio.CancelledError:
            raise
        except Exception:
            self.logger().exception("Error sending interval ping")

    async def _refresh_snapshot(self, trading_pair: str):
        if self._snapshot_output is None:
            return
        snapshot = await self._order_book_snapshot(trading_pair)
        self._snapshot_output.put_nowait(snapshot)

    async def _public_get(
        self,
        path_url: str,
        limit_id: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        rest_assistant = await self._api_factory.get_rest_assistant()
        response = await rest_assistant.execute_request(
            url=web_utils.public_rest_url(path_url),
            throttler_limit_id=limit_id,
            params=params,
            method=RESTMethod.GET,
        )
        return self._connector.ensure_successful_response(response)

    async def _cancel_ping_task(self):
        if self._ping_task is not None:
            self._ping_task.cancel()
            try:
                await self._ping_task
            except asyncio.CancelledError:
                pass
            self._ping_task = None

    @staticmethod
    def _depth_rows(levels: List[Any], contract_size: Decimal) -> List[List[str]]:
        rows = []
        for level in levels:
            price = level[0]
            contracts = Decimal(str(level[1]))
            amount = Decimal("0") if contracts == 0 else contracts * contract_size
            rows.append([str(price), MexcPerpetualAPIOrderBookDataSource._decimal_str(amount)])
        return rows

    @staticmethod
    def _decimal_str(value: Decimal) -> str:
        text = format(value, "f")
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text or "0"

    @staticmethod
    def _optional_int(value: Any) -> Optional[int]:
        if value is None or value == "":
            return None
        return int(value)
