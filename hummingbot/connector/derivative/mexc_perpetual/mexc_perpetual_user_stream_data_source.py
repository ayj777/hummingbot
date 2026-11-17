import asyncio
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from hummingbot.connector.derivative.mexc_perpetual import (
    mexc_perpetual_constants as CONSTANTS,
    mexc_perpetual_web_utils as web_utils,
)
from hummingbot.connector.derivative.mexc_perpetual.mexc_perpetual_auth import MexcPerpetualAuth
from hummingbot.core.data_type.user_stream_tracker_data_source import UserStreamTrackerDataSource
from hummingbot.core.web_assistant.connections.data_types import WSJSONRequest
from hummingbot.core.web_assistant.web_assistants_factory import WebAssistantsFactory
from hummingbot.core.web_assistant.ws_assistant import WSAssistant

if TYPE_CHECKING:
    from hummingbot.connector.derivative.mexc_perpetual.mexc_perpetual_derivative import MexcPerpetualDerivative


class MexcPerpetualUserStreamDataSource(UserStreamTrackerDataSource):
    """
    Private websocket. Login uses the futures open-api signature, and the server then pushes
    orders, fills, positions, and assets without a separate subscribe call.
    """

    def __init__(
        self,
        auth: MexcPerpetualAuth,
        trading_pairs: List[str],
        connector: "MexcPerpetualDerivative",
        api_factory: WebAssistantsFactory,
    ):
        super().__init__()
        self._auth = auth
        self._trading_pairs = trading_pairs
        self._connector = connector
        self._api_factory = api_factory
        self._ping_task: Optional[asyncio.Task] = None

    async def _connected_websocket_assistant(self) -> WSAssistant:
        websocket_assistant = await self._api_factory.get_ws_assistant()
        await websocket_assistant.connect(
            ws_url=web_utils.wss_url(),
            message_timeout=CONSTANTS.SECONDS_TO_WAIT_TO_RECEIVE_MESSAGE,
        )
        await self._authenticate(websocket_assistant)
        return websocket_assistant

    async def _authenticate(self, websocket_assistant: WSAssistant):
        login_payload = self._auth.get_ws_login_payload()
        await websocket_assistant.send(WSJSONRequest(login_payload))
        response = await websocket_assistant.receive()
        message = response.data if response is not None else None
        if not isinstance(message, dict) or message.get("channel") != CONSTANTS.WS_LOGIN_CHANNEL or message.get("data") != "success":
            self.logger().error(f"Private websocket login failed: {message}")
            raise IOError(f"Private websocket login failed: {message}")
        self.logger().info("Private websocket login succeeded.")

    async def _subscribe_channels(self, websocket_assistant: WSAssistant):
        # Login with the default subscribe flag pushes order, order.deal, position, and asset.
        self.logger().info("Personal order, fill, position, and asset channels follow a successful login.")

    async def _process_event_message(self, event_message: Dict[str, Any], queue: asyncio.Queue):
        if not isinstance(event_message, dict):
            return
        channel = event_message.get("channel")
        if channel == CONSTANTS.WS_PONG_CHANNEL:
            self.logger().debug("Received private websocket pong.")
            return
        if channel == CONSTANTS.WS_ERROR_CHANNEL:
            self.logger().error(f"Private websocket error: {event_message}")
            return
        if channel in {
            CONSTANTS.WS_ORDER_CHANNEL,
            CONSTANTS.WS_ORDER_DEAL_CHANNEL,
            CONSTANTS.WS_POSITION_CHANNEL,
            CONSTANTS.WS_ASSET_CHANNEL,
        }:
            queue.put_nowait(event_message)

    async def listen_for_user_stream(self, output: asyncio.Queue):
        while True:
            try:
                self._ws_assistant = await self._connected_websocket_assistant()
                await self._subscribe_channels(websocket_assistant=self._ws_assistant)
                self._ping_task = asyncio.create_task(self._send_interval_ping(self._ws_assistant))
                await self._process_websocket_messages(websocket_assistant=self._ws_assistant, queue=output)
            except asyncio.CancelledError:
                raise
            except ConnectionError as connection_exception:
                self.logger().warning(f"The websocket connection was closed ({connection_exception})")
            except Exception:
                self.logger().exception("Unexpected error while listening to user stream. Retrying after 5 seconds...")
                await self._sleep(5.0)
            finally:
                await self._cancel_ping_task()
                await self._on_user_stream_interruption(websocket_assistant=self._ws_assistant)
                self._ws_assistant = None

    async def _send_interval_ping(self, websocket_assistant: WSAssistant):
        try:
            while True:
                await websocket_assistant.send(WSJSONRequest({"method": CONSTANTS.WS_PING_METHOD}))
                await asyncio.sleep(CONSTANTS.WS_HEARTBEAT_TIME_INTERVAL)
        except asyncio.CancelledError:
            raise
        except Exception:
            self.logger().exception("Error sending interval ping")

    async def _cancel_ping_task(self):
        if self._ping_task is not None:
            self._ping_task.cancel()
            try:
                await self._ping_task
            except asyncio.CancelledError:
                pass
            self._ping_task = None
