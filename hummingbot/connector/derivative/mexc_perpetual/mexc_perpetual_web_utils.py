import gzip
import json
from typing import Callable, Optional, Union

from hummingbot.connector.derivative.mexc_perpetual import mexc_perpetual_constants as CONSTANTS
from hummingbot.connector.time_synchronizer import TimeSynchronizer
from hummingbot.connector.utils import TimeSynchronizerRESTPreProcessor
from hummingbot.core.api_throttler.async_throttler import AsyncThrottler
from hummingbot.core.web_assistant.auth import AuthBase
from hummingbot.core.web_assistant.connections.data_types import RESTMethod, WSResponse
from hummingbot.core.web_assistant.web_assistants_factory import WebAssistantsFactory
from hummingbot.core.web_assistant.ws_post_processors import WSPostProcessorBase


def public_rest_url(path_url: str, domain: str = "mexc.com") -> str:
    # Futures REST is a single host. The domain argument matches the connector base class.
    return CONSTANTS.REST_URL + path_url


def private_rest_url(path_url: str, domain: str = "mexc.com") -> str:
    return CONSTANTS.REST_URL + path_url


def wss_url() -> str:
    return CONSTANTS.WSS_URL


def build_api_factory(
    throttler: Optional[AsyncThrottler] = None,
    time_synchronizer: Optional[TimeSynchronizer] = None,
    time_provider: Optional[Callable] = None,
    auth: Optional[AuthBase] = None,
) -> WebAssistantsFactory:
    throttler = throttler or create_throttler()
    time_synchronizer = time_synchronizer or TimeSynchronizer()
    time_provider = time_provider or (lambda: get_current_server_time(throttler=throttler))
    return WebAssistantsFactory(
        throttler=throttler,
        auth=auth,
        rest_pre_processors=[
            TimeSynchronizerRESTPreProcessor(synchronizer=time_synchronizer, time_provider=time_provider),
        ],
        ws_post_processors=[MexcPerpetualWSPostProcessor()],
    )


def build_api_factory_without_time_synchronizer_pre_processor(throttler: AsyncThrottler) -> WebAssistantsFactory:
    return WebAssistantsFactory(throttler=throttler, ws_post_processors=[MexcPerpetualWSPostProcessor()])


def create_throttler() -> AsyncThrottler:
    return AsyncThrottler(CONSTANTS.RATE_LIMITS)


async def get_current_server_time(
    throttler: Optional[AsyncThrottler] = None,
    domain: str = "mexc.com",
) -> float:
    """
    Server time in milliseconds. The ping payload is the timestamp itself.
    """
    throttler = throttler or create_throttler()
    api_factory = build_api_factory_without_time_synchronizer_pre_processor(throttler=throttler)
    rest_assistant = await api_factory.get_rest_assistant()
    response = await rest_assistant.execute_request(
        url=public_rest_url(CONSTANTS.PING_PATH),
        throttler_limit_id=CONSTANTS.PING_PATH,
        method=RESTMethod.GET,
        return_err=True,
    )
    return float(response["data"])


class MexcPerpetualWSPostProcessor(WSPostProcessorBase):
    """
    Depth and deal streams are gzip-compressed unless the subscription sets compress to false.
    Decode either form so the rest of the connector always sees a dictionary.
    """

    async def post_process(self, response: WSResponse) -> WSResponse:
        response.data = decode_ws_message(response.data)
        return response


def decode_ws_message(data: Union[bytes, bytearray, str, dict, None]):
    if data is None or isinstance(data, dict):
        return data
    raw = data
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = gzip.decompress(raw)
        except OSError:
            pass
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw
    return raw
