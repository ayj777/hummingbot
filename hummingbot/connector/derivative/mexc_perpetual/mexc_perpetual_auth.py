import hashlib
import hmac
from typing import Any, Dict, Mapping, Optional
from urllib.parse import quote, urlencode

from hummingbot.connector.derivative.mexc_perpetual import mexc_perpetual_constants as CONSTANTS
from hummingbot.connector.time_synchronizer import TimeSynchronizer
from hummingbot.core.web_assistant.auth import AuthBase
from hummingbot.core.web_assistant.connections.data_types import RESTMethod, RESTRequest, WSRequest


class MexcPerpetualAuth(AuthBase):
    """
    REST and websocket login signatures for the MEXC futures API.

    Signature = HMAC_SHA256(secret, accessKey + timestamp + paramString).
    paramString is the sorted query string for GET and DELETE, and the raw JSON body for POST.
    Websocket login uses an empty paramString, so the signed payload is accessKey + reqTime.
    """

    def __init__(self, api_key: str, secret_key: str, time_provider: TimeSynchronizer):
        self._api_key = api_key
        self._secret_key = secret_key
        self._time_provider = time_provider

    def header_timestamp(self) -> str:
        return str(int(self._time_provider.time() * 1e3))

    def _sign(self, timestamp: str, param_string: str) -> str:
        payload = f"{self._api_key}{timestamp}{param_string}"
        return hmac.new(
            self._secret_key.encode("utf-8"),
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    @staticmethod
    def param_string_for_query(params: Optional[Mapping[str, Any]]) -> str:
        if not params:
            return ""
        items = []
        for key in sorted(params):
            value = params[key]
            if value is None:
                continue
            if isinstance(value, bool):
                rendered = "true" if value else "false"
            else:
                rendered = str(value)
            items.append((str(key), rendered))
        return urlencode(items, quote_via=quote, safe="")

    def rest_headers(self, timestamp: str, signature: str) -> Dict[str, str]:
        return {
            "ApiKey": self._api_key,
            "Request-Time": timestamp,
            "Signature": signature,
            "Recv-Window": CONSTANTS.RECV_WINDOW,
            "Language": "English",
        }

    async def rest_authenticate(self, request: RESTRequest) -> RESTRequest:
        timestamp = self.header_timestamp()
        if request.method in (RESTMethod.GET, RESTMethod.DELETE):
            param_string = self.param_string_for_query(request.params)
        else:
            param_string = request.data if isinstance(request.data, str) else ""
        signature = self._sign(timestamp, param_string)
        headers = dict(request.headers or {})
        headers.update(self.rest_headers(timestamp, signature))
        request.headers = headers
        return request

    async def ws_authenticate(self, request: WSRequest) -> WSRequest:
        return request

    def get_ws_login_payload(self) -> Dict[str, Any]:
        timestamp = self.header_timestamp()
        signature = self._sign(timestamp, "")
        return {
            "method": CONSTANTS.WS_LOGIN_METHOD,
            "param": {
                "apiKey": self._api_key,
                "reqTime": timestamp,
                "signature": signature,
            },
        }
