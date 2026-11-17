import asyncio
import hashlib
import hmac
from typing import Awaitable
from unittest import TestCase
from unittest.mock import MagicMock

from hummingbot.connector.derivative.mexc_perpetual.mexc_perpetual_auth import MexcPerpetualAuth
from hummingbot.core.web_assistant.connections.data_types import RESTMethod, RESTRequest, WSJSONRequest


class MexcPerpetualAuthTests(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.api_key = "mx0aBYs33eIilxBW5C"
        self.secret_key = "test_secret_key"
        self.time_provider = MagicMock()
        self.time_provider.time.return_value = 1657186536.762
        self.auth = MexcPerpetualAuth(
            api_key=self.api_key,
            secret_key=self.secret_key,
            time_provider=self.time_provider,
        )

    def async_run_with_timeout(self, coroutine: Awaitable, timeout: float = 1):
        return asyncio.get_event_loop().run_until_complete(asyncio.wait_for(coroutine, timeout))

    def _signature(self, timestamp: str, param_string: str) -> str:
        payload = f"{self.api_key}{timestamp}{param_string}"
        return hmac.new(self.secret_key.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()

    def test_get_signature_sorts_query_and_omits_nulls(self):
        request = RESTRequest(
            method=RESTMethod.GET,
            url="https://api.mexc.com/api/v1/private/position/funding_records",
            params={"symbol": "BTC_USDT", "page_num": 1, "empty": None},
            is_auth_required=True,
            headers={},
        )
        self.async_run_with_timeout(self.auth.rest_authenticate(request))

        timestamp = "1657186536762"
        param_string = "page_num=1&symbol=BTC_USDT"
        self.assertEqual(timestamp, request.headers["Request-Time"])
        self.assertEqual(self.api_key, request.headers["ApiKey"])
        self.assertEqual("10", request.headers["Recv-Window"])
        self.assertEqual(self._signature(timestamp, param_string), request.headers["Signature"])

    def test_post_signature_uses_the_raw_json_body(self):
        body = '{"symbol":"BTC_USDT","externalOid":"abc"}'
        request = RESTRequest(
            method=RESTMethod.POST,
            url="https://api.mexc.com/api/v1/private/order/create",
            data=body,
            is_auth_required=True,
            headers={},
        )
        self.async_run_with_timeout(self.auth.rest_authenticate(request))

        timestamp = "1657186536762"
        self.assertEqual(body, request.data)
        self.assertEqual(self._signature(timestamp, body), request.headers["Signature"])

    def test_post_signature_signs_a_json_array_body_exactly(self):
        body = '["877759993012810880"]'
        request = RESTRequest(
            method=RESTMethod.POST,
            url="https://api.mexc.com/api/v1/private/order/cancel",
            data=body,
            is_auth_required=True,
            headers={},
        )
        self.async_run_with_timeout(self.auth.rest_authenticate(request))

        timestamp = "1657186536762"
        self.assertEqual(body, request.data)
        self.assertEqual(self._signature(timestamp, body), request.headers["Signature"])

    def test_post_signature_serializes_a_list_body_before_signing(self):
        request = RESTRequest(
            method=RESTMethod.POST,
            url="https://api.mexc.com/api/v1/private/order/cancel",
            data=["877759993012810880"],
            is_auth_required=True,
            headers={},
        )
        self.async_run_with_timeout(self.auth.rest_authenticate(request))

        body = '["877759993012810880"]'
        timestamp = "1657186536762"
        self.assertEqual(body, request.data)
        self.assertEqual(self._signature(timestamp, body), request.headers["Signature"])

    def test_ws_login_signs_api_key_plus_timestamp(self):
        payload = self.auth.get_ws_login_payload()
        timestamp = "1657186536762"
        self.assertEqual("login", payload["method"])
        self.assertEqual(self.api_key, payload["param"]["apiKey"])
        self.assertEqual(timestamp, payload["param"]["reqTime"])
        self.assertEqual(self._signature(timestamp, ""), payload["param"]["signature"])
        self.assertEqual(self.api_key + timestamp, "mx0aBYs33eIilxBW5C1657186536762")

    def test_ws_request_is_left_unchanged(self):
        request = WSJSONRequest(payload={"method": "ping"}, is_auth_required=True)
        self.async_run_with_timeout(self.auth.ws_authenticate(request))
        self.assertEqual({"method": "ping"}, request.payload)
