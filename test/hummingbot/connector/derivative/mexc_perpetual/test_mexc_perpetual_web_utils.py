import asyncio
import gzip
import json
import unittest

from aioresponses import aioresponses

from hummingbot.connector.derivative.mexc_perpetual import (
    mexc_perpetual_constants as CONSTANTS,
    mexc_perpetual_web_utils as web_utils,
)
from hummingbot.core.web_assistant.connections.data_types import WSResponse


class MexcPerpetualWebUtilsTests(unittest.TestCase):
    def test_rest_and_ws_urls(self):
        self.assertEqual(
            "https://api.mexc.com/api/v1/contract/ping",
            web_utils.public_rest_url(CONSTANTS.PING_PATH),
        )
        self.assertEqual(web_utils.public_rest_url(CONSTANTS.PING_PATH), web_utils.private_rest_url(CONSTANTS.PING_PATH))
        self.assertEqual("wss://contract.mexc.com/edge", web_utils.wss_url())

    @aioresponses()
    def test_get_current_server_time(self, api_mock):
        url = web_utils.public_rest_url(CONSTANTS.PING_PATH)
        payload = {"success": True, "code": 0, "data": 1761875313209}
        api_mock.get(url, status=400, body=json.dumps(payload))

        server_time = asyncio.get_event_loop().run_until_complete(
            asyncio.wait_for(web_utils.get_current_server_time(), 1)
        )

        self.assertEqual(1761875313209, server_time)

    def test_decode_plain_and_gzip_websocket_messages(self):
        message = {"channel": "push.depth", "symbol": "BTC_USDT", "data": {"version": 1}}
        self.assertEqual(message, web_utils.decode_ws_message(message))
        self.assertEqual(message, web_utils.decode_ws_message(json.dumps(message)))
        compressed = gzip.compress(json.dumps(message).encode("utf-8"))
        self.assertEqual(message, web_utils.decode_ws_message(compressed))

    def test_ws_post_processor_decodes_gzip(self):
        message = {"channel": "pong", "data": 1}
        response = WSResponse(gzip.compress(json.dumps(message).encode("utf-8")))
        processor = web_utils.MexcPerpetualWSPostProcessor()
        processed = asyncio.get_event_loop().run_until_complete(processor.post_process(response))
        self.assertEqual(message, processed.data)
