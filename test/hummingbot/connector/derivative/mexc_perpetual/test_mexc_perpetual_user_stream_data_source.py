import asyncio
from test.isolated_asyncio_wrapper_test_case import IsolatedAsyncioWrapperTestCase
from typing import Awaitable
from unittest.mock import AsyncMock, MagicMock

from hummingbot.connector.derivative.mexc_perpetual import mexc_perpetual_constants as CONSTANTS
from hummingbot.connector.derivative.mexc_perpetual.mexc_perpetual_auth import MexcPerpetualAuth
from hummingbot.connector.derivative.mexc_perpetual.mexc_perpetual_derivative import MexcPerpetualDerivative
from hummingbot.connector.derivative.mexc_perpetual.mexc_perpetual_user_stream_data_source import (
    MexcPerpetualUserStreamDataSource,
)


class MexcPerpetualUserStreamDataSourceTests(IsolatedAsyncioWrapperTestCase):
    level = 0

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.trading_pair = "BTC-USDT"

    def setUp(self) -> None:
        super().setUp()
        self.log_records = []
        self.time_provider = MagicMock()
        self.time_provider.time.return_value = 1611038237.237
        self.auth = MexcPerpetualAuth(api_key="api-key", secret_key="secret", time_provider=self.time_provider)
        self.connector = MexcPerpetualDerivative(
            mexc_perpetual_api_key="api-key",
            mexc_perpetual_api_secret="secret",
            trading_pairs=[self.trading_pair],
            trading_required=False,
        )
        self.data_source = MexcPerpetualUserStreamDataSource(
            auth=self.auth,
            trading_pairs=[self.trading_pair],
            connector=self.connector,
            api_factory=self.connector._web_assistants_factory,
        )
        self.data_source.logger().setLevel(1)
        self.data_source.logger().addHandler(self)

    def handle(self, record):
        self.log_records.append(record)

    def async_run_with_timeout(self, coroutine: Awaitable, timeout: float = 1):
        return self.run_async_with_timeout(coroutine, timeout)

    def test_login_sends_openapi_signature_and_requires_success(self):
        ws = AsyncMock()
        ws.receive.return_value = AsyncMock(data={"channel": "rs.login", "data": "success", "ts": "1587442022003"})
        self.async_run_with_timeout(self.data_source._authenticate(ws))
        payload = ws.send.await_args.args[0].payload
        self.assertEqual("login", payload["method"])
        self.assertEqual("api-key", payload["param"]["apiKey"])
        self.assertEqual("1611038237237", payload["param"]["reqTime"])
        self.assertEqual(self.auth._sign("1611038237237", ""), payload["param"]["signature"])
        self.assertTrue(any("login succeeded" in record.getMessage() for record in self.log_records))

    def test_login_failure_raises(self):
        ws = AsyncMock()
        ws.receive.return_value = AsyncMock(data={"channel": "rs.error", "data": "auth failed"})
        with self.assertRaises(IOError):
            self.async_run_with_timeout(self.data_source._authenticate(ws))

    def test_personal_events_are_queued_and_pong_is_ignored(self):
        queue = asyncio.Queue()
        order_event = {"channel": CONSTANTS.WS_ORDER_CHANNEL, "data": {"orderId": 1}}
        self.async_run_with_timeout(self.data_source._process_event_message(order_event, queue))
        self.async_run_with_timeout(
            self.data_source._process_event_message({"channel": CONSTANTS.WS_PONG_CHANNEL, "data": 1}, queue)
        )
        self.assertEqual(order_event, queue.get_nowait())
        self.assertTrue(queue.empty())

    def test_interval_ping_sends_json_ping(self):
        ws = AsyncMock()

        async def stop_after_send(*args, **kwargs):
            raise asyncio.CancelledError

        ws.send.side_effect = stop_after_send
        with self.assertRaises(asyncio.CancelledError):
            self.async_run_with_timeout(self.data_source._send_interval_ping(ws))
        self.assertEqual({"method": "ping"}, ws.send.await_args.args[0].payload)
