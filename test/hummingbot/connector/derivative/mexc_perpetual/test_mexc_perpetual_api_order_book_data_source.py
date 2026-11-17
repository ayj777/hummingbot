import asyncio
import json
import re
from decimal import Decimal
from test.isolated_asyncio_wrapper_test_case import IsolatedAsyncioWrapperTestCase
from typing import Awaitable
from unittest.mock import AsyncMock

from aioresponses import aioresponses
from bidict import bidict

import hummingbot.connector.derivative.mexc_perpetual.mexc_perpetual_constants as CONSTANTS
import hummingbot.connector.derivative.mexc_perpetual.mexc_perpetual_web_utils as web_utils
from hummingbot.connector.derivative.mexc_perpetual.mexc_perpetual_api_order_book_data_source import (
    MexcPerpetualAPIOrderBookDataSource,
)
from hummingbot.connector.derivative.mexc_perpetual.mexc_perpetual_derivative import MexcPerpetualDerivative
from hummingbot.core.data_type.order_book_message import OrderBookMessageType
from hummingbot.core.web_assistant.connections.data_types import WSJSONRequest


class MexcPerpetualAPIOrderBookDataSourceTests(IsolatedAsyncioWrapperTestCase):
    level = 0

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.base_asset = "BTC"
        cls.quote_asset = "USDT"
        cls.trading_pair = f"{cls.base_asset}-{cls.quote_asset}"
        cls.exchange_symbol = f"{cls.base_asset}_{cls.quote_asset}"

    def setUp(self) -> None:
        super().setUp()
        self.log_records = []
        self.connector = MexcPerpetualDerivative(
            mexc_perpetual_api_key="",
            mexc_perpetual_api_secret="",
            trading_pairs=[self.trading_pair],
            trading_required=False,
        )
        self.connector._set_trading_pair_symbol_map(bidict({self.exchange_symbol: self.trading_pair}))
        self.connector._contract_sizes[self.trading_pair] = Decimal("10")
        self.data_source = MexcPerpetualAPIOrderBookDataSource(
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

    def _is_logged(self, message: str) -> bool:
        return any(message in record.getMessage() for record in self.log_records)

    @aioresponses()
    def test_order_book_snapshot_converts_contracts_to_base_size(self, mock_api):
        url = web_utils.public_rest_url(CONSTANTS.DEPTH_PATH.format(symbol=self.exchange_symbol))
        mock_api.get(
            re.compile(f"^{re.escape(url)}.*"),
            body=json.dumps({
                "success": True,
                "code": 0,
                "data": {
                    "asks": [[100, 2, 1]],
                    "bids": [[99, 0, 1], [98, 3, 1]],
                    "version": 25,
                    "timestamp": 1761879567135,
                },
            }),
        )
        snapshot = self.async_run_with_timeout(self.data_source._order_book_snapshot(self.trading_pair))
        self.assertEqual(OrderBookMessageType.SNAPSHOT, snapshot.type)
        self.assertEqual(25, snapshot.update_id)
        self.assertEqual(["100", "20"], snapshot.content["asks"][0])
        self.assertEqual(["99", "0"], snapshot.content["bids"][0])
        self.assertEqual(["98", "30"], snapshot.content["bids"][1])
        self.assertEqual(25, self.data_source._last_depth_version[self.exchange_symbol])

    def test_depth_diff_applies_absolute_contract_quantities(self):
        queue = asyncio.Queue()
        message = {
            "channel": CONSTANTS.WS_DEPTH_CHANNEL,
            "symbol": self.exchange_symbol,
            "data": {
                "asks": [[100, 1, 1]],
                "bids": [[99, 0, 0]],
                "begin": 26,
                "end": 27,
                "version": 27,
            },
            "ts": 1761879567135,
        }
        self.async_run_with_timeout(self.data_source._parse_order_book_diff_message(message, queue))
        diff = queue.get_nowait()
        self.assertEqual(OrderBookMessageType.DIFF, diff.type)
        self.assertEqual(27, diff.update_id)
        self.assertEqual(["100", "10"], diff.content["asks"][0])
        self.assertEqual(["99", "0"], diff.content["bids"][0])

    def test_depth_gap_does_not_apply_the_stale_diff(self):
        self.data_source._last_depth_version[self.exchange_symbol] = 10
        queue = asyncio.Queue()
        message = {
            "channel": CONSTANTS.WS_DEPTH_CHANNEL,
            "symbol": self.exchange_symbol,
            "data": {"asks": [[1, 1, 1]], "bids": [], "begin": 40, "end": 41},
            "ts": 1,
        }
        self.async_run_with_timeout(self.data_source._parse_order_book_diff_message(message, queue))
        self.assertTrue(queue.empty())
        self.assertNotIn(self.exchange_symbol, self.data_source._last_depth_version)
        self.assertTrue(self._is_logged("Order book depth gap"))

    def test_trade_message_converts_contract_volume(self):
        queue = asyncio.Queue()
        message = {
            "channel": CONSTANTS.WS_DEAL_CHANNEL,
            "symbol": self.exchange_symbol,
            "data": [{"p": "100", "v": "4", "T": 1, "t": 1761883066648, "i": "150"}],
            "ts": 1761883066648,
        }
        self.async_run_with_timeout(self.data_source._parse_trade_message(message, queue))
        trade = queue.get_nowait()
        self.assertEqual(OrderBookMessageType.TRADE, trade.type)
        self.assertEqual(150, trade.content["trade_id"])
        self.assertEqual("40", trade.content["amount"])
        self.assertEqual(1.0, trade.content["trade_type"])

    @aioresponses()
    def test_get_funding_info(self, mock_api):
        url = web_utils.public_rest_url(CONSTANTS.FUNDING_RATE_PATH.format(symbol=self.exchange_symbol))
        mock_api.get(
            re.compile(f"^{re.escape(url)}.*"),
            body=json.dumps({
                "success": True,
                "code": 0,
                "data": {
                    "symbol": self.exchange_symbol,
                    "fundingRate": "0.000018",
                    "nextSettleTime": 1761897600000,
                    "idxPrice": "81254.9",
                    "fairPrice": "81212.8",
                },
            }),
        )
        info = self.async_run_with_timeout(self.data_source.get_funding_info(self.trading_pair))
        self.assertEqual(Decimal("81254.9"), info.index_price)
        self.assertEqual(Decimal("81212.8"), info.mark_price)
        self.assertEqual(Decimal("0.000018"), info.rate)
        self.assertEqual(1761897600, info.next_funding_utc_timestamp)

    def test_channel_routing(self):
        self.assertEqual(
            self.data_source._diff_messages_queue_key,
            self.data_source._channel_originating_message({"channel": CONSTANTS.WS_DEPTH_CHANNEL}),
        )
        self.assertEqual(
            self.data_source._trade_messages_queue_key,
            self.data_source._channel_originating_message({"channel": CONSTANTS.WS_DEAL_CHANNEL}),
        )
        self.assertEqual(
            self.data_source._funding_info_messages_queue_key,
            self.data_source._channel_originating_message({"channel": CONSTANTS.WS_FUNDING_CHANNEL}),
        )
        self.assertIsNone(self.data_source._channel_originating_message({"channel": CONSTANTS.WS_PONG_CHANNEL}))

    def test_subscribe_payload_disables_compression(self):
        ws = AsyncMock()
        self.async_run_with_timeout(self.data_source._subscribe_channels(ws))
        methods = [call.args[0].payload["method"] for call in ws.send.await_args_list]
        self.assertEqual(["sub.depth", "sub.deal", "sub.funding.rate"], methods)
        depth = ws.send.await_args_list[0].args[0]
        self.assertIsInstance(depth, WSJSONRequest)
        self.assertFalse(depth.payload["param"]["compress"])
        self.assertFalse(ws.send.await_args_list[1].args[0].payload["param"]["reduce"])
