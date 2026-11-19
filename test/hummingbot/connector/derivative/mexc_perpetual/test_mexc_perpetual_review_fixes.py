import asyncio
import hashlib
import hmac
import json
import random
import re
from decimal import Decimal
from test.isolated_asyncio_wrapper_test_case import IsolatedAsyncioWrapperTestCase
from unittest.mock import AsyncMock

from aioresponses import aioresponses
from bidict import bidict

import hummingbot.connector.derivative.mexc_perpetual.mexc_perpetual_constants as CONSTANTS
from hummingbot.connector.derivative.mexc_perpetual.mexc_perpetual_derivative import (
    MarginMode,
    MexcPerpetualDerivative,
    MexcResponseError,
)
from hummingbot.connector.derivative.mexc_perpetual.mexc_perpetual_web_utils import private_rest_url
from hummingbot.connector.trading_rule import TradingRule
from hummingbot.core.data_type.common import OrderType, PositionAction, PositionMode, PositionSide, TradeType
from hummingbot.core.data_type.in_flight_order import OrderState


class MexcPerpetualReviewFixTests(IsolatedAsyncioWrapperTestCase):
    def setUp(self):
        super().setUp()
        self.trading_pair = "AIOT-USDT"
        self.exchange = MexcPerpetualDerivative(
            mexc_perpetual_api_key="key",
            mexc_perpetual_api_secret="secret",
            trading_pairs=[self.trading_pair],
            trading_required=True,
        )
        self.exchange._set_trading_pair_symbol_map(bidict({"AIOT_USDT": self.trading_pair}))
        self.exchange._set_current_timestamp(1_700_000_000)
        self.exchange._trading_rules[self.trading_pair] = TradingRule(
            trading_pair=self.trading_pair,
            min_order_size=Decimal("10"),
            min_price_increment=Decimal("0.00001"),
            min_base_amount_increment=Decimal("10"),
            buy_order_collateral_token="USDT",
            sell_order_collateral_token="USDT",
        )

    def _track(self, amount: Decimal = Decimal("1000")):
        self.exchange._contract_sizes[self.trading_pair] = Decimal("10")
        self.exchange.start_tracking_order(
            order_id="oid-1",
            exchange_order_id="ex-1",
            trading_pair=self.trading_pair,
            trade_type=TradeType.BUY,
            price=Decimal("0.07816"),
            amount=amount,
            order_type=OrderType.LIMIT,
            position_action=PositionAction.OPEN,
        )
        return self.exchange.in_flight_orders["oid-1"]

    async def test_rest_fill_poll_replaces_the_synthetic_fill_without_double_counting(self):
        order = self._track(Decimal("1000"))
        self.exchange._order_tracker.TRADE_FILLS_WAIT_TIMEOUT = 0.05
        self.exchange._process_order_event_message({
            "orderId": "ex-1",
            "externalOid": "oid-1",
            "state": 4,
            "dealVol": "34",
            "dealAvgPrice": "0.07816",
        })
        await asyncio.sleep(0.2)
        self.assertEqual(Decimal("340"), order.executed_amount_base)

        async def get(path_url, **kwargs):
            return {
                "success": True,
                "code": 0,
                "data": [{
                    "id": "555",
                    "price": "0.07816",
                    "vol": "34",
                    "fee": "0.01",
                    "feeCurrency": "USDT",
                    "timestamp": 1_700_000_000_000,
                }],
            }

        self.exchange._api_get = get
        await self.exchange._update_orders_fills([order])
        await self.exchange._update_orders_fills([order])
        self.assertEqual(Decimal("340"), order.executed_amount_base)
        self.assertNotIn(f"mexc-reported-deal-{order.client_order_id}", order.order_fills)
        self.assertEqual(Decimal("340"), order.order_fills["555"].fill_base_amount)

    async def test_http_500_does_not_count_as_order_not_found(self):
        order = self._track()
        await self.exchange._handle_update_error_for_active_order(
            order,
            IOError("Error executing request GET https://api.mexc.com/order. HTTP status is 500. Error: unavailable"),
        )
        self.assertEqual(0, self.exchange._order_tracker._order_not_found_records.get(order.client_order_id, 0))
        self.assertNotEqual(OrderState.FAILED, order.current_state)

    async def test_cancel_is_confirmed_only_after_cancel_or_not_found(self):
        order = self._track()
        self.assertFalse(self.exchange.is_cancel_confirmed(order.client_order_id))
        self.exchange._cancel_requests.add(order.client_order_id)
        order.current_state = OrderState.CANCELED
        self.assertFalse(self.exchange.is_cancel_confirmed(order.client_order_id))
        self.exchange._cancel_requests.discard(order.client_order_id)
        self.assertTrue(self.exchange.is_cancel_confirmed(order.client_order_id))
        self.exchange._confirmed_absent.add("missing-id")
        self.assertTrue(self.exchange.is_cancel_confirmed("missing-id"))

    async def test_cancel_2041_retries_the_follow_up_read(self):
        order = self._track(Decimal("100"))
        attempts = {"n": 0}

        async def post(*args, **kwargs):
            return {
                "success": True,
                "code": 0,
                "data": [{"orderId": "ex-1", "errorCode": 2041, "errorMsg": "order state cannot be cancelled"}],
            }

        async def get(path_url, **kwargs):
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise IOError("temporary read failure")
            if "deal_details" in path_url:
                return {"success": True, "code": 0, "data": []}
            return {
                "success": True,
                "code": 0,
                "data": {
                    "orderId": "ex-1",
                    "externalOid": "oid-1",
                    "state": 3,
                    "dealVol": "10",
                    "dealAvgPrice": "2",
                    "price": "2",
                    "vol": "10",
                },
            }

        self.exchange._api_post = post
        self.exchange._api_get = get
        await self.exchange._place_cancel("oid-1", order)
        await asyncio.sleep(0.3)
        self.assertGreaterEqual(attempts["n"], 3)
        self.assertEqual(Decimal("100"), order.executed_amount_base)
        self.assertEqual(OrderState.FILLED, order.current_state)

    async def test_resolve_unknown_order_reads_client_id_then_order_id(self):
        order = self._track(Decimal("100"))
        calls = []

        async def get(path_url, **kwargs):
            calls.append(path_url)
            if "external" in path_url:
                raise IOError("client id lookup failed")
            return {
                "success": True,
                "code": 0,
                "data": {
                    "orderId": "ex-1",
                    "externalOid": "oid-1",
                    "state": 3,
                    "dealVol": "10",
                    "dealAvgPrice": "2",
                    "price": "2",
                    "vol": "10",
                },
            }

        self.exchange._api_get = get
        state = await self.exchange.resolve_unknown_order(self.trading_pair, "oid-1", "ex-1")
        self.assertIsNone(state)
        self.assertTrue(any("external" in path for path in calls))
        self.assertTrue(any("/order/get/" in path for path in calls))
        self.assertEqual(Decimal("100"), order.executed_amount_base)

    async def test_canceled_client_read_does_not_beat_an_open_order_id_read(self):
        order = self._track(Decimal("1000"))
        order.current_state = OrderState.OPEN

        async def get(path_url, **kwargs):
            if "external" in path_url:
                data = {
                    "orderId": "ex-1",
                    "externalOid": "oid-1",
                    "state": 4,
                    "dealVol": 0,
                    "price": "0.07816",
                    "vol": "100",
                }
            else:
                data = {
                    "orderId": "ex-1",
                    "externalOid": "oid-1",
                    "state": 2,
                    "dealVol": 0,
                    "price": "0.07816",
                    "vol": "100",
                }
            return {"success": True, "code": 0, "data": data}

        self.exchange._api_get = get
        state = await self.exchange.resolve_unknown_order(self.trading_pair, "oid-1", "ex-1")
        self.assertNotEqual(OrderState.CANCELED, state)
        self.assertNotIn(state, (OrderState.CANCELED, OrderState.FAILED, OrderState.FILLED))
        self.assertEqual(OrderState.OPEN, order.current_state)
        self.assertEqual(Decimal("0"), order.executed_amount_base)

    async def test_agreed_zero_size_cancel_is_final(self):
        async def get(path_url, **kwargs):
            return {
                "success": True,
                "code": 0,
                "data": {
                    "orderId": "ex-1",
                    "externalOid": "oid-1",
                    "state": 4,
                    "dealVol": 0,
                    "price": "0.07816",
                    "vol": "100",
                },
            }

        self.exchange._api_get = get
        state = await self.exchange.resolve_unknown_order(self.trading_pair, "oid-1", "ex-1")
        self.assertEqual(OrderState.CANCELED, state)

    async def test_zero_size_cancel_with_a_failed_read_stays_unknown(self):
        async def get(path_url, **kwargs):
            if "external" in path_url:
                raise IOError("http status is 500")
            return {
                "success": True,
                "code": 0,
                "data": {
                    "orderId": "ex-1",
                    "externalOid": "oid-1",
                    "state": 4,
                    "dealVol": 0,
                    "price": "0.07816",
                    "vol": "100",
                },
            }

        self.exchange._api_get = get
        state = await self.exchange.resolve_unknown_order(self.trading_pair, "oid-1", "ex-1")
        self.assertIsNone(state)

    async def test_order_not_exist_on_both_reads_is_confirmed_absent(self):
        async def get(path_url, **kwargs):
            return {"success": False, "code": 2040, "message": "Order does not exist", "data": None}

        self.exchange._api_get = get
        state = await self.exchange.resolve_unknown_order(self.trading_pair, "oid-1", "ex-1")
        self.assertEqual("confirmed_absent", state)

    async def test_empty_result_on_both_reads_is_confirmed_absent(self):
        async def get(path_url, **kwargs):
            return {"success": True, "code": 0, "data": None}

        self.exchange._api_get = get
        state = await self.exchange.resolve_unknown_order(self.trading_pair, "oid-1", "ex-1")
        self.assertIsNone(state)

    async def test_http_500_lookup_stays_unknown(self):
        async def get(path_url, **kwargs):
            raise IOError("http status is 500")

        self.exchange._api_get = get
        state = await self.exchange.resolve_unknown_order(self.trading_pair, "oid-1", "ex-1")
        self.assertIsNone(state)

    def _order_urls(self, client_order_id: str, exchange_order_id: str):
        client_path = CONSTANTS.ORDER_BY_EXTERNAL_PATH.format(symbol="AIOT_USDT", external_oid=client_order_id)
        order_path = CONSTANTS.ORDER_GET_PATH.format(order_id=exchange_order_id)
        return private_rest_url(client_path), private_rest_url(order_path)

    @aioresponses()
    async def test_websocket_open_is_remembered_and_empty_lookup_stays_unknown(self, mock_api):
        order = self._track(Decimal("1000"))
        self.exchange._process_order_event_message({
            "orderId": "ex-1",
            "externalOid": "oid-1",
            "state": 2,
            "dealVol": 0,
            "price": "0.07816",
            "vol": "100",
        })
        await asyncio.sleep(0.05)
        self.assertEqual(OrderState.OPEN, order.current_state)
        self.assertTrue(self.exchange.order_was_seen_live("oid-1"))
        client_url, order_url = self._order_urls("oid-1", "ex-1")
        body = json.dumps({"success": True, "code": 0, "data": None})
        mock_api.get(client_url, body=body, repeat=True)
        mock_api.get(order_url, body=body, repeat=True)
        state = await self.exchange.resolve_unknown_order(self.trading_pair, "oid-1", "ex-1")
        self.assertIsNone(state)
        self.assertNotEqual("confirmed_absent", state)

    @aioresponses()
    async def test_real_2040_body_is_confirmed_absent_only_before_the_exchange_acks(self, mock_api):
        client_url, order_url = self._order_urls("never-1", "missing-1")
        body = json.dumps({"success": False, "code": 2040, "message": "Order does not exist"})
        mock_api.get(client_url, body=body, repeat=True)
        mock_api.get(order_url, body=body, repeat=True)
        state = await self.exchange.resolve_unknown_order(self.trading_pair, "never-1", None)
        self.assertEqual("confirmed_absent", state)

        order = self._track(Decimal("1000"))
        self.exchange.record_order_observation(
            "oid-1",
            self.trading_pair,
            exchange_order_id="ex-1",
            new_state=OrderState.OPEN,
            acked=True,
        )
        await asyncio.sleep(0.05)
        client_url, order_url = self._order_urls("oid-1", "ex-1")
        mock_api.get(client_url, body=body, repeat=True)
        mock_api.get(order_url, body=body, repeat=True)
        later = await self.exchange.resolve_unknown_order(self.trading_pair, "oid-1", "ex-1")
        self.assertIsNone(later)
        self.assertEqual(OrderState.OPEN, order.current_state)

    async def test_status_2040_does_not_fail_a_seen_order(self):
        order = self._track(Decimal("1000"))
        order.current_state = OrderState.OPEN
        order.update_exchange_order_id("ex-1")
        self.exchange.record_order_observation(
            "oid-1",
            self.trading_pair,
            exchange_order_id="ex-1",
            acked=True,
        )
        error = MexcResponseError(2040, "Order does not exist")
        for _ in range(5):
            await self.exchange._handle_update_error_for_active_order(order, error)
        self.assertNotEqual(OrderState.FAILED, order.current_state)
        self.assertIn("oid-1", self.exchange.in_flight_orders)
        self.assertEqual(0, self.exchange._order_tracker._order_not_found_records.get("oid-1", 0))
        self.assertIn("oid-1", self.exchange.orders_awaiting_resolution())

    async def test_deal_reconcile_records_the_fill_in_order_memory(self):
        order = self._track(Decimal("1000"))

        async def get(path_url, **kwargs):
            return {
                "success": True,
                "code": 0,
                "data": [{
                    "id": "9",
                    "price": "0.07816",
                    "vol": "34",
                    "fee": "0",
                    "feeCurrency": "USDT",
                    "timestamp": 1_700_000_000_000,
                    "orderId": "ex-1",
                }],
            }

        self.exchange._api_get = get
        await self.exchange._reconcile_exchange_deals(order)
        base, _quote = self.exchange.counted_fill("oid-1")
        self.assertEqual(Decimal("340"), base)
        self.assertEqual(Decimal("340"), order.executed_amount_base)

    async def test_counted_quote_does_not_shrink_when_the_size_grows(self):
        self._track(Decimal("1000"))

        async def get(path_url, **kwargs):
            deal = "10" if "external" in path_url else "20"
            price = "10" if "external" in path_url else "0.1"
            return {
                "success": True,
                "code": 0,
                "data": {
                    "orderId": "ex-1",
                    "externalOid": "oid-1",
                    "state": 2,
                    "dealVol": deal,
                    "dealAvgPrice": price,
                    "price": price,
                    "vol": "100",
                },
            }

        self.exchange._api_get = get
        await self.exchange.resolve_unknown_order(self.trading_pair, "oid-1", "ex-1")
        base, quote = self.exchange.counted_fill("oid-1")
        self.assertEqual(Decimal("200"), base)
        self.assertGreaterEqual(quote, Decimal("1000"))

    async def test_filled_versus_open_keeps_the_fill_and_a_later_2040_does_not_clear_it(self):
        order = self._track(Decimal("1000"))
        order.current_state = OrderState.OPEN
        phase = {"step": "conflict"}

        async def get(path_url, **kwargs):
            if phase["step"] == "absent":
                return {"success": False, "code": 2040, "message": "Order does not exist", "data": None}
            if "external" in path_url:
                data = {
                    "orderId": "ex-1",
                    "externalOid": "oid-1",
                    "state": 3,
                    "dealVol": "100",
                    "dealAvgPrice": "0.07816",
                    "price": "0.07816",
                    "vol": "100",
                }
            else:
                data = {
                    "orderId": "ex-1",
                    "externalOid": "oid-1",
                    "state": 2,
                    "dealVol": 0,
                    "price": "0.07816",
                    "vol": "100",
                }
            return {"success": True, "code": 0, "data": data}

        self.exchange._api_get = get
        state = await self.exchange.resolve_unknown_order(self.trading_pair, "oid-1", "ex-1")
        self.assertIsNone(state)
        self.assertEqual(Decimal("1000"), order.executed_amount_base)
        self.assertEqual(Decimal("78.16"), order.executed_amount_quote)
        base, quote = self.exchange.counted_fill("oid-1")
        self.assertEqual(Decimal("1000"), base)
        self.assertEqual(Decimal("78.16"), quote)
        phase["step"] = "absent"
        later = await self.exchange.resolve_unknown_order(self.trading_pair, "oid-1", "ex-1")
        self.assertNotEqual("confirmed_absent", later)
        self.assertIsNone(later)
        self.assertEqual(Decimal("1000"), order.executed_amount_base)
        self.assertTrue(self.exchange.order_was_seen_live("oid-1"))

    async def test_http_500_whose_url_contains_2040_stays_unknown(self):
        order = self._track(Decimal("1000"))
        order.current_state = OrderState.OPEN

        async def get(path_url, **kwargs):
            raise IOError(
                f"Error executing request GET https://api.mexc.com{path_url}. HTTP status is 500."
            )

        self.exchange._api_get = get
        state = await self.exchange.resolve_unknown_order(self.trading_pair, "cid-2040", "2040")
        self.assertIsNone(state)
        self.assertEqual(OrderState.OPEN, order.current_state)
        self.assertEqual(Decimal("0"), order.executed_amount_base)

    async def test_random_read_sequences_never_drop_a_fill_or_exceed_the_total(self):
        self.exchange._order_tracker.TRADE_FILLS_WAIT_TIMEOUT = 0.01
        rng = random.Random(20261117)
        total = Decimal("1000")
        sequences = [self._review_fill_then_absent_sequence()]
        sequences.extend(self._random_read_sequence(rng) for _ in range(20))
        for sequence in sequences:
            order = self._track(Decimal("1000"))
            order.current_state = OrderState.OPEN
            reported = Decimal("0")
            seen_live = False
            extra = Decimal("0")
            resolved = False
            previous_base = Decimal("0")
            for step in sequence:
                kind = step[0]
                client_read = step[1] if len(step) > 1 and kind not in ("ws_order", "ws_deal", "deal_reconcile") else None
                order_read = step[2] if len(step) > 2 else None
                if kind == "status_error":
                    from hummingbot.connector.derivative.mexc_perpetual.mexc_perpetual_derivative import (
                        MexcResponseError,
                    )
                    await self.exchange._handle_update_error_for_active_order(
                        order,
                        MexcResponseError(2040, "Order does not exist"),
                    )
                    if (
                        self.exchange.order_was_seen_live("oid-1")
                        or "oid-1" in self.exchange._acked_client_ids
                        or "oid-1" in self.exchange._submitted_client_ids
                    ):
                        self.assertNotEqual(OrderState.FAILED, order.current_state)
                        self.assertEqual(0, self.exchange._order_tracker._order_not_found_records.get("oid-1", 0))
                        self.assertIn("oid-1", self.exchange.orders_awaiting_resolution())
                    state = None
                elif kind == "deal_reconcile":
                    fill = dict(step[1])
                    reported = max(reported, Decimal(str(fill["vol"])) * Decimal("10"))
                    seen_live = True

                    async def deals(path_url, fill=fill, **kwargs):
                        return {"success": True, "code": 0, "data": [fill]}

                    self.exchange._api_get = deals
                    await self.exchange._reconcile_exchange_deals(order)
                    state = None
                elif kind == "ws_order":
                    payload = dict(step[1])
                    payload["externalOid"] = "oid-1"
                    reported = max(reported, self._payload_fill_base(payload))
                    seen_live = True
                    self.exchange._process_order_event_message(payload)
                    state = None
                elif kind == "ws_deal":
                    payload = dict(step[1])
                    payload["externalOid"] = "oid-1"
                    reported = max(reported, self._payload_fill_base(payload))
                    seen_live = True
                    self.exchange._process_trade_event_message(payload)
                    state = None
                elif kind == "rest_status":
                    reported = max(reported, self._fill_base(step[1]))
                    seen_live = seen_live or self._read_shows_live_order(step[1]) or self._read_shows_live_order(step[2])
                    self._stub_reads(step[1], step[2])
                    try:
                        await self.exchange._request_order_status(order)
                    except Exception:
                        pass
                    state = None
                elif kind == "cancel_read":
                    payload = step[1] if isinstance(step[1], dict) else (step[1].get("data") if isinstance(step[1], dict) else None)
                    if not isinstance(payload, dict):
                        state = None
                    else:
                        payload = dict(payload)
                        payload.setdefault("externalOid", "oid-1")
                        reported = max(reported, self._payload_fill_base(payload))
                        if self._payload_state_is_live(payload):
                            seen_live = True
                        self.exchange.record_order_observation(
                            "oid-1",
                            self.trading_pair,
                            payload=payload,
                            apply_state=False,
                            materialize_fill=True,
                        )
                        state = None
                else:
                    client_read, order_read = step[1], step[2]
                    reported = max(reported, self._fill_base(client_read), self._fill_base(order_read))
                    seen_live = seen_live or self._read_shows_live_order(client_read) or self._read_shows_live_order(order_read)
                    self._stub_reads(client_read, order_read)
                    state = await self.exchange.resolve_unknown_order(self.trading_pair, "oid-1", "ex-1")
                    if seen_live:
                        self.assertNotEqual("confirmed_absent", state)
                    if self._read_is_unknown(client_read) or self._read_is_unknown(order_read):
                        self.assertIsNone(state)
                    if (
                        not self._read_is_unknown(client_read)
                        and not self._read_is_unknown(order_read)
                        and self._read_is_open_state(client_read)
                        and self._read_is_open_state(order_read)
                    ):
                        self.assertIn(state, (OrderState.OPEN, OrderState.PARTIALLY_FILLED))
                tracked = self.exchange._find_order("oid-1")
                counted_base, _counted_quote = self.exchange.counted_fill("oid-1")
                if tracked is None:
                    continue
                observed = max(tracked.executed_amount_base, counted_base)
                self.assertGreaterEqual(
                    observed,
                    reported,
                    f"{kind} reported {reported} observed {observed} counted {counted_base} executed {tracked.executed_amount_base}",
                )
                self.assertGreaterEqual(observed, previous_base, kind)
                previous_base = observed
                if state in (OrderState.CANCELED, OrderState.FILLED) or (
                    state == "confirmed_absent" and not seen_live
                ):
                    child = max(Decimal("0"), total - order.executed_amount_base)
                    self.assertLessEqual(order.executed_amount_base + child, total)
                    if not resolved:
                        extra = child
                        resolved = True
                elif seen_live and not resolved:
                    self.assertLessEqual(max(order.amount, order.executed_amount_base), total)
            self.exchange._order_tracker._in_flight_orders.clear()
            if hasattr(self.exchange, "_max_fill_base"):
                self.exchange._max_fill_base.clear()
                self.exchange._max_fill_quote.clear()
                self.exchange._seen_live_client_ids.clear()

    def _stub_reads(self, client_read, order_read):
        async def get(path_url, client_read=client_read, order_read=order_read, **kwargs):
            chosen = order_read if "external" not in path_url else client_read
            if isinstance(chosen, Exception):
                raise chosen
            return chosen

        self.exchange._api_get = get

    @staticmethod
    def _payload_fill_base(payload: dict) -> Decimal:
        if payload.get("dealVol") not in (None, ""):
            return Decimal(str(payload.get("dealVol") or "0")) * Decimal("10")
        if payload.get("id") not in (None, "") and payload.get("vol") not in (None, ""):
            return Decimal(str(payload.get("vol") or "0")) * Decimal("10")
        return Decimal("0")

    @staticmethod
    def _payload_state_is_live(payload: dict) -> bool:
        try:
            state = int(payload.get("state") or 0)
            deal = Decimal(str(payload.get("dealVol") or payload.get("vol") or "0"))
        except Exception:
            return False
        return state in (1, 2, 3) or deal > 0

    def _review_fill_then_absent_sequence(self):
        filled = {
            "success": True,
            "code": 0,
            "data": {
                "orderId": "ex-1",
                "externalOid": "oid-1",
                "state": 3,
                "dealVol": "100",
                "dealAvgPrice": "0.07816",
                "price": "0.07816",
                "vol": "100",
            },
        }
        opened = {
            "success": True,
            "code": 0,
            "data": {
                "orderId": "ex-1",
                "externalOid": "oid-1",
                "state": 2,
                "dealVol": 0,
                "price": "0.07816",
                "vol": "100",
            },
        }
        absent = {"success": False, "code": 2040, "message": "Order does not exist", "data": None}
        empty = {"success": True, "code": 0, "data": None}
        pending = {
            "success": True,
            "code": 0,
            "data": {"orderId": "ex-1", "externalOid": "oid-1", "state": 1, "dealVol": 0, "price": "1", "vol": "100"},
        }
        unfilled = {
            "success": True,
            "code": 0,
            "data": {"orderId": "ex-1", "externalOid": "oid-1", "state": 2, "dealVol": 0, "price": "1", "vol": "100"},
        }
        http_500 = IOError("Error executing request GET https://api.mexc.com/api/v1/private/order/get/2040. HTTP status is 500.")
        return [
            ("ws_order", {"state": 2, "dealVol": 0, "price": "0.07816", "vol": "100", "orderId": "ex-1"}),
            ("status_error",),
            ("deal_reconcile", {"id": "r-1", "price": "0.07816", "vol": "10", "fee": "0", "orderId": "ex-1", "timestamp": 1_700_000_000_000}),
            ("lookup", empty, empty),
            ("lookup", filled, opened),
            ("lookup", absent, absent),
            ("ws_deal", {"id": "d-1", "price": "0.07816", "vol": "10", "fee": "0", "orderId": "ex-1"}),
            ("rest_status", opened, opened),
            ("cancel_read", {"state": 4, "dealVol": "10", "dealAvgPrice": "0.07816", "orderId": "ex-1"}),
            ("lookup", pending, unfilled),
            ("lookup", http_500, http_500),
        ]

    def _random_read_sequence(self, rng: random.Random):
        steps = []
        kinds = ("lookup", "ws_order", "ws_deal", "rest_status", "cancel_read", "status_error", "deal_reconcile")
        for _ in range(6):
            kind = rng.choice(kinds)
            if kind == "lookup":
                steps.append((kind, self._random_read(rng), self._random_read(rng)))
            elif kind == "ws_order":
                steps.append((kind, {
                    "state": rng.choice([1, 2, 3, 4]),
                    "dealVol": rng.choice([0, 34, 100]),
                    "dealAvgPrice": "0.07816",
                    "price": "0.07816",
                    "vol": "100",
                    "orderId": "ex-1",
                }))
            elif kind == "ws_deal":
                steps.append((kind, {
                    "id": f"d-{rng.randrange(1000)}",
                    "price": "0.07816",
                    "vol": rng.choice(["1", "10", "34"]),
                    "fee": "0",
                    "orderId": "ex-1",
                }))
            elif kind == "status_error":
                steps.append((kind,))
            elif kind == "deal_reconcile":
                steps.append((kind, {
                    "id": f"r-{rng.randrange(1000)}",
                    "price": "0.07816",
                    "vol": rng.choice(["1", "10"]),
                    "fee": "0",
                    "orderId": "ex-1",
                    "timestamp": 1_700_000_000_000,
                }))
            else:
                first = self._random_read(rng)
                second = self._random_read(rng)
                if isinstance(first, Exception) or isinstance(second, Exception):
                    steps.append(("lookup", first, second))
                else:
                    steps.append((kind, first, second))
        return steps

    def _random_read(self, rng: random.Random):
        kind = rng.randrange(6)
        if kind == 0:
            return IOError("Error executing request GET https://api.mexc.com/api/v1/private/order/get/2040. HTTP status is 500.")
        if kind == 1:
            return {"success": False, "code": 2040, "message": "Order does not exist", "data": None}
        if kind == 2:
            return {"success": True, "code": 0, "data": None}
        if kind == 3:
            return {"success": False, "code": 401, "message": "Not logged in", "data": None}
        if kind == 4:
            return IOError("connection reset")
        deal = rng.choice(["0", "34", "100"])
        state = rng.choice([1, 2, 3, 4, 5])
        return {
            "success": True,
            "code": 0,
            "data": {
                "orderId": "ex-1",
                "externalOid": "oid-1",
                "state": state,
                "dealVol": deal,
                "dealAvgPrice": "0.07816",
                "price": "0.07816",
                "vol": "100",
            },
        }

    @staticmethod
    def _fill_base(read) -> Decimal:
        if not isinstance(read, dict):
            return Decimal("0")
        data = read.get("data")
        if not isinstance(data, dict) or data.get("dealVol") in (None, ""):
            return Decimal("0")
        return Decimal(str(data["dealVol"])) * Decimal("10")

    @staticmethod
    def _read_is_unknown(read) -> bool:
        if isinstance(read, Exception) or not isinstance(read, dict):
            return True
        code = read.get("code")
        try:
            numeric = 0 if code in (None, "") else int(code)
        except (TypeError, ValueError):
            return True
        if numeric == 2040:
            return False
        if read.get("success") is False or numeric != 0:
            return True
        return False

    @staticmethod
    def _read_is_open_state(read) -> bool:
        if not isinstance(read, dict):
            return False
        data = read.get("data")
        if not isinstance(data, dict):
            return False
        try:
            return int(data.get("state") or 0) in (1, 2)
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _read_shows_live_order(read) -> bool:
        if not isinstance(read, dict):
            return False
        data = read.get("data")
        if not isinstance(data, dict):
            return False
        try:
            state = int(data.get("state") or 0)
            deal = Decimal(str(data.get("dealVol") or "0"))
        except Exception:
            return False
        return state in (1, 2, 3) or deal > 0

    async def test_cancel_push_records_deal_volume_before_the_order_is_done(self):
        order = self._track(Decimal("1000"))
        self.exchange._order_tracker.TRADE_FILLS_WAIT_TIMEOUT = 0.05
        self.exchange._process_order_event_message({
            "orderId": "ex-1",
            "externalOid": "oid-1",
            "state": 4,
            "dealVol": "34",
            "dealAvgPrice": "0.07816",
            "price": "0.07816",
            "vol": "100",
        })
        await asyncio.sleep(0.2)
        self.assertEqual(Decimal("340"), order.executed_amount_base)
        self.assertEqual(Decimal("0.07816"), order.average_executed_price)
        self.assertEqual(OrderState.CANCELED, order.current_state)

    async def test_filled_push_synthesizes_the_missing_fill_after_the_wait(self):
        order = self._track(Decimal("410"))
        self.exchange._order_tracker.TRADE_FILLS_WAIT_TIMEOUT = 0.05
        self.exchange._process_order_event_message({
            "orderId": "ex-1",
            "externalOid": "oid-1",
            "state": 3,
            "dealVol": "41",
            "dealAvgPrice": "0.07816",
        })
        await asyncio.sleep(0.3)
        self.assertEqual(Decimal("410"), order.executed_amount_base)
        self.assertEqual(OrderState.FILLED, order.current_state)

    async def test_maintenance_and_timeout_do_not_count_as_order_not_found(self):
        order = self._track()
        for code in (604, 801):
            await self.exchange._handle_update_error_for_active_order(
                order,
                IOError(f"Error: {code} - Futures API is under maintenance (retry)"),
            )
        await self.exchange._handle_update_error_for_active_order(order, asyncio.TimeoutError())
        self.assertEqual(0, self.exchange._order_tracker._order_not_found_records.get(order.client_order_id, 0))
        self.assertNotEqual(OrderState.FAILED, order.current_state)

    async def test_create_timeout_keeps_the_same_client_id_in_flight(self):
        self.exchange._position_mode_confirmed = True
        self.exchange._margin_mode_explicit = True
        self.exchange._contract_sizes[self.trading_pair] = Decimal("10")

        async def post(*args, **kwargs):
            raise asyncio.TimeoutError()

        async def get(*args, **kwargs):
            raise asyncio.TimeoutError()

        self.exchange._api_post = post
        self.exchange._api_get = get
        order_id = self.exchange.buy(
            trading_pair=self.trading_pair,
            amount=Decimal("410"),
            order_type=OrderType.LIMIT,
            price=Decimal("0.07816"),
            position_action=PositionAction.OPEN,
        )
        await asyncio.sleep(0.3)
        self.assertIn(order_id, self.exchange.in_flight_orders)
        self.assertNotEqual(OrderState.FAILED, self.exchange.in_flight_orders[order_id].current_state)

    async def test_orders_are_refused_until_position_mode_is_read(self):
        self.exchange._contract_sizes[self.trading_pair] = Decimal("10")
        self.exchange._margin_mode_explicit = True
        sent = []

        async def post(*args, **kwargs):
            sent.append(kwargs.get("data"))
            return {"success": True, "code": 0, "data": {"orderId": "1", "ts": 1}}

        self.exchange._api_post = post
        with self.assertRaises(IOError):
            await self.exchange._place_order(
                order_id="client-1",
                trading_pair=self.trading_pair,
                amount=Decimal("100"),
                trade_type=TradeType.BUY,
                order_type=OrderType.LIMIT,
                price=Decimal("0.08"),
                position_action=PositionAction.OPEN,
            )
        self.assertEqual([], sent)

    async def test_missing_contract_size_refuses_the_order(self):
        self.exchange._position_mode_confirmed = True
        sent = []

        async def post(*args, **kwargs):
            sent.append(kwargs.get("data"))
            return {"success": True, "code": 0, "data": {"orderId": "1", "ts": 1}}

        self.exchange._api_post = post
        with self.assertRaises(ValueError):
            await self.exchange._place_order(
                order_id="client-1",
                trading_pair=self.trading_pair,
                amount=Decimal("410"),
                trade_type=TradeType.BUY,
                order_type=OrderType.LIMIT,
                price=Decimal("0.08"),
                position_action=PositionAction.OPEN,
            )
        self.assertEqual([], sent)
        self.assertNotIn(Decimal("1"), self.exchange._contract_sizes.values())

    async def test_cancel_2041_reconciles_the_fill_and_does_not_stay_open(self):
        order = self._track(Decimal("100"))

        async def post(*args, **kwargs):
            return {
                "success": True,
                "code": 0,
                "data": [{"orderId": "ex-1", "errorCode": 2041, "errorMsg": "order state cannot be cancelled"}],
            }

        async def get(path_url, **kwargs):
            if "deal_details" in path_url:
                return {
                    "success": True,
                    "code": 0,
                    "data": [{
                        "id": "9",
                        "price": "2",
                        "vol": "4",
                        "fee": "0.1",
                        "feeCurrency": "USDT",
                        "timestamp": 1_700_000_000_000,
                    }],
                }
            return {
                "success": True,
                "code": 0,
                "data": {
                    "orderId": "ex-1",
                    "externalOid": "oid-1",
                    "state": 3,
                    "dealVol": "10",
                    "dealAvgPrice": "2",
                    "price": "2",
                    "vol": "10",
                },
            }

        self.exchange._api_post = post
        self.exchange._api_get = get
        await self.exchange._place_cancel("oid-1", order)
        await asyncio.sleep(0.2)
        self.assertEqual(Decimal("100"), order.executed_amount_base)
        self.assertEqual(OrderState.FILLED, order.current_state)
        self.assertTrue(self.exchange.is_cancel_confirmed("oid-1"))

    async def test_cancel_2041_already_cancelled_is_done(self):
        order = self._track(Decimal("100"))
        captured = []

        async def post(*args, **kwargs):
            captured.append(kwargs.get("data"))
            return {
                "success": True,
                "code": 0,
                "data": [{"orderId": "877759993012810880", "errorCode": 2041, "errorMsg": "already cancelled"}],
            }

        async def get(path_url, **kwargs):
            return {
                "success": True,
                "code": 0,
                "data": {
                    "orderId": "877759993012810880",
                    "externalOid": "oid-1",
                    "state": 4,
                    "dealVol": "0",
                    "dealAvgPrice": "0",
                    "price": "0.07816",
                    "vol": "10",
                },
            }

        order.update_exchange_order_id("877759993012810880")
        self.exchange._api_post = post
        self.exchange._api_get = get
        result = await self.exchange._place_cancel("oid-1", order)
        await asyncio.sleep(0.2)
        self.assertFalse(result)
        self.assertEqual(["877759993012810880"], captured[0])
        self.assertEqual(OrderState.CANCELED, order.current_state)
        self.assertTrue(self.exchange.is_cancel_confirmed("oid-1"))

    async def test_cancel_item_error_fails_and_leaves_the_order_open(self):
        order = self._track(Decimal("100"))
        order.update_exchange_order_id("877759993012810880")

        async def post(*args, **kwargs):
            return {
                "success": True,
                "code": 0,
                "data": [{"orderId": "877759993012810880", "errorCode": 600, "errorMsg": "Parameter error"}],
            }

        self.exchange._api_post = post
        with self.assertRaises(MexcResponseError) as caught:
            await self.exchange._place_cancel("oid-1", order)
        self.assertEqual(600, caught.exception.code)
        self.assertNotEqual(OrderState.CANCELED, order.current_state)
        self.assertFalse(self.exchange.is_cancel_confirmed("oid-1"))

    def _raw_post(self, mock_api, path: str) -> tuple:
        pattern = re.compile(f"^{re.escape(private_rest_url(path))}.*")
        calls = []
        for key, value in mock_api.requests.items():
            if pattern.search(str(key[1])):
                calls.extend(value)
        self.assertTrue(calls)
        call = calls[0]
        return call.kwargs["data"], call.kwargs["headers"]

    @aioresponses()
    async def test_cancel_wire_body_is_a_signed_json_array(self, mock_api):
        order = self._track(Decimal("100"))
        order.update_exchange_order_id("877759993012810880")
        mock_api.get(
            re.compile(r".*/api/v1/contract/ping.*"),
            body=json.dumps({"success": True, "code": 0, "data": 1700000000000}),
            repeat=True,
        )
        mock_api.post(
            re.compile(r".*/api/v1/private/order/cancel.*"),
            body=json.dumps({
                "success": True,
                "code": 0,
                "data": [{"orderId": "877759993012810880", "errorCode": 600, "errorMsg": "Parameter error"}],
            }),
        )
        with self.assertRaises(MexcResponseError) as caught:
            await self.exchange._place_cancel("oid-1", order)
        self.assertEqual(600, caught.exception.code)
        raw, headers = self._raw_post(mock_api, CONSTANTS.CANCEL_ORDER_PATH)
        self.assertEqual('["877759993012810880"]', raw)
        self.assertIsInstance(json.loads(raw), list)
        signed = f"key{headers['Request-Time']}{raw}"
        signature = hmac.new(b"secret", signed.encode("utf-8"), hashlib.sha256).hexdigest()
        self.assertEqual(signature, headers["Signature"])
        self.assertNotEqual(OrderState.CANCELED, order.current_state)

    @aioresponses()
    async def test_cancel_with_external_wire_body_is_a_signed_array_of_objects(self, mock_api):
        order = self._track(Decimal("100"))
        order.exchange_order_id = None
        mock_api.get(
            re.compile(r".*/api/v1/contract/ping.*"),
            body=json.dumps({"success": True, "code": 0, "data": 1700000000000}),
            repeat=True,
        )
        mock_api.post(
            re.compile(r".*/api/v1/private/order/cancel_with_external.*"),
            body=json.dumps({
                "success": True,
                "code": 0,
                "data": [{"orderId": None, "errorCode": 0, "errorMsg": "success"}],
            }),
        )
        client_path = CONSTANTS.ORDER_BY_EXTERNAL_PATH.format(symbol="AIOT_USDT", external_oid="oid-1")
        mock_api.get(
            re.compile(f"^{re.escape(private_rest_url(client_path))}.*"),
            body=json.dumps({
                "success": True,
                "code": 0,
                "data": {
                    "orderId": "1",
                    "externalOid": "oid-1",
                    "state": 4,
                    "dealVol": "0",
                    "dealAvgPrice": "0",
                    "price": "0.07816",
                    "vol": "10",
                },
            }),
        )
        mock_api.get(
            re.compile(r".*/api/v1/private/order/deal_details/.*"),
            body=json.dumps({"success": True, "code": 0, "data": []}),
            repeat=True,
        )
        await self.exchange._place_cancel("oid-1", order)
        raw, headers = self._raw_post(mock_api, CONSTANTS.CANCEL_BY_EXTERNAL_PATH)
        expected = json.dumps([{"symbol": "AIOT_USDT", "externalOid": "oid-1"}])
        self.assertEqual(expected, raw)
        self.assertIsInstance(json.loads(raw), list)
        signed = f"key{headers['Request-Time']}{raw}"
        signature = hmac.new(b"secret", signed.encode("utf-8"), hashlib.sha256).hexdigest()
        self.assertEqual(signature, headers["Signature"])

    async def test_stop_cancels_an_order_that_is_still_being_created(self):
        self.exchange._position_mode_confirmed = True
        self.exchange._margin_mode_explicit = True
        self.exchange._contract_sizes[self.trading_pair] = Decimal("10")
        release = asyncio.Event()
        started = asyncio.Event()
        calls = []

        async def post(path_url, data=None, **kwargs):
            calls.append((path_url, data))
            if path_url == CONSTANTS.PLACE_ORDER_PATH:
                started.set()
                await release.wait()
                return {"success": True, "code": 0, "data": {"orderId": "55", "ts": 1}}
            return {
                "success": True,
                "code": 0,
                "data": [{"orderId": 55, "errorCode": 0, "errorMsg": "success"}],
            }

        async def get(path_url, **kwargs):
            return {
                "success": True,
                "code": 0,
                "data": {
                    "orderId": "55",
                    "externalOid": "ignored",
                    "state": 4,
                    "dealVol": "0",
                    "dealAvgPrice": "0",
                    "price": "0.08",
                    "vol": "10",
                },
            }

        self.exchange._api_post = post
        self.exchange._api_get = get
        order_id = self.exchange.buy(
            trading_pair=self.trading_pair,
            amount=Decimal("100"),
            order_type=OrderType.LIMIT,
            price=Decimal("0.08"),
            position_action=PositionAction.OPEN,
        )
        await asyncio.wait_for(started.wait(), 1)
        self.exchange.cancel(self.trading_pair, order_id)
        release.set()
        for _ in range(50):
            order = self.exchange._order_tracker.fetch_order(order_id)
            if order is not None and order.current_state in (OrderState.CANCELED, OrderState.FAILED):
                break
            await asyncio.sleep(0.05)
        order = self.exchange._order_tracker.fetch_order(order_id)
        self.assertIsNotNone(order)
        self.assertIn(order.current_state, (OrderState.CANCELED, OrderState.FAILED))
        self.assertTrue(any(path == CONSTANTS.CANCEL_BY_EXTERNAL_PATH or path == CONSTANTS.CANCEL_ORDER_PATH for path, _data in calls))

    def test_fee_rebate_is_stored_as_a_negative_cost(self):
        order = self._track(Decimal("100"))
        rebate = self.exchange._parse_trade_update(
            {"id": "1", "price": "1", "vol": "1", "fee": "-0.01", "feeCurrency": "USDT", "timestamp": 1},
            order,
        )
        paid = self.exchange._parse_trade_update(
            {"id": "2", "price": "1", "vol": "1", "fee": "0.01", "feeCurrency": "USDT", "timestamp": 1},
            order,
        )
        self.assertEqual(Decimal("-0.01"), rebate.fee.flat_fees[0].amount)
        self.assertEqual(Decimal("0.01"), paid.fee.flat_fees[0].amount)

    async def test_websocket_balance_keeps_the_equity_gap(self):
        async def get(*args, **kwargs):
            return {
                "success": True,
                "code": 0,
                "data": [{
                    "currency": "USDT",
                    "availableBalance": "10",
                    "frozenBalance": "0",
                    "positionMargin": "0",
                    "equity": "15",
                }],
            }

        self.exchange._api_get = get
        await self.exchange._update_balances()
        self.assertEqual(Decimal("15"), self.exchange.get_balance("USDT"))
        self.exchange._process_wallet_event_message({
            "currency": "USDT",
            "availableBalance": "10",
            "frozenBalance": "0",
            "positionMargin": "0",
        })
        self.assertEqual(Decimal("15"), self.exchange.get_balance("USDT"))
        self.assertEqual(Decimal("10"), self.exchange.available_balances["USDT"])

    async def test_price_and_volume_are_json_numbers_and_use_contract_size_ten(self):
        self.exchange._position_mode_confirmed = True
        self.exchange._contract_sizes[self.trading_pair] = Decimal("10")
        self.exchange._perpetual_trading.set_leverage(self.trading_pair, 2)
        await self.exchange.set_margin_mode(MarginMode.ISOLATED)
        captured = {}

        async def post(path_url, data=None, **kwargs):
            captured["data"] = data
            return {"success": True, "code": 0, "data": {"orderId": "9", "ts": 1}}

        self.exchange._api_post = post
        await self.exchange._place_order(
            order_id="client-1",
            trading_pair=self.trading_pair,
            amount=Decimal("100"),
            trade_type=TradeType.BUY,
            order_type=OrderType.LIMIT,
            price=Decimal("0.085"),
            position_action=PositionAction.OPEN,
        )
        payload = captured["data"]
        self.assertIsInstance(payload["price"], (int, float))
        self.assertIsInstance(payload["vol"], (int, float))
        self.assertNotIsInstance(payload["price"], str)
        self.assertEqual(10, payload["vol"])
        self.assertEqual(Decimal("0.085"), Decimal(str(payload["price"])))
        self.assertEqual(2, payload["leverage"])
        self.assertEqual(CONSTANTS.OPEN_TYPE_ISOLATED, payload["openType"])

    async def test_leverage_change_sends_position_id_when_a_position_exists(self):
        self.exchange._position_mode_confirmed = True
        await self._hold_hedge_position(PositionSide.LONG, 1, CONSTANTS.OPEN_TYPE_CROSS, 42, hold_vol="10")
        self.exchange._perpetual_trading.set_position_mode(PositionMode.ONEWAY)
        await self.exchange.set_margin_mode(MarginMode.ISOLATED)
        captured = []

        async def post(path_url, data=None, **kwargs):
            captured.append(data)
            return {"success": True, "code": 0}

        self.exchange._api_post = post
        success, message = await self.exchange._set_trading_pair_leverage(self.trading_pair, 2)
        self.assertTrue(success, message)
        self.assertEqual(42, captured[0]["positionId"])
        self.assertEqual(2, captured[0]["leverage"])
        self.assertNotIn("openType", captured[0])

    async def test_confirm_leverage_and_margin_returns_false_when_the_exchange_rejects_it(self):
        self.exchange._position_mode_confirmed = True

        async def post(*args, **kwargs):
            raise IOError("Error: 2006 - Leverage multiplier must be within the upper limit")

        self.exchange._api_post = post
        confirmed = await self.exchange.confirm_leverage_and_margin(self.trading_pair, 2, MarginMode.ISOLATED)
        self.assertFalse(confirmed)
        self.assertNotEqual(2, self.exchange.get_leverage(self.trading_pair))

    async def _hold_hedge_position(self, side: PositionSide, leverage: int, open_type: int, position_id: int, hold_vol: str = "60770"):
        self.exchange._perpetual_trading.set_position_mode(PositionMode.HEDGE)
        self.exchange._contract_sizes[self.trading_pair] = Decimal("1")
        parsed = await self.exchange._parse_position({
            "symbol": "AIOT_USDT",
            "positionType": 1 if side is PositionSide.LONG else 2,
            "openType": open_type,
            "state": 1,
            "holdVol": hold_vol,
            "holdAvgPrice": "0.08",
            "leverage": leverage,
            "positionId": position_id,
            "unRealizedPnl": "0",
        })
        pos_key, position = parsed
        self.exchange._perpetual_trading.set_position(pos_key, position)
        return position

    async def test_set_leverage_omits_open_type_when_no_margin_mode_was_requested(self):
        self.exchange._position_mode_confirmed = True
        await self._hold_hedge_position(PositionSide.LONG, 1, CONSTANTS.OPEN_TYPE_CROSS, 42, hold_vol="10")
        self.exchange._perpetual_trading.set_position_mode(PositionMode.ONEWAY)
        await self.exchange.set_margin_mode(MarginMode.ISOLATED)
        captured = []

        async def post(path_url, data=None, **kwargs):
            captured.append(data)
            return {"success": True, "code": 0}

        self.exchange._api_post = post
        await self.exchange._execute_set_leverage(self.trading_pair, 4)
        self.assertEqual(1, len(captured))
        self.assertNotIn("openType", captured[0])
        self.assertEqual(4, captured[0]["leverage"])
        self.assertEqual(42, captured[0]["positionId"])
        self.assertEqual(4, self.exchange.get_leverage(self.trading_pair))

    async def test_set_leverage_refuses_when_the_position_is_not_loaded(self):
        self.exchange._position_mode_confirmed = True
        captured = []

        async def post(path_url, data=None, **kwargs):
            captured.append(data)
            return {"success": True, "code": 0}

        self.exchange._api_post = post
        confirmed = await self.exchange.confirm_leverage_and_margin(self.trading_pair, 2, None)
        self.assertFalse(confirmed)
        self.assertEqual([], captured)
        self.assertNotEqual(2, self.exchange.get_leverage(self.trading_pair))
        reason = self.exchange.leverage_confirmation_error()
        self.assertIn("AIOT-USDT LONG", reason)
        self.assertIn("open position is loaded", reason)
        self.assertIn("Refusing to trade", reason)

    async def test_hedge_leverage_read_skips_a_matching_long_and_does_not_change_the_short(self):
        self.exchange._position_mode_confirmed = True
        await self._hold_hedge_position(PositionSide.LONG, 2, CONSTANTS.OPEN_TYPE_ISOLATED, 101)
        await self._hold_hedge_position(PositionSide.SHORT, 5, CONSTANTS.OPEN_TYPE_CROSS, 202, hold_vol="10")
        captured = []

        async def post(path_url, data=None, **kwargs):
            captured.append(data)
            return {"success": True, "code": 0}

        self.exchange._api_post = post
        confirmed = await self.exchange.confirm_leverage_and_margin(
            self.trading_pair,
            2,
            None,
            position_side=PositionSide.LONG,
        )
        self.assertTrue(confirmed)
        self.assertEqual([], captured)
        self.assertEqual(2, self.exchange.get_leverage(self.trading_pair))

        confirmed = await self.exchange.confirm_leverage_and_margin(
            self.trading_pair,
            3,
            None,
            position_side=PositionSide.LONG,
        )
        self.assertTrue(confirmed)
        self.assertEqual(1, len(captured))
        self.assertEqual(101, captured[0]["positionId"])
        self.assertEqual(3, captured[0]["leverage"])
        self.assertNotIn("openType", captured[0])
        self.assertNotIn(202, captured[0].values())
        captured.clear()
        await self.exchange._execute_set_leverage(self.trading_pair, 6, PositionSide.SHORT)
        self.assertEqual([202], [item.get("positionId") for item in captured])
        self.assertNotIn("openType", captured[0])
        self.assertEqual(6, captured[0]["leverage"])

    async def test_hedge_set_leverage_for_an_empty_long_does_not_send_the_short(self):
        self.exchange._position_mode_confirmed = True
        self.exchange._perpetual_trading.set_position_mode(PositionMode.HEDGE)
        captured = []

        async def post(path_url, data=None, **kwargs):
            captured.append(data)
            return {"success": True, "code": 0}

        self.exchange._api_post = post
        success, message = await self.exchange._set_trading_pair_leverage(
            self.trading_pair,
            2,
            margin_mode=None,
            position_side=PositionSide.LONG,
        )
        self.assertFalse(success)
        self.assertIn("LONG", message)
        self.assertIn("open position is loaded", message)
        self.assertEqual([], captured)

        success, message = await self.exchange._set_trading_pair_leverage(
            self.trading_pair,
            2,
            margin_mode=None,
            position_side=None,
        )
        self.assertFalse(success)
        self.assertIn("position side", message)
        self.assertEqual([], captured)

    async def test_hedge_margin_switch_rejection_halts_without_touching_the_other_side(self):
        self.exchange._position_mode_confirmed = True
        self.exchange._margin_mode = MarginMode.CROSS
        await self._hold_hedge_position(PositionSide.LONG, 2, CONSTANTS.OPEN_TYPE_CROSS, 101)
        await self._hold_hedge_position(PositionSide.SHORT, 5, CONSTANTS.OPEN_TYPE_ISOLATED, 202, hold_vol="10")
        captured = []

        async def post(path_url, data=None, **kwargs):
            captured.append(data)
            raise IOError("Error: 2027 - There is only one cross or isolated in the same direction")

        self.exchange._api_post = post
        confirmed = await self.exchange.confirm_leverage_and_margin(
            self.trading_pair,
            2,
            MarginMode.ISOLATED,
            position_side=PositionSide.LONG,
        )
        self.assertFalse(confirmed)
        self.assertEqual(1, len(captured))
        self.assertEqual(101, captured[0]["positionId"])
        self.assertEqual(CONSTANTS.OPEN_TYPE_ISOLATED, captured[0]["openType"])
        self.assertNotIn(202, [item.get("positionId") for item in captured])
        reason = self.exchange.leverage_confirmation_error()
        self.assertIn("AIOT-USDT LONG", reason)
        self.assertIn("CROSS", reason)
        self.assertIn("ISOLATED", reason)
        self.assertIn("Refusing to trade", reason)
        self.assertEqual(MarginMode.CROSS, self.exchange._margin_mode)
        self.assertFalse(self.exchange._margin_mode_explicit)

    async def test_hedge_child_orders_use_the_side_for_that_position(self):
        self.exchange._position_mode_confirmed = True
        self.exchange._perpetual_trading.set_position_mode(PositionMode.HEDGE)
        self.exchange._contract_sizes[self.trading_pair] = Decimal("10")
        await self._hold_hedge_position(PositionSide.LONG, 2, CONSTANTS.OPEN_TYPE_CROSS, 101, hold_vol="6077")
        await self._hold_hedge_position(PositionSide.SHORT, 5, CONSTANTS.OPEN_TYPE_ISOLATED, 202, hold_vol="10")
        captured = []

        async def post(path_url, data=None, **kwargs):
            captured.append(data)
            return {"success": True, "code": 0, "data": {"orderId": str(len(captured)), "ts": 1}}

        self.exchange._api_post = post

        async def place(trade_type, position_action):
            await self.exchange._place_order(
                order_id=f"client-{len(captured) + 1}",
                trading_pair=self.trading_pair,
                amount=Decimal("100"),
                trade_type=trade_type,
                order_type=OrderType.LIMIT,
                price=Decimal("0.08"),
                position_action=position_action,
            )

        await place(TradeType.BUY, PositionAction.OPEN)
        await place(TradeType.SELL, PositionAction.OPEN)
        await place(TradeType.SELL, PositionAction.CLOSE)
        await place(TradeType.BUY, PositionAction.CLOSE)
        open_long, open_short, close_long, close_short = captured
        self.assertEqual(2, open_long["leverage"])
        self.assertEqual(5, open_short["leverage"])
        self.assertEqual(2, close_long["leverage"])
        self.assertEqual(5, close_short["leverage"])
        self.assertEqual(CONSTANTS.SIDE_OPEN_LONG, open_long["side"])
        self.assertEqual(CONSTANTS.POSITION_MODE_HEDGE, open_long["positionMode"])
        self.assertEqual(CONSTANTS.OPEN_TYPE_CROSS, open_long["openType"])
        self.assertNotIn("reduceOnly", open_long)
        self.assertNotIn("positionId", open_long)
        self.assertEqual(CONSTANTS.SIDE_OPEN_SHORT, open_short["side"])
        self.assertEqual(CONSTANTS.POSITION_MODE_HEDGE, open_short["positionMode"])
        self.assertEqual(CONSTANTS.OPEN_TYPE_ISOLATED, open_short["openType"])
        self.assertEqual(CONSTANTS.SIDE_CLOSE_LONG, close_long["side"])
        self.assertEqual(101, close_long["positionId"])
        self.assertEqual(CONSTANTS.POSITION_MODE_HEDGE, close_long["positionMode"])
        self.assertNotIn("reduceOnly", close_long)
        self.assertEqual(CONSTANTS.OPEN_TYPE_CROSS, close_long["openType"])
        self.assertEqual(CONSTANTS.SIDE_CLOSE_SHORT, close_short["side"])
        self.assertEqual(202, close_short["positionId"])
        self.assertEqual(CONSTANTS.POSITION_MODE_HEDGE, close_short["positionMode"])
        self.assertNotIn("reduceOnly", close_short)
        self.assertEqual(CONSTANTS.OPEN_TYPE_ISOLATED, close_short["openType"])

    async def test_order_is_refused_until_that_side_open_type_is_read(self):
        self.exchange._position_mode_confirmed = True
        self.exchange._perpetual_trading.set_position_mode(PositionMode.HEDGE)
        self.exchange._contract_sizes[self.trading_pair] = Decimal("10")
        captured = []

        async def post(path_url, data=None, **kwargs):
            captured.append(data)
            return {"success": True, "code": 0, "data": {"orderId": "1", "ts": 1}}

        self.exchange._api_post = post
        with self.assertRaises(IOError) as raised:
            await self.exchange._place_order(
                order_id="client-1",
                trading_pair=self.trading_pair,
                amount=Decimal("100"),
                trade_type=TradeType.BUY,
                order_type=OrderType.LIMIT,
                price=Decimal("0.08"),
                position_action=PositionAction.OPEN,
            )
        self.assertIn("openType", str(raised.exception))
        self.assertIn("LONG", str(raised.exception))
        self.assertEqual([], captured)
        self.exchange._position_margin_modes[(self.trading_pair, PositionSide.LONG)] = MarginMode.CROSS
        await self.exchange._place_order(
            order_id="client-2",
            trading_pair=self.trading_pair,
            amount=Decimal("100"),
            trade_type=TradeType.BUY,
            order_type=OrderType.LIMIT,
            price=Decimal("0.08"),
            position_action=PositionAction.OPEN,
        )
        self.assertEqual(CONSTANTS.OPEN_TYPE_CROSS, captured[0]["openType"])
        self.assertNotEqual(CONSTANTS.OPEN_TYPE_ISOLATED, captured[0]["openType"])

    async def test_hedge_leverage_without_a_side_changes_neither_side(self):
        self.exchange._position_mode_confirmed = True
        await self._hold_hedge_position(PositionSide.LONG, 2, CONSTANTS.OPEN_TYPE_ISOLATED, 101)
        await self._hold_hedge_position(PositionSide.SHORT, 5, CONSTANTS.OPEN_TYPE_CROSS, 202, hold_vol="10")
        captured = []

        async def post(path_url, data=None, **kwargs):
            captured.append(data)
            return {"success": True, "code": 0}

        self.exchange._api_post = post
        confirmed = await self.exchange.confirm_leverage_and_margin(self.trading_pair, 3, None)
        self.assertFalse(confirmed)
        self.assertEqual([], captured)
        self.assertIn("position side", self.exchange.leverage_confirmation_error())

    async def test_side_less_leverage_raises_until_hedge_mode_is_read(self):
        await self._hold_hedge_position(PositionSide.LONG, 2, CONSTANTS.OPEN_TYPE_ISOLATED, 101)
        await self._hold_hedge_position(PositionSide.SHORT, 5, CONSTANTS.OPEN_TYPE_CROSS, 202, hold_vol="10")
        # The read has not succeeded. The connector still defaults to one-way.
        self.exchange._perpetual_trading.set_position_mode(PositionMode.ONEWAY)
        self.exchange._position_mode_confirmed = False
        captured = []

        async def post(path_url, data=None, **kwargs):
            captured.append(data)
            return {"success": True, "code": 0}

        self.exchange._api_post = post
        with self.assertRaises(IOError) as unread:
            await self.exchange._execute_set_leverage(self.trading_pair, 9)
        self.assertIn("position mode", str(unread.exception))
        self.assertEqual([], captured)
        self.assertNotEqual(9, self.exchange.get_leverage(self.trading_pair))

        self.exchange._position_mode_confirmed = True
        self.exchange._perpetual_trading.set_position_mode(PositionMode.HEDGE)
        with self.assertRaises(IOError) as side_less:
            await self.exchange._execute_set_leverage(self.trading_pair, 9)
        self.assertIn("position side", str(side_less.exception))
        self.assertEqual([], captured)
        self.assertNotEqual(9, self.exchange.get_leverage(self.trading_pair))

        await self.exchange._execute_set_leverage(self.trading_pair, 2, PositionSide.LONG)
        self.assertEqual([], captured)
        self.assertEqual(2, self.exchange.get_leverage(self.trading_pair))

    async def test_explicit_margin_on_an_empty_hedge_side_sends_only_that_side(self):
        self.exchange._position_mode_confirmed = True
        self.exchange._perpetual_trading.set_position_mode(PositionMode.HEDGE)
        captured = []

        async def post(path_url, data=None, **kwargs):
            captured.append(data)
            return {"success": True, "code": 0}

        self.exchange._api_post = post
        success, message = await self.exchange._set_trading_pair_leverage(
            self.trading_pair,
            2,
            margin_mode=MarginMode.ISOLATED,
            position_side=PositionSide.LONG,
        )
        self.assertTrue(success, message)
        self.assertEqual(1, len(captured))
        self.assertEqual(CONSTANTS.POSITION_TYPE_LONG, captured[0]["positionType"])
        self.assertEqual(CONSTANTS.OPEN_TYPE_ISOLATED, captured[0]["openType"])
        self.assertNotEqual(CONSTANTS.POSITION_TYPE_SHORT, captured[0].get("positionType"))

    def _orders_open(self, read, error: Exception):
        self.exchange._position_mode_confirmed = True
        self.exchange._perpetual_trading.set_position_mode(PositionMode.HEDGE)
        self.exchange._api_get = read

        async def post(path_url, data=None, **kwargs):
            raise error

        self.exchange._api_post = post

    async def test_open_orders_accept_a_matching_leverage_read(self):
        async def get(path_url, params=None, **kwargs):
            self.assertEqual(CONSTANTS.POSITION_LEVERAGE_PATH, path_url)
            return {"data": [
                {"symbol": "AIOT_USDT", "positionType": 2, "openType": 2, "leverage": 8},
                {"symbol": "AIOT_USDT", "positionType": 1, "openType": 1, "leverage": 1},
            ]}

        self._orders_open(get, MexcResponseError(2019, "Leverage adjustment unavailable"))
        confirmed = await self.exchange.confirm_leverage_and_margin(
            self.trading_pair, 1, MarginMode.ISOLATED, position_side=PositionSide.LONG,
        )
        self.assertTrue(confirmed)
        self.assertEqual(1, self.exchange.get_leverage(self.trading_pair))
        self.assertEqual(MarginMode.ISOLATED, self.exchange._position_margin_modes[(self.trading_pair, PositionSide.LONG)])
        self.assertNotIn((self.trading_pair, PositionSide.SHORT), self.exchange._position_margin_modes)

    async def test_open_orders_refuse_when_the_live_leverage_differs(self):
        async def get(path_url, params=None, **kwargs):
            return {"data": [{"symbol": "AIOT_USDT", "positionType": 1, "openType": 2, "leverage": 5}]}

        self._orders_open(get, MexcResponseError(2019, "Leverage adjustment unavailable"))
        confirmed = await self.exchange.confirm_leverage_and_margin(
            self.trading_pair, 1, MarginMode.ISOLATED, position_side=PositionSide.LONG,
        )
        self.assertFalse(confirmed)
        self.assertFalse(self.exchange._leverage_explicit)
        reason = self.exchange.leverage_confirmation_error()
        self.assertIn("leverage 5", reason)
        self.assertIn("CROSS", reason)
        self.assertIn("Refusing to trade", reason)

    async def test_open_orders_refuse_when_the_leverage_read_fails(self):
        async def get(path_url, params=None, **kwargs):
            raise IOError("leverage read down")

        self._orders_open(get, IOError("Leverage adjustment unavailable while orders are open"))
        confirmed = await self.exchange.confirm_leverage_and_margin(
            self.trading_pair, 1, MarginMode.ISOLATED, position_side=PositionSide.LONG,
        )
        self.assertFalse(confirmed)
        self.assertFalse(self.exchange._leverage_explicit)
        self.assertIn("live leverage read failed", self.exchange.leverage_confirmation_error())

    async def test_other_leverage_errors_do_not_read_position_leverage(self):
        called = []

        async def get(path_url, params=None, **kwargs):
            called.append(path_url)
            return {"data": []}

        self._orders_open(get, MexcResponseError(2018, "Cannot exceed the maximum reducible margin 2019"))
        confirmed = await self.exchange.confirm_leverage_and_margin(
            self.trading_pair, 1, MarginMode.ISOLATED, position_side=PositionSide.LONG,
        )
        self.assertFalse(confirmed)
        self.assertEqual([], called)
        self.assertFalse(self.exchange._leverage_explicit)

    async def test_hedge_close_without_a_position_id_is_refused(self):
        self.exchange._position_mode_confirmed = True
        self.exchange._perpetual_trading.set_position_mode(PositionMode.HEDGE)
        self.exchange._contract_sizes[self.trading_pair] = Decimal("10")
        self.exchange._position_margin_modes[(self.trading_pair, PositionSide.LONG)] = MarginMode.CROSS
        captured = []

        async def post(path_url, data=None, **kwargs):
            captured.append(data)
            return {"success": True, "code": 0, "data": {"orderId": "1", "ts": 1}}

        self.exchange._api_post = post
        with self.assertRaises(IOError) as raised:
            await self.exchange._place_order(
                order_id="client-1",
                trading_pair=self.trading_pair,
                amount=Decimal("100"),
                trade_type=TradeType.SELL,
                order_type=OrderType.LIMIT,
                price=Decimal("0.09"),
                position_action=PositionAction.CLOSE,
            )
        self.assertIn("positionId", str(raised.exception))
        self.assertIn("LONG", str(raised.exception))
        self.assertEqual([], captured)
        with self.assertRaises(IOError):
            await self.exchange.place_plan_order(
                trading_pair=self.trading_pair,
                amount=Decimal("100"),
                trade_type=TradeType.SELL,
                trigger_price=Decimal("0.09"),
                trigger_type=CONSTANTS.TRIGGER_TYPE_GTE,
                price=Decimal("0.09"),
                position_action=PositionAction.CLOSE,
            )
        self.assertEqual([], captured)

    async def test_one_way_close_uses_buy_or_sell_side_and_reduce_only(self):
        self.exchange._position_mode_confirmed = True
        self.exchange._margin_mode_explicit = True
        self.exchange._perpetual_trading.set_position_mode(PositionMode.ONEWAY)
        self.exchange._contract_sizes[self.trading_pair] = Decimal("10")
        captured = []

        async def post(path_url, data=None, **kwargs):
            captured.append(data)
            return {"success": True, "code": 0, "data": {"orderId": str(len(captured)), "ts": 1}}

        self.exchange._api_post = post
        await self.exchange._place_order(
            order_id="client-sell",
            trading_pair=self.trading_pair,
            amount=Decimal("100"),
            trade_type=TradeType.SELL,
            order_type=OrderType.LIMIT,
            price=Decimal("0.09"),
            position_action=PositionAction.CLOSE,
        )
        await self.exchange._place_order(
            order_id="client-buy",
            trading_pair=self.trading_pair,
            amount=Decimal("100"),
            trade_type=TradeType.BUY,
            order_type=OrderType.LIMIT,
            price=Decimal("0.09"),
            position_action=PositionAction.CLOSE,
        )
        sell_close, buy_close = captured
        self.assertEqual(CONSTANTS.SIDE_OPEN_SHORT, sell_close["side"])
        self.assertTrue(sell_close["reduceOnly"])
        self.assertEqual(CONSTANTS.POSITION_MODE_ONEWAY, sell_close["positionMode"])
        self.assertEqual(CONSTANTS.SIDE_OPEN_LONG, buy_close["side"])
        self.assertTrue(buy_close["reduceOnly"])
        self.assertEqual(CONSTANTS.POSITION_MODE_ONEWAY, buy_close["positionMode"])

    async def test_hedge_exposure_reads_one_side_of_the_pair(self):
        self.exchange._perpetual_trading.set_position_mode(PositionMode.HEDGE)
        self.exchange._contract_sizes[self.trading_pair] = Decimal("10")

        async def get(path_url, params=None, **kwargs):
            if path_url == CONSTANTS.OPEN_POSITIONS_PATH:
                return {
                    "success": True,
                    "code": 0,
                    "data": [
                        {"symbol": "AIOT_USDT", "positionType": 1, "holdVol": 34, "leverage": 2},
                        {"symbol": "AIOT_USDT", "positionType": 2, "holdVol": 8, "leverage": 5},
                        {"symbol": "BTC_USDT", "positionType": 1, "holdVol": 5, "leverage": 2},
                    ],
                }
            return {
                "success": True,
                "code": 0,
                "data": [
                    {"symbol": "AIOT_USDT", "side": 1, "vol": 10, "dealVol": 0},
                    {"symbol": "AIOT_USDT", "side": 2, "vol": 7, "dealVol": 0},
                    {"symbol": "AIOT_USDT", "side": 3, "vol": 4, "dealVol": 0},
                    {"symbol": "AIOT_USDT", "side": 4, "vol": 5, "dealVol": 0},
                ],
            }

        self.exchange._api_get = get
        long_position, long_open = await self.exchange.ground_truth_exposure(
            self.trading_pair,
            TradeType.SELL,
            position_side=PositionSide.LONG,
        )
        short_position, short_open = await self.exchange.ground_truth_exposure(
            self.trading_pair,
            TradeType.BUY,
            position_side=PositionSide.SHORT,
        )
        self.assertEqual(Decimal("340"), long_position)
        self.assertEqual(Decimal("150"), long_open)
        self.assertEqual(Decimal("80"), short_position)
        self.assertEqual(Decimal("110"), short_open)

    async def test_plan_and_stop_cancels_use_the_documented_array_body(self):
        captured = {}

        async def post(path_url, data=None, **kwargs):
            captured[path_url] = data
            return {"success": True, "code": 0}

        self.exchange._api_post = post
        await self.exchange.cancel_plan_orders([{"symbol": "AIOT_USDT", "order_id": "900"}])
        await self.exchange.cancel_stop_orders(["77"])
        plan_body = captured[CONSTANTS.PLAN_ORDER_CANCEL_PATH]
        stop_body = captured[CONSTANTS.STOP_ORDER_CANCEL_PATH]
        self.assertIsInstance(plan_body, list)
        self.assertEqual({"symbol": "AIOT_USDT", "orderId": 900}, plan_body[0])
        self.assertIsInstance(stop_body, list)
        self.assertEqual({"stopPlanOrderId": 77}, stop_body[0])

    def test_funding_parser_handles_the_result_list_and_the_published_rate_snapshot(self):
        listed = {
            "success": True,
            "data": {
                "resultList": [{
                    "symbol": "AIOT_USDT",
                    "funding": "-1.5",
                    "rate": "0.0001",
                    "settleTime": 1_700_000_000_000,
                }]
            },
        }
        records = self.exchange._funding_records(listed)
        self.assertEqual("-1.5", str(records[0]["funding"]))
        sample = {
            "success": True,
            "code": 0,
            "data": {
                "symbol": "BTC_USDT",
                "fundingRate": -0.000019,
                "maxFundingRate": 0.0018,
                "minFundingRate": -0.0018,
                "collectCycle": 8,
                "nextSettleTime": 1761897600000,
                "timestamp": 1761887166142,
            },
        }
        parsed = self.exchange.parse_funding_response(sample)
        self.assertEqual("rate_snapshot", parsed["kind"])
        self.assertIsNone(parsed["payment"])
        self.assertEqual(Decimal("-0.000019"), parsed["rate"])
        self.assertEqual([], self.exchange._funding_records(sample))

    async def test_close_order_sends_position_id(self):
        self.exchange._position_mode_confirmed = True
        self.exchange._margin_mode_explicit = True
        self.exchange._contract_sizes[self.trading_pair] = Decimal("10")
        self.exchange._position_ids[(self.trading_pair, PositionSide.LONG)] = 99
        captured = {}

        async def post(path_url, data=None, **kwargs):
            captured["data"] = data
            return {"success": True, "code": 0, "data": {"orderId": "9", "ts": 1}}

        self.exchange._api_post = post
        await self.exchange._place_order(
            order_id="client-1",
            trading_pair=self.trading_pair,
            amount=Decimal("100"),
            trade_type=TradeType.SELL,
            order_type=OrderType.LIMIT,
            price=Decimal("0.09"),
            position_action=PositionAction.CLOSE,
        )
        self.assertEqual(99, captured["data"]["positionId"])
        self.assertEqual(CONSTANTS.SIDE_OPEN_SHORT, captured["data"]["side"])
        self.assertTrue(captured["data"]["reduceOnly"])

    async def test_mixed_contract_page_counts_only_this_symbol(self):
        self.exchange._contract_sizes[self.trading_pair] = Decimal("10")
        calls = []

        async def get(path_url, params=None, **kwargs):
            calls.append((path_url, params))
            if path_url == CONSTANTS.OPEN_POSITIONS_PATH:
                return {
                    "success": True,
                    "code": 0,
                    "data": [
                        {"symbol": "AIOT_USDT", "positionType": 1, "holdVol": 34},
                        {"symbol": "BTC_USDT", "positionType": 1, "holdVol": 5},
                    ],
                }
            return {
                "success": True,
                "code": 0,
                "data": [
                    {"symbol": "AIOT_USDT", "side": 1, "vol": 10, "dealVol": 0},
                    {"symbol": "AIOT_USDT", "side": 2, "vol": 7, "dealVol": 0},
                    {"symbol": "BTC_USDT", "side": 1, "vol": 10, "dealVol": 0},
                    {"symbol": "AIOT_USDT", "side": 4, "vol": 50, "dealVol": 0},
                ],
            }

        self.exchange._api_get = get
        position, open_size = await self.exchange.ground_truth_exposure(self.trading_pair, TradeType.BUY)
        self.assertEqual(Decimal("340"), position)
        self.assertEqual(Decimal("170"), open_size)
        order_calls = [call for call in calls if call[0] != CONSTANTS.OPEN_POSITIONS_PATH]
        self.assertEqual([(CONSTANTS.OPEN_ORDERS_PATH, {"page_num": 1, "page_size": CONSTANTS.OPEN_ORDERS_PAGE_SIZE})], order_calls)
        self.assertEqual({"symbol": "AIOT_USDT"}, calls[0][1])

    async def test_open_orders_are_counted_across_pages(self):
        self.exchange._contract_sizes[self.trading_pair] = Decimal("10")
        previous_size = CONSTANTS.OPEN_ORDERS_PAGE_SIZE
        CONSTANTS.OPEN_ORDERS_PAGE_SIZE = 2
        pages = {
            1: [
                {"symbol": "AIOT_USDT", "side": 1, "vol": 10, "dealVol": 0},
                {"symbol": "BTC_USDT", "side": 1, "vol": 10, "dealVol": 0},
            ],
            2: [
                {"symbol": "AIOT_USDT", "side": 1, "vol": 6, "dealVol": 1},
            ],
        }
        seen = []

        async def get(path_url, params=None, **kwargs):
            if path_url == CONSTANTS.OPEN_POSITIONS_PATH:
                return {"success": True, "code": 0, "data": []}
            seen.append(params)
            page = int((params or {}).get("page_num") or 0)
            return {"success": True, "code": 0, "data": pages.get(page, [])}

        self.exchange._api_get = get
        try:
            position, open_size = await self.exchange.ground_truth_exposure(self.trading_pair, TradeType.BUY)
        finally:
            CONSTANTS.OPEN_ORDERS_PAGE_SIZE = previous_size
        self.assertEqual(Decimal("0"), position)
        self.assertEqual(Decimal("150"), open_size)
        self.assertEqual([{"page_num": 1, "page_size": 2}, {"page_num": 2, "page_size": 2}], seen)

    async def test_open_orders_stop_when_the_documented_total_is_reached(self):
        self.exchange._contract_sizes[self.trading_pair] = Decimal("10")
        previous_size = CONSTANTS.OPEN_ORDERS_PAGE_SIZE
        CONSTANTS.OPEN_ORDERS_PAGE_SIZE = 2
        seen = []

        async def get(path_url, params=None, **kwargs):
            if path_url == CONSTANTS.OPEN_POSITIONS_PATH:
                return {"success": True, "code": 0, "data": []}
            seen.append(params)
            page = int((params or {}).get("page_num") or 0)
            if page == 1:
                rows = [
                    {"symbol": "AIOT_USDT", "side": 1, "vol": 1, "dealVol": 0},
                    {"symbol": "AIOT_USDT", "side": 1, "vol": 1, "dealVol": 0},
                ]
            elif page == 2:
                rows = [
                    {"symbol": "AIOT_USDT", "side": 1, "vol": 1, "dealVol": 0},
                    {"symbol": "BTC_USDT", "side": 1, "vol": 9, "dealVol": 0},
                ]
            else:
                rows = [{"symbol": "AIOT_USDT", "side": 1, "vol": 50, "dealVol": 0}]
            return {
                "success": True,
                "code": 0,
                "data": {"totalCount": 4, "totalPage": 2, "resultList": rows},
            }

        self.exchange._api_get = get
        try:
            position, open_size = await self.exchange.ground_truth_exposure(self.trading_pair, TradeType.BUY)
        finally:
            CONSTANTS.OPEN_ORDERS_PAGE_SIZE = previous_size
        self.assertEqual(Decimal("0"), position)
        self.assertEqual(Decimal("30"), open_size)
        self.assertEqual([{"page_num": 1, "page_size": 2}, {"page_num": 2, "page_size": 2}], seen)

    async def test_open_order_page_limit_fails_the_read(self):
        self.exchange._contract_sizes[self.trading_pair] = Decimal("10")
        previous_size = CONSTANTS.OPEN_ORDERS_PAGE_SIZE
        previous_pages = CONSTANTS.OPEN_ORDERS_MAX_PAGES
        CONSTANTS.OPEN_ORDERS_PAGE_SIZE = 1
        CONSTANTS.OPEN_ORDERS_MAX_PAGES = 2

        async def get(path_url, params=None, **kwargs):
            if path_url == CONSTANTS.OPEN_POSITIONS_PATH:
                return {"success": True, "code": 0, "data": []}
            return {
                "success": True,
                "code": 0,
                "data": [{"symbol": "AIOT_USDT", "side": 1, "vol": 1, "dealVol": 0}],
            }

        self.exchange._api_get = get
        try:
            with self.assertRaises(IOError):
                await self.exchange.ground_truth_exposure(self.trading_pair, TradeType.BUY)
        finally:
            CONSTANTS.OPEN_ORDERS_PAGE_SIZE = previous_size
            CONSTANTS.OPEN_ORDERS_MAX_PAGES = previous_pages
