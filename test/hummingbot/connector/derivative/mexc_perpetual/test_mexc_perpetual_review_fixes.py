import asyncio
import json
import random
from decimal import Decimal
from test.isolated_asyncio_wrapper_test_case import IsolatedAsyncioWrapperTestCase
from unittest.mock import AsyncMock

from aioresponses import aioresponses
from bidict import bidict

import hummingbot.connector.derivative.mexc_perpetual.mexc_perpetual_constants as CONSTANTS
from hummingbot.connector.derivative.mexc_perpetual.mexc_perpetual_derivative import (
    MarginMode,
    MexcPerpetualDerivative,
)
from hummingbot.connector.trading_rule import TradingRule
from hummingbot.core.data_type.common import OrderType, PositionAction, PositionSide, TradeType
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
        self.assertEqual("confirmed_absent", state)

    async def test_http_500_lookup_stays_unknown(self):
        async def get(path_url, **kwargs):
            raise IOError("http status is 500")

        self.exchange._api_get = get
        state = await self.exchange.resolve_unknown_order(self.trading_pair, "oid-1", "ex-1")
        self.assertIsNone(state)

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
            for client_read, order_read in sequence:
                reported = max(reported, self._fill_base(client_read), self._fill_base(order_read))
                seen_live = seen_live or self._read_shows_live_order(client_read) or self._read_shows_live_order(order_read)

                async def get(path_url, client_read=client_read, order_read=order_read, **kwargs):
                    chosen = order_read if "external" not in path_url else client_read
                    if isinstance(chosen, Exception):
                        raise chosen
                    return chosen

                self.exchange._api_get = get
                state = await self.exchange.resolve_unknown_order(self.trading_pair, "oid-1", "ex-1")
                self.assertGreaterEqual(order.executed_amount_base, reported)
                self.assertGreaterEqual(order.executed_amount_base, previous_base)
                previous_base = order.executed_amount_base
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
                if state in (OrderState.CANCELED, OrderState.FILLED) or (
                    state == "confirmed_absent" and not seen_live
                ):
                    child = max(Decimal("0"), total - order.executed_amount_base)
                    self.assertLessEqual(order.executed_amount_base + child, total)
                    if not resolved:
                        extra = child
                        resolved = True
                if not resolved:
                    self.assertLessEqual(max(order.amount, order.executed_amount_base), total)
                else:
                    self.assertLessEqual(extra, total)
            self.exchange._order_tracker._in_flight_orders.clear()
            if hasattr(self.exchange, "_max_fill_base"):
                self.exchange._max_fill_base.clear()
                self.exchange._max_fill_quote.clear()
                self.exchange._seen_live_client_ids.clear()

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
            (filled, opened),
            (absent, absent),
            (pending, unfilled),
            (http_500, http_500),
        ]

    def _random_read_sequence(self, rng: random.Random):
        steps = []
        for _ in range(6):
            steps.append((self._random_read(rng), self._random_read(rng)))
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

    async def test_stop_cancels_an_order_that_is_still_being_created(self):
        self.exchange._position_mode_confirmed = True
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
        self.exchange._position_ids[(self.trading_pair, PositionSide.LONG)] = 42
        await self.exchange.set_margin_mode(MarginMode.ISOLATED)
        captured = []

        async def post(path_url, data=None, **kwargs):
            captured.append(data)
            return {"success": True, "code": 0}

        self.exchange._api_post = post
        success, _message = await self.exchange._set_trading_pair_leverage(self.trading_pair, 2)
        self.assertTrue(success)
        self.assertEqual(42, captured[0]["positionId"])
        self.assertEqual(2, captured[0]["leverage"])
        self.assertEqual(CONSTANTS.OPEN_TYPE_ISOLATED, captured[0]["openType"])

    async def test_confirm_leverage_and_margin_returns_false_when_the_exchange_rejects_it(self):
        self.exchange._position_mode_confirmed = True

        async def post(*args, **kwargs):
            raise IOError("Error: 2006 - Leverage multiplier must be within the upper limit")

        self.exchange._api_post = post
        confirmed = await self.exchange.confirm_leverage_and_margin(self.trading_pair, 2, MarginMode.ISOLATED)
        self.assertFalse(confirmed)
        self.assertNotEqual(2, self.exchange.get_leverage(self.trading_pair))

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
        self.assertEqual(CONSTANTS.SIDE_CLOSE_LONG, captured["data"]["side"])
