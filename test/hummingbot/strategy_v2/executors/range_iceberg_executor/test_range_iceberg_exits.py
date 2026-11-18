from decimal import Decimal
from test.isolated_asyncio_wrapper_test_case import IsolatedAsyncioWrapperTestCase
from unittest.mock import MagicMock, PropertyMock

from pydantic import ValidationError

from hummingbot.connector.derivative.mexc_perpetual.mexc_perpetual_constants import (
    SIDE_CLOSE_LONG,
    SIDE_CLOSE_SHORT,
)
from hummingbot.connector.derivative.mexc_perpetual.mexc_perpetual_derivative import MexcPerpetualDerivative
from hummingbot.connector.trading_rule import TradingRule
from hummingbot.core.data_type.common import OrderType, PositionAction, PositionMode, PriceType, TradeType
from hummingbot.core.data_type.in_flight_order import InFlightOrder, OrderState
from hummingbot.strategy_v2.executors.range_iceberg_executor.data_types import (
    ExitArming,
    ExitExecution,
    ExitOrderType,
    RangeIcebergExecutorConfig,
    SizeUnit,
)
from hummingbot.strategy_v2.executors.range_iceberg_executor.range_iceberg_executor import RangeIcebergExecutor
from hummingbot.strategy_v2.models.base import RunnableStatus
from hummingbot.strategy_v2.models.executors import CloseType


class RangeIcebergExitTests(IsolatedAsyncioWrapperTestCase):
    """
    Take profit and stop loss for the range iceberg executor.

    A started stop-loss iceberg keeps closing after price returns to the take-profit
    side. An iceberg exit schedule starts at the trigger and does not wait for the
    first exit fill. The tests below lock that behavior.
    """

    def setUp(self):
        super().setUp()
        self.now = 1_000.0
        self.orders = {}
        self.buy_ids = []
        self.strategy = MagicMock()
        type(self.strategy).current_timestamp = PropertyMock(side_effect=lambda: self.now)
        self.connector = MagicMock()
        self.connector.trading_rules = {
            "AIOT-USDT": TradingRule(
                trading_pair="AIOT-USDT",
                min_order_size=Decimal("10"),
                min_price_increment=Decimal("0.00001"),
                min_base_amount_increment=Decimal("10"),
            )
        }
        self.connector.get_contract_size.return_value = Decimal("10")
        self.connector.get_price_by_type.return_value = Decimal("0.085")
        self.connector._order_tracker.fetch_order.side_effect = lambda client_order_id: self.orders.get(client_order_id)
        self.strategy.connectors = {"mexc_perpetual": self.connector}

        def place(connector_name, trading_pair, amount, order_type, price, position_action):
            order_id = f"buy-{len(self.buy_ids) + 1}"
            self.buy_ids.append((order_id, amount, price, order_type, position_action))
            self.orders[order_id] = InFlightOrder(
                client_order_id=order_id,
                trading_pair=trading_pair,
                order_type=order_type,
                trade_type=TradeType.BUY,
                amount=amount,
                price=price if price == price else Decimal("0"),
                initial_state=OrderState.OPEN,
                creation_timestamp=self.now,
            )
            return order_id

        self.strategy.buy.side_effect = place
        self.strategy.sell.side_effect = place
        self.strategy.cancel.return_value = None

    def _run(self, coro):
        return self.run_async_with_timeout(coro, 1)

    def _book(self, bid: Decimal, ask: Decimal):
        self.connector.get_price_by_type.side_effect = lambda pair, price_type: {
            PriceType.BestBid: bid,
            PriceType.BestAsk: ask,
            PriceType.MidPrice: (bid + ask) / Decimal("2"),
        }.get(price_type, ask)

    def _mid(self, price: Decimal):
        self._book(price, price)

    def _config(self, **overrides) -> RangeIcebergExecutorConfig:
        values = dict(
            timestamp=self.now,
            connector_name="mexc_perpetual",
            trading_pair="AIOT-USDT",
            side=TradeType.BUY,
            total_amount=Decimal("100"),
            size_unit=SizeUnit.BASE,
            price_low=Decimal("0.08"),
            price_high=Decimal("0.09"),
            levels=1,
            iceberg=False,
            min_order_interval=0,
            random_seed=1,
        )
        values.update(overrides)
        return RangeIcebergExecutorConfig(**values)

    def _executor(self, **overrides) -> RangeIcebergExecutor:
        executor = RangeIcebergExecutor(self.strategy, self._config(**overrides), update_interval=0.1)
        executor._status = RunnableStatus.RUNNING
        return executor

    def _fill(self, order_id: str, base: Decimal = None, price: Decimal = None):
        order = self.orders[order_id]
        base = order.amount if base is None else base
        price = order.price if price is None else price
        order.executed_amount_base = base
        order.executed_amount_quote = base * price
        order.current_state = OrderState.FILLED
        return base, price

    def _close_calls(self):
        calls = []
        for method, side in ((self.strategy.buy, TradeType.BUY), (self.strategy.sell, TradeType.SELL)):
            for call in method.call_args_list:
                if call.args[5] == PositionAction.CLOSE:
                    calls.append((side, call.args))
        return calls

    def _open_entry(self, executor: RangeIcebergExecutor):
        self._run(executor.control_task())
        order_id = self.buy_ids[-1][0]
        base, price = self._fill(order_id)
        return order_id, base, price

    def test_config_without_exits_completes_and_leaves_the_position_open(self):
        executor = self._executor()
        self.assertIsNone(executor.config.take_profit)
        self.assertIsNone(executor.config.stop_loss)
        self.assertEqual(ExitArming.ON_FULL_FILL, executor.config.exit_arming)
        order_id, _, _ = self._open_entry(executor)
        self._run(executor.control_task())
        self.assertEqual(CloseType.COMPLETED, executor.close_type)
        self.assertEqual([], self._close_calls())
        info = executor.get_custom_info()
        self.assertIsNone(info["take_profit"])
        self.assertIsNone(info["stop_loss"])
        self.assertEqual("idle", info["exit_status"])
        self.assertIsNone(info["exit_active"])
        self.assertEqual("0", info["exit_filled_amount"])
        self.assertEqual("0", info["realized_pnl_quote"])
        self.assertEqual(order_id, info["child_order_ids"][0])

    def test_take_profit_single_limit_closes_a_long_at_the_trigger(self):
        executor = self._executor(take_profit={"percent": "0.10", "execution": "SINGLE", "order_type": "LIMIT"})
        _, base, price = self._open_entry(executor)
        self._run(executor.control_task())
        closes = self._close_calls()
        self.assertEqual(1, len(closes))
        side, args = closes[0]
        self.assertEqual(TradeType.SELL, side)
        self.assertEqual(PositionAction.CLOSE, args[5])
        self.assertEqual(OrderType.LIMIT, args[3])
        self.assertEqual(base, args[2])
        self.assertEqual(price * Decimal("1.10"), args[4])
        self.assertEqual(RunnableStatus.RUNNING, executor.status)

        exit_id = self.buy_ids[-1][0]
        self._fill(exit_id, base, args[4])
        self._run(executor.control_task())
        self.assertEqual(CloseType.TAKE_PROFIT, executor.close_type)
        info = executor.get_custom_info()
        self.assertEqual("done", info["exit_status"])
        self.assertEqual("take_profit", info["exit_active"])
        self.assertEqual(Decimal(base), Decimal(info["exit_filled_amount"]))
        self.assertEqual(price * Decimal("1.10"), Decimal(info["exit_average_price"]))
        self.assertEqual(price * Decimal("1.10"), Decimal(info["take_profit_trigger_price"]))
        self.assertEqual((args[4] - price) * base, Decimal(info["realized_pnl_quote"]))
        self.assertEqual("0.10", info["take_profit"]["percent"])
        self.assertEqual("SINGLE", info["take_profit"]["execution"])

    def test_take_profit_single_market_waits_for_the_trigger(self):
        self._mid(Decimal("0.085"))
        executor = self._executor(take_profit={"percent": "0.10", "execution": "SINGLE", "order_type": "MARKET"})
        self._open_entry(executor)
        self._run(executor.control_task())
        self.assertEqual([], self._close_calls())
        self.assertEqual("armed", executor.get_custom_info()["exit_status"])

        self._mid(Decimal("0.090"))
        self._run(executor.control_task())
        side, args = self._close_calls()[0]
        self.assertEqual(TradeType.SELL, side)
        self.assertEqual(OrderType.MARKET, args[3])
        self.assertEqual(PositionAction.CLOSE, args[5])
        self.assertEqual(Decimal("100"), args[2])
        self.assertTrue(args[4].is_nan())

    def test_take_profit_single_limit_closes_a_short_with_a_buy(self):
        executor = self._executor(
            side=TradeType.SELL,
            take_profit={"price": "0.081", "execution": "SINGLE"},
        )
        _, base, entry_price = self._open_entry(executor)
        self._run(executor.control_task())
        side, args = self._close_calls()[0]
        self.assertEqual(TradeType.BUY, side)
        self.assertEqual(PositionAction.CLOSE, args[5])
        self.assertEqual(OrderType.LIMIT, args[3])
        self.assertEqual(Decimal("0.081"), args[4])
        self.assertEqual(base, args[2])
        self._fill(self.buy_ids[-1][0], base, args[4])
        self._run(executor.control_task())
        self.assertEqual(CloseType.TAKE_PROFIT, executor.close_type)
        info = executor.get_custom_info()
        self.assertEqual((entry_price - Decimal("0.081")) * base, Decimal(info["realized_pnl_quote"]))

    def test_stop_loss_single_market_closes_a_long(self):
        self._mid(Decimal("0.070"))
        executor = self._executor(stop_loss={"percent": "0.05", "execution": "SINGLE"})
        _, base, price = self._open_entry(executor)
        self._run(executor.control_task())
        side, args = self._close_calls()[0]
        self.assertEqual(TradeType.SELL, side)
        self.assertEqual(OrderType.MARKET, args[3])
        self.assertEqual(PositionAction.CLOSE, args[5])
        self.assertEqual(base, args[2])
        self.assertEqual(price * Decimal("0.95"), Decimal(executor.get_custom_info()["stop_loss_trigger_price"]))
        self.assertEqual(ExitOrderType.MARKET, executor.config.stop_loss.order_type)

    def test_stop_loss_single_market_closes_a_short(self):
        self._mid(Decimal("0.095"))
        executor = self._executor(side=TradeType.SELL, stop_loss={"percent": "0.05", "execution": "SINGLE"})
        self._open_entry(executor)
        self._run(executor.control_task())
        side, args = self._close_calls()[0]
        self.assertEqual(TradeType.BUY, side)
        self.assertEqual(OrderType.MARKET, args[3])
        self.assertEqual(PositionAction.CLOSE, args[5])
        self.assertEqual(Decimal("100"), args[2])
        self.assertEqual(Decimal("0.0945"), Decimal(executor.get_custom_info()["stop_loss_trigger_price"]))

    def test_take_profit_iceberg_reuses_the_range_and_shows_one_slice_at_a_time(self):
        self._mid(Decimal("0.089"))
        executor = self._executor(
            take_profit={
                "percent": "0.10",
                "execution": "ICEBERG",
                "slices": 2,
                "visible_count": 1,
                "price_low": "0.088",
                "price_high": "0.090",
            }
        )
        self._open_entry(executor)
        self._run(executor.control_task())
        self.assertEqual(1, len(self._close_calls()))
        _, first = self._close_calls()[0]
        self.assertEqual(TradeType.SELL, self._close_calls()[0][0])
        self.assertEqual(PositionAction.CLOSE, first[5])
        self.assertEqual(OrderType.LIMIT, first[3])
        self.assertGreaterEqual(first[4], Decimal("0.088"))
        self.assertLessEqual(first[4], Decimal("0.090"))
        self.assertLess(first[2], Decimal("100"))
        self._run(executor.control_task())
        self.assertEqual(1, len(self._close_calls()))

        self._fill(self.buy_ids[-1][0])
        self._run(executor.control_task())
        self.assertEqual(2, len(self._close_calls()))
        self.assertLessEqual(sum(call[1][2] for call in self._close_calls()), Decimal("100"))
        self._fill(self.buy_ids[-1][0])
        self._run(executor.control_task())
        self.assertEqual(CloseType.TAKE_PROFIT, executor.close_type)
        self.assertEqual(Decimal("100"), Decimal(executor.get_custom_info()["exit_filled_amount"]))

    def test_take_profit_iceberg_interval_holds_the_next_slice(self):
        self._mid(Decimal("0.089"))
        executor = self._executor(
            take_profit={
                "percent": "0.10",
                "execution": "ICEBERG",
                "slices": 2,
                "visible_count": 1,
                "interval": "10s",
                "limit_price": "0.088",
            }
        )
        self._open_entry(executor)
        self._run(executor.control_task())
        self.assertEqual(1, len(self._close_calls()))
        started = executor.get_custom_info()["exit"]["started_at"]
        self.assertEqual(self.now, started)
        self.assertEqual(Decimal("0"), Decimal(executor.get_custom_info()["exit_filled_amount"]))
        self.now += 3
        self._fill(self.buy_ids[-1][0])
        self._run(executor.control_task())
        self.assertEqual(started, executor.get_custom_info()["exit"]["started_at"])
        self.assertEqual(1, len(self._close_calls()))
        self.now += 7
        self._run(executor.control_task())
        self.assertEqual(2, len(self._close_calls()))
        self.assertGreaterEqual(self._close_calls()[1][1][4], Decimal("0.088"))

    def test_stop_loss_iceberg_closes_a_long_and_a_short(self):
        self._mid(Decimal("0.070"))
        long_executor = self._executor(
            stop_loss={
                "percent": "0.05",
                "execution": "ICEBERG",
                "slices": 2,
                "visible_count": 1,
                "price_low": "0.070",
                "price_high": "0.076",
            }
        )
        self._open_entry(long_executor)
        self._run(long_executor.control_task())
        side, args = self._close_calls()[0]
        self.assertEqual(TradeType.SELL, side)
        self.assertEqual(PositionAction.CLOSE, args[5])
        self.assertGreaterEqual(args[4], Decimal("0.070"))
        self.assertLessEqual(args[4], Decimal("0.076"))

        self.buy_ids.clear()
        self.strategy.buy.reset_mock()
        self.strategy.sell.reset_mock()
        self._mid(Decimal("0.095"))
        short_executor = self._executor(
            side=TradeType.SELL,
            stop_loss={
                "price": "0.094",
                "execution": "ICEBERG",
                "slices": 2,
                "visible_count": 1,
                "limit_price": "0.100",
            },
        )
        self._open_entry(short_executor)
        self._run(short_executor.control_task())
        side, args = self._close_calls()[0]
        self.assertEqual(TradeType.BUY, side)
        self.assertEqual(PositionAction.CLOSE, args[5])
        self.assertLessEqual(args[4], Decimal("0.100"))

    def test_full_fill_arming_waits_and_any_fill_arming_protects_the_partial(self):
        self._mid(Decimal("0.070"))
        waiting = self._executor(
            levels=1,
            stop_loss={"percent": "0.05", "execution": "SINGLE"},
            exit_arming="ON_FULL_FILL",
        )
        self._run(waiting.control_task())
        order = self.orders["buy-1"]
        order.executed_amount_base = Decimal("40")
        order.executed_amount_quote = Decimal("3.2")
        order.current_state = OrderState.PARTIALLY_FILLED
        self._run(waiting.control_task())
        self.assertEqual([], self._close_calls())
        self.assertEqual(0, self.strategy.cancel.call_count)
        self.assertIsNone(waiting.close_type)

        self.strategy.cancel.reset_mock()
        armed = self._executor(
            levels=1,
            stop_loss={"percent": "0.05", "execution": "SINGLE"},
            exit_arming="ON_ANY_FILL",
        )
        self._run(armed.control_task())
        partial = self.orders[self.buy_ids[-1][0]]
        partial.executed_amount_base = Decimal("40")
        partial.executed_amount_quote = Decimal("3.2")
        partial.current_state = OrderState.PARTIALLY_FILLED
        self._run(armed.control_task())
        self.strategy.cancel.assert_called_with("mexc_perpetual", "AIOT-USDT", self.buy_ids[-1][0])
        self.assertEqual([], self._close_calls())
        self._run(armed.control_task())
        _, args = self._close_calls()[0]
        self.assertEqual(Decimal("40"), args[2])
        self.assertEqual(OrderType.MARKET, args[3])
        self.assertEqual(PositionAction.CLOSE, args[5])
        self.assertLess(args[2], Decimal("100"))

    def test_trigger_cancels_a_working_entry_before_the_exit(self):
        self._mid(Decimal("0.085"))
        executor = self._executor(
            levels=2,
            iceberg=False,
            exit_arming="ON_ANY_FILL",
            stop_loss={"percent": "0.05", "execution": "SINGLE"},
        )
        self._run(executor.control_task())
        self._run(executor.control_task())
        self.assertEqual(2, self.strategy.buy.call_count)
        self._fill("buy-1")
        self._mid(Decimal("0.070"))
        self._run(executor.control_task())
        self.strategy.cancel.assert_called_with("mexc_perpetual", "AIOT-USDT", "buy-2")
        self.assertEqual([], self._close_calls())
        self._run(executor.control_task())
        self.assertEqual(1, len(self._close_calls()))
        self.assertEqual(self.orders["buy-1"].executed_amount_base, self._close_calls()[0][1][2])
        self.assertGreater(self._close_calls()[0][1][2], Decimal("0"))
        self.assertLess(self._close_calls()[0][1][2], Decimal("100"))

    def test_stop_replaces_an_unfilled_take_profit_and_the_iceberg_keeps_going(self):
        self._mid(Decimal("0.085"))
        executor = self._executor(
            take_profit={"percent": "0.10", "execution": "SINGLE", "order_type": "LIMIT"},
            stop_loss={
                "percent": "0.05",
                "execution": "ICEBERG",
                "slices": 2,
                "visible_count": 1,
                "price_low": "0.070",
                "price_high": "0.076",
            },
        )
        self._open_entry(executor)
        self._run(executor.control_task())
        tp_id = self.buy_ids[-1][0]
        self.assertEqual(Decimal("0.088"), self.orders[tp_id].price)
        self.assertEqual("take_profit", executor.get_custom_info()["exit_active"])

        self._mid(Decimal("0.070"))
        self._run(executor.control_task())
        self.strategy.cancel.assert_called_with("mexc_perpetual", "AIOT-USDT", tp_id)
        self.assertEqual(1, len(self._close_calls()))
        self._run(executor.control_task())
        self.assertEqual("stop_loss", executor.get_custom_info()["exit_active"])
        self.assertEqual(2, len(self._close_calls()))
        sl_price = self._close_calls()[1][1][4]
        self.assertGreaterEqual(sl_price, Decimal("0.070"))
        self.assertLessEqual(sl_price, Decimal("0.076"))

        self.strategy.cancel.reset_mock()
        self._mid(Decimal("0.090"))
        self._run(executor.control_task())
        self.strategy.cancel.assert_not_called()
        self.assertEqual(2, len(self._close_calls()))
        self.assertNotEqual(CloseType.TAKE_PROFIT, executor.close_type)

        self._fill(self.buy_ids[-1][0])
        self._run(executor.control_task())
        self.assertEqual(3, len(self._close_calls()))
        self.assertLessEqual(self._close_calls()[2][1][4], Decimal("0.076"))
        self._fill(self.buy_ids[-1][0])
        self._run(executor.control_task())
        self.assertEqual(CloseType.STOP_LOSS, executor.close_type)
        self.assertEqual("stop_loss", executor.get_custom_info()["exit_active"])

    def test_exit_size_never_exceeds_the_filled_position(self):
        self._mid(Decimal("0.089"))
        executor = self._executor(
            take_profit={
                "percent": "0.10",
                "execution": "ICEBERG",
                "slices": 4,
                "visible_count": 4,
                "price_low": "0.088",
                "price_high": "0.092",
            }
        )
        _, base, _ = self._open_entry(executor)
        for _ in range(6):
            self._run(executor.control_task())
            placed = sum(call[1][2] for call in self._close_calls())
            self.assertLessEqual(placed, base)
            self.assertGreater(placed, Decimal("0"))
        self.assertEqual(base, sum(call[1][2] for call in self._close_calls()))
        placed_before = len(self._close_calls())
        self._run(executor.control_task())
        self.assertEqual(placed_before, len(self._close_calls()))

    def test_early_stop_cancels_exit_orders_and_keep_position_is_unchanged(self):
        executor = self._executor(take_profit={"percent": "0.10", "execution": "SINGLE", "order_type": "LIMIT"})
        self._open_entry(executor)
        self._run(executor.control_task())
        exit_id = self.buy_ids[-1][0]
        executor.early_stop(keep_position=True)
        self.strategy.cancel.assert_called_with("mexc_perpetual", "AIOT-USDT", exit_id)
        self.assertEqual(CloseType.POSITION_HOLD, executor.close_type)
        self.assertEqual(RunnableStatus.TERMINATED, executor.status)

        self.strategy.cancel.reset_mock()
        other = self._executor(take_profit={"price": "0.091", "execution": "SINGLE"})
        self._open_entry(other)
        self._run(other.control_task())
        other_exit = self.buy_ids[-1][0]
        other.early_stop()
        self.strategy.cancel.assert_called_with("mexc_perpetual", "AIOT-USDT", other_exit)
        self.assertEqual(CloseType.EARLY_STOP, other.close_type)

    def test_restore_keeps_the_active_stop_and_the_remaining_size(self):
        self._mid(Decimal("0.085"))
        executor = self._executor(
            take_profit={"percent": "0.10", "execution": "SINGLE", "order_type": "LIMIT"},
            stop_loss={
                "percent": "0.05",
                "execution": "ICEBERG",
                "slices": 2,
                "visible_count": 1,
                "limit_price": "0.070",
            },
        )
        self._open_entry(executor)
        self._run(executor.control_task())
        snapshot = executor.get_custom_info()
        snapshot["exit"]["active"] = "stop_loss"
        snapshot["exit"]["status"] = "executing"
        snapshot["exit"]["trigger_price"] = snapshot["stop_loss_trigger_price"]
        snapshot["exit"]["order_ids"] = []
        snapshot["exit"]["filled_by_order"] = {}

        restored = self._executor(
            take_profit={"percent": "0.10", "execution": "SINGLE", "order_type": "LIMIT"},
            stop_loss={
                "percent": "0.05",
                "execution": "ICEBERG",
                "slices": 2,
                "visible_count": 1,
                "limit_price": "0.070",
            },
        )
        restored.restore_from_custom_info(snapshot)
        self._mid(Decimal("0.090"))
        before = len(self._close_calls())
        self._run(restored.control_task())
        new_closes = self._close_calls()[before:]
        self.assertGreaterEqual(len(new_closes), 1)
        self.assertEqual("stop_loss", restored.get_custom_info()["exit_active"])
        for _, args in new_closes:
            self.assertEqual(PositionAction.CLOSE, args[5])
            self.assertLessEqual(args[2], Decimal("100"))
            self.assertNotEqual(Decimal("0.088"), args[4])
        self.assertLessEqual(sum(args[2] for _, args in new_closes), Decimal("100"))

    def test_restore_reattaches_a_live_exit_and_does_not_close_twice(self):
        executor = self._executor(take_profit={"percent": "0.10", "execution": "SINGLE", "order_type": "LIMIT"})
        self._open_entry(executor)
        self._run(executor.control_task())
        exit_id = self.buy_ids[-1][0]
        self.orders[exit_id].executed_amount_base = Decimal("40")
        self.orders[exit_id].executed_amount_quote = Decimal("3.52")
        snapshot = executor.get_custom_info()

        restored = self._executor(take_profit={"percent": "0.10", "execution": "SINGLE", "order_type": "LIMIT"})
        restored.restore_from_custom_info(snapshot)
        before = len(self._close_calls())
        self._run(restored.control_task())
        self.assertEqual(before, len(self._close_calls()))
        self.assertEqual(Decimal("40"), Decimal(restored.get_custom_info()["exit_filled_amount"]))
        self.assertLessEqual(restored._position_base(), Decimal("100"))

    def test_exit_config_validation_and_example_json(self):
        legacy = RangeIcebergExecutorConfig(
            timestamp=self.now,
            connector_name="mexc_perpetual",
            trading_pair="AIOT-USDT",
            side="BUY",
            total_amount="10000",
            price_low="0.08",
            price_high="0.09",
        )
        self.assertIsNone(legacy.take_profit)
        self.assertIsNone(legacy.stop_loss)

        example = {
            "timestamp": self.now,
            "connector_name": "mexc_perpetual",
            "trading_pair": "AIOT-USDT",
            "side": "BUY",
            "total_amount": "10000",
            "size_unit": "QUOTE",
            "mode": "RANGE",
            "price_low": "0.08000",
            "price_high": "0.09000",
            "levels": 10,
            "iceberg": True,
            "visible_count": 2,
            "leverage": 5,
            "margin_mode": "ISOLATED",
            "exit_arming": "ON_FULL_FILL",
            "take_profit": {
                "percent": "0.02",
                "execution": "SINGLE",
                "order_type": "LIMIT",
            },
            "stop_loss": {
                "percent": "0.01",
                "execution": "ICEBERG",
                "slices": 4,
                "visible_count": 1,
                "interval": "30s",
                "limit_price": "0.07000",
            },
        }
        parsed = RangeIcebergExecutorConfig(**example)
        self.assertEqual(ExitExecution.SINGLE, parsed.take_profit.execution)
        self.assertEqual(ExitOrderType.LIMIT, parsed.take_profit.order_type)
        self.assertEqual(ExitExecution.ICEBERG, parsed.stop_loss.execution)
        self.assertEqual(Decimal("30"), Decimal(str(parsed.stop_loss.interval_seconds)))
        self.assertEqual(ExitArming.ON_FULL_FILL, parsed.exit_arming)

        with self.assertRaises(ValidationError):
            self._config(take_profit={"percent": "0.01", "price": "0.09"})
        with self.assertRaises(ValidationError):
            self._config(reduce_only=True, side="SELL", take_profit={"percent": "0.01"})
        with self.assertRaises(ValidationError):
            self._config(stop_loss={"percent": "0.01", "execution": "SINGLE", "order_type": "LIMIT"})
        with self.assertRaises(ValidationError):
            self._config(take_profit={"percent": "0.01", "execution": "ICEBERG", "slices": 2})
        with self.assertRaises(ValidationError):
            self._config(
                take_profit={"price": "0.07"},
                stop_loss={"price": "0.08", "execution": "SINGLE"},
            )

    def test_close_long_and_close_short_use_the_mexc_hedge_sides(self):
        connector = MagicMock()
        connector.position_mode = PositionMode.HEDGE
        self.assertEqual(
            SIDE_CLOSE_LONG,
            MexcPerpetualDerivative._exchange_side(connector, TradeType.SELL, PositionAction.CLOSE),
        )
        self.assertEqual(
            SIDE_CLOSE_SHORT,
            MexcPerpetualDerivative._exchange_side(connector, TradeType.BUY, PositionAction.CLOSE),
        )
        self.assertEqual(4, SIDE_CLOSE_LONG)
        self.assertEqual(2, SIDE_CLOSE_SHORT)
