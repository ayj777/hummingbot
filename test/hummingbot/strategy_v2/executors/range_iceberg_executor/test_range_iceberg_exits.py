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
from types import SimpleNamespace

from hummingbot.core.data_type.common import OrderType, PositionAction, PositionMode, PositionSide, PriceType, TradeType
from hummingbot.core.event.events import MarketOrderFailureEvent
from hummingbot.core.data_type.in_flight_order import InFlightOrder, OrderState
from hummingbot.connector.derivative.position import Position
from hummingbot.strategy_v2.executors.range_iceberg_executor.data_types import (
    ExecutionMode,
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
        self.assertEqual("0.1", info["take_profit"]["percent"])
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

    def test_trigger_prices_follow_the_average_fill_and_are_null_before_a_fill(self):
        bare = self._executor()
        self.assertIsNone(bare.get_custom_info()["take_profit_trigger_price"])
        self.assertIsNone(bare.get_custom_info()["stop_loss_trigger_price"])
        self.assertIsNone(bare.get_custom_info()["exchange_stop_trigger_price"])
        self._open_entry(bare)
        self.assertIsNone(bare.get_custom_info()["take_profit_trigger_price"])
        self.assertIsNone(bare.get_custom_info()["stop_loss_trigger_price"])
        self.assertIsNone(bare.get_custom_info()["exchange_stop_trigger_price"])

        executor = self._executor(
            take_profit={"percent": "0.10"},
            stop_loss={"percent": "0.05"},
        )
        self.assertIsNone(executor.get_custom_info()["take_profit_trigger_price"])
        self.assertIsNone(executor.get_custom_info()["stop_loss_trigger_price"])
        self.assertIsNone(executor.get_custom_info()["exchange_stop_trigger_price"])
        self._run(executor.control_task())
        order_id = self.buy_ids[-1][0]
        self._fill(order_id, Decimal("40"), Decimal("0.08"))
        info = executor.get_custom_info()
        self.assertEqual(Decimal("0.088"), Decimal(info["take_profit_trigger_price"]))
        self.assertEqual(Decimal("0.076"), Decimal(info["stop_loss_trigger_price"]))
        order = self.orders[order_id]
        order.executed_amount_base = Decimal("80")
        order.executed_amount_quote = Decimal("7.2")
        info = executor.get_custom_info()
        self.assertEqual(Decimal("0.099"), Decimal(info["take_profit_trigger_price"]))
        self.assertEqual(Decimal("0.0855"), Decimal(info["stop_loss_trigger_price"]))
        self.assertEqual(Decimal(info["filled_base"]) - Decimal(info["closed_base"]), Decimal(info["open_base"]))

        absolute = self._executor(take_profit={"price": "0.091"}, stop_loss={"price": "0.07"})
        self.assertIsNone(absolute.get_custom_info()["take_profit_trigger_price"])
        self.assertIsNone(absolute.get_custom_info()["stop_loss_trigger_price"])
        self._run(absolute.control_task())
        self._fill(self.buy_ids[-1][0])
        info = absolute.get_custom_info()
        self.assertEqual("0.091", info["take_profit_trigger_price"])
        self.assertEqual("0.07", info["stop_loss_trigger_price"])

    def test_on_any_fill_limit_take_profit_keeps_the_entry_and_grows_the_close(self):
        executor = self._executor(
            levels=2,
            exit_arming="ON_ANY_FILL",
            take_profit={"price": "0.10", "execution": "SINGLE", "order_type": "LIMIT"},
        )
        self._run(executor.control_task())
        self._run(executor.control_task())
        self.assertEqual(2, self.strategy.buy.call_count)
        first = self.orders["buy-1"]
        first.executed_amount_base = Decimal("40")
        first.executed_amount_quote = Decimal("40") * first.price
        first.current_state = OrderState.PARTIALLY_FILLED
        self.strategy.cancel.reset_mock()
        self._run(executor.control_task())
        self.strategy.cancel.assert_not_called()
        closes = self._close_calls()
        self.assertEqual(1, len(closes))
        self.assertEqual(TradeType.SELL, closes[0][0])
        self.assertEqual(OrderType.LIMIT, closes[0][1][3])
        self.assertEqual(PositionAction.CLOSE, closes[0][1][5])
        self.assertEqual(Decimal("40"), closes[0][1][2])
        self.assertEqual("RESTING", executor.get_custom_info()["exit_state"])
        self.assertEqual(RunnableStatus.RUNNING, executor.status)
        self.assertIsNone(executor.close_type)
        self.assertTrue(executor._slices[1].is_working)

        first.executed_amount_base = first.amount
        first.executed_amount_quote = first.amount * first.price
        first.current_state = OrderState.FILLED
        self._run(executor.control_task())
        closes = self._close_calls()
        self.assertGreaterEqual(len(closes), 2)
        self.assertEqual(first.amount, sum(call[1][2] for call in closes))
        self.assertLessEqual(sum(call[1][2] for call in closes), executor.filled_base)
        self.assertIn(self.buy_ids[-1][0], executor.get_custom_info()["exit_order_ids"])

    def test_stop_loss_replaces_a_partially_filled_limit_take_profit(self):
        self._mid(Decimal("0.085"))
        executor = self._executor(
            take_profit={"percent": "0.10", "execution": "SINGLE", "order_type": "LIMIT"},
            stop_loss={"percent": "0.05", "execution": "SINGLE"},
        )
        self._open_entry(executor)
        self._run(executor.control_task())
        tp_id = self.buy_ids[-1][0]
        self.orders[tp_id].executed_amount_base = Decimal("50")
        self.orders[tp_id].executed_amount_quote = Decimal("50") * self.orders[tp_id].price
        self.orders[tp_id].current_state = OrderState.PARTIALLY_FILLED
        self._mid(Decimal("0.04"))
        self.strategy.cancel.reset_mock()
        self._run(executor.control_task())
        self.strategy.cancel.assert_called_with("mexc_perpetual", "AIOT-USDT", tp_id)
        self.assertEqual("stop_loss", executor.get_custom_info()["exit_active"])
        self._run(executor.control_task())
        markets = [call for call in self._close_calls() if call[1][3] == OrderType.MARKET]
        self.assertEqual(1, len(markets))
        self.assertEqual(Decimal("50"), markets[0][1][2])
        self.assertEqual(PositionAction.CLOSE, markets[0][1][5])

    def test_stop_loss_overrides_a_resting_take_profit_iceberg(self):
        self._mid(Decimal("0.10"))
        executor = self._executor(
            take_profit={
                "percent": "0.10",
                "execution": "ICEBERG",
                "slices": 2,
                "visible_count": 1,
                "limit_price": "0.10",
            },
            stop_loss={"percent": "0.05", "execution": "SINGLE"},
        )
        self._open_entry(executor)
        self._run(executor.control_task())
        self.assertEqual(1, len(self._close_calls()))
        self.assertEqual(OrderType.LIMIT, self._close_calls()[0][1][3])
        tp_id = self.buy_ids[-1][0]
        self._mid(Decimal("0.04"))
        self._run(executor.control_task())
        self.strategy.cancel.assert_called_with("mexc_perpetual", "AIOT-USDT", tp_id)
        self._run(executor.control_task())
        self.assertEqual("stop_loss", executor.get_custom_info()["exit_active"])
        markets = [call for call in self._close_calls() if call[1][3] == OrderType.MARKET]
        self.assertEqual(1, len(markets))
        self.assertEqual(Decimal("100"), markets[0][1][2])

    def test_partial_fill_at_duration_end_arms_exits_instead_of_stopping(self):
        self._mid(Decimal("0.085"))
        executor = self._executor(
            duration=10,
            schedule_slices=1,
            schedule_start="SUBMIT",
            stop_loss={"percent": "0.05", "execution": "SINGLE"},
        )
        self._run(executor.control_task())
        order = self.orders[self.buy_ids[-1][0]]
        order.executed_amount_base = Decimal("40")
        order.executed_amount_quote = Decimal("40") * order.price
        order.current_state = OrderState.PARTIALLY_FILLED
        self.now += 10
        self.strategy.cancel.reset_mock()
        self._run(executor.control_task())
        self.assertIsNone(executor.close_type)
        self.assertEqual(RunnableStatus.RUNNING, executor.status)
        self.assertEqual("ARMED", executor.get_custom_info()["exit_state"])
        self.assertEqual([], self._close_calls())
        self.strategy.cancel.assert_called()
        self._mid(Decimal("0.07"))
        self._run(executor.control_task())
        self._run(executor.control_task())
        self.assertEqual(1, len(self._close_calls()))
        self.assertEqual(Decimal("40"), self._close_calls()[0][1][2])
        self.assertEqual(OrderType.MARKET, self._close_calls()[0][1][3])

    def test_partial_fill_at_expiry_arms_exits_instead_of_stopping(self):
        self._mid(Decimal("0.085"))
        executor = self._executor(
            expiry_seconds=5,
            stop_loss={"percent": "0.05", "execution": "SINGLE"},
        )
        self._run(executor.control_task())
        order = self.orders[self.buy_ids[-1][0]]
        order.executed_amount_base = Decimal("40")
        order.executed_amount_quote = Decimal("40") * order.price
        order.current_state = OrderState.PARTIALLY_FILLED
        self.now += 5
        self._run(executor.control_task())
        self.assertNotEqual(CloseType.EXPIRED, executor.close_type)
        self.assertEqual(RunnableStatus.RUNNING, executor.status)
        self.assertEqual("ARMED", executor.get_custom_info()["exit_state"])
        self.assertEqual(Decimal("40"), Decimal(executor.get_custom_info()["open_base"]))

    def test_armed_exit_is_not_killed_by_the_entry_time_limit(self):
        self._mid(Decimal("0.085"))
        executor = self._executor(
            duration=30,
            schedule_slices=1,
            expiry_seconds=30,
            stop_loss={"percent": "0.05", "execution": "SINGLE"},
        )
        self._open_entry(executor)
        self._run(executor.control_task())
        self.assertEqual("ARMED", executor.get_custom_info()["exit_state"])
        self.assertEqual([], self._close_calls())
        self.now += 31
        self._run(executor.control_task())
        self.assertIsNone(executor.close_type)
        self.assertEqual(RunnableStatus.RUNNING, executor.status)
        self.assertEqual("ARMED", executor.get_custom_info()["exit_state"])
        self.assertEqual([], self._close_calls())

        market = self._executor(
            duration=30,
            schedule_slices=1,
            take_profit={"percent": "0.10", "execution": "SINGLE", "order_type": "MARKET"},
        )
        self._open_entry(market)
        self._run(market.control_task())
        self.assertEqual("ARMED", market.get_custom_info()["exit_state"])
        self.now += 31
        self._run(market.control_task())
        self.assertIsNone(market.close_type)
        self.assertEqual(RunnableStatus.RUNNING, market.status)
        self.assertEqual([], [call for call in self._close_calls() if call[1][3] == OrderType.MARKET])

    def test_exchange_stop_survives_keep_position_and_is_cancelled_when_take_profit_finishes(self):
        placed = []
        cancelled = []

        async def place_stop(trading_pair, position_id, amount, stop_loss_price=None, **kwargs):
            placed.append((trading_pair, str(position_id), amount, stop_loss_price))
            return "stop-1"

        async def cancel_stop(order_ids):
            cancelled.extend(order_ids)
            return {}

        self.connector.place_position_stop_order = place_stop
        self.connector.cancel_stop_orders = cancel_stop
        self.connector._position_id_for_order = lambda trading_pair, trade_type, position_action, reduce_only: 4242

        self._mid(Decimal("0.085"))
        executor = self._executor(stop_loss={"percent": "0.05", "execution": "SINGLE"})
        _, base, price = self._open_entry(executor)
        self.assertIsNone(executor.get_custom_info()["exchange_stop_order_id"])
        self._run(executor.control_task())
        info = executor.get_custom_info()
        self.assertEqual("ARMED", info["exit_state"])
        self.assertEqual("stop-1", info["exchange_stop_order_id"])
        self.assertEqual("4242", placed[0][1])
        self.assertEqual(base, placed[0][2])
        software = price * Decimal("0.95")
        self.assertEqual(software * (Decimal("1") - executor.config.protective_stop_buffer_pct), placed[0][3])
        self.assertEqual(str(placed[0][3]), info["exchange_stop_trigger_price"])
        executor.early_stop(keep_position=True)
        self.assertEqual([], cancelled)
        self.assertEqual("stop-1", executor.get_custom_info()["exchange_stop_order_id"])
        self.assertEqual(CloseType.POSITION_HOLD, executor.close_type)
        snapshot = executor.get_custom_info()

        restored = self._executor(stop_loss={"percent": "0.05", "execution": "SINGLE"})
        restored.restore_from_custom_info(snapshot)
        self.assertEqual("ARMED", restored.get_custom_info()["exit_state"])
        self._run(restored.control_task())
        self.assertEqual(1, len(placed))
        self.assertEqual("stop-1", restored.get_custom_info()["exchange_stop_order_id"])

        disabled = self._executor(
            stop_loss={"percent": "0.05", "execution": "SINGLE"},
            protective_exchange_stop=False,
        )
        self._open_entry(disabled)
        self._run(disabled.control_task())
        self.assertEqual(1, len(placed))
        self.assertIsNone(disabled.get_custom_info()["exchange_stop_order_id"])

        both = self._executor(
            take_profit={"percent": "0.10", "execution": "SINGLE", "order_type": "LIMIT"},
            stop_loss={"percent": "0.05", "execution": "SINGLE"},
        )
        self._open_entry(both)
        self._run(both.control_task())
        self.assertEqual("stop-1", both.get_custom_info()["exchange_stop_order_id"])
        # The helper returns the same id. A second position still records a place.
        self.assertGreaterEqual(len(placed), 2)
        tp_id = self.buy_ids[-1][0]
        self._fill(tp_id)
        self._run(both.control_task())
        self.assertEqual(CloseType.TAKE_PROFIT, both.close_type)
        self.assertIn("stop-1", cancelled)
        self.assertIsNone(both.get_custom_info()["exchange_stop_order_id"])

    def test_restore_resumes_armed_resting_and_closing(self):
        resting = self._executor(take_profit={"percent": "0.10", "execution": "SINGLE", "order_type": "LIMIT"})
        self._open_entry(resting)
        self._run(resting.control_task())
        self.assertEqual("RESTING", resting.get_custom_info()["exit_state"])
        snapshot = resting.get_custom_info()
        resumed = self._executor(take_profit={"percent": "0.10", "execution": "SINGLE", "order_type": "LIMIT"})
        resumed.restore_from_custom_info(snapshot)
        self.assertEqual("RESTING", resumed.get_custom_info()["exit_state"])
        before = len(self._close_calls())
        self._run(resumed.control_task())
        self.assertEqual(before, len(self._close_calls()))
        self.assertEqual(snapshot["exit_order_ids"], resumed.get_custom_info()["exit_order_ids"])

        exit_id = snapshot["exit_order_ids"][0]
        self.orders[exit_id].executed_amount_base = Decimal("40")
        self.orders[exit_id].executed_amount_quote = Decimal("3.52")
        self.orders[exit_id].current_state = OrderState.PARTIALLY_FILLED
        self._run(resting.control_task())
        self.assertEqual("CLOSING", resting.get_custom_info()["exit_state"])
        closing_snapshot = resting.get_custom_info()
        closer = self._executor(take_profit={"percent": "0.10", "execution": "SINGLE", "order_type": "LIMIT"})
        closer.restore_from_custom_info(closing_snapshot)
        self.assertEqual("CLOSING", closer.get_custom_info()["exit_state"])
        before = len(self._close_calls())
        self._run(closer.control_task())
        self.assertEqual(before, len(self._close_calls()))
        self.assertEqual(Decimal("40"), Decimal(closer.get_custom_info()["closed_base"]))
        self.assertEqual(Decimal("60"), Decimal(closer.get_custom_info()["open_base"]))

        self._mid(Decimal("0.085"))
        armed = self._executor(stop_loss={"percent": "0.05", "execution": "SINGLE"})
        armed.restore_from_custom_info(
            {
                "exit_state": "ARMED",
                "filled_base": "100",
                "filled_quote": "8",
                "exit_order_ids": [],
                "exchange_stop_order_id": None,
            }
        )
        self.assertEqual("ARMED", armed.get_custom_info()["exit_state"])
        before = len(self._close_calls())
        self._run(armed.control_task())
        self.assertEqual(before, len(self._close_calls()))
        self.assertEqual(Decimal("100"), Decimal(armed.get_custom_info()["open_base"]))

    def test_exit_order_ids_are_in_custom_info_and_child_order_ids(self):
        self._mid(Decimal("0.089"))
        executor = self._executor(
            take_profit={
                "percent": "0.10",
                "execution": "ICEBERG",
                "slices": 2,
                "visible_count": 2,
                "price_low": "0.088",
                "price_high": "0.090",
            }
        )
        self._open_entry(executor)
        entry_id = self.buy_ids[0][0]
        self._run(executor.control_task())
        self._run(executor.control_task())
        info = executor.get_custom_info()
        self.assertGreaterEqual(len(info["exit_order_ids"]), 1)
        for exit_id in info["exit_order_ids"]:
            self.assertIn(exit_id, info["child_order_ids"])
        self.assertIn(entry_id, info["child_order_ids"])
        self.assertIsNone(info["exchange_stop_order_id"])
        self.assertEqual(Decimal("0"), Decimal(info["closed_base"]))
        self.assertEqual(Decimal(info["filled_base"]), Decimal(info["open_base"]))

    def test_up_to_exit_closes_a_fill_that_arrives_after_the_plan(self):
        self._mid(Decimal("0.10"))
        executor = self._executor(
            exit_arming="ON_ANY_FILL",
            take_profit={
                "percent": "0.01",
                "execution": "ICEBERG",
                "slices": 1,
                "visible_count": 1,
                "limit_price": "0.08",
            },
        )
        self._run(executor.control_task())
        order = self.orders[self.buy_ids[-1][0]]
        order.executed_amount_base = Decimal("40")
        order.executed_amount_quote = Decimal("3.2")
        order.current_state = OrderState.PARTIALLY_FILLED
        self._run(executor.control_task())
        self._run(executor.control_task())
        self.assertEqual(1, len(self._close_calls()))
        self.assertEqual(Decimal("40"), self._close_calls()[0][1][2])
        order.executed_amount_base = Decimal("80")
        order.executed_amount_quote = Decimal("6.4")
        exit_id = self.buy_ids[-1][0]
        self._fill(exit_id)
        self._run(executor.control_task())
        self.assertEqual(2, len(self._close_calls()))
        self.assertEqual(Decimal("80"), sum(call[1][2] for call in self._close_calls()))
        self.assertNotEqual(CloseType.TAKE_PROFIT, executor.close_type)

    def test_stop_loss_iceberg_rejects_a_bound_on_the_wrong_side_of_the_trigger(self):
        with self.assertRaises(ValidationError):
            self._config(
                stop_loss={
                    "price": "0.09",
                    "execution": "ICEBERG",
                    "slices": 2,
                    "limit_price": "0.095",
                }
            )
        with self.assertRaises(ValidationError):
            self._config(
                side=TradeType.SELL,
                stop_loss={
                    "price": "0.09",
                    "execution": "ICEBERG",
                    "slices": 2,
                    "limit_price": "0.08",
                },
            )
        with self.assertRaises(ValidationError):
            self._config(
                stop_loss={
                    "price": "0.09",
                    "execution": "ICEBERG",
                    "slices": 2,
                    "price_low": "0.091",
                    "price_high": "0.095",
                }
            )
        with self.assertRaises(ValidationError):
            self._config(
                side=TradeType.SELL,
                stop_loss={
                    "price": "0.09",
                    "execution": "ICEBERG",
                    "slices": 2,
                    "price_low": "0.08",
                    "price_high": "0.085",
                },
            )

    def test_stop_loss_iceberg_rests_at_the_cap_then_falls_back_to_market(self):
        self._mid(Decimal("0.05"))
        executor = self._executor(
            cap_behavior="WAIT",
            stop_loss_fallback_seconds=5,
            stop_loss={
                "price": "0.08",
                "execution": "ICEBERG",
                "slices": 2,
                "visible_count": 1,
                "limit_price": "0.07",
            },
        )
        self._open_entry(executor)
        self._run(executor.control_task())
        closes = self._close_calls()
        self.assertEqual(1, len(closes))
        self.assertEqual(OrderType.LIMIT, closes[0][1][3])
        self.assertEqual(Decimal("0.07"), closes[0][1][4])
        self.now += 5
        self._run(executor.control_task())
        self._run(executor.control_task())
        markets = [call for call in self._close_calls() if call[1][3] == OrderType.MARKET]
        self.assertEqual(1, len(markets))
        self.assertEqual(Decimal("100"), markets[0][1][2])
        self.assertEqual(PositionAction.CLOSE, markets[0][1][5])

    def test_exit_rejections_retry_then_market_close_before_failed(self):
        self._mid(Decimal("0.085"))
        executor = RangeIcebergExecutor(
            self.strategy,
            self._config(take_profit={"price": "0.10", "execution": "SINGLE", "order_type": "LIMIT"}),
            update_interval=0.1,
            max_retries=1,
        )
        executor._status = RunnableStatus.RUNNING
        self._open_entry(executor)
        self._run(executor.control_task())
        tp_id = self.buy_ids[-1][0]
        executor.process_order_failed_event(
            0, None, MarketOrderFailureEvent(self.now, tp_id, OrderType.LIMIT),
        )
        self.assertNotEqual(CloseType.FAILED, executor.close_type)
        self.assertIn("retry", executor.get_custom_info()["exit_error"])
        self.now += 1
        self._run(executor.control_task())
        self.assertEqual(OrderType.LIMIT, self._close_calls()[-1][1][3])
        failed = self.buy_ids[-1][0]
        executor.process_order_failed_event(
            0, None, MarketOrderFailureEvent(self.now, failed, OrderType.LIMIT),
        )
        self.assertNotEqual(CloseType.FAILED, executor.close_type)
        self.assertIn("market close", executor.get_custom_info()["exit_error"])
        self._run(executor.control_task())
        self.assertEqual(OrderType.MARKET, self._close_calls()[-1][1][3])
        market_id = self.buy_ids[-1][0]
        executor.process_order_failed_event(
            0, None, MarketOrderFailureEvent(self.now, market_id, OrderType.MARKET),
        )
        self.assertEqual(CloseType.FAILED, executor.close_type)
        self.assertIn("market close", executor.get_custom_info()["exit_error"])
        self.assertGreater(executor.filled_base, Decimal("0"))
        self.assertEqual(RunnableStatus.TERMINATED, executor.status)

    def test_unknown_order_halt_blocks_a_new_exit_order(self):
        self._mid(Decimal("0.085"))
        executor = self._executor(stop_loss={"percent": "0.05", "execution": "SINGLE"})
        self._open_entry(executor)
        self._run(executor.control_task())
        self.assertEqual("ARMED", executor.get_custom_info()["exit_state"])
        executor._unknown_order_ids.add("buy-1")
        executor.pause()
        self._mid(Decimal("0.07"))
        before = len(self._close_calls())
        self._run(executor.control_task())
        self.assertEqual(before, len(self._close_calls()))
        self.assertIsNone(executor.close_type)
        self.assertEqual(RunnableStatus.RUNNING, executor.status)

    def test_resting_limit_take_profit_reports_exit_state_resting(self):
        executor = self._executor(take_profit={"percent": "0.10", "execution": "SINGLE", "order_type": "LIMIT"})
        self.assertEqual("NONE", executor.get_custom_info()["exit_state"])
        self._open_entry(executor)
        self._run(executor.control_task())
        info = executor.get_custom_info()
        self.assertEqual("RESTING", info["exit_state"])
        self.assertEqual("take_profit", info["exit_active"])
        self.assertEqual("resting", info["exit_status"])
        self.assertEqual(RunnableStatus.RUNNING, executor.status)
        self.assertIsNone(executor.close_type)

    def test_take_profit_price_on_the_wrong_side_of_the_average_is_rejected(self):
        with self.assertRaises(ValidationError):
            self._config(take_profit={"price": "0.07"})
        with self.assertRaises(ValidationError):
            self._config(side=TradeType.SELL, take_profit={"price": "0.091"})
        allowed = self._config(take_profit={"price": "0.07"}, allow_adverse_take_profit=True)
        self.assertEqual(Decimal("0.07"), allowed.take_profit.price)

        executor = self._executor(take_profit={"price": "0.09"})
        self._run(executor.control_task())
        self._fill(self.buy_ids[-1][0], Decimal("100"), Decimal("0.10"))
        self._run(executor.control_task())
        self.assertEqual([], self._close_calls())
        self.assertIn("allow_adverse_take_profit", executor.get_custom_info()["exit_error"])
        self.assertNotEqual(CloseType.FAILED, executor.close_type)
        self.assertEqual(RunnableStatus.RUNNING, executor.status)

        flagged = RangeIcebergExecutor(
            self.strategy,
            self._config(take_profit={"price": "0.09"}, allow_adverse_take_profit=True),
            update_interval=0.1,
        )
        flagged._status = RunnableStatus.RUNNING
        self._run(flagged.control_task())
        self._fill(self.buy_ids[-1][0], Decimal("100"), Decimal("0.10"))
        self._run(flagged.control_task())
        self.assertEqual(1, len(self._close_calls()))
        self.assertEqual(Decimal("0.09"), self._close_calls()[0][1][4])

    def _install_mexc_stop_book(self, *, fail_times=0, cancel_result=None):
        """
        MEXC stoporder/place rules the executor has to live with.

        The new stop vol, plus other live stop vols, plus resting reduce-only
        close limit vol, must be within the closable position. A place at the
        same trigger price as a live stop returns that id and replaces its vol.
        """
        state = {
            "stops": {},
            "seq": 0,
            "fail": fail_times,
            "placed": [],
            "cancelled": [],
            "ids": [],
        }

        def closable_and_resting():
            entry = Decimal("0")
            closed = Decimal("0")
            resting = Decimal("0")
            checker = getattr(self.connector, "is_cancel_confirmed", None)
            checker_is_real = (
                checker is not None
                and not getattr(type(checker), "__module__", "").startswith("unittest.mock")
            )
            for order_id, _amount, _price, order_type, action in self.buy_ids:
                order = self.orders.get(order_id)
                if order is None:
                    continue
                filled = Decimal(str(getattr(order, "executed_amount_base", 0) or 0))
                if action == PositionAction.OPEN:
                    entry += filled
                    continue
                if action != PositionAction.CLOSE:
                    continue
                closed += filled
                if order_type != OrderType.LIMIT:
                    continue
                if order.current_state in (OrderState.CANCELED, OrderState.FILLED, OrderState.FAILED):
                    continue
                if checker_is_real and bool(checker(order_id)):
                    continue
                remaining = Decimal(str(order.amount or 0)) - filled
                if remaining > 0:
                    resting += remaining
            closable = entry - closed
            if closable < 0:
                closable = Decimal("0")
            return closable, resting

        async def place_stop(trading_pair, position_id, amount, stop_loss_price=None, **kwargs):
            vol = Decimal(str(amount))
            price = None if stop_loss_price is None else Decimal(str(stop_loss_price))
            state["placed"].append((str(position_id), vol, price, kwargs.get("vol_type")))
            if state["fail"] > 0:
                state["fail"] -= 1
                raise RuntimeError("closable quantity")
            closable, resting = closable_and_resting()
            same = None
            others = Decimal("0")
            for stop_id, stop in state["stops"].items():
                if price is not None and stop["price"] == price and same is None:
                    same = stop_id
                else:
                    others += stop["amount"]
            if vol + others + resting > closable:
                raise RuntimeError("closable quantity")
            if same is not None:
                state["stops"][same] = {"amount": vol, "price": price}
                state["ids"].append(same)
                return same
            state["seq"] += 1
            stop_id = f"stop-{state['seq']}"
            state["stops"][stop_id] = {"amount": vol, "price": price}
            state["ids"].append(stop_id)
            return stop_id

        async def cancel_stop(order_ids):
            ids = [str(order_id) for order_id in order_ids]
            state["cancelled"].extend(ids)
            if isinstance(cancel_result, Exception):
                raise cancel_result
            if callable(cancel_result):
                result = cancel_result(ids)
                if hasattr(result, "__await__"):
                    result = await result
            else:
                result = {} if cancel_result is None else cancel_result
            text = str(result).lower()
            fired = any(
                marker in text
                for marker in ("not found", "already triggered", "already executed", "already filled")
            )
            if fired or (isinstance(result, dict) and result.get("success") is False):
                return result
            rejected = set()
            if isinstance(result, dict):
                for key in ("rejected", "rejected_ids", "failed", "failed_ids"):
                    for item in result.get(key) or []:
                        rejected.add(str(item))
            for stop_id in ids:
                if stop_id not in rejected:
                    state["stops"].pop(stop_id, None)
            return result

        previous_cancel = self.strategy.cancel.side_effect

        def note_cancel(connector_name, trading_pair, order_id, *args, **kwargs):
            checker = getattr(self.connector, "is_cancel_confirmed", None)
            checker_is_real = (
                checker is not None
                and not getattr(type(checker), "__module__", "").startswith("unittest.mock")
            )
            order = self.orders.get(order_id)
            if order is not None and (not checker_is_real or bool(checker(order_id))):
                if order.current_state in (OrderState.OPEN, OrderState.PARTIALLY_FILLED):
                    order.current_state = OrderState.CANCELED
            if previous_cancel is not None:
                return previous_cancel(connector_name, trading_pair, order_id, *args, **kwargs)
            return None

        self.strategy.cancel.side_effect = note_cancel
        self.connector.place_position_stop_order = place_stop
        self.connector.cancel_stop_orders = cancel_stop
        self.connector._position_id_for_order = lambda trading_pair, trade_type, position_action, reduce_only: 4242
        return state

    def _install_exchange_stop(self, place_result="stop-1", cancel_result=None, place_error=None):
        fail_times = place_error if isinstance(place_error, int) else 0
        state = self._install_mexc_stop_book(fail_times=fail_times, cancel_result=cancel_result)
        return state["placed"], state["cancelled"]

    def test_exchange_stop_is_a_backstop_and_is_cleared_before_the_software_stop(self):
        placed, cancelled = self._install_exchange_stop()
        events = []
        original = self.strategy.sell.side_effect

        def sell(*args, **kwargs):
            if args[5] == PositionAction.CLOSE:
                events.append("close")
            return original(*args, **kwargs)

        self.strategy.sell.side_effect = sell
        self._mid(Decimal("0.085"))
        executor = self._executor(stop_loss={"percent": "0.05", "execution": "SINGLE"})
        _, _, price = self._open_entry(executor)
        self._run(executor.control_task())
        software = price * Decimal("0.95")
        backstop = software * (Decimal("1") - executor.config.protective_stop_buffer_pct)
        self.assertEqual(backstop, placed[0][2])
        self.assertEqual(1, placed[0][3])
        self.assertLess(backstop, software)
        self.assertEqual(backstop, Decimal(executor.get_custom_info()["exchange_stop_trigger_price"]))
        self.assertEqual([], events)

        async def cancel_stop(order_ids):
            cancelled.extend(list(order_ids))
            events.append("cancel")
            return {"success": True}

        self.connector.cancel_stop_orders = cancel_stop
        self._mid(Decimal("0.07"))
        self._run(executor.control_task())
        self.assertEqual(["cancel", "close"], events)
        self.assertIsNone(executor.get_custom_info()["exchange_stop_order_id"])
        self.assertNotEqual(CloseType.FAILED, executor.close_type)
        self.assertEqual(OrderType.MARKET, self._close_calls()[-1][1][3])

        fired, _ = self._install_exchange_stop(cancel_result={"success": False, "message": "order not found"})
        other = self._executor(stop_loss={"percent": "0.05", "execution": "SINGLE"})
        self._mid(Decimal("0.085"))
        self._open_entry(other)
        self._run(other.control_task())
        self._mid(Decimal("0.07"))
        before = len(self._close_calls())
        self._run(other.control_task())
        self.assertEqual(before, len(self._close_calls()))
        self.assertEqual(CloseType.STOP_LOSS, other.close_type)
        self.assertEqual("DONE", other.get_custom_info()["exit_state"])
        self.assertNotEqual(CloseType.FAILED, other.close_type)
        self.assertEqual(Decimal("0"), Decimal(other.get_custom_info()["open_base"]))
        self.assertGreaterEqual(len(fired), 1)

    def test_exchange_stop_is_placed_before_the_take_profit_and_backs_off(self):
        state = self._install_mexc_stop_book(fail_times=1)
        attempts = []
        place_stop = self.connector.place_position_stop_order

        async def place_and_stamp(*args, **kwargs):
            attempts.append(self.now)
            return await place_stop(*args, **kwargs)

        self.connector.place_position_stop_order = place_and_stamp
        executor = self._executor(
            take_profit={"percent": "0.10", "execution": "SINGLE", "order_type": "LIMIT"},
            stop_loss={"percent": "0.05", "execution": "SINGLE"},
        )
        self._open_entry(executor)
        self._run(executor.control_task())
        self.assertEqual(1, len(attempts))
        self.assertEqual(1, state["placed"][0][3])
        self.assertEqual([], self._close_calls())
        self.assertIn("Next try", executor.get_custom_info()["exchange_stop_error"])
        self.assertIsNone(executor.get_custom_info()["exchange_stop_order_id"])
        self._run(executor.control_task())
        self.assertEqual(1, len(attempts))
        self.now += 0.5
        self._run(executor.control_task())
        self.assertEqual(2, len(attempts))
        self.assertEqual("stop-1", executor.get_custom_info()["exchange_stop_order_id"])
        self.assertEqual("", executor.get_custom_info()["exchange_stop_error"])
        self.assertEqual(1, len(self._close_calls()))
        self.assertGreater(attempts[1], attempts[0])

    def test_exchange_stop_resize_keeps_the_old_id_until_cancel_is_confirmed(self):
        cancel_error = {"on": True}

        async def cancel_stop(order_ids):
            if cancel_error["on"]:
                raise RuntimeError("cancel timed out")
            return {"success": True}

        state = self._install_mexc_stop_book(cancel_result=cancel_stop)
        self._mid(Decimal("0.085"))
        executor = self._executor(
            exit_arming="ON_ANY_FILL",
            stop_loss={"percent": "0.05", "execution": "SINGLE"},
        )
        self._run(executor.control_task())
        order = self.orders[self.buy_ids[-1][0]]
        order.executed_amount_base = Decimal("40")
        order.executed_amount_quote = Decimal("3.2")
        order.current_state = OrderState.PARTIALLY_FILLED
        self._run(executor.control_task())
        self.assertEqual(["stop-1"], list(state["stops"]))
        self.assertEqual("stop-1", executor.get_custom_info()["exchange_stop_order_id"])
        order.executed_amount_base = Decimal("100")
        order.executed_amount_quote = Decimal("8")
        self._run(executor.control_task())
        self.assertEqual(["stop-1"], list(state["stops"]))
        self.assertEqual("stop-1", executor.get_custom_info()["exchange_stop_order_id"])
        self.assertIn("stop-1", executor.get_custom_info()["exchange_stop_error"])
        cancel_error["on"] = False
        self._run(executor.control_task())
        self.assertEqual(["stop-2"], list(state["stops"]))
        self.assertEqual(Decimal("100"), state["stops"]["stop-2"]["amount"])
        self.assertEqual("stop-2", executor.get_custom_info()["exchange_stop_order_id"])

    def test_exchange_stop_is_cancelled_on_terminal_paths_except_keep_position(self):
        placed, cancelled = self._install_exchange_stop()
        self._mid(Decimal("0.085"))
        held = self._executor(stop_loss={"percent": "0.05", "execution": "SINGLE"})
        self._open_entry(held)
        self._run(held.control_task())
        held.early_stop(keep_position=True)
        self.assertEqual([], cancelled)
        self.assertEqual("stop-1", held.get_custom_info()["exchange_stop_order_id"])
        self.assertEqual(RunnableStatus.TERMINATED, held.status)

        stopped = self._executor(stop_loss={"percent": "0.05", "execution": "SINGLE"})
        self._open_entry(stopped)
        self._run(stopped.control_task())
        stopped.early_stop()
        self.assertEqual(CloseType.EARLY_STOP, stopped.close_type)
        self.assertNotEqual(RunnableStatus.TERMINATED, stopped.status)
        self.assertEqual("stop-1", stopped.get_custom_info()["exchange_stop_order_id"])
        self._run(stopped.control_task())
        self.assertIn("stop-1", cancelled)
        self.assertIsNone(stopped.get_custom_info()["exchange_stop_order_id"])
        self.assertEqual(RunnableStatus.TERMINATED, stopped.status)

        failed = self._executor(stop_loss={"percent": "0.05", "execution": "SINGLE"})
        self._open_entry(failed)
        self._run(failed.control_task())
        failed_id = failed.get_custom_info()["exchange_stop_order_id"]
        failed.early_stop(close_type=CloseType.FAILED)
        self._run(failed.control_task())
        self.assertEqual(CloseType.FAILED, failed.close_type)
        self.assertIsNone(failed.get_custom_info()["exchange_stop_order_id"])
        self.assertEqual(RunnableStatus.TERMINATED, failed.status)
        self.assertIn("stop-1", cancelled)
        self.assertIn(failed_id, cancelled)

    def test_entry_retries_rest_the_limit_take_profit_on_the_filled_amount(self):
        executor = RangeIcebergExecutor(
            self.strategy,
            self._config(take_profit={"price": "0.10", "execution": "SINGLE", "order_type": "LIMIT"}),
            update_interval=0.1,
            max_retries=0,
        )
        executor._status = RunnableStatus.RUNNING
        self._run(executor.control_task())
        order_id = self.buy_ids[-1][0]
        self._fill(order_id, Decimal("40"), Decimal("0.08"))
        self.orders[order_id].current_state = OrderState.PARTIALLY_FILLED
        executor.process_order_failed_event(
            0, None, MarketOrderFailureEvent(self.now, order_id, OrderType.LIMIT),
        )
        self.assertNotEqual(CloseType.FAILED, executor.close_type)
        self.assertNotIn("market close", executor.get_custom_info()["exit_error"])
        self.assertIn("limit take profit", executor.get_custom_info()["exit_error"])
        self._run(executor.control_task())
        closes = self._close_calls()
        self.assertEqual(1, len(closes))
        self.assertEqual(OrderType.LIMIT, closes[0][1][3])
        self.assertEqual(Decimal("40"), closes[0][1][2])
        self.assertEqual("RESTING", executor.get_custom_info()["exit_state"])
        self.assertEqual(RunnableStatus.RUNNING, executor.status)
        self.assertIsNone(executor.close_type)

    def test_stop_iceberg_falls_back_after_a_partial_fill_goes_quiet(self):
        self._mid(Decimal("0.05"))
        executor = self._executor(
            stop_loss_fallback_seconds=5,
            stop_loss={
                "price": "0.08",
                "execution": "ICEBERG",
                "slices": 2,
                "visible_count": 1,
                "limit_price": "0.07",
            },
        )
        self._open_entry(executor)
        self._run(executor.control_task())
        child = self.orders[self.buy_ids[-1][0]]
        self.now = 1004
        child.executed_amount_base = Decimal("40")
        child.executed_amount_quote = Decimal("2.8")
        child.current_state = OrderState.PARTIALLY_FILLED
        self._run(executor.control_task())
        self.assertEqual([], [call for call in self._close_calls() if call[1][3] == OrderType.MARKET])
        self.now = 1008
        self._run(executor.control_task())
        self.assertEqual([], [call for call in self._close_calls() if call[1][3] == OrderType.MARKET])
        self.now = 1009
        self._run(executor.control_task())
        self._run(executor.control_task())
        markets = [call for call in self._close_calls() if call[1][3] == OrderType.MARKET]
        self.assertEqual(1, len(markets))
        self.assertEqual(Decimal("60"), markets[0][1][2])

    def test_stop_loss_retry_backoff_is_capped_near_two_seconds(self):
        self._mid(Decimal("0.07"))
        executor = self._executor(stop_loss={"percent": "0.05", "execution": "SINGLE"})
        self._open_entry(executor)
        delays = []
        for _ in range(4):
            self._run(executor.control_task())
            failed = self.buy_ids[-1][0]
            executor.process_order_failed_event(
                0, None, MarketOrderFailureEvent(self.now, failed, OrderType.MARKET),
            )
            delays.append(executor._exit_retry_after - self.now)
            self.now = executor._exit_retry_after
        self.assertEqual(2.0, delays[-1])
        self.assertGreater(0.5 * (2 ** 3), 2.0)
        self.assertNotEqual(CloseType.FAILED, executor.close_type)

        take_profit = self._executor(take_profit={"price": "0.10", "execution": "SINGLE", "order_type": "LIMIT"})
        self._open_entry(take_profit)
        self._run(take_profit.control_task())
        tp_delays = []
        for _ in range(4):
            failed = self.buy_ids[-1][0]
            take_profit.process_order_failed_event(
                0, None, MarketOrderFailureEvent(self.now, failed, OrderType.LIMIT),
            )
            tp_delays.append(take_profit._exit_retry_after - self.now)
            self.now = take_profit._exit_retry_after
            self._run(take_profit.control_task())
        self.assertGreater(tp_delays[-1], 2.0)

    def test_up_to_take_profit_is_checked_against_the_fill_not_the_limit(self):
        parsed = self._config(
            side=TradeType.SELL,
            mode=ExecutionMode.UP_TO,
            limit_price=Decimal("0.08"),
            take_profit={"price": "0.09", "execution": "SINGLE"},
        )
        self.assertEqual(Decimal("0.09"), parsed.take_profit.price)
        buy = self._config(
            mode=ExecutionMode.UP_TO,
            limit_price=Decimal("0.08"),
            take_profit={"price": "0.09", "execution": "SINGLE"},
        )
        self.assertEqual(Decimal("0.09"), buy.take_profit.price)

    def test_percent_take_profit_reprices_when_the_average_moves(self):
        executor = self._executor(
            exit_arming="ON_ANY_FILL",
            take_profit={"percent": "0.10", "execution": "SINGLE", "order_type": "LIMIT"},
        )
        self._run(executor.control_task())
        order = self.orders[self.buy_ids[-1][0]]
        order.executed_amount_base = Decimal("40")
        order.executed_amount_quote = Decimal("3.2")
        order.current_state = OrderState.PARTIALLY_FILLED
        self._run(executor.control_task())
        self.assertEqual(Decimal("0.088"), self._close_calls()[-1][1][4])
        order.executed_amount_base = Decimal("80")
        order.executed_amount_quote = Decimal("7.2")
        self._run(executor.control_task())
        self.strategy.cancel.assert_called()
        self.assertEqual(Decimal("0.088"), self._close_calls()[-1][1][4])
        self._run(executor.control_task())
        self.assertEqual(Decimal("0.099"), self._close_calls()[-1][1][4])
        self.assertEqual(Decimal("80"), self._close_calls()[-1][1][2])
        placed = len(self._close_calls())
        self._run(executor.control_task())
        self.assertEqual(placed, len(self._close_calls()))
        order.executed_amount_quote = Decimal("8")
        self.strategy.cancel.reset_mock()
        self._run(executor.control_task())
        self.strategy.cancel.assert_not_called()
        self.now += 1
        self._run(executor.control_task())
        self.strategy.cancel.assert_called()

    def test_restore_reposts_a_missing_exit_and_keeps_the_exchange_stop(self):
        placed, _ = self._install_exchange_stop()
        resting = self._executor(
            take_profit={"percent": "0.10", "execution": "SINGLE", "order_type": "LIMIT"},
            stop_loss={"percent": "0.05", "execution": "SINGLE"},
        )
        self._open_entry(resting)
        self._run(resting.control_task())
        self.assertEqual("RESTING", resting.get_custom_info()["exit_state"])
        self.assertEqual("stop-1", resting.get_custom_info()["exchange_stop_order_id"])
        snapshot = resting.get_custom_info()
        for order_id in list(snapshot["exit_order_ids"]):
            self.orders.pop(order_id, None)
        resumed = self._executor(
            take_profit={"percent": "0.10", "execution": "SINGLE", "order_type": "LIMIT"},
            stop_loss={"percent": "0.05", "execution": "SINGLE"},
        )
        resumed.restore_from_custom_info(snapshot)
        before_stops = len(placed)
        before_closes = len(self._close_calls())
        self._run(resumed.control_task())
        self.assertEqual(before_stops, len(placed))
        self.assertEqual("stop-1", resumed.get_custom_info()["exchange_stop_order_id"])
        self.assertEqual(before_closes + 1, len(self._close_calls()))
        self.assertEqual(Decimal(snapshot["open_base"]), self._close_calls()[-1][1][2])
        self.assertEqual("RESTING", resumed.get_custom_info()["exit_state"])

    def _open_order_count(self) -> int:
        count = 0
        for method in (self.strategy.buy, self.strategy.sell):
            for call in method.call_args_list:
                if call.args[5] == PositionAction.OPEN:
                    count += 1
        return count

    def test_resume_entries_false_places_no_entry_orders_for_a_quote_config(self):
        self._mid(Decimal("0.085"))
        executor = self._executor(
            size_unit=SizeUnit.QUOTE,
            total_amount=Decimal("8"),
            take_profit={"percent": "0.10", "execution": "SINGLE", "order_type": "LIMIT"},
            stop_loss={"percent": "0.05", "execution": "SINGLE"},
        )
        executor.restore_from_custom_info({
            "resume_entries": False,
            "size_unit": "BASE",
            "total_amount": "60",
            "open_base": "60",
            "average_fill_price": "0.08",
            "exit_state": "RESTING",
            "exit": {"status": "resting", "active": "take_profit"},
        })
        for _ in range(3):
            self._run(executor.control_task())
        self.assertEqual(0, self._open_order_count())
        self.assertEqual(Decimal("60"), executor.filled_base)
        self.assertEqual(Decimal("60"), Decimal(executor.get_custom_info()["open_base"]))
        self.assertEqual(SizeUnit.BASE, executor._size_unit_override)

    def test_exits_only_resume_rebuilds_average_and_does_not_fire_stop(self):
        self._mid(Decimal("0.085"))
        executor = self._executor(
            size_unit=SizeUnit.QUOTE,
            total_amount=Decimal("8"),
            take_profit={"percent": "0.10", "execution": "SINGLE", "order_type": "LIMIT"},
            stop_loss={"percent": "0.05", "execution": "SINGLE"},
        )
        executor.restore_from_custom_info({
            "resume_entries": False,
            "size_unit": "BASE",
            "total_amount": "60",
            "open_base": "60",
            "filled_base": "100",
            "closed_base": "40",
            "average_fill_price": "0.08",
            "exit_state": "RESTING",
            "exit": {
                "status": "resting",
                "active": "take_profit",
                "entry_filled_base": "60",
                "entry_filled_quote": "20",
                "filled_by_order": {"old-tp": {"base": "40", "quote": "4.4"}},
                "order_ids": ["old-tp"],
            },
        })
        before = len(self._close_calls())
        for _ in range(3):
            self._run(executor.control_task())
        info = executor.get_custom_info()
        self.assertEqual(Decimal("60"), executor.filled_base)
        self.assertEqual(Decimal("0"), Decimal(info["closed_base"]))
        self.assertEqual(Decimal("0.088"), Decimal(info["take_profit_trigger_price"]))
        self.assertEqual(Decimal("0.076"), Decimal(info["stop_loss_trigger_price"]))
        self.assertEqual(0, self._open_order_count())
        markets = [call for call in self._close_calls()[before:] if call[1][3] == OrderType.MARKET]
        self.assertEqual([], markets)
        self.assertNotEqual(CloseType.STOP_LOSS, executor.close_type)
        self.assertEqual("RESTING", info["exit_state"])

    def test_growing_take_profit_resizes_the_stop_only_after_the_limit_is_gone(self):
        state = self._install_mexc_stop_book()
        self._mid(Decimal("0.085"))
        executor = self._executor(
            exit_arming="ON_ANY_FILL",
            take_profit={"price": "0.10", "execution": "SINGLE", "order_type": "LIMIT"},
            stop_loss={"price": "0.07", "execution": "SINGLE"},
        )
        self._run(executor.control_task())
        order = self.orders[self.buy_ids[-1][0]]
        order.executed_amount_base = Decimal("50")
        order.executed_amount_quote = Decimal("4")
        order.current_state = OrderState.PARTIALLY_FILLED
        self._run(executor.control_task())
        self.assertEqual({"stop-1": state["stops"]["stop-1"]}, state["stops"])
        self.assertEqual(Decimal("50"), state["stops"]["stop-1"]["amount"])
        self.assertEqual(Decimal("50"), self._close_calls()[-1][1][2])
        order.executed_amount_base = Decimal("100")
        order.executed_amount_quote = Decimal("8")
        order.current_state = OrderState.FILLED
        self._run(executor.control_task())
        self.assertEqual(1, len(state["placed"]))
        self.assertEqual([], state["cancelled"])
        self.assertEqual(Decimal("50"), executor._exchange_stop_amount)
        self._run(executor.control_task())
        self.assertEqual([Decimal("50"), Decimal("100")], [item[1] for item in state["placed"]])
        self.assertEqual(state["placed"][0][2], state["placed"][1][2])
        self.assertEqual([], state["cancelled"])
        self.assertEqual(["stop-1"], list(state["stops"]))
        self.assertEqual(Decimal("100"), state["stops"]["stop-1"]["amount"])
        info = executor.get_custom_info()
        self.assertEqual(["stop-1"], info["exchange_stop_order_ids"])
        self.assertEqual("stop-1", info["exchange_stop_order_id"])
        self.assertEqual("", info["exchange_stop_error"])
        self.assertEqual(Decimal("100"), self._close_calls()[-1][1][2])
        self.assertEqual(OrderType.LIMIT, self._close_calls()[-1][1][3])
        placed = len(state["placed"])
        self._run(executor.control_task())
        self._run(executor.control_task())
        self.assertEqual(placed, len(state["placed"]))
        open_closes = [
            self.orders[order_id]
            for order_id, _amount, _price, order_type, action in self.buy_ids
            if action == PositionAction.CLOSE
            and order_type == OrderType.LIMIT
            and self.orders[order_id].current_state == OrderState.OPEN
        ]
        self.assertEqual(1, len(open_closes))
        self.assertEqual(Decimal("100"), open_closes[0].amount)

        failed = self._install_mexc_stop_book()
        other = self._executor(
            exit_arming="ON_ANY_FILL",
            take_profit={"price": "0.10", "execution": "SINGLE", "order_type": "LIMIT"},
            stop_loss={"price": "0.07", "execution": "SINGLE"},
        )
        self._run(other.control_task())
        other_order = self.orders[self.buy_ids[-1][0]]
        other_order.executed_amount_base = Decimal("50")
        other_order.executed_amount_quote = Decimal("4")
        other_order.current_state = OrderState.PARTIALLY_FILLED
        self._run(other.control_task())
        failed["fail"] = 1
        other_order.executed_amount_base = Decimal("100")
        other_order.executed_amount_quote = Decimal("8")
        other_order.current_state = OrderState.FILLED
        self._run(other.control_task())
        self._run(other.control_task())
        self.assertEqual(Decimal("50"), failed["stops"]["stop-1"]["amount"])
        self.assertEqual("stop-1", other.get_custom_info()["exchange_stop_order_id"])
        self.assertIn("closable quantity", other.get_custom_info()["exchange_stop_error"])
        self.assertEqual(Decimal("50"), self._close_calls()[-1][1][2])
        places = len(failed["placed"])
        cancels = self.strategy.cancel.call_count
        self._run(other.control_task())
        self.assertEqual(places, len(failed["placed"]))
        self.assertEqual(cancels, self.strategy.cancel.call_count)

    def test_percent_stop_grows_by_the_delta_without_a_coverage_gap(self):
        state = self._install_mexc_stop_book()
        self._mid(Decimal("0.085"))
        executor = self._executor(
            exit_arming="ON_ANY_FILL",
            take_profit={"percent": "0.10", "execution": "SINGLE", "order_type": "LIMIT"},
            stop_loss={"percent": "0.05", "execution": "SINGLE"},
        )
        self._run(executor.control_task())
        order = self.orders[self.buy_ids[-1][0]]
        order.executed_amount_base = Decimal("50")
        order.executed_amount_quote = Decimal("4")
        order.current_state = OrderState.PARTIALLY_FILLED
        self._run(executor.control_task())
        old_price = state["stops"]["stop-1"]["price"]
        self.assertEqual(Decimal("50"), state["stops"]["stop-1"]["amount"])
        order.executed_amount_base = Decimal("100")
        order.executed_amount_quote = Decimal("9")
        order.current_state = OrderState.FILLED
        self._mid(Decimal("0.092"))
        self._run(executor.control_task())
        self.assertEqual([], state["cancelled"])
        self.assertEqual(Decimal("50"), executor._exchange_stop_amount)
        self.assertEqual(old_price, state["stops"]["stop-1"]["price"])
        self._run(executor.control_task())
        self.assertEqual(["stop-1"], state["cancelled"])
        self.assertEqual([Decimal("50"), Decimal("100")], [item[1] for item in state["placed"]])
        self.assertNotEqual(old_price, state["placed"][1][2])
        self.assertEqual(1, len(state["stops"]))
        stop = next(iter(state["stops"].values()))
        self.assertEqual(Decimal("100"), stop["amount"])
        self.assertEqual(state["placed"][1][2], stop["price"])
        info = executor.get_custom_info()
        self.assertEqual(1, len(info["exchange_stop_order_ids"]))
        self.assertEqual(info["exchange_stop_order_ids"][0], info["exchange_stop_order_id"])
        self.assertEqual("", info["exchange_stop_error"])
        self.assertEqual(Decimal("100"), self._close_calls()[-1][1][2])
        self.assertEqual(OrderType.LIMIT, self._close_calls()[-1][1][3])
        placed = len(state["placed"])
        self._run(executor.control_task())
        self._run(executor.control_task())
        self.assertEqual(placed, len(state["placed"]))
        open_closes = [
            self.orders[order_id]
            for order_id, _amount, _price, order_type, action in self.buy_ids
            if action == PositionAction.CLOSE
            and order_type == OrderType.LIMIT
            and self.orders[order_id].current_state == OrderState.OPEN
        ]
        self.assertEqual(1, len(open_closes))
        self.assertEqual(Decimal("100"), open_closes[0].amount)

    def _open_limit_closes(self):
        return [
            self.orders[order_id]
            for order_id, _amount, _price, order_type, action in self.buy_ids
            if action == PositionAction.CLOSE
            and order_type == OrderType.LIMIT
            and self.orders[order_id].current_state == OrderState.OPEN
        ]

    def test_failed_stop_move_restores_protection_before_the_take_profit(self):
        state = self._install_mexc_stop_book()
        self._mid(Decimal("0.085"))
        executor = self._executor(
            exit_arming="ON_ANY_FILL",
            take_profit={"percent": "0.10", "execution": "SINGLE", "order_type": "LIMIT"},
            stop_loss={"percent": "0.05", "execution": "SINGLE"},
        )
        self._run(executor.control_task())
        order = self.orders[self.buy_ids[-1][0]]
        order.executed_amount_base = Decimal("50")
        order.executed_amount_quote = Decimal("4")
        order.current_state = OrderState.PARTIALLY_FILLED
        self._run(executor.control_task())
        old_price = state["stops"]["stop-1"]["price"]
        order.executed_amount_base = Decimal("100")
        order.executed_amount_quote = Decimal("9")
        order.current_state = OrderState.FILLED
        self._mid(Decimal("0.092"))
        self._run(executor.control_task())
        self.assertEqual([], state["cancelled"])
        self.assertEqual(Decimal("50"), state["stops"]["stop-1"]["amount"])
        state["fail"] = 2
        closes = len(self._close_calls())
        self._run(executor.control_task())
        self.assertEqual(["stop-1"], state["cancelled"])
        self.assertEqual({}, state["stops"])
        self.assertIsNone(executor.get_custom_info()["exchange_stop_order_id"])
        self.assertFalse(executor.get_custom_info()["has_exchange_stop"])
        self.assertEqual(closes, len(self._close_calls()))
        self.assertLessEqual(executor._exchange_stop_retry_after - self.now, 2.0)
        self._run(executor.control_task())
        self.assertEqual(closes, len(self._close_calls()))
        self.assertIsNone(executor.get_custom_info()["exchange_stop_order_id"])
        self.now += 0.5
        self._run(executor.control_task())
        self.assertTrue(executor.get_custom_info()["has_exchange_stop"])
        self.assertGreater(executor._exchange_stop_amount, 0)
        self.assertEqual(closes, len(self._close_calls()))
        for _ in range(4):
            self._run(executor.control_task())
        info = executor.get_custom_info()
        self.assertTrue(info["has_exchange_stop"])
        self.assertEqual(1, len(state["stops"]))
        stop = next(iter(state["stops"].values()))
        self.assertEqual(Decimal("100"), stop["amount"])
        self.assertNotEqual(old_price, stop["price"])
        self.assertEqual(1, len(info["exchange_stop_order_ids"]))
        open_closes = self._open_limit_closes()
        self.assertEqual(1, len(open_closes))
        self.assertEqual(Decimal("100"), open_closes[0].amount)
        placed = len(state["placed"])
        self._run(executor.control_task())
        self._run(executor.control_task())
        self.assertEqual(placed, len(state["placed"]))

    def test_sync_sizes_the_stop_to_the_closable_quantity(self):
        state = self._install_mexc_stop_book()
        self._mid(Decimal("0.085"))
        executor = self._executor(
            exit_arming="ON_ANY_FILL",
            take_profit={"price": "0.10", "execution": "SINGLE", "order_type": "LIMIT"},
            stop_loss={"price": "0.07", "execution": "SINGLE"},
        )
        self._run(executor.control_task())
        order = self.orders[self.buy_ids[-1][0]]
        order.executed_amount_base = Decimal("100")
        order.executed_amount_quote = Decimal("8")
        order.current_state = OrderState.FILLED
        self._run(executor.control_task())
        self.assertEqual(Decimal("100"), state["stops"]["stop-1"]["amount"])
        state["stops"].pop("stop-1")
        executor._exchange_stops.clear()
        executor._sync_exchange_stop_fields()
        tp = self._open_limit_closes()[0]
        tp.amount = Decimal("40")
        for slot in executor._exit_slices:
            if slot.tracked is not None and slot.tracked.order is tp:
                slot.target_amount = Decimal("40")
        before = len(state["placed"])
        self._run(executor.control_task())
        self.assertEqual(before + 1, len(state["placed"]))
        self.assertEqual(Decimal("60"), state["placed"][before][1])
        info = executor.get_custom_info()
        self.assertEqual(Decimal("60"), state["stops"][info["exchange_stop_order_id"]]["amount"])
        self.assertIn("partial coverage", info["exchange_stop_error"])
        placed = len(state["placed"])
        self._run(executor.control_task())
        self.assertEqual(placed, len(state["placed"]))
        for _ in range(3):
            self._run(executor.control_task())
        self.assertEqual(1, len(state["stops"]))
        self.assertEqual(Decimal("100"), next(iter(state["stops"].values()))["amount"])
        open_closes = self._open_limit_closes()
        self.assertEqual(1, len(open_closes))
        self.assertEqual(Decimal("100"), open_closes[0].amount)

    def test_take_profit_posts_when_the_connector_cannot_place_a_stop(self):
        executor = self._executor(
            protective_exchange_stop=True,
            take_profit={"percent": "0.10", "execution": "SINGLE", "order_type": "LIMIT"},
            stop_loss={"percent": "0.05", "execution": "SINGLE"},
        )
        self._open_entry(executor)
        self._run(executor.control_task())
        closes = [call for call in self._close_calls() if call[1][3] == OrderType.LIMIT]
        self.assertEqual(1, len(closes))
        self.assertIn("place_position_stop_order", executor.get_custom_info()["exchange_stop_error"])
        self.assertIsNone(executor.get_custom_info()["exchange_stop_order_id"])

    def test_take_profit_posts_after_the_stop_hold_when_placement_keeps_failing(self):
        state = self._install_mexc_stop_book(fail_times=50)
        self._mid(Decimal("0.085"))
        executor = self._executor(
            take_profit={"percent": "0.10", "execution": "SINGLE", "order_type": "LIMIT"},
            stop_loss={"percent": "0.05", "execution": "SINGLE"},
        )
        self.assertIsNone(executor.config.protective_exchange_stop)
        _, _, price = self._open_entry(executor)
        self._run(executor.control_task())
        self.assertEqual([], [call for call in self._close_calls() if call[1][3] == OrderType.LIMIT])
        self.assertIn("Next try", executor.get_custom_info()["exchange_stop_error"])
        self.now += executor.config.exchange_stop_hold_seconds
        self._run(executor.control_task())
        limits = [call for call in self._close_calls() if call[1][3] == OrderType.LIMIT]
        self.assertEqual(1, len(limits))
        self.assertEqual(price * Decimal("1.10"), limits[0][1][4])
        self.assertNotEqual("", executor.get_custom_info()["exchange_stop_error"])
        self.assertIsNone(executor.get_custom_info()["exchange_stop_order_id"])
        self.assertGreaterEqual(len(state["placed"]), 1)

    def test_take_profit_fills_when_price_is_past_it_and_the_stop_is_rejected(self):
        self._install_mexc_stop_book(fail_times=50)
        self._mid(Decimal("0.085"))
        executor = self._executor(
            take_profit={"percent": "0.10", "execution": "SINGLE", "order_type": "LIMIT"},
            stop_loss={"percent": "0.05", "execution": "SINGLE"},
        )
        _, base, price = self._open_entry(executor)
        self._run(executor.control_task())
        self.now += executor.config.exchange_stop_hold_seconds
        self._mid(price * Decimal("1.20"))
        self._run(executor.control_task())
        limits = [call for call in self._close_calls() if call[1][3] == OrderType.LIMIT]
        self.assertEqual(1, len(limits))
        self.assertNotEqual(CloseType.STOP_LOSS, executor.close_type)
        self._fill(self.buy_ids[-1][0], base, limits[0][1][4])
        self._run(executor.control_task())
        self.assertEqual(CloseType.TAKE_PROFIT, executor.close_type)
        self.assertEqual(Decimal("0"), Decimal(executor.get_custom_info()["open_base"]))
        self.assertLess(self.now - 1_000.0, 60)

    def test_stop_loss_iceberg_still_sends_limit_slices_when_the_stop_is_rejected(self):
        state = self._install_mexc_stop_book(fail_times=50)
        self._mid(Decimal("0.070"))
        executor = self._executor(
            stop_loss={
                "percent": "0.05",
                "execution": "ICEBERG",
                "slices": 2,
                "visible_count": 1,
                "price_low": "0.070",
                "price_high": "0.076",
            }
        )
        self._open_entry(executor)
        self._run(executor.control_task())
        limits = [call for call in self._close_calls() if call[1][3] == OrderType.LIMIT]
        self.assertEqual(1, len(limits))
        self.assertGreaterEqual(limits[0][1][4], Decimal("0.070"))
        self.assertLessEqual(limits[0][1][4], Decimal("0.076"))
        self.assertEqual([], [call for call in self._close_calls() if call[1][3] == OrderType.MARKET])
        slice_id = self.buy_ids[-1][0]
        self._run(executor.control_task())
        self.assertEqual(OrderState.OPEN, self.orders[slice_id].current_state)
        self.assertEqual([], [call for call in self._close_calls() if call[1][3] == OrderType.MARKET])
        self.assertEqual(0, len(state["placed"]))
        self.assertTrue(executor.get_custom_info()["exchange_stop_released"])
        self.assertIn(executor.get_custom_info()["exit_state"], ("TRIGGERED", "CLOSING"))

        snapshot = executor.get_custom_info()
        self.strategy.cancel.reset_mock()
        resumed = self._executor(
            stop_loss={
                "percent": "0.05",
                "execution": "ICEBERG",
                "slices": 2,
                "visible_count": 1,
                "price_low": "0.070",
                "price_high": "0.076",
            }
        )
        resumed.restore_from_custom_info(snapshot)
        self._run(resumed.control_task())
        self._run(resumed.control_task())
        self.assertEqual(OrderState.OPEN, self.orders[slice_id].current_state)
        cancelled_ids = [call.args[2] for call in self.strategy.cancel.call_args_list]
        self.assertNotIn(slice_id, cancelled_ids)
        self.assertIn(slice_id, resumed.get_custom_info()["exit_order_ids"])

    def test_resumed_stop_loss_close_does_not_place_a_new_exchange_stop(self):
        state = self._install_mexc_stop_book()
        self._mid(Decimal("0.085"))
        executor = self._executor(stop_loss={"percent": "0.05", "execution": "SINGLE"})
        self._open_entry(executor)
        self._run(executor.control_task())
        self.assertIsNotNone(executor.get_custom_info()["exchange_stop_order_id"])
        self._mid(Decimal("0.07"))
        self._run(executor.control_task())
        info = executor.get_custom_info()
        self.assertTrue(info["exchange_stop_released"])
        self.assertIsNone(info["exchange_stop_order_id"])
        self.assertEqual(1, len([call for call in self._close_calls() if call[1][3] == OrderType.MARKET]))
        placed = len(state["placed"])
        resumed = self._executor(stop_loss={"percent": "0.05", "execution": "SINGLE"})
        resumed.restore_from_custom_info(info)
        self._run(resumed.control_task())
        self._run(resumed.control_task())
        self.assertEqual(placed, len(state["placed"]))
        self.assertFalse(resumed.get_custom_info()["has_exchange_stop"])
        self.assertTrue(resumed.get_custom_info()["exchange_stop_released"])

    def _places_in_any_two_seconds(self, times) -> int:
        worst = 0
        for stamp in times:
            count = sum(1 for other in times if stamp <= other < stamp + 2)
            if count > worst:
                worst = count
        return worst

    def test_failed_grow_keeps_the_take_profit_on_the_book(self):
        state = self._install_mexc_stop_book()
        self._mid(Decimal("0.085"))
        executor = self._executor(
            exit_arming="ON_ANY_FILL",
            take_profit={"price": "0.10", "execution": "SINGLE", "order_type": "LIMIT"},
            stop_loss={"price": "0.07", "execution": "SINGLE"},
        )
        self._run(executor.control_task())
        order = self.orders[self.buy_ids[-1][0]]
        order.executed_amount_base = Decimal("50")
        order.executed_amount_quote = Decimal("4")
        order.current_state = OrderState.PARTIALLY_FILLED
        self._run(executor.control_task())
        self.assertEqual(Decimal("50"), state["stops"]["stop-1"]["amount"])
        self.assertEqual(1, len(self._open_limit_closes()))
        state["fail"] = 100
        order.executed_amount_base = Decimal("100")
        order.executed_amount_quote = Decimal("8")
        order.current_state = OrderState.FILLED
        on_book = 0
        place_times = []
        seen = len(state["placed"])
        for _ in range(180):
            self._run(executor.control_task())
            added = len(state["placed"]) - seen
            place_times.extend([self.now] * added)
            seen = len(state["placed"])
            if self._open_limit_closes():
                on_book += 1
            self.now += 1
        self.assertGreaterEqual(on_book, int(180 * 0.95))
        self.assertLessEqual(self._places_in_any_two_seconds(place_times), 5)
        self.assertEqual(Decimal("50"), state["stops"]["stop-1"]["amount"])
        self.assertNotEqual("", executor.get_custom_info()["exchange_stop_error"])
        self.assertTrue(self._open_limit_closes())
        self.assertEqual(Decimal("50"), self._open_limit_closes()[0].amount)

    def test_percent_stop_reprices_in_place_before_it_resizes(self):
        state = self._install_mexc_stop_book()
        repriced = []

        async def change_price(order_id, stop_loss_price=None, **kwargs):
            price = Decimal(str(stop_loss_price))
            repriced.append((str(order_id), price))
            state["stops"][str(order_id)]["price"] = price
            return {"success": True}

        self.connector.change_stop_plan_price = change_price
        self._mid(Decimal("0.085"))
        executor = self._executor(
            exit_arming="ON_ANY_FILL",
            take_profit={"percent": "0.10", "execution": "SINGLE", "order_type": "LIMIT"},
            stop_loss={"percent": "0.05", "execution": "SINGLE"},
        )
        self._run(executor.control_task())
        order = self.orders[self.buy_ids[-1][0]]
        order.executed_amount_base = Decimal("50")
        order.executed_amount_quote = Decimal("4")
        order.current_state = OrderState.PARTIALLY_FILLED
        self._run(executor.control_task())
        stop_id = "stop-1"
        old_price = state["stops"][stop_id]["price"]
        order.executed_amount_base = Decimal("100")
        order.executed_amount_quote = Decimal("9")
        order.current_state = OrderState.FILLED
        self._mid(Decimal("0.092"))
        for _ in range(6):
            self._run(executor.control_task())
            self.assertIn(stop_id, state["stops"])
            self.assertNotIn(stop_id, state["cancelled"])
        self.assertEqual([], state["cancelled"])
        self.assertGreaterEqual(len(repriced), 1)
        self.assertEqual(stop_id, repriced[0][0])
        self.assertNotEqual(old_price, state["stops"][stop_id]["price"])
        self.assertEqual(Decimal("100"), state["stops"][stop_id]["amount"])
        self.assertEqual([stop_id], executor.get_custom_info()["exchange_stop_order_ids"])
        self.assertEqual("", executor.get_custom_info()["exchange_stop_error"])
        self.assertEqual(Decimal("100"), self._open_limit_closes()[0].amount)

    def test_failed_reprice_posts_the_take_profit_within_the_hold(self):
        state = self._install_mexc_stop_book()

        async def change_price(*args, **kwargs):
            raise RuntimeError("reprice rejected")

        self.connector.change_stop_plan_price = change_price
        self._mid(Decimal("0.085"))
        executor = self._executor(
            exit_arming="ON_ANY_FILL",
            take_profit={"percent": "0.10", "execution": "SINGLE", "order_type": "LIMIT"},
            stop_loss={"percent": "0.05", "execution": "SINGLE"},
        )
        self._run(executor.control_task())
        order = self.orders[self.buy_ids[-1][0]]
        order.executed_amount_base = Decimal("50")
        order.executed_amount_quote = Decimal("4")
        order.current_state = OrderState.PARTIALLY_FILLED
        self._run(executor.control_task())
        order.executed_amount_base = Decimal("100")
        order.executed_amount_quote = Decimal("9")
        order.current_state = OrderState.FILLED
        self._mid(Decimal("0.092"))
        self._run(executor.control_task())
        state["fail"] = 100
        before = len(state["placed"])
        start = self.now
        posted_at = None
        for _ in range(180):
            self._run(executor.control_task())
            if posted_at is None and self._open_limit_closes():
                posted_at = self.now
            self.now += 1
        self.assertIsNotNone(posted_at)
        self.assertLessEqual(posted_at - start, float(executor.config.exchange_stop_hold_seconds))
        self.assertLessEqual(len(state["placed"]) - before, 10)
        self.assertNotEqual("", executor.get_custom_info()["exchange_stop_error"])
        self.assertTrue(self._open_limit_closes())

    def test_unsupported_stop_posts_the_take_profit_once(self):
        executor = self._executor(
            protective_exchange_stop=True,
            take_profit={"percent": "0.10", "execution": "SINGLE", "order_type": "LIMIT"},
            stop_loss={"percent": "0.05", "execution": "SINGLE"},
        )
        self._open_entry(executor)
        self._run(executor.control_task())
        limits = [call for call in self._close_calls() if call[1][3] == OrderType.LIMIT]
        self.assertEqual(1, len(limits))
        order_id = self.buy_ids[-1][0]
        for _ in range(30):
            self.now += 2
            self._run(executor.control_task())
        limits = [call for call in self._close_calls() if call[1][3] == OrderType.LIMIT]
        self.assertEqual(1, len(limits))
        self.assertEqual(OrderState.OPEN, self.orders[order_id].current_state)
        cancelled_ids = [call.args[2] for call in self.strategy.cancel.call_args_list]
        self.assertNotIn(order_id, cancelled_ids)
        self.assertIn("place_position_stop_order", executor.get_custom_info()["exchange_stop_error"])

    def _drive_stop_retries(self, executor, state, seconds):
        """
        One control tick per second. Returns place times, the moment MEXC
        starts accepting, the moment a full stop is live, and whether a
        take profit was open at the end of each tick.
        """
        accept_at = None
        recovered_at = None
        place_times = []
        seen = len(state["placed"])
        on_book = []
        for _ in range(seconds):
            self._run(executor.control_task())
            added = len(state["placed"]) - seen
            place_times.extend([self.now] * added)
            seen = len(state["placed"])
            if accept_at is None and state["fail"] == 0 and seen > 0:
                accept_at = self.now
            stop = next(iter(state["stops"].values()), None)
            if (
                recovered_at is None
                and stop is not None
                and stop["amount"] >= Decimal("100")
            ):
                recovered_at = self.now
            on_book.append(bool(self._open_limit_closes()))
            self.now += 1
        return place_times, accept_at, recovered_at, on_book

    def _assert_full_stop_and_take_profit(self, executor, state):
        self.assertEqual(1, len(state["stops"]))
        stop = next(iter(state["stops"].values()))
        self.assertEqual(Decimal("100"), stop["amount"])
        self.assertTrue(executor.get_custom_info()["has_exchange_stop"])
        opens = self._open_limit_closes()
        self.assertEqual(1, len(opens))
        self.assertEqual(Decimal("100"), opens[0].amount)

    def test_k2_two_rejects_then_a_full_stop_and_take_profit(self):
        state = self._install_mexc_stop_book(fail_times=2)
        self._mid(Decimal("0.085"))
        executor = self._executor(
            take_profit={"price": "0.10", "execution": "SINGLE", "order_type": "LIMIT"},
            stop_loss={"price": "0.07", "execution": "SINGLE"},
        )
        self._open_entry(executor)
        place_times, accept_at, recovered_at, _on_book = self._drive_stop_retries(executor, state, 300)
        self._assert_full_stop_and_take_profit(executor, state)
        self.assertIsNotNone(accept_at)
        self.assertIsNotNone(recovered_at)
        self.assertLessEqual(self._places_in_any_two_seconds(place_times), 5)

    def test_k3_three_rejects_then_a_full_stop_and_take_profit(self):
        state = self._install_mexc_stop_book(fail_times=3)
        self._mid(Decimal("0.085"))
        executor = self._executor(
            take_profit={"price": "0.10", "execution": "SINGLE", "order_type": "LIMIT"},
            stop_loss={"price": "0.07", "execution": "SINGLE"},
        )
        self._open_entry(executor)
        place_times, _accept_at, recovered_at, _on_book = self._drive_stop_retries(executor, state, 300)
        self._assert_full_stop_and_take_profit(executor, state)
        self.assertIsNotNone(recovered_at)
        self.assertLessEqual(self._places_in_any_two_seconds(place_times), 5)

    def test_k4_four_rejects_recover_on_the_slow_retry(self):
        state = self._install_mexc_stop_book(fail_times=4)
        self._mid(Decimal("0.085"))
        executor = self._executor(
            take_profit={"price": "0.10", "execution": "SINGLE", "order_type": "LIMIT"},
            stop_loss={"price": "0.07", "execution": "SINGLE"},
        )
        self._open_entry(executor)
        place_times, accept_at, recovered_at, _on_book = self._drive_stop_retries(executor, state, 300)
        self._assert_full_stop_and_take_profit(executor, state)
        self.assertIsNotNone(accept_at)
        self.assertIsNotNone(recovered_at)
        self.assertLessEqual(
            recovered_at - accept_at,
            float(executor.config.stop_loss_fallback_seconds),
        )
        self.assertLessEqual(self._places_in_any_two_seconds(place_times), 5)
        self.assertGreaterEqual(len(state["placed"]), 5)

    def test_k6_six_rejects_recover_on_the_slow_retry(self):
        state = self._install_mexc_stop_book(fail_times=6)
        self._mid(Decimal("0.085"))
        executor = self._executor(
            take_profit={"price": "0.10", "execution": "SINGLE", "order_type": "LIMIT"},
            stop_loss={"price": "0.07", "execution": "SINGLE"},
        )
        self._open_entry(executor)
        place_times, accept_at, recovered_at, _on_book = self._drive_stop_retries(executor, state, 300)
        self._assert_full_stop_and_take_profit(executor, state)
        self.assertLessEqual(
            recovered_at - accept_at,
            float(executor.config.stop_loss_fallback_seconds),
        )
        self.assertLessEqual(self._places_in_any_two_seconds(place_times), 5)

    def test_k4_no_tp_four_rejects_still_place_the_stop(self):
        state = self._install_mexc_stop_book(fail_times=4)
        self._mid(Decimal("0.085"))
        executor = self._executor(stop_loss={"price": "0.07", "execution": "SINGLE"})
        self._open_entry(executor)
        _times, _accept_at, recovered_at, on_book = self._drive_stop_retries(executor, state, 300)
        self.assertIsNotNone(recovered_at)
        self.assertEqual(1, len(state["stops"]))
        self.assertEqual(Decimal("100"), next(iter(state["stops"].values()))["amount"])
        self.assertTrue(executor.get_custom_info()["has_exchange_stop"])
        self.assertFalse(any(on_book))
        self.assertEqual([], self._open_limit_closes())

    def test_permanent_stop_rejects_clear_the_take_profit_for_at_most_two_ticks(self):
        state = self._install_mexc_stop_book(fail_times=100)
        self._mid(Decimal("0.085"))
        executor = self._executor(
            take_profit={"price": "0.10", "execution": "SINGLE", "order_type": "LIMIT"},
            stop_loss={"price": "0.07", "execution": "SINGLE"},
        )
        self._open_entry(executor)
        place_times, _accept_at, recovered_at, on_book = self._drive_stop_retries(executor, state, 300)
        self.assertIsNone(recovered_at)
        self.assertIsNone(executor.get_custom_info()["exchange_stop_order_id"])
        self.assertIn("Next try", executor.get_custom_info()["exchange_stop_error"])
        self.assertNotIn("4.0s", executor.get_custom_info()["exchange_stop_error"])
        self.assertLessEqual(self._places_in_any_two_seconds(place_times), 5)
        posted = next((index for index, on in enumerate(on_book) if on), None)
        self.assertIsNotNone(posted)
        resting = on_book[posted:]
        for start in range(len(resting)):
            window = resting[start:start + 30]
            off = sum(1 for on in window if not on)
            self.assertLessEqual(off, 2)
        self.assertTrue(self._open_limit_closes())

    def test_h3_three_rejects_then_accept_leaves_a_live_stop(self):
        state = self._install_mexc_stop_book(fail_times=3)
        self._mid(Decimal("0.085"))
        executor = self._executor(
            take_profit={"percent": "0.10", "execution": "SINGLE", "order_type": "LIMIT"},
            stop_loss={"percent": "0.05", "execution": "SINGLE"},
        )
        self._open_entry(executor)
        _times, _accept_at, recovered_at, _on_book = self._drive_stop_retries(executor, state, 300)
        self.assertIsNotNone(recovered_at)
        self.assertTrue(executor.get_custom_info()["has_exchange_stop"])
        self.assertEqual(Decimal("100"), next(iter(state["stops"].values()))["amount"])

    def test_h4_four_rejects_then_accept_leaves_a_live_stop(self):
        state = self._install_mexc_stop_book(fail_times=4)
        self._mid(Decimal("0.085"))
        executor = self._executor(
            take_profit={"percent": "0.10", "execution": "SINGLE", "order_type": "LIMIT"},
            stop_loss={"percent": "0.05", "execution": "SINGLE"},
        )
        self._open_entry(executor)
        _times, accept_at, recovered_at, _on_book = self._drive_stop_retries(executor, state, 300)
        self.assertIsNotNone(recovered_at)
        self.assertLessEqual(
            recovered_at - accept_at,
            float(executor.config.stop_loss_fallback_seconds),
        )
        self.assertTrue(executor.get_custom_info()["has_exchange_stop"])

    def test_gr3_failed_reprice_lands_a_stop_once_mexc_accepts(self):
        state = self._install_mexc_stop_book()

        async def change_price(*args, **kwargs):
            raise RuntimeError("reprice rejected")

        self.connector.change_stop_plan_price = change_price
        self._mid(Decimal("0.085"))
        executor = self._executor(
            exit_arming="ON_ANY_FILL",
            take_profit={"percent": "0.10", "execution": "SINGLE", "order_type": "LIMIT"},
            stop_loss={"percent": "0.05", "execution": "SINGLE"},
        )
        self._run(executor.control_task())
        order = self.orders[self.buy_ids[-1][0]]
        order.executed_amount_base = Decimal("50")
        order.executed_amount_quote = Decimal("4")
        order.current_state = OrderState.PARTIALLY_FILLED
        self._run(executor.control_task())
        self.assertIn("stop-1", state["stops"])
        order.executed_amount_base = Decimal("100")
        order.executed_amount_quote = Decimal("9")
        order.current_state = OrderState.FILLED
        self._mid(Decimal("0.092"))
        state["fail"] = 4
        _times, _accept_at, recovered_at, _on_book = self._drive_stop_retries(executor, state, 300)
        self.assertIsNotNone(recovered_at)
        self.assertTrue(executor.get_custom_info()["has_exchange_stop"])
        self.assertGreaterEqual(next(iter(state["stops"].values()))["amount"], Decimal("50"))

    def test_rs2_repeated_rejects_then_accept_leaves_a_live_stop(self):
        state = self._install_mexc_stop_book(fail_times=6)
        self._mid(Decimal("0.085"))
        executor = self._executor(
            take_profit={"percent": "0.10", "execution": "SINGLE", "order_type": "LIMIT"},
            stop_loss={"percent": "0.05", "execution": "SINGLE"},
        )
        self._open_entry(executor)
        place_times, accept_at, recovered_at, _on_book = self._drive_stop_retries(executor, state, 300)
        self.assertIsNotNone(recovered_at)
        self.assertLessEqual(
            recovered_at - accept_at,
            float(executor.config.stop_loss_fallback_seconds),
        )
        self.assertTrue(executor.get_custom_info()["has_exchange_stop"])
        self.assertEqual(Decimal("100"), next(iter(state["stops"].values()))["amount"])
        self.assertLessEqual(self._places_in_any_two_seconds(place_times), 5)

    def test_repeated_stop_rejects_leave_the_take_profit_resting(self):
        state = self._install_mexc_stop_book(fail_times=100)
        self._mid(Decimal("0.085"))
        executor = self._executor(
            take_profit={"percent": "0.10", "execution": "SINGLE", "order_type": "LIMIT"},
            stop_loss={"percent": "0.05", "execution": "SINGLE"},
        )
        self._open_entry(executor)
        place_times, _accept_at, recovered_at, on_book = self._drive_stop_retries(executor, state, 120)
        self.assertIsNone(recovered_at)
        self.assertIsNone(executor.get_custom_info()["exchange_stop_order_id"])
        self.assertNotEqual("", executor.get_custom_info()["exchange_stop_error"])
        self.assertLessEqual(self._places_in_any_two_seconds(place_times), 5)
        self.assertLessEqual(len(state["placed"]), 12)
        posted = next((index for index, on in enumerate(on_book) if on), None)
        self.assertIsNotNone(posted)
        resting = on_book[posted:]
        for start in range(len(resting)):
            off = sum(1 for on in resting[start:start + 30] if not on)
            self.assertLessEqual(off, 2)
        self.assertTrue(self._open_limit_closes())

    def test_stop_loss_keeps_the_backstop_while_the_take_profit_cancel_is_unconfirmed(self):
        cancelled = []
        self._install_mexc_stop_book()

        async def cancel_stop(order_ids):
            cancelled.extend(list(order_ids))
            return {"success": True}

        self.connector.cancel_stop_orders = cancel_stop
        self.connector.is_cancel_confirmed = lambda order_id: False
        self._mid(Decimal("0.085"))
        executor = self._executor(
            stop_loss_fallback_seconds=5,
            take_profit={"percent": "0.10", "execution": "SINGLE", "order_type": "LIMIT"},
            stop_loss={"percent": "0.05", "execution": "SINGLE"},
        )
        self._open_entry(executor)
        self._run(executor.control_task())
        executor._exchange_stop_amount = Decimal("40")
        take_profit = self.orders[self.buy_ids[-1][0]]
        take_profit.amount = Decimal("30")
        self._mid(Decimal("0.07"))
        for _ in range(4):
            self._run(executor.control_task())
            self.assertEqual([], cancelled)
            self.assertEqual("stop-1", executor.get_custom_info()["exchange_stop_order_id"])
            self.assertEqual(["stop-1"], executor.get_custom_info()["exchange_stop_order_ids"])
            self.assertTrue(executor.get_custom_info()["has_exchange_stop"])
            self.assertEqual([], [call for call in self._close_calls() if call[1][3] == OrderType.MARKET])
        self.now += 5
        self._run(executor.control_task())
        markets = [call for call in self._close_calls() if call[1][3] == OrderType.MARKET]
        self.assertEqual(1, len(markets))
        self.assertEqual(Decimal("30"), markets[0][1][2])
        self.assertEqual([], cancelled)
        self.assertEqual("stop-1", executor.get_custom_info()["exchange_stop_order_id"])
        self.assertTrue(executor.get_custom_info()["has_exchange_stop"])
        self.assertEqual("stop_loss", executor.get_custom_info()["exit_active"])

    def test_missing_position_is_not_a_flat_close(self):
        self._install_exchange_stop()
        self._mid(Decimal("0.085"))
        executor = self._executor(stop_loss={"percent": "0.05", "execution": "SINGLE"})
        self._open_entry(executor)
        self._run(executor.control_task())
        self.connector.account_positions = {}
        self._mid(Decimal("0.07"))
        before = len(self._close_calls())
        self._run(executor.control_task())
        self.assertEqual(before + 1, len(self._close_calls()))
        self.assertEqual(OrderType.MARKET, self._close_calls()[-1][1][3])
        self.assertNotEqual("DONE", executor.get_custom_info()["exit_state"])

        stale = self._executor(stop_loss={"percent": "0.05", "execution": "SINGLE"})
        self._mid(Decimal("0.085"))
        self._open_entry(stale)
        self._run(stale.control_task())
        stale._last_fill_timestamp = self.now
        self.connector.account_positions = {
            "aiot": SimpleNamespace(
                trading_pair="AIOT-USDT",
                position_side=PositionSide.LONG,
                amount=Decimal("0"),
                update_timestamp=self.now - 5,
            )
        }
        self._mid(Decimal("0.07"))
        before = len(self._close_calls())
        self._run(stale.control_task())
        self.assertEqual(before + 1, len(self._close_calls()))
        self.assertNotEqual(CloseType.STOP_LOSS, stale.close_type)

        fresh = self._executor(stop_loss={"percent": "0.05", "execution": "SINGLE"})
        self._mid(Decimal("0.085"))
        self._open_entry(fresh)
        self._run(fresh.control_task())
        fresh._last_fill_timestamp = self.now
        self.connector.account_positions = {
            "aiot": SimpleNamespace(
                trading_pair="AIOT-USDT",
                position_side=PositionSide.LONG,
                amount=Decimal("0"),
                update_timestamp=self.now + 1,
            )
        }
        self._mid(Decimal("0.07"))
        before = len(self._close_calls())
        self._run(fresh.control_task())
        self.assertEqual(before, len(self._close_calls()))
        self.assertEqual(CloseType.STOP_LOSS, fresh.close_type)
        self.assertEqual("DONE", fresh.get_custom_info()["exit_state"])

        queried = self._executor(stop_loss={"percent": "0.05", "execution": "SINGLE"})
        self._mid(Decimal("0.085"))
        self._open_entry(queried)
        self._run(queried.control_task())
        self.connector.account_positions = {}

        async def read_stop(order_id):
            return {"state": 3}

        self.connector.get_position_stop_order = read_stop
        self._mid(Decimal("0.07"))
        before = len(self._close_calls())
        self._run(queried.control_task())
        self.assertEqual(before, len(self._close_calls()))
        self.assertEqual(CloseType.STOP_LOSS, queried.close_type)
        self.assertEqual(Decimal("0"), Decimal(queried.get_custom_info()["open_base"]))

    def test_fired_exchange_stop_credits_its_own_size(self):
        _, cancelled = self._install_exchange_stop(
            cancel_result={"success": False, "message": "already triggered"},
        )
        self._mid(Decimal("0.085"))
        executor = self._executor(stop_loss={"percent": "0.05", "execution": "SINGLE"})
        self._open_entry(executor)
        self._run(executor.control_task())
        executor._exchange_stop_amount = Decimal("40")
        if executor._exchange_stops:
            executor._exchange_stops[0]["amount"] = Decimal("40")
        self.connector.account_positions = {
            "aiot": SimpleNamespace(
                trading_pair="AIOT-USDT",
                position_side=PositionSide.LONG,
                amount=Decimal("0"),
                update_timestamp=None,
            )
        }
        self._mid(Decimal("0.07"))
        before = len(self._close_calls())
        self._run(executor.control_task())
        self.assertEqual(Decimal("40"), Decimal(executor.get_custom_info()["closed_base"]))
        self.assertEqual(Decimal("60"), Decimal(executor.get_custom_info()["open_base"]))
        self.assertEqual(before + 1, len(self._close_calls()))
        self.assertEqual(Decimal("60"), self._close_calls()[-1][1][2])
        self.assertEqual(OrderType.MARKET, self._close_calls()[-1][1][3])
        self.assertNotEqual(CloseType.FAILED, executor.close_type)
        self.assertGreaterEqual(len(cancelled), 1)

    def test_one_of_two_stops_firing_credits_only_that_stop(self):
        state = self._install_mexc_stop_book()
        self._mid(Decimal("0.085"))
        executor = self._executor(stop_loss={"percent": "0.05", "execution": "SINGLE"})
        self._open_entry(executor)
        self._run(executor.control_task())
        price = executor._exchange_stop_price
        other_price = price + Decimal("0.001")
        executor._exchange_stops = [
            {"id": "stop-1", "amount": Decimal("50"), "price": price},
            {"id": "stop-2", "amount": Decimal("50"), "price": other_price},
        ]
        executor._exchange_stop_order_id = "stop-1"
        executor._exchange_stop_amount = Decimal("100")
        executor._exchange_stop_price = price
        state["stops"] = {
            "stop-1": {"amount": Decimal("50"), "price": price},
            "stop-2": {"amount": Decimal("50"), "price": other_price},
        }
        seen = {}

        async def read_stop(order_id):
            if str(order_id) == "stop-2":
                return {"state": 3}
            return {"state": 1}

        async def cancel_stop(order_ids):
            seen["ids"] = [str(order_id) for order_id in order_ids]
            seen["tracked"] = list(executor.get_custom_info()["exchange_stop_order_ids"])
            seen["open"] = executor.get_custom_info()["open_base"]
            for stop_id in seen["ids"]:
                state["stops"].pop(stop_id, None)
            state["cancelled"].extend(seen["ids"])
            return {"success": True}

        self.connector.get_position_stop_order = read_stop
        self.connector.cancel_stop_orders = cancel_stop
        self._mid(Decimal("0.07"))
        self._run(executor.control_task())
        self.assertEqual(["stop-1"], seen["tracked"])
        self.assertEqual(Decimal("50"), Decimal(seen["open"]))
        self.assertEqual(["stop-1"], seen["ids"])
        self.assertEqual(Decimal("50"), Decimal(executor.get_custom_info()["closed_base"]))
        markets = [call for call in self._close_calls() if call[1][3] == OrderType.MARKET]
        self.assertEqual(1, len(markets))
        self.assertEqual(Decimal("50"), markets[0][1][2])
        self.assertNotIn("stop-2", seen["ids"])

    def test_unconfirmed_stop_cancel_backs_off_then_closes_the_remainder(self):
        calls = []
        self._install_mexc_stop_book()

        async def cancel_stop(order_ids):
            calls.append(self.now)
            raise RuntimeError("cancel timed out")

        self.connector.cancel_stop_orders = cancel_stop
        self._mid(Decimal("0.085"))
        executor = self._executor(
            stop_loss_fallback_seconds=5,
            stop_loss={"percent": "0.05", "execution": "SINGLE"},
        )
        self._open_entry(executor)
        self._run(executor.control_task())
        executor._exchange_stop_amount = Decimal("40")
        if executor._exchange_stops:
            executor._exchange_stops[0]["amount"] = Decimal("40")
        self._mid(Decimal("0.07"))
        for _ in range(8):
            self._run(executor.control_task())
        self.assertEqual(1, len(calls))
        self.assertEqual([], self._close_calls())
        self.assertEqual("stop-1", executor.get_custom_info()["exchange_stop_order_id"])
        self.now += 5
        self._run(executor.control_task())
        self.assertEqual(2, len(calls))
        self.assertEqual(1, len(self._close_calls()))
        self.assertEqual(Decimal("60"), self._close_calls()[-1][1][2])
        self.assertEqual(OrderType.MARKET, self._close_calls()[-1][1][3])
        self.assertEqual("stop-1", executor.get_custom_info()["exchange_stop_order_id"])
        self.assertTrue(executor.get_custom_info()["has_exchange_stop"])

    def test_rejected_stop_closes_replace_the_exchange_stop_before_failed(self):
        state = self._install_mexc_stop_book()
        placed = state["ids"]
        cancelled = state["cancelled"]
        self._mid(Decimal("0.085"))
        executor = RangeIcebergExecutor(
            self.strategy,
            self._config(stop_loss={"percent": "0.05", "execution": "SINGLE"}),
            update_interval=0.1,
            max_retries=0,
        )
        executor._status = RunnableStatus.RUNNING
        self._open_entry(executor)
        self._run(executor.control_task())
        self.assertEqual(["stop-1"], placed)
        self._mid(Decimal("0.07"))
        self._run(executor.control_task())
        self.assertIn("stop-1", cancelled)
        self.assertEqual(OrderType.MARKET, self._close_calls()[-1][1][3])
        market_id = self.buy_ids[-1][0]
        executor.process_order_failed_event(
            0, None, MarketOrderFailureEvent(self.now, market_id, OrderType.MARKET),
        )
        self.assertNotEqual(CloseType.FAILED, executor.close_type)
        self.now += 1
        self._run(executor.control_task())
        self.assertEqual(OrderType.MARKET, self._close_calls()[-1][1][3])
        second = self.buy_ids[-1][0]
        executor.process_order_failed_event(
            0, None, MarketOrderFailureEvent(self.now, second, OrderType.MARKET),
        )
        self.assertNotEqual(CloseType.FAILED, executor.close_type)
        self._run(executor.control_task())
        third = self.buy_ids[-1][0]
        self.assertEqual(OrderType.MARKET, self._close_calls()[-1][1][3])
        executor.process_order_failed_event(
            0, None, MarketOrderFailureEvent(self.now, third, OrderType.MARKET),
        )
        self.assertNotEqual(CloseType.FAILED, executor.close_type)
        self.assertNotEqual(RunnableStatus.TERMINATED, executor.status)
        self.assertIn("Re-placing", executor.get_custom_info()["exit_error"])
        self._run(executor.control_task())
        self.assertEqual(CloseType.FAILED, executor.close_type)
        self.assertEqual("stop-2", executor.get_custom_info()["exchange_stop_order_id"])
        self.assertTrue(executor.get_custom_info()["has_exchange_stop"])
        self.assertNotIn("stop-2", cancelled)
        self.assertEqual(RunnableStatus.TERMINATED, executor.status)

    def test_c4_no_stop_is_placed_while_the_stop_loss_is_closing(self):
        state = self._install_mexc_stop_book(fail_times=5)
        self._mid(Decimal("0.085"))
        executor = self._executor(stop_loss={"price": "0.08", "execution": "SINGLE"})
        self._open_entry(executor)
        self._run(executor.control_task())
        self.assertEqual(1, len(state["placed"]))
        self.assertIsNone(executor.get_custom_info()["exchange_stop_order_id"])
        self.assertFalse(executor.get_custom_info()["exchange_stop_released"])
        self._mid(Decimal("0.07"))
        self._run(executor.control_task())
        self.assertIn(executor.get_custom_info()["exit_state"], ("TRIGGERED", "CLOSING"))
        self.assertTrue(executor.get_custom_info()["exchange_stop_released"])
        placed = len(state["placed"])
        self.now += 2
        self._run(executor.control_task())
        self.now += 2
        self._run(executor.control_task())
        self.assertEqual(placed, len(state["placed"]))
        self.assertIn(executor.get_custom_info()["exit_state"], ("TRIGGERED", "CLOSING"))

    def _install_old_stop_book(self, fail_times):
        state = self._install_mexc_stop_book(fail_times=fail_times)
        self.connector.is_cancel_confirmed = None
        cancelled_at = {}

        def cancel(connector_name, trading_pair, order_id, *args, **kwargs):
            cancelled_at[order_id] = self.now
            return None

        self.strategy.cancel.side_effect = cancel
        state["cancelled_at"] = cancelled_at
        return state

    def _drive_old_stop_book(self, executor, state, seconds):
        for _ in range(seconds):
            for order_id, stamp in list(state["cancelled_at"].items()):
                order = self.orders.get(order_id)
                if order is not None and self.now > stamp and order.current_state == OrderState.OPEN:
                    order.current_state = OrderState.CANCELED
            self._run(executor.control_task())
            self.now += 1

    def test_old_mock_k4_recovers_after_the_cancel_tick(self):
        state = self._install_old_stop_book(4)
        self._mid(Decimal("0.085"))
        executor = self._executor(
            take_profit={"price": "0.10", "execution": "SINGLE", "order_type": "LIMIT"},
            stop_loss={"price": "0.07", "execution": "SINGLE"},
        )
        self._open_entry(executor)
        self._drive_old_stop_book(executor, state, 300)
        self.assertEqual(Decimal("100"), next(iter(state["stops"].values()))["amount"])
        self.assertTrue(executor.get_custom_info()["has_exchange_stop"])
        opens = self._open_limit_closes()
        self.assertEqual(1, len(opens))
        self.assertEqual(Decimal("100"), opens[0].amount)

    def test_old_mock_k6_recovers_after_the_cancel_tick(self):
        state = self._install_old_stop_book(6)
        self._mid(Decimal("0.085"))
        executor = self._executor(
            take_profit={"price": "0.10", "execution": "SINGLE", "order_type": "LIMIT"},
            stop_loss={"price": "0.07", "execution": "SINGLE"},
        )
        self._open_entry(executor)
        self._drive_old_stop_book(executor, state, 300)
        self.assertEqual(Decimal("100"), next(iter(state["stops"].values()))["amount"])
        self.assertTrue(executor.get_custom_info()["has_exchange_stop"])
        opens = self._open_limit_closes()
        self.assertEqual(1, len(opens))
        self.assertEqual(Decimal("100"), opens[0].amount)

    def _register_executors(self, *executors):
        self.strategy.executor_orchestrator = SimpleNamespace(
            active_executors={"main": list(executors)},
        )

    def test_second_long_is_refused(self):
        first = self._executor()
        second = self._executor()
        self._register_executors(first, second)
        before = len(self.buy_ids)
        second.start()
        self.assertEqual(RunnableStatus.TERMINATED, second.status)
        self.assertEqual(CloseType.FAILED, second.close_type)
        self.assertEqual(
            "another range iceberg is already open on AIOT-USDT LONG",
            second.get_custom_info()["side_guard_error"],
        )
        self.assertEqual(
            "another range iceberg is already open on AIOT-USDT LONG",
            second.get_custom_info()["status_reason"],
        )
        self.assertEqual(before, len(self.buy_ids))
        self._run(second.control_task())
        self.assertEqual(before, len(self.buy_ids))

    def test_long_and_short_are_both_allowed(self):
        first = self._executor()
        second = self._executor(side=TradeType.SELL)
        self._register_executors(first, second)
        second.start()
        self.assertEqual(RunnableStatus.RUNNING, second.status)
        self.assertEqual("", second.get_custom_info()["side_guard_error"])

    def test_second_long_is_allowed_after_the_first_terminates(self):
        first = self._executor()
        first.stop()
        second = self._executor()
        self._register_executors(first, second)
        second.start()
        self.assertEqual(RunnableStatus.RUNNING, second.status)
        self.assertEqual("", second.get_custom_info()["side_guard_error"])

    def test_different_account_is_allowed_on_the_same_side(self):
        self.connector.account_name = "alpha"
        first = self._executor()
        self.connector.account_name = "beta"
        second = self._executor()
        self._register_executors(first, second)
        second.start()
        self.assertEqual("alpha", first._account_identity)
        self.assertEqual("beta", second._account_identity)
        self.assertEqual(RunnableStatus.RUNNING, second.status)
        self.assertEqual("", second.get_custom_info()["side_guard_error"])

    def test_parked_long_blocks_the_next_long(self):
        self._install_mexc_stop_book()
        self._mid(Decimal("0.085"))
        first = self._executor(stop_loss={"price": "0.07", "execution": "SINGLE"})
        self._open_entry(first)
        self._run(first.control_task())
        self.assertTrue(first.get_custom_info()["has_exchange_stop"])
        first.early_stop(keep_position=True)
        self.assertEqual(RunnableStatus.TERMINATED, first.status)
        self.assertEqual(CloseType.POSITION_HOLD, first.close_type)
        second = self._executor(stop_loss={"price": "0.07", "execution": "SINGLE"})
        self._register_executors(first, second)
        before = len(self.buy_ids)
        second.start()
        self.assertEqual(RunnableStatus.TERMINATED, second.status)
        self.assertEqual(CloseType.FAILED, second.close_type)
        self.assertEqual(
            "another range iceberg is already open on AIOT-USDT LONG",
            second.get_custom_info()["side_guard_error"],
        )
        self.assertEqual(before, len(self.buy_ids))

    def test_resumed_long_is_refused_when_the_side_is_open(self):
        occupant = self._executor()
        resumed = self._executor(stop_loss={"price": "0.07", "execution": "SINGLE"})
        resumed.restore_from_custom_info(resumed.get_custom_info())
        self._register_executors(occupant, resumed)
        before = len(self.buy_ids)
        self._run(resumed.control_task())
        self.assertEqual(RunnableStatus.TERMINATED, resumed.status)
        self.assertEqual(CloseType.FAILED, resumed.close_type)
        self.assertEqual(
            "another range iceberg is already open on AIOT-USDT LONG",
            resumed.get_custom_info()["side_guard_error"],
        )
        self.assertEqual(before, len(self.buy_ids))

    def test_foreign_position_stop_is_not_overwritten(self):
        state = self._install_mexc_stop_book()
        state["stops"]["stop-foreign"] = {
            "amount": Decimal("100"),
            "price": Decimal("0.06965"),
        }

        async def list_open(trading_pair=None):
            return [
                {"id": stop_id, "positionType": 1, "symbol": "AIOT_USDT"}
                for stop_id in state["stops"]
            ]

        self.connector.list_open_stop_orders = list_open
        self._mid(Decimal("0.085"))
        executor = self._executor(stop_loss={"price": "0.07", "execution": "SINGLE"})
        self._open_entry(executor)
        placed = len(state["placed"])
        self._run(executor.control_task())
        self.assertEqual(placed, len(state["placed"]))
        self.assertIsNone(executor.get_custom_info()["exchange_stop_order_id"])
        self.assertIn("does not own", executor.get_custom_info()["exchange_stop_error"])
        self.assertIn("stop-foreign", executor.get_custom_info()["exchange_stop_error"])
        self.assertEqual(Decimal("100"), state["stops"]["stop-foreign"]["amount"])
        self._mid(Decimal("0.06"))
        self._run(executor.control_task())
        markets = [call for call in self._close_calls() if call[1][3] == OrderType.MARKET]
        self.assertEqual(1, len(markets))
        self.assertEqual(placed, len(state["placed"]))
        self.assertEqual(Decimal("100"), state["stops"]["stop-foreign"]["amount"])
        self.assertNotIn("stop-foreign", state["cancelled"])

    def test_same_price_foreign_stop_id_is_not_remembered(self):
        state = self._install_mexc_stop_book()
        state["stops"]["stop-foreign"] = {
            "amount": Decimal("50"),
            "price": Decimal("0.06965"),
        }

        async def list_open(trading_pair=None):
            return [{"id": "stop-foreign", "positionType": 1, "symbol": "AIOT_USDT"}]

        self.connector.list_open_stop_orders = list_open
        self._mid(Decimal("0.085"))
        executor = self._executor(stop_loss={"price": "0.07", "execution": "SINGLE"})
        self._open_entry(executor)
        self._run(executor.control_task())
        self.assertEqual([], state["placed"])
        self.assertIsNone(executor.get_custom_info()["exchange_stop_order_id"])
        self.assertEqual(Decimal("50"), state["stops"]["stop-foreign"]["amount"])
        self.assertIn("stop-foreign", executor.get_custom_info()["exchange_stop_error"])

    def test_a_short_exchange_stop_does_not_block_the_long(self):
        state = self._install_mexc_stop_book()

        async def list_open(trading_pair=None):
            return [{"id": "stop-short", "positionType": 2, "symbol": "AIOT_USDT"}]

        self.connector.list_open_stop_orders = list_open
        self._mid(Decimal("0.085"))
        executor = self._executor(stop_loss={"price": "0.07", "execution": "SINGLE"})
        self._open_entry(executor)
        self._run(executor.control_task())
        self.assertEqual("stop-1", executor.get_custom_info()["exchange_stop_order_id"])
        self.assertEqual(1, len(state["placed"]))
        self.assertNotIn("stop-short", state["cancelled"])

    def test_update_config_grows_the_total_and_schedules_another_slice(self):
        executor = self._executor(levels=2, total_amount=Decimal("100"))
        self._run(executor.control_task())
        self.assertEqual(Decimal("50"), self.buy_ids[-1][1])
        result = executor.update_config({"total_amount": "200"})
        self.assertEqual({"accepted": True, "reason": ""}, result)
        self.assertEqual("200", executor.get_custom_info()["total_amount"])
        self._run(executor.control_task())
        self.assertEqual(Decimal("200"), executor.config.total_amount)
        placed = [
            amount for _order_id, amount, _price, _order_type, action in self.buy_ids
            if action == PositionAction.OPEN
        ]
        self.assertGreater(sum(placed), Decimal("50"))
        self.assertEqual(Decimal("200"), sum(slot.target_amount for slot in executor._slices))

    def test_update_config_shrinks_an_unfilled_slice(self):
        executor = self._executor(total_amount=Decimal("100"))
        self._run(executor.control_task())
        self.assertEqual(Decimal("100"), self.buy_ids[-1][1])
        result = executor.update_config({"total_amount": "40"})
        self.assertTrue(result["accepted"])
        self._run(executor.control_task())
        order_id = self.buy_ids[0][0]
        self.orders[order_id].current_state = OrderState.CANCELED
        self._run(executor.control_task())
        self.assertEqual(Decimal("40"), executor.config.total_amount)
        self.assertEqual(Decimal("40"), self.buy_ids[-1][1])
        self.assertGreaterEqual(self.strategy.cancel.call_count, 1)

    def test_update_config_rejects_a_shrink_below_the_fill(self):
        executor = self._executor()
        self._open_entry(executor)
        result = executor.update_config({"total_amount": "50"})
        self.assertFalse(result["accepted"])
        self.assertIn("below the filled amount", result["reason"])
        self.assertEqual(Decimal("100"), executor.config.total_amount)
        self.assertEqual(0, self.strategy.cancel.call_count)

    def test_update_config_reprices_a_slice_outside_the_new_range(self):
        executor = self._executor()
        self._run(executor.control_task())
        self.assertEqual(Decimal("0.08"), self.buy_ids[-1][2])
        result = executor.update_config({"price_low": "0.091", "price_high": "0.10"})
        self.assertTrue(result["accepted"], result)
        self._run(executor.control_task())
        self.orders[self.buy_ids[0][0]].current_state = OrderState.CANCELED
        self._run(executor.control_task())
        self.assertEqual(Decimal("0.091"), executor.config.price_low)
        self.assertEqual(Decimal("0.10"), executor.config.price_high)
        self.assertEqual(Decimal("0.091"), self.buy_ids[-1][2])
        self.assertGreaterEqual(self.strategy.cancel.call_count, 1)

    def test_update_config_reprices_one_slice_per_rate_limit_window(self):
        executor = self._executor(levels=2, min_order_interval=2)
        self._run(executor.control_task())
        self.now += 2
        self._run(executor.control_task())
        self.assertEqual(2, len(self.buy_ids))
        self.now += 2
        result = executor.update_config({"price_low": "0.11", "price_high": "0.12"})
        self.assertTrue(result["accepted"], result)
        self._run(executor.control_task())
        self.assertEqual(1, self.strategy.cancel.call_count)
        self._run(executor.control_task())
        self.assertEqual(1, self.strategy.cancel.call_count)
        self.now += 2
        self._run(executor.control_task())
        self.assertEqual(2, self.strategy.cancel.call_count)

    def test_update_config_reprices_the_take_profit_and_the_exchange_stop(self):
        state = self._install_mexc_stop_book()
        changes = []

        async def change_price(order_id, stop_loss_price=None, **kwargs):
            price = Decimal(str(stop_loss_price))
            changes.append((str(order_id), price))
            if order_id in state["stops"]:
                state["stops"][order_id]["price"] = price
            return {"success": True}

        self.connector.change_stop_plan_price = change_price
        executor = self._executor(
            take_profit={"percent": "0.10", "execution": "SINGLE", "order_type": "LIMIT"},
            stop_loss={"price": "0.07", "execution": "SINGLE"},
        )
        _order_id, _base, price = self._open_entry(executor)
        self._run(executor.control_task())
        self.assertEqual("stop-1", executor.get_custom_info()["exchange_stop_order_id"])
        result = executor.update_config({
            "take_profit": {"percent": "0.20", "execution": "SINGLE", "order_type": "LIMIT"},
            "stop_loss": {"price": "0.06", "execution": "SINGLE"},
        })
        self.assertTrue(result["accepted"], result)
        self._run(executor.control_task())
        self.assertEqual([("stop-1", Decimal("0.0597"))], changes)
        self.assertEqual(Decimal("0.0597"), Decimal(executor.get_custom_info()["exchange_stop_trigger_price"]))
        self.assertEqual([], state["cancelled"])
        self.assertEqual("stop-1", executor.get_custom_info()["exchange_stop_order_id"])
        closes = [call for call in self._close_calls() if call[1][3] == OrderType.LIMIT]
        self.assertEqual(price * Decimal("1.20"), closes[-1][1][4])

    def test_update_config_rejects_an_edit_while_the_stop_loss_is_closing(self):
        self._mid(Decimal("0.085"))
        executor = self._executor(stop_loss={"price": "0.07", "execution": "SINGLE"})
        self._open_entry(executor)
        self._mid(Decimal("0.06"))
        self._run(executor.control_task())
        self.assertIn(executor.get_custom_info()["exit_state"], ("TRIGGERED", "CLOSING"))
        result = executor.update_config({"total_amount": "150"})
        self.assertFalse(result["accepted"])
        self.assertEqual("a stop loss close is in progress", result["reason"])
        self.assertEqual(Decimal("100"), executor.config.total_amount)

    def test_update_config_rejects_a_stop_on_the_wrong_side_of_the_range(self):
        executor = self._executor()
        result = executor.update_config({
            "stop_loss": {"price": "0.085", "execution": "SINGLE"},
            "take_profit": {"price": "0.07", "execution": "SINGLE"},
        })
        self.assertFalse(result["accepted"])
        self.assertTrue(result["reason"])
        self.assertIsNone(executor.config.stop_loss)
        self.assertIsNone(executor.config.take_profit)

    def test_update_config_is_restored_from_custom_info(self):
        executor = self._executor(
            take_profit={"percent": "0.10", "execution": "SINGLE", "order_type": "LIMIT"},
            stop_loss={"price": "0.07", "execution": "SINGLE"},
        )
        self._open_entry(executor)
        result = executor.update_config({
            "total_amount": "150",
            "price_low": "0.07",
            "price_high": "0.11",
            "take_profit": {"percent": "0.20", "execution": "SINGLE", "order_type": "LIMIT"},
            "stop_loss": {"price": "0.05", "execution": "SINGLE"},
        })
        self.assertTrue(result["accepted"], result)
        self._run(executor.control_task())
        info = executor.get_custom_info()
        self.assertEqual("150", info["total_amount"])
        self.assertEqual("0.07", info["price_low"])
        self.assertEqual("0.11", info["price_high"])
        self.assertEqual("0.2", info["take_profit"]["percent"])
        self.assertEqual("0.05", info["stop_loss"]["price"])
        resumed = self._executor()
        resumed.restore_from_custom_info(info)
        self.assertEqual(Decimal("150"), resumed.config.total_amount)
        self.assertEqual(Decimal("0.07"), resumed.config.price_low)
        self.assertEqual(Decimal("0.11"), resumed.config.price_high)
        self.assertEqual(Decimal("0.20"), resumed.config.take_profit.percent)
        self.assertEqual(Decimal("0.05"), resumed.config.stop_loss.price)
        self.assertEqual("150", resumed.get_custom_info()["total_amount"])

    def test_update_config_does_not_open_a_second_executor_on_the_side(self):
        first = self._executor()
        second = self._executor()
        self._register_executors(first, second)
        result = first.update_config({"total_amount": "120"})
        self.assertTrue(result["accepted"])
        before = len(self.buy_ids)
        second.start()
        self.assertEqual(RunnableStatus.TERMINATED, second.status)
        self.assertEqual(CloseType.FAILED, second.close_type)
        self.assertEqual(
            "another range iceberg is already open on AIOT-USDT LONG",
            second.get_custom_info()["side_guard_error"],
        )
        self.assertEqual(before, len(self.buy_ids))

    def test_update_config_rejects_a_parked_position_hold(self):
        executor = self._executor()
        executor.early_stop(keep_position=True)
        result = executor.update_config({"total_amount": "120"})
        self.assertFalse(result["accepted"])
        self.assertEqual("a parked position hold cannot be edited", result["reason"])

    def test_update_config_rejects_a_ladder_that_cannot_be_built(self):
        executor = self._executor(total_amount=Decimal("100"))
        self._run(executor.control_task())
        result = executor.update_config({"total_amount": "5"})
        self.assertFalse(result["accepted"])
        self.assertIn("minimum order size", result["reason"])
        self.assertEqual(Decimal("100"), executor.config.total_amount)
        self.assertFalse(executor.get_custom_info()["pending_config_edit"])
        self.assertEqual(0, self.strategy.cancel.call_count)

    def test_update_config_keeps_the_old_plan_when_apply_cannot_build(self):
        executor = self._executor(total_amount=Decimal("100"))
        self._run(executor.control_task())
        result = executor.update_config({"total_amount": "40"})
        self.assertTrue(result["accepted"], result)
        self.assertTrue(executor.get_custom_info()["pending_config_edit"])
        self.connector.get_contract_size.side_effect = RuntimeError("missing")
        cancels = self.strategy.cancel.call_count
        self._run(executor.control_task())
        self.assertEqual(Decimal("100"), executor.config.total_amount)
        self.assertEqual(cancels, self.strategy.cancel.call_count)
        info = executor.get_custom_info()
        self.assertFalse(info["pending_config_edit"])
        self.assertIn("contractSize", info["config_edit_error"])
        self.assertEqual({}, info["config_edits"])

    def test_update_config_clamps_a_shrink_when_fills_land_before_apply(self):
        executor = self._executor(total_amount=Decimal("100"))
        self._run(executor.control_task())
        order = self.orders[self.buy_ids[-1][0]]
        order.executed_amount_base = Decimal("30")
        order.executed_amount_quote = Decimal("30") * order.price
        order.current_state = OrderState.PARTIALLY_FILLED
        result = executor.update_config({"total_amount": "40"})
        self.assertTrue(result["accepted"], result)
        order.executed_amount_base = Decimal("70")
        order.executed_amount_quote = Decimal("70") * order.price
        self._run(executor.control_task())
        info = executor.get_custom_info()
        self.assertEqual(Decimal("70"), executor.config.total_amount)
        self.assertIn("clamped to the filled amount 70", info["config_edit_error"])
        self.assertEqual("70", info["config_edits"]["total_amount"])
        self.assertFalse(info["pending_config_edit"])
        self.assertIsNotNone(info["last_config_edit_applied_at"])
        self.assertGreaterEqual(self.strategy.cancel.call_count, 1)

    def test_update_config_records_a_dropped_edit_when_the_executor_stops(self):
        executor = self._executor()
        result = executor.update_config({"total_amount": "120"})
        self.assertTrue(result["accepted"], result)
        self.assertTrue(executor.get_custom_info()["pending_config_edit"])
        executor.early_stop(keep_position=True)
        info = executor.get_custom_info()
        self.assertFalse(info["pending_config_edit"])
        self.assertEqual("dropped: executor stopped before apply", info["config_edit_error"])
        self.assertEqual(Decimal("100"), executor.config.total_amount)
        self.assertEqual({}, info["config_edits"])

    def test_clearing_stop_loss_cancels_the_protective_exchange_stop(self):
        state = self._install_mexc_stop_book()
        executor = self._executor(stop_loss={"price": "0.07", "execution": "SINGLE"})
        self._open_entry(executor)
        self._run(executor.control_task())
        self.assertEqual("stop-1", executor.get_custom_info()["exchange_stop_order_id"])
        result = executor.update_config({"stop_loss": None})
        self.assertEqual(
            {"accepted": True, "reason": "clearing stop_loss cancels the protective exchange stop"},
            result,
        )
        self._run(executor.control_task())
        self.assertIsNone(executor.config.stop_loss)
        self.assertIsNone(executor.get_custom_info()["exchange_stop_order_id"])
        self.assertEqual(["stop-1"], state["cancelled"])
        self.assertNotIn("stop-1", state["stops"])
        self.assertIsNone(executor.get_custom_info()["config_edits"]["stop_loss"])

    def test_exit_decimals_round_trip(self):
        executor = self._executor(
            take_profit={
                "percent": "0.10",
                "execution": "ICEBERG",
                "slices": 2,
                "visible_count": 1,
                "interval": 30.0,
                "limit_price": "0.120",
            },
        )
        view = executor.get_custom_info()["take_profit"]
        self.assertEqual("0.1", view["percent"])
        self.assertEqual("30", view["interval"])
        self.assertEqual("0.12", view["limit_price"])
        again = self._executor(take_profit=view)
        self.assertEqual(view, again.get_custom_info()["take_profit"])

    def test_quote_base_and_contracts_resume_ignores_reported_totals_unless_edited(self):
        state = self._install_mexc_stop_book()
        quote = self._executor(
            size_unit=SizeUnit.QUOTE,
            total_amount=Decimal("8.5"),
            exit_arming="ON_ANY_FILL",
            take_profit={"percent": "0.10", "execution": "SINGLE", "order_type": "LIMIT"},
            stop_loss={"price": "0.07", "execution": "SINGLE"},
        )
        self._run(quote.control_task())
        order = self.orders[self.buy_ids[-1][0]]
        order.executed_amount_base = Decimal("50")
        order.executed_amount_quote = Decimal("50") * order.price
        order.current_state = OrderState.PARTIALLY_FILLED
        self._run(quote.control_task())
        snapshot = quote.get_custom_info()
        self.assertEqual("8.5", snapshot["total_amount"])
        self.assertEqual({}, snapshot["config_edits"])
        self.assertEqual("stop-1", snapshot["exchange_stop_order_id"])
        resumed = self._executor(
            size_unit=SizeUnit.BASE,
            total_amount=Decimal("50"),
            exit_arming="ON_ANY_FILL",
            take_profit={"percent": "0.10", "execution": "SINGLE", "order_type": "LIMIT"},
            stop_loss={"price": "0.07", "execution": "SINGLE"},
        )
        payload = dict(snapshot)
        payload["resume_entries"] = False
        payload["size_unit"] = "BASE"
        payload.pop("open_base")
        resumed.restore_from_custom_info(payload)
        self.assertEqual(SizeUnit.BASE, resumed.config.size_unit)
        self.assertEqual(Decimal("50"), resumed.config.total_amount)
        self.assertNotEqual(CloseType.FAILED, resumed.close_type)
        self.assertEqual(Decimal("50"), resumed.filled_base)
        self._run(resumed.control_task())
        self.assertNotEqual(CloseType.FAILED, resumed.close_type)
        self.assertNotEqual(RunnableStatus.TERMINATED, resumed.status)
        self.assertEqual("stop-1", resumed.get_custom_info()["exchange_stop_order_id"])
        self.assertNotIn("stop-1", state["cancelled"])

        quoted_edit = self._executor(size_unit=SizeUnit.QUOTE, total_amount=Decimal("8.5"))
        self.assertTrue(quoted_edit.update_config({"total_amount": "12"})["accepted"])
        self._run(quoted_edit.control_task())
        quote_info = quoted_edit.get_custom_info()
        self.assertEqual("12", quote_info["config_edits"]["total_amount"])
        self.assertEqual("QUOTE", quote_info["config_edits"]["size_unit"])
        same_quote = self._executor(size_unit=SizeUnit.QUOTE, total_amount=Decimal("8.5"))
        same_quote.restore_from_custom_info(quote_info)
        self.assertEqual(Decimal("12"), same_quote.config.total_amount)
        self.assertEqual(SizeUnit.QUOTE, same_quote.config.size_unit)
        self.assertNotEqual(CloseType.FAILED, same_quote.close_type)
        quote_as_base = self._executor(size_unit=SizeUnit.BASE, total_amount=Decimal("50"))
        quote_as_base.restore_from_custom_info(quote_info)
        self.assertEqual(Decimal("150"), quote_as_base.config.total_amount)
        self.assertEqual(SizeUnit.BASE, quote_as_base.config.size_unit)
        self.assertNotEqual(CloseType.FAILED, quote_as_base.close_type)

        base = self._executor(total_amount=Decimal("100"))
        bare = base.get_custom_info()
        bare["total_amount"] = "8.5"
        bare["size_unit"] = "BASE"
        untouched = self._executor(total_amount=Decimal("100"))
        untouched.restore_from_custom_info(bare)
        self.assertEqual(Decimal("100"), untouched.config.total_amount)
        self.assertEqual({}, untouched.get_custom_info()["config_edits"])
        self.assertTrue(base.update_config({"total_amount": "150"})["accepted"])
        self._run(base.control_task())
        base_info = base.get_custom_info()
        self.assertEqual("BASE", base_info["config_edits"]["size_unit"])
        base_resume = self._executor(total_amount=Decimal("100"))
        base_resume.restore_from_custom_info(base_info)
        self.assertEqual(Decimal("150"), base_resume.config.total_amount)
        self.assertNotEqual(CloseType.FAILED, base_resume.close_type)

        contracts = self._executor(size_unit=SizeUnit.CONTRACTS, total_amount=Decimal("10"))
        contract_bare = contracts.get_custom_info()
        self.assertEqual("10", contract_bare["total_amount"])
        contract_resume = self._executor(size_unit=SizeUnit.BASE, total_amount=Decimal("50"))
        contract_resume.restore_from_custom_info(contract_bare)
        self.assertEqual(Decimal("50"), contract_resume.config.total_amount)
        self.assertNotEqual(CloseType.FAILED, contract_resume.close_type)
        self.assertTrue(contracts.update_config({"total_amount": "8"})["accepted"])
        self._run(contracts.control_task())
        contract_info = contracts.get_custom_info()
        self.assertEqual("8", contract_info["config_edits"]["total_amount"])
        self.assertEqual("CONTRACTS", contract_info["config_edits"]["size_unit"])
        contract_as_base = self._executor(size_unit=SizeUnit.BASE, total_amount=Decimal("50"))
        contract_as_base.restore_from_custom_info(contract_info)
        self.assertEqual(Decimal("80"), contract_as_base.config.total_amount)
        self.assertEqual(SizeUnit.BASE, contract_as_base.config.size_unit)
        self.assertNotEqual(CloseType.FAILED, contract_as_base.close_type)
        same_contracts = self._executor(size_unit=SizeUnit.CONTRACTS, total_amount=Decimal("10"))
        same_contracts.restore_from_custom_info(contract_info)
        self.assertEqual(Decimal("8"), same_contracts.config.total_amount)
        self.assertEqual(SizeUnit.CONTRACTS, same_contracts.config.size_unit)
        self.assertNotEqual(CloseType.FAILED, same_contracts.close_type)

    def _place_four_rungs(self, executor: RangeIcebergExecutor):
        for _ in range(4):
            self._run(executor.control_task())
        amounts = [self.buy_ids[index][1] for index in range(4)]
        self.assertEqual(
            [Decimal("20"), Decimal("20"), Decimal("20"), Decimal("40")],
            amounts,
        )
        return [self.buy_ids[index][0] for index in range(4)]

    def test_d11c_shrink_cancels_slices_beyond_the_new_ladder(self):
        executor = self._executor(levels=4, total_amount=Decimal("100"))
        order_ids = self._place_four_rungs(executor)
        result = executor.update_config({"total_amount": "30"})
        self.assertTrue(result["accepted"], result)
        self._run(executor.control_task())
        self.assertEqual(Decimal("30"), executor.config.total_amount)
        self.assertLessEqual(
            sum(slot.target_amount for slot in executor._slices),
            Decimal("30"),
        )
        kept = [
            slot.order_id
            for slot in executor._slices
            if slot.order_id in order_ids
            and slot.is_working
            and not slot.cancel_requested
            and not getattr(slot, "edit_cancel", False)
        ]
        self.assertEqual([order_ids[0]], kept)
        for order_id in order_ids[1:]:
            slot = next(slot for slot in executor._slices if slot.order_id == order_id)
            self.assertTrue(slot.cancel_requested or getattr(slot, "edit_cancel", False))
            self.assertEqual(Decimal("0"), slot.target_amount)
        for _ in range(6):
            for order_id in order_ids[1:]:
                order = self.orders[order_id]
                if order.is_open:
                    order.current_state = OrderState.CANCELED
            self._run(executor.control_task())
        open_amount = sum(
            (
                order.amount - order.executed_amount_base
                for order in self.orders.values()
                if order.is_open
            ),
            Decimal("0"),
        )
        self.assertLessEqual(open_amount + executor.filled_base, Decimal("30"))
        for order_id in order_ids[1:]:
            self.assertFalse(self.orders[order_id].is_open)

    def test_d11d_shrink_aborts_when_remaining_targets_exceed_the_total(self):
        executor = self._executor(levels=4, total_amount=Decimal("100"))
        order_ids = self._place_four_rungs(executor)
        result = executor.update_config({"total_amount": "30"})
        self.assertTrue(result["accepted"], result)
        for order_id in order_ids[:3]:
            self._fill(order_id)
        cancels = self.strategy.cancel.call_count
        self._run(executor.control_task())
        self.assertEqual(Decimal("100"), executor.config.total_amount)
        self.assertIn("against the new total", executor.get_custom_info()["config_edit_error"])
        self.assertFalse(executor.get_custom_info()["pending_config_edit"])
        self.assertEqual(cancels, self.strategy.cancel.call_count)
        self.assertEqual(
            [Decimal("20"), Decimal("20"), Decimal("20"), Decimal("40")],
            [slot.target_amount for slot in executor._slices],
        )
        self.assertFalse(any(getattr(slot, "edit_cancel", False) for slot in executor._slices))

    def test_update_config_rejects_a_grow_when_entries_are_not_placed(self):
        executor = self._executor()
        executor.restore_from_custom_info({"resume_entries": False})
        self.assertFalse(executor._resume_entries)
        grown = executor.update_config({"total_amount": "150"})
        self.assertFalse(grown["accepted"])
        self.assertEqual(
            "cannot grow total_amount while entries are not being placed",
            grown["reason"],
        )
        self.assertEqual(Decimal("100"), executor.config.total_amount)
        self.assertIsNone(executor._queued_config)
        shrunk = executor.update_config({"total_amount": "40"})
        self.assertTrue(shrunk["accepted"], shrunk)
        self.assertEqual(Decimal("40"), executor._queued_config.total_amount)

    def test_custom_info_reports_size_unit_beside_total_amount(self):
        base = self._executor()
        base_info = base.get_custom_info()
        self.assertEqual("100", base_info["total_amount"])
        self.assertEqual("BASE", base_info["size_unit"])
        quote = self._executor(size_unit=SizeUnit.QUOTE, total_amount=Decimal("8.5"))
        quote_info = quote.get_custom_info()
        self.assertEqual("8.5", quote_info["total_amount"])
        self.assertEqual("QUOTE", quote_info["size_unit"])
        contracts = self._executor(size_unit=SizeUnit.CONTRACTS, total_amount=Decimal("10"))
        contract_info = contracts.get_custom_info()
        self.assertEqual("10", contract_info["total_amount"])
        self.assertEqual("CONTRACTS", contract_info["size_unit"])
