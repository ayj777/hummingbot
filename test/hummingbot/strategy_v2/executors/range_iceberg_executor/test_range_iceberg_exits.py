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
