from decimal import Decimal
from test.isolated_asyncio_wrapper_test_case import IsolatedAsyncioWrapperTestCase
from unittest.mock import MagicMock, PropertyMock

from hummingbot.connector.trading_rule import TradingRule
from hummingbot.core.data_type.common import OrderType, PositionAction, PriceType, TradeType
from hummingbot.core.data_type.in_flight_order import InFlightOrder, OrderState
from hummingbot.strategy_v2.executors.range_iceberg_executor.data_types import (
    RangeIcebergExecutorConfig,
    SizeUnit,
)
from hummingbot.strategy_v2.executors.range_iceberg_executor.range_iceberg_executor import RangeIcebergExecutor
from hummingbot.strategy_v2.models.base import RunnableStatus
from hummingbot.strategy_v2.models.executors import CloseType


class RangeIcebergExecutorTests(IsolatedAsyncioWrapperTestCase):
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

        def buy(connector_name, trading_pair, amount, order_type, price, position_action):
            order_id = f"buy-{len(self.buy_ids) + 1}"
            self.buy_ids.append((order_id, amount, price, order_type, position_action))
            self.orders[order_id] = InFlightOrder(
                client_order_id=order_id,
                trading_pair=trading_pair,
                order_type=order_type,
                trade_type=TradeType.BUY,
                amount=amount,
                price=price,
                initial_state=OrderState.OPEN,
                creation_timestamp=self.now,
            )
            return order_id

        self.strategy.buy.side_effect = buy
        self.strategy.sell.side_effect = buy
        self.strategy.cancel.return_value = None

    def _config(self, **overrides) -> RangeIcebergExecutorConfig:
        values = dict(
            timestamp=self.now,
            connector_name="mexc_perpetual",
            trading_pair="AIOT-USDT",
            side=TradeType.BUY,
            total_amount=Decimal("10000"),
            size_unit=SizeUnit.QUOTE,
            price_low=Decimal("0.08"),
            price_high=Decimal("0.09"),
            levels=3,
            iceberg=True,
            visible_count=1,
            min_order_interval=0,
            random_seed=1,
        )
        values.update(overrides)
        return RangeIcebergExecutorConfig(**values)

    def _executor(self, **overrides) -> RangeIcebergExecutor:
        executor = RangeIcebergExecutor(self.strategy, self._config(**overrides), update_interval=0.1)
        executor._status = RunnableStatus.RUNNING
        return executor

    def _run(self, coro):
        return self.run_async_with_timeout(coro, 1)

    def test_iceberg_buy_shows_one_slice_then_replaces_it_inside_the_range(self):
        executor = self._executor()
        self._run(executor.control_task())
        self.assertEqual(1, len(self.buy_ids))
        order_id, amount, price, order_type, position_action = self.buy_ids[0]
        self.assertEqual(OrderType.LIMIT, order_type)
        self.assertEqual(PositionAction.OPEN, position_action)
        self.assertGreaterEqual(price, Decimal("0.08"))
        self.assertLessEqual(price, Decimal("0.09"))
        self.assertEqual(0, amount % Decimal("10"))

        self._run(executor.control_task())
        self.assertEqual(1, len(self.buy_ids))

        self.orders[order_id].current_state = OrderState.FILLED
        self._run(executor.control_task())
        self.assertEqual(2, len(self.buy_ids))
        self.assertGreater(self.buy_ids[1][2], self.buy_ids[0][2])
        info = executor.get_custom_info()
        self.assertEqual(["buy-1", "buy-2"], info["child_order_ids"])
        self.assertEqual("BUY", info["side"])

    def test_pause_and_resume(self):
        executor = self._executor(iceberg=False, visible_count=3)
        executor.pause()
        self._run(executor.control_task())
        self.assertEqual(0, self.strategy.buy.call_count)
        self.assertTrue(executor.get_custom_info()["paused"])
        executor.resume()
        self._run(executor.control_task())
        self.assertEqual(1, self.strategy.buy.call_count)

    def test_cancel_parent_cancels_children(self):
        executor = self._executor()
        self._run(executor.control_task())
        executor.early_stop()
        self.strategy.cancel.assert_called_once_with("mexc_perpetual", "AIOT-USDT", "buy-1")
        self.assertEqual(CloseType.EARLY_STOP, executor.close_type)
        self.assertEqual(RunnableStatus.TERMINATED, executor.status)

    def test_expiry_cancels_the_working_slice(self):
        executor = self._executor(expiry_seconds=5)
        self._run(executor.control_task())
        self.now = 1_006
        self._run(executor.control_task())
        self.strategy.cancel.assert_called_once()
        self.assertEqual(CloseType.EXPIRED, executor.close_type)

    def test_slippage_guard_does_not_chase_above_the_range(self):
        self.connector.get_price_by_type.side_effect = lambda pair, price_type: (
            Decimal("0.2") if price_type is PriceType.BestAsk else Decimal("0.085")
        )
        executor = self._executor(max_slippage=Decimal("0"))
        self._run(executor.control_task())
        self.assertEqual(0, self.strategy.buy.call_count)

        self.connector.get_price_by_type.side_effect = lambda pair, price_type: Decimal("0.089")
        self._run(executor.control_task())
        self.assertEqual(1, self.strategy.buy.call_count)

    def test_reduce_only_sell_uses_close(self):
        executor = self._executor(side="SELL", reduce_only=True, size_unit=SizeUnit.BASE, total_amount=Decimal("30"))
        self._run(executor.control_task())
        self.assertEqual(PositionAction.CLOSE, self.strategy.sell.call_args.args[5])

    def test_order_spacing_respects_the_place_interval(self):
        executor = self._executor(iceberg=False, visible_count=3, min_order_interval=2)
        self._run(executor.control_task())
        self._run(executor.control_task())
        self.assertEqual(1, self.strategy.buy.call_count)
        self.now += 2
        self._run(executor.control_task())
        self.assertEqual(2, self.strategy.buy.call_count)

    def test_post_only_uses_limit_maker(self):
        executor = self._executor(post_only=True)
        self._run(executor.control_task())
        self.assertEqual(OrderType.LIMIT_MAKER, self.buy_ids[0][3])

    def test_average_fill_uses_child_fills(self):
        executor = self._executor(levels=1, iceberg=False, size_unit=SizeUnit.BASE, total_amount=Decimal("20"))
        self._run(executor.control_task())
        order = self.orders["buy-1"]
        order.executed_amount_base = Decimal("20")
        order.executed_amount_quote = Decimal("1.6")
        order.current_state = OrderState.FILLED
        self._run(executor.control_task())
        self.assertEqual(Decimal("0.08"), executor.average_fill_price)
        self.assertEqual(Decimal("20"), executor.filled_base)
        self.assertEqual(CloseType.COMPLETED, executor.close_type)
