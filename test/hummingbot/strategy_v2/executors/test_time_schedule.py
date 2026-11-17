from decimal import Decimal
from unittest import TestCase

from hummingbot.strategy_v2.executors.time_schedule import SliceSchedule, parse_duration, release_timestamps


class TimeScheduleTests(TestCase):
    def test_duration_strings_and_bare_seconds(self):
        self.assertEqual(86400.0, parse_duration("24h"))
        self.assertEqual(3600.0, parse_duration("1H"))
        self.assertEqual(90.0, parse_duration("90s"))
        self.assertEqual(120.0, parse_duration("2m"))
        self.assertEqual(86400.0, parse_duration(86400))

    def test_release_timestamps_match_the_twap_clock(self):
        self.assertEqual([1.0, 6.0, 11.0], release_timestamps(1, 3, 5))

    def test_skip_drops_only_the_unplaced_part_of_a_blocked_window(self):
        schedule = SliceSchedule(
            amounts=[Decimal("2500"), Decimal("2500")],
            offsets=[0.0, 25.0],
            duration=50.0,
            skip_unplaced=True,
        )
        schedule.advance(0, Decimal("0"), Decimal("0"), outside_cap=True)
        self.assertEqual(Decimal("0"), schedule.skipped)
        self.assertEqual(Decimal("2500"), schedule.opened(0))
        schedule.advance(25, Decimal("0"), Decimal("0"), outside_cap=True)
        self.assertEqual(Decimal("2500"), schedule.skipped)
        self.assertEqual(Decimal("5000"), schedule.opened(25))

    def test_carry_keeps_a_blocked_window_available(self):
        schedule = SliceSchedule(
            amounts=[Decimal("2500"), Decimal("2500")],
            offsets=[0.0, 25.0],
            duration=50.0,
            skip_unplaced=False,
        )
        schedule.advance(25, Decimal("0"), Decimal("0"), outside_cap=True)
        self.assertEqual(Decimal("0"), schedule.skipped)
        self.assertEqual(Decimal("5000"), schedule.opened(25))

    def test_skip_drops_later_blocked_windows_while_one_slice_is_resting(self):
        count = 24
        total = Decimal("10000")
        share = total / Decimal(count)
        schedule = SliceSchedule(
            amounts=[share] * count,
            offsets=[index * 3600.0 for index in range(count)],
            duration=86400.0,
            skip_unplaced=True,
        )
        schedule.advance(23 * 3600.0, Decimal("0"), schedule.amounts[0], outside_cap=True)
        self.assertEqual(sum(schedule.amounts[1:23]), schedule.skipped)
        self.assertEqual(23, schedule._closed)

    def test_skip_with_nothing_resting_drops_every_closed_window(self):
        count = 24
        total = Decimal("10000")
        share = total / Decimal(count)
        schedule = SliceSchedule(
            amounts=[share] * count,
            offsets=[index * 3600.0 for index in range(count)],
            duration=86400.0,
            skip_unplaced=True,
        )
        schedule.advance(23 * 3600.0, Decimal("0"), Decimal("0"), outside_cap=True)
        self.assertEqual(sum(schedule.amounts[:23]), schedule.skipped)

    def test_one_outside_tick_does_not_skip_the_window_after_price_returns(self):
        schedule = SliceSchedule(
            amounts=[Decimal("100")] * 4,
            offsets=[0.0, 10.0, 20.0, 30.0],
            duration=40.0,
            skip_unplaced=True,
        )
        schedule.advance(0, Decimal("0"), Decimal("0"), outside_cap=True)
        schedule.advance(40, Decimal("0"), Decimal("0"), outside_cap=False)
        self.assertEqual(Decimal("0"), schedule.skipped)

    def test_skip_does_not_keep_a_window_blocked_after_price_returns(self):
        schedule = SliceSchedule(
            amounts=[Decimal("2500"), Decimal("2500")],
            offsets=[0.0, 25.0],
            duration=50.0,
            skip_unplaced=True,
        )
        schedule.advance(0, Decimal("0"), Decimal("0"), outside_cap=True)
        schedule.advance(25, Decimal("0"), Decimal("0"), outside_cap=False)
        self.assertEqual(Decimal("0"), schedule.skipped)

    def test_closing_several_windows_at_once_skips_only_the_unplaced_part(self):
        schedule = SliceSchedule(
            amounts=[Decimal("100")] * 4,
            offsets=[0.0, 10.0, 20.0, 30.0],
            duration=40.0,
            skip_unplaced=True,
        )
        schedule.advance(40, Decimal("0"), Decimal("100"), outside_cap=True)
        self.assertEqual(Decimal("300"), schedule.skipped)

    def test_carry_keeps_a_delayed_fill_reserved_until_it_is_reported(self):
        schedule = SliceSchedule(
            amounts=[Decimal("100"), Decimal("100")],
            offsets=[0.0, 10.0],
            duration=20.0,
            skip_unplaced=False,
        )
        schedule.advance(10, Decimal("0"), Decimal("100"), outside_cap=False)
        self.assertEqual(Decimal("0"), schedule.skipped)
        placeable = schedule.opened(10) - schedule.skipped - Decimal("0") - Decimal("100")
        self.assertEqual(Decimal("100"), placeable)
        schedule.advance(10, Decimal("40"), Decimal("60"), outside_cap=False)
        placeable = schedule.opened(10) - schedule.skipped - Decimal("40") - Decimal("60")
        self.assertEqual(Decimal("100"), placeable)
        self.assertEqual(Decimal("0"), schedule.skipped)

    def test_window_accounting_round_trips(self):
        schedule = SliceSchedule(
            amounts=[Decimal("100"), Decimal("100")],
            offsets=[0.0, 10.0],
            duration=20.0,
            skip_unplaced=True,
        )
        schedule.advance(10, Decimal("0"), Decimal("0"), outside_cap=True)
        restored = SliceSchedule(
            amounts=[Decimal("100"), Decimal("100")],
            offsets=[0.0, 10.0],
            duration=20.0,
            skip_unplaced=True,
        )
        restored.restore_state(schedule.export_state())
        self.assertEqual(schedule.skipped, restored.skipped)
        self.assertEqual(schedule._closed, restored._closed)
        self.assertEqual(schedule._assigned, restored._assigned)
