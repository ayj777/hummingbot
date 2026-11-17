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
