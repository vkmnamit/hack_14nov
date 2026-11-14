import unittest

from backend.live_anuga import _rainfall_rate


class LiveAnugaRainfallTests(unittest.TestCase):
    def test_rainfall_stops_at_duration_for_every_pattern(self) -> None:
        for pattern in ("constant", "increasing", "peak"):
            with self.subTest(pattern=pattern):
                self.assertEqual(_rainfall_rate(pattern, 60.0, 600.0, 600.0), 0.0)
                self.assertEqual(_rainfall_rate(pattern, 60.0, 600.0, 900.0), 0.0)

    def test_increasing_rainfall_starts_at_zero(self) -> None:
        self.assertEqual(_rainfall_rate("increasing", 60.0, 600.0, 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
