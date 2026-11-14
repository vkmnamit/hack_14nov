import unittest

import numpy as np

from backend.model_service import (
    CITY_MODELS,
    load_city_assets,
    make_rollout_data,
    rainfall_rate,
)


class ModelServiceInputTests(unittest.TestCase):
    def test_city_runtime_assets_load(self) -> None:
        for location_id, config in CITY_MODELS.items():
            with self.subTest(location_id=location_id):
                self.assertTrue(config.elevation_path.is_file())
                elevation, normalization = load_city_assets(location_id)
                self.assertEqual(elevation.shape, (50, 50))
                self.assertIn("state_mean", normalization)

    def test_builds_real_rollout_inputs_for_one_hour_forecast(self) -> None:
        elevation = np.arange(50 * 50, dtype=np.float32).reshape(50, 50)

        data = make_rollout_data(
            elevation,
            pattern="constant",
            rainfall_intensity_mm_hr=45.0,
            duration_min=15,
        )

        self.assertEqual(data["state_sequence"].shape, (1, 61, 3, 50, 50))
        self.assertEqual(data["raw_rainfall_mm_hr"].shape, (1, 60, 50, 50))
        self.assertEqual(int(np.count_nonzero(data["state_sequence"])), 0)
        self.assertTrue(np.all(data["raw_rainfall_mm_hr"][0, :15] == 45.0))
        self.assertTrue(np.all(data["raw_rainfall_mm_hr"][0, 15:] == 0.0))
        self.assertAlmostEqual(
            float(data["rainfall_effective_m_s"][0, 0, 0, 0]),
            40.0 / 3_600_000.0,
        )
        self.assertTrue(np.array_equal(data["elevation_m"], elevation))

    def test_rainfall_patterns_match_selected_duration(self) -> None:
        self.assertEqual(rainfall_rate("constant", 60.0, 10, 4), 60.0)
        self.assertEqual(rainfall_rate("constant", 60.0, 10, 10), 0.0)
        self.assertEqual(rainfall_rate("increasing", 60.0, 10, 5), 30.0)
        self.assertEqual(rainfall_rate("peak", 60.0, 10, 5), 60.0)
        self.assertEqual(rainfall_rate("peak", 60.0, 10, 10), 0.0)


if __name__ == "__main__":
    unittest.main()
