import unittest

import torch

from backend.shallow_water import needs_physics_fallback


class AdaptivePhysicsFallbackTests(unittest.TestCase):
    def test_accepts_prediction_with_expected_rainfall_volume(self) -> None:
        current = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
        current[:, 0] = 0.1
        rainfall = torch.full((1, 1, 2, 2), 1e-4)
        predicted = current.clone()
        predicted[:, 0] = 0.106

        self.assertFalse(
            needs_physics_fallback(
                predicted,
                current,
                rainfall,
                threshold_pct=1.0,
            )
        )

    def test_triggers_when_prediction_breaks_water_volume_budget(self) -> None:
        current = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
        current[:, 0] = 0.1
        rainfall = torch.full((1, 1, 2, 2), 1e-4)
        predicted = current.clone()
        predicted[:, 0] = 0.2

        self.assertTrue(
            needs_physics_fallback(
                predicted,
                current,
                rainfall,
                threshold_pct=1.0,
            )
        )

    def test_triggers_when_prediction_is_non_finite(self) -> None:
        current = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
        predicted = current.clone()
        predicted[:, 0, 0, 0] = float("nan")

        self.assertTrue(
            needs_physics_fallback(
                predicted,
                current,
                torch.zeros((1, 1, 2, 2)),
                threshold_pct=1.0,
            )
        )


if __name__ == "__main__":
    unittest.main()
