import unittest

import torch

from backend.shallow_water import (
    conserve_depth_volume,
    is_dry_without_rain,
    shallow_water_step,
)


class ShallowWaterStepTests(unittest.TestCase):
    def test_preserves_lake_at_rest_over_sloped_bed(self) -> None:
        size = 8
        rows, columns = torch.meshgrid(
            torch.arange(size, dtype=torch.float32),
            torch.arange(size, dtype=torch.float32),
            indexing="ij",
        )
        elevation = 0.02 * columns + 0.01 * rows
        depth = 2.0 - elevation
        state = torch.zeros((1, 3, size, size), dtype=torch.float32)
        state[:, 0] = depth

        result = shallow_water_step(
            state,
            torch.zeros((size, size)),
            elevation,
            torch.full((size, size), 0.03),
        )

        self.assertTrue(torch.allclose(result[:, 0], state[:, 0], atol=1e-5))
        self.assertLess(float(result[:, 1:].abs().max()), 1e-5)

    def test_conserves_water_with_reflective_boundaries(self) -> None:
        size = 8
        rows, columns = torch.meshgrid(
            torch.arange(size, dtype=torch.float32),
            torch.arange(size, dtype=torch.float32),
            indexing="ij",
        )
        depth = 0.1 + 0.2 * torch.exp(-((columns - 3.5) ** 2 + (rows - 3.5) ** 2))
        state = torch.zeros((1, 3, size, size), dtype=torch.float32)
        state[:, 0] = depth

        result = shallow_water_step(
            state,
            torch.zeros((size, size)),
            torch.zeros((size, size)),
            torch.full((size, size), 0.03),
        )

        self.assertAlmostEqual(
            float(result[:, 0].sum()),
            float(state[:, 0].sum()),
            places=5,
        )

    def test_adds_effective_rainfall_to_depth(self) -> None:
        size = 8
        state = torch.zeros((1, 3, size, size), dtype=torch.float32)
        state[:, 0] = 0.2
        rainfall_rate = torch.full((size, size), 1e-5)

        result = shallow_water_step(
            state,
            rainfall_rate,
            torch.zeros((size, size)),
            torch.full((size, size), 0.03),
        )

        self.assertTrue(
            torch.allclose(
                result[:, 0],
                state[:, 0] + 60.0 * rainfall_rate,
                atol=1e-6,
            )
        )
        self.assertLess(float(result[:, 1:].abs().max()), 1e-6)

    def test_depth_correction_preserves_reference_volume(self) -> None:
        reference = torch.full((1, 1, 4, 4), 0.2)
        predicted = reference.clone()
        predicted[:, :, 0, 0] = -0.1
        predicted[:, :, 1, 1] = 0.7

        result = conserve_depth_volume(predicted, reference)

        self.assertGreaterEqual(float(result.min()), 0.0)
        self.assertAlmostEqual(
            float(result.sum()),
            float(reference.sum()),
            places=5,
        )
        self.assertTrue(
            torch.equal(
                conserve_depth_volume(torch.zeros_like(reference), reference),
                reference,
            )
        )

    def test_detects_dry_state_without_effective_rainfall(self) -> None:
        state = torch.zeros((1, 3, 4, 4), dtype=torch.float32)
        rainfall = torch.zeros((1, 1, 4, 4), dtype=torch.float32)

        self.assertTrue(is_dry_without_rain(state, rainfall))

        rainfall[:, :, 0, 0] = 1e-5
        self.assertFalse(is_dry_without_rain(state, rainfall))

        rainfall.zero_()
        state[:, 0, 0, 0] = 0.01
        self.assertFalse(is_dry_without_rain(state, rainfall))


if __name__ == "__main__":
    unittest.main()
