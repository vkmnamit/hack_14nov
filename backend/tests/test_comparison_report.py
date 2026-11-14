from types import SimpleNamespace
import unittest

from backend.comparison_report import compare_final_frames, verdict_facts


class ComparisonMetricsTests(unittest.TestCase):
    def test_compares_depth_wet_area_and_wet_cell_velocity(self) -> None:
        reference = SimpleNamespace(
            depths=[0.02, 0.0, 0.02, 0.0],
            velocityX=[1.0, 0.0, 0.0, 0.0],
            velocityY=[0.0, 0.0, 0.0, 0.0],
        )
        prediction = SimpleNamespace(
            depths=[0.02, 0.02, 0.0, 0.0],
            velocityX=[0.5, 0.0, 0.0, 0.0],
            velocityY=[0.0, 0.0, 0.0, 0.0],
        )

        metrics = compare_final_frames(reference, prediction)

        self.assertAlmostEqual(metrics["depth_mae_m"], 0.01)
        self.assertAlmostEqual(metrics["depth_rmse_m"], (0.0002) ** 0.5)
        self.assertAlmostEqual(metrics["wet_area_csi_1cm"], 1.0 / 3.0)
        self.assertAlmostEqual(
            metrics["velocity_vector_mae_m_s_on_reference_wet_cells"],
            0.25,
        )
        self.assertEqual(metrics["water_volume_relative_error_pct"], 0.0)

    def test_csi_is_perfect_when_both_reference_and_prediction_are_dry(self) -> None:
        dry_frame = SimpleNamespace(
            depths=[0.0, 0.0],
            velocityX=[0.0, 0.0],
            velocityY=[0.0, 0.0],
        )
        metrics = compare_final_frames(dry_frame, dry_frame)
        self.assertEqual(metrics["wet_area_csi_1cm"], 1.0)
        self.assertIsNone(
            metrics["velocity_vector_mae_m_s_on_reference_wet_cells"]
        )
        self.assertIsNone(metrics["water_volume_relative_error_pct"])

    def test_verdict_facts_select_only_neural_agreement_and_fastest_runtime(self) -> None:
        comparison = {
            "metrics": {
                "physics": {"runtime_seconds": 9.0},
                "convlstm": {
                    "runtime_seconds": 0.4,
                    "depth_mae_m": 0.03,
                    "wet_area_csi_1cm": 0.7,
                },
                "adaptive": {
                    "runtime_seconds": 0.8,
                    "depth_mae_m": 0.02,
                    "wet_area_csi_1cm": 0.6,
                },
            }
        }

        facts = verdict_facts(comparison)

        self.assertEqual(
            facts["best_neural_depth_agreement"]["model"],
            "adaptive",
        )
        self.assertEqual(
            facts["best_neural_wet_area_agreement"]["model"],
            "convlstm",
        )
        self.assertEqual(facts["fastest_runtime_model"]["model"], "convlstm")


if __name__ == "__main__":
    unittest.main()
