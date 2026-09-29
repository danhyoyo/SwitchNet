from __future__ import annotations

import unittest

import torch

from switchnet.config import DataConfig, ExperimentConfig, ModelConfig, apply_overrides
from switchnet.data import build_dataloaders
from switchnet.models import MODEL_NAMES, build_model
from switchnet.utils import count_parameters


class SmokeTests(unittest.TestCase):
    def test_all_models_forward_binary_and_multiclass(self) -> None:
        cfg = ModelConfig(feature_dim=16, encoder_width=24, expert_bottleneck=4, dropout=0.0)
        for classes in (2, 5):
            x = torch.randn(7, 8)
            y = torch.arange(7) % classes
            for name in MODEL_NAMES:
                model = build_model(name, cfg, (8,), classes)
                output = model(x, y)
                self.assertEqual(output.logits.shape, (7, classes))
                self.assertTrue(torch.isfinite(output.logits).all())

    def test_dense_is_parameter_matched(self) -> None:
        cfg = ModelConfig(feature_dim=32, encoder_width=48, expert_bottleneck=8, dropout=0.0)
        dense = build_model("dense", cfg, (10,), 10)
        switch = build_model("switchnet_v1", cfg, (10,), 10)
        relative_error = abs(count_parameters(dense) - count_parameters(switch)) / count_parameters(switch)
        self.assertLess(relative_error, 0.03)

    def test_gate_is_parameter_matched(self) -> None:
        cfg = ModelConfig(feature_dim=32, encoder_width=48, expert_bottleneck=8, dropout=0.0)
        gate = build_model("switchnet_gate", cfg, (10,), 10)
        switch = build_model("switchnet_v1", cfg, (10,), 10)
        relative_error = abs(count_parameters(gate) - count_parameters(switch)) / count_parameters(switch)
        self.assertLess(relative_error, cfg.parameter_match_tolerance)

    def test_synthetic_long_tail_counts(self) -> None:
        cfg = DataConfig(
            name="synthetic",
            num_classes=4,
            input_dim=5,
            train_size=200,
            val_size=40,
            test_size=40,
            imbalance_ratio=20.0,
            batch_size=16,
        )
        bundle = build_dataloaders(cfg, seed=3)
        self.assertEqual(sum(bundle.class_counts), 200)
        self.assertGreater(bundle.class_counts[0], bundle.class_counts[-1])
        self.assertEqual(bundle.input_shape, (5,))

    def test_invalid_override_fails_early(self) -> None:
        with self.assertRaises(ValueError):
            apply_overrides(ExperimentConfig(), ["model.dropout=1.0"])


if __name__ == "__main__":
    unittest.main()
