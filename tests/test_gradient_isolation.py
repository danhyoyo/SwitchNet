from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from switchnet.config import ModelConfig
from switchnet.models import MultiExpertModel, build_model


def flat_private_gradient(model: MultiExpertModel, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x, y).logits, y)
    loss.backward()
    pieces = []
    for parameter in model.parameter_spaces()["private"]:
        pieces.append(
            torch.zeros_like(parameter).flatten()
            if parameter.grad is None
            else parameter.grad.detach().flatten().clone()
        )
    return torch.cat(pieces)


class GradientIsolationTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        cfg = ModelConfig(feature_dim=12, encoder_width=20, expert_bottleneck=4, dropout=0.0)
        self.model = build_model("switchnet_v1", cfg, (6,), 3)
        assert isinstance(self.model, MultiExpertModel)
        self.model.train()

    def test_non_target_private_gradients_are_zero(self) -> None:
        x = torch.randn(8, 6)
        y = torch.zeros(8, dtype=torch.long)
        F.cross_entropy(self.model(x, y).logits, y).backward()
        target_private = [
            *self.model.experts[0].adapter.parameters(),
            *self.model.experts[0].norm.parameters(),
        ]
        target_norm = sum(float(p.grad.abs().sum()) for p in target_private if p.grad is not None)
        self.assertGreater(target_norm, 0.0)
        for expert in self.model.experts[1:]:
            for parameter in (*expert.adapter.parameters(), *expert.norm.parameters()):
                self.assertTrue(parameter.grad is None or torch.count_nonzero(parameter.grad) == 0)

    def test_non_target_evidence_receives_negative_gradient(self) -> None:
        x = torch.randn(8, 6)
        y = torch.zeros(8, dtype=torch.long)
        F.cross_entropy(self.model(x, y).logits, y).backward()
        self.assertLess(float(self.model.experts[0].evidence.bias.grad), 0.0)
        for expert in self.model.experts[1:]:
            self.assertGreater(float(expert.evidence.bias.grad), 0.0)

    def test_cross_class_private_gradient_inner_product_is_zero(self) -> None:
        x0, y0 = torch.randn(5, 6), torch.zeros(5, dtype=torch.long)
        x1, y1 = torch.randn(5, 6), torch.ones(5, dtype=torch.long)
        grad0 = flat_private_gradient(self.model, x0, y0)
        grad1 = flat_private_gradient(self.model, x1, y1)
        self.assertGreater(float(grad0.norm()), 0.0)
        self.assertGreater(float(grad1.norm()), 0.0)
        self.assertEqual(float(torch.dot(grad0, grad1)), 0.0)

    def test_all_experts_run_and_labels_do_not_change_forward_values(self) -> None:
        calls = [0, 0, 0]
        handles = []
        for index, expert in enumerate(self.model.experts):
            handles.append(
                expert.register_forward_hook(
                    lambda _module, _inputs, _output, idx=index: calls.__setitem__(idx, calls[idx] + 1)
                )
            )
        x = torch.randn(4, 6)
        y = torch.tensor([0, 1, 2, 0])
        with torch.no_grad():
            controlled = self.model(x, y).logits
            uncontrolled = self.model(x, None).logits
        for handle in handles:
            handle.remove()
        self.assertEqual(calls, [2, 2, 2])
        torch.testing.assert_close(controlled, uncontrolled, rtol=0, atol=0)

    def test_label_values_do_not_change_dropout_forward(self) -> None:
        cfg = ModelConfig(feature_dim=12, encoder_width=20, expert_bottleneck=4, dropout=0.3)
        model = build_model("switchnet_v1", cfg, (6,), 3)
        model.train()
        x = torch.randn(6, 6)
        first_labels = torch.tensor([0, 1, 2, 0, 1, 2])
        second_labels = first_labels.roll(1)
        torch.manual_seed(123)
        first = model(x, first_labels).logits
        torch.manual_seed(123)
        second = model(x, second_labels).logits
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_multi_branch_has_non_target_gradients(self) -> None:
        cfg = ModelConfig(feature_dim=12, encoder_width=20, expert_bottleneck=4, dropout=0.0)
        model = build_model("multi_branch", cfg, (6,), 3)
        assert isinstance(model, MultiExpertModel)
        x = torch.randn(8, 6)
        y = torch.zeros(8, dtype=torch.long)
        F.cross_entropy(model(x, y).logits, y).backward()
        non_target_norm = sum(
            float(parameter.grad.abs().sum())
            for parameter in model.experts[1].parameters()
            if parameter.grad is not None
        )
        self.assertGreater(non_target_norm, 0.0)

    def test_isolation_preserves_shared_encoder_gradient(self) -> None:
        cfg = ModelConfig(feature_dim=12, encoder_width=20, expert_bottleneck=4, dropout=0.0)
        control = build_model("multi_branch", cfg, (6,), 3)
        isolated = build_model("switchnet_v1", cfg, (6,), 3)
        isolated.load_state_dict(control.state_dict())
        x = torch.randn(8, 6)
        y = torch.arange(8) % 3
        shared_gradients = []
        for model in (control, isolated):
            model.zero_grad(set_to_none=True)
            F.cross_entropy(model(x, y).logits, y).backward()
            shared_gradients.append(
                torch.cat([parameter.grad.flatten() for parameter in model.encoder.parameters()])
            )
        torch.testing.assert_close(shared_gradients[0], shared_gradients[1], rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
