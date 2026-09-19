"""Check the candidate budget and that all layers can learn."""

import pytest
import torch
from torch import nn

from .model import MatchedConvSmall


def test_budget_forward_and_backward():
    torch.manual_seed(7)
    model = MatchedConvSmall()
    logits = model(torch.rand(4, 1, 8, 8))
    assert logits.shape == (4, 10)
    assert model.parameter_count() == 9930
    assert model.dense_equivalent_macs_per_sample() == 88064
    nn.CrossEntropyLoss()(logits, torch.tensor([0, 1, 2, 3])).backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_rejects_wrong_input_size():
    with pytest.raises(ValueError, match="shape"):
        MatchedConvSmall()(torch.zeros(2, 1, 28, 28))
