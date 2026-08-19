"""Tests for the frozen-model input-resolution preprocessing pipeline."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from evaluate_resolution_compatibility import (
    CONDITION_EFFECTIVE_SIZE,
    ResolutionCompatibilityTransform,
    build_mnist_resolution_transform,
)
from onn_model.data import MNIST_MEAN, MNIST_STD


@pytest.fixture
def ramp_tensor() -> torch.Tensor:
    return torch.arange(28 * 28, dtype=torch.float32).reshape(1, 28, 28) / (28 * 28 - 1)


@pytest.mark.parametrize("condition", CONDITION_EFFECTIVE_SIZE)
def test_output_shape_and_determinism(condition: str, ramp_tensor: torch.Tensor) -> None:
    transform = ResolutionCompatibilityTransform(condition)
    first = transform(ramp_tensor)
    second = transform(ramp_tensor)
    assert first.shape == (1, 28, 28)
    assert torch.equal(first, second)


def test_native_condition_preserves_tensor(ramp_tensor: torch.Tensor) -> None:
    output = ResolutionCompatibilityTransform("native_28")(ramp_tensor)
    assert torch.equal(output, ramp_tensor)
    assert output.data_ptr() != ramp_tensor.data_ptr()


@pytest.mark.parametrize("condition,size", [("down8_up28", 8), ("down4_up28", 4)])
def test_area_then_nearest_matches_torch_reference(
    condition: str, size: int, ramp_tensor: torch.Tensor
) -> None:
    expected = F.interpolate(
        F.interpolate(ramp_tensor.unsqueeze(0), size=(size, size), mode="area"),
        size=(28, 28),
        mode="nearest",
    ).squeeze(0)
    actual = ResolutionCompatibilityTransform(condition)(ramp_tensor)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("condition,size", [("down8_up28", 8), ("down4_up28", 4)])
def test_effective_pixel_degrees_of_freedom(
    condition: str, size: int, ramp_tensor: torch.Tensor
) -> None:
    output = ResolutionCompatibilityTransform(condition)(ramp_tensor)
    # A strictly increasing ramp gives each area-downsampled cell a distinct
    # value; nearest upsampling may repeat values but cannot create new ones.
    assert torch.unique(output).numel() == size * size


def test_pipeline_order_and_normalized_value() -> None:
    pixels = np.arange(28 * 28, dtype=np.uint8).reshape(28, 28)
    image = Image.fromarray(pixels, mode="L")
    pipeline = build_mnist_resolution_transform("down8_up28")

    assert isinstance(pipeline.transforms[0], transforms.ToTensor)
    assert isinstance(pipeline.transforms[1], ResolutionCompatibilityTransform)
    assert isinstance(pipeline.transforms[2], transforms.Normalize)

    tensor_01 = transforms.ToTensor()(image)
    resized = ResolutionCompatibilityTransform("down8_up28")(tensor_01)
    expected = (resized - MNIST_MEAN) / MNIST_STD
    assert torch.allclose(pipeline(image), expected)


def test_unknown_condition_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown condition"):
        ResolutionCompatibilityTransform("down2_up28")


@pytest.mark.parametrize(
    "bad_input,expected_exception",
    [
        (torch.zeros(28, 28), ValueError),
        (torch.zeros(1, 27, 28), ValueError),
        (torch.zeros(1, 28, 28, dtype=torch.uint8), TypeError),
    ],
)
def test_invalid_tensor_is_rejected(
    bad_input: torch.Tensor, expected_exception: type[Exception]
) -> None:
    with pytest.raises(expected_exception):
        ResolutionCompatibilityTransform("down4_up28")(bad_input)
