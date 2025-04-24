# %% SETUP
import math
import os
import sys
from pathlib import Path

import einops
import numpy as np
import torch as t
from torch import Tensor

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part0_prereqs"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part0_prereqs.tests as tests
from part0_prereqs.utils import display_array_as_img, display_soln_array_as_img

MAIN = __name__ == "__main__"


# %% Exercises - einops operations (match images)
arr = np.load(section_dir / "numbers.npy")
arr


# %% (1) Column-stacking
arr1 = einops.rearrange(arr, "b c h w -> c (b h) w")
display_array_as_img(arr1)

# %% (2) Column-stacking and copying
arr2 = einops.repeat(arr[0], "c h w -> c (2 h) w")
display_array_as_img(arr2)

# %% (3) Row-stacking and double-copying
arr3 = einops.repeat(arr[:2], "b c h w -> c (b h) (2 w)")
display_array_as_img(arr3)


# %% (4) Stretching
arr4 = einops.repeat(arr[0], "c h w -> c (h 2) w")
display_array_as_img(arr4)

# %% (5) Split channels
arr5 = einops.rearrange(arr[0], "c h w -> h (c w)")
display_array_as_img(arr5)

# %% (6) Stack into rows & cols
arr6 = einops.rearrange(arr, "(b1 b2) c h w -> c (b1 h) (b2 w)", b1=2)
display_array_as_img(arr6)


# %% (7) Transpose
arr7 = einops.rearrange(arr[1], "c h w -> c w h")
display_array_as_img(arr7)

# %% (8) Shrinking
arr8 = einops.reduce(
    arr, "(b1 b2) c (h h2) (w w2) -> c (b1 h) (b2 w)", "max", h2=2, w2=2, b1=2
)
display_array_as_img(arr8)


# %% Exercises - einops operations & broadcasting
def assert_all_equal(actual: Tensor, expected: Tensor) -> None:
    assert actual.shape == expected.shape, f"Shape mismatch, got: {actual.shape}"
    assert (actual == expected).all(), f"Value mismatch, got: {actual}"
    print("Tests passed!")


def assert_all_close(actual: Tensor, expected: Tensor, atol=1e-3) -> None:
    assert actual.shape == expected.shape, f"Shape mismatch, got: {actual.shape}"
    t.testing.assert_close(actual, expected, atol=atol, rtol=0.0)
    print("Tests passed!")


# %% (A1) rearrange
def rearrange_1() -> Tensor:
    """Return the following tensor using only t.arange and einops.rearrange:

    [[3, 4],
     [5, 6],
     [7, 8]]
    """
    arr = t.arange(3, 9)
    return einops.rearrange(arr, "(l1 l2) -> l1 l2", l1=3)


expected = t.tensor([[3, 4], [5, 6], [7, 8]])
assert_all_equal(rearrange_1(), expected)


# %% (A2) rearrange
def rearrange_2() -> Tensor:
    """Return the following tensor using only t.arange and einops.rearrange:

    [[1, 2, 3],
     [4, 5, 6]]
    """
    arr = t.arange(1, 7)
    return einops.rearrange(arr, "(l1 l2) -> l1 l2", l1=2)


assert_all_equal(rearrange_2(), t.tensor([[1, 2, 3], [4, 5, 6]]))


# %% (B1) temperature average
def temperatures_average(temps: Tensor) -> Tensor:
    """Return the average temperature for each week.

    temps: a 1D temperature containing temperatures for each day.
    Length will be a multiple of 7 and the first 7 days are for the first week, second 7 days for the second week, etc.

    You can do this with a single call to reduce.
    """
    assert len(temps) % 7 == 0
    raise NotImplementedError()


temps = t.tensor([71, 72, 70, 75, 71, 72, 70, 75, 80, 85, 80, 78, 72, 83]).float()
expected = [71.571, 79.0]
assert_all_close(temperatures_average(temps), t.tensor(expected))
