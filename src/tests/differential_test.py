from typing import Tuple

import numpy as np
import pytest
import torch

from common.autograd import Value

N = 20  # size 20 for the array


def random_tensors(
    shape: Tuple[int, ...], ranges: Tuple[float, float] = (-10.0, 10.0)
) -> Tuple[torch.Tensor, Value]:
    # random tensor for both PyTorch and Value
    low, high = ranges
    t_torch = (high - low) * torch.rand(shape) + low
    t_value = Value(t_torch.clone().numpy())
    return t_torch, t_value


# base class for differential tests
class OperationTest:
    def __init__(self, torch_fn, our_fn, ranges=(-10.0, 10.0)) -> None:  # type: ignore
        self.torch_fn = torch_fn
        self.our_fn = our_fn
        self.ranges = (
            ranges  # ranges repersent what values the random tensors will take on.
        )
        self.rtol = 1e-5
        self.atol = 1e-8

    def run_test(self, shape: Tuple[int, ...]) -> None:
        a_torch, a = random_tensors(shape, self.ranges)
        b_torch, b = random_tensors(shape, self.ranges)

        # compute both results
        res_our = self.our_fn(a, b)
        res_torch = self.torch_fn(a_torch, b_torch)

        # convert our Value result to numpy for comparison
        if isinstance(res_our.data, np.ndarray):
            our_data = res_our.data
        else:
            our_data = np.array(res_our.data)

        # compare against PyTorch
        assert np.allclose(our_data, res_torch.numpy(), rtol=self.rtol, atol=self.atol)


# tests
@pytest.mark.parametrize("shape", [(3, 3), (5, 2), (2, 4)])
def test_add(shape: Tuple[int, ...]) -> None:
    tester = OperationTest(torch.add, lambda a, b: a + b)
    tester.run_test(shape)


@pytest.mark.parametrize("shape", [(3, 3), (5, 2), (2, 4)])
def test_sub(shape: Tuple[int, ...]) -> None:
    tester = OperationTest(torch.sub, lambda a, b: a - b)
    tester.run_test(shape)


@pytest.mark.parametrize("shape", [(3, 3), (5, 2), (2, 4)])
def test_mul(shape: Tuple[int, ...]) -> None:
    tester = OperationTest(torch.mul, lambda a, b: a * b)
    tester.run_test(shape)


@pytest.mark.parametrize("shape", [(3, 3), (5, 2), (2, 4)])
def test_div(shape: Tuple[int, ...]) -> None:
    tester = OperationTest(torch.div, lambda a, b: a / b, (1e-10, 10))
    tester.run_test(shape)


# compound test
@pytest.mark.parametrize("shape", [(3, 3), (2, 4)])
def test_compound(shape: Tuple[int, ...]) -> None:
    # torchTensors = []
    # ourTensors = []
    # for i in range(N): # init the tensors

    a_torch, a = random_tensors(shape)
    b_torch, b = random_tensors(shape)

    # example expression: a*b + a/b
    # res_torch = a_torch * b_torch + a_torch / b_torch
    # res_our = a * b + a / b

    for n in range(0, 10):
        # pick random number from 1-9
        i = np.random.randint(1, 10)
        match i:
            case 1:  # "+" operator
                res_torch = a_torch + b_torch
                res_our = a + b
            case 2:  # "*" operator
                res_torch = a_torch * b_torch
                res_our = a * b
            case 3:  # "-" operator
                a_torch - b_torch
                res_our = a - b
            case 4:  # "/" operator
                a_torch / b_torch
                res_our = a / b
            case 5:  # "exp" operator(uniary)
                res_torch = a_torch.exp()
                res_our = a.exp()
            case 7:  # "-"   operator(uniary)
                res_torch = a_torch * -1
                res_our = a * -1
            case 8:  # "@"   operator
                res_torch = a_torch @ b_torch
                res_our = a @ b
            case 9:  # "**"  operator
                res_torch = a_torch**3
                res_our = a**3

    our_data = (
        res_our.data if isinstance(res_our.data, np.ndarray) else np.array(res_our.data)
    )
    assert np.allclose(
        our_data, res_torch.numpy(), rtol=1e-5, atol=1e-8
    )  # nessacry when doing compound tests?
