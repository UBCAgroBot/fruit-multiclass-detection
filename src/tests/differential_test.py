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
def test_compound() -> None:
    # Start with two random tensors (random shape each run)
    ndim = np.random.randint(1, 3)
    shape = tuple(int(x) for x in np.random.randint(1, 6, size=ndim))
    a_torch, a = random_tensors(shape)
    b_torch, b = random_tensors(shape)

    # Keep track of all results in a list (the "pool")
    torch_pool = [a_torch, b_torch]
    our_pool = [a, b]

    for n in range(0, 10):
        # Pick two random items from the pool
        i1 = np.random.randint(0, len(torch_pool))
        i2 = np.random.randint(0, len(torch_pool))

        t1, t2 = torch_pool[i1], torch_pool[i2]
        v1, v2 = our_pool[i1], our_pool[i2]

        i = np.random.randint(1, 10)
        res_torch, res_our = None, None

        match i:
            case 1:  # "+" operator
                res_torch = t1 + t2
                res_our = v1 + v2
            case 2:  # "*" operator
                res_torch = t1 * t2
                res_our = v1 * v2
            case 3:  # "-" operator
                res_torch = t1 - t2
                res_our = v1 - v2
            case 4:  # "/" operator
                res_torch = t1 / (t2 + 0.0000000000000001)  # avoid 0
                res_our = v1 / (v2 + 0.0000000000000001)
            case 5:  # "exp" operator
                res_torch = t1.exp()
                res_our = v1.exp()
            case 7:  # unary "-" operator
                res_torch = t1 * -1
                res_our = v1 * -1
            case 8:  # "@" operator
                # gcd approach:
                # 1) flatten each tensor to a total element count (n and m)
                # 2) pick the largest shared factor k so we can build a valid matmul
                #    with shapes (n/k, k) @ (k, m/k)
                # 3) reshape both tensors into those 2D matrices
                # 4) perform matmul on the reshaped views
                # This keeps all elements, avoids invalid shapes, and guarantees
                # the inner dimension matches for matrix multiply.
                n = int(np.prod(t1.shape))  # total elements in t1
                m = int(np.prod(t2.shape))  # total elements in t2

                k = np.gcd(n, m)  # shared factor for a valid inner dimension

                shape1 = (n // k, k)  # (rows, inner)
                shape2 = (k, m // k)  # (inner, cols)

                # reshape into 2D matrices that are guaranteed to be compatible
                res_torch = t1.reshape(shape1) @ t2.reshape(shape2)
                res_our = v1.reshape(shape1) @ v2.reshape(shape2)
            case 9:  # "**" operator
                res_torch = t1**3
                res_our = v1**3

        # add result to pool so it can be used again
        if res_torch is not None:
            torch_pool.append(res_torch)
            our_pool.append(res_our)

    # compare final result
    final_torch = torch_pool[-1]
    final_our = our_pool[-1]

    our_data = (
        final_our.data
        if isinstance(final_our.data, np.ndarray)
        else np.array(final_our.data)
    )
    assert np.allclose(
        our_data, res_torch.numpy(), rtol=1e-5, atol=1e-8
    )  # nessacry when doing compound tests?


# view operations tests
@pytest.mark.parametrize("shape", [(3, 4), (2, 5)])
def test_reshape(shape: Tuple[int, ...]) -> None:
    # Test reshaping into a 1D array
    target_shape = (int(np.prod(shape)),)
    tester = OperationTest(
        lambda a, _: torch.reshape(a, target_shape),
        lambda a, _: a.reshape(target_shape),
    )
    tester.run_test(shape)


@pytest.mark.parametrize("shape", [(3, 4), (2, 5)])
def test_unsqueeze(shape: Tuple[int, ...]) -> None:
    # Test adding a dimension at axis 0
    tester = OperationTest(
        lambda a, _: torch.unsqueeze(a, dim=0), lambda a, _: a.unsqueeze(axis=0)
    )
    tester.run_test(shape)


@pytest.mark.parametrize("shape", [(1, 3, 4), (2, 1, 5)])
def test_squeeze(shape: Tuple[int, ...]) -> None:
    # Test removing dimensions of size 1
    tester = OperationTest(lambda a, _: torch.squeeze(a), lambda a, _: a.squeeze())
    tester.run_test(shape)
