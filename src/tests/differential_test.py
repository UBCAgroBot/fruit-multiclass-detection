from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import torch

from common.autograd import Value

N = 20  # number of random shape cases used in parametrized tests
SEED = 1337
RTOL = 1e-5
ATOL = 1e-8

Shape = tuple[int, ...]
ValueBinaryOp = Callable[[Value, Value], Value]
TorchBinaryOp = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def random_tensors(
    shape: Shape,
    ranges: tuple[float, float] = (-10.0, 10.0),
) -> tuple[torch.Tensor, Value]:
    low, high = ranges
    t_torch = (high - low) * torch.rand(shape) + low
    t_value = Value(t_torch.clone().numpy())
    return t_torch, t_value


def random_shapes(
    rng: np.random.Generator,
    *,
    count: int,
    min_ndim: int = 1,
    max_ndim: int = 6,
    min_size: int = 1,
    max_size: int = 6,
) -> list[Shape]:
    shapes: list[Shape] = []
    for _ in range(count):
        ndim = int(rng.integers(min_ndim, max_ndim + 1))
        dims = tuple(int(x) for x in rng.integers(min_size, max_size + 1, size=ndim))
        shapes.append(dims)
    return shapes


RNG = np.random.default_rng(SEED)
RANDOM_SHAPES = random_shapes(RNG, count=N)
COMPOUND_SHAPES = random_shapes(RNG, count=8, min_ndim=1, max_ndim=8)


class OperationTest:
    def __init__(
        self,
        torch_fn: TorchBinaryOp,
        our_fn: ValueBinaryOp,
        ranges: tuple[float, float] = (-10.0, 10.0),
    ) -> None:
        self.torch_fn = torch_fn
        self.our_fn = our_fn
        self.ranges = ranges
        self.rtol = RTOL
        self.atol = ATOL

    def run_test(self, shape: Shape) -> None:
        a_torch, a = random_tensors(shape, self.ranges)
        b_torch, b = random_tensors(shape, self.ranges)

        res_our = self.our_fn(a, b)
        res_torch = self.torch_fn(a_torch, b_torch)

        if isinstance(res_our.data, np.ndarray):
            our_data = res_our.data
        else:
            our_data = np.array(res_our.data)

        assert np.allclose(our_data, res_torch.numpy(), rtol=self.rtol, atol=self.atol)


@pytest.mark.parametrize("shape", RANDOM_SHAPES)
def test_add(shape: Shape) -> None:
    tester = OperationTest(torch.add, lambda a, b: a + b)
    tester.run_test(shape)


@pytest.mark.parametrize("shape", RANDOM_SHAPES)
def test_sub(shape: Shape) -> None:
    tester = OperationTest(torch.sub, lambda a, b: a - b)
    tester.run_test(shape)


@pytest.mark.parametrize("shape", RANDOM_SHAPES)
def test_mul(shape: Shape) -> None:
    tester = OperationTest(torch.mul, lambda a, b: a * b)
    tester.run_test(shape)


@pytest.mark.parametrize("shape", RANDOM_SHAPES)
def test_div(shape: Shape) -> None:
    tester = OperationTest(torch.div, lambda a, b: a / b, (1e-10, 10.0))
    tester.run_test(shape)


@pytest.mark.parametrize("shape", COMPOUND_SHAPES)
def test_compound(shape: Shape) -> None:
    seed = 0
    for i, dim in enumerate(shape):
        seed += (i + 1) * dim * 9973
    seed %= 2**32
    rng = np.random.default_rng(seed)

    # Keep values bounded so long random expression chains stay numerically stable.
    a_torch, a = random_tensors(shape, ranges=(0.5, 2.0))
    b_torch, b = random_tensors(shape, ranges=(0.5, 2.0))

    torch_pool: list[torch.Tensor] = [a_torch, b_torch]
    our_pool: list[Value] = [a, b]

    for _ in range(6):
        i1 = int(rng.integers(0, len(torch_pool)))
        i2 = int(rng.integers(0, len(torch_pool)))

        t1, t2 = torch_pool[i1], torch_pool[i2]
        v1, v2 = our_pool[i1], our_pool[i2]

        op = int(rng.integers(1, 10))
        res_torch: torch.Tensor | None = None
        res_our: Value | None = None

        match op:
            case 1:
                res_torch = t1 + t2
                res_our = v1 + v2
            case 2:
                res_torch = t1 * t2
                res_our = v1 * v2
            case 3:
                res_torch = t1 - t2
                res_our = v1 - v2
            case 4:
                res_torch = t1 / (t2 + 1e-3)
                res_our = v1 / (v2 + 1e-3)
            case 5:
                res_torch = t1.exp()
                res_our = v1.exp()
            case 7:
                res_torch = t1 * -1
                res_our = v1 * -1
            case 8:
                n = int(np.prod(t1.shape))
                m = int(np.prod(t2.shape))

                k = int(np.gcd(n, m))

                shape1 = (n // k, k)
                shape2 = (k, m // k)

                res_torch = t1.reshape(shape1) @ t2.reshape(shape2)
                res_our = v1.reshape(shape1) @ v2.reshape(shape2)
            case 9:
                res_torch = t1**3
                res_our = v1**3

        if res_torch is not None and res_our is not None:
            torch_pool.append(res_torch)
            our_pool.append(res_our)

    final_torch = torch_pool[-1]
    final_our = our_pool[-1]

    our_data = (
        final_our.data
        if isinstance(final_our.data, np.ndarray)
        else np.array(final_our.data)
    )
    assert np.allclose(our_data, final_torch.numpy(), rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("shape", RANDOM_SHAPES)
def test_add_is_commutative(shape: Shape) -> None:
    a_torch, a = random_tensors(shape)
    b_torch, b = random_tensors(shape)

    torch_ab = (a_torch + b_torch).numpy()
    torch_ba = (b_torch + a_torch).numpy()

    our_ab = np.array((a + b).data)
    our_ba = np.array((b + a).data)

    assert np.allclose(our_ab, torch_ab, rtol=RTOL, atol=ATOL)
    assert np.allclose(our_ba, torch_ba, rtol=RTOL, atol=ATOL)
    assert np.allclose(our_ab, our_ba, rtol=RTOL, atol=ATOL)


# view operations tests
@pytest.mark.parametrize("shape", [(3, 4), (2, 5)])
def test_reshape(shape: Shape) -> None:
    # Test reshaping into a 1D array
    target_shape = (int(np.prod(shape)),)
    tester = OperationTest(
        lambda a, _: torch.reshape(a, target_shape),
        lambda a, _: a.reshape(target_shape),
    )
    tester.run_test(shape)


@pytest.mark.parametrize("shape", [(3, 4), (2, 5)])
def test_unsqueeze(shape: Shape) -> None:
    # Equivalent to unsqueeze at axis 0 for current Value API.
    target_shape = (1, *shape)
    tester = OperationTest(
        lambda a, _: torch.unsqueeze(a, dim=0),
        lambda a, _: a.reshape(target_shape),
    )
    tester.run_test(shape)


@pytest.mark.parametrize("shape", [(1, 3, 4), (2, 1, 5)])
def test_squeeze(shape: Shape) -> None:
    # Equivalent to squeeze for these fixed shapes using reshape.
    target_shape = tuple(dim for dim in shape if dim != 1)
    tester = OperationTest(
        lambda a, _: torch.squeeze(a),
        lambda a, _: a.reshape(target_shape),
    )
    tester.run_test(shape)
