from typing import Union

try:
    import cupy as np
except ImportError:
    print("CUDA not available, defaulting to numpy")
    import numpy as np


class Value:
    """stores value's and its gradient's"""

    def __init__(
        self,
        data: float | np.ndarray,
        _children: tuple["Value", ...] = (),
        _op: str = "",
    ) -> None:
        self.data = np.asarray(
            data, dtype=float
        )  # ensure even if its a scalar its an "1d" array
        self.grad = np.zeros_like(
            self.data, dtype=float
        )  # make sure the grad is an "1d" array as well
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op  # the op that produced this node, for graphviz / debugging / etc

    def sum_to_shape(
        self, tensor: np.ndarray, target_shape: tuple[int, ...]
    ) -> np.ndarray:
        """Sum a tensor along axes to match a target shape.

        This handles two cases:
        1. Extra dimensions that were added during broadcasting (sum them away)
        2. Dimensions that were size 1 but broadcasted to larger (sum along them, keep dim=1)

        Args:
            tensor: numpy array to reshape
            target_shape: tuple of desired shape

        Returns:
            tensor with shape matching target_shape
        """
        if tensor.shape == target_shape:  # the do nothing case
            return tensor
        # Start with the tensor as-is
        result = tensor

        # Case 1: Handle extra leading dimensions
        # If tensor has more dimensions than target, sum away the extra leading ones
        ndim_diff = result.ndim - len(target_shape)
        if ndim_diff > 0:
            # Sum along the extra leading axes
            axes_to_sum = tuple(range(ndim_diff))
            result = result.sum(axis=axes_to_sum, keepdims=False)

        # Case 2: Handle dimensions that were broadcast from size 1
        # Now result and target_shape should have same number of dimensions
        for i, (result_dim, target_dim) in enumerate(zip(result.shape, target_shape)):
            if target_dim == 1 and result_dim > 1:
                # This dimension was broadcast from 1 to result_dim
                # Sum along it and keep it as dimension 1
                result = result.sum(axis=i, keepdims=True)
        return result

    def __add__(self, other: "Value") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            # 1. old way
            grad_self = out.grad
            grad_other = out.grad

            # 2. Fix shapes to handle broadcasting (new code)
            grad_self = self.sum_to_shape(grad_self, self.data.shape)
            grad_other = self.sum_to_shape(grad_other, other.data.shape)

            # 3. Accumulate gradients
            self.grad += grad_self
            other.grad += grad_other

        out._backward = _backward

        return out

    def __mul__(
        self, other: Union["Value", int]
    ) -> (
        "Value"
    ):  # some weird python black thing, so we made it use the union from typing instead(cuz it thinks its a string)
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            # 1. Compute gradients (the old way)
            grad_self = other.data * out.grad
            grad_other = self.data * out.grad

            # 2. Fix shapes to handle broadcasting (new code)
            grad_self = self.sum_to_shape(grad_self, self.data.shape)
            grad_other = self.sum_to_shape(grad_other, other.data.shape)

            # 3. Accumulate gradients
            self.grad += grad_self
            other.grad += grad_other

        out._backward = _backward

        return out

    # new code!
    def __matmul__(self, other: "Value") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        assert (
            len(self.data.shape) == 2 and len(other.data.shape) == 2
        )  # runtime error: both operands must be 2D matrices
        assert (
            self.data.shape[-1] == other.data.shape[0]  # more general mat mul condition
        )  # make sure u can actually multiply them (dimension check)
        out = Value(self.data @ other.data, (self, other), "@")  # do the multiplication

        def _backward() -> None:
            grad_self = out.grad @ other.data.T
            grad_other = self.data.T @ out.grad
            self.grad += grad_self
            other.grad += grad_other

        out._backward = _backward

        return out

    def __pow__(self, other: float | int) -> "Value":
        assert isinstance(
            other, (int, float)
        ), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward() -> None:
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    # new code
    def exp(self) -> "Value":
        out = Value(np.exp(self.data), (self,), "exp")

        def _backward() -> None:
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def relu(self) -> "Value":
        out = Value(0 if self.data.any() < 0 else self.data, (self,), "ReLU")

        def _backward() -> None:
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward

        return out

    def backward(self) -> None:
        # topological order all of the children in the graph
        topo = []
        visited = set()

        def build_topo(v: "Value") -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = np.ones_like(
            self.data, dtype=float
        )  # make sure that grad is a full on matrix(in the matix case)
        for v in reversed(topo):
            v._backward()

    def __neg__(self) -> "Value":  # -self
        return self * -1

    def __radd__(self, other: "Value") -> "Value":  # other + self
        return self + other

    def __sub__(self, other: "Value") -> "Value":  # self - other
        return self + (-other)

    def __rsub__(self, other: "Value") -> "Value":  # other - self
        return other + (-self)

    def __rmul__(self, other: "Value") -> "Value":  # other * self
        return self * other

    def __truediv__(self, other: "Value") -> "Value":  # self / other
        return self * other**-1

    def __rtruediv__(self, other: "Value") -> "Value":  # other / self
        return other * self**-1

    def __repr__(self) -> "str":
        return f"Value(data={self.data}, grad={self.grad})"
