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

    def __add__(self, other: "Value") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __mul__(self, other: "Value" | int) -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    # new code!
    def __matmul__(self, other: "Value") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        assert (
            self.data.shape[1] == other.data.shape[0]
        )  # make sure u can actually multiply them (dimension check)
        out = Value(self.data @ other.data, (self, other), "@")  # do the multiplication

        def _backward() -> None:
            self.grad += out.grad @ (other.data.T)
            other.grad += (self.data.T) @ out.grad

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
            self.grad += out.data * out.grad  # what is out.grad again?

        out._backward = _backward
        return out

    def relu(self) -> "Value":
        out = Value(0 if self.data < 0 else self.data, (self,), "ReLU")

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
