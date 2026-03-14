import random
from typing import Any

from common.autograd import Value
from common.Module import Module


class Neuron(Module):
    def __init__(self, nin: int, nonlin: bool = True) -> None:
        super().__init__()
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def forward(self, x: list[Value]) -> Value:
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self) -> list[Value]:
        return self.w + [self.b]

    def __repr__(self) -> str:
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


class Layer(Module):
    def __init__(self, nin: int, nout: int, **kwargs: Any) -> None:
        super().__init__()
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def forward(self, x: list[Value]) -> Value | list[Value]:
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self) -> list[Value]:
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self) -> str:
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"


class MLP(Module):
    def __init__(self, nin: int, nouts: list[int]) -> None:
        super().__init__()
        sz = [nin] + nouts
        self.layers = [
            Layer(sz[i], sz[i + 1], nonlin=i != len(nouts) - 1)
            for i in range(len(nouts))
        ]

    def forward(self, x: list[Value]) -> Any:
        current: Any = x
        for layer in self.layers:
            current = layer(current)
        return current

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self) -> str:
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
