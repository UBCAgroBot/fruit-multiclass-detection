from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, Dict, Union

from common.autograd import Value


class Module(ABC):
    """Base class for all neural network modules.

    Your models should also subclass this class.

    Modules can also contain other Modules, allowing them to be nested in
    a tree structure. You can assign the submodules as regular attributes::

        import torch.nn as nn
        import torch.nn.functional as F


        class Model(Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)

            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))

    Submodules assigned in this way will be registered, and will also have their
    parameters converted when you call :meth:`to`, etc.

    .. note::
        As per the example above, an ``__init__()`` call to the parent class
        must be made before assignment on the child."""

    params: Dict[
        str, Union[Value, "Module"]
    ]  # to bypass mypys complaints that Module has no attribute of params

    def __init__(self) -> None:
        # Bypass the overwritten __setattr__ to avoid self.params registering itself
        super().__setattr__("params", {})  # real param store: name -> Value or Module

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> None:
        # abstract method
        print("forward method not implemented")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # will call self.forward!
        return self.forward(*args, **kwargs)

    def __setattr__(self, key: str, Val: Any) -> None:
        # Always set the actual attribute first
        super().__setattr__(key, Val)

        if key == "params":
            return

        if isinstance(Val, (Value, Module)):
            self.params[key] = Val
        else:
            self.params.pop(key, None)

    def parameters(self) -> Iterator[Value]:
        # recursively collects parameters and yields them (use a Generator)
        for key, param in self.params.items():
            if isinstance(param, Module):
                # walk through submodules
                yield from param.parameters()
            else:
                yield param

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0
