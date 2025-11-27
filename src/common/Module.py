from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, Union

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
    def __init__(self) -> None:
        self.params: dict[str, Value | "Module"] = {} # real param store: name -> Value or Module
  
        
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Value:
        """Compute the forward pass for this module."""

        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Value:
        # will call self.forward!
        return self.forward(*args, **kwargs)

    def __setattr__(self, key: str, value: Any) -> None:
        # Always set the actual attribute first
        super().__setattr__(key, value)

        # Skip bookkeeping for the param store itself. Without this guard, the
        # act of constructing the store would try to register "params" inside
        # itself, creating incorrect entries and breaking parameter traversal in
        # ``parameters``/``zero_grad``.
        if key == "params":
            return

        # Attribute names arrive without a "self." prefix (e.g., "x"), so we
        # can register them directly under that key. Only Value and Module
        # instances belong in the parameter registry; other attribute types
        # should not be exposed to ``parameters``. When an attribute is replaced
        # with a non-parameter value, we drop the stale entry to keep the
        # registry in sync with the current state of the object.
        if isinstance(value, (Value, Module)):
            self.params[key] = value

    def parameters(self) -> Generator[Value,None,None]:
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
