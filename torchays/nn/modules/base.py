import torch
import torch.nn as nn

WEIGHT_GRAPH = "weight_graph"
BIAS_GRAPH = "bias_graph"


class Tensor(torch.Tensor):
    weight_graph: torch.Tensor = None
    bias_graph: torch.Tensor = None


def add(a: Tensor, b: Tensor):
    new = a + b
    wg = a.weight_graph + b.weight_graph
    bg = a.bias_graph + b.bias_graph
    return set_graph(new, wg, bg)


def get_size_to_one(size: torch.Size):
    assert isinstance(size, torch.Size), 'Input must be a torch.Size'
    return torch.Size((1,) * len(size))


def get_origin_size(input: Tensor):
    if not hasattr(input, WEIGHT_GRAPH) or input.weight_graph is None:
        return input.size()[1:]
    return input.weight_graph.size()[input.dim() :]


def check_graph(x: Tensor):
    if hasattr(x, WEIGHT_GRAPH) and hasattr(x, BIAS_GRAPH):
        return x, x.weight_graph, x.bias_graph
    return set_graph(x), None, None


def set_graph(x: Tensor, wg: torch.Tensor = None, bg: torch.Tensor = None):
    setattr(x, WEIGHT_GRAPH, wg)
    setattr(x, BIAS_GRAPH, bg)
    return x


class Module(nn.Module):
    """Extend a PyTorch module with affine-coefficient propagation in graph mode.

    The output tensor carries ``weight_graph`` and ``bias_graph`` attributes:

        >>> net.graph()
        >>> with torch.no_grad():
        ...     output = net(input)
        ...     weight, bias = output.weight_graph, output.bias_graph
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.graphing = False
        self.origin_size: torch.Size = None

    def _origin_size(self, input: Tensor):
        if self.origin_size is None:
            self.origin_size = get_origin_size(input)
        return self.origin_size

    def forward_graph(self, input):
        raise NotImplementedError()

    def train(self, mode: bool = True):
        self.graphing = False
        return nn.Module.train(self, mode)

    def graph(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = False
        self.graphing = mode
        for module in self.children():
            if isinstance(module, Module):
                module.graph()
        return self

    def forward(self, *args, **kwargs):
        if self.graphing:
            return self.forward_graph(*args, **kwargs)
        else:
            return super().forward(*args, **kwargs)
