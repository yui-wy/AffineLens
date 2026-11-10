# AffineLens

Research code for **AffineLens: Capturing the Continuous Piecewise Affine Functions of Neural Networks**.

**Yi Wei, Xuan Qi, Furao Shen, Jian Zhao, Vittorio Murino, and Cigdem Beyan**

[Paper on arXiv](https://arxiv.org/abs/2605.06218) · ACML 2026 Journal Track

The repository is named **CPAExploration**; its Python library is imported as **`torchays`**.

## Overview

AffineLens studies how a piecewise affine neural network partitions a bounded input domain. Under a fixed activation pattern, the network is an affine map of its input. The method combines affine propagation, linear programming, and breadth-first exploration of neighboring regions to recover the activation-induced partition and interior representative points.

This repository provides:

- **Local affine representations:** effective weights and biases propagated through graph-aware layers.
- **Region enumeration:** region counts, halfspace constraints, and interior representatives on a bounded domain.
- **Visualization:** two-dimensional region partitions and network-output geometry.
- **Experiments:** model training, checkpoint analysis, and comparisons across architectures and training stages.

The regions are **activation-induced affine regions**. Different activation patterns may produce the same final affine map; this code does not merge such cells into maximal affine regions of the realized function. Counts depend on the analysis domain.

![Example of an affine-region partition](docs/images/regions.png)

## Installation

Use Python 3.11 and a virtual environment. From the repository root:

```bash
python -m pip install "numpy<2" -r requirements.txt
```

Dependencies include PyTorch, torchvision, NumPy, SciPy, CVXOPT, polytope, scikit-learn, and Matplotlib. The requirements file pins Matplotlib to 3.6.3 but does not fully pin the environment. The command above keeps NumPy below 2 for this older binary stack; see the [NumPy compatibility guidance](https://numpy.org/doc/2.0/user/troubleshooting-importerror.html). Use a compatible PyTorch/torchvision pair and a PyTorch build appropriate for your CPU or CUDA setup.

Documentation smoke checks used an existing environment with Python 3.11.6, PyTorch 2.1.0+cu121, torchvision 0.16.0+cu121, NumPy 1.24.1, SciPy 1.11.3, and Matplotlib 3.6.3. A fresh installation and compatibility with every newer release have not been verified.

There is no `setup.py` or `pyproject.toml` in this checkout. Run examples as modules **from the repository root** so Python can find `torchays`:

```bash
python -m examples.graph_demo
python -m examples.regions_demo
```

## Quick start

This CPU example creates two orthogonal ReLU boundaries in `[-1, 1]^2`, collects four regions, and verifies the affine representation at two inputs. No dataset or trained checkpoint is required. Run it from the repository root, or save it there as a Python script.

```python
import torch

from torchays import nn
from torchays.cpa import BaseHandler, CPAFactory


class SmallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.depth = 0  # Zero-based index of the last activation-producing layer.
        self.hidden = nn.Linear(2, 2)
        self.relu = nn.ReLU()
        self.readout = nn.Linear(2, 1)
        with torch.no_grad():
            self.hidden.weight.copy_(torch.eye(2))
            self.hidden.bias.zero_()

    def forward(self, x):
        return self.readout(self.relu(self.hidden(x)))

    def forward_layer(self, x, depth=0):
        # Return pre-activations; the final affine readout adds no boundaries.
        if depth != 0:
            raise ValueError("SmallNet has only one activation-producing layer.")
        return self.hidden(x)


class RegionCollector(BaseHandler):
    with_region = True

    def __init__(self):
        self.regions = []

    def region(self, fun, region, point):
        self.regions.append((fun.clone(), region.clone(), point.clone()))


if __name__ == "__main__":
    torch.manual_seed(5)
    net = SmallNet()
    collector = RegionCollector()
    cpa = CPAFactory(workers=1, device=torch.device("cpu")).CPA(
        net, handler=collector
    )
    count = cpa.start(input_size=(2,), bounds=(-1.0, 1.0))
    print("Regions:", count)  # Expected: 4.
    assert count == len(collector.regions) == 4

    x = torch.tensor([[-0.5, 0.25], [0.5, -0.25]])
    net.graph()
    with torch.no_grad():
        y = net(x)
        reconstructed = (
            torch.einsum("boi,bi->bo", y.weight_graph, x) + y.bias_graph
        )
        torch.testing.assert_close(y, reconstructed)
```

## Using the library

### Graph-aware models

Build the analyzed model with `torchays.nn` layers. `net.graph()` enables affine propagation and puts custom modules into evaluation mode. Run graph forward passes inside `torch.no_grad()`.

The output is a tensor with two attributes, **not** an `(output, graph)` tuple:

- `output.weight_graph`: effective weights of shape `(*output.shape, *input.shape[1:])`.
- `output.bias_graph`: effective biases with the same shape as `output`.

For each sample, contracting its effective weights with its input and adding its bias reconstructs its output. These coefficients describe the affine map selected by that sample's activation pattern. Calling `net.train()` or `net.eval()` disables graph mode; call `net.graph()` again before extracting coefficients.

Layers include `Linear`, `Conv2d`, `Flatten`, `ReLU`, `BatchNorm1d/2d/3d`, and average-pooling modules. LeakyReLU is exposed under the existing name **`nn.LeakyRule`**. Batch normalization requires stored running statistics (`track_running_stats=True`). Residual models use the graph-aware addition helper in `torchays.nn.modules.base` to propagate values and coefficients.

For enumeration, a model must also expose:

- `depth`: the **zero-based index** of the last activation-producing layer. For two ReLU layers, use `depth = 1`.
- `forward_layer(x, depth)`: the pre-activation tensor at that index, with affine attributes. Include earlier activations, but stop before the current activation.

A plain `torch.nn.Module` is not automatically converted. Custom operations must preserve affine attributes. The presence of `MaxPool2d` in the library does not establish complete enumeration of its switching boundaries; the enumeration examples use ReLU boundaries.

### Enumeration and region data

`CPAFactory(...).CPA(net, depth=-1, handler=None)` creates an enumerator; `depth=-1` uses `net.depth`.

```python
count = cpa.start(input_size=(2,), bounds=(-1.0, 1.0))
```

`input_size` excludes the batch dimension. The public `start` interface constructs **axis-aligned boxes**:

| `bounds` | Analysis domain |
| --- | --- |
| `1.0` | `[-1, 1]` in every coordinate |
| `(-2.0, 3.0)` | `[-2, 3]` in every coordinate |
| `((-1.0, 1.0), (0.0, 2.0))` | Separate intervals for a two-dimensional input |

Use a tuple of tuples for per-coordinate bounds. The paper's general bounded-polytope formulation is broader than this convenience interface.

With `with_region=True`, the handler receives `(fun, region, point)` for each final region. For input dimension `d`, `fun` has shape `(m, d + 1)`, `region` holds `m` signs, and `point` is an interior representative. The closed halfspace representation is:

```text
region[j] * (fun[j, :-1] @ x + fun[j, -1]) >= 0, for every j.
```

These constraints are not the final network-output coefficients; obtain those with a graph forward pass at a representative. Convert to the `polytope` convention `A @ x <= b` as follows:

```python
signed = region[:, None] * fun
A = -signed[:, :-1]
b = signed[:, -1]
```

For intermediate arrangements, set `with_hpas=True` and implement `inner_hyperplanes(...)`. `cpa.start(point=point, bounds=...)` instead follows the region containing one input; omit the batch dimension from `point`. Create a fresh enumerator for each independent analysis.

### Examples

| File | Purpose |
| --- | --- |
| [graph_demo.py](examples/graph_demo.py) | Extract affine coefficients from a convolutional network and print reconstructed outputs. |
| [regions_demo.py](examples/regions_demo.py) | Enumerate and plot a randomly initialized two-hidden-layer ReLU MLP on `[-1, 1]^2`. |

Both examples use CUDA device 0 when available and otherwise use the CPU. The region example opens a Matplotlib window; it does not save an image by default. These are API demonstrations, not trained models reproducing the paper's tables.

## Experiments

[main.py](main.py) defines configurations for width, depth, residual connections, hyperplane arrangements, and image-model slices. [experiment/config.py](experiment/config.py) holds shared settings; [experiment/run.py](experiment/run.py) builds datasets and models.

Before running `python main.py`, select the intended entries in its `configs` list and review epochs, workers, devices, and data paths. The current driver schedules multiple experiments, with defaults including 5,000 epochs and 64 enumeration workers. It is not a quick-start command. The known limitations below also affect some presets.

Relevant settings include:

- `TRAIN.MAX_EPOCH`, `TRAIN.SAVE_EPOCH`, and `TRAIN.LR` for training.
- `EXPERIMENT.BOUND`, `DEPTH`, and `WORKERS` for enumeration.
- `EXPERIMENT.PROJ_DIM` and `PROJ_VALUES` for image-coordinate slices.
- `COMMON.GPU_ID`, `PATH.DIR`, and `IMAGENET.ROOT` for local resources.

Experiment-specific functions in `main.py` override shared settings. Results are stored beneath `PATH.DIR` (default `./cache`), including configuration snapshots, checkpoints, logs, and plots. MNIST and CIFAR-10 wrappers support downloads; ImageNet and Tiny ImageNet require locally prepared data. Datasets and trained checkpoints are not included.

## Scope and known limitations

- The paper describes exact enumeration of non-empty, full-dimensional activation-induced cells. This implementation uses floating-point propagation and LP solvers; numerical tolerances and degeneracies can affect results.
- Enumeration cost grows with region count and input dimension. Begin with small models and `workers=1`.
- A two-dimensional slice describes the partition on that slice, not the full high-dimensional partition.
- Keep the default `logging=True`: the current timing decorator can raise `UnboundLocalError` when logging is disabled.
- The full experiment driver has unresolved edge cases: random projection indices can exceed the input range, residual-model configuration can persist into later presets, and disabling drawing can leave region buffers unavailable during result saving. Review the selected preset before using it for reproduction.

## Repository layout

```text
CPAExploration/
├── torchays/
│   ├── nn/          # Graph-aware layers and affine propagation
│   ├── cpa/         # Enumeration, LP helpers, and callbacks
│   ├── graph/       # Region plotting
│   ├── models/      # MLP, CNN, and residual models
│   └── utils/       # Logging and CSV utilities
├── dataset/         # Toy and image dataset wrappers
├── experiment/      # Training and geometric analysis
├── examples/        # API demonstrations
├── docs/images/     # Documentation images
├── main.py          # Experiment driver
└── requirements.txt
```

## Citation

```bibtex
@misc{wei2026affinelens,
  title         = {AffineLens: Capturing the Continuous Piecewise Affine Functions of Neural Networks},
  author        = {Yi Wei and Xuan Qi and Furao Shen and Jian Zhao and Vittorio Murino and Cigdem Beyan},
  year          = {2026},
  eprint        = {2605.06218},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2605.06218}
}
```

## License

This code is distributed under the [MIT License](LICENSE).
