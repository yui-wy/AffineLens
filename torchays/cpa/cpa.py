import multiprocessing as mp
import os
import time
from logging import Logger
from multiprocessing.pool import AsyncResult
from multiprocessing.reduction import ForkingPickler
from typing import Callable, List, Tuple

import numpy as np
import torch

from torchays.nn.modules.base import Tensor

from ..nn import Module
from ..utils import get_logger
from .handler import BaseHandler
from .model import Model
from .optimization import cheby_ball, lineprog_intersect
from .regions import CPACache, CPACacheFactory, CPAFunc, CPAHandler, CPASet, WapperRegion
from .util import check_point, err_callback, find_projection, generate_bound_regions, get_regions, log_time, vertify


class CPAFactory:
    def __init__(
        self,
        workers: int = 1,
        device: torch.device = torch.device("cpu"),
        logger: Logger = None,
        logging: bool = True,
    ):
        self.device = device
        self.logger = logger or get_logger("CPAExploration-Console", multi=(workers > 1))
        self.logging = logging
        self.workers = workers if workers > 1 else 1

    def CPA(
        self,
        net: Model,
        depth: int = -1,
        handler: BaseHandler = None,
        logger: Logger = None,
    ):
        if logger is None:
            logger = self.logger
        return CPA(
            net=net,
            depth=depth,
            handler=handler,
            device=self.device,
            workers=self.workers,
            logger=logger,
            logging=self.logging,
        )

    def __call__(self, *args, **kwds):
        return self.CPA(*args, **kwds)


class CPA:
    """Analyze activation-induced affine regions of a piecewise affine network.

    Count regions within a bounded input box, or obtain the boundary constraints
    of the region containing a specified input point.
    """

    def __init__(
        self,
        net: Model,
        depth: int = -1,
        handler: BaseHandler = None,
        workers: int = 1,
        device: torch.device = torch.device("cpu"),
        logger: Logger = None,
        logging: bool = True,
    ):
        assert isinstance(net, Module), "the type of net must be \"BaseModule\"."
        self.net = net.graph().to(device)
        self.depth = net.depth
        if depth >= 0 and depth <= net.depth:
            self.depth = depth
        self.cpa_handler = CPAHandler(handler, self.depth)
        self.device = device
        self.logging = logging
        self.logger = logger
        # Multiprocessing configuration.
        self.workers, self.pool = 1, None
        if workers > 1:
            self.workers = workers

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict["pool"], self_dict["net"], self_dict["cpa_handler"]
        return self_dict

    def start(
        self,
        point: torch.Tensor = None,
        input_size: tuple = (2,),
        bounds: float | int | Tuple[float, float] | Tuple[Tuple[float, float]] = 1.0,
    ) -> int | Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        self._check_multiprocessing_method()
        if input_size is None and point is None:
            raise ValueError("input_size and point can not be None at the same time.")
        if point is None:
            return self._start(bounds=bounds, input_size=input_size)
        return self._start_point(point, bounds=bounds)

    def _check_spawn(self) -> bool:
        method = mp.get_start_method()
        if method == "spawn":
            return True
        return False

    def _check_multiprocessing_method(self):
        if self.workers == 1 or self._check_spawn():
            return
        self.logger.warning(f"The start method of multi-processing is not \"spawn ({mp.get_start_method()})\", which will cause some problems when performing certain operations in Pytorch.")

    def _start_point(
        self,
        point: torch.Tensor,
        *,
        bounds: float | int | Tuple[float, float] | Tuple[Tuple[float, float]] = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Find the region containing ``point`` within the specified bounds.

        Returns:
            Boundary functions, their halfspace signs, and the neuron-induced
            affine functions computed at the final analyzed depth.
        """
        # Initialize the input shape and dimension.
        self.net.origin_size = point.size()
        dim = self.net.origin_size.numel()
        # Construct the input-box boundary constraints.
        p_funcs, p_region, _, self.o_bounds = generate_bound_regions(bounds, dim)
        # Initialize the region set.
        cpa_set, cpa = CPASet(), CPAFunc(p_funcs, p_region, point)
        cpa_set.register(cpa)
        # Follow the input point through successive region partitions.
        self.logger.info("Start Get point cpa.")
        start = time.time()
        point_cpa, neural_funcs = self._get_point_cpa(cpa_set)
        self.logger.info(f"End Get point cpa. Times: {time.time()-start}s")
        return point_cpa.funcs, point_cpa.region, neural_funcs

    def _get_point_cpa(self, cpa_set: CPASet) -> Tuple[CPAFunc, torch.Tensor]:
        for p_cpa in cpa_set:
            cpa_cache = self.cpa_handler.cpa_cache_factory().cpa_cache()
            # Compute the neuron-induced hyperplanes at this depth.
            c_funcs = self._functions(p_cpa.point, p_cpa.depth)
            intersect_funcs, _ = self._find_intersect(p_cpa, c_funcs)
            if intersect_funcs is None:
                c_cpa = CPAFunc(p_cpa.funcs, p_cpa.region, p_cpa.point, p_cpa.depth + 1)
                self._nn_region_counts(c_cpa, cpa_set.register, cpa_cache.cpa)
            else:
                c_region = get_regions(p_cpa.point.reshape(1, -1), intersect_funcs)[0]
                # Validate the child region; neighbor candidates are unused here.
                c_cpa, _, _ = self._optimize_child_region(p_cpa, intersect_funcs, c_region)
                if c_cpa is None:
                    return None, None
                # Keep the original query point as the next layer's reference.
                c_cpa.point = p_cpa.point
                self._nn_region_counts(c_cpa, cpa_set.register, cpa_cache.cpa)
            # Record the parent constraints, layer functions, intersections, and child count.
            cpa_cache.hyperplane(p_cpa, c_funcs, intersect_funcs, 1)
            self.cpa_handler.extend(cpa_cache)
        self.cpa_handler()
        return c_cpa, c_funcs

    def _start(
        self,
        *,
        bounds: float | int | Tuple[float, float] | Tuple[Tuple[float, float]] = 1.0,
        input_size: tuple = (2,),
    ) -> int:
        # Initialize the input shape and dimension.
        self.net.origin_size = torch.Size(input_size)
        dim = self.net.origin_size.numel()
        # Construct the input-box boundary constraints.
        p_funcs, p_region, p_inner_point, self.o_bounds = generate_bound_regions(bounds, dim)
        # Initialize the region set.
        cpa_set, cpa = CPASet(), CPAFunc(p_funcs, p_region, p_inner_point)
        cpa_set.register(cpa)
        # Enumerate the network's activation-induced regions.
        self.logger.info("Start Get region number.")
        start = time.time()
        counts = self._get_counts(cpa_set)
        self.logger.info(f"CPAFunc counts: {counts}. Times: {time.time()-start}s")
        return counts

    @log_time("CPAFunc counts")
    def _get_counts(self, cpa_set: CPASet) -> int:
        if self.workers == 1:
            return self._single_get_counts(cpa_set)

        # For non-spawn workers, temporarily disable custom tensor reducers.
        # Sharing many tensors can exhaust file descriptors or shared memory.
        _save_reducers = {}
        if not self._check_spawn():
            _save_reducers = ForkingPickler._extra_reducers
            ForkingPickler._extra_reducers = {}

        self.pool = mp.Pool(processes=self.workers)
        counts = self._multiprocess_get_counts(cpa_set)

        if not self._check_spawn():
            ForkingPickler._extra_reducers = _save_reducers
        return counts

    def _single_get_counts(self, cpa_set: CPASet) -> int:
        counts: int = 0
        for cpa in cpa_set:
            c_funcs = self._functions(cpa.point, cpa.depth)
            self.logger.info(f"Start to get regions. Depth: {cpa.depth+1}, ")
            # Enumerate child regions and count those at the final depth.
            count, c_cpa_set, cpa_cache = self._handler_region(cpa, c_funcs, self.cpa_handler.cpa_cache_factory())
            counts += count
            self.cpa_handler.extend(cpa_cache)
            cpa_set.extend(c_cpa_set)
        self.cpa_handler()
        return counts

    def _work(self, cpa: CPAFunc, c_funcs: torch.Tensor, factory: CPACacheFactory):
        self.logger.info(f"Start to get regions. Depth: {cpa.depth+1}, Process-PID: {os.getpid()}. ")
        print(f"Start to get regions. Depth: {cpa.depth+1}, Process-PID: {os.getpid()}. ")
        return self._handler_region(cpa, c_funcs, factory)

    def _multiprocess_get_counts(self, cpa_set: CPASet) -> int:
        """Parallelize parent-region searches after the first layer.

        The first layer has a single parent; its child searches are dispatched
        separately by ``_search_regions``.
        """
        counts: int = 0

        def callback(args) -> None:
            nonlocal counts
            count, c_cpa_set, cpa_cache = args
            counts += count
            self.cpa_handler.extend(cpa_cache)
            cpa_set.extend(c_cpa_set)

        factory = self.cpa_handler.cpa_cache_factory()
        results: List[AsyncResult] = []
        for cpa in cpa_set:
            # Compute affine coefficients in the main process to keep CUDA
            # execution and tensor transfers out of the worker processes.
            c_funcs = self._functions(cpa.point, cpa.depth)

            if cpa.depth == 0:
                # The first layer has only the input box as its parent.
                # Parallelize its child searches inside _search_regions.
                self.logger.info(f"Start to get regions. Depth: {cpa.depth+1}.")
                args = self._handler_region(cpa, c_funcs, factory)
                callback(args)
                continue

            # Search this parent's child regions in a worker process.
            res = self.pool.apply_async(
                func=self._work,
                args=(cpa, c_funcs, factory),
                callback=callback,
                error_callback=err_callback,
            )
            # Discard completed result handles.
            results = [r for r in results if not r.ready()]
            results.append(res)
            if len(cpa_set) != 0:
                continue
            for res in results:
                res.wait()
                if len(cpa_set) > 0:
                    break

        results.clear()
        self.pool.close()
        self.pool.join()
        self.cpa_handler()
        return counts

    def _functions(self, x: torch.Tensor, depth: int):
        # Extract the layer's affine functions in the original input coordinates.
        functions = self._net_2_cpa(x, depth)
        # Rescale small coefficients without changing the represented hyperplanes.
        return self._scale_functions(functions)

    def _net_2_cpa(self, x: torch.Tensor, depth: int) -> torch.Tensor:
        x = x.float().to(self.device)
        x = x.reshape(*self.net.origin_size).unsqueeze(dim=0)
        with torch.no_grad():
            out: Tensor = self.net.forward_layer(x, depth=depth)
            # Weight shape: (1, *output_shape, *input_shape); bias shape: (1, *output_shape).
            weight_graph, bias_graph = out.weight_graph, out.bias_graph
            # Flatten weights to (number_of_outputs, number_of_inputs).
            weight_graph = weight_graph.reshape(-1, x.size()[1:].numel())
            # Flatten biases to (number_of_outputs, 1).
            bias_graph = bias_graph.reshape(-1, 1)
            # Each returned row contains the input coefficients followed by the bias.
        return torch.cat([weight_graph, bias_graph], dim=1).cpu()

    def _scale_functions(self, functions: torch.Tensor) -> torch.Tensor:
        def _scale_function(function: torch.Tensor):
            v, _ = function.abs().max(dim=0)
            if v == 0:
                return function
            while v < 1:
                function *= 10
                v *= 10
            return function

        for i in range(functions.shape[0]):
            functions[i] = _scale_function(functions[i])
        return functions

    @log_time("Handler region", 2, logging=False)
    def _handler_region(
        self,
        p_cpa: CPAFunc,
        c_funcs: torch.Tensor,
        factory: CPACacheFactory,
    ):
        """Enumerate the child regions of one parent region."""
        c_cpa_set, cpa_cache = CPASet(), factory.cpa_cache()
        intersect_funcs, inner_points = self._find_intersect(p_cpa, c_funcs)
        if intersect_funcs is None:
            c_cpa = CPAFunc(p_cpa.funcs, p_cpa.region, p_cpa.point, p_cpa.depth + 1)
            n_regions = 1
            counts = self._nn_region_counts(c_cpa, c_cpa_set.register, cpa_cache.cpa)
        else:
            counts, n_regions = self._search_regions(
                p_cpa=p_cpa,
                intersect_funcs=intersect_funcs,
                inner_points=inner_points,
                c_cpa_set=c_cpa_set,
                cpa_cache=cpa_cache,
            )
        # Record the parent constraints, layer functions, intersections, and child count.
        cpa_cache.hyperplane(p_cpa, c_funcs, intersect_funcs, n_regions)
        return counts, c_cpa_set, cpa_cache

    def _search_regions(
        self,
        p_cpa: CPAFunc,
        intersect_funcs: torch.Tensor,
        inner_points: torch.Tensor,
        c_cpa_set: CPASet,
        cpa_cache: CPACache,
    ):

        counts, n_regions = 0, 0
        inner_points: torch.Tensor = torch.cat((inner_points, p_cpa.point.reshape(1, -1)))
        c_regions = get_regions(inner_points, intersect_funcs)
        # Seed the queue with candidate sign patterns.
        layer_regions = WapperRegion(c_regions)

        def callback(args):
            nonlocal counts, n_regions
            c_cpa, filter_region, neighbor_regions = args
            if c_cpa is None:
                return
            # Record the region's boundary-sign filter to avoid duplicate counts.
            if not layer_regions.update_filter(filter_region):
                return
            # Queue candidate neighboring sign patterns.
            layer_regions.extend(neighbor_regions)
            # Count accepted children of this parent region.
            n_regions += 1
            # Count a final region or queue the child for the next depth.
            counts += self._nn_region_counts(c_cpa, c_cpa_set.register, cpa_cache.cpa)

        # Process children sequentially with one worker or after the first layer.
        # At later layers, multiprocessing operates across parent regions instead.
        if self.workers == 1 or p_cpa.depth != 0:
            for c_region in layer_regions:
                args = self._optimize_child_region(p_cpa, intersect_funcs, c_region)
                callback(args)
                layer_regions.down()
            return counts, n_regions

        # Parallelize candidate child searches for the first layer.
        results: List[AsyncResult] = []
        while len(layer_regions) != 0:
            i = 0
            for c_region in layer_regions:
                res = self.pool.apply_async(
                    func=self._optimize_child_region,
                    args=(p_cpa, intersect_funcs, c_region),
                    callback=callback,
                    error_callback=err_callback,
                )
                results.append(res)
                i += 1
                if i == self.workers:
                    break
            # Wait for this batch of child searches to finish.
            for res in results:
                res.wait()
            layer_regions.down()
            results.clear()

        return counts, n_regions

    @log_time("Find intersect", 2, logging=False)
    def _find_intersect(self, p_cpa: CPAFunc, funcs: torch.Tensor) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
        """Find the hyperplanes intersecting the region."""
        intersect_idx, points_idx = [], []
        optim_funcs, optim_x = funcs.numpy(), p_cpa.point.double().numpy()
        pn_funcs = (p_cpa.region.view(-1, 1) * p_cpa.funcs).numpy()
        p_points, w_s = find_projection(p_cpa.point, funcs)
        for i in range(funcs.size(0)):
            if not check_point(p_points[i], w_s[i]):
                continue
            if vertify(p_points[i], p_cpa.funcs, p_cpa.region):
                points_idx.append(i)
                intersect_idx.append(i)
                continue
            if lineprog_intersect(optim_funcs[i], pn_funcs, optim_x, self.o_bounds):
                intersect_idx.append(i)
        if len(intersect_idx) == 0:
            return None, None
        intersect_funs, inner_points = funcs[intersect_idx], p_points[points_idx]
        return intersect_funs, inner_points

    def _optimize_child_region(self, p_cpa: CPAFunc, c_funcs: torch.Tensor, c_region: torch.Tensor) -> Tuple[CPAFunc, torch.Tensor, List[torch.Tensor]]:
        """Validate a candidate region and find its boundary constraints.

        Return the region, its duplicate-detection filter, and neighboring sign
        patterns to check next.
        """
        funcs, region = torch.cat([c_funcs, p_cpa.funcs], dim=0), torch.cat([c_region, p_cpa.region], dim=0)
        # Orient each affine constraint so that a @ x + b >= 0 inside the region.
        constraint_funcs = region.view(-1, 1) * funcs
        # Require a positive-radius inscribed ball and obtain its center.
        c_inner_point: torch.Tensor | None = self._find_region_inner_point(constraint_funcs)
        if c_inner_point is None:
            return None, None, None
        # Remove redundant constraints and generate candidate neighboring sign patterns.
        c_edge_funcs, c_edge_region, filter_region, neighbor_regions = self._optimize_region(funcs, region, constraint_funcs, c_region, c_inner_point)
        c_cpa = CPAFunc(c_edge_funcs, c_edge_region, c_inner_point, p_cpa.depth + 1)
        return c_cpa, filter_region, neighbor_regions

    @log_time("Find inner point", 2, False)
    def _find_region_inner_point(self, functions: torch.Tensor) -> torch.Tensor:
        """Return a Chebyshev center only when the inscribed radius is positive.

        For constraints A @ x + B >= 0, minimize -r subject to
        -A @ x + r * ||A_i|| <= B. Reject unsuccessful LPs and nonpositive radii.
        """
        funcs = functions.numpy()
        x, _, success = cheby_ball(funcs)
        if not success or x is None:
            return None
        return torch.from_numpy(x).float()

    @log_time("Optimize region", 2, False)
    def _optimize_region(
        self,
        funcs: torch.Tensor,
        region: torch.Tensor,
        constraint_funcs: torch.Tensor,
        c_region: torch.Tensor,
        c_inner_point: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Select boundary constraints, build a duplicate filter, and propose neighbors."""
        neighbor_regions: List[torch.Tensor] = list()
        filter_region = torch.zeros_like(c_region).type(torch.int8)

        optim_funcs, optim_x = constraint_funcs.numpy(), c_inner_point.double().numpy()
        p_points, _ = find_projection(c_inner_point, funcs)
        c_edge_idx, redundant_idx = [], []
        # TODO: Reduce the cost of these per-constraint intersection checks.
        for i in range(optim_funcs.shape[0]):
            # First check whether the projected point lies in the region closure.
            if not vertify(p_points[i], funcs, region):
                # Otherwise test intersection after removing this constraint.
                pn_funcs = np.delete(optim_funcs, [i, *redundant_idx], axis=0)
                success = lineprog_intersect(optim_funcs[i], pn_funcs, optim_x, self.o_bounds)
                if not success:
                    redundant_idx.append(i)
                    continue
            c_edge_idx.append(i)
        for i in c_edge_idx:
            if i >= c_region.shape[0]:
                continue
            # Propose a neighbor by flipping one current-layer constraint sign.
            neighbor_region = c_region.clone()
            neighbor_region[i] = -c_region[i]
            neighbor_regions.append(neighbor_region)
            # Retain boundary signs in the duplicate-detection filter.
            filter_region[i] = c_region[i]
        c_edge_funcs, c_edge_region = funcs[c_edge_idx], region[c_edge_idx]
        return c_edge_funcs, c_edge_region, filter_region, neighbor_regions

    def _nn_region_counts(self, cpa: CPAFunc, set_register: Callable[[CPAFunc], None], cpa_register: Callable[[CPAFunc], bool]) -> int:
        # Record and count regions that have reached the final analyzed depth.
        if cpa_register(cpa):
            return 1
        # Otherwise, enqueue this region as a parent for the next depth.
        set_register(cpa)
        return 0
