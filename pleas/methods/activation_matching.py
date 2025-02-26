import torch
import tqdm.auto as tqdm

from collections.abc import Collection

from torch.nn import Module
from torch.utils.data import DataLoader

from pleas.core.utils import PermutationSpec, Permutation, Axis
from copy import copy
from pleas.core.solvers import scipy_solve_lsa


def cross_features_inner_product(x, y, a: int):
    """
    Compute the inner product between features across a specific axis.
    
    Args:
        x (torch.Tensor): First tensor
        y (torch.Tensor): Second tensor
        a (int): Axis along which to compute inner products
        
    Returns:
        torch.Tensor: Matrix of inner products with shape (x.shape[a], y.shape[a])
    """
    x = torch.movedim(x, a, 0).reshape(x.shape[a], -1)
    y = torch.movedim(y, a, 0).reshape(y.shape[a], -1)
    return x @ y.T


def cross_features_cdist(x, y, a: int):
    """
    Compute the negative Euclidean distance between features across a specific axis.
    Used as a similarity metric for permutation matching.
    
    Args:
        x (torch.Tensor): First tensor
        y (torch.Tensor): Second tensor
        a (int): Axis along which to compute distances
        
    Returns:
        torch.Tensor: Matrix of negative distances with shape (x.shape[a], y.shape[a])
    """
    x = torch.movedim(x, a, 0).reshape(x.shape[a], -1)
    y = torch.movedim(y, a, 0).reshape(y.shape[a], -1)
    return -torch.cdist(x[None], y[None])[0]


def build_cross_module(
    model1: torch.nn.Module,
    model2: torch.nn.Module,
    axes: Collection[Axis],
    cross_features,
):
    """
    Build a module that computes cross-feature similarity matrices between
    activations of two models.
    
    Args:
        model1 (torch.nn.Module): First model
        model2 (torch.nn.Module): Second model
        axes (Collection[Axis]): Collection of axes to compute similarities for
        cross_features (callable): Function to compute cross-feature similarities
        
    Returns:
        torch.nn.Module: A module that computes cross-feature similarities
    """
    gm = torch.fx.symbolic_trace(model1)
    nodes = [n for n in gm.graph.nodes]
    g = torch.fx.graph.Graph()

    all_features = {}
    for ax in axes:
        all_features.setdefault(ax.key, set()).add(ax.axis)

    value_remap_a = {nodes[0]: g.node_copy(nodes[0])}
    value_remap_b = copy(value_remap_a)
    cross = {}

    for node in nodes[1:-1]:
        na, nb = copy(node), copy(node)
        if na.op in ("call_module", "get_attr"):
            assert nb.op == na.op
            na.target = f"0.{na.target}"
            nb.target = f"1.{nb.target}"

        nac = value_remap_a[node] = g.node_copy(na, lambda n: value_remap_a[n])
        nbc = value_remap_b[node] = g.node_copy(nb, lambda n: value_remap_b[n])

        if node.name in all_features:
            for a in all_features[node.name]:
                cross[node.name, a] = g.call_function(cross_features, (nac, nbc, a))

    g.output(
        ([value_remap_a[nodes[-1].args[0]], value_remap_b[nodes[-1].args[0]]], cross)
    )
    gmp = torch.fx.graph_module.GraphModule(torch.nn.ModuleList([model1, model2]), g)
    gmp.graph.lint()

    return gmp


def compute_matching_costs(
    spec: PermutationSpec, gm_cross: Module, dataloader: DataLoader, num_batches
):
    """
    Compute matching costs between networks using activations on data.
    
    Args:
        spec (PermutationSpec): Permutation specification for the networks
        gm_cross (Module): Module that computes cross-feature similarities
        dataloader (DataLoader): DataLoader providing input batches
        num_batches (int): Number of batches to process
        
    Returns:
        dict: Dictionary mapping axes to cost matrices
    """
    cross_sum = {}
    with torch.inference_mode():
        for (x, _), _ in zip(dataloader, tqdm.trange(num_batches)):
            x = x.cuda()
            out, cross = gm_cross(x)
            for ka, v in cross.items():
                if ka not in cross_sum:
                    cross_sum[Axis(*ka)] = v
                else:
                    cross_sum[Axis(*ka)].add_(v)

    costs = {
        next(kax for kax in spec.keys() if kax in pg.state): sum(
            cross_sum[nax] for nax in pg.node if nax in cross_sum
        )
        for pg in spec.values()
    }

    return costs   


def activation_matching(
    spec: PermutationSpec,
    model1: Module,
    model2: Module,
    dataloader: DataLoader,
    num_batches=1000,
    cross_features=cross_features_cdist,
    lsa_solver=scipy_solve_lsa,
    output_costs=False,
) -> Permutation:
    """
    Permute one network to match the activations of another.
    
    This function finds permutations that align the activations of model1 and model2
    on the same inputs, making their internal representations more similar.
    
    Args:
        spec (PermutationSpec): Specification of permutable axes in the networks
        model1 (Module): First model
        model2 (Module): Second model to be permuted to match model1
        dataloader (DataLoader): DataLoader providing input batches
        num_batches (int, optional): Number of batches to process. Defaults to 1000.
        cross_features (callable, optional): Function to compute cross-feature similarities. 
                                         Defaults to cross_features_cdist.
        lsa_solver (callable, optional): Function to solve linear sum assignment problem. 
                                    Defaults to scipy_solve_lsa.
        output_costs (bool, optional): Whether to return cost matrices. Defaults to False.
        
    Returns:
        Permutation or tuple: Permutation mapping, or (permutation, costs) if output_costs=True
    """
    axes = [ax for pg in spec.values() for ax in pg.node]
    gm_cross = build_cross_module(model1, model2, axes, cross_features)
    costs = compute_matching_costs(spec, gm_cross, dataloader, num_batches)
    perm = {k: lsa_solver(v) for k, v in tqdm.tqdm(costs.items())}
    if output_costs:
        return perm, costs

    return perm


def check_multi_axis(spec):
    """
    Check if a permutation specification has multiple axes for the same key.
    
    Args:
        spec (PermutationSpec): Permutation specification
        
    Returns:
        bool: True if there are multiple axes for the same key, False otherwise
    """
    counter = set()
    for pg in spec.values():
        for ax in pg.node:
            if ax.key in counter:
                return True
    return False