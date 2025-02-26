import torch
import tqdm.auto as tqdm

from collections.abc import Collection

from torch.nn import Module
from torch.utils.data import DataLoader

from utils import PermutationSpec, Permutation, Axis
from copy import copy
from lsa_solvers import scipy_solve_lsa


def cross_features_inner_product(x, y, a: int):
    x = torch.movedim(x, a, 0).reshape(x.shape[a], -1)
    y = torch.movedim(y, a, 0).reshape(y.shape[a], -1)
    return x @ y.T


def cross_features_cdist(x, y, a: int):
    x = torch.movedim(x, a, 0).reshape(x.shape[a], -1)
    y = torch.movedim(y, a, 0).reshape(y.shape[a], -1)
    return -torch.cdist(x[None], y[None])[0]


def build_cross_module(
    model1: torch.nn.Module,
    model2: torch.nn.Module,
    axes: Collection[Axis],
    cross_features,
):
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


## CODE BELOW IS FOR PROGRESSIVE MERGING ##



def get_cross_module_progressive_merge(merged_model: Module, spec: PermutationSpec, merged_layers: list[str]):
    merged_model.cuda()
    g_merged = torch.fx.symbolic_trace(merged_model)
    axes = [ax for pg in spec.values() for ax in pg.node]

    # To do that, we first create a set of nodes which have been merged
    merged_nodes = set()
    # print("Merged layers", merged_layers)
    for k, v in spec.items():
        # print(f"Checking {k}")
        if k.key.split(":")[0] in merged_layers:
            # print(f"{k} already merged")
            for i in v.node:
                # print(f"{i.key} adding to merged nodes")
                merged_nodes.add(i.key.split(":")[0])
    print("Merged nodes", merged_nodes)
    # Next, we go through the merged model in a manner similar to the activation matching algorithm
    g_merged = torch.fx.symbolic_trace(merged_model)
    nodes = [n for n in g_merged.graph.nodes]
    g = torch.fx.graph.Graph()

    all_features = {}
    for ax in axes:
        all_features.setdefault(ax.key, set()).add(ax.axis)

    value_remap_a = {nodes[0]: g.node_copy(nodes[0])}
    value_remap_b = copy(value_remap_a)
    cross = {}

    for node in nodes[1:-1]:
        nmerged = copy(node)
        nmergedc = value_remap_a[node] = g.node_copy(nmerged, lambda n: value_remap_a[n])
        if node.name in all_features:
            if node.name not in merged_nodes:
                for a in all_features[node.name]:
                    cross[node.name, a] = g.call_function(cross_features_cdist_halved, (nmergedc, a))
            else:
                for a in all_features[node.name]:
                    cross[node.name, a] = g.call_function(cross_features_constant, (nmergedc, a))  # This is where we change depending on whether the node is already merged or not

    g.output(
        ([value_remap_a[nodes[-1].args[0]]], cross)
    )
    gmp = torch.fx.graph_module.GraphModule(merged_model, g)
    gmp.graph.lint()

    return gmp, merged_nodes


def cross_features_cdist_halved(x, a: int):
    x = torch.movedim(x, a, 0).reshape(x.shape[a], -1)
    x_top_half = x[: x.shape[0] // 2]  # Check if this is correct, else modify the get_blocks function to make this correct
    x_bottom_half = x[x.shape[0] // 2 :]
    
    return -torch.cdist(x_top_half[None], x_bottom_half[None])[0]

def cross_features_constant(x, a: int):
    x = torch.movedim(x, a, 0).reshape(x.shape[a], -1)

    return torch.ones(x.shape[0], x.shape[0])

def compute_progressive_matching_costs(
    spec, gm_cross, dataloader, num_batches
):
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
    perm = {k: scipy_solve_lsa(v).cuda() for k, v in (costs.items())}    
    return perm, costs




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
    axes = [ax for pg in spec.values() for ax in pg.node]
    gm_cross = build_cross_module(model1, model2, axes, cross_features)
    costs = compute_matching_costs(spec, gm_cross, dataloader, num_batches)
    perm = {k: lsa_solver(v) for k, v in tqdm.tqdm(costs.items())}
    if output_costs:
        return perm, costs

    return perm


def check_multi_axis(spec):
    counter = set()
    for pg in spec.values():
        for ax in pg.node:
            if ax.key in counter:
                return True
    return False
