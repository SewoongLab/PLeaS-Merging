from copy import copy, deepcopy
from typing import Union

import gurobipy as gp
import torch
import math
import random

from gurobipy import GRB
from torch.nn import Module


from activation_matching import build_cross_module, compute_matching_costs, activation_matching, get_cross_module_progressive_merge, compute_progressive_matching_costs
from lsa_solvers import scipy_solve_lsa
from utils import Axis, Permutation, PermutationSpec, set_attr
from torch.utils.data import DataLoader

Ratios = Union[float, dict[Axis, float]]


def expand_ratios(spec: PermutationSpec, ratios: Ratios) -> dict[Axis, float]:
    if not isinstance(ratios, dict):
        return {ax: ratios for ax in spec}

    return ratios


def get_blocks(
    spec: PermutationSpec,
    perm: Permutation,
    costs: dict[Axis, torch.Tensor],
    ratios: Ratios,
    zero_augmented: bool = False,
    lsa_solver=scipy_solve_lsa,
    logging=False,
) -> dict[Axis, tuple[torch.Tensor, ...]]:
    ratios = expand_ratios(spec, ratios)
    blocks = {}
    # Just tells which ones to keep merged and which ones to keep separate
    for axis, P in perm.items():
        print(axis)
        if abs(ratios[axis] - 1.0) < 1e-3:
            # Special case, makes life easier for me later
            P = torch.arange(len(P))
        C = costs[axis]
        if zero_augmented:
            # # Zero augmented activation matching
            # Algorithm just adds a bunch of nodes with zero edges to the graph
            # This forces the LSA matching to pick up the actual top-k edges to merge
            n, _ = C.shape
            dim_to_add = int(n * (ratios[axis]))
            aug_C = torch.zeros((n + dim_to_add, n + dim_to_add)).to(C.device)
            aug_C[:n, :n] = -C
            aug_C[n:, n:] = 100000.
            P_ = lsa_solver(aug_C, maximize=False)
            P_ = P_[:n]  # Since we only care for the first n elements
            P_ = P_.to(C.device)
            Q_ = torch.arange(len(P_))
            Q_ = Q_.to(C.device)
            c_matched = aug_C[Q_, P_]
            mask = c_matched > torch.quantile(c_matched, ratios[axis])
            mask = mask.to(C.device)  # Mask contains the merged units
            B_merged = P_[mask]
            B_separate = torch.arange(n)
            B_separate = torch.tensor([x for x in B_separate if x not in B_merged])
            A_merged = Q_[mask]
            A_separate = Q_[~mask]
            blocks[axis] = A_merged.cuda(), B_merged.cuda(), A_separate.cuda(), B_separate.cuda(), 
            
            if logging:
                P = P.to(C.device)
                Q = torch.arange(len(P)).to(P.device)
                c__matched = C[Q, P]
                mask_ = c__matched > torch.quantile(c__matched, ratios[axis])
                mask_ = mask_.to(C.device)

                A_seperate_intersection = [x for x in A_separate if x in Q[~mask_]]
                A_separate_union = torch.unique(torch.cat((Q[~mask_], A_separate)))
                B_separate_intersection = [x for x in B_separate if x in P[~mask_]]
                B_separate_union = torch.unique(torch.cat((P[~mask_].cuda(), B_separate.cuda())))
                total_cost_new = c_matched[mask].sum()
                total_cost_orig = -c__matched[mask_].sum()
                print(f"Axis - {axis}, Non Merge ratio - {ratios[axis]}, New cost - {total_cost_new.item()}, Orig cost - {total_cost_orig.item()}, Cost ratio - {total_cost_new.item()/(total_cost_orig.item() + 1e-15)},IoU for A separate - {len(A_seperate_intersection)/ (len(A_separate_union)+1e-15)}, IoU for B separate - {len(B_separate_intersection)/ (len(B_separate_union)+1e-15)}")
        else:
            P = P.to(C.device)
            Q = torch.arange(len(P)).to(P)
            c_matched = C[Q, P]
            mask = c_matched >= torch.quantile(c_matched, ratios[axis])  # Should I change this back to strict inequality?
            # print(axis, len(P), len(Q), len(P[mask]), len(P[~mask]), len(Q[mask]), len(Q[~mask]), C.shape)
            # four cases: merged and seperate for Q and P resp.
            blocks[axis] = Q[mask].cuda(), P[mask].cuda(), Q[~mask].cuda(), P[~mask].cuda()
            # print(axis, len(Q[mask]), len(P[mask]), len(Q[~mask]), len(P[~mask]), ratios[axis], len(P))

    return blocks



def build_partial_merge_model(
    spec: PermutationSpec,
    model1: Module,
    model2: Module,
    blocks: dict[Axis, tuple[torch.Tensor, ...]],
) -> Module:

    # Takes two models along with a permutation spec and returns a module which is merged partially
    # Permutation spec is a dict with 'layer' -> PermutationGroup (having node and state, state contains things which are actually permuted together)
    blocks = copy(blocks)
    for axis, pg in spec.items():
        for ax in pg.state:
            blocks[ax] = blocks[axis]

    # blocks contained for each layer, which

    axes_by_tensor = {}
    for pg in spec.values():
        for ax in pg.state:
            axes_by_tensor.setdefault(ax.key, set()).add(ax.axis)

    D1, D2, D3 = model1.state_dict(), model2.state_dict(), {}
    for tensor_name, axes in axes_by_tensor.items():
        try:
            W1, W2 = D1[tensor_name], D2[tensor_name]
        except KeyError:
            print(f"Could not find - {tensor_name}")
            D3[tensor_name] = None
            continue
        assert len(axes) in {1, 2}

        if len(axes) == 1:
            ax = next(iter(axes))
            b1, b2, b1c, b2c = blocks[Axis(tensor_name, ax)]
            W1b = torch.index_select(W1, ax, b1)
            W1bc = torch.index_select(W1, ax, b1c)
            W2b = torch.index_select(W2, ax, b2)
            W2bc = torch.index_select(W2, ax, b2c)
            W3 = torch.cat(((W1b + W2b) / 2, W1bc, W2bc), ax)


        elif len(axes) == 2:
            # We assume 0 is output and 1 is input. (not tested)
            # This will NOT work if the axes are flipped.
            assert axes == {0, 1}

            # Inputs
            bi1, bi2, bi1c, bi2c = blocks[Axis(tensor_name, 1)]
            ni, mi = len(bi1), len(bi1c)
            si12, si1, si2 = (
                slice(0, ni),
                slice(ni, ni + mi),
                slice(ni + mi, ni + 2 * mi),
            )

            # Outputs
            bo1, bo2, bo1c, bo2c = blocks[Axis(tensor_name, 0)]
            no, mo = len(bo1), len(bo1c)
            so12, so1, so2 = (
                slice(0, no),
                slice(no, no + mo),
                slice(no + mo, no + 2 * mo),
            )

            # print(ni, mi, no, mo)
            # target block matrix
            W3 = torch.zeros((no + 2 * mo), (ni + 2 * mi), *W1.shape[2:])

            # 1 "merged to merged" block (i.e. the Git Re-Basin case)
            W3[so12, si12] = (W1[bo1][:, bi1] + W2[bo2][:, bi2]) / 2

            # 2 "seperate to seperate" blocks (run subsets of each network independently)
            W3[so1, si1] = W1[bo1c][:, bi1c]
            W3[so2, si2] = W2[bo2c][:, bi2c]

            # 2 "merged to seperate" blocks (substitute missing inputs with merged version)
            W3[so1, si12] = W1[bo1c][:, bi1]
            W3[so2, si12] = W2[bo2c][:, bi2]

            # 2 "seperate to merged" blocks (just merge the activations)
            W3[so12, si1] = W1[bo1][:, bi1c] / 2
            W3[so12, si2] = W2[bo2][:, bi2c] / 2

            # if

        D3[tensor_name] = W3

    model3 = deepcopy(model1).eval().cpu()
    for axis, param in D3.items():
        submodules = axis.split(".")
        param = torch.nn.Parameter(param, requires_grad=False)
        param.cpu()
        set_attr(model3, submodules, param)

    return model3


def partial_merge(
    spec: PermutationSpec,
    model1: Module,
    model2: Module,
    perm: Permutation,
    costs: dict[Axis, torch.Tensor],
    ratios: Ratios,
    zero_augmented: bool = False,
    return_blocks=False
):
    blocks = get_blocks(spec, perm, costs, ratios, zero_augmented)
    model3 = build_partial_merge_model(spec, model1, model2, blocks)
    if return_blocks:
        return model3, blocks
    return model3


def partial_merge_flops(
    spec: PermutationSpec, terms: list[tuple[int, Axis, ...]], ratios: Ratios
):
    ratios = expand_ratios(spec, ratios)

    total_flops = 0
    for coeff, *axes in terms:
        assert len(axes) in {1, 2}
        if len(axes) == 1:
            (axis,) = axes
            base_flops = coeff * spec[axis].size
            flops = base_flops * (1 + ratios[axis])

        elif len(axes) == 2:
            ax1, ax2 = axes
            s1, s2, r1, r2 = spec[ax1].size, spec[ax2].size, ratios[ax1], ratios[ax2]
            base_flops = coeff * s1 * s2
            flops = base_flops * ((1 + r1) * (1 + r2) - 2 * r1 * r2)

        total_flops += flops

    return total_flops


def qp_ratios(
    spec: PermutationSpec, terms, flops_budget: float, obj_weights: dict[Axis, float]
):
    try:
        m = gp.Model("partial_match_qp")
    except:
        print("Gurobi not available")
        return {key: random.random() for key in spec}
    m.setParam("OutputFlag", 0)
    m.setParam("NonConvex", 2)
    grb_ratios = {
        key: m.addVar(vtype=GRB.CONTINUOUS, name=str(key), lb=0, ub=1) for key in spec
    }
    m.update()

    m.addConstr(
        partial_merge_flops(spec, terms, grb_ratios)
        / partial_merge_flops(spec, terms, 0)
        <= flops_budget,
        name="flop_limit",
    )
    m.setObjective(
        sum(grb_ratios[key] * max(obj_weights[key], 1e-5)
            for key in spec), GRB.MAXIMIZE
    )
    m.optimize()
    grb_ratios = {k.key: max(min(float(v.X), 1), 0)
                  for k, v in zip(spec, m.getVars())}
    return grb_ratios


## CODE BELOW IS FOR PROGRESSIVE MERGING ##


def build_progressive_merge_model(spec: PermutationSpec, model1: Module, model2: Module, merged_model: Module, blocks: dict[Axis, tuple[torch.Tensor, ...]], merged_layers: list[str]) -> Module:
    # Takes two models along with a permutation spec and returns a module which is merged partially
    # Permutation spec is a dict with 'layer' -> PermutationGroup (having node and state, state contains things which are actually permuted together)
    blocks = copy(blocks)
    for axis, pg in spec.items():
        for ax in pg.state:
            blocks[ax] = blocks[axis]

    # blocks contained for each layer, which

    axes_by_tensor = {}
    for pg in spec.values():
        for ax in pg.state:
            axes_by_tensor.setdefault(ax.key, set()).add(ax.axis)

    D1, D2, D3 = model1.state_dict(), model2.state_dict(), {}
    DMerged = merged_model.state_dict()
    for tensor_name, axes in axes_by_tensor.items():
        W1, W2 = D1[tensor_name], D2[tensor_name]
        assert len(axes) in {1, 2}

        # Check if this was already merged
        already_merged = '_'.join(tensor_name.split('.')[:-1]) in merged_layers  # Converting from parameter name to node name
        print(tensor_name, already_merged)
        if already_merged:
            W3 = DMerged[tensor_name]
        else:
            if len(axes) == 1:
                ax = next(iter(axes))
                b1, b2, b1c, b2c = blocks[Axis(tensor_name, ax)]
                
                # I think this is incorrect, we may have to add something here
                WMerged = DMerged[tensor_name]
                # print(WMerged.shape, W1.shape, W2.shape, len(b1), len(b2), len(b1c), len(b2c))
                
                if 'fc.weight' in tensor_name:
                    # Another hack to ensure that last layer has sufficient output size
                    # Note that only for the last layer will the len of axes be 1
                    W1 = WMerged[:, :WMerged.shape[1]//2]
                    W2 = WMerged[:, WMerged.shape[1]//2:]
                else:
                    # This assumes that ax == 0
                    # For ax = 1, we need to chunk along the appropriate dimension
                    W1 = WMerged[:WMerged.shape[0]//2]
                    W2 = WMerged[WMerged.shape[0]//2:]
                
                W1b = torch.index_select(W1, ax, b1)
                W1bc = torch.index_select(W1, ax, b1c)
                W2b = torch.index_select(W2, ax, b2)
                W2bc = torch.index_select(W2, ax, b2c)
                W3 = torch.cat(((W1b + W2b) / 2, W1bc, W2bc), ax)

            elif len(axes) == 2:
                # We assume 0 is output and 1 is input. (not tested)
                # This will NOT work if the axes are flipped.
                assert axes == {0, 1}
                WMerged = DMerged[tensor_name]
                # Inputs
                bi1, bi2, bi1c, bi2c = blocks[Axis(tensor_name, 1)]
                ni, mi = len(bi1), len(bi1c)


                
                if mi == WMerged.shape[1]:  # If the input is already merged
                    W1 = WMerged[:WMerged.shape[0]//2, :]  # This works because output is not merged, else already_merged would be true
                    W2 = WMerged[WMerged.shape[0]//2:, :]

                    # This is a hack. What we want to say is that the input is already merged
                    # Hence, we need to share the same input for both the conv operations
                    # There should be a more principled way to do this via ratios passed to get_blocks
                    # But here, I just swap the separate and merged things, to make it work
                    bi1, bi1c = bi1c, bi1
                    bi2, bi2c = bi1, bi1c
                    ni, mi = mi, ni
                    
                else:
                    W1 = WMerged[:WMerged.shape[0]//2, :WMerged.shape[1]//2]
                    W2 = WMerged[WMerged.shape[0]//2:, WMerged.shape[1]//2:]
                    
                si12, si1, si2 = (
                    slice(0, ni),
                    slice(ni, ni + mi),
                    slice(ni + mi, ni + 2 * mi),
                )

                # Outputs
                bo1, bo2, bo1c, bo2c = blocks[Axis(tensor_name, 0)]
                no, mo = len(bo1), len(bo1c)
                so12, so1, so2 = (
                    slice(0, no),
                    slice(no, no + mo),
                    slice(no + mo, no + 2 * mo),
                )

                # target block matrix
                W3 = torch.zeros((no + 2 * mo), (ni + 2 * mi), *W1.shape[2:])

                # 1 "merged to merged" block (i.e. the Git Re-Basin case)
                W3[so12, si12] = (W1[bo1][:, bi1] + W2[bo2][:, bi2]) / 2

                # 2 "seperate to seperate" blocks (run subsets of each network independently)
                W3[so1, si1] = W1[bo1c][:, bi1c]
                W3[so2, si2] = W2[bo2c][:, bi2c]

                # 2 "merged to seperate" blocks (substitute missing inputs with merged version)
                W3[so1, si12] = W1[bo1c][:, bi1]
                W3[so2, si12] = W2[bo2c][:, bi2]

                # 2 "seperate to merged" blocks (just merge the activations)
                W3[so12, si1] = W1[bo1][:, bi1c] / 2
                W3[so12, si2] = W2[bo2][:, bi2c] / 2

        D3[tensor_name] = W3

    model3 = deepcopy(model1).eval().cpu()
    for axis, param in D3.items():
        submodules = axis.split(".")
        param = torch.nn.Parameter(param, requires_grad=False)
        param.cpu()
        set_attr(model3, submodules, param)

    return model3

def progressive_merge(spec: PermutationSpec, model1: Module, model2: Module, dataloader: DataLoader, num_batches: int, ratios: Ratios, return_all_costs: bool=False):
    # In this function, we progressively merge the two models. We first merge the two models upto layer k
    # then, we recompute the activations of the merged model upto layer k+1. We then merge the two models upto layer k+1
    # and so on

    # We first compute the matching costs between the two models
    all_costs = []
    perm, costs = activation_matching(
        spec,
        model1,
        model2,
        dataloader,
        num_batches,
        output_costs=True,
    )
    if return_all_costs:
        all_costs.append((perm, costs))
    layer_merging = 0
    init_ratios = copy(ratios)
    merged_layers = []
    for idx, k in enumerate(init_ratios):
        if idx > layer_merging:
            init_ratios[k] = 1.0
        else:
            merged_layers.append(k)
    merged_model = partial_merge(
        spec, model1, model2, perm, costs, init_ratios)

    merged_model.cuda()
    merged_model.train()
    # Reset batchnorm statistics
    for (x, _), _ in zip(dataloader, range(num_batches)):
        x = x.cuda()
        merged_model(x)
    merged_model.eval()
    
    # We want to do progressive merging for each block
    # i.e. we first merge block 1, then block 2 and so on
    # this is to handle residual connections
    # I would want to change this to do this in a more fine-grained manner later
    layer_groups = []
    curr_group = []
    curr_prefix = ""
    for k, v in spec.items():
        if len(curr_group) == 0:
            curr_group.append(k.key.split(":")[0])
            curr_prefix = k.key.split(".")[0]
        else:
            if k.key.split(".")[0] == curr_prefix:
                curr_group.append(k.key.split(":")[0])
            else:
                layer_groups.append(curr_group)
                curr_group = [k.key.split(":")[0]]
                curr_prefix = k.key.split(".")[0]
    layer_groups.append(curr_group)
    print(layer_groups)
    # We merge all layers in a layer group at once
    # Then proceed to the next group
     
    merged_layers = ['conv1.weight']
    for layer_merging in range(1, len(layer_groups)):
        init_ratios = copy(ratios)

        # for idx, k in enumerate(init_ratios):
        #     if idx > layer_merging:
        #         init_ratios[k] = 1.0
        #     elif idx < layer_merging:
        #         merged_layers.append(k) # .key.split(":")[0])
        #         init_ratios[k] = 0.0  # Doing this will ensure that we go from merged to seperate while building the model
        for idx, k in enumerate(init_ratios):
            if k.key.split(":")[0] not in layer_groups[layer_merging]:
                init_ratios[k] = 1.0 
        # We now compute the matching costs between the merged model and the original model
        gm_cross, merged_nodes = get_cross_module_progressive_merge(
            merged_model, spec, merged_layers)
        perm, costs = compute_progressive_matching_costs(
            spec, gm_cross, dataloader, num_batches)
        # print(init_ratios)
        if return_all_costs:
            all_costs.append((perm, costs))
        blocks = get_blocks(spec, perm, costs, init_ratios)
        model1.cpu()
        model2.cpu()
        merged_model.cpu()
        for b in blocks:
            blocks[b] = tuple(t.cpu() for t in blocks[b])
        merged_model = build_progressive_merge_model(
            spec, model1, model2, merged_model, blocks, merged_nodes)
        merged_model.cuda()
        merged_model.train()
        model1.cuda()
        model2.cuda()
        
        # for param in merged_model.named_parameters():
        #     print(param[0], param[1].shape)
        
        # Reset batchnorm statistics
        for m in merged_model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.reset_running_stats()
        for (x, _), _ in zip(dataloader, range(num_batches)):
            x = x.cuda()
            merged_model(x)
        merged_model.eval()
        merged_layers += layer_groups[layer_merging]

    if return_all_costs:
        return merged_model, all_costs
    return merged_model









import torch

from collections import deque
from collections.abc import Sequence
from copy import copy, deepcopy
from functools import wraps
from typing import Union

from utils import (
    Permutation,
    PermutationSpec,
    StateDict,
    apply_perm,
    # apply_perm_with_padding,
    lerp,
    lslerp,
    make_identity_perm,
    # remove_zero_block,
    slerp,
    tree_mean,
    
)
from compiler import PermutationProp
from lsa_solvers import scipy_solve_lsa
from activation_matching import cross_features_inner_product

# def remove_zero_block(tensor, axis, block_size):
#     # Find the indices where the sum along the specified axis is zero
#     if len(tensor.shape) == 1:
#         zero_indices = (tensor == 0).nonzero().squeeze()
#     else:
#         zero_indices = (torch.sum(tensor, dim=1-axis) == 0).nonzero().squeeze()

#     # Find the start and end of the zero block
#     if len(zero_indices) == 0:
#         return tensor
#     start_idx = zero_indices[0].item()
#     end_idx = start_idx + block_size

#     # Slice the tensor to exclude the block of zeros
#     if axis == 0:
#         return torch.cat((tensor[:start_idx], tensor[end_idx:]), axis=0)
#     elif axis == 1:
#         return torch.cat((tensor[:, :start_idx], tensor[:, end_idx:]), axis=1)



# I think we can simplify this function, because zero blocks will always be at the beginning or in a specific position given the ratio
# def remove_zero_block(tensor, axis, block_size):
#     # Handling 1D tensors directly
#     if tensor.dim() == 1:
#         nonzero_indices = torch.nonzero(tensor, as_tuple=True)[0]
#         return tensor[nonzero_indices]

#     # Original shape of the tensor
#     original_shape = tensor.shape
    
#     # Flatten the tensor around the specified axis
#     flattened_tensor = tensor.flatten(0, axis - 1).flatten(1) if axis > 0 else tensor.flatten()

#     # Find the indices where the sum along the axis is zero
#     zero_indices = (torch.sum(flattened_tensor, dim=1) == 0).nonzero(as_tuple=True)[0]
    
#     # Handling the case where there are no zero blocks of the specified size
#     if zero_indices.size(0) == 0 or zero_indices.size(0) < block_size:
#         return tensor

#     # Find the start index of the zero block
#     start_idx = zero_indices[0].item()

#     # Check for contiguous block of zeros of size 'block_size'
#     if not torch.all(zero_indices[:block_size] == torch.arange(start_idx, start_idx + block_size)):
#         return tensor

#     # Remove the block of zeros and reshape back
#     new_flattened_tensor = torch.cat((flattened_tensor[:start_idx], flattened_tensor[start_idx + block_size:]))
#     new_shape = list(original_shape)
#     new_shape[axis] -= block_size
#     return new_flattened_tensor.view(new_shape)

def remove_zero_block(tensor, axis, block_size, final_size, beginning=False):
    if tensor.shape[axis] < final_size:
        return tensor
    if axis == 0:
        if beginning:
            assert tensor[:block_size].sum() == 0., print(final_size, beginning, block_size)
            return tensor[block_size:]
        else:
            return torch.cat([tensor[:block_size], tensor[block_size*2:]], dim=0)
    elif axis == 1:
        if beginning:
            assert tensor[:, :block_size].sum() == 0.
            return tensor[:, block_size:]
        else:
            return torch.cat([tensor[:, :block_size], tensor[:, block_size*2:]], dim=1)


def apply_perm_with_padding(
    perm: Permutation,
    padding: Permutation,
    size: int,
    pad_ahead: bool,
    spec: PermutationSpec,
    state: Union[torch.nn.Module, StateDict],
    inplace=False,
    skip_missing=True,
):
    if isinstance(state, torch.nn.Module):
        assert inplace == True
        state.load_state_dict(
            apply_perm(perm, spec, state.state_dict(), inplace=inplace)
        )
        return state

    if not inplace:
        state = copy(state)

    for key, P in perm.items():
        if P is None:
            continue
        padding = padding[key]
        pg = spec[key]
        # assert P.shape == (pg.size,)
        # print(f"Applying {key} with {P.shape} and {padding.shape} to {size}, {pg.size}")
        for ax in pg.state:
            if skip_missing and ax.key not in state:
                continue
            
            weight = remove_zero_block(state[ax.key], ax.axis, len(padding), size,pad_ahead)
            if weight.shape[ax.axis] == size:
                state[ax.key] = torch.index_select(weight, ax.axis, P.to(weight.device))
            
            # select indices from the original weight
            permuted_weights = torch.index_select(weight, ax.axis, P.to(weight.device))
            separate_weights = torch.index_select(weight, ax.axis, padding.to(weight.device))
            if ax.axis == 0:
                padding_weights = torch.zeros((len(padding), *weight.shape[1:])).to(weight.device)
            else:
                padding_weights = torch.zeros((weight.shape[0], len(padding), *weight.shape[2:])).to(weight.device)
                
            if pad_ahead:
                final_weights = torch.cat((padding_weights, separate_weights, permuted_weights), ax.axis)
            else:
                final_weights = torch.cat((separate_weights, padding_weights, permuted_weights), ax.axis)
            if not torch.abs(final_weights).sum() > 0.:
                print(final_weights)
                print(weight)
                print(separate_weights, permuted_weights)
                print(ax, P, padding)
                print(weight.shape, final_weights.shape)
                print(torch.abs(final_weights).sum())
                raise ValueError("Zero norm")
            state[ax.key] = final_weights   
            
    return state


def weight_matching_partial(
    spec: PermutationSpec,
    state_as: Union[StateDict, Sequence[StateDict]],
    state_bs: Union[StateDict, Sequence[StateDict]],
    ratios:Ratios,
    max_iter=100,
    init_perm=None,
    inplace=False,
    skip_suffixes=("running_mean", "running_var"),
    skip_missing=True,
    lsa_solver=scipy_solve_lsa,
    cross_weights=cross_features_inner_product,
    verbose=True,
    seed=0,

) -> Permutation:
    if isinstance(state_as, dict):
        state_as = [state_as]
    if isinstance(state_bs, dict):
        state_bs = [state_bs]

    assert len(state_as) == len(state_bs)

    if not inplace:
        state_bs = [copy(state_b) for state_b in state_bs]

    perm = make_identity_perm(spec) if init_perm is None else deepcopy(init_perm)
    if init_perm is not None:
        for state_b in state_bs:
            apply_perm(init_perm, spec, state_b, inplace=True)

    perm_names = list(perm.keys())
    device = next(iter(state_as[0].values())).device

    rng = torch.Generator()
    rng.manual_seed(seed)

    # The algorithm is almost the same as weight matching
    # However, instead of just permuting the weights to get new weights
    # we also append 0s to the beginning of the weight or middle of the weight matrix
    # this is to signify that some units will be coming in as separate units from the other model 

    with torch.no_grad():
        for iteration in range(max_iter):
            progress = False
            for p_ix in torch.randperm(len(perm_names), generator=rng):
                p = perm_names[p_ix]  # Which weight to permute
                pg = spec[p]  # Permutation group of the weight
                ratio = ratios[p]  # Dimension of final weight
                n, axes = pg.size, pg.state  # dimension of permutation, and axes to permute along 
                repeat_size = int(n * (1-ratio))
                merge_size = int(n * ratio)
                if merge_size + repeat_size < n:
                    merge_size += 1
                final_dimension = 2 * repeat_size + merge_size
                A = torch.zeros(n, n, device=device)
                
                for ax in axes:
                    if ax.key.endswith(skip_suffixes):
                        continue
                    for state_a, state_b in zip(state_as, state_bs):
                        if skip_missing and not (
                            ax.key in state_a and ax.key in state_b
                        ):
                            continue
                        # Do some manipulations here to handle separate and merged cases 
                        # We need to remove the chunk of zero'd out rows or columns along ax.axis
                        # the len of these is n*ratio
                        # print(state_a[ax.key].shape, ax, ax.axis)
                        w_a, w_b = remove_zero_block(state_a[ax.key], ax.axis, repeat_size, final_dimension, False), remove_zero_block(state_b[ax.key], ax.axis, repeat_size, final_dimension, True)
                        # print(ax, w_a.shape, w_b.shape, ratio, n)
                        try:
                            A.add_(cross_weights(w_a, w_b, ax.axis))
                            if A.norm() == 0:
                                print("Zero norm")
                                print(merge_size, repeat_size, ratio, n)
                                print(w_a, w_b)
                        except Exception as e:
                            print(e)
                            print(w_a, w_b)
                            raise e
                assert A.norm() > 0
                newP = lsa_solver(A).to(A.device)

                # This also needs to be changed to get the top-k indices
                oldL, newL = A.diag().sum(), A[torch.arange(n), newP].sum()
                
                # Coding up the first iteration (when the size needs to be expanded)
                sums = A[torch.arange(n), newP]
                mask = sums > torch.quantile(sums, 1. - ratio)  # Check this
                mask = mask.to(A.device)
                if mask.sum().item() != merge_size:
                    if mask.sum().item() > merge_size:
                        mask[sums.argmin()] = False
                    else:
                        # Very bad solution, but I don't know what else to do!!!
                        for i in range(len(mask)):
                            if mask[i] == False:
                                mask[i] = True
                                if mask.sum().item() == merge_size:
                                    break
                # print(mask.sum().item(), merge_size, ratio, n, repeat_size, merge_size + repeat_size)
                model_a_perm = torch.arange(n).to(mask.device)[mask]
                model_b_perm = newP[mask]
                model_a_separate = torch.arange(n).to(mask.device)[~mask]
                model_b_separate = newP[~mask]
                
                assert len(model_a_perm) == len(model_b_perm) == merge_size, print(len(model_a_perm), len(model_b_perm), merge_size)
                assert len(model_a_separate) == len(model_b_separate) == repeat_size, print(len(model_a_separate), len(model_b_separate), repeat_size)
                
                progress = progress or newL > oldL + 1e-12
                if verbose:
                    print(f"{iteration}/{p.key}:{p.axis}: {newL - oldL}")

                # This also needs to be changed to get the top-k indices
                perm[p] = perm[p][newP.to(perm[p].device)]

                # Change this to add 0s at beginning, add similar loop for state_a
                for state_b in state_bs:
                    apply_perm_with_padding({p: model_b_perm}, {p: model_b_separate}, final_dimension, True, spec, state_b, inplace=True)
                for state_a in state_as:
                    apply_perm_with_padding({p: model_a_perm}, {p: model_a_separate}, final_dimension, False, spec, state_a, inplace=True)

            # Also beware of having correct magnitudes, we may have to multiply some units by 2
            if not progress:
                break

        return perm