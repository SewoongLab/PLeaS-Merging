from copy import copy, deepcopy
from typing import Union

import gurobipy as gp
import torch
import math
import random

from gurobipy import GRB
from torch.nn import Module

from pleas.core.solvers import scipy_solve_lsa
from pleas.core.utils import Axis, Permutation, PermutationSpec, set_attr
from collections.abc import Sequence

from pleas.core.utils import (
    Permutation,
    PermutationSpec,
    StateDict,
    apply_perm,
    make_identity_perm,  
)
from pleas.core.solvers import scipy_solve_lsa
from pleas.methods.activation_matching import cross_features_inner_product


Ratios = Union[float, dict[Axis, float]]


def expand_ratios(spec: PermutationSpec, ratios: Ratios) -> dict[Axis, float]:
    """
    Expand a ratio or dictionary of ratios to cover all axes in a specification.
    
    Args:
        spec (PermutationSpec): Permutation specification
        ratios (Ratios): A single float ratio to apply to all axes, or a dict mapping axes to ratios
        
    Returns:
        dict[Axis, float]: Dictionary mapping all axes in spec to their ratios
    """
    if not isinstance(ratios, dict):
        return {ax: ratios for ax in spec}

    return ratios


def get_blocks(
    spec: PermutationSpec,
    perm: Permutation,
    costs: dict[Axis, torch.Tensor],
    ratios: Ratios,
    lsa_solver=scipy_solve_lsa,
) -> dict[Axis, tuple[torch.Tensor, ...]]:
    """
    Determine which units/features to merge versus keep separate for each axis.
    
    For each permutation group in the specification, this function identifies which 
    units should be merged and which should remain separate in each model based on
    the matching quality and specified ratios.
    
    Args:
        spec (PermutationSpec): Permutation specification
        perm (Permutation): Permutation mapping
        costs (dict[Axis, torch.Tensor]): Cost matrices for each permutation axis
        ratios (Ratios): Ratios of units to merge versus keep separate
        lsa_solver (callable, optional): Function to solve linear sum assignment. Defaults to scipy_solve_lsa.
        
    Returns:
        dict[Axis, tuple[torch.Tensor, ...]]: Dictionary mapping axes to tuples of 
            (merged_indices_model1, merged_indices_model2, separate_indices_model1, separate_indices_model2)
    """
    ratios = expand_ratios(spec, ratios)
    blocks = {}
    # Just tells which ones to keep merged and which ones to keep separate
    for axis, P in perm.items():
        print(axis)
        if abs(ratios[axis] - 1.0) < 1e-3:
            # Special case, makes life easier for me later
            P = torch.arange(len(P))
        C = costs[axis]
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