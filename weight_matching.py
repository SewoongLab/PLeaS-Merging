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
    lerp,
    lslerp,
    make_identity_perm,
    slerp,
    tree_mean,
    
)
from compiler import PermutationProp
from lsa_solvers import scipy_solve_lsa
from activation_matching import cross_features_inner_product


# TODO - rewrite this in the activation matching style
# Might actually be straightforward, we just rewrite the original formulation
# to take into account the partial merges
# Might have to figure out how to deal with axes

def weight_matching(
    spec: PermutationSpec,
    state_as: Union[StateDict, Sequence[StateDict]],
    state_bs: Union[StateDict, Sequence[StateDict]],
    max_iter=100,
    init_perm=None,
    inplace=False,
    skip_suffixes=("running_mean", "running_var"),
    skip_missing=True,
    lsa_solver=scipy_solve_lsa,
    cross_weights=cross_features_inner_product,
    verbose=True,
    seed=0,
    return_costs=False,
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
    all_costs = {}
    rng = torch.Generator()
    rng.manual_seed(seed)

    with torch.no_grad():
        for iteration in range(max_iter):
            progress = False
            for p_ix in torch.randperm(len(perm_names), generator=rng):
                p = perm_names[p_ix]
                pg = spec[p]
                n, axes = pg.size, pg.state
                A = torch.zeros(n, n, device=device)
                for ax in axes:
                    if ax.key.endswith(skip_suffixes):
                        continue
                    for state_a, state_b in zip(state_as, state_bs):
                        if skip_missing and not (
                            ax.key in state_a and ax.key in state_b
                        ):
                            continue
                        w_a, w_b = state_a[ax.key], state_b[ax.key]
                        A.add_(cross_weights(w_a, w_b, ax.axis))

                assert A.norm() > 0
                newP = lsa_solver(A)

                oldL, newL = A.diag().sum(), A[torch.arange(n), newP].sum()
                progress = progress or newL > oldL + 1e-12
                if verbose:
                    print(f"{iteration}/{p.key}:{p.axis}: {newL - oldL}")

                perm[p] = perm[p][newP]
                all_costs[p] = A  # [torch.arange(n), newP]
                for state_b in state_bs:
                    apply_perm({p: newP}, spec, state_b, inplace=True)

            if not progress:
                break
        
        if return_costs:
            return perm, all_costs
        return perm









def weight_multi_matching(
    spec: PermutationSpec,
    states: Union[Sequence[StateDict], Sequence[Sequence[StateDict]]],
        max_iter=100,
        inplace=False
) -> list[Permutation]:
    n = len(states)
    I = make_identity_perm(spec)

    if not inplace:
        states = [copy(state) for state in states]

    state_cycle = deque((state, make_identity_perm(spec)) for state in states)

    for round in range(max_iter):
        progress = False
        for i in range(n):
            print(f"Permuting network {i}")
            s, perm = state_cycle.popleft()
            # TODO: I believe this should work with sequences of
            # states but untested
            s_rest = tree_mean(*(state for state, _ in state_cycle))

            new_perm = weight_matching(spec, s_rest, s, inplace=True, max_iter=1)
            perm = {k: p[new_perm[k]] for k, p in perm.items()}

            if not perm_eq(new_perm, I):
                progress = True

            state_cycle.append((s, perm))

        if not progress:
            break

    return [perm for _, perm in state_cycle]


