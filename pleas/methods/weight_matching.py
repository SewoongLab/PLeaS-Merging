import torch

from collections import deque
from collections.abc import Sequence
from copy import copy, deepcopy
from functools import wraps
from typing import Union

from pleas.core.utils import (
    Permutation,
    PermutationSpec,
    StateDict,
    apply_perm,
    make_identity_perm,

    
)
from pleas.core.solvers import scipy_solve_lsa
from .activation_matching import cross_features_inner_product


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








