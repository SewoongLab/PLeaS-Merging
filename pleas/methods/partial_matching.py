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