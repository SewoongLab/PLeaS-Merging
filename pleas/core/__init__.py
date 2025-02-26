"""
Core utilities and data structures for the PLeaS merging algorithm.

This module provides the fundamental data structures and utilities
for permutation-based model merging, including permutation specifications,
application functions, and model analysis tools.
"""

from pleas.core.utils import (
    Axis,
    PermutationGroup,
    PermutationSpec,
    Permutation,
    apply_perm,
    make_identity_perm,
    make_random_perm,
    invert_perm,
    count_linear_flops,
)

from pleas.core.solvers import scipy_solve_lsa