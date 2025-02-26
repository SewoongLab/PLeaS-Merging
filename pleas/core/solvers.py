import bisect

import scipy as sp
import torch
import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

try:
    from ortools.graph.python import linear_sum_assignment as or_lsa
except ImportError:
    pass

try:
    import lapjv
except ImportError:
    pass

def scipy_solve_lsa(A, maximize=True):
    ri, ci = map(
        torch.tensor, sp.optimize.linear_sum_assignment(A.cpu().numpy(), maximize=maximize)
    )
    assert (ri == torch.arange(len(ri))).all()
    return ci


def or_solve_lsa(A):
    scale = 2 / A.ravel().sort().values.diff().min()
    print(1, scale)
    costs = (-scale * A).cpu().numpy()

    end_nodes_unraveled, start_nodes_unraveled = np.meshgrid(
        np.arange(costs.shape[1]), np.arange(costs.shape[0])
    )
    start_nodes = start_nodes_unraveled.ravel()
    end_nodes = end_nodes_unraveled.ravel()
    arc_costs = costs.ravel()

    assn = or_lsa.SimpleLinearSumAssignment()
    assn.add_arcs_with_cost(start_nodes, end_nodes, arc_costs)

    status = assn.solve()
    assert status == assn.OPTIMAL

    return torch.tensor([assn.right_mate(i) for i in range(assn.num_nodes())])


def lapjv_solve_lsa(A):
    ri, _, _ = lapjv.lapjv(-A.cpu().numpy())
    return torch.tensor(ri).long()


def scipy_solve_minimax_assignment(A):
    B = A.detach().cpu().numpy()
    values = B.copy().ravel()
    values.sort()
    best_t, best_matching = -float('inf'), None

    def check(t):
        nonlocal best_t, best_matching
        E = csr_matrix(B >= t)
        matching = maximum_bipartite_matching(E, perm_type='column')
        test = not (matching == -1).any()
        if test and t > best_t:
            best_t, best_matching = t, matching
        return not test

    bisect.bisect_right(values, False, key=check)

    return torch.from_numpy(best_matching).long().to(A.device)
