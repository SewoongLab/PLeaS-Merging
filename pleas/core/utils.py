import operator as op
from collections.abc import Sequence
from copy import copy, deepcopy
from dataclasses import dataclass
from functools import partial, reduce
from typing import Generic, Optional, TypeVar, Union

import numpy as np
import torch
import torch.utils._pytree as pytree
from torch import nn
from sklearn.model_selection import train_test_split
from torch.fx.passes.shape_prop import ShapeProp


@dataclass(frozen=True)
class Axis:
    """
    Represents a permutable axis in a tensor.
    
    Args:
        key (str): Identifier for the tensor parameter
        axis (int): Axis index within the tensor that can be permuted
    """
    key: str
    axis: int

    def __str__(self):
        return f"{self.key}:{self.axis}"

    __repr__ = __str__


@dataclass
class PermutationGroup:
    """
    Group of axes that should be permuted together.
    
    Args:
        size (int): Size of the permutation (number of units/neurons)
        state (set[Axis]): Set of axes in the state dict
        node (set[Axis]): Set of axes in the computation graph
    """
    size: int
    state: set[Axis]
    node: set[Axis]


PermutationKey = Axis
PermutationSpec = dict[PermutationKey, PermutationGroup]
Permutation = dict[PermutationKey, torch.tensor]
Compression = dict[PermutationKey, torch.tensor]  # The second element in the dict contains the weights that need to be assigned to the remaining units to reconstruct this unit
# For pruning with fine-tuning, these can be computed using the original and pruned matrices.
PyTreePath = Sequence[Union[str, int]]
StateDict = dict[str, torch.tensor]
InputsOrShapes = Union[tuple[tuple, ...], tuple[torch.Tensor, ...]]


def tree_normalize_path(path: PyTreePath):
    """
    Normalize a path to a nested attribute or element.
    
    Args:
        path (PyTreePath): Path to normalize
        
    Returns:
        list: Normalized path
    """
    def process_atom(a):
        try:
            return int(a)
        except ValueError:
            return a

    def process_molecule(m):
        if isinstance(m, str):
            return m.split(".")
        return m

    path = pytree.tree_map(process_molecule, path)
    path = pytree.tree_map(process_atom, path)
    path = pytree.tree_flatten(path)[0]
    return path


def tree_index(tree, path: PyTreePath):
    """
    Access a nested element in a tree structure.
    
    Args:
        tree: Tree structure to access
        path (PyTreePath): Path to the element
        
    Returns:
        The element at the specified path
    """
    path = tree_normalize_path(path)
    subtree = tree
    for i, atom in enumerate(path):
        if hasattr(subtree, str(atom)):
            subtree = getattr(subtree, str(atom))
        else:
            subtree = subtree[atom]

    return subtree


def get_attr(obj, names):
    """
    Access a nested attribute in an object.
    
    Args:
        obj: Object to access attribute from
        names (list): List of attribute names to access in sequence
        
    Returns:
        The requested attribute
    """
    if len(names) == 1:
        return getattr(obj, names[0])
    else:
        return get_attr(getattr(obj, names[0]), names[1:])


def set_attr(obj, names, val):
    """
    Set a nested attribute in an object.
    
    Args:
        obj: Object to set attribute in
        names (list): List of attribute names to traverse
        val: Value to set
        
    Returns:
        The result of setting the attribute
    """
    if len(names) == 1:
        return setattr(obj, names[0], val)
    else:
        return set_attr(getattr(obj, names[0]), names[1:], val)


def make_identity_perm(spec: PermutationSpec) -> Permutation:
    """
    Create an identity permutation for each axis in the specification.
    
    Args:
        spec (PermutationSpec): Permutation specification
        
    Returns:
        Permutation: Identity permutation mapping
    """
    return {k: torch.arange(pg.size) for k, pg in spec.items()}


def make_random_perm(spec: PermutationSpec) -> Permutation:
    """
    Create a random permutation for each axis in the specification.
    
    Args:
        spec (PermutationSpec): Permutation specification
        
    Returns:
        Permutation: Random permutation mapping
    """
    return {k: torch.randperm(pg.size) for k, pg in spec.items()}


def invert_perm(perm: Union[torch.tensor, Permutation]):
    """
    Invert a permutation or dictionary of permutations.
    
    Args:
        perm (Union[torch.tensor, Permutation]): Permutation to invert
        
    Returns:
        Union[torch.tensor, Permutation]: Inverted permutation
    """
    if isinstance(perm, dict):
        return {k: invert_perm(p) for k, p in perm.items()}

    p = torch.empty_like(perm)
    p[perm] = torch.arange(len(p))
    return p


def perm_eq(perm1: Permutation, perm2: Permutation) -> bool:
    """
    Check if two permutations are equal.
    
    Args:
        perm1 (Permutation): First permutation
        perm2 (Permutation): Second permutation
        
    Returns:
        bool: True if permutations are equal, False otherwise
    """
    return len(perm1) == len(perm2) and all(
        (perm2[k] == p).all() for (k, p) in perm1.items()
    )


def apply_perm(
    perm: Permutation,
    spec: PermutationSpec,
    state: Union[nn.Module, StateDict],
    inplace=False,
    skip_missing=True,
):
    """
    Apply a permutation to a model or state dictionary.
    
    Args:
        perm (Permutation): Permutation to apply
        spec (PermutationSpec): Permutation specification
        state (Union[nn.Module, StateDict]): Model or state dictionary to permute
        inplace (bool, optional): Whether to modify state in place. Defaults to False.
        skip_missing (bool, optional): Whether to skip missing keys. Defaults to True.
        
    Returns:
        Union[nn.Module, StateDict]: Permuted model or state dictionary
    """
    if isinstance(state, nn.Module):
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

        pg = spec[key]
        assert P.shape == (pg.size,)
        for ax in pg.state:
            if skip_missing and ax.key not in state:
                continue

            weight = state[ax.key]
            state[ax.key] = torch.index_select(weight, ax.axis, P.to(weight.device))

    return state


def apply_perm_with_padding(
    perm: Permutation,
    padding: Permutation,
    size: int,
    pad_ahead: bool,
    spec: PermutationSpec,
    state: Union[nn.Module, StateDict],
    inplace=False,
    skip_missing=True,
):
    """
    Apply a permutation to a model or state dict with padding.
    
    This function applies a permutation to a model's weights and adds padding to
    support partial merging operations.
    
    Args:
        perm (Permutation): Permutation to apply
        padding (Permutation): Indices to use for padding
        size (int): Target size for the permuted dimension
        pad_ahead (bool): Whether to add padding before the permuted weights
        spec (PermutationSpec): Permutation specification
        state (Union[torch.nn.Module, StateDict]): Model or state dict to permute
        inplace (bool, optional): Whether to modify the state in place. Defaults to False.
        skip_missing (bool, optional): Whether to skip missing keys. Defaults to True.
        
    Returns:
        Union[torch.nn.Module, StateDict]: Permuted state
    """
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

        pg = spec[key]
        # assert P.shape == (pg.size,)
        # print(f"Applying {key} with {P.shape} and {padding.shape} to {size}, {pg.size}")
        for ax in pg.state:
            if skip_missing and ax.key not in state:
                continue
            
            weight = remove_zero_block(state[ax.key], ax.axis, len(padding), size, pad_ahead)
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


T = TypeVar("T")


class UnionFind(Generic[T]):
    """
    Union-find data structure for grouping related elements.
    
    This is used to identify groups of axes that should be permuted together.
    """
    def __init__(self, items: Sequence[T] = ()):
        self.parent_node = {}
        self.rank = {}
        self.extend(items)

    def extend(self, items: Sequence[T]):
        for x in items:
            self.parent_node.setdefault(x, x)
            self.rank.setdefault(x, 0)

    def find(self, item: T, add: bool = False) -> T:
        assert ":" in item
        if add:
            if item not in self.parent_node:
                self.extend([item])

        if self.parent_node[item] != item:
            self.parent_node[item] = self.find(self.parent_node[item])

        return self.parent_node[item]

    def union(self, item1: T, item2: T, add=False):
        p1 = self.find(item1, add)
        p2 = self.find(item2, add)

        if p1 == p2:
            return
        if self.rank[p1] > self.rank[p2]:
            self.parent_node[p2] = p1
        elif self.rank[p1] < self.rank[p2]:
            self.parent_node[p1] = p2
        else:
            self.parent_node[p1] = p2
            self.rank[p2] += 1

    def groups(self):
        sets = {}
        for x in self.parent_node.keys():
            p = self.find(x)
            sets.setdefault(p, set()).add(x)
        return sets

    def __repr__(self):
        return f"UnionFind{repr(self.groups())}"


def reset_running_stats(net):
    """
    Reset running statistics for BatchNorm layers in a network.
    
    Args:
        net (torch.nn.Module): Network to reset statistics for
    """
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()


def tree_multimap(f, *trees):
    """
    Apply a function to corresponding elements in multiple trees.
    
    Args:
        f: Function to apply
        *trees: Trees to map over
        
    Returns:
        Result tree after applying the function
    """
    flats, specs = zip(*(pytree.tree_flatten(tree) for tree in trees))

    def eq_checker(a, b):
        assert a == b
        return a

    reduce(eq_checker, specs)
    spec = next(iter(specs))
    mapped = list(map(lambda t: f(*t), zip(*flats)))
    return pytree.tree_unflatten(mapped, spec)


def tree_reduce(f, tree):
    """
    Reduce a tree to a single value using a binary function.
    
    Args:
        f: Binary function to apply
        tree: Tree to reduce
        
    Returns:
        Reduced value
    """
    flat, _ = pytree.tree_flatten(tree)
    return reduce(f, flat)


def tree_linear(*terms):
    """
    Compute a linear combination of trees.
    
    Args:
        *terms: Pairs of (coefficient, tree)
        
    Returns:
        Result tree
    """
    assert len(terms) > 0

    def inner(*tensors):
        return reduce(op.add, (a * t for t, (a, _) in zip(tensors, terms)))

    return tree_multimap(inner, *(t for _, t in terms))


def tree_mean(*sds):
    """
    Compute the mean of multiple trees.
    
    Args:
        *sds: Trees to average
        
    Returns:
        Mean tree
    """
    return tree_linear(*((1 / len(sds), sd) for sd in sds))


def tree_vdot(tree1, tree2):
    """
    Compute the dot product of two trees.
    
    Args:
        tree1: First tree
        tree2: Second tree
        
    Returns:
        Scalar dot product
    """
    def vdot(a, b):
        return torch.sum(a * b)
        # return torch.vdot(a.ravel(), b.ravel())

    return tree_reduce(torch.add, tree_multimap(vdot, tree1, tree2))


def tree_norm(tree):
    """
    Compute the Euclidean norm of a tree.
    
    Args:
        tree: Tree to compute norm for
        
    Returns:
        Scalar norm
    """
    return torch.sqrt(tree_vdot(tree, tree))


def tree_cosine_sim(tree1, tree2):
    """
    Compute the cosine similarity between two trees.
    
    Args:
        tree1: First tree
        tree2: Second tree
        
    Returns:
        Cosine similarity
    """
    return tree_vdot(tree1, tree2) / tree_norm(tree1) / tree_norm(tree2)


def lerp(lam, tree1, tree2):
    """
    Linear interpolation between two trees.
    
    Args:
        lam (float): Interpolation parameter (0: tree1, 1: tree2)
        tree1: First tree
        tree2: Second tree
        
    Returns:
        Interpolated tree
    """
    # return {k: (1 - lam) * a + lam * state_b[k] for k, a in state_a.items()}
    return tree_linear(((1 - lam), tree1), (lam, tree2))


def slerp(lam, tree1, tree2):
    """
    Spherical linear interpolation between two trees.
    
    Args:
        lam (float): Interpolation parameter (0: tree1, 1: tree2)
        tree1: First tree
        tree2: Second tree
        
    Returns:
        Interpolated tree
    """
    omega = torch.acos(tree_cosine_sim(tree1, tree2))
    a, b = torch.sin((1 - lam) * omega), torch.sin(lam * omega)
    denom = torch.sin(omega)
    return tree_linear((a / denom, tree1), (b / denom, tree2))


def lslerp(lam, tree1, tree2):
    """
    Local spherical linear interpolation between two trees.
    
    Args:
        lam (float): Interpolation parameter (0: tree1, 1: tree2)
        tree1: First tree
        tree2: Second tree
        
    Returns:
        Interpolated tree
    """
    return tree_multimap(partial(slerp, lam), tree1, tree2)


def count_linear_flops(
    spec: PermutationSpec, model: torch.nn.Module, inputs_or_shapes: InputsOrShapes
) -> tuple[int, list[tuple[int, Axis, ...]]]:
    """
    Count the FLOPs for linear and convolutional operations in a model.
    
    Args:
        spec (PermutationSpec): Permutation specification
        model (torch.nn.Module): Model to analyze
        inputs_or_shapes (InputsOrShapes): Input shapes or tensors
        
    Returns:
        tuple: (flops, terms) where terms represents the operations
    """
    # TODO: Only works for nn.Conv2d and nn.Linear ATM

    # Prepare the inputs and perform the shape propagation
    device = next(iter(model.parameters())).device
    inputs = [
        torch.randn(*ios).to(device) if isinstance(ios, tuple) else ios.to(device)
        for ios in inputs_or_shapes
    ]
    gm = torch.fx.symbolic_trace(model)
    sp = ShapeProp(gm)
    sp.propagate(*inputs)

    # Scan through all Conv2d and Linear
    terms, sizes = [], {}
    for node in gm.graph.nodes:
        if node.op == "call_module":
            mod = sp.fetch_attr(node.target)
            shape = node.meta["tensor_meta"].shape

            if isinstance(mod, (nn.Conv2d, nn.Linear)):
                coeff = shape[0]
                if isinstance(mod, nn.Conv2d):
                    coeff *= np.prod(shape[2:]) * np.prod(mod.kernel_size)

                ain = Axis(f"{node.target}.weight", 1)
                aout = Axis(f"{node.target}.weight", 0)
                sout, sin, *_ = mod.weight.shape
                terms.append((coeff, ain, aout))
                assert sizes.setdefault(ain, sin) == sin
                assert sizes.setdefault(aout, sout) == sout

    # Simplify the terms and count flops
    flops, new_terms = 0, []
    axis_keys = {ax: k for k, pg in spec.items() for ax in pg.state}
    for coef, *axes in terms:
        flops += coef * np.prod([sizes[axis] for axis in axes])
        new_axes = []
        for axis in axes:
            if axis not in axis_keys:
                coef *= sizes[axis]
            else:
                new_axes.append(axis_keys[axis])

        new_terms.append((coef, *new_axes))

    return flops, new_terms


def remove_zero_block(tensor, axis, block_size, final_size, beginning=False):
    """
    Remove a block of zeros from a tensor along a specified axis.
    
    Args:
        tensor (torch.Tensor): Input tensor
        axis (int): Axis along which to remove zeros
        block_size (int): Size of the zero block to remove
        final_size (int): Final size of the tensor along the axis
        beginning (bool, optional): Whether the zero block is at the beginning. Defaults to False.
        
    Returns:
        torch.Tensor: Tensor with zero block removed
    """
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