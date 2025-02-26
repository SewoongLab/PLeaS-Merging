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
    key: str
    axis: int

    def __str__(self):
        return f"{self.key}:{self.axis}"

    __repr__ = __str__


@dataclass
class PermutationGroup:
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
    path = tree_normalize_path(path)
    subtree = tree
    for i, atom in enumerate(path):
        if hasattr(subtree, str(atom)):
            subtree = getattr(subtree, str(atom))
        else:
            subtree = subtree[atom]

    return subtree


def get_attr(obj, names):
    if len(names) == 1:
        return getattr(obj, names[0])
    else:
        return get_attr(getattr(obj, names[0]), names[1:])


def set_attr(obj, names, val):
    if len(names) == 1:
        return setattr(obj, names[0], val)
    else:
        return set_attr(getattr(obj, names[0]), names[1:], val)


def make_identity_perm(spec: PermutationSpec) -> Permutation:
    return {k: torch.arange(pg.size) for k, pg in spec.items()}


def make_random_perm(spec: PermutationSpec) -> Permutation:
    return {k: torch.randperm(pg.size) for k, pg in spec.items()}


def invert_perm(perm: Union[torch.tensor, Permutation]):
    if isinstance(perm, dict):
        return {k: invert_perm(p) for k, p in perm.items()}

    p = torch.empty_like(perm)
    p[perm] = torch.arange(len(p))
    return p


def perm_eq(perm1: Permutation, perm2: Permutation) -> bool:
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
            if weight.shape[ax.axis] == size:
                state[ax.key] = torch.index_select(weight, ax.axis, P.to(weight.device))
            
            # select indices from the original weight
            permuted_weights = torch.index_select(weight, ax.axis, P.to(weight.device))
            separate_weights = torch.index_select(weight, ax.axis, padding.to(weight.device))
            if ax.axis == 0:
                padding_weights = torch.zeros((size - weight.shape[ax.axis], *weight.shape[1:]))
            else:
                padding_weights = torch.zeros((weight.shape[0], size - weight.shape[ax.axis], *weight.shape[2:]))
                
            if pad_ahead:
                final_weights = torch.cat((padding_weights, separate_weights, permuted_weights), ax.axis)
            else:
                final_weights = torch.cat((separate_weights, padding_weights, permuted_weights), ax.axis)
            state[ax.key] = final_weights   
            
    return state


T = TypeVar("T")


class UnionFind(Generic[T]):
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
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.reset_running_stats()


def tree_multimap(f, *trees):
    flats, specs = zip(*(pytree.tree_flatten(tree) for tree in trees))

    def eq_checker(a, b):
        assert a == b
        return a

    reduce(eq_checker, specs)
    spec = next(iter(specs))
    mapped = list(map(lambda t: f(*t), zip(*flats)))
    return pytree.tree_unflatten(mapped, spec)


def tree_reduce(f, tree):
    flat, _ = pytree.tree_flatten(tree)
    return reduce(f, flat)


def tree_linear(*terms):
    assert len(terms) > 0

    def inner(*tensors):
        return reduce(op.add, (a * t for t, (a, _) in zip(tensors, terms)))

    return tree_multimap(inner, *(t for _, t in terms))


def tree_mean(*sds):
    return tree_linear(*((1 / len(sds), sd) for sd in sds))


def tree_vdot(tree1, tree2):
    def vdot(a, b):
        return torch.sum(a * b)
        # return torch.vdot(a.ravel(), b.ravel())

    return tree_reduce(torch.add, tree_multimap(vdot, tree1, tree2))


def tree_norm(tree):
    return torch.sqrt(tree_vdot(tree, tree))


def tree_cosine_sim(tree1, tree2):
    return tree_vdot(tree1, tree2) / tree_norm(tree1) / tree_norm(tree2)


def lerp(lam, tree1, tree2):
    # return {k: (1 - lam) * a + lam * state_b[k] for k, a in state_a.items()}
    return tree_linear(((1 - lam), tree1), (lam, tree2))


def slerp(lam, tree1, tree2):
    omega = torch.acos(tree_cosine_sim(tree1, tree2))
    a, b = torch.sin((1 - lam) * omega), torch.sin(lam * omega)
    denom = torch.sin(omega)
    return tree_linear((a / denom, tree1), (b / denom, tree2))


def lslerp(lam, tree1, tree2):
    return tree_multimap(partial(slerp, lam), tree1, tree2)


def count_linear_flops(
    spec: PermutationSpec, model: torch.nn.Module, inputs_or_shapes: InputsOrShapes
) -> tuple[int, list[tuple[int, Axis, ...]]]:
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

import math
import random
from torch.utils.data import DataLoader
class FractionalDataloader:
    def __init__(self, dataloader, fraction, seed=None):
        self.dataloader_numel = len(dataloader.dataset)
        self.numel = int(fraction * self.dataloader_numel)

        self.batch_size = self.dataloader_numel / len(dataloader)
        self.num_batches = int(math.ceil(self.numel / self.batch_size))
        self.dataloader = dataloader
        self.dataset = self.dataloader.dataset
        self.seed = seed
    
    def __iter__(self):
        cur_elems = 0
        if self.seed is not None:
            self.dataloader.dataset.set_seed(self.seed)
            torch.manual_seed(self.seed)
            random.seed(self.seed)
            np.random.seed(self.seed)
        it = iter(self.dataloader)
        while cur_elems < self.numel:
            try:
                x, y = next(it)
                cur_elems += x.shape[0]
                yield x, y
            except StopIteration:
                it = iter(self.dataloader)
                
        
    def __len__(self):
        return self.num_batches
def create_heldout_split(dataset, fraction):
    root = dataset.root_og
    val_set, test_set = train_test_split(dataset.dataset, test_size=fraction)
    val_set = dataset.__class__(root, train=dataset.train, transform=dataset.transform, base_set=val_set)
    test_set = dataset.__class__(root, train=dataset.train, transform=dataset.transform, base_set=test_set)
    return val_set, test_set

def prepare_data(config, device='cuda'):
    """ Load all dataloaders required for experiment. """
    if isinstance(config, list):
        return [prepare_data(c, device) for c in config]
    
    dataset_name = config['name']
    
    from datasets_module import configs as config_module
    data_config = deepcopy(getattr(config_module, dataset_name))
    data_config.update(config)
    data_config['device'] = device

    if data_config['type'] == 'cifar':
        from datasets_module.cifar import prepare_train_loaders, prepare_test_loaders
        train_loaders = prepare_train_loaders(data_config)
        test_loaders = prepare_test_loaders(data_config)
    elif data_config['type'] == 'imagenet':
        # train_loaders, test_loaders = None,  None
        from datasets_module.imagenet import prepare_loaders, ImageNet
        # train_loaders, test_loaders = prepare_loaders(data_config)
        im = ImageNet()
        train_loaders = {"full": im.train_loader}
        test_loaders = {"full": im.val_loader}
    elif data_config['type'] == 'nabird':
        from datasets_module.nabird import prepare_train_loaders, prepare_test_loaders
        train_loaders = prepare_train_loaders(data_config)
        test_loaders = prepare_test_loaders(data_config)
    elif data_config['type'] == 'cub':
        from datasets_module.cub import prepare_train_loaders, prepare_test_loaders
        train_loaders = prepare_train_loaders(data_config)
        test_loaders = prepare_test_loaders(data_config)
    elif data_config['type'] == 'oxford_pets':
        from datasets_module.oxford_pets import prepare_train_loaders, prepare_test_loaders
        train_loaders = prepare_train_loaders(data_config)
        test_loaders = prepare_test_loaders(data_config)
    elif data_config['type'] == 'stanford_dogs':
        from datasets_module.stanford_dogs import prepare_train_loaders, prepare_test_loaders
        train_loaders = prepare_train_loaders(data_config)
        test_loaders = prepare_test_loaders(data_config)
    elif 'domainnet' in data_config['type']:
        from datasets_module.domainnet import prepare_test_loaders, prepare_train_loaders
        train_loaders = prepare_train_loaders(data_config, data_config['type'].split('_')[-1])
        test_loaders = prepare_test_loaders(data_config, data_config['type'].split('_')[-1])
    else:
        raise NotImplementedError(config['type'])
    
    if 'train_fraction' in data_config:
        for k, v in dict(train_loaders.items()).items():
            if k == 'splits':
                train_loaders[k] = [FractionalDataloader(x, data_config['train_fraction']) for x in v]
            elif not isinstance(v, list) and not isinstance(v, torch.Tensor):
                train_loaders[k] = FractionalDataloader(v, data_config['train_fraction'])

    return {
        'train': train_loaders,
        'test': test_loaders
    }


def remove_zero_block(tensor, axis, block_size):
    # Find the indices where the sum along the specified axis is zero
    zero_indices = (torch.sum(tensor, dim=1-axis) == 0).nonzero().squeeze()

    # Find the start and end of the zero block
    start_idx = zero_indices[0].item()
    end_idx = start_idx + block_size

    # Slice the tensor to exclude the block of zeros
    if axis == 0:
        return torch.cat((tensor[:start_idx], tensor[end_idx:]), axis=0)
    elif axis == 1:
        return torch.cat((tensor[:, :start_idx], tensor[:, end_idx:]), axis=1)