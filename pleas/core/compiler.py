import collections
import operator as op
import re
import sys
import traceback
from copy import copy, deepcopy
from typing import Any, Dict, NamedTuple, Optional, Tuple

import torch
import torch.fx.passes
import torchvision.ops as tvops
from torch import nn
from torch.fx._compatibility import compatibility
from torch.fx.node import Argument, Node, Target, map_aggregate, map_arg
from torch.nn import functional as F

from .utils import (
    Axis,
    InputsOrShapes,
    PermutationGroup,
    PermutationSpec,
    UnionFind,
    apply_perm,
    tree_normalize_path,
)


class PermutationProp(torch.fx.Interpreter):
    module_handlers = {}
    function_handlers = {}
    method_handlers = {}

    def __init__(self, module, verbose=True):
        super().__init__(module)
        self.axis_groups = UnionFind()
        self.inputs = []
        self.result = None
        self.verbose = verbose

    def run_node(self, n: Node) -> Any:
        args, kwargs = self.fetch_args_kwargs_from_env(n)
        assert isinstance(args, tuple)
        assert isinstance(kwargs, dict)
        result = getattr(self, n.op)(n, args, kwargs)

        return result

    def call_function(
        self, n: Node, args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Any:
        target = n.target
        assert not isinstance(target, str)

        if self.verbose:
            print(
                "call_function",
                target,
                [a.shape if torch.is_tensor(a) else a for a in args],
                {k: v.shape if torch.is_tensor(v) else v for k, v in kwargs.items()},
            )

        # Execute the function and return the result
        result = target(*args, **kwargs)

        # If function does not touch a tensor, ignore it
        if not any(torch.is_tensor(a) for a in args) and not any(
            torch.is_tensor(a) for a in kwargs.values()
        ):
            return result

        handler = self.function_handlers.get(target)
        if handler:
            links = handler(self, target, args, kwargs, result)
            for link in links:
                self.axis_groups.union(
                    *{self.get_absolute_path(n, endpoint) for endpoint in link},
                    add=True,
                )

        else:
            raise NotImplementedError(f"No handler for function {target}")

        return result

    def call_method(
        self, n: Node, args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Any:
        target = n.target
        # args[0] is the `self` object for this method call
        self_obj, *args_tail = args

        # Execute the method and return the result
        assert isinstance(target, str)

        if self.verbose:
            print(
                "call_method",
                self_obj.shape if torch.is_tensor(self_obj) else self_obj,
                target,
                [a.shape if torch.is_tensor(a) else a for a in args_tail],
                {k: v.shape if torch.is_tensor(v) else v for k, v in kwargs.items()},
            )

        result = getattr(self_obj, target)(*args_tail, **kwargs)

        handler = self.method_handlers.get((self_obj.__class__, target))
        if handler:
            links = handler(self, self_obj, target, args_tail, kwargs, result)
            for link in links:
                self.axis_groups.union(
                    *{self.get_absolute_path(n, endpoint) for endpoint in link},
                    add=True,
                )
        else:
            raise NotImplementedError(
                f"No handler for method call {self_obj.__class__}.{target}"
            )

        return result

    def call_module(
        self, n: Node, args: Tuple[Argument, ...], kwargs: Dict[str, Any]
    ) -> Any:
        target = n.target
        # Retrieve executed args and kwargs values from the environment

        # Execute the method and return the result
        assert isinstance(target, str)
        submod = self.fetch_attr(target)

        if self.verbose:
            print(
                "call_module",
                submod,
                [a.shape if torch.is_tensor(a) else a for a in args],
                {k: v.shape if torch.is_tensor(v) else v for k, v in kwargs.items()},
            )

        result = submod(*args, **kwargs)

        handler = self.module_handlers.get(submod.__class__)
        if handler:
            links = handler(self, submod, args, kwargs, result)
            for link in links:
                self.axis_groups.union(
                    *{self.get_absolute_path(n, endpoint) for endpoint in link},
                    add=True,
                )

        else:
            raise NotImplementedError(f"No handler for module {submod}")

        return result

    def placeholder(self, n, args, kwargs):
        assert not args
        assert not kwargs
        result = super().placeholder(n.target, args, kwargs)

        self.inputs.append(result)

        for i in range(len(result.shape)):
            self.axis_groups.union(
                f"placeholders.{len(self.inputs) - 1}:{i}",
                self.get_absolute_path(n, f"output:{i}"),
                add=True,
            )

        return result

    def get_attr(self, n, *args):
        result = super().get_attr(n.target, *args)

        assert n.target in self.module.state_dict()

        for i in range(len(result.shape)):
            self.axis_groups.union(
                f"state.{n.target}:{i}",
                f"node.{n.name}:{i}",
                add=True,
            )

        return result

    def output(self, n, args, kwargs):
        if self.verbose:
            print(args)

        assert len(args) == 1
        assert not kwargs

        result = super().output(n.target, args, kwargs)
        self.result = result

        for i in range(len(result.shape)):
            self.axis_groups.union(
                f"result:{i}",
                self.get_absolute_path(n, f"args.0:{i}"),
                add=True,
            )

        return result

    def register_module_handler(*modules):
        def inner(f):
            handlers = sys._getframe(1).f_locals.get("module_handlers")
            for module in modules:
                handlers[module] = f
            return f

        return inner

    def register_function_handler(*functions):
        def inner(f):
            handlers = sys._getframe(1).f_locals.get("function_handlers")
            for function in functions:
                handlers[function] = f
            return f

        return inner

    def register_method_handler(*methods):
        def inner(f):
            handlers = sys._getframe(1).f_locals.get("method_handlers")
            for method in methods:
                if not isinstance(method, tuple):
                    handlers[torch.Tensor, method] = f
                    handlers[torch.nn.Parameter, method] = f
                else:
                    handlers[method] = f
            return f

        return inner

    def get_absolute_path(self, n, relpath):
        assert ":" in relpath
        if match := re.match(r"(args|kwargs).(?P<tail>.*)", relpath):
            relpath, axis = relpath.split(":")
            path = tree_normalize_path(relpath)

            subtree = {"args": n.args, "kwargs": n.kwargs}

            # This is kind of gross
            if n.op == "call_method":
                subtree["args"] = subtree["args"][1:]

            for i, atom in enumerate(path):
                if hasattr(subtree, str(atom)):
                    subtree = getattr(subtree, str(atom))
                else:
                    subtree = subtree[atom]
                if isinstance(subtree, Node):
                    break

            assert isinstance(subtree, Node)
            tail = ".".join(map(str, path[i + 1 :]))

            return f"node.{subtree.name}{'.' if tail else ''}{tail}:{axis}"

        elif match := re.match(r"state.(?P<tail>.*)", relpath):
            tail = match.group("tail")
            return f"state.{n.target}.{tail}"
        elif match := re.match(r"output(?P<tail>[.:].*)", relpath):
            tail = match.group("tail")
            return f"node.{n.name}{tail}"
        elif match := re.match(r"object(?P<tail>[.:].*)", relpath):
            tail = match.group("tail")
            return f"node.{n.args[0]}{tail}"
        else:
            raise ValueError(f"Unknown link group {relpath.split('.')[0]}!")

    @register_module_handler(nn.Conv2d)
    def handle_conv2d(self, module, args, kwargs, output):
        assert len(args) == 1
        assert module.groups in {1, module.in_channels}
        assert not kwargs
        x = args[0]
        assert len(x.shape) == 4  # NCHW

        if module.groups == 1:
            return [
                {"args.0:0", "output:0"},
                {"args.0:1", "state.weight:1"},
                {"output:1", "state.weight:0"},
                *([{"output:1", "state.bias:0"}] if module.bias is not None else []),
            ]

        else:
            # Depthwise separable conv
            assert module.groups == x.shape[1]
            assert x.shape[:2] == output.shape[:2]
            return [
                {"args.0:0", "output:0"},
                {"args.0:1", "output:1"},
                {"output:1", "state.weight:0"},
                *([{"output:1", "state.bias:0"}] if module.bias is not None else []),
            ]

    @register_function_handler(torch.conv2d)
    def handle_conv2d_function(self, _, args, kwargs, output):
        x = args[0]
        assert kwargs.get("groups") == x.shape[1]
        assert not kwargs.get("bias")

        print("TODO" * 10)
        return [
            {"args.0:0", "output:0"},
            {"args.0:1", "output:1"},
            {"args.0:1", "args.1:0"},
        ]

    @register_module_handler(nn.BatchNorm2d)
    def handle_batchnorm2d(self, module, args, kwargs, output):
        assert len(args) == 1
        assert not kwargs
        x = args[0]
        assert len(x.shape) == 4  # NCHW
        assert x.shape == output.shape
        return [
            {"args.0:0", "output:0"},
            {"args.0:1", "output:1"},
            {"args.0:1", "state.weight:0"},
            {"args.0:1", "state.bias:0"},
            {"args.0:1", "state.running_mean:0"},
            {"args.0:1", "state.running_var:0"},
            {"args.0:2", "output:2"},
            {"args.0:3", "output:3"},
        ]

    @register_function_handler(
        torch.sigmoid, tvops.stochastic_depth, F.gelu, F.relu, torch.sqrt
    )
    @register_module_handler(
        nn.ReLU, nn.GELU, nn.Identity, nn.Dropout, nn.Sigmoid, nn.SiLU
    )
    def handle_elementwise(self, _, args, kwargs, output):
        x = args[0]
        assert x.shape == output.shape
        return [{f"args.0:{i}", f"output:{i}"} for i in range(len(x.shape))]

    @register_function_handler(F.adaptive_avg_pool2d, F.avg_pool2d)
    @register_module_handler(nn.AdaptiveAvgPool2d, nn.AvgPool2d)
    def handle_adaptiveavgpool2d(self, module, args, kwargs, output):
        # assert len(args) == 1
        # assert not kwargs
        x = args[0]
        # assert len(x.shape) == 4  # NCHW
        # assert len(output.shape) == 4
        # assert module.output_size in {1, (1, 1)}
        return [{"args.0:0", "output:0"}, {"args.0:1", "output:1"}]

    @register_module_handler(nn.MaxPool2d)
    def handle_maxpool2d(self, module, args, kwargs, output):
        assert len(args) == 1
        assert not kwargs
        x = args[0]
        assert len(x.shape) == 4  # NCHW
        assert len(output.shape) == 4
        return [{"args.0:0", "output:0"}, {"args.0:1", "output:1"}]

    @register_module_handler(nn.Linear)
    def handle_linear(self, module, args, kwargs, output):
        assert len(args) == 1
        assert not kwargs
        x = args[0]
        assert x.shape[:1] == output.shape[:1]
        last = len(x.shape) - 1
        return [
            *[{f"args.0:{i}", f"output:{i}"} for i in range(last)],
            {f"args.0:{last}", "state.weight:1"},
            {f"output:{last}", "state.weight:0"},
            *([{f"output:{last}", "state.bias:0"}] if module.bias is not None else []),
        ]

    @register_function_handler(op.add, op.sub, op.mul, op.truediv)
    def handle_binop(self, f, args, kwargs, output):
        assert len(args) == 2
        assert not kwargs
        x, y = args
        if isinstance(x, float):
            x = torch.tensor(x)
        if isinstance(y, float):
            y = torch.tensor(y)

        n, m = len(x.shape), len(y.shape)
        npad, mpad = max(m - n, 0), max(n - m, 0)
        xs, ys = (*[1] * npad, *x.shape), (*[1] * mpad, *y.shape)

        links = []
        for i in range(max(n, m)):
            if xs[i] == ys[i] and i >= max(npad, mpad):
                links.extend(
                    [
                        {f"args.0:{i-npad}", f"args.1:{i-mpad}"},
                        {f"args.0:{i-npad}", f"output:{i}"},
                    ]
                )
            elif xs[i] > ys[i] == 1 or (i < mpad and xs[i] == ys[i] == 1):
                links.append({f"args.0:{i-npad}", f"output:{i}"})
            elif ys[i] > xs[i] == 1 or (i < npad and xs[i] == ys[i] == 1):
                links.append({f"args.1:{i-mpad}", f"output:{i}"})
            else:
                # I think the previous three cases should cover everything
                assert False

        if self.verbose:
            print("shapes", x.shape, y.shape)
            print("links", links)

        return links

    @register_method_handler("mul", "add", "sub")
    def handle_binop_method(self, obj, method, args, kwargs, result):
        assert not kwargs
        assert len(args) == 1
        x, y = obj, args[0]
        links = self.handle_binop(None, (x, y), {}, result)
        links = [
            {l.replace("args.0", "object").replace("args.1", "args.0") for l in link}
            for link in links
        ]
        return links

    @register_function_handler(op.matmul)
    def handle_matmul(self, f, args, kwargs, output):
        assert len(args) == 2
        assert not kwargs
        x, y = args
        assert isinstance(x, (torch.Tensor, nn.Parameter))
        assert isinstance(y, (torch.Tensor, nn.Parameter))
        assert len(x.shape) == len(y.shape) == 2

        return [
            {"args.0:0", "output:0"},
            {"args.0:1", "args.1:0"},
            {"args.1:1", "output:1"},
        ]

    @register_function_handler(torch.flatten)
    def handle_flatten(self, f, args, kwargs, output):
        assert 1 <= len(args) <= 3
        assert not kwargs
        x = args[0]
        dims = len(x.shape)
        start_dim = range(dims)[args[1] if len(args) > 1 else 0]
        end_dim = range(dims)[args[2] if len(args) > 2 else -1]

        # Check that we are only squeezing
        # TODO: can't actually tell which one is the non-unit axis, so we guess
        assert sum(l != 1 for l in x.shape[start_dim : end_dim + 1]) == 1
        j = next(i for i in range(start_dim, end_dim + 1) if x.shape[i] != 1)

        return [
            *[{f"args.0:{i}", f"output:{i}"} for i in range(start_dim)],
            {f"args.0:{j}", f"output:{start_dim}"},
            *[
                {f"args.0:{i}", f"output:{i - (end_dim - start_dim)}"}
                for i in range(end_dim + 1, dims)
            ],
        ]

    @register_module_handler(nn.Flatten)
    def handle_flatten_wrapper(self, module, args, kwargs, output):
        assert len(args) == 1

        return self.handle_flatten(
            torch.flatten, (args[0], module.start_dim, module.end_dim), {}, output
        )

    @register_function_handler(getattr)
    def handle_getattr(self, f, args, kwargs, output):
        assert not isinstance(output, torch.Tensor)
        return []

    @register_function_handler(op.getitem)
    def handle_getitem(self, f, args, kwargs, output):
        if not isinstance(output, torch.Tensor):
            return []

        assert len(args) == 2
        x, index = args
        if isinstance(x, (tuple, list)):
            assert isinstance(index, int)
            return [
                {f"args.0.{index}:{i}", f"output:{i}"} for i in range(len(output.shape))
            ]

        elif isinstance(x, (torch.Tensor, nn.Parameter)):
            if not isinstance(index, tuple):
                index = (index,)

            links = []
            in_axis, out_axis = 0, 0
            for axis in index:
                assert isinstance(axis, (slice, int)) or axis is None
                if isinstance(axis, slice):
                    assert axis == slice(None, None, None)
                    links.append({f"args.0:{in_axis}", f"output:{out_axis}"})
                    out_axis += 1

                elif axis is None:
                    out_axis += 1
                    in_axis -= 1

                in_axis += 1

            while in_axis < len(x.shape):
                links.append({f"args.0:{in_axis}", f"output:{out_axis}"})
                in_axis += 1
                out_axis += 1

            return links

    @register_method_handler("reshape", "view")
    def handle_reshape(self, obj, method, args, kwargs, output):
        before, after = list(obj.shape), list(output.shape)
        aaxis, baxis, abuf, bbuf, links = 0, 0, 1, 1, []

        while aaxis < len(after) and baxis < len(before):
            if abuf == bbuf and after[aaxis] == before[baxis]:
                links.append({f"object:{baxis}", f"output:{aaxis}"})
                abuf = bbuf = 1
                aaxis += 1
                baxis += 1
            elif abuf <= bbuf:
                abuf *= after[aaxis]
                aaxis += 1
            else:
                bbuf *= before[baxis]
                baxis += 1

        return links

    @register_method_handler("expand")
    def handle_expand(self, obj, method, args, kwargs, output):
        before, after = list(obj.shape), list(output.shape)
        links = []
        for i in range(1, len(before) + 1):
            if args[-i] == -1:
                links.append({f"object:{len(before) - i}", f"output:{len(after) - i}"})

        return links

    @register_function_handler(torch.permute)
    def handle_permute_function(self, f, args, kwargs, output):
        assert len(args) == 2
        obj, args = args
        assert len(obj.shape) == len(output.shape)
        links = [{f"args.0:{args[i]}", f"output:{i}"} for i in range(len(obj.shape))]
        return links

    @register_method_handler("permute")
    def handle_permute_method(self, obj, method, args, kwargs, output):
        assert len(obj.shape) == len(output.shape)
        links = [{f"object:{args[i]}", f"output:{i}"} for i in range(len(obj.shape))]
        return links

    @register_method_handler("to", "type")
    def handle_cast(self, obj, method, args, kwargs, output):
        assert obj.shape == output.shape
        links = [{f"object:{i}", f"output:{i}"} for i in range(len(obj.shape))]
        return links

    @register_method_handler("pow", "sqrt")
    def handle_method_elementwise(self, obj, method, args, kwargs, output):
        assert obj.shape == output.shape
        links = [{f"object:{i}", f"output:{i}"} for i in range(len(obj.shape))]
        return links

    @register_method_handler("size", "dim")
    def handle_size(self, obj, method, args, kwargs, output):
        assert isinstance(obj, torch.Tensor)
        return []

    @register_method_handler("tile")
    def handle_tile(self, obj, method, args, kwargs, output):
        assert not kwargs
        n, m = len(args), len(obj.shape)

        # TODO: double check this
        assert set(args[max(0, n - m) :]) == {1}
        assert output.shape[-len(obj.shape) :] == obj.shape

        return [
            {f"object:{i}", f"output:{i+max(n-m,0)}"} for i in range(len(obj.shape))
        ]

    @register_method_handler("sum", "mean")
    def handle_sum(self, obj, method, args, kwargs, output):
        if len(args) == 0:
            return []

        n = len(obj.shape)
        dims = args[0]
        if not isinstance(dims, (list, tuple)):
            dims = [
                dims,
            ]
        else:
            dims = list(dims)

        for i in range(len(dims)):
            if dims[i] < 0:
                dims[i] += n
        keepdim = kwargs.get("keepdim") or (len(args) >= 2 and args[1])

        out_axis = 0
        links = []
        for i in range(n):
            if i in dims:
                if keepdim:
                    out_axis += 1
            else:
                links.append({f"object:{i}", f"output:{out_axis}"})
                out_axis += 1

        return links

    @register_function_handler(torch.cat)
    def handle_cat(self, f, args, kwargs, output):
        # TODO: handle "out" arg
        assert (len(args) == 2) != (len(kwargs) == 1 and "dim" in kwargs)
        dim = kwargs.get("dim") or args[1]

        links = []
        for i in range(len(args[0])):
            for j in range(len(output.shape)):
                if j == dim:
                    continue
                links.append({f"args.0.{i}:{j}", f"output:{j}"})

        return links

    @register_function_handler(F.layer_norm)
    def handle_layernorm(self, f, args, kwargs, output):
        assert not (len(args) >= 3 and "weight" in kwargs)
        assert not (len(args) >= 4 and "bias" in kwargs)
        x, normalized_shape = args[:2]
        n = len(x.shape)

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        assert x.shape[-len(normalized_shape) :] == normalized_shape
        assert x.shape == output.shape

        links = []
        # TODO: should really make a helper for this sort of thing
        weight = args[2] if len(args) >= 3 else kwargs.get("weight")
        weight_name = "args.2" if len(args) >= 3 else "kwargs.weight"
        if weight is not None:
            m = len(weight.shape)
            links.extend(
                [{f"{weight_name}:{i}", f"args.0:{n - m + i}"} for i in range(m)]
            )

        bias = args[3] if len(args) >= 4 else kwargs.get("bias")
        bias_name = "args.3" if len(args) >= 4 else "kwargs.bias"
        if bias is not None:
            m = len(bias.shape)
            links.extend(
                [{f"{bias_name}:{i}", f"args.0:{n - m + i}"} for i in range(m)]
            )

        links.extend([{f"args.0:{i}", f"output:{i}"} for i in range(n)])

        return links

    @register_module_handler(nn.LayerNorm)
    def handle_layernorm2(self, module, args, kwargs, output):
        assert len(args) == 1
        x, normalized_shape = args[0], module.normalized_shape
        n = len(x.shape)

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        assert x.shape[-len(normalized_shape) :] == normalized_shape
        assert x.shape == output.shape

        links = []
        if module.elementwise_affine:
            m = len(module.weight.shape)
            assert len(module.bias.shape) == m

            links.extend(
                [{f"state.weight:{i}", f"args.0:{n - m + i}"} for i in range(m)]
            )
            links.extend([{f"state.bias:{i}", f"args.0:{n - m + i}"} for i in range(m)])

        links.extend([{f"args.0:{i}", f"output:{i}"} for i in range(n)])

        return links

    @register_module_handler(nn.MultiheadAttention)
    def handle_multiheadattention(self, module, args, kwargs, output):
        print("WARNING " * 10)
        print("MultiheadAttention is not supported!")
        return []

    def get_permutation_spec(self) -> PermutationSpec:
        def process_node_path(path):
            key, axis = path.replace("node.", "", 1).split(":")
            return Axis(key, int(axis))

        def process_state_path(path):
            key, axis = path.replace("state.", "", 1).split(":")
            return Axis(key, int(axis))

        # We assume no invariance of inputs or outputs
        io_axes = set.union(
            {
                f"placeholders.{i}:{j}"
                for i, t in enumerate(self.inputs)
                for j in range(len(t.shape))
            },
            {f"result:{i}" for i in range(len(self.result.shape))},
        )

        groups = self.axis_groups.groups().values()
        groups = [
            PermutationGroup(
                None,  # unknown
                {process_state_path(x) for x in g if x.startswith("state")},
                {process_node_path(x) for x in g if x.startswith("node")},
            )
            for g in groups
            if not g.intersection(io_axes)
        ]

        groups = [g for g in groups if len(g.state) > 1]

        state_dict = self.module.state_dict()
        order = {k: i for i, k in enumerate(state_dict.keys())}

        # this is the convention that axis 0 is the output and axis 1 is the input
        groups = {
            min(g.state, key=lambda k: (order[k.key], -k.axis)): g for g in groups
        }
        # groups.sort(key=lambda g: min(map(lambda k: (order[k[0]], -k[1]), g)))

        # populate sizes
        for key in list(groups.keys()):
            group = groups[key]
            sizes = {state_dict[k.key].shape[k.axis] for k in group.state}
            assert len(sizes) == 1
            if sizes == {1}:
                del groups[key]
            else:
                groups[key].size = next(iter(sizes))

        return groups

    def test_permutation_spec(self, spec: PermutationSpec, x):
        state_dict = deepcopy(self.module.state_dict())

        with torch.no_grad():
            reference = self.module(x)
            fail = set()

            for i, (key, pg) in enumerate(spec.items()):
                print(f"Testing group {i} ({key}, size {pg.size})")
                # TODO: make sure P is not identity
                P = torch.randperm(pg.size)

                apply_perm({key: P}, spec, self.module, inplace=True)

                print((self.module(x) - reference).abs().max())

                if not torch.allclose(self.module(x), reference, rtol=1e-2, atol=1e-3):
                    print(f"Group {i} failed!")
                    fail.add(key)

                self.module.load_state_dict(state_dict)

            if fail:
                print("At least one test failed!")
                print(fail)
                return False

            else:
                print("All tests passed")
                return True


def get_permutation_spec(
    model: nn.Module, inputs_or_shapes: InputsOrShapes, verbose=False
) -> PermutationSpec:
    device = next(iter(model.parameters())).device
    inputs = [
        torch.randn(*ios).to(device) if isinstance(ios, tuple) else ios.to(device)
        for ios in inputs_or_shapes
    ]
    gm = torch.fx.symbolic_trace(model)
    pp = PermutationProp(gm, verbose=verbose)
    pp.run(*inputs)
    return pp.get_permutation_spec()