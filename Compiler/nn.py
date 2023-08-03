from tensor import *
import warnings
import functools
from collections import OrderedDict, namedtuple
from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List
from tensor import Tensor
import math
import re
from itertools import islice
from Compiler import mpc_math, util
from Compiler.types import *
from Compiler.types import _unreduced_squant
from Compiler.library import *
from Compiler.util import is_zero, tree_reduce
from Compiler.comparison import CarryOutRawLE
from Compiler.GC.types import sbitint
from functools import reduce


def _addindent(s_, numSpaces):
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s


r"""This tracks hooks common to all modules that are executed immediately before
.registering the buffer/module/parameter"""
_global_buffer_registration_hooks: Dict[int, Callable] = OrderedDict()
_global_module_registration_hooks: Dict[int, Callable] = OrderedDict()
_global_parameter_registration_hooks: Dict[int, Callable] = OrderedDict()


class Parameter(Tensor):
    r"""A kind of Tensor that is to be considered a module parameter.

    Parameters are :class:`~torch.Tensor` subclasses, that have a
    very special property when used with :class:`Module` s - when they're
    assigned as Module attributes they are automatically added to the list of
    its parameters, and will appear e.g. in :meth:`~Module.parameters` iterator.
    Assigning a Tensor doesn't have such effect. This is because one might
    want to cache some temporary state, like last hidden state of the RNN, in
    the model. If there was no such class as :class:`Parameter`, these
    temporaries would get registered too.

    Args:
        data (Tensor): parameter tensor.
        requires_grad (bool, optional): if the parameter requires gradient. Note that
            the torch.no_grad() context does NOT affect the default behavior of
            Parameter creation--the Parameter will still have `requires_grad=True` in
            :class:`~no_grad` mode. See :ref:`locally-disable-grad-doc` for more
            details. Default: `True`
    """
    def __new__(cls, data=None, requires_grad=True):
        if data is None:
            raise RuntimeError(f"Data cannot be None when creating Parameters")
        if type(data) is Tensor or type(data) is Parameter:
            # For ease of BC maintenance, keep this path for standard Tensor.
            # Eventually (tm), we should change the behavior for standard Tensor to match.
            data.set_req_grad(requires_grad)
            return data
        else:
            raise RuntimeError(f"Parameter can only be created from tensor or parameter")

    def __repr__(self):
        return 'Parameter containing:\n' + super().__repr__()


class Module():
    def __init__(self):
        self.training = True
        self._parameters: Dict[str, Optional[Parameter]] = OrderedDict()
        self._buffers: Dict[str, Optional[Tensor]] = OrderedDict()
        self._modules: Dict[str, Optional['Module']] = OrderedDict()
        self._non_persistent_buffers_set: Set[str] = set()

    def register_buffer(self, name: str, tensor: Optional[Tensor], persistent: bool = True) -> None:
        r"""Adds a buffer to the module.

        This is typically used to register a buffer that should not to be
        considered a model parameter. For example, BatchNorm's ``running_mean``
        is not a parameter, but is part of the module's state. Buffers, by
        default, are persistent and will be saved alongside parameters. This
        behavior can be changed by setting :attr:`persistent` to ``False``. The
        only difference between a persistent buffer and a non-persistent buffer
        is that the latter will not be a part of this module's
        :attr:`state_dict`.

        Buffers can be accessed as attributes using given names.

        Args:
            name (str): name of the buffer. The buffer can be accessed
                from this module using the given name
            tensor (Tensor or None): buffer to be registered. If ``None``, then operations
                that run on buffers, such as :attr:`cuda`, are ignored. If ``None``,
                the buffer is **not** included in the module's :attr:`state_dict`.
            persistent (bool): whether the buffer is part of this module's
                :attr:`state_dict`.

        Example::
            >>> # xdoctest: +SKIP("undefined vars")
            >>> self.register_buffer('running_mean', torch.zeros(num_features))
        """

        if '_buffers' not in self.__dict__:
            raise AttributeError(
                "cannot assign buffer before Module.__init__() call")
        elif not isinstance(name, str):
            raise TypeError("buffer name should be a string. "
                            "Got {}".format((type(name).__name__)))
        elif '.' in name:
            raise KeyError("buffer name can't contain \".\"")
        elif name == '':
            raise KeyError("buffer name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._buffers:
            raise KeyError("attribute '{}' already exists".format(name))
        elif tensor is not None and not isinstance(tensor, Tensor):
            raise TypeError("cannot assign '{}' object to buffer '{}' "
                            "(Tensor or None required)"
                            .format((type(tensor).__name__), name))
        else:
            self._buffers[name] = tensor

    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        r"""Adds a parameter to the module.

        The parameter can be accessed as an attribute using given name.

        Args:
            name (str): name of the parameter. The parameter can be accessed
                from this module using the given name
            param (Parameter or None): parameter to be added to the module. If
                ``None``, then operations that run on parameters, such as :attr:`cuda`,
                are ignored. If ``None``, the parameter is **not** included in the
                module's :attr:`state_dict`.
        """
        if '_parameters' not in self.__dict__:
            raise AttributeError(
                "cannot assign parameter before Module.__init__() call")

        elif not isinstance(name, str):
            raise TypeError("parameter name should be a string. "
                            "Got {}".format(type(name).__name__))
        elif '.' in name:
            raise KeyError("parameter name can't contain \".\"")
        elif name == '':
            raise KeyError("parameter name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._parameters:
            raise KeyError("attribute '{}' already exists".format(name))

        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, Parameter):
            raise TypeError("cannot assign '{}' object to parameter '{}' "
                            "(torch.nn.Parameter or None required)"
                            .format(type(param).__name__, name))
        elif param.grad_fn:
            raise ValueError(
                "Cannot assign non-leaf Tensor to parameter '{0}'. Model "
                "parameters must be created explicitly. To express '{0}' "
                "as a function of another Tensor, compute the value in "
                "the forward() method.".format(name))
        else:
            for hook in _global_parameter_registration_hooks.values():
                output = hook(self, name, param)
                if output is not None:
                    param = output
            self._parameters[name] = param

    def add_module(self, name: str, module: Optional['Module']) -> None:
        r"""Adds a child module to the current module.

        The module can be accessed as an attribute using the given name.

        Args:
            name (str): name of the child module. The child module can be
                accessed from this module using the given name
            module (Module): child module to be added to the module.
        """
        if not isinstance(module, Module) and module is not None:
            raise TypeError("{} is not a Module subclass".format(
                type(module).__name__))
        elif not isinstance(name, str):
            raise TypeError("module name should be a string. Got {}".format(
                type(name).__name__))
        elif hasattr(self, name) and name not in self._modules:
            raise KeyError("attribute '{}' already exists".format(name))
        elif '.' in name:
            raise KeyError("module name can't contain \".\", got: {}".format(name))
        elif name == '':
            raise KeyError("module name can't be empty string \"\"")
        for hook in _global_module_registration_hooks.values():
            output = hook(self, name, module)
            if output is not None:
                module = output
        self._modules[name] = module

    def register_module(self, name: str, module: Optional['Module']) -> None:
        r"""Alias for :func:`add_module`."""
        self.add_module(name, module)

    def get_submodule(self, target: str) -> "Module":
        """
        Returns the submodule given by ``target`` if it exists,
        otherwise throws an error.

        For example, let's say you have an ``nn.Module`` ``A`` that
        looks like this:

        .. code-block:: text

            A(
                (net_b): Module(
                    (net_c): Module(
                        (conv): Conv2d(16, 33, kernel_size=(3, 3), stride=(2, 2))
                    )
                    (linear): Linear(in_features=100, out_features=200, bias=True)
                )
            )

        (The diagram shows an ``nn.Module`` ``A``. ``A`` has a nested
        submodule ``net_b``, which itself has two submodules ``net_c``
        and ``linear``. ``net_c`` then has a submodule ``conv``.)

        To check whether or not we have the ``linear`` submodule, we
        would call ``get_submodule("net_b.linear")``. To check whether
        we have the ``conv`` submodule, we would call
        ``get_submodule("net_b.net_c.conv")``.

        The runtime of ``get_submodule`` is bounded by the degree
        of module nesting in ``target``. A query against
        ``named_modules`` achieves the same result, but it is O(N) in
        the number of transitive modules. So, for a simple check to see
        if some submodule exists, ``get_submodule`` should always be
        used.

        Args:
            target: The fully-qualified string name of the submodule
                to look for. (See above example for how to specify a
                fully-qualified string.)

        Returns:
            torch.nn.Module: The submodule referenced by ``target``

        Raises:
            AttributeError: If the target string references an invalid
                path or resolves to something that is not an
                ``nn.Module``
        """
        if target == "":
            return self

        atoms: List[str] = target.split(".")
        mod: Module = self

        for item in atoms:

            if not hasattr(mod, item):
                raise AttributeError(mod._get_name() + " has no "
                                     "attribute `" + item + "`")

            mod = getattr(mod, item)

            if not isinstance(mod, Module):
                raise AttributeError("`" + item + "` is not "
                                     "an nn.Module")

        return mod

    def get_parameter(self, target: str) -> "Parameter":
        """
        Returns the parameter given by ``target`` if it exists,
        otherwise throws an error.

        See the docstring for ``get_submodule`` for a more detailed
        explanation of this method's functionality as well as how to
        correctly specify ``target``.

        Args:
            target: The fully-qualified string name of the Parameter
                to look for. (See ``get_submodule`` for how to specify a
                fully-qualified string.)

        Returns:
            Parameter: The Parameter referenced by ``target``

        Raises:
            AttributeError: If the target string references an invalid
                path or resolves to something that is not an
                ``nn.Parameter``
        """
        module_path, _, param_name = target.rpartition(".")

        mod: Module = self.get_submodule(module_path)

        if not hasattr(mod, param_name):
            raise AttributeError(mod._get_name() + " has no attribute `"
                                 + param_name + "`")

        param: Parameter = getattr(mod, param_name)

        if not isinstance(param, Parameter):
            raise AttributeError("`" + param_name + "` is not an "
                                 "nn.Parameter")

        return param

    def get_buffer(self, target: str) -> "Tensor":
        """
        Returns the buffer given by ``target`` if it exists,
        otherwise throws an error.

        See the docstring for ``get_submodule`` for a more detailed
        explanation of this method's functionality as well as how to
        correctly specify ``target``.

        Args:
            target: The fully-qualified string name of the buffer
                to look for. (See ``get_submodule`` for how to specify a
                fully-qualified string.)

        Returns:
            torch.Tensor: The buffer referenced by ``target``

        Raises:
            AttributeError: If the target string references an invalid
                path or resolves to something that is not a
                buffer
        """
        module_path, _, buffer_name = target.rpartition(".")

        mod: Module = self.get_submodule(module_path)

        if not hasattr(mod, buffer_name):
            raise AttributeError(mod._get_name() + " has no attribute `"
                                 + buffer_name + "`")

        buffer: Tensor = getattr(mod, buffer_name)

        if buffer_name not in mod._buffers:
            raise AttributeError("`" + buffer_name + "` is not a buffer")

        return buffer

    def __getattr__(self, name: str) -> Union[Tensor, 'Module']:
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

    def __setattr__(self, name: str, value: Union[Tensor, 'Module']) -> None:
        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                    else:
                        d.discard(name)

        params = self.__dict__.get('_parameters')
        if isinstance(value, Parameter):
            if params is None:
                raise AttributeError(
                    "cannot assign parameters before Module.__init__() call")
            remove_from(self.__dict__, self._buffers, self._modules, self._non_persistent_buffers_set)
            self.register_parameter(name, value)
        elif params is not None and name in params:
            self.register_parameter(name, value)
        else:
            modules = self.__dict__.get('_modules')
            if isinstance(value, Module):
                if modules is None:
                    raise AttributeError(
                        "cannot assign module before Module.__init__() call")
                remove_from(self.__dict__, self._parameters, self._buffers, self._non_persistent_buffers_set)
                modules[name] = value
            else:
                buffers = self.__dict__.get('_buffers')
                object.__setattr__(self, name, value)

    def __delattr__(self, name):
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._buffers:
            del self._buffers[name]
            self._non_persistent_buffers_set.discard(name)
        elif name in self._modules:
            del self._modules[name]
        else:
            object.__delattr__(self, name)

    def _named_members(self, get_members_fn, prefix='', recurse=True):
        r"""Helper method for yielding various names + members of modules."""
        memo = set()
        modules = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                if v is None or v in memo:
                    continue
                memo.add(v)
                name = module_prefix + ('.' if module_prefix else '') + k
                yield name, v

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        r"""Returns an iterator over module parameters.

        This is typically passed to an optimizer.

        Args:
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.

        Yields:
            Parameter: module parameter

        Example::

            >>> for param in model.parameters():
            >>>     print(type(param), param.size())
            <class 'torch.Tensor'> (20L,)
            <class 'torch.Tensor'> (20L, 1L, 5L, 5L)

        """
        for name, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Parameter]]:
        r"""Returns an iterator over module parameters, yielding both the
        name of the parameter as well as the parameter itself.

        Args:
            prefix (str): prefix to prepend to all parameter names.
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.

        Yields:
            (string, Parameter): Tuple containing the name and parameter

        Example::

            >>> for name, param in self.named_parameters():
            >>>    if name in ['bias']:
            >>>        print(param.size())

        """
        gen = self._named_members(
            lambda module: module._parameters.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

    def buffers(self, recurse: bool = True) -> Iterator[Tensor]:
        r"""Returns an iterator over module buffers.

        Args:
            recurse (bool): if True, then yields buffers of this module
                and all submodules. Otherwise, yields only buffers that
                are direct members of this module.

        Yields:
            torch.Tensor: module buffer

        Example::

            >>> for buf in model.buffers():
            >>>     print(type(buf), buf.size())
            <class 'torch.Tensor'> (20L,)
            <class 'torch.Tensor'> (20L, 1L, 5L, 5L)

        """
        for _, buf in self.named_buffers(recurse=recurse):
            yield buf

    def named_buffers(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Tensor]]:
        r"""Returns an iterator over module buffers, yielding both the
        name of the buffer as well as the buffer itself.

        Args:
            prefix (str): prefix to prepend to all buffer names.
            recurse (bool): if True, then yields buffers of this module
                and all submodules. Otherwise, yields only buffers that
                are direct members of this module.

        Yields:
            (string, torch.Tensor): Tuple containing the name and buffer

        Example::

            >>> for name, buf in self.named_buffers():
            >>>    if name in ['running_var']:
            >>>        print(buf.size())

        """
        gen = self._named_members(
            lambda module: module._buffers.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

    def children(self) -> Iterator['Module']:
        r"""Returns an iterator over immediate children modules.

        Yields:
            Module: a child module
        """
        for name, module in self.named_children():
            yield module

    def named_children(self) -> Iterator[Tuple[str, 'Module']]:
        r"""Returns an iterator over immediate children modules, yielding both
        the name of the module as well as the module itself.

        Yields:
            (string, Module): Tuple containing a name and child module

        Example::

            >>> for name, module in model.named_children():
            >>>     if name in ['conv4', 'conv5']:
            >>>         print(module)

        """
        memo = set()
        for name, module in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield name, module

    def modules(self) -> Iterator['Module']:
        r"""Returns an iterator over all modules in the network.

        Yields:
            Module: a module in the network

        Note:
            Duplicate modules are returned only once. In the following
            example, ``l`` will be returned only once.

        Example::

            >>> l = nn.Linear(2, 2)
            >>> net = nn.Sequential(l, l)
            >>> for idx, m in enumerate(net.modules()):
                    print(idx, '->', m)

            0 -> Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            )
            1 -> Linear(in_features=2, out_features=2, bias=True)

        """
        for _, module in self.named_modules():
            yield module

    def named_modules(self, memo: Optional[Set['Module']] = None, prefix: str = '', remove_duplicate: bool = True):
        r"""Returns an iterator over all modules in the network, yielding
        both the name of the module as well as the module itself.

        Args:
            memo: a memo to store the set of modules already added to the result
            prefix: a prefix that will be added to the name of the module
            remove_duplicate: whether to remove the duplicated module instances in the result
                or not

        Yields:
            (string, Module): Tuple of name and module

        Note:
            Duplicate modules are returned only once. In the following
            example, ``l`` will be returned only once.

        Example::

            >>> l = nn.Linear(2, 2)
            >>> net = nn.Sequential(l, l)
            >>> for idx, m in enumerate(net.named_modules()):
                    print(idx, '->', m)

            0 -> ('', Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            ))
            1 -> ('0', Linear(in_features=2, out_features=2, bias=True))

        """

        if memo is None:
            memo = set()
        if self not in memo:
            if remove_duplicate:
                memo.add(self)
            yield prefix, self
            for name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ('.' if prefix else '') + name
                for m in module.named_modules(memo, submodule_prefix, remove_duplicate):
                    yield m

    def train(self: T, mode: bool = True) -> T:
        r"""Sets the module in training mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def setup(self, data_loader):
        # todo, setup tensor space of the model, call it before training or evaluation
        self.forward()
        train()
        return self

    def requires_grad_(self: T, requires_grad: bool = True) -> T:
        r"""Change if autograd should record operations on parameters in this
        module.

        This method sets the parameters' :attr:`requires_grad` attributes
        in-place.

        This method is helpful for freezing part of the module for finetuning
        or training parts of a model individually (e.g., GAN training).

        See :ref:`locally-disable-grad-doc` for a comparison between
        `.requires_grad_()` and several similar mechanisms that may be confused with it.

        Args:
            requires_grad (bool): whether autograd should record operations on
                                  parameters in this module. Default: ``True``.

        Returns:
            Module: self
        """
        for p in self.parameters():
            p.requires_grad_(requires_grad)
        return self

    def zero_grad(self, set_to_none: bool = False) -> None:
        r"""Sets gradients of all model parameters to zero. See similar function
        under :class:`torch.optim.Optimizer` for more context.

        Args:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                See :meth:`torch.optim.Optimizer.zero_grad` for details.
        """
        if getattr(self, '_is_replica', False):
            warnings.warn(
                "Calling .zero_grad() from a module created with nn.DataParallel() has no effect. "
                "The parameters are copied (in a differentiable manner) from the original module. "
                "This means they are not leaf nodes in autograd and so don't accumulate gradients. "
                "If you need gradients in your forward method, consider using autograd.grad instead.")

        for p in self.parameters():
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    if p.grad.grad_fn is not None:
                        p.grad.detach_()
                    else:
                        p.grad.requires_grad_(False)
                    p.grad.zero_()

    def _get_name(self):
        return self.__class__.__name__

    def extra_repr(self) -> str:
        r"""Set the extra representation of the module

        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.
        """
        return ''

    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str

    def __dir__(self):
        module_attrs = dir(self.__class__)
        attrs = list(self.__dict__.keys())
        parameters = list(self._parameters.keys())
        modules = list(self._modules.keys())
        buffers = list(self._buffers.keys())
        keys = module_attrs + attrs + parameters + modules + buffers

        # Eliminate attrs that are not legal Python variable names
        keys = [key for key in keys if not key[0].isdigit()]

        return sorted(keys)

    def _call_impl(self, *args, **kwargs):
        forward_call = self.forward
        break_point()
        result = forward_call(*args, **kwargs)
        break_point()

        return result

    __call__: Callable[..., Any] = _call_impl


class Sequential(Module):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the
    constructor. Alternatively, an ``OrderedDict`` of modules can be
    passed in. The ``forward()`` method of ``Sequential`` accepts any
    input and forwards it to the first module it contains. It then
    "chains" outputs to inputs sequentially for each subsequent module,
    finally returning the output of the last module.

    The value a ``Sequential`` provides over manually calling a sequence
    of modules is that it allows treating the whole container as a
    single module, such that performing a transformation on the
    ``Sequential`` applies to each of the modules it stores (which are
    each a registered submodule of the ``Sequential``).

    What's the difference between a ``Sequential`` and a
    :class:`torch.nn.ModuleList`? A ``ModuleList`` is exactly what it
    sounds like--a list for storing ``Module`` s! On the other hand,
    the layers in a ``Sequential`` are connected in a cascading way.

    Example::

        # Using Sequential to create a small model. When `model` is run,
        # input will first be passed to `Conv2d(1,20,5)`. The output of
        # `Conv2d(1,20,5)` will be used as the input to the first
        # `ReLU`; the output of the first `ReLU` will become the input
        # for `Conv2d(20,64,5)`. Finally, the output of
        # `Conv2d(20,64,5)` will be used as input to the second `ReLU`
        model = nn.Sequential(
                  nn.Conv2d(1,20,5),
                  nn.ReLU(),
                  nn.Conv2d(20,64,5),
                  nn.ReLU()
                )

        # Using Sequential with OrderedDict. This is functionally the
        # same as the above code
        model = nn.Sequential(OrderedDict([
                  ('conv1', nn.Conv2d(1,20,5)),
                  ('relu1', nn.ReLU()),
                  ('conv2', nn.Conv2d(20,64,5)),
                  ('relu2', nn.ReLU())
                ]))
    """

    _modules: Dict[str, Module]  # type: ignore[assignment]

    @overload
    def __init__(self, *args: Module) -> None:
        ...

    @overload
    def __init__(self, arg: 'OrderedDict[str, Module]') -> None:
        ...

    def __init__(self, *args):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def _get_item_by_idx(self, iterator, idx) -> T:
        """Get the idx-th item of the iterator"""
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError('index {} is out of range'.format(idx))
        idx %= size
        return next(islice(iterator, idx, None))

    def __getitem__(self, idx: Union[slice, int]) -> Union['Sequential', T]:
        if isinstance(idx, slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(), idx)

    def __setitem__(self, idx: int, module: Module) -> None:
        key: str = self._get_item_by_idx(self._modules.keys(), idx)
        return setattr(self, key, module)

    def __delitem__(self, idx: Union[slice, int]) -> None:
        if isinstance(idx, slice):
            for key in list(self._modules.keys())[idx]:
                delattr(self, key)
        else:
            key = self._get_item_by_idx(self._modules.keys(), idx)
            delattr(self, key)
        # To preserve numbering
        str_indices = [str(i) for i in range(len(self._modules))]
        self._modules = OrderedDict(list(zip(str_indices, self._modules.values())))

    def __len__(self) -> int:
        return len(self._modules)

    def __add__(self, other) -> 'Sequential':
        if isinstance(other, Sequential):
            ret = Sequential()
            for layer in self:
                ret.append(layer)
            for layer in other:
                ret.append(layer)
            return ret
        else:
            raise ValueError('add operator supports only objects '
                             'of Sequential class, but {} is given.'.format(
                                 str(type(other))))

    def pop(self, key: Union[int, slice]) -> Module:
        v = self[key]
        del self[key]
        return v

    def __iadd__(self, other) -> 'Sequential':
        if isinstance(other, Sequential):
            offset = len(self)
            for i, module in enumerate(other):
                self.add_module(str(i + offset), module)
            return self
        else:
            raise ValueError('add operator supports only objects '
                             'of Sequential class, but {} is given.'.format(
                                 str(type(other))))

    def __mul__(self, other: int) -> 'Sequential':
        if not isinstance(other, int):
            raise TypeError(f"unsupported operand type(s) for *: {type(self)} and {type(other)}")
        elif (other <= 0):
            raise ValueError(f"Non-positive multiplication factor {other} for {type(self)}")
        else:
            combined = Sequential()
            offset = 0
            for _ in range(other):
                for module in self:
                    combined.add_module(str(offset), module)
                    offset += 1
            return combined

    def __rmul__(self, other: int) -> 'Sequential':
        return self.__mul__(other)

    def __imul__(self, other: int) -> 'Sequential':
        if not isinstance(other, int):
            raise TypeError(f"unsupported operand type(s) for *: {type(self)} and {type(other)}")
        elif (other <= 0):
            raise ValueError(f"Non-positive multiplication factor {other} for {type(self)}")
        else:
            len_original = len(self)
            offset = len(self)
            for _ in range(other - 1):
                for i in range(len_original):
                    self.add_module(str(i + offset), self._modules[str(i)])
                offset += len_original
            return self

    def __dir__(self):
        keys = super().__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    # NB: We can't really type check this function as the type of input
    # may change dynamically (as is tested in
    # TestScript.test_sequential_intermediary_types).  Cannot annotate
    # with Any as TorchScript expects a more precise type
    def forward(self, input):
        for module in self:
            input = module(input)
        return input

    def append(self, module: Module) -> 'Sequential':
        r"""Appends a given module to the end.

        Args:
            module (nn.Module): module to append
        """
        self.add_module(str(len(self)), module)
        return self

    def insert(self, index: int, module: Module) -> 'Sequential':
        if not isinstance(module, Module):
            raise AssertionError(
                'module should be of type: {}'.format(Module))
        n = len(self._modules)
        if not (-n <= index <= n):
            raise IndexError(
                'Index out of range: {}'.format(index))
        if index < 0:
            index += n
        for i in range(n, index, -1):
            self._modules[str(i)] = self._modules[str(i - 1)]
        self._modules[str(index)] = module
        return self

    def extend(self, sequential) -> 'Sequential':
        for layer in sequential:
            self.append(layer)
        return self
