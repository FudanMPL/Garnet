import warnings
from typing import Any, Callable, Dict, List, Tuple
from collections import OrderedDict, defaultdict, abc as container_abcs
from tensor import *
import tensor as TS
from copy import deepcopy
from itertools import chain
class _RequiredParameter:
    """Singleton class representing a required parameter for an Optimizer."""
    def __repr__(self):
        return "<required parameter>"

required = _RequiredParameter()
class Optimizer:
    r"""Base class for all optimizers.

    .. warning::
        Parameters need to be specified as collections that have a deterministic
        ordering that is consistent between runs. Examples of objects that don't
        satisfy those properties are sets and iterators over values of dictionaries.

    Args:
        params (iterable): an iterable of :class:`torch.Tensor` s or
            :class:`dict` s. Specifies what Tensors should be optimized.
        defaults: (dict): a dict containing default values of optimization
            options (used when a parameter group doesn't specify them).
    """

    def __init__(self, params, defaults):
        self.defaults = defaults
        self._optimizer_step_pre_hooks: Dict[int, Callable] = OrderedDict()
        self._optimizer_step_post_hooks: Dict[int, Callable] = OrderedDict()


        if isinstance(params, Tensor):
            raise TypeError("params argument given to the optimizer should be "
                            "an iterable of Tensors or dicts, but got " +
                           (type(params).__name__))

        self.state = defaultdict(dict)
        self.param_groups = []

        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        for param_group in param_groups:
            self.add_param_group(param_group)

        # Allows _cuda_graph_capture_health_check to rig a poor man's TORCH_WARN_ONCE in python,
        # which I don't think exists
        # https://github.com/pytorch/pytorch/issues/72948
        self._warned_capturable_if_run_uncaptured = True


    def __getstate__(self):
        return {
            'defaults': self.defaults,
            'state': self.state,
            'param_groups': self.param_groups,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)
        if '_optimizer_step_pre_hooks' not in self.__dict__:
            self._optimizer_step_pre_hooks = OrderedDict()
        if '_optimizer_step_post_hooks' not in self.__dict__:
            self._optimizer_step_post_hooks = OrderedDict()
        self._patch_step_function()  # To support multiprocessing pickle/unpickle
        self.defaults.setdefault('differentiable', False)

    def __repr__(self):
        format_string = self.__class__.__name__ + ' ('
        for i, group in enumerate(self.param_groups):
            format_string += '\n'
            format_string += 'Parameter Group {0}\n'.format(i)
            for key in sorted(group.keys()):
                if key != 'params':
                    format_string += '    {0}: {1}\n'.format(key, group[key])
        format_string += ')'
        return format_string

    def state_dict(self):
        r"""Returns the state of the optimizer as a :class:`dict`.

        It contains two entries:

        * state - a dict holding current optimization state. Its content
            differs between optimizer classes.
        * param_groups - a list containing all parameter groups where each
            parameter group is a dict
        """
        # Save order indices instead of Tensors
        param_mappings = {}
        start_index = 0

        def pack_group(group):
            nonlocal start_index
            packed = {k: v for k, v in group.items() if k != 'params'}
            param_mappings.update({id(p): i for i, p in enumerate(group['params'], start_index)
                                   if id(p) not in param_mappings})
            packed['params'] = [param_mappings[id(p)] for p in group['params']]
            start_index += len(packed['params'])
            return packed
        param_groups = [pack_group(g) for g in self.param_groups]
        # Remap state to use order indices as keys
        packed_state = {(param_mappings[id(k)] if isinstance(k, Tensor) else k): v
                        for k, v in self.state.items()}
        return {
            'state': packed_state,
            'param_groups': param_groups,
        }

    @staticmethod
    def _process_value_according_to_param_policy(param: Tensor, value: Tensor, param_id: int = None,
                                                 param_groups: List[Dict[Any, Any]] = None, key=None) -> Tensor:
        # Floating-point types are a bit special here. They are the only ones
        # that are assumed to always match the type of params.
        # Make sure state['step'] is not casted https://github.com/pytorch/pytorch/issues/74424
        # UNLESS fused or capturable, see note [special device hosting for step]
        fused = False
        capturable = False
        for pg in param_groups:
            if param_id in pg["params"]:
                fused = pg["fused"] if "fused" in pg else False
                capturable = pg["capturable"] if "capturable" in pg else False
                break

        if key != "step" or capturable or fused:
            if param.is_floating_point():
                return value.to(dtype=param.dtype, device=param.device)
            return value.to(device=param.device)
        return value

    def load_state_dict(self, state_dict):
        r"""Loads the optimizer state.

        Args:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        state_dict = deepcopy(state_dict)
        # Validate the state_dict
        groups = self.param_groups
        saved_groups = state_dict['param_groups']

        if len(groups) != len(saved_groups):
            raise ValueError("loaded state dict has a different number of "
                             "parameter groups")
        param_lens = (len(g['params']) for g in groups)
        saved_lens = (len(g['params']) for g in saved_groups)
        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
            raise ValueError("loaded state dict contains a parameter group "
                             "that doesn't match the size of optimizer's group")

        # Update the state
        id_map = dict(zip(chain.from_iterable((g['params'] for g in saved_groups)),
                      chain.from_iterable((g['params'] for g in groups))))

        def cast(param, value, param_id=None, param_groups=None, key=None):
            r"""Make a deep copy of value, casting all tensors to device of param."""
            if isinstance(value, Tensor):
                return Optimizer._process_value_according_to_param_policy(param, value, param_id, param_groups, key)
            elif isinstance(value, dict):
                return {k: cast(param, v, param_id=param_id, param_groups=param_groups, key=k) for k, v in value.items()}
            elif isinstance(value, container_abcs.Iterable):
                return type(value)(cast(param, v, param_id=param_id, param_groups=param_groups) for v in value)
            else:
                return value

        # Copy state assigned to params (and cast tensors to appropriate types).
        # State that is not assigned to params is copied as is (needed for
        # backward compatibility).
        state = defaultdict(dict)
        for k, v in state_dict['state'].items():
            if k in id_map:
                param = id_map[k]
                state[param] = cast(param, v, param_id=k, param_groups=state_dict['param_groups'])
            else:
                state[k] = v

        # Update parameter groups, setting their 'params' value
        def update_group(group, new_group):
            new_group['params'] = group['params']
            return new_group
        param_groups = [
            update_group(g, ng) for g, ng in zip(groups, saved_groups)]
        self.__setstate__({'state': state, 'param_groups': param_groups})

    def zero_grad(self, set_to_none: bool = True):
        r"""Resets the gradients of all optimized :class:`torch.Tensor` s.

        Args:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                This will in general have lower memory footprint, and can modestly improve performance.
                However, it changes certain behaviors. For example:
                1. When the user tries to access a gradient and perform manual ops on it,
                a None attribute or a Tensor full of 0s will behave differently.
                2. If the user requests ``zero_grad(set_to_none=True)`` followed by a backward pass, ``.grad``\ s
                are guaranteed to be None for params that did not receive a gradient.
                3. ``torch.optim`` optimizers have a different behavior if the gradient is 0 or None
                (in one case it does the step with a gradient of 0 and in the other it skips
                the step altogether).
        """
        # for group in self.param_groups:
        #     for p in group['params']:
        #         if p.grad is not None:
        #                 p.zero_grad()
        for name, p in TS.tensors.items():
            p.zero_grad()

    def step(self, closure):
        r"""Performs a single optimization step (parameter update).

        Args:
            closure (Callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.

        .. note::
            Unless otherwise specified, this function should not modify the
            ``.grad`` field of the parameters.
        """
        raise NotImplementedError

    def add_param_group(self, param_group):
        r"""Add a param group to the :class:`Optimizer` s `param_groups`.

        This can be useful when fine tuning a pre-trained network as frozen layers can be made
        trainable and added to the :class:`Optimizer` as training progresses.

        Args:
            param_group (dict): Specifies what Tensors should be optimized along with group
                specific optimization options.
        """
        assert isinstance(param_group, dict), "param group must be a dict"

        params = param_group['params']
        if isinstance(params, Tensor):
            param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError('optimizer parameters need to be organized in ordered collections, but '
                            'the ordering of tensors in sets will change between runs. Please use a list instead.')
        else:
            param_group['params'] = list(params)

        for param in param_group['params']:
            if not isinstance(param, Tensor):
                raise TypeError("optimizer can only optimize Tensors, "
                                "but one of the params is " + type(param).__name__)

        for name, default in self.defaults.items():
            if default is required and name not in param_group:
                raise ValueError("parameter group didn't specify a value of required optimization parameter " +
                                 name)
            else:
                param_group.setdefault(name, default)

        params = param_group['params']
        if len(params) != len(set(params)):
            warnings.warn("optimizer contains a parameter group with duplicate parameters; "
                          "in future, this will cause an error; "
                          "see github.com/pytorch/pytorch/issues/40967 for more information", stacklevel=3)

        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group['params']))

        if not param_set.isdisjoint(set(param_group['params'])):
            raise ValueError("some parameters appear in more than one parameter group")

        self.param_groups.append(param_group)


class SGD(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, *, maximize: bool = False, foreach: Optional[bool] = None,
                 differentiable: bool = False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                weight_decay=weight_decay, nesterov=nesterov,
                maximize=maximize, foreach=foreach,
                differentiable=differentiable)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)
        if momentum != 0:
            for group in self.param_groups:
                self.init_momentum(group)
        self.iter = regint(0)
        
    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)
            group.setdefault('maximize', False)
            group.setdefault('differentiable', False)

    def init_momentum(self, group):
        for p in group['params']:
            if p.grad is not None:
                state = self.state[p]
                buf = p.grad.same_shape()
                buf.assign_all(0)
                state['momentum_buffer'] = buf
                
                
    def _init_group(self, group, params_with_grad, d_p_list, momentum_buffer_list):
        has_sparse_grad = False
        for p in group['params']:
            if p.grad is not None:
                params_with_grad.append(p)
                d_p_list.append(p.grad)

                state = self.state[p]
                if group['momentum'] !=0:
                    if 'momentum_buffer' not in state:
                        raise CompilerError("momentum should be inited")
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

        return has_sparse_grad
    @buildingblock("sgd")
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        # if closure is not None:
        #     with torch.enable_grad():
        #         loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []

            has_sparse_grad = self._init_group(group, params_with_grad, d_p_list, momentum_buffer_list)

            sgd(params_with_grad,
                d_p_list,
                momentum_buffer_list,
                iter=self.iter,
                weight_decay=group['weight_decay'],
                momentum=group['momentum'],
                lr=group['lr'],
                dampening=group['dampening'],
                nesterov=group['nesterov'],
                maximize=group['maximize'],
                has_sparse_grad=has_sparse_grad,
                foreach=group['foreach'],)

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss
    
def sgd(params: List[Tensor],
        d_p_list: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        iter: int,
        # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
        # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
        has_sparse_grad: bool = None,
        foreach: Optional[bool] = None,
        *,
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool,
        maximize: bool):
    r"""Functional API that performs SGD algorithm computation.

    See :class:`~torch.optim.SGD` for details.
    """

    # if foreach is None:
    #     # why must we be explicit about an if statement for torch.jit.is_scripting here?
    #     _, foreach = _default_to_fused_or_foreach(params, differentiable=False, use_fused=False)


    if foreach:
        func = _multi_tensor_sgd
    else:
        func = _single_tensor_sgd

    func(params,
         d_p_list,
         momentum_buffer_list,
         iter = iter,
         weight_decay=weight_decay,
         momentum=momentum,
         lr=lr,
         dampening=dampening,
         nesterov=nesterov,
         has_sparse_grad=has_sparse_grad,
         maximize=maximize)

def _single_tensor_sgd(params: List[Tensor],
                       d_p_list: List[MultiArray],
                       momentum_buffer_list: List[Optional[MultiArray]],
                       iter,
                       *,
                       weight_decay: float,
                       momentum: float,
                       lr: float,
                       dampening: float,
                       nesterov: bool,
                       maximize: bool,
                       has_sparse_grad: bool):
    for i, param in enumerate(params):
        d_p = d_p_list[i] if not maximize else -d_p_list[i]


        if weight_decay != 0:
            # d_p = d_p.add(param, alpha=weight_decay)
            d_p[:] = d_p[:] + param.value[:] * weight_decay

        if momentum != 0:
            buf = momentum_buffer_list[i]
            @if_e(iter == 0)
            def _():
                buf[:] = d_p[:]
            @else_
            def _():
                buf[:] = buf[:] * momentum + d_p[:] * (1 - dampening)
            if nesterov:
                # d_p = d_p.add(buf, alpha=momentum)
                d_p[:] = d_p[:] + buf[:] * momentum
            else:
                d_p[:] = buf[:]
        param.value[:] = param.value[:] -  d_p[:] * lr
        break_point()

    iter.update(iter + 1)

def _multi_tensor_sgd(params: List[Tensor],
                      grads: List[MultiArray],
                      momentum_buffer_list: List[Optional[MultiArray]],
                      *,
                      weight_decay: float,
                      momentum: float,
                      lr: float,
                      dampening: float,
                      nesterov: bool,
                      maximize: bool,
                      has_sparse_grad: bool):

    raise NotImplementedError

class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, n_threads = 1, approx = True, 
                 weight_decay=0, amsgrad=False, *, foreach: Optional[bool] = None,
                 maximize: bool = False, capturable: bool = False,
                 differentiable: bool = False, fused: Optional[bool] = None):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        beta_power = [MemValue(cfix(1)), MemValue(cfix(1))]
        defaults = dict(lr=MemValue(cfix(lr)), betas=betas, beta_power = beta_power, eps=eps,
                        n_threads = n_threads, approx = approx,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        maximize=maximize, foreach=foreach, capturable=capturable,
                        differentiable=differentiable, fused=fused)
        super().__init__(params, defaults)
        for group in self.param_groups:
            self.init_state(group)
        
    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
            group.setdefault('capturable', False)
            group.setdefault('differentiable', False)
            group.setdefault('fused', None)
        # Currently, we do not support GPU
        # state_values = list(self.state.values())
        # step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]['step'])
        # if not step_is_tensor:
        #     for s in state_values:
        #         s['step'] = torch.tensor(float(s['step']))
            

    def init_state(self, group):
        for p in group['params']:
            if p.req_grad:   
                state = self.state[p]
                if len(state) == 0:
                    # note(crcrpar): [special device hosting for step]
                    # Deliberately host `step` on CPU if both capturable and fused are off.
                    # This is because kernel launches are costly on CUDA and XLA.
                    # state['step'] = (
                    #     torch.zeros((), dtype=torch.float, device=p.device)
                    #     if group['capturable'] or group['fused']
                    #     else torch.tensor(0.)
                    # )
                    # Exponential moving average of gradient values
                    exp_avg = p.grad.same_shape()
                    exp_avg.assign_all(0)
                    state['exp_avg'] = exp_avg
                    # Exponential moving average of squared gradient values
                    exp_avg_sq = p.grad.same_shape()
                    exp_avg_sq.assign_all(0)
                    state['exp_avg_sq'] = exp_avg_sq
                    if group['amsgrad']:
                        max_exp_avg_sq = p.grad.same_shape()
                        max_exp_avg_sq.assign_all(0)
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = max_exp_avg_sq
     
    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps
    ):
        for p in group['params']:
            if p.grad is not None:
                params_with_grad.append(p)
                grads.append(p.grad)

                state = self.state[p]

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])

                if group['amsgrad']:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])
                if group['differentiable'] and state['step'].requires_grad:
                    raise RuntimeError('`requires_grad` is not supported for `step` in differentiable mode')
                # state_steps.append(state['step'])

    @buildingblock("Adam")
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        # if closure is not None:
        #     with torch.enable_grad():
        #         loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']
            beta_power = group['beta_power']
            beta_power[0] *= beta1
            beta_power[1] *= beta2
            n_threads = group['n_threads']
            approx = group['approx']
            
            self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps)

            adam(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=group['amsgrad'],
                beta1=beta1,
                beta2=beta2,
                beta_power1 = beta_power[0],
                beta_power2 = beta_power[1],
                n_threads = n_threads,
                approx = approx,
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                eps=group['eps'],
                maximize=group['maximize'],
                foreach=group['foreach'],
                capturable=group['capturable'],
                differentiable=group['differentiable'],
                fused=group['fused'],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
            )

        return loss

def adam(params: List[Tensor],
         grads: List[Tensor],
         exp_avgs: List[Tensor],
         exp_avg_sqs: List[Tensor],
         max_exp_avg_sqs: List[Tensor],
         state_steps: List[Tensor],
         # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
         # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
         foreach: Optional[bool] = None,
         capturable: bool = False,
         differentiable: bool = False,
         fused: Optional[bool] = None,
         grad_scale: Optional[Tensor] = None,
         found_inf: Optional[Tensor] = None,
         *,
         amsgrad: bool,
         beta1: float,
         beta2: float,
         beta_power1,
         beta_power2,
         n_threads,
         approx,         
         lr: float,
         weight_decay: float,
         eps: float,
         maximize: bool):
    r"""Functional API that performs Adam algorithm computation.
    See :class:`~torch.optim.Adam` for details.
    """

    # Respect when the user inputs False/True for foreach or fused. We only want to change
    # the default when neither have been user-specified. Note that we default to foreach
    # and pass False to use_fused. This is not a mistake--we want to give the fused impl
    # bake-in time before making it the default, even if it is typically faster.
    if fused is None:
        fused = False
    if foreach is None:
        foreach = False

    if not all(isinstance(t, Tensor) for t in state_steps):
        raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")


    if foreach:
        func = _multi_tensor_adam
    else:
        func = _single_tensor_adam

    func(params,
         grads,
         exp_avgs,
         exp_avg_sqs,
         max_exp_avg_sqs,
         state_steps,
         amsgrad=amsgrad,
         beta1=beta1,
         beta2=beta2,
         beta_power1 = beta_power1,
         beta_power2 = beta_power2,  
         n_threads = n_threads,
         approx = approx,
         lr=lr,
         weight_decay=weight_decay,
         eps=eps,
         maximize=maximize,
         capturable=capturable,
         differentiable=differentiable,
         grad_scale=grad_scale,
         found_inf=found_inf)


def _single_tensor_adam(params: List[Tensor],
                        grads: List[Tensor],
                        exp_avgs: List[Tensor],
                        exp_avg_sqs: List[Tensor],
                        max_exp_avg_sqs: List[Tensor],
                        state_steps: List[Tensor],
                        grad_scale: Optional[Tensor],
                        found_inf: Optional[Tensor],
                        *,
                        amsgrad: bool,
                        beta1: float,
                        beta2: float,
                        beta_power1,
                        beta_power2, 
                        n_threads,
                        approx, 
                        lr: float,
                        weight_decay: float,
                        eps: float,
                        maximize: bool,
                        capturable: bool,
                        differentiable: bool):

    assert grad_scale is None and found_inf is None
    m_factor = MemValue(1 / (1 - beta_power1))
    v_factor = MemValue(1 / (1 - beta_power2))
    for i, param in enumerate(params):

        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]

        # update step


        if weight_decay != 0:
            grad[:] = grad[:] + param.value[:] * weight_decay

        # Decay the first and second moment running average coefficient
        m = exp_avg
        v = exp_avg_sq
        g = grad

        # if capturable or differentiable:
        #     step = step_t

        #     # 1 - beta1 ** step can't be captured in a CUDA graph, even if step is a CUDA tensor
        #     # (incurs "RuntimeError: CUDA error: operation not permitted when stream is capturing")
        #     bias_correction1 = 1 - torch.pow(beta1, step)
        #     bias_correction2 = 1 - torch.pow(beta2, step)

        #     step_size = lr / bias_correction1
        #     step_size_neg = step_size.neg()

        #     bias_correction2_sqrt = bias_correction2.sqrt()

        #     if amsgrad:
        #         # Maintains the maximum of all 2nd moment running avg. till now
        #         if differentiable:
        #             max_exp_avg_sqs_i = max_exp_avg_sqs[i].clone()
        #         else:
        #             max_exp_avg_sqs_i = max_exp_avg_sqs[i]
        #         max_exp_avg_sqs[i].copy_(torch.maximum(max_exp_avg_sqs_i, exp_avg_sq))
        #         # Uses the max. for normalizing running avg. of gradient
        #         # Folds in (admittedly ugly) 1-elem step_size math here to avoid extra param-set-sized read+write
        #         # (can't fold it into addcdiv_ below because addcdiv_ requires value is a Number, not a Tensor)
        #         denom = (max_exp_avg_sqs[i].sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(eps / step_size_neg)
        #     else:
        #         denom = (exp_avg_sq.sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(eps / step_size_neg)

        #     param.addcdiv_(exp_avg, denom)
        # else:

        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            vhats = max_exp_avg_sqs[i]
        
        @multithread(n_threads, m.total_size(),
                         max_size=get_program().budget)
        def _(base, size):
            m_part = m.get_vector(base, size)
            v_part = v.get_vector(base, size)
            g_part = g.get_vector(base, size)
            m_part = beta1 * m_part + (1 - beta1) * g_part
            v_part = beta2 * v_part + (1 - beta2) * g_part ** 2
            m.assign_vector(m_part, base)
            v.assign_vector(v_part, base)
            mhat = m_part * m_factor.expand_to_vector(size)
            vhat = v_part * v_factor.expand_to_vector(size)
            if amsgrad:
                v_max = vhats.get_vector(base, size)
                vhat = util.max(vhat, v_max)
                vhats.assign_vector(vhat, base)
            diff = lr.expand_to_vector(size) * mhat
            if approx:
                diff *= mpc_math.InvertSqrt(vhat + eps ** 2)
            else:
                diff /= mpc_math.sqrt(vhat) + eps
            param.value.assign_vector(param.value.get_vector(base, size) - diff, base)


def _multi_tensor_adam(params: List[Tensor],
                       grads: List[Tensor],
                       exp_avgs: List[Tensor],
                       exp_avg_sqs: List[Tensor],
                       max_exp_avg_sqs: List[Tensor],
                       state_steps: List[Tensor],
                       grad_scale: Optional[Tensor],
                       found_inf: Optional[Tensor],
                       *,
                       amsgrad: bool,
                       beta1: float,
                       beta2: float,
                       lr: float,
                       weight_decay: float,
                       eps: float,
                       maximize: bool,
                       capturable: bool,
                       differentiable: bool):
    pass