import math
import torch
from torch.optim.optimizer import Optimizer


class Adahessian_J(Optimizer):
    """Implements Adahessian algorithm.
    It has been proposed in `ADAHESSIAN: An Adaptive Second Order Optimizer for Machine Learning`.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 0.15)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-4)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        hessian_power (float, optional): Hessian power (default: 1). You can also try 0.5. For some tasks we found this to result in better performance.
        single_gpu (Bool, optional): Do you use distributed training or not "torch.nn.parallel.DistributedDataParallel" (default: True)
        update_d_every (int): update the Hessian diagonal estimate (default:1)
        d_warmup (int, optional): update the Hessian diagonal estimate for the first d_warmup steps 
                                regardless of update_d_every
    """

    def __init__(self, params, lr=0.15, betas=(0.9, 0.999), lr_warmup=0.99, eps=1e-4,
                 weight_decay=0, hessian_power=1, single_gpu=True, update_d_every=1, d_warmup=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(
                    betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(
                    betas[1]))
        if not 0. <= lr_warmup < 1.:
            raise ValueError(f'Invalid lr warmup parameter: {lr_warmup:g}')
        if not 0.0 <= hessian_power <= 1.0:
            raise ValueError("Invalid Hessian power value: {}".format(hessian_power))
        defaults = dict(lr=lr, betas=betas, lr_warmup=lr_warmup, eps=eps,
                        weight_decay=weight_decay, hessian_power=hessian_power,update_d_every=update_d_every, d_warmup=d_warmup)
        self.single_gpu = single_gpu 
        super(Adahessian_J, self).__init__(params, defaults)
        self.update_d_every = update_d_every
        self.d_warmup = d_warmup
        self.steps = 0
        self.steps_since_d = 0


    def state_dict(self):
        r"""Returns the state of the optimizer as a :class:`dict`.
        It contains three entries:
        * global_state - a dict holding global state.
        * state - a dict holding current optimization state. Its content
            differs between optimizer classes.
        * param_groups - a dict containing all parameter groups
        """
        global_state = {'update_d_every': self.update_d_every,
                        'd_warmup': self.d_warmup,
                        'steps': self.steps,
                        'steps_since_d': self.steps_since_d}
        return {'global_state': global_state, **super().state_dict()}
    

    def load_state_dict(self, state_dict):
        r"""Loads the optimizer state.
        Arguments:
            state_dict: optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        super().load_state_dict(state_dict)
        self.update_d_every = state_dict['global_state']['update_d_every']
        self.d_warmup = state_dict['global_state']['d_warmup']
        self.steps = state_dict['global_state']['steps']
        self.steps_since_d = state_dict['global_state']['steps_since_d']

    def should_create_graph(self):
        """Returns True if the optimizer will update the squared Hessian diagonal estimate
        on the next call to .step() and thus you need to enable create_graph:
        >>> loss.backward(create_graph=optimizer.should_create_graph())
        """
        return self.steps < self.d_warmup or self.steps_since_d >= self.update_d_every


    def get_trace(self, params, grads):
        """
        compute the Hessian vector product with a random vector v, at the current gradient point,
        i.e., compute the gradient of <gradsH,v>.
        :param gradsH: a list of torch variables
        :return: a list of torch tensors
        """

        # Check backward was called with create_graph set to True
        for i, grad in enumerate(grads):
            if grad.grad_fn is None:
                raise RuntimeError('Gradient tensor {:} does not have grad_fn. When calling\n'.format(i) +
                           '\t\t\t  loss.backward(), make sure the option create_graph is\n' +
                           '\t\t\t  set to True.')

        v = [2 * torch.randint_like(p, high=2) - 1 for p in params]

        # this is for distributed setting with single node and multi-gpus, 
        # for multi nodes setting, we have not support it yet.
        if not self.single_gpu:
            for v_i in v:
                v_i[v_i < 0.] = -1.
                v_i[v_i >= 0.] = 1.

        hvs = torch.autograd.grad(
            grads,
            params,
            grad_outputs=v,
            only_inputs=True,
            retain_graph=True)

        hutchinson_trace = []
        for hv in hvs:
            param_size = hv.size()
            if len(param_size) <= 2:  # for 0/1/2D tensor
                # Hessian diagonal block size is 1 here.
                # We use that torch.abs(hv * vi) = hv.abs()
                tmp_output = hv.abs()

            elif len(param_size) == 4:  # Conv kernel
                # Hessian diagonal block size is 9 here: torch.sum() reduces the dim 2/3.
                # We use that torch.abs(hv * vi) = hv.abs()
                tmp_output = torch.mean(hv.abs(), dim=[2, 3], keepdim=True)
            hutchinson_trace.append(tmp_output)

        # this is for distributed setting with single node and multi-gpus, 
        # for multi nodes setting, we have not support it yet.
        if not self.single_gpu:
            for output1 in hutchinson_trace:
                dist.all_reduce(output1 / torch.cuda.device_count())
        
        return hutchinson_trace

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            gradsH: The gradient used to compute Hessian vector product.
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        params = []
        groups = []
        grads = []

        # Flatten groups into lists, so that
        #  hut_traces can be called with lists of parameters
        #  and grads 
        hut_traces_iter = None

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    params.append(p)
                    groups.append(group)
                    grads.append(p.grad)

            if self.should_create_graph():
                # get the Hessian diagonal
                hut_traces = self.get_trace(params, grads)
                self.steps_since_d = 0
                hut_traces_iter = iter(hut_traces)
    

        
        # if self.should_create_graph():
        #     for group in self.param_groups:
        #         for p in group['params']:
        #             if p.grad is not None:
        #                 params.append(p)
        #                 groups.append(group)
        #                 grads.append(p.grad)

        #     # get the Hessian diagonal
        #     hut_traces = self.get_trace(params, grads)
        #     self.steps_since_d = 0
        #     hut_traces_iter = iter(hut_traces)
    

        ## do the step
        #for (p, group, grad, hut_trace) in zip(params, groups, grads, hut_traces):
        for (p, group, grad) in zip(params, groups, grads):

            state = self.state[p]

            # State initialization
            if len(state) == 0:
                state['step'] = 0
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p.data)
                # Exponential moving average of Hessian diagonal square values
                state['exp_hessian_diag_sq'] = torch.zeros_like(p.data)
                state['lr_warmup_cumprod'] = 1.

            exp_avg, exp_hessian_diag_sq = state['exp_avg'], state['exp_hessian_diag_sq']

            beta1, beta2 = group['betas']

            state['step'] += 1

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad.detach_(), alpha=1 - beta1)
            
            if hut_traces_iter is not None:
                hut_trace = next(hut_traces_iter)
                #print(p.shape, hut_trace.shape)
                exp_hessian_diag_sq.mul_(beta2).addcmul_(hut_trace, torch.ones_like(hut_trace), value=1 - beta2)
    
                # Learning rate schedule
            if hut_traces_iter is not None:
                state['lr_warmup_cumprod'] *= group['lr_warmup']
    
            bias_correction1 = 1 - beta1 ** state['step']
            bias_correction2 = 1 - beta2 ** state['step']

            # make the square root, and the Hessian power
            k = group['hessian_power']
            denom = (
                (exp_hessian_diag_sq.sqrt() ** k) /
                math.sqrt(bias_correction2) ** k).add_(
                group['eps'])

            # make update
            p.data = p.data - \
                group['lr'] * (exp_avg / bias_correction1 / denom + group['weight_decay'] * p.data)


        self.steps += 1
        self.steps_since_d += 1
        
        return loss