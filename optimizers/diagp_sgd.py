""" This optimizer class offers two preconditioners
- Equilbrated (Hv)**2
- Jacobi (Hv)
"""

import torch
from .optimizer import Optimizer 
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec

def normalization(v):
    """
    Normalization of a list of vectors
    
    """
    s = group_product(v, v)
    s = s ** 0.5
    v = [ vi / (s+1e-6) for vi in v]
    return v
    
    
def group_product(xs, ys):
    """
    Computes the inner product of two lists of variables xs, ys
    """
    return sum([torch.sum(x*y) for (x,y) in zip(xs, ys)])
    

class Diag_PSGD(Optimizer):
    """ESGD-M (ESGD from "Equilibrated adaptive learning rates for non-convex optimization"
    with Nesterov momentum.
    To use this optimizer you must call .backward() with the create_graph=True option.
    Gradient accumulation steps and distributed training are currently not supported.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of the gradient and squared Hessian diagonal estimate
            (default: (0.9, 0.999))
        lr_warmup (float, optional): exponential learning rate warmup coefficient
            (same units as betas, 0 means no warmup, closer to 1 means a longer
            warmup) (default: 0.99)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-4)
        weight_decay (float, optional): weight decay coefficient (default: 0)
        update_d_every (int, optional): update the squared Hessian diagonal
            estimate every update_d_every steps (default: 10)
        d_warmup (int, optional): update the squared Hessian diagonal estimate
            for the first d_warmup steps regardless of update_d_every (default: 20)
    .. _Equilibrated adaptive learning rates for non-convex optimization:
        https://proceedings.neurips.cc/paper/2015/file/430c3626b879b4005d41b8a46172e0c0-Paper.pdf
    """

    def __init__(self, params,  lr=10, betas=(0.9, 0.999), lr_warmup=0.99, eps=1e-4,
                 weight_decay=0, update_d_every=100, d_warmup=50, preconditioner_type="equilibrated"):
        if not 0. <= lr:
            raise ValueError(f'Invalid learning rate: {lr:g}')
        if not 0. <= betas[0] < 1.:
            raise ValueError(f'Invalid beta parameter at index 0: {betas[0]:g}')
        if not 0. <= betas[1] <= 1.:
            raise ValueError(f'Invalid beta parameter at index 1: {betas[1]:g}')
        if not 0. <= lr_warmup < 1.:
            raise ValueError(f'Invalid lr warmup parameter: {lr_warmup:g}')
        if not 0. <= eps:
            raise ValueError(f'Invalid epsilon value: {eps:g}')
        if not 0. <= weight_decay:
            raise ValueError('Invalid weight_decay value: {weight_decay:g}')
        if not int(update_d_every) or not 1 <= update_d_every:
            raise ValueError(f'Invalid update_d_every parameter: {update_d_every}')
        if not int(d_warmup) or not 1 <= d_warmup:
            raise ValueError(f'Invalid d_warmup parameter: {d_warmup}')
        defaults = dict( lr=lr, betas=betas, lr_warmup=lr_warmup, eps=eps,
                        weight_decay=weight_decay, update_d_every=update_d_every, d_warmup=d_warmup)
        super(Diag_PSGD, self).__init__(params, defaults)
        self.update_d_every = update_d_every
        self.d_warmup = d_warmup
        self.steps = 0
        self.steps_since_d = 0
        self.preconditioner_type = preconditioner_type


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

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                

        # Compute the squared Hessian diagonal estimate
        hvps_iter = None
        if self.should_create_graph():
            params, grads, vs = [], [], []
            with torch.enable_grad():
                for group in self.param_groups:
                    for p in group['params']:
                        if p.grad is None:
                            continue
                        if p.grad.is_sparse:
                            raise RuntimeError('ESGD does not support sparse gradients')
                        if p.grad.grad_fn is None:
                            msg = f'Gradient tensor shaped like {tuple(p.grad.shape)} does not have ' \
                                'a grad_fn. When calling loss.backward(), make sure the option ' \
                                'create_graph is set to True.'
                            raise RuntimeError(msg)
                        params.append(p)
                        grads.append(p.grad)
                        #vs.append(torch.normal(0, 1, size=p.grad.shape, device="cuda"))
                        vs.append(torch.randint_like(p.grad, 2) * 2 - 1)
                hvps = torch.autograd.grad(grads, params, grad_outputs=vs)
                
            hvps_iter = iter(hvps)
            self.steps_since_d = 0

        # Do the step
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Perform optimization step
                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state['exp_avg_bias_corr'] = 1.
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponentially weighted infinity norm of squared Hessian diagonal estimates
                    state['exp_avg_d'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Learning rate warmup cumulative product
                    state['lr_warmup_cumprod'] = 1.

                exp_avg, exp_avg_d = state['exp_avg'], state['exp_avg_d']
                beta1, beta2 = group['betas']

                # Update the running averages
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                state['exp_avg_bias_corr'] *= beta1
                if hvps_iter is not None:
                    hvp = next(hvps_iter)
                    #exp_avg_d.mul_(beta2).addcmul_(hvp, hvp.conj(), value=1-beta2)
                    exp_avg_d.mul_(beta2)
                    if self.preconditioner_type == "equilibrated":
                        torch.maximum(exp_avg_d, hvp.square_(), out=exp_avg_d)
                    elif self.preconditioner_type == "jacobi":
                        ## (Theoretically) hadamard product ## v o Hv ## 
                        # (We have to use abs in implementation because of sqrt term) #
                        torch.maximum(exp_avg_d, hvp.abs_(), out=exp_avg_d)
                    else:
                        TypeError("Unknown preconditioner")
                denom = exp_avg_d.sqrt().add_(group['eps'])

                # Learning rate schedule
                if hvps_iter is not None:
                    state['lr_warmup_cumprod'] *= group['lr_warmup']
                step_size = group['lr'] * (1 - state['lr_warmup_cumprod'])  

                # Momentum
                exp_avg_est = torch.lerp(exp_avg, grad, 1 - beta1)
                
                mom_bias_corr = 1 - state['exp_avg_bias_corr'] * beta1



                #print("exp_avg_bias_corr {} | mom_bias_corr {}".format(state['exp_avg_bias_corr'],mom_bias_corr))

                # Weight decay
                p.mul_(1 - group['weight_decay'] * step_size)
                # Do the step
                p.addcdiv_(exp_avg_est, denom, value=-step_size / mom_bias_corr)



        self.steps += 1
        self.steps_since_d += 1

        return

        #return torch.cat([p.flatten() for p in preconditioner])
