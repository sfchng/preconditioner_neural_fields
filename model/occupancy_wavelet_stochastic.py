import numpy as np
import os
import torch
import torch.nn.functional as torch_F
import PIL
import importlib
from . import base
from util import log
import time
from easydict import EasyDict as edict
import tqdm
from torch.utils.data import DataLoader

from bocc_utils import sdf_mesh
from data.bocc_stanford import MeshOccupancy
import optimizers as torch_optimizers
from optimizers.precond_kfac import kfac as pkfac
from optimizers.shampoo import shampoo
from optimizers.precond_adahessian import adahessian
from optimizers.precond_adahessian import adahessian_j



class Model(base.Model):
    
    def __init__(self, opt):
        super().__init__(opt)

    def load_dataset(self, opt):
        self.sdf_sampler = MeshOccupancy(opt)

    def build_networks(self, opt):
        return super().build_networks(opt)

    def setup_optimizer(self,opt):
        log.info("setting up optimizers...{}".format(opt.optim.algo))
           
        if opt.optim.algo == "Adam":
            optimizer = getattr(torch_optimizers, opt.optim.algo)
            optim_list = [
                dict(params=self.graph.neural_bocc.parameters(),lr=opt.optim.Adam.lr)
            ]
            self.optim = optimizer(optim_list)
        elif opt.optim.algo == "SGD":
            optimizer = getattr(torch.optim, opt.optim.algo)
            optim_list = [
                dict(params=self.graph.neural_bocc.parameters(),lr=opt.optim.SGD.lr)
            ]
            self.optim = optimizer(optim_list)            
        elif opt.optim.algo == "Shampoo":
            self.optim = shampoo.Shampoo(self.graph.neural_bocc.parameters(), lr=opt.optim.Shampoo.lr)
        elif opt.optim.algo == "Adahessian":
            self.optim = adahessian.Adahessian(self.graph.neural_bocc.parameters(),lr=opt.optim.Adahessian.lr)            
        elif opt.optim.algo =="Adahessian_J":
            self.optim = adahessian_j.Adahessian_J(self.graph.neural_bocc.parameters(),lr=opt.optim.Adahessian.lr)
        elif opt.optim.algo == "Preconditioner_KFAC":
            self.optim = torch_optimizers.Adam(self.graph.neural_bocc.parameters(), lr=opt.optim.Adam.lr)
            self.preconditioner = pkfac.KFAC(self.graph.neural_bocc, opt.optim.PKFac.lr, opt.optim.PKFac.update_freq)
        elif opt.optim.algo == "ESGD" or opt.optim.algo == "ESGD_Max":
            optimizer = getattr(torch_optimizers, opt.optim.algo)
            self.optim = optimizer(params=self.graph.neural_bocc.parameters(),lr=opt.optim.ESGD.lr, update_d_every=opt.optim.ESGD.update, d_warmup=opt.optim.ESGD.d_warmup,eps=opt.optim.ESGD.damping,preconditioner_type="equilbrated")            
        else:
            raise NotImplementedError
        log.status(self.optim)

    def train(self, opt):
        ## before training ##
        log.title("Training start")
        self.timer = edict(start=time.time(), it_mean=None)
        self.ep = self.it = self.vis_it = 0

        self.training_time = 0

        time_fname = "{}/time.txt".format(opt.output_path)
        with open(time_fname, "w") as file:
            file.close() 
        self.graph.train()

        ## trainloader ##
        self.sdf_sampler.reset()
        num_batches = len(self.sdf_sampler) // opt.batch_size
        var = edict(idx=torch.arange(opt.batch_size))

        ## timeloader ##
        timeloader = tqdm.trange(opt.max_iter, desc="training occupancy", leave=False)

        for it in timeloader:
            
            for batch_id in range(num_batches):

                points, labels = self.sdf_sampler.points[batch_id * opt.batch_size: (batch_id + 1) * opt.batch_size], self.sdf_sampler.labels[batch_id * opt.batch_size: (batch_id + 1) * opt.batch_size]
                var.points = points.cuda()
                var.labels = labels.cuda()

                if opt.optim.algo == "ESGD" or opt.optim.algo == "ESGD_M" :
                    loss = self.train_iteration_esgd(opt, var, batch_id)
                elif opt.optim.algo =="Adahessian" or opt.optim.algo =="Adahessian_J":
                    loss = self.train_iteration_ada(opt, var)
                elif opt.optim.algo == "Preconditioner_KFAC":
                    loss = self.train_iteration_pkfac(opt, var)
                else:
                    loss = self.train_iteration(opt, var, batch_id)

            self.sdf_sampler.reset()

            print("Epoch {} Loss {:.5f} IOU {:.5f} | Train time {:.5f}\n".format(self.it, loss.all, loss.iou, self.training_time))
            ## save ckpt ##
            if it % opt.freq.ckpt == 0:
                self.save_checkpoint(opt,ep=None,it=it)
                
            if it % opt.freq.scalar == 0 or it == 0:
                self.log_scalars(opt, loss, var, step=it, split="train")
            self.it+=1
            
            ## save logs ##
            with open(time_fname, "a") as f:
                f.write("{} {:.5f} {:.5f} {:.6f} {:.6f}\n".format(self.it, self.train_per_iter, self.training_time, loss.all, loss.iou ))

    def train_iteration(self, opt, var, batch_id):

        self.timer.it_start = time.time()

        ## train iteration ##

        self.optim.zero_grad()
        var = self.graph.forward(opt, var, mode="train")
        loss = self.graph.compute_loss(opt, var, mode="train")
        loss.all.backward()
        adam_train_start = time.time()
        self.optim.step()
        adam_train_end = time.time()

        self.train_per_iter = adam_train_end - adam_train_start
        self.training_time += self.train_per_iter
        ## compute IOU #
        loss.iou = self.graph.compute_IOU(var.labels, var.occupancy, mode="train")

        return loss
    
    def train_iteration_ada(self, opt, var):

        self.timer.it_start = time.time()

        ## train iteration ##
        self.optim.zero_grad()
        var_out = self.graph.forward(opt, var, mode="train")
        loss = self.graph.compute_loss(opt, var_out, mode="train")
        loss.all.backward(create_graph=True)
    
        train_start = time.time()
        self.optim.step()
        train_end = time.time()

        self.train_per_iter = train_end - train_start
        self.training_time += self.train_per_iter
        
        ## compute IOU #
        loss.iou = self.graph.compute_IOU(var.labels, var.occupancy, mode="train")

        return loss

    def train_iteration_esgd(self, opt, var, batch_id):

        self.timer.it_start = time.time()


        ## train iteration ##
        self.optim.zero_grad(set_to_none=True)
        var = self.graph.forward(opt, var, mode="train")
        loss = self.graph.compute_loss(opt, var, mode="train")
        loss.all.backward(create_graph=self.optim.should_create_graph())

        esgd_train_start = time.time()
        self.optim.step()
        esgd_train_end = time.time()

        self.train_per_iter = esgd_train_end - esgd_train_start
        self.training_time += self.train_per_iter

        ## compute IOU #
        loss.iou = self.graph.compute_IOU(var.labels, var.occupancy, mode="train")
        return loss


    def train_iteration_pkfac(self, opt, var):

        self.timer.it_start = time.time()

        ## train iteration ##
        self.optim.zero_grad()
        var = self.graph.forward(opt, var, mode="train")
        loss = self.graph.compute_loss(opt, var, mode="train")
        loss.all.backward()
        train_start = time.time()
        self.preconditioner.step()
        self.optim.step()
        train_end = time.time()
        self.train_per_iter = train_end - train_start
        self.training_time += self.train_per_iter
        return loss
    
    @torch.no_grad()
    def log_scalars(self,opt,loss,var, metric=None,step=0,split="train"):
        for key,value in loss.items():
            self.tb.add_scalar("{0}/loss_{1}".format(split,key),value,step)
         

    @torch.no_grad()
    def evaluate(self,opt):
        sdf_mesh.create_mesh_from_occupancy(opt, self.graph.neural_bocc)
        

class Graph(base.Graph):

    def __init__(self, opt):
        super().__init__(opt)
        self.neural_bocc = NeuralBinaryOccupancy(opt)
        log.status(self.neural_bocc)


    def forward(self, opt, var, mode=None):
        var.occupancy = self.neural_bocc.forward(var.points)
        return var
        
    
    def compute_loss(self, opt, var, mode="train"):
        loss = edict()
        loss_bcc = torch.nn.functional.binary_cross_entropy_with_logits(var.occupancy, var.labels, reduction="none")
        loss.all = loss_bcc.mean()
        return loss
   

    def compute_IOU(self, gt, preds, mode="train", threshold=0.5):
        if threshold is not None:
            preds[preds < threshold] = 0.0
            preds[preds >= threshold] = 1.0
            
        if type(preds) == np.ndarray:
            intersection = np.logical_and(preds, gt).sum()
            union =  np.logical_or(preds, gt).sum()
        else:
            intersection =  torch.logical_and(preds.cuda(), gt.cuda()).sum()
            union = torch.logical_or(preds.cuda(), gt.cuda()).sum()
        
        IOU = intersection / union
        return IOU


    def gradient(self, y, x, grad_outputs=None):
        if grad_outputs is None:
            grad_outputs = torch.ones_like(y)

        grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
        return grad
    

class GaussianLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, sigma=0.05):
        super().__init__()
        self.sigma = sigma
        self.linear = torch.nn.Linear(in_features, out_features)


    def forward(self, input):
        return self.gaussian(self.linear(input))


    def gaussian(self, input):
        """
        Args:
            opt
            x (torch.Tensor [B,num_rays,])
        """
        k1 = (-0.5*(input)**2/self.sigma**2).exp()
        return k1
    


class GaborLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega=10, sigma=10, trainable=False ):
        super().__init__()
        self.omega = omega
        self.scale = sigma
        self.is_first = is_first
        self.input_features = in_features

        if self.is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat

        ## set trainable parameters if they are to be simulaneously optimized ##
        self.omega = torch.nn.Parameter(self.omega*torch.ones(1), trainable)
        self.scale = torch.nn.Parameter(self.scale*torch.ones(1), trainable)

        self.linear = torch.nn.Linear(in_features, out_features, bias=bias, dtype=dtype)

        ## second Gaussian window ##
        self.scale_orth = torch.nn.Linear(in_features, out_features, bias=bias, dtype=dtype)

    
    def forward(self, input):
        lin = self.linear(input)
        scale_x = lin
        scale_y = self.scale_orth(input)
        
        freq_term = torch.exp(1j*self.omega*lin)
        
        arg = scale_x.abs().square() + scale_y.abs().square()
        gauss_term = torch.exp(-self.scale*self.scale*arg)
                
        return freq_term*gauss_term
    

class RealGaborLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega=10, sigma=10, trainable=False ):
        super().__init__()
        self.omega_0 = omega
        self.scale_0 = sigma
        self.is_first = is_first
        self.input_features = in_features

        self.freqs = torch.nn.Linear(in_features, out_features, bias=bias)
        self.scale = torch.nn.Linear(in_features, out_features, bias=bias)
        
    def forward(self, input):
        omega = self.omega_0 * self.freqs(input)
        scale = self.scale(input) * self.scale_0
        
        return torch.cos(omega)*torch.exp(-(scale**2))        



class NeuralBinaryOccupancy(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.define_network(opt)

    def define_network(self,opt):
        self.mlp = []
        input_features = 3
        output_features = 1

        hidden_features = opt.arch.wavelet.hidden_features

        self.mlp.append(RealGaborLayer(input_features, hidden_features, omega=opt.arch.wavelet.first_omega, sigma=opt.arch.wavelet.scale, is_first=True))
        for i in range(opt.arch.wavelet.hidden_layers):
            self.mlp.append(RealGaborLayer(hidden_features, hidden_features, omega=opt.arch.wavelet.hidden_omega, sigma=opt.arch.wavelet.scale, is_first=False))

        self.mlp.append(torch.nn.Linear(hidden_features, output_features))

        self.mlp = torch.nn.Sequential(*self.mlp)



    def forward(self,input): # [B,...,3]
        bocc = self.mlp(input)
        return bocc



