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

                if opt.optim.algo in ["ESGD","ESGD_Max"] :
                    loss = self.train_iteration_esgd(opt, var, batch_id)
                elif opt.optim.algo in ["Adahessian", "Adahessian_J"]:
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
        loss.iou = self.graph.compute_IOU(var.labels, var.occupancy, mode="train")
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
        var.occupancy = self.neural_bocc.forward(opt, var.points)
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
    

class ReLULayer(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.input_features = in_features
        self.linear = torch.nn.Linear(in_features, out_features)
        self.relu = torch.nn.ReLU()

    def forward(self, input):
        return self.relu(self.linear(input))


class NeuralBinaryOccupancy(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.define_network(opt)

    def define_network(self,opt):
        self.mlp = []

        output_features = 1

        if opt.arch.relu.posenc.enabled:
            log.warning("Positional encoding enabled...")
            input_features = 3+ (6*opt.arch.relu.posenc.L_3D)
        else:
            input_features = 3

        self.mlp.append(ReLULayer(input_features, opt.arch.relu.hidden_features))
        for i in range(opt.arch.relu.hidden_layers):
            self.mlp.append(ReLULayer(opt.arch.relu.hidden_features, opt.arch.relu.hidden_features))

        final_linear = torch.nn.Linear(opt.arch.relu.hidden_features, output_features)
        self.mlp.append(final_linear)
        self.mlp = torch.nn.Sequential(*self.mlp)



    def forward(self,opt, input): # [B,...,3]
        if opt.arch.relu.posenc.enabled:
            coord_3D_enc = self.positional_encoding(opt, input, opt.arch.relu.posenc.L_3D)
            points_enc = torch.cat([input, coord_3D_enc], dim=-1)
        else:
            points_enc = input

        bocc = self.mlp(points_enc)
        return bocc
    
    
    def positional_encoding(self,opt,input,L): 
        """
        Args:
            opt (edict)
            input (torch.Tensor [B,C]): 2d coordinates
            L (int): length of positional encoding

        Return:
            input_enc (torch.Tensor [B,2LC]): positional encoded coordinates
        """
        shape = input.shape
        freq = 2**torch.arange(L,dtype=torch.float32,device=opt.device)*np.pi # [L]
        spectrum = input[...,None]*freq # [B,...,C,L]
        sin,cos = spectrum.sin(),spectrum.cos() # [B,...,C,L]
        input_enc = torch.stack([sin,cos],dim=-2) # [B,...,C,2,L]
        input_enc = input_enc.view(*shape[:-1],-1) # [B,...,2CL]
        return input_enc





