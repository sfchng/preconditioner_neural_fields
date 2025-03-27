import numpy as np
import os,sys,time
import torch
import torch.nn.functional as torch_F
import torchvision
import torchvision.transforms.functional as torchvision_F
import tqdm
from easydict import EasyDict as edict
import PIL
import PIL.Image,PIL.ImageDraw
from PIL import ImageOps
import imageio

import util,util_vis
from util import log
from . import base
import warp
import matplotlib.pyplot as plt
import math
import scipy.stats as ss

import time

from torch.utils.data import DataLoader
from data.image import Image
import optimizers as torch_optimizers
from optimizers.precond_kfac import kfac as pkfac
from optimizers.shampoo import shampoo
from optimizers.precond_adahessian import adahessian
from optimizers.precond_adahessian import adahessian_j

# ============================ main engine for training and evaluation ============================

class Model(base.Model):

    def __init__(self,opt):
        super().__init__(opt)

    def load_dataset(self,opt,eval_split=None):
        self.dataset = Image(opt)
    
    def build_networks(self,opt):
        super().build_networks(opt)
        
    def setup_optimizer(self,opt):
        log.info("setting up optimizers...{}".format(opt.optim.algo))
          
        if opt.optim.algo == "Adam":
            optimizer = getattr(torch_optimizers, opt.optim.algo)
            optim_list = [
                dict(params=self.graph.neural_image.parameters(),lr=opt.optim.Adam.lr)
            ]
            self.optim = optimizer(optim_list)
        elif opt.optim.algo == "SGD":
            optimizer = getattr(torch.optim, opt.optim.algo)
            optim_list = [
                dict(params=self.graph.neural_image.parameters(),lr=opt.optim.SGD.lr)
            ]
            self.optim = optimizer(optim_list)            
        elif opt.optim.algo == "Shampoo":
            self.optim = shampoo.Shampoo(self.graph.neural_image.parameters(), lr=opt.optim.Shampoo.lr)
        elif opt.optim.algo == "Adahessian":
            self.optim = adahessian.Adahessian(self.graph.neural_image.parameters(),lr=opt.optim.Adahessian.lr)            
        elif opt.optim.algo =="Adahessian_J":
            self.optim = adahessian_j.Adahessian_J(self.graph.neural_image.parameters(),lr=opt.optim.Adahessian.lr)
        elif opt.optim.algo == "Preconditioner_KFAC":
            self.optim = torch_optimizers.Adam(self.graph.neural_image.parameters(), lr=opt.optim.Adam.lr)
            self.preconditioner = pkfac.KFAC(self.graph.neural_image, opt.optim.PKFac.lr, opt.optim.PKFac.update_freq)
        elif opt.optim.algo == "ESGD" or opt.optim.algo == "ESGD_Max":
            optimizer = getattr(torch_optimizers, opt.optim.algo)
            self.optim = optimizer(params=self.graph.neural_image.parameters(),lr=opt.optim.ESGD.lr, update_d_every=opt.optim.ESGD.update, d_warmup=opt.optim.ESGD.d_warmup, eps=opt.optim.ESGD.damping, preconditioner_type="equilbrated")            
        else:
            raise NotImplementedError
        log.status(self.optim)

    def setup_visualizer(self,opt):
        super().setup_visualizer(opt)   


    def train(self,opt):
        # before training
        log.title("TRAINING START")
        self.timer = edict(start=time.time(),it_mean=None)
        self.ep = self.it = self.iter = self.vis_it = 0

        self.training_time = 0
        self.forward_time = 0
        self.graph.train()

        trainloader = DataLoader(self.dataset, batch_size=opt.data.batch_size, shuffle=True)

        var = edict(idx=torch.arange(opt.batch_size))
        var.test_coords = self.dataset.coords #(HW,2)
        var.gt_rgb = self.dataset.labels #(HW,3)

        self.graph.eval()
        var = util.move_to_device(var, opt.device)
        var = self.graph.forward(opt, var)
        loss = self.graph.compute_loss(opt, var, mode='initial')
        self.visualize(opt, var, step=0, split="initial")
        self.log_scalars(opt, loss, var, step=0, split='initial')


        ####################### save log ############################################
        outfile = "{}/log.txt".format(opt.output_path)
        with open(outfile, 'w') as f:
            f.close()


        ####################### Training ############################################

        loader = tqdm.trange(opt.max_iter,desc="training neural image function",leave=False)

        for it in loader:
            self.graph.train()
            ## train iteration ## 
            if opt.optim.algo  == "ESGD":
                loss = self.train_iteration_esgd(opt,var, loader, trainloader)             
            elif opt.optim.algo =="Preconditioner_KFAC":
                loss = self.train_iteration_pkfac(opt, var, loader, trainloader)
            elif opt.optim.algo in ["Adahessian","Adahessian_J"]:
                loss = self.train_iteration_adahessian(opt, var, loader, trainloader)                    
            else:
                loss = self.train_iteration(opt,var, loader, trainloader)
            
            final_loss = loss.render

            if self.it % opt.freq.val ==0:
                with torch.no_grad():
                    var = self.graph.forward(opt,var,mode="evaluate")
                    loss = self.graph.compute_loss(opt, var, mode="evaluate")
                    loss = self.summarize_loss(opt, var, loss)
                    self.log_scalars(opt,loss,var,step=self.it, split="evaluate")
                    self.visualize(opt,var, step=self.it, split="evaluate")

            if self.it % opt.freq.ckpt == 0:
                self.save_checkpoint(opt,ep=None,it=self.it)

            with open(outfile, 'a') as f:
                f.write("{} {:.5f} {:.5f} {:.5f}\n".format(self.it, self.train_per_epoch, self.training_time, final_loss))
                    
            
        ## after training ##
        print("Converged at {} with {} Loss {}".format(self.it, self.training_time, final_loss))
        self.save_checkpoint(opt,ep=None,it=self.it)

        ## final image ##
        self.graph.eval()
        var = self.graph.forward(opt,var,mode="evaluate")
        self.log_scalars(opt,loss,var,step=self.it, split="evaluate")
        self.visualize(opt,var, step=0, split="final")
    
        if opt.tb:
            self.tb.flush()
            self.tb.close()
        if opt.visdom: self.vis.close()
        log.title("TRAINING DONE")

        
    def train_iteration(self,opt,var,loader, trainloader):
        # before train iteration
        self.timer.it_start = time.time()

        self.train_per_epoch = 0
        ## dataloader ##
        for i, (input, gt) in enumerate(trainloader):

            var.input = input
            var.label = gt
            var = util.move_to_device(var, opt.device)

            time_start = time.time()                     
            ## train iteration ##
            self.optim.zero_grad()
            self.graph.forward(opt,var, mode="train")
            loss = self.graph.compute_loss(opt,var,mode="train")
            loss = self.summarize_loss(opt,var,loss)

            loss.all.backward()
                 
            self.optim.step()
            train_per_iter = time.time() - time_start
            self.training_time += train_per_iter
            self.train_per_epoch += train_per_iter

            if self.iter == 0:
                self.log_scalars(opt,loss,var,step=self.iter+1,split="initial")
            self.iter+=1
            #print("train_per_iteration {}... total training time {:.5f}\n".format(train_per_iter, self.training_time))
        
        ## after train epoch ##
        if (self.it+1)%opt.freq.scalar==0: self.log_scalars(opt,loss,var,step=self.it+1,split="train")
        self.it += 1

        loader.set_postfix(it=self.it,loss="{:.6f}".format(loss.all), train_per_iteration="{:.4f}".format(train_per_iter))
        self.timer.it_end = time.time()
        util.update_timer(opt,self.timer,self.ep,len(loader))
        return loss
    

    def train_iteration_esgd(self,opt,var,loader, trainloader):
        # before train iteration
        self.timer.it_start = time.time()
        self.train_per_epoch = 0
        ## dataloader ##
        for i, (input, gt) in enumerate(trainloader):

            var.input = input
            var.label = gt
            var = util.move_to_device(var, opt.device)
            
            time_start = time.time()       
            ## train iteration ##
            self.optim.zero_grad(set_to_none=True)
            self.graph.forward(opt,var, mode="train")
            loss = self.graph.compute_loss(opt,var,mode="train")
            loss = self.summarize_loss(opt,var,loss)

            loss.all.backward(create_graph=self.optim.should_create_graph())
            
            self.hvp_vis = self.optim.step()
            train_per_iter = time.time() - time_start
            self.training_time += train_per_iter
            self.train_per_epoch += train_per_iter
            self.iter+=1

        ## after train epoch ##
        if (self.it+1)%opt.freq.scalar==0: self.log_scalars(opt,loss,var,step=self.it+1,split="train")
        self.it += 1
        loader.set_postfix(it=self.it,loss="{:.6f}".format(loss.all), train_per_iteration="{:.4f}".format(train_per_iter))
        self.timer.it_end = time.time()
        util.update_timer(opt,self.timer,self.ep,len(loader))
        return loss


    def train_iteration_pkfac(self,opt,var,loader, trainloader):
        # before train iteration
        self.timer.it_start = time.time()

        self.train_per_epoch = 0

        ## dataloader ##
        for i, (input, gt) in enumerate(trainloader):

            var.input = input
            var.label = gt
            var = util.move_to_device(var, opt.device)
            
            time_start = time.time()       
            ## train iteration ##
            self.optim.zero_grad()
            self.graph.forward(opt,var, mode="train")
            loss = self.graph.compute_loss(opt,var,mode="train")
            loss = self.summarize_loss(opt,var,loss)

            loss.all.backward()
            
            self.preconditioner.step()
            self.optim.step()
            train_per_iter = time.time() - time_start
            self.training_time += train_per_iter
            self.train_per_epoch += train_per_iter
            
            if self.iter == 0:
                self.log_scalars(opt,loss,var,step=self.iter+1,split="initial")
            self.iter+=1
        if (self.it+1)%opt.freq.scalar==0: self.log_scalars(opt,loss,var,step=self.it+1,split="train")   
        self.it += 1
        loader.set_postfix(it=self.it,loss="{:.6f}".format(loss.all), train_per_iteration="{:.4f}".format(train_per_iter))
        self.timer.it_end = time.time()
        util.update_timer(opt,self.timer,self.ep,len(loader))

        return loss
    

    def train_iteration_adahessian(self,opt,var,loader, trainloader):
        # before train iteration
        self.timer.it_start = time.time()
        
        self.train_per_epoch = 0
        ## dataloader ##
        for i, (input, gt) in enumerate(trainloader):
            hessian = None
            var.input = input
            var.label = gt
            var = util.move_to_device(var, opt.device)
            
            time_start = time.time()       
            ## train iteration ##
            self.optim.zero_grad(set_to_none=True)
            self.graph.forward(opt,var, mode="train")
            loss = self.graph.compute_loss(opt,var,mode="train")
            loss = self.summarize_loss(opt,var,loss)

            loss.all.backward(create_graph=True)
            self.optim.step()
                    
            train_per_iter = time.time() - time_start
            self.training_time += train_per_iter
            self.train_per_epoch += train_per_iter

            ## after train epoch ##
                         
            self.iter+=1
        ## log after every epoch ##
        if (self.it+1)%opt.freq.scalar==0: self.log_scalars(opt,loss,var,step=self.it+1,split="train")   
        self.it += 1
        loader.set_postfix(it=self.it,loss="{:.6f}".format(loss.all), train_per_iteration="{:.4f}".format(train_per_iter))
        self.timer.it_end = time.time()
        util.update_timer(opt,self.timer,self.ep,len(loader))
        return loss

    def compute_condition_num(self, hessian):
        eig_vals = torch.linalg.eigvals(hessian)
        if eig_vals.abs().min() == 0:
            eig_vals_min = eig_vals.abs().min() + 1e-31
        else:
            eig_vals_min = eig_vals.abs().min()
        condition_num = eig_vals.abs().max() / eig_vals_min
        return condition_num
    
    @torch.no_grad()
    def log_cond_num(self,cond_num,step=0,split="train", mode="condition_num"):
        
        ## log learning rate ##
        if split == "train":
            self.tb.add_scalar("{0}/{1}".format(split, mode),cond_num,step)


    def train_iteration_lbfgs(self, opt, var, loader):
        ## before train iteration ##
        self.timer.it_start = time.time()

        def closure():
            #torch.set_grad_enabled(True)
            # train iteration
            self.optim.zero_grad()
            var_out = self.graph.forward(opt,var,mode="train")
            loss = self.graph.compute_loss(opt,var_out,mode="train")
            loss = self.summarize_loss(opt,var_out,loss)
            loss.all.backward()
            return loss.all

        time_start = time.time()
        ## update weights ##
        self.optim.step(closure)
        loss = closure()

        end_time = time.time()
        train_per_iter = end_time - time_start
        self.training_time += train_per_iter
        print("train_per_iteration {:.5f}".format(train_per_iter))
        print("total training time {}\n".format(self.training_time))
        ## after train iteration ##
        if (self.it+1)%opt.freq.scalar==0: self.log_scalars(opt,loss,var=None,step=self.it+1,split="train")
        self.it += 1
        loader.set_postfix(it=self.it,loss="{:.5f}".format(loss), train_per_iteration="{:.4f}".format(train_per_iter))
        self.timer.it_end = time.time()
        util.update_timer(opt,self.timer,self.ep,len(loader))
        return loss
    
    def compute_hessian(self, loss):

        grads = torch.autograd.grad(loss.all, self.graph.neural_image.parameters(), create_graph=True,retain_graph=True)
        grads = torch.cat([grad.flatten() for grad in grads])       
        # Compute the Hessian matrix
        hessian = []
        
        for grad_i in range(grads.shape[0]):
            hessian_row = []
            for k, param in enumerate(self.graph.neural_image.parameters()):
                #log.info("Computing the hessian with respect to layer {}".format(k))
                hessian_i = torch.autograd.grad(grads[grad_i], param, retain_graph=True)[0]
                hessian_row.append(hessian_i.view(-1))
            hessian.append(torch.cat(hessian_row, dim=0))
            
        hessian = torch.stack(hessian,dim=0)
        
        return hessian

    @torch.no_grad()
    def log_scalars(self,opt,loss,var,metric=None,step=0,split="train"):
        
        ## log learning rate ##
        if split == "train":
            lr = self.optim.param_groups[0]["lr"]
            self.tb.add_scalar("{0}/{1}".format(split,"lr"),lr,step)
        if type(loss) is edict:
            for key,value in loss.items():
                if key=="all": continue
                if opt.loss_weight[key] is not None:
                    self.tb.add_scalar("{0}/loss_{1}".format(split,key),value,step)
            if metric is not None:
                for key,value in metric.items():
                    self.tb.add_scalar("{0}/{1}".format(split,key),value,step)
            # compute PSNR
            psnr = -10*loss.render.log10()
        else:
            psnr = -10*loss.log10()
            self.tb.add_scalar("{0}/loss_render".format(split),loss,step)
        self.tb.add_scalar("{0}/{1}".format(split,"PSNR"),psnr,step)


    @torch.no_grad()
    def visualize(self,opt,var,step=0,split="train"):
        var.rgb_map = var.rgb.detach().clone()
        if split != "train":
            self.vis_it +=1
            if opt.tb:
                util_vis.tb_image(opt,self.tb,step,split,"groundtruth",var.gt_rgb.view(opt.H, opt.W, 3).permute(2,0,1)[None])
                util_vis.tb_image(opt,self.tb,step,split,"predicted",var.rgb_map.view(opt.batch_size,opt.H,opt.W,3).permute(0,3,1,2))

# ============================ computation graph for forward/backprop ============================

class Graph(base.Graph):

    def __init__(self,opt):
        super().__init__(opt)
        self.neural_image = NeuralImageFunction(opt)
        log.status(self.neural_image)
        self.mse_loss = torch.nn.MSELoss()
        if opt.grayscale:
            self.channel=1
        else:
            self.channel=3
    

    def forward(self,opt,var,mode=None):
        if mode == "train":
            var.rgb = self.neural_image.forward(opt, var.input) 
        else:
            var.rgb = self.neural_image.forward(opt, var.test_coords)
        return var
        
    def compute_loss(self,opt,var,mode=None):
        loss = edict()
        if opt.loss_weight.render is not None:
            if mode == "train":
                loss.render = self.mse_loss(var.rgb,var.label)
            else:
                loss.render = self.mse_loss(var.rgb, var.gt_rgb)
        return loss


class ReLULayer(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.input_features = in_features
        self.linear = torch.nn.Linear(in_features, out_features)
        self.relu = torch.nn.ReLU()

    def forward(self, input):
        return self.relu(self.linear(input))


class NeuralImageFunction(torch.nn.Module):
    def __init__(self,opt):
        super().__init__()
        self.define_network(opt)


    def define_network(self,opt):
        self.mlp = []
        
        ## positional encoding ##
        if opt.arch.relu.posenc.enabled:
            log.warning("Positional encoding enabled...")
            input_features = 2 + 4*opt.arch.relu.posenc.L_2D 
        else:
            input_features = 2
        output_features = 3
        
        self.mlp.append(ReLULayer(input_features, opt.arch.relu.hidden_features))
        for i in range(opt.arch.relu.hidden_layers):
            self.mlp.append(ReLULayer(opt.arch.relu.hidden_features, opt.arch.relu.hidden_features))

        final_linear = torch.nn.Linear(opt.arch.relu.hidden_features, output_features)
        self.mlp.append(final_linear)

        self.mlp = torch.nn.Sequential(*self.mlp)


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


    def forward(self,opt,coord_2D): 

        if opt.arch.relu.posenc.enabled:
            coord_2D_enc = self.positional_encoding(opt, coord_2D, opt.arch.relu.posenc.L_2D)
            points_enc = torch.cat([coord_2D, coord_2D_enc], dim=-1)
        else:
            points_enc = coord_2D

        rgb = self.mlp(points_enc)
        
        return rgb