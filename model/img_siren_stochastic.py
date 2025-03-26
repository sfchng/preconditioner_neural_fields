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
import scipy
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

        trainloader = DataLoader(self.dataset, batch_size=opt.data.batch_size, shuffle=True)

        var = edict(idx=torch.arange(opt.batch_size))
        var.test_coords = self.dataset.coords
        var.test_rgbs = self.dataset.labels
        var.image = self.dataset.image_raw

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
    

    def train_iteration_adahessian(self,opt,var,loader, trainloader):
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

    def get_params_grad(self):
        """
        Get model parameters with corresponding gradients
        """
        params = []
        grads = []
        
        for param in self.graph.neural_image.parameters():
            params.append(param)
            grads.append(param.grad)
        
        return params, grads
    
    def compute_eigenvals_hvp(self, top_n=1, max_iter=5000, tol=1.e-3):
        """
        Compute the top_n eigenvalues using power iteration algorithm
        
        top_n: top n eigenvalues
        max_iter: maximum number of iterations used to compute each single eigenvalues
        tol: the relative tolerance between two consecutive eigenvalue computations from power iterations
        """
        
        assert top_n >= 1
        eigenvalues =  []
        eigenvectors = []
        computed_n = 0
        
        params, grads = self.get_params_grad()
        while computed_n < top_n:
            eigenvalue = None
            v = [torch.randn(p.size()).to("cuda") for p in params]
            
            v = self.normalization(v)
        
            """ Compute Rayleigh quotient """
            for i in range(max_iter):
                v = self.orthnormal(v, eigenvectors)
                self.graph.zero_grad()
                Hv = self.hvp(grads, params, v)
                tmp_eigenvalue = self.group_product(Hv, v).cpu().item()
                
                
                if eigenvalue == None:
                    eigenvalue = tmp_eigenvalue
                else:
                    if abs(eigenvalue - tmp_eigenvalue) / (abs(eigenvalue) + 1e-6) < tol:
                        break
                    else:
                        eigenvalue = tmp_eigenvalue
                        
            eigenvalues.append(eigenvalue)
            eigenvectors.append(v)
            computed_n += 1
        
    
    def hvp(self, grads, params, v):
        Hv = torch.autograd.grad(grads, params, grad_outputs=v, only_inputs=True,  retain_graph=True)
        return Hv
    
    
    def compute_hessian(self, loss, D=None):

        grads = torch.autograd.grad(loss.all, self.graph.neural_image.parameters(), create_graph=True, retain_graph=True)
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
    
    def compute_condition_num(self, hessian):
        eig_vals = torch.linalg.eigvals(hessian)
        condition_num = eig_vals.abs().max() / eig_vals.abs().min()
        return condition_num


    def normalization(self, v):
        """
        normalization of a list of vectors
        return: normalized vectors v
        """
        s = self.group_product(v, v)
        s = s**0.5
        s = s.cpu().item()
        v = [vi / (s + 1e-6) for vi in v]
        return v
    
    def group_product(self, xs, ys):
        """
        the inner product of two lists of variables xs,ys
        :param xs:
        :param ys:
        :return:
        """
        return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])    
        
    def orthnormal(self, w, v_list):
        """
        make vector w orthogonal to each vector in v_list.
        afterwards, normalize the output w
        """
        for v in v_list:
            w = self.group_add(w, v, alpha=-self.group_product(w, v))
        return self.normalization(w) 
    
    def group_add(self, params, update, alpha=1):
        """
        params = params + update*alpha
        :param params: list of variable
        :param update: list of data
        :return:
        """
        for i, p in enumerate(params):
            params[i].data.add_(update[i] * alpha)
        return params       

    @torch.no_grad()
    def log_cond_num(self,cond_num,step=0,split="train", mode="condition_num"):
        
        ## log learning rate ##
        if split == "train":
            self.tb.add_scalar("{0}/{1}".format(split, mode),cond_num,step)


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
                util_vis.tb_image(opt,self.tb,step,split,"groundtruth",var.image[None])
                util_vis.tb_image(opt,self.tb,step,split,"predicted",var.rgb_map.view(opt.batch_size,opt.H,opt.W,var.image.shape[0]).permute(0,3,1,2))

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
            var.rgb = self.neural_image.forward(var.input)
        else:
            var.rgb = self.neural_image.forward(var.test_coords.squeeze(0))
        return var
        
    def compute_loss(self,opt,var,mode=None):
        loss = edict()
        if opt.loss_weight.render is not None:
            image = var.image.view(opt.batch_size,self.channel,opt.H*opt.W).permute(0,2,1)
            if mode == "train":
                loss.render = self.mse_loss(var.rgb,var.label.unsqueeze(0))
            else:
                loss.render = self.mse_loss(var.rgb, image)
        return loss


class SineLayer(torch.nn.Module):    
    def __init__(self, in_features, out_features, opt, bias=True, is_first=False, omega=30, trainable=True):
        super().__init__()
        self.omega =omega
        self.is_first = is_first
        self.input_features = in_features

        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
        self.init_weights() ## migrate to linear layer
        
        if trainable != True:
            self.linear.weight.requires_grad_(False)
            self.linear.bias.requires_grad_(False)

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1/self.input_features, 1/self.input_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6/self.input_features)/self.omega,
                                            np.sqrt(6/self.input_features)/self.omega)

    def forward(self, input):
        return torch.sin(self.omega * self.linear(input))


class NeuralImageFunction(torch.nn.Module):
    def __init__(self,opt):
        super().__init__()
        self.define_network(opt)


    def define_network(self,opt):
        self.mlp = []
        input_features = 2
        output_features = 3

        self.mlp.append(SineLayer(input_features, opt.arch.siren.hidden_features, opt, is_first=True, omega=opt.arch.siren.first_omega))
        for i in range(opt.arch.siren.hidden_layers):
            self.mlp.append(SineLayer(opt.arch.siren.hidden_features, opt.arch.siren.hidden_features, opt, is_first=False, omega=opt.arch.siren.hidden_omega))
                
        if opt.arch.siren.outermost_linear:
            final_linear = torch.nn.Linear(opt.arch.siren.hidden_features, output_features)
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6/opt.arch.siren.hidden_features) / opt.arch.siren.hidden_omega,
                                                np.sqrt(6/opt.arch.siren.hidden_features)/ opt.arch.siren.hidden_omega)
                self.mlp.append(final_linear)
        else:
            self.mlp.append(SineLayer(opt.arch.siren.hidden_features, output_features, is_first=False, omega=opt.arch.siren.hidden_omega))

        self.mlp = torch.nn.Sequential(*self.mlp)


    def forward(self,coord_2D): 

        rgb = self.mlp(coord_2D)
        
        return rgb


