import glob
import math
import os

import matplotlib.colors as colors
import numpy as np
import scipy.ndimage
import torch
import PIL
import PIL.Image
from torch.utils.data import Dataset
import util
from util import log
import matplotlib.pyplot as plt
import warp


class Image(Dataset):
    def __init__(self, opt):
        super().__init__()
        
        self.load_image(opt)

    def load_div(self,opt):
        """
        npz_data["train_data"] and npz_data["test_data"] have different sets of images

        Return:
            self.image_raw ([C,H,W])
        """
        id = opt.data.div2k_id
        npz_data = np.load('data/data_div2k.npz')
        out = {
            "data_grid_train": npz_data["train_data"]/255., 
            "data_test": npz_data["test_data"]/255.,
        }
        if opt.data.div2k_mode == "test":
            image_raw = torch.from_numpy(out["data_test"][id]).type(torch.FloatTensor).permute(2,0,1)
        else:
            image_raw = torch.from_numpy(out["data_grid_train"][id]).type(torch.FloatTensor).permute(2,0,1)            
        image = torch.nn.functional.interpolate(image_raw.unsqueeze(0), size=(opt.data.image_size[0], opt.data.image_size[1]), mode='bilinear')
        return image.squeeze(0)

    def load_image(self, opt):


        self.image_raw = self.load_div(opt)
        opt.H = self.image_raw.shape[1]
        opt.W = self.image_raw.shape[2]

        self.channel = self.image_raw.shape[0]
        if self.channel == 1:
            log.info("Loaded grayscale image")
        self.coords = warp.get_normalized_pixel_grid_equal(opt)
        self.image_raw = self.image_raw[:3]
        self.labels = self.image_raw.view(opt.batch_size, self.channel, opt.H*opt.W).permute(0,2,1)

       
    def __len__(self):
        return self.coords.shape[1]
    
    def __getitem__(self, idx):
        
        gt_dict = {"labels": self.labels[0]}
        return self.coords[0, idx], self.labels[0,idx]
            

        