'''From the DeepSDF repository https://github.com/facebookresearch/DeepSDF
'''
import torch
import numpy as np
import plyfile
import time
from bocc_utils import files_utils
import skimage.measure
import itertools
import mcubes
import trimesh
import os


        
def create_mesh_from_occupancy(opt, decoder, res=512, return_sdf=False):
    
    os.makedirs('{}/meshes'.format(opt.output_path),  exist_ok=True)
    ply_filename = "{}/meshes/{}".format(opt.output_path, opt.data.scene)
    decoder.eval()
    import tqdm
    # write output
    x = torch.linspace(-1, 1, res)

    x, y, z = torch.meshgrid(x, x, x)
    render_coords = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=-1).cuda()
    sdf_values = [np.zeros((res**3, 1)) for i in range(1)]

    # render in a batched fashion to save memory
    bsize = int(128**2)
    for i in tqdm.tqdm(range(int(res**3 / bsize))):
        coords = render_coords[i*bsize:(i+1)*bsize, :]

        if opt.model == "occupancy_relu_stochastic":
            out  = decoder.forward(opt, coords)
        else:
            out = decoder.forward(coords)


        if not isinstance(out, list):
            out = [out, ]

        for idx, sdf in enumerate(out):
            sdf_values[idx][i*bsize:(i+1)*bsize] = sdf.detach().cpu().numpy()

    if return_sdf:
        return [sdf.reshape(res, res, res) for sdf in sdf_values]

    for idx, sdf in enumerate(sdf_values):
        sdf = sdf.reshape(res, res, res)
        vertices, triangles = mcubes.marching_cubes(-sdf, 0.3)
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        #mesh.vertices = (mesh.vertices / res - 0.5) + 0.5/res

        os.makedirs('{}/meshes'.format(opt.output_path),  exist_ok=True)
        mesh.export(f"{opt.output_path}/meshes/{opt.data.scene}.obj")
    
    
