import os
import sys
import time
import pickle
if pickle.HIGHEST_PROTOCOL < 5:
    import pickle5
from shutil import copyfile
from PIL import Image
import stl
import matplotlib.pyplot as plt
import plyfile
import torch
from typing import Tuple, List, Union, Callable, Type, Iterator, Dict, Set, Optional, Any, Sized
import numpy as np

N = type(None)
V = np.array
ARRAY = np.ndarray
ARRAYS = Union[Tuple[ARRAY, ...], List[ARRAY]]
VS = Union[Tuple[V, ...], List[V]]
VN = Union[V, N]
VNS = Union[VS, N]
T = torch.Tensor
TS = Union[Tuple[T, ...], List[T]]
TN = Optional[T]
TNS = Union[Tuple[TN, ...], List[TN]]
TSN = Union[TS, N]
TA = Union[T, ARRAY]

V_Mesh = Tuple[V, V]
T_Mesh = Tuple[T, Optional[T]]
T_Mesh_T = Union[T_Mesh, T]
COLORS = Union[T, V, Tuple[int, int, int]]


CPU = torch.device('cpu')


def add_suffix(path: str, suffix: str) -> str:
    if len(path) < len(suffix) or path[-len(suffix):] != suffix:
        path = f'{path}{suffix}'
    return path

def load_ply(path: str):
    path = add_suffix(path, '.ply')
    plydata = plyfile.PlyData.read(path)
    keys = {element.name for element in plydata.elements}
    points = [torch.from_numpy(plydata['vertex'][axis]) for axis in ('x', 'y', 'z')]
    points = torch.stack(points, dim=1).float()
    faces = []
    if 'face' in keys:
        keys = {element.name for element in plydata['face'].properties}
        for key in ('vertex_indices', 'vertex_index'):
            if key in keys and 'vertex' in key:
                faces_ = plydata['face'][key]
                if faces_.shape[0] > 0:
                    faces = torch.from_numpy(np.stack(faces_.tolist(), axis=0)).long()
                break
    return points, faces


def load_stl(path: str):
    path = add_suffix(path, '.stl')
    mesh = stl.mesh.Mesh.from_file(path)
    triangles = torch.from_numpy(mesh.vectors.copy()).float()
    vs = triangles.view(-1, 3)
    faces = torch.arange(vs.shape[0]).view(-1, 3)
    return vs, faces


def load_mesh(file_name: str, dtype: Union[type(T), type(V)] = T,
              device: torch.device = CPU) -> Union[T, T_Mesh, V_Mesh, Tuple[T, List[List[int]]]]:

    def off_parser():
        header = None

        def parser_(clean_line: list):
            nonlocal header
            if not clean_line:
                return False
            if len(clean_line) == 3 and not header:
                header = True
            elif len(clean_line) == 3:
                return 0, 0, float
            elif len(clean_line) > 3:
                return 1, -int(clean_line[0]), int

        return parser_

    def obj_parser(clean_line: list):
        nonlocal is_quad
        if not clean_line:
            return False
        elif clean_line[0] == 'v':
            return 0, 1, float
        elif clean_line[0] == 'f':
            is_quad = is_quad or len(clean_line) != 4
            return 1, 1, int
        return False

    def fetch(lst: list, idx: int, dtype: type):
        if '/' in lst[idx]:
            lst = [item.split('/') for item in lst[idx:]]
            lst = [item[0] for item in lst]
            idx = 0
        face_vs_ids = [dtype(c.split('/')[0]) for c in lst[idx:]]
        if dtype is float and len(face_vs_ids) > 3:
            face_vs_ids = face_vs_ids[:3]
        return face_vs_ids

    def load_from_txt(parser) -> TS:
        mesh_ = [[], []]
        with open(file_name, 'r') as f:
            for line in f:
                clean_line = line.strip().split()
                info = parser(clean_line)
                if not info:
                    continue
                data = fetch(clean_line, info[1], info[2])
                mesh_[info[0]].append(data)
        if is_quad:
            faces = mesh_[1]
            for face in faces:
                for i in range(len(face)):
                    face[i] -= 1
        else:
            faces = torch.tensor(mesh_[1], dtype=torch.int64)
            if len(faces) > 0 and faces.min() != 0:
                faces -= 1
        mesh_ = [torch.tensor(mesh_[0], dtype=torch.float32), faces]
        return mesh_

    for suffix in ['.obj', '.off', '.ply', '.stl']:
        file_name_tmp = add_suffix(file_name, suffix)
        if os.path.isfile(file_name_tmp):
            file_name = file_name_tmp
            break

    is_quad = False
    name, extension = os.path.splitext(file_name)
    if extension == '.obj':
        mesh = load_from_txt(obj_parser)
    elif extension == '.off':
        mesh = load_from_txt(off_parser())
    elif extension == '.ply':
        mesh = load_ply(name)
    elif extension == '.stl':
        mesh = load_stl(name)
    else:
        raise ValueError(f'mesh file {file_name} is not exist or not supported')
    if type(mesh[1]) is T and not ((mesh[1] >= 0) * (mesh[1] < mesh[0].shape[0])).all():
        print(f"err: {file_name}")
    assert type(mesh[1]) is not T or ((mesh[1] >= 0) * (mesh[1] < mesh[0].shape[0])).all()
    if dtype is V:
        mesh = mesh[0].numpy(), mesh[1].numpy()
    elif device != CPU:
        mesh = mesh[0].to(device), mesh[1].to(device)
    else:
        mesh = (mesh[0], mesh[1])
        if len(mesh[1]) == 0:
            return mesh[0]
    return mesh

def init_folders(*folders):
    for f in folders:
        dir_name = os.path.dirname(f)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)


def export_mesh(mesh: Union[V_Mesh, T_Mesh, T, Tuple[T, List[List[int]]]], file_name: str,
                colors: Optional[COLORS] = None, normals: TN = None, edges=None, spheres=None):
    # return
    if type(mesh) is not tuple and type(mesh) is not list:
        mesh = mesh, None
    vs, faces = mesh
    if vs.shape[1] < 3:
        vs = torch.cat((vs, torch.zeros(len(vs), 3 - vs.shape[1], device=vs.device)), dim=1)
    file_name = add_suffix(file_name, '.obj')
    if colors is not None:
        colors = colors_to_colors(colors, mesh)
    init_folders(file_name)
    if not os.path.isdir(os.path.dirname(file_name)):
        return
    if faces is not None:
        if type(faces) is T:
            faces: T = faces + 1
            faces_lst = faces.tolist()
        else:
            faces_lst_: List[List[int]] = faces
            faces_lst = []
            for face in faces_lst_:
                faces_lst.append([face[i] + 1 for i in range(len(face))])
    with open(file_name, 'w') as f:
        for vi, v in enumerate(vs):
            if colors is None or colors[vi, 0] < 0:
                v_color = ''
            else:
                v_color = ' %f %f %f' % (colors[vi, 0].item(), colors[vi, 1].item(), colors[vi, 2].item())
            f.write("v %f %f %f%s\n" % (v[0], v[1], v[2], v_color))
        if normals is not None:
            for n in normals:
                f.write("vn %f %f %f\n" % (n[0], n[1], n[2]))
        if faces is not None:
            for face in faces_lst:
                face = [str(f) for f in face]
                f.write(f'f {" ".join(face)}\n')
        if edges is not None:
            for edges_id in range(edges.shape[0]):
                f.write(f'\ne {edges[edges_id][0].item():d} {edges[edges_id][1].item():d}')
        if spheres is not None:
            for sphere_id in range(spheres.shape[0]):
                f.write(f'\nsp {spheres[sphere_id].item():d}')


def unnormalize_image(img: T):
    if img.dim() == 4:
        img = img[0]
    if img.shape[0] == 3:
        img: T = img.permute(1, 2, 0)
        mean = torch.as_tensor((0.485, 0.456, 0.406), dtype=img.dtype, device=img.device)
        std = torch.as_tensor((0.229, 0.224, 0.225), dtype=img.dtype, device=img.device)
        img = (img * std[None, None, :] + mean[None, None, :]) * 255
        img = img.clamp(0, 255)
    return img


def image_to_display(img) -> ARRAY:
    if type(img) is T:
        img = unnormalize_image(img)
        if img.max() <= 1. and img.max() != 0:
            img = img * 255
        img = img.detach().cpu().numpy().astype(np.uint8)
    if type(img) is str:
        img = Image.open(str(img))
    if type(img) is not V:
        img = V(img)
    return img


def imshow(img):
    img = image_to_display(img)
    plt.imshow(img)
    plt.axis("off")
    plt.show()
    plt.close('all')


def add_suffix_if_is_file(path: str, *suffix: str):
    for suffix_ in suffix:
        path_ = add_suffix(path, suffix_)
        if is_file(path_):
            return path_
    raise ValueError




def measure_time(func, num_iters: int, *args):
    start_time = time.time()
    for i in range(num_iters):
        func(*args)
    total_time = time.time() - start_time
    avg_time = total_time / num_iters
    print(f"{str(func).split()[1].split('.')[-1]} total time: {total_time}, average time: {avg_time}")
