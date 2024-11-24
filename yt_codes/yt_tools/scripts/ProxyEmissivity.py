import argparse
import numpy as np
import yt
import aglio
import glob
import re
import os
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager, Value, Lock

from ..yt_spherical_3D import spherical_data
from .. import geometry
from ..stream_line import field_line, trilinear_interpolation

def ProxyEmissivity_Process_Cartesian(start_idx, end_idx, points, progress_list, lock, print_interval=100, t0=0, max_steps=10000):
    res_list = []
    for idx in range(start_idx, end_idx):
        point = points[idx]
        npoints = len(points)
        fline = field_line(bxyz, point, dxyz=dxyz, max_step=max_step)
        is_skip = len(fline)<2 or (fline[0,2]>2) or (fline[-1,2]>2)
        if is_skip:
            with lock:
                progress_list[0]+=1
                progress_list[1]+=1
                if progress_list[0] % print_interval == 0 or progress_list[0] == npoints or progress_list[0]==1:
                    ti = time.time()
                    wall_time = (ti-t0)/60
                    print(
                        f'Process: {progress_list[0]:6d}/{npoints}, '+\
                        f'Percent: {(progress_list[0]/npoints*100):6.2f} %, '+\
                        f'Wall_time: {wall_time:6.3f} min, '+\
                        f'Skip_count: {progress_list[1]:6d}', flush=True
                    )
            continue
        J_alone = trilinear_interpolation(J_dat, fline)
        ss = np.linalg.norm(fline[1:]-fline[:-1], axis=1)
        J2_av = np.sum(ss*J_alone[1:]**2)/ss.sum()
        ix,iy,iz = fline.astype(int).T
        ix       = np.clip(ix,0,nx-1)
        iy       = np.clip(iy,0,ny-1)
        iz       = np.clip(iz,0,nz-1)
        # proxy_emissivity_dat[ix,iy,iz] += J2_av
        res_list.append([ix,iy,iz,J2_av])

        with lock:
            progress_list[0] += 1  # 更新计数器
            if progress_list[0] % print_interval == 0 or progress_list[0] == npoints or progress_list[0]==1:
                ti = time.time()
                wall_time = (ti-t0)/60
                print(
                    f'Process: {progress_list[0]:6d}/{npoints}, '+\
                    f'Percent: {(progress_list[0]/npoints*100):6.2f} %, '+\
                    f'Wall_time: {wall_time:6.3f} min, '+\
                    f'Skip_count: {progress_list[1]:6d}', flush=True
                )   
    return res_list

def parallel_ProxyEmissivity_Cartesian(points, n_cores=10, print_interval=100, max_steps=10000):
    print('### =========== Parallel computing ============ ###')
    print(f'#       Available CPU cores: {multiprocessing.cpu_count():3d}                  #')
    print(f'#            Used CPU cores: {n_cores:3d}                  #')
    print('### =========================================== ###')
    t0 = time.time()
    n_points = len(points)
    chunk_size = n_points // n_cores

    proxy_emissivity = []

    # 使用 Manager 来共享进度信息
    manager = Manager()
    progress_list = manager.list([0,0])  # 用于存储已完成的任务数，初始化为0
    lock = manager.Lock()  # 使用Manager提供的Lock
    print('Initialization OK')

    # 开始并行计算
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        futures = []
        for i in range(n_cores):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < n_cores - 1 else n_points
            futures.append(executor.submit(ProxyEmissivity_Process_Cartesian, start_idx, end_idx, points, progress_list, lock, print_interval, t0, max_step))

        print('Assigning task OK', flush=True)
        for future in as_completed(futures):
            proxy_emissivity.extend(future.result())

    ret = np.zeros((nx,ny,nz))
    for ix,iy,iz,J2_av in proxy_emissivity:
        ret[ix,iy,iz]=J2_av

    return ret

def rk45(rfun, x, dl, **kwargs):
    sig = kwargs.get('sig', 1)
    x0  = x
    k1  = rfun(x0            , **kwargs)
    k2  = rfun(x0+sig*k1*dl/2, **kwargs)
    k3  = rfun(x0+sig*k2*dl/2, **kwargs)
    k4  = rfun(x0+sig*k3*dl  , **kwargs)
    k   = (k1+2*k2+2*k3+k4)/6*sig
    ret = x0+k*dl
    return ret

def magline_stepper(xyz, **kwargs):
    eps = kwargs.get('eps', 1e-10)
    if np.any(np.isnan(xyz)):
        return np.full(3,np.nan)
    x,y,z = xyz
    idx   = instance.xyz2idx(xyz)
    bvec  = trilinear_interpolation(bxyz.transpose(1,2,3,0),idx)
    bmag  = np.linalg.norm(bvec)
    bhat  = bvec/(bmag+eps)
    ret   = bhat if bmag>eps else np.full(3,np.nan)
    return ret

def magline_solver(points, **kwargs):
    rb  = kwargs.get('rmin', 1.0)
    Ns  = kwargs.get('max_steps', 10000)
    dl  = kwargs.get('step_size', dxyz.min())
    # print(f'max_steps: {Ns}')
    ret = []
    for xyz in points:
        xyz0   = xyz
        idx0   = instance.xyz2idx(xyz0)
        line_f = [idx0]
        line_b = []
        # forward
        flag = True
        iter = 0
        while flag and iter<Ns:
            # print(f'iter: {iter:04d}')
            xyz  = rk45(magline_stepper, xyz, dl, **kwargs)
            iter+=1
            flag = not np.any(np.isnan(xyz)) and np.linalg.norm(xyz)>=rb
            idx  = instance.xyz2idx(xyz)
            if flag:
                line_f.append(idx)
        # backward
        xyz  = xyz0
        flag = True
        iter = 0
        while flag and iter<Ns:
            xyz = rk45(magline_stepper, xyz, dl, sig=-1, **kwargs)
            iter+=1
            flag = not np.any(np.isnan(xyz)) and np.linalg.norm(xyz)>=rb
            idx  = instance.xyz2idx(xyz)
            if flag:
                line_b.append(idx)
        line = np.array(line_b[::-1]+line_f)
        ret.append(line)
    return ret

def ProxyEmissivity_Process_Spherical(idx_i,idx_f,points,progress,lock,t0,n_print=100,max_step=10000,rmin=1.01):
    ret = []
    for idx in range(idx_i,idx_f):
        point = [points[idx]]
        fline = magline_solver(point, max_step=max_step, rmin=rb, step_size=dl)[0]
        # print(fline)
        npoints = len(points)
        xyz = instance.idx2xyz(fline)
        end_point = xyz[-1]
        is_skip = (len(fline)<2) or (np.linalg.norm(end_point)>rmin) or (np.linalg.norm(xyz[0])>rmin)
        if not is_skip:
            J_along = trilinear_interpolation(J_dat, fline)
            ss      = np.linalg.norm(fline[1:]-fline[:-1], axis=1)
            J2_av   = np.sum(ss*J_along[1:]**2)/ss.sum()
            ix,iy,iz = np.around(fline).astype(int).T
            ix       = np.clip(ix,0,nx-1)
            iy       = np.clip(iy,0,ny-1)
            iz       = np.clip(iz,0,nz-1)
            ret.append([ix,iy,iz,J2_av])
        with lock:
            progress[0]+=1
            progress[1]+=1 if is_skip else 0
            if progress[0] % n_print==0 or progress[0]==npoints or progress[0]==1:
                ti = time.time()
                wall_time = (ti-t0)/60
                print(
                    f"Process: {progress[0]:6d}/{npoints}, "
                    f"Percent: {progress[0]/npoints*100: 6.2f}%, "
                    f"Wall_time: {wall_time:6.3f} min, "
                    f"Skip_count: {progress[1]:6d}",
                flush=True
                )
    return ret

def parallel_ProxyEmissivity_Spherical(points, n_cores=10, print_interval=100, max_steps=10000):
    print('### =========== Parallel computing ============ ###')
    print(f'#       Available CPU cores: {multiprocessing.cpu_count():3d}                  #')
    print(f'#            Used CPU cores: {n_cores:3d}                  #')
    print('### =========================================== ###')
    t0 = time.time()
    n_points = len(points)
    chunk_size = n_points // n_cores

    proxy_emissivity = []

    # 使用 Manager 来共享进度信息
    manager = Manager()
    progress = manager.list([0,0])  # 用于存储已完成的任务数，初始化为0
    lock = manager.Lock()  # 使用Manager提供的Lock
    print('Initialization OK')

    # 开始并行计算
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        futures = []
        for i in range(n_cores):
            idx_i = i * chunk_size
            idx_f = (i + 1) * chunk_size if i < n_cores - 1 else n_points
            futures.append(executor.submit(ProxyEmissivity_Process_Spherical, idx_i, idx_f, points, progress, lock, t0, print_interval, max_steps))

        print('Assigning task OK', flush=True)
        for future in as_completed(futures):
            proxy_emissivity.extend(future.result())

    ret = np.zeros((nx,ny,nz))
    for ix,iy,iz,J2_av in proxy_emissivity:
        ret[ix,iy,iz]+=J2_av

    return ret

parser = argparse.ArgumentParser(description='Parallel computing ProxyEmmisivity')
parser.add_argument('-i'        , type=str, help='Path to .npz file for required information'   , required=False, default='./proxy_emissivity_info.npz')
parser.add_argument('-n'        , type=int, help='Number of cores to use'                       , required=False, default=10                           )
parser.add_argument('--max_step', type=int, help='maximum step of the line iteration'           , required=False, default=10000                        )
parser.add_argument('--n_print' , type=int, help='Print interval'                               , required=False, default=100                          )
parser.add_argument('--geometry', type=str, help="Coordinate system, 'cartesian' or 'spherical'", required=False, default='cartesian'                  )

args           = parser.parse_args()
info           = np.load(args.i)
max_steps      = args.max_step
print_interval = args.n_print
n_cores        = args.n
bxyz           = info['bxyz']
dxyz           = info['dxyz']
points         = info['points']
ti             = time.time()
nx,ny,nz       = bxyz.shape[1:]
coords         = args.geometry
if coords == 'cartesian':
    current = geometry.rot(bxyz, dxyz=dxyz)
    J_dat   = np.linalg.norm(current, axis=0)
    proxy_emissivity = parallel_ProxyEmissivity_Cartesian(points, n_cores=n_cores, print_interval=print_interval, max_steps=max_steps)
    save_name = f'./proxy_emissivity_temp.npy'
    np.save(save_name, proxy_emissivity)
elif coords=='spherical':
    X  = info['X']
    Y  = info['Y']
    Z  = info['Z']
    rb = info['rb']
    dl = info.get('dl',dxyz.min())
    instance = spherical_data.load('instance_temp.pkl')
    current = geometry.rot(bxyz, dxyz=dxyz)
    J_dat   = np.linalg.norm(current, axis=0)
    proxy_emissivity = parallel_ProxyEmissivity_Spherical(points, n_cores=n_cores, print_interval=print_interval, max_steps=max_steps)
    save_name = f'./proxy_emissivity_temp.npy'
    np.save(save_name, proxy_emissivity)
else:
    raise ValueError("`coords` is not 'spherical' or 'cartesian'...")
# print(f'!!!    Save file: {save_name}    !!!\n\n')