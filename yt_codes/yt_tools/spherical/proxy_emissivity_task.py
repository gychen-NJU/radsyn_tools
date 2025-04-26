import os
import time
import argparse
import multiprocessing

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager, Value, Lock
from scipy.interpolate import RegularGridInterpolator

from ..need import *
from .. import geometry
from yt_tools.geometry import rtp2xyz
from ..stream_line import field_line, trilinear_interpolation,rk45

# ==========================================================
def magline_stepper(rtp, **kwargs):
    eps = kwargs.get('eps', 1e-16)
    r,t,p = rtp if rtp.ndim==1 else rtp.T
    t = t % (np.pi)
    p = p % (np.pi*2)
    if rtp.ndim==1:
        if r>rlist.max() or r<rlist.min() or np.any(np.isnan(rtp)):
            return np.full(3,np.nan)
        rtp = np.array([r,t,p])
        br = br_interp(rtp)
        bt = bt_interp(rtp)
        bp = bp_interp(rtp)
        bn = np.linalg.norm([br,bt,bp])
        if bn<eps:
            return np.full(3,np.nan)
        ret = np.array([br[0]/bn,bt[0]/bn/r,bp[0]/bn/(r*np.sin(t))])
        return ret
    else:
        br = br_interp(rtp)
        bt = bt_interp(rtp)
        bp = bp_interp(rtp)
        bn = np.linalg.norm(np.stack([br,bt,bp],axis=-1), axis=-1)
        null_point = np.where(bn<eps)
        bn[null_point]+=eps
        ret = np.stack([br/bn,bt/bn,bp/bn], axis=-1)/np.stack([np.ones_like(r),1/r,1/(r*np.sin(t))], axis=-1)
        ret[null_point]=np.full(3,np.nan)
    return ret

def magline_solver(rtp, **kwargs):
    Rl = kwargs.get('Rmin',rr.min())
    Ru = kwargs.get('Rmax',rr.max())
    Ns = kwargs.get('max_steps', int(1e5))
    dl = kwargs.get('step_length', 5e-3)
    rtp0 = rtp
    forward  = [rtp0]
    backward = []
    # forward integral
    for i in range(Ns):
        if rtp0[0]<Rl or rtp0[0]>Ru or np.any(np.isnan(rtp0)):
            if len(forward)<2:
                forward.pop()
                break
            rtp0 = forward[-2]
            kk   = magline_stepper(rtp0)
            dl1  = (Rl-rtp0[0])/kk[0]
            dl1  = 1e4 if dl1<0 else dl1
            dl2  = (Ru-rtp0[0])/kk[0]
            dl2  = 1e4 if dl2<0 else dl2
            dl0  = np.min([dl1,dl2])
            rtp0 = rtp0+dl0*kk
            if dl1<dl2:
                rtp0[0]=Rl
            else:
                rtp0[0]=Ru
            forward[-1]=rtp0
            break
        rtp0 = rk45(magline_stepper, rtp0, dl, sig= 1)
        forward.append(rtp0)
    # backward integral
    rtp0 = rtp
    for i in range(Ns):
        if rtp0[0]<Rl or rtp0[0]>Ru or np.any(np.isnan(rtp0)):
            if len(backward)<2:
                rtp0 = rtp
            else:
                rtp0 = backward[-2]
            kk   = magline_stepper(rtp0)
            dl1  = (Rl-rtp0[0])/kk[0]*(-1)
            dl1  = 1e4 if dl1<0 else dl1
            dl2  = (Ru-rtp0[0])/kk[0]*(-1)
            dl2  = 1e4 if dl2<0 else dl2
            dl0  = np.min([dl1,dl2])
            rtp0 = rtp0-kk*dl0
            if dl1<dl2:
                rtp0[0]=Rl
            else:
                rtp0[0]=Ru
            backward[-1]=rtp0
            break
        rtp0 = rk45(magline_stepper, rtp0, dl, sig=-1)
        backward.append(rtp0)
    magline = np.array(backward[::-1]+forward)
    return magline

def rtp2idx(rtp):
    rmin  = rlist.min()
    dr    = rlist[1]-rlist[0]
    dt    = tlist[1]-tlist[0]
    dp    = plist[1]-plist[0]
    if np.array(rtp).ndim==1:
        r,t,p = rtp
        ir    = (r-rmin)/dr
        it    = (t-np.pi)/dt
        ip    = (p-0)/dp
        return np.array([ir,it,ip])
    else:
        r,t,p = rtp.T
        ir    = (r-rmin)/dr
        it    = (t-np.pi)/dt
        ip    = (p-0)/dp
        return np.stack([ir,it,ip], axis=-1)

def ProxyEmissivity_Process_Spherical(idx_i,idx_f,points,progress,lock,t0,n_print=100,max_step=100000,rmin=1.01):
    ret = []
    for idx in range(idx_i,idx_f):
        point = points[idx]
        fline = magline_solver(point, Rmin=Rmin, Rmax=Rmax, step_size=dl, max_step=Ns)
        non_nan_row = ~np.isnan(fline).any(axis=1)
        fline = fline[non_nan_row]
        fline[:,1] = fline[:,1] % np.pi
        fline[:,2] = fline[:,2] % (np.pi*2)
        npoints = len(points)
        is_skip = (len(fline)<2) or (fline[0,0]>rmin) or (fline[-1,0]>rmin)
        if not is_skip:
            J_along  = J_interp(fline)
            xyz      = rtp2xyz(fline.T).T
            ss       = np.linalg.norm(xyz[1:]-xyz[:-1], axis=1)
            J2_av    = np.sum(ss*J_along[1:]**2)/(ss.sum()+1e-16)
            ir,it,ip = np.around(rtp2idx(fline).T).astype(int)
            ret.append([ir,it,ip,J2_av])
        with lock:
            progress[0]+=1
            progress[1]+=1 if is_skip else 0
            if progress[0] % n_print==0 or progress[0]==npoints or progress[0]==1:
                ti = time.time()
                wall_time = (ti-t0)/60
                print(
                    f"Process: {progress[0]:6d}/{npoints}, "
                    f"Percent: {progress[0]/npoints*100: 7.2f}%, "
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
    n_tasks  = len(points)
    n_chunks = n_tasks//n_cores
    res      = n_tasks % n_cores
    chunks   = [n_chunks+1]*res+[n_chunks]*(n_cores-res)
    idx_i    = 0
    idx_f    = 0

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
            idx_f+=chunks[i]
            futures.append(
                executor.submit(
                    ProxyEmissivity_Process_Spherical, 
                    idx_i, idx_f, points, progress, lock, t0, print_interval, max_steps))
            idx_i = idx_f

        print('Assigning task OK', flush=True)
        for future in as_completed(futures):
            proxy_emissivity.extend(future.result())

    ret = np.zeros((Nr,Nt,Np))
    for ir,it,ip,J2_av in proxy_emissivity:
        ret[ir,it,ip]+=J2_av

    return ret

# ==========================================================
parser = argparse.ArgumentParser(description='Parallel computing ProxyEmmisivity')
parser.add_argument('-i'        , type=str, help='Path to .npz file for required information'   , required=False, default='./proxy_emissivity_info.npz')

args = parser.parse_args()
info = np.load(args.i)

points = info['points']
rtp    = info['rtp']
Brtp   = info['Brtp']
rlist  = rtp[0,:,0,0]
tlist  = rtp[1,0,:,0]
plist  = rtp[2,0,0,:]
Jrtp   = geometry.rot(Brtp, rtp=rtp)
Jmag   = np.linalg.norm(Jrtp,axis=0)
Rmin   = info.get('Rmin',rlist.min())
Rmax   = info.get('Rmax',rlist.max())
Ns     = info.get('max_steps',int(1e5))
dl     = info.get('step_length',rlist[1]-rlist[0])
Nc     = info.get('n_cores', 50)
NP     = info.get('n_print', 1000)
SN     = info.get('save_name', './proxy_emissivity_temp.npy')

Nr,Nt,Np  = Jmag.shape
br_interp = RegularGridInterpolator((rlist,tlist,plist),Brtp[0], method='linear')
bt_interp = RegularGridInterpolator((rlist,tlist,plist),Brtp[1], method='linear')
bp_interp = RegularGridInterpolator((rlist,tlist,plist),Brtp[2], method='linear')
J_interp  = RegularGridInterpolator((rlist,tlist,plist),   Jmag, method='linear')

proxy_emissivity = parallel_ProxyEmissivity_Spherical(
    points, 
    n_cores=Nc, 
    print_interval=NP, 
    max_steps=Ns
)

np.save(SN, proxy_emissivity)