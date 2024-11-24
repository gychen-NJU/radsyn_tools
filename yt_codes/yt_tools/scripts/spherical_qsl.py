# spherical_qsl.py
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from ..need import *
from .. import geometry
from ..funcs import trilinear_interpolation
from ..funcs import Generalized_Jacobian, grad, div, rot
from ..spherical.data_loader import spherical_data
from ..geometry import rtp2xyz, xyz2rtp

# ======================================================================

def initializeUV(rtp=None, xyz=None,  **kwargs):
    brtp = instance.get_Brtp(rtp=rtp,xyz=xyz,**kwargs)
    bmag = np.linalg.norm(brtp)
    bhat = brtp/bmag
    err  = kwargs.get('err', 1e-10)
    Vvec  = np.zeros(3)
    Uvec  = np.zeros(3)
    if (np.abs(bhat[2])<err):
        if (np.abs(bhat[0])<err):
            if (np.abs(bhat[1])<err):
                Vvec = np.zeros(3)
            else:
                Vvec[0] = 1.
                Vvec[1] = -bhat[0]/bhat[1]
                Vvec[2] = 0.
        else:
            Vvec[0] = -bhat[1]/bhat[0]
            Vvec[1] = 1.
            Vvec[2] = 0.
    else:
        Vvec[0] = 0.
        Vvec[1] = 1.
        Vvec[2] = -bhat[1]/bhat[2]
    Uvec = np.cross(bhat, Vvec)
    Vn   = np.linalg.norm(Vvec)
    Un   = np.linalg.norm(Uvec)
    Vvec = Vvec/Vn if Vn>=err else Vvec
    Uvec = Uvec/Un if Un>=err else Uvec
    return Uvec, Vvec

def rk45_stepper(rfun, x, y, dl=0.01, **kwargs):
    sig = kwargs.get('sig', 1)
    x0  = x
    k1  = rfun(x0         , y            , **kwargs)
    k2  = rfun(x0+sig*dl/2, y+sig*k1*dl/2, **kwargs)
    k3  = rfun(x0+sig*dl/2, y+sig*k2*dl/2, **kwargs)
    k4  = rfun(x0+sig*dl  , y+sig*k3*dl  , **kwargs)
    k   = (k1+2*k2+2*k3+k4)/6*sig
    ret = y+k*dl
    return ret

def rfun(l,y,**kwargs):
    bl     = kwargs.get('bottom_left', instance.bbox.T[0])
    ur     = kwargs.get('upper_right', instance.bbox.T[1])
    r,t,p  = y[ :3]
    X0     = y[ :3] # r,t,p
    if np.any(np.isnan(y)) or np.any((X0<bl) | (X0>ur)):
        return np.array([np.nan]).repeat(len(y))
    U0     = y[3:6]
    V0     = y[6: ]
    idx    = instance.rtp2idx(X0)
    bhat   = trilinear_interpolation(    Bhat.transpose(1,2,3,0), idx)
    jacobi = trilinear_interpolation(Jacobi.transpose(2,3,4,0,1), idx)
    dX     = bhat/np.array([1,r,r*np.sin(t)])
    dU     = U0[0]*jacobi[:,0]+U0[1]*jacobi[:,1]+U0[2]*jacobi[:,2]
    dV     = V0[0]*jacobi[:,0]+V0[1]*jacobi[:,1]+V0[2]*jacobi[:,2]
    ret    = np.hstack([dX,dU,dV])
    return ret

def Line_Integration(rtp, **kwargs):
    dl    = kwargs.get('dl'         , instance.drtp[0]  )
    Ns    = kwargs.get('max_steps'  , 1000000           )
    bl    = kwargs.get('bottom_left', instance.bbox.T[0])
    ur    = kwargs.get('upper_right', instance.bbox.T[1])
    xyz   = kwargs.pop('xyz', None)
    X0    = rtp
    U0,V0 = initializeUV(rtp=X0,xyz=xyz,**kwargs)
    yf_list = list()
    yb_list = list()
    # forward integration
    yy    = np.hstack([X0,U0,V0])
    sig   = 1
    iter  = 0
    lf    = 0
    stop  = False
    while iter<Ns and not stop:
        # yf_list.append(yy)
        iter+=1
        yf = rk45_stepper(rfun, lf, yy, dl=dl, sig=sig)
        if np.any(np.isnan(yf)) or np.min(yf[:3]-bl)<0 or np.min(ur-yf[:3])<0:
            xx  = yy[:3]
            idx = instance.rtp2idx(xx)
            bb  = trilinear_interpolation(Bhat.transpose(1,2,3,0), idx)
            dl1 = ((bl-xx)/(sig*bb))*np.array([1,xx[0],xx[0]*np.sin(xx[1])])
            dl1 = 1e4 if len(dl1[dl1>=0])==0 else np.min(dl1[dl1>=0])
            dl2 = ((ur-xx)/(sig*bb))*np.array([1,xx[0],xx[0]*np.sin(xx[1])])
            dl2 = 1e4 if len(dl2[dl2>=0])==0 else np.min(dl2[dl2>=0])
            dl0 = np.min([dl1,dl2])
            yf   = yy+sig*dl0*rfun(lf,yy)
            lf   = lf+sig*dl0
            if dl1<dl2:
                if np.min(yf[:3]-bl)<0:
                    idir = np.argmin(np.abs(yf[:3]-bl))
                    yf[idir]=bl[idir]
            else:
                if np.min(ur-yf[:3])<0:
                    idir = np.argmin(np.abs(yf[:3]-ur))
                    yf[idir]=ur[idir]
            stop = True
        else:
            yy = yf
            lf = lf+sig*dl
    if iter==Ns:
        print(f'Forward Integration over {Ns} iterations')
    # backward integration
    yy    = np.hstack([X0,U0,V0])
    sig   = -1
    iter  = 0
    lb    = 0
    stop  = False
    while iter<Ns and not stop:
        # yb_list.append(yy)
        iter+=1
        yb = rk45_stepper(rfun, lb, yy, dl=dl, sig=sig)
        if np.any(np.isnan(yb)) or np.min(yb[:3]-bl)<0 or np.min(ur-yb[:3])<0:
            xx  = yy[:3]
            idx = instance.rtp2idx(xx)
            bb  = trilinear_interpolation(Bhat.transpose(1,2,3,0), idx)
            dl1 = ((bl-xx)/(sig*bb))
            dl1 = 1e4 if len(dl1[dl1>=0])==0 else np.min(dl1[dl1>=0])
            dl2 = ((ur-xx)/(sig*bb))
            dl2 = 1e4 if len(dl2[dl2>=0])==0 else np.min(dl2[dl2>=0])
            dl0 = np.min([dl1,dl2])
            yb   = yy+sig*dl0*rfun(lb,yy)
            lb   = lb+sig*dl0
            if dl1<dl2:
                if np.min(yb[:3]-bl)<0:
                    idir = np.argmin(np.abs(yb[:3]-bl))
                    yb[idir]=bl[idir]
            else:
                if np.min(ur-yb[:3])<0:
                    idir = np.argmin(np.abs(yb[:3]-ur))
                    yb[idir]=ur[idir]
            stop = True
        else:
            yy = yb
            lb = lb+sig*dl
    if iter==Ns:
        print(f'Backward Integration over {Ns} iterations')

    return yf,yb,lf,lb #,yf_list,yb_list

def calculate_qsl_range(idx_i, idx_e, points, progress, total_tasks, lock, 
                        print_interval=100, t0=None):
    t0 = time.time() if t0 is None else t0
    res_list = []
    for icnt in range(idx_i, idx_e):
        X0          = points[icnt]
        yf,yb,lf,lb = Line_Integration(X0, dl=dl, max_steps=Ns)
        Xf,UF,VF    = yf[:3],yf[3:6],yf[6:]
        Xb,UB,VB    = yb[:3],yb[3:6],yb[6:]
        idx_0       = instance.rtp2idx(X0)
        idx_f       = instance.rtp2idx(Xf)
        idx_b       = instance.rtp2idx(Xb)
        BF          = trilinear_interpolation(Brtp.transpose(1,2,3,0), idx_f)
        bF          = BF/np.linalg.norm(BF)
        BB          = trilinear_interpolation(Brtp.transpose(1,2,3,0), idx_b)
        bB          = BB/np.linalg.norm(BB)
        B0          = trilinear_interpolation(Brtp.transpose(1,2,3,0), idx_0)
        b0          = B0/np.linalg.norm(B0)
        B0,BF,BB    = np.linalg.norm(B0),np.linalg.norm(BF),np.linalg.norm(BB)
        UF          = UF-np.dot(bF,UF)*bF
        VF          = VF-np.dot(bF,VF)*bF
        UB          = UB-np.dot(bB,UB)*bB
        VB          = VB-np.dot(bB,VB)*bB
        Det         = B0**2/(BF*BB)
        Norm        = np.dot(UF,UF)*np.dot(VB,VB)+np.dot(UB,UB)*np.dot(VF,VF)-2*np.dot(UF,VF)*np.dot(UB,VB)
        logQ        = np.log10(Norm)-np.log10(Det)
        logQ        = np.log10(2) if logQ<np.log10(2) else logQ
        logQ        = -logQ if np.dot(rtp2xyz(X0), b0)<0 else logQ
        length      = np.abs(lf)+np.abs(lb)
        res_list.append([icnt,logQ, length])

        with lock:
            progress[0] += 1  # 更新计数器
            if progress[0] % print_interval == 0 or progress[0] == total_tasks or progress[0]==1:
                ti = time.time()
                print(f"Progress: {progress[0]:6d}/{total_tasks} tasks completed.  Wall_time: {(ti-t0)/60:6.3f} min", flush=True)
                
    return res_list

def parallel_qsl(points, n_cores=10, print_interval=100):
    print('### =========== Parallel computing ============ ###')
    print(f'#       Available CPU cores: {multiprocessing.cpu_count():3d}                  #')
    print(f'#            Used CPU cores: {n_cores:3d}                  #')
    print('### =========================================== ###')
    t0 = time.time()
    n_points = len(points)
    chunk_size = n_points // n_cores

    qsl_results = []

    manager = Manager()
    progress= manager.list([0])  # 用于存储已完成的任务数，初始化为0
    lock    = manager.Lock()  # 使用Manager提供的Lock

    total_tasks = n_points
    print('Initialization OK')

    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        futures = []
        for i in range(n_cores):
            idx_i = i * chunk_size
            idx_e = (i + 1) * chunk_size if i < n_cores - 1 else n_points
            futures.append(executor.submit(calculate_qsl_range, idx_i, idx_e, points, progress, total_tasks, lock, print_interval, t0))

        print('Assigning task OK', flush=True)
        for future in as_completed(futures):
            qsl_results.extend(future.result())

    return qsl_results

# =======================================================================
t0     = time.time()
parser = argparse.ArgumentParser(description='Parallel computing QSL in Spherical Coordinates System')
parser.add_argument('-p', type=str  , help='Path to .npy file for target points array', required=False, default='./qsl_points.npy')
parser.add_argument('-i', type=str  , help='Path to .pkl file for instance'           , required=False, default='./sph_temp.pkl'  )
parser.add_argument('-f', type=int  , help='Which frame is?'                          , required=False, default=0                 )
parser.add_argument('-n', type=int  , help='Number of cores to use'                   , required=False, default=10                )
parser.add_argument('-e', type=int  , help='Print interval'                           , required=False, default=1                 )
parser.add_argument('--max_step', type=int  , help='maximum step for qsl intergral'   , required=False, default=1000000           )
parser.add_argument('--step_size',type=float, help='step size to integration'         , required=False, default=-1                )

args           = parser.parse_args()
frame          = args.f
instance       = spherical_data.load(args.i)
points         = instance.info[f'frame_{frame:04d}']['cal_qsl_setting'].get('points', None)
n_cores        = instance.info[f'frame_{frame:04d}']['cal_qsl_setting'].get('n_cores', 10)
print_interval = instance.info[f'frame_{frame:04d}']['cal_qsl_setting'].get('print_interval', 1000)
Ns             = instance.info[f'frame_{frame:04d}']['cal_qsl_setting'].get('max_step', 100000)
dl             = instance.info[f'frame_{frame:04d}']['cal_qsl_setting'].get('step_length', -1)

if points is None:
    raise ValueError('Target `points` to calculate QSL missed...')
RTP    = instance.get_rtp()
Brtp   = instance.get_Brtp()
Bmag   = np.linalg.norm(Brtp, axis=0)
Bhat   = Brtp/Bmag[None,:,:,:]
Jacobi = Generalized_Jacobian(Bhat, rtp=RTP)
dl     = dl if dl>0 else instance.drtp[0]*4

qsl_results = parallel_qsl(points, n_cores=n_cores, print_interval=print_interval)
qsl_res     = np.array(qsl_results)
sort_idx    =np.argsort(qsl_res[:,0])
logQ        = np.abs(qsl_res[sort_idx,1])
logQ        = np.where(logQ<np.log10(2), np.log10(2), logQ)
Len         = np.abs(qsl_res[sort_idx,2])
qsl_array   = np.stack([logQ, Len])

np.save('spherical_qsl.npy', qsl_array)