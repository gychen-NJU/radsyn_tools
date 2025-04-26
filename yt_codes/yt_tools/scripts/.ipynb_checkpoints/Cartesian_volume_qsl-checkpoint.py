# parallel_qsl.py
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from ..need import *
from ..yt_spherical_3D import spherical_data
from .. import geometry
from ..stream_line import trilinear_interpolation
from ..yt_cartesian_3D import cartesian_data

# ==============================================

def initializeUV(xyz, **kwargs):
    idx  = instance.xyz2idx(xyz)
    bxyz = instance.sample_with_idx(idx, give_field=Bxyz.transpose(1,2,3,0))
    bmag = np.linalg.norm(bxyz)
    bhat = bxyz/bmag
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
    bl     = kwargs.get('bottom_left', instance.domain_left_edge.value )
    ur     = kwargs.get('upper_right', instance.domain_right_edge.value)
    X0     = y[ :3]
    if np.any(np.isnan(y)) or np.any((X0<bl) | (X0>ur)):
        return np.array([np.nan]).repeat(len(y))
    U0     = y[3:6]
    V0     = y[6: ]
    idx    = instance.xyz2idx(X0)
    bhat   = instance.sample_with_idx(idx,give_field=Bhat.transpose(1,2,3,0))
    jacobi = instance.sample_with_idx(idx,give_field=Jacobi.transpose(2,3,4,0,1))
    dX     = bhat
    dU     = U0[0]*jacobi[:,0]+U0[1]*jacobi[:,1]+U0[2]*jacobi[:,2]
    dV     = V0[0]*jacobi[:,0]+V0[1]*jacobi[:,1]+V0[2]*jacobi[:,2]
    ret    = np.hstack([dX,dU,dV])
    return ret

def Line_Integration(xyz, **kwargs):
    dl    = kwargs.get('dl'         , dxyz.min()                      )
    Ns    = kwargs.get('max_steps'  , 1000000                         )
    bl    = kwargs.get('bottom_left', instance.domain_left_edge.value )
    ur    = kwargs.get('upper_right', instance.domain_right_edge.value)
    RL    = kwargs.get('return_line', False                           )
    X0    = xyz
    U0,V0 = initializeUV(X0, **kwargs)
    # forward integration
    yy    = np.hstack([X0,U0,V0])
    sig   = 1
    iter  = 0
    lf    = 0
    stop  = False
    yf_list = [yy]
    yb_list = []
    while iter<Ns and not stop:
        iter+=1
        yf = rk45_stepper(rfun, lf, yy, dl=dl, sig=sig)
        if np.any(np.isnan(yf)) or np.min(yf[:3]-bl)<0 or np.min(ur-yf[:3])<0:
            xx  = yy[:3]
            idx = instance.xyz2idx(xx)
            bb  = instance.sample_with_idx(idx,give_field=Bhat.transpose(1,2,3,0))
            dl1 = ((bl-xx)/(sig*bb))
            dl1 = 1e4 if len(dl1[dl1>=0])==0 else np.min(dl1[dl1>=0])
            dl2 = ((ur-xx)/(sig*bb))
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
            yf_list.append(yf)
        else:
            yy = yf
            lf = lf+sig*dl
            yf_list.append(yy)
    if iter==Ns:
        print(f'Forward Integration over {Ns} iterations')
    # backward integration
    yy    = np.hstack([X0,U0,V0])
    sig   = -1
    iter  = 0
    lb    = 0
    stop  = False
    while iter<Ns and not stop:
        iter+=1
        yb = rk45_stepper(rfun, lb, yy, dl=dl, sig=sig)
        if np.any(np.isnan(yb)) or np.min(yb[:3]-bl)<0 or np.min(ur-yb[:3])<0:
            xx  = yy[:3]
            idx = instance.xyz2idx(xx)
            bb  = instance.sample_with_idx(idx,give_field=Bhat.transpose(1,2,3,0))
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
            yb_list.append(yb)
        else:
            yy = yb
            lb = lb+sig*dl
            yb_list.append(yy)
    if iter==Ns:
        print(f'Backward Integration over {Ns} iterations')
    if not RL:
        return yf, yb, lf, lb
    else:
        return yf,yb,lf,lb,np.array(yb_list[::-1]+yf_list)

def point_task_qsl(point):
    X0               = point
    yf,yb,lf,lb,line = Line_Integration(X0, dl=dl, max_steps=Ns, return_line=True)
    Xf,UF,VF         = yf[:3],yf[3:6],yf[6:]
    Xb,UB,VB         = yb[:3],yb[3:6],yb[6:]
    idx_0            = instance.xyz2idx(X0)
    idx_f            = instance.xyz2idx(Xf)
    idx_b            = instance.xyz2idx(Xb)
    BF               = trilinear_interpolation(Bxyz.transpose(1,2,3,0), idx_f)
    bF               = BF/np.linalg.norm(BF)
    BB               = trilinear_interpolation(Bxyz.transpose(1,2,3,0), idx_b)
    bB               = BB/np.linalg.norm(BB)
    B0               = trilinear_interpolation(Bxyz.transpose(1,2,3,0), idx_0)
    b0               = B0/np.linalg.norm(B0)
    B0,BF,BB         = np.linalg.norm(B0),np.linalg.norm(BF),np.linalg.norm(BB)
    UF               = UF-np.dot(bF,UF)*bF
    VF               = VF-np.dot(bF,VF)*bF
    UB               = UB-np.dot(bB,UB)*bB
    VB               = VB-np.dot(bB,VB)*bB
    Det              = B0**2/(BF*BB)
    Norm             = np.dot(UF,UF)*np.dot(VB,VB)+np.dot(UB,UB)*np.dot(VF,VF)-2*np.dot(UF,VF)*np.dot(UB,VB)
    logQ             = np.log10(Norm)-np.log10(Det)
    logQ             = np.log10(2) if logQ<np.log10(2) else logQ
    # logQ             = -logQ if np.dot(X0, b0)<0 else logQ
    length           = np.abs(lf)+np.abs(lb)
    # if logQ == 0:
    #     print(f"logQ: {logQ};  X0: {X0};  idx: {idx0}")
    return line,logQ,length

def update_progress(progress,lock, nprint, ntasks, t0):
    with lock:
        progress[0]+=1
        if progress[0] % nprint == 0 or progress[0] == ntasks or progress[0]==1:
            ti = time.time()
            print(f"  # Progress: {progress[0]:6d}/{ntasks},  Wall_time: {(ti-t0)/60:6.3f} min", flush=True)

def process_qsl_task(idx_i, idx_f, surface_points, progress, lock, t0, n_cores=10, nprint=1000):
    ntasks = len(surface_points)
    res_list = []
    for icnt in range(idx_i, idx_f):
        point = surface_points[icnt]
        ix0,iy0,iz0 = np.rint(instance.xyz2idx(point)).astype(int)
        # point_value = [f"{value:4.2f}" for value in point]
        # print(f"idx: {ix0:3d},{iy0:3d},{iz0:3d};  points: {', '.join(point_value)}")
        if np.abs(qsl_cube[ix0,iy0,iz0])>np.log10(2):
            # update_progress(progress, lock, nprint, ntasks, t0)
            with lock:
                progress[0]+=1
                if progress[0] % nprint == 0 or progress[0] == ntasks or progress[0]==1:
                    ti = time.time()
                    print(f"  # Progress: {progress[0]:6d}/{ntasks},  Wall_time: {(ti-t0)/60:6.3f} min", flush=True)
            continue
        else:
            line,logQ,length = point_task_qsl(point)
            xyz = line[:,:3]
            idx = instance.xyz2idx(xyz)
            ix,iy,iz = np.unique(np.rint(idx.T).astype(int), axis=1)
            res_list.append([ix,iy,iz,logQ,length,np.array([ix0,iy0,iz0])])
            # point_value = [f"{value:4.2f}" for value in point]
            # print(f"idx: {ix0:3d},{iy0:3d},{iz0:3d};  points: {', '.join(point_value)};  logQ: {logQ:+.4e}")
            # update_progress(progress, lock, nprint, ntasks, t0)
            with lock:
                progress[0]+=1
                if progress[0] % nprint == 0 or progress[0] == ntasks or progress[0]==1:
                    ti = time.time()
                    print(f"  # Progress: {progress[0]:6d}/{ntasks},  Wall_time: {(ti-t0)/60:6.3f} min", flush=True)
    return res_list

def surface_task_qsl(iz, t0, n_cores=10, nprint=1000):
    print(f' ## =================== Layer:{(iz+1):04d}/{nz:04d} =================== ## ')
    # print(f' ## ======================================================= ## ')
    points = XYZ[:,:,:,iz].reshape(3,-1).T
    checkQ = np.abs(qsl_cube[:,:,iz])
    select = np.where(checkQ.flatten()<=np.log10(2))
    points = points[select]
    return_line = True
    ntasks = len(points)
    nchunk = ntasks // n_cores
    res    = ntasks % n_cores
    chunks = [nchunk+1]*res+[nchunk]*(n_cores-res)
    res_list = []
    manager = Manager()
    progress = manager.list([0])  # 用于存储已完成的任务数，初始化为0
    lock = manager.Lock()  # 使用Manager提供的Lock
    # print('  Initialization OK')

    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        futures = []
        idx_i = 0
        idx_f = 0
        for i in range(n_cores):
            idx_f += chunks[i]
            futures.append(
                executor.submit(
                    process_qsl_task,
                    idx_i, idx_f, points, progress, lock, t0, n_cores, nprint
                )
            )
            idx_i = idx_f

        # print('  Assigning task OK', flush=True)
        for future in as_completed(futures):
            res_list.extend(future.result())
            
    # for ix,iy,iz,lgQ,Len,idx0 in res_list:
    #     qsl_cube[ix,iy,iz] = np.where(np.abs(qsl_cube[ix,iy,iz])<np.abs(lgQ),lgQ,qsl_cube[ix,iy,iz])
    #     len_cube[ix,iy,iz] = np.where(np.abs(len_cube[ix,iy,iz])<np.abs(Len),Len,len_cube[ix,iy,iz])
    #     qsl_cube[tuple(idx0)] = lgQ
    #     len_cube[tuple(idx0)] = Len

    return res_list

# ==============================================

t0     = time.time()
parser = argparse.ArgumentParser(description='Parallel computing QSL')
parser.add_argument('-i', type=str  , help='Path to .pkl file for instance'           , required=False, default='./instance.pkl'  )
parser.add_argument('-f', type=int  , help='Which frame is?'                          , required=False, default=0                 )
parser.add_argument('-n', type=int  , help='Number of cores to use'                   , required=False, default=10                )
parser.add_argument('-e', type=int  , help='Print interval'                           , required=False, default=1                 )
parser.add_argument('--max_step', type=int  , help='maximum step for qsl intergral'   , required=False, default=1000000           )
parser.add_argument('--step_size',type=float, help='step size to integration'         , required=False, default=-1                )

args           = parser.parse_args()
frame          = args.f
n_cores        = args.n
print_interval = args.e
instance       = cartesian_data.load(args.i)
Ns             = args.max_step
dl             = args.step_size

nxyz = instance.dimensions*instance.sample_level
dxyz = (instance.bbox[:,1]-instance.bbox[:,0])/(nxyz-1)
Bxyz = instance.return_bxyz(sample_level=instance.sample_level, frame=frame)
Bhat = Bxyz/(np.linalg.norm(Bxyz,axis=0)[np.newaxis,:,:,:])
Jacobi = geometry.jacobi_matrix(Bhat,dxyz=dxyz).detach().cpu().numpy()
dl   = dl if dl>0 else dxyz.min()
nx,ny,nz = nxyz
XYZ      = instance._xyz()
X,Y,Z    = XYZ

qsl_cube = np.zeros(nxyz)
len_cube = np.zeros(nxyz)
nprint   = print_interval
# print(f'### ======================================================= ###')
# print(f'###           Parallel Computing Domain QSL                 ###')
# print(f'###           Available CPU cores: {multiprocessing.cpu_count():3d}                      ###')
# print(f'###                Used CPU cores: {n_cores:3d}                      ###')
# print(f'### ======================================================= ###')
# t0 = time.time()
# for iz in range(nz):
#     surface_task_qsl(iz, t0, n_cores, nprint)
# np.savez('Cartesian_volume_qsl.npz', **dict(logQ=qsl_cube, length=len_cube))

# print('!!! Code Ending !!!')
print(f'### ======================================================= ###')
print(f'###           Parallel Computing Domain QSL                 ###')
print(f'###           Available CPU cores: {multiprocessing.cpu_count():3d}                      ###')
print(f'###                Used CPU cores: {n_cores:3d}                      ###')
print(f'### ======================================================= ###')
t0 = time.time()
for iz in range(nz):
    res_list = surface_task_qsl(iz, t0, n_cores, nprint)
    for ix,iy,iz,lgQ,Len,idx0 in res_list:
        qsl_cube[ix,iy,iz] = np.where(np.abs(qsl_cube[ix,iy,iz])<np.abs(lgQ),lgQ,qsl_cube[ix,iy,iz])
        len_cube[ix,iy,iz] = np.where(np.abs(len_cube[ix,iy,iz])<np.abs(Len),Len,len_cube[ix,iy,iz])
        qsl_cube[tuple(idx0)] = lgQ
        len_cube[tuple(idx0)] = Len
    np.savez('Cartesian_volume_qsl.npz', **dict(logQ=qsl_cube, length=len_cube))
        
np.savez('Cartesian_volume_qsl.npz', **dict(logQ=qsl_cube, length=len_cube))
print('!!! Code Ending !!!')