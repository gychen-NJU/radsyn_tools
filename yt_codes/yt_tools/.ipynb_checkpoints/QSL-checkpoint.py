import multiprocessing
from .need import *
from . import geometry
from .stream_line import *
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from scipy import ndimage
from matplotlib.colors import LogNorm, PowerNorm
# from .yt_spherical_3D import spherical_data

def initializeUV(lineP, **kwargs):
    bhat  = lineP[9:12]
    bmag  = lineP[12]
    eps_N = kwargs.get('eps_N', 1e-8)
    Vvec  = np.zeros(3)
    Uvec  = np.zeros(3)
    if (np.abs(bhat[2])<eps_N):
        if (np.abs(bhat[0])<eps_N):
            if (np.abs(bhat[1])<eps_N):
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
    lineP[3:6] = Uvec/np.linalg.norm(Uvec) if np.linalg.norm(Uvec)>eps_N else Uvec
    lineP[6:9] = Vvec/np.linalg.norm(Vvec) if np.linalg.norm(Vvec)>eps_N else Vvec
    return lineP

def tangent(self, lineP, bxyz, **kwargs):
    xyz  = lineP[:3]
    idx  = self.xyz2idx(xyz)
    bvec = self.sample_with_idx(idx, give_field=bxyz.transpose(1,2,3,0))
    bmag = np.linalg.norm(bvec)
    bhat = bvec/bmag if bmag>=1e-6 else np.array([0,0,0])
    lineP[9:12] = bhat
    lineP[12]   = bmag
    # if bmag<1e-6:
    #     print('bmag= ', bmag)
    return lineP

def rfunction(self, ipt, bxyz, Jacobi, dxyz=[1.,1.,1.], **kwargs):
    lineP     = np.zeros(13)
    lineP[:9] = ipt.copy()
    lineP     = self.tangent(lineP.copy() , bxyz)
    xyz       = lineP[:3]
    Uvec      = lineP[3:6]
    Vvec      = lineP[6:9]
    bhat      = lineP[9:12]
    bmag      = lineP[12]
    ret       = np.zeros(9)
    idx       = self.xyz2idx(xyz)
    ret[:3]   = bhat     
    jacobi    = self.sample_with_idx(idx, give_field=Jacobi.transpose(2,3,4,0,1))
    tmp       = jacobi[:,0]*Uvec[0]+jacobi[:,1]*Uvec[1]+jacobi[:,2]*Uvec[2]
    ret[3:6]  = tmp
    tmp       = jacobi[:,0]*Vvec[0]+jacobi[:,1]*Vvec[1]+jacobi[:,2]*Vvec[2] #np.tensordot(Vvec, jacobi, axes=1)
    ret[6:9]  = tmp
    return ret

def rk4_stepping(self, init, bxyz, Jacobi, dxyz=[1,1,1], sig=1, ds=0.001):
    # print(f'rk4_stepping called with sig={sig}, ds={ds}')
    ds    = ds*sig
    k1    = self.rfunction(init,         bxyz, Jacobi, dxyz)
    k2    = self.rfunction(init+k1*ds/2, bxyz, Jacobi, dxyz)
    k3    = self.rfunction(init+k2*ds/2, bxyz, Jacobi, dxyz)
    k4    = self.rfunction(init+k3*ds,   bxyz, Jacobi, dxyz)
    k     = (k1+2*k2+2*k3+k4)/6
    ret   = init+k*ds
    return ret

def LineIntegrate(self, lineP, bxyz, Jacobi, dxyz, **kwargs):
    coords    = kwargs.get('geometry', 'spherical')
    is_print  = kwargs.get('is_print', False)
    if coords=='spherical':
        r_min = kwargs.get('r_min', self.bbox[0][0])
    elif coords=='cartesian':
        bbox = self.bbox
    else:
        raise ValueError(f"{coords} is not 'spherical' or 'cartesian'")
    eps_B     = kwargs.get('eps_B', 1e-6)
    max_step  = kwargs.get('max_step', 10000)
    xyz       = lineP[:3]
    lineF = lineP.copy()
    lineB = lineP.copy()
    # forward integral
    sig = 1.
    nround = 0
    lambda_F = 0
    ds = kwargs.get('ds',np.min(dxyz))
    while(True):
        lineI = lineF.copy()
        if nround >= max_step:
            if is_print:
                print(f'over the max step: {max_step:06} during the forward integrating', flush=True)
            break
        nround+=1
        lineF[:9] = self.rk4_stepping(lineF[:9].copy(), bxyz, Jacobi, dxyz, sig, ds)
        lineF = self.tangent(lineF.copy(), bxyz)
        if coords=='spherical':
            is_stop = lineF[12]<= eps_B or np.linalg.norm(lineF[:3])<r_min
        else:
            is_stop = lineF[12]<= eps_B or not (bbox[0,0]<lineF[0]<bbox[0,1]) or not (bbox[1,0]<lineF[1]<bbox[1,1]) or not (bbox[2,0]<lineF[2]<bbox[2,1])
        if is_stop:
            lineF = lineI
            # print(f'!!!!! Stop !!!!! BF={lineI[12]:7.4f}')
            break
        lambda_F+=ds
        # print(f'lambda_F: {lambda_F:5.2f}, BF:{lineF[12]:5.2f}')

    # backward integral
    sig = -1.
    nround = 0
    lambda_B = 0
    while(True):
        lineI = lineB.copy()
        if nround >= max_step:
            print(f'over the max step: {max_step:06} during the forward integrating')
            break
        nround+=1
        lineB[:9] = self.rk4_stepping(lineB[:9], bxyz, Jacobi, dxyz=dxyz, sig=-1, ds=ds)
        lineB = self.tangent(lineB, bxyz)
        
        if coords=='spherical':
            is_stop = lineB[12]<= eps_B or np.linalg.norm(lineB[:3])<r_min
        else:
            is_stop = lineB[12]<= eps_B or not (bbox[0,0]<lineB[0]<bbox[0,1]) or not (bbox[1,0]<lineB[1]<bbox[1,1]) or not (bbox[2,0]<lineB[2]<bbox[2,1])
        if is_stop:
            lineB = lineI
            break
        lambda_B+=ds*sig
        # print(f'lambda_B:{lambda_B:5.2f}, BB:{BB:5.2f}')
    return lineF, lineB, lambda_F, lambda_B

def _SquanshingQ(self, xyz, frame=0, **kwargs):
    # print('SquanshingQ Starting OK', flush=True)
    # x_sample, y_sample, z_sample, bbox_cart, bbox = self.load_xyz()
    coords    = kwargs.get('coords', 'spherical')
    max_step  = kwargs.get('max_step', 10000)
    if coords=='spherical':
        bbox      = np.array(self.bbox)
        bbox_cart = self.bbox_cart
        r_min     = kwargs.get('r_min', bbox[0][0])
        dxyz      = kwargs.get('dxyz', (bbox_cart[:,1]-bbox_cart[:,0])/self.n_car)
        bxyz_file = os.path.join(self.bxyz_path, 'bxyz_'+str(frame).zfill(4)+'.npy')
        bxyz      = kwargs.get('bxyz',np.load(bxyz_file))
    elif coords=='cartesian':
        bbox      = np.array(self.bbox)
        bxyz      = kwargs.get('bxyz', None)
        dxyz      = kwargs.get('dxyz', None)
        r_min     = None
        if bxyz is None or dxyz is None:
            bxyz,dxyz = self.return_bxyz(frame=frame, sample_level=self.sample_level)
    ds        = kwargs.get('ds',np.min(dxyz))
    Jacobi    = kwargs.get('Jacobi', geometry.jacobi_matrix(bxyz, dxyz).detach().cpu().numpy())
    lineP     = np.zeros(13)
    lineP[:3] = xyz
    lineP     = self.tangent(lineP, bxyz)
    if np.max(lineP[9:12])<1e-8:
        return 0,0
    # print('SquanshingQ initialization OK', flush=True)
    lineP     = initializeUV(lineP)
    # print('SquanshingQ initializeUV OK', flush=True)
    lineF, lineB, lambda_F, lambda_B = self.LineIntegrate(lineP, bxyz, Jacobi, dxyz, r_min=r_min, geometry=coords, max_step=max_step, ds=ds)
    # print('Squanshing Integrate OK', flush=True)
    B0        = lineP[12]
    BF        = lineF[12]
    BB        = lineB[12]
    UF        = lineF[3:6]
    VF        = lineF[6:9]
    UB        = lineB[3:6]
    VB        = lineB[6:9]
    bF        = lineF[9:12]
    bB        = lineB[9:12]
    UF        = UF-np.dot(bF,UF)*bF
    VF        = VF-np.dot(bF,VF)*bF
    UB        = UB-np.dot(bB,UB)*bB
    VB        = VB-np.dot(bB,VB)*bB
    Det       = B0**2/(BF*BB)
    Norm      = np.dot(UF,UF)*np.dot(VB,VB)+np.dot(UB,UB)*np.dot(VF,VF)-2*np.dot(UF,VF)*np.dot(UB,VB)
    logQ      = np.log10(Norm)-np.log10(Det)
    logQ      = np.log10(2) if logQ<np.log10(2) else logQ
    logQ      = -logQ if np.dot(xyz, lineP[9:12])<0 else logQ
    length    = np.abs(lambda_F)+np.abs(lambda_B)
    # print(f'lambda_F: {lambda_F:5.2f}, lambda_B:{lambda_B:5.2f}, BF:{BF:5.2f}, BB:{BB:5.2f}')
    return logQ, length


# parallel computing #
def parallel_QSL(self, points, n_cores=10, print_interval=100, **kwargs):
    frame       = kwargs.get('frame', 0)
    bbox        = np.array(self.bbox)
    bbox_cart   = self.bbox_cart
    r_min       = kwargs.get('r_min', bbox[0][0])
    dxyz        = kwargs.get('dxyz', (bbox_cart[:,1]-bbox_cart[:,0])/self.n_car)
    bxyz_file   = os.path.join(self.bxyz_path, 'bxyz_'+str(frame).zfill(4)+'.npy')
    bxyz        = kwargs.get('bxyz',np.load(bxyz_file))
    # Jacobi      = kwargs.get('Jacobi', geometry.jacobi_matrix(bxyz, dxyz).detach().cpu().numpy())
    Jacobi      = kwargs.get('Jacobi', None)
    p_name      = kwargs.get('p_name', './qsl_points.npy')
    is_save     = kwargs.get('is_save', False)
    if Jacobi is None:
        bmag = np.linalg.norm(bxyz,axis=0)[np.newaxis,:,:,:]
        bhat = bxyz/bmag
        Jacobi = geometry.jacobi_matrix(bhat, dxyz).detach().cpu().numpy()

    np.save(p_name, points)
    if is_save or (self.output_name is None):
        self.save('instance.pkl')
    
    command     = f"python -m yt_tools.scripts.qsl_scripts -i {self.output_name} -p {p_name} -f {frame} -n {n_cores} -e {print_interval} -r {r_min}"
    ret         = os.system(command)
    qsl_results = np.load('./qsl_results.npy')
    os.remove('./qsl_results.npy')
    os.remove(p_name)
    qsl_idx     = np.argsort(qsl_results[:,0])
    qsl_dat     = qsl_results[qsl_idx][:,1:]
    return qsl_dat

def plot2D_QSL(self, qsl, nx=None,ny=None,**kwargs):
    qsl       = np.array(qsl)
    idx       = np.argsort(qsl[:,0])
    logQ      = qsl[idx,1]
    length    = qsl[idx,2]
    ax   = kwargs.get('ax'  , None)
    Qlim = kwargs.get('Qlim', None)
    Llim = kwargs.get('Llim', None)
    Slim = kwargs.get('Slim', None)
    tr   = kwargs.get('transpose', False)
    logQ = np.abs(np.nan_to_num(logQ))
    length[length < 1e-10] = 1e-10
    lens_val = length
    sobel_x = ndimage.sobel(lens_val.reshape((nx, ny)).astype(np.float32), axis=0)
    sobel_y = ndimage.sobel(lens_val.reshape((nx, ny)).astype(np.float32), axis=1)
    edges = np.hypot(sobel_x, sobel_y)
    edges[edges < 1e-10] = 1e-10
    logQ   = logQ.reshape((nx, ny))   if not tr else logQ.T
    length = length.reshape((nx, ny)) if not tr else length.T
    edges  = edges                    if not tr else edges.T
        

    if ax is None:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    
    # Plot for Log(Q)
    im1 = ax[0].imshow(logQ, origin='lower', cmap='jet',
                       norm=PowerNorm(1), interpolation='bicubic')
    if Qlim is not None:
        min = Qlim['min'] if 'min' in Qlim else np.log10(2)
        max = Qlim['max'] if 'max' in Qlim else None
        im1.set_clim(min, max)  
    ax[0].set_title('Log(Q)')
    ax[0].axis('off')
    cbar1 = plt.colorbar(im1, ax=ax[0], orientation='horizontal', fraction=0.046, pad=0.04)
    
    # Plot for Length
    im2 = ax[1].imshow(length, origin='lower', norm=LogNorm())
    if Llim is not None:
        min = Llim['min'] if 'min' in Llim else None
        max = Llim['max'] if 'max' in Llim else None
        im2.set_clim(min, max)   
    ax[1].set_title('Length')
    ax[1].axis('off')
    cbar2 = plt.colorbar(im2, ax=ax[1], orientation='horizontal', fraction=0.046, pad=0.04)
    
    # Plot for Log(S)
    im3 = ax[2].imshow(np.log10(edges), origin='lower', cmap='gray')
    if Slim is not None:
        min = Slim['min'] if 'min' in Slim else None
        max = Slim['max'] if 'max' in Slim else None
        im3.set_clim(min, max)  
    ax[2].set_title('Log(S)')
    ax[2].axis('off')
    plt.colorbar(im3, ax=ax[2], orientation='horizontal', fraction=0.046, pad=0.04)
    