import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import time

# ===================================================== #
#               Integrate the field line                #
# ===================================================== #
def sphere_sample(point, r=1, nsample=10):
    if nsample==1:
        return point
    nsample = nsample-1
    x0, y0, z0 = point
    rlist = np.random.uniform(0, r, nsample)
    theta = np.random.uniform(0,np.pi, nsample)
    phi   = np.random.uniform(0, 2*np.pi, nsample)
    xlist = rlist*np.sin(theta)*np.cos(phi)+x0
    ylist = rlist*np.sin(theta)*np.sin(phi)+y0
    zlist = rlist*np.cos(theta)+z0
    xlist = np.append(xlist, x0)
    ylist = np.append(ylist, y0)
    zlist = np.append(zlist, z0)
    ret = np.vstack([xlist,ylist,zlist]).T
    
    return ret

def trilinear_interpolation(field, idx):
    xi,yi,zi    = np.floor(idx).astype(int).T if np.array(idx).ndim>1 else np.floor(idx).astype(int)
    xd,yd,zd    = (idx-np.floor(idx)).T
    nx,ny,nz    = field.shape[:3]
    xf,yf,zf    = np.where(xi+1>nx-1,nx-1,xi+1), np.where(yi+1>ny-1,ny-1,yi+1), np.where(zi+1>nz-1,nz-1,zi+1)
    f           = field
    out         = (xi>nx-1) | (yi>ny-1) | (zi>nz-1) | (xi<0) | (yi<0) | (zi<0)
    out         = out | (xf<0) | (yf<0) | (zf<0) | (xf>nx-1) | (yf>ny-1) | (zf>nz-1)
    xi,yi,zi    = np.where(xi>nx-1,nx-1,xi), np.where(yi>ny-1,ny-1,yi), np.where(zi>nz-1,nz-1,zi)
    xi,yi,zi    = np.where(xi<0,0,xi), np.where(yi<0,0,yi), np.where(zi<0,0,zi)
    xf,yf,zf    = np.where(xi+1>nx-1,nx-1,xi+1), np.where(yi+1>ny-1,ny-1,yi+1), np.where(zi+1>nz-1,nz-1,zi+1)
    xf,yf,zf    = np.where(xf<0,0,xf),np.where(yf<0,0,yf),np.where(zf<0,0,zf)
    if isinstance(xd, np.ndarray):
        xd              = xd[(slice(None),)+(np.newaxis,)*(f[xi,yi,zi].ndim-1)]
        yd              = yd[(slice(None),)+(np.newaxis,)*(f[xi,yi,zi].ndim-1)]
        zd              = zd[(slice(None),)+(np.newaxis,)*(f[xi,yi,zi].ndim-1)]
    ret         = f[xi,yi,zi]*(1-xd)*(1-yd)*(1-zd)+f[xf,yi,zi]*xd*(1-yd)*(1-zd)+f[xi,yf,zi]*(1-xd)*yd*(1-zd)+\
                  f[xi,yi,zf]*(1-xd)*(1-yd)*zd+f[xf,yf,zi]*xd*yd*(1-zd)+f[xf,yi,zf]*xd*(1-yd)*zd+\
                  f[xi,yf,zf]*(1-xd)*yd*zd+f[xf,yf,zf]*xd*yd*zd
    ret[out]=np.full_like(ret[out], np.nan)
    return ret
# def trilinear_interpolation(field, idx):
#     xi,yi,zi    = np.floor(idx).astype(int).T if len(np.shape(idx))>1 else np.floor(idx).astype(int)
#     xd,yd,zd    = (idx-np.floor(idx)).T
#     nx,ny,nz    = field.shape[:3]
#     # if not 0<=xi<nx or not 0<=yi<ny or not 0<=zi<nz:
#     # if xi>=nx or yi>=ny or zi>=nz:
#     #     print(f"{idx} is out of dimension {field.shape[:3]}")
#     xf,yf,zf    = np.where(xi+1>nx-1,nx-1,xi+1), np.where(yi+1>ny-1,ny-1,yi+1), np.where(zi+1>nz-1,nz-1,zi+1)
#     f           = field
#     if isinstance(xd, np.ndarray):
#         xd              = xd[(slice(None),)+(np.newaxis,)*(f[xi,yi,zi].ndim-1)]
#         yd              = yd[(slice(None),)+(np.newaxis,)*(f[xi,yi,zi].ndim-1)]
#         zd              = zd[(slice(None),)+(np.newaxis,)*(f[xi,yi,zi].ndim-1)]
#     ret         = f[xi,yi,zi]*(1-xd)*(1-yd)*(1-zd)+f[xf,yi,zi]*xd*(1-yd)*(1-zd)+f[xi,yf,zi]*(1-xd)*yd*(1-zd)+\
#                   f[xi,yi,zf]*(1-xd)*(1-yd)*zd+f[xf,yf,zi]*xd*yd*(1-zd)+f[xf,yi,zf]*xd*(1-yd)*zd+\
#                   f[xi,yf,zf]*(1-xd)*yd*zd+f[xf,yf,zf]*xd*yd*zd
#     return ret

def field_line(field, star_point, ds=1.0, max_step=10000, dxyz=[1,1,1]):
    dxyz_min_idx=np.argmin(dxyz)
    Dx,Dy,Dz=np.array(dxyz)/dxyz[dxyz_min_idx]
    scale = np.array([1/Dx,1/Dy,1/Dz])
    first = True
    flag  = True
    fieldline1 = [star_point]
    fieldline2 = []
    x0, y0, z0 = star_point
    iters = 0
    lb = np.array([0,0,0])
    ub = np.array(field.shape[-3:])-1
    over_max_step = False
    
    while True:
        xi, yi, zi = x0, y0, z0
        bvec = trilinear_interpolation(field.transpose(1,2,3,0), np.array([[xi,yi,zi]]))[0]
        bb = np.sum(bvec**2)**0.5
        if bb<1e-10:
            break
        k1 = bvec/bb
        xi, yi, zi = np.array([x0,y0,z0])+k1*ds/2*scale
        bvec = trilinear_interpolation(field.transpose(1,2,3,0), np.array([[xi,yi,zi]]))[0]
        bb = np.sum(bvec**2)**0.5
        if bb<1e-10:
            break
        k2 = bvec/bb
        xi, yi, zi = np.array([x0,y0,z0])+k2*ds/2*scale
        bvec = trilinear_interpolation(field.transpose(1,2,3,0), np.array([[xi,yi,zi]]))[0]
        bb = np.sum(bvec**2)**0.5
        if bb<1e-10:
            break
        k3 = bvec/bb
        xi, yi, zi = np.array([x0,y0,z0])+k3*ds*scale
        bvec = trilinear_interpolation(field.transpose(1,2,3,0), np.array([[xi,yi,zi]]))[0]
        bb = np.sum(bvec**2)**0.5
        if bb<1e-10:
            # print('encounter the boundary')
            break
        k4 = bvec/bb
        x0, y0, z0 = np.array([x0,y0,z0])+(k1+2*k2+2*k3+k4)*ds/6.*scale
        iters += 1
        if x0<lb[0] or x0>ub[0] or y0<lb[1] or y0>ub[1] or z0<lb[2] or z0>ub[2] or iters>=max_step:
            # if iters >= max_step:
            #     print('over the max step: %05d during the forward integrating' % iters)
            break
        if np.any(np.isnan(np.array([x0, y0, z0]))):
            break
        fieldline1.append(np.array([x0, y0, z0]))
        
# back ward stream line
    x0, y0, z0 = star_point
    iters = 0
    while True:
        xi, yi, zi = x0,y0,z0
        bvec = trilinear_interpolation(field.transpose(1,2,3,0), np.array([[xi,yi,zi]]))[0]
        bb = np.sum(bvec**2)**0.5
        if bb<1e-10:
            # print('encounter the boundary')
            break
        k1 = -bvec/bb
        xi, yi, zi = np.array([x0,y0,z0])+k1*ds/2*scale
        bvec = trilinear_interpolation(field.transpose(1,2,3,0), np.array([[xi,yi,zi]]))[0]
        bb = np.sum(bvec**2)**0.5
        if bb<1e-10:
            # print('encounter the boundary')
            break
        k2 = -bvec/bb
        xi, yi, zi = np.array([x0,y0,z0])+k2*ds/2*scale
        bvec = trilinear_interpolation(field.transpose(1,2,3,0), np.array([[xi,yi,zi]]))[0]
        bb = np.sum(bvec**2)**0.5
        if bb<1e-10:
            # print('encounter the boundary')
            break
        k3 = -bvec/bb
        xi, yi, zi = np.array([x0,y0,z0])+k3*ds*scale
        bvec = trilinear_interpolation(field.transpose(1,2,3,0), np.array([[xi,yi,zi]]))[0]
        bb = np.sum(bvec**2)**0.5
        if bb<1e-10:
            # print('encounter the boundary')
            break
        k4 = -bvec/bb
        x0, y0, z0 = np.array([x0,y0,z0])+(k1+2*k2+2*k3+k4)*ds/6.*scale
        iters += 1
        if x0<lb[0] or x0>ub[0] or y0<lb[1] or y0>ub[1] or z0<lb[2] or z0>ub[2] or iters>=max_step:
            # if iters >= max_step:
            #     print('over the max step: %05d during the backward integrating' % iters)
            break
        if np.any(np.isnan(np.array([x0, y0, z0]))):
            break
        fieldline2.append(np.array([x0, y0, z0]))
        
    filedline = fieldline2[::-1]+fieldline1
    filedline = np.array(filedline)
    return filedline

def brtp2bxyz(brtp,xyz):
    '''
    Convert the [Br, Bp, Bt] to [Bx, By, Bz] under the Cartesian meshgrid.
    '''
    br,bt,bp = brtp
    X, Y, Z  = xyz
    R = np.sqrt(X**2+Y**2+Z**2)
    T = np.arcsin(Z/R)
    P = np.arctan(Y/X)
    P = np.where(X<0,P+np.pi,P)
    bx = br*np.cos(T)*np.cos(P)+bt*np.sin(T)*np.cos(P)-bp*np.sin(P)
    by = br*np.cos(T)*np.sin(P)+bt*np.sin(T)*np.sin(P)+bp*np.cos(P)
    bz = br*np.sin(T)-bt*np.cos(T)
    b_vec = np.stack([bx,by,bz])
    return b_vec

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

class Spherical_magline():
    """
    Spherical_magline

    This class is used to compute magnetic field lines in spherical coordinates (r, t, p). It performs 3D interpolation of the magnetic field data and integrates the field lines using the Runge-Kutta 45 (RK45) method.

    Methods:
    --------
    __init__(self, Brtp, rtp, **kwargs)
        Initializes the Spherical_magline class.

    Parameters:
    -----------
    Brtp : numpy.ndarray
        4D array of shape (N, M, L, 3) representing the magnetic field strength in spherical coordinates.
        - N, M, L: Number of grid points in the r, t, and p directions, respectively.
        - 3: Magnetic field components in the r, t, and p directions.
    rtp : numpy.ndarray
        4D array of shape (N, M, L, 3) representing the grid points in spherical coordinates.
        - N, M, L: Number of grid points in the r, t, and p directions, respectively.
        - 3: r (radius), t (polar angle), and p (azimuthal angle) coordinates.
    **kwargs : dict
        Additional parameters for the initialization.
        - step_length : float, optional
            Step length for the integration (default is the difference between the first and second grid points in the r direction).
        - max_steps : int, optional
            Maximum number of integration steps (default is 100000).
        - Rmin : float, optional
            Minimum radius for the integration (default is the minimum value in the rtp array).
        - Rmax : float, optional
            Maximum radius for the integration (default is the maximum value in the rtp array).

    Methods:
    --------
    rtp2idx(self, rtp)
        Converts spherical coordinates to grid indices.
    get_B(self, rtp)
        Interpolates the magnetic field at the given spherical coordinates.
    integrate_field_line(self, start_point)
        Integrates the magnetic field line starting from the given spherical coordinates using the RK45 method.

    Example:
    --------
    >>> import numpy as np
    >>> r = np.linspace(0, 1, 10)
    >>> t = np.linspace(0, np.pi, 10)
    >>> p = np.linspace(0, 2 * np.pi, 10)
    >>> R, T, P = np.meshgrid(r, t, p, indexing='ij')
    >>> Br = np.sin(R) * np.cos(T) * np.cos(P)
    >>> Bt = np.cos(R) * np.sin(T) * np.sin(P)
    >>> Bp = np.cos(R) * np.cos(T) * np.sin(P)
    >>> Brtp = np.array([Br, Bt, Bp]).transpose((1, 2, 3, 0))
    >>> rtp = np.array([R, T, P]).transpose((1, 2, 3, 0))
    >>> spherical_magline = Spherical_magline(Brtp, rtp)
    >>> start_point = [1.0, np.pi/2, np.pi]
    >>> magline = spherical_magline.magline_solver(start_point)
    """
    def __init__(self, Brtp, rtp, **kwargs):
        self.Brtp = Brtp
        self.rtp  = rtp
        self.Nrtp = rtp.shape
        self.drtp = rtp[1,1,1]-rtp[0,0,0]
        self.dl   = kwargs.get('step_length', rtp[1,0,0]-rtp[0,0,0])
        self.Ns   = kwargs.get('max_steps', 100000)
        self.Rl   = kwargs.get('Rmin', rtp[0,0,0,0])
        self.Ru   = kwargs.get('Rmax', rtp[-1,0,0,0])

    def rtp2idx(self, rtp):
        r ,t ,p  = rtp
        r0,t0,p0 = self.rtp[ 0, 0, 0]
        dr,dt,dp = self.drtp
        irtp     = np.stack([(r-r0)/dr,(t-t0)/dt,(p-p0)/dp], axis=-1)
        return irtp

    def get_Brtp(self, rtp, **kwargs):
        irtp = self.rtp2idx(rtp).T
        Brtp = trilinear_interpolation(self.Brtp, irtp)
        return Brtp

    def stepper(self, rtp, **kwargs):
        eps   = kwargs.get('eps', 1e-10)
        if np.any(np.isnan(rtp)):
            return np.full(3,np.nan)
        r,t,p = rtp
        Brtp  = self.get_Brtp(rtp, **kwargs)
        Bn    = np.linalg.norm(Brtp)
        if Bn<eps:
            return np.full(3,np.nan)
        Bhat  = Brtp/np.linalg.norm(Brtp, axis=0)
        ret   = Bhat/np.array([1,r,r*np.sin(t)])
        return ret

    def single_task(self, rtp, dl, Ns):
        Rl = self.Rl
        Ru = self.Ru
        rtp0 = rtp
        forward  = [rtp0]
        backward = []
        # forward integral
        for i in range(Ns):
            if rtp0[0]<Rl or rtp0[0]>Ru or np.any(np.isnan(rtp0)):
                rtp0 = forward[-2]
                kk   = self.stepper(rtp0)
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
            rtp0 = rk45(self.stepper, rtp0, dl, sig= 1)
            forward.append(rtp0)
        # backward integral
        rtp0 = rtp
        for i in range(Ns):
            if rtp0[0]<Rl or rtp0[0]>Ru or np.any(np.isnan(rtp0)):
                rtp0 = backward[-2]
                kk   = self.stepper(rtp0)
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
            rtp0 = rk45(self.stepper, rtp0, dl, sig=-1)
            backward.append(rtp0)
        magline = np.array(backward[::-1]+forward)
        return magline

    def parallel_task(self, idx_i, idx_f, rtps, dl, Ns):
        ret = []
        for i in range(idx_i,idx_f):
            rtp     = rtps[i]
            magline = self.single_task(rtp, dl, Ns)
            ret.append([i, magline])
        return ret

    def magline_solver(self, rtps, **kwargs):
        t0 = time.time()
        dl = kwargs.get('step_length', self.dl)
        Ns = kwargs.get('max_steps'  , self.Ns)
        Nc = kwargs.get('n_cores'    , -1     )
        Nc = min(Nc, multiprocessing.cpu_count())
        self.Rl = kwargs.get('rmin', self.Rl)
        self.Ru = kwargs.get('rmax', self.Ru)
        parallel = (Nc > 1)
        if parallel:
            n_tasks  = len(rtps)
            n_chunks = n_tasks//Nc
            res      = n_tasks % Nc
            chunks   = [n_chunks+1]*res+[n_chunks]*(n_tasks-res)
            idx_i    = 0
            idx_f    = 0
            print('Parallel Computing', flush=True)
            magline_res = []
            total_tasks = len(rtps)
            with ProcessPoolExecutor(max_workers=Nc) as executor:
                futures = []
                for i in range(Nc):
                    idx_f+=chunks[i]
                    futures.append(executor.submit(self.parallel_task, idx_i, idx_f, rtps, dl, Ns))
                    idx_i = idx_f
                for future in futures:
                    magline_res.extend(future.result())
            print(f'Time Used: {(time.time()-t0)/60:8.3f} min')
            return magline_res
        else:
            magline_res = []
            for rtp in rtps:
                imagline = self.single_task(rtp, dl, Ns)
                magline_res.append(imagline)
            print(f'Time Used: {(time.time()-t0)/60:8.3f} min')
            return magline_res

class Cartesian_magline():
    def __init__(self, Bxyz, xyz, **kwargs):
        self.Bxyz = Bxyz
        self.xyz  = xyz
        self.Nxyz = xyz.shape
        self.dxyz = xyz[1,1,1]-xyz[0,0,0]
        self.dl   = kwargs.get('step_length', self.dxyz.min())
        self.Ns   = kwargs.get('max_steps'  , 100000)
        self.lb   = kwargs.get('left_bottom', xyz[0,0,0])
        self.rt   = kwargs.get('right_top'  , xyz[-1,-1,-1])

    def xyz2idx(self, xyz):
        x ,y ,z  = xyz
        x0,y0,z0 = self.xyz[0,0,0]
        dx,dy,dz = self.dxyz
        ixyz     = np.stack([(x-x0)/dx,(y-y0)/dy,(z-z0)/dz], axis=-1)
        return ixyz

    def get_Bxyz(self, xyz, **kwargs):
        ixyz = self.xyz2idx(xyz).T
        Bxyz = trilinear_interpolation(self.Brtp, ixyz)
        return Bxyz

    def stepper(self, xyz, **kwargs):
        eps = kwargs.get('eps', 1e-10)
        if np.any(np.isnan(xyz)):
            return np.full(3, np.nan)
        x,y,z = xyz
        Bxyz  = self.get_Bxyz(xyz, **kwargs)
        Bn    = np.linalg.norm(Bxyz)
        if Bn<eps:
            return np.full(3, np.nan)
        Bhat  = Bxyz/Bn
        ret   = Bhat
        return ret

    def single_task(self, xyz, dl, Ns):
        lb       = self.lb
        rt       = self.rt
        xyz0     = xyz
        forward  = [xyz0]
        backward = []
        # forward integral
        for i in range(Ns):
            if np.any(xyz0<lb) or np.any(xyz0>rt) or np.any(np.isnan(xyz0)):
                break
            xyz0 = rk45(self.stepper, xyz0, dl, sig= 1)
            forward.append(xyz0)
        # backward integral
        xyz0 = xyz
        for i in range(Ns):
            if np.any(xyz0<lb) or np.any(xyz0>rt) or np.any(np.isnan(xyz0)):
                break
            xyz0 = rk45(self.stepper, xyz0, dl, sig=-1)
            backward.append(xyz0)
        magline = np.array(backward[::-1]+forward)
        return magline

    def parallel_task(self, idx_i, idx_f, xyzs, dl, Ns):
        ret = []
        for i in range(idx_i, idx_f):
            xyz     = xyzs[i]
            magline = self.single_task(xyz, dl, Ns)
            ret.append([i, magline])
        return ret

    def magline_solver(self, xyzs, **kwargs):
        t0 = time.time()
        dl = kwargs.get('step_length', self.dl)
        Ns = kwrags.get('max_steps'  , self.Ns)
        Nc = kwargs.get('n_cores'    , -1)
        Nc = min(Nc, multiprocessing.cpu_count())
        self.lb = kwargs.get('left_bottom', self.lb)
        self.rt = kwargs.get('right_top'  , self.rt)
        parallel = (Nc>1)
        if np.array(xyzs).ndim==1:
            xyzs = [xyzs]
        if parallel:
            n_tasks  = len(xyzs)
            n_chunks = n_tasks//Nc
            res      = n_tasks % Nc
            chunks   = [n_chunsk+1]*res+[n_chunks]*(n_tasks-res)
            idx_i    = 0
            idx_f    = 0
            print('Parallel Computing', flush=True)
            magline_res = []
            with ProcesPoolExecutor(max_workers=Nc) as executor:
                futures = []
                for i in range(Nc):
                    idx_f+=chunks[i]
                    futures.append(executor.submit(self.parallel_task, idx_i, idx_f, rtps, dl, Ns))
                    idx_i = idx_f
                for future in futures:
                    magline_res.extend(future.result())
            print(f'Time Used: {(time.time()-t0):8.3f} sec...')
            return magline_res
        else:
            magline_res = []
            for xyz in xyzs:
                imagline = self.single_task(xyz, dl, Ns)
                magline_res.append(imagline)
            print(f'Time Used: {(time.time()-t0):8.3f} sec...')
            return magline_res
                           