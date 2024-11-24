from .need import *
import torch


def trilinear_interpolation(field, idx):
    if np.any(np.isnan(np.array(idx))):
        if np.array(idx).ndim==1:
            ret = np.nan if field.ndim==3 else np.full_like(field[0,0,0], np.nan)
        else:
            ret = np.full(len(idx), np.nan) if field.ndim==3 else np.full_like(field[0,0,0],np.nan).repeat(len(idx), axis=0)
        return ret
    x0,y0,z0    = np.array(idx).T             if np.array(idx).ndim>1 else np.array(idx)
    xi,yi,zi    = np.floor(idx).astype(int).T if np.array(idx).ndim>1 else np.floor(idx).astype(int)
    xd,yd,zd    = (idx-np.floor(idx)).T
    nx,ny,nz    = field.shape[:3]
    xf,yf,zf    = np.ceil(idx).astype(int).T if np.array(idx).ndim>1 else np.ceil(idx).astype(int)
    # xf,yf,zf    = np.where(xi+1>nx-1,nx-1,xi+1), np.where(yi+1>ny-1,ny-1,yi+1), np.where(zi+1>nz-1,nz-1,zi+1)
    f           = field
    # out         = (xi>nx-1) | (yi>ny-1) | (zi>nz-1) | (xi<0) | (yi<0) | (zi<0)
    # out         = out | (xf<0) | (yf<0) | (zf<0) | (xf>nx-1) | (yf>ny-1) | (zf>nz-1)
    out         = (x0>nx-1) | (y0>ny-1) | (z0>nz-1) | (x0<0) | (y0<0) | (z0<0)
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

def Generalized_Jacobian(Vfield, Lame=None, dq=None, **kwargs):
    device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device)
    geometry = kwargs.pop('geometry', 'spherical')
    if isinstance(Vfield, np.ndarray):
        is_array = True
        Vfield   = torch.from_numpy(Vfield).to(device)
        Lame,dq  = Lame_coefficient(geometry,**kwargs)
        Lame     = torch.from_numpy(Lame  ).to(device)
        # print(f"Lame's shape: ", Lame.shape)
    else:
        is_array = False
    d1,d2,d3 = dq if dq is not None else [1,1,1]
    H1,H2,H3 = Lame
    H        = H1*H2*H3
    V1,V2,V3 = Vfield
    F1,F2,F3 = Lame*Vfield
    G1       = V1*H2*H3
    G2       = H1*V2*H3
    G3       = H1*H2*V3
    GF1      = torch.stack([G1,F2,F3])
    GF2      = torch.stack([F1,G2,F3])
    GF3      = torch.stack([F1,F2,G3])
    pd1      = torch.cat([(GF1[:,1:2,:,:]*4-GF1[:,0:1,:,:]*3  -GF1[:,2:3,:,:])/2,
                          (GF1[:,2: ,:,:]  -GF1[:,:-2,:,:])/2,
                          (GF1[:,-1:,:,:]*3-GF1[:,-2:-1,:,:]*4-GF1[:,-3:-2,:,:])/2
                         ], dim=1)/d1
    pd2      = torch.cat([(GF2[:,:,1:2,:]*4-GF2[:,:,0:1  ,:]*3-GF2[:,:, 2: 3,:])/2,
                          (GF2[:,:,2: ,:]  -GF2[:,:,:-2  ,:])/2,
                          (GF2[:,:,-1:,:]*3-GF2[:,:,-2:-1,:]*4-GF2[:,:,-3:-2,:])/2
                         ], dim=2)/d2
    pd3      = torch.cat([(GF3[:,:,:,1:2]*4-GF3[:,:,:,0:1  ]*3-GF3[:,:,:, 2: 3])/2,
                          (GF3[:,:,:,2: ]  -GF3[:,:,:,:-2  ])/2,
                          (GF3[:,:,:,-1:]*3-GF3[:,:,:,-2:-1]*4-GF3[:,:,:,-3:-2])/2
                         ], dim=3)/d3
    GA       = torch.stack([pd1,pd2,pd3], dim=0)/H
    if is_array:
        return GA.detach().cpu().numpy()
    else:
        return GA

def Lame_coefficient(geometry='spherical', **kwargs):
    if geometry=='spherical':
        rtp = kwargs.get('rtp', None)
        if rtp is None:
            raise ValueError('Please give a spherical coordinates `rtp`...')
        r,t,p = rtp
        dr    = r[1,0,0]-r[0,0,0]
        dt    = t[0,1,0]-t[0,0,0]
        dp    = p[0,0,1]-p[0,0,0]
        dq    = [dr,dt,dp]
        H1    = np.ones_like(r)
        H2    = r
        H3    = r*np.sin(t)
        Lame  = np.stack([H1,H2,H3])
    elif geometry=='cylindrical':
        rpz = kwargs.get('rpz', None)
        if rpz is None:
            raise ValueError('Please give a cylindrical coordinates `rpz`...')
        r,p,z = rpz
        dr    = r[1,0,0]-r[0,0,0]
        dp    = p[0,1,0]-p[0,0,0]
        dz    = z[0,0,1]-z[0,0,0]
        dq    = [dr,dp,dz]
        H1    = np.ones_like(r)
        H2    = np.ones_like(r)
        H3    = r
        Lame  = np.stack([H1,H2,H3])
    elif geometry=='cartesian':
        xyz  = kwargs.get('xyz' , None)
        dxyz = kwargs.get('dxyz', None)
        if dxyz is None:
            if xyz is None:
                raise ValueError('Please give a cartesian coordinates `xyz`...')
            else:
                x,y,z = xyz
                dx = x[1,0,0]-x[0,0,0]
                dy = y[0,1,0]-y[0,0,0]
                dz = z[0,0,1]-z[0,0,0]
                dq = [dx,dy,dz]
                Lame = np.zeros(xyz.shape)
    else:
        raise TypeError("`geometry` only support to be 'spherical','cylindrical' and 'cartesian'...")
    return Lame, dq

def div(V, geometry='spherical', **kwargs):
    Lame, dq = Lame_coefficient(geometry, **kwargs)
    GA       = Generalized_Jacobian(V, Lame, dq=dq, **kwargs)
    ret      = GA[0,0]+GA[1,1]+GA[2,2]
    return ret

def rot(V, geometry='spherical', **kwargs):
    Lame, dq = Lame_coefficient(geometry, **kwargs)
    GA       = Generalized_Jacobian(V, Lame, dq=dq, **kwargs)
    ret      = np.stack([(GA[1,2]-GA[2,1])*Lame[0],
                         (GA[2,0]-GA[0,2])*Lame[1],
                         (GA[0,1]-GA[1,0])*Lame[2]
                        ], axis=0)
    return ret

def grad(V, geometry='spherical', **kwargs):
    Lame, dq = Lame_coefficient(geometry, **kwargs)
    h1,h2,h3 = Lame
    g1,g2,g3 = np.gradient(V,axis=(0,1,2))
    ret      = np.stack([g1/(h1*dq[0]),g2/(h2*dq[1]),g3/(h3*dq[2])],axis=0)
    return ret