import numpy as np

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
    phi = np.random.uniform(0, 2*np.pi, nsample)
    xlist = rlist*np.sin(theta)*np.cos(phi)+x0
    ylist = rlist*np.sin(theta)*np.sin(phi)+y0
    zlist = rlist*np.cos(theta)+z0
    xlist = np.append(xlist, x0)
    ylist = np.append(ylist, y0)
    zlist = np.append(zlist, z0)
    ret = np.vstack([xlist,ylist,zlist]).T
    return ret

def trilinear_interpolation(field, idx):
    xi,yi,zi    = np.floor(idx).astype(int).T if len(np.shape(idx))>1 else np.floor(idx).astype(int)
    xd,yd,zd    = (idx-np.floor(idx)).T
    nx,ny,nz    = field.shape[:3]
    xf,yf,zf    = np.where(xi+1>nx-1,nx-1,xi+1), np.where(yi+1>ny-1,ny-1,yi+1), np.where(zi+1>nz-1,nz-1,zi+1)
    f           = field
    xd              = xd[(slice(None),)+(np.newaxis,)*(f[xi,yi,zi].ndim-1)]
    yd              = yd[(slice(None),)+(np.newaxis,)*(f[xi,yi,zi].ndim-1)]
    zd              = zd[(slice(None),)+(np.newaxis,)*(f[xi,yi,zi].ndim-1)]
    ret         = f[xi,yi,zi]*(1-xd)*(1-yd)*(1-zd)+f[xf,yi,zi]*xd*(1-yd)*(1-zd)+f[xi,yf,zi]*(1-xd)*yd*(1-zd)+\
                  f[xi,yi,zf]*(1-xd)*(1-yd)*zd+f[xf,yf,zi]*xd*yd*(1-zd)+f[xf,yi,zf]*xd*(1-yd)*zd+\
                  f[xi,yf,zf]*(1-xd)*yd*zd+f[xf,yf,zf]*xd*yd*zd
    return ret

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
