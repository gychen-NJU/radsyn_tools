import numpy as np

# ===================================================== #
#               Integrate the field line                #
# ===================================================== #
def sphere_sample(point, r=1, nsample=10):
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
    
    while True:
        xi, yi, zi = np.around(np.array([x0, y0, z0])).astype(int)
        bvec = np.array(field[:,xi,yi,zi])
        bb = np.sum(bvec**2)**0.5
        if bb<1e-10:
            # print('encounter the boundary')
            break
        k1 = bvec/bb
        xi, yi, zi = np.around(np.array(np.array([x0,y0,z0])+k1*ds/2*scale)).astype(int)
        bvec = np.array(field[:,xi,yi,zi])
        bb = np.sum(bvec**2)**0.5
        if bb<1e-10:
            # print('encounter the boundary')
            break
        k2 = bvec/bb
        xi, yi, zi = np.around((np.array(np.array([x0,y0,z0])+k2*ds/2*scale))).astype(int)
        bvec = np.array(field[:,xi,yi,zi])
        bb = np.sum(bvec**2)**0.5
        if bb<1e-10:
            # print('encounter the boundary')
            break
        k3 = bvec/bb
        xi, yi, zi = np.around(np.array(np.array([x0,y0,z0])+k3*ds*scale)).astype(int)
        bvec = np.array(field[:,xi,yi,zi])
        bb = np.sum(bvec**2)**0.5
        if bb<1e-10:
            # print('encounter the boundary')
            break
        k4 = bvec/bb
        x0, y0, z0 = np.array([x0,y0,z0])+(k1+2*k2+2*k3+k4)*ds/6.*scale
        iters += 1
        if x0<lb[0] or x0>ub[0] or y0<lb[1] or y0>ub[1] or z0<lb[2] or z0>ub[2] or iters>=max_step:
            if iters >= max_step:
                print('over the max step: %05d during the forward integrating' % iters)
            break
        fieldline1.append(np.array([x0, y0, z0]))
        
# back ward stream line
    x0, y0, z0 = star_point
    iters = 0
    while True:
        xi, yi, zi = np.around(np.array([x0, y0, z0])).astype(int)
        bvec = np.array(field[:,xi,yi,zi])
        bb = np.sum(bvec**2)**0.5
        if bb<1e-10:
            # print('encounter the boundary')
            break
        k1 = -bvec/bb
        xi, yi, zi = np.around(np.array(np.array([x0,y0,z0])+k1*ds/2)).astype(int)
        bvec = np.array(field[:,xi,yi,zi])
        bb = np.sum(bvec**2)**0.5
        if bb<1e-10:
            # print('encounter the boundary')
            break
        k2 = -bvec/bb
        xi, yi, zi = np.around(np.array(np.array([x0,y0,z0])+k2*ds/2)).astype(int)
        bvec = np.array(field[:,xi,yi,zi])
        bb = np.sum(bvec**2)**0.5
        if bb<1e-10:
            # print('encounter the boundary')
            break
        k3 = -bvec/bb
        xi, yi, zi = np.around(np.array(np.array([x0,y0,z0])+k3*ds)).astype(int)
        bvec = np.array(field[:,xi,yi,zi])
        bb = np.sum(bvec**2)**0.5
        if bb<1e-10:
            # print('encounter the boundary')
            break
        k4 = -bvec/bb
        x0, y0, z0 = np.array([x0,y0,z0])+(k1+2*k2+2*k3+k4)*ds/6.
        iters += 1
        if x0<lb[0] or x0>ub[0] or y0<lb[1] or y0>ub[1] or z0<lb[2] or z0>ub[2] or iters>=max_step:
            if iters >= max_step:
                print('over the max step: %05d during the backward integrating' % iters)
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