from .need import *
import torch

# ===================================================== #
#                    vector analysis                    #
# ===================================================== #
# differential oprator
def jacobi_matrix(b_cube ,dxyz=[1.,1.,1.]):
    dx,dy,dz = dxyz/np.min(dxyz)
    is_array = False
    if isinstance(b_cube, np.ndarray):
        b_cube = torch.from_numpy(b_cube)
        is_array=True
    elif not isinstance(b_cube, torch.Tensor):
        raise ValueError("Input must be a NumPy array or a PyTorch tensor.")
    
    b_dx = torch.cat([(b_cube[:,1:2,:,:]*4-b_cube[:,0:1,:,:]*3-b_cube[:,2:3,:,:])/2,
                      (b_cube[:,2:,:,:]-b_cube[:,:-2,:,:])/2,
                      (b_cube[:,-1:,:,:]*3-b_cube[:,-2:-1,:,:]*4+b_cube[:,-3:-2,:,:])/2], dim=1)/dx
    
    b_dy = torch.cat([(b_cube[:,:,1:2,:]*4-b_cube[:,:,0:1,:]*3-b_cube[:,:,2:3,:])/2,
                      ((b_cube[:,:,2:,:]-b_cube[:,:,:-2,:])/2),
                      (b_cube[:,:,-1:,:]*3-b_cube[:,:,-2:-1,:]*4+b_cube[:,:,-3:-2,:])/2], dim=2)/dy
    
    b_dz = torch.cat([(b_cube[:,:,:,1:2]*4-b_cube[:,:,:,0:1]*3-b_cube[:,:,:,2:3])/2,
                      ((b_cube[:,:,:,2:]-b_cube[:,:,:,:-2])/2),
                      (b_cube[:,:,:,-1:]*3-b_cube[:,:,:,-2:-1]*4+b_cube[:,:,:,-3:-2])/2], dim=3)/dz
    
    jacobi = torch.stack([b_dx, b_dy, b_dz], dim=1)
    return jacobi

def rot(vec, dxyz=[1.,1.,1.]):
    jacobi = jacobi_matrix(vec, dxyz)
    # rotB: (dBy/dz-dBz/dy, dBz/dx-dBx/dz, dBy/dx-dBx/dy)
    rot_vec = torch.stack([jacobi[2,1]-jacobi[1,2],
                          jacobi[0,2]-jacobi[2,0],
                          jacobi[1,0]-jacobi[0,1]], dim=0)
    
    if isinstance(vec, torch.Tensor):
        return rot_vec
    else:
        return rot_vec.cpu().numpy()

def div(vec, dxyz=[1.,1.,1.]):
    jacobi = jacobi_matrix(vec, dxyz)
    div_vec = (jacobi[0,0] + jacobi[1,1] + jacobi[2,2]).unsqueeze(0)
    
    if isinstance(vec, torch.Tensor):
        return div_vec
    else:
        return div_vec.cpu().numpy()

def grad(vec, dxyz=[1.,1.,1.]):
    jacobi = jacobi_matrix(vec, dxyz)
    grad_vec = torch.stack([jacobi[0,0], jacobi[0,1], jacobi[0,2]], dim=0)
    
    if isinstance(vec, torch.Tensor):
        return grad_vec
    else:
        return grad_vec.cpu().numpy()

def cube_cross(a, b):
    is_array=False
    if isinstance(a, np.ndarray):
        a = torch.from_numpy(a)
        is_array=True
    if isinstance(b, np.ndarray):
        b = torch.from_numpy(b)
    size = a.permute(1,2,3,0).shape
        
    a = a.permute(1,2,3,0).reshape(-1,3)
    b = b.permute(1,2,3,0).reshape(-1,3)
    axb = torch.cross(a, b).reshape(size).permute(3,0,1,2)
    
    if is_array:
        return axb.cpu().numpy()
    else:
        return axb

def cube_dot(a, b):
    is_array = False
    if isinstance(a, np.ndarray):
        a = torch.from_numpy(a)
        is_array = True
    if isinstance(b, np.ndarray):
        b = torch.from_numpy(b)
    
    a_dot_b = torch.sum(a*b, dim=0).unsqueeze(0)
    if is_array:
        a_dot_b = a_dot_b.detach().numpy()
    
    return a_dot_b

def xyz2rtp(xyz):
    """
    Convert Cartesian coordinates (x, y, z) to spherical coordinates (r, theta, phi).
    
    Parameters:
    x (float): x-coordinate
    y (float): y-coordinate
    z (float): z-coordinate
    
    Returns:
    tuple: (r, theta, phi)
        r (float): radial distance
        theta (float): polar angle (in radians)
        phi (float): azimuthal angle (in radians)
    """
    x,y,z=xyz
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.where(r!=0,np.arccos(z/r),0)
    phi = np.arctan2(y, x) % (np.pi*2)
    return np.stack([r, theta, phi])

def rtp2xyz(rtp):
    """
    Convert spherical coordinates (r, theta, phi) to Cartesian coordinates (x, y, z).
    
    Parameters:
    r (float): radial distance
    theta (float): polar angle (in radians)
    phi (float): azimuthal angle (in radians)
    
    Returns:
    tuple: (x, y, z)
        x (float): x-coordinate
        y (float): y-coordinate
        z (float): z-coordinate
    """
    r,theta,phi=rtp
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.stack([x, y, z])

def Vrtp2Vxyz(Vrtp, rtp):
    """
    Convert a vector from spherical coordinates to Cartesian coordinates.

    Parameters:
    Vrtp: A list or array of three elements, representing the vector in spherical coordinates (Vr, Vtheta, Vphi).
    rtp: A list or array of three elements, representing the position in spherical coordinates (r, theta, phi).
         Theta and phi should be in radians.

    Returns:
    Vxyz: An array of three elements, representing the vector in Cartesian coordinates (Vx, Vy, Vz).
    """
    Vr,Vt,Vp = Vrtp
    r , t, p = rtp
    # Calculate the direction of radial unit vector
    e_rx = np.sin(t)*np.cos(p)
    e_ry = np.sin(t)*np.sin(p)
    e_rz = np.cos(t)
    # Calculate the direction of the polor angle unit vector (from z-axis to xy-plane)
    e_tx = np.cos(t)*np.cos(p)
    e_ty = np.cos(t)*np.sin(p)
    e_tz =-np.sin(t)
    # Calculate the direction of the azimuthal angle unit vector (in xy-plane)
    e_px =-np.sin(p)
    e_py = np.cos(p)
    e_pz = 0
    # Convert the vector from spherical to Cartesian coordinates
    Vx = Vr*e_rx+Vt*e_tx+Vp*e_px
    Vy = Vr*e_ry+Vt*e_ty+Vp*e_py
    Vz = Vr*e_rz+Vt*e_tz+Vp*e_pz
    Vxyz = np.stack([Vx,Vy,Vz])
    return Vxyz