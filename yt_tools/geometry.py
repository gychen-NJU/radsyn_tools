from .need import *
import torch

# ===================================================== #
#                    vector analysis                    #
# ===================================================== #
# differential oprator
def jacobi_matrix(b_cube):
    is_array = False
    if isinstance(b_cube, np.ndarray):
        b_cube = torch.from_numpy(b_cube)
        is_array=True
    elif not isinstance(b_cube, torch.Tensor):
        raise ValueError("Input must be a NumPy array or a PyTorch tensor.")
    
    b_dx = torch.cat([(b_cube[:,-1:,:,:]-b_cube[:,-2:-1,:,:]),
                      (b_cube[:,2:,:,:]-b_cube[:,:-2,:,:])/2,
                      (b_cube[:,1:2,:,:]-b_cube[:,0:1,:,:])], dim=1)
    
    b_dy = torch.cat([(b_cube[:,:,-1:,:]-b_cube[:,:,-2:-1,:]),
                      ((b_cube[:,:,2:,:]-b_cube[:,:,:-2,:])/2),
                      (b_cube[:,:,1:2,:]-b_cube[:,:,0:1,:])], dim=2)
    
    b_dz = torch.cat([(b_cube[:,:,:,-1:]-b_cube[:,:,:,-2:-1]),
                      ((b_cube[:,:,:,2:]-b_cube[:,:,:,:-2])/2),
                      (b_cube[:,:,:,1:2]-b_cube[:,:,:,0:1])], dim=3)
    
    jacobi = torch.stack([b_dx, b_dy, b_dz], dim=1)
    return jacobi

def rot(vec):
    jacobi = jacobi_matrix(vec)
    # rotB: (dBy/dz-dBz/dy, dBz/dx-dBx/dz, dBy/dx-dBx/dy)
    rot_vec = torch.stack([jacobi[2,1]-jacobi[1,2],
                          jacobi[0,2]-jacobi[2,0],
                          jacobi[1,0]-jacobi[0,1]], dim=0)
    
    if isinstance(vec, torch.Tensor):
        return rot_vec
    else:
        return rot_vec.cpu().numpy()

def div(vec):
    jacobi = jacobi_matrix(vec)
    div_vec = (jacobi[0,0] + jacobi[1,1] + jacobi[2,2]).unsqueeze(0)
    
    if isinstance(vec, torch.Tensor):
        return div_vec
    else:
        return div_vec.cpu().numpy()

def grad(vec):
    jacobi = jacobi_matrix(vec)
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
