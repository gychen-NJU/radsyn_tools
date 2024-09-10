# parallel_qsl.py
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from ..need import *
from ..yt_spherical_3D import spherical_data
from .. import geometry
from ..QSL import initializeUV

def calculate_qsl_range(start_idx, end_idx, boundary_points, progress_list, total_tasks, lock, print_interval=100, t0=0, r_min=1.0):
    res_list = []
    for icnt in range(start_idx, end_idx):
        point = boundary_points[icnt]
        # print('flag 1', flush=True)
        # logQ, length = instance._SquanshingQ(point, bxyz=bxyz, Jacobi=Jacobi, dxyz=dxyz, r_min=1.01)
        lineP     = np.zeros(13)
        lineP[:3] = point
        lineP     = instance.tangent(lineP, bxyz)
        lineP     = initializeUV(lineP)
        if np.max(lineP[9:12])<1e-8:
            logQ, length = 0,0
        else:
            lineF, lineB, lambda_F, lambda_B = instance.LineIntegrate(lineP, bxyz, Jacobi, dxyz, r_min=r_min)
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
            logQ      = -logQ if np.dot(point, lineP[9:12])<0 else logQ
            length    = np.abs(lambda_F)+np.abs(lambda_B)
        # logQ, length = np.random.randn(2)
        # print('flag 2', flush=True)
        res_list.append([icnt,logQ, length])
        # print('flag 3', flush=True)

        with lock:
            progress_list[0] += 1  # 更新计数器
            ti = time.time()
            if progress_list[0] % print_interval == 0 or progress_list[0] == total_tasks or progress_list[0]==1:
                print(f"Progress: {progress_list[0]:6d}/{total_tasks} tasks completed.  Wall_time: {(ti-t0)/60:6.3f} min", flush=True)
                
    # print('flag 4', flush=True)
    return res_list

def parallel_qsl(boundary_points, n_cores=10, print_interval=100, r_min=1.0):
    print('### =========== Parallel computing ============ ###')
    print(f'#       Available CPU cores: {multiprocessing.cpu_count():3d}                  #')
    print(f'#            Used CPU cores: {n_cores:3d}                  #')
    print('### =========================================== ###')
    t0 = time.time()
    n_points = len(boundary_points)
    chunk_size = n_points // n_cores

    # 用于保存结果的列表
    qsl_results = []

    # 使用 Manager 来共享进度信息
    manager = Manager()
    progress_list = manager.list([0])  # 用于存储已完成的任务数，初始化为0
    lock = manager.Lock()  # 使用Manager提供的Lock

    total_tasks = n_points
    print('Initialization OK')

    # 开始并行计算
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        futures = []
        for i in range(n_cores):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < n_cores - 1 else n_points
            futures.append(executor.submit(calculate_qsl_range, start_idx, end_idx, boundary_points, progress_list, total_tasks, lock, print_interval, t0, r_min))

        print('Assigning task OK', flush=True)
        for future in as_completed(futures):
            qsl_results.extend(future.result())

    return qsl_results

t0     = time.time()
parser = argparse.ArgumentParser(description='Parallel computing QSL')
parser.add_argument('-p', type=str  , help='Path to .npy file for target points array', required=False, default='./qsl_points.npy'    )
parser.add_argument('-i', type=str  , help='Path to .pkl file for instance'           , required=False, default='./spherical_data.pkl')
parser.add_argument('-f', type=int  , help='Which frame is?'                          , required=False, default=0                     )
parser.add_argument('-n', type=int  , help='Number of cores to use'                   , required=False, default=10                    )
parser.add_argument('-e', type=int  , help='Print interval'                           , required=False, default=1                     )
parser.add_argument('-r', type=float, help='r_min of the bottom boundary'             , required=False, default=1.01                  )

args           = parser.parse_args()
points         = np.load(args.p)
instance       = spherical_data.load(args.i)
frame          = args.f
n_cores        = args.n
print_interval = args.e
r_min          = args.r

bbox_cart      = np.array(instance.bbox_cart)
bbox           = np.array(instance.bbox)
n              = instance.n_sph
deg            = np.pi / 180
n_points       = points.shape[0]
dxyz           = (bbox_cart[:,1]-bbox_cart[:,0])/instance.n_car

bxyz           = instance.load_bxyz(frame=frame)
Jacobi         = geometry.jacobi_matrix(bxyz, dxyz).detach().cpu().numpy()
ti             = time.time()
print(f'Preparation finished in {(ti-t0):5.2f} seconds...')
    
results        = parallel_qsl(points, n_cores=n_cores, print_interval=print_interval, r_min=r_min)
np.save('./qsl_results.npy', np.array(results))
print('!!! Code Ending !!!')
