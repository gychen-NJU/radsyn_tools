import multiprocessing
from .need import *
from . import geometry
from .stream_line import *
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager

def twist_number(self, point, frame=0, **kwargs):
    # print('Begin twist calculation', flush=True)
    bbox_cart = self.bbox_cart
    n    = self.n_sph
    err  = kwargs.get('err', 1.e-10)
    dxyz = kwargs.get('dxyz', (bbox_cart[:,1]-bbox_cart[:,0])/self.n_car)
    bxyz_file = os.path.join(self.bxyz_path, 'bxyz_'+str(frame).zfill(4)+'.npy')
    bxyz      = kwargs.get('bxyz',np.load(bxyz_file))
    current   = kwargs.get('current', None)
    if current is None:
        current = geometry.rot(bxyz, dxyz=dxyz)
        print('TEST PROBE 4', flush=True)
    mag_line = self.magline(point, n_lines=1, radius=0.005, frame=frame)[0]
    if len(mag_line)==1:
        return 0
    xyz = self.idx2xyz(mag_line)
    end_point = xyz[-1]
    bvec  = self.sample_with_idx(mag_line, give_field=bxyz.transpose(1,2,3,0))
    jvec  = self.sample_with_idx(mag_line, give_field=current.transpose(1,2,3,0))
    alpha = np.sum(bvec*jvec, axis=1)/(np.linalg.norm(bvec, axis=1)**2+err)
    ss    = np.linalg.norm(xyz[1:]-xyz[:-1], axis=1)
    twist = 1/(4*np.pi)*np.sum(ss*alpha[1:])/(dxyz.min())
    # print('End twist calculation', flush=True)
    return twist

def calculate_twist_range(self, 
                          start_idx, end_idx, boundary_points, progress_list, total_tasks, lock, 
                          print_interval, t0, frame,dxyz, bxyz_file, bxyz, err, current):
    twist_list = []
    # print(f'start_idx:{start_idx}, end_idx:{end_idx}, boundary_points_shape:{boundary_points.shape}, total_tasks:{total_tasks}, frame:{frame}', flush=True)
    for icnt in range(start_idx, end_idx):
        point = boundary_points[icnt]
        twist = self._twist(point, frame=frame, err=err, dxyz=dxyz, bxyz=bxyz, current=current)
        twist_list.append([icnt,twist])

        with lock:
            progress_list[0] += 1  # 更新计数器
            ti = time.time()
            if progress_list[0] % print_interval == 0 or progress_list[0] == total_tasks or progress_list[0]==1:
                print(f"Progress: {progress_list[0]:6d}/{total_tasks} tasks completed.  Wall_time: {(ti-t0)/60:6.3f} min", flush=True)
    
    return twist_list

def parallel_twist(self, boundary_points, n_cores=10, print_interval=100, **kwargs):
    print('### =========== Parallel computing ============ ###')
    print(f'#       Available CPU cores: {multiprocessing.cpu_count():3d}                  #')
    print(f'#            Used CPU cores: {n_cores:3d}                  #')
    print('### =========================================== ###')
    t0 = time.time()
    n_points = len(boundary_points)
    chunk_size = n_points // n_cores

    frame = kwargs.get('frame', 0)
    bbox_cart = self.bbox_cart
    err  = kwargs.get('err', 1.e-10)
    dxyz = kwargs.get('dxyz', (bbox_cart[:,1]-bbox_cart[:,0])/self.n_car)
    bxyz_file = os.path.join(self.bxyz_path, 'bxyz_'+str(frame).zfill(4)+'.npy')
    bxyz      = kwargs.get('bxyz',np.load(bxyz_file))
    current   = kwargs.get('current', geometry.rot(bxyz, dxyz=dxyz))

    # 用于保存结果的列表
    twist_results = []

    # 使用 Manager 来共享进度信息
    manager = Manager()
    progress_list = manager.list([0])  # 用于存储已完成的任务数，初始化为0
    lock = manager.Lock()  # 使用Manager提供的Lock

    total_tasks = n_points

    print('initial OK')
    # 开始并行计算
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        futures = []
        for i in range(n_cores):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < n_cores - 1 else n_points
            futures.append(executor.submit(self.calculate_twist_range, start_idx, end_idx, boundary_points, progress_list, total_tasks, lock, print_interval, t0, frame, dxyz, bxyz_file, bxyz, err, current))

        print('assigning task OK', flush=True)
        for future in as_completed(futures):
            twist_results.extend(future.result())

    return twist_results
