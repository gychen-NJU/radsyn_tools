import numpy as np 
import yt 
import aglio
import glob
import re
import os
import time
import logging
import pickle
import pkg_resources
import sunpy.map
import vtk
import gc

from typing import List
from multiprocessing import Pool
from scipy.spatial import cKDTree
from astropy import units as u
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.graph_objects as go
from sunpy.visualization.colormaps import color_tables as ct
from vtk.util.numpy_support import numpy_to_vtk
from scipy.ndimage import gaussian_filter
from scipy import ndimage

from .stream_line import *
from . import geometry
from .need import *
from .QSL import tangent, rfunction, rk4_stepping, LineIntegrate, _SquanshingQ, parallel_QSL, plot2D_QSL
from .twist import twist_number, parallel_twist, calculate_twist_range
from .yt_spherical_3D import aia_response, is_ipython_environment

class cartesian_data():
    def __init__(self, folder='./data/', sample_level=1):
        start_time             = time.time()
        self.folder            = folder
        self.ts                = yt.load(os.path.join(folder,'*.dat'))
        self.sample_level      = sample_level
        ds                     = self.ts[0]
        self.domain_left_edge  = ds.domain_left_edge
        self.domain_right_edge = ds.domain_right_edge
        self.bbox              = np.array([ds.domain_left_edge, ds.domain_right_edge]).T
        self.dimensions        = ds.domain_dimensions
        self.nxyz              = self.dimensions*sample_level
        self.dxyz              = (self.bbox[:,1]-self.bbox[:,0])/(self.nxyz-1)
        self.info              = dict()
        end_time               = time.time()
        print(f"Initialization completed in {end_time - start_time} seconds")

    def __getstate__(self):
        state = self.__dict__.copy()
        # 移除无法序列化的属性
        del state['ts']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # 重新加载或初始化 'ts' 属性
        self.ts      = yt.load(os.path.join(self.folder, '*.dat'))
        self.fig     = None
        self.rad_fig = None

    def save(self, output_name='cartesian_data.pkl'):
        self.output_name = output_name
        directory = os.path.dirname(output_name)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        with open(output_name, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        print(f"Instance saved to {output_name}")

    def _xyz(self):
        sample_level = self.sample_level
        nx,ny,nz     = self.nxyz
        bbox         = self.bbox
        X,Y,Z        = np.meshgrid(np.linspace(*bbox[0],nx),
                                   np.linspace(*bbox[1],ny),
                                   np.linspace(*bbox[2],nz),
                                   indexing='ij')
        return np.stack([X,Y,Z])

    @classmethod
    def load(cls, load_name='cartesian_data.pkl'):
        with open(load_name, 'rb') as input:
            return pickle.load(input)

    def uniform_sample(self, field, sample_level=None, frame=0):
        ds                = self.ts[frame]
        if sample_level==None:
            sample_level = self.sample_level
        else:
            self.sample_level = sample_level
        nx,ny,nz          = self.dimensions*sample_level
        region            = ds.r[::nx*1j,::ny*1j,::nz*1j]
        ret_dict          = {}
        if isinstance(field, list):
            for f in field:
                ret_dict[f]=region[f]
        else:
            ret_dict[field]=region[field]
        return  ret_dict

    def return_bxyz(self, sample_level=1, frame=0):
        ds = self.ts[frame]
        self.sample_level = sample_level
        nx,ny,nz = ds.domain_dimensions*sample_level
        region   = ds.r[::nx*1j,::ny*1j,::nz*1j]
        dxyz     = (self.bbox[:,1]-self.bbox[:,0])/(np.array([nx,ny,nz])-1)
        bx       = region['b1']
        by       = region['b2']
        bz       = region['b3']
        bxyz     = np.stack([bx,by,bz], axis=0)
        return np.array(bxyz)

    def proxy_emissivity(self, sample_level=1, frame=0, **kwargs):
        n_cores        = kwargs.get('n_cores', 10)
        max_step       = kwargs.get('max_step', 10000)
        print_interval = kwargs.get('print_interval', 100)
        bxyz, dxyz = self.return_bxyz(sample_level, frame)
        coords   = 'cartesian'
        nx,ny,nz = self.dimensions*sample_level
        x = np.arange(nx)
        y = np.arange(ny)
        z = np.array([1])
        X,Y,Z = np.meshgrid(x,y,z,indexing='ij')
        points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        info_dict   = dict(bxyz=bxyz, dxyz=dxyz, points=points)
        np.savez('proxy_emissivity_info.npz', **info_dict)
        command     = f"python -m yt_tools.scripts.ProxyEmissivity -n {n_cores} --max_step {max_step} --n_print {print_interval}"
        ret         = os.system(command)
        proxy_emissivity = np.load('./proxy_emissivity_temp.npy')
        os.remove('./proxy_emissivity_temp.npy')
        os.remove('./proxy_emissivity_info.npz')
        return proxy_emissivity

    def show_proxy_emissivity(self, 
                              sample_level=1, 
                              frame=0, 
                              is_view3d=False, 
                              proxy_emissivity=None, 
                              n_cores=10, 
                              max_step=10000, 
                              print_interval=100, 
                              **kwargs):
        # is_view3d = kwargs.get('is_view3d', False)
        if (proxy_emissivity is None):
            proxy_emissivity = self.proxy_emissivity(sample_level, frame, n_cores=n_cores, max_step=max_step, print_interval=print_interval)
        shp = proxy_emissivity.shape
        ds_dict = dict(proxy_emissivity=proxy_emissivity)
        ds = yt.load_uniform_grid(ds_dict, shp, bbox=self.bbox, length_unit='code_length')
        if is_view3d:
            view3D(ds, 'proxy_emissivity')
        else:
            p = projection(ds, 'proxy_emissivity', **kwargs)
            return p

    def xyz2idx(self, xyz):
        nxyz = self.dimensions*self.sample_level
        dxyz = (self.bbox[:,1]-self.bbox[:,0])/(nxyz-1)
        idx  = (xyz-self.bbox[:,0])/dxyz
        return idx

    def idx2xyz(self, idx):
        nxyz = self.dimensions*self.sample_level
        dxyz = (self.bbox[:,1]-self.bbox[:,0])/(nxyz-1)
        xyz  = idx*dxyz+self.bbox[:,0]
        return xyz

    def sample_with_idx(self, idx=None, field=None, frame=0, **kwargs):
        give_field      = kwargs.get('give_field', None)
        if give_field is None:
            if field is None:
                raise ValueError("`field` is needed or `give_field` should be provided")
            else:
                f = self.uniform_sample(field, sample_level=self.sample_level, frame=frame)
        else:
            f = give_field
        ret = trilinear_interpolation(f, idx)
        return ret

    # from .QSL import tangent, rfunction, rk4_stepping, LineIntegrate, _SquanshingQ
    def parallel_qsl(self, point, **kwargs):
        Nc     = kwargs.get('n_cores'       , 50     )
        Np     = kwargs.get('print_interval', 1000   )
        dl     = kwargs.get('step_size'     , -1     )
        PY     = kwargs.get('python'    , 'python '  )
        F      = kwargs.get('frame'         , 0      )
        Ns     = kwargs.get('max_steps'     , 1000000)
        p_file = './qsl_point.npy'
        i_file = './cartesian_data.pkl'
        np.save(p_file, point)
        self.save()
        command = PY+f' -u -m yt_tools.scripts.cartesian_qsl '+\
                  f'-p {p_file} -i {i_file} -f {F} -n {Nc} -e {Np} --max_step {Ns} --step_size {dl}'
        ret         = os.system(command)
        qsl_results = np.load('./cartesian_qsl.npy')
        os.remove('./cartesian_qsl.npy')
        os.remove(p_file)
        return qsl_results

    def volume_qsl_task(self,**kwargs):
        Nc     = kwargs.get('n_cores'       , 50     )
        Np     = kwargs.get('print_interval', 1000   )
        dl     = kwargs.get('step_size'     , -1     )
        PY     = kwargs.get('python'    , 'python '  )
        F      = kwargs.get('frame'         , 0      )
        Ns     = kwargs.get('max_steps'     , 1000000)
        sigma  = kwargs.get('sigma', 2)
        tc     = kwargs.get('trancate', 1)
        FM     = kwargs.get('fast_mode', True)
        i_file = './cartesian_data.pkl'
        self.save()
        command = PY+f' -u -m yt_tools.scripts.Cartesian_volume_qsl '+\
                  f'-i {i_file} -f {F} -n {Nc} -e {Np} --max_step {Ns} --step_size {dl}'
        ret         = os.system(command)
        results = np.load('./Cartesian_volume_qsl.npz')
        os.remove('./Cartesian_volume_qsl.npz')
        lgQ = results['logQ']
        Len = results['length']
        smoothed_lgQ = gaussian_filter(lgQ, sigma=sigma, truncate=tc)
        smoothed_Len = gaussian_filter(Len, sigma=sigma, truncate=tc)
        sobel_x = ndimage.sobel(smoothed_Len, axis=0)
        sobel_y = ndimage.sobel(smoothed_Len, axis=1)
        sobel_z = ndimage.sobel(smoothed_Len, axis=2)
        Scube   = np.sqrt(sobel_x**2+sobel_y**2+sobel_z**2)
        return lgQ, Len, smoothed_lgQ, smoothed_Len, Scube

    def plot2D_QSL(self, qsl, nx=None, ny=None, **kwargs):
        return plot2D_QSL(self, qsl, nx=nx, ny=ny, **kwargs)

    def get_magline(self, xyzs, **kwargs):
        sample_level = kwargs.pop('sample_level', self.sample_level)
        frame        = kwargs.pop('frame', 0)
        Bxyz = self.return_bxyz(sample_level=sample_level, frame=frame).transpose(1,2,3,0)
        xyz  = self._xyz().transpose(1,2,3,0)
        cm   = Cartesian_magline(Bxyz, xyz, **kwargs)
        ret  = cm.magline_solver(xyzs, **kwargs)
        return ret

def view3D(ds, field):
    rc = yt_idv.render_context()
    sc=rc.add_scene(ds, field)
    rc.run()

def projection(ds, field, **kwargs):
    time0     = time.time()
    center    = kwargs.get('center', 'c')
    width     = kwargs.get('width', ds.quan(1, 'unitary'))
    norm      = kwargs.get('norm',  [0,0,1])
    norm      = np.array(norm)
    norm      = norm/np.linalg.norm(norm)
    save_path = kwargs.get('save_path', None)
    zmin      = kwargs.get('zmin', None)
    zmax      = kwargs.get('zmax', None)
    north     = kwargs.get('north', None)
    cmap      = kwargs.get('cmap', None)
    xlabel    = kwargs.get('xlabel', '')
    ylabel    = kwargs.get('ylabel', '')
    clabel    = kwargs.get('clabel', f'{field}')
    set_cb    = kwargs.get('set_colorbar', True)
    show      = kwargs.get('img_show', True)
    plot_norm = kwargs.get('plot_norm', None)
    save_name = kwargs.get('save_name', f'projection_{field}.png')
    p = yt.OffAxisProjectionPlot(ds, norm, field, center=center, width=width, north_vector=north)
    if cmap!=None:
        p.set_cmap(field, cmap)
    p.set_xlabel(xlabel)
    p.set_ylabel(ylabel)
    p.set_colorbar_label(field, clabel)
    if zmin!=None and zmax!=None:
        p.set_zlim(field, zmin, zmax)
    if not set_cb:
        p.hide_colorbar()
    if plot_norm is not None:
        p.set_norm(field, plot_norm)

    fig = p.plots[field].figure
    ax = fig.axes[0]
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    plt.tight_layout()
    if save_path != None:
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        p.save(os.path.join(save_path, save_name))
        # print('Time spent: %6.3f sec' % (time.time()-time0))
    if is_ipython_environment() and show:
        p.show()
    return p