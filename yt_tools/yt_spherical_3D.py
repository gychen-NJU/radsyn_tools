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

from .stream_line import *

def is_ipython_environment():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:  # 检查是否在IPython内核中
            return False
    except Exception:
        return False
    return True

class aia_response():
    def __init__(self):
        self.response_curves = {}
        self.curve_files_path = pkg_resources.resource_filename('yt_tools', 'AIA_response')
        self.wavebands = ['94', '131', '171', '193', '211', '304', '335']
        self.curve_files = glob.glob(self.curve_files_path+'/*.npz')
        for iwave in self.wavebands:
            pattern = re.compile(iwave)
            matches = filter(lambda x: pattern.search(x) is not None, self.curve_files)
            matches = list(matches)[0]
            self.response_curves[iwave]=np.load(matches)['a']

    def response(self, Log_T, iwave='171'):
        Log_T0 = self.response_curves[iwave][:,0]
        res0   = self.response_curves[iwave][:,1]
        # log_T  = np.log10(T_in*1e6)
        res    = np.interp(Log_T, Log_T0, res0)
        return res

    def res_plot(self):
        T_in = np.linspace(5.0, 8.0, 100)
        for iwave in self.wavebands:
            plt.plot(T_in, self.response(T_in, iwave), '--', label=iwave)
            plt.legend()
            plt.yscale('log')
            plt.ylim(1e-28, 1e-23)
            plt.xlabel(r'Log$_{10}(T)$')
            plt.ylabel('Response [DN~$\mathrm{cm^5 s^{-1} pix^{-1}}]$')

    def __call__(self, Log_T, iwave='171'):
        return self.response(Log_T, iwave=iwave)



class spherical_data():
    def __init__(self, folder='./data/', xyz_file='./x_y_z.npz', n_sph=400, n_car=256, fields_save_path='./yt_fields/'):
        start_time             = time.time()
        self.folder            = folder
        self.ts                = yt.load(os.path.join(folder,'*.dat'))
        self.n_sph             = n_sph
        self.n_car             = n_car
        ds                     = self.ts[0]
        self.domain_left_edge  = ds.domain_left_edge
        self.domain_right_edge = ds.domain_right_edge
        self.r                 = np.linspace(float(self.domain_left_edge[0]), float(self.domain_right_edge[0]), self.n_sph)
        self.lat               = np.linspace(90-float(self.domain_left_edge[1])*180/np.pi, 90-float(self.domain_right_edge[1])*180/np.pi, self.n_sph)
        self.lon               = np.linspace(float(self.domain_left_edge[2])*180/np.pi, float(self.domain_right_edge[2])*180/np.pi, self.n_sph)
        r_g, lat_g, lon_g      = self.get_test_array(self.r, self.lat, self.lon)
        self.bbox              = [[self.r.min(), self.r.max()], [self.lat.min(), self.lat.max()], [self.lon.min(), self.lon.max()]]
        self.the_tree          = self.get_kdtree([r_g, lat_g, lon_g,])
        xyz_g                  = aglio.coordinate_transformations.geosphere2cart(lat_g, lon_g, r_g)
        self.bbox_cart         = np.array([[dim.min(), dim.max()] for dim in xyz_g])

        x_sample               = np.linspace(self.bbox_cart[0][0], self.bbox_cart[0][1], self.n_car)
        y_sample               = np.linspace(self.bbox_cart[1][0], self.bbox_cart[1][1], self.n_car)
        z_sample               = np.linspace(self.bbox_cart[2][0], self.bbox_cart[2][1], self.n_car)
        self.xyz_file          = xyz_file
        self.fields_save_path  = fields_save_path
        x_sample, y_sample, z_sample = np.meshgrid(x_sample, y_sample, z_sample, indexing='ij')
        np.savez(self.xyz_file,x_sample= x_sample,y_sample= y_sample,z_sample= z_sample,bbox_cart=self.bbox_cart,bbox=self.bbox)
        self.fields_file_name  = None
        self.output_name       = None
        self.fig               = None
        self.n_drop            = 0
        self.rad_fig           = None
        self.bxyz_path         = None
        self.index             = None
        self.n_cut             = 0

        end_time               = time.time()
        print(f"Initialization completed in {end_time - start_time} seconds")

    def __getstate__(self):
        state = self.__dict__.copy()
        # 移除无法序列化的属性
        del state['ts']
        del state['fig']
        del state['rad_fig']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # 重新加载或初始化 'ts' 属性
        self.ts      = yt.load(os.path.join(self.folder, '*.dat'))
        self.fig     = None
        self.rad_fig = None
    
    def get_test_array(self, r, lat, lon):
        r_g, lat_g, lon_g = np.meshgrid(r, lat , lon, indexing='ij')
        return r_g, lat_g, lon_g

    def get_kdtree(self, coord_arrays: List[np.ndarray], **kwargs):
        n_dims = len(coord_arrays)
        normalized_coords = []
        for idim in range(n_dims):
            assert coord_arrays[idim].shape == coord_arrays[0].shape
    
        for idim in range(n_dims):
            dim_1d = coord_arrays[idim].ravel()        
            normalized_coords.append(self.normalize_coord(dim_1d, idim))

        normalized_coords = np.column_stack(normalized_coords)
    
        # print(normalized_coords)
        return cKDTree(normalized_coords, **kwargs)

    def normalize_coord(self, dim_vals, dim_index):
        return (dim_vals - self.bbox[dim_index][0]) / (self.bbox[dim_index][1] -  self.bbox[dim_index][0])

    def load_xyz(self):
        sample=np.load(self.xyz_file)
        x_sample=sample['x_sample']
        y_sample=sample['y_sample']
        z_sample=sample['z_sample']
        bbox_cart=sample['bbox_cart']
        bbox=sample['bbox']
        return x_sample, y_sample, z_sample, bbox_cart, bbox

    def _mask_outside_bounds(self, r, lat, lon, n_drop=0):
        self.n_drop = n_drop
        # lon    = np.where(lon < 0, lon+360, lon)
        dr     = (self.bbox[0][1]-self.bbox[0][0])/self.n_sph*n_drop
        inside = r >= self.bbox[0][0]+dr
        inside = (inside) & (r   < self.bbox[0][1])
        inside = (inside) & (lat < self.bbox[1][1])
        inside = (inside) & (lat > self.bbox[1][0])
        inside = (inside) & (lon < self.bbox[2][1])
        inside = (inside) & (lon > self.bbox[2][0])
        return ~inside

    def cart2sph(self, X, Y, Z):
        R = np.sqrt(X**2+Y**2+Z**2)
        T = np.arcsin(Z/R)
        P = np.arctan(Y/X)
        P = np.where(X<0,P+np.pi,P)
        return R,T*180/np.pi,P*180/np.pi

    def get_index(self, x, y, z, n_drop):
        r, lat, lon   = self.cart2sph(x, y, z)
        r             = r.ravel()
        lat           = lat.ravel()
        lon           = lon.ravel()
        outside       = self._mask_outside_bounds(r, lat, lon, n_drop)
        r             = self.normalize_coord(r, 0)
        lat           = self.normalize_coord(lat, 1)
        lon           = self.normalize_coord(lon, 2)
        # query the tree
        dists, indexs = self.the_tree.query(np.column_stack([r, lat, lon]), k=1)
        return indexs, outside

    def sample_field(self, val_1d, field_name, x,y,z, dr=0.0, **kwargs):
        n_drop   = np.ceil(dr/(self.bbox[0][1]-self.bbox[0][0])*self.n_sph)
        fill_val = kwargs.get('fill_val',0)
        orig_shape = x.shape
        indexs = kwargs.get('indexs', None)
        outside = kwargs.get('outside', None)
        if not (isinstance(indexs,np.ndarray) and isinstance(outside, np.ndarray)):
            print("Don't give a index, try to calculate")
            indexs,outside = self.get_index(x,y,z, n_drop)
            np.savez(os.path.join(self.folder,'indexs.npz'), **dict(indexs=indexs,outside=outside))
        # select field values
        indexs[outside] = -1
        vals = np.zeros_like(indexs).astype(float)
        vals[indexs>=0] = val_1d[indexs[indexs>=0]]
        vals[indexs<0]  = fill_val
        index = indexs
        return vals.reshape(orig_shape), index.reshape(orig_shape)

    def save_fields(self, fields_list=['rho','e','m1','m2','m3','b1','b2','b3'], fields_file_name='fields', auto_skip=False, dr=0.0, n_cut='auto',begin=0, end=-1):
        '''
        dr            :   要切除掉的层的厚度，可缺省，默认为 0
        fields_list   :   需要提取的物理场，可以缺省（不推荐），默认为 ['rho','e','m1','m2','m3','b1','b2','b3']
        auto_skip     :   检测是否已经存在文件，并自动跳过，可以确认，默认为 False
        n_cut         :   要切掉几层，可以缺省，默认为 0
        begin         :   从第几帧开始，可以缺省，默认为 0
        end           :   结束帧，可以缺省，默认为 -1 (全部）
        '''
        n_drop   = np.ceil(dr/(self.bbox[0][1]-self.bbox[0][0])*self.n_sph)
        end = len(self.ts) if end<0 else min(len(self.ts), end+1)
        self.fields_file_name = fields_file_name
        n = self.n_sph
        x_sample, y_sample, z_sample, bbox_cart, bbox = self.load_xyz()
        n_cut = self.n_sph/self.n_car*max(self.bbox_cart[:,1]-self.bbox_cart[:,0])/(self.bbox[0][1]-self.bbox[0][0])*np.sqrt(3) if n_cut=='auto' else n_cut
        n_cut = np.ceil(n_cut)
        self.n_cut = n_cut
        if not os.path.exists(self.fields_save_path):
            os.makedirs(self.fields_save_path)
        for idx, ds in enumerate(self.ts[begin:end]):
            idx+=begin
            file_prob = os.path.join(self.fields_save_path,fields_file_name+str(idx).zfill(4)+'.npz')
            if auto_skip:
                if os.path.exists(file_prob):
                    print(f'File {file_prob} already exists. Skipping...')
                    continue
            region = ds.r[::n*1j, ::n*1j, ::n*1j]
            save_dict = {}
            time0 = time.time()
            indexs, outside = self.get_index(x_sample, y_sample, z_sample, n_drop)
            print('Get indexs time: %6.3f sec' % (time.time()-time0))
            np.savez(os.path.join(self.folder,'indexs.npz'), **dict(indexs=indexs,outside=outside))
            for field in fields_list:
                time0 = time.time()
                f = field
                val_pre = np.array(region[('amrvac', f)])
                val_1d  = val_pre.ravel()
                save_dict[field], index = self.sample_field(val_1d, f, x_sample, y_sample, z_sample, 
                                                            dr=dr, indexs=indexs, outside=outside)
                indices = np.where((index < (self.n_drop+n_cut+1)*n*n) & (index >= self.n_drop*n*n))
                save_dict[field][indices] = 0
                self.index = index
                print('process: %04d/%04d,  field: %5s, time: %6.3f sec' % (idx+1, end-begin, f, time.time()-time0))
            np.savez(os.path.join(self.fields_save_path,fields_file_name+str(idx).zfill(4)+'.npz'), **save_dict)
            print(f"!!!    File {file_prob} is created     !!!")
            
    def save(self, output_name='spherical_data.pkl'):
        self.output_name = output_name
        directory = os.path.dirname(output_name)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        with open(output_name, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        print(f"Instance saved to {output_name}")

    def return_ds(self, frame=0, fields_path='default'):
        if fields_path == 'default':
            fields_path = self.fields_save_path
        fields_file = glob.glob(os.path.join(fields_path, '*.npz'))
        fields_file = sorted(fields_file)
        if fields_file == []:
            raise SystemError(f'{fields_path} has no *.npz files')
        # field = np.load(fields_file[frame])
        fields_file = os.path.join(self.fields_save_path,self.fields_file_name+str(frame).zfill(4)+'.npz')
        # print(f'load the fields_file: {fields_file}')
        field = np.load(fields_file)
        new_dict = {}
        keys = field.files
        shp = field[keys[0]].shape
        for f in keys:
            new_dict[f] = np.nan_to_num(field[f], nan=0)
        ds = yt.load_uniform_grid(new_dict, shp, bbox=self.bbox_cart, length_unit = 'Mm')
        return ds

    def add_radiation(self, iwave='171', fields_path='default', frame=0):
        ds = self.return_ds(frame=frame, fields_path=fields_path)
        ds.add_field(('gas', 'sdoaia'+iwave), 
                     function=lambda field,data: aia_response().response(np.nan_to_num(np.log10(data['e']/data['rho']*0.66*1e6)), iwave)*data['rho']**2, 
                     sampling_type='cell', force_override=True)
        return ds

    def radiation_synthesis(self, norm=[1,0,0], iwave='171', fields_path='default', frame=0, zmin='auto', zmax='auto', save_path=None):
        wavelength = int(iwave) * u.angstrom
        aia_cmap = ct.aia_color_table(wavelength)
        ds = self.add_radiation(iwave=iwave, fields_path=fields_path, frame=frame)
        center = 'c'
        width = ds.quan(1, 'unitary')
        norm = np.array(norm)
        norm = norm/np.linalg.norm(norm)
        field = ('gas', f'sdoaia{iwave}')
        zmin = None if zmin=='auto' else zmin
        zmax = None if zmax=='auto' else zmax
        p = yt.OffAxisProjectionPlot(ds, norm, ('gas', 'sdoaia'+iwave), center=center, width=width)
        p.set_cmap(('gas','sdoaia'+iwave), aia_cmap)
        p.set_xlabel('x')
        p.set_ylabel('y')
        p.set_colorbar_label(('gas', 'sdoaia'+iwave), 'AIA SDO %s$\mathrm{\AA}$' % iwave)
        if zmin!=None and zmax!=None:
            p.set_zlim(field, zmin, zmax)

        self.fig = p.plots[('gas', 'sdoaia'+iwave)].figure
        ax = self.fig.axes[0]
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        plt.tight_layout()
        self.rad_fig = p
        if save_path != None:
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            p.save(os.path.join(save_path, f'radiation_{iwave}_{frame:02}.png'))
        if is_ipython_environment():
            p.show()

    def projection(self, field='rho', norm=[1,0,0], fields_path='default', frame=0, zmin='auto', zmax='auto', save_path=None,ds=None):
        ds     = self.return_ds(frame=frame) if ds==None else ds
        center = 'c'
        width  = ds.quan(1, 'unitary')
        norm   = np.array(norm)
        norm   = norm/np.linalg.norm(norm)
        zmin   = None if zmin=='auto' else zmin
        zmax   = None if zmax=='auto' else zmax
        p = yt.OffAxisProjectionPlot(ds, norm, field, center=center, width=width)
        # p.set_cmap(('gas','sdoaia'+iwave), aia_cmap)
        p.set_xlabel('x')
        p.set_ylabel('y')
        p.set_colorbar_label(field, f'{field}')
        if zmin!=None and zmax!=None:
            p.set_zlim(field, zmin, zmax)

        self.fig = p.plots[field].figure
        ax = self.fig.axes[0]
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        plt.tight_layout()
        self.rad_fig = p
        if save_path != None:
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            p.save(os.path.join(save_path, f'projection_{field}_{frame:02}.png'))
        if is_ipython_environment():
            p.show()
            
    def rad_review(self,iwave='171', frame=0, fields_path='default'):
        ds = self.add_radiation(iwave=iwave, fields_path=fields_path, frame=frame)
        rc = yt_idv.render_context()
        sc=rc.add_scene(ds, ('gas', 'sdoaia'+iwave))
        rc.run()       
        print("Camera's Position: ", sc.camera.position)
        print("Camera's Focus   : ", sc.camera.focus)
        print("Normal verctor   : ", sc.camera.position-sc.camera.focus)

    def rad_vtk(self, iwave='171', start_frame=0, end_frame='auto', fields_path='default' ,sight='top', save_path='vtk', auto_skip=True):
        end_frame   = min(end_frame, len(self.ts)) if end_frame!='auto' else len(self.ts)
        n           = self.n_car
        field       = ('gas', 'sdoaia'+iwave)
        xlb,ylb,zlb = self.bbox_cart[:,0]
        xub,yub,zub = self.bbox_cart[:,1]
        dx          = (self.bbox_cart[0][1]-self.bbox_cart[0][0])/n
        dy          = (self.bbox_cart[1][1]-self.bbox_cart[1][0])/n
        dz          = (self.bbox_cart[2][1]-self.bbox_cart[2][0])/n
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
            print(f'create folder: {save_path}')
        for iframe in range(start_frame, end_frame):
            t_start   = time.time()
            vtk_name  = os.path.join(save_path,f'sdoaia{iwave}_{sight}_{iframe:02}.vtk')
            ds        = self.add_radiation(iwave=iwave, fields_path=fields_path, frame=iframe)
            region    = ds.r[::n*1j,::n*1j,::n*1j]
            rad       = np.nan_to_num(np.array(region[field]))
            imageData = vtk.vtkImageData()
            nx,nz,ny  = rad.shape
            if os.path.exists(vtk_name):
                print(f'File {vtk_name} already exists. Skipping...')
                continue
            if sight=='top':
                emission = rad.sum(axis=0).transpose()
                imageData.SetDimensions(1,ny,nz)
                imageData.SetSpacing(dx,dy,dz)
                imageData.SetOrigin(xlb-1e-3,ylb,zlb)
            elif sight=='right':
                emission = rad.sum(axis=1)
                imageData.SetDimensions(nx,1,nz)
                imageData.SetSpacing(dx,dy,dz)
                imageData.SetOrigin(xlb,ylb-1.e-3,zlb)
            elif sight=='left':
                emission = rad.sum(axis=1)
                imageData.SetDimensions(nx,1,nz)
                imageData.SetSpacing(dx,dy,dz)
                imageData.SetOrigin(xlb,yub+1.e-3,zlb)
            elif sight=='up':
                emission = rad.sum(axis=2).transpose()
                imageData.SetDimensions(nx,ny,1)
                imageData.SetSpacing(dx,dy,dz)
                imageData.SetOrigin(xlb,ylb,zlb-1.e-3)
            elif sight=='down':
                emission = rad.sum(axis=2).transpose()
                imageData.SetDimensions(nx,ny,1)
                imageData.SetSpacing(dx,dy,dz)
                imageData.SetOrigin(xlb,ylb,zub+1.e-3)
            else:
                raise SystemError("sight should be set within  'top','left','right','up' or 'down'")
            vtk_data = numpy_to_vtk(emission.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
            vtk_data.SetName('SDOAIA'+iwave)
            imageData.GetPointData().SetScalars(vtk_data)

            writer = vtk.vtkStructuredPointsWriter()
            writer.SetFileName(vtk_name)
            writer.SetInputData(imageData)
            writer.Write()
            t_end = time.time()
            dt    = t_end-t_start
            print(f'Save the vtk file: {vtk_name};  Time to import: {dt:6.2f} sec')
            del rad, imageData, vtk_data
            gc.collect()  # 强制垃圾回收

    def add_field(self, f_names,  frame='all', dr=0.0, n_cut=0):
        print(f'frame: {frame}')
        choice = self.ts if frame=='all' else self.ts[frame]
        begin  = 0 if frame=='all' else frame
        end    = len(self.ts) if frame=='all' else frame+1
        x_sample, y_sample, z_sample, bbox_cart, bbox = self.load_xyz()
        n = self.n_sph
        for idx in range(begin,end):
            ds = self.ts[idx]
            field_file = os.path.join(self.fields_save_path,self.fields_file_name+str(idx).zfill(4)+'.npz')
            dict = np.load(field_file)
            check = set(f_names)-set(dict.files)
            new_dict = {}
            for f in dict.files:
                new_dict[f]=dict[f]
            if not check:
                print(f'All {f_names} have been add in file: {field_file}')
                continue
            else:
                region  = ds.r[::n*1j,::n*1j,::n*1j]
                temp    = np.load(os.path.join(self.folder,'indexs.npz'))
                indexs  = temp['indexs']
                outside = temp['outside']
                for field in check:
                    time0 = time.time()
                    f = field
                    val_pre = np.array(region[('amrvac', f)])
                    val_1d  = val_pre.ravel()
                    new_dict[field], index = self.sample_field(val_1d, f, x_sample, y_sample, z_sample, 
                                                               dr=dr, indexs=indexs, outside=outside)
                    indices = np.where((index < (self.n_drop+self.n_cut+2)*n*n) & (index > (self.n_drop+1)*n*n))
                    new_dict[field][indices] = 0
                    print('process: %04d/%04d,  field: %5s, time: %6.3f sec' % (idx+1, len(self.ts), f, time.time()-time0))
                np.savez(field_file, **new_dict)
                print(f"!!!    File {field_file} is recreated     !!!")

    def brtp2bxyz(self, frame=0, save_path='default',**kwargs):
        save_path = os.path.join(self.folder,'bxyz') if save_path=='default' else save_path
        self.bxyz_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        field_file = os.path.join(self.fields_save_path,self.fields_file_name+str(frame).zfill(4)+'.npz')
        data = np.load(field_file)
        check = {'b1','b2','b3'}-set(data.files)
        if not (not check):
            self.add_field(f_names=['b1','b2','b3'], frame=frame, **kwargs)
            data = np.load(field_file)
        X, Y, Z, bbox_cart, bbox = self.load_xyz()
        br = data['b1']
        bt = data['b2']
        bp = data['b3']
        b_vec = brtp2bxyz([br,bt,bp], [X,Y,Z])
        bxyz_file_name = os.path.join(save_path, 'bxyz_'+str(frame).zfill(4)+'.npy')
        np.save(bxyz_file_name,b_vec)
        print(f'Create file: {bxyz_file_name}')

    def magline(self, p, n_lines=10, radius=0.005, frame=0):
        if self.bxyz_path==None:
            print('Start to save bxyz_file')
            self.brtp2bxyz(frame=frame)
            bxyz_file = os.path.join(self.bxyz_path, 'bxyz_'+str(frame).zfill(4)+'.npy')
        else:
            bxyz_file = os.path.join(self.bxyz_path, 'bxyz_'+str(frame).zfill(4)+'.npy')
            if not os.path.exists(bxyz_file):
                print('Start to save bxyz_file')
                self.brtp2bxyz(frame=frame)
        x0,y0,z0     = p
        bbox_cart    = self.bbox_cart
        bbox         = np.array(self.bbox)
        dxyz         = bbox_cart[:,1]-bbox_cart[:,0]
        n            = self.n_car
        s_unit       = dxyz.min()/n
        n_dr         = np.ceil(radius/s_unit)
        x0_idx       = round((x0-bbox_cart[0][0])/(bbox_cart[0][1]-bbox_cart[0][0])*n)
        y0_idx       = round((y0-bbox_cart[1][0])/(bbox_cart[1][1]-bbox_cart[1][0])*n)
        z0_idx       = round((z0-bbox_cart[2][0])/(bbox_cart[2][1]-bbox_cart[2][0])*n)
        start_points = sphere_sample([x0_idx, y0_idx, z0_idx], r=n_dr, nsample=n_lines)
        b_vec        = np.load(bxyz_file)
        field_lines = []
        for p in start_points:
            fieldline = field_line(b_vec, p, dxyz=dxyz)
            field_lines.append(fieldline)
        return field_lines

    def show_magline(self, p, n_lines=10, radius=0.005, frame=0, **kwargs):
        '''
        spherical_data.show_magline need the following input:
        p ([x0,y0,z0])    :  磁力线的初始点，需要参数
        radius            :  在一个球形内取样磁力线，可以缺省，默认为 0.01
        frame             :  对某一帧画磁力线，可缺省，默认为 0
        n_lines           :  取多少条磁力线，可缺省，默认为 10
        r_slice           :  显示一个底边界的切片，默认为数据的下界
        补充：一般第一次可视化会涉及到分量转化和坐标存取，因此稍微慢一些，后续重复会比较快
        '''
        bbox         = np.array(self.bbox)
        fields_file  = os.path.join(self.fields_save_path,self.fields_file_name+str(frame).zfill(4)+'.npz')
        data         = np.load(fields_file)
        br           = data['b1']
        r_slice      = kwargs.get('r_slice',bbox[0][0])
        n_slice      = np.ceil((r_slice-bbox[0][0])/(bbox[0][1]-bbox[0][0])*self.n_sph)
        n            = self.n_sph
        x_sample, y_sample, z_sample, bbox_cart, bbox = self.load_xyz()
        if isinstance(self.index, np.ndarray):
            index = self.index
            indices = np.where((index < (n_slice+2)*n*n) & (index > (n_slice+1)*n*n))
        else:
            val_pre = br
            val_1d  = val_pre.ravel()
            if os.path.exists(os.path.join(self.folder,'indexs.npz')):
                temp = np.load(os.path.join(self.folder,'indexs.npz'))
                indexs = temp['indexs']
                outside = temp['outside']
            else:
                indexs = None
                outside = None
            Br, index = self.sample_field(val_1d, 'b1', x_sample, y_sample, z_sample, 
                                          dr=0.0, indexs=indexs, outside=outside)
            indices = np.where((index < (n_slice+2)*n*n) & (index > (n_slice+1)*n*n))
            self.index = index
        field_lines = self.magline(p,n_lines=n_lines,radius=radius,frame=frame)
        fig = go.Figure(data=[go.Mesh3d(x=x_sample[indices], 
                                        y=y_sample[indices], 
                                        z=z_sample[indices],
                                        intensity=br[indices],
                                        colorscale='gray',
                                        colorbar=dict(title='Br'),
                                        opacity=0.8)])
        # fig.data=[]
        for fieldline in field_lines:
            flx_idx,fly_idx, flz_idx = np.floor(fieldline).astype(int).transpose()
            fig.add_trace(go.Scatter3d(x=x_sample[flx_idx,fly_idx,flz_idx],
                                       y=y_sample[flx_idx,fly_idx,flz_idx],
                                       z=z_sample[flx_idx,fly_idx,flz_idx],
                                       mode='lines',
                                       showlegend=False,
                                       line=dict(color='white',width=2)))
        fig.update_layout(width=800,height=800)
        fig.show()
              
    @classmethod
    def load(cls, load_name='spherical_data.pkl'):
        with open(load_name, 'rb') as input:
            return pickle.load(input)
    
    
    if __name__ == "__main__":
        # 创建一个示例对象
        data_instance = spherical_data()
        # 调用 save 方法，保存对象
        data_instance.save('saved_instances/spherical_data_instance.pkl')