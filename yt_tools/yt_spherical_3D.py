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
        self.memory            = None

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

    def save_fields(self, fields_list=['rho','e','m1','m2','m3','b1','b2','b3'], fields_file_name='fields', auto_skip=False, dr=0.0,begin=0, end=-1):
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
        first = True
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
            if first:
                time0 = time.time()
                indexs, outside = self.get_index(x_sample, y_sample, z_sample, n_drop)
                print('Get indexs time: %6.3f sec' % (time.time()-time0))
                np.savez(os.path.join(self.folder,'indexs.npz'), **dict(indexs=indexs,outside=outside))
                first=False
            save_dict['indexs']  = indexs
            save_dict['outside'] = outside
            for field in fields_list:
                time0 = time.time()
                f = field
                val_pre = np.array(region[('amrvac', f)])
                val_1d  = val_pre.ravel()
                save_dict[field], index = self.sample_field(val_1d, f, x_sample, y_sample, z_sample, 
                                                            dr=dr, indexs=indexs, outside=outside)
                save_dict['index'] = index
                indices = np.where((index < (self.n_drop+1)*n*n) & (index >= self.n_drop*n*n))
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

    def return_ds(self, frame=0, fields_path='default',n_cut=0):
        self.n_cut=n_cut
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
        keys = list(set(field.files)-{'indexs','outside','index'})
        shp = field[keys[0]].shape
        index = field['index']
        n = self.n_sph
        for f in keys:
            fval          = field[f]
            indices       = np.where((index<(n_cut+1+self.n_drop)*n*n))
            fval[indices] = 0
            new_dict[f]   = np.nan_to_num(fval, nan=0)
        ds = yt.load_uniform_grid(new_dict, shp, bbox=self.bbox_cart, length_unit = 'Mm')
        return ds

    def add_radiation(self, iwave='171', fields_path='default', frame=0, n_cut=0):
        unit = (1.4e9)**2
        ds = self.return_ds(frame=frame, fields_path=fields_path,n_cut=n_cut)
        ds.add_field(('gas', 'sdoaia'+iwave), 
                     function=lambda field,data: aia_response().response(np.nan_to_num(np.log10(data['e']/data['rho']*0.66*1e6)), iwave)*data['rho']**2*unit, 
                     sampling_type='cell', force_override=True)
        return ds

    def radiation_synthesis(self, norm=[1,0,0], iwave='171', fields_path='default', frame=0, save_path=None, **kwargs):
        wavelength = int(iwave) * u.angstrom
        aia_cmap = ct.aia_color_table(wavelength)
        n_cut    = kwargs.get('n_cut',0)
        ds = self.add_radiation(iwave=iwave, fields_path=fields_path, frame=frame,n_cut=n_cut)
        center = 'c'
        width = ds.quan(1, 'unitary')
        norm = np.array(norm)
        norm = norm/np.linalg.norm(norm)
        field = ('gas', f'sdoaia{iwave}')
        zmin   = kwargs.get('zmin', None)
        zmax   = kwargs.get('zmax', None)
        north  = kwargs.get('nort_vector', [0,0,1])
        xlabel = kwargs.get('xlabel', '')
        ylabel = kwargs.get('ylabel', '')
        clabel = kwargs.get('clabel', 'AIA SDO %s$\mathrm{\AA}~[\mathrm{DN/s}]$' % iwave)
        p = yt.OffAxisProjectionPlot(ds, norm, ('gas', 'sdoaia'+iwave), center=center, width=width)
        p.set_cmap(('gas','sdoaia'+iwave), aia_cmap)
        p.set_xlabel(xlabel)
        p.set_ylabel(ylabel)
        p.set_colorbar_label(('gas', 'sdoaia'+iwave), clabel)
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

    def projection(self, field='rho', norm=[1,0,0], fields_path='default', frame=0, save_path=None,ds=None, **kwargs):
        time0     = time.time()
        ds        = self.return_ds(frame=frame) if ds==None else ds
        center    = 'c'
        width     = ds.quan(1, 'unitary')
        norm      = np.array(norm)
        norm      = norm/np.linalg.norm(norm)
        zmin      = kwargs.get('zmin', None)
        zmax      = kwargs.get('zmax', None)
        north     = kwargs.get('nort_vector', [0,0,1])
        cmap      = kwargs.get('cmap', None)
        xlabel    = kwargs.get('xlabel', '')
        ylabel    = kwargs.get('ylabel', '')
        clabel    = kwargs.get('clabel', f'{field}')
        set_cb    = kwargs.get('set_colorbar', True)
        show      = kwargs.get('img_show', True)
        plot_norm = kwargs.get('plot_norm', None)
        p = yt.OffAxisProjectionPlot(ds, norm, field, center=center, width=width, north_vector=north)
        if cmap!=None:
            p.set_cmap(field, cmap)
        p.set_xlabel(xlabel)
        p.set_ylabel(ylabel)
        p.set_colorbar_label(field, clabel)
        if not set_cb:
            p.hide_colorbar()
        if plot_norm is not None:
            p.set_norm(field, plot_norm)
        if zmin!=None and zmax!=None:
            p.set_zlim(field, zmin, zmax)

        self.fig = p.plots[field].figure
        ax = self.fig.axes[0]
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        plt.tight_layout()
        self.p = p
        if save_path != None:
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            p.save(os.path.join(save_path, f'projection_{field}_{frame:02}.png'))
            print('Time spent: %6.3f sec' % (time.time()-time0))
        if is_ipython_environment() and show:
            p.show()
            
    def rad_review(self,iwave='171', frame=0, fields_path='default'):
        ds = self.add_radiation(iwave=iwave, fields_path=fields_path, frame=frame)
        rc = yt_idv.render_context()
        sc=rc.add_scene(ds, ('gas', 'sdoaia'+iwave))
        rc.run()       
        print("Camera's Position: ", sc.camera.position)
        print("Camera's Focus   : ", sc.camera.focus)
        print("Normal verctor   : ", sc.camera.position-sc.camera.focus)

    def rad_vtk(self, iwave='171', start_frame=0, end_frame='auto', fields_path='default' ,sight='top', save_path='vtk', auto_skip=True, **kwargs):
        end_frame   = min(end_frame, len(self.ts)) if end_frame!='auto' else len(self.ts)
        n           = self.n_car
        field       = ('gas', 'sdoaia'+iwave)
        xlb,ylb,zlb = self.bbox_cart[:,0]
        xub,yub,zub = self.bbox_cart[:,1]
        dx          = (self.bbox_cart[0][1]-self.bbox_cart[0][0])/n
        dy          = (self.bbox_cart[1][1]-self.bbox_cart[1][0])/n
        dz          = (self.bbox_cart[2][1]-self.bbox_cart[2][0])/n
        n_cut       = kwargs.get('n_cut', self.n_cut)
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
            print(f'create folder: {save_path}')
        for iframe in range(start_frame, end_frame):
            t_start   = time.time()
            vtk_name  = os.path.join(save_path,f'sdoaia{iwave}_{sight}_{iframe:02}.vtk')
            if os.path.exists(vtk_name):
                print(f'File {vtk_name} already exists. Skipping...')
                continue
            ds        = self.add_radiation(iwave=iwave, fields_path=fields_path, frame=iframe, n_cut=n_cut)
            region    = ds.r[::n*1j,::n*1j,::n*1j]
            rad       = np.nan_to_num(np.array(region[field]))
            imageData = vtk.vtkImageData()
            nx,nz,ny  = rad.shape
            L0        = 6.95e10
            if sight=='top':
                emission = rad.sum(axis=0).transpose()*dx*L0
                imageData.SetDimensions(1,ny,nz)
                imageData.SetSpacing(dx,dy,dz)
                imageData.SetOrigin(xlb-1e-3,ylb,zlb)
            elif sight=='right':
                emission = rad.sum(axis=1)*dy*L0
                imageData.SetDimensions(nx,1,nz)
                imageData.SetSpacing(dx,dy,dz)
                imageData.SetOrigin(xlb,ylb-1.e-3,zlb)
            elif sight=='left':
                emission = rad.sum(axis=1)*dy*L0
                imageData.SetDimensions(nx,1,nz)
                imageData.SetSpacing(dx,dy,dz)
                imageData.SetOrigin(xlb,yub+1.e-3,zlb)
            elif sight=='up':
                emission = rad.sum(axis=2).transpose()*dz*L0
                imageData.SetDimensions(nx,ny,1)
                imageData.SetSpacing(dx,dy,dz)
                imageData.SetOrigin(xlb,ylb,zlb-1.e-3)
            elif sight=='down':
                emission = rad.sum(axis=2).transpose()*dz*L0
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

    def add_field(self, f_names,  frame='all', dr=0.0, n_cut=0, f_vals=None):
        if not isinstance(f_vals, list):
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
                    # temp    = np.load(os.path.join(self.folder,'indexs.npz'))
                    # indexs  = temp['indexs']
                    # outside = temp['outside']
                    indexs  = new_dict['indexs']
                    outside = new_dict['outside']
                    for field in check:
                        time0 = time.time()
                        f = field
                        val_pre = np.array(region[('amrvac', f)])
                        val_1d  = val_pre.ravel()
                        new_dict[field], index = self.sample_field(val_1d, f, x_sample, y_sample, z_sample, 
                                                                   dr=dr, indexs=indexs, outside=outside)
                        indices = np.where((index < (self.n_drop+1)*n*n) & (index > (self.n_drop)*n*n))
                        new_dict[field][indices] = 0
                        print('process: %04d/%04d,  field: %5s, time: %6.3f sec' % (idx+1, len(self.ts), f, time.time()-time0))
                    np.savez(field_file, **new_dict)
                    print(f"!!!    File {field_file} is recreated     !!!")
        elif len(f_names)==len(f_vals):
            begin = 0            if frame=='all' else frame
            end   = len(self.ts) if frame=='all' else frame+1
            for idx in range(begin, end):
                field_file = os.path.join(self.fields_save_path, self.fields_file_name+str(idx).zfill(4)+'.npz')
                old_dict = np.load(field_file)
                new_dict = {}
                for f in old_dict.files:
                    new_dict[f] = old_dict[f]
                for f, v in zip(f_names, f_vals):
                    new_dict[f] = v
                np.savez(field_file, **new_dict)
                print(f'### Frame: {idx},  Add {f_names} in fields. ###')
        else:
            raise SystemError(f"There is no fields: {f_names} in .dat files \nor \nthe numbers of the given f_vals not match that of the f_names")

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
        
    def xyz2idx(self, p):
        x0,y0,z0     = p
        bbox_cart    = self.bbox_cart
        dxyz         = bbox_cart[:,1]-bbox_cart[:,0]
        n            = self.n_car
        x0_idx       = (x0-bbox_cart[0][0])/(bbox_cart[0][1]-bbox_cart[0][0])*n
        y0_idx       = (y0-bbox_cart[1][0])/(bbox_cart[1][1]-bbox_cart[1][0])*n
        z0_idx       = (z0-bbox_cart[2][0])/(bbox_cart[2][1]-bbox_cart[2][0])*n
        return np.array([x0_idx,y0_idx,z0_idx])
    
    def idx2xyz(self, xyz_idx):
        bbox_cart = self.bbox_cart
        dxyz      = bbox_cart[:,1]-bbox_cart[:,0]
        n         = self.n_car
        xyz       = xyz_idx/n*dxyz+bbox_cart[:,0]
        return xyz
    
    def sample_with_idx(self, idx, field, frame=0):
        xi,yi,zi    = np.floor(idx).astype(int).T
        xd,yd,zd    = (idx-np.floor(idx)).T
        fields_file = os.path.join(self.fields_save_path,self.fields_file_name+str(frame).zfill(4)+'.npz')
        fields      = np.load(fields_file)
        f           = fields[field]
        ret         = f[xi,yi,zi]*(1-xd)*(1-yd)*(1-zd)+f[xi+1,yi,zi]*xd*(1-yd)*(1-zd)+f[xi,yi+1,zi]*(1-xd)*yd*(1-zd)+\
                      f[xi,yi,zi+1]*(1-xd)*(1-yd)*zd+f[xi+1,yi+1,zi]*xd*yd*(1-zd)+f[xi+1,yi,zi+1]*xd*(1-yd)*zd+\
                      f[xi,yi+1,zi+1]*(1-xd)*yd*zd+f[xi+1,yi+1,zi+1]*xd*yd*zd
        return ret

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
        bbox_cart    = self.bbox_cart
        bbox         = np.array(self.bbox)
        dxyz         = bbox_cart[:,1]-bbox_cart[:,0]
        n            = self.n_car
        s_unit       = dxyz.min()/n
        n_dr         = np.ceil(radius/s_unit)
        x0_idx,y0_idx,z0_idx=self.xyz2idx(p)
        start_points = sphere_sample([x0_idx, y0_idx, z0_idx], r=n_dr, nsample=n_lines)
        if n_lines==1:
            start_points = [start_points]
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
        slice_cmap      = kwargs.get('slice_cmap', 'Viridis')
        point_size      = kwargs.get('point_size', 10)
        magline_width   = kwargs.get('magline_width', 2)
        magline_color   = kwargs.get('magline_color', 'white')
        camera          = kwargs.get('camera', None)
        save_path       = kwargs.get('save_path', None)
        is_show         = kwargs.get('is_show', True)
        remember        = kwargs.get('remember', True)
        is_multi_points = (len(np.shape(p))>1)
        ps = p if is_multi_points else [p]
        self.add_field(f_names=['b1','b2','b3'],frame=frame,n_cut=self.n_cut)
        bbox         = np.array(self.bbox)
        # fields_file  = os.path.join(self.fields_save_path,self.fields_file_name+str(frame).zfill(4)+'.npz')
        # data         = np.load(fields_file)
        # br           = data['b1']
        r_slice      = kwargs.get('r_slice',bbox[0][0])
        n_slice      = np.ceil((r_slice-bbox[0][0])/(bbox[0][1]-bbox[0][0])*self.n_sph).astype(int)
        n            = self.n_sph
        deg = np.pi/180
        r = np.array([r_slice])
        t = np.linspace(bbox[1][0],bbox[1][1], n)*deg
        p = np.linspace(bbox[2][0],bbox[2][1], n)*deg
        R,T,P = np.meshgrid(r,t,p, indexing='ij')
        Z = R*np.sin(T)
        X = R*np.cos(T)*np.cos(P)
        Y = R*np.cos(T)*np.sin(P)
        if self.memory is not None:
            if remember and (self.memory['n_slice']==n_slice):
                br = self.memory['br']
        if (not remember) or (self.memory is None):
            ds = self.ts[frame]
            region = ds.r[::n*1j,::n*1j,::n*1j]
            br = np.array(region[('amrvac','b1')])[n_slice,:,:]
            if remember:
                self.memory = {}
                self.memory['n_slice'] = n_slice
                self.memory['br']      = br
        fig = go.Figure(data=[go.Scatter3d(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            mode='markers',
            marker=dict(
                size=point_size,  # Adjust size to your liking
                color=br[::-1].flatten(),  # Color of markers
                colorscale=slice_cmap,  # This is a visual preference
                colorbar=dict(title='Br'),
                opacity=0.8
            ),
            showlegend=False
        )])
        dx,dy,dz = (self.bbox_cart[:,1]-self.bbox_cart[:,0])/self.n_car
        x_sample, y_sample, z_sample, bbox_cart, bbox = self.load_xyz()
        for p in ps:
            field_lines = self.magline(p,n_lines=n_lines,radius=radius,frame=frame)
            for fieldline in field_lines:
                flx_idx,fly_idx, flz_idx = np.floor(fieldline).astype(int).transpose()
                dxx,dyy,dzz = ((fieldline-np.stack([flx_idx,fly_idx,flz_idx], axis=1))*np.array([dx,dy,dz])).T
                xx = x_sample[flx_idx,fly_idx,flz_idx]+dxx
                yy = y_sample[flx_idx,fly_idx,flz_idx]+dyy
                zz = z_sample[flx_idx,fly_idx,flz_idx]+dzz
                fig.add_trace(go.Scatter3d(x=xx,
                                           y=yy,
                                           z=zz,
                                           mode='lines',
                                           showlegend=False,
                                           line=dict(color=magline_color,width=magline_width)))
        fig.update_layout(width=800,height=800)
        if camera is not None:
            fig.update_layout(
                scene=dict(camera=camera)
            )
        if save_path is not None:
            fig.write_image(save_path)
        self.fig = fig
        if is_show:
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
