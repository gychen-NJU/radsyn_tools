import argparse
import json

from .need import *
from .yt_spherical_3D import aia_response, spherical_data

# ============================
parser = argparse.ArgumentParser(description='Radiation Systhesis with AMRVAC .dat files')
parser.add_argument('-f', type=str, help='file path for the AMRVAC .dat files', required=True)
parser.add_argument('-frame', type=int, help='frame for visualizing', required=False, default=0)
parser.add_argument('-p', type=str, help='frame for visualizing', required=False, default=0)
parser.add_argument('-i', type=str, help='reload instance', required=False, default=None)
parser.add_argument('-s', type=str, help='radius slice position', required=False, default='1.02')
parser.add_argument('--config', type=str, help='configuration file path', required=False, default=None)

args = parser.parse_args()
def parse_norm(norm_str):
    return [float(x) for x in norm_str.split(',')]

config   = args.config

if config==None:
    dat_path         = args.f
    p                = parse_norm(args.p)
    frame            = args.frame
    reload_file      = args.i
    r_slice          = float(args.s)

    xyz_file         = './x_y_z.npz'
    n_sph            = 400
    n_car            = 400
    fields_save_path = os.path.join(dat_path,'/yt_fields/')
    save_pkl         = False
    pkl_name         = 'spherical_data.pkl'
    fields_file_name = 'fields'
    auto_skip        = True
    radius           = 0.02
    n_lines          = 10
    dr               = 0.0
    n_cut            = 0
else:
    with open(config, 'r') as json_file:
        config_data  = json.load(json_file)
    dat_path         = config_data['magline']['dat_path']
    p                = config_data['magline']['p']
    frame            = config_data['magline']['frame']
    reload_file      = config_data['magline']['reload_file']
    r_slice          = config_data['magline']['r_slice']

    xyz_file         = config_data['magline']['xyz_file']
    n_sph            = config_data['magline']['n_sph']
    n_car            = config_data['magline']['n_car']
    fields_save_path = config_data['magline']['fields_save_path']
    save_pkl         = config_data['magline']['save_pkl']
    pkl_name         = config_data['magline']['pkl_name']
    fields_file_name = config_data['magline']['fields_file_name']
    auto_skip        = config_data['magline']['auto_skip']
    radius           = config_data['magline']['radius']
    n_lines          = config_data['magline']['n_lines']
    dr               = config_data['magline']['dr']
    n_cut            = config_data['magline']['n_cut']

# ===========================
fields_list = ['b1','b2','b3']
if reload_file!=None:
    instance = spherical_data.load(reload_file)
else:
    instance = spherical_data(dat_path, fields_save_path=fields_save_path, n_car=n_car,n_sph=n_sph, xyz_file=xyz_file)
    if save_pkl:
        instance.save(pkl_name)
        
if instance.fields_file_name==None:
    instance.save_fields(dr=dr, fields_list=fields_list, auto_skip=auto_skip, n_cut=n_cut, begin=frame, end=frame)
prob_file = os.path.join(instance.fields_save_path,instance.fields_file_name+str(frame).zfill(4)+'.npz')
if os.path.exists(prob_file):
    test  = np.load(prob_file)
    check = {'b1','b2','b3'}-set(test.files)
    if check:
        print(f'The existed .npz missing the fields: {check}')
        instance.add_field(fields_list, frame='all', dr=dr, n_cut=n_cut)

print('# ====================================== #')
print('#           Show magnetic line           #')
print('# ====================================== #')
x0,y0,z0 = p
instance.show_magline([x0,y0,z0], radius=radius, frame=frame, n_lines=n_lines, r_slice=1.05)

print('!!! Process Finished !!!')
