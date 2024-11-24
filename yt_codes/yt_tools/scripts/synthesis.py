import argparse
import json

from ..need import *
from ..yt_spherical_3D import aia_response, spherical_data

# ============================
parser = argparse.ArgumentParser(description='Radiation Systhesis with AMRVAC .dat files')
parser.add_argument('-f', type=str, help='file path for the AMRVAC .dat files', required=False,default='./data')
parser.add_argument('-w', type=str, help='waveband', default= '171')
parser.add_argument('-n', type=str, help='radiation normal direction', default= '1,0,0')
parser.add_argument('-s', type=str, help='save pics folder path', default= 'default')
parser.add_argument('-zmin', type=str, help='zmin of the radiation figure', default= 'auto')
parser.add_argument('-zmax', type=str, help='zmax of the radiation figure', default= 'auto')
parser.add_argument('-begin', type=int, help='begin frame', default= 0)
parser.add_argument('-end', type=int, help='end frame', default=-1)
parser.add_argument('--config', type=str, help='configuration file path', required=False, default=None)

args = parser.parse_args()
def parse_norm(norm_str):
    return [float(x) for x in norm_str.split(',')]

config   = args.config

if config==None:
    dat_path         = args.f
    iwave            = args.w
    norm             = parse_norm(args.n)
    rad_path         = args.s
    zmin             = float(args.zmin) if args.zmin!='auto' else 'auto'
    zmax             = float(args.zmax) if args.zmax!='auto' else 'auto'
    begin            = args.begin
    end              = args.end

    xyz_file         = './x_y_z.npz'
    n_sph            = 400
    n_car            = 400
    fields_save_path = os.path.join(dat_path,'yt_fields/')
    is_reload        = False
    reload_file      = None
    save_pkl         = True
    pkl_name         = 'spherical_data.pkl'
    fields_file_name = 'fields'
    auto_skip        = True
    fields_list      = ['rho','e']
    dr               = 0.0
    n_cut            = 0
else:
    with open(config, 'r') as json_file:
        config_data  = json.load(json_file)
    dat_path         = config_data['synthesis']['dat_path']
    iwave            = config_data['synthesis']['iwave']
    norm             = config_data['synthesis']['norm']
    rad_path         = config_data['synthesis']['rad_path']
    zmin             = config_data['synthesis']['zmin']
    zmax             = config_data['synthesis']['zmax']
    begin            = config_data['synthesis']['begin']
    end              = config_data['synthesis']['end']
    
    xyz_file         = config_data['synthesis']['xyz_file']
    n_sph            = config_data['synthesis']['n_sph']
    n_car            = config_data['synthesis']['n_car']
    fields_save_path = config_data['synthesis']['fields_save_path']
    is_reload        = config_data['synthesis']['is_reload']
    reload_file      = config_data['synthesis']['reload_file']
    save_pkl         = config_data['synthesis']['save_pkl']
    pkl_name         = config_data['synthesis']['pkl_name']
    fields_file_name = config_data['synthesis']['fields_file_name']
    auto_skip        = config_data['synthesis']['auto_skip']
    fields_list      = config_data['synthesis']['fields_list']
    dr               = config_data['synthesis']['dr']
    n_cut            = config_data['synthesis']['n_cut']
print('norm: ', norm)
rad_path = os.path.join(dat_path, 'sdoaia'+iwave) if rad_path=='default' else rad_path

# ===========================
if is_reload:
    instance = spherical_data.load(reload_file)
else:
    instance = spherical_data(dat_path, fields_save_path=fields_save_path, n_car=n_car,n_sph=n_sph, xyz_file=xyz_file)
end = len(instance.ts) if end<0 else min(end+1, len(instance.ts))
instance.save_fields(dr=dr, fields_list=fields_list, auto_skip=auto_skip, n_cut=n_cut, begin=begin, end=end)

x_sample, y_sample, z_sample, bbox_cart, bbox = instance.load_xyz()
n_frames = len(instance.ts)

print('# ====================================== #')
print('#           Radiation Synthesis          #')
print('# ====================================== #')
if not os.path.exists(rad_path):
    os.makedirs(rad_path, exist_ok=True)            
for iframe in range(n_frames):
    if iframe==0:
        zmin=zmin if zmin!='auto' else 'auto'
        zmax=zmax if zmax!='auto' else 'auto'
    instance.radiation_synthesis(norm=norm, iwave=iwave, fields_path='default', frame=iframe, zmin=zmin, zmax=zmax, save_path=rad_path)
    plt.tight_layout()
    plt.close()
if save_pkl:
    instance.save(pkl_name)

print('!!! Process Finished !!!')
