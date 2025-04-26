import argparse
import json

from .need import *
from .yt_spherical_3D import aia_response, spherical_data

# ============================
parser = argparse.ArgumentParser(description='Radiation Systhesis with AMRVAC .dat files')
parser.add_argument('-f', type=str, help='file path for the AMRVAC .dat files', required=True)
parser.add_argument('-n', type=str, help='radiation normal direction', default= '1,0,0')
parser.add_argument('-s', type=str, help='save pics folder path', default= 'default')
parser.add_argument('-field', type=str, help='field', default= 'rho')
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
    norm             = parse_norm(args.n)
    save_path        = args.s
    field            = args.field
    zmin             = float(args.zmin) if args.zmin!='auto' else 'auto'
    zmax             = float(args.zmax) if args.zmax!='auto' else 'auto'
    begin            = args.begin
    end              = args.end

    xyz_file         = './x_y_z.npz'
    n_sph            = 400
    n_car            = 400
    fields_save_path = os.path.join(dat_path,'/yt_fields/')
    is_reload        = False
    reload_file      = None
    save_pkl         = True
    pkl_name         = 'spherical_data.pkl'
    fields_file_name = 'fields'
    auto_skip        = True
    fields_list      = ['rho','e']
else:
    with open(config, 'r') as json_file:
        config_data  = json.load(json_file)
    dat_path         = config_data['projection']['dat_path']
    field            = config_data['projection']['field']
    norm             = config_data['projection']['norm']
    save_path        = config_data['projection']['save_path']
    zmin             = config_data['projection']['zmin']
    zmax             = config_data['projection']['zmax']
    begin            = config_data['projection']['begin']
    end              = config_data['projection']['end']
    
    xyz_file         = config_data['projection']['xyz_file']
    n_sph            = config_data['projection']['n_sph']
    n_car            = config_data['projection']['n_car']
    fields_save_path = config_data['projection']['fields_save_path']
    is_reload        = config_data['projection']['is_reload']
    reload_file      = config_data['projection']['reload_file']
    save_pkl         = config_data['projection']['save_pkl']
    pkl_name         = config_data['projection']['pkl_name']
    fields_file_name = config_data['projection']['fields_file_name']
    auto_skip        = config_data['projection']['auto_skip']
    fields_list      = config_data['projection']['fields_list']
    dr               = config_data['projection']['dr']
    n_cut            = config_data['projection']['n_cut']
print('norm: ', norm)
save_path = os.path.join(dat_path,field) if save_path=='default' else save_path

# ===========================
if is_reload:
    instance = spherical_data.load(reload_file)
else:
    instance = spherical_data(dat_path, fields_save_path=fields_save_path, n_car=n_car,n_sph=n_sph, xyz_file=xyz_file)
    if save_pkl:
        instance.save(pkl_name)
end = len(instance.ts) if end<0 else min(end+1, len(instance.ts))
instance.save_fields(dr=dr, fields_list=fields_list, auto_skip=auto_skip, n_cut=n_cut, begin=begin, end=end)
if auto_skip:
    prob_file = os.path.join(instance.fields_save_path,instance.fields_file_name+str(begin).zfill(4)+'.npz')
    test      = np.load(prob_file)
    check     = set(fields_list)-set(test.files)
    if check:
        print(f'The existed .npz missing the fields: {check}')
        instance.add_field(fields_list, frame='all', dr=dr, n_cut=n_cut)

print('# ====================================== #')
print('#               Projection               #')
print('# ====================================== #')
if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)            
if not os.exists(img_path):
    os.makedirs(img_path, exist_ok=True)
for iframe in range(begin, end):
    instance.projection(frame=iframe, field=field, norm=norm, save_path=save_path)
    plt.tight_layout()
    plt.close()

print('!!! Process Finished !!!')
