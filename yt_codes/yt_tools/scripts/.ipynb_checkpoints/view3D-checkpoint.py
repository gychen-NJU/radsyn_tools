import argparse
import json

from ..need import *
from ..yt_spherical_3D import aia_response, spherical_data

# ============================
parser = argparse.ArgumentParser(description='Radiation Systhesis with AMRVAC .dat files')
parser.add_argument('-f', type=str, help='file path for the AMRVAC .dat files', required=False, default='./data')
parser.add_argument('-w', type=str, help='waveband', default= None)
parser.add_argument('-field', type=str, help='field', default= None)
parser.add_argument('-frame', type=int, help='frame', default= 0)
parser.add_argument('-zmin', type=str, help='zmin of the radiation figure', default= 'auto')
parser.add_argument('-zmax', type=str, help='zmax of the radiation figure', default= 'auto')
parser.add_argument('--fields_path', type=str, help='fields dictory path',  default= './fields')
parser.add_argument('--config', type=str, help='configuration file path', required=False, default=None)

args = parser.parse_args()
config   = args.config

if config==None:
    dat_path         = args.f
    iwave            = args.w
    field            = args.field
    frame            = args.frame
    zmin             = float(args.zmin) if args.zmin!='auto' else 'auto'
    zmax             = float(args.zmax) if args.zmax!='auto' else 'auto'

    xyz_file         = './x_y_z.npz'
    n_sph            = 400
    n_car            = 400
    fields_save_path = os.path.join(dat_path,'fields/')
    save_pkl         = False
    pkl_name         = 'spherical_data.pkl'
    fields_file_name = 'fields'
    auto_skip        = True
    dr               = 0.0
    n_cut            = 0
    reload_file      = None
else:
    with open(config, 'r') as json_file:
        config_data  = json.load(json_file)
    dat_path         = config_data['view3D']['dat_path']
    iwave            = config_data['view3D']['iwave']
    field            = config_data['view3D']['field']
    frame            = config_data['view3D']['frame']
    zmin             = config_data['view3D']['zmin']
    zmin             = config_data['view3D']['zmax']

    xyz_file         = config_data['view3D']['xyz_file']
    reload_file      = config_data['view3D']['reload_file']
    n_sph            = config_data['view3D']['n_sph']
    n_car            = config_data['view3D']['n_car']
    fields_save_path = config_data['view3D']['fields_save_path']
    save_pkl         = config_data['view3D']['save_pkl']
    pkl_name         = config_data['view3D']['pkl_name']
    fields_file_name = config_data['view3D']['fields_file_name']
    auto_skip        = config_data['view3D']['auto_skip']
    dr               = config_data['view3D']['dr']
    n_cut            = config_data['view3D']['n_cut']

# ===========================
print(fields_save_path)
if reload_file is not None:
    instance = spherical_data.load(reload_file)
else:
    print('Create instance')
    instance = spherical_data(dat_path, fields_save_path=fields_save_path, n_car=n_car,n_sph=n_sph, xyz_file=xyz_file)

if iwave is None:
    recreate=False
    if os.path.exists(fields_save_path):
        load = np.load(sorted(glob.glob(os.path.join(fields_save_path,'*.npz')))[frame])
        if field in load.files:
            print(f"{field} already in fields file...")
            instance.fields_file_name='fields'
            ds = instance.return_ds(frame=frame)
        else:
            recreate=True
    else:
        recreate=True
    if recreate:
        print('Recreate the instance')
        fields_list = [field]
        instance.save_fields(dr=dr, fields_list=fields_list, auto_skip=auto_skip, begin=frame, end=frame)
        if auto_skip:
            instance.add_field(fields_list,frame=frame,dr=dr,n_cut=n_cut)
        ds = instance.return_ds(frame=frame)
else:
    fields_list = ['rho','e']
    instance.save_fields(dr=dr, fields_list=fields_list, auto_skip=auto_skip, n_cut=n_cut, begin=frame, end=frame)
    if auto_skip:
        instance.add_field(fields_list, frame=frame, dr=dr, n_cut=n_cut)
    ds = instance.add_radiation(frame=frame)
    field = ('gas','sdoaia'+iwave)
        
rc = yt_idv.render_context()
sc=rc.add_scene(ds, field)
rc.run()

if save_pkl:
    instance.save(pkl_name)