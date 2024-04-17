from yt_tools.need import *
from yt_tools.yt_spherical_3D import aia_response, spherical_data

# ============================
parser = argparse.ArgumentParser(description='Radiation Systhesis with AMRVAC .dat files')
parser.add_argument('-f', type=str, help='file path for the AMRVAC .dat files', required=True)
parser.add_argument('-w', type=str, help='waveband', default= None)
parser.add_argument('-field', type=str, help='field', default= None)
parser.add_argument('-frame', type=int, help='frame', default= 0)
parser.add_argument('-zmin', type=str, help='zmin of the radiation figure', default= 'auto')
parser.add_argument('-zmax', type=str, help='zmax of the radiation figure', default= 'auto')
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
    fields_save_path = os.path.join(dat_path,'/yt_fields/')
    save_pkl         = False
    pkl_name         = 'spherical_data.pkl'
    fields_file_name = 'fields'
    auto_skip        = True
    dr               = 0.0
    n_cut            = 0

# ===========================
if reload_file!=None:
    instance = spherical_data.load(reload_file)
else:
    instance = spherical_data(dat_path, fields_save_path=fields_save_path, n_car=n_car,n_sph=n_sph, xyz_file=xyz_file)
    if save_pkl:
        instance.save(pkl_name)
if iwave==None:
    fields_list = [field]
    instance.save_fields(dr=dr, fields_list=fields_list, auto_skip=auto_skip, n_cut=n_cut, begin=frame, end=frame)
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