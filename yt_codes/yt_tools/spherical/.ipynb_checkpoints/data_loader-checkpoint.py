from ..need import *
from .. import geometry
from ..funcs import trilinear_interpolation
from ..geometry import xyz2rtp, rtp2xyz, Vrtp2Vxyz
from ..stream_line import Spherical_magline

class spherical_data():
    def __init__(self, folder='./data/', sample_level=1, **kwargs):
        start_time             = time.time()
        self.folder            = folder
        self.ts                = yt.load(os.path.join(folder,'*.dat'))
        self.sample_level      = sample_level
        ds                     = self.ts[0]
        self.domain_left_edge  = ds.domain_left_edge
        self.domain_right_edge = ds.domain_right_edge
        self.bbox              = np.array([ds.domain_left_edge, ds.domain_right_edge]).T
        self.dimensions        = ds.domain_dimensions
        self.Nrtp              = self.dimensions*sample_level
        self.drtp              = (self.bbox.T[1]-self.bbox.T[0])/(self.Nrtp-1)
        end_time               = time.time()
        self.fig               = None
        self.save_name         = kwargs.get('save_name', 'spherical_data.pkl')
        self.info              = dict()
        print(f"Initialization completed in {end_time - start_time} seconds")

    def __getstate__(self):
        state = self.__dict__.copy()
        # 移除无法序列化的属性
        del state['ts']
        del state['fig']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # 重新加载或初始化 'ts' 属性
        self.ts      = yt.load(os.path.join(self.folder, '*.dat'))
        self.fig     = None

    def save(self, save_name='spherical_data.pkl'):
        self.save_name = save_name
        directory = os.path.dirname(save_name)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        with open(save_name, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        print(f"Instance saved to {save_name}")

    @classmethod
    def load(cls, load_name='spherical_data.pkl'):
        with open(load_name, 'rb') as input:
            return pickle.load(input)

    def _drtp(self, **kwargs):
        sample_level = kwargs.get('sample_level', self.sample_level)
        nrtp         = self.dimensions*sample_level
        Drtp         = self.bbox[:,1]-self.bbox[:,0]
        drtp         = Drtp/(nrtp-1)
        return drtp

    def _recordB(self, **kwargs):
        frame = kwargs.get('frame', 0)
        B     = self.uniform_sample(['b1','b2','b3'], **kwargs)
        Brtp  = np.stack([B['b1'],B['b2'],B['b3']])
        self.info[f'frame_{frame:04d}']=dict(Brtp=Brtp)

    def uniform_sample(self, field, **kwargs):
        frame        = kwargs.get('frame', 0)
        sample_level = kwargs.get('sample_level', self.sample_level)
        ds           = self.ts[frame]
        if sample_level==None:
            sample_level = self.sample_level
        else:
            self.sample_level = sample_level
        nx,ny,nz          = self.dimensions*sample_level
        region            = ds.r[::nx*1j,::ny*1j,::nz*1j]
        ret_dict          = {}
        if isinstance(field, list):
            for f in field:
                ret_dict[f]=np.array(region[f])
        else:
            ret_dict[field]=np.array(region[field])
        return  ret_dict

    def get_rtp(self, **kwargs):
        sample_level = kwargs.get('sample_level', self.sample_level)
        bbox         = self.bbox
        Nr,Nt,Np     = self.dimensions*sample_level
        rlist        = np.linspace(*bbox[0], Nr)
        tlist        = np.linspace(*bbox[1], Nt)
        plist        = np.linspace(*bbox[2], Np)
        R,T,P        = np.meshgrid(rlist, tlist, plist, indexing='ij')
        return np.stack([R,T,P])

    def rtp2idx(self, rtp, **kwargs):
        sample_level = kwargs.get('sample_level', self.sample_level)
        dr,dt,dp     = self._drtp(**kwargs)
        rm,tm,pm     = self.bbox.T[0]
        r,t,p        = rtp
        ir           = (r-rm)/dr
        it           = (t-tm)/dt
        ip           = (p-pm)/dp
        idx          = np.stack([ir,it,ip], axis=0)
        return idx
        
    def get_Brtp(self, rtp=None, xyz=None, **kwargs):
        frame=kwargs.get('frame',0)
        Brtp = self.info.get(f'frame_{frame:04d}', dict()).get('Brtp', None)
        if Brtp is None:
            self._recordB(**kwargs)
            Brtp=self.info[f'frame_{frame:04d}'].get('Brtp', None)
        Brtp = Brtp.transpose(1,2,3,0)
        if rtp is not None:
            idx  = self.rtp2idx(rtp, **kwargs)
            ret  = trilinear_interpolation(Brtp ,idx)
        elif xyz is not None:
            rtp  = xyz2rtp(xyz)
            idx  = self.rtp2idx(rtp, **kwargs)
            ret  = trilinear_interpolation(Brtp, idx)
        else:
            ret  = Brtp.transpose(3,0,1,2)
        return ret

    def sample_with_idx(self, fields, idx, **kwargs):
        if isinstance(fields, list):
            fval = []
            fdic = self.uniform_sample(fields, **kwargs)
            for f in fields:
                fval.append(fdic[f])
            fval = np.stack(fval, axis=-1)
            ret  = trilinear_interpolation(fval, idx)
        else:
            fval = self.uniform_sample(fields, **kwargs)[fields]
            ret  = trilinear_interpolation(fval, idx)
        return ret

    def sample_with_rtp(self, fields, rtp, **kwargs):
        idx = self.rtp2idx(rtp, **kwargs)
        ret = self.sample_with_idx(fields, idx, **kwargs)
        return ret

    def parallel_qsl(self, points, **kwargs):
        dl    = kwargs.get('step_length', self.drtp[0]*4)
        Ns    = kwargs.get('max_steps', 100000)
        frame = kwargs.get('frame', 0)
        Nc    = kwargs.get('n_cores', 50)
        PI    = kwargs.get('print_interval', 10000)
        PY    = kwargs.get('python', 'python')
        SN    = kwargs.get('save_name', 'sph_temp.pkl')
        Brtp  = self.get_Brtp(**kwargs)
        self.info[f'frame_{frame:04d}']['cal_qsl_setting']=dict(
            points=points,
            n_cores=Nc,
            print_interval=PI,
            max_steps=Ns,
            step_length=dl,
        )
        self.save(SN)
        command = PY+f' -u -m yt_tools.scripts.spherical_qsl -i {SN} -f {frame}'
        tem_ret = os.system(command)
        if tem_ret != 0:
            raise SystemError('Scripts execution error...')
        else:
            ret = np.load('spherical_qsl.npy')
            os.remove(SN)
            os.remove('spherical_qsl.npy')
            return ret

    def get_magline(self, rtp, **kwargs):
        Brtp = self.get_Brtp()
        rtp  = self.get_rtp()
        sm   = Spherical_magline(Brtp, rtp, **kwargs)
        ret  = sm.magline_solver(rtp, **kwargs)
        return ret