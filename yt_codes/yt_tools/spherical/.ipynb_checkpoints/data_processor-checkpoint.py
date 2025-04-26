import torch
from ..need import *
from .. import geometry
from ..funcs import trilinear_interpolation,rot
from ..geometry import xyz2rtp, rtp2xyz, Vrtp2Vxyz
from ..stream_line import Spherical_magline

class spherical_data():
    def __init__(self, grid, **kwargs):
        self.grid   = grid
        self.Nrtp   = np.array([len(g) for g in grid])
        self.bbox   = np.array([[g.min(),g.max()] for g in grid])
        self.drtp   = np.array([g[1]-g[0] for g in grid])
        self.fnames = kwargs.get('fnames', None)
        fields      = kwargs.get('fields', None)
        self.save_name = kwargs.get('save_name', 'spherical_data.pkl')
        self.info      = dict()
        if self.fnames is not None:
            self.set_fields(self.fnames, fields)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['fig']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
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

    def set_fields(self,fnames, fvals, **kwargs):
        if isinstance(fnames, list):
            for fname,fval in zip(fnames,fvals):
                self.info[fname]=fval
        else:
            self.info[fnames]=fvals

    def get_rtp(self, **kwargs):
        r,t,p = self.grid
        R,T,P = np.meshgrid(r,t,p,indexing='ij')
        rtp   = np.stack([R,T,P], axis=0)
        return rtp

    def get_Brtp(self, **kwargs):
        Br = self.info.get('Br', None)
        Bt = self.info.get('Bt', None)
        Bp = self.info.get('Bp', None)
        Brtp = np.stack([Br,Bt,Bp], axis=0)
        return Brtp

    def rtp2idx(self, rtp, **kwargs):
        dr,dt,dp = self.drtp
        rm,tm,pm = self.bbox.T[0]
        r,t,p    = rtp
        ir       = (r-rm)/dr
        it       = (t-tm)/dt
        ip       = (p-pm)/dp
        idx      = np.stack([ir,it,ip], axis=0)
        return idx

    def sample_with_idx(self, fields, idx, **kwargs):
        fields = np.array(fields)
        ret    = trilinear_interpolation(fields, idx)
        return ret

    def sample_with_rtp(self, fields, rtp, **kwargs):
        idx = self.rtp2idx(rtp, **kwargs)
        ret = self.sample_with_idx(fields, idx, **kwargs)
        return ret

    def parallel_qsl(self, Brtp, points, **kwargs):
        dl    = kwargs.get('step_length', self.drtp[0]*4)
        Ns    = kwargs.get('max_steps', 100000)
        frame = kwargs.get('frame', 0)
        Nc    = kwargs.get('n_cores', 50)
        PI    = kwargs.get('print_interval', 10000)
        PY    = kwargs.get('python', 'python')
        SN    = kwargs.get('save_name', 'sph_temp.pkl')
        Brtp  = np.array(Brtp)
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
        Brtp = self.get_Brtp().transpose(1,2,3,0)
        RTP  = self.get_rtp().transpose(1,2,3,0)
        sm   = Spherical_magline(Brtp, RTP, **kwargs)
        ret  = sm.magline_solver(rtp, **kwargs)
        return ret

class Proxy_Emissivity(spherical_data):
    def __init__(self,grid,Brtp,**kwargs):
        super(Proxy_Emissivity, self).__init__(grid,**kwargs)
        Br,Bt,Bp = Brtp
        fnames = ['Br','Bt','Bp']
        fields = [Br,Bt,Bp]
        self.set_fields(fnames,fields)

    def __call__(self, **kwargs):
        rtp      = self.get_rtp()
        Brtp     = self.get_Brtp()
        rb,tb,pb = rtp[:,0,:,:]
        points   = np.stack([rb.ravel(),tb.ravel(),pb.ravel()],axis=-1)
        options  = ['n_cores','max_steps','step_length','n_print','save_name','device']
        info     = dict(rtp=rtp,Brtp=Brtp,points=points)
        for option in options:
            if option in kwargs.keys():
                info[option]=kwargs.get(option)
        info_name = kwargs.get('info_name', './proxy_emissivity_info.npz')
        np.savez(info_name, **info)
        PY = kwargs.get('python', 'python')
        command = PY+f' -u -m yt_tools.scripts.SphericalProxyEmissivity -i {info_name}'
        ret = os.system(command)
        SN  = kwargs.get('save_name', './proxy_emissivity_temp.npy')
        PE  = np.load(SN)
        os.remove(SN)
        os.remove(info_name)
        return PE