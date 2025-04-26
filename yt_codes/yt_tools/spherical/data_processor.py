import torch
from ..need import *
from .. import geometry
from ..funcs import trilinear_interpolation,rot
from ..geometry import xyz2rtp, rtp2xyz, Vrtp2Vxyz
from ..stream_line import Spherical_magline

class spherical_data():
    """
    A class for handling and processing data in spherical coordinates.

    Parameters
    ----------
    grid : list or tuple
        A list/tuple containing the r, theta, phi coordinate arrays that define the spherical grid
    kwargs : dict, optional
        Additional keyword arguments:
        - fnames : list, optional
            List of field names to initialize
        - fields : list, optional 
            List of field data arrays corresponding to fnames
        - save_name : str, optional
            Name of file to save the data (default: 'spherical_data.pkl')

    Attributes
    ----------
    grid : list
        The r, theta, phi coordinate arrays
    Nrtp : numpy.ndarray
        Number of points in each dimension [Nr, Ntheta, Nphi]
    bbox : numpy.ndarray 
        Bounding box min/max values for each dimension
    drtp : numpy.ndarray
        Grid spacing in each dimension
    fnames : list
        List of field names
    info : dict
        Dictionary storing the field data
    save_name : str
        Name of file to save the data
    fig : plotly.graph_objects.Figure
        Figure object for plotting

    Methods
    -------
    save()
        Save instance to file
    load()
        Load instance from file
    set_fields()
        Set field data
    get_rtp()
        Get meshgrid arrays of r, theta, phi coordinates
    """
    def __init__(self, grid, **kwargs):
        self.grid   = grid
        self.Nrtp   = np.array([len(g) for g in grid])
        self.bbox   = np.array([[g[0],g[-1]] for g in grid])
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
        r,t,p    = rtp.T if np.array(rtp).ndim==2 else rtp
        ir       = (r-rm)/dr
        it       = (t-tm)/dt
        ip       = (p-pm)/dp
        idx      = np.stack([ir,it,ip], axis=-1) if np.array(rtp).ndim==2 else np.stack([ir,it,ip], axis=0)
        return idx

    def sample_with_idx(self, fields, idx, **kwargs):
        fields = np.array(fields)
        # idx    = np.array(idx).T if idx.ndim==2 else idx
        ret    = trilinear_interpolation(fields, idx)
        return ret

    def sample_with_rtp(self, fields, rtp, **kwargs):
        # rtp = np.array(rtp).T if rtp.ndim==2 else rtp
        idx = self.rtp2idx(rtp, **kwargs)
        # idx = idx.T if idx.ndim==2 else idx
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

    def show_maglines(self, maglines=None, **kwargs):
        fig = kwargs.get('fig', go.Figure())
        show = kwargs.get('show', True)
        frame = kwargs.get('frame', True)
        c = kwargs.get('color', 'magfield')
        lw = kwargs.get('lw', 5)
        width = kwargs.get('width', 800)
        height = kwargs.get('height', 800)
        if maglines is None:
            nlines   = kwargs.get('nlines',100)
            seeds = np.stack([np.random.uniform(g.min(),g.max(),nlines) for g in self.grid])
            maglines = self.get_magline(seeds)
        maglines_xyz = []
        for imagline in maglines:
            ri,ti,pi = imagline.T
            xi = ri*np.sin(ti)*np.cos(pi)
            yi = ri*np.sin(ti)*np.sin(pi)
            zi = ri*np.cos(ti)
            maglines_xyz.append(np.stack([xi,yi,zi], axis=1))
        if c == 'magfield':
            vmin = kwargs.get('vmin', None)
            vmax = kwargs.get('vmax', None)
            if vmin is None or vmax is None:
                vmin = 1e10
                vmax = -1
                color_list = []
                for imagline in maglines:
                    brtp_maglines = self.sample_with_rtp(self.get_Brtp().transpose(1,2,3,0), imagline)
                    color = np.nan_to_num(np.linalg.norm(brtp_maglines, axis=-1))
                    color_list = color_list+list(np.log10(color+1e-10))
                c_mean = np.mean(color_list)
                c_std  = np.std(color_list)
                vmin   = c_mean-c_std*3
                vmax   = c_mean+c_std*3
        for i,imagline in enumerate(maglines_xyz):
            if c == 'magfield':
                brtp_maglines = self.sample_with_rtp(self.get_Brtp().transpose(1,2,3,0), maglines[i])
                color = np.nan_to_num(np.linalg.norm(brtp_maglines, axis=-1))
            xx,yy,zz = imagline.T
            fig.add_trace(go.Scatter3d(x=xx,
                                    y=yy,
                                    z=zz,
                                    mode='lines',
                                    showlegend=True if i==0 else False,
                                    name='Magnetic Field Lines',
                                    line=dict(
                                        color=np.log10(color+1e-10),
                                        width=lw,
                                        colorscale=kwargs.get('colorscale', 'jet'),
                                        showscale=True if i==0 else False,
                                        cmin=vmin,
                                        cmax=vmax,
                                        colorbar=dict(
                                            title=dict(
                                                text='log10(B)',
                                                side='right'
                                            ),
                                            thickness=20,
                                            len=0.75
                                        )
                                    ) if c == 'magfield' else dict(color=c,width=lw)
                                    ))
        # Add a semi-transparent unit sphere
        u = np.linspace(0, 2*np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))        
        fig.add_surface(
            x=x, y=y, z=z,
            opacity=0.5,
            showscale=False,
            surfacecolor=np.ones_like(x),
            colorscale=[[0, 'rgb(200,200,200)'], [1, 'rgb(200,200,200)']],
            showlegend=False
        )
        if not frame:
            fig = unframe(fig)
        if show:
            fig.update_layout(width=width,
                            height=height,
                            scene=dict(
                                aspectmode='data'
                            ))
            fig.show()
            return fig
        else:
            return fig

def unframe(fig, **kwargs):
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                showgrid=False,       
                showline=False,       
                showticklabels=False, 
                ticks='',             
                showbackground=False, 
                title='',
            ),
            yaxis=dict(
                showgrid=False, 
                showline=False, 
                showticklabels=False, 
                ticks='', 
                showbackground=False,
                title='',
            ),
            zaxis=dict(
                showgrid=False, 
                showline=False, 
                showticklabels=False, 
                ticks='', 
                showbackground=False,
                title='',
            )
        )
    )
    return fig

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