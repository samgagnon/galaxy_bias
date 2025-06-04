import numpy as np

from astropy.cosmology import Planck18
from astropy import units as u
from astropy.constants import c, G, m_p, k_B

def get_tang_data():
    # TANG LAE data
    # https://arxiv.org/pdf/2402.06070
    MUV, MUV_err, z, ew_lya, ew_lya_err, dv_lya, dv_lya_err, fescA, \
        fescA_err, fescB, fescB_err, ID = np.load('../data/tang24.npy').T
    return MUV, MUV_err, z, ew_lya, ew_lya_err, dv_lya, dv_lya_err, \
        fescA, fescA_err, fescB, fescB_err, ID

class LymanAlphaPopulation():

    """
    Lyman Alpha Population Model
    ==========
    This class implements the Lyman Alpha Population Model of Gagnon-Hartman et al. in prep 
    and provides tools to compare their statistics to the TANG LAE data 
    and SILVERRUSH narrow band survey statistics.
    It is used to generate synthetic Lyman Alpha emission line properties
    based on the input parameters.
    
    assumes case A recombination
    possibly inaccurate, see https://arxiv.org/pdf/2503.21896
    """

    def __init__(self):

        # histogram spaces
        muv_space = np.linspace(-24, -16, 100)
        dv_space = np.linspace(0, 1000, 100)
        dv_logspace = 10**np.linspace(0, 3, 100)
        w_space = 10**np.linspace(0, 3, 100)
        fesc_space = np.linspace(0, 1.5, 100)

        self.muv_space = muv_space[1:] - (muv_space[1] - muv_space[0])
        self.dv_space = dv_space[1:] - (dv_space[1] - dv_space[0])
        self.dv_logspace = dv_logspace[1:] - (dv_logspace[1] - dv_logspace[0])
        self.w_space = w_space[1:] - (w_space[1] - w_space[0])
        self.fesc_space = fesc_space[1:] - (fesc_space[1] - fesc_space[0])

        # constants
        self.nu_lya = (c/(1215.67*u.Angstrom)).to('Hz').value
        self.l_lya = 1215.67
        self.kuv = 1.15e-28  # UV continuum opacity in erg/s/cm^2/Hz

        # prior
        self.theta_min = np.array([-1e-3, -100, 41, 1.5, -1.5e3, 0.1, 0.1, 50, 0.0])
        self.theta_max = np.array([-1e-2, -50, 42, 3.0, -1e3, 0.5, 0.5, 200, 0.8])

    def load_data(self, redshift):
        """
        Setter function for halo field
        """
        _, _, _, self.halo_masses, _, self.sfr = np.load(f'../data/halo_field_{redshift}.npy')
        self.muv = 51.64 - np.log10(self.sfr*3.1557e7/1.15e-28) / 0.4
        self.halo_masses = self.halo_masses[self.muv<=-16]
        self.sfr = self.sfr[self.muv<=-16]
        self.muv = self.muv[self.muv<=-16]
        self.NSAMPLES = len(self.muv)
        self.lum_flux_factor = 4*np.pi*(Planck18.luminosity_distance(redshift).to('cm').value)**2

    def prior(self):
        """
        Inputs sampled from a uniform distribution
        """
        theta = np.random.uniform(0, 1, 9)
        self.theta = self.theta_min + \
            (self.theta_max - self.theta_min)*theta
        return self.theta

    def dice_loss(self, a, b):
        """
        Calculate the DICE loss between two probability distributions.
        
        Parameters:
        a : array-like
            The first probability distribution (true distribution).
        b : array-like
            The second probability distribution (approximate distribution).
            
        Returns:
        float
            The DICE loss between p and q.
        """
        a = np.asarray(a, dtype=np.float16)
        b = np.asarray(b, dtype=np.float16)

        if a.max() == 0 or b.max() == 0:
            return 1.0

        # normalize the distributions
        a /= a.max()
        b /= b.max()

        c_num = np.sum(a * b)
        c_den = np.sum(a * np.sign(b))
        if c_den == 0:
            c_den = 1
        c = c_num / c_den
        numerator = 2 * np.sum(a * b)
        denominator = c*np.sum(a) + np.sum(b)
        
        return numerator / denominator

    def get_beta_bouwens14(self, muv):
        """
        Returns the beta slope of the UV continuum
        based on the Bouwens et al. 2014 relation
        https://arxiv.org/pdf/1306.2950
        """
        return -2.05 + -0.2*(muv+19.5)

    def set_lya_properties(self, theta):

        a_w, a_v, b_h_mean, b_w_mean, b_v_mean,\
            sigma_h, sigma_w, sigma_v, _ = theta
        
        NSAMPLES = len(self.muv)

        # three random variables and two fixed slopes
        b_h = np.random.normal(b_h_mean, sigma_h, NSAMPLES)
        b_w = np.random.normal(b_w_mean, sigma_w, NSAMPLES)
        b_v = np.random.normal(b_v_mean, sigma_v, NSAMPLES)

        # we could alternatively set this to -2
        beta = self.get_beta_bouwens14(self.muv)
        log10w = a_w*a_v*self.muv + b_w + a_w*b_v
        log10fesc = log10w - b_h + 39.19 + 0.81*(beta + 2)
        dv = a_v*self.muv + b_v
        fesc = 10**log10fesc
        w = 10**log10w

        lha = (10**b_h) * self.sfr
        fha = lha / self.lum_flux_factor

        lum_dens_alpha = (w / 1215.67) * (10**(-0.4*(self.muv - 51.6))) * (1215.67/1500)**(beta + 2)
        intensity = lum_dens_alpha/self.lum_flux_factor
        mab = -2.5 * np.log10(intensity) - 48.6

        self.NSAMPLES = NSAMPLES
        self.mab = mab
        self.fha = fha
        self.dv = dv
        self.w = w
        self.fesc = fesc
    
    def get_select(self, NSAMPLES, mode='wide'):

        muv_limit_draw = -15
        fha_draw = np.random.normal(loc=2e-18, scale=2e-18/5, size=NSAMPLES)
        fha_draw[fha_draw > 2e-18] = 2e-18

        if mode == 'wide':
            # ROW ONE: MUSE-Wide
            flux_limit_draw = np.random.normal(loc=2e-17, scale=2e-17/5, size=NSAMPLES)
            flux_limit_draw[flux_limit_draw > 2e-17] = 2e-17
            flux_limit_draw[flux_limit_draw <= 0.0] = 1e-100
            mab_lim = -2.5 * np.log10(flux_limit_draw/self.nu_lya) - 48.6
            w_draw = np.random.normal(loc=80, scale=80/5, size=NSAMPLES)
            w_draw[w_draw > 80] = 80

        elif mode == 'deep':
            # SECOND ROW: MUSE-Deep
            flux_limit_draw = np.random.normal(loc=2e-18, scale=2e-18/5, size=NSAMPLES)
            flux_limit_draw[flux_limit_draw > 2e-18] = 2e-18
            flux_limit_draw[flux_limit_draw <= 0.0] = 1e-100
            mab_lim = -2.5 * np.log10(flux_limit_draw/self.nu_lya) - 48.6
            w_draw = np.random.normal(loc=8, scale=8/5, size=NSAMPLES)
            w_draw[w_draw > 8] = 8

        return mab_lim, muv_limit_draw, fha_draw, w_draw

    def get_t24_histograms(self):

        # TODO check that get_lya_properties is called before this function
            
        # appy selection effects
        _mab = np.copy(self.mab)
        _dv = np.copy(self.dv)
        _w = np.copy(self.w)
        _muv = np.copy(self.muv)
        _fesc = np.copy(self.fesc)
        _fha = np.copy(self.fha)

        mab_lim, muv_limit_draw, fha_draw, w_draw  = self.get_select(self.NSAMPLES, 'wide')

        print(self.theta)
        print(_fha.mean(), fha_draw.mean())
        SELECT = (_mab<mab_lim)*(_muv<muv_limit_draw)*(_fha>fha_draw)*(_w>w_draw)
        print(np.sum(_mab<mab_lim), np.sum(_muv<muv_limit_draw), np.sum(_fha>fha_draw), np.sum(_w>w_draw))
        quit()
        dv = _dv[SELECT]
        fesc = _fesc[SELECT]
        w = _w[SELECT]
        muv = _muv[SELECT]

        if len(w) == 0:
            return np.zeros((12, 9801))

        h_mdv, _, _ = np.histogram2d(muv, dv, bins=(self.muv_space, self.dv_space), density=True)
        model00 = h_mdv.T
        h_mw, _, _ = np.histogram2d(muv, w, bins=(self.muv_space, self.w_space), density=True)
        model01 = h_mw.T
        h_wdv, _, _ = np.histogram2d(w, dv, bins=(self.w_space, self.dv_space), density=True)
        model02 = h_wdv.T
        h_dvfesc, _, _ = np.histogram2d(dv, fesc, bins=(self.dv_logspace, self.fesc_space), density=True)
        model03 = h_dvfesc.T
        h_wfesc, _, _ = np.histogram2d(w, fesc, bins=(self.w_space, self.fesc_space), density=True)
        model04 = h_wfesc.T
        h_mfesc, _, _ = np.histogram2d(muv, fesc, bins=(self.muv_space, self.fesc_space), density=True)
        model05 = h_mfesc.T

        mab_lim, muv_limit_draw, fha_draw, w_draw  = self.get_select(self.NSAMPLES, 'deep')
        SELECT = (_mab<mab_lim)*(_muv<-15)*(_fha>fha_draw)*(_w>w_draw)
        dv = _dv[SELECT]
        fesc = _fesc[SELECT]
        w = _w[SELECT]
        muv = _muv[SELECT]

        if len(w) == 0:
            return np.zeros((12, 9801))

        # create 2d histogram of muv, dv
        h_mdv, _, _ = np.histogram2d(muv, dv, bins=(self.muv_space, self.dv_space), density=True)
        model10 = h_mdv.T
        h_mw, _, _ = np.histogram2d(muv, w, bins=(self.muv_space, self.w_space), density=True)
        model11 = h_mw.T
        h_wdv, _, _ = np.histogram2d(w, dv, bins=(self.w_space, self.dv_space), density=True)
        model12 = h_wdv.T
        h_dvfesc, _, _ = np.histogram2d(dv, fesc, bins=(self.dv_logspace, self.fesc_space), density=True)
        model13 = h_dvfesc.T
        h_wfesc, _, _ = np.histogram2d(w, fesc, bins=(self.w_space, self.fesc_space), density=True)
        model14 = h_wfesc.T
        h_mfesc, _, _ = np.histogram2d(muv, fesc, bins=(self.muv_space, self.fesc_space), density=True)
        model15 = h_mfesc.T

        # array of all flattened histograms
        out = np.array([model00.flatten(), model01.flatten(), model02.flatten(),
                        model03.flatten(), model04.flatten(), model05.flatten(),
                        model10.flatten(), model11.flatten(), model12.flatten(),
                        model13.flatten(), model14.flatten(), model15.flatten()])

        return out
    
    def load_t24_histograms(self):
        """
        Load the TANG LAE histograms from the precomputed file
        """
        hist00 = np.load('../data/muse_hist/hist00.npy').flatten()
        hist10 = np.load('../data/muse_hist/hist10.npy').flatten()
        hist01 = np.load('../data/muse_hist/hist01.npy').flatten()
        hist11 = np.load('../data/muse_hist/hist11.npy').flatten()
        hist02 = np.load('../data/muse_hist/hist02.npy').flatten()
        hist12 = np.load('../data/muse_hist/hist12.npy').flatten()
        hist03 = np.load('../data/muse_hist/hist03.npy').flatten()
        hist13 = np.load('../data/muse_hist/hist13.npy').flatten()
        hist04 = np.load('../data/muse_hist/hist04.npy').flatten()
        hist14 = np.load('../data/muse_hist/hist14.npy').flatten()
        hist05 = np.load('../data/muse_hist/hist05.npy').flatten()
        hist15 = np.load('../data/muse_hist/hist15.npy').flatten()

        out =  np.array([hist00.flatten(), hist01.flatten(), hist02.flatten(),
                        hist03.flatten(), hist04.flatten(), hist05.flatten(),
                        hist10.flatten(), hist11.flatten(), hist12.flatten(),
                        hist13.flatten(), hist14.flatten(), hist15.flatten()])
        
        return out
    
    def likelihood(self):

        redshift = 5.0
        self.load_data(redshift)
        theta = self.prior()
        self.set_lya_properties(theta)
        likelihood = 0.0
        if redshift == 5.0:
            model_hist_list = self.get_t24_histograms()
            data_hist_list = self.load_t24_histograms()
            hist_likelihood = np.sum([self.dice_loss(m, d) for \
                m, d in zip(model_hist_list, data_hist_list)])
            likelihood += hist_likelihood
        return likelihood

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    rc = {"font.family" : "serif", 
        "mathtext.fontset" : "stix"}
    plt.rcParams.update(rc) 
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams.update({'font.size': 14})
    plt.style.use('dark_background')
    
    model = LymanAlphaPopulation()

    model.load_data(5.0)
    model.set_lya_properties(model.prior())
    print("Likelihood:", model.likelihood())
    quit()
    
    fig, axs = plt.subplots(3, 1, figsize=(12, 12), constrained_layout=True)
    axs[0].hist(np.log10(model.w), bins=50, density=True, color='blue', alpha=0.5, label='MUV')
    axs[0].set_xlabel('W')
    axs[0].set_ylabel('Density')
    axs[1].hist(model.dv, bins=50, density=True, color='green', alpha=0.5, label='dv')
    axs[1].set_xlabel('dv (km/s)')
    axs[1].set_ylabel('Density')
    axs[2].hist(np.log10(model.fesc), bins=50, density=True, color='red', alpha=0.5, label='fesc')
    axs[2].set_xlabel('fesc')
    axs[2].set_ylabel('Density')
    plt.show()