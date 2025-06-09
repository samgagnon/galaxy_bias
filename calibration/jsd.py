import numpy as np

from astropy.cosmology import Planck18
from astropy import units as u
from astropy.constants import c, G, m_p, k_B


def get_gaus_mvar(x, y, mean, sigma):
    x_gaus = np.exp(-0.5*((x - mean[0])/sigma[0])**2)/(sigma[0]*np.sqrt(2*np.pi))
    y_gaus = np.exp(-0.5*((y - mean[1])/sigma[1])**2)/(sigma[1]*np.sqrt(2*np.pi))
    gaus_mvar = np.array([x_gaus]*len(y_gaus))*np.array([y_gaus]*len(x_gaus)).T
    return gaus_mvar
    
def dat_to_hist(x, y, dat, dat_err):
    """
    Given a 2D array of data and errors, return the histogram of the data
    """
    hist = np.zeros((len(x), len(y)))
    for datum, datum_err in zip(dat.T, dat_err.T):
        gaus_mvar = get_gaus_mvar(x, y, datum, datum_err)
        gaus_mvar /= gaus_mvar.max()
        hist += gaus_mvar
    hist /= hist.sum()
    return x, y, hist

def kl_divergence(p, q):
    """
    Compute the Kullback-Leibler divergence between two probability distributions.
    """
    p = p.flatten()
    q = q.flatten()
    # Remove zero entries
    select = (p > 0)*(q>0)
    p = p[select]
    q = q[select]  
    return np.sum(p * np.log(p / q))

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    rc = {"font.family" : "serif", 
        "mathtext.fontset" : "stix"}
    plt.rcParams.update(rc) 
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams.update({'font.size': 14})
    plt.style.use('dark_background')

    xspace = np.linspace(-10, 10, 100)
    yspace = np.linspace(-10, 10, 100)
    x = xspace[1:] - (xspace[1] - xspace[0])
    y = yspace[1:] - (yspace[1] - yspace[0])
    p = get_gaus_mvar(xspace, yspace, np.array([0, 0]), np.array([1, 1]))
    # p /= p.max()
    p /= p.sum()

    from scipy.special import erfinv

    for i in range(10):
        q = get_gaus_mvar(xspace, yspace, np.array([0, 0]), np.array([i+1, 1]))
        # q /= q.max()
        q /= q.sum()
        m = 0.5 * (p + q)
        js_div = 0.5*(kl_divergence(p, m) + kl_divergence(q, m))/np.log(2)
        print(f"Jensen-Shannon divergence for breadth {i:.2f}: {erfinv(js_div):.2f}")

        # prob = np.sum(q*p)
        # print(f"Probability of overlap for breadth {i+1:.2f}: {prob:.2f}")
