"""
Experiment with prior bounds and determine implied mean escape fraction
for each.
"""

import numpy as np

mu_w_prior_bounds = {
    "min": 1.0,
    "max": 2.7
    }

sd_w_prior_bounds = {
    "min": 0.25,
    "max": 0.5
    }

mu_v_prior_bounds = {
    "min": -1500,
    "max": -1200
    }

sd_v_prior_bounds = {
    "min": 85,
    "max": 200
    }

mu_h_prior_bounds = {
    "min": 41,
    "max": 41.54
    }

sd_h_prior_bounds = {
    "min": 0.1,
    "max": 0.5
    }

def get_prior_bounds():
    """
    Returns a dictionary containing the prior bounds for mu_w, sd_w, mu_v, sd_v, mu_h, and sd_h.
    """
    return {
        "mu_w": mu_w_prior_bounds,
        "sd_w": sd_w_prior_bounds,
        "mu_v": mu_v_prior_bounds,
        "sd_v": sd_v_prior_bounds,
        "mu_h": mu_h_prior_bounds,
        "sd_h": sd_h_prior_bounds
    }

def sample_prior(n=1):
    """
    Samples from the prior distributions defined by the bounds.
    
    Returns:
        dict: A dictionary containing sampled values for mu_w, sd_w, mu_v, sd_v, mu_h, and sd_h.
        n: Number of samples to draw (default is 1).
    """
    prior = get_prior_bounds()
    return {
        "mu_w": np.random.uniform(prior["mu_w"]["min"], prior["mu_w"]["max"], n),
        "sd_w": np.random.uniform(prior["sd_w"]["min"], prior["sd_w"]["max"], n),
        "mu_v": np.random.uniform(prior["mu_v"]["min"], prior["mu_v"]["max"], n),
        "sd_v": np.random.uniform(prior["sd_v"]["min"], prior["sd_v"]["max"], n),
        "mu_h": np.random.uniform(prior["mu_h"]["min"], prior["mu_h"]["max"], n),
        "sd_h": np.random.uniform(prior["sd_h"]["min"], prior["sd_h"]["max"], n)
    }

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    rc = {"font.family" : "serif", 
        "mathtext.fontset" : "stix"}
    plt.rcParams.update(rc)
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams.update({'font.size': 14})
    plt.style.use('dark_background')
    import matplotlib as mpl
    label_size = 20
    font_size = 30
    mpl.rcParams['xtick.labelsize'] = label_size 
    mpl.rcParams['ytick.labelsize'] = label_size

    # Example usage
    samples = sample_prior(1000)

    mu_w = samples["mu_w"]
    sd_w = samples["sd_w"]
    mu_v = samples["mu_v"]
    sd_v = samples["sd_v"]
    mu_h = samples["mu_h"]
    sd_h = samples["sd_h"]

    num = mu_w - mu_v/303 - mu_h + 0.64
    denom = np.sqrt(sd_w**2 + (sd_v/303)**2 + sd_h**2)
    s = -1* num / denom

    print(s.min(), s.max(), s.mean())
    plt.hist(s, bins=20, density=True, alpha=1.0, color='blue')
    plt.xlabel(r'$s$', fontsize=label_size)
    plt.ylabel('Density', fontsize=label_size)
    plt.grid()
    plt.show()