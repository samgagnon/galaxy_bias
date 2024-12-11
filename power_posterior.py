import numpy as np

from scipy import special
from scipy.stats import skewnorm, lognorm
from scipy.optimize import minimize
from astropy.cosmology import Planck18

def get_skewnorm_params(mean, lower_limit, upper_limit):
    # Objective function to minimize

    def objective(params):
        alpha, loc, scale = params
        skewed_dist = lognorm(s=alpha, loc=loc, scale=scale)
        lower_cdf = skewed_dist.cdf(mean+lower_limit)
        upper_cdf = skewed_dist.cdf(mean+upper_limit)
        # fit one sigma bounds and mean
        return (lower_cdf - 0.1587)**2 + (upper_cdf - 0.8413)**2 \
            + (skewed_dist.mean() - mean)**2 #+ (np.exp(-2*alpha) - mean)**2

    # Initial guess for alpha, loc, and scale
    initial_guess = [1, 0, 1]

    # Minimize the objective function
    result = minimize(objective, initial_guess, bounds=[(-10, 10), (0.01, 10), (0.1, 10)])
    alpha, loc, scale = result.x

    skewed_dist = lognorm(s=alpha, loc=loc, scale=scale)
    lower_cdf = skewed_dist.cdf(mean+lower_limit)
    upper_cdf = skewed_dist.cdf(mean+upper_limit)
    print(f"Lower CDF: {lower_cdf}, Upper CDF: {upper_cdf}, Mean: {skewed_dist.mean()}")
    # quit()

    return alpha, loc, scale


def p_lognorm(x, mu, sigma, skew):
    y = (x - mu) / sigma
    factor = 1/(y*sigma*np.sqrt(2*np.pi))
    exp = np.exp(-0.5*np.log(y)**2/skew**2)
    return factor*exp

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    rc = {"font.family" : "serif", 
        "mathtext.fontset" : "stix"}
    plt.rcParams.update(rc)
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams.update({'font.size': 14})
    plt.style.use('dark_background')

    from matplotlib.colors import ListedColormap

    newcmp = ListedColormap(['black', 'cyan', 'blue', 'navy'])

    # r0 = 8
    # r0_upper = 8 + 1.9
    # r0_lower = 8 - 5.8
    # gamma = 1.4
    # gamma_upper = 1.4 + 0.58
    # gamma_lower = 1.4 - 0.88

    r0 = 4
    r0_upper = r0 + 0.6
    r0_lower = r0 - 0.7
    gamma = 1.4
    gamma_upper = gamma + 0.17
    gamma_lower = gamma - 0.17

    gamma_up_err = gamma_upper - gamma
    gamma_low_err = gamma - gamma_lower
    r0_up_err = r0_upper - r0
    r0_low_err = r0 - r0_lower

    gamma_up_rel_err = gamma_up_err / gamma
    gamma_low_rel_err = gamma_low_err / gamma
    r0_up_rel_err = r0_up_err / r0
    r0_low_rel_err = r0_low_err / r0

    gamma_up_special_err = special.gamma(gamma_upper) - special.gamma(gamma)
    gamma_low_special_err = special.gamma(gamma) - special.gamma(gamma_lower)
    r0_up_special_err = special.gamma(r0_upper) - special.gamma(r0)
    r0_low_special_err = special.gamma(r0) - special.gamma(r0_lower)

    slope_up_err = gamma_up_err
    slope_low_err = gamma_low_err

    slope = gamma - 1

    NSAMPLES = 100000

    r0_alpha, r0_loc, r0_scale = get_skewnorm_params(r0, -1*r0_low_err, r0_up_err)

    gamma_alpha, gamma_loc, gamma_scale = get_skewnorm_params(gamma, -1*gamma_low_err, gamma_up_err)

    gamma_range = np.linspace(1, 3, 100)
    r0_range = np.linspace(2, 8, 100)

    verify = True
    # verify = False
    if verify:

        r0_fit = p_lognorm(r0_range, mu=r0_loc, sigma=r0_scale, skew=r0_alpha)
        gamma_fit = p_lognorm(gamma_range, mu=gamma_loc, sigma=gamma_scale, skew=gamma_alpha)

        fig, axs = plt.subplots(1, 2, sharey=False, figsize=(12, 6), constrained_layout=True)
        axs[0].plot(r0_range, r0_fit, color='white')
        axs[0].axvline(r0-r0_low_err, color='blue', linestyle='dashed', linewidth=2)
        axs[0].axvline(r0, color='red', linestyle='dashed', linewidth=2)
        axs[0].axvline(r0+r0_up_err, color='blue', linestyle='dashed', linewidth=2)
        axs[0].set_xlabel(r'$r_0$ [h$_{70}$ Mpc$^{-1}$]', fontsize=14)
        axs[0].set_ylabel('PDF', fontsize=14)

        axs[1].plot(gamma_range, gamma_fit, color='white')
        axs[1].axvline(gamma-gamma_low_err, color='blue', linestyle='dashed', linewidth=2)
        axs[1].axvline(gamma, color='red', linestyle='dashed', linewidth=2)
        axs[1].axvline(gamma+gamma_up_err, color='blue', linestyle='dashed', linewidth=2)
        axs[1].set_xlabel(r'$\gamma$', fontsize=14)
        plt.show()
        plt.close()

    r_gamma_mesh = np.meshgrid(r0_range, gamma_range)
    p_r0 = p_lognorm(r_gamma_mesh[0], mu=r0_loc, sigma=r0_scale, skew=r0_alpha)
    p_gamma = p_lognorm(r_gamma_mesh[1], mu=gamma_loc, sigma=gamma_scale, skew=gamma_alpha)
    p_joint = p_r0 * p_gamma

    fig, axs = plt.subplots(1, 1, figsize=(6, 6), constrained_layout=True)

    im = axs.contourf(p_joint, extent=[2, 8, 1, 2], \
                cmap='hot')
    # plt.colorbar(im, ax=axs)
    axs.scatter(r0, gamma, color='orange', marker='x', s=100)
    axs.set_xlabel(r'$r_0$ [h$_{70}$ Mpc$^{-1}$]', fontsize=14)
    axs.set_ylabel(r'$\gamma$', fontsize=14)
    plt.show()
