import numpy as np

from scipy import special
from scipy.stats import skewnorm, lognorm
from scipy.optimize import minimize
from astropy.cosmology import Planck18

def ellipse(x, a, b, h, k):
    return b*np.sqrt(1 - (x-h)**2/a**2) + k

def get_slope(gamma):
    return gamma - 1

def get_intercept(r0, gamma):
    return np.real(gamma * np.log10(1j) + 0.5*np.log10(np.pi/2) + \
        -1*gamma*np.log10(r0) - np.log10(special.gamma(gamma)))

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

    r0 = 8
    r0_upper = 8 + 1.9
    r0_lower = 8 - 5.8
    gamma = 1.4
    gamma_upper = 1.4 + 0.58
    gamma_lower = 1.4 - 0.88

    # r0 = 4
    # r0_upper = r0 + 0.6
    # r0_lower = r0 - 0.7
    # gamma = 1.4
    # gamma_upper = gamma + 0.17
    # gamma_lower = gamma - 0.17

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

    skew = False
    # skew = True

    if skew:

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

    else:

        resolution = 10000

        r0_range_right = np.linspace(r0, r0+r0_up_err, resolution)
        r0_range_left = np.linspace(r0-r0_low_err, r0, resolution)

        r0_range_right2 = np.linspace(r0, r0+2*r0_up_err, resolution)
        r0_range_left2 = np.linspace(r0-2*r0_low_err, r0, resolution)

        r0_range_right3 = np.linspace(r0, r0+3*r0_up_err, resolution)
        r0_range_left3 = np.linspace(r0-3*r0_low_err, r0, resolution)

        r0_range_1 = np.concatenate([r0_range_right, r0_range_right[::-1], r0_range_left[::-1], r0_range_left])
        r0_range_2 = np.concatenate([r0_range_right2, r0_range_right2[::-1], r0_range_left2[::-1], r0_range_left2])
        r0_range_3 = np.concatenate([r0_range_right3, r0_range_right3[::-1], r0_range_left3[::-1], r0_range_left3])

        sigma1_upper_right = ellipse(r0_range_right, r0_up_err, gamma_up_err, r0, gamma)
        sigma1_upper_left = ellipse(r0_range_left, r0_low_err, gamma_up_err, r0, gamma)
        sigma1_lower_right = gamma-1*(ellipse(r0_range_right, r0_up_err, gamma_low_err, r0, gamma) - gamma)
        sigma1_lower_left = gamma-1*(ellipse(r0_range_left, r0_low_err, gamma_low_err, r0, gamma) - gamma)

        sigma2_upper_right = ellipse(r0_range_right2, 2*r0_up_err, 2*gamma_up_err, r0, gamma)
        sigma2_upper_left = ellipse(r0_range_left2, 2*r0_low_err, 2*gamma_up_err, r0, gamma)
        sigma2_lower_right = gamma-1*(ellipse(r0_range_right2, 2*r0_up_err, 2*gamma_low_err, r0, gamma) - gamma)
        sigma2_lower_left = gamma-1*(ellipse(r0_range_left2, 2*r0_low_err, 2*gamma_low_err, r0, gamma) - gamma)

        sigma3_upper_right = ellipse(r0_range_right3, 3*r0_up_err, 3*gamma_up_err, r0, gamma)
        sigma3_upper_left = ellipse(r0_range_left3, 3*r0_low_err, 3*gamma_up_err, r0, gamma)
        sigma3_lower_right = gamma-1*(ellipse(r0_range_right3, 3*r0_up_err, 3*gamma_low_err, r0, gamma) - gamma)
        sigma3_lower_left = gamma-1*(ellipse(r0_range_left3, 3*r0_low_err, 3*gamma_low_err, r0, gamma) - gamma)

        sigma1 = np.concatenate([sigma1_upper_right, sigma1_lower_right[::-1], sigma1_lower_left[::-1], sigma1_upper_left])
        sigma2 = np.concatenate([sigma2_upper_right, sigma2_lower_right[::-1], sigma2_lower_left[::-1], sigma2_upper_left])
        sigma3 = np.concatenate([sigma3_upper_right, sigma3_lower_right[::-1], sigma3_lower_left[::-1], sigma3_upper_left])

        r0_range_1[r0_range_1<0] = 0
        r0_range_2[r0_range_2<0] = 0
        r0_range_3[r0_range_3<0] = 0

        sigma1[sigma1<0] = 0
        sigma2[sigma2<0] = 0
        sigma3[sigma3<0] = 0

        fig, axs = plt.subplots(1, 2, figsize=(6, 6), constrained_layout=True)

        axs[0].plot(r0, gamma, 'x', color='cyan', markersize=10)

        axs[0].plot(r0_range_1, sigma1, color='cyan', linestyle='solid', linewidth=2)
        axs[0].plot(r0_range_2, sigma2, color='cyan', linestyle='dashed', linewidth=2)
        axs[0].plot(r0_range_3, sigma3, color='cyan', linestyle='dotted', linewidth=2)

        axs[0].set_ylabel(r'$\gamma$', fontsize=14)
        axs[0].set_xlabel(r'$r_0$ [h$_{70}$ Mpc$^{-1}$]', fontsize=14)

        axs[1].plot(get_intercept(r0, gamma), get_slope(gamma), 'x', color='cyan', markersize=10)

        axs[1].plot(get_intercept(r0_range_1, sigma1), get_slope(sigma1), color='cyan', linestyle='solid', markersize=10)
        axs[1].plot(get_intercept(r0_range_2, sigma2), get_slope(sigma2), color='cyan', linestyle='dashed', markersize=10)
        axs[1].plot(get_intercept(r0_range_3, sigma3), get_slope(sigma3), color='cyan', linestyle='dotted', markersize=10)

        # axs[1].plot((np.linspace(-2, 2) - 1), np.linspace(-2, 2), '--', color='cyan', markersize=10)

        axs[1].set_ylabel(r'$\gamma-1$', fontsize=14)
        axs[1].set_xlabel(r'$\log_{10}k_0$', fontsize=14)

        # axs[1].set_xlim(-2, 5)

        plt.show()
