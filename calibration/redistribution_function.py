import numpy as np

from scipy.special import erf
from scipy.integrate import trapezoid

from astropy.cosmology import Planck18
from astropy import units as u
from astropy.constants import c, G, m_p, k_B

# lyman alpha parameters
wave_Lya = 1215.67*u.AA
freq_Lya = (c.to(u.AA/u.s)/wave_Lya).to('Hz')
omega_Lya = 2*np.pi*freq_Lya
decay_factor = 6.25*1e8*u.s**-1
wavelength_range = np.linspace(1215, 1220, 1000)*u.AA

def delta_nu(T):
    """
    Returns the Doppler width of a Lya line with temperature T
    """
    return ((freq_Lya/c) * np.sqrt(2*k_B*T/m_p.to('kg'))).to('Hz')

def get_a(dnu):
    """
    Returns the damping parameter of a Lya line with Doppler width dnu
    """
    return (decay_factor/(4*np.pi*dnu)).to('')

def voigt_tasitsiomi(x, dnu):
    """
    Returns the Voigt profile of a line with damping parameter a
    """
    dL = 9.936e7*u.Hz
    a = 0.5*(dL/dnu)
    xt = x**2
    z = (xt - 0.855)/(xt + 3.42)
    q = np.zeros(len(x))
    IDX = (z > 0.0)
    q[IDX] = z[IDX]*(1 + 21/xt[IDX])*(a/np.pi/(xt[IDX] + 1.0))*\
        (0.1117 + z[IDX]*(4.421 + z[IDX]*(-9.207 + 5.674*z[IDX])))
    return (q + np.exp(-xt)/1.77245385)*np.sqrt(np.pi)

def directional_redistribution_func(x_out, u_space, xin, mu):
    """
    Returns the redistribution function for a given mu and xin
    """
    delta_x = x_out - xin
    gas_temp = 1e4*u.K
    dnu = delta_nu(gas_temp)
    a_nu = get_a(dnu)
    voigt_profile = voigt_tasitsiomi(x_out, dnu)
    norm = a_nu/(np.pi**(3/2)*voigt_profile[np.argmin(np.abs(x_out - xin))]*np.sqrt(1-mu**2))
    integrand_one = np.exp(-u_space**2)/((xin - u_space)**2 + a_nu**2)
    integrand_two = np.exp(-1*((delta_x[:,np.newaxis] + u_space*(mu - 1))/(np.sqrt(1 - mu**2)))**2)
    return norm*integrand_one*integrand_two
    # return norm*trapezoid(integrand_one*integrand_two, u_space, axis=1)

def redistribution_func(x_out, u_space, x_in):
    # draw mu from distribution and take the average
    theta_space = np.linspace(0, 2*np.pi, 1000)
    p_theta = 1+np.cos(theta_space)**2
    p_theta = p_theta/p_theta.sum()
    theta_sample = np.random.choice(theta_space, size=1000, p=p_theta)
    mu_sample = np.cos(theta_sample)
    delta_x = x_out - x_in
    gas_temp = 1e4*u.K
    dnu = delta_nu(gas_temp)
    a_nu = get_a(dnu)
    voigt_profile = voigt_tasitsiomi(x_out, dnu)
    norm = a_nu/(np.pi**(3/2)*voigt_profile[np.argmin(np.abs(x_out - x_in))]*np.sqrt(1-mu_sample[np.newaxis, np.newaxis, :]**2))
    integrand_one = np.exp(-u_space[np.newaxis,:,np.newaxis]**2)/((x_in - u_space[np.newaxis,:,np.newaxis])**2 + a_nu**2)
    integrand_two = np.exp(-1*((delta_x[:, np.newaxis, np.newaxis] + u_space[np.newaxis,:,np.newaxis]*(mu_sample[np.newaxis, np.newaxis, :] - 1))/\
                               (np.sqrt(1 - mu_sample[np.newaxis, np.newaxis, :]**2)))**2)
    return norm*integrand_one*integrand_two
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    rc = {"font.family" : "serif", 
        "mathtext.fontset" : "stix"}
    plt.rcParams.update(rc) 
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams.update({'font.size': 14})
    plt.style.use('dark_background')

    x_in = np.linspace(-10, 10, 1000)
    x_out = np.copy(x_in)
    x_out[np.abs(x_in)>3] -= 1/x_in[np.abs(x_in)>3]
    u_space = np.linspace(-100, 100, 1000)
    voigt = voigt_tasitsiomi(x_in, delta_nu(1e4*u.K))
    # voigt = voigt_tasitsiomi(x_out, delta_nu(1e4*u.K))

    plt.plot(x_in, voigt, color='cyan')
    plt.plot(x_out, voigt, color='lime')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$\phi(x)$')
    plt.yscale('log')
    plt.show()
    quit()
    fig, axs = plt.subplots(2, 5, figsize=(12, 6), sharex=True, constrained_layout=True)
    for i, mu in enumerate([0.1, 0.3, 0.5, 0.7, 0.9]):
        theta = np.arctan(np.sqrt(1 - mu**2)/mu)
        f = directional_redistribution_func(x_out, u_space, 5, mu)
        f_line = trapezoid(f, u_space, axis=1)
        axs[0, i].pcolormesh(x_out, u_space, np.log10(f.T), cmap='hot')
        axs[1, i].plot(x_out, f_line, color='red')
        axs[0, i].text(-8, 80, r'$\theta=$'+str(np.around(theta*180/np.pi, 2))+r'$^\circ$', color='white')

    axs[1,0].set_ylabel(r'$p(x_{\rm out})$')
    axs[0,0].set_ylabel(r'$v/v_{\rm th}$')
    axs[1,0].set_xlabel(r'$x_{\rm out}$')
    axs[1,1].set_xlabel(r'$x_{\rm out}$')
    axs[1,2].set_xlabel(r'$x_{\rm out}$')
    axs[1,3].set_xlabel(r'$x_{\rm out}$')
    axs[1,4].set_xlabel(r'$x_{\rm out}$')

    plt.show()
    quit()

    theta_space = np.linspace(0, 2*np.pi, 1000)
    p_theta = 1+np.cos(theta_space)**2
    p_theta = p_theta/p_theta.sum()
    theta_sample = np.random.choice(theta_space, size=1000, p=p_theta)
    mu_sample = np.cos(theta_sample)

    plt.hist(mu_sample, bins=100, histtype='step', color='white')
    plt.xlabel(r'$\mu$')
    plt.ylabel(r'$p(\mu)$')
    plt.show()
