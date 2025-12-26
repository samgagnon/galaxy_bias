import numpy as np

from astropy.cosmology import Planck18, z_at_value
from astropy import units as u
from astropy.constants import c, G, m_p, k_B

from scipy import odr

def gaussian(x, mu, sigma):
    return np.exp(-0.5*((x - mu)/sigma)**2)/(sigma*np.sqrt(2*np.pi))

def I(x):
    return x**(9/2)/(1 - x) + (9/7)*x**(7/2) + \
        (9/5)*x**(5/2) + 3*x**(3/2) + 9*x**(1/2) - \
        (9/2)*np.log(np.abs((1 + x**0.5)/(1 - x**0.5)))

def miralda_escude(z, delta_z):
    delta_z = 1e-3*delta_z
    tau_gp = 7.16e5*((1 + z)/10)**1.5
    nu_lya = (c/(1215.67*u.Angstrom)).to('Hz').value
    r_alpha = 6.25e8/(4*np.pi*nu_lya)
    z_b = z_s - delta_z
    x_range_b = (1 + z_b)/(1 + z)
    x_range_e = (1 + z_e)/(1 + z)
    return tau_gp*r_alpha*((1+z_b)/(1+z))**1.5*(I(x_range_b) - I(x_range_e))/np.pi

def asymmetric_gaussian(x, mu, sigma, a):
    sigma_a = a*(x - mu) + sigma
    return np.exp(-0.5*(x - mu)**2/sigma_a**2)


def double_asymmetric_gaussian(velocity_range, mu, sigma, a):
    out = asymmetric_gaussian(velocity_range, mu, sigma, a) + \
        asymmetric_gaussian(velocity_range, -1*mu, sigma, -1*a)
    out /= np.max(out)
    return out

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    rc = {"font.family" : "serif", 
        "mathtext.fontset" : "stix"}
    plt.rcParams.update(rc) 
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams.update({'font.size': 14})
    # plt.style.use('dark_background')

    cm = plt.get_cmap('bwr')

    tau_igm_table = np.load('../data/tau_igm_table.npy')
    # density_pencil = np.load('density_pencil.npy')
    # xHI_pencil = np.load('xHI_pencil.npy')
    # dvz_pencil = np.load('dvz_pencil.npy')
    # vz_pencil = np.load('vz_pencil.npy')

    z_s = 7.0
    # end of EoR
    z_e = 5.5
    wavelength_range = np.linspace(1215, 1220, 1000)
    velocity_range = c.to('km/s').value*(wavelength_range/1215.67 - 1)
    z_range = (wavelength_range/1215.67)*(1 + z_s) - 1
    # delta = (wavelength_range - 1215.67*(1 + z_s))/(1215.67*(1 + z_s))

    model = odr.Model(miralda_escude)
    data = odr.RealData(z_range[velocity_range>0], tau_igm_table.T[1000, velocity_range>0])
    odr_obj = odr.ODR(data, model, beta0=[4])
    out = odr_obj.run()
    delta_z = out.beta[0]
    print('delta_z', delta_z)

    # mock_igm = np.exp(-1*miralda_escude(z_range, 50))
    mock_igm_base = np.exp(-1*tau_igm_table.T[1000])
    mock_igm = mock_igm_base

    # fig, axs = plt.subplots()
    # axs.set_xlabel('velocity [km/s]', fontsize=16)
    # axs.set_ylabel('flux [arb. units]', fontsize=16)
    # axs.set_xlim(-150, 500)
    # axs.set_ylim(-0.1, 1.1)
    # axs.plot(velocity_range, double_asymmetric_gaussian(velocity_range, 50, 20, 0.2), color='blue', linewidth=2, label='Intrinsic Lyα line')
    # plt.show()
    # quit()

    from matplotlib import animation

    fig, ax = plt.subplots()
    ax.set_xlabel('velocity [km/s]', fontsize=16)
    ax.set_ylabel('flux [arb. units]', fontsize=16)
    ax.set_xlim(-150, 500)

    mock_line = double_asymmetric_gaussian(velocity_range, 50, 20, 0.2)

    line1 = ax.plot(velocity_range, mock_line, color='blue', linewidth=2)
    line2 = ax.plot(velocity_range, mock_igm_base, color='orange', linewidth=2)
    line3 = ax.plot(velocity_range, mock_line*mock_igm_base, color='magenta', linewidth=2)

    def update(frame):
        if frame < 100:
            # mock_igm = np.interp(velocity_range+10-frame, velocity_range, mock_igm_base)
            mock_line = double_asymmetric_gaussian(velocity_range, frame, 20, 0.2)
            line1[0].set_ydata(mock_line)
            line2[0].set_ydata(mock_igm)
            line3[0].set_ydata(mock_line*mock_igm)
        else:
            mock_line = double_asymmetric_gaussian(velocity_range, 150-(frame-50), 20, 0.2)
            # mock_igm = np.interp(velocity_range+10-100+(frame-100), velocity_range, mock_igm_base)
            line1[0].set_ydata(mock_line)
            line2[0].set_ydata(mock_igm)
            line3[0].set_ydata(mock_line*mock_igm)


    ani = animation.FuncAnimation(fig=fig, func=update, frames=200, interval=15)

    # plt.show()
    ani.save('/mnt/c/Users/sgagn/Downloads/test.gif', writer='pillow', fps=30)
