import numpy as np

from astropy.constants import c, G
from astropy import units as u
from astropy.cosmology import Planck18, z_at_value

def gaussian(x, mu, sigma):
    return np.exp(-0.5*((x - mu)/sigma)**2)/(sigma*np.sqrt(2*np.pi))

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

    cm = plt.get_cmap('bwr')

    tau_igm_table = np.load('../data/tau_igm_table.npy')
    density_pencil = np.load('../data/density_pencil.npy')
    xHI_pencil = np.load('../data/xHI_pencil.npy')
    dvz_pencil = np.load('../data/dvz_pencil.npy')
    vz_pencil = np.load('../data/vz_pencil.npy')

    dvz_pencil[xHI_pencil>0.0] = 0.0

    color_grad = np.zeros_like(xHI_pencil)
    last_neutral = False
    counting = False
    for i, xHI in enumerate(xHI_pencil):
        if xHI == 0.0:
            if last_neutral or counting:
                color_grad[i] += last_color + 1
                counting = True
            last_neutral = False
        else:
            counting = False
            last_neutral = True
        last_color = color_grad[i]

    color_grad[color_grad>0.0] = 0.5
    color_grad[xHI_pencil>0.0] = 1.0

    # BLUE: ionized region, no neutral on LoS
    # WHITE: ionized region, neutral on LoS (emitter inside bubble)
    # RED: neutral region

    wavelength_range = np.linspace(1215, 1220, 1000)
    velocity_range = c.to('km/s').value*(wavelength_range/1215.67 - 1)

    dv = 300
    emission = gaussian(velocity_range, dv, dv)
    emission /= emission.max()

    Mh = 1e10*u.solMass
    v_circ = ((10*G*Mh*Planck18.H(7.0))**(1/3)).to('km/s')

    fig, axs = plt.subplots(1, 1, figsize=(10, 8), sharex=True, constrained_layout=True)

    for i, tau_igm in enumerate(tau_igm_table.T):
        tau_igm = np.interp(velocity_range, dvz_pencil[i] + velocity_range, tau_igm)
        axs.plot(velocity_range, np.exp(-1*tau_igm), c=cm(color_grad[i]), alpha=0.1, rasterized=True)

    axs.set_xlabel('Velocity [km/s]', fontsize=font_size)
    axs.set_ylabel('Transmission [%]', fontsize=font_size)
    axs.set_ylim(0, 1)
    axs.set_xlim(velocity_range.min(), velocity_range.max())
    
    plt.show()
    