"""
A code to prepare properties for computing the Lyman alpha transmission lightcone
Needs to be run once per cosmology. Be sure to use the same cosmology as the lightcone, 
pay attention to the Planck18 import statement.
"""

import os

import py21cmfast as p21c

from utils import *

from scipy import interpolate

from astropy.cosmology import Planck18, z_at_value

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    rc = {"font.family" : "serif", \
        "mathtext.fontset" : "stix"}
    plt.rcParams.update(rc)
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams.update({'font.size': 14})

    # import the lightcone and extract the relevant quantities
    LC = p21c.LightCone.read('./data/lightcones/LC.h5')
    xHI = LC.xH_box
    vz = LC.velocity_z
    density = LC.density
    node_redshifts = LC.node_redshifts

    # rewrite these to load in the values from the lightcone
    DIM = xHI.shape[0]
    LoS_DIM = xHI.shape[-1]
    L = 300*u.Mpc
    res = L/DIM

    # this is a choice
    NCELLS_HIRES = int(100)

    d_LoS = np.linspace(0, 1.5*LoS_DIM, int(LoS_DIM))*u.Mpc + Planck18.comoving_distance(5.00)
    d_LoS_edges = np.linspace(0, 1.5*LoS_DIM + 1.5, int(LoS_DIM)+1)*u.Mpc + Planck18.comoving_distance(5.00) - 1.5*u.Mpc/2

    z_LoS = np.array([z_at_value(Planck18.comoving_distance, d) for d in d_LoS])
    z_LoS_edges = np.array([z_at_value(Planck18.comoving_distance, d) for d in d_LoS_edges])
    d_transverse = np.linspace(0, 300, 200)

    z_LoS_cube = np.array([[z_LoS]*DIM]*DIM)
    vz_proper = (vz*u.Mpc/u.s).to('km/s')/(1+z_LoS_cube)

    # get width of the Lyman alpha line for the assumed 
    # temperature of all neutral gas (~10^4 K)
    dnu = delta_nu(1e4*u.K)
    a = get_a(dnu)

    # compute the redshift jacobian for the lowres sightline
    jacobian = (c.c/Planck18.H(z_LoS)).to('cm').value
    
    # compute the critical number density of hydrogen along the LoS
    rho_crit = Planck18.critical_density(z_LoS)
    n_HI = 0.75*(rho_crit/c.m_p).to('1/cm^3') #precompute this!
    n_HI = n_HI.value

    # produce x range for template voigt profile
    wavelength_range_extd = np.linspace(1210, 1250, 1000)*u.AA
    velocity_range = c.c.to('km/s')*(wavelength_range_extd/wave_Lya - 1)
    x0 = (velocity_range/c.c).to('')*(freq_Lya/dnu).to('')

    # define cross section constants
    v_thermal = np.sqrt(2*c.k_B*1e4*u.K/c.m_p).to('km/s').value
    sigma_0 = 5.88e-14*u.cm**2
    prefactor = (sigma_0*a/np.pi).value
    av = 4.7e-4
    voigt_constant = av/(np.sqrt(np.pi))

    # voigt profile template for interpolation
    voigt_template = voigt_tasitsiomi(x0, dnu.value)
    voigt_interpolator = interpolate.interp1d(x0, voigt_template, bounds_error=False,\
                        fill_value=(voigt_template[0], voigt_template[-1]))
    
    # remove units for speed
    dnu = dnu.value
    c_kps = c.c.to('km/s').value
    wavelength_range = wavelength_range.to('Angstrom').value
    freq_Lya = freq_Lya.to('Hz').value
    wave_Lya = wave_Lya.to('Angstrom').value
    vz_proper = vz_proper.to('km/s').value

    voigt_tables = []
    rel_idcs_list = []
    z_LoS_highres_list = []
    z_table_constructor_list = []
    dz_constructor_list = []
    all_idcs = np.array(list(range(LoS_DIM)))

    # precompute voigt tables in rest frame of each systemic redshift in the LoS
    for k in range(len(z_LoS)):
        
        dist_to_observer = k*res
        zs = z_LoS[k]
        if dist_to_observer < 50*u.Mpc:
            rel_idcs = all_idcs[:k+1]
        else:
            rel_idcs = all_idcs[k-int(50/res.value):k+1]

        z_LoS_rel = z_LoS[rel_idcs]
        jacobian_rel = jacobian[rel_idcs]

        z_LoS_highres = np.linspace(z_LoS_rel[0], z_LoS_rel[-1], NCELLS_HIRES)

        jacobian_rel = np.interp(z_LoS_highres, z_LoS_rel, jacobian_rel)
        dz = np.diff(z_LoS_highres)

        # these should be log quantities, or at least the first should be
        z_table_constructor_list.append(np.log10(jacobian_rel))
        dz_constructor_list.append(dz)

        rel_idcs_list.append(rel_idcs)

        z_LoS_highres_list.append(z_LoS_highres)

        voigt_table = np.zeros((len(z_LoS_highres), len(wavelength_range)))

        for l in range(len(z_LoS_highres)):
            # interpolate to frame of reference
            velocity_range = c_kps*(wavelength_range*(1+zs)/(1+z_LoS_highres[l])/wave_Lya - 1)
            x_range_stationary = (velocity_range/c_kps)*(freq_Lya/dnu)
            
            voigt_table[l] = voigt_interpolator(x_range_stationary)

        voigt_tables.append(np.log10(voigt_table).astype(np.float16))
        # voigt_tables.append(np.log10(voigt_table))

    voigt_tables = np.array(voigt_tables)
    z_LoS_highres_list = np.array(z_LoS_highres_list)
    z_table_list = np.array(z_table_constructor_list)
    dz_list = np.array(dz_constructor_list)

    rel_idcs_txt = []
    for IDX in rel_idcs_list:
        rel_idcs_txt += list(IDX)
        rel_idcs_txt.append(-1)

    # save the properties
    os.makedirs('./data/absorption_properties', exist_ok=True)
    np.save('./data/absorption_properties/voigt_tables.npy', voigt_tables)
    del voigt_tables
    np.savetxt('./data/absorption_properties/rel_idcs_list.txt', rel_idcs_txt)
    np.save('./data/absorption_properties/z_LoS_highres_list.npy', z_LoS_highres_list)
    np.save('./data/absorption_properties/z_table_list.npy', z_table_list)
    np.save('./data/absorption_properties/dz_list.npy', dz_list)
    np.save('./data/absorption_properties/z_LoS.npy', z_LoS)
    np.save('./data/absorption_properties/z_LoS_edges.npy', z_LoS_edges)
    np.save('./data/absorption_properties/d_LoS_edges.npy', d_LoS_edges.value)
    np.save('./data/absorption_properties/prefactor.npy', prefactor)
    np.save('./data/absorption_properties/n_HI.npy', n_HI)
    