"""
This script reads the lightcone data and saves the xHI, density and velocity fields as numpy arrays.
"""

import numpy as np
import py21cmfast as p21c

if __name__ == "__main__":

    LC = p21c.LightCone.read('./data/lightcones/LC.h5')
    xHI = LC.xH_box
    vz = LC.velocity_z
    density = LC.density

    np.save('./data/absorption_properties/xHI.npy', xHI)
    np.save('./data/absorption_properties/density.npy', density)
    np.save('./data/absorption_properties/vz.npy', vz)
