"""
Generates and saves a PerturbHaloField object.

Samuel Gagnon-Hartman, 6 December 2024
Scuola Normale Superiore, Pisa, Italy
"""


import argparse
import numpy as np

import py21cmfast as p21c
from py21cmfast import cache_tools
from py21cmfast.c_21cmfast import ffi, lib

import logging, os, json
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('21cmFAST')

from timeit import default_timer as timer

from astropy.cosmology import Planck18

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Generate a perturbed halo field')
    parser.add_argument('-m', type=float, default=1e9, help='Minimum halo mass')
    parser.add_argument('-z', type=float, default=7.2, help='Redshift')

    args = parser.parse_args()

    MIN_MASS = args.m
    redshift = args.z

    inputs = p21c.InputParameters.from_template('./sgh24.toml',\
                random_seed=1).evolve_input_structs(SAMPLER_MIN_MASS=MIN_MASS)

    initial_conditions = p21c.compute_initial_conditions(inputs=inputs)

    # how is this so fast????
    pth_field = p21c.determine_halo_list(
        redshift = redshift,
        inputs = inputs,
        initial_conditions = initial_conditions
    )

    os.makedirs(f'./data/halo_fields/z{np.around(redshift, 1)}', exist_ok=True)
    pth_field.write(direc=f'./data/halo_fields/z{np.around(redshift, 1)}/', \
                    fname=f'sgh24_MIN_MASS_{np.around(np.log10(MIN_MASS), 1)}.h5')
