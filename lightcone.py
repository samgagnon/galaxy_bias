import py21cmfast as p21c

import logging, os

if __name__ == "__main__":

    # logging and cacheing
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    import argparse

    parser = argparse.ArgumentParser(description='run multiple ionbox')
    parser.add_argument('--cache', default='./data/lc_cache/sgh25/', \
                        help='cache folder (set to scratch for submitted jobs)')

    args = parser.parse_args()

    if not os.path.exists(args.cache):
        os.mkdir(args.cache)

    p21c.config['direc'] = args.cache

    min_redshift=5.0
    max_redshift=35.0

    node_redshifts = p21c.get_logspaced_redshifts(min_redshift  = min_redshift,
                             max_redshift  = max_redshift,
                             z_step_factor = 1.02)

    inputs = p21c.InputParameters.from_template('./sgh25.toml',\
                random_seed=1,\
                node_redshifts=node_redshifts).evolve_input_structs(N_THREADS=4)

    lcn = p21c.RectilinearLightconer.with_equal_cdist_slices(
        min_redshift=inputs.node_redshifts[-1],
        max_redshift=inputs.node_redshifts[0],
        resolution=inputs.user_params.cell_size,
        cosmo=inputs.cosmo_params.cosmo,
        quantities=['brightness_temp', 'density', 'xH_box', 'velocity_z']
    )

    _, _, _, lc = p21c.exhaust_lightcone(
        lightconer=lcn,
        inputs=inputs,
    )

    lc.save('./data/lightcones/sgh25.h5', clobber=True)
