import py21cmfast as p21c

if __name__ == "__main__":

    min_redshift=5.0
    max_redshift=35.0

    node_redshifts = p21c.get_logspaced_redshifts(min_redshift  = min_redshift,
                             max_redshift  = max_redshift,
                             z_step_factor = 1.02)

    inputs = p21c.InputParameters.from_template('./sgh24.toml',\
                random_seed=1,\
                node_redshifts=node_redshifts).evolve_input_structs(SAMPLER_MIN_MASS=1e8)

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

    # we need to do something about the caches, does cache_files exist?

    lc.save('./data/test_lc.h5', clobber=True)
