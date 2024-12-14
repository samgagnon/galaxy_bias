import py21cmfast as p21c

if __name__ == "__main__":


    inputs = p21c.InputParameters.from_template('./sgh24.toml',\
                random_seed=1).evolve_input_structs(SAMPLER_MIN_MASS=1e8)

    lcn = p21c.RectilinearLightconer.with_equal_cdist_slices(
        min_redshift=inputs.node_redshifts[-1],
        max_redshift=inputs.node_redshifts[0],
        resolution=inputs.user_params.cell_size,
        cosmo=inputs.cosmo_params.cosmo,
        quantities=['brightness_temp', 'density', 'xH_box', 'velocity_z'],
    )

    _, _, _, lc = p21c.exhaust_lightcone(
        lightconer=lcn,
        inputs=inputs,
    )

    # we need to do something about the caches, does cache_files exist?

    lc.save('./data/test_lc.h5', clobber=True)
