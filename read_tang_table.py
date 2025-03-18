import numpy as np

if __name__ == "__main__":
    line_lengths = []

    galaxies = {}

    # Read the table
    with open("tang_table.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            line_len = len(line.strip().split('&'))
            if line_len not in line_lengths:
                print("Line length: ", line_len)
                line_lengths.append(line_len)
            if line_len == 10:
                galaxy_name, ra, dec, MUV, z, flya, ewlya, dvlya, fescB, fescA = line.strip().split('&')
                try:
                    _ = float(ra.strip().replace('$', ''))
                except:
                    continue

                galaxies[galaxy_name] = {}
                galaxies[galaxy_name]['ID'] = galaxy_name.strip()
                galaxies[galaxy_name]['RA'] = float(ra.strip().replace('$', ''))
                galaxies[galaxy_name]['DEC'] = float(dec.strip().replace('$', ''))

                MUV, MUV_err = MUV.split('\\pm')

                galaxies[galaxy_name]['MUV'] = float(MUV.strip().replace('$', ''))
                galaxies[galaxy_name]['MUV_err'] = float(MUV_err.strip().replace('$', ''))
                galaxies[galaxy_name]['z'] = float(z.strip().replace('$', ''))
                f_lya, f_lya_err = flya.split('\\pm')

                f_lya_err = f_lya_err.split('(')[0].strip()

                galaxies[galaxy_name]['f_lya'] = float(f_lya.replace('$', ''))
                galaxies[galaxy_name]['f_lya_err'] = float(f_lya_err.replace('$', ''))

                EW, EW_err = ewlya.split('\\pm')

                galaxies[galaxy_name]['ew_lya'] = float(EW.strip().replace('$', ''))
                galaxies[galaxy_name]['ew_lya_err'] = float(EW_err.strip().replace('$', ''))

                dv, dv_err = dvlya.split('\\pm')

                galaxies[galaxy_name]['dv_lya'] = float(dv.strip().replace('$', ''))
                galaxies[galaxy_name]['dv_lya_err'] = float(dv_err.strip().replace('$', ''))

                fescB, fescB_err = fescB.split('\\pm')

                galaxies[galaxy_name]['fescB'] = float(fescB.strip().replace('$', ''))
                galaxies[galaxy_name]['fescB_err'] = float(fescB_err.strip().replace('$', ''))

                fescA, fescA_err = fescA.split('\\pm')

                galaxies[galaxy_name]['fescA'] = float(fescA.strip().replace('$', ''))
                galaxies[galaxy_name]['fescA_err'] = float(fescA_err.strip().replace('$', '').replace('\\', ''))



            elif line_len == 7:
                galaxy_name, jades_name, m_star, age, AHalpha, EW_OIII_HB, ion_eff = line.strip().split('&')
                try:
                    _ = float(m_star.split('^')[0].strip().replace('$', ''))
                except:
                    continue

                galaxies[galaxy_name]['jades_name'] = jades_name.strip()
                galaxies[galaxy_name]['m_star'] = float(m_star.split('^')[0].strip().replace('$', ''))
                galaxies[galaxy_name]['m_star_up_err'] = float(m_star.split('^')[1].split('_')[0].strip()\
                                                               .replace('$', '').replace('{', '').replace('}', ''))
                galaxies[galaxy_name]['m_star_lo_err'] = float(m_star.split('^')[1].split('_')[1].strip()\
                                                               .replace('$', '').replace('{', '').replace('}', ''))
                galaxies[galaxy_name]['age'] = float(age.split('^')[0].strip().replace('$', ''))
                galaxies[galaxy_name]['age_up_err'] = float(age.split('^')[1].split('_')[0].strip()\
                                                               .replace('$', '').replace('{', '').replace('}', ''))
                galaxies[galaxy_name]['age_lo_err'] = float(age.split('^')[1].split('_')[1].strip()\
                                                               .replace('$', '').replace('{', '').replace('}', ''))
                galaxies[galaxy_name]['AHalpha'] = float(AHalpha.split('^')[0].strip().replace('$', ''))
                galaxies[galaxy_name]['AHalpha_up_err'] = float(AHalpha.split('^')[1].split('_')[0].strip()\
                                                               .replace('$', '').replace('{', '').replace('}', ''))
                galaxies[galaxy_name]['AHalpha_lo_err'] = float(AHalpha.split('^')[1].split('_')[1].strip()\
                                                               .replace('$', '').replace('{', '').replace('}', ''))
                galaxies[galaxy_name]['EW_OIII_HB'] = float(EW_OIII_HB.split('^')[0].strip().replace('$', ''))
                galaxies[galaxy_name]['EW_OIII_HB_up_err'] = float(EW_OIII_HB.split('^')[1].split('_')[0].strip()\
                                                               .replace('$', '').replace('{', '').replace('}', ''))
                galaxies[galaxy_name]['EW_OIII_HB_lo_err'] = float(EW_OIII_HB.split('^')[1].split('_')[1].strip()\
                                                                .replace('$', '').replace('{', '').replace('}', ''))
                galaxies[galaxy_name]['ion_eff'] = float(ion_eff.split('^')[0].strip().replace('$', ''))
                galaxies[galaxy_name]['ion_eff_up_err'] = float(ion_eff.split('^')[1].split('_')[0].strip()\
                                                               .replace('$', '').replace('{', '').replace('}', ''))
                galaxies[galaxy_name]['ion_eff_lo_err'] = float(ion_eff.split('^')[1].split('_')[1].strip()\
                                                                .replace('$', '').replace('{', '').replace('}', '').replace('\\', ''))
    
    # now I have a dictionary object containing all relevant information
    # from this, I wish to create a npy file containing relevant information
    # AB magnitude, redshift, Lya EW

    gal_props = np.zeros((len(galaxies), 12))

    for i, galaxy in enumerate(galaxies):
        if len(galaxy) > 10:
            if galaxy[:4]=='MUSE':
                # MUSE Wide limiting flux 2e-18
                ID = 0
            elif 'vdrop' in galaxy:
                # vdrop
                ID = 2
            elif '_' in galaxy:
                # idrop
                ID = 3
            else:
                # bdrop
                ID = 4
        else:
            # MUSE Deep limiting flux 2e-19
            ID = 1
        
        MUV = galaxies[galaxy]['MUV']
        MUV_err = galaxies[galaxy]['MUV_err']
        z = galaxies[galaxy]['z']
        ew_lya = galaxies[galaxy]['ew_lya']
        ew_lya_err = galaxies[galaxy]['ew_lya_err']
        dv_lya = galaxies[galaxy]['dv_lya']
        dv_lya_err = galaxies[galaxy]['dv_lya_err']
        fescA = galaxies[galaxy]['fescA']
        fescA_err = galaxies[galaxy]['fescA_err']
        fescB = galaxies[galaxy]['fescB']
        fescB_err = galaxies[galaxy]['fescB_err']
        gal_props[i] = np.array([MUV, MUV_err, z, ew_lya, ew_lya_err, dv_lya, dv_lya_err, fescA,\
                                 fescA_err, fescB, fescB_err, ID])
    
    np.save("data/tang24.npy", gal_props)
