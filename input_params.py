import os
import argparse
import py21cmfast as p21c

parser = argparse.ArgumentParser(description='Build a Coeval Cube')
parser.add_argument('--hires', type=int, default=300, help='DIM in py21cmfast, length of hires grid')
parser.add_argument('--lores', type=int, default=200, help='HII_DIM in py21cmfast, length of lores grid')
parser.add_argument('--boxlen', type=int, default=200, help='BOX_LEN in py21cmfast, size of box')
parser.add_argument('--redshift', type=float, help='central z', default=6.6)
parser.add_argument('--nthreads', type=int, default=4, help='number of OMP threads')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--regenerate', action='store_false')
parser.add_argument('--cache', default='_cache', help='cache folder (set to scratch for submitted jobs)')

parser.add_argument('--exp_filter', type=bool, default=False, help='USE_EXP_FILTER')
parser.add_argument('--avg_below', type=int, default=1, help='AVG_BELOW_SAMPLER')
parser.add_argument('--sampler_min_mass', type=float, default=1e10, help='SAMPLER_MIN_MASS')
parser.add_argument('--avg_below_sampler', type=int, default=1, help='AVG_BELOW_SAMPLER')

parser.add_argument('--use_halo_field', type=bool, default=True, help='USE_HALO_FIELD')

args = parser.parse_args()

if not os.path.exists(args.cache):
    os.mkdir(args.cache)

p21c.config['direc'] = args.cache

if args.regenerate:
    cache_tools.clear_cache(direc=args.cache)

random_seed = args.seed

#set globals(no changes here)
p21c.global_params.HII_FILTER = 0
p21c.global_params.ZPRIME_STEP_FACTOR = 1.02
p21c.global_params.DELTA_R_FACTOR = 1.03
p21c.global_params.MAXHALO_FACTOR = 2
p21c.global_params.HALO_MTURN_FACTOR = 16
p21c.global_params.HALO_SAMPLE_FACTOR = 2

p21c.global_params.AVG_BELOW_SAMPLER = args.avg_below_sampler
p21c.global_params.SAMPLER_MIN_MASS = args.sampler_min_mass
# p21c.global_params.MINIMIZE_MEMORY = 0

#fill param arrays
cosmo_params = p21c.CosmoParams()
user_params = p21c.UserParams(USE_INTERPOLATION_TABLES=True, N_THREADS=args.nthreads, BOX_LEN=args.boxlen, DIM=args.hires,
                                HII_DIM=args.lores, HMF=1, STOC_MINIMUM_Z=args.redshift,
                                INTEGRATION_METHOD_ATOMIC=0, INTEGRATION_METHOD_MINI=0, INTEGRATION_METHOD_HALOS=0)

ap = p21c.AstroParams(SIGMA_STAR=0.5, SIGMA_SFR=0.6)
fo = p21c.FlagOptions(HALO_STOCHASTICITY=args.use_halo_field, USE_HALO_FIELD=args.use_halo_field, 
                    USE_UPPER_STELLAR_TURNOVER=True, USE_MASS_DEPENDENT_ZETA=True, USE_TS_FLUCT=False,
                    INHOMO_RECO=False, PHOTON_CONS=False, USE_EXP_FILTER=args.exp_filter, CELL_RECOMB=args.exp_filter)
# added upper stellar turnover