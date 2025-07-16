import os

import numpy as np
import py21cmfast as p21c

from tqdm import tqdm

from astropy.cosmology import Planck18, z_at_value
from astropy import units as u
from astropy.constants import c, k_B, m_p, e, m_e

from scipy.integrate import trapezoid
from scipy.special import gamma, erf
from scipy.optimize import differential_evolution

import matplotlib.pyplot as plt
rc = {"font.family" : "serif", 
    "mathtext.fontset" : "stix"}
plt.rcParams.update(rc) 
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams.update({'font.size': 14})
import matplotlib as mpl
from matplotlib import patches
label_size = 20
font_size = 30
mpl.rcParams['xtick.labelsize'] = label_size 
mpl.rcParams['ytick.labelsize'] = label_size

presentation = False  # Set to True for presentation style
# presentation = True  # Set to True for presentation style
if presentation == True:
    plt.style.use('dark_background')
    color = 'white'
    color1 = 'cyan'
else:
    color = 'black'
    color1 = 'orange'

def get_beta_bouwens14(muv):
    # https://arxiv.org/pdf/1306.2950
    return -2.05 + -0.2*(muv+19.5)

muv_space = np.linspace(-22, -16, 1000)
log10w_space = np.linspace(0.5, 3, 1000)

NSAMPLES = 500
MEAN_MUV = -18
STD_MUV = 1.0
MEAN_W = 2.0
STD_W = 0.2
muv_samples = np.random.normal(MEAN_MUV, STD_MUV, NSAMPLES)
log10w_samples = np.random.normal(MEAN_W, STD_W, NSAMPLES)
beta_samples = get_beta_bouwens14(muv_samples)
f_samples = ((10**log10w_samples) / \
             (4*np.pi*Planck18.luminosity_distance(5.0).to('cm').value**2)) * \
             (2.47e15/1215.67) * (10 ** (0.4*(51.6 - muv_samples))) * \
             (1215.67/1500)**(beta_samples + 2)

w_select = log10w_samples > np.log10(80)
muv_select = muv_samples < -18
f_select = f_samples > 2e-17
select = w_select & muv_select & f_select

w_minf = (1215.67/2.47e15) * (4*np.pi*Planck18.luminosity_distance(5.0).to('cm').value**2) * \
          2e-17 * (10 ** (-0.4*(51.6 - muv_space))) * \
          (1215.67/1500)**(-1*get_beta_bouwens14(muv_space) - 2)

e01 = patches.Ellipse((MEAN_MUV, MEAN_W), 2*STD_MUV, 2*STD_W,
                     angle=0, linewidth=0, color=color1, alpha=0.2, fill=True, zorder=2)
e02 = patches.Ellipse((MEAN_MUV, MEAN_W), 4*STD_MUV, 4*STD_W,
                     angle=0, linewidth=0, color=color1, alpha=0.2, fill=True, zorder=2)
e03 = patches.Ellipse((MEAN_MUV, MEAN_W), 6*STD_MUV, 6*STD_W,
                     angle=0, linewidth=0, color=color1, alpha=0.2, fill=True, zorder=2)

e11 = patches.Ellipse((MEAN_MUV-1, MEAN_W-0.1), 2*STD_MUV, 2*STD_W,
                     angle=20, linewidth=0, color=color1, alpha=0.2, fill=True, zorder=2)
e12 = patches.Ellipse((MEAN_MUV-1, MEAN_W-0.1), 4*STD_MUV, 4*STD_W,
                     angle=20, linewidth=0, color=color1, alpha=0.2, fill=True, zorder=2)
e13 = patches.Ellipse((MEAN_MUV-1, MEAN_W-0.1), 6*STD_MUV, 6*STD_W,
                     angle=20, linewidth=0, color=color1, alpha=0.2, fill=True, zorder=2)


mean_obs_w = np.mean(log10w_samples[select])
mean_obs_muv = np.mean(muv_samples[select])
std_obs_w = np.std(log10w_samples[select])
std_obs_muv = np.std(muv_samples[select])

e31 = patches.Ellipse((mean_obs_muv, mean_obs_w), 2*std_obs_muv, 2*std_obs_w,
                     angle=20, linewidth=0, color=color1, alpha=0.2, fill=True, zorder=2)
e32 = patches.Ellipse((mean_obs_muv, mean_obs_w), 4*std_obs_muv, 4*std_obs_w,
                     angle=20, linewidth=0, color=color1, alpha=0.2, fill=True, zorder=2)
e33 = patches.Ellipse((mean_obs_muv, mean_obs_w), 6*std_obs_muv, 6*std_obs_w,
                     angle=20, linewidth=0, color=color1, alpha=0.2, fill=True, zorder=2)

e41 = patches.Ellipse((mean_obs_muv, mean_obs_w), 2*std_obs_muv, 2*std_obs_w,
                     angle=20, linewidth=0, color=color1, alpha=0.2, fill=True, zorder=2)
e42 = patches.Ellipse((mean_obs_muv, mean_obs_w), 4*std_obs_muv, 4*std_obs_w,
                     angle=20, linewidth=0, color=color1, alpha=0.2, fill=True, zorder=2)
e43 = patches.Ellipse((mean_obs_muv, mean_obs_w), 6*std_obs_muv, 6*std_obs_w,
                     angle=20, linewidth=0, color=color1, alpha=0.2, fill=True, zorder=2)


# fig, axs = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)

# axs.plot(muv_samples[select], log10w_samples[select], 'o',
#              markersize=5, alpha=1.0, color=color, label='Sampled points')
# # axs.plot(muv_samples[~select], log10w_samples[~select], 'o',
# #              markersize=5, alpha=0.2, color=color, label='Sampled points')
# axs.set_xlabel(r'${\rm M}_{\rm UV}$', fontsize=font_size)
# axs.set_ylabel(r'$\log_{10}W_{\rm emerg}^{\rm Ly\alpha}$', fontsize=font_size)
# axs.set_xlim(-22, -16)
# axs.set_ylim(1.5, 3)
# plt.show()

# fig, axs = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)

# axs.plot(muv_samples[select], log10w_samples[select], 'o',
#              markersize=5, alpha=1.0, color=color, label='Sampled points')
# axs.add_patch(e31)
# axs.add_patch(e32)
# axs.add_patch(e33)
# axs.set_xlabel(r'${\rm M}_{\rm UV}$', fontsize=font_size)
# axs.set_ylabel(r'$\log_{10}W_{\rm emerg}^{\rm Ly\alpha}$', fontsize=font_size)
# axs.set_xlim(-22, -16)
# axs.set_ylim(1.5, 3)
# plt.show()

# plt.close()


# fig, axs = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)

# axs.plot(muv_samples[select], log10w_samples[select], 'o',
#              markersize=5, alpha=1.0, color=color, label='Sampled points')
# axs.plot(muv_space, np.log10(w_minf), '--', color=color, label='Minimum flux limit')
# axs.axvline(-18, color=color, linestyle='--', linewidth=2, label=r'${\rm M}_{\rm UV}=-18$')
# axs.axhline(np.log10(80), color=color, linestyle='--', linewidth=2, label=r'$\log_{10}W_{\rm emerg}^{\rm Ly\alpha}=2$')
# axs.add_patch(e41)
# axs.add_patch(e42)
# axs.add_patch(e43)
# axs.set_xlabel(r'${\rm M}_{\rm UV}$', fontsize=font_size)
# axs.set_ylabel(r'$\log_{10}W_{\rm emerg}^{\rm Ly\alpha}$', fontsize=font_size)
# axs.set_xlim(-22, -16)
# axs.set_ylim(1.5, 3)
# plt.show()
# quit()

fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True, constrained_layout=True)

axs[0].plot(muv_samples[select], log10w_samples[select], 'o',
             markersize=5, alpha=1.0, color=color, label='Sampled points')
axs[0].plot(muv_samples[~select], log10w_samples[~select], 'o',
             markersize=5, alpha=0.2, color=color, label='Sampled points')
axs[0].plot(muv_space, np.log10(w_minf), '--', color=color, label='Minimum flux limit')
axs[0].axvline(-18, color=color, linestyle='--', linewidth=2, label=r'${\rm M}_{\rm UV}=-18$')
axs[0].axhline(np.log10(80), color=color, linestyle='--', linewidth=2, label=r'$\log_{10}W_{\rm emerg}^{\rm Ly\alpha}=2$')
axs[0].add_patch(e01)
axs[0].add_patch(e02)
axs[0].add_patch(e03)
axs[0].set_xlabel(r'${\rm M}_{\rm UV}$', fontsize=font_size)
axs[0].set_ylabel(r'$\log_{10}W_{\rm emerg}^{\rm Ly\alpha}$', fontsize=font_size)
axs[0].set_xlim(-22, -16)
axs[0].set_ylim(1.5, 3)

axs[1].plot(muv_samples[select], log10w_samples[select], 'o',
             markersize=5, alpha=1.0, color=color, label='Sampled points')
axs[1].plot(muv_samples[~select], log10w_samples[~select], 'o',
             markersize=5, alpha=0.2, color=color, label='Sampled points')
axs[1].plot(muv_space, np.log10(w_minf), '--', color=color, label='Minimum flux limit')
axs[1].axvline(-18, color=color, linestyle='--', linewidth=2, label=r'${\rm M}_{\rm UV}=-18$')
axs[1].axhline(np.log10(80), color=color, linestyle='--', linewidth=2, label=r'$\log_{10}W_{\rm emerg}^{\rm Ly\alpha}=2$')
axs[1].add_patch(e11)
axs[1].add_patch(e12)
axs[1].add_patch(e13)
axs[1].set_xlabel(r'${\rm M}_{\rm UV}$', fontsize=font_size)
axs[1].set_xlim(-22, -16)
axs[1].set_ylim(1.5, 3)
# plt.show()
savedir = '/mnt/c/Users/sgagn/Documents/phd/lyman_alpha/figures/'
savedir += 'pedagogy/'
os.makedirs(savedir, exist_ok=True)
plt.savefig(f'{savedir}spurious_vs_true_correlation.pdf', dpi=300)