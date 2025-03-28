import os

import numpy as np

def get_tang_data():
    # TANG LAE data
    # https://arxiv.org/pdf/2402.06070
    MUV, MUV_err, z, ew_lya, ew_lya_err, dv_lya, dv_lya_err, fescA, \
        fescA_err, fescB, fescB_err, ID = np.load('../data/tang24.npy').T
    return MUV, MUV_err, z, ew_lya, ew_lya_err, dv_lya, dv_lya_err, \
        fescA, fescA_err, fescB, fescB_err, ID

def get_gaus_mvar(x, y, mean, sigma):
    x_gaus = np.exp(-0.5*((x - mean[0])/sigma[0])**2)/(sigma[0]*np.sqrt(2*np.pi))
    y_gaus = np.exp(-0.5*((y - mean[1])/sigma[1])**2)/(sigma[1]*np.sqrt(2*np.pi))
    gaus_mvar = np.array([x_gaus]*len(y_gaus))*np.array([y_gaus]*len(x_gaus)).T
    return gaus_mvar
    
def dat_to_hist(x, y, dat, dat_err):
    """
    Given a 2D array of data and errors, return the histogram of the data
    """
    hist = np.zeros((len(x), len(y)))
    for datum, datum_err in zip(dat.T, dat_err.T):
        gaus_mvar = get_gaus_mvar(x, y, datum, datum_err)
        gaus_mvar /= gaus_mvar.max()
        hist += gaus_mvar
    hist /= hist.sum()
    return x, y, hist

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    rc = {"font.family" : "serif", 
        "mathtext.fontset" : "stix"}
    plt.rcParams.update(rc) 
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams.update({'font.size': 14})
    presentation = False
    if presentation == True:
        plt.style.use('dark_background')

    MUV, MUV_err, z, ew_lya, ew_lya_err, dv_lya, dv_lya_err, \
        fescA, fescA_err, fescB, fescB_err, ID = get_tang_data()

    muv_space = np.linspace(-24, -16, 100)
    dv_space = np.linspace(0, 1000, 100)
    dv_logspace = 10**np.linspace(0, 3, 100)
    w_space = 10**np.linspace(0, 3, 100)
    fesc_space = np.linspace(0, 1, 100)

    muv_space = muv_space[1:] - (muv_space[1] - muv_space[0])
    dv_space = dv_space[1:] - (dv_space[1] - dv_space[0])
    dv_logspace = dv_logspace[1:] - (dv_logspace[1] - dv_logspace[0])
    w_space = w_space[1:] - (w_space[1] - w_space[0])
    fesc_space = fesc_space[1:] - (fesc_space[1] - fesc_space[0])

    # dv-muv
    data00 = np.array([MUV[ID==0], dv_lya[ID==0]])
    data_err00 = np.array([MUV_err[ID==0], dv_lya_err[ID==0]])
    _, _, h00 = dat_to_hist(muv_space, dv_space, data00, data_err00)

    data10 = np.array([MUV[ID==1], dv_lya[ID==1]])
    data_err10 = np.array([MUV_err[ID==1], dv_lya_err[ID==1]])
    _, _, h10 = dat_to_hist(muv_space, dv_space, data10, data_err10)

    data20 = np.array([MUV[ID>1], dv_lya[ID>1]])
    data_err20 = np.array([MUV_err[ID>1], dv_lya_err[ID>1]])
    _, _, h20 = dat_to_hist(muv_space, dv_space, data20, data_err20)

    # w-muv
    data01 = np.array([MUV[ID==0], ew_lya[ID==0]])
    data_err01 = np.array([MUV_err[ID==0], ew_lya_err[ID==0]])
    _, _, h01 = dat_to_hist(muv_space, w_space, data01, data_err01)

    data11 = np.array([MUV[ID==1], ew_lya[ID==1]])
    data_err11 = np.array([MUV_err[ID==1], ew_lya_err[ID==1]])
    _, _, h11 = dat_to_hist(muv_space, w_space, data11, data_err11)

    data21 = np.array([MUV[ID>1], ew_lya[ID>1]])
    data_err21 = np.array([MUV_err[ID>1], ew_lya_err[ID>1]])
    _, _, h21 = dat_to_hist(muv_space, w_space, data21, data_err21)

    # dv-w
    data02 = np.array([ew_lya[ID==0], dv_lya[ID==0]])
    data_err02 = np.array([ew_lya_err[ID==0], dv_lya_err[ID==0]])
    _, _, h02 = dat_to_hist(w_space, dv_space, data02, data_err02)

    data12 = np.array([ew_lya[ID==1], dv_lya[ID==1]])
    data_err12 = np.array([ew_lya_err[ID==1], dv_lya_err[ID==1]])
    _, _, h12 = dat_to_hist(w_space, dv_space, data12, data_err12)

    data22 = np.array([ew_lya[ID>1], dv_lya[ID>1]])
    data_err22 = np.array([ew_lya_err[ID>1], dv_lya_err[ID>1]])
    _, _, h22 = dat_to_hist(w_space, dv_space,data22, data_err22)

    # fesc-dv
    data03 = np.array([dv_lya[ID==0], fescA[ID==0]])
    data_err03 = np.array([dv_lya_err[ID==0], fescA_err[ID==0]])
    _, _, h03 = dat_to_hist(dv_logspace, fesc_space, data03, data_err03)

    data13 = np.array([dv_lya[ID==1], fescA[ID==1]])
    data_err13 = np.array([dv_lya_err[ID==1], fescA_err[ID==1]])
    _, _, h13 = dat_to_hist(dv_logspace, fesc_space, data13, data_err13)

    data23 = np.array([dv_lya[ID>1], fescA[ID>1]])
    data_err23 = np.array([dv_lya_err[ID>1], fescA_err[ID>1]])
    _, _, h23 = dat_to_hist(dv_logspace, fesc_space, data23, data_err23)

    # fesc-w
    data04 = np.array([ew_lya[ID==0], fescA[ID==0]])
    data_err04 = np.array([ew_lya_err[ID==0], fescA_err[ID==0]])
    _, _, h04 = dat_to_hist(w_space, fesc_space, data04, data_err04)

    data14 = np.array([ew_lya[ID==1], fescA[ID==1]])
    data_err14 = np.array([ew_lya_err[ID==1], fescA_err[ID==1]])
    _, _, h14 = dat_to_hist(w_space, fesc_space, data14, data_err14)

    data24 = np.array([ew_lya[ID>1], fescA[ID>1]])
    data_err24 = np.array([ew_lya_err[ID>1], fescA_err[ID>1]])
    _, _, h24 = dat_to_hist(w_space, fesc_space, data24, data_err24)

    os.makedirs('../data/muse_hist', exist_ok=True)
    np.save('../data/muse_hist/hist00.npy', h00)
    np.save('../data/muse_hist/hist10.npy', h10)
    np.save('../data/muse_hist/hist20.npy', h20)
    np.save('../data/muse_hist/hist01.npy', h01)
    np.save('../data/muse_hist/hist11.npy', h11)
    np.save('../data/muse_hist/hist21.npy', h21)
    np.save('../data/muse_hist/hist02.npy', h02)
    np.save('../data/muse_hist/hist12.npy', h12)
    np.save('../data/muse_hist/hist22.npy', h22)
    np.save('../data/muse_hist/hist03.npy', h03)
    np.save('../data/muse_hist/hist13.npy', h13)
    np.save('../data/muse_hist/hist23.npy', h23)
    np.save('../data/muse_hist/hist04.npy', h04)
    np.save('../data/muse_hist/hist14.npy', h14)
    np.save('../data/muse_hist/hist24.npy', h24)