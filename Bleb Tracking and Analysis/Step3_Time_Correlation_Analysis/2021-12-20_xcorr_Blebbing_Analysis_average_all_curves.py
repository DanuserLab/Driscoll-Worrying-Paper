# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 14:03:42 2021

@author: fyz11
"""

def mkdir(directory):

    import os

    if not os.path.exists(directory):
        os.makedirs(directory)

    return []


def plot_pair_signal(signal1, signal2, Ts=1,
                     color1='k', color2='r',
                     xlabel1=None, xlabel2=None,
                     ylabel1=None, ylabel2=None,
                     lw=2,
                     figsize=(8,5),
                     savefile=None): # if there is a savefolder then we will save to this automatically.

    import pylab as plt

    data1 = np.squeeze(signal1);
    data1_smooth = als_smooth_1D(data1, lam=1e3, p=0.25, niter=10)
    data2 = np.squeeze(signal2)
    data2_smooth = als_smooth_1D(data2, lam=1e3, p=0.25, niter=10)
    t = np.arange(len(data1)) * Ts - len(data1)//2 * Ts # so we center the time.

    fig, ax1 = plt.subplots(figsize=figsize)
    color = color1
    ax1.set_xlabel(xlabel1, fontsize=16, fontname='Arial')
    ax1.set_ylabel(ylabel1, color=color, fontsize=16, fontname='Arial')
    ax1.plot(t, data1, color=color)
    # ax1.vlines(t[len(t)//2], .5, 2.0, color='k', linestyles='dashed')
    ax1.plot(t, data1_smooth, '-.', color=color, lw=2)
    ax1.tick_params(axis='y', labelcolor=color, labelsize=18)
    ax1.tick_params(axis='x', labelcolor=color, labelsize=18)
    # ax1.set_ylim([0,2.0])
    for tick in ax1.xaxis.get_major_ticks():
        tick.label1.set_fontsize(14)
        tick.label1.set_fontname('Arial')
    for tick in ax1.yaxis.get_major_ticks():
        tick.label1.set_fontsize(14)
        tick.label1.set_fontname('Arial')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = color2
    ax2.set_ylabel(ylabel2, color=color, fontsize=16, fontname='Arial')  # we already handled the x-label with ax1
    ax2.plot(t, data2, color=color)
    ax2.plot(t, data2_smooth, '-.', color=color, lw=2)
    # ax2.yaxis.set_tick_params(labelsize=18, color='r')
    ax2.tick_params(axis='y', labelcolor=color, labelsize=18)
    # ax2.set_yticklabels(ax2, fontname='Arial')
    # # ax2.tick_params(axis='x', labelcolor=color, labelsize=18)
    for tick in ax2.yaxis.get_major_ticks():
        # tick.label.set_fontsize(14)
        # tick.label.set_fontname('Arial')
        # tick.label1.set_fontsize(14)
        # tick.label1.set_fontname('Arial')
        tick.label2.set_fontsize(14)
        tick.label2.set_fontname('Arial')
    # ax2.set_ylim([0,1.5])

    ax1.tick_params(length=10)
    ax2.tick_params(length=10)
    # plt.legend()
    # ax1.set_xticklabels(fontname='Arial', fontsize=18)
    # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    if savefile is not None:
        plt.savefig(savefile, dpi=300, bbox_inches='tight')
    plt.show()

    return []


def plot_pair_mean_signal(signal1, signal2,
                          stderr1, stderr2,
                          t,
                          color1='k', color2='r',
                          xlabel1=None, xlabel2=None,
                          ylabel1=None, ylabel2=None,
                          lw=2,
                          alpha=.5, 
                          figsize=(8,5),
                          savefile=None): # if there is a savefolder then we will save to this automatically.

    import pylab as plt

    data1 = np.squeeze(signal1);
    # data1_smooth = als_smooth_1D(data1, lam=1e3, p=0.25, niter=10)
    data2 = np.squeeze(signal2)
    # data2_smooth = als_smooth_1D(data2, lam=1e3, p=0.25, niter=10)
    # t = np.arange(len(data1)) * Ts - len(data1)//2 * Ts # so we center the time.

    fig, ax1 = plt.subplots(figsize=figsize)
    color = color1
    ax1.set_xlabel(xlabel1, fontsize=16, fontname='Arial')
    ax1.set_ylabel(ylabel1, color=color, fontsize=16, fontname='Arial')
    ax1.plot(t, data1, color=color)
    ax1.fill_between(t, data1 - stderr1, data1+stderr1, color=color1, alpha=alpha)
    # ax1.vlines(t[len(t)//2], .5, 2.0, color='k', linestyles='dashed')
    # ax1.plot(t, data1_smooth, '-.', color=color, lw=2)
    ax1.tick_params(axis='y', labelcolor=color, labelsize=18)
    ax1.tick_params(axis='x', labelcolor=color, labelsize=18)
    # ax1.set_ylim([0,2.0])
    for tick in ax1.xaxis.get_major_ticks():
        tick.label1.set_fontsize(14)
        tick.label1.set_fontname('Arial')
    for tick in ax1.yaxis.get_major_ticks():
        tick.label1.set_fontsize(14)
        tick.label1.set_fontname('Arial')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = color2
    ax2.set_ylabel(ylabel2, color=color, fontsize=16, fontname='Arial')  # we already handled the x-label with ax1
    ax2.plot(t, data2, color=color)
    ax2.fill_between(t, data2 - stderr2, data2+stderr2, color=color2, alpha=alpha)
    # ax2.plot(t, data2_smooth, '-.', color=color, lw=2)
    # ax2.yaxis.set_tick_params(labelsize=18, color='r')
    ax2.tick_params(axis='y', labelcolor=color, labelsize=18)
    # ax2.set_yticklabels(ax2, fontname='Arial')
    # # ax2.tick_params(axis='x', labelcolor=color, labelsize=18)
    for tick in ax2.yaxis.get_major_ticks():
        # tick.label.set_fontsize(14)
        # tick.label.set_fontname('Arial')
        # tick.label1.set_fontsize(14)
        # tick.label1.set_fontname('Arial')
        tick.label2.set_fontsize(14)
        tick.label2.set_fontname('Arial')
    # ax2.set_ylim([0,1.5])

    ax1.tick_params(length=10)
    ax2.tick_params(length=10)
    # plt.legend()
    # ax1.set_xticklabels(fontname='Arial', fontsize=18)
    # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    if savefile is not None:
        plt.savefig(savefile, dpi=300, bbox_inches='tight')
    plt.show()

    return []


# add the asymmetric least squares smoothing algorithm.
def baseline_als(y, lam, p, niter=10):
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

def als_smooth_1D(ysig, lam=1, p=0.5, niter=10):

    # pad the signal to fit.
    N = len(ysig)
    ysig_ = np.hstack([ysig[::-1], ysig, ysig[::-1]])
    ysig_smooth = baseline_als(ysig_, lam=lam, p=p, niter=niter)

    return ysig_smooth[N:-N]


def univariate_spl(x,y, xnew, s=0, k=1):

    from scipy import interpolate

    tck = interpolate.splrep(x, y, s=s, k=k)
    ynew = interpolate.splev(xnew, tck, der=0)

    return xnew, ynew


def parse_movie_regions(namelist, splitkey='_', splitpos=0):

    import numpy as np 
    movies = []
    for name in namelist:
        pre = name.split(splitkey)[splitpos]
        try:
            movie = int(pre.split('cell')[1])
        except:
            movie = int(pre.split('movie')[1])    
        movies.append(movie)
    return np.hstack(movies)
    
def find_Ts_movie_region(movies, Ts_table):
    
    Ts_table_movies = Ts_table['Filename'].values
    Ts_table_Ts = Ts_table['Ts(s)'].values
    
    Ts_out = []
    for movie in movies:
        movie_lookup = movie.split('_')[0]
        Ts_lookup = Ts_table_Ts[Ts_table_movies==movie_lookup]
        Ts_lookup = float(Ts_lookup)
        Ts_out.append(Ts_lookup)
    Ts_out = np.hstack(Ts_out)
    
    return Ts_out
    

def norm_xcorr(x, y, eps=1e-12): 
    
    from scipy.signal import correlate
    "Plot cross-correlation (full) between two signals."
    N = max(len(x), len(y)) 
    n = min(len(x), len(y)) 

    if N == len(y): 
        lags = np.arange(-N + 1, n) 
    else: 
        lags = np.arange(-n + 1, N) 
    
    # i think we do need to demean. 
    c = correlate( (x - np.nanmean(x)) / (np.nanstd(x)+eps), (y - np.nanmean(y)) / (np.nanstd(y)+eps), 'full') 

    return lags, c/n


if __name__=="__main__":

    import numpy as np
    import pylab as plt
    import scipy.io as spio
    import os
    import glob
    import scipy.stats as spstats
    import pandas as pd

    savefolder = 'C:\\Users\\fyz11\\Documents\\Work\\Projects\\Danuser-3D Causality\\Data\\Etai\\Worrying\\2021-10-12_Analysis\\Responsive_Cells\\Plots_final_2021-12-20'
    mkdir(savefolder)
    # savefolder = 'C:\\Users\\fyz11\\Documents\\Work\\Projects\\Danuser-3D Causality\\Data\\Etai\\Worrying\\2021-10-12_Analysis\\NonResponsive_Cells\\Plots_final_2021-12-20'
    # mkdir(savefolder)


    # these are the statistics files.
    rootfolder = 'C:\\Users\\fyz11\\Documents\\Work\\Projects\\Danuser-3D Causality\\Data\\Etai\\Worrying\\2021-10-12_Analysis'
    all_datafolders = [os.path.join(rootfolder, 'Responsive_Cells', 'responsive_cells_combined_time-series.mat'),
                       os.path.join(rootfolder, 'Responsive_Cells', '2021-12-19_responsive_cells_combined_time-series_2.mat'),
                       os.path.join(rootfolder, 'Responsive_Cells', '2021-12-19_responsive_cells_combined_time-series_3.mat'),
                       os.path.join(rootfolder, 'Responsive_Cells', '2021-12-19_responsive_cells_combined_time-series_4.mat')]

    # time sampling files.
    Ts_folder = 'C:\\Users\\fyz11\\Documents\\Work\\Projects\\Danuser-3D Causality\\Manuscripts\\2022_Worrying-Paper'
    all_Ts_files = [os.path.join(Ts_folder, 'Ts-firstdataset_responsive.xlsx'),
                    os.path.join(Ts_folder, 'Ts-two_color_20211215_tiam1_day2.xlsx'),
                    os.path.join(Ts_folder, 'Ts-two_color_20211212_tagRFP_tiam1_Halo_arp3_J646_day2.xlsx'),
                    os.path.join(Ts_folder, 'Ts-two_color_20211218_day1.xlsx')]


    # master dataset table in order to tell us which movies to plot and analyse for each dataset.
    master_dataset_file = os.path.join(Ts_folder, 'Datasets.xlsx')
    master_datatable = pd.read_excel(master_dataset_file, sheet_name='ResponsiveCells(Tiam1)')
    # movies_to_analyse = master_datatable['Movies to Analyse for RFP']
    movies_to_analyse = master_datatable['Movies to Analyse for Arp3']
    
# =============================================================================
#     Set up some analytical parameters.
# =============================================================================
    trim_samples = 20

# =============================================================================
#     Go through and load in all the signals to analyse.
# =============================================================================

    # containers for collecting everything.
    all_resp_cells_R = []
    # all_resp_cells_localization = []
    all_resp_cells_arp = []
    all_resp_cells_movies = []
    all_resp_cells_maxsize = []
    
    all_Ts = []
    
    # iterate over all the datasets. 
    for ii in np.arange(len(all_datafolders))[:]:

        # load all the data for each file.
        datafile = all_datafolders[ii]
        cells = spio.loadmat(datafile)

        # load all the signals to analyse.
        resp_cells_R = np.squeeze(cells['bleb_radius'])
        # resp_cells_localization = np.squeeze(cells['bleb_norm_RFP'])
        resp_cells_Arp = np.squeeze(cells['bleb_norm_Arp'])
        resp_cells_movies = cells['ROI_names']
        resp_cells_maxsize = np.squeeze(cells['bleb_size_times'])

        # grab the Ts for all of these and save these in the global container. 
        Ts_movie = pd.read_excel(all_Ts_files[ii])
    
        # parse the region names to get which movie we are getting from.
        resp_movies = parse_movie_regions(resp_cells_movies, splitkey='_', splitpos=0)
        
        """
        match the resp_movies to the Ts_movie
        """
        Ts_movies = find_Ts_movie_region(resp_cells_movies, Ts_movie)
        
        # analyse_movies
        analyse_movies = movies_to_analyse[ii] # turn into a movie array
        
        """
        Take those we are going to keep and put them in the master container. 
        """
        if '-' not in analyse_movies:
            # then at least a subset of these will be used for analysis. 
            analyse_movies = np.hstack(list(analyse_movies.split(','))).astype(np.int64)
            use_regions = np.hstack([resp_movie in analyse_movies for resp_movie in resp_movies])
            
            ######
            # use the use_regions to mask the reequired entities.
            ######
            all_resp_cells_R.append(resp_cells_R[use_regions])
            # all_resp_cells_localization.append(resp_cells_localization[use_regions])
            all_resp_cells_arp.append(resp_cells_Arp[use_regions])
            all_resp_cells_movies.append(resp_cells_movies[use_regions])
            all_resp_cells_maxsize.append(resp_cells_maxsize[use_regions])
            
            all_Ts.append(Ts_movies[use_regions])
                    
            
# =============================================================================
#    Get the unique movies!. that we used!.          
# =============================================================================
    
    def parse_uniq_conditions(fnames):
        unique_fnames = []
        for ii in np.arange(len(fnames)):
            if len(unique_fnames) == 0:
                unique_fnames.append(fnames[ii])
            else:
                if fnames[ii] == unique_fnames[-1]:
                    continue
                else:
                    unique_fnames.append(fnames[ii])
        return np.hstack(unique_fnames)        
    
    all_movies = np.hstack([mm.split('_Region')[0] for mm in np.hstack(all_resp_cells_movies)])
    all_uniq_movies = parse_uniq_conditions(all_movies) 
    
    print('N_regions:, ', len(np.hstack(all_resp_cells_movies)))
    print('N_movies:, ', len(all_uniq_movies))
                    
    
# =============================================================================
#     1. iterate all the signals and in particular the Ts to resample the signals to the same time period. 
# =============================================================================

    # find the temporal sampling. 
    t_all = []    
    # double layer the iteration 
    for ii in np.arange(len(all_resp_cells_R))[:]:
        resp_cells_R_ii = all_resp_cells_R[ii]
        for jj in np.arange(len(resp_cells_R_ii)):
            signal_ii = np.squeeze(resp_cells_R_ii[jj])
            t = np.arange(len(signal_ii)) * all_Ts[ii][jj] - len(signal_ii)//2*all_Ts[ii][jj] # in terms of seconds.
            t_all.append(t)
    
    # determine the resampled timeseries.
    t_min = np.min(np.hstack(t_all))
    t_max = np.max(np.hstack(t_all))
    sample_no = np.hstack([len(tt) for tt in t_all])
    n_samples = int(spstats.mode(sample_no)[0]) # is this the best option? # will this adversely effect the lag analysis?  
    # n_samples = int(np.min(sample_no))
    t_new = np.linspace(t_min, t_max, n_samples)


# =============================================================================
#     2. Linearly resample all the sequences 
# =============================================================================

    # these are now the final time series for averaging and correlation lag analysis.
    corr_nonresp_lags = []
    # resp_blebs_loc = []
    corr_nonresp_blebs= []

    for ii in np.arange(len(all_resp_cells_R)):
        
        resp_cells_R = all_resp_cells_R[ii]
        # resp_cells_localization = all_resp_cells_localization[ii]
        resp_cells_arp = all_resp_cells_arp[ii]
        resp_cells_maxsize = all_resp_cells_maxsize[ii]
        Ts = all_Ts[ii]
        
        for jj in np.arange(len(resp_cells_R))[:]:

            cell_size = np.squeeze(resp_cells_maxsize[jj])
            signal_ii = np.squeeze(resp_cells_R[jj])
            arp_ii = np.squeeze(resp_cells_arp[jj])
            # localization_ii = np.squeeze(resp_cells_localization[jj])
        
            # arp_ii = np.squeeze(resp_cells_Arp[ii])
            t = np.arange(len(signal_ii)) * Ts[jj] - len(signal_ii)//2*Ts[jj] # in terms of seconds.
    
            # we have to resample all of these, using linear interpolation. 
            signal_ii_new = univariate_spl(t, signal_ii, t_new, s=0., k=1)[1]
            # localization_ii_new = univariate_spl(t, localization_ii, t_new, s=0, k=1)[1]
            arp_ii_new = univariate_spl(t, arp_ii, t_new, s=0)[1]
    
    
            """
            Cross correlation analysis -> and then we put into the container. 
            """
            # get the xcorr between two signals? 
            xcorr_lag, xcorr_signal_arp = norm_xcorr(signal_ii_new, 
                                                     arp_ii_new, 
                                                     eps=1e-12)
            
            plt.figure()
            plt.plot(xcorr_lag, xcorr_signal_arp)
            plt.show()
    
            corr_nonresp_lags.append(xcorr_lag)
            corr_nonresp_blebs.append(xcorr_signal_arp)

    """
    # stack everything together 
    """
    corr_nonresp_lags = np.vstack(corr_nonresp_lags)
    # resp_blebs_arp = np.vstack(resp_blebs_loc)
    corr_nonresp_blebs = np.vstack(corr_nonresp_blebs)


# # =============================================================================
# #     Plotting the xcorr 
# # =============================================================================

    mean_func = np.nanmean
    # mean_func = np.nanmedian

    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(corr_nonresp_lags[0] * (t_new[1]-t_new[0]), 
            mean_func(corr_nonresp_blebs, axis=0), 'k-')
    # fill 
    ax.fill_between(corr_nonresp_lags[0] * (t_new[1]-t_new[0]), 
                    mean_func(corr_nonresp_blebs, axis=0) - np.nanstd(mean_func(corr_nonresp_blebs, axis=0)), 
                    mean_func(corr_nonresp_blebs, axis=0) + np.nanstd(mean_func(corr_nonresp_blebs, axis=0)), alpha=0.5, color='gray')
    ax.vlines(0, -1, 1, linestyles = 'dashed', color='k')
    plt.tick_params(length=10, right=True)
    plt.xlabel('Time Lag (s)', fontsize=16, fontname='Arial')
    plt.ylabel('Correlation', fontsize=16, fontname='Arial')
    plt.xticks(fontname='Arial', fontsize=14)
    plt.yticks(fontname='Arial', fontsize=14)
    plt.ylim([-1,1])
    # plt.xlim([-200,200])
    plt.savefig(os.path.join(savefolder, 
                             'all_blebR-vs-arp3_xcorr_full.png'), dpi=300, bbox_inches='tight')
    plt.show()


    # create the zoom in. 
    fig, ax = plt.subplots(figsize=(5,5))
    ax.plot(corr_nonresp_lags[0] * (t_new[1]-t_new[0]), 
            mean_func(corr_nonresp_blebs, axis=0), 'k-')
    # fill 
    ax.fill_between(corr_nonresp_lags[0] * (t_new[1]-t_new[0]), 
                    mean_func(corr_nonresp_blebs, axis=0) - np.nanstd(mean_func(corr_nonresp_blebs, axis=0)), 
                    mean_func(corr_nonresp_blebs, axis=0) + np.nanstd(mean_func(corr_nonresp_blebs, axis=0)), alpha=0.5, color='gray')
    ax.vlines(0, -1, 1, linestyles = 'dashed', color='k')
    plt.tick_params(length=10, right=True)
    plt.xlabel('Time Lag (s)', fontsize=16, fontname='Arial')
    plt.ylabel('Correlation', fontsize=16, fontname='Arial')
    plt.xticks(fontname='Arial', fontsize=14)
    plt.yticks(fontname='Arial', fontsize=14)
    plt.ylim([-1,1])
    plt.xlim([-100,100])
    plt.savefig(os.path.join(savefolder, 
                             'all_blebR-vs-arp3_xcorr_zoomin.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    
    # save the compiled statistics now for potentially coplotting 
    savematfile = os.path.join(savefolder, 'xcorr_stats_responsive_vs_arp3.mat')
    spio.savemat(savematfile, 
                 {'xcorr_all': corr_nonresp_blebs,
                  'lags_all': corr_nonresp_lags, 
                  'movies_all': np.hstack(all_resp_cells_movies), 
                  't_new': t_new})
    
    
# all_resp_cells_R = []
#     # all_resp_cells_localization = []
#     all_resp_cells_arp = []
#     all_resp_cells_movies = []
#     all_resp_cells_maxsize = []



#     # derive the mean signals and standard deviation 
#     # mean_resp_blebs = np.nanmean(resp_blebs, axis=0)[trim_samples:-trim_samples]
#     # mean_resp_blebs_arp = np.nanmean(resp_blebs_arp, axis=0)[trim_samples:-trim_samples]
#     mean_resp_blebs = np.nanmedian(resp_blebs, axis=0)[trim_samples:-trim_samples]
#     mean_resp_blebs_arp = np.nanmedian(resp_blebs_arp, axis=0)[trim_samples:-trim_samples]
#     std_resp_blebs = np.nanstd(resp_blebs, axis=0)[trim_samples:-trim_samples]
#     std_resp_blebs_arp = np.nanstd(resp_blebs_arp, axis=0)[trim_samples:-trim_samples]
#     t_new = t_new[trim_samples:-trim_samples]
    
#     # plot the mean signal
#     fig, ax = plt.subplots(figsize=(8,5))
#     ax.plot(t_new, mean_resp_blebs, 'k-')
#     ax.plot(t_new, mean_resp_blebs_arp, 'r-')
#     # ax.plot(t_new, np.nanmean(resp_blebs_arp, axis=0), 'g-')
#     plt.show()


#     """
#     blebs vs arp3
#     """
#     # savefile_average_curves_plot = os.path.join(savefolder,
#                                                 # 'all_blebR-vs-arp3_meancurve.png') # as png for now.
#     savefile_average_curves_plot = os.path.join(savefolder,
#                                                 'all_blebR-vs-arp3_mediancurve.png') # as png for now.
#     plot_pair_mean_signal(mean_resp_blebs,
#                           mean_resp_blebs_arp,
#                            # spstats.sem(resp_blebs, axis=0, nan_policy='omit')[trim_samples:-trim_samples],
#                            # spstats.sem(resp_blebs_loc, axis=0, nan_policy='omit')[trim_samples:-trim_samples],
#                            std_resp_blebs,
#                            std_resp_blebs_arp,
#                           t=t_new,
#                           color1='k', color2='g',
#                           xlabel1='Time (s)', xlabel2='Time (s)',
#                           ylabel1='Bleb Radius (um)', ylabel2='Arp3 (A.U)',
#                           lw=2,
#                           alpha=.25,
#                           figsize=(8,5),
#                           savefile=savefile_average_curves_plot)

#     """
#     blebs vs arp3
#     """
#     savefile_average_curves_plot = os.path.join(savefolder_response,
#                                                 'all_blebR-vs-arp3_avgcurve.png') # as png for now.
#     plot_pair_mean_signal(np.nanmean(resp_blebs, axis=0)[trim_samples:-trim_samples],
#                           np.nanmean(resp_blebs_arp, axis=0)[trim_samples:-trim_samples],
#                           # spstats.sem(nonresp_blebs, axis=0, nan_policy='omit')[trim_samples:-trim_samples],
#                           # spstats.sem(nonresp_blebs_loc, axis=0, nan_policy='omit')[trim_samples:-trim_samples],
#                           np.nanstd(resp_blebs, axis=0)[trim_samples:-trim_samples],
#                           np.nanstd(resp_blebs_arp, axis=0)[trim_samples:-trim_samples],
#                           t=t_new[trim_samples:-trim_samples],
#                           color1='k', color2='g',
#                           xlabel1='Time (s)', xlabel2='Time (s)',
#                           ylabel1='Bleb Radius (um)', ylabel2='Arp3 (A.U)',
#                           lw=2,
#                           figsize=(8,5),
#                           savefile=savefile_average_curves_plot)



#     # # b) non responsive cells.
#     # get the most popular temporal sampling time in order to build a complete timescale. ?
#     # spline fit time and signal .... then average.

#     # figure out the time_min and time_max
#     Ts = [4.96, 4.96, 4.96, 4.96, 4.96, 4.97, 4.96, 4.96, 4.96, 4.96, 4.96, 4.96, 4.96, 4.97, 4.96, 4.96]

#     t_all = []

#     for ii in np.arange(len(nonresp_cells_R))[:]:
#         signal_ii = np.squeeze(nonresp_cells_R[ii])
#         t = np.arange(len(signal_ii)) * Ts[ii] - len(signal_ii)//2*Ts[ii] # in terms of seconds.
#         t_all.append(t)

#     # determine the resampled timeseries.
#     t_min = np.min(np.hstack(t_all))
#     t_max = np.max(np.hstack(t_all))
#     sample_no = np.hstack([len(tt) for tt in t_all])
#     n_samples = int(spstats.mode(sample_no)[0])
#     t_new = np.linspace(t_min, t_max, n_samples)

# # =============================================================================
# #     Now we fit the series .. (check this ) and then resample onto the common time for averaging.
# # =============================================================================

#     nonresp_blebs = []
#     nonresp_blebs_loc = []
#     nonresp_blebs_arp = []

#     for ii in np.arange(len(nonresp_cells_R))[:]:

#         # use the localization as the control for the delivery.
#         # signal localisation?
#         cell_size = np.squeeze(nonresp_cells_maxsize[ii])
#         signal_ii = np.squeeze(nonresp_cells_R[ii])
#         localization_ii = nonresp_cells_localization[ii]
#         arp_ii = nonresp_cells_Arp[ii]
#         t = np.arange(len(signal_ii)) * Ts[ii] - len(signal_ii)//2*Ts[ii] # in terms of seconds.

#         # we have to resample all of these.
#         signal_ii_new = univariate_spl(t, signal_ii, t_new, s=0)[1]
#         localization_ii_new = univariate_spl(t, localization_ii, t_new, s=0)[1]
#         arp_ii_new = univariate_spl(t, arp_ii, t_new, s=0)[1]

#         fig, ax = plt.subplots(figsize=(10,10))
#         ax.plot(t,signal_ii, 'k.')
#         ax.plot(t_new, signal_ii_new, 'k-')
#         ax.plot(t,localization_ii, 'r.')
#         ax.plot(t_new, localization_ii_new, 'r-')
#         ax.plot(t,arp_ii, 'g.')
#         ax.plot(t_new, arp_ii_new, 'g-')
#         plt.show()

#         # append the unified signals.
#         nonresp_blebs.append(signal_ii_new)
#         nonresp_blebs_loc.append(localization_ii_new)
#         nonresp_blebs_arp.append(arp_ii_new)

#     nonresp_blebs = np.vstack(nonresp_blebs)
#     nonresp_blebs_loc = np.vstack(nonresp_blebs_loc)
#     nonresp_blebs_arp = np.vstack(nonresp_blebs_arp)


# # =============================================================================
# #     Generate the mean curve and the standard deviation intervals.
# # =============================================================================
#     # plot the mean signal
#     fig, ax = plt.subplots(figsize=(8,5))
#     ax.plot(t_new, np.nanmean(nonresp_blebs, axis=0), 'k-')
#     ax.plot(t_new, np.nanmean(nonresp_blebs_loc, axis=0), 'r-')
#     ax.plot(t_new, np.nanmean(nonresp_blebs_arp, axis=0), 'g-')
#     plt.show()


#     trim_samples = 5

#     """
#     blebs vs localisation
#     """
#     savefile_average_curves_plot = os.path.join(savefolder_nonresponse,
#                                                 'all_blebR-vs-localization_avgcurve.png') # as png for now.
#     plot_pair_mean_signal(np.nanmean(nonresp_blebs, axis=0)[trim_samples:-trim_samples],
#                           np.nanmean(nonresp_blebs_loc, axis=0)[trim_samples:-trim_samples],
#                           # spstats.sem(nonresp_blebs, axis=0, nan_policy='omit')[trim_samples:-trim_samples],
#                           # spstats.sem(nonresp_blebs_loc, axis=0, nan_policy='omit')[trim_samples:-trim_samples],
#                           np.nanstd(nonresp_blebs, axis=0)[trim_samples:-trim_samples],
#                           np.nanstd(nonresp_blebs_loc, axis=0)[trim_samples:-trim_samples],
#                           t=t_new[trim_samples:-trim_samples],
#                           color1='k', color2='r',
#                           xlabel1='Time (s)', xlabel2='Time (s)',
#                           ylabel1='Bleb Radius (um)', ylabel2='Localization (A.U)',
#                           lw=2,
#                           figsize=(8,5),
#                           savefile=savefile_average_curves_plot)

#     """
#     blebs vs arp3
#     """
#     savefile_average_curves_plot = os.path.join(savefolder_nonresponse,
#                                                 'all_blebR-vs-arp3_avgcurve.png') # as png for now.
#     plot_pair_mean_signal(np.nanmean(nonresp_blebs, axis=0)[trim_samples:-trim_samples],
#                           np.nanmean(nonresp_blebs_arp, axis=0)[trim_samples:-trim_samples],
#                           # spstats.sem(nonresp_blebs, axis=0, nan_policy='omit')[trim_samples:-trim_samples],
#                           # spstats.sem(nonresp_blebs_loc, axis=0, nan_policy='omit')[trim_samples:-trim_samples],
#                           np.nanstd(nonresp_blebs, axis=0)[trim_samples:-trim_samples],
#                           np.nanstd(nonresp_blebs_arp, axis=0)[trim_samples:-trim_samples],
#                           t=t_new[trim_samples:-trim_samples],
#                           color1='k', color2='g',
#                           xlabel1='Time (s)', xlabel2='Time (s)',
#                           ylabel1='Bleb Radius (um)', ylabel2='Arp3 (A.U)',
#                           lw=2,
#                           figsize=(8,5),
#                           savefile=savefile_average_curves_plot)







    # data1 = np.squeeze(resp_cells_R[0]);
    # data1_smooth = als_smooth_1D(data1, lam=1e1, p=0.5, niter=10)
    # data2 = np.squeeze(resp_cells_localization[0])
    # data2_smooth = als_smooth_1D(data2, lam=1e1, p=0.5, niter=10)
    # t = np.arange(len(data1))

    # fig, ax1 = plt.subplots(figsize=(8,5))
    # color = 'k'
    # ax1.set_xlabel('time (Frame #)', fontsize=16, fontname='Arial')
    # ax1.set_ylabel('Bleb Radius (um)', color=color, fontsize=16, fontname='Arial')
    # ax1.plot(t, data1, color=color)
    # # ax1.plot(t, data1_smooth, '-.', color=color, lw=2)
    # ax1.tick_params(axis='y', labelcolor=color, labelsize=18)
    # ax1.tick_params(axis='x', labelcolor=color, labelsize=18)
    # ax1.set_ylim([0,1.5])
    # for tick in ax1.xaxis.get_major_ticks():
    #     tick.label.set_fontsize(14)
    #     tick.label.set_fontname('Arial')
    # for tick in ax1.yaxis.get_major_ticks():
    #     tick.label.set_fontsize(14)
    #     tick.label.set_fontname('Arial')


    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    # color = 'r'
    # ax2.set_ylabel('Localization (A.U.)', color=color, fontsize=16, fontname='Arial')  # we already handled the x-label with ax1
    # ax2.plot(t, data2, color=color)
    # # ax2.plot(t, data2_smooth, '-.', color=color, lw=2)
    # # ax2.yaxis.set_tick_params(labelsize=18, color='r')
    # ax2.tick_params(axis='y', labelcolor=color, labelsize=18)
    # # ax2.set_yticklabels(ax2, fontname='Arial')
    # # # ax2.tick_params(axis='x', labelcolor=color, labelsize=18)
    # for tick in ax2.yaxis.get_major_ticks():
    #     tick.label.set_fontsize(14)
    #     tick.label.set_fontname('Arial')
    #     tick.label1.set_fontsize(14)
    #     tick.label1.set_fontname('Arial')
    #     tick.label2.set_fontsize(14)
    #     tick.label2.set_fontname('Arial')
    # # ax2.set_ylim([0,1.5])

    # ax1.tick_params(length=10)
    # ax2.tick_params(length=10)
    # plt.legend()
    # # ax1.set_xticklabels(fontname='Arial', fontsize=18)
    # # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # plt.show()

