# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 03:17:38 2021

@author: fyz11
"""
def mkdir(directory):

    import os

    if not os.path.exists(directory):
        os.makedirs(directory)

    return []

if __name__=="__main__":
    
    import numpy as np
    import pylab as plt 
    import os 
    import scipy.io as spio 
    import scipy.stats as spstats
    
    picformat = '.pdf'
    
    basefolder = 'C:\\Users\\fyz11\\Documents\\Work\\Projects\\Danuser-3D Causality\\Data\\Etai\\Worrying\\2021-10-12_Analysis'
    savefolder = os.path.join(basefolder, '2021-12-21_paper_plots'); mkdir(savefolder)
    
    responsive_xcorrs_file_arp3 = os.path.join(basefolder, 'Responsive_Cells', 'Plots_final_2021-12-20', 'xcorr_stats_responsive_vs_arp3.mat')
    responsive_xcorrs_file_loc = os.path.join(basefolder, 'Responsive_Cells', 'Plots_final_2021-12-20', 'xcorr_stats_responsive_vs_loc.mat')
    nonresponsive_xcorrs_file_arp3 = os.path.join(basefolder, 'NonResponsive_Cells', 'Plots_final_2021-12-20', 'xcorr_stats_nonresponsive_vs_arp3.mat')
    nonresponsive_xcorrs_file_loc = os.path.join(basefolder, 'NonResponsive_Cells', 'Plots_final_2021-12-20', 'xcorr_stats_nonresponsive_vs_loc.mat')
    
    responsive_xcorrs_arp3 = spio.loadmat(responsive_xcorrs_file_arp3)['xcorr_all'] ; responsive_xcorrs_arp3_tlag = spio.loadmat(responsive_xcorrs_file_arp3)['lags_all']
    responsive_xcorrs_arp3_tnew =  np.squeeze(spio.loadmat(responsive_xcorrs_file_arp3)['t_new'])
    responsive_xcorrs_loc = spio.loadmat(responsive_xcorrs_file_loc)['xcorr_all']; responsive_xcorrs_loc_tlag = spio.loadmat(responsive_xcorrs_file_loc)['lags_all']
    responsive_xcorrs_loc_tnew = np.squeeze(spio.loadmat(responsive_xcorrs_file_loc)['t_new'])
    
    nonresponsive_xcorrs_arp3 = spio.loadmat(nonresponsive_xcorrs_file_arp3)['xcorr_all']; nonresponsive_xcorrs_arp3_tlag = spio.loadmat(nonresponsive_xcorrs_file_arp3)['lags_all']
    nonresponsive_xcorrs_arp3_tnew = np.squeeze(spio.loadmat(nonresponsive_xcorrs_file_arp3)['t_new'])
    nonresponsive_xcorrs_loc = spio.loadmat(nonresponsive_xcorrs_file_loc)['xcorr_all']; nonresponsive_xcorrs_loc_tlag = spio.loadmat(nonresponsive_xcorrs_file_loc)['lags_all']
    nonresponsive_xcorrs_loc_tnew = np.squeeze(spio.loadmat(nonresponsive_xcorrs_file_loc)['t_new'])
    
    
# =============================================================================
#   derive the mean and coplot the responsive and nonresponsive on the same curve for comparison (RFP and arp3).   
# =============================================================================

    aspect_ratio = (5,5)
    mean_func = np.nanmean
    min_lim, max_lim = -.6, .6
    min_x, max_x = -600, 600
    
    sem_factor = 1 # .96 # for the 95% confidence interval 
    
    """
    1. correlation for the localisation RFP.
    """
    fig, ax = plt.subplots(figsize=aspect_ratio)
    # responsive 
    ax.plot(responsive_xcorrs_loc_tlag[0] * (responsive_xcorrs_loc_tnew[1]-responsive_xcorrs_loc_tnew[0]), 
            mean_func(responsive_xcorrs_loc, axis=0), 'r-')
    ax.fill_between(responsive_xcorrs_loc_tlag[0] * (responsive_xcorrs_loc_tnew[1]-responsive_xcorrs_loc_tnew[0]), 
                    mean_func(responsive_xcorrs_loc, axis=0) - sem_factor*np.nanstd(mean_func(responsive_xcorrs_loc, axis=0)), 
                    mean_func(responsive_xcorrs_loc, axis=0) + sem_factor*np.nanstd(mean_func(responsive_xcorrs_loc, axis=0)), alpha=0.5, color='r')
    # nonresponsive
    ax.plot(nonresponsive_xcorrs_loc_tlag[0] * (nonresponsive_xcorrs_loc_tnew[1]-nonresponsive_xcorrs_loc_tnew[0]), 
            mean_func(nonresponsive_xcorrs_loc, axis=0), 'k-')
    ax.fill_between(nonresponsive_xcorrs_loc_tlag[0] * (nonresponsive_xcorrs_loc_tnew[1]-nonresponsive_xcorrs_loc_tnew[0]), 
                    mean_func(nonresponsive_xcorrs_loc, axis=0) - sem_factor*np.nanstd(mean_func(nonresponsive_xcorrs_loc, axis=0)), 
                    mean_func(nonresponsive_xcorrs_loc, axis=0) + sem_factor*np.nanstd(mean_func(nonresponsive_xcorrs_loc, axis=0)), alpha=0.5, color='gray')

    ax.vlines(0, min_lim, max_lim, linestyles = 'dashed', color='k')
    ax.hlines(0, min_x, max_x, linestyles = 'dashed', color='k')
    plt.tick_params(length=10, right=True)
    plt.xlabel('Time Lag (s)', fontsize=16, fontname='Arial')
    plt.ylabel('Correlation', fontsize=16, fontname='Arial')
    plt.xticks(fontname='Arial', fontsize=14)
    plt.yticks(fontname='Arial', fontsize=14)
    plt.ylim([min_lim,max_lim])
    plt.xlim([min_x,max_x])
    plt.savefig(os.path.join(savefolder, 
                              'all_blebR-vs-tagRFP_xcorr_confinterval%s' %(picformat)), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()


    """
    2. correlation for the localisation Arp3 signal.
    """
    fig, ax = plt.subplots(figsize=aspect_ratio)
    # responsive 
    ax.plot(responsive_xcorrs_arp3_tlag[0] * (responsive_xcorrs_arp3_tnew[1]-responsive_xcorrs_arp3_tnew[0]), 
            mean_func(responsive_xcorrs_arp3, axis=0), 'g-')
    ax.fill_between(responsive_xcorrs_arp3_tlag[0] * (responsive_xcorrs_arp3_tnew[1]-responsive_xcorrs_arp3_tnew[0]), 
                    mean_func(responsive_xcorrs_arp3, axis=0) - np.nanstd(mean_func(responsive_xcorrs_arp3, axis=0)), 
                    mean_func(responsive_xcorrs_arp3, axis=0) + np.nanstd(mean_func(responsive_xcorrs_arp3, axis=0)), alpha=0.5, color='g')
    # nonresponsive
    ax.plot(nonresponsive_xcorrs_arp3_tlag[0] * (nonresponsive_xcorrs_arp3_tnew[1]-nonresponsive_xcorrs_arp3_tnew[0]), 
            mean_func(nonresponsive_xcorrs_arp3, axis=0), 'k-')
    ax.fill_between(nonresponsive_xcorrs_arp3_tlag[0] * (nonresponsive_xcorrs_loc_tnew[1]-nonresponsive_xcorrs_arp3_tnew[0]), 
                    mean_func(nonresponsive_xcorrs_arp3, axis=0) - np.nanstd(mean_func(nonresponsive_xcorrs_arp3, axis=0)), 
                    mean_func(nonresponsive_xcorrs_arp3, axis=0) + np.nanstd(mean_func(nonresponsive_xcorrs_arp3, axis=0)), alpha=0.5, color='gray')

    ax.vlines(0, min_lim, max_lim, linestyles = 'dashed', color='k')
    ax.hlines(0, min_x, max_x, linestyles = 'dashed', color='k')
    plt.tick_params(length=10, right=True)
    plt.xlabel('Time Lag (s)', fontsize=16, fontname='Arial')
    plt.ylabel('Correlation', fontsize=16, fontname='Arial')
    plt.xticks(fontname='Arial', fontsize=14)
    plt.yticks(fontname='Arial', fontsize=14)
    plt.ylim([min_lim,max_lim])
    plt.xlim([min_x,max_x])
    plt.savefig(os.path.join(savefolder, 
                              'all_blebR-vs-arp3_xcorr_confinterval%s' %(picformat)), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()
    
