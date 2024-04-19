# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 12:52:08 2021

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
    import scipy.io as spio 
    import scipy.stats as spstats
    import os 
    
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
    
    
    savefolder = 'C:\\Users\\fyz11\\Documents\\Work\\Projects\\Danuser-3D Causality\\Data\\Etai\\Worrying\\2021-10-12_Analysis\\2021-12-21_paper_plots'
    # mkdir(savefolder)

    
    aspect = (3,5)
    jitter = 0.02
    savepicformat = '.svg'
    
    
    resp_mat = '2021-12-21_before-after-bleb-stats-responsive_for_coanalysis-Area.mat'
    # spio.savemat(savematfile, 
    #              {'bleb_size': all_resp_cells_R, 
    #               'bleb_size_mean': all_resp_blebs_mean, 
    #               'bleb_size_sem': all_resp_blebs_sem, 
    #               'loc_rfp': all_resp_cells_localization, 
    #               'loc_rfp_mean' : all_resp_blebs_loc_mean, 
    #               'loc_rfp_sem': all_resp_blebs_loc_sem, 
    #               'arp3': all_resp_cells_arp, 
    #               'arp3_mean': all_resp_blebs_arp_mean,
    #               'arp3_sem': all_resp_blebs_arp_sem, 
    #               'arp_select': arp_select, 
    #               'roi_names': all_resp_cells_movies})
    nonresp_mat = '2021-12-21_before-after-bleb-stats-nonresponsive_for_coanalysis-Area.mat'
    
    
    resp_mat_obj = spio.loadmat(resp_mat)
    nonresp_mat_obj = spio.loadmat(nonresp_mat)
    
    
    # 1. bleb size ( change in area assuming a circle. ) 
    resp_delta_bleb_size = ((resp_mat_obj['bleb_size'][:,1] - resp_mat_obj['bleb_size'][:,0]) / (resp_mat_obj['bleb_size'][:,0] + 1e-8)) * 100 # (square the scale factor)
    # resp_delta_bleb_size = ((resp_mat_obj['bleb_size'][:,1]) / (resp_mat_obj['bleb_size'][:,0] + 1e-8)) **2 * 100 - 100 
    nonresp_delta_bleb_size = (nonresp_mat_obj['bleb_size'][:,1] - nonresp_mat_obj['bleb_size'][:,0])  / (nonresp_mat_obj['bleb_size'][:,0] + 1e-8) * 100  
    # nonresp_delta_bleb_size = ((nonresp_mat_obj['bleb_size'][:,1]) / (nonresp_mat_obj['bleb_size'][:,0] + 1e-8)) **2 * 100 - 100 
    
    resp_roi_bleb_size = resp_mat_obj['roi_names'].copy(); resp_roi_bleb_size = np.hstack([mm.split('_Region')[0] for mm in np.hstack(resp_roi_bleb_size)])
    nonresp_roi_bleb_size = nonresp_mat_obj['roi_names'].copy(); nonresp_roi_bleb_size = np.hstack([mm.split('_Region')[0] for mm in np.hstack(nonresp_roi_bleb_size)])
    print('N_resp_bleb: ', len(resp_delta_bleb_size), len(parse_uniq_conditions(resp_roi_bleb_size)), ' movies')
    print('N_nonresp_bleb: ', len(nonresp_delta_bleb_size), len(parse_uniq_conditions(nonresp_roi_bleb_size)), ' movies')
    
    
    # 2. localisation  
    resp_delta_bleb_rfp = (resp_mat_obj['loc_rfp'][:,1] - resp_mat_obj['loc_rfp'][:,0]) #/ (resp_mat_obj['loc_rfp'][:,0] + 1e-8) * 100 
    nonresp_delta_bleb_rfp = (nonresp_mat_obj['loc_rfp'][:,1] - nonresp_mat_obj['loc_rfp'][:,0]) #/ (nonresp_mat_obj['loc_rfp'][:,0] + 1e-8) * 100 
    
    
    resp_roi_bleb_rfp = resp_mat_obj['roi_names'].copy(); resp_roi_bleb_rfp = np.hstack([mm.split('_Region')[0] for mm in np.hstack(resp_roi_bleb_rfp)])
    nonresp_roi_bleb_rfp = nonresp_mat_obj['roi_names'].copy(); nonresp_roi_bleb_rfp = np.hstack([mm.split('_Region')[0] for mm in np.hstack(nonresp_roi_bleb_rfp)])
    print('N_resp_rfp: ', len(resp_roi_bleb_rfp), len(parse_uniq_conditions(resp_roi_bleb_rfp)), ' movies')
    print('N_nonresp_rfp: ', len(nonresp_roi_bleb_rfp), len(parse_uniq_conditions(nonresp_roi_bleb_rfp)), ' movies')
    
    # 3. arp3     
    resp_delta_bleb_arp3 = (resp_mat_obj['arp3'][:,1] - resp_mat_obj['arp3'][:,0]) #/ (resp_mat_obj['arp3'][:,0] + 1e-8)  * 100 
    nonresp_delta_bleb_arp3 = (nonresp_mat_obj['arp3'][:,1] - nonresp_mat_obj['arp3'][:,0]) #/ (nonresp_mat_obj['arp3'][:,0] + 1e-8) * 100  
    
    resp_roi_bleb_arp3 = resp_mat_obj['roi_names'][resp_mat_obj['arp_select'].ravel()>0]; resp_roi_bleb_arp3 = np.hstack([mm.split('_Region')[0] for mm in np.hstack(resp_roi_bleb_arp3)])
    nonresp_roi_bleb_arp3 = nonresp_mat_obj['roi_names'][nonresp_mat_obj['arp_select'].ravel()>0]; nonresp_roi_bleb_arp3 = np.hstack([mm.split('_Region')[0] for mm in np.hstack(nonresp_roi_bleb_arp3)])
    print('N_resp_arp3: ', len(resp_roi_bleb_arp3), len(parse_uniq_conditions(resp_roi_bleb_arp3)), ' movies')
    print('N_nonresp_arp3: ', len(nonresp_roi_bleb_arp3), len(parse_uniq_conditions(nonresp_roi_bleb_arp3)), ' movies')
    
    
# =============================================================================
#     Plots 
# =============================================================================
    
    """
    1. bleb size change between responsive and non responsive 
    """
    all_blebs_R = [resp_delta_bleb_size, nonresp_delta_bleb_size]
    all_blebs_mean = np.hstack([np.nanmean(resp_delta_bleb_size), np.nanmean(nonresp_delta_bleb_size)])
    all_blebs_sem = np.hstack([spstats.sem(resp_delta_bleb_size, nan_policy='omit'), spstats.sem(nonresp_delta_bleb_size, nan_policy='omit')])
    
    # plot
    fig, ax = plt.subplots(figsize=aspect)
    # ax.bar(np.arange(resp_blebs.shape[1]), 
    #        resp_blebs_mean)
    ax.plot([-0.5,1.5], 
            [0,0], 'k-')
    ax.errorbar(np.arange(len(all_blebs_mean)), 
                all_blebs_mean, 
                yerr = all_blebs_sem, 
                fmt='none', 
                capsize=5, capthick=2, lw=2, ecolor='k', zorder=10000)
    for group_ii in np.arange(len(all_blebs_R)):
        # data = all_blebs_R[group_ii]
        # data_xx = group_ii + jitter*np.random.randn(len(data))
        # ax.plot(data_xx, data, 'o', mfc='none', zorder=100)
        # # ax.plot([group_ii-0.1, group_ii+0.1],
        # #         [all_blebs_mean[group_ii], all_blebs_mean[group_ii]], 
        # #         'k-', lw=2, zorder=10000)
        ax.bar(group_ii, [all_blebs_mean[group_ii], all_blebs_mean[group_ii]], color='lightgray', edgecolor='k', width=0.8)
    # plt.ylim([-0.05, 0.2])
    plt.ylim([-10,60])
    plt.xlim([-1+0.5,2-0.5])
    # plt.tick_params(length=10, right=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xticks(np.arange(len(all_blebs_mean)), ['sspb tagRFP Tiam1', 'sspb tagRFP'], fontsize=14, fontname='Arial', rotation=30, ha='right')
    plt.yticks(fontsize=14, fontname='Arial')
    plt.ylabel('% Change in Bleb Area', fontname='Arial', fontsize=16)
    # # take the spines out. 
    # # save the analysis out. 
    plt.savefig(os.path.join(savefolder, 
                             'Delta-bleb-size_percent_area%s' %(savepicformat)), dpi=300, bbox_inches='tight')
    plt.show()
    
    
    """
    2. Localisation signal 
    """
    all_rfp_R = [resp_delta_bleb_rfp, nonresp_delta_bleb_rfp]
    all_rfp_mean = np.hstack([np.nanmean(resp_delta_bleb_rfp), np.nanmean(nonresp_delta_bleb_rfp)])
    all_rfp_sem = np.hstack([spstats.sem(resp_delta_bleb_rfp, nan_policy='omit'), spstats.sem(nonresp_delta_bleb_rfp, nan_policy='omit')])
    
    # plot
    fig, ax = plt.subplots(figsize=aspect)
    # ax.bar(np.arange(resp_blebs.shape[1]), 
    #        resp_blebs_mean)
    ax.plot([-0.5,1.5], 
            [0,0], 'k-')
    ax.errorbar(np.arange(len(all_rfp_R)), 
                all_rfp_mean, 
                yerr = all_rfp_sem, 
                fmt='none', 
                capsize=5, capthick=2, lw=2, ecolor='k', zorder=10000)
    for group_ii in np.arange(len(all_rfp_R)):
        # data = all_blebs_R[group_ii]
        # data_xx = group_ii + jitter*np.random.randn(len(data))
        # ax.plot(data_xx, data, 'o', mfc='none', zorder=100)
        # # ax.plot([group_ii-0.1, group_ii+0.1],
        # #         [all_blebs_mean[group_ii], all_blebs_mean[group_ii]], 
        # #         'k-', lw=2, zorder=10000)
        ax.bar(group_ii, [all_rfp_mean[group_ii], all_rfp_mean[group_ii]], color='lightgray', edgecolor='k', width=0.8)
    plt.ylim([-0.1, 0.8])
    plt.xlim([-1+0.5,2-0.5])
    # plt.tick_params(length=10, right=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xticks(np.arange(len(all_blebs_mean)), ['sspb tagRFP Tiam1', 'sspb tagRFP'], fontsize=14, fontname='Arial', rotation=30, ha='right')
    plt.yticks(fontsize=14, fontname='Arial')
    plt.ylabel('$\Delta$ Photoinduction (a.u.)', fontname='Arial', fontsize=16)
    # # take the spines out. 
    # # save the analysis out. 
    plt.savefig(os.path.join(savefolder, 
                             'Delta-rfp-size%s' %(savepicformat)), dpi=300, bbox_inches='tight')
    plt.show()
    
    
    """
    3. Arp3 signal
    """
    all_arp3_R = [resp_delta_bleb_arp3, nonresp_delta_bleb_arp3]
    all_arp3_mean = np.hstack([np.nanmean(resp_delta_bleb_arp3), np.nanmean(nonresp_delta_bleb_arp3)])
    all_arp3_sem = np.hstack([spstats.sem(resp_delta_bleb_arp3, nan_policy='omit'), spstats.sem(nonresp_delta_bleb_arp3, nan_policy='omit')])
    
    # plot
    fig, ax = plt.subplots(figsize=aspect)
    # ax.bar(np.arange(resp_blebs.shape[1]), 
    #        resp_blebs_mean)
    ax.plot([-0.5,1.5], 
            [0,0], 'k-')
    ax.errorbar(np.arange(len(all_arp3_R)), 
                all_arp3_mean, 
                yerr = all_arp3_sem, 
                fmt='none', 
                capsize=5, capthick=2, lw=2, ecolor='k', zorder=10000)
    for group_ii in np.arange(len(all_arp3_R)):
        # data = all_blebs_R[group_ii]
        # data_xx = group_ii + jitter*np.random.randn(len(data))
        # ax.plot(data_xx, data, 'o', mfc='none', zorder=100)
        # # ax.plot([group_ii-0.1, group_ii+0.1],
        # #         [all_blebs_mean[group_ii], all_blebs_mean[group_ii]], 
        # #         'k-', lw=2, zorder=10000)
        ax.bar(group_ii, [all_arp3_mean[group_ii], all_arp3_mean[group_ii]], color='lightgray', edgecolor='k', width=0.8)
    plt.ylim([-0.1, 0.8])
    plt.xlim([-1+0.5,2-0.5])
    # plt.tick_params(length=10, right=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xticks(np.arange(len(all_blebs_mean)), ['sspb tagRFP Tiam1', 'sspb tagRFP'], fontsize=14, fontname='Arial', rotation=30, ha='right')
    plt.yticks(fontsize=14, fontname='Arial')
    plt.ylabel('$\Delta$ Arp3 (a.u.)', fontname='Arial', fontsize=16)
    # # take the spines out. 
    # # save the analysis out. 
    plt.savefig(os.path.join(savefolder, 
                             'Delta-arp3-size%s' %(savepicformat)), dpi=300, bbox_inches='tight')
    plt.show()
    

    
    