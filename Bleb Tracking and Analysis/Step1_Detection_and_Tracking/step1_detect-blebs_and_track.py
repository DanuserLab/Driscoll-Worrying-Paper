# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 10:55:53 2021

@author: fyz11
"""

def largest_component(binary):
    
    from skimage.measure import regionprops, label
    
    label_img = label(binary)
    props = regionprops(label_img)
    
    uniq_lab = np.setdiff1d(np.unique(label_img),0)
    areas = np.hstack([prop.area for prop in props])
    keep_label = label_img == uniq_lab[np.argmax(areas)]
    
    return keep_label
    

def parse_roifilezip(roifile, min_size=10):
    
    # only reads rectangle
    from read_roi import read_roi_file, read_roi_zip
    
    roi = read_roi_zip(roifile)

    regions = []
    
    for region in list(roi.keys()):
        entry = roi[region]
        # print(entry)
        region_name = entry['name']
        x = entry['left']
        y = entry['top']
        w = entry['width']
        h = entry['height']
        x1 = x
        x2 = x + w
        y1 = y
        y2 = y + h
        box = np.hstack([region_name, x1,y1,x2,y2])
        regions.append(box)
        
    regions = np.vstack(regions)
    regions_bbox = regions[:,1:].astype(np.float64)
    regions_size = (regions_bbox[:,2]-regions_bbox[:,0]) * ( regions_bbox[:,3] - regions_bbox[:,1])
    
    keep = regions_size > min_size
    regions = regions[keep]
    
    return regions


def mkdir(fpath):
    
    import os 
    
    if not os.path.exists(fpath):
        os.makedirs(fpath)
        
    return []

    
def normalize_vid(vid):

    from skimage import exposure 
    import numpy as np 

    gray_ = vid.copy() # this overestimates. 
    gray_ = np.uint8(np.array([255.*(gg - gg.min())/(gg.max()-gg.min()) for gg in gray_]))
    gray_ = np.array([exposure.rescale_intensity(gg) for gg in gray_])

    return gray_


def circles_to_boxes(circle_list):
    
    import numpy as np 
    
    bbox_list = []
    
    for roi_region_ii in np.arange(len(circle_list)):
        circle_list_region = circle_list[roi_region_ii]
    
        bbox_list_region = []
        
        # iterate over the time. 
        for nframe_ii in np.arange(len(circle_list_region)):
            
            circle_list_frame = circle_list_region[nframe_ii]
            if len(circle_list_frame)>0:
                x1 = circle_list_frame[:,1] - circle_list_frame[:,-1]
                x2 = circle_list_frame[:,1] + circle_list_frame[:,-1]
                y1 = circle_list_frame[:,0] - circle_list_frame[:,-1]
                y2 = circle_list_frame[:,0] + circle_list_frame[:,-1]
                bbox_list_region.append(np.hstack([x1[:,None], y1[:,None], x2[:,None], y2[:,None]]))
            else:
                bbox_list_region.append(np.array([[np.nan, np.nan, np.nan, np.nan]])) # initiate a nan list. 

        bbox_list.append(bbox_list_region)

    return bbox_list 

def thresh_obj(im):

    import skimage.filters as skfilters
    import scipy.ndimage as ndimage
    binary = im>skfilters.threshold_otsu(obj_vid)
    return ndimage.morphology.binary_fill_holes(skmorph.binary_closing(binary,skmorph.disk(1)))


def bleb_score(bbox_track, obj_vid):
    """
    Parameters
    ----------
    bbox_track : TYPE
        N x T x 4 
    obj_vid : TYPE
        T x m x n binary

    Returns
    -------
        N x T score. 
    """
    
    T = bbox_track.shape[1] # length of the track. 
    score = []
    
    for t in np.arange(T):
        obj_vid_t = obj_vid[t].copy()
        bbox_track_t = bbox_track[:,t].copy()
        
        score_t = np.zeros(len(bbox_track_t)); score_t[:] = np.nan
        nonnan = np.arange(len(score_t))[np.logical_not(np.isnan(bbox_track_t[:,0]))] 
        
        if len(nonnan) > 0: 
            for ind in nonnan:
                x1,y1,x2,y2 = bbox_track_t[ind].astype(np.int)
                score_t_ind = np.nanmean(obj_vid_t[y1:y2,x1:x2])
                score_t[ind] = score_t_ind
                
        score.append(score_t[:,None])
    score = np.concatenate(score, axis=-1)
    
    return score

def bleb_R(bbox_track):
    
    delta_X = (bbox_track[...,2] - bbox_track[...,0]) / 2.
    delta_Y = (bbox_track[...,3] - bbox_track[...,1]) / 2.
    
    return .5*(delta_X + delta_Y)


if __name__=="__main__":
    
    import numpy as np
    import pylab as plt 
    import scipy.io as spio 
    # from czifile import CziFile
    from aicspylibczi import CziFile
    import scipy.ndimage as ndimage
    from skimage.feature import blob_dog, blob_log, blob_doh
    from skimage.exposure import equalize_hist
    import skimage.exposure as exposure
    import skimage.filters as skfilters
    import skimage.morphology as skmorph
    from skimage.measure import find_contours
    # from pylibczi import CziScene
    # from pathlib import Path
    import os 
    import glob
    import skimage.io as skio 
    import bleb_tracking # custom module for bleb tracking 
    import skimage.measure as skmeasure
    import seaborn as sns 
    import os 
    from aicsimageio import AICSImage
    
# =============================================================================
#     Specify the separate paths of the movies + the Rois
# =============================================================================


    moviefolder = '/archive/bioinformatics/Danuser_lab/Fiolka/LabMembers/Etai/worring_paper_data/Zeiss_780/Movie_for_analysis_felix'
    roifolder = '/project/bioinformatics/Danuser_lab/3Dmorphogenesis/analysis/fzhou/FiolkaLab/Etai/2021_WorryingPaper_Bleb'
    
    moviefiles = glob.glob(os.path.join(moviefolder, '*.czi')) # 8 movies. 
    moviefiles = np.sort(moviefiles) # so we always get it in the same order.
    
    # hard code in whether we do the image inversion or not. 
    invert_image = [False, True, False, False, True, True, True, True] # for the 7th we miss the largest. 
    
    # save the analysis in the same folder
    
# =============================================================================
#     Pipeline -> bleb detection, filteration of blebs by track lenght + background. 
# =============================================================================
    
    for ii in np.arange(len(moviefiles))[-1:]:
        
        moviefile = moviefiles[ii]
        fname = os.path.split(moviefile)[-1]
        czi = CziFile(moviefile)
        im, im_shape = czi.read_image()
        im = np.squeeze(im) # time x channels x nrows x ncols
        gray = im[:,1].copy()
        
        # plt.figure()
        # plt.title(ii)
        # plt.imshow(gray[0])
        # plt.show()

        roifile = os.path.join(roifolder, fname.split('_')[0])+'_RoiSet.zip'
        roi_regions = parse_roifilezip(roifile)

        print(len(roi_regions))


        # grab the pixel size.
        img_obj = AICSImage(moviefile)
        pix_res = img_obj.physical_pixel_sizes[1]
        
        multiply_ratio = 2.63576456e-7 / pix_res


        saveanalysisfolder = os.path.join(roifolder, fname.split('_')[0]) + '_revised'
        mkdir(saveanalysisfolder)
        
        vizanalysisfolder = os.path.join(saveanalysisfolder, 'viz_detections')
        mkdir(vizanalysisfolder)
        
        gray_ = normalize_vid(gray)
        obj_vid = np.array([np.abs(skfilters.sobel(gg)) for gg in gray])
        obj_vid_binary = np.array([thresh_obj(gg) for gg in obj_vid]) # convert the edgeness signature into a blebby proxy signature.
        
#         # get the background segmentations for all frames first so we can infer if the blebs to track should be inverted or not.
        
#         all_binary = []
        
#         for frame in np.arange(len(gray))[:]:
            
#             # 1. grab the cell boundary initially. 
#             gray_eq = equalize_hist(gray[frame])
#             gray_dog = gray_eq - ndimage.gaussian_filter(gray_eq, sigma=1) + np.mean(gray_eq) # how much to smooth? 
#             gray_dog[gray_dog<0] = 0 
#             gray_dog = (gray_dog - gray_dog.min()) / (gray_dog.max()-gray_dog.min())
            
#             # segmentation based on edgeness. 
#             edge_mag = np.abs(skfilters.sobel(gray_eq)); 
#             edge_mag = ndimage.gaussian_filter(edge_mag, sigma=3)
#             binary = edge_mag >= skfilters.threshold_multiotsu(edge_mag)[0] 
#             binary = skmorph.binary_closing(binary, skmorph.disk(5)); 
#             binary = ndimage.binary_fill_holes(binary);
#             binary = skmorph.binary_closing(binary, skmorph.disk(5)); 
#             binary = ndimage.binary_fill_holes(binary)
            
#             #  """
#             # remove using a border
#             # """
#             binary[:20, :] = 0
#             binary[-20:,:] = 0
#             binary[:, :20] = 0
#             binary[:,-20:] = 0
            
#             binary = skmorph.remove_small_objects(binary, min_size=250)
#             # binary_bg = binary.copy()
#             all_binary.append(binary)
        
#         # check if need to invert this. 
#         I_ratio_time = np.hstack([np.nanmean(gray_[tt][obj_vid_binary[tt]>0]) / np.nanmean(gray_[tt][all_binary[tt]>0]) for tt in np.arange(len(all_binary))])
        
#         I_ratio = np.nanmean(I_ratio_time)
        
#         if I_ratio < 1:
#             print('invert')
        
# #         img_frame = gray_[frame].copy()
        
# #         I_ratio = np.nanmean(img_frame[obj_vid_binary[frame]>0])  / np.nanmean(img_frame[binary_bg>0])
# #         print(I_ratio)
        
# #         if I_ratio < 1:
# #             img_frame = 255 - img_frame
        
    # =============================================================================
    #     Attempt to capture the cellular boundary -> to inform which blebs to keep. 
    # =============================================================================
        
        all_binary = []
        all_detections = []
        roi_detections = [[] for ii in np.arange(len(roi_regions))] # pre-initialise with the number of ROI regions.
    
        for frame in np.arange(len(gray))[:]:
            
            # 1. grab the cell boundary initially. 
            gray_eq = equalize_hist(gray[frame])
            gray_dog = gray_eq - ndimage.gaussian_filter(gray_eq, sigma=1) + np.mean(gray_eq) # how much to smooth? 
            gray_dog[gray_dog<0] = 0 
            gray_dog = (gray_dog - gray_dog.min()) / (gray_dog.max()-gray_dog.min())
            
            
            # segmentation based on edgeness. 
            edge_mag = np.abs(skfilters.sobel(gray_eq)); 
            edge_mag = ndimage.gaussian_filter(edge_mag, sigma=3)
            binary = edge_mag >= skfilters.threshold_multiotsu(edge_mag)[0] 
            binary = skmorph.binary_closing(binary, skmorph.disk(5)); 
            binary = ndimage.binary_fill_holes(binary);
            binary = skmorph.binary_closing(binary, skmorph.disk(5)); 
            binary = ndimage.binary_fill_holes(binary)
            
            
            binary = skmorph.remove_small_objects(binary, min_size=250)
            binary_bg = binary.copy()
            
            # # largest_component. 
            # binary = largest_component(binary) # this is a good outer version
            # binary_inner = skmorph.binary_erosion(binary, skmorph.disk(21)) # how much to erode? 21 at the moment -> larger = capture more blebs... 
            binary_inner = skmorph.binary_erosion(binary, skmorph.disk(35))
            
            binary = np.logical_and(binary, np.logical_not(binary_inner)) # should save this?
            
            """
            remove using a border
            """
            binary[:20, :] = 0
            binary[-20:,:] = 0
            binary[:, :20] = 0
            binary[:,-20:] = 0
            
            
            all_binary.append(binary)
            # plt.figure()
            # plt.imshow(gray_eq)
            # plt.show()
            
            # plt.figure(figsize=(10,10))
            # plt.subplot(121)
            # plt.imshow(equalize_hist(gray[frame]), cmap='gray')
            # plt.subplot(122)
            # plt.imshow(binary)
            # plt.show()
            
            # 1b. grab the cell boundary initially. 
            cell_boundary = find_contours(binary, level=0)
            cell_boundary_main = cell_boundary[np.argmax([len(bb) for bb in cell_boundary])]
            
            
            # check if need to invert this. 
            img_frame = gray_[frame].copy()
            
            # I_ratio = np.nanmean(img_frame[obj_vid_binary[frame]>0])  / np.nanmean(img_frame[binary_bg>0])
            # print(I_ratio)
            # if I_ratio < 1:
            if invert_image[ii]:
                img_frame = 255 - img_frame
            
            # img_frame = (img_frame - img_frame.min())/ (img_frame.max()- img_frame.min()) * 255
            # img_frame = np.abs(skfilters.sobel(img_frame))
            
            # multiply_ratio = 1
            
            # 2. multiscale blob detection -> but how to really set this ???? hmmm .... 
            blobs_log = blob_dog(equalize_hist(ndimage.gaussian_filter(img_frame, sigma=1)), 
                                    # img_frame/255.,
                                    # gray_eq,
                                    # equalize_hist(img_frame),
                                  min_sigma=int(2*multiply_ratio), 
                                  max_sigma=int(10*multiply_ratio), # 30 too high! consider determining this by pixel resolution! ?
                                  sigma_ratio=1.2, #1.6 prev
                                  threshold=.05, # .05
                                  overlap=0.75) # we may need a custom version of this!. 
            
            # blobs_log_2 = blob_dog(equalize_hist(ndimage.gaussian_filter(255-img_frame, sigma=1)), 
            #                       # equalize_hist(gray[frame]),
            #                       # gray_eq
            #                       min_sigma=int(2*multiply_ratio), 
            #                       max_sigma=int(30*multiply_ratio), # consider determining this by pixel resolution! ?
            #                       sigma_ratio=1.6, 
            #                       threshold=.05, 
            #                       overlap=0.5) # we may need a custom version of this!. 
            
            # blobs_log = np.vstack([blobs_log, 
            #                        blobs_log_2])
            
            """
            consider non max suppression using objectness. 
            """
            
            
            
            
            # blobs_log = blob_log(equalize_hist(ndimage.gaussian_filter(gray[frame], sigma=1)), 
            #                       min_sigma=2, 
            #                       max_sigma=35, 
            #                       num_sigma=35, 
            #                       threshold=0.1, 
            #                       overlap=0.5)
            blobs_log[:, 2] = blobs_log[:, 2] * np.sqrt(2)
            
            
            # 3. gating by mask. 
            blobs_log_keep = binary[blobs_log[:,0].astype(np.int), 
                                    blobs_log[:,1].astype(np.int)] > 0 
            
            blobs_log = blobs_log[blobs_log_keep]
            # blobs_log = blob_doh(equalize_hist(ndimage.gaussian_filter(gray[0], sigma=1)), 
            #                       min_sigma=2, 
            #                       max_sigma=30, 
            #                       num_sigma=20, 
            #                       threshold=0.01, 
            #                       overlap=0.5)
            
            # # size constraints, 
            # keep_maxsize = blobs_log[...,-1] < maxsize
            # blobs_log = blobs_log[keep_maxsize]
            
            
            fig, ax = plt.subplots(figsize=(15,15))
            plt.imshow(gray[frame], cmap='gray')
            # plt.plot(blobs_log[:,1], 
            #           blobs_log[:,0], 'ro')
            plt.plot(cell_boundary_main[:,1], 
                      cell_boundary_main[:,0], 'g--', lw=5)
            
            for roi_ii, roi_region in enumerate(roi_regions):
                x1,y1,x2,y2 = roi_region[1:].astype(np.float64)
                plt.plot([x1,x2,x2,x1,x1], 
                          [y1,y1,y2,y2,y1], 'k-', lw=3)
                # put the region name
                plt.text(.5*(x1+x2),y1-5, roi_region[0], fontsize=24, ha='center', fontname='Liberation Sans')
                
                # create a mask. 
                roi_mask = np.zeros(gray[frame].shape)
                roi_mask[int(y1):int(y2)+1, int(x1):int(x2)+1] = 1
                
                roi_blobs = roi_mask[blobs_log[:,0].astype(np.int), 
                                      blobs_log[:,1].astype(np.int)] > 0
                roi_detections[roi_ii].append(blobs_log[roi_blobs])
                
            
            for blob in blobs_log:
                y, x, r = blob
                c = plt.Circle((x, y), r, color='r', linewidth=2, fill=False)
                ax.add_patch(c)
                # ax.set_axis_off()
            plt.axis('off')
            plt.grid('off')
            fig.savefig(os.path.join(vizanalysisfolder, 'Frame_%s.png' %(str(frame).zfill(4))), dpi=120, bbox_inches='tight')
            plt.show()
            
        
# =============================================================================
#         How best to save these out? 
# =============================================================================

        # extraction of data points from the ROI of interest and counting ... / sizes
        
        # this last is useful the rest is not!. 
        mean_roi_signal = np.vstack([np.hstack([roi_sig[:,-1].mean() for roi_sig in roi_detection]) for roi_detection in roi_detections])
        # max_roi_signal = np.vstack([np.hstack([roi_sig[:,-1].max() for roi_sig in roi_detection]) for roi_detection in roi_detections])
        # median_roi_signal = np.hstack([[np.median(roi_sig[:,-1]) for roi_sig in roi_detection] for roi_detection in roi_detections])
        
        
        # save the data into 1 .mat. 
        savematfile = os.path.join(saveanalysisfolder, 'bleb_statistics-'+os.path.split(saveanalysisfolder)[-1])
        spio.savemat(savematfile, {'mean_radius_pixels': mean_roi_signal, 
                                    'roi_names': np.ascontiguousarray(roi_regions[:,0]),
                                    'roi_box_coords': np.ascontiguousarray(roi_regions[:,1:].astype(np.float)),
                                    'roi_detections':  roi_detections,
                                    'all_detections': all_detections})
                                   
        # save the binary separately. 
        all_binary = np.array(all_binary)
        skio.imsave(os.path.join(saveanalysisfolder, 'binary_cellborders.tif'), np.uint8(all_binary*255))
        
        
        # convert the detections into bounding boxes for easier processing and saving. 
        roi_detections_bbox = circles_to_boxes(roi_detections) # save this with the tracks. 
        
        print('found ' + str(len(roi_detections_bbox)) + ' regions')

# =============================================================================
#     Predictive tracking of individual blebs.    
# =============================================================================
        
        """
        optical flow
        """
        # gray_ = normalize_vid(gray)
        # obj_vid = np.array([np.abs(skfilters.sobel(gg)) for gg in gray])
        # obj_vid_binary = np.array([thresh_obj(gg) for gg in obj_vid]) # convert the edgeness signature into a blebby proxy signature.

        optical_flow_params = dict(pyr_scale=0.5, levels=3, winsize=5, iterations=5, poly_n=3, poly_sigma=1.2, flags=0) # add a larger window size to be smoother. 
        optflow = bleb_tracking.extract_optflow(np.uint8(gray_), 
                                                optical_flow_params=optical_flow_params, 
                                                rescale_intensity=True, intensity_range=[0,100]) # don't do the intensity range. 

    # =============================================================================
    #     Precompute the 2D optical flow in the video here for use in predictive tracking.   
    # =============================================================================
        
        """
        # Specify the various tracking parameters. 
        """
        iou_match = 0.15 # this is the required iou match across frames.
        # ds_factor = 4
        ds_factor = 1
        min_aspect_ratio = 3 
        max_dense_bbox_cover = 20 # set something really large for Xiaoyues.... 
        wait_time = 5 # the sampling is more so we can wait longer? 


        roi_detection_bleb_tracks = []
        roi_detection_bleb_tracks_blebbiness_score = []
        roi_detection_bleb_tracks_R = []
        
        vizanalysisfolder_tracks = os.path.join(saveanalysisfolder, 'viz_tracks')
        mkdir(vizanalysisfolder_tracks)
        
        for jjj in np.arange(len(roi_detections_bbox)):
            
            vizanalysisfolder_tracks_jjj = os.path.join(vizanalysisfolder_tracks, 'Region_'+str(jjj))
            mkdir(vizanalysisfolder_tracks_jjj)
            
            roi_region_box = roi_regions[jjj]
            
            # do this over each region !
            vid_bbox_tracks_prob, vid_bbox_tracks, vid_bbox_match_checks, (vid_bbox_tracks_all_lens, vid_bbox_tracks_all_start_time, vid_bbox_tracks_lifetime_ratios) = bleb_tracking.track_bleb_bbox(optflow,
                                                                                                                                                                                                        roi_detections_bbox[jjj], # bboxes are already loaded in yolo format.   
                                                                                                                                                                                                            vid = gray,
                                                                                                                                                                                                            iou_match=iou_match,
                                                                                                                                                                                                            ds_factor = ds_factor,
                                                                                                                                                                                                            wait_time = wait_time,
                                                                                                                                                                                                            min_aspect_ratio=min_aspect_ratio,
                                                                                                                                                                                                            max_dense_bbox_cover=max_dense_bbox_cover,
                                                                                                                                                                                                            to_viz=False, # suppress here the bounding box.
                                                                                                                                                                                                            saveanalysisfolder=None) # do not save here. 
                
            """
            # track postprocessing parameters - remove short tracks, remove duplicates. 
            """
            filter_prob_thresh = 0.05 # higher for confocal # doesn't apply for non deep learning. 
            filter_life_thresh =.25 # do we need anything here? 
            
            """
            filter the bbox_tracks - keep only minimal length tracks
            """
            filter_prob_index = np.nanmean(vid_bbox_tracks_prob, axis=1) >= filter_prob_thresh
            filter_lifetime_index = vid_bbox_tracks_lifetime_ratios >= filter_life_thresh
            keep_track_index = np.logical_and(filter_prob_index, filter_lifetime_index)  # which tracklets to keep.
            
            
            """
            filter the bbox_tracks - suppress duplicate detections. (may occur due to the predictive tracking?)
            """
            blebs_all = [vid_bbox_tracks] # load in all the produced tracks into a list and pad....  
            blebs_all_nan = [bleb_tracking.filter_nan_tracks(tra) for tra in blebs_all]
            
            remove_index = np.hstack([len(bb) == 0 for bb in blebs_all_nan])
            blebs_all_nan = [ blebs_all_nan[ii] for ii in np.arange(len(blebs_all_nan)) if remove_index[ii]==False]
                

            """
            filtering using the objectness signature.
            """        
            blebs_all_filter = bleb_tracking.non_stable_track_suppression_filter(#(gray_.max() - gray_)[...,None], # grayscale needs to be padded to at least 1 channel. 
                                                                              obj_vid_binary[...,None],     
                                                                              blebs_all_nan,
                                                                              track_overlap_thresh=0.25, # lower numbers remove more, (less tolerant of )
                                                                              weight_nan=0., weight_smooth=0.1, 
                                                                              max_obj_frames=10, # not sure about this ....
                                                                              obj_mean_func=np.nanmean,
                                                                              smoothness_mean_func=np.nanmean,
                                                                              debug_viz=False)
        
            # repeat the above removal of index. 
            remove_index = np.hstack([len(bb) == 0 for bb in blebs_all_filter])
            
            # channels_files = [ channels_files[ii] for ii in np.arange(len(channels_files)) if remove_index[ii]==False]
            blebs_all_filter = [ blebs_all_filter[ii] for ii in np.arange(len(blebs_all_filter)) if remove_index[ii]==False]
            
            # use the objectness video to produce objectness for each bleb. based on the bounding box! 
            blebbiness_score_all_filter = [bleb_score(bb, obj_vid_binary) for bb in blebs_all_filter]
            
            # compute the mean R of bleb. 
            bleb_R_all_filter = [bleb_R(bb) for bb in blebs_all_filter]
            
            # """
            # appending statistics. 
            # """
            # roi_detection_bleb_tracks.append(blebs_all_filter[0]) # append the tracks to the master. 
            # roi_detection_bleb_tracks_blebbiness_score.append(blebbiness_score_all_filter[0])
            # roi_detection_bleb_tracks_R.append(bleb_R_all_filter[0])
            
            """
            check with the plotting. 
            """
            img_shape = gray[0].shape
            
            plot_colors = sns.color_palette('hls', n_colors=len(blebs_all_filter[0]))
            np.random.shuffle(plot_colors)
            
            # plotfolder = '2021-12-01_blebtrack_' + pth.split('.czi')[0]
            # bleb_tracking.mkdir(plotfolder)

            for frame_no in range(len(gray)):
                fig, ax = plt.subplots(figsize=(10,10))
                plt.title('Frame: %d' %(frame_no+1))
                vid_overlay = gray[frame_no].copy()

                ax.imshow(vid_overlay, alpha=1., cmap='gray')
                
                #### show the ROI box #### 
                x1,y1,x2,y2 = roi_region_box[1:].astype(np.float64)
                ax.plot([x1,x2,x2,x1,x1], 
                          [y1,y1,y2,y2,y1], 'k-', lw=3)
                # put the region name
                plt.text(.5*(x1+x2),y1-5, roi_region_box[0], fontsize=24, ha='center', fontname='Liberation Sans')

                # iterate over each channel. 
                for bb_i, bb in enumerate(blebs_all_filter[:]): 
                    
                    for bbb_ii, bbb in enumerate(bb):
                        x1,y1,x2,y2 = bbb[frame_no]
                        
                        ax.plot([x1,x2,x2,x1,x1], 
                                [y1,y1,y2,y2,y1], color=plot_colors[bbb_ii], lw=1)
                        
                        # plot the last 5 track? 
                        f1 = np.maximum(0, frame_no - 5)
                        f2 = frame_no
                        bb_track_x = .5*(bbb[f1:f2+1,0] + bbb[f1:f2+1,2])
                        bb_track_y = .5*(bbb[f1:f2+1,1] + bbb[f1:f2+1,3])
                        ax.plot(bb_track_x, bb_track_y, color=plot_colors[bbb_ii], lw=1)
                        # ax.plot(bbb[frame_no][:,1], 
                        #         bbb[frame_no][:,0], color=plot_colors[bbb_ii], lw=3)
                
                ax.set_xlim([0, img_shape[1]-1])
                ax.set_ylim([img_shape[0]-1, 0])
                plt.axis('off')
                plt.grid('off')
                
                # fig.savefig(os.path.join(plotfolder, str(frame_no).zfill(3)+'.png'), bbox_inches='tight', dpi=120)
                # fig.savefig(os.path.join(plotfolderfinal, 
                #                          '%s-Frame-%s.png' %(expt, str(frame_no).zfill(3))), bbox_inches='tight', dpi=80)
                fig.savefig(os.path.join(vizanalysisfolder_tracks_jjj, 'Frame_%s.png' %(str(frame_no).zfill(4))), dpi=120, bbox_inches='tight')
                plt.show()
            
            
            """
            Save the final statistics. 
            """
            roi_detection_bleb_tracks.append(blebs_all_filter[0]) # append the tracks to the master. 
            roi_detection_bleb_tracks_blebbiness_score.append(blebbiness_score_all_filter[0])
            roi_detection_bleb_tracks_R.append(bleb_R_all_filter[0])
        
            savematfile = os.path.join(saveanalysisfolder, 'bleb_track_statistics-'+os.path.split(saveanalysisfolder)[-1])
            spio.savemat(savematfile, {'bbox_tracks': np.ascontiguousarray(roi_detection_bleb_tracks), 
                                        'bbox_tracks_blebbiness': np.ascontiguousarray(roi_detection_bleb_tracks_blebbiness_score),
                                        'bbox_tracks_meanR': np.ascontiguousarray(roi_detection_bleb_tracks_R),
                                        'optflow':  optflow.astype(np.float32),
                                        'blebbiness_binary': obj_vid_binary})
    
    
    
    