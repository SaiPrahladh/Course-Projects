from helper import briefMatch, computeBrief, detectCorners, convert2Gray, plotMatches
from planarH import computeH_ransac

import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy


def matchPics(img1, img2, opts):
    # img1, img2 : Images to match
    # opts: input opts
    ratio = opts.ratio  # 'ratio for BRIEF feature descriptor'
    sigma = opts.sigma  # 'threshold for corner detection using FAST feature detector'

    # Convert Images to GrayScale
    img1 = convert2Gray(img1)
    img2 = convert2Gray(img2)

    # Detect Features in Both Images
    locs1 = detectCorners(img1,sigma)
    locs2 = detectCorners(img2,sigma)
    
    # Obtain descriptors for the computed feature locations
    desc1,locs1 = computeBrief(img1,locs1)
    desc2,locs2 = computeBrief(img2,locs2)

    # Match features using the descriptors
    matches = briefMatch(desc1,desc2,ratio)
    #plotMatches(img1,img2,matches,locs1,locs2)

    return matches, locs1, locs2


# Q2.1.5
def briefRotTest(img, opts):
   
    angles = [10,20,30]
    labels = [i for i in range(0,360,10)]
    hists = []

    for angle in range(0,360,10):
        print(angle)
        rot_im = scipy.ndimage.rotate(img,angle)
        matches,locs1, locs2 = matchPics(img,rot_im,opts)
        hists.append(len(matches))

        if angle in angles:
            title = 'sigma = 0.15, ratio = 0.7, angle = ' + str(angle) + '°'
            plotMatches(img,rot_im,matches,locs1,locs2,title)

    
    plt.bar(labels,hists,width = 8, log = True)
    plt.xlabel('rotation angles')
    plt.ylabel('Number of Matches')
    plt.title('Histogram of Matches')
    plt.xticks(np.arange(0,360,10),rotation = 90)
    plt.show()


def composeWarpedImg(img_source, img_target, img_replacement, opts):
    # Obtain the homography that warps img_source to img_target, then use it to overlay img_replacement over img_target
    # Write script for Q2.2.4

    # Obtain features from both source and target images
    matches, locs1, locs2 = matchPics(img_target,img_source,opts)

    locs_mat1 = locs1[matches[:,0],0:2]
    locs_mat2 = locs2[matches[:,1],0:2]

    locs_mat1[:, [1, 0]] = locs_mat1[:, [0, 1]]
    locs_mat2[:, [1, 0]] = locs_mat2[:, [0, 1]] 
    # Get homography by RANSAC
    best_H2to1, inliers = computeH_ransac(locs_mat1, locs_mat2, opts)
    
    scale_x = img_source.shape[0]/img_replacement.shape[0]
    scale_y = img_source.shape[1]/img_replacement.shape[1]

    res = cv2.resize(img_replacement,(0,0),fx = scale_y, fy = scale_x ,interpolation = cv2.INTER_CUBIC)

    
    dst = cv2.warpPerspective(res,best_H2to1,(img_target.shape[1],img_target.shape[0]))

    mask = np.ones((img_source.shape))
    
    mask_warp = cv2.warpPerspective(mask,best_H2to1,(img_target.shape[1],img_target.shape[0]))

    indices1 = mask_warp==1
    indices2 = mask_warp==0

    mask_warp[indices1] = 0
    mask_warp[indices2] = 1

    res = img_target*mask_warp


    # Create a composite image after warping the replacement 
    # image on top of the target image using the homography
    composite_img = cv2.addWeighted(res,1.0,dst,1.0,0,dtype = cv2.CV_64F)
    

    

    return composite_img
