# Import necessary functions
import skimage
import skimage.io
from loadVid import loadVid
import imageio
from opts import get_opts
from ar_helper import composeWarpedImg
import matplotlib.pyplot as plt
import numpy as np

import cv2
import time
opts = get_opts()

def superimpose(ref,samp_src,samp_book,opts):
    _,cov_w,_ = ref.shape
    _,W,_ = samp_src.shape

    # crop src_frames so that it has the same aspect ratio as the cv_cover
    crop_img = samp_src[:,W//2 - cov_w//2:W//2 + cov_w//2,:]
    
    # use ar_helper.composeWarpedImg to place each src_frame correctly over each book_frame, using cv_cover as a reference for warping
    cmp_img = composeWarpedImg(ref, samp_book, crop_img, opts)

    return cmp_img


def main():
    # Write function for Q3.1

    # load in necessary data
    src_frames = loadVid('../data/ar_source.mov')
    
    # Cut the zero padded region and convert to RGB
    src_frames = src_frames[:, 48:-48, :, ::-1]
    book_frames = loadVid('../data/book.mov')
    book_frames = book_frames[:, :, :, ::-1]  # convert to RGB

    rem_src_frames = src_frames[:len(book_frames)-len(src_frames)]

    # also pad at the end of src_frames with the beginning of src_frames until number of srcframes equals number of book_frames
    src_frames = np.concatenate((src_frames,rem_src_frames),axis=0)
    
    cv_cover = skimage.io.imread('../data/cv_cover.jpg')
    
    cv_cover = np.stack((cv_cover,)*3, axis = -1)

    composite_frames = []
 
    t = time.time()
    for f_num in range(len(src_frames)):
        print(f_num)
        samp_src = src_frames[f_num]

        samp_book = book_frames[f_num]

        cmp_img = superimpose(cv_cover,samp_src,samp_book,opts)
        cmp_img = cmp_img.astype(np.uint8)

        # save composite frames into a video, where composite_frames is a list of image arrays of shape (h, w, 3) obtained above
        composite_frames.append(cmp_img)
        
    
    imageio.mimwrite('../result/ar.avi', composite_frames, fps=30)
    el = time.time() - t
    print('time elapsed for 10 frames is: ',el)
    #--------------------------------------------------------------------------
    
    



if __name__ == '__main__':
    main()
