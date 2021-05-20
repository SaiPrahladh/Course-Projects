import cv2
from helper import plotMatches
from opts import get_opts

from ar_helper import matchPics, briefRotTest, composeWarpedImg
from planarH import computeH, computeH_norm, computeH_ransac

def main():
    opts = get_opts()

    cv_cover = cv2.imread('../data/cv_cover.jpg')
    cv_desk = cv2.imread('../data/cv_desk.png')
    hp_cover = cv2.imread('../data/hp_cover.jpg')

    # Q2.1.4
    matches, locs1, locs2 = matchPics(cv_cover, cv_desk, opts)
    # Display matched features
    plotMatches(cv_cover, cv_desk, matches, locs1, locs2)

    # Q2.1.6
    briefRotTest(cv_cover, opts)


    # Q2.2.4
    composite_img = composeWarpedImg(cv_cover, cv_desk, hp_cover, opts)
    
    cv2.imwrite('../result/hp_desk_800_0.2.png', composite_img)


if __name__ == '__main__':
    main()
