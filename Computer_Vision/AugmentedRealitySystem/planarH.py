import numpy as np


def computeH(locs1, locs2):
    # Q2.2.1
    # Compute the homography between two sets of points
    
    P = []

    # locs1 and locs2 are now of shape 2xN and not Nx2

    _,N = locs1.shape
    for i in range(N):
        p_i = np.array([[-locs2[0,i],-locs2[1,i],-1,0,0,0,locs2[0,i]*locs1[0,i],locs2[1,i]*locs1[0,i],locs1[0,i]]\
                         ,[0,0,0,-locs2[0,i],-locs2[1,i],-1,locs2[0,i]*locs1[1,i],locs2[1,i]*locs1[1,i],locs1[1,i]]])
        
        P.append(p_i)
    
    P = np.vstack(P)

    _,_,vh = np.linalg.svd(P)

    H2to1 = vh[-1].reshape(3,3)
    H2to1 /= (H2to1[-1,-1]+1e-16)
   
    return H2to1

def get_max_Euclidean(locs_cent):
    locs_cent = locs_cent.T
    og = [0,0]
    dist = (locs_cent -og)**2
    dist = np.sqrt(np.sum(dist,axis = 1))

    return np.max(dist)


def computeH_norm(locs1, locs2):
    # Q2.2.2
    # Compute the centroid of the points

    # Transposing the locations matrix to fit the dimensions
    # of matrix multiplication with H
    locs1 = locs1.T
    locs2 = locs2.T


    cent1 = np.mean(locs1,axis = 1).reshape(-1,1)
    cent2 = np.mean(locs2,axis = 1).reshape(-1,1)
    
    # Shift the origin of the points to the centroid
 
    dif_locs1 = locs1 - cent1
    dif_locs2 = locs2 - cent2
    
    
    s1 = get_max_Euclidean(dif_locs1)
    s2 = get_max_Euclidean(dif_locs2)

    
    # Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    dif_locs1/=(s1 + 1e-16)
    dif_locs2/=(s2 + 1e-16)
    

    # Similarity transform 1
    T1 = computeH(dif_locs1,locs1)
    # Similarity transform 2
    T2 = computeH(dif_locs2,locs2)
    # Compute homography
    H2to1_norm = computeH(dif_locs1,dif_locs2)
    # Denormalization
    try:
        H2to1 = np.matmul(np.matmul(np.linalg.inv(T1),H2to1_norm),T2)
        H2to1 /= (H2to1[-1,-1]+1e-16)
    except:
        H2to1 = np.zeros((3,3))
    return H2to1


def computeH_ransac(locs1, locs2, opts):
    # Q2.2.3
    # Compute the best fitting homography given a list of matching points
    max_iters = opts.max_iters  # the number of iterations to run RANSAC for
    # the tolerance value for considering a point to be an inlier
    inlier_tol = opts.inlier_tol
    H_score = 0
    
    for i in range(max_iters):
        idxs = np.random.choice(locs1.shape[0],4)

        l1_samp = locs1[idxs]    
        l2_samp = locs2[idxs]

        H = computeH_norm(l1_samp,l2_samp)

        locs2_tmp = np.concatenate((locs2,np.ones((locs2.shape[0],1))),axis=1)

        l2_proj = np.matmul(H,locs2_tmp.T)

        l2_proj/=( (l2_proj[-1,:]) + 1e-16)

        l2_proj = l2_proj[:2,:]

        dis_mat = np.linalg.norm(l2_proj.T-locs1,axis=1)

        tmp_inliers = np.argwhere(dis_mat < inlier_tol).flatten()

        cnt_inliers = len(tmp_inliers)

        if cnt_inliers > H_score:
            H_score = cnt_inliers
            indices = tmp_inliers
            best_H2to1 = H
        
    inliers = np.zeros(locs1.shape[0])
    inliers[indices] = 1
    

    return best_H2to1, inliers
