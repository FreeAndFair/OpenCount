import sys, os, pdb, pickle, time
sys.path.append('..')
from os.path import join as pathjoin
import numpy as np
import scipy.cluster.vq
import pylab, cv

import specify_voting_targets.util_gui as util_gui
import cluster_fns
import pixel_reg.imagesAlign as imagesAlign

"""
A script designed to cluster images.
"""

def cluster_imgs_pca_kmeans(imgpaths, bb_map=None, k=2, N=3, do_align=True):
    """ Using PCA and K-means, cluster the imgpaths into 'k' clusters,
    using the first 'N' principal components.
    Algorithm details:
        Input: Set of image patches A, of size NxM
        0.) Discretize the image patch into K N'xM' equally-sized slices.
        1.) Using the discretized image patches A', run PCA to extract
            the slices S that maximize the variance
        2.) Run k-means (k=2) on the slices S.
    Input:
        list imgpaths: (imgpath_i, ...)
        dict bb_map: If you want to only cluster based on a sub-region
                     of each image, pass in 'bb_map', which is:
                         {str imgpath: (y1,y2,x1,x2)}
        int k: number of clusters
        int N: Number of principle components to use. (NOT USED)
    Output:
        dict clusters, maps {str clusterID: [imgpath_i, ...]}
    """
    data = imgpaths_to_mat(imgpaths, bb_map=bb_map, do_align=do_align)
    '''
    if bb_map == None:
        bb_map = {}
        h_big, w_big = get_largest_img_dims(imgpaths)
    else:
        bb_big = get_largest_bb(bb_map.values())
        h_big = int(abs(bb_big[0] - bb_big[1]))
        w_big = int(abs(bb_big[2] - bb_big[3]))
    #TEMP: Cut off first half, last quarter
    #w_big = (w_big / 2) - (w_big / 4)
    # 0.) First, convert images into MxN array, where M is the number
    #     of images, and N is the number of pixels of each image.
    data = np.zeros((len(imgpaths), h_big*w_big))
    for row, imgpath in enumerate(imgpaths):
        img = scipy.misc.imread(imgpath, flatten=True)
        # TEMP: Cut off first half
        #img = img[:, img.shape[1]/2:(3*img.shape[1])/4]
        #img = util_gui.autothreshold_numpy(img, method="otsu")

        bb = bb_map.get(imgpath, None)
        if bb == None:
            patch = resize_mat(img, (h_big, w_big))
        else:
            # Must make sure that all patches are the same shape.
            patch = resize_mat(img[bb[0]:bb[1], bb[2]:bb[3]], (h_big, w_big))
        # Reshape 'patch' to be a single row of pixels, instead of rows
        # of pixels.
        patch = patch.reshape(1, patch.shape[0]*patch.shape[1])
        data[row,:] = patch
    '''
    # Inspiration for PCA-related code comes from:
    #     http://glowingpython.blogspot.it/2011/07/pca-and-image-compression-with-numpy.html

    # 1.) Call PCA on the data matrix, extract first N principle comps
    M = (data - np.mean(data.T, axis=1)) # subtract mean, along cols
    (latent, coeff) = np.linalg.eig(np.cov(M))
    p = np.size(coeff, axis=1)

    idx = pylab.argsort(latent)  # sort eigenvalues
    idx = idx[::-1]        # ascending order (i.e. by 'relevance')
    # idx is a sorted list of indices into imgpaths, i.e. if there
    # are 5 images, and idx is:
    #   idx := [4, 1, 3, 2, 0]
    # then this means that imgpaths[4] most explains the variance,
    # followed by imgpaths[1], etc.
    idx = idx[:k]
    cluster_centers = data[idx, :]
    clustering = {} # maps {int clusterID: [imgpath_i, ...]}
    
    # 2.) Nearest-Neighbors to cluster_centers
    for i, imgarray in enumerate(data):
        best_dist, best_j = None, None
        for j, clustercenter in enumerate(cluster_centers):
            dist = np.linalg.norm(imgarray - clustercenter)
            if best_dist == None or dist < best_dist:
                best_dist = dist
                best_j = j
        clustering.setdefault(best_j, []).append(imgpaths[i])
    return clustering

def cluster_imgs_kmeans(imgpaths, bb_map=None, k=2, do_chopmid=False, chop_prop=0.3,
                        do_downsize=False, downsize_amt=0.5, do_align=True):
    """ Using k-means, cluster the images given by 'imgpaths' into 'k'
    clusters.
    Note: This uses the Euclidean distance as the distance metric:
        dist(P,Q) = sum{x,y} sqrt((P[x,y] - Q[x,y])^2)

    Input:
        list imgpaths:
        dict bb_map: If you want to only cluster based on a sub-region
                     of each image, pass in 'bb_map', which is:
                         {str imgpath: (y1,y2,x1,x2)}
        int k:
        bool do_chopmid: If True, then this will only consider the middle
            portion of the patch.
    Output:
        Returns the clustering, in the form:
            {clusterID: [impath_i, ...]}
    """
    data = imgpaths_to_mat(imgpaths, bb_map=bb_map, do_align=do_align)
    '''
    if bb_map == None:
        bb_map = {}
        h_big, w_big = get_largest_img_dims(imgpaths)
    else:
        bb_big = get_largest_bb(bb_map.values())
        h_big = int(abs(bb_big[0] - bb_big[1]))
        w_big = int(abs(bb_big[2] - bb_big[3]))
    # 0.) First, convert images into MxN array, where M is the number
    #     of images, and N is the number of pixels of each image.
    data = np.zeros((len(imgpaths), h_big*w_big))
    for row, imgpath in enumerate(imgpaths):
        img = scipy.misc.imread(imgpath, flatten=True)
        bb = bb_map.get(imgpath, None)
        if bb == None:
            patch = resize_mat(img, (h_big, w_big))
        else:
            # Must make sure that all patches are the same shape.
            patch = resize_mat(img[bb[0]:bb[1], bb[2]:bb[3]], (h_big, w_big))
        # Reshape 'patch' to be a single row of pixels, instead of rows
        # of pixels.
        patch = patch.reshape(1, patch.shape[0]*patch.shape[1])
        data[row,:] = patch
    '''

    # 1.) Call scipy's kmeans implementation
    centroids, _ = scipy.cluster.vq.kmeans(data, k)
    # 2.) Assign each image to a cluster
    idx, dists = scipy.cluster.vq.vq(data, centroids)
    clusters = {}
    for i, clusterid in enumerate(idx):
        clusters.setdefault(clusterid, []).append(imgpaths[i])
    return clusters

def cluster_imgs_kmeans_mine(imgpaths, bb_map=None, k=2, distfn_method='L2', centroidfn_method='mean', do_align=True):
    data = imgpaths_to_mat(imgpaths, bb_map=bb_map, do_align=do_align)
    print "Running k-means..."
    t = time.time()
    assigns = cluster_fns.kmeans(data, K=k, distfn_method=distfn_method, centroidfn_method=centroidfn_method)
    dur = time.time() - t
    print "...Finished k-means ({0} s).".format(dur)
    clusters = {} # maps {clusterID: [imgpath_i, ...]}
    for i in xrange(k):
        cluster = []
        for idx in np.where(assigns == i)[0]:
            cluster.append(imgpaths[idx])
        print "    cluster {0}: {1} elements.".format(i, len(cluster))
        clusters[i] = cluster
    return clusters

def cluster_imgs_kmeans_alignerr(imgpaths, bb_map=None, k=2, distfn_method=None, centroidfn_method=None, do_align=True):
    data, errs = imgpaths_to_mat(imgpaths, bb_map=bb_map, do_align=do_align, return_align_errs=True)
    print "Running k-means..."
    t = time.time()
    assigns = cluster_fns.kmeans(errs, K=k, distfn_method=distfn_method, centroidfn_method=centroidfn_method)
    dur = time.time() - t
    print "...Finished k-means ({0} s).".format(dur)
    clusters = {}
    for i in xrange(k):
        cluster = []
        for idx in np.where(assigns == i)[0]:
            cluster.append(imgpaths[idx])
        print "    cluster {0}: {1} elements.".format(i, len(cluster))
        clusters[i] = cluster
    return clusters

def cluster_imgs_hag(imgpaths, bb_map=None, do_align=True, distfn_method='L2', centroidfn_method='simple'):
    data = imgpaths_to_mat(imgpaths, bb_map=bb_map, do_align=do_align)
    print "Running HAG-Clustering..."
    t = time.time()
    assigns = cluster_fns.hag_cluster_flatten(data, C=0.8, distfn_method=distfn_method, centroidfn_method=centroidfn_method)
    dur = time.time() - t
    print "...Finished HAG-Clustering ({0} s).".format(dur)
    clusters = {}
    k = len(set(list(assigns)))
    for i in xrange(k):
        cluster = []
        for idx in np.where(assigns == i)[0]:
            cluster.append(imgpaths[idx])
        print "    cluster {0}: {1} elements.".format(i, len(cluster))
        clusters[i] = cluster
    return clusters

def kmeans_2D(imgpaths, bb_map=None, k=2, distfn_method=None,
              do_align=True, do_edgedetect=False):
    data = imgpaths_to_mat2D(imgpaths, bb_map=bb_map, do_align=do_align, do_edgedetect=do_edgedetect)
    
    print "Running k-means..."
    t = time.time()
    assigns = cluster_fns.kmeans_2D(data, K=k, distfn_method=distfn_method)
    dur = time.time() - t
    print "...Finished k-means ({0} s).".format(dur)
    clusters = {} # maps {clusterID: [imgpath_i, ...]}
    for i in xrange(k):
        cluster = []
        for idx in np.where(assigns == i)[0]:
            cluster.append(imgpaths[idx])
        print "    cluster {0}: {1} elements.".format(i, len(cluster))
        clusters[i] = cluster
    return clusters    

def kmediods_2D(imgpaths, bb_map=None, k=2, distfn_method=None,
                do_align=True, do_edgedetect=False):
    data = imgpaths_to_mat2D(imgpaths, bb_map=bb_map, do_align=do_align, do_edgedetect=do_edgedetect)
    
    print "Running k-mediods..."
    t = time.time()
    assigns = cluster_fns.kmediods_2D(data, K=k, distfn_method=distfn_method)
    dur = time.time() - t
    print "...Finished k-mediods ({0} s).".format(dur)
    clusters = {} # maps {clusterID: [imgpath_i, ...]}
    for i in xrange(k):
        cluster = []
        for idx in np.where(assigns == i)[0]:
            cluster.append(imgpaths[idx])
        print "    cluster {0}: {1} elements.".format(i, len(cluster))
        clusters[i] = cluster
    return clusters  

def imgpaths_to_mat(imgpaths, bb_map=None, do_align=False, return_align_errs=False):
    """ Reads in a series of imagepaths, and converts it to an NxM
    matrix, where N is the number of images, and M is the (w*h), where
    w,h are the width/height of the largest image in IMGPATHS.
    If BB_MAP is given, then this will extract a patch from the 
    associated IMGPATH.
    """
    if bb_map == None:
        bb_map = {}
        h_big, w_big = get_largest_img_dims(imgpaths)
    else:
        bb_big = get_largest_bb(bb_map.values())
        h_big = int(abs(bb_big[0] - bb_big[1]))
        w_big = int(abs(bb_big[2] - bb_big[3]))
    # 0.) First, convert images into MxN array, where M is the number
    #     of images, and N is the number of pixels of each image.
    data = np.zeros((len(imgpaths), h_big*w_big))
    Iref = None
    alignerrs = [None] * len(imgpaths) # [float err_i, ...]
    alignerrs = np.zeros((len(imgpaths), 1))
    for row, imgpath in enumerate(imgpaths):
        img = scipy.misc.imread(imgpath, flatten=True)
        bb = bb_map.get(imgpath, None)
        if bb == None:
            patch = resize_mat(img, (h_big, w_big))
        else:
            # Must make sure that all patches are the same shape.
            patch = resize_mat(img[bb[0]:bb[1], bb[2]:bb[3]], (h_big, w_big))
        if do_align and Iref == None:
            Iref = patch
        elif do_align:
            H, patch, err = imagesAlign.imagesAlign(patch, Iref)
            if return_align_errs:
                alignerrs[row] = err
            patch = np.nan_to_num(patch)
        # Reshape 'patch' to be a single row of pixels, instead of rows
        # of pixels.
        patch = patch.reshape(1, patch.shape[0]*patch.shape[1])
        data[row,:] = patch
    if return_align_errs:
        return data, alignerrs
    return data

def imgpaths_to_mat2D(imgpaths, bb_map=None, do_align=False, return_align_errs=False,
                      do_edgedetect=False, LOW_T=75, RATIO=3):
    if bb_map == None:
        bb_map = {}
        h_big, w_big = get_largest_img_dims(imgpaths)
    else:
        bb_big = get_largest_bb(bb_map.values())
        h_big = int(abs(bb_big[0] - bb_big[1]))
        w_big = int(abs(bb_big[2] - bb_big[3]))
    # 0.) First, convert images into MxHxW array, where M is the number
    #     of images, and (H,W) are image sizes.
    data = np.zeros((len(imgpaths), h_big, w_big))
    Iref = None
    alignerrs = [None] * len(imgpaths) # [float err_i, ...]
    alignerrs = np.zeros((len(imgpaths), 1))
    for row, imgpath in enumerate(imgpaths):
        img = scipy.misc.imread(imgpath, flatten=True)
        bb = bb_map.get(imgpath, None)
        if bb == None:
            patch = resize_mat(img, (h_big, w_big))
        else:
            # Must make sure that all patches are the same shape.
            patch = resize_mat(img[bb[0]:bb[1], bb[2]:bb[3]], (h_big, w_big))
        
        if do_edgedetect:
            patch = edgedetect(patch)
        if do_align and Iref == None:
            Iref = patch
        elif do_align:
            H, patch, err = imagesAlign.imagesAlign(patch, Iref)
            patch_img = np.nan_to_num(patch)
            #patch = patch_img
            try:
                os.makedirs("alignedimgs")
            except:
                pass
            scipy.misc.imsave(os.path.join("alignedimgs", "{0}_{1}.png".format(row, err)),
                              patch_img)
            print "alignment err:", err
            if return_align_errs:
                alignerrs[row] = err

        data[row,:,:] = patch
    if return_align_errs:
        return data, alignerrs
    return data

def edgedetect(I_np, LOW_T=75, RATIO=3):
    I_cv = cv.fromarray(I_np)
    I_cv8U = cv.CreateMat(I_cv.rows, I_cv.cols, cv.CV_8U)
    cv.Convert(I_cv, I_cv8U)
    edges = cv.CreateMat(I_cv8U.rows, I_cv8U.cols, cv.CV_8U)

    cv.Canny(I_cv8U, edges, LOW_T, LOW_T*RATIO)
    edges_32f = cv.CreateMat(edges.rows, edges.cols, cv.CV_32F)
    cv.Convert(edges, edges_32f)
    edges_np = np.array(edges_32f)
    return edges_np

def smooth(I):
    pass

def get_largest_img_dims(imgpaths):
    """ Returns the largest dimensions of the images in imgpaths. """
    h, w = None, None
    for imgpath in imgpaths:
        img = scipy.misc.imread(imgpath)
        if h == None or img.shape[0] > h:
            h = img.shape[0]
        if w == None or img.shape[1] > w:
            w = img.shape[1]
    return (h, w)
def get_largest_bb(bbs):
    """ Returns the largest bb in bb_map.
    Input:
        List of (y1,y2,x1,x2)
    Output:
        The largest bb in all dims, (y1',y2',x1',x2').
    """
    outbb = []
    for i in range(4):
        outbb.append(max(map(lambda _bb: _bb[i], bbs)))
    return outbb

def resize_mat(mat, shape):
    """ Takes in an NxM matrix 'mat', and either truncates or pads
    it so that it has the 'shape'.
    Input:
        obj mat: an NxM numpy array
        tuple shape: (h,w)
    Output:
        An hxw sized array.
    """
    out = np.zeros(shape)
    i = min(mat.shape[0], out.shape[0])
    j = min(mat.shape[1], out.shape[1])
    out[0:i,0:j] = mat[0:i, 0:j]
    return out

def _test_resize_mat():
    foo = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print 'foo is:', foo
    out1 = resize_mat(foo, (2, 2))
    print 'out1 is:', out1
    out2 = resize_mat(foo, (4, 5))
    print 'out2 is:', out2
    out3 = resize_mat(foo, (3, 3))
    print 'out3 is:', out3

def test_clusterimgs_kmeans():
    def get_bb(digitsdir):
        shapes = []
        for dirname, dirnames, filenames in os.walk(digitsdir):
            for f in filenames:
                if f.lower().endswith('.png'):
                    imgpath = pathjoin(dirname, f)
                    shape = scipy.misc.imread(imgpath, flatten=True).shape
                    shapes.append((0,shape[0],0,shape[1]))
        return get_largest_bb(shapes)
            
    rootdir = 'test_cluster_kmeans_imgs'
    basedir = os.path.join(rootdir, 'orangecounty')
    
    dir_0 = pathjoin(basedir, '0')
    dir_1 = pathjoin(basedir, '1')
    dir_3 = pathjoin(basedir, '3')
    dir_4 = pathjoin(basedir, '4')
    dir_6 = pathjoin(basedir, '6')
    dir_8 = pathjoin(basedir, '8')
    
    imgs_0 = get_imgpaths(dir_0)
    imgs_1 = get_imgpaths(dir_1)
    imgs_3 = get_imgpaths(dir_3)
    imgs_4 = get_imgpaths(dir_4)
    imgs_6 = get_imgpaths(dir_6)
    imgs_8 = get_imgpaths(dir_8)
    
    bb = get_bb(basedir)
    bb_map_01 = {}
    for imgpath in imgs_0 + imgs_1:
        bb_map_01[imgpath] = bb
    
    clusters = cluster_imgs_kmeans(imgs_0 + imgs_1, bb_map=bb_map_01)
    for clusterid, path in clusters.iteritems():
        print '{0} : {1}'.format(clusterid, path)

    
def get_imgpaths(dir):
    paths = []
    for dirpath, dirnames, filenames in os.walk(dir):
        for filename in filenames:
            if filename.lower().endswith('.png'):
                paths.append(pathjoin(dirpath, filename))
    return paths

def main():
    test_clusterimgs_kmeans()

if __name__ == '__main__':
    main()

