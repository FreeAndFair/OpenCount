import sys
import pdb
import random
import traceback
import numpy as np
import matplotlib.pyplot as plt

from util import debug, warn, error


import pixel_reg.shared as shared

"""
A collection of general-purpose clustering algorithms.
"""


def kmeans(data, initial=None, K=2, distfn_method='L2', centroidfn_method='mean',
           VERBOSE=True):
    """ Performs K-Means clustering on DATA, with an optional initial
    set of means INITIAL.
    Input:
        obj data: NxM array, where N is number of observations, and M
            is the dimension of the data.
        list initial: If given, this should be a list of K initial
            means, used to initialize the algorithm.
        int K: number of clusters
        fn distfn: The distance function to use. Should be a function
            of two arguments, and returns a float. Defaults to L2-Distance.
        fn centroidfn: The function used to compute the centroid of a
            cluster, during the update_means step. Should be a function
            that, given a NxM array, returns the 'centroid' of the N
            observations (where 'centroid' can be the mean, etc.).
            Defaults to the mean of all data points in a cluster.
    Output:
        assigns, an array of N indices, where each N_i says which of
        the K clusters observation N_i belongs to.
    """
    def assignment(data, assigns, means, distfn):
        """ For each observation A in DATA, assign A to the closest
        mean in MEANS, by mutating ASSIGNS.
        """
        for i in xrange(data.shape[0]):
            bestidx, mindist = None, None
            for idx, mean in enumerate(means):
                dist = distfn(data[i, :], mean)
                if bestidx is None or dist < mindist:
                    bestidx = idx
                    mindist = dist
            assigns[i] = bestidx
        return assigns

    def update_means(data, assigns, means, distfn, centfn):
        """ For the clustering specified by ASSGNS, compute new means
        by mutating MEANS.
        """
        for i in xrange(len(means)):
            rows = data[np.where(assigns == i)]
            means[i] = centfn(rows)
        return means
    if distfn_method == 'L2':
        distfn = lambda a, b: np.linalg.norm(a - b)
    elif distfn_method == 'vardiff':
        distfn = vardiff
    else:
        distfn = lambda a, b: np.linalg.norm(a - b)
    if centroidfn_method == 'mean':
        centroidfn = np.mean
    elif centroidfn_method == 'median':
        centroidfn = np.median
    else:
        centroidfn = np.mean

    if initial is None:
        initial_idxs = []
        _len = range(len(data))
        for _ in xrange(K):
            _i = random.choice(_len)
            while _i in initial_idxs:
                _i = random.choice(_len)
            initial_idxs.append(_i)
        initial = data[initial_idxs]
    debug("initial means: {0}", initial)
    means = initial
    assigns = np.zeros(data.shape[0])
    done = False
    iters = 0
    while not done:
        debug("kmeans iteration", iters)
        # 1.) Assignment of data to current means
        prev_assigns = assigns.copy()
        assigns = assignment(data, assigns, means, distfn)
        # 2.) Halt if assignments don't change
        if np.all(np.equal(prev_assigns, assigns)):
            done = True
        else:
            # 3.) Re-compute means from new clusters
            means = update_means(data, assigns, means, distfn, centroidfn)
            iters += 1
    return assigns


def kmeans_2D(data, initial=None, K=2, distfn_method='L2',
              assigns=None,
              MAX_ITERS=200, VERBOSE=True):
    """ Runs K-means on data.
    Input:
        nparray data: An NxHxW matrix, where N is the number of
            observations, and (H,W) is the shape of each image.
        list initial: A list of initial means to use.
            If not given, then this will be randomly generated.
        int K:
        str distfn_method:
    Output:
        An ASSIGNS list of idxs.
    """
    def assignment(data, assigns, means, distfn):
        """ For each observation A in DATA, assign A to the closest
        mean in MEANS, by mutating ASSIGNS.
        """
        for i in xrange(data.shape[0]):
            bestidx, mindist = None, None
            for idx, mean in enumerate(means):
                I = data[i, :, :]
                try:
                    dist = distfn(I, mean)
                except:
                    traceback.print_exc()
                    pdb.set_trace()
                if dist == np.nan:
                    error("Uhoh, nan dist.")
                    pdb.set_trace()
                if bestidx is None or dist < mindist:
                    if dist == mindist:
                        # To prevent cycles, always tie-break via smallest
                        # index.
                        bestidx = min(bestidx, idx)
                    else:
                        bestidx = idx
                    mindist = dist
            assigns[i] = bestidx
        return assigns

    def update_means(data, assigns, means):
        """ For the clustering specified by ASSGNS, update MEANS. """
        for i in xrange(len(means)):
            cluster_i = data[np.where(assigns == i)]
            if len(cluster_i) == 0:
                # Empty cluster - reinitialize with a random datapoint
                warn("Empty cluster for mean {0}, reinitializing.", i)
                means[i] = random.choice(data)
                continue
            mean_i = mean_nan(cluster_i)
            if len(mean_i[~np.isnan(mean_i)]) == 0:
                error("Uhoh, only NaN's here.")
                pdb.set_trace()
            means[i] = mean_i
        return means

    def init_means(data):
        initial_idxs = []
        _len = range(data.shape[0])
        for _ in xrange(K):
            _i = random.choice(_len)
            while _i in initial_idxs:
                _i = random.choice(_len)
            initial_idxs.append(_i)
        return initial_idxs

    distfn = _get_distfn(distfn_method)
    if initial is None:
        means = data[init_means(data)]
    else:
        means = initial
    # TODO: Why infinite loop?
    # initial_idxs = [np.array([16]), np.array([23])]
    debug("...initial means: {0}", means)
    assigns = np.zeros(data.shape[0])
    done = False
    iters = 0
    prevprev_assigns = None
    while not done:
        debug("kmeans iteration {0}", iters)
        if iters >= MAX_ITERS:
            debug("Exceeded MAX_ITERS: {0}", MAX_ITERS)
            done = True
        # 1.) Assignment of data to current means
        prev_assigns = assigns.copy()
        assigns = assignment(data, assigns, means, distfn)
        # 2.) Halt if assignments don't change
        if np.all(np.equal(prev_assigns, assigns)):
            done = True
        elif prevprev_assigns is not None and np.all(np.equal(prevprev_assigns, assigns)):
            warn("len-2 Cycle detected, restarting")
            means = update_means(data, assigns, means)
            iters += 1
            # means = data[init_means(data)]
            # assigns = np.zeros(data.shape[0])
            # prevprev_assigns = None
        else:
            # 3.) Re-compute clusters from new clusters
            means = update_means(data, assigns, means)
            prevprev_assigns = prev_assigns
            iters += 1
    if np.all(assigns == assigns[0]):
        # Currently, this happens if all elements in DATA are 'too close',
        # i.e. distfn always outputs 0.0.
        debug("Degenerate clustering detected - splitting evenly.")
        _chunk = int(len(assigns) / K)
        out = np.zeros(data.shape[0])
        for i in xrange(K - 1):
            out[i * _chunk:(i + 1) * _chunk] = i
        out[(K - 1) * _chunk:] = (K - 1)
        return out
    return assigns


def kmediods_2D(data, initial=None, K=2, distfn_method='L2',
                MAX_ITERS=200, VERBOSE=True):
    """ Implements the K-Mediods algorithm. DATA should be an NxWxH matrix,
    where N is the number of observations, and WxH is the dimension of the
    images.
    """
    def compute_distmat(data, distfn):
        """ Computes the pairwise distance matrix. """
        out = np.zeros((data.shape[0], data.shape[0]))
        for i in xrange(data.shape[0]):
            for j in xrange(data.shape[0]):
                if i == j:
                    continue
                out[i, j] = distfn(data[i, :, :], data[j, :, :])
        return out

    def assignment(data, assigns, mediods, distfn, distmat):
        """ Assigns each data point to the nearest mediod, by mutating
        the input ASSIGNS.
        """
        for row in xrange(data.shape[0]):
            if row in mediods:
                # Data pt. is a mediod, should be assigned to itself
                assigns[row] = row
                continue
            mindist, bestidx = None, None
            for i, idx in enumerate(mediods):
                dist = distmat[row, idx]
                try:
                    foo = dist < mindist
                    bar = mindist is None
                    baz = foo or bar
                except:
                    pdb.set_trace()
                if mindist is None or dist < mindist:
                    mindist = dist
                    bestidx = idx
            assigns[row] = bestidx
        return assigns

    def update_mediods(data, assigns, mediods, distmat):
        """ Re-computes the optimal mediod for each cluster. Mutates the
        input MEDIODS.
        """
        for i, idx_mediod in enumerate(mediods):
            # elem_idxs: indices of elements in current mediod
            elem_idxs = np.where(assigns == idx_mediod)[0]
            # 1.) Choose the element in M that minimizes cost.
            mincost, minidx = None, None
            for elem_idx1 in elem_idxs:
                cost = 0
                for elem_idx2 in elem_idxs:
                    if elem_idx1 == elem_idx2:
                        continue
                    cost += distmat[elem_idx1, elem_idx2]
                if mincost is None or cost < mincost:
                    debug("swapped mediod: cost {0} -> {1}", mincost, cost)
                    mincost = cost
                    minidx = elem_idx1
            # 2.) Update the mediod of M.
            if minidx is None:
                error("Uhoh, problem.")
                pdb.set_trace()
            mediods[i] = minidx
        return mediods
    distfn = _get_distfn(distfn_method)
    debug("computing distance matrix")
    distmat = compute_distmat(data, distfn)
    debug("Finished computing distance matrix.")

    if initial is None:
        initial_idxs = []
        _len = range(data.shape[0])
        for _ in xrange(K):
            _i = random.choice(_len)
            while _i in initial_idxs:
                _i = random.choice(_len)
            initial_idxs.append(_i)
    if len(set(initial_idxs)) != K:
        error("Invalid setting of initial mediods.")
        pdb.set_trace()
    debug("...initial idxs:", initial_idxs)
    mediods = initial_idxs
    assigns = np.zeros(data.shape[0])

    done = False
    iters = 0
    prevprev_assigns = None
    while not done:
        debug("...kmediods iteration {0}", iters)
        if iters >= MAX_ITERS:
            debug("Exceeded MAX_ITERS:", MAX_ITERS)
            done = True
        # 1.) Assignment of data to current mediods
        prev_assigns = assigns.copy()
        assigns = assignment(data, assigns, mediods, distfn, distmat)
        # 2.) Halt if assignments don't change
        if np.all(np.equal(prev_assigns, assigns)):
            done = True
        elif prevprev_assigns is not None and np.all(np.equal(prevprev_assigns, assigns)):
            debug("len-2 Cycle detected, aborting.")
            done = True
        else:
            # 3.) Re-compute clusters from new clusters
            mediods = update_mediods(data, assigns, mediods, distmat)
            prevprev_assigns = prev_assigns
            iters += 1
    # 4.) Munge ASSIGNS to only be values from 0 to K-1.
    foo = set(assigns)
    assert len(foo) == K
    for k, val in enumerate(foo):
        assigns[assigns == val] = k
    return assigns


def _get_distfn(distfn_method):
    if distfn_method == 'L2':
        distfn = lambda a, b: np.linalg.norm(a - b, 2)
    elif distfn_method == 'L1':
        distfn = _L1
    elif distfn_method == 'vardiff':
        distfn = vardiff
    elif distfn_method == 'vardiff_align':
        distfn = vardiff_align
    elif distfn_method == 'imgdistortion':
        distfn = imgdistortiondiff
    elif distfn_method == 'imgdistortion_vardiff':
        distfn = imgdistortion_vardiff
    else:
        distfn = _L1
    return distfn

""" The following are distance functions, between two images A, B """


def _L1(A, B, debug=False):
    diff = np.abs(A - B)
    # size: Exclude NaN's
    size = len(diff[~np.isnan(diff)])
    err = np.sum(diff[np.nonzero(diff > 0)])
    if debug and err == 0:
        pdb.set_trace()
    return err / size


def vardiff(A, B, debug=False):
    """ Computes the difference between A and B, but with an attempt to
    account for background color. Basically a 1-D version of
    shared.variableDiffThr
    """
    A_nonan = A[~np.isnan(A)]
    B_nonan = B[~np.isnan(B)]

    def estimateBg(I):
        hist = np.histogram(I, bins=10)
        return hist[1][np.argmax(hist[0])]
    A_bg = estimateBg(A_nonan)
    B_bg = estimateBg(B_nonan)

    Athr = (A_bg - A_nonan.min()) / 2
    Bthr = (B_bg - B_nonan.min()) / 2
    thr = min(Athr, Bthr)
    diff = np.abs(A - B)

    # sum values of diffs above  threshold
    err = np.sum(diff[np.nonzero(diff > thr)])
    dist = err / float(A_nonan.size)
    if debug and dist == 0:
        pdb.set_trace()
    return dist


def vardiff_align(A, B):
    err, diff, Ireg = shared.lkSmallLarge(A, B, 0, B.shape[0], 0, B.shape[1])
    return err / diff.size


def imgdistortiondiff(A, B, M=3):
    """ Returns the diff between A and B, but for each pixel P in A,
    compares P to the P' in B that is most similar, within a window of
    size MxM. Utilizes the 'Image Distortion Model'.
    TODO: This is probably very slow - might have to do this in Cython.
    """
    diff = 0.0
    for i in xrange(A.shape[0]):
        for j in xrange(A.shape[1]):
            p = A[i, j]
            if np.isnan(p):
                continue
            bb = (max(0, i - M),
                  min(A.shape[0] - 1, i + M),
                  max(0, j - M),
                  min(A.shape[1] - 1, j + M))
            region = B[bb[0]:bb[1], bb[2]:bb[3]]
            dist = np.nanmin(np.abs(p - region))
            if np.isnan(dist):
                continue
            diff += dist
    return diff


def imgdistortion_vardiff(A, B, M=3):
    """ Just like imgdistortiondiff, but also tries to estimate the
    background, just like vardiff.
    """
    A_nonan = A[~np.isnan(A)]
    B_nonan = B[~np.isnan(B)]

    def estimateBg(I):
        hist = np.histogram(I, bins=10)
        return hist[1][np.argmax(hist[0])]
    A_bg = estimateBg(A_nonan)
    B_bg = estimateBg(B_nonan)

    Athr = (A_bg - A_nonan.min()) / 2
    Bthr = (B_bg - B_nonan.min()) / 2
    thr = min(Athr, Bthr)
    A = A.copy()
    B = B.copy()
    A[A < thr] = 0
    B[B < thr] = 0
    return imgdistortiondiff(A, B)

"""
For the following, I is one data pt, C is a cluster of data pts. Used
in k-mediods to compute the distance between a point and a cluster.
"""


def mean_nan(A):
    """ Computes the mean of A, ignoring NaN's. A is an NxHxW matrix,
    where N is the number of elements in the cluster. """
    dat = np.ma.masked_array(A, np.isnan(A))
    mean = np.mean(dat, axis=0)
    return mean.filled(np.nan)


class Node(object):

    def __init__(self, row=None, children=None, parent=None):
        raise NotImplementedError


class HAG_Node(Node):

    def __init__(self, children=None, parent=None, dist=None):
        self.children = children
        self.parent = parent
        self.dist = dist

    def get_idxs(self):
        idxs = []
        for c in self.children:
            idxs.extend(c.get_idxs())
        return idxs

    def size(self):
        return 1 + sum([child.size() for child in self.children])

    def __eq__(self, o):
        return (o and isinstance(o, HAG_Node) and self.children == o.children)

    def __repr__(self):
        return "HAG_Node({0} elements)".format(self.size())

    def __str__(self):
        return "HAG_Node({0} elements)".format(self.size())


class HAG_Leaf(Node):

    def __init__(self, row, parent=None):
        self.row = row
        self.parent = parent

    def get_idxs(self):
        return (self.row,)

    def size(self):
        return 1

    def __eq__(self, o):
        return (o and isinstance(o, HAG_Leaf) and self.row == o.row)

    def __repr__(self):
        return "HAG_Leaf({0})".format(repr(self.row))

    def __str__(self):
        return "HAG_Leaf"


def test_kmeans():
    data1 = np.array([[1, 0],
                      [2, 1],
                      [1, 1],
                      [0, 1],
                      [25, 25],
                      [44, 45],
                      [44, 43],
                      [32, 45],
                      [48, 45]])
    data1 = np.random.random((400, 2))
    assigns = kmeans(data1)
    debug("assigns is: {0}", assigns)
    cluster1 = data1[np.where(assigns == 0)]
    cluster2 = data1[np.where(assigns == 1)]
    plt.plot(cluster1[:, 0], cluster1[:, 1], 'ro')
    plt.plot(cluster2[:, 0], cluster2[:, 1], 'bo')
    plt.ylabel('The Y Axis')
    plt.show()


def test_hac():
    import scipy.cluster.hierarchy as sch
    data = np.array([[1, 0],
                     [2, 1],
                     [1, 1],
                     [0, 1],
                     [25, 25],
                     [44, 45],
                     [44, 43],
                     [32, 45],
                     [48, 45]])
    # data = np.random.random((400, 2))
    Z = sch.linkage(data)
    T = sch.fcluster(Z, 0.5)
    colors = ('bo', 'go', 'ro', 'co', 'mo', 'yo', 'ko', 'wo')
    clusters = []
    clusterIDs = set(list(T))
    for i, clusterID in enumerate(clusterIDs):
        cluster_i = data[np.where(T == clusterID)]
        plt.plot(cluster_i[:, 0], cluster_i[:, 1], colors[i])

    plt.ylabel('The Y Axis')
    plt.show()


def main():
    args = sys.argv[1:]
    if args[0] == 'k':
        test_kmeans()
    elif args[0] == 'hac':
        test_hac()

if __name__ == '__main__':
    main()
