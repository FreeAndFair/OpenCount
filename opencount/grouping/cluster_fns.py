import sys, os, pdb, time, random
import numpy as np
import scipy
import matplotlib.pyplot as plt

"""
A collection of general-purpose clustering algorithms.
"""

def kmeans(data, initial=None, K=2, distfn=None, centroidfn=None,
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
            of two arguments, and returns a float.
        fn centroidfn: The function used to compute the centroid of a
            cluster, during the update_means step. Should be a function
            that, given a NxM array, returns the 'centroid' of the N
            observations (where 'centroid' can be the mean, etc.).
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
                dist = distfn(data[i,:], mean)
                if bestidx == None or dist < mindist:
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
            means[i] = np.mean(rows)
        return means

    if distfn == None:
        distfn = lambda a,b: np.linalg.norm(a-b)
    if centroidfn == None:
        centroidfn = np.mean

    if initial == None:
        initial_idxs = []
        _len = range(len(data))
        for _ in xrange(K):
            _i = random.choice(_len)
            while _i in initial_idxs:
                _i = random.choice(_len)
            initial_idxs.append(_i)
        initial = data[initial_idxs]
    if VERBOSE:
        print "...initial means:", initial
    means = initial
    assigns = np.zeros(data.shape[0])
    done = False
    iters = 0
    while not done:
        if VERBOSE:
            print "...kmeans iteration", iters
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
    print "assigns is:", assigns
    cluster1 = data1[np.where(assigns == 0)]
    cluster2 = data1[np.where(assigns == 1)]
    plt.plot(cluster1[:, 0], cluster1[:, 1], 'ro')
    plt.plot(cluster2[:, 0], cluster2[:, 1], 'bo')
    plt.ylabel('The Y Axis')
    plt.show()

def main():
    test_kmeans()

if __name__ == '__main__':
    main()

    
