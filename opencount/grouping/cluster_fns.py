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

def hag_cluster_maketree(data, distfn=None, VERBOSE=True):
    """ Performs Hierarchical-Agglomerative Clustering on DATA. Returns
    a dendrogram (i.e. tree), where the children of a node N is 
    considered to have been 'merged' into the cluster denoted by N.
    Input:
        array DATA: An NxM array, where N is the number of observations,
            and M is the dimensionality of the feature space.
    Output:
        A tree structure, which consists of the results of merges during
        the agglomerative clustering. The leaf nodes contain indices into
        rows of the original DATA matrix.
    """
    if distfn == None:
        distfn = lambda a,b: np.linalg.norm(a-b)
    #dists = scipy.spatial.distance.pdist(data, metric='euclidean')
    clusters = [HAG_Leaf(rowidx) for rowidx in xrange(data.shape[0])]
    curiter = 0
    while len(clusters) != 1:
        if VERBOSE:
            print "...HAG Cluster iteration: {0}".format(curiter)
        # 0.) Compute pair-wise distances between all clusters
        c1_min, c2_min, mindist = None, None, None
        #dists = np.zeros((len(clusters), len(clusters)))
        for i, c1 in enumerate(clusters):
            for j, c2 in enumerate(clusters):
                if i == j: 
                    continue
                dist = distfn(c1.compute_centroid(data), c2.compute_centroid(data))
                if mindist == None or dist < mindist:
                    c1_min = c1
                    c2_min = c2
                    mindist = dist
        # 1.) Merge two-closest clusters.
        if VERBOSE:
            print "...Merging clusters {0}, {1}. Dist: {2}".format(c1_min, c2_min, mindist)
        parent = HAG_Node((c1_min, c2_min), dist=mindist)
        c1_min.parent = parent
        c2_min.parent = parent
        clusters = [c for c in clusters if c not in (c1_min, c2_min)]
        clusters.append(parent)
        curiter += 1
    return clusters[0]

def hag_cluster_flatten(data, C=0.8):
    """ Performs Hierarchichal-Agglomerative Clustering on DATA, and 
    through the use of a threshold C, tries to infer the 'natural'
    clustering by returning a clustering of DATA.
    Input:
        array DATA: An NxM array, N being the number of observations,
        M being the dimensionality.
        float C: Threshold value to use for inferring when to split
        groups. Should range from [0.0, 1.0].
    Output:
        array assigns: An N-vector, where each N[i] contains an integer
            saying which cluster N[i] belongs to.
    """
    # TODO: Normalize (somehow) the node.dist, so that the C threshold
    # makes sense.
    def flatten_hag_tree(node, C):
        if node.isleaf():
            return ((node.row,),)
        elif node.dist < C:
            return (node.get_idxs(),)
        else:
            idxs = []
            for child in node.children:
                idxs.extend(flatten_hag_tree(child, C))
            return idxs
    # 0.) Create HAG Cluster Tree
    tree = hag_cluster_maketree(data)
    # 1.) Create the ASSIGNS return value.
    assigns = np.zeros(data.shape[0])
    clusters = flatten_hag_tree(tree, C)
    for clusterID, cluster in enumerate(clusters):
        for row in cluster:
            assigns[row] = clusterID
    return assigns
    
class Node(object):
    def __init__(self, row=None, children=None, parent=None):
        raise NotImplementedError

    def compute_centroid(self, method='mean'):
        raise NotImplementedError

class HAG_Node(Node):
    def __init__(self, children=None, parent=None, dist=None):
        self.children = children
        self.parent = parent
        self.dist=dist
    def isleaf(self):
        return False
    def compute_centroid(self, data, method='mean'):
        _sum = 0.0
        if method == 'mean':
            for child in self.children:
                _sum += child.compute_centroid(data)
            return _sum / len(self.children)
        print "Unrecognized method:", method
        return 1.0
    def get_idxs(self):
        idxs = []
        for c in self.children:
            idxs.extend(c.get_idxs())
        return idxs
        
    def __eq__(self, o):
        return (o and isinstance(o, HAG_Node) and self.children == o.children)
    def __repr__(self):
        return "HAG_Node({0})".format([repr(c) for c in self.children])
    def __str__(self):
        return "HAG_Node({0} children)".format(len(self.children))

class HAG_Leaf(Node):
    def __init__(self, row, parent=None):
        self.row = row
        self.parent = parent
    def isleaf(self):
        return True
    def get_idxs(self):
        return (self.row,)
    def compute_centroid(self, data, method='mean'):
        return data[self.row]
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
    print "assigns is:", assigns
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
    #data = np.random.random((400, 2))
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

    
