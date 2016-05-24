import os
import pdb
import time
import traceback
from os.path import join as pathjoin

try:
    import cPickle as pickle
except ImportError as e:
    import pickle

import wx
import numpy as np
import scipy.cluster.vq
from scipy import misc
from util import error


import pixel_reg.shared as sh
import pixel_reg.part_match as part_match

from imageviewer import WorldState as WorldState
from imageviewer import BoundingBox as BoundingBox
import cust_attrs
import cluster_imgs
import partask

DUMMY_ROW_ID = -42  # Also defined in label_attributes.py

# Special ID's used for Attributes
TABULATION_ONLY_ID = 1
DIGIT_BASED_ID = 2


class AttributeBox(BoundingBox):
    """
    Represents a bounding box around a ballot attribute.
    """

    def __init__(self, x1, y1, x2, y2, label='', color=None,
                 id=None, is_contest=False, contest_id=None,
                 target_id=None,
                 line_width=None, children=None,
                 is_new=False, is_selected=False,
                 attrs=None, side='front', is_digitbased=False,
                 is_tabulationonly=False):
        """
        attrs is a dict mapping:
            {str attrtype: str attrval}
        is_tabulationonly: True if this Attribute is not used for grouping
                           purposes, i.e. is only for tabulation purposes.
        """
        BoundingBox.__init__(self, x1, y1, x2, y2, label=label, color=color,
                             id=id, is_contest=is_contest, contest_id=contest_id,
                             target_id=target_id,
                             line_width=line_width, children=children,
                             is_new=is_new, is_selected=is_selected)
        self.attrs = attrs if attrs else {}
        self.side = side
        self.is_digitbased = is_digitbased
        self.is_tabulationonly = is_tabulationonly

    def get_attrtypes(self):
        return tuple(self.attrs.keys())

    def copy(self):
        """ Return a copy of myself """
        return AttributeBox(self.x1, self.y1,
                            self.x2, self.y2, label=self.label,
                            color=self.color, id=self.id, is_contest=self.is_contest,
                            contest_id=self.contest_id, is_new=self.is_new,
                            is_selected=self.is_selected,
                            target_id=self.target_id,
                            attrs=self.attrs,
                            side=self.side, is_digitbased=self.is_digitbased,
                            is_tabulationonly=self.is_tabulationonly)

    def marshall(self):
        """
        Return a dict-equivalent of myself.
        """
        data = BoundingBox.marshall(self)
        data['attrs'] = self.attrs
        data['side'] = self.side
        data['is_digitbased'] = self.is_digitbased
        data['is_tabulationonly'] = self.is_tabulationonly
        return data

    @staticmethod
    def unmarshall(data):
        box = AttributeBox(0, 0, 0, 0)
        for (propname, propval) in data.iteritems():
            setattr(box, propname, propval)
        return box

    def __eq__(self, a):
        return (a and self.x1 == a.x1 and self.y1 == a.y1 and self.x2 == a.x2
                and self.y2 == a.y2 and self.is_contest == a.is_contest
                and self.label == a.label and self.attrs == a.attrs
                and self.side == a.side and self.is_digitbased == a.is_digitbased
                and self.is_tabulationonly == a.is_tabulationonly)

    def __repr__(self):
        return "AttributeBox({0},{1},{2},{3},attrs={4},side={5},is_digitbased{6},is_tabonly{7})".format(self.x1, self.y1, self.x2, self.y2, self.attrs, self.side, self.is_digitbased, self.is_tabulationonly)

    def __str___(self):
        return "AttributeBox({0},{1},{2},{3},attrs={4},side={5},is_digitbased{6},is_tabonly{7})".format(self.x1, self.y1, self.x2, self.y2, self.attrs, self.side, self.is_digitbased, self.is_tabulationonly)


def resize_img_norescale(img, size):
    """ Resizes img to be a given size without rescaling - it only
    pads/crops if necessary.
    Input:
        obj img: a numpy array
        tuple size: (w,h)
    Output:
        A numpy array with shape (h,w)
    """
    w, h = size
    shape = (h, w)
    out = np.zeros(shape)
    i = min(img.shape[0], out.shape[0])
    j = min(img.shape[1], out.shape[1])
    out[0:i, 0:j] = img[0:i, 0:j]
    return out


def get_attrtypes(project, with_digits=True):
    """
    Returns all attribute types in this election. Excludes CustomAttributes,
    but does include DigitAttributes (if with_digits is True).
    """
    attrtypes = pickle.load(open(project.ballot_attributesfile, 'rb'))
    result = set()
    for attrdict in attrtypes:
        if not attrdict['is_digitbased']:
            attrs_str = get_attrtype_str(attrdict['attrs'])
            result.add(attrs_str)
        elif with_digits and attrdict['is_digitbased']:
            attrs_str = get_attrtype_str(attrdict['attrs'])
            result.add(attrs_str)
    return result


def exists_imgattrs(proj):
    """ Returns True if there exists at least one image based attribute
    (i.e. a non-custom+non-digit based attr).
    """
    # attrs does NOT include CustomAttributes (these are stored in
    # custom_attrs.p), so no need to check for them.
    attrs = pickle.load(open(proj.ballot_attributesfile, 'rb'))
    for attr in attrs:
        if not attr['is_digitbased']:
            return True
    return False


def is_tabulationonly(project, attrtype):
    """ Returns True if the attrtype is for tabulationonly. """
    # 1.) Try imgbased+digitbased attributes
    attrtypes_dicts = pickle.load(open(project.ballot_attributesfile, 'rb'))
    for attrdict in attrtypes_dicts:
        attrs_str = '_'.join(attrdict['attrs'])
        if attrs_str == attrtype:
            return attrdict['is_tabulationonly']
    # 2.) Try custom attributes
    customattrs = cust_attrs.load_custom_attrs(project)
    for cattr in customattrs:
        if cattr.attrname == attrtype:
            return cattr.is_tabulationonly
    # Means we can't find attrtype anywhere.
    assert False, "Can't find attrtype: {0}".format(attrtype)


def is_digitbased(project, attrtype):
    """ Returns True if the attrtype is digit-based. """
    attrtypes_dicts = pickle.load(open(project.ballot_attributesfile, 'rb'))
    for attrdict in attrtypes_dicts:
        attrs_str = '_'.join(attrdict['attrs'])
        if attrs_str == attrtype:
            return attrdict['is_digitbased']
    # 2.) Try custom attributes
    customattrs = cust_attrs.load_custom_attrs(project)
    if customattrs:
        for cattr in customattrs:
            if cattr.attrname == attrtype:
                return False
    # Means we can't find attrtype anywhere.
    assert False, "Can't find attrtype: {0}".format(attrtype)


def is_quarantined(project, path):
    """ Returns True if the image path was quarantined by the user. """
    if not os.path.exists(project.quarantined):
        return False
    f = open(project.quarantined, 'r')
    for line in f:
        if line:
            l = line.strip()
            if os.path.abspath(l) == os.path.abspath(path):
                return True
    f.close()
    return False


def get_attr_prop(project, attrtype, prop):
    """ Returns the property of the given attrtype. """
    ballot_attrs = pickle.load(open(project.ballot_attributesfile, 'rb'))
    for attrdict in ballot_attrs:
        attrstr = get_attrtype_str(attrdict['attrs'])
        if attrstr == attrtype:
            return attrdict[prop]
    error("couldn't find attribute: {0}", attrtype)
    pdb.set_trace()
    return None


def get_attrtype_str(attrtypes):
    """Returns a 'canonical' string-representation of the attributes
    of a ballot attribute.
    Useful if an AttributeBox has multiple attribute defined within
    it. The 'canonical' representation is:
        each attribute sorted in alphabetical order, separated by
        underscores '_'.
    Input:
        lst attrtypes: List of strings
    """
    return '_'.join(sorted(attrtypes))


def num_common_prefix(*args):
    """
    For each input list L, return the number of common elements amongst
    all lists (starting from L-R ordering).
    Assumes all input lists are of the same length.
    """
    result = 0
    for idx in range(len(args[0])):
        val = args[0][idx]
        for lst in args[1:]:
            if val != lst[idx]:
                return result
        result += 1
    return result


def is_img_ext(f):
    return os.path.splitext(f.lower())[1].lower() in ('.bmp', '.jpg',
                                                      '.jpeg', '.png',
                                                      '.tif', '.tiff')


def get_imagepaths(dir):
    """ Given a directory, return all imagepaths. """
    results = []
    for dirpath, dirnames, filenames in os.path.walk(dir):
        results.append([pathjoin(dirpath, imname)
                        for imname in filter(is_img_ext, filenames)])
    return results


def get_avglightest_img(imgpaths):
    """ Given a list of image paths, return the image with the lightest
    average intensity.
    """
    bestpath, bestscore, idx = None, None, None
    for i, imgpath in enumerate(imgpaths):
        img = scipy.misc.imread(imgpath, flatten=True)
        score = np.average(img)
        if bestpath is None or score > bestscore:
            bestpath = imgpath
            bestscore = score
            idx = i
    return bestpath, bestscore, idx


class GroupClass(object):
    """
    A class that represents a potential group of images.
    """
    # A dict mapping {str label: int count}
    ctrs = {}

    def __init__(self, elements, no_overlays=False, user_data=None):
        """
        TODO: Is it really 'sampleid'? Or what?

        elements: A list of (str sampleid, rankedlist, str imgpatch),
                 where sampleid is the ID for this data point.
                 rankedlist is a list of grouplabels, which should be
                 sorted by confidence (i.e. the most-likely grouplabel
                 should be at index 0).
                 imgpatch is a path to the image that this element
                 represents.
        user_data: Whatever you want it to be. For 'Digit' attributes,
                   this will be a dict that maps:
                     {str patchpath: float score}
                   This will be used during 'Split', for smarter split
                   behavior. TODO: UNUSED.
        """
        # Converting to Tuples didn't seem to help - if anything, it hurt?
        # self.elements = tuple(elements) if type(elements) != tuple else elements
        self.elements = elements
        # for i in range(len(elements)):  # Why did I do this again?
        #    if not issubclass(type(elements[i][1]), list):
        #        self.elements[i] = list((elements[i][0], list(elements[i][1]), elements[i][2]))
        self.no_overlays = no_overlays
        # self.is_misclassify: Used to mark a GroupClass that the user
        # said was 'Misclassified'
        self.is_misclassify = False
        # orderedAttrVals is a list of grouplabels, whose order is
        # predetermined by some score-metric. Should not change after it
        # is first set.
        self.orderedAttrVals = ()

        # The index of the grouplabel (w.r.t self.orderedAttrVals) that
        # this group ostensibly represents. Is 'finalized' when the user
        # clicks 'Ok' within the VerifyOverlay UI.
        self.index = 0

        # self.user_data can be several things. For "digits" attributes,
        # it's a dict mapping {str patchpath: float score}
        self.user_data = user_data

        self.processElements()

        s = str(self.getcurrentgrouplabel())
        if s not in GroupClass.ctrs:
            GroupClass.ctrs[s] = 1
        else:
            GroupClass.ctrs[s] += 1

        # is_manual: A flag used by MODE_YESNO2, indicates this group
        # should be labeled manually.
        self.is_manual = False

    @property
    def label(self):
        s = str(self.getcurrentgrouplabel())
        if s not in GroupClass.ctrs:
            GroupClass.ctrs[s] = 1
        return '{0}-{1}'.format(s,
                                GroupClass.ctrs[s])

    @staticmethod
    def merge(*groups):
        """ Given a bunch of GroupClass objects G, return a new GroupClass
        object that 'merges' all of the elements in G.
        """
        new_elements = []  # a list, ((sampleid_i, rlist_i, patchpath_i), ...)
        # TODO: Merge user_data's, though, it's not being used at the moment.
        label = None
        g_type = None
        for group in groups:
            if g_type is None:
                g_type = type(group)
            elif type(group) != g_type:
                error("Can't merge groups with different types.")
                pdb.set_trace()
                return None

            if label is None:
                label = group.label
            elif group.label != label:
                error("Can't merge groups with different labels.")
                pdb.set_trace()
                return None
            new_elements.extend(group.elements)
        if type(g_type) == GroupClass:
            return GroupClass(new_elements)
        else:
            return DigitGroupClass(new_elements)

    def get_overlays(self):
        """ Returns overlayMin, overlayMax """
        if self.no_overlays:
            return None, None
        with util.time("Generating min/max overlays"):
            return do_generate_overlays(self)

    def __eq__(self, o):
        return (o and issubclass(type(o), GroupClass) and
                self.elements == o.elements)

    def __str__(self):
        return "GroupClass({0} elems)".format(len(self.elements))

    def __repr__(self):
        return "GroupClass({0} elems)".format(len(self.elements))

    def getcurrentgrouplabel(self):
        return self.orderedAttrVals[self.index]

    def processElements(self):
        """
        Go through the elements, and compile an ordered list of
        gropulabels for self.orderedAttrVals.
        """
        # weightedAttrVals is a dict mapping {[attrval, flipped]: float weight}
        weightedAttrVals = {}
        # self.elements is a list of the form [(imgpath_i, rankedlist_i, patchpath_i), ...]
        # where each rankedlist_i is tuples of the form: (attrval_i, flipped_i,
        # imageorder_i)
        for element in self.elements:
            # element := (imgpath, rankedlist, patchpath)
            """
            Ordered templates
            """
            vote = 1.0
            rankedlist = element[1]
            for group in rankedlist:
                if (group not in weightedAttrVals):
                    weightedAttrVals[group] = vote
                else:
                    weightedAttrVals[group] = weightedAttrVals[group] + vote

                vote = vote / 2.0
        self.orderedAttrVals = tuple([group
                                      for (group, weight)
                                      in sorted(weightedAttrVals.items(),
                                                key=lambda t: t[1],
                                                reverse=True)])

    def split_kmeans(self, K=2):
        """ Uses k-means (k=2) to try to split this group. """
        if len(self.elements) == 2:
            if type(self) == GroupClass:
                return (GroupClass((self.elements[0],),
                                   user_data=self.user_data),
                        GroupClass((self.elements[1],),
                                   user_data=self.user_data))
            elif type(self) == DigitGroupClass:
                return (DigitGroupClass((self.elements[0],),
                                        user_data=self.user_data),
                        DigitGroupClass((self.elements[1],),
                                        user_data=self.user_data))
            else:
                error("Inexplicable error")
                pdb.set_trace()

        # 1.) Gather images
        patchpaths = []
        # patchpath_map used to re-construct 'elements' later on.
        patchpath_map = {}  # maps {patchpath: (sampleid, rlist)}
        for (sampleid, rlist, patchpath) in self.elements:
            patchpaths.append(patchpath)
            patchpath_map[patchpath] = (sampleid, rlist)
        # 2.) Call kmeans clustering
        _t = time.time()
        with util.time_operation("running k-means."):
            clusters = cluster_imgs.cluster_imgs_kmeans(patchpaths,
                                                        k=K,
                                                        do_downsize=True,
                                                        do_align=True)
        # 3.) Create GroupClasses
        groups = []
        for clusterid, patchpaths in clusters.iteritems():
            debug("For clusterid {0}, there are {1} elements.",
                  clusterid,
                  len(patchpaths))
            elements = []
            for patchpath in patchpaths:
                elements.append(patchpath_map[patchpath] + (patchpath,))
            if type(self) == GroupClass:
                groups.append(GroupClass(elements,
                                         user_data=self.user_data))
            elif type(self) == DigitGroupClass:
                groups.append(DigitGroupClass(elements,
                                              user_data=self.user_data))

        assert len(groups) == K
        return groups

    def split_pca_kmeans(self, K=2, N=3):
        """ Use PCA to help with the split process.
        Input: Set of image patches A, of size NxM
        0.) Discretize the image patch into K N'xM' equally-sized slices.
        1.) Using the discretized image patches A', run PCA to extract
            the slices S that maximize the variance
        2.) Run k-means (k=2) on the slices S.
        """
        if len(self.elements) <= K:
            groups = []
            for element in self.elements:
                if type(self) == GroupClass:
                    groups.append(GroupClass((element,),
                                             user_data=self.user_data))
                elif type(self) == DigitGroupClass:
                    groups.append(DigitGroupClass((element,),
                                                  user_data=self.user_data))
                else:
                    error("Unknown element type: {0}", type(self))
                    pdb.set_trace()

            return groups
        # 1.) Gather images
        patchpaths = []
        # patchpath_map used to re-construct 'elements' later on.
        patchpath_map = {}  # maps {patchpath: (sampleid, rlist)}
        for (sampleid, rlist, patchpath) in self.elements:
            patchpaths.append(patchpath)
            patchpath_map[patchpath] = (sampleid, rlist)
        # 2.) Call kmeans clustering
        with util.time_operation("running PCA+k-means."):
            clusters = cluster_imgs.cluster_imgs_pca_kmeans(
                patchpaths, k=K, do_align=True)
        # 3.) Create GroupClasses
        groups = []
        for clusterid, patchpaths in clusters.iteritems():
            debug("For clusterid {0}, there are {1} elements.",
                  clusterid,
                  len(patchpaths))
            elements = []
            for patchpath in patchpaths:
                elements.append(patchpath_map[patchpath] + (patchpath,))
            if type(self) == GroupClass:
                groups.append(GroupClass(elements,
                                         user_data=self.user_data))
            elif type(self) == DigitGroupClass:
                groups.append(DigitGroupClass(elements,
                                              user_data=self.user_data))
            else:
                error("Unknown type: {0}", type(self))
                pdb.set_trace()

        assert len(groups) == K
        return groups

    def split_rankedlist(self):
        """ Perform a split by using the rankedlist outputted by
        Kai's grouping algorithm.
        """
        groups = []
        new_elements = {}
        all_rankedlists = [t[1] for t in self.elements]

        n = num_common_prefix(*all_rankedlists)

        def naive_split(elements):
            mid = int(round(len(elements) / 2.0))
            group1 = elements[:mid]
            group2 = elements[mid:]
            if type(self) == GroupClass:
                groups.append(GroupClass(group1, user_data=self.user_data))
                groups.append(GroupClass(group2, user_data=self.user_data))
            elif type(self) == DigitGroupClass:
                groups.append(DigitGroupClass(
                    group1, user_data=self.user_data))
                groups.append(DigitGroupClass(
                    group2, user_data=self.user_data))
            else:
                error("Unknown group class: {0}", type(self))
                pdb.set_trace()
            return groups

        if n == len(all_rankedlists[0]):
            debug('rankedlists were same for all voted ballots -- '
                  'doing a naive split instead.')
            return naive_split(self.elements)

        if n == 0:
            warn('group class splitting shouldn\'t result in a count '
                 'of zero; setting it to one.')
            n = 1

        # group by index 'n' into each ballots attrslist (i.e. ranked list)
        for (samplepath, rankedlist, patchpath) in self.elements:
            if len(rankedlist) <= 1:
                debug("Can't split anymore.")
                return [self]
            new_group = rankedlist[n]
            new_elements.setdefault(new_group, []).append(
                (samplepath, rankedlist, patchpath))

        if len(new_elements) == 1:
            # no new groups were made -- just do a naive split
            debug('After a "smart" split, no new groups were made. So, '
                  'just doing a naive split.')
            return naive_split(self.elements)

        debug('number of new groups after split: {0}', len(new_elements))
        for grouplabel, elements in new_elements.iteritems():
            if type(self) == GroupClass:
                groups.append(GroupClass(elements, user_data=self.user_data))
            elif type(self) == DigitGroupClass:
                groups.append(DigitGroupClass(
                    elements, user_data=self.user_data))
            else:
                error('Unsupported group class: {0}', type(self))
                pdb.set_trace()
        return groups

    def split_kmeans2(self, K=2):
        """ Performs our hand-rolled K-Means implementation. """
        if len(self.elements) == 2:
            if type(self) == GroupClass:
                return (GroupClass((self.elements[0],),
                                   user_data=self.user_data),
                        GroupClass((self.elements[1],),
                                   user_data=self.user_data))
            elif type(self) == DigitGroupClass:
                return (DigitGroupClass((self.elements[0],),
                                        user_data=self.user_data),
                        DigitGroupClass((self.elements[1],),
                                        user_data=self.user_data))
            else:
                print "Wat?"
                pdb.set_trace()

        # 1.) Gather images
        patchpaths = []
        # patchpath_map used to re-construct 'elements' later on.
        patchpath_map = {}  # maps {patchpath: (sampleid, rlist)}
        for (sampleid, rlist, patchpath) in self.elements:
            patchpaths.append(patchpath)
            patchpath_map[patchpath] = (sampleid, rlist)
        # 2.) Call kmeans clustering
        _t = time.time()
        print "...running k-means2"
        clusters = cluster_imgs.kmeans_2D(patchpaths, k=K,
                                          distfn_method='vardiff',
                                          do_align=True)
        print "...Completed running k-means2 ({0} s).".format(time.time() - _t)
        # 3.) Create GroupClasses
        groups = []
        found_patchpaths = set()
        for clusterid, c_patchpaths in clusters.iteritems():
            print "For clusterid {0}, there are {1} elements.".format(clusterid, len(c_patchpaths))
            elements = []
            for c_patchpath in c_patchpaths:
                if c_patchpath in found_patchpaths:
                    print "Uhoh, element {0} was present in multiple clusters.".format(c_patchpath)
                    pdb.set_trace()
                found_patchpaths.add(c_patchpath)
                elements.append(patchpath_map[c_patchpath] + (c_patchpath,))
            if type(self) == GroupClass:
                groups.append(GroupClass(elements,
                                         user_data=self.user_data))
            elif type(self) == DigitGroupClass:
                groups.append(DigitGroupClass(elements,
                                              user_data=self.user_data))
        if len(found_patchpaths) != len(patchpaths):
            print "Uhoh, only found {0} patchpaths, but should have found {1}".format(len(found_patchpaths),
                                                                                      len(patchpaths))
            pdb.set_trace()
        assert len(groups) == K
        return groups

    def split_kmediods(self, K=2):
        """ Performs our hand-rolled K-Mediods implementation. """
        if len(self.elements) == 2:
            if type(self) == GroupClass:
                return (GroupClass((self.elements[0],),
                                   user_data=self.user_data),
                        GroupClass((self.elements[1],),
                                   user_data=self.user_data))
            elif type(self) == DigitGroupClass:
                return (DigitGroupClass((self.elements[0],),
                                        user_data=self.user_data),
                        DigitGroupClass((self.elements[1],),
                                        user_data=self.user_data))
            else:
                print "Wat?"
                pdb.set_trace()

        # 1.) Gather images
        patchpaths = []
        # patchpath_map used to re-construct 'elements' later on.
        patchpath_map = {}  # maps {patchpath: (sampleid, rlist)}
        for (sampleid, rlist, patchpath) in self.elements:
            patchpaths.append(patchpath)
            patchpath_map[patchpath] = (sampleid, rlist)
        # 2.) Call kmeans clustering
        _t = time.time()
        print "...running k-mediods."
        clusters = cluster_imgs.kmediods_2D(patchpaths, k=K,
                                            distfn_method='vardiff',
                                            do_align=True)
        print "...Completed running k-mediods ({0} s).".format(time.time() - _t)
        # 3.) Create GroupClasses
        groups = []
        for clusterid, patchpaths in clusters.iteritems():
            print "For clusterid {0}, there are {1} elements.".format(clusterid, len(patchpaths))
            elements = []
            for patchpath in patchpaths:
                elements.append(patchpath_map[patchpath] + (patchpath,))
            if type(self) == GroupClass:
                groups.append(GroupClass(elements,
                                         user_data=self.user_data))
            elif type(self) == DigitGroupClass:
                groups.append(DigitGroupClass(elements,
                                              user_data=self.user_data))

        assert len(groups) == K
        return groups

    def split(self, mode='kmeans'):
        if mode == 'rankedlist':
            return self.split_rankedlist()
        elif mode == 'kmeans':
            return self.split_kmeans(K=2)
        elif mode == 'pca_kmeans':
            return self.split_pca_kmeans(K=3)
        elif mode == 'kmeans2':
            return self.split_kmeans2(K=2)
        elif mode == 'kmediods':
            return self.split_kmediods(K=2)
        else:
            print "Unrecognized mode: {0}. Defaulting to kmeans.".format(mode)
            return self.split_kmeans(K=2)


class DigitGroupClass(GroupClass):
    """
    A class that represents a potential group of digits.
    """

    def __init__(self, elements, no_overlays=False, user_data=None,
                 *args, **kwargs):
        GroupClass.__init__(self, elements, no_overlays=no_overlays,
                            user_data=user_data)

    def split_kmeans(self, K=2):
        """ Uses k-means (k=2) to try to split this group. """
        if len(self.elements) == 2:
            return (DigitGroupClass((self.elements[0],),
                                    user_data=self.user_data),
                    DigitGroupClass((self.elements[1],),
                                    user_data=self.user_data))
        # 1.) Gather images
        patchpaths = []
        # patchpath_map used to re-construct 'elements' later on.
        patchpath_map = {}  # maps {patchpath: (sampleid, rlist)}
        for (sampleid, rlist, patchpath) in self.elements:
            patchpaths.append(patchpath)
            patchpath_map[patchpath] = (sampleid, rlist)
        # 2.) Call kmeans clustering
        _t = time.time()
        print "...running k-means."
        clusters = cluster_imgs.cluster_imgs_kmeans(
            patchpaths, k=K, do_align=True)
        print "...Completed running k-means ({0} s).".format(time.time() - _t)
        # 3.) Create DigitGroupClasses
        groups = []
        for clusterid, patchpaths in clusters.iteritems():
            print "For clusterid {0}, there are {1} elements.".format(clusterid, len(patchpaths))
            elements = []
            for patchpath in patchpaths:
                elements.append(patchpath_map[patchpath] + (patchpath,))
            groups.append(DigitGroupClass(elements,
                                          user_data=self.user_data))
        assert len(groups) == K
        return groups


def do_generate_overlays(group):
    """ Given a GroupClass, generate the Min/Max overlays. """
    if len(group.elements) <= 20:
        # Just do it all in serial.
        return _generate_overlays(group.elements)
    else:
        return partask.do_partask(_generate_overlays,
                                  group.elements,
                                  combfn=_my_combfn_overlays,
                                  init=(None, None))


def _generate_overlays(elements):
    overlayMin, overlayMax = None, None
    for element in elements:
        path = element[2]
        img = misc.imread(path, flatten=1)
        if (overlayMin is None):
            overlayMin = img
        else:
            if overlayMin.shape != img.shape:
                h, w = overlayMin.shape
                img = resize_img_norescale(img, (w, h))
            overlayMin = np.fmin(overlayMin, img)
        if (overlayMax is None):
            overlayMax = img
        else:
            if overlayMax.shape != img.shape:
                h, w = overlayMax.shape
                img = resize_img_norescale(img, (w, h))
            overlayMax = np.fmax(overlayMax, img)

    rszFac = sh.resizeOrNot(overlayMax.shape, sh.MAX_PRECINCT_PATCH_DISPLAY)
    overlayMax = sh.fastResize(overlayMax, rszFac)  # / 255.0
    overlayMin = sh.fastResize(overlayMin, rszFac)  # / 255.0
    return overlayMin, overlayMax


def _my_combfn_overlays(result, subresult):
    """ result, subresult are (np img_min, np img_max). Overlay the
    min's and max's together.
    """
    imgmin, imgmax = result
    imgmin_sub, imgmax_sub = subresult
    if imgmin is None:
        imgmin = imgmin_sub
    else:
        if imgmin.shape != imgmin_sub.shape:
            h, w = imgmin.shape
            imgmin_sub = resize_img_norescale(imgmin_sub, (w, h))
        imgmin = np.fmin(imgmin, imgmin_sub)
    if imgmax is None:
        imgmax = imgmax_sub
    else:
        if imgmax.shape != imgmax_sub.shape:
            h, w = imgmax.shape
            imgmax_sub = resize_img_norescale(imgmax_sub, (w, h))
        imgmax = np.fmax(imgmax, imgmax_sub)
    return imgmin, imgmax


class TextInputDialog(wx.Dialog):
    """
    A dialog to accept N user inputs.
    """

    def __init__(self, parent, caption="Please enter your input(s).",
                 labels=('Input 1:',),
                 vals=('',),
                 *args, **kwargs):
        """
        labels: A list of strings. The number of strings determines the
              number of inputs desired.
        vals: An optional list of values to pre-populate the inputs.
        """
        wx.Dialog.__init__(
            self, parent, title='Input required', *args, **kwargs)
        self.parent = parent

        # self.results maps {str label: str value}
        self.results = {}

        self.input_pairs = []
        for idx, label in enumerate(labels):
            txt = wx.StaticText(self, label=label)
            input_ctrl = wx.TextCtrl(self, style=wx.TE_PROCESS_ENTER)
            if idx == len(labels) - 1:
                input_ctrl.Bind(wx.EVT_TEXT_ENTER, self.onButton_ok)
            try:
                input_ctrl.SetValue(vals[idx])
            except:
                pass
            self.input_pairs.append((txt, input_ctrl))
        panel_btn = wx.Panel(self)
        btn_ok = wx.Button(panel_btn, id=wx.ID_OK)
        btn_ok.Bind(wx.EVT_BUTTON, self.onButton_ok)
        btn_cancel = wx.Button(panel_btn, id=wx.ID_CANCEL)
        btn_cancel.Bind(wx.EVT_BUTTON, self.onButton_cancel)
        panel_btn.sizer = wx.BoxSizer(wx.HORIZONTAL)
        panel_btn.sizer.Add(btn_ok, border=10, flag=wx.RIGHT)
        panel_btn.sizer.Add(btn_cancel, border=10, flag=wx.LEFT)
        panel_btn.SetSizer(panel_btn.sizer)

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        caption_txt = wx.StaticText(self, label=caption)
        self.sizer.Add(caption_txt, border=10, flag=wx.ALL)
        gridsizer = wx.GridSizer(rows=0, cols=2, hgap=5, vgap=3)
        gridsizer.Add(self.input_pairs[0][0])
        gridsizer.Add(self.input_pairs[0][1])
        for txt, input_ctrl in self.input_pairs[1:]:
            gridsizer.Add(txt, border=10, flag=wx.ALL)
            gridsizer.Add(input_ctrl, border=10, flag=wx.ALL)
        self.gridsizer = gridsizer
        self.sizer.Add(gridsizer)
        self.sizer.Add(panel_btn, border=10, flag=wx.ALL | wx.ALIGN_CENTER)
        self.SetSizer(self.sizer)

        self.Fit()

        self.input_pairs[0][1].SetFocus()

    def onButton_ok(self, evt):
        for txt, input_ctrl in self.input_pairs:
            self.results[txt.GetLabel()] = input_ctrl.GetValue()
        self.EndModal(wx.ID_OK)

    def onButton_cancel(self, evt):
        self.EndModal(wx.ID_CANCEL)


class SingleChoiceDialog(wx.Dialog):
    """
    A Dialog to allow the user to select one of N choices.
    """

    def __init__(self, parent, message="Please make a choice.", choices=[], *args, **kwargs):
        wx.Dialog.__init__(self, parent, *args, **kwargs)

        # self.result will be the user-selected choice
        self.result = None

        sizer = wx.BoxSizer(wx.VERTICAL)
        txt1 = wx.StaticText(self, label=message)
        sizer.Add(txt1)
        radio_btns = []  # List of [(str choice_i, obj RadioButton_i), ...]
        self.radio_btns = radio_btns
        for i, choice in enumerate(choices):
            if i == 0:
                radiobtn = wx.RadioButton(
                    self, label=choice, style=wx.RB_GROUP)
            else:
                radiobtn = wx.RadioButton(self, label=choice)
            radio_btns.append((choice, radiobtn))
            sizer.Add(radiobtn)
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        btn_ok = wx.Button(self, label="Ok")
        btn_ok.Bind(wx.EVT_BUTTON, self.onButton_ok)
        btn_cancel = wx.Button(self, label="Cancel")
        btn_cancel.Bind(wx.EVT_BUTTON, self.onButton_cancel)
        btn_sizer.AddMany([(btn_ok,), (btn_cancel,)])
        sizer.Add(btn_sizer, flag=wx.ALIGN_CENTER)
        self.SetSizer(sizer)
        self.Fit()

    def onButton_ok(self, evt):
        for choice, radiobtn in self.radio_btns:
            if radiobtn.GetValue() == True:
                self.result = choice
        self.EndModal(wx.ID_OK)

    def onButton_cancel(self, evt):
        self.EndModal(wx.ID_CANCEL)
