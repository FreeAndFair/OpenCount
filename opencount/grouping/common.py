import sys, os, csv, pickle, pdb
import numpy as np
from os.path import join as pathjoin
from scipy import misc
sys.path.append('../')
sys.path.append('../pixel_reg/')
import pixel_reg.shared as sh
import pixel_reg.part_match as part_match

import wx, pickle
from specify_voting_targets.imageviewer import WorldState as WorldState
from specify_voting_targets.imageviewer import BoundingBox as BoundingBox
from util import encodepath
import specify_voting_targets.util_gui as util_gui

DUMMY_ROW_ID = -42 # Also defined in label_attributes.py

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

    def has_attrtype(self, attrtype):
        return attrtype in self.attrs

    def get_attrval(self, attrtype):
        return self.attrs.get(attrtype, None)

    def get_attrtypes(self):
        return tuple(self.attrs.keys())

    def add_attrtypes(self, attrtypes, attrvals=None):
        if not attrvals:
            attrvals = (None,)*len(attrtypes)
        assert len(attrtypes) == len(attrvals)
        for i, attrtype in enumerate(attrtypes):
            self.set_attrtype(attrtype, attrvals[i])

    def add_attrtype(self, attrtype, attrval=None):
        self.attrs[attrtype] = attrval

    def set_attrtype(self, attrtype, attrval=None):
        self.attrs[attrtype] = attrval

    def remove_attrtype(self, attrtype):
        assert attrtype in self.attrs, "Error - {0} was not found in \
self.attrs: {1}".format(attrtype, self.attrs)
        self.attrs.pop(attrtype)

    def relabel_attrtype(self, oldname, newname):
        assert oldname in self.attrs, "Error - {0} was not found in \
self.attrs: {1}".format(oldname, self.attrs)
        oldval = self.get_attrval(oldname)
        self.remove_attrtype(oldname)
        self.set_attrtype(newname, oldval)

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
        box = AttributeBox(0,0,0,0)
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

class IWorldState(WorldState):
    def __init__(self, box_locations=None):
        WorldState.__init__(self, box_locations=box_locations)

    def get_attrpage(self, attrtype):
        return self.get_attrbox(attrtype).side

    def get_attrtypes(self):
        """
        Return a list of all attribute types.
        """
        result = set()
        for b in self.get_attrboxes():
            for attrtype in b.get_attrtypes():
                result.add(attrtype)
        return list(result)

    def get_attrbox(self, attrtype):
        """
        Return the AttributeBox with given attrtype.
        """
        for b in self.get_boxes_all_list():
            if b.has_attrtype(attrtype):
                return b
        print "== Error: In IWorldState.get_attrbox, no AttributeBox \
with type {0} was found."
        return None

    def get_attrboxes(self):
        """
        Return all AttributeBoxes in a flat list.
        """
        return self.get_boxes_all_list()

    def remove_attrtype(self, attrtype):
        """
        Removes the attrtype from all instances of AttributeBoxes.
        """
        for temppath, boxes in self.box_locations.iteritems():
            for b in boxes:
                if attrtype in b.get_attrtypes():
                    b.remove_attrtype(attrtype)
        self._remove_empties()

    def _remove_empties(self):
        newboxlocs = {}
        for temppath, boxes in self.box_locations.iteritems():
            newboxlocs[temppath] = [b for b in boxes if b.get_attrtypes()]
        self.box_locations = newboxlocs

    def remove_attrbox(self, box):
        for temppath in self.box_locations:
            if box in self.get_boxes(temppath):
                self.remove_box(temppath, box)

    def mutate(self, iworld):
        WorldState.mutate(self, iworld)

def dump_attrboxes(attrboxes, filepath):
    listdata = [b.marshall() for b in attrboxes]
    f = open(filepath, 'wb')
    pickle.dump(listdata, f)
    f.close()
def load_attrboxes(filepath):
    f = open(filepath, 'rb')
    listdata = pickle.load(f)
    f.close()
    return [AttributeBox.unmarshall(b) for b in listdata]

def marshall_iworldstate(world):
    """
    Marshall world.box_locations such that it's 
    possible to pickle them.
    """
    boxlocs = {}
    for temppath, boxes in world.box_locations.iteritems():
        for b in boxes:
            b_data = b.marshall()
            boxlocs.setdefault(temppath, []).append(b_data)
    return boxlocs

def unmarshall_iworldstate(data):
    """
    Unmarshall data, which is of the form:
        <boxlocations data>
    to return a new IWorldState.
    """
    iworld = IWorldState()
    new_boxlocations = {}
    boxlocsdata = data
    for temppath, boxesdata in boxlocsdata:
        for b_data in boxesdata:
            box = AttributeBox.unmarshall(b_data)
            new_boxlocations.setdefault(temppath, []).append(box)
    iworld.box_locations = new_boxlocations
    return iworld

def load_iworldstate(filepath):
    f = open(filepath, 'rb')
    data = pickle.load(filepath)
    return unmarshall_iworldstate(data)

def dump_iworldstate(iworld, filepath):
    f = open(filepath, 'wb')
    pickle.dump(marshall_iworldstate(iworld), f)
    f.close()

def resize_img_norescale(img, size):
    """ Resizes img to be a given size without rescaling - it only
    pads/crops if necessary.
    """
    w, h = size
    newimg = np.zeros((h,w), dtype='float32')
    h_img, w_img = img.shape
    if w_img > w:
        w_new = w
    else:
        w_new = w_img
    if h_img > h:
        h_new = h
    else:
        h_new = h_img
    newimg[0:h_new, 0:w_new] = img[0:h_new, 0:w_new]
    return newimg

def get_attrtypes(project):
    """
    Returns all attribute types in this election.
    """
    attrtypes = pickle.load(open(project.ballot_attributesfile, 'rb'))
    result = set()
    for attrdict in attrtypes:
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

def exists_digitattrs(proj):
    """ Returns True if there exists at least one digit-based attr. """
    attrs = pickle.load(open(proj.ballot_attributesfile, 'rb'))
    for attr in attrs:
        if attr['is_digitbased']:
            return True
    return False

def exists_customattrs(proj):
    """ Returns True if there exists at least one Custom attr. """
    customattrsP = pathjoin(proj.projdir_path, proj.custom_attrs)
    return os.path.exists(customattrsP)

def is_tabulationonly(project, attrtype):
    """ Returns True if the attrtype is for tabulationonly. """
    attrtypes_dicts = pickle.load(open(project.ballot_attributesfile, 'rb'))
    for attrdict in attrtypes_dicts:
        attrs_str = '_'.join(attrdict['attrs'])
        if attrs_str == attrtype:
            return attrdict['is_tabulationonly']
    # Means we can't find attrtype anywhere.
    assert False, "Can't find attrtype: {0}".format(attrtype)

def is_digitbased(project, attrtype):
    """ Returns True if the attrtype is digit-based. """
    attrtypes_dicts = pickle.load(open(project.ballot_attributesfile, 'rb'))
    for attrdict in attrtypes_dicts:
        attrs_str = '_'.join(attrdict['attrs'])
        if attrs_str == attrtype:
            return attrdict['is_digitbased']
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

def get_attrpair_grouplabel(project, grouplabel):
    """ Given a grouplabel, return both the attrtype of the grouplabel
    and the attrval.
    """
    ballot_attributes = pickle.load(open(project.ballot_attributesfile, 'rb'))
    for attrdict in ballot_attributes:
        attrtype_str = get_attrtype_str(attrdict['attrs'])
        if get_propval(grouplabel, attrtype_str):
            return attrtype_str, get_propval(grouplabel, attrtype_str)
    print "Uhoh, couldn't find an attrpair in this grouplabel:", str_grouplabel(grouplabel)
    pdb.set_trace()
    return None

def get_attr_side(project, attrtype):
    """ Returns which side of the ballot this attrtype was defined
    on.
    """
    ballot_attrs = pickle.load(open(project.ballot_attributesfile, 'rb'))
    for attrdict in ballot_attrs:
        attrstr = get_attrtype_str(attrdict['attrs'])
        if attrstr == attrtype:
            return attrdict['side']
    print "Uhoh, couldn't find attribute:", attrtype
    pdb.set_trace()
    return None

def get_attr_prop(project, attrtype, prop):
    """ Returns the property of the given attrtype. """
    ballot_attrs = pickle.load(open(project.ballot_attributesfile, 'rb'))
    for attrdict in ballot_attrs:
        attrstr = get_attrtype_str(attrdict['attrs'])
        if attrstr == attrtype:
            return attrdict[prop]
    print "Uhoh, couldn't find attribute:", attrtype
    pdb.set_trace()
    return None

def get_numdigits(project, attr):
    """Return the number of digits that this digit-based attribute
    has. If this attr is not a digits attribute (or num_digitsmap
    hasn't been created yet), this returns None.
    """
    if not os.path.exists(pathjoin(project.projdir_path,
                                   project.num_digitsmap)):
        return None
    numdigits_map = pickle.load(open(pathjoin(project.projdir_path,
                                              project.num_digitsmap),
                                     'rb'))
    if attr not in numdigits_map:
        print "Uhoh, {0} not in numdigits_map".format(attr)
        pdb.set_trace()
        return None
    return int(numdigits_map[attr])

def get_digitbased_attrs(project):
    allattrs = get_attrtypes(project)
    return [attr for attr in allattrs if is_digitbased(project, attr)]

def is_digit_grouplabel(grouplabel, project):
    """ Return True if this grouplabel is digit-based. """
    attrtypes = get_attrtypes(project)
    for attrtype in attrtypes:
        if get_propval(grouplabel, attrtype):
            if is_digitbased(project, attrtype):
                return True
    return False

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
    
def remove_common_pathpart(rootdir, path):
    """ Given two paths, a root and a path, return just the part of
    path that starts at root:
    >>> remove_common_pathpart('/media/data1/election', 'election/blanks1/bar.png')
    blanks1/bar.png
    """
    rootdir_abs = os.path.abspath(rootdir)
    path_abs = os.path.abspath(path)
    if not rootdir.endswith('/'):
        rootdir += '/'
    if not path_abs.startswith(rootdir_abs):
        print "Wait, wat? Perhaps invalid arguments to remove_common_pathpart"
        pdb.set_trace()
    return path_abs[:len(rootdir_abs)]

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

def importPatches(project):
    """
    Reads in all .csv files in precinct_locations/, and returns
    them as:
      {str temppath: [(y1,y2,x1,x2),grouplabel_i,side_i,is_digitbased_i,is_tabulationonly_i), ...]}
    Output:
      A dict that contains all attribute types for all blank ballots,
      of the form:
        {str temppath: [(y1,y2,x1,x2),grouplabel_i,side_i,is_digitbased_i,is_tabulationonly_i), ...]}
    """
    if not project or not project.patch_loc_dir:
        return
    def is_csvfile(p):
        return os.path.splitext(p)[1].lower() == '.csv'
    fields = ('imgpath', 'id', 'x', 'y', 'width', 'height',
              'attr_type', 'attr_val', 'side', 'is_digitbased', 'is_tabulationonly')
    boxes = {}
    for dirpath, dirnames, filenames in os.walk(project.patch_loc_dir):
        for csvfilepath in [f for f in filenames if is_csvfile(f)]:
            try:
                csvfile = open(os.path.join(dirpath, csvfilepath), 'rb')
                dictreader = csv.DictReader(csvfile)
                for row in dictreader:
                    imgpath = os.path.abspath(row['imgpath'])
                    id = int(row['id'])
                    if id == DUMMY_ROW_ID:
                        boxes.setdefault(imgpath, [])
                        continue
                    x1 = int(row['x'])
                    y1 = int(row['y'])
                    x2 = x1 + int(row['width'])
                    y2 = y1 + int(row['height'])
                    side = row['side']
                    is_digitbased = row['is_digitbased']
                    is_tabulationonly = row['is_tabulationonly']
                    if not(boxes.has_key(imgpath)):
                        boxes[imgpath]=[]
                    # Currently, we don't create an exemplar attrpatch
                    # for flipped/wrong-imgorder. For now, just fake it.
                    # QUESTION: Do we have to merge attrboxes if they're
                    # at the same location? (I.e. multi-attr boxes)
                    for flip in (0,1):
                        for imgorder in (0,1):
                            grouplabel = make_grouplabel((row['attr_type'],row['attr_val']),
                                                         ('flip',flip),
                                                         ('imageorder', imgorder))
                            boxes[imgpath].append(((y1, y2, x1, x2), 
                                                   grouplabel,
                                                   side, is_digitbased, is_tabulationonly))
            except IOError as e:
                print "Unable to open file: {0}".format(csvfilepath)
    return boxes

""" GroupLabel Data Type """

def make_grouplabel(*args):
    """ Given k-v tuples, returns a grouplabel.
    >>> make_grouplabel(('precinct', '380400'), ('side', 0))
    """
    return frozenset(args)

def get_propval(grouplabel, property):
    """ Returns the value of a property in a grouplabel, or None
    if the property isn't present.
    >>> grouplabel = make_grouplabel(('precinct', '380400'), ('side', 0))
    >>> get_propval(grouplabel, 'precinct')
    380400
    >>> get_propval(grouplabel, 'foo') == None
    True
    """
    t = tuple(grouplabel)
    for key,v in t:
        if key == property:
            return v
    return None

def grouplabel_keys(grouplabel):
    """ Returns the keys of a grouplabel. """
    return tuple([k for (k,v) in tuple(grouplabel)])

def str_grouplabel(grouplabel):
    """ Returns a string-representation of the grouplabel. """
    kv_pairs = tuple(grouplabel)
    out = ''
    for (k, v) in kv_pairs:
        out += '{0}->{1}, '.format(k, v)
    return out

def get_median_img(imgdir):
    """ Given a directory of images, choose the median image. """
    # TODO: By choosing the median exemplar image, this might
    #       improve accuracy (somewhat). Can't hurt.
    imgs = []

class GroupClass(object):
    """
    A class that represents a potential group of images.
    """
    # A dict mapping {str label: int count}
    ctrs = {}
    def __init__(self, elements, no_overlays=False, is_digit=False, user_data=None):
        """
        TODO: Is it really 'sampleid'? Or what?

        elements: A list of (str sampleid, rankedlist, str imgpatch),
                 where sampleid is the ID for this data point. 
                 rankedlist is a list of grouplabels, which should be
                 sorted by confidence (i.e. the most-likely grouplabel
                 should be at index 0).
                 imgpatch is a path to the image that this element
                 represents.
        bool is_digit: True if this is a digit, False o.w.
        user_data: Whatever you want it to be. For 'Digit' attributes,
                   this will be a dict that maps:
                     {str patchpath: float score}
                   This will be used during 'Split', for smarter split
                   behavior.
        """
        self.elements = list(elements)
        for i in range(len(elements)):
            if not issubclass(type(elements[i][1]), list):
                self.elements[i] = list((elements[i][0], list(elements[i][1]), elements[i][2]))
        self.no_overlays=no_overlays
        self.is_digit = is_digit
        # orderedAttrVals is a list of grouplabels, whose order is 
        # predetermined by some score-metric. Should not change after it
        # is first set.
        self.orderedAttrVals = []
        
        # The index of the grouplabel (w.r.t self.orderedAttrVals) that
        # this group ostensibly represents. Is 'finalized' when the user
        # clicks 'Ok' within the VerifyOverlay UI.
        self.index = 0

        # self.user_data can be several things. For "digits" attributes,
        # it's a dict mapping {str patchpath: float score}
        self.user_data = user_data

        self.processElements()

        # The label that will be displayed in the ListBoxes to 
        # the user, i.e. a public name for this GroupClass.
        try:
            self.label = str(self.getcurrentgrouplabel())
        except Exception as e:
            print e
            pdb.set_trace()
                     
        if self.label not in GroupClass.ctrs:
            GroupClass.ctrs[self.label] = 1
        else:
            GroupClass.ctrs[self.label] += 1
        self.label += '-{0}'.format(GroupClass.ctrs[self.label])

        # is_manual: A flag used by MODE_YESNO2, indicates this group
        # should be labeled manually.
        self.is_manual = False

    def get_overlays(self):
        """ Returns overlayMin, overlayMax """
        overlayMin, overlayMax = None, None
        for element in self.elements:
            path = element[2]
            if not self.no_overlays:
                try:
                    img = misc.imread(path, flatten=1)
                    if (overlayMin == None):
                        overlayMin = img
                    else:
                        if overlayMin.shape != img.shape:
                            h, w = overlayMin.shape
                            img = resize_img_norescale(img, (w,h))
                        overlayMin = np.fmin(overlayMin, img)
                    if (overlayMax == None):
                        overlayMax = img
                    else:
                        if overlayMax.shape != img.shape:
                            h, w = overlayMax.shape
                            img = resize_img_norescale(img, (w,h))
                        overlayMax = np.fmax(overlayMax, img)
                except Exception as e:
                    print e
                    print "Cannot open patch @ {0}".format(path)
                    pdb.set_trace()

        rszFac=sh.resizeOrNot(overlayMax.shape,sh.MAX_PRECINCT_PATCH_DISPLAY)
        overlayMax = sh.fastResize(overlayMax, rszFac) / 255.0
        overlayMin = sh.fastResize(overlayMin, rszFac) / 255.0

        return overlayMin, overlayMax

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
        def sanitycheck_rankedlists(elements):
            """Make sure that the first grouplabel for each rankedlist
            are all the same grouplabel.
            """
            grouplabel = None
            for (elementid, rankedlist, patchpath) in elements:
                if grouplabel == None:
                    if rankedlist:
                        grouplabel = rankedlist[0]
                        continue
                    else:
                        print 'wat, no rankedlist?!'
                        pdb.set_trace()
                elif rankedlist[0] != grouplabel:
                    print "Error, first element of all rankedlists are \
not equal."
                    pdb.set_trace()
            return True
        sanitycheck_rankedlists(self.elements)
        # weightedAttrVals is a dict mapping {[attrval, flipped]: float weight}
        weightedAttrVals = {}
        # self.elements is a list of the form [(imgpath_i, rankedlist_i, patchpath_i), ...]
        # where each rankedlist_i is tuples of the form: (attrval_i, flipped_i, imageorder_i)
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
        self.orderedAttrVals = [group
                                for (group, weight) in sorted(weightedAttrVals.items(), 
                                                                   key=lambda t: t[1],
                                                                   reverse=True)]
        
    def split(self):
        if self.is_digit:
            # Assumes that only Digit attributes is using self.user_data.
            # Split the elements based on the partmatch scores: the top
            # 50%, and the bottom 50%.
            # self.user_data: {str patchpath: float score}
            # 0.) Check degenerate case
            if len(self.elements) == 2:
                return (GroupClass((self.elements[0],), is_digit=True,
                                  user_data=self.user_data),
                        GroupClass((self.elements[1],), is_digit=True,
                                   user_data=self.user_data))
            # 1.) Compute median score
            scores = []
            for (sampleid, rlist, patchpath) in self.elements:
                if patchpath not in self.user_data:
                    print "Uhoh, patchpath not in self.user_data."
                    pdb.set_trace()
                score = self.user_data[patchpath]
                scores.append(score)
            scores = sorted(scores)
            median = scores[int(len(scores) / 2)]
            print "MEDIAN WAS:", median
            # 2.) Group high and low scores
            elements1, elements2 = [], []
            for (sampleid, rlist, patchpath) in self.elements:
                score = self.user_data[patchpath]
                if score > median:
                    elements1.append((sampleid, rlist, patchpath))
                else:
                    elements2.append((sampleid, rlist, patchpath))
            return (GroupClass(elements1, is_digit=True, user_data=self.user_data), GroupClass(elements2, is_digit=True, user_data=self.user_data))
            
        groups = []
        new_elements = {}
        all_rankedlists = [t[1] for t in self.elements]

        n = num_common_prefix(*all_rankedlists)

        def naive_split(elements):
            mid = int(round(len(elements) / 2.0))
            group1 = elements[:mid]
            group2 = elements[mid:]
            groups.append(GroupClass(group1, user_data=self.user_data))
            groups.append(GroupClass(group2, user_data=self.user_data))
            return groups
            
        if n == len(all_rankedlists[0]):
            print "rankedlists were same for all voted ballots -- \
doing a naive split instead."
            return naive_split(self.elements)

        if n == 0:
            print "== Wait, n shouldn't be 0 here (in GroupClass.split). \
Changing to n=1, since that makes some sense."
            print "Enter in 'c' for 'continue' to continue execution."
            pdb.set_trace()
            n = 1

        # group by index 'n' into each ballots attrslist (i.e. ranked list)
        for (samplepath, rankedlist, patchpath) in self.elements:
            if len(rankedlist) <= 1:
                print "==== Can't split anymore."
                return [self]
            new_group = rankedlist[n]
            new_elements.setdefault(new_group, []).append((samplepath, rankedlist, patchpath))

        if len(new_elements) == 1:
            # no new groups were made -- just do a naive split
            print "After a 'smart' split, no new groups were made. So, \
just doing a naive split."
            return naive_split(self.elements)

        print 'number of new groups after split:', len(new_elements)
        for grouplabel, elements in new_elements.iteritems():
            groups.append(GroupClass(elements, user_data=self.user_data))
        return groups

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
        wx.Dialog.__init__(self, parent, title='Input required', *args, **kwargs)
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
        radio_btns = [] # List of [(str choice_i, obj RadioButton_i), ...]
        self.radio_btns = radio_btns
        for i, choice in enumerate(choices):
            if i == 0:
                radiobtn = wx.RadioButton(self, label=choice, style=wx.RB_GROUP)
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

def do_digitocr(imgpaths, digit_exs, num_digits, bb=None,
                rejected_hashes=None, accepted_hashes=None):
    """ Basically does what sh.digitParse does, but checks to see if
    the image might be flipped, and if it is, to flip it and return
    the match with the best response.
    Input:
        list imgpaths: list of image paths to perform digit ocr over
        dict digit_exs: maps {str digit: obj img}
        tuple bb: If given, this is a tuple (y1,y2,x1,x2), which 
                  restricts the ocr search to the given bb.
        dict rejected_hashes: maps {imgpath: {str digit: [((y1,y2,x1,x2), side_i), ...]}}
        dict accepted_hashes: maps {imgpath: {str digit: [((y1,y2,x1,x2), side_i), ...]}}
    Output:
        list of [(imgpath_i, ocrstr_i, meta_i, isflip_i), ...]
    """
    def get_best_flip(results_noflip, results_flip):
        """ Given the results of digitParse (both not flipped and
        flipped), return a new results list that, for each voted ballot,
        chooses the match (either flipped/not-flipped) with the highest
        score.
        Input:
            lst results_noflip: list of [(imgpath_i, ocrstr_i, meta_i), ...]
            lst results_flip: list of [(imgpath_i, ocrstr_i, meta_i), ...]
        Output:
            lst results: list of [(imgpath_i, ocrstr_i, meta_i, isflip_i), ...]
        """
        assert len(results_noflip) == len(results_flip)
        results_noflip = sorted(results_noflip, key=lambda tup: tup[0])
        results_flip = sorted(results_flip, key=lambda tup: tup[0])
        results = []
        for idx, (path_noflip, ocrstr_noflip, meta_noflip) in enumerate(results_noflip):
            path_flip, ocrstr_flip, meta_flip = results_flip[idx]
            assert path_noflip == path_flip
            assert len(meta_noflip) == len(meta_flip)
            avg_score_noflip = sum([tup[6] for tup in meta_noflip]) / float(len(meta_noflip))
            avg_score_flip = sum([tup[6] for tup in meta_flip]) / float(len(meta_flip))
            if avg_score_noflip > avg_score_flip:
                results.append((path_noflip, ocrstr_noflip, meta_noflip, 0))
            else:
                results.append((path_flip, ocrstr_flip, meta_flip, 1))
        assert len(results) == len(results_noflip)
        assert len(results) == len(results_flip)
        return results
    
    if not bb:
        imgsize = misc.imread(imgpath, flatten=True).shape
        bb = (0, imgsize[0], 0, imgsize[1])
    results_noflip = part_match.digitParse(digit_exs, imgpaths, bb,
                                           num_digits, do_flip=False,
                                           rejected_hashes=rejected_hashes,
                                           accepted_hashes=accepted_hashes)
    results_flip = part_match.digitParse(digit_exs, imgpaths, bb,
                                         num_digits, do_flip=True,
                                         rejected_hashes=rejected_hashes,
                                         accepted_hashes=accepted_hashes)
    results_noflip = munge_pm_results(results_noflip)
    results_flip = munge_pm_results(results_flip)
    results_best = get_best_flip(results_noflip, results_flip)
    return results_best

def munge_pm_results(results):
    """ Converts results format from part_match.digitParse to the output
    format of shared.digitParse.
    Input:
        lst results: List of [(imgpath_i, ocrstr_i, imgpatches_i, patchcoords_i, scores_i), ...]
    Output:
        lst of [(imgpath_i, ocrstr_i, meta_i), ...], where each meta_i
        is a list of the form:
            meta_i := (y1,y2,x1,x2,str digit, obj digitimg, float score)
    """
    out = []
    for (imgpath, ocrstr, imgpatches, patchcoords, scores) in results:
        meta = []
        if not (len(imgpatches) == len(patchcoords) == len(scores)):
            print "Uhoh, not all were equal length."
            pdb.set_trace()
        assert len(imgpatches) == len(patchcoords) == len(scores)
        for i, (y1,y2,x1,x2) in enumerate(patchcoords):
            try:
                digit = ocrstr[i]
            except Exception as e:
                print e
                print "OH DEAR. Did you accidently not type in the digit \
during LabelDigitAttrs?"
                pdb.set_trace()
            digitimg = imgpatches[i]
            score = scores[i]
            meta.append((y1,y2,x1,x2,digit,digitimg,score))
        out.append((imgpath, ocrstr, meta))
    return out

if __name__ == '__main__':
    class MyFrame(wx.Frame):
        def __init__(self, parent, *args, **kwargs):
            wx.Frame.__init__(self, parent, *args, **kwargs)
            btn = wx.Button(self, label="Click me")
            btn.Bind(wx.EVT_BUTTON, self.onButton)
            
        def onButton(self, evt):
            dlg = TextInputDialog(self, labels=("Input 1:", "Input 2:", "Input 3:"))
            dlg.ShowModal()
            print dlg.results
    app = wx.App(False)
    frame = MyFrame(None)
    frame.Show()
    app.MainLoop()
