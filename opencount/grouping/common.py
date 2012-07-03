import sys, os, csv, pickle, pdb
import numpy as np
from os.path import join as pathjoin
from scipy import misc
sys.path.append('../')
sys.path.append('../pixel_reg/')
import pixel_reg.shared as sh
from pixel_reg.imagesAlign import fastResize

import wx, pickle
from specify_voting_targets.imageviewer import WorldState as WorldState
from specify_voting_targets.imageviewer import BoundingBox as BoundingBox
from util import encodepath
import specify_voting_targets.util_gui as util_gui

DUMMY_ROW_ID = -42 # Also defined in label_attributes.py

class AttributeBox(BoundingBox):
    """
    Represents a bounding box around a ballot attribute.
    """
    def __init__(self, x1, y1, x2, y2, label='', color=None, 
                 id=None, is_contest=False, contest_id=None, 
                 target_id=None,
                 line_width=None, children=None,
                 is_new=False, is_selected=False,
                 attrs=None, side='front'):
        """
        attrs is a dict mapping:
            {str attrtype: str attrval}
        """
        BoundingBox.__init__(self, x1, y1, x2, y2, label=label, color=color,
                             id=id, is_contest=is_contest, contest_id=contest_id,
                             target_id=target_id,
                             line_width=line_width, children=children,
                             is_new=is_new, is_selected=is_selected)
        self.attrs = attrs if attrs else {}
        self.side = side

    def has_attrtype(self, attrtype):
        return attrtype in self.attrs

    def get_attrval(self, attrtype):
        return self.attrs.get(attrtype, None)

    def get_attrtypes(self):
        return self.attrs.keys()

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
                            side=self.side)

    def marshall(self):
        """
        Return a dict-equivalent of myself.
        """
        data = BoundingBox.marshall(self)
        data['attrs'] = self.attrs
        data['side'] = self.side
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
                and self.side == a.side)
    def __repr__(self):
        return "AttributeBox({0},{1},{2},{3},attrs={4},side={5})".format(self.x1, self.y1, self.x2, self.y2, self.attrs, self.side)
    def __str___(self):
        return "AttributeBox({0},{1},{2},{3},attrs={4},side={5})".format(self.x1, self.y1, self.x2, self.y2, self.attrs, self.side)

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

def get_attrtypes(project):
    """
    Returns all attribute types in this election.
    """
    attr_types = set()
    for dirpath, dirnames, filenames in os.walk(project.patch_loc_dir):
        for filename in [f for f in filenames if f.lower().endswith('.csv')]:
            csvfile = open(pathjoin(dirpath, filename), 'r')
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['attr_type'] != '_dummy_':
                    attr_types.add(row['attr_type'])
            csvfile.close()
    return tuple(attr_types)

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

def importPatches(project):
    """
    Reads in all .csv files in precinct_locations/, and returns
    them as {str templatepath: ((y1,y2,x1,x2), grouplabel, side)}
    """
    if not project or not project.patch_loc_dir:
        return
    def is_csvfile(p):
        return os.path.splitext(p)[1].lower() == '.csv'
    fields = ('imgpath', 'id', 'x', 'y', 'width', 'height',
              'attr_type', 'attr_val', 'side')
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
                    if not(boxes.has_key(imgpath)):
                        boxes[imgpath]=[]
                    # Currently, we don't create an exemplar attrpatch
                    # for flipped/wrong-imgorder. For now, just fake it.
                    for flip in (0,1):
                        for imgorder in (0,1):
                            grouplabel = make_grouplabel((row['attr_type'],row['attr_val']),
                                                         ('flip',flip),
                                                         ('imageorder', imgorder))
                            boxes[imgpath].append(((y1, y2, x1, x2), 
                                                   grouplabel,
                                                   side))
            except IOError as e:
                print "Unable to open file: {0}".format(csvfilepath)
    return boxes

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

class GroupClass(object):
    """
    A class that represents a potential group of images.
    """
    # A dict mapping {str label: int count}
    ctrs = {}
    def __init__(self, elements):
        """
        elements: A list of (str sampleid, rankedlist, str imgpatch),
                 where sampleid is the ID for this data point. 
                 rankedlist is a list of grouplabels, which should be
                 sorted by confidence (i.e. the most-likely grouplabel
                 should be at index 0).
                 imgpatch is a path to the image that this element
                 represents.
        """
        self.elements = list(elements)
        for i in range(len(elements)):
            if not issubclass(type(elements[i][1]), list):
                self.elements[i] = list((elements[i][0], list(elements[i][1]), elements[i][2]))
        self.overlayMax = None
        self.overlayMin = None
        # orderedAttrVals is a list of tuples of the form:
        #   (str attrval, int flipped, int imageorder)
        self.orderedAttrVals = []
        
        # Index into the attrs_list that this group is currently using.
        # Is 'finalized' in OnClickOK
        self.index = 0

        self.processElements()

        # The label that will be displayed in the ListBoxes to 
        # the user, i.e. a public name for this GroupClass.
        self.label = str(self.getcurrentgrouplabel())
                     
        if self.label not in GroupClass.ctrs:
            GroupClass.ctrs[self.label] = 1
        else:
            GroupClass.ctrs[self.label] += 1
        self.label += '-{0}'.format(GroupClass.ctrs[self.label])

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
        Go through the elements generating overlays and compiling an ordered list
        of candidate templates
        """
        # weightedAttrVals is a dict mapping {[attrval, flipped]: float weight}
        weightedAttrVals = {}
        # self.elements is a list of the form [(imgpath_i, attrlist_i, patchpath_i), ...]
        # where each attrlist_i is tuples of the form: (attrval_i, flipped_i, imageorder_i)
        for element in self.elements:
            # element := (imgpath, attrlist, patchpath)
            """
            Overlays
            """
            path = element[2]
            try:
                img = misc.imread(path, flatten=1)
                if (self.overlayMin == None):
                    self.overlayMin = img
                else:
                    self.overlayMin = np.fmin(self.overlayMin, img)
                if (self.overlayMax == None):
                    self.overlayMax = img
                else:
                    self.overlayMax = np.fmax(self.overlayMax, img)
            except:
                print "Cannot open patch @ {0}".format(path)
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

        rszFac=sh.resizeOrNot(self.overlayMax.shape,sh.MAX_PRECINCT_PATCH_DISPLAY)
        self.overlayMax = fastResize(self.overlayMax, rszFac) / 255.0
        self.overlayMin = fastResize(self.overlayMin, rszFac) / 255.0
        
    def split(self):
        groups = []
        new_elements = {}
        all_rankedlists = [t[1] for t in self.elements]

        n = num_common_prefix(*all_rankedlists)

        def naive_split(elements):
            mid = int(round(len(elements) / 2.0))
            group1 = elements[:mid]
            group2 = elements[mid:]
            # TODO: Is this groupname/patchDir setting correct?
            groups.append(GroupClass(group1))
            groups.append(GroupClass(group2))
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
            groups.append(GroupClass(elements))
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
