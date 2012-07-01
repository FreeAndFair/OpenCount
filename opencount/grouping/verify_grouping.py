import sys, csv, copy, pdb, os
import threading, time
import timeit
sys.path.append('../')

from util import MyGauge
from specify_voting_targets import util_gui as util_gui
from specify_voting_targets import imageviewer as imageviewer
from specify_voting_targets.util_gui import *
from verify_overlays import VerifyPanel
import label_attributes, util, common


from pixel_reg.imagesAlign import *
import pixel_reg.shared as sh
from pixel_reg.doGrouping import  groupImagesMAP, encodepath

####
## Import 3rd party libraries
####

try:
    import wx
    import wx.animate
    import wx.lib.scrolledpanel as scrolled
except ImportError:
    print """Error importing wxPython (wx) -- to install wxPython (a Python GUI \
library), do (if you're on Linux):
    sudo apt-get install python-wxgtk2.8
Or go to: 
    http://www.wxpython.org/download.php
For OS-specific installation instructions."""
    exit(1)
try:
    import Image
except ImportError:
    print """Error importing Python Imaging Library (Image) -- to install \
PIL (a Python image-processing library), go to: 
    http://www.pythonware.com/products/pil/"""
    exit(1)
try:
    import cv2
except ImportError:
    print """Error importing OpenCV w/ Python bindings (cv2) -- to install \
OpenCV w/ Python bindings (a Python computer vision library), go to:
    http://opencv.willowgarage.com/wiki/
Note that documentation for installing OpenCV is pretty shaky in my \
experience. A README section on installing OpenCV will be created soon.
On Windows, to get the Python bindings, copy/paste the contents of:
    opencv/build/python/2.7 (or 2.6)
to the site-packages directory of your Python installation, i.e.:
    C:/Python27/Lib/site-packages/
For me, this means that you'll be adding two new files to that directory:
    C:/Python27/Lib/site-packages/cv.py
    C:/Python27/Lib/site-packages/cv2.pyd"""
    exit(1)
try:
    import numpy as np
except ImportError:
    print """Error importing Numpy (numpy) -- to install Numpy, go to:
    http://numpy.scipy.org/
You'll probably want to install both scipy and numpy."""
    exit(1)
    
try:
    import scipy
    from scipy import misc
except ImportError:
    print """Error importing SciPy (scipy) -- to install Scipy, go to:
    http://scipy.org/
You'll probably want to install both scipy and numpy."""
    exit(1)
    
import wx.lib.inspection
from wx.lib.pubsub import Publisher

# Set by MainFrame
TIMER = None

# Get this script's directory. Necessary to know this information
# since the current working directory may not be the same as where
# this script lives (which is important for loading resources like
# imgs)
try:
    MYDIR = os.path.abspath(os.path.dirname(__file__))
except NameError:
    # This script is being run directly
    MYDIR = os.path.abspath(sys.path[0])

class GroupingMasterPanel(wx.Panel):
    """
    Panel that contains both RunGroupingPanel and VerifyGroupingPanel.
    """
    def __init__(self, parent, *args, **kwargs):
        wx.Panel.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.run_grouping = RunGroupingPanel(self)
        self.verify_grouping = VerifyPanel(self)
        
        self.sizer.Add(self.run_grouping, proportion=0, flag=wx.EXPAND)
        self.sizer.Add(self.verify_grouping, proportion=0, flag=wx.EXPAND)
        self.run_grouping.Hide()
        self.verify_grouping.Hide()
        self.SetSizer(self.sizer)

    def set_timer(self, timer):
        self.TIMER = timer
        global TIMER
        TIMER = timer

    def start(self):
        self.run_grouping.Show()
        self.run_grouping.start()

    def grouping_done(self, groups):
        """
        If groups is not None, then it means we ran grouping
        computation.
        If groups is None, then we skipped it, so we should
        load in the previous verify-grouping state.
        """
        self.run_grouping.Hide()
        self.verify_grouping.Show()
        if groups:
            self.verify_grouping.start(groups, self.run_grouping.patches)
            self.verify_grouping.SendSizeEvent()
            self.Refresh()
            self.Fit()
        else:
            self.verify_grouping.load_state()
            self.verify_grouping.start_verifygrouping()
            self.SendSizeEvent()
            self.Refresh()
            self.Fit()

    def dump_state(self):
        self.verify_grouping.dump_state()

    def exportResults(self):
        self.verify_grouping.exportResults()
        
    def checkCanMoveOn(self):
        return True

class RunGroupingPanel(wx.Panel):
    def __init__(self, parent, *args, **kwargs):
        wx.Panel.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.project = None

        self.patches = None

        self.sizer = wx.BoxSizer(wx.VERTICAL)

        self.run_button = wx.Button(self, label="Run Grouping")
        self.run_button.Bind(wx.EVT_BUTTON, self.onButton_run)
        self.rerun_button = wx.Button(self, label="Re-run Grouping")
        self.rerun_button.Bind(wx.EVT_BUTTON, self.onButton_rerun)
        self.continue_button = wx.Button(self, label="Continue Correcting Grouping")
        self.continue_button.Bind(wx.EVT_BUTTON, self.onButton_continue)
        
        self.sizer.AddMany(([self.run_button, self.rerun_button, 
                             self.continue_button]))

        self.SetSizer(self.sizer)
        
        Publisher().subscribe(self._pubsub_project, "broadcast.project")

    def start(self):
        assert self.project
        if self.is_grouping_done():
            self.run_button.Hide()
        else:
            self.rerun_button.Hide()
            self.continue_button.Hide()
        # Load in all attribute patch regions
        self.importPatches()

    def onButton_run(self, evt):
        """ Start Grouping """
        try:
            self.TIMER.start_task(('cpu', 'Group Ballots Computation'))
        except Exception as e:
            print e
            print "grouping.RunGroupingPanel can't output time to TIMER."
        self.run_button.Disable()
        self.start_grouping()

    def groupBallotsProcess(self, stopped, deleteall):
        num = 0
        for dirpath, dirnames, filenames in os.walk(self.project.samplesdir):
            for f in filenames:
                if not is_image_ext(f):
                    continue
                num += 1
        kind = len(self.patches.items()[0])

        wx.CallAfter(Publisher().sendMessage, "signals.MyGauge.nextjob", num*kind)
        bal2imgs=pickle.load(open(self.project.ballot_to_images,'rb'))
        tpl2imgs=pickle.load(open(self.project.template_to_images,'rb'))

        groupImagesMAP(bal2imgs,
                       tpl2imgs,
                       self.patches,
                       self.project.extracted_precinct_dir, 
                       self.project.ballot_grouping_metadata, 
                       stopped,
                       deleteall=deleteall)
        
        wx.CallAfter(Publisher().sendMessage, "signals.MyGauge.done")

    def start_grouping(self):
        r = ProcessClass(self.groupBallotsProcess, True)
        r.start()

        def fn():
            try:
                meta = os.path.abspath(self.project.ballot_grouping_metadata)
                num = 0
                for f in os.listdir(self.project.projdir_path):
                    path = os.path.join(self.project.projdir_path, f)
                    path = os.path.abspath(path)
                    if path.startswith(meta):
                        num += len(os.listdir(path))
                return num
            except:
                return 0
        x = MyGauge(self, 1, pos=(200,300), funs=[fn], ondone=self.on_grouping_done, thread=r)
        x.Show()

    def on_grouping_done(self):
        """
        Called right after grouping is done. Creates lists of GroupClass
        objects, based on the results of grouping.
        """
        try:
            self.TIMER.stop_task(('cpu', 'Group Ballots Computation'))
            self.TIMER.start_task(('user', 'Verify Ballot Grouping'))
        except Exception as e:
            print e
            print "grouping can't output time to TIMER."

        bal2imgs=pickle.load(open(self.project.ballot_to_images,'rb'))
        
        grouping_results = {}
        attr_types = set()
        for values_list in self.patches.values():
            for (r, attrtype, attrval, side) in values_list:
                attr_types.add(attrtype)
        if not util.is_multipage(self.project):
            for attr_type in attr_types:
                for ballotid in bal2imgs:
                    metadata_dir = self.project.ballot_grouping_metadata + '-' + attr_type
                    path = os.path.join(metadata_dir, encodepath(ballotid))
                    try:
                        file = open(path, 'rb')
                    except IOError as e:
                        print e
                        pdb.set_trace()

                    data = pickle.load(file)
                    file.close()
                    dummies = [0]*len(data["attrOrder"])
                    attrs_list = zip(data["attrOrder"], data["flipOrder"], dummies)
                    bestMatch = attrs_list[0]
                    grouping_results.setdefault(attr_type, {}).setdefault(bestMatch, []).append((ballotid, attrs_list))
        else:
            # Multipage
            for attr_type in attr_types:
                for ballotid in bal2imgs:
                    metadata_dir = self.project.ballot_grouping_metadata + '-' + attr_type
                    path = os.path.join(metadata_dir, encodepath(ballotid))
                    try:
                        file = open(path, 'rb')
                    except IOError as e:
                        print e
                        pdb.set_trace()
                    # data is a dict with keys 'attrOrder', 'flipOrder', 'err',
                    # and 'imageOrder'
                    data = pickle.load(file)
                    file.close()
                    attrs_list = zip(data["attrOrder"], data["flipOrder"], data["imageOrder"])
                    bestMatch = attrs_list[0]
                    bestAttr = bestMatch[0]
                    grouping_results.setdefault(attr_type, {}).setdefault(bestAttr, []).append((ballotid, attrs_list))
        
        groups = []
        # Seed initial set of groups
        i = 1
        for attrtype, _dict in grouping_results.items():
            for attrval, samples in _dict.items():
                extracted_attr_dir = self.project.extracted_precinct_dir + '-' + attrtype
                group = GroupClass((attrtype, attrval), samples, extracted_attr_dir)
                groups.append(group)
                i += 1        

        self.parent.grouping_done(groups)

    def onButton_rerun(self, evt):
        dirs = sum(self._get_outputdirs(), [])
        msg = """Warning: Choosing to re-run grouping will result in \
all current grouping progress being lost. Grouping is a \
computationally-intensive task (potentially on the order of hours \
or days), so think twice before accepting! \n
Do you really want to re-run grouping?"""
        YES, NO = 510, 805
        dlg = util.WarningDialog(self,
                                 msg, 
                                 ("Yes", "No"),
                                 (YES, NO))
        statusval = dlg.ShowModal()
        if statusval == YES:
            self.start_grouping()

    def onButton_continue(self, evt):
        """
        Skip grouping computation, load in previous group-verify state.
        """
        self.parent.grouping_done(None)

    def importPatches(self):
        """
        Reads in all .csv files in precinct_locations/, and stores
        then in self.patches
        """
        self.patches = common.importPatches(self.project)
        
    def _get_outputdirs(self):
        """
        Return all output directories for this election.
        Outputs:
            (<extractedpatches dirs>, <balgroupmetainit dirs>, <balgroupmeta dirs>)
        """
        attrtypes = common.get_attrtypes(self.project)
        extractedpatches_dirs = []
        balgroupmeta_dirs = []
        for attrtype in attrtypes:
            path1 = self.project.extracted_precinct_dir + '-' + attrtype
            path2 = self.project.ballot_grouping_metadata + '-' + attrtype
            extractedpatches_dirs.append(path1)
            balgroupmeta_dirs.append(path2)
        return extractedpatches_dirs, balgroupmeta_dirs
        
    def is_grouping_done(self):
        """
        Return True if the voted ballots have already been grouped in
        some previous execution. To check this, see if there exist
        images in:
            <projdir>/<project.extracted_precinct_dir+'-'+attrtype>/*
        And if files exist in:
            <projdir>/<project.ballot_grouping_metadata+'-'+attrtype+'_init'>/*
            <projdir>/<project.ballot_grouping_metadata+'-'+attrtype>/*
        """
        extractedpatches_dirs, balgroupmeta_dirs = self._get_outputdirs()
        for dir in extractedpatches_dirs:
            if not util.contains_image(dir):
                print '{0} has no images.'.format(dir)
                return False
        for dir in balgroupmeta_dirs:
            if not util.contains_file(dir):
                print '{0} has no files'.format(dir)
                return False
        return True

    def _pubsub_project(self, msg):
        proj = msg.data
        self.project = proj

class GroupClass(object):
    """
    A class that represents a potential group of images.
    """
    # A dict mapping {str label: int count}
    ctrs = {}
    def __init__(self, groupname, elements, patchDir):
        """
        groupname: A tuple [str attrtype, str attrval]. Represents the 'name'
                   of this Group.
        elements: A list of (str sampleid, attrs_list), where attrs_list
                 is a list [(attrval_1, flip_1, imageorder), ..., (attrval_N, flip_N, imageorder)]
        str patchDir: path to the .png image representing the attribute patch
        """
        self.groupname = groupname
        self.elements = elements
        self.patchDir = patchDir
        self.overlayMax = None
        self.overlayMin = None
        # orderedAttrVals is a list of tuples of the form:
        #   (str attrval, int flipped, int imageorder, "<imgname>+[flipped]")
        self.orderedAttrVals = []
        
        # Index into the attrs_list that this group is currently using.
        # Is 'finalized' in OnClickOK
        self.index = 0
        
        # The label that will be displayed in the ListBoxes to 
        # the user, i.e. a public name for this GroupClass.
        self.label = 'Type: {0} Value: {1}'.format(self.groupname[0],
                                                   self.groupname[1])
        if self.label not in GroupClass.ctrs:
            GroupClass.ctrs[self.label] = 1
        else:
            GroupClass.ctrs[self.label] += 1
        self.label += '-{0}'.format(GroupClass.ctrs[self.label])

        self.processElements()

    def __eq__(self, o):
        return (o and issubclass(type(o), GroupClass) and
                self.groupname == o.groupname and
                self.elements == o.elements)
        
    @property
    def attrtype(self):
        return self.groupname[0]
    @property
    def attrval(self):
        return self.groupname[1]

    def processElements(self):
        """
        Go through the elements generating overlays and compiling an ordered list
        of candidate templates
        """
        # weightedAttrVals is a dict mapping {[attrval, flipped]: float weight}
        weightedAttrVals = {}
        # self.elements is a list of the form [(imgpath_1, attrlist_1), ..., (imgpath_N, attrlist_N)]
        # where each attrlist_i is tuples of the form: (attrval_i, flipped_i, imageorder_i)
        for element in self.elements:
            # element := (imgpath, attrlist)
            """
            Overlays
            """
            path = os.path.join(self.patchDir, encodepath(element[0])+'.png')
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
                pdb.set_trace()
            """
            Ordered templates
            """
            vote = 1.0
            for attrval_t in element[1]:
                # attrval_t := (attrval, flipped, imageorder)
                if (attrval_t not in weightedAttrVals):
                    weightedAttrVals[attrval_t] = vote
                else:
                    weightedAttrVals[attrval_t] = weightedAttrVals[attrval_t] + vote
                
                vote = vote / 2.0
                
        self.orderedAttrVals = [(attrval_t[0], 
                                 attrval_t[1], 
                                 attrval_t[2],
                                 "{0}{1}".format(get_filename(attrval_t[0]), 
                                                 ", flipped" if attrval_t[1] == 1 else ""))
                                for (attrval_t, weight) in sorted(weightedAttrVals.items(), 
                                                                   key=lambda t: t[1],
                                                                   reverse=True)]

        rszFac=sh.resizeOrNot(self.overlayMax.shape,sh.MAX_PRECINCT_PATCH_DISPLAY)
        self.overlayMax = fastResize(self.overlayMax, rszFac) / 255.0
        self.overlayMin = fastResize(self.overlayMin, rszFac) / 255.0
        
    def split(self):
        groups = []
        new_elements = {}
        all_attrslist = [t[1] for t in self.elements]
        # for common_prefix, strip out imageOrder, since that's not
        # important for splitting.
        all_attrslist2 = []
        for lst in all_attrslist:
            t = []
            for (attrval, flipped, imageorder) in lst:
                t.append((attrval,flipped))
            all_attrslist2.append(t)
            
        n = num_common_prefix(*all_attrslist2)

        def naive_split(elements):
            mid = int(round(len(elements) / 2.0))
            group1 = elements[:mid]
            group2 = elements[mid:]
            # TODO: Is this groupname/patchDir setting correct?
            groups.append(GroupClass(self.groupname, group1, self.patchDir))
            groups.append(GroupClass(self.groupname, group2, self.patchDir))
            return groups
            
        if n == len(all_attrslist[0]):
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
        for (samplepath, attrslist) in self.elements:
            if len(attrslist) <= 1:
                print "==== Can't split anymore."
                return [self]
            new_attrval = attrslist[n][0]
            new_groupname = (self.attrtype, new_attrval)
            new_elements.setdefault(new_groupname, []).append((samplepath, attrslist))

        if len(new_elements) == 1:
            # no new groups were made -- just do a naive split
            print "After a 'smart' split, no new groups were made. So, \
just doing a naive split."
            return naive_split(self.elements)

        print 'number of new groups after split:', len(new_elements)
        for groupname in new_elements:
            elements = new_elements[groupname]
            newPatchDir = self.patchDir # TODO: Is this actually used?            
            groups.append(GroupClass(groupname, elements, newPatchDir))
        return groups
            
class ProcessClass(threading.Thread):
    def __init__(self, fn, *args):
        self.fn = fn
        self.args = args
        threading.Thread.__init__(self)
        self.stop = threading.Event()

    def abort(self):
        self.stop.set()

    def stopped(self):
        return self.stop.isSet()

    def run(self):
        self.fn(self.stopped, *self.args)

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

def get_votedballot_img(ballotid, imageorder, project, bal2img=None):
    """
    Returns the numpy image for ballotid, imageorder. Abstracts away the
    annoying details of multipage elections.
    """
    if not util.is_multipage(project):
        return util_gui.open_as_grayscale(ballotid)
    else:
        return util_gui.open_as_grayscale(bal2img[ballotid][imageorder])

def to_templateid(temppath, project, img2temp):
    """
    Returns the TemplateID of temppath. Abstracts away the annoying details
    of multipage elections.
    """
    if not util.is_multipage(project):
        return temppath
    else:
        return img2temp[temppath]


