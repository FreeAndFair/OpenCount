import sys, csv, copy, pdb, os
import threading, time
import timeit
sys.path.append('../')

from util import MyGauge
from specify_voting_targets import util_gui as util_gui
from specify_voting_targets import imageviewer as imageviewer
from specify_voting_targets.util_gui import *
import label_attributes, util

#from gridAlign import *
from pixel_reg.imagesAlign import *
import pixel_reg.shared as sh
from pixel_reg.doGrouping import  groupImagesMAP, encodepath#,groupImages

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

def MyDebug(f):
    def res(*args, **kargs):
        #print "***********"
        #print 'call', f.__name__, args, kargs
        #print "***********"
        #time.sleep(3)
        v = f(*args, **kargs)
        #print "***********"
        #print 'done', f.__name__
        #print "***********"
        return v
    return res

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

@MyDebug
def adjustSize(I, Iref):
    height, width = Iref.shape
    origHeight, origWidth = I.shape
    
    newI = np.zeros((height, width))
    newI[0:min(origHeight, height), 0:min(origWidth, width)] = I[0:min(origHeight, height), 0:min(origWidth, width)]
    
    return newI   

@MyDebug
def generateOverlays(templateImg, sampleImages, region):
    #resMin = resMax = templateImg[region[0]:region[1],region[2]:region[3]]
    resMin = resMax = None
    for img in sampleImages:
        #img = img[region[0]:region[1],region[2]:region[3]]
        if resMax == None:
            resMax = img
        else:
            resMax = np.fmax(resMax, img)
        if resMin == None:
            resMin = img
        else:
            resMin = np.fmin(resMin, img)
            
    return (resMin, resMax) 

@MyDebug
def splitArray(arr):
    if (len(arr) > 1):
        mid = int(round(len(arr) / 2.0))
        arr1 = arr[:mid]
        arr2 = arr[mid:]
        return (arr1, arr2)
    else:
        return None

THRESHOLD = 0.9

class GroupClass(object):
    """
    A class that represents a cluster within a grouping.
    """
    # A dict mapping {str label: int count}
    ctrs = {}
    def __init__(self, groupname, samples, patchDir):
        """
        groupname: A tuple [str attrtype, str attrval]. Represents the 'name'
                   of this Group.
        samples: A list of (str samplepath, attrs_list), where attrs_list
                 is a list [(attrval_1, flip_1, imageorder), ..., (attrval_N, flip_N, imageorder)]
        str patchDir: path to the .png image representing the attribute patch
        """
        self.groupname = groupname
        self.samples = samples
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

        self.processSamples()

    def __eq__(self, o):
        return (o and issubclass(type(o), GroupClass) and
                self.groupname == o.groupname and
                self.samples == o.samples)
        
    @property
    def attrtype(self):
        return self.groupname[0]
    @property
    def attrval(self):
        return self.groupname[1]

    def processSamples(self):
        """
        Go through the samples generating overlays and compiling an ordered list
        of candidate templates
        """
        # weightedAttrVals is a dict mapping {[attrval, flipped]: float weight}
        weightedAttrVals = {}
        # self.samples is a list of the form [(imgpath_1, attrlist_1), ..., (imgpath_N, attrlist_N)]
        # where each attrlist_i is tuples of the form: (attrval_i, flipped_i, imageorder_i)
        for sample in self.samples:
            # sample := (imgpath, attrlist)
            """
            Overlays
            """
            path = os.path.join(self.patchDir, encodepath(sample[0])+'.png')
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
            for attrval_t in sample[1]:
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
        new_samples = {}
        all_attrslist = [t[1] for t in self.samples]
        # for common_prefix, strip out imageOrder, since that's not
        # important for splitting.
        all_attrslist2 = []
        for lst in all_attrslist:
            t = []
            for (attrval, flipped, imageorder) in lst:
                t.append((attrval,flipped))
            all_attrslist2.append(t)
            
        #n = num_common_prefix(*all_attrslist)
        n = num_common_prefix(*all_attrslist2)
        #pdb.set_trace()
        def naive_split(samples):
            mid = int(round(len(samples) / 2.0))
            group1 = samples[:mid]
            group2 = samples[mid:]
            # TODO: Is this groupname/patchDir setting correct?
            groups.append(GroupClass(self.groupname, group1, self.patchDir))
            groups.append(GroupClass(self.groupname, group2, self.patchDir))
            return groups
            
        if n == len(all_attrslist[0]):
            print "rankedlists were same for all voted ballots -- \
doing a naive split instead."
            return naive_split(self.samples)

        if n == 0:
            print "== Wait, n shouldn't be 0 here (in GroupClass.split). \
Changing to n=1, since that makes some sense."
            print "Enter in 'c' for 'continue' to continue execution."
            pdb.set_trace()
            n = 1

        # group by index 'n' into each ballots attrslist (i.e. ranked list)
        for (samplepath, attrslist) in self.samples:
            if len(attrslist) <= 1:
                print "==== Can't split anymore."
                return [self]
            new_attrval = attrslist[n][0]
            new_groupname = (self.attrtype, new_attrval)
            new_samples.setdefault(new_groupname, []).append((samplepath, attrslist))

        if len(new_samples) == 1:
            # no new groups were made -- just do a naive split
            print "After a 'smart' split, no new groups were made. So, \
just doing a naive split."
            return naive_split(self.samples)

        print 'number of new groups after split:', len(new_samples)
        for groupname in new_samples:
            samples = new_samples[groupname]
            newPatchDir = self.patchDir # TODO: Is this actually used?            
            groups.append(GroupClass(groupname, samples, newPatchDir))
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

class VerifyPanel(wx.Panel):
    @MyDebug
    def __init__(self, parent, *args, **kwargs):
        wx.Panel.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.project = None

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        
        # List of groups to-be-verified
        self.queue = []
        self.currentGroup = None
        # List of groups that have been verified
        self.finished = []
        # Stores the final mapping from samplepath to blank ballot - or,
        # in my case, to the correct attrvalues:
        #   {str samplepath: {str attrtype: (str attrval, int flip)}}
        # For multipage elections, this is:
        #   {str ballotid: {str attrtype: (str attrval, int flip)}}
        self.results = dict()
        # A dict mapping {str temppath: list of ((y1,y2,x1,x2), attr_type, attr_val, str side)}
        self.patches = {}
        # res is a dict mapping:
        #       {str attrtype:
        #         {str attrval:
        #           list of (str samplepath, attrs_list)}}
        # where attrs_list is a list of [(attrval_1, flip_1), ..., (attrval_N, flip_N)].
        # attrs_list is the rankedlist. 
        # For multipage, attrs_list is instead:
        #   list of [(attrval_1, flip_1, imageorder_1), ..., (attrval_N, flip_N, imageorderN)]
        # The idea is that each group in res (i.e. all samples with the same
        # key) is a cluster within the ballot grouping.
        self.res = {}
        
        self.resultsPath = None
        self.csvdir = None
        
        self.templatesdir = None
        self.samplesdir = None
        # templates is a dict mapping
        #    {str attrtype: 
        #      {str attrval: obj attrpatch_img}}
        # where attrpatch_img is the image patch on the template corresponding to the
        # attrtype attribute type.
        self.templates = None
        self.templatePaths = None
        
        self.canMoveOn = False
        
        self.mainPanel = scrolled.ScrolledPanel(self, size=self.GetSize(), pos=(0,0))
        
        self.initLayout()
        self.initBindings()
        
        Publisher().subscribe(self._pubsub_project, "broadcast.project")
        Publisher().subscribe(self._pubsub_message_dialog, "message_dialog")
        #Publisher().subscribe(self._pubsub_templatesdir, "broadcast.templatesdir")

    def _pubsub_message_dialog(self, msg):
        """
        Triggered when the UI wants to create a message dialog.
        """
        msg, style = msg.data
        dlg = wx.MessageDialog(self, message=msg, style=style)
        dlg.ShowModal()

    def set_timer(self, timer):
        self.TIMER = timer
        global TIMER
        TIMER = timer

    def fitPanel(self):
        w, h = self.parent.GetClientSize()
        self.mainPanel.SetMinSize((w * 0.95, h * 0.9))
        self.mainPanel.GetSizer().SetSizeHints(self)
        self.mainPanel.SetupScrolling()
    
    @MyDebug
    def initLayout(self):
        gridsizer = wx.GridSizer(rows=4, cols=2, hgap=5, vgap=5)
        
        # HBOX 1 (min overlay)
        st1 = wx.StaticText(self.mainPanel, -1, "min:     ", style=wx.ALIGN_LEFT)
        self.minOverlayImg = wx.StaticBitmap(self.mainPanel, bitmap=wx.EmptyBitmap(1, 1))
        gridsizer.Add(st1, flag=wx.ALIGN_LEFT)
        gridsizer.Add(self.minOverlayImg, flag=wx.ALIGN_LEFT)
        
        # HBOX 2 (max overlay)
        st2 = wx.StaticText(self.mainPanel, -1, "max:     ", style=wx.ALIGN_LEFT)
        self.maxOverlayImg = wx.StaticBitmap(self.mainPanel, bitmap=wx.EmptyBitmap(1, 1))
        gridsizer.Add(st2, flag=wx.ALIGN_LEFT)
        gridsizer.Add(self.maxOverlayImg, flag=wx.ALIGN_LEFT)
        
        # HBOX 3 (template patch)
        st3 = wx.StaticText(self.mainPanel, -1, "Attribute Patch:", style=wx.ALIGN_LEFT)
        self.templateImg = wx.StaticBitmap(self.mainPanel, bitmap=wx.EmptyBitmap(1, 1))
        gridsizer.Add(st3, flag=wx.ALIGN_LEFT)
        gridsizer.Add(self.templateImg, flag=wx.ALIGN_LEFT)
        
        # HBOX 6 (diff patch)
        st4 = wx.StaticText(self.mainPanel, -1, "diff:", style=wx.ALIGN_LEFT)
        self.diffImg = wx.StaticBitmap(self.mainPanel, bitmap=wx.EmptyBitmap(1, 1))
        gridsizer.Add(st4, flag=wx.ALIGN_LEFT)
        gridsizer.Add(self.diffImg, flag=wx.ALIGN_LEFT)
        
        # HBOX 5 (ComboBox and buttons)
        hbox5 = wx.BoxSizer(wx.HORIZONTAL)
        self.templateChoice = wx.ComboBox(self.mainPanel, choices=[], style=wx.CB_READONLY)
        self.okayButton = wx.Button(self.mainPanel, label='OK')
        self.splitButton = wx.Button(self.mainPanel, label='Split')
        self.debugButton = wx.Button(self.mainPanel, label='DEBUG')
        self.quarantineButton = wx.Button(self.mainPanel, label='Quarantine')
        hbox5.Add((5,-1))
        hbox5.Add(self.templateChoice, flag=wx.LEFT | wx.CENTRE)
        hbox5.Add((25,-1))
        hbox5.Add(self.okayButton, flag=wx.LEFT | wx.CENTRE)
        hbox5.Add((25,-1))
        hbox5.Add(self.splitButton, flag=wx.LEFT | wx.CENTRE)
        hbox5.Add((25,-1))
        hbox5.Add(self.debugButton, flag=wx.LEFT | wx.CENTRE)
        hbox5.Add((40,-1))
        hbox5.Add(self.quarantineButton, flag=wx.LEFT | wx.CENTRE)

        # HBOX8 (# of ballots)
        hbox8 = wx.BoxSizer(wx.HORIZONTAL)
        st5 = wx.StaticText(self.mainPanel, -1, "# of ballots in the group: ", style=wx.ALIGN_LEFT)
        self.tNumBallots = wx.TextCtrl(self.mainPanel, value='0')
        self.tNumBallots.SetEditable(False)
        hbox8.Add(st5, flag=wx.CENTRE)
        hbox8.Add(self.tNumBallots, flag=wx.CENTRE)
        
        # VBOX2 (right half)
        vbox2 = wx.BoxSizer(wx.VERTICAL)
        vbox2.Add((-1,5))
        vbox2.Add(hbox8, flag=wx.LEFT | wx.CENTRE)
        #vbox2.Add(hbox4, flag=wx.LEFT | wx.CENTRE)
        #vbox2.Add(vbox1, flag=wx.LEFT | wx.CENTRE)
        vbox2.Add(gridsizer, flag=wx.LEFT | wx.CENTRE)
        vbox2.Add(hbox5, flag=wx.LEFT | wx.CENTRE)
        
        # HBOX7
        
        hbox7 = wx.BoxSizer(wx.HORIZONTAL) 
        self.queueList = wx.ListBox(self.mainPanel)
        self.queueList.Bind(wx.EVT_LISTBOX, self.onSelect_queuelistbox)
        self.finishedList = wx.ListBox(self.mainPanel)
        self.finishedList.Hide() # Not using this for UI simplicity
        #self.queueList.Enable(False)
        hbox7.Add(self.queueList, flag = wx.LEFT | wx.TOP)
        hbox7.Add((25,-1))
        hbox7.Add(self.finishedList, flag = wx.ALIGN_LEFT | wx.ALIGN_TOP)
        hbox7.Add((25,-1))
        hbox7.Add(vbox2, flag = wx.ALIGN_LEFT | wx.ALIGN_CENTRE)
        
        self.mainPanel.SetSizer(hbox7)
        
        self.runButton = wx.Button(self, label='Run Grouping')
        self.rerunButton = wx.Button(self, label="Re-run Grouping")
        self.rerunButton.Bind(wx.EVT_BUTTON, self.onButton_rerun)
        self.continueButton = wx.Button(self, label="Continue Correcting Grouping")
        self.continueButton.Bind(wx.EVT_BUTTON, self.onButton_continue)
        sizerprelude = wx.BoxSizer(wx.VERTICAL)
        sizerprelude.Add(self.runButton)
        sizerprelude.Add(self.rerunButton)
        sizerprelude.Add(self.continueButton)
        sizerprelude.ShowItems(False)
        self.sizerprelude = sizerprelude
        
        self.sizer.Add(self.mainPanel)
        
        self.mainPanel.Hide()

        #self.mainPanel.Show()

        self.SetSizer(self.sizer, deleteOld=False)
        
        #self.Fit()
        #self.parent.Fit()
        
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
            self.groupBallots()

    def onButton_continue(self, evt):
        self.load_state()
        self.start_verifygrouping()

    def onSelect_queuelistbox(self, evt):
        idx = evt.GetSelection()
        if idx >= 0:
            assert idx < len(self.queue)
            self.select_group(self.queue[idx])
            
    @MyDebug
    def start(self, samplesdir, templatesdir):
        self.samplesdir = samplesdir
        self.templatesdir = templatesdir
        self.sizerprelude.ShowItems(True)
        if self.is_grouping_done():
            self.runButton.Hide()
        else:
            self.rerunButton.Hide()
            self.continueButton.Hide()
        self.SetSizer(self.sizerprelude, deleteOld=False)
        self.Layout()
        self.importPatches()
    
    @MyDebug
    def getTemplates(self):
        """
        Load in all attribute patches - in particular, loading in the
        attribute patch image for each attrtype->attrval pair.
        TODO: Don't keep all patches in memory.
        All exemplar attribute patches are stored in:
        <projdir>/ballot_grouping_metadata-<attrname>_exemplars/<attrval>.png
        """
        self.templates = {}
        dirs = os.listdir(self.project.projdir_path)
        for dir in dirs:
            pre, post = 'ballot_grouping_metadata-', '_exemplars'
            if (dir.startswith(pre) and
                dir.endswith(post)):
                attrtype = dir[len(pre):-len(post)]
                for dirpath, dirnames, filenames in os.walk(pathjoin(self.project.projdir_path,
                                                                     dir)):
                    for f in filenames:
                        attrval = os.path.splitext(f)[0]
                        imgpatch = misc.imread(pathjoin(dirpath, f), flatten=1)
                        rszFac = sh.resizeOrNot(imgpatch.shape, sh.MAX_PRECINCT_PATCH_DISPLAY)
                        self.templates.setdefault(attrtype, {})[attrval] = fastResize(imgpatch, rszFac) / 255.0
                    
    @MyDebug
    def countTemplates(self):
        if (self.templates == None):
            i = 0
            for dirpath, dirnames, filenames in os.walk(self.templatesdir):
                for f in filenames:
                    if util_gui.is_image_ext(f):
                        i += 1
            return i
        else:
            return len(self.templates)
    
    @MyDebug
    def groupBallots(self):
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
                #print "Found", num
                return num
            except:
                return 0
        x = MyGauge(self, 1, pos=(200,300), funs=[fn], ondone=self.on_grouping_done, thread=r)
        x.Show()

    def on_grouping_done(self):
        """
        Called right after grouping is done. This is called only if
        grouping (Kai's code) happens - sets up the self.res variable,
        which is used to seed self.queue.
        """
        try:
            self.TIMER.stop_task(('cpu', 'Group Ballots Computation'))
            self.TIMER.start_task(('user', 'Verify Ballot Grouping'))
        except Exception as e:
            print e
            print "grouping can't output time to TIMER."

        bal2imgs=pickle.load(open(self.project.ballot_to_images,'rb'))
        
        groups = {}
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
                    groups.setdefault(attr_type, {}).setdefault(bestMatch, []).append((ballotid, attrs_list))
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
                    groups.setdefault(attr_type, {}).setdefault(bestAttr, []).append((ballotid, attrs_list))
        
        self.res = groups

        i = 1
        for attrtype, _dict in self.res.items():
            for attrval, samples in _dict.items():
                extracted_attr_dir = self.project.extracted_precinct_dir + '-' + attrtype
                group = GroupClass((attrtype, attrval), samples, extracted_attr_dir)
                self.add_group(group)
                #self.queue.append(group)
                i += 1        

        self.start_verifygrouping()

    def dump_state(self):
        if self.project:
            fqueue = open(pathjoin(self.project.projdir_path, 'verifygroupstate.p'), 'wb')
            d = {}
            q = list(self.queue)
            if self.currentGroup:
                q.insert(0, self.currentGroup)
            q.extend(self.finished)
            d['todo'] = q
            #d['finished'] = self.finished
            d['finished'] = []
            pickle.dump(d, fqueue)
        
    def load_state(self):
        fstate = open(pathjoin(self.project.projdir_path, 'verifygroupstate.p'), 'rb')
        d = pickle.load(fstate)
        todo = d['todo']
        todo.extend(d['finished'])
        for group in todo:
            self.add_group(group)
        #self.queue = d['todo']
        # Don't worry about keeping 'finished' separated from 'queue'
        # for now.
        #self.queue.extend(d['finished'])
        self.finished = d['finished']
 
        #self.getTemplates()

    def start_verifygrouping(self):
        """
        Called after sample ballots have been grouped by Kai's grouping
        code. Sets up UI widgets of the verification UI.
        """
        self.SetSizer(self.sizer, deleteOld=False)

        #for group in self.queue:
        #    self.queueList.Append(group.label)        
        #for finished_group in self.finished:
        #    self.finishedList.Append(finished_group.label)

        self.getTemplates()

        if self.queue:
            self.select_group(self.queue[0])
        #self.queueList.SetSelection(0)

        self.runButton.Disable()
        self.runButton.Hide()
        self.mainPanel.Show()
        self.Fit()

    @MyDebug
    def groupBallotsProcess(self, stopped, deleteall):
        num = 0
        for dirpath, dirnames, filenames in os.walk(self.samplesdir):
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
        
    @MyDebug
    def initBindings(self):
        self.templateChoice.Bind(wx.EVT_COMBOBOX, self.OnSelectTemplate)
        self.Bind(wx.EVT_BUTTON, self.OnClickOK, self.okayButton)
        self.Bind(wx.EVT_BUTTON, self.OnClickSplit, self.splitButton)
        self.Bind(wx.EVT_BUTTON, self.OnClickRun, self.runButton)
        self.Bind(wx.EVT_BUTTON, self.OnClickDebug, self.debugButton)
        self.Bind(wx.EVT_BUTTON, self.OnClickQuarantine, self.quarantineButton)
        self.Bind(wx.EVT_SIZE, self.OnSize)
        
    @MyDebug
    def exportResults(self):
        """
        Export all attrtype->attrval mappings for each sample ballot
        to a csv file.
        """
        if len(self.results.items()) == 0:
            return
        if not self.is_done_verifying():
            return
        attr_types = set()
        for values_list in self.patches.values():
            for (r, attrtype, attrval, side) in values_list:
                # Question: 'side' isn't used here. Does it need to be?
                attr_types.add(attrtype)
        fields = ('samplepath','templatepath') + tuple(sorted(tuple(attr_types))) + ('flipped_front', 'flipped_back')
        csvfilepath = self.resultsPath
        csvfile = open(csvfilepath, 'wb')
        dictwriter = csv.DictWriter(csvfile, fieldnames=fields)
        try:
            dictwriter.writeheader()
        except AttributeError:
            util_gui._dictwriter_writeheader(csvfile, fields)
        dictwriter.writerows(self._rows)
    
    @MyDebug
    def displayNextGroup(self):
        '''
        if (len(self.queue) == 0):
            self.okayButton.Disable()
            self.splitButton.Disable()
            self.canMoveOn = True
            return
            
        self.currentGroup = self.queue.pop(0)
        
        overlayMin = self.currentGroup.overlayMin
        overlayMax = self.currentGroup.overlayMax
        ordered_attrvals = self.currentGroup.orderedAttrVals
        samples = self.currentGroup.samples
        
        self.minOverlayImg.SetBitmap(NumpyToWxBitmap(overlayMin * 255.0))
        self.maxOverlayImg.SetBitmap(NumpyToWxBitmap(overlayMax * 255.0))
        
        self.tNumBallots.SetValue("{0}".format(len(samples)))
        
        self.templateChoice.Clear()
        history = set()
        for (attrval, flipped, imageorder, foo) in ordered_attrvals:
            if attrval not in history:
                display_string = attrval
                self.templateChoice.Append(display_string)
                history.add(attrval)
        
        #self.templateChoice.SetSelection(0)
        self.templateChoice.SetSelection(self.currentGroup.index)
        
        self.updateTemplateThumb()
        
        if (len(samples) <= 1):
            self.splitButton.Enable(False)
        else:
            self.splitButton.Enable(True)
        
        self.fitPanel()
        #self.Fit()
        #self.parent.Fit()
        '''
        pass
        
    @MyDebug
    def updateTemplateThumb(self):
        """
        Updates the 'Attribute Patch' and 'Diff' image patches.
        """
        overlayMin = self.currentGroup.overlayMin
        overlayMax = self.currentGroup.overlayMax
        templates = self.currentGroup.orderedAttrVals
        samples = self.currentGroup.samples

        attrtype = self.currentGroup.attrtype

        attrval = self.templateChoice.GetStringSelection()
        attrpatch_img = self.templates[attrtype][attrval]
        
        height, width = attrpatch_img.shape
        IO = imagesAlign(overlayMax, attrpatch_img)
        Dabs=np.abs(IO[1]-attrpatch_img)
        diffImg = np.vectorize(lambda x: x * 255.0 if x >= THRESHOLD else 0.0)(Dabs)
        
        self.templateImg.SetBitmap(NumpyToWxBitmap(attrpatch_img * 255.0))
        self.diffImg.SetBitmap(NumpyToWxBitmap(diffImg))
        self.Refresh()

    @MyDebug
    def OnSelectTemplate(self, event):
        """
        Triggered when the user selects a different attribute value
        in the dropdown menu.
        """
        self.updateTemplateThumb()
        
    def add_finalize_group(self, group, final_index):
        group.index = final_index
        self.finished.append(group)
        self.finishedList.Append(group.label)

    def add_group(self, group):
        """
        Adds a new GroupClass to internal datastructures, and updates
        relevant UI components.
        """
        assert group not in self.queue
        self.queue.insert(0, group)
        self.queueList.Insert(group.label, 0)
        
    def remove_group(self, group):
        """
        Removes a GroupClass from my internal datastructures, and
        updates relevant UI components.
        """
        assert group in self.queue
        idx = self.queue.index(group)
        self.queue.remove(group)
        self.queueList.Delete(idx)

    def select_group(self, group):
        """
        Displays the GroupClass to the user, to allow him/her to work
        with it. 
        """
        assert group in self.queue
        self.currentGroup = group
        idx = self.queue.index(group)
        self.queueList.SetSelection(idx)

        overlayMin = self.currentGroup.overlayMin
        overlayMax = self.currentGroup.overlayMax
        ordered_attrvals = self.currentGroup.orderedAttrVals
        samples = self.currentGroup.samples
        
        self.minOverlayImg.SetBitmap(NumpyToWxBitmap(overlayMin * 255.0))
        self.maxOverlayImg.SetBitmap(NumpyToWxBitmap(overlayMax * 255.0))
        
        self.tNumBallots.SetValue("{0}".format(len(samples)))
        
        self.templateChoice.Clear()
        history = set()
        for (attrval, flipped, imageorder, foo) in ordered_attrvals:
            if attrval not in history:
                display_string = attrval
                self.templateChoice.Append(display_string)
                history.add(attrval)
        
        #self.templateChoice.SetSelection(0)
        self.templateChoice.SetSelection(self.currentGroup.index)
        
        self.updateTemplateThumb()
        
        if (len(samples) <= 1):
            self.splitButton.Disable()
        else:
            self.splitButton.Enable()
        
        self.fitPanel()
    
    @MyDebug
    def OnClickOK(self, event):
        #templates = self.currentGroup.orderedAttrVals
        #samples = self.currentGroup.samples
        #attrtype = self.currentGroup.attrtype
        index = self.templateChoice.GetCurrentSelection()
        self.add_finalize_group(self.currentGroup, index)
        #for samplepath, attrs_list in samples:
        #    self.results.setdefault(samplepath, {})[attrtype] = attrs_list[index]
        
        self.remove_group(self.currentGroup)

        #self.displayNextGroup()
        
        #curIndex = self.queueList.GetSelection()
        #self.queueList.Delete(curIndex)
        #self.queueList.SetSelection(curIndex)
        if self.is_done_verifying():
            self.currentGroup = None
            self.done_verifying()
        else:
            self.select_group(self.queue[0])

    def is_done_verifying(self):
        return not self.queue
        
    def done_verifying(self):
        """
        When the user has finished verifying all groups, do some
        fancy computing.
        """
        # First populate results
        for group in self.finished:
            samples = group.samples
            attrtype = group.attrtype
            index = group.index
            for samplepath, attrs_list in samples:
                self.results.setdefault(samplepath, {})[attrtype] = attrs_list[index]

        attr_types = set()
        for values_list in self.patches.values():
            for (r, attrtype, attrval, side) in values_list:
                # Question: 'side' isn't used here. Does it need to be?
                attr_types.add(attrtype)
        fields = ('samplepath','templatepath') + tuple(sorted(tuple(attr_types))) + ('flipped_front', 'flipped_back')
        # maps {str ballotid: {str attrtype: int imageorder}}
        # this is useful because we can then infer which 
        # voted image is front/back (in bal2imgs) by comparing
        # to the temp2imgs, at which we know the correct ordering.
        sample_attrmap = {}
        bal2tmp = {}
        sample_flips = {} # {str ballotid: [flip0, flip1]}
        if util.is_multipage(self.project):
            img2tmp = pickle.load(open(self.project.image_to_template, 'rb'))
        else:
            img2tmp = None
        # _rows is a list of rows
        self._rows = []
        hosed_bals = []
        for samplepath, attrdict in self.results.items():
            row = {}
            row['samplepath'] = samplepath
            for attrtype, (attrval, flip, imageorder) in attrdict.items():
                row[attrtype] = attrval
                sample_flips.setdefault(samplepath, [None, None])[imageorder] = flip
                sample_attrmap.setdefault(samplepath, {})[attrtype] = imageorder
            munged_patches = munge_patches(self.patches,
                                           util.is_multipage(self.project),
                                           img2tmp)
            templateid = determine_template(attrdict, munged_patches)
            if not templateid:
                hosed_bals.append((samplepath, attrdict, munged_patches))
                continue
            row['templatepath'] = templateid
            bal2tmp[samplepath] = templateid
            self._rows.append(row)
        if hosed_bals:
            msg = """Warning: There were {0} voted ballots for which \
OpenCount wasn't able to determine the corresponding blank ballot. \
OpenCount has quarantined these voted ballots - you will have the \
opportunity to manually group these ballots.\n""".format(len(hosed_bals))
            msg2 = "The hosed ballots were:\n"
            qfile = open(self.project.quarantined, 'a')
            for (samplepath, attrdict, munged_patches) in hosed_bals:
                msg2 += """    Imagepath: {0}
        Attributes: {1}\n""".format(os.path.relpath(samplepath), attrdict)
                print >>qfile, os.path.abspath(samplepath)
            qfile.close()
            msg3 = "\nFor reference, the template attr patches were: {0}".format(munged_patches)
            HOSED_FILENAME = os.path.join(self.project.projdir_path, 'hosed_votedballots.log')
            msg4 = "\n(This information has been dumped to '{0}'".format(HOSED_FILENAME)
            msg = msg + msg2 + msg3 + msg4
            dlg = wx.MessageDialog(self, message=msg, style=wx.OK)
            try:
                f = open(HOSED_FILENAME, 'w')
                print >>f, msg
                f.close()
            except IOError as e:
                print e
            self.Disable()
            dlg.ShowModal()
            self.Enable()
        # Force one export, to allow add_flipinfo to do its magic
        self.exportResults()
                                          
        # For multipage, we need to both:
        #  a.) Correct the ordering in ballot_to_images
        #  b.) Add in the correct flipped_front/flipped_back column values
        correctedflips = fix_ballot_to_images(self.project, bal2tmp, sample_attrmap, self.patches, sample_flips)
        # but always 'correct' the flipinfo, even for single page elections
        add_flipinfo(self.project, correctedflips, fields, self.resultsPath)
        
    @MyDebug
    def OnClickSplit(self, event):
        newGroups = self.currentGroup.split()
        for group in newGroups:
            #self.queue.insert(0, group)
            self.add_group(group)
        
        #self.displayNextGroup()
        self.remove_group(self.currentGroup)
        self.select_group(self.queue[0])
        
        # Visual
        '''
        curIndex = self.queueList.GetSelection()
        curGroup = self.queueList.GetStringSelection()
        i = len(newGroups)
        while i > 0:
            self.queueList.Insert("{0}_{1}".format(curGroup, i), curIndex)
            i -= 1
        
        self.queueList.Delete(curIndex+len(newGroups))
        self.queueList.SetSelection(curIndex)
        '''
        self.queueList.Fit()
        self.Fit()
        
    @MyDebug
    def OnClickRun(self, event):
        try:
            self.TIMER.start_task(('cpu', 'Group Ballots Computation'))
        except Exception as e:
            print e
            print "grouping.VerifyPanel can't output time to TIMER."
        self.runButton.Disable()
        self.groupBallots()
        
    def OnClickDebug(self, event):
        if (self.currentGroup != None):
            samples = self.currentGroup.samples
            for sample in samples:
                print sample[0]
            
    def quarantine_group(self, group):
        """
        Quarantines group.
        """
        samples = group.samples
        qfile = open(self.project.quarantined, 'a')
        for sample in samples:
            print >>qfile, os.path.abspath(sample[0])
        qfile.close()
        self.remove_group(group)
        
    def OnClickQuarantine(self, event):
        if (self.currentGroup != None):
            '''
            self.displayNextGroup()
            curIndex = self.queueList.GetSelection()
            self.queueList.Delete(curIndex)
            self.queueList.SetSelection(curIndex)
            '''
            self.quarantine_group(self.currentGroup)
            if self.is_done_verifying():
                self.done_verifying()
            else:
                self.select_group(self.queue[0])
        
    @MyDebug
    def checkCanMoveOn(self):
        # TODO: Fix this implementation. Currently, self.templatesdir
        # happens to be None, which causes errors.
        return True

        if (self.countTemplates() == 1):
            return True

        return self.canMoveOn
    
    def _get_outputdirs(self):
        """
        Return all output directories for this election.
        Outputs:
            (<extractedpatches dirs>, <balgroupmetainit dirs>, <balgroupmeta dirs>)
        """
        attrtypes = self.get_attrtypes()
        extractedpatches_dirs = []
        #balgroupmetainit_dirs = []
        balgroupmeta_dirs = []
        for attrtype in attrtypes:
            path1 = self.project.extracted_precinct_dir + '-' + attrtype
            path2 = self.project.ballot_grouping_metadata + '-' + attrtype
            extractedpatches_dirs.append(path1)
            #balgroupmetainit_dirs.append(path2+'_init')
            balgroupmeta_dirs.append(path2)
        #return extractedpatches_dirs, balgroupmetainit_dirs, balgroupmeta_dirs
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
        #extractedpatches_dirs, balgroupmetainit_dirs, balgroupmeta_dirs = self._get_outputdirs()
        extractedpatches_dirs, balgroupmeta_dirs = self._get_outputdirs()
        for dir in extractedpatches_dirs:
            if not util.contains_image(dir):
                print '{0} has no images.'.format(dir)
                return False
        #for dir in balgroupmetainit_dirs:
        #    if not util.contains_file(dir):
        #        print '{0} has no files'.format(dir)
        #        return False
        for dir in balgroupmeta_dirs:
            if not util.contains_file(dir):
                print '{0} has no files'.format(dir)
                return False
        return True

    def get_attrtypes(self):
        """
        Returns all attribute types in this election.
        """
        attr_types = set()
        for dirpath, dirnames, filenames in os.walk(self.project.patch_loc_dir):
            for filename in [f for f in filenames if f.lower().endswith('.csv')]:
                csvfile = open(pathjoin(dirpath, filename), 'r')
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if row['attr_type'] != '_dummy_':
                        attr_types.add(row['attr_type'])
                csvfile.close()
        return tuple(attr_types)

    @MyDebug
    def importPatches(self):
        """
        Reads in all .csv files in precinct_locations/, and stores
        then in self.patches
        """
        if not self.csvdir:
            return
        def is_csvfile(p):
            return os.path.splitext(p)[1].lower() == '.csv'
        fields = ('imgpath', 'id', 'x', 'y', 'width', 'height',
                  'attr_type', 'attr_val', 'side')
        boxes = {}
        for dirpath, dirnames, filenames in os.walk(self.csvdir):
            for csvfilepath in [f for f in filenames if is_csvfile(f)]:
                try:
                    csvfile = open(os.path.join(dirpath, csvfilepath), 'rb')
                    dictreader = csv.DictReader(csvfile)
                    for row in dictreader:
                        imgpath = os.path.abspath(row['imgpath'])
                        id = int(row['id'])
                        if id == label_attributes.DUMMY_ROW_ID:
                            boxes.setdefault(imgpath, [])
                            continue
                        x1 = int(row['x'])
                        y1 = int(row['y'])
                        x2 = x1 + int(row['width'])
                        y2 = y1 + int(row['height'])
                        side = row['side']
                        if not(boxes.has_key(imgpath)):
                            boxes[imgpath]=[]
                        boxes[imgpath].append(((y1, y2, x1, x2), 
                                               row['attr_type'], 
                                               row['attr_val'],
                                               side))
                except IOError as e:
                    print "Unable to open file: {0}".format(csvfilepath)
        self.patches = boxes
        
    def _pubsub_project(self, msg):
        project = msg.data
        self.project = project
        
        self.csvdir = project.patch_loc_dir
        self.resultsPath = project.grouping_results
        
        self.project.addCloseEvent(self.exportResults)
        
    def OnSize(self, event):
        self.fitPanel()
        event.Skip()

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
    
def munge_patches(patches, is_multipage=False, img2tmp=None):
    """
    Convert self.patches dict to the template_attrs dict needed for
    determine_template.
    If multipage, this should return a dict:
        {str tempid: {str attrtype: (str attrval, str side)}}
    else:
        {str temppath: {str attrtype: str attrval, 'front'}}
    Input:
      dict patches: {str temppath: list of ((y1,y2,x1,x2), attrtype, attrval, side)}
    """
    result = {}
    if not is_multipage:
        for temppath, tuples in patches.iteritems():
            for (r, attrtype, attrval, side) in tuples:
                result.setdefault(temppath, {})[attrtype] = (attrval, 'front')
    else:
        for temppath, tuples in patches.iteritems():
            for (r, attrtype, attrval, side) in tuples:
                result.setdefault(img2tmp[temppath], {})[attrtype] = (attrval, side)

    return result

def determine_template(sample_attrs, template_attrs):
    """
    Given a sample image's attrtype->attrval mappings, return the
    template that has the same attrtype->attrval mapping.
    Also returns the side ordering of the sample (i.e. that '0' is 'front',
    and '1' is 'back').
    Input:
      dict sample_attrs: {str attrtype: (str attrval, int flip, int imageorder)}
      dict template_attrs: {str temppath: {str attrtype: str attrval, int side}}
    """
    for temppath, temp_attrdict in template_attrs.iteritems():
        flag = True
        for attrtype, (temp_attrval, temp_side) in temp_attrdict.iteritems():
            if attrtype not in sample_attrs:
                # I.e. an attr on the back of a Template will not be found
                # on a sample image of a Front side.
                continue
            sample_attrval, flip, imageorder = sample_attrs[attrtype]
            if sample_attrval != temp_attrval:
                flag = False
                break
        if flag:
            return temppath
    # if we get here, we're hosed
    print "== Error, determine_template couldn't find a template. We're hosed."
    return None

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

def fix_ballot_to_images(project, bal2tmp, sample_attrmap, patches, sample_flips):
    """
    Fix the ordering in the ballot_to_images mapping.
    dict bal2tmp: {str ballotid: str templateid}
    dict sample_attrmap: {str ballotid: {str attrtype: int imageorder}}
    dict patches: {str temppath: list of ((y1,y2,x1,x2),attrtype,attrval,side)}
    dict sample_flips: {str ballotid: [flip_0, flip_1]}
    Returns a dict that tells you, for each sample Ballot, whether the front/back
    is flipped:
      {str ballotid: [flip_front, flip_back]}
    """
    if not util.is_multipage(project):
        # Don't do anything, just return a 'dummy' correctedflips.
        # In this case, it is just sample_flips, since for singepage,
        # imageorder '0' is always 'front'
        return sample_flips
    else:
        b2imgs = pickle.load(open(project.ballot_to_images, 'rb'))
        tmp2imgs = pickle.load(open(project.template_to_images, 'rb'))
        correctedflips = {}
        for ballotid, templateid in bal2tmp.iteritems():
            side0, side1 = None, None
            attr_tuples = patches[tmp2imgs[templateid][0]] + patches[tmp2imgs[templateid][1]]
            for (r, attrtype, attrval, side) in attr_tuples:
                imageorder = sample_attrmap[ballotid][attrtype]
                if imageorder == 0:
                    side0 = side
                else:
                    side1 = side
            img0, img1 = b2imgs[ballotid]
            front, back = None,None
            flip0, flip1 = sample_flips[ballotid]
            if side0 == 'front':
                front = img0
                back = img1
                correctedflips[ballotid] = flip0, flip1
            else:
                front = img1
                back = img0
                flip0, flip1 = sample_flips[ballotid]
                correctedflips[ballotid] = flip1, flip0
            b2imgs[ballotid] = front, back
        pickle.dump(b2imgs, open(project.ballot_to_images, 'wb'))
        return correctedflips

def add_flipinfo(project, correctedflips, fields, csvpath):
    """
    In the .csv file in grouping_results.csv, fill in the 'flipped_front'
    and 'flipped_back' columns.
    sample_flips: dict {str ballotid: [flipfront, flipback]}
    """
    newrows = []
    csvfile = open(csvpath, 'r')
    for row in csv.DictReader(csvfile):
        sampleid = row['samplepath']
        flipfront, flipback = correctedflips[sampleid]
        row['flipped_front'] = flipfront if flipfront != None else 'None'
        row['flipped_back'] = flipback if flipback != None else 'None'
        newrows.append(row)
    csvfile.close()
    writefile = open(csvpath, 'w')
    util_gui._dictwriter_writeheader(writefile, fields)
    writer = csv.DictWriter(writefile, fieldnames=fields)
    writer.writerows(newrows)
    writefile.close()
