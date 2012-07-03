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
        self.project = None
        self.sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.run_grouping = RunGroupingPanel(self)
        self.verify_grouping = VerifyPanel(self)

        self._rows = [] # Used for exportResults
        
        self.sizer.Add(self.run_grouping, proportion=1, flag=wx.EXPAND)
        self.sizer.Add(self.verify_grouping, proportion=1, flag=wx.EXPAND)
        self.run_grouping.Hide()
        self.verify_grouping.Hide()
        self.SetSizer(self.sizer)

        Publisher().subscribe(self._pubsub_project, "broadcast.project")

    def _pubsub_project(self, msg):
        project = msg.data
        self.project = project
        
        self.project.addCloseEvent(self.exportResults)

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
        def get_exemplar_paths():
            """ Return a dict {grouplabel: str imgpath} """
            # TODO: Currently we dont' create exemplars for
            # upsidedown/wrong-side attribute patches. So,
            # we fake it. 
            exemplar_paths = {}
            dirs = os.listdir(self.project.projdir_path)
            pre, post = 'ballot_grouping_metadata-', '_exemplars'
            for dir in dirs:
                if (dir.startswith(pre) and
                    dir.endswith(post)):
                    attrtype = dir[len(pre):-len(post)]
                    for dirpath, dirnames, filenames in os.walk(pathjoin(self.project.projdir_path,
                                                                         dir)):
                        for f in filenames:
                            attrval = os.path.splitext(f)[0]
                            for flip in (0, 1):
                                for imageorder in (0, 1):
                                    exemplar_paths[common.make_grouplabel((attrtype,attrval),
                                                                          ('flip', flip),
                                                                          ('imageorder', imageorder))] = pathjoin(dirpath, f)
                             
            return exemplar_paths

        self.run_grouping.Hide()
        self.verify_grouping.Show()
        if groups:
            exemplar_paths = get_exemplar_paths()
            self.verify_grouping.start(groups, self.run_grouping.patches, exemplar_paths, ondone=self.verifying_done)
            self.verify_grouping.SendSizeEvent()
            self.SendSizeEvent()
            self.Refresh()
            self.Fit()
        else:
            #self.verify_grouping.start(groups, patches, exemplar_paths)
            self.verify_grouping.load_state()
            exemplar_paths = get_exemplar_paths()
            self.verify_grouping.ondone = self.verifying_done
            self.verify_grouping.load_exemplar_attrpatches(exemplar_paths)
            self.verify_grouping.start_verifygrouping()
            self.SendSizeEvent()
            self.Refresh()
            self.Fit()

    def verifying_done(self, results):
        """
        Called when the user is finished with grouping verification.
        results is a dict of the form:
            {grouplabel: elements}
        """
        attr_types = set(common.get_attrtypes(self.project))
        # munge results -> results_foo
        results_foo = {} # {samplepath: {attrtype: (attrval, flip, imgorder)}}
        for grouplabel, elements in results.iteritems():
            # elements := list of (samplepath, rankedlist, patchpath)
            if not elements:
                # This grouplabel never got assigned any samples
                continue
            for attrtype in attr_types:
                attrval = common.get_propval(grouplabel, attrtype)
                if attrval:
                    break
            assert attrval != None
            flip = common.get_propval(grouplabel, 'flip')
            imgorder = common.get_propval(grouplabel, 'imageorder')
            assert flip != None
            assert imgorder != None
            for (samplepath, rankedlist, patchpath) in elements:
                results_foo.setdefault(samplepath, {})[attrtype] = (attrval,
                                                                    flip,
                                                                    imgorder)
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
        # _rows is a list of rows, used in exportResults
        self._rows = []
        hosed_bals = []

        munged_patches = munge_patches(self.verify_grouping.patches,
                                       self.project,
                                       util.is_multipage(self.project),
                                       img2tmp)

        for samplepath, attrdict in results_foo.items():
            row = {}
            row['samplepath'] = samplepath
            for attrtype, (attrval, flip, imageorder) in attrdict.items():
                row[attrtype] = attrval
                sample_flips.setdefault(samplepath, [None, None])[imageorder] = flip
                sample_attrmap.setdefault(samplepath, {})[attrtype] = imageorder
            
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
        correctedflips = fix_ballot_to_images(self.project, bal2tmp, sample_attrmap, self.verify_grouping.patches, sample_flips)
        # but always 'correct' the flipinfo, even for single page elections
        add_flipinfo(self.project, correctedflips, fields, self.project.grouping_results)
        
    def exportResults(self):
        """
        Export all attrtype->attrval mappings for each sample ballot
        to a csv file.
        """
        if not self._rows:
            return
        if not self.is_done_verifying():
            return
        attr_types = common.get_attrtypes(self.project)
        fields = ('samplepath','templatepath') + tuple(sorted(tuple(attr_types))) + ('flipped_front', 'flipped_back')
        csvfilepath = self.project.grouping_results
        csvfile = open(csvfilepath, 'wb')
        dictwriter = csv.DictWriter(csvfile, fieldnames=fields)
        try:
            dictwriter.writeheader()
        except AttributeError:
            util_gui._dictwriter_writeheader(csvfile, fields)
        dictwriter.writerows(self._rows)

    def is_done_verifying(self):
        return self.verify_grouping.is_done_verifying()

    def dump_state(self):
        self.verify_grouping.dump_state()

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

        # munge patches into format that groupImagesMAP wants
        all_attrtypes = common.get_attrtypes(self.project)
        def munge_patches(patches, attrtypes):
            """ Converts {str templatepath: ((y1,y2,x1,x2),grouplabel,side)}
            to: {str templatepath: ((y1,y2,x1,x2), attrtype, attrval, side)}
            """
            result = {}
            for temppath, patchtriple in patches.iteritems():
                for (bb, grouplabel, side) in patchtriple:
                    for attrtype in attrtypes:
                        if common.get_propval(grouplabel, attrtype):
                            attrval = common.get_propval(grouplabel, attrtype)
                            result.setdefault(temppath, []).append((bb, attrtype, attrval, side))
            assert len(result) == len(patches)
            return result
        munged = munge_patches(self.patches, all_attrtypes)
        groupImagesMAP(bal2imgs,
                       tpl2imgs,
                       munged,
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
        attr_types = common.get_attrtypes(self.project)
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
                    patchpath = pathjoin(self.project.extracted_precinct_dir+"-"+attr_type,
                                    encodepath(ballotid)+'.png')
                    grouping_results.setdefault(attr_type, {}).setdefault(bestMatch, []).append((ballotid, attrs_list, patchpath))
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
                    patchpath = pathjoin(self.project.extracted_precinct_dir+"-"+attr_type,
                                    encodepath(ballotid)+'.png')
                                                                
                    grouping_results.setdefault(attr_type, {}).setdefault(bestAttr, []).append((ballotid, attrs_list, patchpath))
        
        groups = []
        # Seed initial set of groups
        i = 1
        # Note: grouping_results is structured strangely, hence, why
        # the strange code below.
        for attrtype, _dict in grouping_results.items():
            for (attrval,flip,imgorder), samples in _dict.items():
                #extracted_attr_dir = self.project.extracted_precinct_dir + '-' + attrtype
                munged_samples = []
                for (path, rankedlist, patchpath) in samples:
                    maps = []
                    for (attrval, flip, imgorder) in rankedlist:
                        maps.append(((attrtype,attrval),('flip',flip),('imageorder',imgorder)))
                    munged_samples.append((path,
                                           [common.make_grouplabel(*mapping) for mapping in maps],
                                          patchpath))
                group = common.GroupClass(munged_samples)
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

def munge_patches(patches, project, is_multipage=False, img2tmp=None):
    """
    Convert self.patches dict to the template_attrs dict needed for
    determine_template.
    If multipage, this should return a dict:
        {str tempid: {str attrtype: (str attrval, str side)}}
    else:
        {str temppath: {str attrtype: str attrval, 'front'}}
    Input:
      dict patches: {str temppath: list of ((y1,y2,x1,x2), grouplabel, side)}
    Output:
      dict result: {str temppath: {str attrype: (str attrval, int side)}}
    """
    def get_attrtypeval(grouplabel,attrtypes):
        v = None
        for attrtype in attrtypes:
            v = common.get_propval(grouplabel, attrtype)
            if v:
                break
        assert v
        return attrtype, v
    result = {}
    attrtypes = common.get_attrtypes(project)
    if not is_multipage:
        for temppath, tuples in patches.iteritems():
            for (r, grouplabel, side) in tuples:
                attrtype, attrval = get_attrtypeval(grouplabel, attrtypes)
                result.setdefault(temppath, {})[attrtype] = (attrval, 'front')
    else:
        for temppath, tuples in patches.iteritems():
            for (r, grouplabel, side) in tuples:
                attrtype, attrval = get_attrtypeval(grouplabel, attrtypes)
                result.setdefault(img2tmp[temppath], {})[attrtype] = (attrval, side)

    return result
