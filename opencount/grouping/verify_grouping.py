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

from common import TABULATION_ONLY_ID, DIGIT_BASED_ID

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
        def get_example_digit(dir):
            for dirpath, dirnames, filenames in os.walk(dir):
                for imgname in [f for f in filenames if util_gui.is_image_ext(f)]:
                    # Just grab the first image, whatevs
                    imgpath = pathjoin(dirpath, imgname)
                    return imgpath
            # If we get here, uh oh, we couldn't find one...
            print "Uhoh, we couldn't find a digit exemplar...?"
            pdb.set_trace()
            return None
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
            # Account for digits
            rootdir = pathjoin(self.project.projdir_path,
                               self.project.digit_exemplars_outdir)
            if os.path.exists(rootdir):
                digitdirs = os.listdir(rootdir)
                for digitdir in digitdirs:
                    # Assumes digitdirs is of the form:
                    #    <digitdirs>/0_examples/
                    #    <digitdirs>/1_examples/
                    #    ...
                    digitfullpath = os.path.join(rootdir, digitdir)
                    digit = digitdir.split('_')[0]
                    digitgrouplabel = common.make_grouplabel(('digit', digit))
                    imgpath = get_example_digit(digitfullpath)
                    assert digitgrouplabel not in exemplar_paths
                    assert imgpath != None
                    exemplar_paths[digitgrouplabel] = imgpath
            return exemplar_paths

        self.run_grouping.Hide()
        self.verify_grouping.Show()
        if groups:
            exemplar_paths = get_exemplar_paths()
            self.verify_grouping.start(groups, exemplar_paths, ondone=self.verifying_done)
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
            {grouplabel: list of GroupClasses}
        """
        attr_types = set(common.get_attrtypes(self.project))
        # 0.) munge digit-grouping-results into results
        results = munge_digit_results(results, attr_types, self.project)
        # munge results -> results_foo
        results_foo = {} # {samplepath: {attrtype: (attrval, flip, imgorder)}}
        # 1.) First, handle non-digit patches
        for grouplabel, groups in results.iteritems():
            # groups := list of GroupClass objects
            if not groups:
                # Maybe never got assigned any samples.
                continue
            ad = {} # maps {str attrtype: str attrval}
            # Gather up all attrtype->attrval mappings into ad
            for attrtype in attr_types:
                attrval = common.get_propval(grouplabel, attrtype)
                if attrval:
                    ad[attrtype] = attrval
            if ad == {}:
                print "Uhoh, an attribute type was not found in the grouplabel:", grouplabel
                pdb.set_trace()
            assert ad != {}
            if common.is_digit_grouplabel(grouplabel, self.project):
                # Temporary hack for digit patches :\
                # For real, we need to do at least imgorder
                flip = 0
                imgorder = 0
            else:
                flip = common.get_propval(grouplabel, 'flip')
                imgorder = common.get_propval(grouplabel, 'imageorder')
            assert flip != None
            assert imgorder != None
            for group in groups:
                for (samplepath, rankedlist, patchpath) in group.elements:
                    for attrtype, attrval in ad.iteritems():
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
        sample_flips = {} # {str samplepath: [flip0, flip1]}
        if util.is_multipage(self.project):
            img2tmp = pickle.load(open(self.project.image_to_template, 'rb'))
        else:
            img2tmp = None
        # _rows is a list of rows, used in exportResults
        self._rows = []
        hosed_bals = []
        munged_patches = munge_patches(self.run_grouping.patches,
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
            templateid = determine_template(attrdict, munged_patches, self.project)
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
        correctedflips = fix_ballot_to_images(self.project, bal2tmp, sample_attrmap, self.run_grouping.patches, sample_flips)
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
        def munge_patches(patches, attrtypes, project):
            """ Converts {str templatepath: ((y1,y2,x1,x2),grouplabel,side, is_digitbased,is_tabulationonly}
            to: {str templatepath: list of ((y1,y2,x1,x2), attrtype, attrval, side, is_digitbased,is_tabulationonly}
            """
            result = {}
            digitsresult = {} # maps {str attrtype: ((y1,y2,x1,x2), side)}
            # patches won't have digit-based attributes
            for temppath, patchtriple in patches.iteritems():
                for (bb, grouplabel, side, is_digitbased, is_tabulationonly) in patchtriple:
                    for attrtype in attrtypes:
                        if common.get_propval(grouplabel, attrtype):
                            attrval = common.get_propval(grouplabel, attrtype)
                            result.setdefault(temppath, []).append((bb, attrtype, attrval, side, is_digitbased,is_tabulationonly))
            # Handle digit-based attributes
            for attrdict in pickle.load(open(project.ballot_attributesfile, 'rb')):
                w_img, h_img = project.imgsize
                if attrdict['is_digitbased']:
                    # TODO: digitbased assumes that each attr patch has only
                    # one attribute. Not sure if the non-digitbased
                    # correctly-handles multi-attribute pathces.
                    for attrtype in attrtypes:
                        if attrtype in attrdict['attrs']:
                            if attrtype not in digitsresult:
                                # Only add attrtype once into digitsresult
                                bb = [int(round(attrdict['y1']*h_img)),
                                      int(round(attrdict['y2']*h_img)),
                                      int(round(attrdict['x1']*w_img)),
                                      int(round(attrdict['x2']*w_img))]
                                digitsresult[attrtype] = (bb, attrdict['side'])
            return result, digitsresult
        munged,digitmunged = munge_patches(self.patches, all_attrtypes, self.project)
        groupImagesMAP(bal2imgs,
                       tpl2imgs,
                       munged,
                       self.project.extracted_precinct_dir, 
                       self.project.ballot_grouping_metadata, 
                       stopped,
                       deleteall=deleteall)
        if digitmunged:
            digitgroup_results = do_digitocr_patches(bal2imgs, digitmunged, self.project)
            outpath = os.path.join(self.project.projdir_path, 
                                   self.project.digitgroup_results)
            f = open(outpath, 'wb')
            pickle.dump(digitgroup_results, f)
            f.close()
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
        digitgroupresultsP = pathjoin(self.project.projdir_path,
                                      self.project.digitgroup_results)
        if os.path.exists(digitgroupresultsP):
            f = open(digitgroupresultsP, 'rb')
            # maps {str imgpath: list of (attrtype_i, ocr_str_i, meta_i)},
            # where meta_i is numDigits-tuples of the form:
            #     (y1,y2,x1,x2, digit_i, digitimgpath_i, score)
            digitgroup_results = pickle.load(f)
            f.close()
        else:
            digitgroup_results = {}

        grouping_results = {} # maps {str attrtype: {bestmatch: list of [ballotid, rankedlist, patchpath]}}
        digits_results = {} # maps {str digit: list of [ballotid, patchpath]}
        attr_types = common.get_attrtypes(self.project)
        def removedups_rankedlist(rlist):
            """ Remove duplicates from the input ranked list, starting
            from L-R. This is needed because Kais' NCC-OCR code can,
            for a given digit patch, signal that a given digit D
            occurs in the patch multiple times.
            """
            result = []
            for i, grouplabel in enumerate(rlist):
                if i >= len(rlist)-1:
                    break
                if grouplabel not in rlist[i+1:]:
                    result.append(grouplabel)
            return result
        def sanity_check_rankedlist(rlist):
            history = {}
            for grouplabel in rlist:
                if grouplabel in history:
                    print 'woah, this grouplabel was already here:', grouplabel
                    pdb.set_trace()
                    return False
                else:
                    history[grouplabel] = True
            return True
        # Munge the grouping results into grouping_results, digit_results
        if not util.is_multipage(self.project):
            for attr_type in attr_types:
                for ballotid in bal2imgs:
                    if common.is_digitbased(self.project, attr_type):
                        for (attrtype_i, ocr_str_i, meta_i) in digitgroup_results[ballotid]:
                            if attrtype_i == attr_type:
                                for (y1,y2,x1,x2, digit, digitpatchpath, score) in meta_i:
                                    digits_results.setdefault(digit, []).append((ballotid, digitpatchpath))
                                break
                        continue
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
                for ballotid, (frontpath, backpath) in bal2imgs.iteritems():
                    if common.is_digitbased(self.project, attr_type):
                        ''' this is all wrong... '''
                        attr_side = common.get_attr_prop(self.project, attr_type, 'side')
                        path = frontpath if attr_side == 'front' else backpath
                        for (attrtype_i, ocr_str_i, meta_i) in digitgroup_results[path]:
                            if attrtype_i == attr_type:
                                for (y1,y2,x1,x2,digit,digitpatchpath,score) in meta_i:
                                    digits_results.setdefault(digit, []).append((path, digitpatchpath))
                                break
                        continue
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
                    grouping_results.setdefault(attr_type, {}).setdefault(bestMatch, []).append((ballotid, attrs_list, patchpath))
        
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
        # Now, seed the digits stuff
        alldigits = digits_results.keys()
        for digit, lst in digits_results.iteritems():
            elements = []
            rankedlist = make_digits_rankedlist(digit, alldigits)
            for (ballotid, patchpath) in lst:
                elements.append((ballotid, rankedlist, patchpath))
            group = common.GroupClass(elements)
            groups.append(group)
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
        try:
            flipfront, flipback = correctedflips[sampleid]
        except Exception as e:
            print e
            pdb.set_trace()
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
    dict patches: {str temppath: list of ((y1,y2,x1,x2),attrtype,attrval,side, is_digitbased,is_tabulationonly)}
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
        img2bal = pickle.load(open(project.image_to_ballot, 'rb'))
        correctedflips = {}
        for ballotid, templateid in bal2tmp.iteritems():
            frontpath, backpath = b2imgs[ballotid]
            side0, side1 = None, None
            attr_tuples = []
            tmp_front, tmp_back = tmp2imgs[templateid]
            if tmp_front in patches:
                attr_tuples.extend(list(patches[tmp_front]))
            if tmp_back in patches:
                attr_tuples.extend(list(patches[tmp_back]))
            for (r, grouplabel, side, is_digitbased, is_tabulationonly) in attr_tuples:
                attrtype, attrval = common.get_attrpair_grouplabel(project, grouplabel)
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
                correctedflips[ballotid] = flip1, flip0
            b2imgs[ballotid] = front, back
        pickle.dump(b2imgs, open(project.ballot_to_images, 'wb'))
        return correctedflips

def determine_template(sample_attrs, template_attrs, project):
    """
    Given a sample image's attrtype->attrval mappings, return the
    template that has the same attrtype->attrval mapping.
    Also returns the side ordering of the sample (i.e. that '0' is 'front',
    and '1' is 'back').
    Input:
      dict sample_attrs: {str attrtype: (str attrval, int flip, int imageorder)}
      dict template_attrs: {str temppath: {str attrtype: str attrval, int side}}
      project
    """
    for temppath, temp_attrdict in template_attrs.iteritems():
        flag = True
        for attrtype, (temp_attrval, temp_side) in temp_attrdict.iteritems():
            if attrtype not in sample_attrs:
                # I.e. an attr on the back of a Template will not be found
                # on a sample image of a Front side.
                continue
            elif common.is_tabulationonly(project, attrtype):
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
            for (r, grouplabel, side, is_digitbased, is_tabulationonly) in tuples:
                attrtype, attrval = get_attrtypeval(grouplabel, attrtypes)
                result.setdefault(temppath, {})[attrtype] = (attrval, 'front')
    else:
        for temppath, tuples in patches.iteritems():
            for (r, grouplabel, side, is_digitbased, is_tabulationonly) in tuples:
                attrtype, attrval = get_attrtypeval(grouplabel, attrtypes)
                result.setdefault(img2tmp[temppath], {})[attrtype] = (attrval, side)

    return result

def munge_digit_results(results, all_attrtypes, project):
    """Given the results of overlay-verification, take all digit-based
    groups, and munge them back into the results (i.e. jump from
    'digits' to 'precinct'.
    Input:
        dict results: maps {grouplabel: list of GroupClasses}
        lst all_attrtypes: List of all attrtypes
    Output:
        dict that maps: {grouplabel: list of GroupClasses} but with
        all digitattributes inside of the result.
    """
    def is_digitbased_grouplabel(grouplabel):
        """Assumes a digit-based grouplabel has a k,v with the
        k being 'digit'. Lousy assumption.
        """
        return common.get_propval(grouplabel, 'digit') != None
    img2bal = pickle.load(open(project.image_to_ballot, 'rb'))
    digitattrs = [a for a in all_attrtypes if common.is_digitbased(project, a)]
    if not digitattrs:
        print "No digit-based attributes in this election."
        return results
    if len(digitattrs) != 1:
        print "Sorry, ack, OpenCount only supports one digit-based \
patch, sorry."
        assert False
    new_results = {} # maps {grouplabel: list of GroupClasses}
    patchlabels = {} # maps {str digitpatchpath: str digit}
    for grouplabel, groups in results.iteritems():
        if not groups or not is_digitbased_grouplabel(grouplabel):
            continue
        curdigit = common.get_propval(grouplabel, 'digit')
        for group in groups:
            for (samplepath, rankedlist, patchpath) in group.elements:
                patchlabels[patchpath] = curdigit
    # a dict mapping {str samplepath: [(attrtype_i, correct_digitlabel_i), ...]
    digit_labels = correct_digit_labels(project, patchlabels)
    samples_map = {} # maps {grouplabel: list of samplepaths}
    for samplepath, lst in digit_labels.iteritems():
        for (attrtype, digitlabel) in lst:
            grouplabel = common.make_grouplabel((attrtype, digitlabel))
            samples_map.setdefault(grouplabel, []).append(samplepath)
    for grouplabel, samplepaths in samples_map.iteritems():
        elements = []
        for samplepath in samplepaths:
            ballotid = img2bal[os.path.abspath(samplepath)]
            elements.append((ballotid, (grouplabel,), None))
        group = common.GroupClass(elements, no_overlays=True)
        new_results.setdefault(grouplabel, []).append(group)
    for grouplabel, groups in results.iteritems():
        if not is_digitbased_grouplabel(grouplabel):
            new_results.setdefault(grouplabel, []).extend(groups)
    return new_results

def correct_digit_labels(project, patchlabels):
    """Given the correct labelings for each voted digit patch, return
    the correct labeled numbers for each digitbased patch.
    Input:
        obj project
        dict patchlabels: maps {str digitpatch: str digit}
    Output:
        A dict mapping {str imgpath: [(attrtype_i, correct_digitlabel_i), ...]}
    """
    p = pathjoin(project.projdir_path,
                 project.digitgroup_results)
    f = open(p, 'rb')
    # maps {str votedpath: list of (attrtype, ocr_str, meta)} where
    # each meta is numDigits-tuples of:
    #    (y1,y2,x1,x2,digit, digitpatchpath, score)
    digitgroup_results = pickle.load(f)
    result = {}
    for imgpath, lst in digitgroup_results.iteritems():
        if common.is_quarantined(project, imgpath):
            continue
        for attrtype, ocr_str, meta in lst:
            correct_str = ''
            for i, (y1,y2,x1,x2,digit,digitpatchpath,score) in enumerate(meta):
                try:
                    correct_digit = patchlabels[digitpatchpath]
                except Exception as e:
                    print e
                    pdb.set_trace()
                correct_str += correct_digit
            result.setdefault(imgpath, []).append((attrtype, correct_str))
    return result

def make_digits_rankedlist(d, digits):
    #intuition = {'0': ('8', '9', ''),
    #             '1': '7',
    #             '2': '0',
    #             '3': '8',
    #             '4': '5',
    #             '5': '4',
    #             '6': 
    cpy = list(digits)[:]
    cpy.remove(d)
    cpy.insert(0, d)
    result = []
    for digit in cpy:
        grouplabel = common.make_grouplabel(('digit', digit))
        result.append(grouplabel)
    return result

def do_digitocr_patches(bal2imgs, digitattrs, project):
    """ For each digitbased attribute, run our NCC-OCR on the patch
    (using our digit exemplars).
    Input:
        dict bal2imgs
        dict digitattrs: maps {attrtype: ((y1,y2,x1,x2), side)}
        obj project
    Output:
        A dict that maps:
          {ballotid: ((attrtype_i, ocrresult_i, meta_i), ...)
        where meta_i is a tuple containing numDigits tuples:
          (y1_i,y2_i,x1_i,x2_i, str digit_i, str digitimgpath_i, score)
    """
    def make_digithashmap(project):
        digitmap = {} # maps {str digit: obj img}
        digit_exemplarsdir = os.path.join(project.projdir_path,
                                          project.digit_exemplars_outdir)
        digitdirs = os.listdir(digit_exemplarsdir)
        for digitdir in digitdirs:
            # Assumes this has directories of the form:
            #    0_examples/*.png
            #    1_examples/*.png
            #    ...
            fullpath = os.path.join(digit_exemplarsdir, digitdir)
            digit = digitdir.split('_')[0]
            for dirpath, dirnames, filenames in os.walk(fullpath):
                for imgname in [f for f in filenames if util_gui.is_image_ext(f)]:
                    imgpath = os.path.join(dirpath, imgname)
                    img = sh.standardImread(imgpath, flatten=True)
                    digitmap[digit] = img
        return digitmap
    def all_ballotimgs_gen(bal2imgs, side):
        """ Generate all ballot images for either side 0 or 1. """
        for ballotid, path in bal2imgs.iteritems():
            if side == 0:
                yield path[0]
            else:
                yield path[1]
        raise StopIteration
    result = {}
    digit_exs = make_digithashmap(project)
    numdigitsmap = pickle.load(open(os.path.join(project.projdir_path, 
                                                 project.num_digitsmap),
                                    'rb'))
    voteddigits_dir = os.path.join(project.projdir_path,
                                     project.voteddigits_dir)
    ctr = 0
    for digitattr, ((y1,y2,x1,x2),side) in digitattrs.iteritems():
        num_digits = numdigitsmap[digitattr]
        # add some border, for good measure
        w, h = abs(x1-x2), abs(y1-y2)
        c = 0.0
        bb = [max(0, y1-int(round(h*c))),
              y2+int(round(h*c)),
              max(0, x1-int(round(w*c))),
              x2+int(round(w*c))]
        results_side0 = sh.digitParse(digit_exs,
                                      all_ballotimgs_gen(bal2imgs, 0),
                                      bb,
                                      num_digits)
        '''
        if util.is_multipage(project):
            results_side1 = sh.digitParse(digit_exs,
                                          all_ballotimgs_gen(bal2imgs, 1),
                                          bb,
                                          num_digits)
        '''
        digitparse_results = results_side0
        for (imgpath, ocr_str, meta) in digitparse_results:
            meta_out = []
            for (y1,y2,x1,x2, digit, digitimg, score) in meta:
                rootdir = os.path.join(voteddigits_dir, digit)
                util.create_dirs(rootdir)
                outpath = os.path.join(rootdir, '{0}_votedextract.png'.format(ctr))
                scipy.misc.imsave(outpath, digitimg)
                meta_out.append((y1,y2,x1,x2, digit, outpath, score))
                ctr += 1
            result.setdefault(imgpath, []).append((digitattr, ocr_str, meta_out))
    return result
    
    
    
