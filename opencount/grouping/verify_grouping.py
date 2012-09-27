import sys, csv, copy, pdb, os, re, shutil, math
import threading, time
import timeit
sys.path.append('../')

from util import MyGauge
from specify_voting_targets import util_gui as util_gui
from specify_voting_targets import imageviewer as imageviewer
from specify_voting_targets.util_gui import *
from verify_overlays import VerifyPanel
import label_attributes, util, common, digit_group, cust_attrs

from common import TABULATION_ONLY_ID, DIGIT_BASED_ID

from pixel_reg.imagesAlign import *
import pixel_reg.shared as sh
from pixel_reg.doGrouping import  groupImagesMAP, encodepath

####
## Import 3rd party libraries
####

import wx
import wx.animate
import wx.lib.scrolledpanel as scrolled
import Image
import cv2
import numpy as np
import scipy
from scipy import misc
    
import wx.lib.inspection
from wx.lib.pubsub import Publisher

# Set by MainFrame
TIMER = None

"""
Output Files:
- <projdir>/grouping_results.csv
    Contains the grouping output for each voted ballot image. This is
    a csv with the following columns:
      samplepath,templatepath,<attrtype_i>,...,flipped_front,flipped_back
    This is outputted as soon as the user finishes verifying the overlays.
- <projdir>/digitgroup_results.p
    Contains the results of running digit-based OCR (i.e. the return 
    value of do_digitocr_patches()). This is a dict that maps:
      {ballotid: ((attrtype_i, ocrstr_i, meta_i, isflip_i, side_i), ...)
    Where meta_i is a tuple containing numDigits-number of tuples:
      (y1_i,y2_i,x1_i,x2_i, str digit_i, str digitimgpath_i, float score)
    This is created as soon as the digitocr computation is completed.
- <projdir>/ballot_to_page.p
    This is a dictionary that maps voted imgpath to its page, i.e:
      0 := front/side0
      1 := back/side1
      2 := side2
      ...
    Created when grouping computation is complete.
"""

class GroupingMasterPanel(wx.Panel):
    """
    Panel that contains both RunGroupingPanel and VerifyGroupingPanel.
    """
    def __init__(self, parent, *args, **kwargs):
        wx.Panel.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.project = None

        self.grouplabel_record = None

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
        self.grouplabel_record = common.load_grouplabel_record(self.project)
        result = sanitycheck_blankballots(self.project)
        if result:
            print "Uhoh, blank ballots failed our sanity check."
            dlg = wx.MessageDialog(self, message="OpenCount detected that \
there exists more than one blank ballot for a given set of attribute \
values. We recommend that the user re-consider the current attribute \
patch selections. If you continue to run grouping, target extraction \
will do strange things, and you will probably get poor results.", style=wx.OK)
            self.Disable()
            dlg.ShowModal()
            self.Enable()
            for attrpairs, badgroup in result.iteritems():
                print "attrs:", attrpairs
                for bpath in badgroup:
                    print "    ", bpath
                print
            
        self.run_grouping.Show()
        self.run_grouping.start()

    def grouping_done(self, groups, do_replacedigits=False):
        """
        If groups is not None, then it means we ran grouping
        computation.
        If groups is None, then we skipped it, so we should
        load in the previous verify-grouping state.
        If do_replacedigits is True, then we only ran digitgrouping, so,
        there must be verifyoverlay state already, and, we just have to
        replace all GroupClass 'digit' things with stuff in digitgroups.
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
        self.project.addCloseEvent(self.verify_grouping.dump_state)
        if groups:
            exemplar_paths = get_exemplar_paths()
            self.verify_grouping.start(groups, exemplar_paths, self.project, ondone=self.verifying_done)
            self.verify_grouping.SendSizeEvent()
            self.SendSizeEvent()
            self.Refresh()
            self.Fit()
        else:
            verifyoverlay_stateP = pathjoin(self.project.projdir_path,
                                            'verifygroupstate.p')
            if not os.path.exists(verifyoverlay_stateP):
                # If grouping computation completes, but the VerifyOverlay
                # UI Crashes, and doesn't save its statefile, then
                # regenerate the state required for the UI.
                dlg = wx.MessageDialog(self, message="Note: No \
VerifyOverlay state file detected ({0}) - starting from the \
beginning..".format(verifyoverlay_stateP),
                                       style=wx.OK)
                self.Disable()
                dlg.ShowModal()
                self.Enable()

                groups = to_groupclasses(self.project)
                digitgroup_results = digit_group.load_digitgroup_results(self.project)
                groups.extend(digit_group.to_groupclasses_digits(self.project, digitgroup_results))
                exemplar_paths = get_exemplar_paths()
                self.verify_grouping.start(groups, exemplar_paths, self.project, ondone=self.verifying_done)
                self.verify_grouping.SendSizeEvent()
                self.SendSizeEvent()
                self.Refresh()
                self.Fit()
                return
            dlg = wx.MessageDialog(self, message="VerifyOverlay state \
file detected ({0}) - loading in state.".format(verifyoverlay_stateP),
                                   style=wx.OK)
            self.Disable()
            dlg.ShowModal()
            self.Enable()
                
            #self.verify_grouping.start(groups, patches, exemplar_paths)
            if do_replacedigits:
                digitgroup_results = digit_group.load_digitgroup_results(self.project)
                digitgroups = digit_group.to_groupclasses_digits(self.project, digitgroup_results)
                self.verify_grouping.load_state(replacedigits=True, digitgroups=digitgroups)
            else:
                self.verify_grouping.load_state()
            exemplar_paths = get_exemplar_paths()
            self.verify_grouping.project = self.project
            self.verify_grouping.ondone = self.verifying_done
            self.verify_grouping.load_exemplar_attrpatches(exemplar_paths)
            self.verify_grouping.start_verifygrouping()
            self.SendSizeEvent()
            self.Refresh()
            self.Fit()

    def verifying_done(self, results, grouplabel_record):
        """
        Called when the user is finished with grouping verification.
        results is a dict of the form:
            {gl_idx: list of GroupClasses}
        """
        self.project.removeCloseEvent(self.verify_grouping.dump_state)
        self.verify_grouping.dump_state()
        attr_types = set(common.get_attrtypes(self.project))
        # 0.) munge digit-grouping-results into results, since digitattrs
        #     are still in 'digit' form.
        results = munge_digit_results(results, attr_types, self.project, grouplabel_record)
        # munge results -> results_foo
        # results_foo has all attribute vals for all voted ballots
        results_foo = {} # {samplepath: {attrtype: (attrval, flip, imgorder)}}

        for gl_idx, groups in results.iteritems():
            # groups := list of GroupClass objects
            if type(gl_idx) != int:
                print "Uhoh, expected gl_idx to be int, not {0}".format(type(gl_idx))
                pdb.set_trace()
            grouplabel = grouplabel_record[gl_idx]
            if not groups:
                # Maybe never got assigned any samples.
                continue
            ad = {} # maps {str attrtype: str attrval}
            # Gather up all attrtype->attrval mappings into ad
            for attrtype in attr_types:
                attrval = common.get_propval(gl_idx, attrtype, self.project, grouplabel_record)
                if attrval:
                    ad[attrtype] = attrval
            if ad == {}:
                print "Uhoh, an attribute type was not found in the grouplabel:", grouplabel
                pdb.set_trace()
            assert ad != {}
            if common.is_digit_grouplabel(gl_idx, self.project, grouplabel_record):
                # Temporary hack for digit patches :\
                # Because the flip/imgorder info got thrown out in
                # on_grouping_done, these grouplabels won't have this
                # info. But, rest assured, it has been taken care of.
                flip = 0
                imgorder = 0
            else:
                flip = common.get_propval(gl_idx, 'flip', self.project, grouplabel_record)
                imgorder = common.get_propval(gl_idx, 'imageorder', self.project, grouplabel_record)
            assert flip != None
            assert imgorder != None
            for group in groups:
                for (samplepath, rankedlist, patchpath) in group.elements:
                    if common.is_quarantined(self.project, samplepath):
                        continue
                    for attrtype, attrval in ad.iteritems():
                        results_foo.setdefault(samplepath, {})[attrtype] = (attrval,
                                                                            flip,
                                                                            imgorder)
        # Finally, add CustomAttributes to results_foo
        results_foo = add_customattrs_info_voted(self.project, results_foo)
            
        fields = ('samplepath','templatepath')
        fields += tuple(sorted(tuple(common.get_attrtypes_all(self.project))))
        fields += ('flipped_front', 'flipped_back')
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
                                       img2tmp,
                                       grouplabel_record)
        # munged_patches doesn't know anything about digitattrs, so add
        # this info in.
        if common.get_digitbased_attrs(self.project):
            munged_patches = add_digitattr_info(self.project, munged_patches)
        munged_patches = add_customattrs_info_blanks(self.project, munged_patches)
        # Finally, for each voted ballot find its corresponding blank
        # ballot.
        for samplepath, attrdict in results_foo.items():
            row = {}
            row['samplepath'] = samplepath
            for attrtype, (attrval, flip, imageorder) in attrdict.items():
                row[attrtype] = attrval
                sample_flips.setdefault(samplepath, [None, None])[imageorder] = flip
                sample_attrmap.setdefault(samplepath, {})[attrtype] = imageorder
            templateid = determine_template(attrdict, munged_patches, samplepath, self.project)
            if type(templateid) == tuple:
                status, data = templateid
                hosed_bals.append((samplepath, attrdict, data))
                continue
            row['templatepath'] = templateid
            bal2tmp[samplepath] = templateid
            self._rows.append(row)
        if hosed_bals:
            msg = """Warning: There were {0} voted ballots for which \
OpenCount wasn't able to determine the corresponding blank ballot. \
OpenCount has quarantined these voted ballots - you will have the \
opportunity to manually group these ballots.\n""".format(len(hosed_bals))
            qfile = open(self.project.quarantined, 'a')
            for (samplepath, attrdict, munged_patches) in hosed_bals:
                print >>qfile, os.path.abspath(samplepath)
            qfile.close()
            HOSED_FILENAME = os.path.join(self.project.projdir_path, 'hosed_votedballots.log')
            msg2 = "\n(This information has been dumped to '{0}'".format(HOSED_FILENAME)
            msg = msg + msg2
            dlg = wx.MessageDialog(self, message=msg, style=wx.OK)
            log_hosed_ballots(hosed_bals, HOSED_FILENAME)
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
        attr_types = common.get_attrtypes_all(self.project)
        fields = ('samplepath','templatepath') + tuple(sorted(tuple(attr_types))) + ('flipped_front', 'flipped_back')
        csvfilepath = self.project.grouping_results
        csvfile = open(csvfilepath, 'wb')
        dictwriter = csv.DictWriter(csvfile, fieldnames=fields)
        try:
            dictwriter.writeheader()
        except AttributeError:
            util_gui._dictwriter_writeheader(csvfile, fields)
        dictwriter.writerows(self._rows)
        csvfile.close()


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

        # self.patches is {str temppath: [((y1,y2,x1,x2),grouplabel_i,side_i,is_digitbased_i,is_tabulationonly_i),...]}
        self.patches = None

        self.sizer = wx.BoxSizer(wx.VERTICAL)

        self.run_button = wx.Button(self, label="Run Grouping")
        self.run_button.Bind(wx.EVT_BUTTON, self.onButton_run)
        self.rerun_button = wx.Button(self, label="Re-run Grouping")
        self.rerun_button.Bind(wx.EVT_BUTTON, self.onButton_rerun)
        self.run_digitgroup_button = wx.Button(self, label="Re-run Digit Grouping Only.")
        self.run_digitgroup_button.Bind(wx.EVT_BUTTON, self.onButton_rundigitgroup)
        self.continue_button = wx.Button(self, label="Continue Correcting Grouping")
        self.continue_button.Bind(wx.EVT_BUTTON, self.onButton_continue)
        
        self.sizer.AddMany(([self.run_button, self.rerun_button, 
                             self.continue_button, self.run_digitgroup_button]))

        self.SetSizer(self.sizer)
        
        Publisher().subscribe(self._pubsub_project, "broadcast.project")

    def start(self):
        assert self.project
        if self.is_grouping_done():
            self.run_button.Hide()
        else:
            self.rerun_button.Hide()
            self.run_digitgroup_button.Hide()
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
        # Remember to remove state files
        projdir = self.project.projdir_path
        util_gui.remove_files(self.project.grouping_results,
                              pathjoin(projdir,
                                       self.project.ballot_to_page),
                              pathjoin(projdir,
                                       'verifygroupstate.p'),
                              pathjoin(projdir,
                                       self.project.digitgroup_results),
                              pathjoin(projdir,
                                       self.project.digitmatch_info),
                              pathjoin(projdir,
                                       self.project.rejected_hashes),
                              pathjoin(projdir,
                                       self.project.accepted_hashes))
        voteddigits_dir = pathjoin(projdir, 'voteddigits_dir')
        if os.path.exists(voteddigits_dir):
            shutil.rmtree(pathjoin(projdir,
                                   'voteddigits_dir'))
        self.start_grouping()

    def groupBallotsProcess(self, stopped, deleteall):
        """ Performs grouping of both img-based and digit-based
        attributes.
        """
        num = 0
        for dirpath, dirnames, filenames in os.walk(self.project.samplesdir):
            for f in filenames:
                if not is_image_ext(f):
                    continue
                num += 1
        # QUESTION: What is kind? The number of img-based attributes?
        #           Is this change correct?
        if not common.exists_imgattrs(self.project):
            kind = 0
        else:
            kind = len(self.patches.items()[0])

        wx.CallAfter(Publisher().sendMessage, "signals.MyGauge.nextjob", num*kind)
        bal2imgs=pickle.load(open(self.project.ballot_to_images,'rb'))
        tpl2imgs=pickle.load(open(self.project.template_to_images,'rb'))

        # munge patches into format that groupImagesMAP wants
        all_attrtypes = common.get_attrtypes(self.project)
        munged,digitmunged = munge_patches_grouping(self.patches, all_attrtypes, self.project)
        print "== calling groupImagesMAP..."
        groupImagesMAP(bal2imgs,
                       tpl2imgs,
                       munged,
                       self.project.extracted_precinct_dir, 
                       self.project.ballot_grouping_metadata, 
                       stopped,
                       self.project,
                       deleteall=deleteall)
        print "== finished groupImagesMAP"
        if digitmunged:
            self.run_digit_grouping(digitmunged=digitmunged)
        wx.CallAfter(Publisher().sendMessage, "signals.MyGauge.done")

    def onButton_rundigitgroup(self, evt):
        dlg = wx.MessageDialog(self, message="This will re-run Digit \
Grouping completely, which means you will lose all progress on \
Digit-Related Attributes. However, you will keep your progress on \
other Attributes (i.e. non Digit-related Attributes).\n\n\
Are you sure you want to proceed?",
                               style=wx.YES_NO | wx.NO_DEFAULT)
        status = dlg.ShowModal()
        if status == wx.ID_YES:
            print "YES"
            self.run_digit_grouping()
            self.parent.grouping_done(None, do_replacedigits=True)
        else:
            print "NO"

    def run_digit_grouping(self, digitmunged=None):
        # Remember to remove all files
        rdir = self.project.projdir_path
        p = self.project
        util_gui.remove_files(pathjoin(rdir,
                                       p.accepted_hashes),
                              pathjoin(rdir, p.rejected_hashes),
                              pathjoin(rdir, p.digitgroup_results),
                              pathjoin(rdir, p.digitmatch_info))
        all_attrtypes = common.get_attrtypes(self.project)
        if digitmunged == None:
            _, digitmunged = munge_patches_grouping(self.patches, all_attrtypes, self.project)
        bal2imgs = pickle.load(open(self.project.ballot_to_images, 'rb'))
        print "== Performing DigitOCR..."
        t = time.time()
        digitgroup_results, digitmatch_info = digit_group.do_digitocr_patches(bal2imgs, digitmunged, self.project)
        dur = time.time() - t
        print "== Finished DigitOCR ({0} s).".format(dur)
        digit_group.save_digitgroup_results(self.project, digitgroup_results)
        digit_group.save_digitmatch_info(self.project, digitmatch_info)

    def start_grouping(self):
        """ Creates the separate Thread that spawns the grouping
        subprocesses. This part will group both img-based and digit-based
        attributes.
        The separate process also performs bkgd-clustering, which
        produces multipleexemplars, for blank ballots where the
        background varies.
        """
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
        digitgroupresultsP = pathjoin(self.project.projdir_path,
                                      self.project.digitgroup_results)
        if os.path.exists(digitgroupresultsP):
            f = open(digitgroupresultsP, 'rb')
            # maps {str ballotid: list of (attrtype_i, ocr_str_i, meta_i, isflip_i, side_i)},
            # where meta_i is numDigits-tuples of the form:
            #     (y1,y2,x1,x2, digit_i, digitimgpath_i, score)
            digitgroup_results = pickle.load(f)
            f.close()
        else:
            digitgroup_results = {}
        
        groups = to_groupclasses(self.project)
        groups.extend(digit_group.to_groupclasses_digits(self.project, digitgroup_results))
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
            # Remember to remove state files
            projdir = self.project.projdir_path
            util_gui.remove_files(self.project.grouping_results,
                                  pathjoin(projdir,
                                           self.project.ballot_to_page),
                                  pathjoin(projdir,
                                           'verifygroupstate.p'),
                                  pathjoin(projdir,
                                           self.project.digitgroup_results),
                                  pathjoin(projdir,
                                           self.project.digitmatch_info),
                                  pathjoin(projdir,
                                           self.project.rejected_hashes),
                                  pathjoin(projdir,
                                           self.project.accepted_hashes))
            voteddigits_dir = pathjoin(projdir, 'voteddigits_dir')
            if os.path.exists(voteddigits_dir):
                shutil.rmtree(pathjoin(projdir,
                                       'voteddigits_dir'))
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
        Return all output directories for this election. Excludes
        digit-based and Custom attributes, since they don't get
        exported in this manner.
        Outputs:
            (<extractedpatches dirs>, <balgroupmetainit dirs>, <balgroupmeta dirs>)
        """
        # Weird thing: ballot_attributesfile includes img-based and
        # digit-based attributes (but not custom-attrs). Filter out the
        # digit-based attrs.
        attrs = pickle.load(open(self.project.ballot_attributesfile, 'rb'))
        extractedpatches_dirs = []
        balgroupmeta_dirs = []
        for attr in attrs:
            assert issubclass(type(attr['is_digitbased']), bool)
            if attr['is_digitbased'] == True:
                continue
            attrtype = common.get_attrtype_str(attr['attrs'])
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
        # The following block only checks img-based attributes
        if common.exists_imgattrs(self.project):
            extractedpatches_dirs, balgroupmeta_dirs = self._get_outputdirs()
            for dir in extractedpatches_dirs:
                if not util.contains_image(dir):
                    print '{0} has no images.'.format(dir)
                    return False
            for dir in balgroupmeta_dirs:
                if not util.contains_file(dir):
                    print '{0} has no files'.format(dir)
                    return False
        # The following block checks digit-based attributes
        if common.exists_digitattrs(self.project):
            digitgroup_resultsP = pathjoin(self.project.projdir_path,
                                           self.project.digitgroup_results)
            if not os.path.exists(digitgroup_resultsP):
                print "Digit Grouping hasn't been done yet:".format(digitgroup_resultsP)
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
    Creates the 'ballot_to_page' dictionary, which maps voted imgpath
    to which 'side' it is on.
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
        img2bal = pickle.load(open(pathjoin(project.image_to_ballot), 'rb'))
        bal2pageP = pathjoin(project.projdir_path, project.ballot_to_page)
        bal2page = {}
        for imgpath in img2bal:
            bal2page[imgpath] = 0
        pickle.dump(bal2page, open(bal2pageP, 'wb'))
        return sample_flips
    else:
        b2imgs = pickle.load(open(project.ballot_to_images, 'rb'))
        tmp2imgs = pickle.load(open(project.template_to_images, 'rb'))
        img2bal = pickle.load(open(project.image_to_ballot, 'rb'))
        gl_record = common.load_grouplabel_record(project)
        correctedflips = {}
        bal2page = {}
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
                gl_idx = gl_record.index(grouplabel)
                attrtype, attrval = common.get_attrpair_grouplabel(project, gl_idx, gl_record)
                try:
                    imageorder = sample_attrmap[ballotid][attrtype]
                except Exception as e:
                    print e
                    pdb.set_trace()
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
            bal2page[front] = 0
            bal2page[back] = 1    # TODO: Account for more-than-2 pages
            #b2imgs[ballotid] = front, back
        #pickle.dump(b2imgs, open(project.ballot_to_images, 'wb'))
        pickle.dump(bal2page, open(pathjoin(project.projdir_path,
                                            project.ballot_to_page), 'wb'))
        return correctedflips

def add_digitattr_info(proj, munged_patches):
    """ Aux function to add DigitAttrVals for blank ballots to the input
    munged_patches.
    Input:
        obj proj:
        dict munged_patches: maps {str temppath: {str attrtype: (str attrval, int side)}}
    Output:
        An updated munged_patches.
    """
    attrs = pickle.load(open(proj.ballot_attributesfile, 'rb'))
    # a dict {temppath: {digitattrtype: (digitval, bb, int side)}}
    digitattrvals_blanks = pickle.load(open(pathjoin(proj.projdir_path,
                                                     proj.digitattrvals_blanks),
                                            'rb'))
    img2tmp = pickle.load(open(proj.image_to_template, 'rb'))
    digitattrs = common.get_digitbased_attrs(proj)
    for tmpimgpath, digitvals in digitattrvals_blanks.iteritems():
        tmpid = img2tmp[tmpimgpath]
        for digitattrtype, (digitval, bb, side) in digitvals.iteritems():
            munged_patches.setdefault(tmpid, {})[digitattrtype] = (digitval, side)
    return munged_patches

def add_customattrs_info_blanks(proj, munged_patches):
    """ Aux function that adds CustomAttributes for blank ballots to
    the input munged_patches.
    Input:
        obj proj:
        dict munged_patches: maps {str temppath: {str attrtype: (str attrval, int side)}}
    Output:
        The (mutated) munged_patches.
    """
    custom_attrs = cust_attrs.load_custom_attrs(proj)
    if custom_attrs == None:
        return munged_patches
    for temppath, attrdict in munged_patches.iteritems():
        for cattr in custom_attrs:
            if cattr.is_votedonly:
                continue
            elif cattr.mode == cust_attrs.CustomAttribute.M_SPREADSHEET:
                (inval, side) = attrdict[cattr.attrin]
                attrval = cust_attrs.custattr_map_inval_ss(proj,
                                                           cattr.attrname,
                                                           inval)
                attrdict[cattr.attrname] = (attrval, 0) # side is irrelevant
            elif cattr.mode == cust_attrs.CustomAttribute.M_FILENAME:
                temp_filename = os.path.split(temppath)[1]
                attrval = cust_attrs.custattr_apply_filename(cattr, temp_filename)
                attrdict[cattr.attrname] = (attrval, 0) # side is irrelevant
    return munged_patches

def add_customattrs_info_voted(proj, results_foo):
    """ Adds CustomAttribute information for all voted ballots to
    results_foo. Mutates input results_foo.
    Input:
        obj proj:
        dict results_foo: Maps {str samplepath: {attrtype: (attrval, flip, imgorder)}}
    Output:
        The (mutated) results_foo.
    """
    # A list of CustomAttribute instances
    custom_attrs = cust_attrs.load_custom_attrs(proj)
    if custom_attrs == None:
        return results_foo
    for samplepath, attrdict in results_foo.iteritems():
        for cattr in custom_attrs:
            if cattr.mode == cust_attrs.CustomAttribute.M_SPREADSHEET:
                (inval, flip, imgorder) = attrdict[cattr.attrin]
                attrval = cust_attrs.custattr_map_inval_ss(proj,
                                                           cattr.attrname,
                                                           inval)
                attrdict[cattr.attrname] = (attrval, 0, 0) # flip,imgorder irrelevant
            elif cattr.mode == cust_attrs.CustomAttribute.M_FILENAME:
                filename = os.path.split(samplepath)[1]
                attrval = cust_attrs.custattr_apply_filename(cattr, filename)
                attrdict[cattr.attrname] = (attrval, 0, 0) # flip,imgorder irrelevant
    return results_foo

def munge_patches_grouping(patches, attrtypes, project):
    """ Converts {str templatepath: ((y1,y2,x1,x2),grouplabel,side, is_digitbased,is_tabulationonly}
    to: {str templatepath: list of ((y1,y2,x1,x2), attrtype, attrval, side, is_digitbased,is_tabulationonly}
    """
    result = {}
    digitsresult = {} # maps {str attrtype: ((y1,y2,x1,x2), side)}
    gl_record = common.load_grouplabel_record(project)
    # patches won't have digit-based attributes
    for temppath, patchtriple in patches.iteritems():
        for (bb, grouplabel, side, is_digitbased, is_tabulationonly) in patchtriple:
            gl_idx = gl_record.index(grouplabel)
            if grouplabel != gl_record[gl_idx]:
                print "Uhoh, grouplabels weren't consistent w.r.t gl_record."
                pdb.set_trace()
            assert grouplabel == gl_record[gl_idx]
            for attrtype in attrtypes:
                if common.get_propval(gl_idx, attrtype, project):
                    attrval = common.get_propval(gl_idx, attrtype, project)
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


def determine_template(sample_attrs, template_attrs, samplepath, project):
    """
    Given a sample image's attrtype->attrval mappings, return the
    template that has the same attrtype->attrval mapping.
    Input:
      dict sample_attrs: {str attrtype: (str attrval, int flip, int imageorder)}
      dict template_attrs: {str temppath: {str attrtype: (str attrval, int side)}}
      str samplepath: Imagepath to the sample ballot in question.
      obj project: 
    Output:
      Path of the associated template, if it's successful. If it isn't,
      then if returns (0, None) if there are no possible matches.
      It returns (<N>, ((str blankpath_i, dict attrs_i), ..)) if there
      are N possible choices.
    """
    # 1.) First, handle 'standard' img-based attributes
    possibles = {}
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
            #return temppath
            possibles[temppath] = temp_attrdict
    if len(possibles) > 1:
        # TODO: Instead of unconditionally-quarantining the voted
        # ballot, instead, do this:
        #    If every template T in possibles contains the same contests,
        #    then arbitrarily return a template in T - this will be fine,
        #    assuming:
        #          a.) The ballot attributes uniquely determine ballot style
        #          b.) Contest ordering doesn't change
        #    Otherwise, we have a problem - quarantine the ballot, and alos
        #    warn the user that the current set of attributes is probably not
        #    'good enough'.
        #    
        if common.is_blankballot_contests_eq(*possibles.keys()):
            print "There were {0} possible blank ballots, but *phew*, \
they're all equivalent.".format(len(possibles))
            return possibles.keys()[0]
        else:
            print "== Error, more than one possible blank ballot: {0} possibles.".format(len(possibles))
            print "   We're hosed, so OpenCount will quarantine this voted ballot."
            print "   Perhaps the current set of Ballot Attributes don't"
            print "   uniquely specify a blank ballot?"
            print "   ", samplepath
            # TODO: Temporary hack, just arbitrarily return the first
            #       blank ballot.
            print "   Choosing the first blank ballot..."
            return possibles.keys()[0]
            #return (len(possibles), [(bpath, attrs) for bpath,attrs in possibles.items()])
    if len(possibles) == 0:
        print "== Error, determine_template couldn't find a blank ballot with a matching set"
        print "   of attributes. We're hosed.  Quarantining this voted ballot."
        print "  ", samplepath
        print "== To proceed, type in 'c', and press ENTER."
        return (0, None)
    assert len(possibles) == 1
    return possibles.keys()[0]

def log_hosed_ballots(hosed_ballots, outpath):
    """ Outputs information about voted ballots that didn't have a
    corresponding blank ballot to an output file.
    Input:
        list hosed_ballots: ((samplepath_i, dict attrs_i, data_i), ...)
    """
    f = open(outpath, 'w')
    for i, (samplepath, attrs, data) in enumerate(hosed_ballots):
        print >>f, "Voted Ballot {0}: {1}".format(i, samplepath)
        print >>f, "    Attributes:"
        for attrtype, (attrval, flip, imgorder) in attrs.iteritems():
            print >>f, "        {0}: {1}, flip: {2}, imgorder: {3}".format(attrtype, attrval,
                                                                           flip, imgorder)
        if data == None:
            print >>f, "    Blank Ballot(s): {0} found.".format(0)
        else:
            print >>f, "    Blank Ballot(s): {0} found.".format(len(data))
            for (blankpath, blankattrs) in data:
                print >>f, "        Path: {0}".format(blankpath)
                print >>f, "        Attributes:"
                for attrtype, (attrval, flip) in blankattrs.iteritems():
                    print >>f, "            {0}: {1}, flip: {2}".format(attrtype,
                                                                        attrval, flip)
        print >>f, ""
    f.close()

def munge_patches(patches, project, is_multipage=False, img2tmp=None, gl_record=None):
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
    def get_attrtypeval(grouplabel,attrtypes, gl_record):
        gl_idx = gl_record.index(grouplabel)
        v = None
        for attrtype in attrtypes:
            v = common.get_propval(gl_idx, attrtype, project, gl_record)
            if v:
                break
        if not v:
            print "Uh oh, v wasn't found in this grouplabel:", common.str_grouplabel(gl_idx, project)
            pdb.set_trace()
        assert v
        return attrtype, v
    if gl_record == None:
        gl_record = common.load_grouplabel_record(project)
    result = {}
    attrtypes = common.get_attrtypes(project)
    if not is_multipage:
        for temppath, tuples in patches.iteritems():
            for (r, grouplabel, side, is_digitbased, is_tabulationonly) in tuples:
                attrtype, attrval = get_attrtypeval(grouplabel, attrtypes, gl_record)
                result.setdefault(temppath, {})[attrtype] = (attrval, 'front')
    else:
        for temppath, tuples in patches.iteritems():
            for (r, grouplabel, side, is_digitbased, is_tabulationonly) in tuples:
                attrtype, attrval = get_attrtypeval(grouplabel, attrtypes, gl_record)
                result.setdefault(img2tmp[temppath], {})[attrtype] = (attrval, side)

    return result

def munge_digit_results(results, all_attrtypes, project, gl_record):
    """Given the results of overlay-verification, take all digit-based
    groups, and munge them back into the results (i.e. jump from
    'digits' to 'precinct'.
    Input:
        dict results: maps {gl_idx: list of GroupClasses}.
        lst all_attrtypes: List of all attrtypes
    Output:
        dict that maps: {gl_idx: list of GroupClasses} but with
        all digitattributes inside of the result.
    """
    def is_digitbased_grouplabel(gl_idx):
        """Assumes a digit-based grouplabel has a k,v with the
        k being 'digit'. Lousy assumption.
        TODO: This 'digit' kv-pair assumption restricts this framework
        to only allow one digit-based attribute at a time. We should
        ideally be able to handle any number of digit-based attributes.
        """
        return common.get_propval(gl_idx, 'digit', project) != None
    img2bal = pickle.load(open(project.image_to_ballot, 'rb'))
    digitattrs = [a for a in all_attrtypes if common.is_digitbased(project, a)]
    if not digitattrs:
        print "No digit-based attributes in this election."
        return results
    if len(digitattrs) != 1:
        print "Sorry, ack, OpenCount only supports one digit-based \
patch, sorry."
        assert False
    new_results = {} # maps {gl_idx: list of GroupClasses}
    patchlabels = {} # maps {str digitpatchpath: str digit}
    for gl_idx, groups in results.iteritems():
        if type(gl_idx) == frozenset:
            pdb.set_trace()
        grouplabel = gl_record[gl_idx]
        if not groups or not is_digitbased_grouplabel(gl_idx):
            continue
        curdigit = common.get_propval(gl_idx, 'digit', project, gl_record)
        for group in groups:
            for (samplepath, rankedlist, patchpath) in group.elements:
                patchlabels[patchpath] = curdigit
    # a dict mapping {str samplepath: [(attrtype_i, correct_digitlabel_i), ...]
    digit_labels = correct_digit_labels(project, patchlabels)
    samples_map = {} # maps {gl_idx: list of samplepaths}
    did_change_glrecord = False
    for samplepath, lst in digit_labels.iteritems():
        for (attrtype, digitlabel) in lst:
            # We fake imageorder/flip, because it's been accounted for
            # in an earlier part of the pipeline.
            grouplabel = common.make_grouplabel((attrtype, digitlabel), ('imageorder', 0), ('flip', 0))
            try:
                gl_idx = gl_record.index(grouplabel)
            except:
                gl_idx = len(gl_record)
                gl_record.append(grouplabel)
                did_change_glrecord = True
            samples_map.setdefault(gl_idx, []).append(samplepath)
    if did_change_glrecord:
        common.save_grouplabel_record(project, gl_record)
    for gl_idx, samplepaths in samples_map.iteritems():
        if type(gl_idx) == frozenset:
            print "Ah."
            pdb.set_trace()
        elements = []
        for samplepath in samplepaths:
            ballotid = img2bal[os.path.abspath(samplepath)]
            elements.append((ballotid, (gl_idx,), None))
        group = common.GroupClass(elements, no_overlays=True)
        new_results.setdefault(gl_idx, []).append(group)
    for gl_idx, groups in results.iteritems():
        if type(gl_idx) == frozenset:
            pdb.set_trace()
        if not is_digitbased_grouplabel(gl_idx):
            new_results.setdefault(gl_idx, []).extend(groups)
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
        for (attrtype, ocr_str, meta, isflip_i, side_i) in lst:
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

def to_groupclasses(proj, grouplabel_record=None):
    """ Converts the result of grouping (non-digits) to a list of 
    GroupClass instances.
    Input:
        obj proj:
        list grouplabel_record: The canonical list of attrtype/attrval
            orderings, created by create_grouplabel_record(). If not
            given, this function will read it from disk.
    Output:
        List of GroupClass instances for non-digit attributes.
    """
    bal2imgs=pickle.load(open(proj.ballot_to_images,'rb'))
    if grouplabel_record == None:
        grouplabel_record = common.load_grouplabel_record(proj)
    attr_types = common.get_attrtypes(proj)
    ## Note: the isflip_i/side_i info from digitgroup_results gets
    ## thrown out after these blocks. Do I actually need them?
    group_elements = {} # maps {int grouplabel_idx: [(ballotid_i, rlist_i, patchpath_i), ...]}
    if not util.is_multipage(proj):
        for attr_type in attr_types:
            if common.is_digitbased(proj, attr_type):
                continue
            for ballotid in bal2imgs:
                metadata_dir = proj.ballot_grouping_metadata + '-' + attr_type
                path = os.path.join(metadata_dir, util.encodepath(ballotid))
                try:
                    file = open(path, 'rb')
                except IOError as e:
                    print e
                    pdb.set_trace()
                # data maps {'attrOrder': [str attrval_i, ...], 'flipOrder': [int i, ...]}
                data = pickle.load(file)
                file.close()
                # 0.) Construct the ranked list
                rlist = []
                attrvals, flips = data['attrOrder'], data['flipOrder']
                for i, attrval in enumerate(attrvals):
                    grouplabel = common.make_grouplabel((attr_type, attrval), ('flip', flips[i]),
                                                        ('imageorder', 0))
                    gl_idx = grouplabel_record.index(grouplabel)
                    rlist.append(gl_idx)
                patchpath = pathjoin(proj.extracted_precinct_dir+"-"+attr_type,
                                     util.encodepath(ballotid)+'.png')
                group_elements.setdefault(rlist[0], []).append((ballotid, rlist, patchpath))
    else:
        # Multipage
        for attr_type in attr_types:
            if common.is_digitbased(proj, attr_type):
                continue
            for ballotid, (frontpath, backpath) in bal2imgs.iteritems():
                metadata_dir = proj.ballot_grouping_metadata + '-' + attr_type
                path = os.path.join(metadata_dir, util.encodepath(ballotid))
                try:
                    file = open(path, 'rb')
                except IOError as e:
                    print e
                    pdb.set_trace()
                # data is a dict with keys 'attrOrder', 'flipOrder', 'err',
                # and 'imageOrder'
                # data maps {'attrOrder': [str attrval_i, ...], 'flipOrder': [int i, ...], 'imageOrder': [int i, ...]}
                data = pickle.load(file)
                file.close()
                # 0.) Construct the ranked list
                rlist = []
                attrvals, flips = data['attrOrder'], data['flipOrder']
                for i, attrval in enumerate(attrvals):
                    grouplabel = common.make_grouplabel((attr_type, attrval), ('flip', flips[i]),
                                                        ('imageorder', 0))
                    gl_idx = grouplabel_record.index(grouplabel)
                    rlist.append(gl_idx)
                patchpath = pathjoin(proj.extracted_precinct_dir+"-"+attr_type,
                                     util.encodepath(ballotid)+'.png')
                group_elements.setdefault(rlist[0], []).append((ballotid, rlist, patchpath))

    groups = []
    # Seed initial set of groups
    for gl_idx, elements in group_elements.iteritems():
        groups.append(common.GroupClass(elements))
    return groups

def sanitycheck_blankballots(proj):
    """ Makes sure that each blank ballot has a unique set of ballot
    attributes. If a set S of blank ballots have the same set of ballot
    attributes, then the check fails if the layouts of the ballots in S
    are not all the same. 
    Two ballots A, B have a different 'layout' if:
        a.) A, B have different number of targets
        b.) The locations of targets between A, B are not 'close enough'
    Input:
        obj proj
    Output:
        dict badblanks: a dict containing all sets S that fail the above
            sanity check, of the form:
                {((attrtype_i, attrval_i), ...): (str blankpath_i, ...)}
    """
    # 0.) Read in blank ballot attributes
    blanks = {} # maps {str blankid: ((attrtype_i, attrval_i), ...)}
    attrtypes = []
    if os.path.exists(pathjoin(proj.projdir_path, proj.digitattrvals_blanks)):
        # dict mapping {str blankpath: {digitattrtype: digitval}}
        digitattrvals = pickle.load(open(pathjoin(proj.projdir_path, proj.digitattrvals_blanks), 'rb'))
    else:
        digitattrvals = None
    for dirpath, dirnames, filenames in os.walk(proj.patch_loc_dir):
        for f in [name for name in filenames if name.lower().endswith('.csv')]:
            csvfile = open(pathjoin(dirpath, f), 'rb')
            reader = csv.DictReader(csvfile)
            attrs = [] # of the form [(str attrtype_i, str attrval_i), ...]
            blankid = None
            for row in reader:
                if blankid == None: blankid = row['imgpath']
                if row['is_tabulationonly'] == 'False':
                    attrs.append((row['attr_type'], row['attr_val']))
            # a.) Handle digitbased-attrs
            if digitattrvals:
                for digitattrtype, (digitattrval, bb, side) in digitattrvals[blankid].iteritems():
                    if not common.is_tabulationonly(proj, digitattrtype):
                        attrs.append((digitattrtype, digitattrval))
            # b.) Handle custom-attrs
            if common.exists_customattrs(proj):
                cattrs = cust_attrs.load_custom_attrs(proj)
                for cattr in cattrs:
                    if not cattr.is_tabulationonly and not cattr.is_votedonly:
                        if cattr.mode == cust_attrs.CustomAttribute.M_SPREADSHEET:
                            inval = [v for (t, v) in attrs if t == cattr.attrin][0]
                            attrval = cust_attrs.custattr_map_inval_ss(proj, cattr.attrname,
                                                                       inval)
                        elif cattr.mode == cust_attrs.CustomAttribute.M_FILENAME:
                            attrval = cust_attrs.custattr_apply_filename(cattr, blankid)
                        else:
                            print "Unexpected CustomAttribute mode."
                            pdb.set_trace()
                        attrs.append((cattr.attrname, attrval))
            blanks[blankid] = attrs
    # 1.) Construct inverse mapping of blanks
    inv_blanks = {} # maps {((attrtype_i, attrval_i), ...): [str blankid_i, ...]}
    for blankid, pairs in blanks.iteritems():
        inv_blanks.setdefault(tuple(sorted(pairs, key=lambda tup: tup[0])), []).append(blankid)
    # 2.) Filter out all buckets with more than one blank ballot
    for attrpairs in inv_blanks.keys():
        if len(inv_blanks[attrpairs]) == 1:
            inv_blanks.pop(attrpairs)
    # 3.) Terminate if no blank ballots have same attribute values
    if not inv_blanks:
        print "No blank ballots exist with same attribute values, done!"
        return {}
    # 4.) Do 'involved' check between contests in each set S
    output = {} # maps {((attrtype_i, attrval_i), ...): [str blankid_i]}
    print "...Exists blank ballots with same attribute values, need to dig deeper."
    for attrpairs, group in inv_blanks.iteritems():
        by_layout = separate_by_layout(group, proj)
        if len(by_layout) != 1:
            # a.) Physical layout is different!
            output.setdefault(attrpairs, []).extend(group)
        else:
            # b.) Layout is same. Check text interpretation, if possible.
            by_text = separate_by_text(group, proj)
            if len(by_text) != 1:
                output.setdefault(attrpairs, []).extend(group)
    return output

def separate_by_layout(blankpaths, proj):
    """ Given a list of blank ballot paths, group the blank ballots
    by ballot layout, purely based on location of contests+voting targets.
    Input:
        list blankpaths: [blankpath_i, ...]
        obj proj:
    Output:
        list groups: [[blankpath_i0, ...], [blankpath_i1, ...], ...]
    """
    csvpath_map = pickle.load(open(pathjoin(proj.target_locs_dir, 'csvpath_map.p'),
                                   'rb'))
    # 0.) Read in all targets/contests information
    layouts = {} # maps {str blankpath: [[x, y, w, h, is_contest], ...]}
    _set_blankpaths = set(blankpaths)
    for csvpath, blankpath in csvpath_map.iteritems():
        if blankpath not in _set_blankpaths: continue
        f = open(csvpath, 'rb')
        reader = csv.DictReader(f)
        for row in reader:
            entry = [row['x'], row['y'], row['width'], row['height'], row['is_contest']]
            entry = [int(n) for n in entry]
            layouts.setdefault(blankpath, []).append(entry)
        f.close()
    if False:
        FLAG = True
    else:
        FLAG = False
    # 1.) Do comparisons.
    blankpaths = blankpaths[:]
    bp_cpy = blankpaths[:]
    output = [] # list of groups
    while len(blankpaths) > 0:
        bp_i = blankpaths.pop()
        layout_i = layouts[bp_i]
        group_i = [bp_i]
        j = 0
        while j < len(blankpaths):
            bp_j = blankpaths[j]
            layout_j = layouts[bp_j]
            if is_layout_same(layout_i, layout_j, debug=FLAG):
                group_i.append(bp_j)
                blankpaths.pop()
            else:
                j += 1
        output.append(group_i)
    return output
    
def is_layout_same(layoutA, layoutB, C=0.75, debug=False):
    """ Returns True iff LAYOUTA is reasonably close to LAYOUTB.
    Input:
        list layoutA: [[x,y,w,h,is_contest], ...]
        list layoutB: [[x,y,w,h,is_contest], ...]
        float C: Param controlling how far away (in terms of target w/h)
            two targets can be and still be considered 'paired'.
    Output:
        True/False.
    """
    def check_boxes(boxesA, boxesB):
        boxesA = boxesA[:]
        boxesB = boxesB[:]
        while len(boxesA) > 0:
            bA = boxesA.pop()
            j = 0
            foundit = False
            if debug:
                print "...Trying to find box:", bA
            minDist, minJ = None, None
            while j < len(boxesB):
                bB = boxesB[j]
                dist = distL2(bA[0], bA[1], bB[0], bB[1])
                if debug:
                    print 'dist is:', dist
                if dist <= (bA[3]*C): # height*C
                    if minDist == None or dist < minDist:
                        minDist = dist
                        minJ = j
                    else:
                        j += 1
                else:
                    j += 1
            if minDist == None:
                # Couldn't find a target close enough to bA
                if debug:
                    pdb.set_trace()
                return False
            else:
                boxesB.pop(minJ)
        return True
    # 0.) Simple sanity checks
    if len(layoutA) != len(layoutB):
        return False
    targetsA, contestsA = [], []
    targetsB, contestsB = [], []
    for (x,y,w,h,is_contest) in layoutA:
        if is_contest == 1:
            contestsA.append((x,y,w,h,is_contest))
        else:
            targetsA.append((x,y,w,h,is_contest))
    for (x,y,w,h,is_contest) in layoutB:
        if is_contest == 1:
            contestsB.append((x,y,w,h,is_contest))
        else:
            targetsB.append((x,y,w,h,is_contest))
    if len(targetsA) != len(targetsB) or len(contestsA) != len(contestsB):
        return False
    # 1.) Check targets and contests pairwise
    return check_boxes(targetsA, targetsB) and check_boxes(contestsA, contestsB)

def distL2(x1,y1,x2,y2):
    return math.sqrt((y1-y2)**2 + (x1-x2)**2)

def separate_by_text(blankpaths, proj):
    """ Given a list of blank ballot paths, group the ballots by text
    interpretation. Assumes that BLANKPATHS contains blank ballots with
    the same layout, i.e. separate_by_layout(BLANKPATHS)[0] == BLANKPATHS.
    Input:
        list blankpaths: [blankpath_i, ...]
    Output:
        list groups: [[blankpath_i0, ...], [blankpath_i1, ...]]
    """
    # TODO: Implement me!
    return [blankpaths]
