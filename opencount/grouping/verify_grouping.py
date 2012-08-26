import sys, csv, copy, pdb, os, re, shutil
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
            self.verify_grouping.start(groups, exemplar_paths, self.project, ondone=self.verifying_done)
            self.project.addCloseEvent(self.verify_grouping.dump_state)
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
                groups = to_groupclasses(self.project)
                digitgroup_results = digit_group.load_digitgroup_results(self.project)
                groups.extend(digit_group.to_groupclasses_digits(self.project, digitgroup_results))
                exemplar_paths = get_exemplar_paths()
                self.verify_grouping.start(groups, exemplar_paths, self.project, ondone=self.verifying_done)
                self.project.addCloseEvent(self.verify_grouping.dump_state)
                self.verify_grouping.SendSizeEvent()
                self.SendSizeEvent()
                self.Refresh()
                self.Fit()
                return
                
            #self.verify_grouping.start(groups, patches, exemplar_paths)
            self.verify_grouping.load_state()
            exemplar_paths = get_exemplar_paths()
            self.verify_grouping.project = self.project
            self.project.addCloseEvent(self.verify_grouping.dump_state)
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
        self.project.removeCloseEvent(self.verify_grouping.dump_state)
        attr_types = set(common.get_attrtypes(self.project))
        # 0.) munge digit-grouping-results into results, since digitattrs
        #     are still in 'digit' form.
        results = munge_digit_results(results, attr_types, self.project)
        # munge results -> results_foo
        # results_foo has all attribute vals for all voted ballots
        results_foo = {} # {samplepath: {attrtype: (attrval, flip, imgorder)}}

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
                # Because the flip/imgorder info got thrown out in
                # on_grouping_done, these grouplabels won't have this
                # info. But, rest assured, it has been taken care of.
                flip = 0
                imgorder = 0
            else:
                flip = common.get_propval(grouplabel, 'flip')
                imgorder = common.get_propval(grouplabel, 'imageorder')
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
                                       img2tmp)
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
        print "== calling groupImagesMAP..."
        groupImagesMAP(bal2imgs,
                       tpl2imgs,
                       munged,
                       self.project.extracted_precinct_dir, 
                       self.project.ballot_grouping_metadata, 
                       stopped,
                       deleteall=deleteall)
        print "== finished groupImagesMAP"
        if digitmunged:
            print "== Performing DigitOCR..."
            digitgroup_results, digitmatch_info = digit_group.do_digitocr_patches(bal2imgs, digitmunged, self.project)
            print "== Finished DigitOCR."
            digit_group.save_digitgroup_results(self.project, digitgroup_results)
            digit_group.save_digitmatch_info(self.project, digitmatch_info)
        wx.CallAfter(Publisher().sendMessage, "signals.MyGauge.done")

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
                attrtype, attrval = common.get_attrpair_grouplabel(project, grouplabel)
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
      Path of the associated template.
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
        print "Uhoh, more than one possible blank ballot: {0} possibles.".format(len(possibles))
        pdb.set_trace()
    if len(possibles) == 0:
        print "== Error, determine_template couldn't find a blank ballot with a matching set"
        print "   of attributes. We're hosed.  Quarantining this voted ballot."
        print "  ", samplepath
        print "== To proceed, type in 'c', and press ENTER."
        pdb.set_trace()
        return None
    assert len(possibles) == 1
    return possibles.keys()[0]

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
        if not v:
            print "Uh oh, v wasn't found in this grouplabel:", common.str_grouplabel(grouplabel)
            pdb.set_trace()
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
        dict results: maps {grouplabel: list of GroupClasses}.
        lst all_attrtypes: List of all attrtypes
    Output:
        dict that maps: {grouplabel: list of GroupClasses} but with
        all digitattributes inside of the result.
    """
    def is_digitbased_grouplabel(grouplabel):
        """Assumes a digit-based grouplabel has a k,v with the
        k being 'digit'. Lousy assumption.
        TODO: This 'digit' kv-pair assumption restricts this framework
        to only allow one digit-based attribute at a time. We should
        ideally be able to handle any number of digit-based attributes.
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

def to_groupclasses(proj):
    """ Converts the result of grouping (non-digits) to a list of 
    GroupClass instances.
    Input:
        obj proj:
    Output:
        List of GroupClass instances for non-digit attributes.
    """
    bal2imgs=pickle.load(open(proj.ballot_to_images,'rb'))

    grouping_results = {} # maps {str attrtype: {bestmatch: list of [ballotid, rankedlist, patchpath]}}
    attr_types = common.get_attrtypes(proj)
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
    ## Note: the isflip_i/side_i info from digitgroup_results gets
    ## thrown out after these blocks. Do I actually need them?
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

                data = pickle.load(file)
                file.close()
                dummies = [0]*len(data["attrOrder"])
                attrs_list = zip(data["attrOrder"], data["flipOrder"], dummies)
                bestMatch = attrs_list[0]
                patchpath = pathjoin(proj.extracted_precinct_dir+"-"+attr_type,
                                     util.encodepath(ballotid)+'.png')
                grouping_results.setdefault(attr_type, {}).setdefault(bestMatch, []).append((ballotid, attrs_list, patchpath))
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
                data = pickle.load(file)
                file.close()
                attrs_list = zip(data["attrOrder"], data["flipOrder"], data["imageOrder"])
                bestMatch = attrs_list[0]
                bestAttr = bestMatch[0]
                patchpath = pathjoin(proj.extracted_precinct_dir+"-"+attr_type,
                                     util.encodepath(ballotid)+'.png')
                grouping_results.setdefault(attr_type, {}).setdefault(bestMatch, []).append((ballotid, attrs_list, patchpath))

    groups = []
    # Seed initial set of groups
    i = 1
    # Note: grouping_results is structured strangely, hence, why
    # the strange code below.
    for attrtype, _dict in grouping_results.items():
        for (attrval,flip,imgorder), samples in _dict.items():
            #extracted_attr_dir = self.proj.extracted_precinct_dir + '-' + attrtype
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
    return groups
