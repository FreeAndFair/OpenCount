import sys, csv, copy, pdb, os
import threading, time
import timeit
sys.path.append('../')

from util import MyGauge
from specify_voting_targets import util_gui as util_gui
from specify_voting_targets import imageviewer as imageviewer
from specify_voting_targets.util_gui import *
import util, common

from pixel_reg.imagesAlign import *
import pixel_reg.shared as sh
from pixel_reg.doGrouping import  groupImagesMAP, encodepath

####
## Import 3rd party libraries
####
import wx
import wx.lib.scrolledpanel as scrolled
import Image
import cv2
import numpy as np
import scipy
from scipy import misc    
import wx.lib.inspection
from wx.lib.pubsub import Publisher

THRESHOLD = 0.9

class VerifyPanel(wx.Panel):
    """
    Modes for verify behavior.
    MODE_NORMAL: The general behavior, where you have N predetermined
                 group categories that you want to assign to some 
                 data.
    MODE_YESNO: A subset of MODE_NORMAL, where you only have two
                groups: groupA, and 'other'.
    MODE_YESNO2: Like MODE_YESNO, but answering the the slightly
                 different question: "Do these images represent
                 /some/ group?" The idea is, this mode doesn't
                 know the groups ahead of time, whereas MODE_YESNO
                 does.
    """
    MODE_NORMAL = 0
    MODE_YESNO  = 1
    MODE_YESNO2 = 2

    YES_IDX = 0
    OTHER_IDX = 1

    # Create the 'dummy' grouplabels used in other modes
    GROUPLABEL_OTHER = common.make_grouplabel(('othertype', 'otherval'))
    GROUPLABEL_MANUAL = common.make_grouplabel(('manualtype', 'manualval'))

    def __init__(self, parent, verifymode=None, *args, **kwargs):
        wx.Panel.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.project = None
        self.ondone = None  # An optional callback function to call
                            # when verifying is done
        self.outfilepath = None   # An optional filepath to output
                                  # grouping results to.
        if not verifymode:
            self.mode = VerifyPanel.MODE_NORMAL
        elif verifymode == VerifyPanel.MODE_YESNO:
            self.mode = verifymode
        elif verifymode == VerifyPanel.MODE_YESNO2:
            self.mode = verifymode
        else:
            print "Unrecognized mode for VerifyPanel:", verifymode
            self.mode = VerifyPanel.MODE_NORMAL

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.patchsizer = None
        
        # List of groups to-be-verified
        self.queue = []
        self.currentGroup = None
        # List of groups that have been verified
        self.finished = []

        self.resultsPath = None
        
        # templates is a dict mapping
        #    {grouplabel: str attrpatch_img}
        # where attrpatch_img is the image patch on the template 
        # corresponding to the grouplabel
        self.templates = None
        
        self.canMoveOn = False
        self.mainPanel = scrolled.ScrolledPanel(self, size=self.GetSize(), pos=(0,0))

        self.initLayout()
        self.initBindings()

        Publisher().subscribe(self._pubsub_project, "broadcast.project")

    def fitPanel(self):
        #w, h = self.parent.GetClientSize()
        #self.mainPanel.SetMinSize((w * 0.95, h * 0.9))
        #self.mainPanel.GetSizer().SetSizeHints(self)
        self.mainPanel.SetupScrolling()
    
    def overlays_layout_vert(self):
        self.patchsizer.Clear(False)
        self.patchsizer.SetRows(4)
        self.patchsizer.SetCols(2)
        
        # HBOX 1 (min overlay)
        self.patchsizer.Add(self.st1, flag=wx.ALIGN_LEFT)
        self.patchsizer.Add(self.minOverlayImg, flag=wx.ALIGN_LEFT)
        
        # HBOX 2 (max overlay)
        self.patchsizer.Add(self.st2, flag=wx.ALIGN_LEFT)
        self.patchsizer.Add(self.maxOverlayImg, flag=wx.ALIGN_LEFT)
        
        # HBOX 3 (template patch)
        self.patchsizer.Add(self.st3, flag=wx.ALIGN_LEFT)
        self.patchsizer.Add(self.templateImg, flag=wx.ALIGN_LEFT)
        
        # HBOX 6 (diff patch)
        self.patchsizer.Add(self.st4, flag=wx.ALIGN_LEFT)
        self.patchsizer.Add(self.diffImg, flag=wx.ALIGN_LEFT)

    def overlays_layout_horiz(self):
        self.patchsizer.Clear(False)
        self.patchsizer.SetRows(2)
        self.patchsizer.SetCols(4)
        # Add texts
        self.patchsizer.Add(self.st1, flag=wx.ALIGN_LEFT)
        self.patchsizer.Add(self.st2, flag=wx.ALIGN_LEFT)
        self.patchsizer.Add(self.st3, flag=wx.ALIGN_LEFT)
        self.patchsizer.Add(self.st4, flag=wx.ALIGN_LEFT)
        # HBOX 1 (min overlay)
        self.patchsizer.Add(self.minOverlayImg, flag=wx.ALIGN_LEFT)
        # HBOX 2 (max overlay)
        self.patchsizer.Add(self.maxOverlayImg, flag=wx.ALIGN_LEFT)
        # HBOX 3 (template patch)
        self.patchsizer.Add(self.templateImg, flag=wx.ALIGN_LEFT)
        # HBOX 6 (diff patch)
        self.patchsizer.Add(self.diffImg, flag=wx.ALIGN_LEFT)

    def set_patch_layout(self, orient='horizontal'):
        """
        Change the orientation of the overlay patch images. Either
        arrange 'horizontal', or stack 'vertical'.
        """
        if orient == 'horizontal':
            sizer = self.overlays_layout_horiz()
        else:
            sizer = self.overlays_layout_vert()
        self.Refresh()
        self.Layout()
        self.Refresh()

    def initLayout(self):
        st1 = wx.StaticText(self.mainPanel, -1, "min:     ", style=wx.ALIGN_LEFT)
        st2 = wx.StaticText(self.mainPanel, -1, "max:     ", style=wx.ALIGN_LEFT)
        st3 = wx.StaticText(self.mainPanel, -1, "Attribute Patch:", style=wx.ALIGN_LEFT)
        st4 = wx.StaticText(self.mainPanel, -1, "diff:", style=wx.ALIGN_LEFT)
        self.st1, self.st2, self.st3, self.st4 = st1, st2, st3, st4

        self.minOverlayImg = wx.StaticBitmap(self.mainPanel, bitmap=wx.EmptyBitmap(1, 1))
        self.maxOverlayImg = wx.StaticBitmap(self.mainPanel, bitmap=wx.EmptyBitmap(1, 1))
        self.templateImg = wx.StaticBitmap(self.mainPanel, bitmap=wx.EmptyBitmap(1, 1))
        self.diffImg = wx.StaticBitmap(self.mainPanel, bitmap=wx.EmptyBitmap(1, 1))

        self.patchsizer = wx.GridSizer()
        self.set_patch_layout('horizontal')
        # HBOX 5 (ComboBox and buttons)
        hbox5 = wx.BoxSizer(wx.HORIZONTAL)
        self.templateChoice = wx.ComboBox(self.mainPanel, choices=[], style=wx.CB_READONLY)
        self.okayButton = wx.Button(self.mainPanel, label='OK')
        self.splitButton = wx.Button(self.mainPanel, label='Split')
        self.debugButton = wx.Button(self.mainPanel, label='DEBUG')
        self.misclassifyButton = wx.Button(self.mainPanel, label="Mis-classified")
        self.misclassifyButton.Bind(wx.EVT_BUTTON, self.OnClickMisclassify)
        self.quarantineButton = wx.Button(self.mainPanel, label='Quarantine')

        # Buttons for MODE_YESNO
        self.yes_button = wx.Button(self.mainPanel, label="Yes")
        self.no_button = wx.Button(self.mainPanel, label="No")
        # Buttons for MODE_YESNO2
        self.manuallylabelButton = wx.Button(self.mainPanel, label='Manually Label This Group')
        self.manuallylabelButton.Bind(wx.EVT_BUTTON, self.OnClickLabelManually)

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
        hbox5.Add(self.misclassifyButton, flag=wx.LEFT | wx.CENTRE)
        hbox5.Add(self.yes_button, flag=wx.LEFT | wx.CENTRE)
        hbox5.Add((40,-1))
        hbox5.Add(self.no_button, flag=wx.LEFT | wx.CENTRE)
        hbox5.Add((40,-1))
        hbox5.Add(self.manuallylabelButton, flag=wx.LEFT | wx.CENTRE)

        # HBOX8 (# of ballots)
        hbox8 = wx.BoxSizer(wx.HORIZONTAL)
        st5 = wx.StaticText(self.mainPanel, -1, "# of ballots in the group: ", style=wx.ALIGN_LEFT)
        self.tNumBallots = wx.TextCtrl(self.mainPanel, value='0')
        self.tNumBallots.SetEditable(False)
        hbox8.Add(st5, flag=wx.CENTRE)
        hbox8.Add(self.tNumBallots, flag=wx.CENTRE)
        
        # VBOX2 (right half)
        vbox2 = wx.BoxSizer(wx.VERTICAL)
        self.vbox2 = vbox2
        vbox2.Add((-1,5))
        vbox2.Add(hbox8, flag=wx.LEFT | wx.CENTRE)
        #vbox2.Add(hbox4, flag=wx.LEFT | wx.CENTRE)
        #vbox2.Add(vbox1, flag=wx.LEFT | wx.CENTRE)
        vbox2.Add(self.patchsizer, flag=wx.LEFT | wx.CENTRE)
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
        
        if self.mode == VerifyPanel.MODE_NORMAL:
            self.yes_button.Hide()
            self.no_button.Hide()
            self.manuallylabelButton.Hide()
        elif self.mode == VerifyPanel.MODE_YESNO:
            self.okayButton.Hide()
            self.quarantineButton.Hide()
            self.misclassifyButton.Hide()
            self.manuallylabelButton.Hide()
        elif self.mode == VerifyPanel.MODE_YESNO2:
            self.okayButton.Hide()
            self.no_button.Hide()
            self.quarantineButton.Hide()
            self.misclassifyButton.Hide()
            self.templateImg.Hide()
            self.diffImg.Hide()
            self.st3.Hide()
            self.st4.Hide()

        self.mainPanel.SetSizer(hbox7)
        self.sizer.Add(self.mainPanel, proportion=1, flag=wx.EXPAND)
        self.mainPanel.Hide()
        self.SetSizer(self.sizer, deleteOld=False)

    def onSelect_queuelistbox(self, evt):
        idx = evt.GetSelection()
        if idx >= 0:
            assert idx < len(self.queue)
            self.select_group(self.queue[idx])
            
    def start(self, groups, exemplar_paths, outfilepath=None, ondone=None):
        """
        Start verifying the overlays. Groups is a list of 
        GroupClass objects, representing pre-determined clusters
        within a data set.
        exemplar_paths is a dict that maps each grouplabel to an
        exemplar image patch. This is the set of possible labels
        for each example in the data set. Only used for MODE_NORMAL
        and MODE_YESNO
          {grouplabel: str imgpath}
        outfilepath is an optional file to store the grouping results.
        ondone is an optional function to call when grouping is done:
        it should be a function that accepts one argument (the grouping
        results). The grouping results is a dict that maps each grouplabel
        to a list of samples:
          {grouplabel: list of (sampleid, rankedlist, patchpath)}
        """
        if exemplar_paths:
            self.load_exemplar_attrpatches(exemplar_paths)
        else:
            self.exemplar_paths = None
            self.templates = {}
        self.outfilepath = outfilepath
        self.ondone = ondone
        for group in groups:
            self.add_group(group)
        self.start_verifygrouping()
    
    def load_exemplar_attrpatches(self, exemplar_paths):
        """
        Load in all attribute patches for each attrtype->attrval pair.
        exemplar_paths: {grouplabel: str patchpath}
        """
        self.templates = exemplar_paths
    
    def dump_state(self):
        if self.project:
            fqueue = open(pathjoin(self.project.projdir_path, 'verifygroupstate.p'), 'wb')
            d = {}
            q = list(self.queue)
            if self.currentGroup:
                if self.currentGroup not in self.queue and self.currentGroup not in self.finished:
                    q.insert(0, self.currentGroup)
            q.extend(self.finished)
            d['todo'] = q
            #d['finished'] = self.finished
            d['finished'] = []
            pickle.dump(d, fqueue)
        
    def load_state(self):
        if os.path.exists(pathjoin(self.project.projdir_path, 'verifygroupstate.p')):
            try:
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
            except Exception as e:
                # If you can't read in the state file, then just don't
                # load in any state.
                print e
                return
 
    def start_verifygrouping(self):
        """
        Called after sample ballots have been grouped by Kai's grouping
        code. Sets up UI widgets of the verification UI.
        """
        self.SetSizer(self.sizer, deleteOld=False)

        if self.mode in (VerifyPanel.MODE_YESNO, VerifyPanel.MODE_YESNO2):
            # Add a dummy group to each GroupClass
            if self.mode == VerifyPanel.MODE_YESNO:
                type, val = 'othertype', 'otherval'
            else:
                type, val = 'manualtype', 'manualval'
            num_descs = 0
            for group in self.queue:
                for element in group.elements:
                    num_descs = len(element[1][0]) - 1
                    break
                break
            dummy_grouplabel = common.make_grouplabel(*((type, val),) + (('dummy',None),)*num_descs)
            for group in self.queue:
                group.orderedAttrVals.append(dummy_grouplabel)
                for element in group.elements:
                    element[1].append(dummy_grouplabel)
            
        if self.queue:
            self.select_group(self.queue[0])

        self.mainPanel.Show()
        self.Fit()

    def initBindings(self):
        self.templateChoice.Bind(wx.EVT_COMBOBOX, self.OnSelectTemplate)
        self.Bind(wx.EVT_BUTTON, self.OnClickOK, self.okayButton)
        self.Bind(wx.EVT_BUTTON, self.OnClickSplit, self.splitButton)
        self.Bind(wx.EVT_BUTTON, self.OnClickDebug, self.debugButton)
        self.Bind(wx.EVT_BUTTON, self.OnClickQuarantine, self.quarantineButton)
        self.Bind(wx.EVT_BUTTON, self.OnClickYes, self.yes_button)
        self.Bind(wx.EVT_BUTTON, self.OnClickNo, self.no_button)
        self.Bind(wx.EVT_SIZE, self.OnSize)
    
    def updateTemplateThumb(self):
        """
        Updates the 'Attribute Patch' and 'Diff' image patches.
        """
        if self.mode == VerifyPanel.MODE_YESNO2:
            # We don't have exemplar patches
            return
        overlayMin, overlayMax = self.currentGroup.get_overlays()
        templates = self.currentGroup.orderedAttrVals
        elements = self.currentGroup.elements

        idx = self.templateChoice.GetSelection()
        curgrouplabel = self.currentGroup.orderedAttrVals[idx]

        try:
            attrpatch_imgpath = self.templates[curgrouplabel]
            attrpatch_img = misc.imread(attrpatch_imgpath, flatten=1)
            rszFac = sh.resizeOrNot(attrpatch_img.shape, sh.MAX_PRECINCT_PATCH_DISPLAY)
            attrpatch_img = sh.fastResize(attrpatch_img, rszFac) / 255.0
        except Exception as e:
            print e
            pdb.set_trace()
        
        h_overlay, w_overlay = overlayMax.shape
        if attrpatch_img.shape != overlayMax.shape:
            attrpatch_img = common.resize_img_norescale(attrpatch_img, (w_overlay, h_overlay))
        IO = imagesAlign(overlayMax, attrpatch_img)
        Dabs=np.abs(IO[1]-attrpatch_img)
        diffImg = np.vectorize(lambda x: x * 255.0 if x >= THRESHOLD else 0.0)(Dabs)
        
        self.templateImg.SetBitmap(NumpyToWxBitmap(attrpatch_img * 255.0))
        self.diffImg.SetBitmap(NumpyToWxBitmap(diffImg))
        self.Refresh()
    
    def OnSelectTemplate(self, event):
        """
        Triggered when the user selects a different attribute value
        in the dropdown menu.
        """
        self.updateTemplateThumb()
        
    def add_finalize_group(self, group, final_index):
        """
        Finalize a group, stating that all images in group match the
        class given by final_index.
        """
        group.index = final_index
        self.finished.append(group)
        self.finishedList.Append(group.label)

    def add_group(self, group):
        """
        Adds a new GroupClass to internal datastructures, and updates
        relevant UI components.
        """
        if group in self.queue:
            # DOGFOOD: Remove this hotfix after the Napa audits
            print "Silently throwing out duplicate group object."
            return
        #assert group not in self.queue
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

        overlayMin, overlayMax = self.currentGroup.get_overlays()
        ordered_attrvals = self.currentGroup.orderedAttrVals
        elements = self.currentGroup.elements
        
        h, w = overlayMin.shape
        if w > h:
            self.set_patch_layout('vertical')
        else:
            self.set_patch_layout('horizontal')

        self.minOverlayImg.SetBitmap(NumpyToWxBitmap(overlayMin * 255.0))
        self.maxOverlayImg.SetBitmap(NumpyToWxBitmap(overlayMax * 255.0))
        
        self.tNumBallots.SetValue("{0}".format(len(elements)))
        
        self.templateChoice.Clear()
        history = set()
        for grouplabel in ordered_attrvals:
            if grouplabel not in history:
                #display_string = str(grouplabel)
                try:
                    display_string = common.str_grouplabel(grouplabel)
                except Exception as e:
                    print e
                    pdb.set_trace()
                self.templateChoice.Append(display_string)
                history.add(grouplabel)
        
        self.templateChoice.SetSelection(self.currentGroup.index)
        
        self.updateTemplateThumb()
        
        if (len(elements) <= 1):
            self.splitButton.Disable()
        else:
            self.splitButton.Enable()
        
        self.parent.Fit()
        self.fitPanel()
    
    def OnClickOK(self, event):
        index = self.templateChoice.GetCurrentSelection()
        self.add_finalize_group(self.currentGroup, index)
        
        self.remove_group(self.currentGroup)

        if self.is_done_verifying():
            self.currentGroup = None
            self.done_verifying()
        else:
            self.select_group(self.queue[0])

    def OnClickYes(self, event):
        """ Used for MODE_YESNO """
        self.add_finalize_group(self.currentGroup, VerifyPanel.YES_IDX)
        self.remove_group(self.currentGroup)
        if self.is_done_verifying():
            self.currentGroup = None
            self.done_verifying()
        else:
            self.select_group(self.queue[0])
        
    def OnClickNo(self, event):
        """ USED FOR MODE_YESNO """
        self.add_finalize_group(self.currentGroup, VerifyPanel.OTHER_IDX)
        self.remove_group(self.currentGroup)
        if self.is_done_verifying():
            self.currentGroup = None
            self.done_verifying()
        else:
            self.select_group(self.queue[0])

    def OnClickLabelManually(self, event):
        """ USED FOR MODE_YESNO2. Signal that the user wants to 
        manually label everything in this group. """
        self.currentGroup.is_manual = True
        self.add_finalize_group(self.currentGroup, VerifyPanel.OTHER_IDX)
        self.remove_group(self.currentGroup)
        if self.is_done_verifying():
            self.currentGroup = None
            self.done_verifying()
        else:
            self.select_group(self.queue[0])

    def OnClickMisclassify(self, evt):
        """ Used for MODE_NORMAL. Signals that the current attr patch
        is misclassified (say, the wrong attribute type), and to handle
        it *somehow*.
        For digit-based attributes, this will re-run partmatch*, but with
        an updated mismatch dict.
        """
        grouplabel = self.currentGroup.getcurrentgrouplabel()
        attrtypestr, attrval = common.get_attrpair_grouplabel(self.project, grouplabel)
        if not common.is_digitbased(attrtypestr):
            dlg = wx.MessageDialog(self, message="'Misclassify' isn't \
supported for non-digitbased attributes. Perhaps you'd like to quarantine \
this instead?", style=wx.OK)
            self.Disable()
            dlg.ShowModal()
            self.Enable()
            return
        rejected_paths = []
        for (sampleid, rlist, imgpatch) in self.currentGroup.elements:
            # TODO: Do I append sampleid, or imgpath? 
            rejected_paths.append(imgpath)
        partmatch_fns.reject_match(rejected_paths, self.project)

    def is_done_verifying(self):
        return not self.queue
        
    def done_verifying(self):
        """
        When the user has finished verifying all groups, do some
        fancy computing, and output results.
        Outputs grouping results into the specified out-directory,
        where each group gets outputted to an output file.
        """
        # First populate results
        print "DONE Verifying!"
        self.Disable()
        results = {} # {grouplabel: list of GroupClasses}
        if self.mode == VerifyPanel.MODE_YESNO2:
            # Hack: Treat each GroupClass as separate categories,
            # instead of trying to merge them. However, we do
            # merge the GROUPLABEL_MANUAL ones.
            for group in self.finished:
                grouplabel = group.orderedAttrVals[0]
                if grouplabel in results:
                    pdb.set_trace()
                assert grouplabel not in results, "grouplabel {0} was duplicated".format(common.str_grouplabel(grouplabel))
                results[grouplabel] = [group]
        else:
            for group in self.finished:
                results.setdefault(group.getcurrentgrouplabel(), []).append(group)
            if self.templates:
                for grouplabel in self.templates:
                    if grouplabel not in results:
                        results[grouplabel] = []

        if self.outfilepath:
            pickle.dump(results, open(self.outfilepath, 'wb'))

        if self.ondone:
            self.ondone(results)

    def OnClickSplit(self, event):
        def collect_ids(newGroups):
            ids = {} # {str attrname: list of ids}
            groups = tuple(newGroups) + tuple(self.queue) + tuple(self.finished)

            for group in groups:
                # In MODE_YESNO2, foo is a list of the form:
                #    ((<attrtype>,), ID)
                #foo = list(group.getcurrentgrouplabel())
                foo = list(group.orderedAttrVals[0])
                attrtype = tuple(sorted([t[0] for t in foo]))
                id = foo[0][1]
                ids.setdefault(attrtype, []).append(id)
            return ids
        def assign_new_id(group, ids):
            """ Given a new GroupClass, and a previous IDS mapping,
            find a unique new id for group to use.
            """
            foo = list(group.getcurrentgrouplabel())
            k = tuple(sorted([t[0] for t in foo]))
            i = 0
            while i >= 0:
                if i not in ids[k]:
                    grouplabel = group.orderedAttrVals[0]
                    newgrouplabel = common.make_grouplabel(*[(a, i) for a in k])
                    group.orderedAttrVals[0] = newgrouplabel
                    ids.setdefault(k, []).append(i)
                    break
                i += 1
            return group
            
        newGroups = self.currentGroup.split()

        if self.mode == VerifyPanel.MODE_YESNO2:
            # For each new group, make sure each GroupClass with a 
            # given attr has unique 'attr' values (i.e. a 
            # global counter)
            ids = collect_ids(newGroups)
            for group in newGroups:
                assign_new_id(group, ids)

        for group in newGroups:
            self.add_group(group)
        
        self.remove_group(self.currentGroup)
        self.select_group(self.queue[0])
        self.queueList.Fit()
        self.parent.Fit()
        self.fitPanel()
        
    def OnClickDebug(self, event):
        if (self.currentGroup != None):
            elements = self.currentGroup.elements
            for element in elements:
                print element[0]
            
    def quarantine_group(self, group):
        """
        Quarantines group.
        """
        elements = group.elements
        qfile = open(self.project.quarantined, 'a')
        for element in elements:
            print >>qfile, os.path.abspath(element[0])
        qfile.close()
        self.remove_group(group)
        
    def OnClickQuarantine(self, event):
        if (self.currentGroup != None):
            self.quarantine_group(self.currentGroup)
            if self.is_done_verifying():
                self.done_verifying()
            else:
                self.select_group(self.queue[0])
    
    def checkCanMoveOn(self):
        # TODO: Fix this implementation.
        return True

        return self.canMoveOn
        
    def _pubsub_project(self, msg):
        project = msg.data
        self.project = project
        
    def OnSize(self, event):
        self.fitPanel()
        event.Skip()

