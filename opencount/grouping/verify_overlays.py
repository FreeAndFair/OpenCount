import sys, csv, copy, pdb, os
import threading, time
import timeit
sys.path.append('../')

from util import MyGauge
from specify_voting_targets import util_gui as util_gui
from specify_voting_targets import imageviewer as imageviewer
from specify_voting_targets.util_gui import *
import util, common, partmatch_fns, digit_group

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

"""
Output Files:
- <projdir>/verifygroupstate.p

  This is a snapshot of the internal state of the VerifyOverlays widget.
  It is just a pickle'd dictionary with the keys:
    {'todo': list of all GroupClass objects. Note that this is a list of
             /all/ GroupClass objects, including groups that the user
             has already labeled. At one point, I considered keeping a
             separate 'finished' list, but for simplicity we are not
             doing this.
     'finished': At one point, intended to store all labeled GroupClass
                 instances, but this is unused - should always be the
                 empty list [].}
- <projdir>/<VeriyPanel.outfilepath>
  Stores the raw results of completing the overlay verification, as a
  pickle'd dictionary, where the keys are grouplabels, and the values
  are a list of GroupClass instances:
    {grouplabel: list of GroupClass's}
  For OpenCount, I don't think it uses/saves this. 
"""

"""
VerifyPanel Modes Documentation

MODE_NORMAL: The general behavior, where you have N pre-determined
group categories, and you want to assign each Group to some category.
Actions:
  'Ok' - All overlays look good, and they correspond to the currently-
         selected label.
  'Split' - The overlays look bad (i.e. two different categories are
            present within the Group), so split the Group into two
            different groups, in the hopes of separating the Group
            into distinct groups with good overlays.
  'Quarantine' - If something awful happened (say, the overlay doesn't
                 look /at all/ correct (say, all white/all black), then
                 signal that something bad happened. For OpenCount, this
                 will quarantine the voted ballot(s) that this overlay
                 came from.

MODE_YESNO: Here, the UI is asking the following question: do these
images correspond to some pre-determined group G? For instance, "Do
these images correspond to a 'two'?"
Actions:
  'Yes' - All overlays look good, and they correspond to G. Choosing this
          will set the currentGroup.index = VerifyPanel.YES_IDX.
  'No' - These overlays definitely do /not/ correspond to G. Under the
         hood, this adds currentGroup to the category labeled by grouplabel
         VerifyPanel.GROUPLABEL_OTHER, and sets the group.index to
         VerifyPanel.OTHER_IDX.
  'Split' - See MODE_NORMAL.
  'Quarantine' - See MODE_NORMAL.
     
MODE_YESNO2: Here, the UI is asking: "Given this set of images A, try
to separate them into N groups. We don't know ahead of time how many
groups there are. This would most likely be used to verify the results
of some clustering algorithm (i.e. k-means). 
Actions:
  'Yes' - All overlays look good, and corresponds to some group G. 
          Choosing this will set the currentGroup.index to
          VerifyPanel.YES_IDX. In addition, this creates a grouplabel
          with a unique ID (global counter) stuffed in. For instance,
          if the grouplabel was originally:
            GroupLabel('party' -> 'democrat')
          Then the grouplabels outputted will be:
            GroupLabel(('party', 'democrat') -> 0)
            GroupLabel(('party', 'democrat') -> 1)
            ...
  'Split' - See MODE_NORMAL.
  'Manually Label' - If the current group has an extremely messy overlay,
                     due to many different categories being present, then
                     the user might choose this option in order to avoid
                     having to manually separate out the group into 
                     its categories (which might take /many/ Splits).
                     This sets the currentGroup.is_manual to True, and
                     sets currentGroup.index to VerifyPanel.OTHER_INDEX.
                     In addition, this adds the currentGroup to the 
                     category labeled by VerifyPanel.GROUPLABEL_MANUAL.
"""

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
        # self._mismatch_cnt: Keeps track of the current number of 'new'
        # MisClassified overlays currently in the queue.
        self._mismatch_cnt = 0
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
        misclassify_sizer = wx.BoxSizer(wx.VERTICAL)
        self.misclassifyButton = wx.Button(self.mainPanel, label="Mis-classified")
        self.misclassifyButton.Bind(wx.EVT_BUTTON, self.OnClickMisclassify)
        self.misclassify_txt = wx.StaticText(self.mainPanel, label="Mismatches \
in queue: 0")
        misclassify_sizer.Add(self.misclassifyButton)
        misclassify_sizer.Add(self.misclassify_txt)
        self.rundigitgroupButton = wx.Button(self.mainPanel, label="Run Digit Grouping")
        self.rundigitgroupButton.Bind(wx.EVT_BUTTON, self.OnClickRunDigitGroup)
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
        hbox5.Add(misclassify_sizer, flag=wx.LEFT | wx.CENTRE)
        #hbox5.Add(self.misclassifyButton, flag=wx.LEFT | wx.CENTRE)
        hbox5.Add(self.rundigitgroupButton, flag=wx.LEFT | wx.CENTRE)
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
            self.misclassify_txt.Hide()
            self.rundigitgroupButton.Hide()
            self.manuallylabelButton.Hide()
        elif self.mode == VerifyPanel.MODE_YESNO2:
            self.okayButton.Hide()
            self.no_button.Hide()
            self.quarantineButton.Hide()
            self.misclassifyButton.Hide()
            self.misclassify_txt.Hide()
            self.rundigitgroupButton.Hide()
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
            # Add a dummy group to each GroupClass's orderedAttrVals and
            # rankedlists of the elements list.
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
            #pdb.set_trace()
            return
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
        """ Used for MODE_NORMAL. Indicates that the currentGroup is 
        indeed represented by the current exemplar. """
        index = self.templateChoice.GetCurrentSelection()
        self.add_finalize_group(self.currentGroup, index)
  
        if common.get_propval(self.currentGroup.getcurrentgrouplabel(), 'digit') != None:
            # For digits-based, update our accepted_hashes.
            # TODO: Assumes that digit-based grouplabels has a key 'digit'
            cur_digit = common.get_propval(self.currentGroup.getcurrentgrouplabel(), 'digit')
            # accepted_hashes: {str imgpath: {str digit: [((y1,y2,x1,x2), side), ...]}}
            accepted_hashes = partmatch_fns.get_accepted_hashes(self.project)
            if accepted_hashes == None:
                accepted_hashes = {}
                partmatch_fns.save_accepted_hashes(self.project, accepted_hashes)
            for (sampleid, rlist, patchpath) in self.currentGroup.elements:
                # digitinfo: ((y1,y2,x1,x2), str side)
                digitinfo = digit_group.get_digitmatch_info(self.project, patchpath)
                accepted_hashes.setdefault(sampleid, {}).setdefault(cur_digit, []).append(digitinfo)
            partmatch_fns.save_accepted_hashes(self.project, accepted_hashes)

            cnt = 0
            for imgpath, digitmap in accepted_hashes.iteritems():
                for digit, lst in digitmap.iteritems():
                    cnt += len(lst)
            print "Total number of accepted regions:", cnt

        self.remove_group(self.currentGroup)

        if self.is_done_verifying():
            self.currentGroup = None
            self.done_verifying()
        else:
            self.select_group(self.queue[0])

    def OnClickYes(self, event):
        """ Used for MODE_YESNO. Indicates that the currentGroup does
        correspond to some pre-determined group G. """
        self.add_finalize_group(self.currentGroup, VerifyPanel.YES_IDX)
        self.remove_group(self.currentGroup)
        if self.is_done_verifying():
            self.currentGroup = None
            self.done_verifying()
        else:
            self.select_group(self.queue[0])
        
    def OnClickNo(self, event):
        """ USED FOR MODE_YESNO. Indicates that the currentGroup does
        NOT correspond to some pre-determined group G. """
        self.add_finalize_group(self.currentGroup, VerifyPanel.OTHER_IDX)
        self.remove_group(self.currentGroup)
        if self.is_done_verifying():
            self.currentGroup = None
            self.done_verifying()
        else:
            self.select_group(self.queue[0])

    def OnClickLabelManually(self, event):
        """ USED FOR MODE_YESNO2. Indicates that the user wants to 
        manually label everything in this group, say, because the current
        group has too many different types, and doesn't want to bother
        repeatedly-performing Splits. """
        self.currentGroup.is_manual = True
        self.add_finalize_group(self.currentGroup, VerifyPanel.OTHER_IDX)
        self.remove_group(self.currentGroup)
        if self.is_done_verifying():
            self.currentGroup = None
            self.done_verifying()
        else:
            self.select_group(self.queue[0])

    def OnClickRunDigitGroup(self, evt):
        """ Used for MODE_NORMAL, for digitbased attributes. Signals
        that the user wants to re-run the digitparsing.
        """
        def get_digitattrtypes(project):
            attrs = pickle.load(open(project.ballot_attributesfile, 'rb'))
            digitattrs = []
            for attr in attrs:
                attrtypestr = common.get_attrtype_str(attr['attrs'])
                if common.is_digitbased(project, attrtypestr):
                    digitattrs.append(attrtypestr)
            return digitattrs
        grouplabel = self.currentGroup.getcurrentgrouplabel()
        digitattrs = get_digitattrtypes(self.project)
        if not digitattrs:
            dlg = wx.MessageDialog(self, message="'Run Digit Group' isn't \
supported for non-digitbased attributes. Perhaps you'd like to quarantine \
this instead?", style=wx.OK)
            self.Disable()
            dlg.ShowModal()
            self.Enable()
            return

        if len(digitattrs) != 1:
            print "Sorry, OpenCount only supports one digit-based attribute \
at a time."
            pdb.set_trace()
            assert False

        attrtypestr = digitattrs[0]  # Assume only one digit-based attr
        num_digits = common.get_numdigits(self.project, attrtypestr)
        w_img, h_img = self.project.imgsize
            
        bal2imgs = pickle.load(open(self.project.ballot_to_images, 'rb'))
        # a.) Reconstruct digit_attrs
        digit_attrs = {} # maps {str attrtype: ((y1,y2,x1,x2),side)}
        attrs = pickle.load(open(self.project.ballot_attributesfile, 'rb'))
        for attrdict in attrs:
            attrstr = common.get_attrtype_str(attrdict['attrs'])
            if common.is_digitbased(self.project, attrstr):
                y1 = int(round(attrdict['y1']*h_img))
                y2 = int(round(attrdict['y2']*h_img))
                x1 = int(round(attrdict['x1']*w_img))
                x2 = int(round(attrdict['x2']*w_img))
                side = attrdict['side']
                digit_attrs[attrstr] = ((y1, y2, x1, x2), side)
        if len(digit_attrs) != 1:
            print "Uhoh, len(digit_attrs) should have been 1, but wasn't."
            pdb.set_trace()
        assert len(digit_attrs) == 1
        # b.) Construct rejected_hashes
        #cur_digit = common.get_propval(self.currentGroup.getcurrentgrouplabel(), 'digit')
        # rejected_hashes maps {imgpath: {digit: [((y1,y2,x1,x2),side_i), ...]}}
        rejected_hashes = partmatch_fns.get_rejected_hashes(self.project)
        if rejected_hashes == None:
            # Hasn't been created yet.
            rejected_hashes = {}
            pickle.dump(rejected_hashes, open(pathjoin(self.project.projdir_path,
                                                       self.project.rejected_hashes),
                                              'wb'))
        ct = 0
        for imgpath, digitsmap in rejected_hashes.iteritems():
            for digit, lst in digitsmap.iteritems():
                ct += len(lst)
        print "Number of rejected regions:", ct

        partmatch_fns.save_rejected_hashes(self.project, rejected_hashes)
        if len(rejected_hashes) == 0:
            print "No need to re-run partmatch, rejected_hashes is empty."
            return
        # c.) Grab accepted_hashes
        # accepted_hashes: {str imgpath: {str digit: [((y1,y2,x1,x2), side_i), ...]}}
        accepted_hashes = partmatch_fns.get_accepted_hashes(self.project)
        if accepted_hashes == None:
            # Hasn't been created yet.
            accepted_hashes = {}
            partmatch_fns.save_accepted_hashes(self.project, accepted_hashes)
        accept_cnt = 0
        for imgpath, digitsmap in accepted_hashes.iteritems():
            for digit, lst in digitsmap.iteritems():
                accept_cnt += len(lst)
        print "Total number of accepted regions:", accept_cnt

        print "Running partmatch digit-OCR computation with updated \
rejected_hashes..."
        digitgroup_results, digitmatch_info = digit_group.do_digitocr_patches(bal2imgs, digit_attrs, self.project,
                                                                              rejected_hashes=rejected_hashes,
                                                                              accepted_hashes=accepted_hashes)
        digit_group.save_digitgroup_results(self.project, digitgroup_results)
        digit_group.save_digitmatch_info(self.project, digitmatch_info)
        groups = digit_group.to_groupclasses_digits(self.project, digitgroup_results)
        print "Finished partmatch digit-OCR. Number of groups:", len(groups)

        # Replace my internal groups (self.queue, etc.) with the
        # GroupClass's given in GROUPS.
        # 1.) First, remove all 'digit' Groups
        for group in self.queue[:]:
            # TODO: Assumes a GroupClass with a grouplabel with 'digit'
            # signals that this is a Digit. This disallows:
            #    a.) Multiple digit-based attributes
            #    b.) A Ballot Attribute called 'digit'
            for grouplabel in group.orderedAttrVals:
                if common.get_propval(grouplabel, 'digit') != None:
                    self.remove_group(group)
                    break
        for group in self.finished[:]:
            for grouplabel in group.orderedAttrVals:
                if common.get_propval(grouplabel, 'digit') != None:
                    self.finished.remove(group)
                    break
        # 2.) Now, add in all new 'digit' Groups
        # TODO: Discard all matches that deal with already-verified
        #       patches, or tell partmatch to not search these imgs.
        for new_digitgroup in groups:
            self.add_group(new_digitgroup)
        self.select_group(self.queue[0])

        self._mismatch_cnt = 0
        self.misclassify_txt.SetLabel("Mismatches in queue: 0")
        
    def OnClickMisclassify(self, evt):
        """ Used for MODE_NORMAL. Signals that the current attr patch
        is misclassified (say, the wrong attribute type), and to handle
        it *somehow*.
        For digit-based attributes, this will update the rejected_hashes
        data structure.
        TODO: Assumes that groups representing a digitbased attribute
              will have a grouplabel with a kv-pair whose key is 'digit',
              and whose value is the digit string ('0','1',etc.). Lousy
              assumption, since I think this restricts the architecture
              to only allowing one digit-based attribute.
        """
        def get_digitattrtypes(project):
            attrs = pickle.load(open(project.ballot_attributesfile, 'rb'))
            digitattrs = []
            for attr in attrs:
                attrtypestr = common.get_attrtype_str(attr['attrs'])
                if common.is_digitbased(project, attrtypestr):
                    digitattrs.append(attrtypestr)
            return digitattrs
        grouplabel = self.currentGroup.getcurrentgrouplabel()
        if common.get_propval(grouplabel, 'digit') == None:
            dlg = wx.MessageDialog(self, message="'Misclassify' isn't \
supported for non-digitbased attributes. Perhaps you'd like to quarantine \
this instead?", style=wx.OK)
            self.Disable()
            dlg.ShowModal()
            self.Enable()
            return
        digitattrs = get_digitattrtypes(self.project)
        if not digitattrs:
            print "Uhoh, digitattrs was empty, when it shouldn't be."
            pdb.set_trace()
        assert len(digitattrs) > 0
        if len(digitattrs) != 1:
            print "Sorry, OpenCount only supports one digit-based attribute \
at a time."
            pdb.set_trace()
            assert False

        attrtypestr = digitattrs[0]  # Assume only one digit-based attr
        num_digits = common.get_numdigits(self.project, attrtypestr)
        w_img, h_img = self.project.imgsize
            
        bal2imgs = pickle.load(open(self.project.ballot_to_images, 'rb'))
        print "==== a.) Reconstruct digit attrs"
        # a.) Reconstruct digit_attrs
        digit_attrs = {} # maps {str attrtype: ((y1,y2,x1,x2),side)}
        attrs = pickle.load(open(self.project.ballot_attributesfile, 'rb'))
        for attrdict in attrs:
            attrstr = common.get_attrtype_str(attrdict['attrs'])
            if common.is_digitbased(self.project, attrstr):
                y1 = int(round(attrdict['y1']*h_img))
                y2 = int(round(attrdict['y2']*h_img))
                x1 = int(round(attrdict['x1']*w_img))
                x2 = int(round(attrdict['x2']*w_img))
                side = attrdict['side']
                digit_attrs[attrstr] = ((y1, y2, x1, x2), side)
        if len(digit_attrs) != 1:
            print "Uhoh, len(digit_attrs) should have been 1, but wasn't."
            pdb.set_trace()
        assert len(digit_attrs) == 1
        # b.) Construct rejected_hashes
        print "==== b.) Construct rejected_hashes"
        cur_digit = common.get_propval(self.currentGroup.getcurrentgrouplabel(), 'digit')
        # rejected_hashes maps {imgpath: {digit: [((y1,y2,x1,x2),side_i), ...]}}
        rejected_hashes = partmatch_fns.get_rejected_hashes(self.project)
        if rejected_hashes == None:
            # Hasn't been created yet.
            rejected_hashes = {}
            pickle.dump(rejected_hashes, open(pathjoin(self.project.projdir_path,
                                                       self.project.rejected_hashes),
                                              'wb'))
        print "== Throw stuff in self.currentGroup.elements into rejected_hashes"
        for (sampleid, rlist, patchpath) in self.currentGroup.elements:
            # TODO: Do I append sampleid, or patchpath? 
            # TODO: Is it sampleid, or imgpath?
            #rejected_hashes.setdefault(sampleid, {})[cur_digit] = digit_attrs[attrtypestr]
            rejected_hashes.setdefault(sampleid, {}).setdefault(cur_digit, []).append(digit_group.get_digitmatch_info(self.project, patchpath))
        print "== Saving rejected_hashes"
        partmatch_fns.save_rejected_hashes(self.project, rejected_hashes)
        print "== Counting..."
        ct = 0
        for imgpath, digitsmap in rejected_hashes.iteritems():
            for digit, lst in digitsmap.iteritems():
                ct += len(lst)
        print "Total Number of rejected regions:", ct
        self._mismatch_cnt += len(self.currentGroup.elements)
        self.misclassify_txt.SetLabel("Mismatches in queue: {0}".format(self._mismatch_cnt))

        # Remove the current group, and display the next one
        self.remove_group(self.currentGroup)
        self.select_group(self.queue[0])
        
    def is_done_verifying(self):
        return not self.queue
        
    def done_verifying(self):
        """
        When the user has finished verifying all groups, do some
        fancy computing, and output results.
        Outputs grouping results into the specified out-directory, if
        given.
        """
        # First populate results
        print "DONE Verifying!"
        self.Disable()
        results = {} # {grouplabel: list of GroupClasses}
        if self.mode == VerifyPanel.MODE_YESNO2:
            # Hack: Treat each GroupClass as separate categories,
            # instead of trying to merge them. However, we do
            # merge the GROUPLABEL_MANUAL ones.
            # TODO: In theory, we could /try/ to
            # merge GroupClasses from the same category, but this isn't
            # trivial -- maybe we'd perform another clustering run on
            # exemplars from each new cluster, but it doesn't seem
            # worth it at the moment.
            for group in self.finished:
                grouplabel = group.orderedAttrVals[0]
                if grouplabel in results:
                    print "Uhoh, gropulabel was in results more than once."
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
            """ Only used in MODE_YESNO2. """
            ids = {} # {str attrname: list of ids}
            groups = tuple(newGroups) + tuple(self.queue) + tuple(self.finished)

            for group in groups:
                # In MODE_YESNO2, foo is a list of the form:
                #    ((<attrtype>,), ID)

                # TODO: group.orderedAttrVals[0] seems troubling. Shouldn't
                # I be using the group.index or something? Or is it guaranteed
                # that group.orderedAttrVals[0] is always the 'current-best'
                # guess?
                #foo = list(group.getcurrentgrouplabel())
                foo = list(group.orderedAttrVals[0])
                attrtype = tuple(sorted([t[0] for t in foo]))
                id = foo[0][1]
                ids.setdefault(attrtype, []).append(id)
            return ids
        def assign_new_id(group, ids):
            """ Given a new GroupClass, and a previous IDS mapping,
            find a unique new id for group to use.
            Only used in MODE_YESNO2.
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

