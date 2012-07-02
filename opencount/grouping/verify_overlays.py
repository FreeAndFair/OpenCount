import sys, csv, copy, pdb, os
import threading, time
import timeit
sys.path.append('../')

from util import MyGauge
from specify_voting_targets import util_gui as util_gui
from specify_voting_targets import imageviewer as imageviewer
from specify_voting_targets.util_gui import *
import label_attributes, util, common

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
    """
    MODE_NORMAL = 0
    MODE_YESNO  = 1

    YES_IDX = 0
    OTHER_IDX = 1

    def __init__(self, parent, verifymode=None, *args, **kwargs):
        wx.Panel.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.project = None
        self.ondone = None  # An optional callback function to call
                            # when verifying is done

        if not verifymode:
            self.mode = VerifyPanel.MODE_NORMAL
        elif verifymode == VerifyPanel.MODE_YESNO:
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

        # A dict mapping {str temppath: list of ((y1,y2,x1,x2), grouplabel)}
        self.patches = {}
        
        self.resultsPath = None
        
        # templates is a dict mapping
        #    {grouplabel: obj attrpatch_img}
        # where attrpatch_img is the image patch on the template corresponding to the
        # grouplabel
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
        self.quarantineButton = wx.Button(self.mainPanel, label='Quarantine')
        # Buttons for MODE_YESNO
        self.yes_button = wx.Button(self.mainPanel, label="Yes")
        self.no_button = wx.Button(self.mainPanel, label="No")
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
        hbox5.Add(self.yes_button, flag=wx.LEFT | wx.CENTRE)
        hbox5.Add((40,-1))
        hbox5.Add(self.no_button, flag=wx.LEFT | wx.CENTRE)

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
        elif self.mode == VerifyPanel.MODE_YESNO:
            self.okayButton.Hide()
            self.quarantineButton.Hide()

        self.mainPanel.SetSizer(hbox7)
        self.sizer.Add(self.mainPanel, proportion=1, flag=wx.EXPAND)
        self.mainPanel.Hide()
        self.SetSizer(self.sizer, deleteOld=False)

    def onSelect_queuelistbox(self, evt):
        idx = evt.GetSelection()
        if idx >= 0:
            assert idx < len(self.queue)
            self.select_group(self.queue[idx])
            
    def start(self, groups, patches, exemplar_paths, outfilepath=None, ondone=None):
        """
        Start verifying the overlays. Groups is a list of 
        GroupClass objects, representing pre-determined clusters
        within a data set.
        patches is a dict that contains information about each possible
        group:
          {str temppath_i: 
            (((x1,y1,x2,y2), patchtype_i, patchval_i, imgpath_i), ...)}
        exemplar_paths is a dict that maps each attrtype->attrval mapping
        to an exemplar image patch:
          {str attrtype: {str attrval: str imgpath}}
        outdir specifies where the grouping results get saved to.
        """
        self.patches = patches
        self.load_exemplar_attrpatches(exemplar_paths)
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
        self.templates = {}
        for grouplabel, patchpath in exemplar_paths.iteritems():
            imgpatch = misc.imread(patchpath, flatten=1)
            rszFac = sh.resizeOrNot(imgpatch.shape, sh.MAX_PRECINCT_PATCH_DISPLAY)
            self.templates[grouplabel] = fastResize(imgpatch, rszFac)/255.0
    
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
 
        # also load in self.patches
        self.patches = common.importPatches(self.project)

    def start_verifygrouping(self):
        """
        Called after sample ballots have been grouped by Kai's grouping
        code. Sets up UI widgets of the verification UI.
        """
        self.SetSizer(self.sizer, deleteOld=False)

        if self.mode == VerifyPanel.MODE_YESNO:
            # Add a dummy group to each GroupClass
            num_descs = 0
            for group in self.queue:
                for element in group.elements:
                    num_descs = len(element[1][0]) - 1
                    break
                break
            for group in self.queue:
                for element in group.elements:
                    element[1].append((('othertype','otherval'),) + (('dummy',None),)*num_descs)
            
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
        overlayMin = self.currentGroup.overlayMin
        overlayMax = self.currentGroup.overlayMax
        templates = self.currentGroup.orderedAttrVals
        elements = self.currentGroup.elements

        idx = self.templateChoice.GetSelection()
        curgrouplabel = self.currentGroup.orderedAttrVals[idx]

        attrpatch_img = self.templates[curgrouplabel]
        
        height, width = attrpatch_img.shape
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
                display_string = str(grouplabel)
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
        assert len(self.patches) == 1
        self.add_finalize_group(self.currentGroup, VerifyPanel.YES_IDX)
        self.remove_group(self.currentGroup)
        if self.is_done_verifying():
            self.currentGroup = None
            self.done_verifying()
        else:
            self.select_group(self.queue[0])
        
    def OnClickNo(self, event):
        """ USED FOR MODE_YESNO """
        assert len(self.patches) == 1
        self.add_finalize_group(self.currentGroup, VerifyPanel.OTHER_IDX)
        self.remove_group(self.currentGroup)
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
        fancy computing, and output results.
        Outputs grouping results into the specified out-directory,
        where each group gets outputted to an output file.
        """
        # First populate results
        print "DONE Verifying!"
        self.Disable()
        results = {} # {grouplabel: elements}
        for group in self.finished:
            results.setdefault(group.getcurrentgrouplabel(), []).extend(group.elements)
        for grouplabel in self.templates:
            if grouplabel not in results:
                results[grouplabel] = []

        if self.outfilepath:
            pickle.dump(results, open(self.outfilepath, 'wb'))

        if self.ondone:
            self.ondone(results)

    def OnClickSplit(self, event):
        newGroups = self.currentGroup.split()
        for group in newGroups:
            self.add_group(group)
        
        self.remove_group(self.currentGroup)
        self.select_group(self.queue[0])
        self.queueList.Fit()
        self.Fit()
        
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

