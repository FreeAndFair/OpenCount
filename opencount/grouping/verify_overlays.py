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

        if not verifymode:
            self.mode = VerifyPanel.MODE_NORMAL
        else:
            self.mode = verifymode

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.patchsizer = None
        
        # List of groups to-be-verified
        self.queue = []
        self.currentGroup = None
        # List of groups that have been verified
        self.finished = []
        self._rows = None# list of csv rows used for exportResults
        # A dict mapping {str temppath: list of ((y1,y2,x1,x2), attr_type, attr_val, str side)}
        self.patches = {}
        
        self.resultsPath = None
        self.csvdir = None
        
        # templates is a dict mapping
        #    {str attrtype: 
        #      {str attrval: obj attrpatch_img}}
        # where attrpatch_img is the image patch on the template corresponding to the
        # attrtype attribute type.
        self.templates = None
        
        self.canMoveOn = False
        self.mainPanel = scrolled.ScrolledPanel(self, size=self.GetSize(), pos=(0,0))

        self.initLayout()
        self.initBindings()
        
        Publisher().subscribe(self._pubsub_project, "broadcast.project")

    def fitPanel(self):
        w, h = self.parent.GetClientSize()
        self.mainPanel.SetMinSize((w * 0.95, h * 0.9))
        self.mainPanel.GetSizer().SetSizeHints(self)
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
        self.sizer.Add(self.mainPanel, flag=wx.EXPAND)
        self.mainPanel.Hide()
        self.SetSizer(self.sizer, deleteOld=False)

    def onSelect_queuelistbox(self, evt):
        idx = evt.GetSelection()
        if idx >= 0:
            assert idx < len(self.queue)
            self.select_group(self.queue[idx])
            
    def start(self, groups, patches):
        """
        Start verifying the overlays. Groups is a list of 
        GroupClass objects, representing pre-determined clusters
        within a data set.
        patches is a dict that contains information about each possible
        group:
          {str temppath_i: 
            (((x1,y1,x2,y2), patchtype_i, patchval_i, imgpath_i), ...)}
        """
        self.patches = patches
        for group in groups:
            self.add_group(group)
        self.start_verifygrouping()
    
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

        self.getTemplates()

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
        csvfilepath = self.resultsPath
        csvfile = open(csvfilepath, 'wb')
        dictwriter = csv.DictWriter(csvfile, fieldnames=fields)
        try:
            dictwriter.writeheader()
        except AttributeError:
            util_gui._dictwriter_writeheader(csvfile, fields)
        dictwriter.writerows(self._rows)
    
    def updateTemplateThumb(self):
        """
        Updates the 'Attribute Patch' and 'Diff' image patches.
        """
        overlayMin = self.currentGroup.overlayMin
        overlayMax = self.currentGroup.overlayMax
        templates = self.currentGroup.orderedAttrVals
        elements = self.currentGroup.elements

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
        for (attrval, flipped, imageorder, foo) in ordered_attrvals:
            if attrval not in history:
                display_string = attrval
                self.templateChoice.Append(display_string)
                history.add(attrval)
        
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
        assert len(self.patches) == 2
        self.add_finalize_group(self.currentGroup, VerifyPanel.YES_IDX)
        self.remove_group(self.currentGroup)
        if self.is_done_verifying():
            self.currentGroup = None
            self.done_verifying()
        else:
            self.select_group(self.queue[0])
        
    def OnClickNo(self, event):
        """ USED FOR MODE_YESNO """
        assert len(self.patches) == 2
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
        fancy computing.
        """
        # First populate results
        print "DONE Verifying!"
        self.Disable()
        results = {}
        for group in self.finished:
            elements = group.elements
            attrtype = group.attrtype
            index = group.index
            for samplepath, attrs_list in elements:
                results.setdefault(samplepath, {})[attrtype] = attrs_list[index]

        attr_types = set(common.get_attrtypes(self.project))
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
        munged_patches = munge_patches(self.patches,
                                       util.is_multipage(self.project),
                                       img2tmp)
        for samplepath, attrdict in results.items():
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
        correctedflips = fix_ballot_to_images(self.project, bal2tmp, sample_attrmap, self.patches, sample_flips)
        # but always 'correct' the flipinfo, even for single page elections
        add_flipinfo(self.project, correctedflips, fields, self.resultsPath)
        
    
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
        
    
    def checkCanMoveOn(self):
        # TODO: Fix this implementation.
        return True

        return self.canMoveOn

        
    def _pubsub_project(self, msg):
        project = msg.data
        self.project = project
        
        self.csvdir = project.patch_loc_dir
        self.resultsPath = project.grouping_results
        
        self.project.addCloseEvent(self.exportResults)
        
    def OnSize(self, event):
        self.fitPanel()
        event.Skip()

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
    Output:
      dict result: {str temppath: {str attrype: (str attrval, int side)}}
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
