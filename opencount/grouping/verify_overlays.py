import sys, pdb, os, textwrap, traceback
import time
sys.path.append('../')

from specify_voting_targets.util_gui import *
import common, partmatch_fns, digit_group, label_imgs

from pixel_reg.imagesAlign import *
import pixel_reg.shared as sh

####
## Import 3rd party libraries
####
import wx
import wx.lib.scrolledpanel as scrolled
import numpy as np
from scipy import misc
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

Digit-related Output Files

- <projdir>/accepted_hashes.p
- <projdir>/rejected_hashes.p

These store the regions on voted ballots for which the user clicked 'Ok'
or 'Misclassified' for. Dictionaries are of the form:
    {str imgpath: {str digit: [(bb_i, side, isflip), ...]}}

- <projdir>/digitgroup_results.p

This is the result of digit_group.do_digitocr_patches. Is a dict of the
form:
    {str ballotid: [(digitattr_i, ocrstr_i, meta_i, isflip_i, side_i), ...]}
where meta_i is numDigits-tuples of the form:
    [(y1,y2,x1,x2, digit_i, outpath_i, score_i), ...]

- <projdir>/digitmatch_info.p

This is from the result of digit_group.do_digitocr_patches, and maps
{str patchpath: (bb, side, isflip, ballotid)}

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

class VerifyPanel(scrolled.ScrolledPanel):
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
        scrolled.ScrolledPanel.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.project = None
        self.ondone = None  # An optional callback function to call
                            # when verifying is done
        self.outfilepath = None   # An optional filepath to output
                                  # grouping results to.

        # A 'canonical' list of grouplabels to use - the idxs in the
        # rlists index into this list.
        self.grouplabel_record = None

        ## The following are only used for digit-based attributes 

        # self._mismatch_cnt: Keeps track of the current number of 'new'
        # MisClassified overlays currently in the queue.
        self._mismatch_cnt = 0
        # self._ok_history: Keeps track of how many times the user clicked
        # 'Ok', for a voted ballot B
        self._ok_history = {} # maps {str votedpath: int count}
        # self._misclassify_history
        self._misclassify_history = {} # maps {str ballotid: int count}

        self.accepted_hashes = None
        self.rejected_hashes = None
        self.digitmatch_info = None
            
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
        
        # Mode for splitting algorithm choice
        # Options are: rankedlist, k-means, PCA+k-means
        # Defaults to rankedlist
        self.splitMode = 'rankedlist'  
        
        # templates is a dict mapping
        #    {grouplabel: str attrpatch_img}
        # where attrpatch_img is the image patch on the template 
        # corresponding to the grouplabel
        self.templates = None
        
        self.canMoveOn = False
        #self.mainPanel = scrolled.ScrolledPanel(self, size=self.GetSize(), pos=(0,0))
        self.initLayout()
        self.initBindings()

        Publisher().subscribe(self._pubsub_project, "broadcast.project")

    def load_accepted_hashes(self):
        if self.accepted_hashes != None:
            return self.accepted_hashes
        accepted_hashes = partmatch_fns.get_accepted_hashes(self.project)
        if accepted_hashes == None:
            accepted_hashes = {}
            partmatch_fns.save_accepted_hashes(self.project, accepted_hashes)
        self.accepted_hashes = accepted_hashes
        return self.accepted_hashes
    def save_accepted_hashes(self):
        partmatch_fns.save_accepted_hashes(self.project, self.accepted_hashes)
        
    def load_rejected_hashes(self):
        if self.rejected_hashes != None:
            return self.rejected_hashes
        rejected_hashes = partmatch_fns.get_rejected_hashes(self.project)
        if rejected_hashes == None:
            rejected_hashes = {}
            partmatch_fns.save_rejected_hashes(self.project, rejected_hashes)
        self.rejected_hashes = rejected_hashes
        return self.rejected_hashes
    def save_rejected_hashes(self):
        partmatch_fns.save_rejected_hashes(self.project, self.rejected_hashes)
        
    def load_digitmatch_info(self):
        if self.digitmatch_info != None:
            return self.digitmatch_info
        digitmatch_info = digit_group.get_digitmatch_info(self.project)
        self.digitmatch_info = digitmatch_info
        return self.digitmatch_info
    def save_digitmatch_info(self, digitmatch_info):
        self.digitmatch_info = digitmatch_info
        digit_group.save_digitmatch_info(self.project, self.digitmatch_info)
        
    def fitPanel(self):
        #w, h = self.parent.GetClientSize()
        #self.mainPanel.SetMinSize((w * 0.95, h * 0.9))
        #self.mainPanel.GetSizer().SetSizeHints(self)
        self.Layout()
        self.SetupScrolling()
    
    def overlays_layout_vert(self):
        """ Layout the overlay patches s.t. there is one row of N columns. 
        Typically called when the patch height > patch width.
        """
        self.sizer_overlays.SetOrientation(wx.VERTICAL)
        self.sizer_overlays_voted.SetOrientation(wx.HORIZONTAL)
        self.sizer_min.SetOrientation(wx.VERTICAL)
        self.sizer_max.SetOrientation(wx.VERTICAL)
        self.sizer_attrpatch.SetOrientation(wx.VERTICAL)
        self.sizer_diff.SetOrientation(wx.VERTICAL)

    def overlays_layout_horiz(self):
        """ Layout the overlay patches s.t. there are N rows of 1 column.
        Typically called when the patch width > patch height.
        """
        self.sizer_overlays.SetOrientation(wx.HORIZONTAL)
        self.sizer_overlays_voted.SetOrientation(wx.VERTICAL)
        self.sizer_min.SetOrientation(wx.HORIZONTAL)
        self.sizer_max.SetOrientation(wx.HORIZONTAL)
        self.sizer_attrpatch.SetOrientation(wx.HORIZONTAL)
        self.sizer_diff.SetOrientation(wx.HORIZONTAL)

    def set_patch_layout(self, orient='horizontal'):
        """
        Change the orientation of the overlay patch images. Either
        arrange 'horizontal', or stack 'vertical'.
        """
        if orient == 'horizontal':
            print 'HORIZ'
            sizer = self.overlays_layout_horiz()
        else:
            print 'VERT'
            sizer = self.overlays_layout_vert()
        self.Layout()
        self.Refresh()

    def initLayout(self):
        st1 = wx.StaticText(self, -1, "min: ")
        st2 = wx.StaticText(self, -1, "max: ")
        st3 = wx.StaticText(self, -1, "Looks like? ")
        st4 = wx.StaticText(self, -1, "diff: ")
        self.st1, self.st2, self.st3, self.st4 = st1, st2, st3, st4

        self.minOverlayImg = wx.StaticBitmap(self, bitmap=wx.EmptyBitmap(1, 1))
        self.maxOverlayImg = wx.StaticBitmap(self, bitmap=wx.EmptyBitmap(1, 1))
        self.templateImg = wx.StaticBitmap(self, bitmap=wx.EmptyBitmap(1, 1))
        self.diffImg = wx.StaticBitmap(self, bitmap=wx.EmptyBitmap(1, 1))

        maxTxtW = max([txt.GetSize()[0] for txt in (st1, st2, st3, st4)]) + 20

        sizer_overlays = wx.BoxSizer(wx.HORIZONTAL)
        self.sizer_overlays = sizer_overlays
        self.sizer_overlays_voted = wx.BoxSizer(wx.VERTICAL)
        self.sizer_min = wx.BoxSizer(wx.HORIZONTAL)
        self.sizer_min.AddMany([(st1,), ((maxTxtW-st1.GetSize()[0],0),), (self.minOverlayImg,)])
        self.sizer_max = wx.BoxSizer(wx.HORIZONTAL)
        self.sizer_max.AddMany([(st2,), ((maxTxtW-st2.GetSize()[0],0),), (self.maxOverlayImg,)])
        self.sizer_attrpatch = wx.BoxSizer(wx.HORIZONTAL)
        self.sizer_attrpatch.AddMany([(st3,), ((maxTxtW-st3.GetSize()[0],0),), (self.templateImg,)])
        self.sizer_diff = wx.BoxSizer(wx.HORIZONTAL)
        self.sizer_diff.AddMany([(st4,), ((maxTxtW-st4.GetSize()[0],0),), (self.diffImg,)])
        self.sizer_overlays_voted.AddMany([(self.sizer_min,), ((50, 50),), (self.sizer_max,), ((50, 50),),
                                           (self.sizer_diff,)])
        self.sizer_overlays.AddMany([(self.sizer_overlays_voted,), ((50, 50),),
                                     (self.sizer_attrpatch, 0, wx.ALIGN_CENTER)])
        self.set_patch_layout('horizontal')

        self.templateChoice = wx.ComboBox(self, choices=[], style=wx.CB_READONLY)
        self.okayButton = wx.Button(self, label='OK')

        self.splitButton = wx.Button(self, label='Split')
        self.SetSplitModeButton = wx.Button(self, label='Set Split Mode')
        self.mergeButton = wx.Button(self, label='Merge Groups...')
        sizer_splitmerge = wx.BoxSizer(wx.VERTICAL)
        sizer_splitmerge.Add(self.splitButton, flag=wx.LEFT | wx.CENTRE)
        sizer_splitmerge.Add(self.SetSplitModeButton, flag=wx.LEFT | wx.CENTRE)
        sizer_splitmerge.Add((20, 20))
        sizer_splitmerge.Add(self.mergeButton, flag=wx.LEFT | wx.CENTRE)

        misclassify_sizer = wx.BoxSizer(wx.VERTICAL)
        self.misclassifyButton = wx.Button(self, label="Mis-classified")
        self.misclassifyButton.Bind(wx.EVT_BUTTON, self.OnClickMisclassify)
        self.misclassify_txt = wx.StaticText(self, label="Mismatches \
in queue: 0")
        self.manuallabelallButton =  wx.Button(self, label="Label All Manually...")
        self.manuallabelallButton.Bind(wx.EVT_BUTTON, self.OnClickManualLabelAll)
        misclassify_sizer.Add(self.misclassifyButton)
        misclassify_sizer.Add((20, 20))
        misclassify_sizer.Add(self.misclassify_txt)
        misclassify_sizer.Add((20, 20))
        misclassify_sizer.Add(self.manuallabelallButton)

        digitgroup_sizer = wx.BoxSizer(wx.VERTICAL)
        self.rundigitgroupButton = wx.Button(self, label="Run Digit Grouping")
        self.rundigitgroupButton.Bind(wx.EVT_BUTTON, self.OnClickRunDigitGroup)
        self.forcedigitgroupButton = wx.Button(self, label="Force Digit Grouping")
        self.forcedigitgroupButton.Bind(wx.EVT_BUTTON, self.OnClickForceDigitGroup)
        digitgroup_sizer.Add(self.rundigitgroupButton)
        digitgroup_sizer.Add((20, 20))
        digitgroup_sizer.Add(self.forcedigitgroupButton)
        quarantine_sizer = wx.BoxSizer(wx.VERTICAL)
        self.viewImgButton = wx.Button(self, label="View Image...")
        self.debugButton = wx.Button(self, label='DEBUG')
        self.quarantineButton = wx.Button(self, label='Quarantine')
        quarantine_sizer.AddMany([(self.viewImgButton,), (self.debugButton,), (self.quarantineButton,)])
        self.viewImgButton.Bind(wx.EVT_BUTTON, self.onButton_viewimg)

        checkpoint_sizer = wx.BoxSizer(wx.VERTICAL)
        self.saveCheckpointButton = wx.Button(self, label="Save Checkpoint")
        self.saveCheckpointButton.Bind(wx.EVT_BUTTON, self.OnClickSaveCheckpoint)
        self.loadCheckpointButton = wx.Button(self, label="Load Checkpoint...")
        self.loadCheckpointButton.Bind(wx.EVT_BUTTON, self.OnClickLoadCheckpoint)
        self.undoButton = wx.Button(self, label="Undo.")
        self.undoButton.Bind(wx.EVT_BUTTON, self.OnClickUndo)
        self.restoreallButton = wx.Button(self, label="Restore All Groups...")
        self.restoreallButton.Bind(wx.EVT_BUTTON, self.OnClickRestoreAll)
        checkpoint_sizer.AddMany([(self.saveCheckpointButton,), ((10,10),), 
                                  (self.loadCheckpointButton,), ((10,10),),
                                  (self.undoButton,), ((10,10),),
                                  (self.restoreallButton,)])

        # Buttons for MODE_YESNO
        self.yes_button = wx.Button(self, label="Yes")
        self.no_button = wx.Button(self, label="No")
        # Buttons for MODE_YESNO2
        self.manuallylabelButton = wx.Button(self, label='Manually Label This Group')
        self.manuallylabelButton.Bind(wx.EVT_BUTTON, self.OnClickLabelManually)

        sizer_btns = wx.BoxSizer(wx.HORIZONTAL)
        sizer_btns.AddMany([(self.templateChoice,), (self.okayButton,),
                            (sizer_splitmerge,), (misclassify_sizer,),
                            (digitgroup_sizer,), (quarantine_sizer,),
                            (checkpoint_sizer,),
                            (self.yes_button,), (self.no_button,),
                            (self.manuallylabelButton,)])

        st5 = wx.StaticText(self, -1, "# of ballots in the group: ", style=wx.ALIGN_LEFT)
        self.tNumBallots = wx.TextCtrl(self, value='0')
        self.tNumBallots.SetEditable(False)
        
        self.queueList = wx.ListBox(self, size=(-1, 140))
        self.queueList.Bind(wx.EVT_LISTBOX, self.onSelect_queuelistbox)
        self.finishedList = wx.ListBox(self)
        self.finishedList.Hide() # Not using this for UI simplicity
        
        sizer_queue = wx.BoxSizer(wx.VERTICAL)
        sizer_numballots = wx.BoxSizer(wx.HORIZONTAL)
        sizer_numballots.AddMany([(st5,), (self.tNumBallots,)])
        sizer_queue.AddMany([(sizer_numballots,), (self.queueList,)])

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
            self.quarantineButton.Hide()
            self.forcedigitgroupButton.Hide()
            self.saveCheckpointButton.Hide()
            self.loadCheckpointButton.Hide()
            self.manuallabelallButton.Hide()
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
            self.quarantineButton.Hide()
            self.forcedigitgroupButton.Hide()
            self.saveCheckpointButton.Hide()
            self.loadCheckpointButton.Hide()
            self.manuallabelallButton.Hide()

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(sizer_queue, proportion=0, flag=wx.EXPAND)
        self.sizer.Add((0, 50))
        self.sizer.Add(sizer_overlays, flag=wx.ALIGN_CENTER)
        self.sizer.Add((0, 100))
        self.sizer.Add(sizer_btns, proportion=0, flag=wx.EXPAND | wx.ALIGN_CENTER)
        self.SetSizer(self.sizer, deleteOld=False)
        self.Layout()
            
        self.Hide()

    def onSelect_queuelistbox(self, evt):
        idx = evt.GetSelection()
        if idx >= 0:
            assert idx < len(self.queue)
            self.select_group(self.queue[idx])
            
    def start(self, groups, exemplar_paths, proj, outfilepath=None, ondone=None,
              grouplabel_record=None):
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
        list grouplabel_record: The canonical list of ordered grouplabels 
            to use. This is what gl_idxs will index into. If it's not
            given, then this will use the 'globally'-defined record,
            which is meant for 'Verify Overlays' at the end of OpenCount.
        """
        if not groups:
            print "Uhoh, there aren't any groups passed into VerifyPanel."
            pdb.set_trace()
        self.project = proj
        if grouplabel_record == None:
            self.grouplabel_record = common.load_grouplabel_record(proj)
        else:
            self.grouplabel_record = grouplabel_record
        if exemplar_paths:
            self.load_exemplar_attrpatches(exemplar_paths)
        else:
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
            print "DUMPING VERIFY GROUP STATE"
            statedict = {}
            if self.currentGroup and self.currentGroup not in self.queue and self.currentGroup not in self.finished:
                # self.currentGroup will be None when verification is done.
                print "Uhoh, why isn't currentGroup anywhere?"
                pdb.set_trace()
            #if self.currentGroup:
            #    if self.currentGroup not in self.queue and self.currentGroup not in self.finished:
            #        q.insert(0, self.currentGroup)
            statedict['todo'] = self.queue
            statedict['finished'] = self.finished
            #if self.currentGroup != None:
            #    statedict['curidx'] = self.queue.index(self.currentGroup)
            #else:
            #    statedict['curidx'] = 0
            #statedict['misclassify_cnt'] = self._mismatch_cnt
            print "Number todo: {0} Number finished: {1}".format(len(self.queue), len(self.finished))
            fqueue = open(pathjoin(self.project.projdir_path, 'verifygroupstate.p'), 'wb')
            print "Dumping statedict..."
            pickle.dump(statedict, fqueue)
            print "...Finished Dumping statedict."
        
    def load_state(self, replacedigits=False, digitgroups=None):
        # TODO: Move the 'verifygroupstate' to the Project class, to 
        #       consolidate all output files into Project.
        if os.path.exists(pathjoin(self.project.projdir_path, 'verifygroupstate.p')):
            try:
                self._mismatch_cnt = 0
                fstate = open(pathjoin(self.project.projdir_path, 'verifygroupstate.p'), 'rb')
                print "Loading statedict..."
                _t = time.time()
                statedict = pickle.load(fstate)
                print "...Finished loading statedict ({0} s).".format(time.time() - _t)
                todo = statedict['todo']
                finished = statedict['finished']
                print "Number todo: {0} Number finished: {1}".format(len(todo), len(finished))
                # TODO: 'curidx' not used at the moment. Might be a useful
                # thing to have at some point.
                if 'curidx' not in statedict:
                    # TODO: Legacy code. Remove me at some future time.
                    curgroupidx = 0
                else:
                    curgroupidx = statedict['curidx']
                # TODO: For Marin backwards compatibility, we'll just recompute
                # the misclassify count. But later, let's just use the misclassify_cnt
                # in the statefile.
                if 'misclassify_cnt' not in statedict:
                    # TODO: Legacy code. Remove me at some future time.
                    misclassify_cnt = None
                else:
                    misclassify_cnt = statedict['misclassify_cnt']

                # 0.) First, clear all my internal state
                self.reset_state()

                img2bal = pickle.load(open(self.project.image_to_ballot, 'rb'))

                todo.reverse() # to not reverse groups in UI
                for group in todo: 
                    # TODO: Code that handles legacy GroupClass instances
                    #       that don't have the self.is_misclassify field.
                    #       Remove me after awhile - is harmless to leave in.
                    if not hasattr(group, 'is_misclassify'):
                        # This is the legacy part
                        group.is_misclassify = False
                    elif replacedigits and type(group) == common.DigitGroupClass:
                        # Skip previous DigitGroupClasses
                        continue
                    elif group.is_misclassify == True:
                        self._mismatch_cnt += len(group.elements)
                    # TODO: Discard user_Data for now.
                    group.user_data = None
                    self.add_group(group)
                for group in finished:
                    if not hasattr(group, 'is_misclassify'):
                        # This is the legacy part
                        group.is_misclassify = False
                    elif replacedigits and type(group) == common.DigitGroupClass:
                        # Skip previous DigitGroupClasses
                        continue
                    elif group.is_misclassify == True:
                        self._mismatch_cnt += len(group.elements)
                        for element in group.elements:
                            balid = img2bal[element[0]]
                            if balid not in self._misclassify_history:
                                self._misclassify_history[balid] = 1
                            else:
                                self._misclassify_history[balid] += 1
                if replacedigits:
                    # Add in new DigitGroups
                    assert digitgroups != None
                    for g in digitgroups:
                        self.add_group(g)
                self.finished = finished
                self.grouplabel_record = common.load_grouplabel_record(self.project)
            except Exception as e:
                # If you can't read in the state file, then just don't
                # load in any state.
                dlg = wx.MessageDialog(self, message="Warning - couldn't \
open the Verify Overlays state file for some reason: {0} \nIf the statefile \
is a nullfile, please delete it, and start over.".format(pathjoin(self.project.projdir_path,
                                                                  'verifygroupstate.p')))
                self.Disable()
                dlg.ShowModal()
                self.Enable()
                traceback.print_exc()
                print e
                return
        # Update the self.misclassify_txt label
        self.misclassify_txt.SetLabel("Mismatches in queue: {0}".format(self._mismatch_cnt))

    def reset_state(self):
        """ Resets all internal state, both data structures and UI
        widgets.
        """
        self._mismatch_cnt = 0
        self.queue = []
        self.currentGroup = None
        self.finished = []
        self.queueList.Clear()

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
            example_gl = self.grouplabel_record[0]
            num_descs = len(example_gl) - 1
            dummy_grouplabel = common.make_grouplabel(*((type, val),) + (('dummy',None),)*num_descs)
            # 0.) Add the dummy_grouplabel to our self.grouplabel_record
            try:
                gl_idx = self.grouplabel_record.index(dummy_grouplabel)
            except:
                gl_idx = len(self.grouplabel_record)
                self.grouplabel_record.append(dummy_grouplabel)
            for group in self.queue:
                #group.orderedAttrVals.append(dummy_grouplabel)
                #for element in group.elements:
                #    element[1].append(dummy_grouplabel)
                group.orderedAttrVals += (gl_idx,)
                new_elements = []
                for sampleid, rlist, patchpath in group.elements:
                    new_elements.append((sampleid, rlist + (gl_idx,), patchpath))
                group.elements = tuple(new_elements)
            
        if self.queue:
            self.select_group(self.queue[0])

        self.Show()
        self.Fit()

    def initBindings(self):
        self.templateChoice.Bind(wx.EVT_COMBOBOX, self.OnSelectTemplate)
        self.Bind(wx.EVT_BUTTON, self.OnClickOK, self.okayButton)
        self.Bind(wx.EVT_BUTTON, self.OnClickSplit, self.splitButton)
        self.Bind(wx.EVT_BUTTON, self.OnClickSetSplitMode, self.SetSplitModeButton)
        self.Bind(wx.EVT_BUTTON, self.OnClickMerge, self.mergeButton)
        self.Bind(wx.EVT_BUTTON, self.OnClickDebug, self.debugButton)
        self.Bind(wx.EVT_BUTTON, self.OnClickQuarantine, self.quarantineButton)
        self.Bind(wx.EVT_BUTTON, self.OnClickYes, self.yes_button)
        self.Bind(wx.EVT_BUTTON, self.OnClickNo, self.no_button)
        self.Bind(wx.EVT_SIZE, self.OnSize)
    
    def updateTemplateThumb(self, votedOverlays=None):
        """
        Updates the 'Attribute Patch' and 'Diff' image patches.
        Input:
            tuple votedOverlays: If given, this will be the Min/Max
                overlays for the current voted ballot: (obj min, obj max).
        """
        if self.mode == VerifyPanel.MODE_YESNO2:
            # We don't have exemplar patches
            return
        if votedOverlays == None:
            overlayMin, overlayMax = self.currentGroup.get_overlays()
        else:
            overlayMin, overlayMax = votedOverlays
            
        templates = self.currentGroup.orderedAttrVals
        elements = self.currentGroup.elements

        idx = self.templateChoice.GetSelection()
        cur_gl_idx = self.currentGroup.orderedAttrVals[idx]
        curgrouplabel = self.grouplabel_record[cur_gl_idx]
        try:
            attrpatch_imgpath = self.templates[curgrouplabel]
            attrpatch_img = misc.imread(attrpatch_imgpath, flatten=1)
            rszFac = sh.resizeOrNot(attrpatch_img.shape, sh.MAX_PRECINCT_PATCH_DISPLAY)
            attrpatch_img = sh.fastResize(attrpatch_img, rszFac) / 255.0
        except Exception as e:
            print "Error in loading exemplar attribute patch."
            traceback.print_exc()
            pdb.set_trace()
        
        h_overlay, w_overlay = overlayMax.shape
        if attrpatch_img.shape != overlayMax.shape:
            attrpatch_img = common.resize_img_norescale(attrpatch_img, (w_overlay, h_overlay))
        IO = imagesAlign(overlayMax, attrpatch_img)
        Iref = np.nan_to_num(IO[1])
        Dabs=np.abs(Iref-attrpatch_img)
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

    def add_misclassify_group(self, group):
        """ Marks a GroupClass as being 'Misclassified.' """
        group.is_misclassify = True
        self.finished.append(group)

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
        gl_idx = group.getcurrentgrouplabel()
        attrtype, attrval = common.get_attrpair_grouplabel(self.project, gl_idx, self.grouplabel_record)
        #self.queueList.Insert(group.label, 0)
        self.queueList.Insert("{0}->{1}: {2} elements".format(attrtype, attrval, len(group.elements)), 0)
        
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
        if w < h:
            self.set_patch_layout('vertical')
        else:
            self.set_patch_layout('horizontal')
            
        self.minOverlayImg.SetBitmap(NumpyToWxBitmap(overlayMin)) #* 255.0))
        self.maxOverlayImg.SetBitmap(NumpyToWxBitmap(overlayMax)) #* 255.0))
        
        self.tNumBallots.SetValue("{0}".format(len(elements)))
        
        self.templateChoice.Clear()
        history = set()
        for gl_idx in ordered_attrvals:
            grouplabel = self.grouplabel_record[gl_idx]
            if grouplabel not in history:
                #display_string = str(grouplabel)
                try:
                    display_string = common.str_grouplabel(gl_idx, self.project, self.grouplabel_record)
                except Exception as e:
                    print e
                    pdb.set_trace()
                self.templateChoice.Append(display_string)
                history.add(grouplabel)
        
        self.templateChoice.SetSelection(self.currentGroup.index)
        
        self.updateTemplateThumb(votedOverlays=(overlayMin, overlayMax))
        
        if (len(elements) <= 1):
            self.splitButton.Disable()
        else:
            self.splitButton.Enable()
        
        #self.parent.Fit() # Causes UI issues
        if self.mode == VerifyPanel.MODE_NORMAL and type(self.currentGroup) == common.DigitGroupClass:
            self.misclassifyButton.Enable()
        else:
            self.misclassifyButton.Disable()
        self.fitPanel()
    
    def OnClickOK(self, event):
        """ Used for MODE_NORMAL. Indicates that the currentGroup is 
        indeed represented by the current exemplar. """
        startTime = time.time()
        times = {} # maps {str job: float duration}
        _t = time.time()
        index = self.templateChoice.GetCurrentSelection()
        self.add_finalize_group(self.currentGroup, index)
        _dur = time.time() - _t
        times['pre_step'] = _dur

        if common.get_propval(self.currentGroup.getcurrentgrouplabel(), 'digit', self.project) != None:
            # For digits-based, update our accepted_hashes.
            # TODO: Assumes that digit-based grouplabels has a key 'digit'
            cur_digit = common.get_propval(self.currentGroup.getcurrentgrouplabel(), 'digit', self.project)
            # accepted_hashes: {str imgpath: {str digit: [((y1,y2,x1,x2), side, isflip), ...]}}
            _t = time.time()
            
            #accepted_hashes = partmatch_fns.get_accepted_hashes(self.project)
            #if accepted_hashes == None:
            #    accepted_hashes = {}
            #    partmatch_fns.save_accepted_hashes(self.project, accepted_hashes)
            accepted_hashes = self.load_accepted_hashes()
            
            #digitmatch_info = digit_group.get_digitmatch_info(self.project)
            digitmatch_info = self.load_digitmatch_info()
            
            _dur = time.time() - _t
            times['load_data'] = _dur
            _t = time.time()
            print "Updating accepted hashes..."
            for (sampleid, rlist, patchpath) in self.currentGroup.elements:
                # digitinfo: ((y1,y2,x1,x2), str side, bool isflip, str ballotid)
                digitinfo = digit_group.get_digitpatch_info(self.project, patchpath, digitmatch_info)
                (bb, side, isflip) = digitinfo[0], digitinfo[1], digitinfo[2]
                accepted_hashes.setdefault(sampleid, {}).setdefault(cur_digit, []).append((bb,side,isflip))
                if sampleid in self._ok_history:
                    self._ok_history[sampleid] += 1
                else:
                    self._ok_history[sampleid] = 1
            _dur = time.time() - _t
            print "...Finished Updating accepted hashes. ({0} s).".format(_dur)
            times['update_acceptedhashes'] = _dur
            _t = time.time()
            #partmatch_fns.save_accepted_hashes(self.project, accepted_hashes)

            self._ok_history
            _dur = time.time() - _t
            times['save_accepted'] = _dur

        _t = time.time()
        self.remove_group(self.currentGroup)
        _dur = time.time() - _t
        times['post_step'] = _dur

        _t = time.time()
        if self.is_done_verifying():
            self.currentGroup = None
            self.done_verifying()
        elif len(self.queue) == 0:
            if self._mismatch_cnt != 0:
                msg = "It looks like you have {0} 'Mis-matched' things in \
the queue. Please click the 'Run Digit Grouping' to make more progress.".format(self._mismatch_cnt)
            else:
                msg = "I'm confused. There are no more overlays, but \
OpenCount claims you're 'done'. Uh oh."
            dlg = wx.MessageDialog(self, message=msg, style=wx.OK)
            self.Disable()
            dlg.ShowModal()
            self.Enable()
        else:
            self.select_group(self.queue[0])
        _dur = time.time() - _t
        times['select_group'] = _dur

        totalTime = time.time() - startTime
        print "==== Total time to process 'Ok': {0} s".format(totalTime)
        for job, duration in times.iteritems():
            print "    job {0} took {1}%.".format(job, 100.0*float(duration / totalTime))


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
        self.run_digit_grouping()

    def OnClickForceDigitGroup(self, evt):
        dlg = ForceDigitGroupDialog(self)
        self.Disable()
        status = dlg.ShowModal()
        self.Enable()
        if status == wx.ID_CANCEL:
            return
        elif status == ForceDigitGroupDialog.ID_SMARTFORCE:
            self.run_digit_grouping(force=True, do_smartforce=True)
        elif status == ForceDigitGroupDialog.ID_BRUTEFORCE:
            self.run_digit_grouping(force=True, do_smartforce=False)

    def run_digit_grouping(self, force=False, do_smartforce=True):
        """ Runs an iteration of DigitGrouping.
        Input:
            bool force: If True, then this forces an iteration.
            bool do_smartforce: If True, then the forced digitgroup 
                iteration will only run on voted ballots 'touched'
                by either an 'Ok' or a 'Misclassify' (i.e. it won't
                run on voted ballots that would possibly change the
                digit group results).
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

        attrtypestr = digitattrs[0]  # TODO: Assume only one digit-based attr
            
        bal2imgs = pickle.load(open(self.project.ballot_to_images, 'rb'))
        # a.) Reconstruct digit_attrs
        digit_attrs = {} # maps {str attrtype: ((y1,y2,x1,x2),side)}
        attrs = pickle.load(open(self.project.ballot_attributesfile, 'rb'))
        for attrdict in attrs:
            attrstr = common.get_attrtype_str(attrdict['attrs'])
            if common.is_digitbased(self.project, attrstr):
                y1 = attrdict['y1']
                y2 = attrdict['y2']
                x1 = attrdict['x1']
                x2 = attrdict['x2']
                side = attrdict['side']
                num_digits = attrdict['num_digits']
                digit_attrs[attrstr] = ((y1, y2, x1, x2), side, num_digits)
        if len(digit_attrs) != 1:
            print "Uhoh, len(digit_attrs) should have been 1, but wasn't."
            pdb.set_trace()
        assert len(digit_attrs) == 1
        # b.) Construct rejected_hashes
        #cur_digit = common.get_propval(self.currentGroup.getcurrentgrouplabel(), 'digit')
        # rejected_hashes maps {imgpath: {digit: [((y1,y2,x1,x2),side_i), ...]}}
        rejected_hashes = self.load_rejected_hashes()
        #rejected_hashes = partmatch_fns.get_rejected_hashes(self.project)
        #if rejected_hashes == None:
        #    # Hasn't been created yet.
        #    rejected_hashes = {}
        #    pickle.dump(rejected_hashes, open(pathjoin(self.project.projdir_path,
        #                                               self.project.rejected_hashes),
        #                                      'wb'))
        ct = 0
        for imgpath, digitsmap in rejected_hashes.iteritems():
            for digit, lst in digitsmap.iteritems():
                ct += len(lst)
        print "Number of rejected regions:", ct

        partmatch_fns.save_rejected_hashes(self.project, rejected_hashes)
        if not force and len(rejected_hashes) == 0:
            print "No need to re-run partmatch, rejected_hashes is empty."
            dlg = wx.MessageDialog(self, message="No need to re-run \
DigitGrouping yet.", style=wx.OK)
            self.Disable()
            dlg.ShowModal()
            self.Enable()
            return
        # c.) Grab accepted_hashes
        # accepted_hashes: {str imgpath: {str digit: [((y1,y2,x1,x2), side_i), ...]}}
        accepted_hashes = self.load_accepted_hashes()
        #accepted_hashes = partmatch_fns.get_accepted_hashes(self.project)
        #if accepted_hashes == None:
        #    # Hasn't been created yet.
        #    #accepted_hashes = {}
        #    #partmatch_fns.save_accepted_hashes(self.project, accepted_hashes)
        accept_cnt = 0
        for imgpath, digitsmap in accepted_hashes.iteritems():
            for digit, lst in digitsmap.iteritems():
                accept_cnt += len(lst)
        print "Total number of accepted regions:", accept_cnt
        partmatch_fns.save_accepted_hashes(self.project, accepted_hashes)
        print "Running partmatch digit-OCR computation with updated \
rejected_hashes..."
        # Filter out ballotids from bal2imgs that we don't need to process.
        # VotedBallots we must process:
        #    i.) If B has any 'mis classify' actions upon it
        bal2imgs_todo = {} # maps {str ballotid: (path_i, ...)}
        img2bal = pickle.load(open(self.project.image_to_ballot, 'rb'))
        todo_jobs = 0
        for votedpath, count in self._misclassify_history.iteritems():
            ballotid = img2bal[votedpath]
            imgs = bal2imgs[ballotid]
            bal2imgs_todo[ballotid] = imgs
            todo_jobs += 1
        if force and do_smartforce:
            # Only run on voted ballots that have been touched by a
            # "Yes" or a "MisClassify"
            for votedpath, count in self._ok_history.iteritems():
                ballotid = img2bal[votedpath]
                imgs = bal2imgs[ballotid]
                bal2imgs_todo[ballotid] = imgs
                todo_jobs += 1
        elif force and not do_smartforce:
            bal2imgs_todo = bal2imgs
            todo_jobs = len(bal2imgs)
        elif todo_jobs == 0:
            dlg = wx.MessageDialog(self, message="No need to run \
DigitGrouping - there's no new information. You must have performed a \
'MisClassify' action in order for DigitGrouping to result in any \
change. \nHowever, if you'd like to force a DigitGroup re-run (say, \
you mistakenely labeled a digit), then choose the 'Force Digit Grouping' \
button in the previous screen.", style=wx.OK)
            self.Disable()
            dlg.ShowModal()
            self.Enable()
            return

        print "==== Number of Jobs fed to do_digitocrpatches: {0}".format(todo_jobs)
        if todo_jobs == 0:
            print "== ...oh wait, no jobs here, return."
            dlg = wx.MessageDialog(self, message="No need to re-run \
DigitGrouping. If you /really/ want to re-run DigitGrouping, try \
choosing the 'Force Digit Group' button.", style=wx.OK)
            self.Disable()
            dlg.ShowModal()
            self.Enable()
            return

        digitgroup_results, digitmatch_info = digit_group.do_digitocr_patches(bal2imgs_todo, digit_attrs, self.project,
                                                                              rejected_hashes=rejected_hashes,
                                                                              accepted_hashes=accepted_hashes)
        prev_digitgroup_results = digit_group.load_digitgroup_results(self.project)
        def _merge_previous_results(digitgroup_results, digitmatch_info,
                                    prev_digitgroup_results, proj):
            if prev_digitgroup_results == None:
                return digitgroup_results, digitmatch_info
            elif force and not do_smartforce:
                return digitgroup_results, digitmatch_info
            ## Merge previous results of digitgrouping with the current
            ## digitgrouping results.
            # digitgroup_results,digitmatch_info will only have results for
            # ballotids from bal2imgs_todo. We need to populate these data
            # structures with the other ballotids that we didn't re-run
            # digitgrouping on.
            prev_digitmatch_info = digit_group.get_digitmatch_info(proj)
            for b_id, tuples in prev_digitgroup_results.iteritems():
                # This is actually b_id (ballotid), not votedpath.
                if b_id not in bal2imgs_todo:
                    digitgroup_results.setdefault(b_id, []).extend(tuples)
            for patchpath, (bb, side, isflip, b_id) in prev_digitmatch_info.iteritems():
                # TODO: Assumes only one digitattribute.
                if b_id not in bal2imgs_todo:
                    digitmatch_info[patchpath] = (bb, side, isflip, b_id)
            return digitgroup_results, digitmatch_info

        digitgroup_results, digitmatch_info = _merge_previous_results(digitgroup_results,
                                                                      digitmatch_info,
                                                                      prev_digitgroup_results,
                                                                      self.project)
        digit_group.save_digitgroup_results(self.project, digitgroup_results)
        #digit_group.save_digitmatch_info(self.project, digitmatch_info)
        self.save_digitmatch_info(digitmatch_info)
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
            for gl_idx in group.orderedAttrVals:
                if common.get_propval(gl_idx, 'digit', self.project, self.grouplabel_record) != None:
                    self.remove_group(group)
                    break
        for group in self.finished[:]:
            for gl_idx in group.orderedAttrVals:
                if common.get_propval(gl_idx, 'digit', self.project, self.grouplabel_record) != None:
                    self.finished.remove(group)
                    break
        self._ok_history = {}
        self._misclassify_history = {}
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
        if common.get_propval(grouplabel, 'digit', self.project, self.grouplabel_record) == None:
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
            
        bal2imgs = pickle.load(open(self.project.ballot_to_images, 'rb'))
        print "==== a.) Reconstruct digit attrs"
        # a.) Reconstruct digit_attrs
        digit_attrs = {} # maps {str attrtype: ((y1,y2,x1,x2),side)}
        attrs = pickle.load(open(self.project.ballot_attributesfile, 'rb'))
        for attrdict in attrs:
            attrstr = common.get_attrtype_str(attrdict['attrs'])
            if common.is_digitbased(self.project, attrstr):
                y1 = attrdict['y1']
                y2 = attrdict['y2']
                x1 = attrdict['x1']
                x2 = attrdict['x2']
                side = attrdict['side']
                num_digits = attrdict['num_digits']
                digit_attrs[attrstr] = ((y1, y2, x1, x2), side, num_digits)
        if len(digit_attrs) != 1:
            print "Uhoh, len(digit_attrs) should have been 1, but wasn't."
            pdb.set_trace()
        assert len(digit_attrs) == 1
        # b.) Construct rejected_hashes
        print "==== b.) Construct rejected_hashes"
        cur_digit = common.get_propval(self.currentGroup.getcurrentgrouplabel(), 'digit', self.project, self.grouplabel_record)
        # rejected_hashes maps {imgpath: {digit: [((y1,y2,x1,x2),side_i,isflip_i), ...]}}
        rejected_hashes = self.load_rejected_hashes()
        #rejected_hashes = partmatch_fns.get_rejected_hashes(self.project)
        #if rejected_hashes == None:
        #    # Hasn't been created yet.
        #    rejected_hashes = {}
        #    pickle.dump(rejected_hashes, open(pathjoin(self.project.projdir_path,
        #                                               self.project.rejected_hashes),
        #                                      'wb'))
        print "== Throw stuff in self.currentGroup.elements into rejected_hashes"
        # Load in digitmatch_info once, since it can be quite large.
        digitmatch_info = digit_group.get_digitmatch_info(self.project)
        common.log_misclassify_ballots(self.project, self.currentGroup.elements)
        for (sampleid, rlist, patchpath) in self.currentGroup.elements:
            # TODO: Do I append sampleid, or patchpath? 
            # TODO: Is it sampleid, or imgpath?
            (bb, side, isflip, b_id) = digit_group.get_digitpatch_info(self.project, patchpath, digitmatch_info)
            rejected_hashes.setdefault(sampleid, {}).setdefault(cur_digit, []).append((bb, side, isflip))
            if sampleid in self._misclassify_history:
                self._misclassify_history[sampleid] += 1
            else:
                self._misclassify_history[sampleid] = 1
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
        #self.remove_group(self.currentGroup)
        self.add_misclassify_group(self.currentGroup)
        self.remove_group(self.currentGroup)

        if self.is_done_verifying():
            self.currentGroup = None
            self.done_verifying()
        elif len(self.queue) == 0:
            if self._mismatch_cnt != 0:
                msg = "It looks like you have {0} 'Mis-matched' things in \
the queue. Please click the 'Run Digit Grouping' to make more progress.".format(self._mismatch_cnt)
            else:
                msg = "I'm confused. There are no more overlays, but \
OpenCount claims you're 'done'. Uh oh."
            dlg = wx.MessageDialog(self, message=msg, style=wx.OK)
            self.Disable()
            dlg.ShowModal()
            self.Enable()
        else:
            self.select_group(self.queue[0])

    def OnClickManualLabelAll(self, evt):
        if common.get_propval(self.currentGroup.orderedAttrVals[0], 'digit', self.project) != None:
            dlg = wx.MessageDialog(self, message="Manually Labeling Digit \
Patches isn't supported right now, sorry.", style=wx.OK)
            self.Disable()
            dlg.ShowModal()
            self.Enable()
            return
            
        dlg = wx.MessageDialog(self, message="Are you sure you want to \
manually label everything in the current group?",
                               style=wx.YES_NO | wx.NO_DEFAULT)
        self.Disable()
        status = dlg.ShowModal()
        self.Enable()
        if status == wx.ID_NO:
            return
        self.Disable()
        dlg = ManualLabelDialog(self, self.currentGroup, self.project)
        status = dlg.ShowModal()
        self.Enable()
        if status == ManualLabelDialog.ID_CANCEL:
            return
        for group, (final_idx, val) in dlg.results:
            if final_idx == None:
                print "== Uhoh, quarantining this group."
                self.quarantine_group(group, doremove=False) # doremove=False since group was never in self.queue
            else:
                self.add_finalize_group(group, final_idx)
        self.remove_group(self.currentGroup)

        if self.is_done_verifying():
            self.currentGroup = None
            self.done_verifying()
        elif len(self.queue) == 0:
            if self._mismatch_cnt != 0:
                msg = "It looks like you have {0} 'Mis-matched' things in \
the queue. Please click the 'Run Digit Grouping' to make more progress.".format(self._mismatch_cnt)
            else:
                msg = "I'm confused. There are no more overlays, but \
OpenCount claims you're 'done'. Uh oh."
            dlg = wx.MessageDialog(self, message=msg, style=wx.OK)
            self.Disable()
            dlg.ShowModal()
            self.Enable()
        else:
            self.select_group(self.queue[0])


    def is_done_verifying(self):
        """ Return True iff overlay verification is complete. False o.w. """
        if not self.queue and self._mismatch_cnt == 0:
            return True
        else:
            return False
        
    def done_verifying(self):
        """
        When the user has finished verifying all groups, do some
        fancy computing, and output results.
        Outputs grouping results into the specified out-directory, if
        given.
        """
        print "DONE Verifying!"
        if self.mode == VerifyPanel.MODE_NORMAL:
            dlg = wx.MessageDialog(self, message="Grouping verification \
finished! Press 'Ok', then you may continue to the next step.",
                                   style=wx.OK)
            self.Disable()
            dlg.ShowModal()
        else:
            self.Disable()
        # First populate results
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
                gl_idx = group.orderedAttrVals[0]
                grouplabel = self.grouplabel_record[gl_idx]
                if grouplabel in results:
                    print "Uhoh, grouplabel was in results more than once."
                    pdb.set_trace()
                assert grouplabel not in results, "grouplabel {0} was duplicated".format(common.str_grouplabel(grouplabel, self.project, self.grouplabel_record))
                results[gl_idx] = [group]
        else:
            for group in self.finished:
                results.setdefault(group.getcurrentgrouplabel(), []).append(group)
            if self.templates:
                for grouplabel in self.templates:
                    gl_idx = self.grouplabel_record.index(grouplabel)
                    if gl_idx not in results:
                        results[gl_idx] = []

        if self.outfilepath:
            pickle.dump(results, open(self.outfilepath, 'wb'))

        if self.ondone:
            self.ondone(results, self.grouplabel_record)

    def OnClickSplit(self, event):
        def collect_ids(newGroups):
            """ Only used in MODE_YESNO2. """
            ids = {} # {str attrname: list of ids}
            groups = tuple(newGroups) + tuple(self.queue) + tuple(self.finished)
            dflag = True
            for group in groups:
                # In MODE_YESNO2, foo is a list of the form:
                #    ((<attrtype>,), ID)
                #foo = list(group.getcurrentgrouplabel())
                gl_idx = group.orderedAttrVals[0]
                grouplabel_tup = tuple(self.grouplabel_record[gl_idx])
                attrtype = tuple(sorted([t[0] for t in grouplabel_tup]))
                id = grouplabel_tup[0][1]
                if dflag:
                    dflag = False
                    print "id is:", id
                    #pdb.set_trace()
                ids.setdefault(attrtype, []).append(id)
            return ids
        def assign_new_id(group, ids):
            """ Given a new GroupClass, and a previous IDS mapping,
            find a unique new id for group to use.
            Only used in MODE_YESNO2.
            """
            grouplabel = self.grouplabel_record[group.getcurrentgrouplabel()]
            foo = tuple(grouplabel)
            k = tuple(sorted([t[0] for t in foo]))
            i = 0
            while i >= 0:
                if i not in ids[k]:
                    newgrouplabel = common.make_grouplabel(*[(a, i) for a in k])
                    try:
                        gl_idx = self.grouplabel_record.index(newgrouplabel)
                    except:
                        gl_idx = len(self.grouplabel_record)
                        self.grouplabel_record.append(newgrouplabel)
                    group.orderedAttrVals = (gl_idx,) + tuple(group.orderedAttrVals[1:])
                    #group.orderedAttrVals[0] = newgrouplabel
                    ids.setdefault(k, []).append(i)
                    break
                i += 1
            return group
        
        if self.mode in (VerifyPanel.MODE_YESNO, VerifyPanel.MODE_YESNO2):
            if self.splitMode == 'rankedlist':
                # RankedList doesn't make sense in this mode right now,
                # since the rankedlist is created arbitrarily. Default
                # to kmeans.
                # TODO: It is feasible for rankedlist to be a reasonable
                # choice, if the rankedlists in this mode are created
                # 'correctly'.
                print "Not using rankedlist, using kmeans instead."
                self.splitMode = 'kmeans'

        newGroups = self.currentGroup.split(mode=self.splitMode)
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
        #self.parent.Fit() # Causes UI issues
        self.fitPanel()
    
    def OnClickSetSplitMode(self, event):
        if self.mode == VerifyPanel.MODE_YESNO2:
            dlg = ChooseSplitModeDialog(self, disable=(ChooseSplitModeDialog.ID_RANKEDLIST,))
        else:
            dlg = ChooseSplitModeDialog(self)
        self.Disable()
        status = dlg.ShowModal()
        self.Enable()
        
        if status == wx.ID_CANCEL:
            return
        elif status == ChooseSplitModeDialog.ID_RANKEDLIST:
            self.splitMode = 'rankedlist'
        elif status == ChooseSplitModeDialog.ID_KMEANS:
            self.splitMode = 'kmeans'
        elif status == ChooseSplitModeDialog.ID_PCA_KMEANS:
            self.splitMode = 'pca_kmeans'
        elif status == ChooseSplitModeDialog.ID_KMEANS2:
            self.splitMode = 'kmeans2'
        elif status == ChooseSplitModeDialog.ID_KMEDIODS:
            self.splitMode = 'kmediods'
        else:
            dlg = wx.MessageDialog(self, message="Unrecognized split mode.", style=wx.OK)
            self.Disable()
            dlg.ShowModal()
            self.Enable()
            return

    def OnClickMerge(self, event):
        """ Take all currently-displayed groups A, and 'condense' them
        into a new set of groups A', where groups G_0, G_1 in A are
        combined if their grouplabels are the same. The 'grouplabel'
        is, for each group G, OpenCount's current 'guess' as to what
        G's overlay represents (i.e. 'party'->'democrat'?).
        In other words, a 'Merge' is an anti-'Split'.
        """
        dlg = wx.MessageDialog(self, message="Not implemented yet!")
        self.Disable()
        dlg.ShowModal()
        self.Enable()
        # TODO: This is more nuanced than I expected. I'd have to
        # manually re-arrange all rankedlists in each new GroupClass
        # to ensure that the label L that the user chose is at the
        # 'front' of the rankedlist (since it's in sorted order).
        # Also, for regular-attributes, arbitrarily re-ordering the
        # ranked list might have bad consequences for 'Split', since
        # 'Split' assumes that the rankedlist is ordered a certain way.
        '''
        newgroups_map = {} # maps {str grouplabel: (GroupClass_i, ...)}
        for group in self.queue:
            newgroups_map.setdefault(group.label, []).append(group)
        pdb.set_trace()
        newgroups = [] # (GroupClass_i, ...)
        for grouplabel, groups in newgroups_map.iteritems():
            newgroups.append(common.GroupClass.merge(*groups))

        # Sanity check
        oldcount = sum([len(g.elements) for g in self.queue])
        newcount = sum([len(g.elements) for g in newgroups])
        if oldcount != newcount:
            print "Uhoh, old num. of elements was {0}, but the new \
elements num. is {1}".format(oldcount, newcount)
            pdb.set_trace()
            return

        # 0.) Remove all current groups
        for group in self.queue:
            self.remove_group(group)
        # 1.) Add in all new groups
        for group in newgroups:
            self.add_group(group)
        self.select_group(self.queue[0])
        '''
        
    def OnClickDebug(self, event):
        if (self.currentGroup != None):
            elements = self.currentGroup.elements
            for element in elements:
                print element[0]

    def onButton_viewimg(self, evt):
        """ Display a pop-up that displays all ballot images of the
        current group.
        """
        class ViewImgsDialog(wx.Dialog):
            def __init__(self, parent, imgpaths, *args, **kwargs):
                wx.Dialog.__init__(self, parent, title="Viewing Images...", size=(600, 900), *args, **kwargs)
                self.imgpaths = imgpaths

                self.curidx = 0

                self.sbitmap = wx.StaticBitmap(self)
                txt0 = wx.StaticText(self, label="Image path: ")
                self.txt_imP = wx.StaticText(self, label="Foo")
                sizer_img = wx.BoxSizer(wx.VERTICAL)
                sizer0 = wx.BoxSizer(wx.HORIZONTAL)
                sizer0.AddMany([(txt0,),(self.txt_imP,)])
                sizer_img.AddMany([(self.sbitmap,), ((0, 10),), (sizer0,)])
                
                btn_next = wx.Button(self, label="Next Image")
                btn_next.Bind(wx.EVT_BUTTON, self.onButton_next)
                btn_prev = wx.Button(self, label="Previous Image")
                btn_prev.Bind(wx.EVT_BUTTON, self.onButton_prev)
                btn_close = wx.Button(self, label="Close")
                btn_close.Bind(wx.EVT_BUTTON, self.onButton_close)
                btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
                btn_sizer.AddMany([(btn_next,), ((10,0),), (btn_prev,), ((10,0),), (btn_close,)])
                
                txt1 = wx.StaticText(self, label="Image ")
                self.txt_n = wx.StaticText(self, label="1")
                txt2 = wx.StaticText(self, label=" / ")
                self.txt_k = wx.StaticText(self, label="{0}.".format(len(imgpaths)))
                txt_sizer = wx.BoxSizer(wx.HORIZONTAL)
                txt_sizer.AddMany([(txt1,), (self.txt_n,), (txt2,), (self.txt_k,)])
                
                sizer = wx.BoxSizer(wx.VERTICAL)
                sizer.Add(sizer_img)
                sizer.Add((0, 20))
                sizer.Add(btn_sizer, flag=wx.ALIGN_CENTER)
                sizer.Add((0, 20))
                sizer.Add(txt_sizer, flag=wx.ALIGN_CENTER)
                sizer.Add((0, 20))

                self.SetSizer(sizer)
                self.Layout()
                self.display_img(self.curidx)
            def display_img(self, idx):
                if idx >= len(self.imgpaths) or idx < 0:
                    return
                self.curidx = idx
                imgpath = self.imgpaths[self.curidx]
                wximg = wx.Image(imgpath, wx.BITMAP_TYPE_PNG)
                c = wximg.GetHeight() / 700.
                new_w = wximg.GetWidth() / c
                wximg = wximg.Rescale(new_w, 700, wx.IMAGE_QUALITY_HIGH)
                self.sbitmap.SetBitmap(wx.BitmapFromImage(wximg))
                self.txt_imP.SetLabel(imgpath)
                self.txt_n.SetLabel(str(self.curidx+1))

                self.Fit()
                
            def onButton_next(self, evt):
                self.display_img(self.curidx+1)
            def onButton_prev(self, evt):
                self.display_img(self.curidx-1)
            def onButton_close(self, evt):
                self.Close()
        imgpaths = []
        for (sampleid, rlist, patchpath) in self.currentGroup.elements:
            imgpaths.append(sampleid)
        dlg = ViewImgsDialog(self, imgpaths)
        dlg.ShowModal()
            
    def quarantine_group(self, group, doremove=True):
        """
        Quarantines group.
        """
        elements = group.elements
        qfile = open(self.project.quarantined, 'a')
        votedpaths = set()
        for element in elements:
            votedpath = os.path.abspath(element[0])
            print >>qfile, votedpath
            votedpaths.add(votedpath)
        qfile.close()
        if doremove:
            self.remove_group(group)
        # Finally, remove the quarantined ballots from the GroupClasses.
        # Remember to remove any resulting 0-element GroupClasses.
        def pred(element, votedpaths):
            return element[0] not in votedpaths
        self.queueList.Clear()
        newqueue = []
        for group in self.queue:
            group.elements = tuple([el for el in group.elements if pred(el, votedpaths)])
            if group.elements:
                newqueue.append(group)
                gl_idx = group.getcurrentgrouplabel()
                attrtype, attrval = common.get_attrpair_grouplabel(self.project, gl_idx, self.grouplabel_record)
                self.queueList.Append("{0}->{1}: {2} elements".format(attrtype, attrval, len(group.elements)))
        newfinished = []
        for group in self.finished:
            group.elements = tuple([el for el in group.elements if pred(el, votedpaths)])
            if group.elements:
                newfinished.append(group)
        self.queue = newqueue
        self.finished = newfinished
        for sampleid in self._misclassify_history.keys():
            if sampleid in votedpaths:
                cnt = self._misclassify_history.pop(sampleid)
                self._mismatch_cnt -= cnt
                self.misclassify_txt.SetLabel("Mismatches in queue: {0}".format(self._mismatch_cnt))
        
    def OnClickQuarantine(self, event):
        if (self.currentGroup != None):
            self.quarantine_group(self.currentGroup)
            if self.is_done_verifying():
                self.done_verifying()
            elif len(self.queue) == 0:
                if self._mismatch_cnt != 0:
                    msg = "It looks like you have {0} 'Mis-matched' things in \
the queue. Please click the 'Run Digit Grouping' to make more progress.".format(self._mismatch_cnt)
                else:
                    msg = "I'm confused. There are no more overlays, but \
OpenCount claims you're 'done'. Uh oh."
                dlg = wx.MessageDialog(self, message=msg, style=wx.OK)
                self.Disable()
                dlg.ShowModal()
                self.Enable()
            else:
                self.select_group(self.queue[0])

    def OnClickSaveCheckpoint(self, event):
        """ Saves the current (exact) UI state, such as which groups
        are visible, not visible, etc.
        """
        print "Saving state."
        dlg = wx.MessageDialog(self, message="Warning - saving this \
Checkpoint will overwrite the last-saved Checkpoint. Are you sure you \
want to continue?", style=wx.YES_NO | wx.NO_DEFAULT)
        self.Disable()
        status = dlg.ShowModal()
        if status == wx.ID_YES:
            self.dump_state()
        self.Enable()
        
    def OnClickLoadCheckpoint(self, event):
        """ Loads the (last) saved checkpoint. """
        dlg = wx.MessageDialog(self, message="Warning - loading the last \
saved Checkpoint will result in all unsaved progress being lost. Are \
you sure you want to continue?", style=wx.YES_NO | wx.NO_DEFAULT)
        self.Disable()
        status = dlg.ShowModal()
        if status == wx.ID_YES:
            self.load_state()
        self.select_group(self.queue[0])
        self.Enable()

    def restore_hidden_groups(self):
        for i in range(len(self.finished)):
            self.add_group(self.finished.pop())

    def OnClickRestoreAll(self, event):
        """ Restores all hidden groups to the UI. """
        dlg = wx.MessageDialog(self, message="Warning - restoring all\
groups will add all previously-OK'd Groups to the UI. You will then \
have to re-click 'Ok' for each of these Groups. \n\n\
Are you sure you want to do this?", style=wx.YES_NO | wx.NO_DEFAULT)
        self.Disable()
        status = dlg.ShowModal()
        if status == wx.ID_YES:
            self.restore_hidden_groups()
        self.select_group(self.queue[0])
        self.Enable()
    
    def OnClickUndo(self, event):
        """ Adds the last-finished GroupClass to the UI. """
        if not self.finished:
            dlg = wx.MessageDialog(self, message="No OK'd Groups to \
undo!", style=wx.OK)
            self.Disable()
            dlg.ShowModal()
            self.Enable()
            return
        self.add_group(self.finished.pop())
        self.select_group(self.queue[0])

    def checkCanMoveOn(self):
        # TODO: Fix this implementation.
        return True
        #return self.canMoveOn
        
    def _pubsub_project(self, msg):
        project = msg.data
        self.project = project
        
    def OnSize(self, event):
        self.fitPanel()
        event.Skip()

class ForceDigitGroupDialog(wx.Dialog):
    ID_SMARTFORCE = 78
    ID_BRUTEFORCE = 79

    def __init__(self, parent, *args, **kwargs):
        wx.Dialog.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        sizer = wx.BoxSizer(wx.VERTICAL)
        msg0 = "Warning: This will re-run \
Digit Grouping on all voted ballots (even if it's not necessary to run \
Digit Grouping). This can take a few hours of computation time."
        
        msg1 = "If you want to run Digit Grouping only on a subset of the set of voted \
ballots, choose 'Cancel', then the 'Run Digit Grouping' button. \
This is the 'normal' mode of operation."
        
        msg2 = "If you wish to force-run Digit Grouping on only voted ballots \
that you've marked either with an 'Ok' or a 'MisClassify', then \
choose 'Smart Force'."
        msg3 = "If you definitely want to force-run Digit Grouping on /all/ voted \
ballots, choose 'Brute Yes'."

        msg0 = textwrap.fill(msg0, 70)
        msg1 = textwrap.fill(msg1, 70)
        msg2 = textwrap.fill(msg2, 70)
        msg3 = textwrap.fill(msg3, 70)
        msg = msg0 + "\n\n" + msg1 + "\n" + msg2 + "\n" + msg3
        txt = wx.StaticText(self, label=msg)
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        btn_cancel = wx.Button(self, label="Cancel")
        btn_cancel.Bind(wx.EVT_BUTTON, self.onButton_cancel)
        btn_cancel.SetDefault()
        btn_smartforce = wx.Button(self, label="Smart Force")
        btn_smartforce.Bind(wx.EVT_BUTTON, self.onButton_smartforce)
        btn_bruteforce = wx.Button(self, label="Brute Force")
        btn_bruteforce.Bind(wx.EVT_BUTTON, self.onButton_bruteforce)
        btn_sizer.AddMany([(btn_cancel,), (btn_smartforce,), (btn_bruteforce,)])
        sizer.Add(txt)
        sizer.Add(btn_sizer, flag=wx.ALIGN_CENTER)
        self.SetSizer(sizer)
        self.Fit()

    def onButton_cancel(self, evt):
        self.EndModal(wx.ID_CANCEL)
    def onButton_smartforce(self, evt):
        self.EndModal(self.ID_SMARTFORCE)
    def onButton_bruteforce(self, evt):
        self.EndModal(self.ID_BRUTEFORCE)

class ManualLabelDialog(wx.Dialog):
    ID_DONE = 42
    ID_CANCEL = 43

    def __init__(self, parent, group, proj, *args, **kwargs):
        wx.Dialog.__init__(self, parent, 
                           style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
                           *args, **kwargs)
        self.parent = parent
        self.group = group
        self.proj = proj
        self.results = None # list of [(GroupClass_i, int final_idx), ...]
        self.map = {} # maps {str patchpath: 

        self.Maximize()
        
        sizer = wx.BoxSizer(wx.VERTICAL)
        self.labelpanel = label_imgs.LabelPanel(self)
        btn_cancel = wx.Button(self, label="Cancel.")
        btn_cancel.Bind(wx.EVT_BUTTON, lambda evt: self.EndModal(ManualLabelDialog.ID_CANCEL))

        sizer.Add(self.labelpanel, proportion=1, flag=wx.EXPAND)
        sizer.Add(btn_cancel, flag=wx.ALIGN_CENTER)
        self.SetSizer(sizer)

        self.attrtype = None
        for (imgpath, rlist, patchpath) in group.elements:
            elements = ((imgpath,rlist,patchpath),)
            if type(group) == common.GroupClass:
                self.map[patchpath] = common.GroupClass(elements)
            elif type(group) == common.DigitGroupClass:
                self.map[patchpath] = common.DigitGroupClass(elements)

            if self.attrtype == None:
                # Infer which attribute type this GroupClass is for.
                bestguess = group.orderedAttrVals[0]
                self.attrtype, attrval = common.get_attrpair_grouplabel(self.proj, bestguess, self.grouplabel_record)

        self.possible_attrvals = common.get_attrtype_possiblevals(self.proj, self.attrtype)
            
        self.imageslist = [el[2] for el in group.elements]
        self.labelpanel.start(self.imageslist, callback=self.done, possibles=self.possible_attrvals)

    def done(self, imagelabels):
        """
        Input:
            dict imagelabels: maps {str imgpath: str label}
        """
        attrtypes = common.get_attrtypes(self.proj, with_digits=True)
        print "MOO"
        self.results = []
        for patchpath, label in imagelabels.iteritems():
            groupclass = self.map[patchpath]
            final_idx = None
            for i, rlabel in enumerate(groupclass.elements[0][1]):  # [0][1] -> rlist
                attrtype, attrval = common.get_attrpair_grouplabel(self.proj, rlabel, self.grouplabel_record)
                if self.attrtype != attrtype:
                    print "Uhoh, inconsistent attrtypes."
                    pdb.set_trace()
                assert self.attrtype == attrtype
                if label == attrval:
                    final_idx = i
                    break
            if final_idx == None:
                print "Uhoh, couldn't find final_idx for label {0}".format(label)
                self.results.append((groupclass, None))
                pdb.set_trace()
            else:
                self.results.append((groupclass, (final_idx, label)))
        self.EndModal(ManualLabelDialog.ID_DONE)
    
class ChooseSplitModeDialog(wx.Dialog):
    ID_RANKEDLIST = 42
    ID_KMEANS = 43
    ID_PCA_KMEANS = 44
    ID_KMEANS2 = 45
    ID_KMEDIODS = 46

    def __init__(self, parent, disable=None, *args, **kwargs):
        """ disable is a list of ID's (ID_RANKEDLIST, etc.) to disable. """
        wx.Dialog.__init__(self, parent, *args, **kwargs)
        if disable == None:
            disable = []
        sizer = wx.BoxSizer(wx.VERTICAL)
        txt = wx.StaticText(self, label="Please choose the desired 'Split' method.")

        self.rankedlist_rbtn = wx.RadioButton(self, label='Ranked-List (fast)', style=wx.RB_GROUP)
        self.kmeans_rbtn = wx.RadioButton(self, label='K-means (not-as-fast)')
        self.pca_kmeans_rbtn = wx.RadioButton(self, label='PCA+K-means (not-as-fast)')
        self.kmeans2_rbtn = wx.RadioButton(self, label="K-means V2 (not-as-fast)")
        self.kmediods_rbtn = wx.RadioButton(self, label="K-Mediods")
        
        if parent.splitMode == 'rankedlist':
            self.rankedlist_rbtn.SetValue(1)
        elif parent.splitMode == 'kmeans':
            self.kmeans_rbtn.SetValue(1)
        elif parent.splitMode == 'pca_kmeans':
            self.pca_kmeans_rbtn.SetValue(1)
        elif parent.splitMode == 'kmeans2':
            self.kmeans2_rbtn.SetValue(1)
        elif parent.splitMode == 'kmediods':
            self.kmediods_rbtn.SetValue(1)
        else:
            print "Unrecognized parent.splitMode: {0}. Defaulting to kmeans.".format(parent.splitMode)
            self.kmeans_rbtn.SetValue(1)

        if ChooseSplitModeDialog.ID_RANKEDLIST in disable:
            self.rankedlist_rbtn.Disable()
        if ChooseSplitModeDialog.ID_KMEANS in disable:
            self.kmeans_rbtn.Disable()
        if ChooseSplitModeDialog.ID_PCA_KMEANS in disable:
            self.pca_kmeans_rbtn.Disable()
        if ChooseSplitModeDialog.ID_KMEANS2 in disable:
            self.kmeans2_rbtn.Disable()
        if ChooseSplitModeDialog.ID_KMEDIODS in disable:
            self.kmediods_rbtn.Disable()
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        btn_ok = wx.Button(self, label="Ok")
        btn_cancel = wx.Button(self, label="Cancel")
        btn_ok.Bind(wx.EVT_BUTTON, self.onButton_ok)
        btn_cancel.Bind(wx.EVT_BUTTON, lambda evt: self.EndModal(wx.ID_CANCEL))
        
        btn_sizer.AddMany([(btn_ok,), (btn_cancel,)])

        sizer.AddMany([(txt,), ((20,20),), (self.rankedlist_rbtn,),
                       (self.kmeans_rbtn,), (self.pca_kmeans_rbtn,),
                       (self.kmeans2_rbtn,), (self.kmediods_rbtn),])
        sizer.Add(btn_sizer, flag=wx.ALIGN_CENTER)

        self.SetSizer(sizer)
        self.Fit()

    def onButton_ok(self, evt):
        if self.rankedlist_rbtn.GetValue():
            self.EndModal(self.ID_RANKEDLIST)
        elif self.kmeans_rbtn.GetValue():
            self.EndModal(self.ID_KMEANS)
        elif self.pca_kmeans_rbtn.GetValue():
            self.EndModal(self.ID_PCA_KMEANS)
        elif self.kmeans2_rbtn.GetValue():
            self.EndModal(self.ID_KMEANS2)
        elif self.kmediods_rbtn.GetValue():
            self.EndModal(self.ID_KMEDIODS)
        else:
            print "Unrecognized split mode. Defaulting to K-means."
            self.EndModal(self.ID_KMEANS)



