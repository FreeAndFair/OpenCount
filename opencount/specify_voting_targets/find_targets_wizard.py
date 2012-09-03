import os, sys, math, csv, copy, random, threading, time, traceback
import pickle, datetime, pdb, Queue, multiprocessing
import numpy as np
import scipy
import scipy.misc
import scipy.ndimage
import cv

from os.path import join as pathjoin
import util_gui
import util_widgets
import util
import grouptargets
import time

from imageviewer import BallotViewer, BallotScreen, BoundingBox, Autodetect_Panel, Autodetect_Confirm, WorldState
from labelcontest.group_contests import find_contests

"""
UI intended for a user to denote all locations of voting targets, on
all template images, using auto-detection.
"""

"""
Bugs:
- When zooming into a ballot image in Mosaic during Mosaic-Verification,
  the scrollbars don't activate on the right panel until you do a resize.
- During MosaicVerification, Deleting a target (with backspace)
  doesn't update the mosaicpanel
- If the UL corner of box A and the LR corner of box B are too close,
  then trying to move A will also resize box B at the same time.

Annoyances:
- 'Modify' mode
- Cursor while creating new targets is too big and obtrusive
- mouse scrolling (with wheelmouse) doesn't get intuitively captured
  between mosaic-panel and ballotviewer. Ideally, the RightThing(tm)
  should happen when the mouse has entered either panel. Should be
  easy to implement with the OnMouseEnter event.
- Target Autodetection doesn't work so well on other template images,
  where differences in rotation/whatnot make it 'harder' to do a 
  simple naive template match. It does 'reasonably' well, but not
  as well as I'd like.

Todo:
- If a round of template-matching doesn't change anything in a
  contest, don't re-compute those contest bounding boxes - it's
  frustrating when adding a new target means losing all manual
  bounding box adjustments
- Add undo feature
- Add 'Add contest' feature
  
"""

####
## Import 3rd party libraries
####

import wx
import wx.animate
import Image
import cv2
import numpy as np
import wx.lib.inspection
from wx.lib.pubsub import Publisher
    
# Get this script's directory. Necessary to know this information
# since the current working directory may not be the same as where
# this script lives (which is important for loading resources like
# imgs). And yes, this is a bit hacky :\
try:
    # If __file__ is defined, then this script is being invoked by
    # another script.  __file__ is the path to this script.
    MYDIR = os.path.abspath(os.path.dirname(__file__))
except NameError:
    # This script is being run directly - then, sys.path[0] contains
    # the path to this script.
    MYDIR = os.path.abspath(sys.path[0])

CONSTANT_CLOSE_TO = 4.0 # Threshold pixel proximity for is_close_to(2)

TIMER = None
        
class SpecifyTargetsPanel(wx.Panel):
    """
    Panel that contains the Mosaic Panel and BallotViewer Panel. 
    Allows a user to specify bounding boxes around all voting targets.
    """
    # Number of times to automatically re-run template matching
    NUM_ITERS = 0
    # Default 'confidence' parameter value for Template Matching
    TEMPMATCH_DEFAULT_PARAM = 0.85
    INFERCONTESTS_JOB_ID = util.GaugeID("InferContestsJobId")

    def __init__(self, parent, *args, **kwargs):
        wx.Panel.__init__(self, parent, *args, **kwargs)
        
        ## Instance vars
        self.project = None
        self.parent = parent
        self.world = None
        # frontback_map maps {str templatepath: 'front'/'back'}
        self.frontback_map = {} 

        # frozen_contests maps {str temppath: List of BoundingBoxes (contests)}. 
        # keeps track of all contests that were modified by the user.
        self.frozen_contests = {}

        self.has_started = False    # If True, then self.start() has already been called. UNUSED
        
        self.setup_widgets()

        self.callbacks = [("signals.autodetect.final_refimg", self._final_refimg),
                          ("signals.MainPanel.export_targets", self.pubsub_export_targets),
                          ("broadcast.mosaicpanel.mosaic_img_selected", self.pubsub_mosaic_img_selected),
                          ("broadcast.updated_world", self._pubsub_updatedworld),
                          ("broadcast.ballotscreen.added_target", self.pubsub_added_target),
                          ("broadcast.deleted_targets", self._pubsub_deleted_targets),
                          ("broadcast.tempmatchdone", self._pubsub_tempmatchdone),
                          ("broadcast.ballotscreen.added_contest", self._pubsub_added_contest),
                          ("broadcast.freeze_contest", self._pubsub_freeze_contest)]
        self.subscribe_pubsubs()

        # Pubsub Subscribing
        Publisher().subscribe(self._pubsub_project, "broadcast.project")
        Publisher().subscribe(self._pubsub_projupdate, "broadcast.projupdate")

    def get_frozen_contests(self, templatepath):
        if templatepath not in self.frozen_contests:
            self.frozen_contests[templatepath] = []
        return self.frozen_contests[templatepath]

    def unsubscribe_pubsubs(self):
        for (topic, callback) in self.callbacks:
            Publisher().unsubscribe(callback, topic)
        self.panel_mosaic.unsubscribe_pubsubs()
        self.ballotviewer.unsubscribe_pubsubs()
        self.world.unsubscribe_pubsubs()

    def subscribe_pubsubs(self):
        for (topic, callback) in self.callbacks:
            Publisher().subscribe(callback, topic)
        self.panel_mosaic.subscribe_pubsubs()
        self.ballotviewer.subscribe_pubsubs()
        self.world.subscribe_pubsubs()
        
    def set_timer(self, timer):
        self.TIMER = timer
        global TIMER
        TIMER = timer

    def setup_widgets(self):
        self.sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.world = WorldState()
        #self.panel_mosaic = util_widgets.MosaicPanel(self, style=wx.SIMPLE_BORDER)
        self.panel_mosaic = MosaicPanel2(self, self.world, style=wx.SIMPLE_BORDER)
        self.panel_mosaic.Hide()
        self.ballotviewer = BallotViewer(self, self.world, ballotscreen=MyBallotScreen, style=wx.SIMPLE_BORDER)
        self.ballotviewer.Hide()
        self.frontbackpanel = FrontBackPanel(self)
        self.frontbackpanel.Hide()
        vertsizer = wx.BoxSizer(wx.VERTICAL)
        vertsizer.Add(self.ballotviewer, border=10, proportion=2, flag=wx.EXPAND | wx.ALL | wx.ALIGN_LEFT)
        vertsizer.Add(self.frontbackpanel, border=10, proportion=0, flag=wx.ALL | wx.ALIGN_LEFT)
        self.sizer.Add(self.panel_mosaic, border=10, proportion=0, flag=wx.EXPAND |wx.ALL | wx.ALIGN_LEFT)
        self.sizer.Add(vertsizer, border=10, proportion=1, flag=wx.EXPAND | wx.ALL | wx.ALIGN_LEFT)
        #self.sizer.Add(self.ballotviewer, border=10, proportion=1, flag=wx.EXPAND | wx.ALL | wx.ALIGN_LEFT)
        #self.sizer.Add(self.frontbackpanel, border=10, proportion=0, flag=wx.ALL)
        self.SetSizer(self.sizer)
        self.Fit()
        
    def _pubsub_freeze_contest(self, msg):
        """
        Triggered when the user performs some freeze-ing action on a
        contest bounding box:
            Splitting
            Resizing
            Moving
        """
        temppath, contest = msg.data
        contests = self.frozen_contests.setdefault(temppath, [])
        if contest not in contests:
            self.frozen_contests[temppath].append(contest)

    def reset(self):
        #self.box_locations = {}
        #self.first_time = True
        self.has_started = False
        # These aren't working - it isn't removing the previous
        # mosaic panels/ballot screens. will tend to later.
        #self.panel_mosaic.reset()
        #self.ballotviewer.ballotscreen.set_image(None)
        # TOOD: RESET ballotscreen and mosaic

    def validate_outputs(self):
        """
        Checks the outputs of this widget, raises warning dialogs
        to the user if things don't seem right.
        Returns True if everything is OK, False otherwise.
        """
        if util.is_multipage(self.project):
            # Make sure # of backsides equals # of frontsides
            if self.frontback_map:
                sides = self.frontback_map.values()
                n1 = len([s for s in sides if s == 'front'])
                n2 = len([s for s in sides if s == 'back'])
                if n1 != n2:
                    dlg = wx.MessageDialog(self, message="Warning: uneven \
number of front/back sides detected. OpenCount currently thinks that there \
are {0} front side blank ballots, and {1} back side blank ballots. If this \
is in error, please correct this by, for each blank ballot, indicating the \
side.".format(n1, n2),
                                           style=wx.OK)
                    self.Disable()
                    dlg.ShowModal()
                    self.Enable()
                    return False
        # Make sure there are voting targets
        if self.world.get_boxes_count_all() == 0:
            dlg = wx.MessageDialog(self, message="""No voting targets \
or contests were created. Please go back and do this. OpenCount won't \
work if it has no idea where the voting targets are, or if it doesn't \
know what contests are.""",
                                   style=wx.OK)
            self.Disable()
            dlg.ShowModal()
            self.Enable()
            return False
        else:
            # Finer-grained search on individual templates
            bad_temps = {}   # maps {str temppath: [bool noTargets, bool noContests]}
            notargets = []
            nocontests = []
            for temppath, boxes in self.world.box_locations.iteritems():
                bad_temps.setdefault(temppath, [False, False])
                targets = [b for b in boxes if not b.is_contest]
                contests = [b for b in boxes if b.is_contest]
                if not targets:
                    bad_temps[temppath][0] = True
                    notargets.append(temppath)
                if not contests:
                    bad_temps[temppath][1] = True
                    nocontests.append(temppath)
            msg = """Warning: {0} blank ballots don't have any voting \
targets, and {1} blank ballots don't have any contests. 

If you think \
this is incorrect, and would like to go back and double-check your \
work, choose the 'I want to double-check' button.

Otherwise, if this is correct, go on ahead by choosing the \
'This is correct, proceed onwards' button. This could be the \
case if, for instance, a blank ballot has a totally-empty back-page.""".format(len(notargets),len(nocontests))
            if notargets or nocontests:
                dlg = WarnNoBoxesDialog(self, msg)
                self.Disable()
                statusval = dlg.ShowModal()
                self.Enable()
                if statusval == WarnNoBoxesDialog.GOBACK:
                    return False

        # Make sure each voting target is enclosed by a contest
        lonely_tmpls = set()
        for temppath, boxes in self.world.box_locations.iteritems():
            targets = [b for b in boxes if not b.is_contest]
            contests = [b for b in boxes if b.is_contest]
            for target in targets:
                contest = util_gui.find_assoc_contest(target, contests)
                if not contest:
                    lonely_tmpls.add(temppath)
        if lonely_tmpls:
            msg = """Warning: {0} blank ballots have voting targets \
that are not enclosed by a contest. Please go back and fix them. \
OpenCount must know the target-contest associations in order to \
function correctly.""".format(len(lonely_tmpls))
            dlg = wx.MessageDialog(self, message=msg, style=wx.OK)
            self.Disable()
            dlg.ShowModal()
            self.Enable()
            return False
        return True
        
    def stop(self):
        """
        Disengage PubSub listeners.
        """
        self.unsubscribe_pubsubs()

    def start(self):
        """
        Load in template images, and allow user to start drawing boxes.
        Also assumes that self.world has been set.
        """
        if False:#self.has_started: # Never reset (for now)
            self.reset()
        self.has_started = True
        self.subscribe_pubsubs()
        if not self.world.get_boxes_all():
            # User hasn't created boxes before
            imgpaths = []
            img_boxes = {}
            for dirpath, dirnames, filenames in os.walk(self.project.templatesdir):
                for imgname in [f for f in filenames if util_gui.is_image_ext(f)]:
                    imgpath = os.path.abspath(pathjoin(dirpath, imgname))
                    imgpaths.append(imgpath)
                    img_boxes[imgpath] = []
            #imgpaths = sorted(imgpaths)
            util.sort_nicely(imgpaths) # Fixes Marin filename ordering
            self.world.box_locations = img_boxes
        else:
            imgpaths = []
            for dirpath, dirnames, filenames in os.walk(self.project.templatesdir):
                for imgname in [f for f in filenames if util_gui.is_image_ext(f)]:
                    imgpath = os.path.abspath(pathjoin(dirpath, imgname))
                    imgpaths.append(imgpath)
            #imgpaths = sorted(imgpaths)
            util.sort_nicely(imgpaths)
            img_boxes = self.world.get_boxes_all()
            
        self.panel_mosaic.set_images(imgpaths)

        # Notify the MosaicPanel about the boxes
        #box_locs = convert_boxes2mosaic(self.project, self.world.box_locations)
        #self.panel_mosaic.set_boxes(box_locs)
        #self.panel_mosaic.set_boxes(self.world.box_locations,
        #                            transfn=make_transfn(self.project))
        self.panel_mosaic.set_transfn(make_transfn(self.project))
        Publisher().sendMessage("broadcast.updated_world")

        # Display first template on BallotScreen
        imgpath = imgpaths[0]
        img = util_gui.open_as_grayscale(imgpath)
        target_locations = self.world.get_boxes(imgpath)
        self.panel_mosaic.select_image(imgpath)
        Publisher().sendMessage("signals.ballotviewer.set_image_pil", (imgpath, img))
        Publisher().sendMessage("signals.BallotScreen.set_bounding_boxes", (imgpath, target_locations))        
        Publisher().sendMessage("signals.BallotScreen.update_state", BallotScreen.STATE_IDLE)

        # Try to pull in previously-saved frontback map
        try:
            self.import_frontback()
        except IOError as e:
            print 'error in importing frontback.'
            pass

        # Prepopulate self.frontback_map with template paths if
        # necessary (i.e. if this is the first time running)
        for templatepath in self.world.box_locations:
            if templatepath not in self.frontback_map:
                # Default to 'front'
                self.frontback_map[templatepath] = 'front'

        # Set frontbackpanel's Front/Back radio buttons accordingly
        self.update_frontbackpanel()
        
        self.panel_mosaic.Show()
        self.ballotviewer.Show()

        if util.is_multipage(self.project):
            self.frontbackpanel.Show()

        self.sanity_check_grouping()
        
        self.Refresh()
        self.Fit()
                
    def update_frontbackpanel(self):
        """
        Set the radio buttons according to the values in
        frontback_map for the given image.
        """
        if self.frontback_map:
            cur_imgpath = self.ballotviewer.ballotscreen.current_imgpath
            side = self.frontback_map[cur_imgpath]
            self.frontbackpanel.set_side(side)

    def update_frontbackmap(self):
        cur_imgpath = self.ballotviewer.ballotscreen.current_imgpath
        if cur_imgpath == None:
            return
        self.frontback_map[cur_imgpath] = self.frontbackpanel.get_side()

    def apply_target_grouping(self):
        """
        Tries to group voting targets into clusters (which should 
        correspond to individual contests). This clustering will be
        reflected in the UI as green bounding boxes. Assumes that
        self.apply_template_matching has already been called, so that
        self.box_locations contains all voting targets and contest
        locations.
        Also updates contest_id's for voting targets and contests.
        """
        cur_id = 0
        for templatepath in self.world.get_boxes_all():
            targets = [box.get_coords() for box in self.world.get_boxes(templatepath) if not box.is_contest]
            # groups is a tuple of ((x1_i,y1_i,x2_i,y2_i), ...)
            groups = grouptargets.do_group_hist(targets, epsilon=0.215) # used to be 1.15
            groups_post = process_groups(groups)
            n1 = sum([len(group) for group in groups])
            n2 = sum([len(group) for group in groups_post])
            assert n1 == n2
            groups = groups_post
            # find bounding box around each group
            assert sum(map(lambda lst: len(lst), groups)) == len(targets), "{0} targets, but there were {1} targets in groups".format(len(targets), sum(map(lambda lst: len(lst), groups)))
            contest_boxes = find_bounding_boxes(groups)
            # 'normalize' bounding box sizes so that they don't extend
            # outside the image
            contest_boxes = normalize_boxes(contest_boxes, (self.ballotviewer.ballotscreen.img_bitmap.GetWidth(),
                                                            self.ballotviewer.ballotscreen.img_bitmap.GetHeight()))
            for contest in sorted(contest_boxes, key=lambda box: box.x1):
                contest.contest_id = cur_id
                cur_id += 1
            '''
            # We want to keep all frozen contests alive
            ignore = []
            for target in [b for b in self.world.get_boxes(templatepath) if not box.is_contest]:
                # If target is associated with a frozen contest...
                if util_gui.find_assoc_contest(target, self.get_frozen_contests(templatepath)):
                    ignore.append(target)
            # Filter out all unnecessary new contests
            for ignore_target in ignore:
                contest = util_gui.find_assoc_contest(ignore_target, contest_boxes)
                if contest:
                    pdb.set_trace()
                    contest_boxes.remove(contest)
            # Add in frozen contests
            contest_boxes.extend(self.get_frozen_contests(templatepath))
            '''
            # Finally, update target-contest associations
            for target in [box for box in self.world.get_boxes(templatepath) if not box.is_contest]:
                assoc_contest = util_gui.find_assoc_contest(target, contest_boxes)
                if assoc_contest:
                    target.contest_id = assoc_contest.contest_id
                else:
                    print "Couldn't find a contest for this target."
            self.remove_contests(templatepath)
            self.world.add_boxes(templatepath, contest_boxes)

        # Notify the MosaicPanel about the new boxes
        #box_locs = convert_boxes2mosaic(self.project, self.world.box_locations)
        #self.panel_mosaic.set_boxes(box_locs)
        #self.panel_mosaic.set_boxes(self.world.box_locations,
        #                            transfn=make_transfn(self.project))
        self.panel_mosaic.set_transfn(make_transfn(self.project))
        self.Refresh()

    def sanity_check_grouping(self):
        """
        Apply sanity checks to the target grouping. Assumes that
        self.apply_target_grouping has already been called.
        Returns True if everything was ok, False o.w.
        """
        atleastone = False
        ctr = 0
        imgpaths = []
        for temppath, boxes in self.world.get_boxes_all().items():
            for contest in [b for b in boxes if b.is_contest]:
                atleastone = True
                assoc_targets = util_gui.associated_targets(contest, self.world.get_boxes(temppath))
                if len(assoc_targets) == 1:
                    contest.set_color("Red")
                    imgpaths.append(temppath)
                    ctr += 1
        Publisher().sendMessage("broadcast.cant_proceed")
        if ctr > 0:
            msg = "Warning: There were {0} contests with only one voting \
bubble detected. \nDouble check these contests (colored Red) to see if any \
voting bubbles were missed. \n\
Press 'Ok' to jump to a blank ballot with a problematic contest.".format(ctr)
            dlg = wx.MessageDialog(self, message=msg, style=wx.OK)
            self.parent.Disable()
            dlg.ShowModal()
            self.parent.Enable()
            # Jump to the 'first' blank ballot with a problem.
            pagenum, row, col = None, None, None
            path = None
            for imgpath in imgpaths:
                pagenumA, rowA, colA = self.panel_mosaic.get_img_info(imgpath)
                if pagenum == None or pagenumA < pagenum:
                    pagenum,row,col = pagenumA, rowA, colA
                    path = imgpath
                elif pagenumA == pagenum:
                    if rowA < row:
                        pagenum,row,col = pagenumA, rowA, colA
                        path = imgpath
                    elif rowA == row:
                        if colA < col:
                            pagenum,row,col = pagenumA, rowA, colA
                            path = imgpath
            print "Jumping to page {0} -- selected blank ballot is on \
row {1}, col {2}".format(pagenum, row, col)
            dlg = wx.MessageDialog(self, message="Jumping to page {0}. \
The problematic blank ballot is on row {1}, col {2}".format(pagenum,row,col),
                                   style=wx.OK)
            self.Disable()
            dlg.ShowModal()
            self.Enable()
            self.panel_mosaic.display_page(pagenum)
            self.panel_mosaic.select_image(path)
            self.Refresh()
            return False
        elif atleastone:
            # User may be done with this task, so emit a message
            # to allow UI to signal to user to move on.
            # '1' stands for my index in the NoteBook's tabs list
            Publisher().sendMessage("broadcast.can_proceed")
        self.Refresh()
        return True

    def remove_contests(self, templatepath):
        """
        Remove all contests from self.box_locations that are for
        templatepath. Keeps all voting targets though.
        """
        self.world.remove_contests(templatepath)
    def remove_targets(self, templatepath):
        """
        Remove all voting targets from self.box_locations that are for
        templatepath. Keeps all contest bounding boxes though.
        """
        self.world.remove_voting_targets(templatepath)
    def set_contests(self, templatepath, contests):
        """ For a given templatepath, set its list of contest bounding
        boxes to contests.
        """
        self.remove_contests(templatepath)
        self.world.add_boxes(templatepath, contests)

    def export_bounding_boxes(self):
        """ 
        Export box locations to csv files. Also, returns the BoundingBox
        instances for each template image as a dict of the form:
            {str templatepath: list boxes}
        Because the user might have manually changed the bounding
        boxes of each contest (by modifying the size of a contest
        box), this function will re-compute the correct contest_ids
        for all voting target BoundingBoxes.
        """
        if not self.world:
            return
        elif len(self.world.get_boxes_all().items()) == 0:
            # No templates loaded
            return
        elif len(self.world.get_boxes_all_list()) == 0:
            # No bounding boxes created
            return
        #util_gui.create_dirs(CSVFILES_DIR)
        util_gui.create_dirs(self.project.target_locs_dir)
        # First update contest ids for all boxes, since user might 
        # have changed bounding boxes
        updated_boxes = {}
        for templatepath, boxes in self.world.get_boxes_all().items():
            updated_boxes[templatepath] = compute_contest_ids(boxes)
        self.world.box_locations = updated_boxes

        if self.world.get_boxes_all().keys():
            h_img, w_img = util_gui.open_img_scipy(self.world.get_boxes_all().keys()[0]).shape
            
        csvpath_map = {} # maps {str csvpath: str template_imgpath}
        # Enforce the rule that all voting targets have the same dimension
        def get_max_dimensions(boxes):
            """
            Return max(width, height) out of all boxes
            """
            w, h = None, None
            for b in boxes:
                _w, _h = b.width, b.height
                if not w or _w > w:
                    w = _w
                elif not h or _h > h:
                    h = _h
            return w, h
        w_target, h_target = get_max_dimensions([b for b in self.world.get_boxes_all_list() if not b.is_contest])
        w_target = int(round(w_target * w_img))
        h_target = int(round(h_target * h_img))
        fields = ('imgpath', 'id', 'x', 'y', 'width', 'height', 'label', 'is_contest', 'contest_id')
        for imgpath in self.world.get_boxes_all():
            ## We comment this out to avoid breaking things downstream,
            ## but at some point this change will be necessary
            ## to handle blank ballots that have the same filenames
            ## (as in Napa)
            # TODO: Apply this change.
            #tdir = self.project.templatesdir
            #if tdir[-1] != '/':
            #    tdir += '/'
            #basedir = imgpath[len(tdir):]
            basedir = ''
            csvfilepath = pathjoin(self.project.target_locs_dir,
                                   basedir, 
                                   "{0}_targetlocs.csv".format(os.path.splitext(os.path.split(imgpath)[1])[0]))
            util_gui.create_dirs(os.path.split(csvfilepath)[0])
            csvfile = open(csvfilepath, 'wb')
            csvpath_map[csvfilepath] = os.path.abspath(imgpath)
            dictwriter = csv.DictWriter(csvfile, fieldnames=fields)
            try:
                dictwriter.writeheader()
            except AttributeError:
                util_gui._dictwriter_writeheader(csvfile, fields)

            for id, bounding_box in enumerate(self.world.get_boxes(imgpath)):
                x1, y1, x2, y2 = bounding_box.get_coords()
                row = {}
                row['imgpath'] = os.path.abspath(imgpath)
                row['id'] = id
                # Convert relative coords back to pixel coords
                row['x'] = int(round(x1 * w_img))
                row['y'] = int(round(y1 * h_img))
                if bounding_box.is_contest:
                    width = int(round(abs(x1-x2)*w_img))
                    height = int(round(abs(y1-y2)*h_img))
                    row['width'] = width
                    row['height'] = height
                else:
                    row['width'] = w_target
                    row['height'] = h_target
                # Replace commas with underscore to avoid problems with csv files
                row['label'] = bounding_box.label.replace(",", "_")
                row['is_contest'] = 1 if bounding_box.is_contest else 0
                row['contest_id'] = bounding_box.contest_id
                dictwriter.writerow(row)
            csvfile.close()
        csvpath_map_filepath = pathjoin(self.project.target_locs_dir, 'csvpath_map.p')
        pickle.dump(csvpath_map, open(csvpath_map_filepath, 'wb'))
        val = copy.deepcopy(self.world.get_boxes_all())
        return val
    
    def export_frontback(self):
        """
        Exports the front/back mappings for each template image to an
        output file, given by: project.frontback_map
        """
        if not self.project:
            return
        # Make sure we write out the front/back info of the currently-
        # viewed template, to avoid losing user's work
        self.update_frontbackmap()
        pickle.dump(self.frontback_map, open(self.project.frontback_map, 'wb'))

    def import_frontback(self):
        if not self.project:
            return
        frontback_map = pickle.load(open(self.project.frontback_map, 'rb'))
        self.frontback_map = frontback_map

    #### Pubsub Callbacks

    def _final_refimg(self, msg):
        """ Receive the reference image as a PIL image """
        self.refimg = msg.data
    def pubsub_export_targets(self, msg):    
        """ Export box locations to csv files """
        self.export_bounding_boxes()
            
    def pubsub_mosaic_img_selected(self, msg):
        """ Triggered when the user clicks a blank ballot in the Mosaic
        panel. """
        imgpath = msg.data
        img = util_gui.open_as_grayscale(imgpath)
        target_locations = self.world.get_boxes(imgpath)
        self.update_frontbackmap()
        Publisher().sendMessage("signals.ballotviewer.set_image_pil", (imgpath, img))
        Publisher().sendMessage("signals.BallotScreen.set_bounding_boxes", (imgpath, target_locations))
        # Update frontbackpanel
        self.update_frontbackpanel()
        
    def _pubsub_updatedworld(self, msg):
        """
        Triggered when the user modifies the voting targets in the
        BallotScreen (during Mosaic verify).
        """
        self.panel_mosaic.Refresh()
        self.Refresh()
    def pubsub_added_target(self, msg):
        """
        Triggered when the user adds a new bounding box to the image.
        Run template matching across all template images with this
        as the reference image. In addition, this will run template
        matching several times in order to detect as many targets as
        it can.
        """
        def sanity_check(box_coords):
            """
            Makes sure box_coords is of the form:
                box_coords : (ul_x, ul_y, lr_x, lr_y)
            In addition, make sure that it's a box with nonzero area,
            and that it's at least 2x2 pixels
            """
            MIN_DIM = 2
            ul_x, ul_y, lr_x, lr_y = box_coords
            return ((abs(ul_x - lr_x) >= MIN_DIM) and
                    (abs(ul_y - lr_y) >= MIN_DIM) and
                    ul_x < lr_x and ul_y < lr_y and 
                    ul_x != lr_x and ul_y != lr_y)
                    
        self.TIMER.stop_task(('user', 'Select/Group Voting Targets'))
        self.TIMER.start_task(('cpu', 'TemplateMatch Targets Computation'))
        imgpath, box = msg.data
        box = normalize_boxes((box,), self.ballotviewer.get_imgsize())[0]
        box_copy = box.copy()
        box_copy.set_color("Green")
        # First do an autofit on the selected region
        img_pil = util_gui.open_as_grayscale(imgpath)
        w_img, h_img = img_pil.size
        ul_x = intround(box.x1*w_img)
        ul_y = intround(box.y1*h_img)
        lr_x = intround(box.x2*w_img)
        lr_y = intround(box.y2*h_img)
        if not sanity_check((ul_x,ul_y,lr_x,lr_y)):
            return
        if self.world.get_boxes_count_all() == 0:
            # The user did drag-and-drop creation, so do autofit
            region_pil = img_pil.crop((ul_x, ul_y, lr_x, lr_y))
            refimg_rect = np.array(util_gui.fit_image(region_pil, padx=0, pady=0), dtype='f')
            exemplar_boxdims = None
        else:
            # Run template matching on the user-selected region,
            # and find a new 'fitted' bounding box around the newly
            # detected voting target
            def get_exemplar_target(world):
                for temppath, boxes in self.world.box_locations.iteritems():
                    targets = [b for b in boxes if not b.is_contest]
                    if targets:
                        return temppath, targets[0]
                print "== Error in get_exemplar_target: Couldn't find a \
single voting target, which violates assumptions."
                return None, None
            exemplar_temppath, b = get_exemplar_target(self.world)
            exemplar_boxdims = b.width, b.height
            if exemplar_temppath == self.ballotviewer.ballotscreen.current_imgpath:
                exemplar_img = img_pil
            else:
                exemplar_img = util_gui.open_as_grayscale(exemplar_temppath)
            target_pil = exemplar_img.crop((intround(b.x1*w_img), intround(b.y1*h_img),
                                           intround(b.x2*w_img), intround(b.y2*h_img)))
            # Extract region from current image
            ex_w = int(round(exemplar_boxdims[0]*w_img))
            ex_h = int(round(exemplar_boxdims[1]*h_img))
            if abs(ul_x - lr_x) < ex_w:
                # Extend region so that it's larger
                delta = abs(ex_w - abs(ul_x - lr_x))
                if lr_x == (img_pil.size[0] - 1):
                    ul_x -= delta
                else:
                    lr_x += delta
            if abs(ul_y - lr_y) < ex_h:
                delta = abs(ex_h - abs(ul_y - lr_y))
                if lr_y == (img_pil.size[1] - 1):
                    ul_y -= delta
                else:
                    lr_y += delta
                
            region_pil = img_pil.crop((ul_x, ul_y, lr_x, lr_y))

            region_np = np.array(region_pil, dtype='f')
            target_np = np.array(target_pil, dtype='f')
                        
            region_cv = cv.fromarray(region_np)
            target_cv = cv.fromarray(target_np)

            try:
                outCv = cv.CreateMat(region_np.shape[0]-target_np.shape[0]+1,
                                     region_np.shape[1]-target_np.shape[1]+1,
                                     region_cv.type)
            except Exception as e:
                print e
                pdb.set_trace()
            cv.MatchTemplate(region_cv, target_cv, outCv, cv.CV_TM_CCOEFF_NORMED)
            Iout = np.asarray(outCv)
            
            (y,x) = np.unravel_index(Iout.argmax(), Iout.shape)
            w_box = intround(b.width*w_img)
            h_box = intround(b.height*h_img)
            r3 = [y, y+h_box, x, x+w_box]
            newtarget = region_np[r3[0]:r3[1], r3[2]:r3[3]]
            refimg_rect = newtarget
        
        # On rare occasions, refimg_rect is a 0-d array, which breaks
        # things. Check for this, and other degenerate cases
        if refimg_rect.shape == ():
            return
        elif refimg_rect.shape[0] in (0,1) or refimg_rect.shape[1] in (0,1):
            return
            
        if not sanity_check_box(refimg_rect):
            x1,y1,x2,y2 = box.get_coords()
            x1, x2 = map(lambda x: int(round(x*w_img)), (x1,x2))
            y1, y2 = map(lambda y: int(round(y*h_img)), (y1,y2))
            user_region = np.array(img_pil.crop((x1,y1,x2,y2)), dtype='f')
            dlg = WarningSelectedRegion(self, refimg_rect, user_region)
            # if I don't disable this, then the click to exit modal
            # mode will end up creating a new target
            self.Disable()
            response = dlg.ShowModal()
            self.Enable()
            if response in (wx.ID_NO, wx.ID_CANCEL):
                return
            elif response == WarningSelectedRegion.OVERRIDE:
                refimg_rect = user_region
        try:
            param = float(self.project.tempmatch_param)
        except AttributeError:
            param = self.TEMPMATCH_DEFAULT_PARAM
        timelogfile = open(pathjoin(self.project.projdir_path, 'tempmatch_timing.log'), 'a')
        first_time = True if self.world.get_boxes_count_all() == 0 else False
        t = ThreadTempMatch(self.world.box_locations, 
                            refimg_rect, 
                            first_time,
                            param,
                            exemplar_boxdims,
                            outfile=timelogfile)
        t.start()
        gauge = TempMatchProgress(self)
        gauge.Show()
        self.Disable()

    def _pubsub_added_contest(self, msg):
        """
        Triggered when the user adds a new contest bounding box to the
        image.
        """
        imgpath, box = msg.data
        box = box.copy()
        box.restore_color()
        self.world.add_box(imgpath, box)
        self.Refresh()

    def _pubsub_tempmatchdone(self, msg):
        """
        Triggered when Template Matching is done - now, do target
        grouping.
        """
        self.apply_target_grouping()
        self.sanity_check_grouping()
        self.TIMER.stop_task(('cpu', 'TemplateMatch Targets Computation'))
        self.TIMER.start_task(('user', 'Select/Group Voting Targets'))
        Publisher().sendMessage("signals.MosaicPanel.update_all_boxes", self.world.get_boxes_all())

    def _pubsub_deleted_targets(self, msg):
        """
        Triggered when an outside source has deleted some 
        BoundingBoxes.
        """
        imgpath, del_boxes = msg.data
        for contest in [b for b in del_boxes if b.is_contest]:
            if contest in self.get_frozen_contests(imgpath):
                self.frozen_contests[imgpath].remove(contest)
        self.Refresh()

    def _pubsub_project(self, msg):
        """
        Triggered when the user selects a Project. Pull in relevant
        state. Also, 'mutate' the current WorldState to reflect the
        project change.
        """
        project = msg.data
        self.project = project
        timelogfile = open(pathjoin(self.project.projdir_path, 'tempmatch_timing.log'), 'a')
        print >>timelogfile, '================================'
        print >>timelogfile, datetime.datetime.now()
        print >>timelogfile, '================================'
        timelogfile.close()

        self.project.addCloseEvent(self.export_bounding_boxes)
        self.project.addCloseEvent(self.export_frontback)
        if not project.templatesdir:
            # user hasn't gotten to this part yet
            # Reset the WorldState
            self.world.reset()
            return
        # annoying: get a template image size
        # Assumes that all templates are the same image size
        imgpath = ''
        for dirpath, dirnames, filenames in os.walk(project.templatesdir):
            imgs = [pathjoin(dirpath, f) for f in filenames if util_gui.is_image_ext(f)]
            if imgs:
                imgpath = imgs[0]
                break
        if not imgpath:
            print "Error: imgpath was {0} in SpecifyTargetPanel._pubsub_project".format(imgpath)
            exit(1)
        imgsize = Image.open(imgpath).size
        worldstate = import_worldstate(project.target_locs_dir, imgsize)
        self.world.mutate(worldstate)
        self.import_frontback()

    def _pubsub_projupdate(self, msg):
        """
        Triggered when a change is made to the Project.
        """
        pass

    def do_infer_contests(self):
        """ Use Nicholas' contest-region-inferring code, display the
        results on the screen, and allow the user to adjust it. 
        Repeated calls to this function will discard previous
        contest-region-inferring results.
        """
        self.queue = Queue.Queue()
        self.export_bounding_boxes()
        t = ThreadDoInferContests(self.queue, self.INFERCONTESTS_JOB_ID,
                                  self.project)

        gauge = util.MyGauge(self, 1, thread=t, ondone=self.on_infer_contests_done,
                             msg="Inferring Contest Regions...",
                             job_id=self.INFERCONTESTS_JOB_ID)
        t.start()
        gauge.Show()

    def on_infer_contests_done(self):
        """ Display the inferring results to the screen, and allow the
        user to manually change the bounds if necessary.
        results will contain the bounding boxes of each contest, for
        every ballot image:
            {str blankpath: list of (x1, y1, x2, y2)}
        """
        results = self.queue.get()
        print "AND I GET THE RESULTS", results
        w_img, h_img = self.project.imgsize
        w_img, h_img = float(w_img), float(h_img)
        for blankpath, contests in results.iteritems():
            contestboxes = []
            for (x1, y1, x2, y2) in contests:
                # Set all contest bounding boxes to this
                x1a = x1 / w_img
                y1a = y1 / h_img
                x2a = x2 / w_img
                y2a = y2 / h_img
                box = BoundingBox(x1a, y1a, x2a, y2a, is_contest=True)
                contestboxes.append(box)
            self.remove_contests(blankpath)
            self.world.add_boxes(blankpath, contestboxes)
        Publisher().sendMessage("broadcast.updated_world")
            
                
class ThreadDoInferContests(threading.Thread):
    def __init__(self, queue, job_id, proj, *args, **kwargs):
        threading.Thread.__init__(self)
        self.job_id = job_id
        self.queue = queue
        self.proj = proj

    def extract_data(self):
        """
        Stolen from labelcontest.py.

        This should be removed in favor of taking the data from
        this panel directly, instead of loading from the file.
        """
        res = []
        dirList = []
        for root,dirs,files in os.walk(self.proj.target_locs_dir):
            util.sort_nicely(files) # Fixes Marin ordering.
            for each in files:
                if each[-4:] != '.csv': continue
                gr = {}
                name = os.path.join(root, each)
                for i, row in enumerate(csv.reader(open(name))):
                    if i == 0:
                        # skip the header row, to avoid adding header
                        # information to our data structures
                        continue
                    # If this one is a target, not a contest
                    if row[7] == '0':
                        if row[8] not in gr:
                            gr[row[8]] = []
                        # 2,3,4,5 are left,up,width,height but need left,up,right,down
                        gr[row[8]].append((int(row[2]), int(row[3]), 
                                           int(row[2])+int(row[4]), 
                                           int(row[3])+int(row[5])))
                    if row[0] not in dirList:
                        dirList.append(row[0])
                res.append(gr.values())
        return res, dirList
        
    def run(self):
        # Do fancy contest-inferring computation
        data, files = self.extract_data()
        bboxes = dict(zip(files,find_contests(self.proj.ocr_tmp_dir, files, data)))
        # Computation done!
        self.queue.put(bboxes)
        self.proj.infer_bounding_boxes = True
        print "AND I SEND THE RESUTS", bboxes
        wx.CallAfter(Publisher().sendMessage, "signals.MyGauge.done",
                     (self.job_id,))

        
def import_worldstate(csvdir, imgsize):
    """
    Given a directory containing csv files, return a new worldstate.
    """
    world = WorldState()
    boxes = util_gui.import_box_locations(csvdir, imgsize)
    for temppath, box_locations in boxes.items():
        world.add_boxes(temppath, box_locations)
    return world
                
class MosaicPanel2(util_widgets.MosaicPanel):
    """
    Behaves just like util_widget's MosaicPanel, but with a few
    OpenCount-specific things added in (like Pubsub hooks).
    """
    def __init__(self, parent, world, *args, **kwargs):
        util_widgets.MosaicPanel.__init__(self, parent, imgmosaicpanel=ImageMosaicPanel_OpenCount, 
                                          _init_args=(world,), 
                                          *args, **kwargs)
        self.world = world
        # Pubsubs
        self.callbacks = [("signals.MosaicPanel.update_all_boxes", self._pubsub_update_all_boxes),
                          ("signals.MosaicPanel.update_boxes", self._pubsub_update_boxes),
                          ("broadcast.deleted_targets", self._pubsub_deleted_targets),
                          ("broadcast.updated_world", self._pubsub_updated_world)]
    def subscribe_pubsubs(self):
        for (topic, callback) in self.callbacks:
            Publisher().subscribe(callback, topic)
        
    def unsubscribe_pubsubs(self):
        for (topic, callback) in self.callbacks:
            Publisher().unsubscribe(callback, topic)

    def _pubsub_update_all_boxes(self, msg):
        """
        Triggered usually when an operation is done that modifies 
        BoundingBoxes for all template images (say, after a 
        template-matching run).
        msg.data = box_locations, where box_locations is a dict 
        mapping templatepath to a list of BoundingBox instances.
        """
        box_locations = msg.data
        self.Refresh()
        #self.update_all_targets(box_locations)
    def _pubsub_update_boxes(self, msg):
        """
        Used to update the box locations for only one template image.
        msg.data := str templatepath, list box_locations
        """
        templatepath, box_locations = msg.data
        self.Refresh()
        #self.update_targets(templatepath, box_locations)
    def _pubsub_deleted_targets(self, msg):
        """
        Triggered when the user deletes a BoundingBox (via 
        MyBallotScreen).
        msg.data := str templatepath, list BoundingBox
        """
        templatepath, boxes = msg.data
        #self.remove_targets(templatepath, boxes)
        self.Refresh()
    def _pubsub_updated_world(self, msg):
        """
        Triggered whenever the WorldState gets updated. Since this
        probably means box locations got changed, I should redraw
        myself. For performance, it'd be a good idea to only redraw
        things that have changed - but for now, just redraw everything.
        """
        self.Refresh()
        self.imagemosaic.Refresh()

class ImageMosaicPanel_OpenCount(util_widgets.ImageMosaicPanel):
    """
    A class that is meant to be integrated directly into OpenCount.
    """
    def __init__(self, parent, world, *args, **kwargs):
        util_widgets.ImageMosaicPanel.__init__(self, parent, *args, **kwargs)
        self.world = world
        
    def get_boxes(self, imgpath):
        return self.world.get_boxes(imgpath)

class MyBallotScreen(BallotScreen):
    """
    Basically, exactly the same as BallotScreen, but, the first 
    bounding box is created via click-and-drag, and subsequent
    boxes are place-and-drop (whose dims are specified by the first
    bounding box).
    """
    
    def __init__(self, parent, world, *args, **kwargs):
        BallotScreen.__init__(self, parent, world, *args, **kwargs)
        
        self.ghost_box = None   # BoundingBox to show on mouse cursor
        self.display_ghost_box = False  # If True, display self.ghost_box
        
    def subscribe_pubsubs(self):
        BallotScreen.subscribe_pubsubs(self)
        callbacks = (("broadcast.undo", self._pubsub_undo),)
        self.callbacks.extend(callbacks)
        for (topic, callback) in callbacks:
            Publisher().subscribe(callback, topic)

    def _pubsub_updated_world(self, msg):
        """
        If no boxes remain on this template, then nuke the ghost_box.
        """
        if not self.world.get_boxes(self.current_imgpath):
            self.ghost_box = None
            self.display_ghost_box = False
            
        BallotScreen._pubsub_updated_world(self, msg)
    def _pubsub_undo(self, msg):
        """
        Triggered whenever the user does 'Ctrl-z' anywhere in the UI.
        Since the user might not be on the tab for 'Select and Group
        Voting Targets', only do undo if this tab is currently the one
        displayed. If nothing is sent in msg.data, then do an 
        unconditional undo.
        msg.data := int page, where 'page' is a 0-indexed integer
            representing which page is displayed
        """
        page_num = msg.data
        if not page_num or page_num == 2: # hard-coded constant for this page.
            self.undo()
        
    def set_state(self, newstate):
        oldstate = self.curstate
        if (oldstate == BallotScreen.STATE_ADD_TARGET
                and newstate != BallotScreen.STATE_ADD_TARGET):
            self.display_ghost_box = False
        if (newstate in (BallotScreen.STATE_ADD_TARGET, BallotScreen.STATE_ADD_CONTEST)):
            self.display_ghost_box = True
        BallotScreen.set_state(self, newstate)
        
    def onLeftDown(self, evt):
        self.SetFocus()
        x, y = self.CalcUnscrolledPosition(evt.GetPositionTuple())
        if (self.curstate in (BallotScreen.STATE_ADD_TARGET, BallotScreen.STATE_ADD_CONTEST)):
            if False: #self.ghost_box:
                # User previously created a bounding box, so 'place'
                # down a pre-sized box at current location
                x_rel, y_rel = x / float(self.img_bitmap.GetWidth()), y / float(self.img_bitmap.GetHeight())
                
                w_box, h_box = self.ghost_box.width, self.ghost_box.height
                new_box = BoundingBox(x_rel, y_rel, x_rel+w_box, y_rel+h_box, color="Purple")
                self.set_new_box(new_box)
                self.Refresh()
            else:
                # User is creating his/her first bounding box via
                # 'drag and drop'
                BallotScreen.onLeftDown(self, evt)
        else:
            BallotScreen.onLeftDown(self, evt)
    
    def onMotion(self, evt):
        x, y = self.CalcUnscrolledPosition(evt.GetPositionTuple())
        x_rel, y_rel = x / float(self.img_bitmap.GetWidth()), y / float(self.img_bitmap.GetHeight())
        if (self.curstate in (BallotScreen.STATE_ADD_TARGET, BallotScreen.STATE_ADD_CONTEST)
                and self.is_new_box()):
            BallotScreen.onMotion(self,evt)
        elif (self.curstate in (BallotScreen.STATE_ADD_TARGET, BallotScreen.STATE_ADD_CONTEST)
                and not self.is_new_box()):
            self.Refresh()
            BallotScreen.onMotion(self, evt)
        else:
            BallotScreen.onMotion(self, evt)
    
    def onPaint(self, evt):
        """ Refresh screen. """
        if self.IsDoubleBuffered():
            dc = wx.PaintDC(self)
        else:
            dc = wx.BufferedPaintDC(self)
        # You must do PrepareDC in order to force the dc to account
        # for scrolling.
        self.PrepareDC(dc)
        
        dc.DrawBitmap(self.img_bitmap, 0, 0)
        self._display_targets(dc)
        if self.curstate == BallotScreen.STATE_AUTODETECT and self._autodet_rect:
            self._draw_autodet_rect(dc)
        #if self.auxstate == BallotScreen.STATE_RESIZE_TARGET:
        if self.is_resize_target():
            self._draw_resize_rect(dc)
        if self.curstate == BallotScreen.STATE_AUTODETECT_VERIFY:
            self._draw_candidate_targets(dc, self.candidate_targets)
        if self._dragselectregion:
            self._draw_dragselectregion(dc)        
        if self.display_ghost_box and self.ghost_box:
            self._draw_box(dc, self.ghost_box)
        evt.Skip()

class Figure(wx.Panel):
    """
    A class that combines an image and a caption.
    """
    def __init__(self, parent, imgpath='', caption='Caption', 
                 img=None, height=100, *args, **kwargs):
        """
        You can pass in the image in several ways: either as a path
        to the image (imgpath), or as the img itself (img), in either
        PIL, numpy array, or wxBitmap/wxImage format.
        
        obj parent: Parent widget
        str imgpath: Path to image
        str caption: Text that will be displayed under the image
        obj img: PIL Image, wxImage, or wxBitmap
        int height: Height (in pixels) of the displayed figure.
        """
        wx.Panel.__init__(self, parent, style=wx.SIMPLE_BORDER, *args, **kwargs)
        
        ## Instance vars
        self.parent = parent
        self.imgpath = imgpath
        self.caption = caption
        self.img = img
        self.img_bitmap = None  # Should be set by _set_image
        self._set_image(imgpath, img, height)
        
        self.panel_img = wx.Panel(self)
        self.panel_img.SetMinSize((self.img_bitmap.GetWidth(), height))
        
        self.panel_caption = wx.Panel(self)
        txt = wx.StaticText(self.panel_caption, label=caption)
        txt.Wrap(self.img_bitmap.GetWidth())
        self.panel_caption.sizer = wx.BoxSizer(wx.VERTICAL)
        self.panel_caption.sizer.Add(txt, flag=wx.ALIGN_CENTER)
        self.panel_caption.SetSizer(self.panel_caption.sizer)
        self.panel_caption.Fit()
        
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.panel_img, border=10, flag=wx.ALL | wx.EXPAND)
        self.sizer.Add(self.panel_caption, border=10, flag=wx.ALL | wx.EXPAND)
        self.SetSizer(self.sizer)
        
        self.panel_img.Bind(wx.EVT_PAINT, self.onPaint_img)
        
        #self.img_bitmap.ConvertToImage().SaveFile('img_bitmap.png', wx.BITMAP_TYPE_PNG)

        self.Fit()
        
    def compute_size(self):
        return self.sizer.GetMinSize()

    def _set_image(self, imgpath, img, height):
        if img != None:
            try:
                # PIL Image
                w_img, h_img = img.size
                c = h_img / float(height)
                new_w = int(round(w_img / c))
                pil_resize = img.resize((new_w, height), Image.ANTIALIAS)
                self.img_bitmap = util_gui.PilImageToWxBitmap(pil_resize)
                return
            except:
                # Well, it's not a PIL Image!
                pass
            try:
                # Numpy array
                h_img, w_img = img.shape
                c = float(height) / h_img
                try:
                    resized_img = scipy.misc.imresize(img, c, interp='bilinear')
                except TypeError as e:
                    # scipy v0.7.0 doesn't take interp kwrd arg, but
                    # v0.10.0rc1 does.
                    resized_img = scipy.misc.imresize(img, c)
                self.img_bitmap = util_gui.NumpyToWxBitmap(resized_img)
                return
            except Exception as e:
                # Well, it's not a Numpy array!
                pass
            try:
                # wxImage
                w_img, h_img = img.GetWidth(), img.GetHeight()
                c = h_img / float(height)
                new_w = int(round(w_img / c))
                self.img_bitmap = img.Rescale(new_w, height, wx.IMAGE_QUALITY_HEIGHT).ConvertToBitmap()
                return
            except:
                # Well, it's not a wxImage
                pass
            try:
                # wxBitmap
                pil = util_gui.WxBitmapToPilImage(img)
                w_img, h_img = pil.size
                c = h_img / float(height)
                new_w = int(round(w_img / c))
                pil_resize = pil.resize((new_w, height), Image.ANTIALIAS)
                self.img_bitmap = util_gui.PilImageToWxBitmap(pil_resize)
                return
            except:
                # wat
                print 'Unrecognized input to Figure._set_image:', type(img)
                raise RuntimeError("Wat, in Figure._set_image")
        elif imgpath:
            self.img_pil = util_gui.open_img_as_grayscale(imgpath)
            w_img, h_img = self.img_pil.size
            c = h_img / float(height)
            new_w = int(round(w_img / c))
            img_pil_resize = self.img_pil.resize((new_w, height), Image.ANTIALIAS)
            
            self.img_bitmap = util_gui.PilImageToWxBitmap(img_pil_resize)
        else:
            # User didn't pass in an img nor imgpath, so set to blank
            # bitmap
            self.img_bitmap = util_gui.make_blank_bitmap((100, 100), 200)
            
    def onPaint_img(self, evt):
        if self.IsDoubleBuffered():
            dc = wx.PaintDC(self.panel_img)
        else:
            dc = wx.BufferedPaintDC(self.panel_img)
        dc.DrawBitmap(self.img_bitmap, 0, 0)
        evt.Skip()
        
class WarningSelectedRegion(wx.Dialog):
    """
    A Dialog that pops up when the system detects that the user selected
    something weird.
    """
    OVERRIDE = 42
    def __init__(self, parent, refimg_rect, user_region, *args, **kwargs):
        """
        obj refimg_prect: A numpy array for the auto-created region
        obj user_region: A numpy array for the exact region that the
                         user created (useful for overriding auto-crop)
        """
        wx.Dialog.__init__(self, parent, title="Warning -- Possible Voting Target Error", *args, **kwargs)
        self.parent = parent
        
        self.panel = wx.Panel(self)
        self.panel.sizer = wx.BoxSizer(wx.VERTICAL)
        self.panel.SetSizer(self.panel.sizer)

        self.panel_figs = wx.Panel(self.panel) # figures, btns
        sizer_figs = wx.BoxSizer(wx.HORIZONTAL)
        self.panel_figs.SetSizer(sizer_figs)
        # Left side (autocrop region, yes/no btns)
        self.fig = Figure(self.panel_figs, img=refimg_rect, caption="An auto-cropped region.")
        self.fig2 = Figure(self.panel_figs, img=user_region, caption="The exact region you selected.")
        sizer_figs.Add(self.fig, border=10, flag=wx.ALL | wx.ALIGN_TOP)
        sizer_figs.Add(self.fig2, border=10, flag=wx.ALL | wx.ALIGN_TOP)
        self.panel_figs.Fit()
        msg = """OpenCount detected \
that you might have made a mistake when selecting the voting target. \
Displayed on the left is the region that OpenCount received. Does this region \
only contain the full voting target? 

If the image on the left contains only voting \
target in its entirety, and nothing else, then click yes to proceed."""
        msg2 = """If parts of the \
target is missing (or there's non-voting target pixels present), then \
click 'No' and retry the selection. Zooming-in might help with precision."""
        msg3 = """Or, if you want to force OpenCount to use exactly what you \
selected (the image on the right), then click 'Use what I selected"."""
        self.txt = wx.StaticText(self.panel, label=msg)
        self.txt.Wrap(450)
        self.txt2 = wx.StaticText(self.panel, label=msg2)
        self.txt2.Wrap(450)
        self.txt3 = wx.StaticText(self.panel, label=msg3)
        self.txt3.Wrap(450)
        
        panel_btn = wx.Panel(self.panel, style=wx.SIMPLE_BORDER)
        self.btn_yes = wx.Button(panel_btn, id=wx.ID_YES)
        self.btn_yes.Bind(wx.EVT_BUTTON, self.onButton_yes)
        
        self.btn_no = wx.Button(panel_btn, id=wx.ID_NO)
        self.btn_no.Bind(wx.EVT_BUTTON, self.onButton_no)

        txt = wx.StaticText(panel_btn, label="Or")

        self.btn_ignore = wx.Button(panel_btn, label="Use what I selected.")
        self.btn_ignore.Bind(wx.EVT_BUTTON, self.onButton_override)

        panel_btn.sizer = wx.BoxSizer(wx.HORIZONTAL)
        panel_btn.sizer.Add(self.btn_yes, border=10, flag=wx.ALL | wx.ALIGN_CENTER)
        panel_btn.sizer.Add(self.btn_no, border=10, flag=wx.ALL | wx.ALIGN_CENTER)
        panel_btn.sizer.Add(txt, border=15, flag=wx.LEFT | wx.RIGHT | wx.ALIGN_CENTER)
        panel_btn.sizer.Add(self.btn_ignore, border=10, flag=wx.ALL | wx.ALIGN_CENTER)
        panel_btn.SetSizer(panel_btn.sizer)
        panel_btn.Fit()

        # Debug Button
        btn_debug = wx.Button(self.panel, label="Debug")
        btn_debug.Bind(wx.EVT_BUTTON, self.onButton_debug)

        self.panel.sizer.Add(self.panel_figs, border=10, flag=wx.LEFT | wx.RIGHT | wx.ALIGN_CENTER)
        self.panel.sizer.Add(self.txt, border=10)
        self.panel.sizer.Add(wx.StaticLine(self.panel), proportion=1, flag=wx.ALIGN_CENTER | wx.EXPAND)
        self.panel.sizer.Add(self.txt2, border=10)
        self.panel.sizer.Add(wx.StaticLine(self.panel), proportion=1, flag=wx.ALIGN_CENTER | wx.EXPAND)
        self.panel.sizer.Add(self.txt3, border=10)
        self.panel.sizer.Add((10, 10))
        self.panel.sizer.Add(panel_btn, border=10, flag=wx.LEFT | wx.RIGHT | wx.ALIGN_CENTER)
        self.panel.sizer.Add(btn_debug, flag=wx.ALIGN_CENTER)
        self.panel.Fit()
        
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.panel, border=10, flag=wx.ALL)
        self.SetSizer(self.sizer)
        self.Fit()
        
    def onButton_yes(self, evt):
        self.EndModal(wx.ID_YES)
    def onButton_no(self, evt):
        self.EndModal(wx.ID_NO)
    def onButton_override(self, evt):
        self.EndModal(self.OVERRIDE)
    def onButton_debug(self, evt):
        
        print "DEBUG"
        
class TempMatchProgress(util_widgets.ProgressGauge):
    """
    A dialog that pops up to display a progress gauge when template
    matching is occuring.
    """
    def __init__(self, parent, *args, **kwargs):
        # get num of jobs - in this case, it's:
        #   <num template images> * <num_iters>
        totaljobs = len(parent.world.box_locations)*(SpecifyTargetsPanel.NUM_ITERS+1)
        msg = "Auto-detecting voting targets..."

        util_widgets.ProgressGauge.__init__(self, parent, totaljobs, msg=msg, *args, **kwargs)
        
    def _pubsub_done(self, msg):
        self.parent.Enable()
        self.Destroy()
        
def print_write(msg, file=None):
    """
    Output msg both to stdout and to an optional file.
    """
    if file:
        print >>file, msg
    print msg

class ThreadTempMatch(threading.Thread):
    def __init__(self, box_locations, refimg_rect, first_time, param, boxdims,outfile=None, *args, **kwargs):
        threading.Thread.__init__(self, *args, **kwargs)
        self.box_locations = copy.deepcopy(box_locations)
        self.refimg_rect = refimg_rect
        self.first_time = first_time
        self.param = param
        self.outfile = outfile
        self.boxdims = boxdims
    def standardize_boxsizes(self):
        """
        Given a dict mapping str temppath -> list BoundingBoxes,
        modify all boxes to have equal size.
        """
        if not self.boxdims:
            # This is the first templatematching call
            return
        w_box, h_box = self.boxdims
        for temppath in self.box_locations.keys():
            self.box_locations[temppath] = [util_gui.standardize_box(b) for b in self.box_locations[temppath]]
            for b in self.box_locations[temppath]:
                x_del = (b.x2-b.x1)-w_box
                y_del = (b.y2-b.y1)-h_box
                b.x1 += x_del / 2.0
                b.x2 -= x_del / 2.0
                b.y1 += y_del / 2.0
                b.y2 -= y_del / 2.0
                if b.x1 < 0:
                    delta = abs(b.x1)
                    b.x1 = 0
                    b.x2 += (x_del / 2.0) + delta
                if b.x2 >= 1.0:
                    delta = abs(1.0 - b.x2)
                    b.x1 -= abs((x_del / 2.0)) + delta
                    b.x2 -= abs((x_del / 2.0)) + delta
                if b.y1 < 0:
                    delta = abs(b.y1)
                    b.y1 = 0
                    b.y2 += abs((y_del / 2.0)) + delta
                if b.y2 >= 1.0:
                    delta = abs(1.0 - b.y2)
                    b.y1 -= abs(y_del / 2.0) + delta
                    b.y2 -= abs(y_del / 2.0) + delta
            
    def run(self):
        _t = time.time()
        newboxes1 = template_match(self.box_locations, 
                                   self.refimg_rect, 
                                   confidence=self.param)
        t1 = time.time() - _t
        avg_time = t1 / len(self.box_locations) if self.box_locations else t1
        msg1 = """Time elapsed (secs) for template_match call: {0}
  Avg. time per ballot: {1}""".format(t1, avg_time)
        print_write(msg1, self.outfile)
        # Also 'center' the targets 
        for temppath, newboxes in newboxes1.items():
            self.box_locations.setdefault(temppath, []).extend(newboxes)
        # Now, grab a random newly-detected voting target, and re-run
        # template matching. 
        new_boxes_flat = reduce(lambda x,y: x+y, newboxes1.values(), [])
        num_new_boxes = len(new_boxes_flat)
        if num_new_boxes:            
            def get_random(new_boxes):
                """
                'randomly' get a box from a template image.
                """
                candidates = [x for x in new_boxes if len(new_boxes[x])]
                temppath = random.choice(candidates)
                box = random.choice(new_boxes[temppath])
                return box, temppath

            for i in range(SpecifyTargetsPanel.NUM_ITERS):
                box, temppath = get_random(newboxes1)
                img = util_gui.open_as_grayscale(temppath)
                refimg_pil = crop_out_box(img, box)
                refimg = np.array(refimg_pil, dtype='f')
                _t = time.time()
                new_boxes2 = template_match(self.box_locations, refimg, False, self.param)
                t2 = time.time() - _t
                msg = """\tTime elapsed (secs) for {0}-th template_match call: {1}
\t  Avg. Time per ballot: {2}""".format(i+1, t2, t2 / len(self.box_locations))
                print_write(msg, self.outfile)
                for temppath, boxes in new_boxes2.items():
                    self.box_locations.setdefault(temppath, []).extend(boxes)
                ct = sum([len(lst) for lst in new_boxes2.values()])
                foo = "Found {0} more additional targets".format(ct)
                print_write(foo, self.outfile)
        # wx.CallAfter is used to avoid crashing Linux - if you
        # directly call Publisher().sendMessage, X11 crashes.
        self.standardize_boxsizes()
        wx.CallAfter(Publisher().sendMessage, "signals.world.set_boxes", self.box_locations)
        wx.CallAfter(Publisher().sendMessage, "signals.ProgressGauge.done")
        wx.CallAfter(Publisher().sendMessage, "broadcast.tempmatchdone")
        self.outfile.close()

class FrontBackPanel(wx.Panel):
    """
    Class to allow user to select if this image is a front or back.
    """
    def __init__(self, parent, *args, **kwargs):
        wx.Panel.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.SetSizer(sizer)
        txt = wx.StaticText(self, label="Which side is this ballot image?")
        radio_sizer = wx.BoxSizer(wx.HORIZONTAL)
        radiobtn_front = wx.RadioButton(self, label="Front side", style=wx.RB_GROUP)
        radiobtn_front.Bind(wx.EVT_RADIOBUTTON, self.onRadioButton)
        radiobtn_back = wx.RadioButton(self, label="Back side")
        radiobtn_back.Bind(wx.EVT_RADIOBUTTON, self.onRadioButton)
        self.radiobtn_front = radiobtn_front
        self.radiobtn_back = radiobtn_back
        radio_sizer.Add(radiobtn_front)
        radio_sizer.Add(radiobtn_back)
        sizer.Add(txt)
        sizer.Add((10, 10))
        sizer.Add(radio_sizer)
        btn_options = wx.Button(self, label="Options...")
        btn_options.Bind(wx.EVT_BUTTON, self.onButton_options)
        sizer.Add((10, 10))
        sizer.Add(wx.StaticText(self, label="-Or-"))
        sizer.Add((10, 10))
        sizer.Add(btn_options)
        
    def onRadioButton(self, evt):
        front_val = self.radiobtn_front.GetValue()
        back_val = self.radiobtn_back.GetValue()

    def onButton_options(self, evt):
        dlg = FrontBackOptsDlg(self)
        retval = dlg.ShowModal()
        if retval == wx.ID_CANCEL:
            return
        if dlg.is_alternating == True:
            # Update parent.frontback_map to alternate by sorted imgpath
            for i, imgpath in enumerate(sorted(self.parent.frontback_map)):
                # TODO: Support more-than 2 sides
                newside = 'front' if i % 2 == 0 else 'back'
                self.parent.frontback_map[imgpath] = newside

    def set_side(self, side):
        """
        Sets the RadioButton to 'side' appropriately. Expects 'side' to
        either be 'front' or 'back'.
        """
        if side == 'front':
            self.radiobtn_front.SetValue(True)
            self.radiobtn_back.SetValue(False)
        else:
            self.radiobtn_back.SetValue(True)
            self.radiobtn_front.SetValue(False)

    def get_side(self):
        return 'front' if self.radiobtn_front.GetValue() else 'back'

class FrontBackOptsDlg(wx.Dialog):
    """
    A dialog that is displayed when the user clicks the 'Options...'
    button in the 'Front/Back' panel.

    EndModal(int status):
        wx.ID_OK  - user clicked 'Ok'
        wx.ID_CANCEl - user clicked 'Cancel'
    """
    def __init__(self, parent, *args, **kwargs):
        wx.Dialog.__init__(self, parent, title="Front/Back Options", *args, **kwargs)

        # 'Output' variables
        self.is_alternating = False

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        txt1 = wx.StaticText(self, label="Blank ballots alternate \
front/back.")
        self.is_alt_chkbox = wx.CheckBox(self)
        opt1_sizer = wx.BoxSizer(wx.HORIZONTAL)
        opt1_sizer.AddMany([(txt1,), (self.is_alt_chkbox,)])
        btn_ok = wx.Button(self, label="Ok")
        btn_ok.Bind(wx.EVT_BUTTON, self.onButton_ok)
        btn_cancel = wx.Button(self, label="Cancel")
        btn_cancel.Bind(wx.EVT_BUTTON, self.onButton_cancel)
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        btn_sizer.AddMany([(btn_ok,), (btn_cancel,)])
        self.sizer.AddMany([(opt1_sizer,), (btn_sizer,)])
        self.SetSizer(self.sizer)

    def onButton_ok(self, evt):
        if self.is_alt_chkbox.IsChecked() == True:
            self.is_alternating = True
        self.EndModal(wx.ID_OK)
    def onButton_cancel(self, evt):
        self.EndModal(wx.ID_CANCEL)

class WarnNoBoxesDialog(wx.Dialog):
    """
    A warning dialog that might be displayed when the user tries
    to leave the 'Specify and Group Targets' page, but the output
    validation fails.
    """
    GOBACK = 42
    PROCEED = 9001

    def __init__(self, parent, msg, *args, **kwargs):
        wx.Dialog.__init__(self, parent, *args, **kwargs)
        txt = wx.StaticText(self, label=msg)
        btn_goback = wx.Button(self, label="I want to go back and double-check")
        btn_goback.Bind(wx.EVT_BUTTON, self.onButton_goback)
        btn_proceed = wx.Button(self, label="This is correct, proceed onwards.")
        btn_proceed.Bind(wx.EVT_BUTTON, self.onButton_proceed)

        sizer_btns = wx.BoxSizer(wx.HORIZONTAL)
        sizer_btns.Add(btn_goback, flag=wx.ALIGN_CENTER)
        sizer_btns.Add((30, 10))
        sizer_btns.Add(btn_proceed, flag=wx.ALIGN_CENTER)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(txt, border=10, flag=wx.ALL)
        sizer.Add((20, 20))
        sizer.Add(sizer_btns, border=10, flag=wx.ALL | wx.ALIGN_CENTER)
        self.SetSizer(sizer)
        self.Fit()
        
    def onButton_goback(self, evt):
        self.EndModal(WarnNoBoxesDialog.GOBACK)
    def onButton_proceed(self, evt):
        self.EndModal(WarnNoBoxesDialog.PROCEED)

def process_groups(groups):
    """ Given the output of grouptargets.do_group_hist, apply a few
    checks:
        1.) Group only by rows/cols. Don't group targets into a grid.
            For now, if a grid is detected, we'll split the grid into
            a separate group for each row.
    Input:
        lst groups: Of the form (group_i, ...), where each group_i is
        of the from ((x1,y1,x2,y2), ...).
    Output:
        A list of groups in the same format as the input groups, but
        with possibly a few groups removed/added in. There will be the
        same number of targets present in both input and output.
    """
    def is_gridlike(group):
        """ Returns True if the targets in group form some NxM grid.
        (Or, at least, has at least 1 row and at least 1 col)."""
        row_targets = []   # represents each row as a target
        for target in group:
            for row_t in row_targets:
                if is_same_row(target, row_t):
                    return True
            row_targets.append(target)
        return False
    def is_same_row(t1, t2):
        """ Returns True if t1 is roughly in the same row as t2. """
        w = abs(t1[0] - t1[2])
        h = abs(t1[1] - t1[3])
        if abs(t1[1] - t2[1]) <= (h / 2.0):
            return True
        return False
    def is_row_any(target, group):
        """ Returns True if target is in the same row as some target
        in group, False otherwise. """
        for t in group:
            if is_same_row(target, t):
                return True
        return False
    def row_idx(target, groups):
        """ If the target is in the same row as some group G in groups,
        then return the index of G. Else, return None. """
        for i, group in enumerate(groups):
            if is_row_any(target, group):
                return i
        return None
    def fuzzy_eq(n1, n2):
        return abs(n1 - n2) <= 0.001
    result = []
    for group in groups:
        if not is_gridlike(group):
            result.append(group)
        else:
            # list of groups
            new_groups = []
            for target in group:
                idx = row_idx(target, new_groups)
                if idx == None:
                    new_groups.append([target])
                else:
                    new_groups[idx].append(target)
            result.extend(new_groups)
    return result

def find_bounding_boxes(groups):
    """
    Given a list of box-groups, returns a single bounding box for each
    group of boxes.
    Input:
        A tuple of tuples of the form (upper, left, lower, right)
    Output:
        A tuple of BoundingBox instances
    """
    bounding_boxes = []
    for group in groups:
        min_x1 = min([x1 for (x1,y1,x2,y2) in group])
        min_y1 = min([y1 for (x1,y1,x2,y2) in group])
        max_x2 = max([x2 for (x1,y1,x2,y2) in group])
        max_y2 = max([y2 for (x1,y1,x2,y2) in group])
        # Now add 'padding'
        min_x1 = max(0, min_x1 - 0.05)
        min_y1 = max(0, min_y1 - 0.025)
        max_x2 = min(1.0, max_x2 + 0.05)
        max_y2 = min(1.0, max_y2 + 0.025)
        bounding_boxes.append(BoundingBox(min_x1, min_y1, max_x2, max_y2,
                                          is_contest=True))
    return bounding_boxes
        
def compute_contest_ids(bounding_boxes):
    """
    Given a list of BoundingBox instances (both contests and voting
    targets), update the contest_ids of all voting targets by
    associating each voting target to the contest box that surrounds
    it.
    A useful function to call whenever contest bounding boxes have
    been changed.
    Output:
        A list of BoundingBoxes, same number as input, but with
        updated contest_id fields.
    """
    new_boxes = []
    targets = [b for b in bounding_boxes if not b.is_contest]
    contest_boxes = [b for b in bounding_boxes if b.is_contest]
    cur_id = 0
    # sort by x1 to process column-by-column
    for contest in sorted(contest_boxes, key=lambda box: box.x1):
        new_contest = contest.copy()
        new_contest.contest_id = cur_id
        cur_id += 1
        new_boxes.append(new_contest)
    for target in [box for box in bounding_boxes if not box.is_contest]:
        new_target = target.copy()
        assoc_contest = util_gui.find_assoc_contest(target, contest_boxes)
        if assoc_contest:
            new_target.contest_id = util_gui.find_assoc_contest(target, contest_boxes).contest_id
        else:
            # If no contest bounds this target, then assign it a dummy contest_id value.
            new_target.contest_id = -1
        new_boxes.append(new_target)
    return new_boxes

def sanity_check_box(img):
    """
    Given the image region that the user selected (that supposedly
    contains a voting target), try a series of heuristics to try to
    detect if the user actually highlighted the entire voting target.
    1.) To detect if part of the box is 'chopped off', we could check
        to see if a vertical and horizontal line passing through the
        center of the region encounters two distinct lines. (This
        assumes that the input to this function is reasonably centered)
    2.) To detect if too much was highlighted (say, some text was 
        inadvertantly included).
    Input:
        obj img: A numpy array
    Output:
        True if img passes the sanity checks, False otherwise.
    """
    h, w = img.shape
    img_threshold = util_gui.autothreshold_numpy(img, method='otsu', slop=15)
    #_img_threshold2 = util_gui.autothreshold_numpy(img, method='otsu', slop=15)
    #scipy.misc.imsave('no_threshold_np.png', img)
    #scipy.misc.imsave('threshold_np_{0}.png'.format('kmeans'), img_threshold)
    #scipy.misc.imsave('threshold_np_otsu.png', _img_threshold2)
    num_x1 = num_collisions(img_threshold, 'x', (h/2)-2)
    num_x2 = num_collisions(img_threshold, 'x', h/2)
    num_x3 = num_collisions(img_threshold, 'x', (h/2)+2)
    num_y1 = num_collisions(img_threshold, 'y', (w/2)-2)
    num_y2 = num_collisions(img_threshold, 'y', w/2)
    num_y3 = num_collisions(img_threshold, 'y', (w/2)+2)
    ## 1,2) Detect chopped-off, too much
    if (num_x1 != 2 and num_x2 != 2 and num_x3 != 2):
        print "Failed sanity check: on X axis, didn't hit 2 times. Hit {0} times instead.".format(max(num_x1, num_x2, num_x3))
        return False
    elif (num_y1 != 2 and num_y2 != 2 and num_y3 != 2):
        print "Failed sanity check: on Y axis, didn't hit 2 times. Hit {0} times instead.".format(max(num_y1, num_y2, num_y3))
        return False
    return True
    
def num_collisions(img, axis, a):
    """
    Return number of times a line (on axis) drawn at row/col 'a' 
    intersects with a line/object.
    Assumes that img has been thresholded.
    Input:
        obj img: a numpy array
        str axis: 'x' for x axis (horiz line), 'y' for y axis (vert line)
        int a: which row/col to search on (0 indexed).
    """
    BLACK = 0   # I always forget 0 is black, 255 is white
    WHITE = 255
    if axis.lower() == 'y':
        img = img.T
    try:
        row_or_col = img[a]
    except IndexError as e:
        print e
        pdb.set_trace()
    num_collisions = 0
    flag_in_line = False
    for val in row_or_col:
        if not flag_in_line and val == BLACK:
            flag_in_line = True
            num_collisions += 1
        elif flag_in_line and val == WHITE:
            flag_in_line = False
    return num_collisions
           
def normalize_boxes(boxes, imgsize):
    """
    Given a list of BoundingBoxes, normalize each bounding box so that
    each box doesn't extend outside of the image. 
    Input:
        list boxes: A list of BoundingBoxes instances
        tuple imgsize: (int width, int height)
    Output:
        A list of BoundingBox instances with 'normalized' sizes.
    """
    def normalize(box, imgsize):
        box_copy = box.copy()
        box_copy.x1 = max(0, box.x1)
        box_copy.y1 = max(0, box.y1)
        box_copy.x2 = min(1.0 - (1.0 / imgsize[0]), box.x2)
        box_copy.y2 = min(1.0 - (1.0 / imgsize[1]), box.y2)
        return box_copy
    return [normalize(box, imgsize) for box in boxes]

def intround(num):
    return int(round(num))

def crop_out_box(img, box):
    """
    Given a PIL image and a BoundingBox, crops out a portion of the
    Pil image and returns it (as specified by the BoundingBox).
    Input:
        obj img: a PIL image
        obj box: a BoundingBox instance
    Output:
        A PIL image.
    """
    w_img, h_img = img.size
    ul_x = intround(box.x1*w_img)
    ul_y = intround(box.y1*h_img)
    lr_x = intround(box.x2*w_img)
    lr_y = intround(box.y2*h_img)
    return img.crop((ul_x, ul_y, lr_x, lr_y))
    
def template_match(boxes, ref_img, add_padding=False, confidence=0.8):
    """
    Input:
        dict boxes: {str templatepath: list BoundingBoxes}
        obj ref_img: A numpy array
    Output:
        Return a dict mapping templatepath to new matched 
        BoundingBoxes.
    """
    def is_overlap(rect1, rect2):
        """
        Returns True if any part of rect1 is contained within rect2.
        Input:
            rect1: Tuple of (x1,y1,x2,y2)
            rect2: Tuple of (x1,y1,x2,y2)
        """
        def is_within_box(pt, box):
            return box[0] < pt[0] < box[2] and box[1] < pt[1] < box[3]
        x1, y1, x2, y2 = rect1
        w, h = abs(x2-x1), abs(y2-y1)
        # Checks (in order): UL, UR, LR, LL corners
        return (is_within_box((x1,y1), rect2) or
                is_within_box((x1+w,y1), rect2) or 
                is_within_box((x1+w,y1+h), rect2) or 
                is_within_box((x1,y1+h), rect2))
    def too_close(b1, b2):
        """
        Input:
            b1: Tuple of (x1,y1,x2,y2)
            b2: Tuple of (x1,y1,x2,y2)
        """
        dist = util_gui.dist_euclidean
        w, h = abs(b1[0]-b1[2]), abs(b1[1]-b1[3])
        return ((abs(b1[0] - b2[0]) <= w / 2.0 and
                 abs(b1[1] - b2[1]) <= h / 2.0) or
                is_overlap(b1, b2) or 
                is_overlap(b2, b1))
    def fix_box(box, imgsize):
        """
        If the box extends outside the image in any direction,
        extend it in the opposite direction to enforce that each
        box is the same size (and chop it off at the image
        boundary).
        """
        flag = False
        w_img, h_img = imgsize
        ul_x, lr_x = map(lambda x: int(round(x*w_img)), (box.x1, box.x2))
        ul_y, lr_y = map(lambda y: int(round(y*h_img)), (box.y1, box.y2))
        if ul_x < 0:
            ul_x = 0
            lr_x += abs(ul_x)
        if lr_x >= w_img:
            delta = abs(lr_x - w_img + 1.0)
            lr_x -= delta
            ul_x -= delta
        if ul_y < 0:
            ul_y = 0
            lr_y += abs(ul_y)
        if lr_y >= h_img:
            delta = abs(lr_y - h_img + 1.0)
            lr_y -= delta
            ul_y -= delta
        box.x1 = ul_x / float(w_img)
        box.x2 = lr_x / float(w_img)
        box.y1 = ul_y / float(h_img)
        box.y2 = lr_y / float(h_img)
        return box

    count = 0
    new_boxes = {}  # {str templatepath: list of BoundingBoxes}
    ref_imgs = [ref_img]
    NUM_TEMPS = len(boxes)
    numComplete = 0  # number of jobs finished so far
    N = multiprocessing.cpu_count()
    manager = multiprocessing.Manager()
    queue = manager.Queue()
    pool = multiprocessing.Pool()
    
    def divy_boxes(boxes, N):
        """ Separates boxes into N jobs. Separate by templatepath.
        Input:
            dict boxes: {str templatepath: list BoundingBoxes}
            int N: Number of processors
        Output:
            A list of N dicts of the form {str templatepath: list Boxes}
        """
        result = []
        if len(boxes) < N:
            # Degerate case, less jobs than procs
            for temppath, tempboxes in boxes.iteritems():
                d = {}
                d[temppath] = tempboxes
                result.append(d)
            return result
        k = int(math.floor(float(len(boxes) / float(N))))

        cur = {}
        for i, temppath in enumerate(boxes):
            cur[temppath] = boxes[temppath]
            if i % k == 0:
                result.append(cur)
                cur = {}
        if cur:
            result.append(cur)
        return result

    for cur_ref_img in ref_imgs:
        _count = 0
        jobs = divy_boxes(boxes, N)

        for i, job_boxes in enumerate(jobs):
            print "Process {0} got {1} jobs".format(i, len(job_boxes))
            pool.apply_async(tempmatch_process, args=(job_boxes, cur_ref_img, queue, confidence))
        while numComplete < NUM_TEMPS:
            match_coords, (h_img, w_img), bounding_boxes, templateimgpath = queue.get()
            wx.CallAfter(Publisher().sendMessage, "signals.ProgressGauge.tick")
            new_bounding_boxes = []
            for (x,y) in match_coords:
                x_rel, y_rel = x / float(w_img), y / float(h_img)
                flag = True
                for box in [b for b in bounding_boxes if not b.is_contest]+new_bounding_boxes:
                    if too_close((x_rel,y_rel,x_rel+box.width,y_rel+box.height),
                                 box.get_coords()):
                        flag = False
                        break
                if flag:
                    # Let's also add some padding
                    if add_padding:
                        # This is the first bounding box, so add some
                        # padding to the auto-fitted thing
                        PAD_X, PAD_Y = 2.0 / w_img, 2.0 / h_img
                    else:
                        # Don't add padding, it was added earlier
                        PAD_X, PAD_Y = 0.0, 0.0
                    w_fitted = cur_ref_img.shape[1] / float(w_img)
                    h_fitted = cur_ref_img.shape[0] / float(h_img)
                    box_new = BoundingBox(x_rel-PAD_X, y_rel-PAD_Y, 
                                          x_rel+abs(w_fitted)+PAD_X,
                                          y_rel+abs(h_fitted)+PAD_Y,
                                          color="Orange")
                    box_new = fix_box(box_new, (w_img,h_img))
                    new_bounding_boxes.append(box_new)
            new_bounding_boxes = normalize_boxes(new_bounding_boxes,
                                                (w_img, h_img))
            count += len(new_bounding_boxes)
            _count += len(new_bounding_boxes)
            new_boxes.setdefault(templateimgpath, []).extend(new_bounding_boxes)
            numComplete += 1
        print 'Number of new voting targets detected:', _count

    pool.close()
    return new_boxes

def tempmatch_process(boxes, cur_ref_img, queue, confidence=0.8):
    for (templateimgpath, bounding_boxes) in boxes.items():
        img_array = util_gui.open_img_scipy(templateimgpath)
        match_coords = util_gui.template_match(img_array,
                                               cur_ref_img, 
                                               confidence=confidence)
        queue.put((match_coords, img_array.shape, bounding_boxes, templateimgpath))

def make_transfn(proj):
    w_img, h_img = proj.imgsize
    def fn(x1,y1,x2,y2):
        x1 = int(round(x1 * w_img))
        y1 = int(round(y1 * h_img))
        x2 = int(round(x2 * w_img))
        y2 = int(round(y2 * h_img))
        return x1, y1, x2, y2
    return fn

def convert_boxes2mosaic(project, box_locations):
    """ Given the box_locations in [0,1] coordinates, return a new dict
    with BoundingBoxes in img coordinates.
    Input:
        obj project;
        dict box_locations: maps {str temppath: [BoundingBox_i, ...]}
    Output:
        A dictionary mapping {str temppath: [BoundingBox_i, ...]} but 
        in image coordinates.
    """
    result = {}
    w_img, h_img = project.imgsize
    # Scale the coords to image coords.
    for temppath, boxes in box_locations.iteritems():
        lst = []
        for box in boxes:
            b_cpy = box.copy()
            b_cpy.x1 = int(round(box.x1 * w_img))
            b_cpy.y1 = int(round(box.y1 * h_img))
            b_cpy.x2 = int(round(box.x2 * w_img))
            b_cpy.y2 = int(round(box.y2 * h_img))
            lst.append(b_cpy)
        result[temppath] = lst
    return result
