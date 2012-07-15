import sys, csv, copy, os, time, pickle
sys.path.append('../')

import wx, wx.animate, Image, cv2, wx.lib.inspection
import numpy as np
from wx.lib.pubsub import Publisher
from specify_voting_targets import find_targets_wizard as find_targets_wizard
from specify_voting_targets import imageviewer as imageviewer
from specify_voting_targets.imageviewer import BallotScreen as BallotScreen
from specify_voting_targets.imageviewer import BallotViewer as BallotViewer
from specify_voting_targets.imageviewer import WorldState as WorldState
from specify_voting_targets.imageviewer import BoundingBox as BoundingBox
from specify_voting_targets.find_targets_wizard import MosaicPanel as MosaicPanel
from specify_voting_targets import util_gui as util_gui
from define_attributes import IBallotScreen as IBallotScreen
from define_attributes import IToolBar as IToolBar

# Get this script's directory. Necessary to know this information
# since the current working directory may not be the same as where
# this script lives (which is important for loading resources like
# imgs)
try:
    MYDIR = os.path.abspath(os.path.dirname(__file__))
except NameError:
    # This script is being run directly
    MYDIR = os.path.abspath(sys.path[0])    
        
class SpecifyPatchesPanel(wx.Panel):
    """
    Panel that contains the Mosaic Panel and IBallotViewer Panel. 
    Allows a user to specify bounding boxes around all identifying patches.
    """
    
    def __init__(self, parent, *args, **kwargs):
        wx.Panel.__init__(self, parent, *args, **kwargs)
        
        ## Instance vars
        self.project = None
        self.parent = parent
        self.world = None
        self.limit = 1
        self.templatesdir = None
        self.samplesdir = None

        #self.first_time = True      # If True, then the user has not created a bounding box yet
        self.has_started = False    # If True, then self.start() has already been called.
        
        self.csvdir = None
        
        self.setup_widgets()

        # Pubsubs
        self.callbacks = [("broadcast.mosaicpanel.mosaic_img_selected", self.pubsub_mosaic_img_selected),
                          ("broadcast.IBallotScreen.targets_update", self.pubsub_update_targets),
                          ("broadcast.IBallotScreen.added_target", self._pubsub_added_target),
                          ("signals.SelectPatches.autoFill", self._pubsub_auto_fill)]
        self.subscribe_pubsubs()

        # Pubsub Subscribing Required
        Publisher().subscribe(self._pubsub_project, "broadcast.project")
       
    def subscribe_pubsubs(self):
        for (topic, callback) in self.callbacks:
            Publisher().subscribe(callback, topic)
        self.panel_mosaic.subscribe_pubsubs()
        self.ballotviewer.subscribe_pubsubs()
        self.world.subscribe_pubsubs()
        
    def unsubscribe_pubsubs(self):
        """
        Selectively unsubscribe pubsub listeners that shouldn't be
        listening when this widget isn't active.
        """
        for (topic, callback) in self.callbacks:
            Publisher().unsubscribe(callback, topic)
        self.panel_mosaic.unsubscribe_pubsubs()
        self.ballotviewer.unsubscribe_pubsubs()
        self.world.unsubscribe_pubsubs()
 
    def setup_widgets(self):
        self.sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.world = WorldState()
        self.panel_mosaic = MosaicPanel(self, self.world, style=wx.SIMPLE_BORDER)
        self.panel_mosaic.Hide()
        self.ballotviewer = BallotViewer(self, self.world,
                                         ballotscreen=IBallotScreen,
                                         toolbar=IToolBar, style=wx.SIMPLE_BORDER)
        self.ballotviewer.Hide()
        self.sizer.Add(self.panel_mosaic, border=10, proportion=0, flag=wx.ALL | wx.ALIGN_LEFT)
        self.sizer.Add(self.ballotviewer, border=10, proportion=0, flag=wx.EXPAND | wx.ALL | wx.ALIGN_LEFT)

        self.SetSizer(self.sizer)
        
    def reset(self):
        #self.box_locations = {}
        #self.first_time = True
        self.has_started = False
        # These aren't working - it isn't removing the previous
        # mosaic panels/ballot screens. will tend to later.
        #self.panel_mosaic.reset()
        #self.ballotviewer.ballotscreen.set_image(None)
        # TOOD: RESET ballotscreen and mosaic
        
    def stop(self):
        pickle.dump(self.attributes, open(self.project.ballot_attributesfile, 'wb'))
        self.unsubscribe_pubsubs()

    def start(self, templatesdir):
        if False:#self.has_started: # Never reset (for now)
            self.reset()
        self.has_started = True
        self.subscribe_pubsubs()
        self.world.reset()
        self.import_bounding_boxes(templatesdir)

        self.panel_mosaic.display_images(self.world.get_boxes_all())
        
        # Display first template on BallotScreen
        imgpath = sorted(self.panel_mosaic.imgs.keys())[0]
        img = util_gui.open_as_grayscale(imgpath)
        target_locations = self.world.get_boxes(imgpath)
        img_panel = self.panel_mosaic.img_panels[imgpath]
        img_panel.static_bitmap.select()
        
        Publisher().sendMessage("signals.ballotviewer.set_image_pil", (imgpath, img))
        Publisher().sendMessage("signals.BallotScreen.set_bounding_boxes", (imgpath, target_locations))        
        Publisher().sendMessage("signals.BallotScreen.update_state", IBallotScreen.STATE_IDLE)
        Publisher().sendMessage("signals.IBallotScreen.makeAutobox")

        self.panel_mosaic.Show()
        self.ballotviewer.Show()
        
        self.Refresh()
        self.parent.Fit()
        self.parent.Refresh()

    def remove_targets(self, templatepath):
        """
        Remove all voting targets from self.box_locations that are for
        templatepath. Keeps all contest bounding boxes though.
        """
        self.world.remove_voting_targets(templatepath)
          
    def export_bounding_boxes(self):
        #Export box locations to csv files. Also, returns the BoundingBox
        #instances for each template image as a dict of the form:
        #    {str templatepath: list boxes}
        #Because the user might have manually changed the bounding
        #boxes of each contest (by modifying the size of a contest
        #box), this function will re-compute the correct contest_ids
        #for all voting target BoundingBoxes.
        now = time.time()

        if not self.world:
            return
        elif len(self.world.get_boxes_all().items()) == 0:
            # No templates loaded
            return
        elif len(self.world.get_boxes_all_list()) == 0:
            # No bounding boxes created
            return

        util_gui.create_dirs(self.csvdir)
        fields = ('imgpath', 'id', 'x', 'y', 'width', 'height', 'label', 'is_contest', 'contest_id')
        for imgpath in self.world.get_boxes_all():
            print 'imgpath:', imgpath
            print self.world.get_boxes_all()
            csvfilepath = os.path.join(self.csvdir, "{0}_patchlocs.csv".format(os.path.splitext(os.path.split(imgpath)[1])[0]))
            csvfile = open(csvfilepath, 'wb')
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
                img = util_gui.open_as_grayscale(imgpath)
                w_img, h_img = img.size
                row['x'] = int(round(x1 * w_img))
                row['y'] = int(round(y1 * h_img))
                width = int(round(abs(x1-x2)*w_img))
                height = int(round(abs(y1-y2)*h_img))
                row['width'] = width
                row['height'] = height
                # Replace commas with underscore to avoid problems with csv files
                row['label'] = bounding_box.label.replace(",", "_")
                row['is_contest'] = 1 if bounding_box.is_contest else 0
                row['contest_id'] = bounding_box.contest_id
                dictwriter.writerow(row)
            csvfile.close()
        val = copy.deepcopy(self.world.get_boxes_all())
        
        return val
    
    #### Pubsub Callbacks
        
    def pubsub_export_targets(self, msg):    
        """ Export box locations to csv files """
        self.export_bounding_boxes()
            
    def pubsub_mosaic_img_selected(self, msg):
        imgpath = msg.data
        img = util_gui.open_as_grayscale(imgpath)
        target_locations = self.world.get_boxes(imgpath)
        Publisher().sendMessage("signals.ballotviewer.set_image_pil", (imgpath, img))
        Publisher().sendMessage("signals.IBallotScreen.update_state", IBallotScreen.STATE_IDLE)
        
    def checkCanMoveOn(self):
        canMoveOn = True
        for boxes in self.world.box_locations.values():
            if len(boxes) != 1:
                canMoveOn = False
                
        if len(self.world.box_locations) == 1:
            return True
        
        return canMoveOn
        
    def pubsub_update_targets(self, msg):
        """
        Triggered when the user modifies the voting targets in the
        BallotScreen (during Mosaic verify).
        """
        imgpath, targets = msg.data
        self.panel_mosaic.Refresh()
        self.Refresh()
        
    def _pubsub_added_target(self, msg):
        self.panel_mosaic.Refresh()
        self.Refresh()

    def _pubsub_project(self, msg):
        """
        Triggered when the user selects a Project. Pull in relevant
        state. Also, 'mutate' the current WorldState to reflect the
        project change.
        """
        project = msg.data
        self.project = project
        self.csvdir = project.patch_loc_dir
        try:
            self.attributes = pickle.load(open(project.ballot_attributesfile, 'rb'))
            print "Successfully loaded ballot_attributes file."
        except IOError as e:
            # This project is 'old' and doesn't have it. Create it to 
            # 'update' this project.
            print "Couldn't find ballot_attributes file. Upgrading \
project."
            self.attributes = {}
            project.ballot_attributesfile = os.path.join(project.projdir_path, 'ballot_attributes.p')
            attrs_file = open(project.ballot_attributesfile, 'wb')
            pdb.set_trace()
            pickle.dump(self.attributes, attrs_file)
            attrs_file.close()
        self.ballotviewer.ballotscreen.attributes = self.attributes
        self.project.addCloseEvent(self.export_bounding_boxes)
        
    def _pubsub_auto_fill(self, msg):        
        if self.world.get_boxes_count_all() == 1:
            for path, boxes in self.world.get_boxes_all().items():
                if (len(boxes) != 0):
                    box = boxes[0]
                    imgpath = path
            
            for path, boxes in self.world.get_boxes_all().items():
                if len(boxes) < self.limit and path != imgpath:
                    self.world.add_box(path, box.copy())
                    
            Publisher().sendMessage("signals.IBallotScreen.updateMode")
            self.panel_mosaic.Refresh()
            self.ballotviewer.Refresh()
            self.Refresh()
        
    def import_bounding_boxes(self, templatesdir):
        self.templatesdir = templatesdir
        
        if not self.templatesdir:
            # user hasn't gotten to this part yet
            # Reset the WorldState
            self.world.reset()
            return
        
        # annoying: get img size
        for dirpath, dirnames, filenames in os.walk(self.templatesdir):
            imgpath = [os.path.join(dirpath, f) for f in filenames if util_gui.is_image_ext(f)][0]
            break
        img = util_gui.open_as_grayscale(imgpath)
        worldstate = find_targets_wizard.import_worldstate(self.csvdir, img.size)
        
        self.world.mutate(worldstate)

        for dirpath, dirnames, filenames in os.walk(self.templatesdir):
            for imgname in [x for x in filenames if util_gui.is_image_ext(x)]:
                imgpath = os.path.join(dirpath, imgname)
                if (imgpath not in self.world.box_locations):
                    self.world.box_locations[imgpath] = []

        if (self.world.get_boxes_count_all() > 0):
            self.first_time = False
            
        #Publisher().sendMessage("signals.IBallotScreen.updateMode")
