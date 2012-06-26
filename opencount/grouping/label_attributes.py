import sys, csv, copy, os, time, pickle, pdb
from os.path import join as pathjoin
sys.path.append('../')

import wx, wx.animate, Image, cv2, wx.lib.inspection
import numpy as np
import util
import common
from wx.lib.pubsub import Publisher
from specify_voting_targets import find_targets_wizard as find_targets_wizard
from specify_voting_targets import imageviewer as imageviewer
from specify_voting_targets.imageviewer import BallotScreen as BallotScreen
from specify_voting_targets.imageviewer import BallotViewer as BallotViewer
from specify_voting_targets.imageviewer import WorldState as WorldState
from specify_voting_targets.imageviewer import BoundingBox as BoundingBox
from specify_voting_targets.find_targets_wizard import MosaicPanel as MosaicPanel
from specify_voting_targets import util_gui as util_gui
from common import AttributeBox, IWorldState, TextInputDialog

"""
Outputs of this module:

project.patch_loc_dir
    This is a directory that contains .csv files. Each csv file
    is for a blank Ballot, which has information about each
    ballot attribute (location, value, and side).

May Modify:

project.ballot_attributesfile
    If the user deletes an Attribute while in 'Label Ballot
    Attributes', then this change must be reflected in 'Define
    Ballot Attributes' as well.
"""

# Get this script's directory. Necessary to know this information
# since the current working directory may not be the same as where
# this script lives (which is important for loading resources like
# imgs)
try:
    MYDIR = os.path.abspath(os.path.dirname(__file__))
except NameError:
    # This script is being run directly
    MYDIR = os.path.abspath(sys.path[0])    

# For .csv files in <projdir>/precinct_locations/*, the first row
# is always a dummy row with an ID given by DUMMY_ROW_ID
DUMMY_ROW_ID = -42

class LabelScreen(BallotScreen):
    """
    Like BallotScreen, but where you just label previously-
    created bounding boxes.
    """
    def __init__(self, parent, world, *args, **kwargs):
        BallotScreen.__init__(self, parent, world, 
                              can_resize=False,
                              can_delete=False,
                              #can_modify=False,
                              *args, **kwargs)

    def onRightDown(self, evt):
        x, y = self.CalcUnscrolledPosition(evt.GetPositionTuple())
        box, mode = self.get_closest_box_any(self.world.get_boxes(self.current_imgpath), (x,y))
        if box:
            attr_types, attr_vals = zip(*box.attrs.items())
            dlg = TextInputDialog(self, caption="Enter the value for \
each Attribute Type:",
                                  labels=attr_types,
                                  vals=attr_vals)
            self.Disable()
            val = dlg.ShowModal()
            self.Enable()
            if val == wx.ID_OK:
                box.attrs = dlg.results

    def onPaint(self, event):
        """ 
        Refresh screen. 
        Note: Regrettably, I couldn't simply call the inherited 
              onPaint() method of the parent, since visual things got
              all wonky.
        """
        if self.IsDoubleBuffered():
             dc = wx.PaintDC(self)
        else:
            dc = wx.BufferedPaintDC(self)
        # You must do PrepareDC in order to force the dc to account
        # for scrolling.
        self.PrepareDC(dc)
        
        dc.DrawBitmap(self.img_bitmap, 0, 0)
        self._display_targets(dc)
        if self.is_resize_target():
            self._draw_resize_rect(dc)
        event.Skip()

    def _display_targets(self, dc):
        BallotScreen._display_targets(self, dc)
        if self.world:
            dc.SetTextForeground("Blue")
            w_img, h_img = self.img_bitmap.GetWidth(), self.img_bitmap.GetHeight()
            for box in self.get_boxes():
                s = ''
                for attrtype in box.get_attrtypes():
                    val = box.get_attrval(attrtype)
                    s += "'{0}': {1}\n".format(attrtype, val if val else '_none_')
                x1, y1 = int(round(box.x1 * w_img)), int(round(box.y1 * h_img))
                w_txt, h_txt = dc.GetTextExtent(s)
                
                x_txt, y_txt = x1, y1 - h_txt
                if y_txt < 0:
                    y_txt = y1 + int(round(box.height * h_img))
                dc.DrawText(s, x_txt, y_txt)
        
class LabelAttributesPanel(wx.Panel):
    """
    Panel that contains the Mosaic Panel and BallotViewer Panel. 
    Allows a user to specify bounding boxes around all identifying patches.
    """
    
    def __init__(self, parent, *args, **kwargs):
        wx.Panel.__init__(self, parent, *args, **kwargs)
        
        ## Instance vars
        self.project = None
        self.parent = parent
        self.world = None
        self.limit = 1

        #self.first_time = True      # If True, then the user has not created a bounding box yet
        self.has_started = False    # If True, then self.start() has already been called.
        
        self.setup_widgets()

        # Pubsubs
        self.callbacks = [("broadcast.mosaicpanel.mosaic_img_selected", self.pubsub_mosaic_img_selected),
                          ("broadcast.IBallotScreen.targets_update", self.pubsub_update_targets),
                          ("broadcast.IBallotScreen.added_target", self._pubsub_added_target)]
        self.subscribe_pubsubs()

        # Pubsub Subscribing Required
        Publisher().subscribe(self._pubsub_project, "broadcast.project")
       
    def setup_widgets(self):
        self.sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.world = IWorldState()
        self.nb = wx.Notebook(self)
        self.nb.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.onPageChange)
        self.panel_mosaic = MosaicPanel(self, self.world, style=wx.SIMPLE_BORDER)
        self.ballotviewer = BallotViewer(self.nb, self.world,
                                         ballotscreen=LabelScreen,
                                         toolbar=LightToolBar, style=wx.SIMPLE_BORDER)
        self.ballotviewer_alt = None
        self.nb.AddPage(self.ballotviewer, "Front")
        self.sizer.Add(self.panel_mosaic, border=10, proportion=0, flag=wx.EXPAND | wx.ALL | wx.ALIGN_LEFT)
        self.sizer.Add(self.nb, border=10, proportion=1, flag=wx.EXPAND | wx.ALL | wx.ALIGN_LEFT)
        self.SetSizer(self.sizer)

    def onPageChange(self, evt):
        """
        Triggered when the user switches tabs from, say, 'Front' to
        'Back'. Only for multi-page elections.
        """
        old = evt.GetOldSelection()
        new = evt.GetSelection()
        if old == 0:
            self.ballotviewer.unsubscribe_pubsubs()
        elif old == 1:
            self.ballotviewer_alt.unsubscribe_pubsubs()
        if new == 0:
            self.ballotviewer.subscribe_pubsubs()
        elif new == 1:
            self.ballotviewer_alt.subscribe_pubsubs()

    def subscribe_pubsubs(self):
        for (topic, callback) in self.callbacks:
            Publisher().subscribe(callback, topic)
        self.panel_mosaic.subscribe_pubsubs()
        self.ballotviewer.subscribe_pubsubs()
        if self.ballotviewer_alt:
            self.ballotviewer_alt.subscribe_pubsubs()
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
        if self.ballotviewer_alt:
            self.ballotviewer_alt.unsubscribe_pubsubs()
        self.world.unsubscribe_pubsubs()
        
    def reset(self):
        self.has_started = False
        
    def stop(self):
        #common.dump_iworldstate(self.world, self.project.ballot_attributesfile)
        self.unsubscribe_pubsubs()

    def start(self):
        self.has_started = True
        self.subscribe_pubsubs()
        # Note: import_bounding_boxes may return {}, if there's nothing to
        # import from in project.patch_loc_dir
        box_locations = self.import_bounding_boxes()
        self.world.box_locations = box_locations
        frontback_map = pickle.load(open(self.project.frontback_map, 'rb'))
        defined_attrtypes = common.load_attrboxes(self.project.ballot_attributesfile)
        if not box_locations:
            # Prepopulate each template image with all attr patches
            for dirpath, dirnames, filenames in os.walk(self.project.templatesdir):
                for imgname in [f for f in filenames if util_gui.is_image_ext(f)]:
                    imgpath = os.path.abspath(os.path.join(dirpath, imgname))
                    self.world.box_locations.setdefault(imgpath, [])
                    for box in defined_attrtypes:
                        side = box.side
                        if side == frontback_map[imgpath]:
                            self.world.add_box(imgpath, box.copy())
        # If the user defined new Ballot Attributes, then update my
        # data structures.
        # Assumption: A given attrtype only exists in /one/ 
        #             AttributeBox.
        for attrbox in defined_attrtypes:
            attr_types = tuple(attrbox.get_attrtypes())
            side = attrbox.side
            for temppath in self.world.box_locations.keys():
                temppath = os.path.abspath(temppath)
                if side != frontback_map[temppath]:
                    continue
                boxes = self.world.box_locations[temppath]
                myattr_types = set(sum([b.get_attrtypes() for b in boxes], []))
                missings = []
                for attrtype in attr_types:
                    if attrtype not in myattr_types:
                        missings.append(attrtype)
                if missings:
                    flag = False
                    # Add in missing attribute box definition, either
                    # by modifying an existing box, or creating new
                    def fuzzy_eqs(nums1, nums2, e=1e-3):
                        def fuzzy_eq(a, b, e=1e-3):
                            return abs(a-b) <= e
                        for i, n1 in enumerate(nums1):
                            if not fuzzy_eq(n1, nums2[i]):
                                return False
                        return True
                    for box in self.world.get_boxes(temppath):
                        if fuzzy_eqs(box.get_coords(), attrbox.get_coords()):
                            flag = True
                            for missing in missings:
                                box.add_attrtype(missing)
                    if not flag:
                        # try not to add duplicate attrtypes
                        newbox = AttributeBox(attrbox.x1, attrbox.y1,
                                              attrbox.x2, attrbox.y2)
                        for missing in missings:
                            newbox.add_attrtype(missing)
                        self.world.box_locations.setdefault(temppath, []).append(newbox)
        # Finally, update sizes, in case the user resized boxes in 
        # 'Define Ballot Attributes'
        for attrbox_def in defined_attrtypes:
            w_def, h_def = attrbox_def.width, attrbox_def.height
            boxes = [b for b in self.world.get_boxes_all_list() if b.get_attrtypes() == attrbox_def.get_attrtypes()]
            util_gui.resize_boxes(boxes, (w_def, h_def), mode='lower-right')
        if util.is_multipage(self.project):
            self.ballotviewer_alt = BallotViewer(self.nb, self.world,
                                                 ballotscreen=LabelScreen,
                                                 toolbar=LightToolBar,
                                                 style=wx.SIMPLE_BORDER)
            self.nb.AddPage(self.ballotviewer_alt, 'Back')
            template_to_images = pickle.load(open(self.project.template_to_images, 'rb'))
            # To provide a consistent ordering of ballot images, sort
            # lexigraphically.
            sorted_items = sorted(template_to_images.items(), key=lambda x: x[0])
            ballot_first, (frontpath_first, backpath_first) = sorted_items[0]
            frontimg = util_gui.open_as_grayscale(frontpath_first)
            backimg  = util_gui.open_as_grayscale(backpath_first)
            self.ballotviewer.set_image_pil(frontpath_first, frontimg)
            self.ballotviewer_alt.set_image_pil(backpath_first, backimg)
            # Display ballots on MosaicPanel
            to_display = {}
            for (ballot, (frontpath, backpath)) in sorted_items:
                # For now, always only show the front page. Maybe later
                # we can allow the user to 'flip' the page within the 
                # MosaicPanel.
                to_display[frontpath] = self.world.get_boxes(frontpath)
            self.panel_mosaic.display_images(to_display)
            self.panel_mosaic.img_panels[frontpath_first].static_bitmap.select()
        else:
            # Display first template on BallotScreen
            self.panel_mosaic.display_images(self.world.get_boxes_all())
            imgpath = sorted(self.panel_mosaic.imgs.keys())[0]
            img = util_gui.open_as_grayscale(imgpath)
            target_locations = self.world.get_boxes(imgpath)
            img_panel = self.panel_mosaic.img_panels[imgpath]
            img_panel.static_bitmap.select()

            Publisher().sendMessage("signals.ballotviewer.set_image_pil", (imgpath, img))
            Publisher().sendMessage("signals.BallotScreen.set_bounding_boxes", (imgpath, target_locations))        
            Publisher().sendMessage("signals.BallotScreen.update_state", BallotScreen.STATE_IDLE)

        self.Refresh()
        self.parent.Fit()
        self.parent.Refresh()
        self.project.addCloseEvent(self.export_bounding_boxes)

    def remove_targets(self, templatepath):
        """
        Remove all voting targets from self.box_locations that are for
        templatepath. Keeps all contest bounding boxes though.
        """
        self.world.remove_voting_targets(templatepath)
          
    def remove_attrtype(self, attrtype):
        """
        Remove the given Attribute Type. Only removes it from 
        data structures directly-created by 'Label Ballot Attributes': 
        be aware, the data structures created by 'Define Ballot Attributes'
        will still contain attrtype.
        """
        self.world.remove_attrtype(attrtype)
        self.Refresh()

    def import_bounding_boxes(self):
        """
        Tries to load in all attributes patches for all blank ballots
        from the 'precinct_locations/' folder (.csv files). Returns
        None if they haven't been created yet.
        This includes both attribute types and attribute values.
        Remember to treat the dummy row specially.
        """
        if not self.world or not self.project:
            return None
        box_locations = {}
        w_img, h_img = None, None
        # frontback_map is a dict {str temppath: str side}
        frontback_map = pickle.load(open(self.project.frontback_map, 'rb'))
        fields = ('imgpath', 'id', 'x', 'y', 'width', 'height', 
                  'attr_type', 'attr_val', 'side')
        boxloc_map = {}  # Maps {str imgpath: {(x1,y1,x2,y2): AttributeBox}}
        for dirpath, dirnames, filenames in os.walk(self.project.patch_loc_dir):
            for filename in [f for f in filenames if f.lower().endswith('.csv')]:
                filepath = pathjoin(dirpath, filename)
                csvfile = open(filepath, 'r')
                dictreader = csv.DictReader(csvfile)
                for row in dictreader:
                    imgpath = row['imgpath']
                    if not w_img:
                        w_img, h_img = util_gui.open_as_grayscale(imgpath).size
                        w_img, h_img = float(w_img), float(h_img)
                    id = int(row['id'])
                    if id == DUMMY_ROW_ID:
                        box_locations.setdefault(imgpath, [])
                        continue
                    x1 = int(row['x']) / w_img
                    y1 = int(row['y']) / h_img
                    x2 = x1 + (int(row['width']) / w_img)
                    y2 = y1 + (int(row['height']) / h_img)
                    attrtype_csv = row['attr_type']
                    attrval_csv  = row['attr_val']
                    attr_type = attrtype_csv if attrtype_csv != "_none_" else None
                    attr_val = attrval_csv if attrval_csv != "_none_" else None
                    side = row['side']
                    box_locations.setdefault(imgpath, [])
                    boxloc_map.setdefault(imgpath, {})
                    if side == frontback_map[imgpath]:
                        if (x1,y1,x2,y2) in boxloc_map[imgpath]:
                            boxloc_map[imgpath][(x1,y1,x2,y2)].add_attrtype(attr_type, attr_val)
                        else:
                            box = AttributeBox(x1, y1, x2, y2)
                            box.add_attrtype(attr_type, attr_val)
                            box_locations.setdefault(imgpath, []).append(box)
                            boxloc_map[imgpath][(x1,y1,x2,y2)] = box
        return box_locations
                                       
    def export_bounding_boxes(self):
        """
        Saves all Attribute patches (location, attr type, attr val, side)
        for all blank ballots to a 'precinct_locations/' folder as
        .csv files.
        Note: The first row is always a DUMMY row, just to indicate the
              imgpath of the blank ballot. DUMMY rows always have an id 
              of -42, for practical (and philosophical) reasons.
        """
        now = time.time()

        if not self.world:
            return
        elif len(self.world.get_boxes_all().items()) == 0:
            # No templates loaded
            return
        elif len(self.world.get_boxes_all_list()) == 0:
            # No bounding boxes created
            return

        util_gui.create_dirs(self.project.patch_loc_dir)
        frontback_map = pickle.load(open(self.project.frontback_map, 'rb'))
        fields = ('imgpath', 'id', 'x', 'y', 'width', 'height',
                  'attr_type', 'attr_val', 'side')
        w_img, h_img = None, None
        for imgpath in self.world.get_boxes_all():
            csvfilepath = pathjoin(self.project.patch_loc_dir, 
                                   "{0}_patchlocs.csv".format(os.path.splitext(os.path.split(imgpath)[1])[0]))
            csvfile = open(csvfilepath, 'wb')
            dictwriter = csv.DictWriter(csvfile, fieldnames=fields)
            try:
                dictwriter.writeheader()
            except AttributeError:
                util_gui._dictwriter_writeheader(csvfile, fields)
            dummyrow = {'imgpath': imgpath, 'id': -42, 'x': 0, 'y':0,
                        'width': 0, 'height': 0, 'attr_type': '_dummy_',
                        'attr_val': '_dummy_', 'side': '_dummy_'}
            dictwriter.writerow(dummyrow)
            for id, box in enumerate(self.world.get_boxes(imgpath)):
                x1, y1, x2, y2 = box.get_coords()
                row = {}
                row['imgpath'] = os.path.abspath(imgpath)
                row['id'] = id
                # Convert relative coords back to pixel coords
                if not w_img or not h_img:
                    w_img, h_img = [float(_) for _ in util_gui.open_as_grayscale(imgpath).size]
                row['x'] = int(round(x1 * w_img))
                row['y'] = int(round(y1 * h_img))
                width = int(round(abs(x1-x2)*w_img))
                height = int(round(abs(y1-y2)*h_img))
                row['width'] = width
                row['height'] = height
                row['side'] = frontback_map[os.path.abspath(imgpath)]
                for attrtype in box.get_attrtypes():
                    attrval = box.get_attrval(attrtype)
                    row_copy = row.copy()
                    row_copy['attr_type'] = '_none_' if not attrtype else attrtype
                    row_copy['attr_val'] = '_none_' if not attrval else attrval
                    dictwriter.writerow(row_copy)
            csvfile.close()
        val = copy.deepcopy(self.world.get_boxes_all())
        return val
    
    #### Pubsub Callbacks
            
    def pubsub_mosaic_img_selected(self, msg):
        imgpath = msg.data
        img = util_gui.open_as_grayscale(imgpath)
        self.ballotviewer.set_image_pil(imgpath, img)
        Publisher().sendMessage("signals.IBallotScreen.update_state", LabelScreen.STATE_IDLE)
        if util.is_multipage(self.project):
            # TODO: Get imgpath of the backside of 'imgpath', then load
            # necessary state (target_locations, img).
            template_to_images = pickle.load(open(self.project.template_to_images, 'rb'))
            image_to_template = pickle.load(open(self.project.image_to_template, 'rb'))
            template_id = image_to_template[imgpath]
            frontpath, backpath = template_to_images[template_id]
            img_back = util_gui.open_as_grayscale(backpath)
            self.ballotviewer_alt.set_image_pil(backpath, img_back)
            old_page = self.nb.GetSelection()
            self.nb.ChangeSelection(0)
            self.nb.SendPageChangedEvent(old_page, 0)
        self.Refresh()

    def validate_outputs(self):
        """
        Checks the output(s) of this widget, raises warning dialogs
        to the user if things aren't right.
        Returns True if everything is OK, False otherwise.
        """
        if (not self.is_attrslabeled() or
            not self.is_attrsvalid() or
            not self.is_attrmaps_unique()):
            return False
        print 'LABEL ATTRIBUTES HAS VALID OUTPUTS'
        return True

    def checkCanMoveOn(self):
        canMoveOn = True
        # TODO: Implement this method to check to see if all
        #       attribute patches have been labeled.
        # Currently not integrated with validate_outputs - need to
        # find the 'unified' approach to this sort of thing.
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
        '''
        try:
            attributes = pickle.load(open(project.ballot_attributesfile, 'rb'))
        except IOError as e:
            # This project is 'old' and doesn't have it. Create it to 
            # 'update' this project.
            attributes = {}
            project.ballot_attributesfile = os.path.join(project.projdir_path, 'ballot_attributes.p')
            attrs_file = open(project.ballot_attributesfile, 'wb')
            pickle.dump(attributes, attrs_file)
            attrs_file.close()
        self.world.attributes = attributes
        self.project.addCloseEvent(self.export_bounding_boxes)
        '''
        
    def _pubsub_projupdate(self, msg):
        """
        Triggered when the Project is updated.
        """
        pass

    def import_attribute_patches(self):
        """
        In 'ballot_attributes.p', there is a dictionary mapping
            {str attr_type: [AttributeBox box, str side]}
        Currently not in use.
        """
        if not self.project.templatesdir:
            # user hasn't gotten to this part yet
            return
        attributes = pickle.load(open(self.project.ballot_attributesfile, 'rb'))
        return attributes

    # Validators

    def is_attrslabeled(self):
        badtemps = set()
        num_unlabeled = 0
        for temppath, boxes in self.world.box_locations.iteritems():
            unlabeled_attrs = []
            for b in boxes:
                for attrtype in b.get_attrtypes():
                    if not b.get_attrval(attrtype):
                        unlabeled_attrs.append(attrtype)
            if unlabeled_attrs:
                badtemps.add(temppath)
            num_unlabeled += len(unlabeled_attrs)
        if badtemps:
            msg = """Warning: There were {0} unlabeled attributes \
across {1} blank ballots. Please go back and assign values to these \
attributes.""".format(num_unlabeled, len(badtemps))
            dlg = wx.MessageDialog(self, message=msg, style=wx.OK)
            dlg.ShowModal()
            return False
        return True

    def is_attrsvalid(self):
        """
        Check to see if any attributes have the same value across all
        blank ballots (i.e. is totally 'useless').
        """
        # maps {str attrtype: set() attrvals}
        attrmap = {}
        for temppath, boxes in self.world.box_locations.iteritems():
            for b in boxes:
                for attrtype in b.get_attrtypes():
                    attrmap.setdefault(attrtype, set()).add(b.get_attrval(attrtype))
        useless_attrs = [] # Tuples of (str attrtype, str attrval)
        for attrtype, attrvals_set in attrmap.iteritems():
            if len(attrvals_set) == 1:
                attrval = list(attrvals_set)[0]
                useless_attrs.append((attrtype, attrval))
        if useless_attrs:
            useless_attrtypes = [foo[0] for foo in useless_attrs]
            useless_attrvals = [foo[1] for foo in useless_attrs]
            msg = """Warning: The Ballot Attribute '{0}' has the same \
value across all blank ballots: '{1}'. If you think you made a mistake, \
please go back and correct this mistake.

On the other hand, if this Ballot Attribute is unnecessary (because it \
only has one value across all blank ballots), then tell OpenCount to \
discard it by choosing the 'Discard and Proceed' button.""".format(useless_attrtypes,
                                                                   useless_attrvals)
            GOBACK = 42
            PROCEED = 9001
            dlg = util.WarningDialog(self, msg, ("Go back", "Discard and Proceed"),
                                     (GOBACK, PROCEED))
            statusval = dlg.ShowModal()
            if statusval == GOBACK:
                return False
            else:
                for (attrtype, attrval) in useless_attrs:
                    self.remove_attrtype(attrtype)
                    # also remove attrtype from 'Define Ballot Attributes'
                    self._remove_attrtype_dba(attrtype)
                return True
        return True

    def _remove_attrtype_dba(self, attrtype):
        """
        Remove attrtype from the output files of 'Define Ballot Attribute',
        since it contains its own data structures.
        Modifies the self.project.ballot_attributesfile, which is a pickle'd
        dictionary {str attrtype: [AttributeBox, str side]}
        """
        attrboxes = common.load_attrboxes(self.project.ballot_attributesfile)
        for b in attrboxes:
            if attrtype in b.get_attrtypes():
                b.remove_attrtype(attrtype)
        attrboxes = [b for b in attrboxes if b.get_attrtypes()]
        common.dump_attrboxes(attrboxes, self.project.ballot_attributesfile)

    def is_attrmaps_unique(self):
        """
        Return True if each blank ballot has a unique attrtype->attrval
        mapping.
        Assumes that:
          - All attributes are labeled
        """
        attrmaps_all = {} # maps {str temppath: {str attrtype: str attrval}}
        for temppath, boxes in self.world.box_locations.iteritems():
            for b in boxes:
                for attrtype in b.get_attrtypes():
                    attrmaps_all.setdefault(temppath, {})[attrtype] = b.get_attrval(attrtype)
        exemplar = None
        duplicate_temps = []
        for temppath, attrmap in attrmaps_all.iteritems():
            if not exemplar:
                exemplar = (temppath, attrmap)
                continue
            if attrmap == exemplar:
                duplicate_temps.append((temppath, attrmap))
        if duplicate_temps:
            msg = """Warning: {0} blank ballots did not have unique \
Ballot Attribute values. OpenCount requires each blank ballot to have \
its own uniquely-identifying ballot attribute values for grouping \
purposes.

Please go back and correct.""".format(len(duplicate_temps))
            dlg = wx.MessageDialog(self, message=msg, style=wx.OK)
            dlg.ShowModal()
            return False
        return True

class LightToolBar(imageviewer.ToolBar):
    """
    Only need zoom in/out and select.
    """
    def __init__(self, parent, *args, **kwargs):
        imageviewer.ToolBar.__init__(self, parent, *args, **kwargs)

        # Instance vars
        self.iconsdir = os.path.join(MYDIR, '..', 'specify_voting_targets', 'imgs','icons')
               
    def _populate_icons(self, iconsdir):
        imageviewer.ToolBar._populate_icons(self, iconsdir)
        self.btn_addtarget.GetParent().Hide()
        self.btn_addcontest.GetParent().Hide()
        self.btn_splitcontest.GetParent().Hide()
        self.btn_undo.GetParent().Hide()
                          
