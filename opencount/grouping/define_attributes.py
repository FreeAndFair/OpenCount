import sys, csv, copy, os, time, pickle, pdb
from os.path import join as pathjoin
sys.path.append('../')

import wx, wx.animate, Image, cv2, wx.lib.inspection
import numpy as np
import util
from wx.lib.pubsub import Publisher
from specify_voting_targets import find_targets_wizard as find_targets_wizard
from specify_voting_targets import imageviewer as imageviewer
from specify_voting_targets.imageviewer import BallotScreen as BallotScreen
from specify_voting_targets.imageviewer import BallotViewer as BallotViewer
from specify_voting_targets.imageviewer import WorldState as WorldState
from specify_voting_targets.imageviewer import BoundingBox as BoundingBox
from specify_voting_targets.imageviewer import ToolBar as ToolBar
from specify_voting_targets.find_targets_wizard import MosaicPanel as MosaicPanel
from specify_voting_targets import util_gui as util_gui
from common import AttributeBox, IWorldState, TextInputDialog
import common

# Get this script's directory. Necessary to know this information
# since the current working directory may not be the same as where
# this script lives (which is important for loading resources like
# imgs)
try:
    MYDIR = os.path.abspath(os.path.dirname(__file__))
except NameError:
    # This script is being run directly
    MYDIR = os.path.abspath(sys.path[0])

"""
Output files of this module:

project.ballot_attributesfile
    A pickle'd file that lists all defined Ballot Attributes,
    and what side they are each defined on.
"""
                          
class IBallotScreen(imageviewer.BallotScreen):
    """
    Just like a BallotScreen, but allow user to draw bounding boxes
    around Ballot Attributes.
    """
    # Modes
    MODE_DRAW = 10
    MODE_PLACE = 11

    def __init__(self, parent, world, limit=-1, *args, **kwargs):
        imageviewer.BallotScreen.__init__(self, parent, world, *args, **kwargs)

        self.mode = IBallotScreen.MODE_DRAW
        self.canResize = True
        
        self._auto_box = None
        # The AttributeBox to display while the user is labeling it.
        self._limbobox = None

        self.cur_attr = None  # Current selected Attribute

    def subscribe_pubsubs(self):
        imageviewer.BallotScreen.subscribe_pubsubs(self)
        callbacks = (("signals.IBallotScreen.update_mode", self._pubsub_update_mode),
                     ("signals.IBallotScreen.makeAutobox", self._pubsub_make_autobox))
        self.callbacks.extend(callbacks)
        for (topic, callback) in callbacks:
            Publisher().subscribe(callback, topic)
                             
    def remove_targets(self):
        self.world.remove_voting_targets(self.current_imgpath)
        
    def relabel_attribute(self, oldattrname, newattrname):
        """
        Update data structures to accomodate renaming a Ballot Attribute.
        In particular, if the user had already labeled ballot attributes
        in 'Label Ballot Attributes' for 'oldattrname', these values should
        be re-associated with the 'newattrname'. Making the user re-enter
        everything after re-naming the box isn't necessary.
        Assumes that all attrtypes are unique.
        """
        attrbox = self.world.get_attrbox(oldattrname)
        attrbox.relabel_attrtype(oldattrname, newattrname)
        #self.world.remove_box(self.current_imgpath, attrbox)
        #for i, oldattrname in enumerate(oldattrnames):
        #    attrbox = self.world.get_attrbox(oldattrname)
        #    newattrname = newattrnames[i]
        #    try:
        #        attrbox.relabel_attrtype(oldattrname, newattrname)
        #    except Exception as e:
        #        print e
        #        pdb.set_trace()
        #self.world.add_box(self.current_imgpath, attrbox)
        self.Refresh()
        # Now, update .csv files in project.patch_loc_dir, replacing all
        # 'attr_type' columns with oldattrname with newattrname.
        try:
            new_rows = {}
            fields = ('imgpath', 'id', 'x', 'y', 'width', 'height', 
                      'attr_type', 'attr_val', 'side', 'is_digitbased', 'is_tabulationonly')
            notebook = self.parent.parent
            project = notebook.GetParent().project
            for dirpath, dirnames, filenames in os.walk(project.patch_loc_dir):
                for filename in [f for f in filenames if f.lower().endswith('.csv')]:
                    filepath = pathjoin(dirpath, filename)
                    csvfile = open(filepath, 'r')
                    dictreader = csv.DictReader(csvfile)
                    for row in dictreader:
                        attrtype = row['attr_type']
                        if attrtype == oldattrname:
                            row['attr_type'] = newattrname
                        new_rows.setdefault(filepath, []).append(row)
            # Now, write back all rows
            for csvfilepath, rows in new_rows.iteritems():
                csvfile = open(csvfilepath, 'w')
                util_gui._dictwriter_writeheader(csvfile, fields)
                dictwriter = csv.DictWriter(csvfile, fields)
                dictwriter.writerows(rows)
                csvfile.flush()
        except IOError as e:
            print "In 'Define Ballot Attributes', user hasn't gotten to 'Label \
Ballot Attributes' yet."

    #### PubSub callbacks

    def select_target(self, p):
        imageviewer.BallotScreen.select_target(self, p)
        Publisher().sendMessage("broadcast.IBallotScreen.targets_update", (self.current_imgpath, self.world.get_boxes(self.current_imgpath)))
        
    def set_state(self, newstate):
        imageviewer.BallotScreen.set_state(self, newstate)
        self.updateMode()
        self.Refresh()

    #### Event handlers
    
    def onLeftDown(self, event):
        """
        Depending on the edit mode, either creates a new voting target
        box, or moves an existing voting target box (if the mouse is
        close to one).
        """
        self.SetFocus()
        x, y = self.CalcUnscrolledPosition(event.GetPositionTuple())
        if not self.is_within_img((x,y)):
            return
        if self.curstate == IBallotScreen.STATE_ADD_TARGET:
            # Since the user might have scrolled, use self.CalcUnscrolledPosition
            w_img, h_img = self.img_bitmap.GetWidth(), self.img_bitmap.GetHeight()
            x_rel, y_rel = x / float(w_img), y / float(h_img)
            if (self.mode == IBallotScreen.MODE_DRAW):
                w_rel, h_rel = 1.0 / float(w_img), 1.0 / float(h_img)
            elif (self.mode == IBallotScreen.MODE_PLACE):
                w_rel, h_rel = self._auto_box

            x2_rel, y2_rel = x_rel + w_rel, y_rel + h_rel
            new_box = AttributeBox(x_rel, y_rel, x2_rel, y2_rel, is_contest=False)
            self.set_new_box(new_box)
            self.Refresh()
        else:
            imageviewer.BallotScreen.onLeftDown(self, event)

    def onLeftUp(self, event):
        """ Drop the voting target box at the current mouse location. """            
        mousepos = self.CalcUnscrolledPosition(event.GetPositionTuple())
        page = 'front' if self.parent.parent.GetSelection() == 0 else 'back'
        if ((self.curstate == IBallotScreen.STATE_ADD_TARGET)
                and self.is_new_box()):
            new_box = self.get_new_box()
            new_box = util_gui.standardize_box(new_box)
            self.unset_new_box()
            caption = """Please enter the name of this Ballot Attribute. 

Previously entered Ballot Attributes are:\n"""
            for attrtype in self.world.get_attrtypes():
                caption += "    {0}\n".format(attrtype)
            dlg = DefineAttributeDialog(self, message=caption, can_add_more=True)
            self.Disable()
            self._limbobox = new_box
            status = dlg.ShowModal()
            self.Enable()
            if status == wx.ID_OK:
                attr_types = dlg.results
            else:
                # User pushed cancel or something
                self._limbobox = None
                return
            for attrtype in attr_types:
                if attrtype in self.world.get_attrtypes():
                    warning_dlg = wx.MessageDialog(self, message="""Warning: You \
entered '{0}' as the name of this Ballot Attribute, yet '{0}' already exists \
as a Ballot Attribute name. Please don't create two different attributes with \
the same name.""".format(attrtype), style=wx.OK)
                    self.Disable()
                    warning_dlg.ShowModal()
                    self.Enable()
                    self._limbobox = None
                    self.Refresh()
                    return
            new_box.add_attrtypes(attr_types)
            new_box.side = page
            new_box.is_digitbased = dlg.is_digitbased
            if new_box.is_digitbased:
                attrtypes_str = common.get_attrtype_str(attr_types)
                update_numdigits_dict(attrtypes_str, dlg.num_digits, self.parent.parent.GetParent().project)
            new_box.is_tabulationonly = dlg.is_tabulationonly
            self.world.add_box(self.current_imgpath, new_box)
            self._limbobox = None
            self.Refresh()
        elif self.is_resize_target():
            Publisher().sendMessage("broadcast.push_state", self.world)
            x, y = mousepos
            x1, y1, x2, y2 = self._resize_rect
            (ul_x, ul_y), (lr_x, lr_y) = util_gui.get_box_corners((x1, y1), (x2, y2))
            self._resize_target.x1 = ul_x / float(self.img_bitmap.GetWidth())
            self._resize_target.y1 = ul_y / float(self.img_bitmap.GetHeight())
            self._resize_target.x2 = lr_x / float(self.img_bitmap.GetWidth())
            self._resize_target.y2 = lr_y / float(self.img_bitmap.GetHeight())
            self.unset_resize_target()
            Publisher().sendMessage("broadcast.updated_world")
            self.set_auxstate(self._prev_auxstate)
            self._prev_auxstate = None
            self.Refresh()
        else:
            imageviewer.BallotScreen.onLeftUp(self, event)
        
    def onRightDown(self, event):
        """
        If the edit mode is 'Modify', then if the user right-clicks a
        voting target box, bring up a selection to modify it.
        """
        x,y = self.CalcUnscrolledPosition(event.GetPositionTuple())
        attrbox = self.get_closest_target((x,y), mode="interior")
        if attrbox:
            # Display a context menu, to allow relabel/delete
            self.PopupMenu(AttributeContextMenu(self, attrbox, self.parent.parent.GetParent().project), (x,y))
        else:
            imageviewer.BallotScreen.onRightDown(self, event)
        
    def onMotion(self, event):
        """
        Depending on the edit mode, move the voting target box 
        currently held by the user.
        """
        x, y = self.CalcUnscrolledPosition(event.GetPositionTuple())
        if not self._oldmousepos:
            self._oldmousepos = x,y
            
        if (self.curstate == IBallotScreen.STATE_ADD_TARGET
                and self.is_new_box()):
            w_img, h_img = self.img_bitmap.GetWidth(), self.img_bitmap.GetHeight()
            x_rel, y_rel = x / float(w_img), y / float(h_img)
            if (self.mode == IBallotScreen.MODE_DRAW
                and event.LeftIsDown()):
                # I am in the middle of resizing a newly-created box
                self._new_box.x2 = x_rel
                self._new_box.y2 = y_rel
                #Publisher().sendMessage("signals.IMosaicPanel.updated_world")
            elif (self.mode == IBallotScreen.MODE_PLACE):
                w_rel, h_rel = self._auto_boxn
                x2_rel, y2_rel = x_rel + w_rel, y_rel + h_rel
                self._new_box.x1 = x_rel
                self._new_box.x2 = x2_rel
                self._new_box.y1 = y_rel
                self._new_box.y2 = y2_rel
            self._new_box.set_color("Red")
            Publisher().sendMessage("broadcast.IBallotScreen.targets_update", (self.current_imgpath, self.world.get_boxes(self.current_imgpath)))
            self.Refresh()
        elif (self.curstate in (IBallotScreen.STATE_IDLE, IBallotScreen.STATE_MODIFY)
                and not self.is_resize_target()
                and not self._resize_cursor_flag
                and self.canResize):
            # Chcek to see if we need to change mouse cursor
            t, mode = self.get_closest_box_any(self.world.get_boxes(self.current_imgpath), (x,y))
            if t:
                if mode in ('upper-left', 'lower-right'):
                    myCursor = wx.StockCursor(wx.CURSOR_SIZENWSE)
                elif mode in ('top', 'bottom'):
                    myCursor = wx.StockCursor(wx.CURSOR_SIZENS)
                elif mode in ('left', 'right'):
                    myCursor = wx.StockCursor(wx.CURSOR_SIZEWE)
                elif mode in ('upper-right', 'lower-left'):
                    myCursor = wx.StockCursor(wx.CURSOR_SIZENESW)
                elif mode == 'interior':
                    myCursor = wx.StockCursor(wx.CURSOR_SIZING)
                self._prev_cursor = self.GetCursor()
                self.SetCursor(myCursor)
                self._resize_cursor_flag = True
                Publisher().sendMessage("broadcast.IBallotScreen.targets_update", (self.current_imgpath, self.world.get_boxes(self.current_imgpath)))
                self.Refresh()
        else:
            imageviewer.BallotScreen.onMotion(self, event)
        
    def delete_attribute(self, attr_type):
        """
        Delete the given attribute type from my internal data 
        structures, and also from any output files.
        """
        if attr_type in self.world.get_attrtypes():
            self.world.remove_attrtype(attr_type)
        else:
            print "Warning -- in delete_attribute, attrtype {0} was \
not found in self.world.attributes".format(attr_type)
        delete_attr_type(self.parent.GetParent().GetParent().project.patch_loc_dir, attr_type)

    def onKeyDown(self, event):
        keycode = event.GetKeyCode()
        if (self.curstate in (IBallotScreen.STATE_IDLE, IBallotScreen.STATE_MODIFY)
                and self.is_select_target()):
            if (keycode == wx.WXK_DELETE or keycode == wx.WXK_BACK):
                for box in self.get_selected_boxes():
                    for attrtype in box.get_attrtypes():
                        self.delete_attribute(attrtype)
                imageviewer.BallotScreen.onKeyDown(self, event)
                self.updateMode()
                Publisher().sendMessage("broadcast.IBallotScreen.targets_update", (self.current_imgpath, self.world.get_boxes(self.current_imgpath)))
            elif (keycode in (wx.WXK_UP, wx.WXK_DOWN, wx.WXK_LEFT, wx.WXK_RIGHT)):
                imageviewer.BallotScreen.onKeyDown(self, event)
                Publisher().sendMessage("broadcast.IBallotScreen.targets_update", (self.current_imgpath, self.world.get_boxes(self.current_imgpath)))
                self.Refresh()
                return  # To avoid event.Skip() propagating to scrollbars
        self.Refresh()
        event.Skip()
        
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
        if self._limbobox:
            self._draw_box(dc, self._limbobox)
        event.Skip()

    def _display_targets(self, dc):
        imageviewer.BallotScreen._display_targets(self, dc)
        curpage = 'front' if self.parent.parent.GetSelection() == 0 else 'back'
        if self.world:
            dc.SetTextForeground("Blue")
            w_img, h_img = self.img_bitmap.GetWidth(), self.img_bitmap.GetHeight()
            for box in self.world.get_attrboxes():
                if curpage == box.side:
                    attrtypes = ",".join(box.get_attrtypes())
                    x1, y1 = int(round(box.x1 * w_img)), int(round(box.y1 * h_img))
                    w_txt, h_txt = dc.GetTextExtent(attrtypes)
                    x_txt, y_txt = x1, y1 - h_txt
                    if y_txt < 0:
                        y_txt = y1 + int(round(box.height * h_img))
                    dc.DrawText(attrtypes, x_txt, y_txt)
        
    def updateMode(self):
        # Always do DRAW mode
        self.mode = IBallotScreen.MODE_DRAW
        return
        count = self.world.get_boxes_count_all()
        self.canResize = True
        Publisher().sendMessage("signals.IToolbar.enableAuto", False)
        
        if count > 0:
            self.mode = IBallotScreen.MODE_PLACE
            if count == 1:
                self.makeAutoBox()
                
                Publisher().sendMessage("signals.IToolbar.enableAuto", True)
                
                if (self.curstate == IBallotScreen.STATE_ADD_TARGET):
                    self.canResize = False
            else:
                self.canResize = False
        else:
            self.mode = IBallotScreen.MODE_DRAW
            
    def makeAutoBox(self):
        box = None
        
        for boxes in self.world.get_boxes_all().values():
            if (len(boxes) != 0):
                box = boxes[0] 
        if box:
            self._auto_box = abs(box.x1-box.x2), abs(box.y1-box.y2)
            
    def _pubsub_update_mode(self, msg):
        self.updateMode()
        
    def _pubsub_make_autobox(self, msg):
        self.updateMode()
        self.makeAutoBox()

class DefineAttributesPanel(wx.Panel):
    """
    Panel that allows the user to specify bounding boxes around
    all attribute patches, and label them.
    """
    def __init__(self, parent, *args, **kwargs):
        wx.Panel.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.project = None
        self.templatesdir = None
        self.samplesdir = None
        
        self.setup_widgets()

        # Pubsubs
        self.callbacks = []
        self.subscribe_pubsubs()

        # Pubsub Subscribing Required
        Publisher().subscribe(self._pubsub_project, "broadcast.project")
        Publisher().subscribe(self._pubsub_projupdate, "broadcast.projupdate")
       
    def subscribe_pubsubs(self):
        for (topic, callback) in self.callbacks:
            Publisher().subscribe(callback, topic)
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
        self.ballotviewer.unsubscribe_pubsubs()
        if self.ballotviewer_alt:
            self.ballotviewer_alt.unsubscribe_pubsubs()
        self.world.unsubscribe_pubsubs()

    def setup_widgets(self):
        self.sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.world = IWorldState()
        self.nb = wx.Notebook(self)
        self.nb.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.onPageChange)
        self.ballotviewer = BallotViewer(self.nb, self.world,
                                         ballotscreen=IBallotScreen,
                                         toolbar=IToolBar, style=wx.SIMPLE_BORDER)
        self.ballotviewer_alt = None
        self.nb.AddPage(self.ballotviewer, "Page 1")
        self.sizer.Add(self.nb, border=10, proportion=1, flag=wx.EXPAND | wx.ALL | wx.ALIGN_LEFT)
        self.SetSizer(self.sizer)
                
    def onPageChange(self, evt):
        """
        Triggered when the user switches tabs, say, from 'Front' to 
        'Back'.
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

    def stop(self):
        self.export_attribute_patches()
        self.unsubscribe_pubsubs()

    def start(self):
        if not self.templatesdir:
            print 'In DefineAttributes, self.templatesdir not defined.'
            return
        # Make UI changes if this is a multipage election
        if util.is_multipage(self.project):
            self.ballotviewer_alt = BallotViewer(self.nb, self.world,
                                                 ballotscreen=IBallotScreen,
                                                 toolbar=IToolBar, style=wx.SIMPLE_BORDER)
            self.nb.AddPage(self.ballotviewer_alt, 'Page 2')
        self.subscribe_pubsubs()
        self.world.reset()
        # attrboxes is a list of AttributeBoxes
        attrboxes = self.import_attribute_patches()
        if not util.is_multipage(self.project):
            # Display first template on BallotScreen
            imgname = None
            for dirpath, dirnames, filenames in sorted(os.walk(self.templatesdir)):
                for imgname in sorted([f for f in filenames if util_gui.is_image_ext(f)]):
                    imgpath = os.path.abspath(os.path.join(dirpath, imgname))
                    break
            if not imgpath:
                print "Something scary happened - no template images exist in {0}?".format(self.templatesdir)
                exit(1)
            img = util_gui.open_as_grayscale(imgpath)
            for box in attrboxes:
                self.world.add_box(imgpath, box)
            Publisher().sendMessage("signals.ballotviewer.set_image_pil", (imgpath, img))
            Publisher().sendMessage("signals.BallotScreen.update_state", IBallotScreen.STATE_IDLE)
        else:
            # Display first template Ballot on BallotScreens
            template_to_images = pickle.load(open(self.project.template_to_images, 'rb'))
            ballot, (imgpath1, imgpath2) = next(template_to_images.iteritems())
            imgpath1 = os.path.abspath(imgpath1)
            imgpath2 = os.path.abspath(imgpath2)
            img1 = util_gui.open_as_grayscale(imgpath1)
            img2 = util_gui.open_as_grayscale(imgpath2)
            self.ballotviewer.set_image_pil(imgpath1, img1)
            self.ballotviewer_alt.set_image_pil(imgpath2, img2)
            frontback_map = pickle.load(open(self.project.frontback_map, 'rb'))            
            for box in attrboxes:
                if box.side == frontback_map[imgpath1]:
                    self.world.add_box(imgpath1, box)
                else:
                    self.world.add_box(imgpath2, box)
            Publisher().sendMessage("broadcast.updated_world")
            Publisher().sendMessage("signals.BallotScreen.update_state", IBallotScreen.STATE_IDLE)
                
        self.ballotviewer.Show()
        
        self.Refresh()
        self.parent.Fit()
        self.parent.Refresh()

    def _pubsub_project(self, msg):
        """
        Triggered when the user selects a Project. Pull in relevant
        state.
        """
        project = msg.data
        self.project = project
        self.templatesdir = project.templatesdir
        self.ballotviewer.ballotscreen.csvdir = project.patch_loc_dir

    def _pubsub_projupdate(self, msg):
        """
        Triggered when the Project object changes.
        """
        if not self.project:
            return
        self.templatesdir = self.project.templatesdir
        
    def export_attribute_patches(self):
        """
        Marshall my definition Attribute patches to a pickle file
        with a list of marshall'd AttributeBoxes
        """
        #output = marshall_iworldstate(self.world)
        #pickle.dump(output, open(self.project.ballot_attributesfile, 'wb'))
        #common.dump_iworldstate(self.world, self.project.ballot_attributesfile)
        m_boxes = [b.marshall() for b in self.world.get_attrboxes()]
        pickle.dump(m_boxes, open(self.project.ballot_attributesfile, 'wb'))

    def import_attribute_patches(self):
        """
        Unmarshall the pickle'd file, which is:
            <marshall'd box_locations>
        Returns a list of AttributeBoxes.
        """
        #thedump = pickle.load(open(self.project.ballot_attributesfile, 'rb'))
        #return unmarshall_iworldstate(thedump)
        #return common.load_iworldstate(self.project.ballot_attributesfile)
        try:
            listdata = pickle.load(open(self.project.ballot_attributesfile, 'rb'))
            return [AttributeBox.unmarshall(bdata) for bdata in listdata]
        except IOError as e:
            return []

class IToolBar(ToolBar):
    """
    Only need zoom in/out, add box, and select.
    """
    def __init__(self, parent, *args, **kwargs):
        ToolBar.__init__(self, parent, *args, **kwargs)

        # Instance vars
        self.iconsdir = os.path.join(MYDIR, '..', 'specify_voting_targets', 'imgs','icons')
               
    def _populate_icons(self, iconsdir):
        ToolBar._populate_icons(self, iconsdir)
        panel_addcustomattr = wx.Panel(self)
        self.btn_addcustomattr = wx.Button(panel_addcustomattr, label="Custom Attr")
        self.btn_addcustomattr.Bind(wx.EVT_BUTTON, self.onButton_customattr)
        font = wx.Font(8, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        txt = wx.StaticText(panel_addcustomattr, label="Add Custom Attribute", style=wx.ALIGN_CENTER)
        txt.SetFont(font)
        sizer = wx.BoxSizer(wx.VERTICAL)
        panel_addcustomattr.SetSizer(sizer)
        sizer.Add(self.btn_addcustomattr)
        sizer.Add(txt, flag=wx.ALIGN_CENTER)
        self.sizer.Add(panel_addcustomattr)

        self.btn_addtarget.GetParent().GetChildren()[1].SetLabel("Define Ballot Attribute")
        self.btn_addcontest.GetParent().Hide()
        self.btn_splitcontest.GetParent().Hide()
        self.btn_undo.GetParent().Hide()
        self.btn_infercontests.GetParent().Hide()

    def onButton_customattr(self, evt):
        dlg = SpreadSheetAttrDialog(self)
        status = dlg.ShowModal()
        if status == wx.ID_CANCEL:
            return
        path = dlg.path
        print "User selected:", path

class AttributeContextMenu(wx.Menu):
    """
    Context Menu to display when user right-clicks on a Ballot Attribute.
    Allow user to either re-label a box, or delete it.
    """
    def __init__(self, parent, attrbox, project, *args, **kwargs):
        wx.Menu.__init__(self, *args, **kwargs)
        self.parent = parent
        self.project = project
        self.attrbox = attrbox
        relabel_mi = wx.MenuItem(self, wx.NewId(), text="Re-Label")
        self.AppendItem(relabel_mi)
        self.Bind(wx.EVT_MENU, self.on_relabel, relabel_mi)
        delete_mi = wx.MenuItem(self, wx.NewId(), text="Delete")
        self.AppendItem(delete_mi)
        self.Bind(wx.EVT_MENU, self.on_delete, delete_mi)

    def on_relabel(self, evt):
        old_attrtypes = self.attrbox.get_attrtypes()
        dlg = DefineAttributeDialog(self.parent, message="New Attribute Name(s):",
                                    vals=old_attrtypes,
                                    can_add_more=True)
        if self.attrbox.is_digitbased == True:
            dlg.chkbox_is_digitbased.SetValue(True)
        if self.attrbox.is_tabulationonly == True:
            dlg.chkbox_is_tabulationonly.SetValue(True)
        attrs_str = common.get_attrtype_str(self.attrbox.get_attrtypes())
        num_digits = common.get_numdigits(self.project, attrs_str)
        if num_digits == None:
            dlg.num_digits.SetValue('')
        else:
            dlg.num_digits.SetValue(str(num_digits))
        val = dlg.ShowModal()
        if val == wx.ID_OK:
            new_attrtypes = dlg.results
            self.attrbox.is_digitbased = dlg.is_digitbased
            self.attrbox.is_tabulationonly = dlg.is_tabulationonly
            if self.attrbox.is_digitbased:
                attrtypes_str = '_'.join(new_attrtypes)
                update_numdigits_dict(attrtypes_str, dlg.num_digits, self.project)
            for i, new_attrtype in enumerate(new_attrtypes):
                if i < len(old_attrtypes):
                    old_attrtype = old_attrtypes[i]
                    self.parent.relabel_attribute(old_attrtype, new_attrtype)
                else:
                    self.attrbox.add_attrtype(new_attrtype)
            #self.attrbox.attr_type = new_attrname
            #oldbox, page = self.parent.world.attributes.pop(old_attrname)
            #self.parent.world.attributes[new_attrname] = self.attrbox, page
        
    def on_delete(self, evt):
        for attrtype in self.attrbox.get_attrtypes():
            self.parent.delete_attribute(attrtype)
        
class DefineAttributeDialog(wx.Dialog):
    """
    A dialog to allow the user to add attribute types to a 
    bounding box.
    """
    def __init__(self, parent, message="Please enter your input(s).", 
                 vals=('',),
                 can_add_more=False,
                 *args, **kwargs):
        """
        vals: An optional list of values to pre-populate the inputs.
        can_add_more: If True, allow the user to add more text entry
                      fields.
        """
        wx.Dialog.__init__(self, parent, title='Input required', *args, **kwargs)
        self.parent = parent
        self.results = []
        self._panel_btn = None
        self.btn_ok = None
        self.is_digitbased = False
        self.num_digits = None
        self.is_tabulationonly = False

        self.input_pairs = []
        for idx, val in enumerate(vals):
            txt = wx.StaticText(self, label="Attribute {0}:".format(idx))
            input_ctrl = wx.TextCtrl(self, style=wx.TE_PROCESS_ENTER)
            if idx == len(vals) - 1:
                input_ctrl.Bind(wx.EVT_TEXT_ENTER, self.onButton_ok)
            input_ctrl.SetValue(vals[idx])
            self.input_pairs.append((txt, input_ctrl))
        if not self.input_pairs:
            txt = wx.StaticText(self, label="Attribute 0")
            input_ctrl = wx.TextCtrl(self, style=wx.TE_PROCESS_ENTER)
            input_ctrl.Bind(wx.EVT_TEXT_ENTER, self.onButton_ok)
            self.input_pairs.append((txt, input_ctrl))

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        caption_txt = wx.StaticText(self, label=message)
        self.sizer.Add(caption_txt, border=10, flag=wx.ALL)
        gridsizer = wx.GridSizer(rows=0, cols=2, hgap=5, vgap=3)
        btn_add = wx.Button(self, label="+")
        self.btn_add = btn_add
        btn_add.Bind(wx.EVT_BUTTON, self.onButton_add)
        btn_add.Bind(wx.EVT_SET_FOCUS, self.onAddButtonFocus)
        
        horizsizer = wx.BoxSizer(wx.HORIZONTAL)
        horizsizer.Add(btn_add, proportion=0, flag=wx.ALIGN_LEFT | wx.ALIGN_TOP)

        gridsizer.Add(self.input_pairs[0][0])
        gridsizer.Add(self.input_pairs[0][1])
        horizsizer.Add(gridsizer)
        for txt, input_ctrl in self.input_pairs[1:]:
            gridsizer.Add((1,1))
            gridsizer.Add(txt, border=10, flag=wx.ALL)
            gridsizer.Add(input_ctrl, border=10, flag=wx.ALL)
        self.gridsizer = gridsizer
        self.sizer.Add(horizsizer)
        
        self.chkbox_is_digitbased = wx.CheckBox(self, label="Is this a digit-based precinct patch?")
        self.chkbox_is_tabulationonly = wx.CheckBox(self, label="Should \
this patch be only used for tabulation (and not for grouping)?")
        numdigits_label = wx.StaticText(self, label="Number of Digits:")
        self.num_digits = wx.TextCtrl(self, value='')
        digit_sizer = wx.BoxSizer(wx.HORIZONTAL)
        digit_sizer.Add(self.chkbox_is_digitbased, proportion=0)
        digit_sizer.Add(numdigits_label, proportion=0)
        digit_sizer.Add(self.num_digits, proportion=0)
        self.sizer.Add(digit_sizer, proportion=0)
        self.sizer.Add(self.chkbox_is_tabulationonly, proportion=0)

        self._add_btn_panel(self.sizer)
        self.SetSizer(self.sizer)
        if not can_add_more:
            btn_add.Hide()
        self.Fit()

        self.input_pairs[0][1].SetFocus()

    def onAddButtonFocus(self, evt):
        """
        Due to tab-traversal issues, do this annoying thing where we
        shift focus away from the '+' button. Sigh.
        """
        if len(self.input_pairs) > 1:
            self.input_pairs[1][1].SetFocus()
        else:
            self.btn_ok.SetFocus()

    def _add_btn_panel(self, sizer):
        """
        Due to tab-traversal issues, do this annoying hack where we
        re-create the button panel every time we dynamically add new
        rows to the dialog.
        """
        if self._panel_btn:
            sizer.Remove(self._panel_btn)
            self._panel_btn.Destroy()
            self._panel_btn = None
        panel_btn = wx.Panel(self)
        self._panel_btn = panel_btn
        btn_ok = wx.Button(panel_btn, id=wx.ID_OK)
        btn_ok.Bind(wx.EVT_BUTTON, self.onButton_ok)
        self.btn_ok = btn_ok
        btn_cancel = wx.Button(panel_btn, id=wx.ID_CANCEL)
        btn_cancel.Bind(wx.EVT_BUTTON, self.onButton_cancel)
        panel_btn.sizer = wx.BoxSizer(wx.HORIZONTAL)
        panel_btn.sizer.Add(btn_ok, border=10, flag=wx.RIGHT)
        panel_btn.sizer.Add(btn_cancel, border=10, flag=wx.LEFT)
        panel_btn.SetSizer(panel_btn.sizer)
        sizer.Add(panel_btn, border=10, flag=wx.ALL | wx.ALIGN_CENTER)

    def onButton_ok(self, evt):
        history = set()
        if self.chkbox_is_digitbased.GetValue() == True:
            self.is_digitbased = True
            self.num_digits = int(self.num_digits.GetValue())
        if self.chkbox_is_tabulationonly.GetValue() == True:
            self.is_tabulationonly = True
        for txt, input_ctrl in self.input_pairs:
            val = input_ctrl.GetValue()
            if val in history:
                dlg = wx.MessageDialog(self, message="{0} was entered \
more than once. Please correct.".format(val),
                                       style=wx.OK)
                dlg.ShowModal()
                return
            self.results.append(input_ctrl.GetValue())
            history.add(val)
        self.EndModal(wx.ID_OK)

    def onButton_cancel(self, evt):
        self.EndModal(wx.ID_CANCEL)

    def onButton_add(self, evt):
        txt = wx.StaticText(self, label="Attribute {0}:".format(len(self.input_pairs)))
        input_ctrl = wx.TextCtrl(self, style=wx.TE_PROCESS_ENTER)
        input_ctrl.Bind(wx.EVT_TEXT_ENTER, self.onButton_ok)
        self.input_pairs[-1][1].Unbind(wx.EVT_TEXT_ENTER)
        self.input_pairs.append((txt, input_ctrl))
        self.gridsizer.Add(txt)
        self.gridsizer.Add(input_ctrl)
        self._add_btn_panel(self.sizer)
        self.Fit()
        input_ctrl.SetFocus()

class SpreadSheetAttrDialog(DefineAttributeDialog):
    def __init__(self, parent, *args, **kwargs):
        DefineAttributeDialog.__init__(self, parent, *args, **kwargs)

        # The path that the user selected
        self.path = ''

        self.parent = parent
        self.chkbox_is_digitbased.Disable()
        self.chkbox_is_tabulationonly.Disable()
        self.num_digits.Disable()
        self.btn_add.Disable()
        txt = wx.StaticText(self, label="Spreadsheet File:")
        file_inputctrl = wx.TextCtrl(self, style=wx.TE_READONLY)
        self.file_inputctrl = file_inputctrl
        btn_select = wx.Button(self, label="Select...")
        btn_select.Bind(wx.EVT_BUTTON, self.onButton_selectfile)

        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(txt)
        sizer.Add((10, 10))
        sizer.Add(file_inputctrl, proportion=1, flag=wx.EXPAND)
        sizer.Add(btn_select)

        self.input_pairs.append((txt, file_inputctrl))

        self.sizer.Insert(len(self.sizer.GetChildren())-1, sizer,
                          proportion=1,
                          border=10,
                          flag=wx.EXPAND | wx.ALL)

        self.Fit()

    def onButton_selectfile(self, evt):
        dlg = wx.FileDialog(self, message="Choose spreadsheet...",
                            defaultDir='.', style=wx.FD_OPEN)
        status = dlg.ShowModal()
        if status == wx.ID_CANCEL:
            return
        path = dlg.GetPath()
        self.file_inputctrl.SetValue(path)
        self.path = path

def delete_attr_type(attrvalsdir, attrtype):
    """
    Call this when the user deletes an Attribute definition. This will
    remove all mentions of the deleted attribute type from the .csv
    files.
    attrvalsdir is the project.patch_loc_dir, i.e. the output directory
    of 'Label Ballot Attributes'. It may or may not exist, btw.
    """
    fields = ('imgpath', 'id', 'x', 'y', 'width', 'height', 'attr_type', 'attr_val', 'side','is_digitbased','is_tabulationonly')
    for dirpath, dirnames, filenames in os.walk(attrvalsdir):
        for filename in [f for f in filenames if f.lower().endswith('.csv')]:
            filepath = pathjoin(dirpath, filename)
            csvfile = open(filepath, 'rb')
            reader = csv.DictReader(csvfile, fields)
            rows = []
            for row in reader:
                if row['attr_type'] != attrtype:
                    rows.append(row)
            csvfile.close()
            # truncate the csvfile
            with open(filepath, 'w'):
                pass
            # now, write the new contents
            csvfile = open(filepath, 'wb')
            writer = csv.DictWriter(csvfile, fields)
            writer.writerows(rows)

def update_numdigits_dict(attrtype, numdigits, project):
    """Updates the num_digitsmap dictionary. """
    num_digitsmappath = os.path.join(project.projdir_path,
                                     project.num_digitsmap)
    if os.path.exists(num_digitsmappath):
        f = open(num_digitsmappath, 'rb')
        num_digitsmap = pickle.load(f)
        f.close()
    else:
        num_digitsmap = {}
    num_digitsmap[attrtype] = int(numdigits)
    outf = open(num_digitsmappath, 'wb')
    pickle.dump(num_digitsmap, outf)
    outf.close()
    return num_digitsmap

def domain():
    class MyFrame(wx.Frame):
        def __init__(self, parent, *args, **kwargs):
            wx.Frame.__init__(self, parent, *args, **kwargs)
            btn = wx.Button(self, label="Push for dlg")
            btn.Bind(wx.EVT_BUTTON, self.onButton)

        def onButton(self, evt):
            attributes = {'precinct number': ['42', '9001'],
                              'location': ['mail in', 'polling place']}
            dlg = SetAttributeDialog(self, attributes, None)
            dlg.ShowModal()
    app = wx.App(False)
    f = MyFrame(None)
    f.Show()
    app.MainLoop()

if __name__ == '__main__':
    domain()
