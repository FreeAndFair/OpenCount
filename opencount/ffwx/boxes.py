'''
These are classes that represent boxes being drawn on-screen in various
UI widgets.
'''

import graphcolour
import math
import numpy as np
import wx
from wx.lib.scrolledpanel import ScrolledPanel


class Box(object):
    # SHADING: (int R, int G, int B)
    #     (Optional) color of transparent shading for drawing
    shading_clr = None
    shading_selected_clr = None
    shading_clr_cycle = None

    def __init__(self, x1, y1, x2, y2):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self._dirty = True

    @property
    def width(self):
        return abs(self.x1 - self.x2)

    @property
    def height(self):
        return abs(self.y1 - self.y2)

    def __repr__(self):
        return "{cls}({0},{1},{2},{3})".format(
            self.x1,
            self.y1,
            self.x2,
            self.y2,
            cls=self.__class__.__name__)

    def __eq__(self, o):
        return (isinstance(o, self.__class__) and
                self.x1 == o.x1 and
                self.x2 == o.x2 and
                self.y1 == o.y1 and
                self.y2 == o.y2)

    def canonicalize(self):
        """ Re-arranges my points (x1,y1),(x2,y2) such that we get:
            (x_upperleft, y_upperleft, x_lowerright, y_lowerright)
        """
        xa, ya, xb, yb = self.x1, self.y1, self.x2, self.y2
        w, h = abs(xa - xb), abs(ya - yb)
        if xa < xb and ya < yb:
            # UpperLeft, LowerRight
            self.x1, self.y1 = xa, ya
            self.x2, self.y2 = xb, yb
        elif xa < xb and ya > yb:
            # LowerLeft, UpperRight
            self.x1, self.y1 = xa, ya - h,
            self.x2, self.y2 = xb, yb + h
        elif xa > xb and ya < yb:
            # UpperRight, LowerLeft
            self.x1, self.y1 = xa - w, ya
            self.x2, self.y2 = xb + w, yb
        else:
            # LowerRight, UpperLeft
            self.x1, self.y1 = xb, yb
            self.x2, self.y2 = xa, ya
        return self

    def scale(self, scale):
        self.x1 = int(round(self.x1 * scale))
        self.y1 = int(round(self.y1 * scale))
        self.x2 = int(round(self.x2 * scale))
        self.y2 = int(round(self.y2 * scale))

    def copy(self):
        return self.__class__(self.x1, self.y1, self.x2, self.y2)

    def get_draw_opts(self):
        """ Given the state of me, return the color+line-width for the
        DC to use.
        """
        return ("Green", 2)

    def marshall(self):
        return {'x1': self.x1, 'y1': self.y1, 'x2': self.x2, 'y2': self.y2}


class TargetBox(Box):
    shading_clr = (0, 255, 0)  # Green
    shading_selected_clr = (255, 0, 0)  # Red

    shading_clr_cycle = None

    def __init__(self, x1, y1, x2, y2, is_sel=False):
        Box.__init__(self, x1, y1, x2, y2)
        self.is_sel = is_sel

    def __str__(self):
        return "TargetBox({0},{1},{2},{3},is_sel={4})".format(
            self.x1, self.y1, self.x2, self.y2, self.is_sel)

    def __repr__(self):
        return "TargetBox({0},{1},{2},{3},is_sel={4})".format(
            self.x1, self.y1, self.x2, self.y2, self.is_sel)

    def get_draw_opts(self):
        """ Given the state of me, return the color+line-width for the
        DC to use.
        """
        if self.is_sel:
            return ("Yellow", 1)
        else:
            return ("Green", 1)

    def copy(self):
        return TargetBox(self.x1,
                         self.y1,
                         self.x2,
                         self.y2,
                         is_sel=self.is_sel)


class ContestBox(Box):
    shading_clr = (0, 0, 200)  # Blue
    shading_selected_clr = (171, 0, 240)  # Purple

    # shading_clr_cycle := A list of colors to alternate from
    shading_clr_cycle = ((0, 0, 200), (100, 0, 0),
                         (0, 150, 245), (0, 230, 150), (100, 0, 190))

    def __init__(self, x1, y1, x2, y2, is_sel=False):
        Box.__init__(self, x1, y1, x2, y2)
        self.is_sel = is_sel
        self.colour = None

    def __str__(self):
        return "ContestBox({0},{1},{2},{3},is_sel={4})".format(
            self.x1, self.y1, self.x2, self.y2, self.is_sel)

    def __repr__(self):
        return "ContestBox({0},{1},{2},{3},is_sel={4})".format(
            self.x1, self.y1, self.x2, self.y2, self.is_sel)

    def get_draw_opts(self):
        """ Given the state of me, return the color+line-width for the
        DC to use.
        """
        if self.is_sel:
            return ("Yellow", 1)
        else:
            return ("Blue", 1)

    def copy(self):
        return ContestBox(self.x1,
                          self.y1,
                          self.x2,
                          self.y2,
                          is_sel=self.is_sel)


class SelectionBox(Box):

    def get_draw_opts(self):
        return ("Black", 1)


def canonicalize_box(box):
    """ Takes two arbitrary (x,y) points and re-arranges them
    such that we get:
        (x_upperleft, y_upperleft, x_lowerright, y_lowerright)
    """
    xa, ya, xb, yb = box
    w, h = abs(xa - xb), abs(ya - yb)
    if xa < xb and ya < yb:
        # UpperLeft, LowerRight
        return (xa, ya, xb, yb)
    elif xa < xb and ya > yb:
        # LowerLeft, UpperRight
        return (xa, ya - h, xb, yb + h)
    elif xa > xb and ya < yb:
        # UpperRight, LowerLeft
        return (xa - w, ya, xb + w, yb)
    else:
        # LowerRight, UpperLeft
        return (xb, yb, xa, ya)


def get_boxes_within(boxes, box):
    """ Returns all boxes in BOXES that lie within BOX.
    Input:
        list boxes: [Box_i, ...]
        Box box: Enclosing box.
    Output:
        list outboxes.
    """
    result = []
    for boxA in boxes:
        wA, hA = int(abs(boxA.x1 - boxA.x2)), int(abs(boxA.y1 - boxA.y2))
        if (((boxA.x1 + (wA / 3)) >= box.x1) and
                ((boxA.x2 - (wA / 3)) <= box.x2) and
                ((boxA.y1 + (hA / 3)) >= box.y1) and
                ((boxA.y2 - (hA / 3)) <= box.y2)):
            result.append(boxA)
    return result


def compute_box_ids(boxes):
    """ Given a list of Boxes, some of which are Targets, others
    of which are Contests, geometrically compute the correct
    target->contest associations. Also outputs all voting targets
    which are not contained in a contest.
    Input:
        list BOXES:
    Output:
        (ASSOCS, LONELY_TARGETS)
    dict ASSOCS: {int contest_id, [ContestBox, [TargetBox_i, ...]]}
    list LONELY_TARGETS: [TargetBox_i, ...]
    """
    def containing_box(box, boxes):
        """ Returns the box in BOXES that contains BOX. """
        w, h = box.width, box.height
        # Allow some slack when checking which targets are contained by a
        # contest
        slack_fact = 0.1
        xEps = int(round(w * slack_fact))
        yEps = int(round(h * slack_fact))
        for i, otherbox in enumerate(boxes):
            if ((box.x1 + xEps) >= otherbox.x1 and
                    (box.y1 + yEps) >= otherbox.y1 and
                    (box.x2 - xEps) <= otherbox.x2 and
                    (box.y2 - yEps) <= otherbox.y2):
                return i, otherbox
        return None, None
    assocs = {}
    contests = [b for b in boxes if isinstance(b, ContestBox)]
    # print contests
    targets = [b for b in boxes if isinstance(b, TargetBox)]
    lonely_targets = []
    # Ensure that each contest C is present in output ASSOCS, even if
    # it has no contained voting targets
    # Note: output contest ids are determined by ordering in the CONTESTS list
    for contestid, c in enumerate(contests):
        assocs[contestid] = (c, [])

    for t in targets:
        id, c = containing_box(t, contests)
        if id is None:
            # print "Warning", t, "is not contained in any box."
            lonely_targets.append(t)
        elif id in assocs:
            assocs[id][1].append(t)
        else:
            assocs[id] = [c, [t]]
    return assocs, lonely_targets


class AttrBox(Box):
    shading_clr = (0, 255, 0)
    shading_selected_clr = (255, 0, 0)

    def __init__(self, x1, y1, x2, y2, is_sel=False, label='', attrtypes=None,
                 attrvals=None,
                 is_digitbased=None, num_digits=None, is_tabulationonly=None,
                 side=None, grp_per_partition=None):
        """
        Input:
            bool GRP_PER_PARTITION:
                If True, then this is an attribute that is consistent
                within a single partition P, where partitions are
                defined by the barcode value(s).
        """
        Box.__init__(self, x1, y1, x2, y2)
        self.is_sel = is_sel
        self.label = label
        # TODO: Assume that there is only one attrtype per AttrBox.
        # I don't think we'll ever need multiple attributes per bounding box.
        self.attrtypes = attrtypes
        self.attrvals = attrvals
        self.is_digitbased = is_digitbased
        self.num_digits = num_digits
        self.is_tabulationonly = is_tabulationonly
        self.side = side
        self.grp_per_partition = grp_per_partition

    def __str__(self):
        return "AttrBox({0},{1},{2},{3},{4})".format(
            self.x1, self.y1, self.x2, self.y2, self.label)

    def __repr__(self):
        return "AttrBox({0},{1},{2},{3},{4})".format(
            self.x1, self.y1, self.x2, self.y2, self.label)

    def __eq__(self, o):
        return (isinstance(o, AttrBox) and
                self.x1 == o.x1 and
                self.x2 == o.x2 and
                self.y1 == o.y1 and
                self.y2 == o.y2 and
                self.label == o.label and
                self.side == o.side and
                self.attrvals == o.attrvals and
                self.attrtypes == o.attrtypes)

    def copy(self):
        return AttrBox(self.x1, self.y1, self.x2, self.y2, label=self.label,
                       attrtypes=self.attrtypes, attrvals=self.attrvals,
                       is_digitbased=self.is_digitbased,
                       num_digits=self.num_digits,
                       is_tabulationonly=self.is_tabulationonly,
                       side=self.side,
                       grp_per_partition=self.grp_per_partition)

    def get_draw_opts(self):
        if self.is_sel:
            return ("Yellow", 3)
        else:
            return ("Green", 3)

    def marshall(self):
        """ Return a dict-equivalent version of myself. """
        data = Box.marshall(self)
        data['attrs'] = self.attrtypes
        data['attrvals'] = self.attrvals
        data['side'] = self.side
        data['is_digitbased'] = self.is_digitbased
        data['num_digits'] = self.num_digits
        data['is_tabulationonly'] = self.is_tabulationonly
        data['grp_per_partition'] = self.grp_per_partition
        return data


class ImagePanel(ScrolledPanel):
    """ Basic widget class that display one image out of N image paths.
    Also comes with a 'Next' and 'Previous' button. Extend me to add
    more functionality (i.e. mouse-related events).
    """

    def __init__(self, parent, *args, **kwargs):
        ScrolledPanel.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        # self.img := a WxImage
        self.img = None
        # self.imgbitmap := A WxBitmap
        self.imgbitmap = None
        # self.npimg := A Numpy-version of an untarnished-version of
        # self.imgbitmap
        self.npimg = None

        # self.scale: Scaling factor used to display self.IMGBITMAP
        self.scale = 1.0

        # If True, a signal to completely-redraw the original image
        self._imgredraw = False

        self._setup_ui()
        self._setup_evts()

    def _setup_ui(self):
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.sizer)

    def _setup_evts(self):
        self.Bind(wx.EVT_PAINT, self.onPaint)
        self.Bind(wx.EVT_LEFT_DOWN, self.onLeftDown)
        self.Bind(wx.EVT_LEFT_UP, self.onLeftUp)
        self.Bind(wx.EVT_MOTION, self.onMotion)
        self.Bind(wx.EVT_CHILD_FOCUS, self.onChildFocus)

    def set_image(self, img, size=None):
        """ Updates internal data-structures to allow viewing a new input
        IMG. If SIZE is given (width, height), then we will scale image
        to match SIZE, maintaining aspect ratio.
        """
        self.img = img

        c = size[0] / float(self.img.GetWidth()) if size else self.scale
        self.set_scale(c)

    def set_scale(self, scale):
        """ Changes scale, i.e. to acommodate zoom in/out. Mutates the
        self.IMGBITMAP.
        Input:
            float scale: Smaller values -> zoomed out images.
        """
        self.scale = scale
        w, h = self.img.GetWidth(), self.img.GetHeight()
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        self.imgbitmap = img_to_wxbitmap(self.img, (new_w, new_h))
        self.npimg = wxBitmap2np_v2(self.imgbitmap, is_rgb=True)

#        self.sizer.Detach(0)
        self.sizer.Add(self.imgbitmap.GetSize())
        self.SetupScrolling()

        self.Refresh()

    def zoomin(self, amt=0.1):
        self.set_scale(self.scale + amt)

    def zoomout(self, amt=0.1):
        self.set_scale(self.scale - amt)

    def client_to_imgcoord(self, x, y):
        """ Transforms client (widget) coordinates to the Image
        coordinate system -- i.e. accounts for image scaling.
        Input:
            int (x,y): Client (UI) Coordinates.
        Output:
            int (X,Y), image coordinates.
        """
        return (int(round(x / self.scale)), int(round(y / self.scale)))

    def c2img(self, x, y):
        """ Convenience method to self.CLIENT_TO_IMGCOORD. """
        return self.client_to_imgcoord(x, y)

    def img_to_clientcoord(self, x, y):
        """ Transforms Image coords to client (widget) coords -- i.e.
        accounts for image scaling.
        Input:
            int (X,Y): Image coordinates.
        Output:
            int (x,y): Client (UI) coordinates.
        """
        return (int(round(x * self.scale)), int(round(y * self.scale)))

    def img2c(self, x, y):
        return self.img_to_clientcoord(x, y)

    def force_new_img_redraw(self):
        """ Forces this widget to completely-redraw self.IMG, even if
        self.imgbitmap contains modifications.
        """
        self._imgredraw = True

    def draw_image(self, dc):
        if not self.imgbitmap:
            return
        if self._imgredraw:
            # Draw the 'virgin' self.img
            self._imgredraw = False
            w, h = self.img.GetWidth(), self.img.GetHeight()
            new_w, new_h = int(round(w * self.scale)
                               ), int(round(h * self.scale))
            self.imgbitmap = img_to_wxbitmap(self.img, (new_w, new_h))

        dc.DrawBitmap(self.imgbitmap, 0, 0)
        return dc

    def onPaint(self, evt):
        if self.IsDoubleBuffered():
            dc = wx.PaintDC(self)
        else:
            dc = wx.BufferedPaintDC(self)
        self.PrepareDC(dc)
        self.draw_image(dc)

        return dc

    def onLeftDown(self, evt):
        x, y = self.CalcUnscrolledPosition(evt.GetPositionTuple())
        evt.Skip()

    def onMotion(self, evt):
        evt.Skip()

    def onChildFocus(self, evt):
        # If I don't override this child focus event, then wx will
        # reset the scrollbars at extremely annoying times. Weird.
        # For inspiration, see:
        #    http://wxpython-users.1045709.n5.nabble.com/ScrolledPanel-mouse-click-resets-scrollbars-td2335368.html
        pass


class BoxDrawPanel(ImagePanel):
    """ A widget that allows a user to draw boxes on a displayed image,
    and each image remembers its list of boxes.
    """

    """ Mouse Mouse:
        M_CREATE: Create a box on LeftDown.
        M_IDLE: Allow user to resize/move/select(multiple) boxes.
    """
    M_CREATE = 0
    M_IDLE = 1

    def __init__(self, parent, *args, **kwargs):
        ImagePanel.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        # self.boxes := [Box_i, ...]
        self.boxes = []

        # self.sel_boxes := [Box_i, ...]
        self.sel_boxes = []

        # Vars to keep track of box-being-created
        self.isCreate = False
        self.box_create = None

        # Vars for resizing behavior
        self.isResize = False
        self.box_resize = None
        self.resize_orient = None  # 'N', 'NE', etc...

        # self.isDragging : Is the user moving-mouse while mouse-left-down
        # is held down?
        self.isDragging = False

        self.mode_m = BoxDrawPanel.M_CREATE

        # BOXTYPE: Class of the Box to create
        self.boxtype = Box

        # _x,_y keep track of last mouse position
        self._x, self._y = 0, 0

    def _setup_evts(self):
        ImagePanel._setup_evts(self)
        self.Bind(wx.EVT_KEY_DOWN, self.onKeyDown)

    def set_mode_m(self, mode):
        """ Sets my MouseMode. """
        self.mode_m = mode
        self.update_cursor()

    def update_cursor(self, force_cursor=None):
        """ Updates the mouse cursor depending on the current state.
        Returns the wx.Cursor that it decides to set.
        To force the mouse cursor, pass in a wx.Cursor as FORCE_CURSOR.
        """
        if force_cursor is not None:
            self.SetCursor(force_cursor)
            return force_cursor
        if self.mode_m == BoxDrawPanel.M_CREATE:
            cursor = wx.StockCursor(wx.CURSOR_CROSS)
        elif self.mode_m == BoxDrawPanel.M_IDLE:
            if self.isResize:
                cursor = wx.StockCursor(wx.CURSOR_SIZING)
            elif self.get_box_to_resize(self._x, self._y)[0]:
                cursor = wx.StockCursor(wx.CURSOR_SIZING)
            elif self.get_boxes_within(self._x, self._y, mode='any'):
                cursor = wx.StockCursor(wx.CURSOR_HAND)
            else:
                cursor = wx.StockCursor(wx.CURSOR_ARROW)
        else:
            cursor = wx.StockCursor(wx.CURSOR_ARROW)
        if self.GetCursor() != cursor:
            self.SetCursor(cursor)
        return cursor

    def set_boxes(self, boxes):
        self.boxes = boxes
        self.dirty_all_boxes()

    def startBox(self, x, y, boxtype=None):
        """ Starts creating a box at (x,y). """
        if boxtype is None:
            boxtype = self.boxtype
        self.isCreate = True
        self.box_create = boxtype(x, y, x + 1, y + 1)
        # Map Box coords to Image coords, not UI coords.
        self.box_create.scale(1 / self.scale)

    def finishBox(self, x, y):
        """ Finishes box creation at (x,y). """
        self.isCreate = False
        # 0.) Canonicalize box coords s.t. order is: UpperLeft, LowerRight.
        self.box_create.canonicalize()
        toreturn = self.box_create
        self.box_create = None
        self.dirty_all_boxes()
        return toreturn

    def set_scale(self, scale, *args, **kwargs):
        self.dirty_all_boxes()
        return ImagePanel.set_scale(self, scale, *args, **kwargs)

    def dirty_all_boxes(self):
        """ Signal to unconditionally-redraw all boxes. """
        for box in self.boxes:
            box._dirty = True

    def select_boxes(self, *boxes):
        for box in boxes:
            box.is_sel = True
        self.sel_boxes.extend(boxes)
        self.dirty_all_boxes()

    def clear_selected(self):
        """ Un-selects all currently-selected boxes, if any. """
        for box in self.sel_boxes:
            box.is_sel = False
        self.sel_boxes = []

    def delete_boxes(self, *boxes):
        """ Deletes the boxes in BOXES. """
        for box in boxes:
            self.boxes.remove(box)
            if box in self.sel_boxes:
                self.sel_boxes.remove(box)
        self.dirty_all_boxes()
        if not self.boxes:
            # Force a redraw of the image - otherwise, the last-removed
            # boxes don't go away.
            self.force_new_img_redraw()
            self.Refresh()

    def get_boxes_within(self, x, y, C=8.0, mode='any'):
        """ Returns a list of Boxes that are at most C units within
        the position (x,y), sorted by distance (increasing).
        Input:
            int (x,y):
        Output:
            list MATCHES, of the form:
                [(obj Box_i, float dist_i), ...]
        """

        results = []
        for box in self.boxes:
            if mode == 'N':
                x1, y1 = self.img2c((box.x1 + (box.width / 2)), box.y1)
            elif mode == 'NE':
                x1, y1 = self.img2c(box.x1 + box.width, box.y1)
            elif mode == 'E':
                x1, y1 = self.img2c(box.x1 + box.width,
                                    box.y1 + (box.height / 2))
            elif mode == 'SE':
                x1, y1 = self.img2c(box.x1 + box.width, box.y1 + box.height)
            elif mode == 'S':
                x1, y1 = self.img2c(
                    box.x1 + (box.width / 2), box.y1 + box.height)
            elif mode == 'SW':
                x1, y1 = self.img2c(box.x1, box.y1 + box.height)
            elif mode == 'W':
                x1, y1 = self.img2c(box.x1, box.y1 + (box.heigth / 2))
            elif mode == 'NW':
                x1, y1 = self.img2c(box.x1, box.y1)
            else:
                # Default to 'any'
                x1, y1 = self.img2c(box.x1, box.y1)
                x2, y2 = self.img2c(box.x2, box.y2)
                if (x > x1 and x < x2 and
                        y > y1 and y < y2):
                    results.append((box, None))
                continue
            dist = distL2(x1, y1, x, y)
            if dist <= C:
                results.append((box, dist))
        if mode == 'any':
            return results
        results = sorted(results, key=lambda t: t[1])
        return results

    def get_box_to_resize(self, x, y, C=5.0):
        """ Returns a Box instance if the current mouse location is
        close enough to a resize location, or None o.w.
        Input:
            int X, Y: Mouse location.
            int C: How close the mouse has to be to a box corner.
        Output:
            (Box, str orientation) or (None, None).
        """
        results = []  # [[orient, box, dist], ...]
        for box in self.boxes:
            locs = {'N': self.img2c(box.x1 + (box.width / 2), box.y1),
                    'NE': self.img2c(box.x1 + box.width, box.y1),
                    'E': self.img2c(box.x1 + box.width, box.y1 + (box.height / 2)),
                    'SE': self.img2c(box.x1 + box.width, box.y1 + box.height),
                    'S': self.img2c(box.x1 + (box.width / 2), box.y1 + box.height),
                    'SW': self.img2c(box.x1, box.y1 + box.height),
                    'W': self.img2c(box.x1, box.y1 + (box.height / 2)),
                    'NW': self.img2c(box.x1, box.y1)}
            for (orient, (x1, y1)) in locs.iteritems():
                dist = distL2(x1, y1, x, y)
                if dist <= C:
                    results.append((orient, box, dist))
        if not results:
            return None, None
        results = sorted(results, key=lambda t: t[2])
        return results[0][1], results[0][0]

    def onLeftDown(self, evt):
        self.SetFocus()
        x, y = self.CalcUnscrolledPosition(evt.GetPositionTuple())
        x_img, y_img = self.c2img(x, y)
        w_img, h_img = self.img.GetSize()
        if x_img >= (w_img - 1) or y_img >= (h_img - 1):
            return

        box_resize, orient = self.get_box_to_resize(x, y)
        if self.mode_m == BoxDrawPanel.M_IDLE and box_resize:
            self.isResize = True
            self.box_resize = box_resize
            self.resize_orient = orient
            self.Refresh()
            self.update_cursor()
            return

        if self.mode_m == BoxDrawPanel.M_CREATE:
            self.clear_selected()
            self.startBox(x, y)
            self.update_cursor()
        elif self.mode_m == BoxDrawPanel.M_IDLE:
            boxes = self.get_boxes_within(x, y, mode='any')
            if boxes:
                b = boxes[0][0]
                if b not in self.sel_boxes:
                    self.clear_selected()
                    self.select_boxes(boxes[0][0])
            else:
                self.clear_selected()
                self.startBox(x, y, SelectionBox)
            self.update_cursor()

    def onLeftUp(self, evt):
        x, y = self.CalcUnscrolledPosition(evt.GetPositionTuple())
        self.isDragging = False
        if self.isResize:
            self.box_resize.canonicalize()
            self.box_resize = None
            self.isResize = False
            self.dirty_all_boxes()
            self.update_cursor()

        if self.mode_m == BoxDrawPanel.M_CREATE and self.isCreate:
            box = self.finishBox(x, y)
            self.boxes.append(box)
            self.update_cursor()
        elif self.mode_m == BoxDrawPanel.M_IDLE and self.isCreate:
            box = self.finishBox(x, y)
            boxes = get_boxes_within(self.boxes, box)
            self.select_boxes(*boxes)
            self.update_cursor()

        self.Refresh()

    def onMotion(self, evt):
        x, y = self.CalcUnscrolledPosition(evt.GetPositionTuple())
        xdel, ydel = x - self._x, y - self._y
        self._x, self._y = x, y

        if self.isResize and evt.Dragging():
            xdel_img, ydel_img = self.c2img(xdel, ydel)
            if 'N' in self.resize_orient:
                self.box_resize.y1 += ydel_img
            if 'E' in self.resize_orient:
                self.box_resize.x2 += xdel_img
            if 'S' in self.resize_orient:
                self.box_resize.y2 += ydel_img
            if 'W' in self.resize_orient:
                self.box_resize.x1 += xdel_img
            self.update_cursor()
            self.Refresh()
            return

        if self.isCreate:
            self.box_create.x2, self.box_create.y2 = self.c2img(x, y)
            self.Refresh()
        elif self.sel_boxes and evt.Dragging():
            self.isDragging = True
            xdel_img, ydel_img = self.c2img(xdel, ydel)
            for box in self.sel_boxes:
                box.x1 += xdel_img
                box.y1 += ydel_img
                box.x2 += xdel_img
                box.y2 += ydel_img
            # Surprisingly, forcing a redraw for each mouse mvmt. is
            # a very fast operation! Very convenient.
            self.dirty_all_boxes()
            self.Refresh()
        self.update_cursor()

    def onKeyDown(self, evt):
        keycode = evt.GetKeyCode()
        if (keycode == wx.WXK_DELETE or keycode == wx.WXK_BACK):
            self.delete_boxes(*self.sel_boxes)
            self.Refresh()
        elif ((keycode in (wx.WXK_UP, wx.WXK_DOWN,
                           wx.WXK_LEFT, wx.WXK_RIGHT)) and
              self.sel_boxes):
            xdel, ydel = 0, 0
            if keycode == wx.WXK_UP:
                ydel -= 1
            elif keycode == wx.WXK_DOWN:
                ydel += 1
            elif keycode == wx.WXK_LEFT:
                xdel -= 1
            else:
                xdel += 1
            xdel_img, ydel_img = self.c2img(xdel, ydel)
            for box in self.sel_boxes:
                box.x1 += xdel_img
                box.y1 += ydel_img
                box.x2 += xdel_img
                box.y2 += ydel_img
            self.dirty_all_boxes()
            self.Refresh()

    def onPaint(self, evt):
        dc = ImagePanel.onPaint(self, evt)
        if self.isResize:
            dboxes = [b for b in self.boxes if b != self.box_resize]
        else:
            dboxes = self.boxes
        self.drawBoxes(dboxes, dc)
        if self.isCreate:
            # Draw Box-Being-Created
            can_box = self.box_create.copy().canonicalize()
            self.drawBox(can_box, dc)
        if self.isResize:
            resize_box_can = self.box_resize.copy().canonicalize()
            self.drawBox(resize_box_can, dc)
        return dc

    def drawBoxes(self, boxes, dc):
        boxes_todo = [b for b in boxes if b._dirty]
        if not boxes_todo:
            return
        # First draw contests, then targets on-top.
        contest_boxes, target_boxes = [], []
        for box in boxes_todo:
            if isinstance(box, ContestBox):
                contest_boxes.append(box)
            else:
                target_boxes.append(box)

        npimg_cpy = self.npimg.copy()

        def draw_border(npimg, box, thickness=2, color=(0, 0, 0)):
            T = thickness
            clr = np.array(color)
            x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2
            x1, y1 = self.img2c(x1, y1)
            x2, y2 = self.img2c(x2, y2)
            x1 = max(x1, 0)
            y1 = max(y1, 0)
            # Top
            npimg[y1:y1 + T, x1:x2] = npimg[y1:y1 + T, x1:x2] * 0.2
            npimg[y1:y1 + T, x1:x2] = npimg[y1:y1 + T, x1:x2] + clr * 0.8
            # Bottom
            npimg[max(0, (y2 - T)):y2,
                  x1:x2] = npimg[max(0, (y2 - T)):y2, x1:x2] * 0.2
            npimg[max(0, (y2 - T)):y2, x1:x2] = \
                npimg[max(0, (y2 - T)):y2, x1:x2] + clr * 0.8
            # Left
            npimg[y1:y2, x1:(x1 + T)] = npimg[y1:y2, x1:(x1 + T)] * 0.2
            npimg[y1:y2, x1:(x1 + T)] = npimg[y1:y2, x1:(x1 + T)] + clr * 0.8
            # Right
            npimg[y1:y2, max(0, (x2 - T)):x2] = \
                npimg[y1:y2, max(0, (x2 - T)):x2] * 0.2
            npimg[y1:y2, max(0, (x2 - T)):x2] = \
                npimg[y1:y2, max(0, (x2 - T)):x2] + clr * 0.8
            return npimg

        # Handle legacy-ContestBoxes that don't have the .colour property
        # TODO: This can be eventually removed. This is more for internal
        #       purposes, to not crash-and-burn on projects created before
        #       this feature was pushed to the repo. Harmless to leave in.
        if contest_boxes and not hasattr(contest_boxes[0], 'colour'):
            recolour_contests(contest_boxes)

        for i, contestbox in enumerate(contest_boxes):
            clr, thickness = contestbox.get_draw_opts()
            draw_border(npimg_cpy, contestbox,
                        thickness=thickness, color=(0, 0, 0))
            if contestbox.is_sel:
                transparent_color = np.array(contestbox.shading_selected_clr) \
                    if contestbox.shading_selected_clr else None
            else:
                transparent_color = np.array(
                    contestbox.colour) if contestbox.colour else None
            if transparent_color is not None:
                _x1, _y1 = self.img2c(contestbox.x1, contestbox.y1)
                _x2, _y2 = self.img2c(contestbox.x2, contestbox.y2)
                np_rect = npimg_cpy[max(0, _y1):_y2, max(0, _x1):_x2]
                np_rect[:, :] = np_rect[:, :] * 0.7
                np_rect[:, :] = np_rect[:, :] + transparent_color * 0.3

            contestbox._dirty = False

        for targetbox in target_boxes:
            clr, thickness = targetbox.get_draw_opts()
            draw_border(npimg_cpy, targetbox,
                        thickness=thickness, color=(0, 0, 0))
            if targetbox.is_sel:
                transparent_color = np.array(targetbox.shading_selected_clr) \
                    if targetbox.shading_selected_clr else None
            else:
                transparent_color = np.array(
                    targetbox.shading_clr) if targetbox.shading_clr else None
            if transparent_color is not None:
                _x1, _y1 = self.img2c(targetbox.x1, targetbox.y1)
                _x2, _y2 = self.img2c(targetbox.x2, targetbox.y2)
                np_rect = npimg_cpy[max(0, _y1):_y2, max(0, _x1):_x2]
                np_rect[:, :] = np_rect[:, :] * 0.7
                np_rect[:, :] = np_rect[:, :] + transparent_color * 0.3

            targetbox._dirty = False

        h, w = npimg_cpy.shape[:2]
        _image = wx.EmptyImage(w, h)
        _image.SetData(npimg_cpy.tobytes())
        bitmap = _image.ConvertToBitmap()

        self.imgbitmap = bitmap
        self.Refresh()

    def drawBox(self, box, dc):
        """ Draws BOX onto DC.
        Note: This draws directly to the PaintDC - this should only be done
        for user-driven 'dynamic' behavior (such as resizing a box), as
        drawing to the DC is much slower than just blitting everything to
        the self.imgbitmap.
        self.drawBoxes does all heavy-lifting box-related drawing in a single
        step.
        Input:
            list box: (x1, y1, x2, y2)
            wxDC DC:
        """

        dc.SetBrush(wx.TRANSPARENT_BRUSH)
        drawops = box.get_draw_opts()
        dc.SetPen(wx.Pen(*drawops))
        w = int(round(abs(box.x2 - box.x1) * self.scale))
        h = int(round(abs(box.y2 - box.y1) * self.scale))
        client_x, client_y = self.img2c(box.x1, box.y1)
        dc.DrawRectangle(client_x, client_y, w, h)

        transparent_color = np.array([200, 0, 0]) if isinstance(
            box, TargetBox) else np.array([0, 0, 200])
        if self.imgbitmap and type(box) in (TargetBox, ContestBox):
            _x1, _y1 = self.img2c(box.x1, box.y1)
            _x2, _y2 = self.img2c(box.x2, box.y2)
            _x1, _y1 = max(0, _x1), max(0, _y1)
            _x2, _y2 = min(self.imgbitmap.Width - 1,
                           _x2), min(self.imgbitmap.Height - 1, _y2)
            if (_x2 - _x1) <= 1 or (_y2 - _y1) <= 1:
                return
            sub_bitmap = self.imgbitmap.GetSubBitmap(
                (_x1, _y1, _x2 - _x1, _y2 - _y1))
            # I don't think I need to do a .copy() here...
            # np_rect = wxBitmap2np_v2(sub_bitmap, is_rgb=True).copy()
            np_rect = wxBitmap2np_v2(sub_bitmap, is_rgb=True)
            np_rect[:, :] = np_rect[:, :] * 0.7
            np_rect[:, :] = np_rect[:, :] + transparent_color * 0.3

            _h, _w, channels = np_rect.shape

            _image = wx.EmptyImage(_w, _h)
            _image.SetData(np_rect.tobytes())
            bitmap = _image.ConvertToBitmap()

            memdc = wx.MemoryDC()
            memdc.SelectObject(bitmap)
            dc.Blit(client_x, client_y, _w, _h, memdc, 0, 0)
            memdc.SelectObject(wx.NullBitmap)

        if isinstance(box, TargetBox) or isinstance(box, ContestBox):
            # Draw the 'grabber' circles
            CIRCLE_RAD = 2
            dc.SetPen(wx.Pen("Black", 1))
            dc.SetBrush(wx.Brush("White"))
            # Upper-Left
            dc.DrawCircle(client_x, client_y, CIRCLE_RAD)
            dc.DrawCircle(client_x + (w / 2), client_y, CIRCLE_RAD)     # Top
            dc.DrawCircle(client_x + w, client_y,
                          CIRCLE_RAD)         # Upper-Right
            dc.DrawCircle(client_x, client_y + (h / 2), CIRCLE_RAD)     # Left
            dc.DrawCircle(client_x + w, client_y +
                          (h / 2), CIRCLE_RAD)   # Right
            dc.DrawCircle(client_x, client_y + h,
                          CIRCLE_RAD)         # Lower-Left
            dc.DrawCircle(client_x + (w / 2), client_y +
                          h, CIRCLE_RAD)     # Bottom
            dc.DrawCircle(client_x + w, client_y + h,
                          CIRCLE_RAD)           # Lower-Right
            dc.SetBrush(wx.TRANSPARENT_BRUSH)


def recolour_contests(contests):
    """ Performs a five-colouring on CONTESTS to improve UI experience.
    Input:
        list CONTESTS: [ContestBox_i, ...]
    Output:
        None. Mutates the input contests.
    """
    def contests2graph():
        contest2node = {}  # maps {ContestBox: Node}
        node2contest = {}  # maps {Node: ContestBox}
        for i, contest in enumerate(contests):
            node = graphcolour.Node(
                (contest.x1, contest.y1, contest.x2, contest.y2, id(contest)))
            contest2node[contest] = node
            node2contest[node] = contest
        for i, contest0 in enumerate(contests):
            for j, contest1 in enumerate(contests):
                if i == j:
                    continue
                if is_adjacent(contest0, contest1):
                    contest2node[contest0].add_neighbor(contest2node[contest1])
        return graphcolour.AdjListGraph(contest2node.values()), node2contest

    graph, node2contest = contests2graph()
    colouring = graphcolour.fivecolour_planar(
        graph, colours=ContestBox.shading_clr_cycle)
    if not colouring:
        colouring = graphcolour.graphcolour(
            graph, colours=ContestBox.shading_clr_cycle)
    for node, colour in colouring.iteritems():
        cbox = node2contest[node]
        cbox.colour = colour


def img_to_wxbitmap(img, size=None):
    """ Converts IMG to a wxBitmap. """
    # TODO: Assumes that IMG is a wxImage
    if size:
        img_scaled = img.Scale(size[0], size[1], quality=wx.IMAGE_QUALITY_HIGH)
    else:
        img_scaled = img
    return wx.BitmapFromImage(img_scaled)


def wxBitmap2np_v2(wxBmp, is_rgb=True):
    """ Converts wxBitmap to numpy array """

    w, h = wxBmp.GetSize()

    npimg = np.zeros(h * w * 3, dtype='uint8')
    wxBmp.CopyToBuffer(npimg, format=wx.BitmapBufferFormat_RGB)
    npimg = npimg.reshape(h, w, 3)

    return npimg


def wxImage2np(Iwx, is_rgb=True):
    """ Converts wxImage to numpy array """
    w, h = Iwx.GetSize()
    Inp_flat = np.frombuffer(Iwx.GetDataBuffer(), dtype='uint8')
    if is_rgb:
        Inp = Inp_flat.reshape(h, w, 3)
    else:
        Inp = Inp_flat.reshape(h, w)
    return Inp


def distL2(x1, y1, x2, y2):
    return math.sqrt((float(y1) - y2)**2.0 + (float(x1) - x2)**2.0)


def is_adjacent(contest0, contest1, C=0.2):
    """ Returns True if the input ContestBoxes are adjacent. """
    def check_topbot(top, bottom):
        return (abs(top.y2 - bottom.y1) < thresh_h and
                is_line_overlap_horiz((top.x1, top.x2),
                                      (bottom.x1, bottom.x2)))

    def check_leftright(left, right):
        return (abs(left.x2 - right.x1) < thresh_w and
                is_line_overlap_vert((left.y1, left.y2),
                                     (right.y1, right.y2)))
    thresh_w = C * min(contest0.width, contest1.width)
    thresh_h = C * min(contest0.height, contest1.height)
    left = contest0 if contest0.x1 <= contest1.x1 else contest1
    right = contest0 if contest0.x1 > contest1.x1 else contest1
    top = left if left.y1 <= right.y1 else right
    bottom = left if left.y1 > right.y1 else right

    if check_topbot(top, bottom):
        return True
    elif check_leftright(left, right):
        return True
    return False


def is_line_overlap_horiz(a, b):
    left = a if a[0] <= b[0] else b
    right = a if a[0] > b[0] else b
    if (left[0] < right[0] and left[1] < right[0]):
        return False
    return True


def is_line_overlap_vert(a, b):
    top = a if a[0] <= b[0] else b
    bottom = a if a[0] > b[0] else b
    if (top[0] < bottom[0] and top[1] < bottom[0]):
        return False
    return True
