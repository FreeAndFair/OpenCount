import sys, os, time, pdb, traceback, threading, Queue
try:
    import cPickle as pickle
except ImportError:
    import pickle

sys.path.append('..')

import wx
from wx.lib.scrolledpanel import ScrolledPanel

import util_gui

class SelectTargetsMainPanel(ScrolledPanel):
    def __init__(self, parent, *args, **kwargs):
        ScrolledPanel.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        self.proj = None
        self.imgpaths = None

    def start(self, proj, imgpaths):
        self.proj = proj
        self.imgpaths = imgpaths

class SelectTargetsPanel(ScrolledPanel):
    def __init__(self, parent, *args, **kwargs):
        ScrolledPanel.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        self.imgpaths = None
        # self.cur_i: Index of currently-displayed image (w.r.t self.IMGPATHS)
        self.cur_i = None

        # self.boxes: {int idx: [(x1, y1, x2, y2), ...]}
        self.boxes = {}

        self.toolbar = Toolbar(self)
        self.imagepanel = TemplateMatchDrawPanel(self, self.do_tempmatch)

        txt = wx.StaticText(self, label="Select all Voting Targets from \
this partition.")

        btn_next = wx.Button(self, label="Next Ballot")
        btn_prev = wx.Button(self, label="Previous Ballot")
        
        btn_next.Bind(wx.EVT_BUTTON, self.onButton_next)
        btn_prev.Bind(wx.EVT_BUTTON, self.onButton_prev)

        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        btn_sizer.AddMany([(btn_prev,), (btn_next,)])
        
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(txt, flag=wx.ALIGN_CENTER)
        self.sizer.Add(self.toolbar, flag=wx.EXPAND)
        self.sizer.Add(self.imagepanel, proportion=1, flag=wx.EXPAND)
        self.sizer.Add(btn_sizer, flag=wx.ALIGN_CENTER)

        self.SetSizer(self.sizer)

    def start(self, imgpaths):
        self.imgpaths = imgpaths
        print "len(imgpaths):", len(imgpaths)

        self.display_image(0)

        self.SetupScrolling()

    def do_tempmatch(self, box, img):
        """ Runs template matching on all of my self.IMGPATHS, using 
        the BOX from IMG as the template.
        Input:
            Box BOX:
            wximg IMG:
        """
        class TM_Thread(threading.Thread):
            TEMPLATE_MATCH_JOBID = 48
            def __init__(self, queue, job_id, patch, img, imgpaths, callback, *args, **kwargs):
                self.queue = queue
                self.job_id = job_id
                self.patch = patch
                self.img = img
                self.imgpaths = imgpaths
                self.callback = callback
            def run(self):
                print "Running template matching..."
                THRESHOLD = 0.85
                # results = partask.do_partask(tempmatch, imgpaths, _args=(patch, THRESHOLD))
        # 1.) If this is the first-created box, then do an autofit.
        # Else, use the exact input BOX.
        if len(self.boxes.values()) == 0:
            patch_prefit = img.crop(box)
            patch = util_gui.fit_image(patch, padx=0, pady=0)
        else:
            patch = img.crop(box)
        # 2.) Run template matching across all images in self.IMGPATHS,
        # using PATCH as the template.
        queue = Queue.Queue()
        thread = TM_Thread(queue, self.TEMPLATE_MATCH_JOBID, patch, img, self.imgpaths, self.on_tempmatch_done)
        thread.start()

    def on_tempmatch_done(self, results):
        """ Invoked after template matching computation is complete. 
        Input:
            dict RESULTS: maps {str imgpath: [Box_i, ...]}. The matches
                that template matching discovered.
        """
        def is_overlap(rect1, rect2):
            def is_within_box(pt, box):
                return box.x1 < pt[0] < box.x2 and box.y1 < pt[1] < box.y2
            x1, y1, x2, y2 = rect1.x1, rect.y1, rect.x2, rect.y2
            w, h = abs(x2-x1), abs(y2-y1)
            # Checks (in order): UL, UR, LR, LL corners
            return (is_within_box((x1,y1), rect2) or
                    is_within_box((x1+w,y1), rect2) or 
                    is_within_box((x1+w,y1+h), rect2) or 
                    is_within_box((x1,y1+h), rect2))
        def too_close(b1, b2):
            w, h = abs(b1.x1-b1.x2), abs(b1.y1-b1.y2)
            return ((abs(b1.x1 - b2.x1) <= w / 2.0 and
                     abs(b1.y1 - b2.y1) <= h / 2.0) or
                    is_overlap(b1, b2) or 
                    is_overlap(b2, b1))
        # 1.) Add the new matches to self.BOXES, but also filter out
        # any matches in RESULTS that are too close to previously-found
        # matches.
        for imgpathA, boxesA in self.boxes.iteritems():
            boxesB = results.get(imgpathA, [])
            i = 0
            while i < len(boxesB):
                boxB = boxesB[i]
                for boxA in boxesA:
                    if too_close(boxA, boxB):
                        boxesB.pop(i)
                    else:
                        self.boxes.setdefault(imgpathA, []).append(boxB)
                        i += 1
        print "...Finished adding results from tempmatch run."

    def display_image(self, idx):
        """ Displays the image at IDX. Also handles reading/saving in
        the currently-created boxes for the old/new image.
        Input:
            int IDX: Index (into self.IMGPATHS) that you want to display.
        Output:
            Returns the IDX we decided to display, if successful.
        """
        if idx < 0 or idx >= len(self.imgpaths):
            print "Invalid idx into self.imgpaths:", idx
            pdb.set_trace()
        # 0.) Save boxes of old image
        if self.cur_i != None:
            self.boxes.setdefault(self.cur_i, []).extend(self.imagepanel.boxes)
            
        self.cur_i = idx
        imgpath = self.imgpaths[self.cur_i]
        
        # 1.) Display New Image
        print "...Displaying image:", imgpath
        wximg = wx.Image(imgpath, wx.BITMAP_TYPE_ANY)
        # 1.a.) Resize image s.t. width is equal to containing width
        wP, hP = self.parent.GetSize()
        _c = wximg.GetWidth() / float(wP)
        wimg = wP
        himg = int(round(wximg.GetHeight() / _c))
        self.imagepanel.set_image(wximg, size=(wimg, himg))
        
        # 2.) Read in previously-created boxes for IDX (if exists)
        boxes = self.boxes.get(self.cur_i, [])
        self.imagepanel.set_boxes(boxes)

        self.SetupScrolling()
        return idx

    def display_next(self):
        """ Displays the next image in self.IMGPATHS. If the end of the
        list is reached, returns None, and does nothing. Else, returns
        the new image index.
        """
        next_idx = self.cur_i + 1
        if next_idx >= len(self.imgpaths):
            return None
        return self.display_image(next_idx)
        
    def display_prev(self):
        prev_idx = self.cur_i - 1
        if prev_idx < 0:
            return None
        return self.display_image(prev_idx)

    def onButton_next(self, evt):
        self.display_next()

    def onButton_prev(self, evt):
        self.display_prev()

class Toolbar(wx.Panel):
    def __init__(self, parent, *args, **kwargs):
        wx.Panel.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        
        self._setup_ui()
        self._setup_evts()
        self.Layout()

    def _setup_ui(self):
        self.btn_addtarget = wx.Button(self, label="Add Target...")
        self.btn_modify = wx.Button(self, label="Modify...")
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        btn_sizer.AddMany([(self.btn_addtarget,), (self.btn_modify,)])
        self.sizer.Add(btn_sizer)
        self.SetSizer(self.sizer)

    def _setup_evts(self):
        self.btn_addtarget.Bind(wx.EVT_BUTTON, lambda evt: self.setmode(BoxDrawPanel.M_CREATE))
        self.btn_modify.Bind(wx.EVT_BUTTON, lambda evt: self.setmode(BoxDrawPanel.M_IDLE))
    def setmode(self, mode_m):
        self.parent.imagepanel.set_mode_m(mode_m)

class ImagePanel(wx.Panel):
    """ Basic widget class that display one image out of N image paths.
    Also comes with a 'Next' and 'Previous' button. Extend me to add
    more functionality (i.e. mouse-related events).
    """
    def __init__(self, parent, *args, **kwargs):
        wx.Panel.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        # self.img := a WxImage
        self.img = None
        # self.imgbitmap := A WxBitmap
        self.imgbitmap = None

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
        self.imgbitmap = img_to_wxbitmap(img, size=size)

        imgsize = self.imgbitmap.GetWidth(), self.imgbitmap.GetHeight()
        self.sizer.Detach(0)
        self.sizer.Add(imgsize)
        self.Layout()
        print 'calling Refresh()...'
        self.Refresh()

    def onPaint(self, evt):
        if self.IsDoubleBuffered():
            dc = wx.PaintDC(self)
        else:
            dc = wx.BufferedPaintDC(self)
        if self.imgbitmap:
            dc.DrawBitmap(self.imgbitmap, 0, 0)

        return dc

    def onLeftDown(self, evt):
        #x, y = self.CalcUnscrolledPosition(evt.GetPositionTuple())
        x, y = evt.GetPositionTuple()
        print "Left Down, at:", (x,y)

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

        # Vars to keep track of box-being-created
        self.isCreate = False
        self.box_create = None

        self.mode_m = BoxDrawPanel.M_CREATE

    def _setup_evts(self):
        ImagePanel._setup_evts(self)

    def set_mode_m(self, mode):
        """ Sets my MouseMode. """
        self.mode_m = mode

    def set_boxes(self, boxes):
        self.boxes = boxes

    def startBox(self, x, y, boxtype=None):
        """ Starts creating a box at (x,y). """
        if boxtype == None:
            boxtype = Box
        print "...Creating Box: {0}, {1}".format((x,y), boxtype)
        self.isCreate = True
        self.box_create = boxtype(x, y, x+1, y+1)
    def finishBox(self, x, y):
        """ Finishes box creation at (x,y). """
        print "...Finished Creating Box:", (x,y)
        self.isCreate = False
        # 0.) Canonicalize box coords s.t. order is: UpperLeft, LowerRight.
        self.box_create.canonicalize()
        toreturn = self.box_create
        self.box_create = None
        return toreturn

    def onLeftDown(self, evt):
        x, y = evt.GetPositionTuple()
        if self.mode_m == BoxDrawPanel.M_CREATE:
            print "...Creating Target box."
            self.startBox(x, y, TargetBox)
        elif self.mode_m == BoxDrawPanel.M_IDLE:
            print "...Creating Selection box."
            self.startBox(x, y, SelectionBox)

    def onLeftUp(self, evt):
        x, y = evt.GetPositionTuple()
        if self.mode_m == BoxDrawPanel.M_CREATE and self.isCreate:
            box = self.finishBox(x, y)
            self.boxes.append(box)
            self.Refresh()
        elif self.mode_m == BoxDrawPanel.M_IDLE and self.isCreate:
            box = self.finishBox(x, y)
            boxes = get_boxes_within(self.boxes, box)
            print "...Selecting {0} boxes.".format(len(boxes))
            for box in boxes:
                box.is_selected = True
            self.Refresh()

    def onMotion(self, evt):
        x, y = evt.GetPositionTuple()
        if self.isCreate:
            self.box_create.x2, self.box_create.y2 = x, y
            self.Refresh()

    def onPaint(self, evt):
        dc = ImagePanel.onPaint(self, evt)
        self.drawBoxes(self.boxes, dc)
        if self.isCreate:
            # Draw Box-Being-Created
            can_box = self.box_create.copy().canonicalize()
            self.drawBox(can_box, dc)
        return dc
        
    def drawBoxes(self, boxes, dc):
        for box in self.boxes:
            self.drawBox(box, dc)

    def drawBox(self, box, dc):
        """ Draws BOX onto DC.
        Input:
            list box: (x1, y1, x2, y2)
            wxDC DC:
        """
        dc.SetBrush(wx.TRANSPARENT_BRUSH)
        drawops = box.get_draw_opts()
        dc.SetPen(wx.Pen(*drawops))
        w = int(abs(box.x2 - box.x1))
        h = int(abs(box.y2 - box.y1))
        dc.DrawRectangle(box.x1, box.y1, w, h)

class TemplateMatchDrawPanel(BoxDrawPanel):
    """ Like a BoxDrawPanel, but when you create a Target box, it runs
    Template Matching to try to find similar instances.
    """
    def __init__(self, parent, tempmatch_fn, *args, **kwargs):
        BoxDrawPanel.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.tempmatch_fn = tempmatch_fn

    def onLeftUp(self, evt):
        x, y = evt.GetPositionTuple()
        if self.mode_m == BoxDrawPanel.M_CREATE and self.isCreate:
            box = self.finishBox(x, y)
            imgpil = util_gui.imageToPil(self.img)
            pdb.set_trace()
            self.tempmatch_fn(box, self.img)
            self.Refresh()
        else:
            BoxDrawPanel.onLeftUp(self, evt)

class Box(object):
    def __init__(self, x1, y1, x2, y2):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
    def __str__(self):
        return "Box({0},{1},{2},{3})".format(self.x1, self.y1, self.x2, self.y2)
    def __repr__(self):
        return "Box({0},{1},{2},{3})".format(self.x1, self.y1, self.x2, self.y2)
    def __eq__(self, o):
        return (isinstance(o, Box) and self.x1 == o.x1 and self.x2 == o.x2
                and self.y1 == o.y1 and self.y2 == o.y2)
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
    def copy(self):
        return Box(self.x1, self.y1, self.x2, self.y2)
class TargetBox(Box):
    def __init__(self, x1, y1, x2, y2, is_sel=False):
        Box.__init__(self, x1, y1, x2, y2)
        self.is_sel = is_sel
    def __str__(self):
        return "TargetBox({0},{1},{2},{3},is_sel={4})".format(self.x1, self.y1, self.x2, self.y2, self.is_sel)
    def __repr__(self):
        return "TargetBox({0},{1},{2},{3},is_sel={4})".format(self.x1, self.y1, self.x2, self.y2, self.is_sel)
    def get_draw_opts(self):
        """ Given the state of me, return the color+line-width for the
        DC to use.
        """
        if self.is_sel:
            return ("Yellow", 3)
        else:
            return ("Green", 3)
    def copy(self):
        return TargetBox(self.x1, self.y1, self.x2, self.y2, is_sel=self.is_sel)
class SelectionBox(Box):
    def __str__(self):
        return "SelectionBox({0},{1},{2},{3})".format(self.x1, self.y1, self.x2, self.y2)
    def __repr__(self):
        return "SelectionBox({0},{1},{2},{3})".format(self.x1, self.y1, self.x2, self.y2)
    def get_draw_opts(self):
        return ("Black", 1)
    def copy(self):
        return SelectionBox(self.x1, self.y1, self.x2, self.y2)

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
    return []

def img_to_wxbitmap(img, size=None):
    """ Converts IMG to a wxBitmap. """
    # TODO: Assumes that IMG is a wxImage
    if size:
        img_scaled = img.Scale(size[0], size[1], quality=wx.IMAGE_QUALITY_HIGH)
    else:
        img_scaled = img
    return wx.BitmapFromImage(img_scaled)

def isimgext(f):
    return os.path.splitext(f)[1].lower() in ('.png', '.bmp', 'jpeg', '.jpg', '.tif')

def main():
    class TestFrame(wx.Frame):
        def __init__(self, parent, imgpaths, *args, **kwargs):
            wx.Frame.__init__(self, parent, size=(600, 400), *args, **kwargs)
            self.parent = parent
            self.imgpaths = imgpaths

            self.st_panel = SelectTargetsPanel(self)
            self.st_panel.start(imgpaths)

    args = sys.argv[1:]
    imgsdir = args[0]
    imgpaths = []
    for dirpath, _, filenames in os.walk(imgsdir):
        for imgname in [f for f in filenames if isimgext(f)]:
            imgpaths.append(os.path.join(dirpath, imgname))
    
    app = wx.App(False)
    f = TestFrame(None, imgpaths)
    f.Show()
    app.MainLoop()

if __name__ == '__main__':
    main()
