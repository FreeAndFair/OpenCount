import sys, os, time, pdb, traceback
try:
    import cPickle as pickle
except ImportError:
    import pickle

sys.path.append('..')

import wx
from wx.lib.scrolledpanel import ScrolledPanel

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

        self.imagepanel = BoxDrawPanel(self)

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
        self.sizer.Add(self.imagepanel, proportion=1, flag=wx.EXPAND)
        self.sizer.Add(btn_sizer, flag=wx.ALIGN_CENTER)

        self.SetSizer(self.sizer)

    def start(self, imgpaths):
        self.imgpaths = imgpaths
        print "len(imgpaths):", len(imgpaths)

        self.display_image(0)

        self.SetupScrolling()

    def display_image(self, idx):
        if idx < 0 or idx >= len(self.imgpaths):
            print "Invalid idx into self.imgpaths:", idx
            pdb.set_trace()
        self.cur_i = idx
        imgpath = self.imgpaths[self.cur_i]
        print "...Displaying image:", imgpath
        wximg = wx.Image(imgpath, wx.BITMAP_TYPE_ANY)
        # 0.) Resize image s.t. width is equal to containing width
        wP, hP = self.parent.GetSize()
        _c = wximg.GetWidth() / float(wP)
        wimg = wP
        himg = int(round(wximg.GetHeight() / _c))
        self.imagepanel.set_image(wximg, size=(wimg, himg))
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
        if prev_idx <= 0:
            return None
        return self.display_image(prev_idx)

    def onButton_next(self, evt):
        self.display_next()

    def onButton_prev(self, evt):
        self.display_prev()

class ImagePanel(wx.Panel):
    """ Basic widget class that display one image out of N image paths.
    Also comes with a 'Next' and 'Previous' button. Extend me to add
    more functionality (i.e. mouse-related events).
    """
    def __init__(self, parent, *args, **kwargs):
        wx.Panel.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        self.img = None
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

        # self.boxes := [(x1, y1, x2, y2), ...]
        self.boxes = []

        # Vars to keep track of box-being-created
        self.isCreate = False
        # (x1,y1) is coords of first mouse click
        self.x1, self.y1 = 0, 0
        # (x2,y2) is coords of second mouse click
        self.x2, self.y2 = 0, 0

        self.mode_m = BoxDrawPanel.M_CREATE

    def _setup_evts(self):
        ImagePanel._setup_evts(self)

    def set_mode_m(self, mode):
        """ Sets my MouseMode. """
        self.mode_m = mode

    def startBox(self, x, y):
        """ Starts creating a box at (x,y). """
        print "...Creating Box:", (x,y)
        self.isCreate = True
        self.x1, self.y1 = x, y

    def finishBox(self, x, y):
        """ Finishes box creation at (x,y). """
        print "...Finished Creating Box:", (x,y)
        self.isCreate = False
        # 0.) Canonicalize box coords s.t. order is: UpperLeft, LowerRight.
        box = canonicalize_box((self.x1, self.y1, self.x2, self.y2))
        return box

    def onLeftDown(self, evt):
        x, y = evt.GetPositionTuple()
        if self.mode_m == BoxDrawPanel.M_CREATE:
            self.startBox(x, y)

    def onLeftUp(self, evt):
        x, y = evt.GetPositionTuple()
        if self.isCreate:
            box = self.finishBox(x, y)
            self.boxes.append(box)
            self.Refresh()
        
    def onMotion(self, evt):
        x, y = evt.GetPositionTuple()
        if self.isCreate:
            self.x2, self.y2 = x, y
            self.Refresh()

    def onPaint(self, evt):
        dc = ImagePanel.onPaint(self, evt)
        self.drawBoxes(self.boxes, dc)
        if self.isCreate:
            # Draw Box-Being-Created
            can_box = canonicalize_box((self.x1, self.y1, self.x2, self.y2))
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
        dc.SetPen(wx.Pen("Green", 3))
        w = int(abs(box[0] - box[2]))
        h = int(abs(box[3] - box[1]))
        dc.DrawRectangle(box[0], box[1], w, h)

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
