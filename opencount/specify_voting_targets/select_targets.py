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

        self.imagepanel = ImagePanel(self)

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
    def __init__(self, parent, *args, **kwargs):
        wx.Panel.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        self.img = None
        self.imgbitmap = None

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.sizer)

        self.Bind(wx.EVT_PAINT, self.onPaint)
        self.Bind(wx.EVT_LEFT_DOWN, self.onLeftDown)
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
