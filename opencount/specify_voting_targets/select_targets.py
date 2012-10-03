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

        self.imagepanel = ImagePanel(self)


        txt = wx.StaticText(self, label="Select all Voting Targets from \
this partition.")

        
        
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(txt, flag=wx.ALIGN_CENTER)
        self.sizer.Add(self.imagepanel, proportion=1, flag=wx.EXPAND)

        self.SetSizer(self.sizer)

    def start(self, imgpaths):
        self.imgpaths = imgpaths

        print "...Displaying first image:", self.imgpaths[0]
        wximg = wx.Image(self.imgpaths[0], wx.BITMAP_TYPE_ANY)
        self.imagepanel.set_image(wximg)

        self.SetupScrolling()

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
        
    def set_image(self, img):
        """ Updates internal data-structures to allow viewing a new input
        IMG.
        """
        self.img = img
        self.imgbitmap = img_to_wxbitmap(img)

        imgsize = img.GetWidth(), img.GetHeight()
        self.sizer.Add(imgsize)
        self.Layout()

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

def img_to_wxbitmap(img):
    """ Converts IMG to a wxBitmap. """
    # TODO: Assumes that IMG is a wxImage
    return wx.BitmapFromImage(img)

def isimgext(f):
    return os.path.splitext(f)[1].lower() in ('.png', '.bmp', 'jpeg', '.jpg', '.tif')

def main():
    class TestFrame(wx.Frame):
        def __init__(self, parent, imgpaths, *args, **kwargs):
            wx.Frame.__init__(self, parent, *args, **kwargs)
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
