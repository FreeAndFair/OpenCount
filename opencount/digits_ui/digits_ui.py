import sys, os, pickle, pdb
import wx, cv, scipy, Image
import wx.lib.colourchooser
import wx.lib.scrolledpanel
import numpy as np
from os.path import join as pathjoin

sys.path.append('..')
from specify_voting_targets import util_gui
from pixel_reg import shared

"""
Assumes extracted_dir looks like:
    <projdir>/extracted_attrs/precinct/*.png
Where each *.png is the result of encodepath'ing a blank ballot
id.
"""

class Box(object):
    def __init__(self, x1, y1, x2, y2):
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2

    @property
    def width(self):
        return abs(self.x1 - self.x2)
    @property
    def height(self):
        return abs(self.y1 - self.y2)
    
    def set_coords(self, pts):
        self.x1, self.y1, self.x2, self.y2 = pts
    def get_coords(self):
        return self.x1, self.y1, self.x2, self.y2

class DigitLabelPanel(wx.lib.scrolledpanel.ScrolledPanel):
    MAX_WIDTH = 200
    NUM_COLS = 4

    def __init__(self, parent, extracted_dir, *args, **kwargs):
        """
        str extracted_dir: Directory containing extracted patches
                           for each blank ballot.
        """
        wx.lib.scrolledpanel.ScrolledPanel.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.extracted_dir = extracted_dir

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.sizer)

        self.gridsizer = wx.GridSizer(rows=0, cols=DigitLabelPanel.NUM_COLS)
        self.sizer.Add(self.gridsizer, proportion=1, flag=wx.EXPAND)

        self.bitmapdc = None
        self.i, self.j = 0, 0
        self.cellw, self.cellh = DigitLabelPanel.MAX_WIDTH, None
        self.imgID2cell = {} # Maps {str imgID: (i,j)}
        self.cell2imgID = {} # Maps {(i,j): str imgID}

        self.boxes = [] 
        self._box = None # A Box that is being created

        self.Bind(wx.EVT_PAINT, self.onPaint)
        self.Bind(wx.EVT_LEFT_DOWN, self.onLeftDown)
        self.Bind(wx.EVT_LEFT_UP, self.onLeftUp)
        self.Bind(wx.EVT_MOTION, self.onMotion)
        self.Bind(wx.EVT_SIZE, self.onSize)

    """ Box-Creation methods """
    def _start_box(self, x, y):
        """ Start creating a boundingbox at (x,y) """
        assert not self._box
        self._box = Box(x, y, x+1, y+1)
    def _finish_box(self, x, y):
        """ Finish creating the self._box Box. """
        assert self._box
        if self._box.width < 4 or self._box.height < 4:
            self._box = None
            return None
        tmp = self._box
        self._box = None
        return tmp
    def _is_box(self):
        """ Returns True if I'm in the middle of creating a box """
        return self._box != None
    def _update_box(self, x, y):
        """ Updates box to have new coordinates """
        assert self._box
        self._box.x2, self._box.y2 = x, y

    def xy2cell(self, x, y):
        """Transforms (x,y) coord to (i,j) cell index."""
        return (int(y / self.cellh), int(x / self.cellw))
    def cell2xy(self, i, j):
        """Transforms (i,j) cell indices to (x,y) coords """
        return (j * self.cellw, i * self.cellh)

    def onLeftDown(self, evt):
        x,y = self.CalcUnscrolledPosition(evt.GetPositionTuple())
        i,j = self.xy2cell(x,y)
        #print "Looking at imgpath:", self.cell2imgID[(i,j)]
        self._start_box(x, y)
        self.Refresh()
        
    def extract_patch(self, x1, y1, x2, y2):
        memory = wx.MemoryDC()
        w, h = abs(x2-x1), abs(y2-y1)
        bitmap = wx.EmptyBitmap(w, h, -1)
        memory.SelectObject(bitmap)
        memory.Blit(0, 0, w, h, self.bitmapdc, x1, y1)
        memory.SelectObject(wx.NullBitmap)
        # Having trouble converting the damn bitmap to a numpy
        # array. Wtf. I'll just save it to a temp file, argh.
        bitmap.SaveFile('patch_tmp.png', wx.BITMAP_TYPE_PNG)
        '''
        bitmap.SaveFile('foo.png', wx.BITMAP_TYPE_PNG)
        foo = np.array((0,)*(w*h))
        bitmap.CopyToBuffer(foo)
        scipy.misc.imsave('fooarray.png', foo)
        '''

    def onLeftUp(self, evt):
        x,y = self.CalcUnscrolledPosition(evt.GetPositionTuple())
        box = self._finish_box(x,y)
        if box:
            #npimg = self.extract_patch(box.x1, box.y1, box.x2, box.y2)
            self.extract_patch(box.x1, box.y1, box.x2, box.y2)
            npimg = scipy.ndimage.imread('patch_tmp.png', flatten=True)
            
            '''
            x_img, y_img = int(x / self.cellw), int(y / self.cellh)
            w, h = box.width, box.height
            
            im = scipy.misc.imread(
            boxesmap = do_template_match(box, self.
            '''
        self.Refresh()

    def onMotion(self, evt):
        x,y = self.CalcUnscrolledPosition(evt.GetPositionTuple())
        if self._is_box() and evt.LeftIsDown():
            self._update_box(x, y)
            self.Refresh()

    def onSize(self, evt):
        self.Refresh()
        evt.Skip()

    def start(self):
        wc, hc = self.GetClientSize()
        print 'wc, hc:', wc, hc
        self.bitmapdc = wx.lib.colourchooser.canvas.BitmapBuffer(wc, hc, wx.BLACK)

        self.setup_grid()
        self.SetVirtualSize((1000, 1000))
        self.SetupScrolling()

    def _get_cur_loc(self):
        """Returns (x,y) of next cell location """
        return (self.j * self.cellw, self.i * self.cellh)

    def add_img(self, imgbitmap, imgID):
        """Adds a new image to this grid. """
        (x, y) = self._get_cur_loc()
        assert imgID not in self.imgID2cell
        assert (self.i, self.j) not in self.cell2imgID
        self.imgID2cell[imgID] = (self.i, self.j)
        self.cell2imgID[(self.i, self.j)] = imgID
        self.bitmapdc.DrawBitmap(imgbitmap, x, y)
        if self.j >= (DigitLabelPanel.NUM_COLS - 1):
            self.i += 1
            self.j = 0
        else:
            self.j += 1
        w, h = imgbitmap.GetSize()
        self.gridsizer.Add((w,h))

    def setup_grid(self):
        """Reads in the digit patches (given by self.extracted_dir),
        and displays them on a grid.
        """
        for dirpath, dirnames, filenames in os.walk(self.extracted_dir):
            for imgname in [f for f in filenames if util_gui.is_image_ext(f)]:
                imgpath = pathjoin(dirpath, imgname)
                pil_img = util_gui.open_as_grayscale(imgpath)
                w, h = pil_img.size
                c = float(w) / self.MAX_WIDTH
                w_scaled, h_scaled = int(self.MAX_WIDTH), int(round(h / c))
                if not self.cellh:
                    self.cellh = h_scaled
                pil_img = pil_img.resize((w_scaled, h_scaled), resample=Image.ANTIALIAS)
                b = util_gui.PilImageToWxBitmap(pil_img)
                self.add_img(b, imgpath)
        self.Refresh()
                
    def onPaint(self, evt):
        """ Refresh screen. """
        if self.IsDoubleBuffered():
            dc = wx.PaintDC(self)
        else:
            dc = wx.BufferedPaintDC(self)
        # You must do PrepareDC in order to force the dc to account
        # for scrolling.
        self.PrepareDC(dc)
        w, h = dc.GetSize()
        dc.Blit(0, 0, w, h, self.bitmapdc, 0, 0)
        self._draw_boxes(dc)
        evt.Skip()

    def _draw_boxes(self, dc):
        """ Draws boxes """
        def make_canonical(xa, ya, xb, yb):
            """ Takes two arbitrary (x,y) points and re-arranges them
            such that we get:
                (x_upperleft, y_upperleft, x_lowerright, y_lowerright)
            """
            w, h = abs(xa - xb), abs(ya - yb)
            if xa < xb and ya < yb:
                # UpperLeft, LowerRight
                return (xa, ya, xb, yb)
            elif xa < xb and ya > yb:
                # LowerLeft, UpperRight
                return (xa, ya - h, xa, ya + h)
            elif xa > xb and ya < yb:
                # UpperRight, LowerLeft
                return (xa - w, ya, xb + w, yb)
            else:
                # LowerRight, UpperLeft
                return (xb, yb, xa, ya)
        dc.SetBrush(wx.TRANSPARENT_BRUSH)
        dc.SetPen(wx.Pen("Green", 2))
        for box in self.boxes:
            x1, y1, x2, y2 = make_canonical(*box.get_coords())
            dc.DrawRectangle(x1, y1, box.width, box.height)
        # Draw box-in-progress
        if self._box:
            dc.SetPen(wx.Pen("Red", 2))
            x1, y1, x2, y2 = make_canonical(*self._box.get_coords())
            dc.DrawRectangle(x1, y1, self._box.width, self._box.height)
        
class TestFrame(wx.Frame):
    def __init__(self, parent, extracted_dir, *args, **kwargs):
        wx.Frame.__init__(self, parent, *args, **kwargs)
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.panel = DigitLabelPanel(self, extracted_dir)
        self.button = wx.Button(self, label="Click me")
        self.button.Bind(wx.EVT_BUTTON, self.onbutton)
        self.sizer.Add(self.panel, proportion=1, flag=wx.EXPAND)
        self.sizer.Add(self.button, proportion=0)
        self.SetSizer(self.sizer)
        self.Show()

    def onbutton(self, evt):
        self.panel.start()

def main():
    app = wx.App(False)
    frame = TestFrame(None, 'test_imgs/extracted_precincts')
    app.MainLoop()

if __name__ == '__main__':
    main()

