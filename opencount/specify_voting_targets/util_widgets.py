import time, threading, sys, os, math, pdb
import wx
import util_gui
from wx.lib.scrolledpanel import ScrolledPanel
from wx.lib.pubsub import Publisher

"""
A module to store widgets that might be useful in several
places.
"""

class ProgressGauge(wx.Frame):
    """
    A dialog that pops up to display a progress gauge when some
    long-running process is running.
    """
    def __init__(self, parent, numjobs, msg="Please wait...", *args, **kwargs):
        wx.Frame.__init__(self, parent, size=(400, 300), 
                          style=wx.DEFAULT_FRAME_STYLE | wx.FRAME_FLOAT_ON_PARENT, 
                          *args, **kwargs)
        self.parent = parent
        panel = wx.Panel(self)
        
        self.val = 0        
        self.numjobs = numjobs
        
        txt1 = wx.StaticText(panel, label=msg)
        self.gauge = wx.Gauge(panel, range=numjobs, size=(200, 25))
        self.btn_abort = wx.Button(panel, label="Abort")
        self.btn_abort.Bind(wx.EVT_BUTTON, self.onbutton_abort)
        
        panel.sizer = wx.BoxSizer(wx.VERTICAL)
        panel.sizer.Add(txt1)
        panel.sizer.Add(self.gauge)
        panel.sizer.Add(self.btn_abort)
        panel.SetSizer(panel.sizer)
        panel.Fit()
        self.Fit()
        
        Publisher().subscribe(self._pubsub_done, "signals.ProgressGauge.done")
        Publisher().subscribe(self._pubsub_tick, "signals.ProgressGauge.tick")
        
    def _pubsub_done(self, msg):
        self.Destroy()
    def _pubsub_tick(self, msg):
        self.val += 1
        self.gauge.SetValue(self.val)
    
    def onbutton_abort(self, evt):
        print "Abort not implemented yet. Maybe never."
        #self.Destroy()

class MosaicPanel(ScrolledPanel):
    """ A widget that contains both an ImageMosaicPanel, and a simple
    button toolbar that allows pageup/pagedown.
    """
    def __init__(self, parent, *args, **kwargs):
        ScrolledPanel.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.imagemosaic = ImageMosaicPanel(self)
        
        btn_pageup = wx.Button(self, label="Page Up")
        btn_pagedown = wx.Button(self, label="Page Down")
        btn_pageup.Bind(wx.EVT_BUTTON, self.onButton_pageup)
        btn_pagedown.Bind(wx.EVT_BUTTON, self.onButton_pagedown)

        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        btn_sizer.Add(btn_pageup)
        btn_sizer.Add(btn_pagedown)
        
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.imagemosaic, proportion=1, flag=wx.EXPAND)
        sizer.Add(btn_sizer)
        
        self.SetSizer(sizer)
        self.Layout()

        self.Bind(wx.EVT_CHILD_FOCUS, self.OnChildFocus)

    def onButton_pageup(self, evt):
        self.imagemosaic.do_page_up()

    def onButton_pagedown(self, evt):
        self.imagemosaic.do_page_down()

    def set_images(self, imgpaths):
        self.imagemosaic.set_images(imgpaths)
        min_w = self.imagemosaic.cell_width * self.imagemosaic.num_cols
        min_h = self.imagemosaic.cell_height * (self.imagemosaic.num_rows)
        self.SetMinSize((min_w, -1))

        self.SetupScrolling()
        self.Fit()

    def set_boxes(self, boxes_dict):
        """ Given a dict that tells us all boxes for all imgpaths,
        update the self.imagemosaic so that the boxes are correctly
        displayed.
        Input:
            dict boxes_dict: maps {str imgpath: list of (y1, y2, x1, x2)}
        """
        for imgpath, boxes in boxes_dict.iteritems():
            self.imagemosaic.set_boxes(imgpath, boxes)

    def select_image(self, imgpath):
        """ Selects an image within the ImageMosaicPanel. """
        self.imagemosaic.select_img(imgpath)

    def OnChildFocus(self, evt):
        # If I don't override this child focus event, then wx will
        # reset the scrollbars at extremely annoying times. Weird.
        # For inspiration, see:
        #    http://wxpython-users.1045709.n5.nabble.com/ScrolledPanel-mouse-click-resets-scrollbars-td2335368.html
        pass


class ImageMosaicPanel(ScrolledPanel):
    """ A widget that (efficiently) displays images in a grid, organized
    in pages. Assumes that the images are the same size.
    """
    def __init__(self, parent, *args, **kwargs):
        ScrolledPanel.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        self.num_rows = 4
        self.num_cols = 2

        self.cell_width = None    # set by display_page
        self.cell_height = 400

        self.imgpaths = []
        self.cur_page = 0

        self.boxes_dict = {}  # maps {str imgpath: list of (y1,y2,x1,x2)}
        # A 2-D array containing all CellPanels. self.cells[i][j]
        # is the CellPanel at row i, col j.
        self.cells = [[None for _ in range(self.num_cols)] for _ in range(self.num_rows)]
        self.gridsizer = wx.GridSizer(self.num_rows, self.num_cols)

        # Pre-populate the gridsizer with StaticBitmaps
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                cellpanel = CellPanel(self, i, j)
                self.cells[i][j] = cellpanel
                self.gridsizer.Add(cellpanel)
        self.sizer = wx.BoxSizer(wx.VERTICAL)

        self.sizer.Add(self.gridsizer)

        self.SetSizer(self.sizer)

    def do_page_up(self):
        """ Handles necessary logic of turning to the previous page. """
        if self.cur_page <= 0:
            self.cur_page = 0
        else:
            self.cur_page -= 1
            self.display_page(self.cur_page)
    def do_page_down(self):
        """ Handles necessary logic of turning to the next page. """
        total_pages = int(math.ceil(len(self.imgpaths) / float((self.num_rows*self.num_cols))))
        if self.cur_page >= (total_pages - 1):
            self.cur_page = (total_pages - 1)
        else:
            self.cur_page += 1
            self.display_page(self.cur_page)

    def set_images(self, imgpaths):
        """Given a list of image paths, display them."""
        self.imgpaths = imgpaths
        # Reset the boxes_dict for all imgpaths
        self.boxes_dict = {}
        for imgpath in imgpaths:
            self.boxes_dict.setdefault(imgpath, [])
        self.cur_page = 0
        self.display_page(self.cur_page)

    def set_boxes(self, imgpath, boxes):
        """ Updates the list of boxes for imgpath.
        Input:
            str imgpath
            list boxes: [(y1,y2,x1,x2), ...]
        """
        assert imgpath in self.boxes_dict
        self.boxes_dict[imgpath] = list(boxes)
        self.Refresh()

    def display_page(self, pagenum):
        """Sets up UI so that all images on the pagenum are displayed.
        """
        assert self.imgpaths
        start_idx = (self.num_rows * self.num_cols) * pagenum
        assert start_idx < len(self.imgpaths)
        i, j = 0, 0
        for idx in range(start_idx, start_idx + (self.num_rows*self.num_cols)):
            if idx >= len(self.imgpaths):
                # No more images to display, just display empty panels.
                cellpanel = self.cells[i][j]
                dummybitmap = wx.EmptyBitmapRGBA(self.cell_width, self.cell_height,
                                                 red=0, green=0, blue=0)
                cellpanel.set_bitmap(dummybitmap)
                cellpanel.set_txtlabel('No image.')
                cellpanel.imgpath = None
            else:
                imgpath = self.imgpaths[idx]
                img = wx.Image(imgpath, wx.BITMAP_TYPE_PNG) # assume PNG
                if img.GetHeight() != self.cell_height:
                    c = img.GetHeight() / float(self.cell_height)
                    new_w = int(round(img.GetWidth() / c))
                    if self.cell_width == None:
                        self.cell_width = new_w
                    img.Rescale(new_w, self.cell_height, quality=wx.IMAGE_QUALITY_HIGH)
                else:
                    c = 1.0
                cellpanel = self.cells[i][j]
                cellpanel.set_bitmap(wx.BitmapFromImage(img))
                imgname = os.path.split(imgpath)[1]
                parentdir = os.path.split(os.path.split(imgpath)[0])[1]
                cellpanel.set_txtlabel(os.path.join(parentdir, imgname))
                cellpanel.imgpath = imgpath
                cellpanel.cellbitmap.rszFac = c
            j += 1
            if j >= self.num_cols:
                j = 0
                i += 1
        self.SetupScrolling()
        self.Refresh()

    def select_img(self, imgpath):
        """ Selects the cell given by imgpath. """
        Publisher().sendMessage("broadcast.mosaicpanel.mosaic_img_selected", imgpath)

class CellPanel(wx.Panel):
    """ A Panel that contains both a StaticText label (displaying
    the imagepath of the blank ballot) and a CellBitmap (which
    displays the actual blank ballot image).
    """
    def __init__(self, parent, i, j, imgpath=None, bitmap=None, *args, **kwargs):
        wx.Panel.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.i, self.j = i, j
        self.imgpath = imgpath
        self.bitmap = bitmap

        self.cellbitmap = CellBitmap(self, i, j, imgpath, bitmap)
        
        self.txtlabel = wx.StaticText(self, label="Label here.")

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.txtlabel)
        sizer.Add(self.cellbitmap, proportion=1, flag=wx.EXPAND)
        self.SetSizer(sizer)
        self.Fit()

        self.Bind(wx.EVT_LEFT_DOWN, self.onLeftDown)
    def onLeftDown(self, evt):
        self.parent.select_img(self.imgpath)
        
    def set_txtlabel(self, label):
        self.txtlabel.SetLabel(label)
        
    def set_bitmap(self, bitmap):
        self.cellbitmap.set_bitmap(bitmap)

class CellBitmap(wx.Panel):
    """ A panel that displays an image, in addition to displaying a
    list of colored boxes, which could indicate voting targets,
    contests, etc.
    To be used by MosaicPanel.
    """

    def __init__(self, parent, i, j, imgpath, bitmap=None, pil_img=None, rszFac=1.0, *args, **kwargs):
        wx.Panel.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.rszFac = rszFac
        if not bitmap:
            bitmap = wx.EmptyBitmap(50, 50, -1)
        self.bitmap = bitmap
        self.pil_img = pil_img
        self.i, self.j = i, j

        self.SetMinSize(bitmap.GetSize())

        self.Bind(wx.EVT_LEFT_DOWN, self.parent.onLeftDown)
        self.Bind(wx.EVT_PAINT, self.onPaint)

    def set_bitmap(self, bitmap):
        """ Given a wx.Bitmap, update me to display bitmap. """
        self.bitmap = bitmap
        self.SetMinSize(bitmap.GetSize())
        self.Refresh()

    def add_box(self, box):
        assert box not in self.parent.boxes_dict[self.parent.imgpath]
        self.parent.boxes_dict[self.parent.imgpath].append(box)

    def onPaint(self, evt):
        """ Refresh screen. """
        if self.IsDoubleBuffered():
            dc = wx.PaintDC(self)
        else:
            dc = wx.BufferedPaintDC(self)
        dc.DrawBitmap(self.bitmap, 0, 0)
        if self.parent.imgpath == None:
            return
        my_boxes = self.parent.parent.boxes_dict[self.parent.imgpath]
        self._draw_boxes(dc, my_boxes)
        evt.Skip()
        
    def _draw_boxes(self, dc, boxes):
        dc.SetBrush(wx.TRANSPARENT_BRUSH)
        dc.SetPen(wx.Pen("Green", 2))
        for box in boxes:
            if box.is_contest:
                dc.SetPen(wx.Pen("Blue", 2))
            else:
                dc.SetPen(wx.Pen("Red", 2))
            x1, y1, x2, y2 = make_canonical(box)
            x1, y1, x2, y2 = map(lambda n: int(round(n / float(self.rszFac))), (x1,y1,x2,y2))
            w, h = int(abs(x1-x2)), int(abs(y1-y2))
            dc.DrawRectangle(x1, y1, w, h)

def make_canonical(box):
    """ Takes two arbitrary (x,y) points and re-arranges them
    such that we get:
        (x_upperleft, y_upperleft, x_lowerright, y_lowerright)
    """
    xa, ya, xb, yb = box.get_coords()
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
    

class _TestMosaicFrame(wx.Frame):
    """
    Frame to demo the MosaicPanel.
    """
    def __init__(self, parent, imgpaths, *args, **kwargs):
        wx.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.imgpaths = imgpaths

        self.SetSize((500, 500))
        sizer = wx.BoxSizer(wx.VERTICAL)
        self.mosaicpanel = MosaicPanel(self)
        self.mosaicpanel.set_images(imgpaths)
        sizer.Add(self.mosaicpanel, proportion=1, flag=wx.EXPAND)
        self.SetSizer(sizer)

    

class _MainFrame(wx.Frame):
    """
    Frame to demo the ProgressGauge
    """
    def __init__(self, parent, *args, **kwargs):
        wx.Frame.__init__(self, parent, wx.ID_ANY, "title", size=(400, 400), *args, **kwargs)

        btn = wx.Button(self, label="Click to start progress bar demo")
        btn.Bind(wx.EVT_BUTTON, self.onbutton)

    def onbutton(self, evt):
        num_tasks = 10
        progressgauge = ProgressGauge(self, num_tasks, msg="Doing work...")
        progressgauge.Show()
        workthread = _WorkThread(num_tasks)
        workthread.start()
class _WorkThread(threading.Thread):
    def __init__(self, num_tasks, *args, **kwargs):
        threading.Thread.__init__(self, *args, **kwargs)
        self.num_tasks = num_tasks
    def run(self):
        for i in range(self.num_tasks):
            # Do 'work', sending a tick message after every step
            #time.sleep(1.0)
            sum(range(5000000))
            print 'a'
            #Publisher().sendMessage("signals.ProgressGauge.tick")
            wx.CallAfter(Publisher().sendMessage, "signals.ProgressGauge.tick")

        # Notify ProgressGauge that the work is done
        #Publisher().sendMessage("signals.ProgressGauge.done")        
        wx.CallAfter(Publisher().sendMessage, "signals.ProgressGauge.done")

def demo_progressgauge():
    app = wx.App(False)
    frame = _MainFrame(None)
    frame.Show()
    app.MainLoop()

def demo_mosaicpanel(imgsdir):
    imgpaths = []
    for dirpath, dirnames, filenames in os.walk(imgsdir):
        for imgname in [f for f in filenames if util_gui.is_image_ext(f)]:
            imgpaths.append(os.path.join(dirpath, imgname))
    app = wx.App(False)
    frame = _TestMosaicFrame(None, imgpaths)
    frame.Show()
    app.MainLoop()

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 0:
        print "Provide an argument!"
    elif args[0] == 'progressgauge':
        demo_progressgauge()
    elif args[0] == 'mosaicpanel':
        imgsdir = args[1]
        demo_mosaicpanel(imgsdir)
