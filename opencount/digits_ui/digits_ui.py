import sys, os, pickle, pdb, Queue, threading, time
import wx, cv, scipy, Image
import wx.lib.colourchooser
import wx.lib.scrolledpanel
import wx.lib.inspection
from wx.lib.pubsub import Publisher

import numpy as np
from os.path import join as pathjoin

sys.path.append('..')
import util
from specify_voting_targets import util_gui
from pixel_reg import shared
from grouping import common, verify_overlays

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

    def __eq__(self, o):
        return (o and issubclass(type(o), Box) and
                self.x1 == o.x1 and self.y1 == o.y1 and
                self.x2 == o.x2 and self.y2 == o.y2)

    @staticmethod
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
            return (xa, ya - h, xa, ya + h)
        elif xa > xb and ya < yb:
            # UpperRight, LowerLeft
            return (xa - w, ya, xb + w, yb)
        else:
            # LowerRight, UpperLeft
            return (xb, yb, xa, ya)

    @staticmethod
    def is_overlap(box1, box2):
        """
        Returns True if any part of rect1 is contained within rect2.
        Input:
            rect1: Tuple of (x1,y1,x2,y2)
            rect2: Tuple of (x1,y1,x2,y2)
        """
        def is_within_box(pt, box):
            return box[0] < pt[0] < box[2] and box[1] < pt[1] < box[3]
        x1, y1, x2, y2 = Box.make_canonical(box1)
        w, h = abs(x2-x1), abs(y2-y1)
        # Checks (in order): UL, UR, LR, LL corners
        return (is_within_box((x1,y1), rect1) or
                is_within_box((x1+w,y1), rect1) or 
                is_within_box((x1+w,y1+h), rect1) or 
                is_within_box((x1,y1+h), rect1))
    @staticmethod
    def too_close(box_a, box_b):
        """
        Input:
            box_a, box_b
        """
        dist = util_gui.dist_euclidean
        w, h = box_a.width, box_a.height
        (x1_a, y1_a, x2_a, y2_a) = Box.make_canonical(box_a)
        (x1_b, y1_b, x2_b, y2_b) = Box.make_canonical(box_b)
        return ((abs(x1_a - x1_b) <= w / 2.0 and
                 abs(y1_a - y1_b) <= h / 2.0) or
                is_overlap(box_a, box_b) or 
                is_overlap(box_b, box_a))
        

class DigitLabelPanel(wx.lib.scrolledpanel.ScrolledPanel):
    MAX_WIDTH = 200

    # Per page
    NUM_COLS = 3
    NUM_ROWS = 2

    DIGITTEMPMATCH_JOB_ID = util.GaugeID('DigitTempMatchID')

    # Temp image files that we currently use.
    PATCH_TMP = '_patch_tmp.png'
    REGION_TMP = '_region_tmp.png'

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

        self.gridsizer = wx.GridSizer(rows=DigitLabelPanel.NUM_ROWS, cols=DigitLabelPanel.NUM_COLS)
        self.sizer.Add(self.gridsizer, proportion=1, flag=wx.EXPAND)

        self.bitmapdc = None
        self.i, self.j = 0, 0    # Keeps track of all boxes
        self.i_cur, self.j_cur = 0, 0  # Keeps track of currently
                                       # displayed boxes
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
        
        self.Bind(wx.EVT_SCROLLWIN_TOP, self.onScrollTop)
        self.Bind(wx.EVT_SCROLLWIN_BOTTOM, self.onScrollBottom)
        self.Bind(wx.EVT_SCROLLWIN, self.onScrollAll)

        self.Bind(wx.EVT_SCROLLWIN_LINEUP, self.onScrollUp)
        self.Bind(wx.EVT_SCROLLWIN_LINEDOWN, self.onScrollDown)
        self.Bind(wx.EVT_SCROLLWIN_THUMBTRACK, self.onScrollThumbTrack)
        self.Bind(wx.EVT_SCROLLWIN_THUMBRELEASE, self.onScrollThumbRelease)
        self.Bind(wx.EVT_SCROLL_CHANGED, self.onScrollChanged)

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

    def add_box(self, box):
        assert box not in self.boxes
        self.boxes.append(box)

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
        bitmap.SaveFile(self.PATCH_TMP, wx.BITMAP_TYPE_PNG)
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
            npimg = scipy.ndimage.imread(self.PATCH_TMP, flatten=True)
            # Save self.bitmapdc to a tmp png file. Ugh.
            memdc = wx.MemoryDC()
            w, h = self.bitmapdc.GetSize()
            bitmap = wx.EmptyBitmap(w, h, -1)
            memdc.SelectObject(bitmap)
            memdc.Blit(0, 0, w, h, self.bitmapdc, 0, 0)
            memdc.SelectObject(wx.NullBitmap)
            bitmap.SaveFile(self.REGION_TMP, wx.BITMAP_TYPE_PNG)
            # End ugh.
            #nppatch = scipy.ndimage.imread(ALLPATCHES_TMP, flatten=True)

            # == Now, have user label the selected-digit
            dlg = LabelDigitDialog(self, caption="What digit is this?",
                                   labels=("Digit?:",),
                                   imgpath=self.PATCH_TMP)
            self.Disable()
            retstat = dlg.ShowModal()
            self.Enable()
            if retstat == wx.ID_CANCEL:
                return
            digitval = dlg.results["Digit?:"]
            self.current_digit = digitval

            # == Now, perform template matching across all patches
            self.start_tempmatch(npimg, self.REGION_TMP)
        self.Refresh()

    def start_tempmatch(self, digitimg, allimgpath):
        self.queue = Queue.Queue()
        t = ThreadDoTempMatch(digitimg, allimgpath, self.queue, self.DIGITTEMPMATCH_JOB_ID)

        gauge = util.MyGauge(self, 1, thread=t, ondone=self.on_tempmatchdone,
                             msg="Finding digit instances...",
                             job_id=self.DIGITTEMPMATCH_JOB_ID)
        t.start()
        gauge.Show()
        
    def on_tempmatchdone(self):
        """Called when the template-matching thread is finished. """
        queue = self.queue
        exemplar_img = queue.get()
        matches = queue.get()

        if not matches:
            print "Couldn't find any matches."
            return

        print "Num. Matches Found:", len(matches)            
        tmp_dir = '_tmp_overlays'
        util_gui.create_dirs(tmp_dir)
        self.overlaymaps = {} # maps {int matchID: (i,j)}
        grouplabel = common.make_grouplabel(('digit', self.current_digit))
        examples = []
        for matchID, (filename,score1,score2,Ireg,y1,y2,x1,x2,rszFac) in enumerate(matches):
            coords = map(lambda c:int(round(c*rszFac)),(x1,y1,x2,y2))
            i, j = self.xy2cell(coords[0], coords[1])
            self.overlaymaps[matchID] = (i,j)
            patchpath = os.path.join(tmp_dir, '{0}_match.png'.format(matchID))
            scipy.misc.imsave(patchpath, Ireg)
            examples.append((filename, (grouplabel,), patchpath))
        group = common.GroupClass(examples)
        exemplar_paths = {grouplabel: self.PATCH_TMP}

        # == Now, verify the found-matches via overlay-verification
        verifypanel = verify_overlays.VerifyPanel(self, verify_overlays.VerifyPanel.MODE_YESNO)
        self.Disable()
        verifypanel.start((group,), exemplar_paths, ondone=self.on_verifydone)

    def on_verifydone(self, results):
        """Invoked once the user has finished verifying the template
        matching on the current digit. Add all 'correct' matches to
        our self.boxes.
        """
        def dont_add(newbox):
            for box in self.boxes:
                if Box.too_close(newbox, box):
                    return True
            return False
        self.Enable()
        print "Verifying done."
        for (filename,score1,score2,Ireg,y1,y2,x1,x2,rszFac) in matches:
            coords = map(lambda c:int(round(c*rszFac)), (x1,y1,x2,y2))
            newbox = Box(*(x1, y1, x2, y2))
            if not dont_add(newbox):
                self.add_box(self.boxes)

    def onMotion(self, evt):
        x,y = self.CalcUnscrolledPosition(evt.GetPositionTuple())
        if self._is_box() and evt.LeftIsDown():
            self._update_box(x, y)
            self.Refresh()

    def onSize(self, evt):
        self.Refresh()
        evt.Skip()

    def start(self):
        def compute_dc_size():
            for dirpath, dirnames, filenames in os.walk(self.extracted_dir):
                for imgname in [f for f in filenames if util_gui.is_image_ext(f)]:
                    imgpath = pathjoin(dirpath, imgname)
                    pil_img = util_gui.open_as_grayscale(imgpath)
                    w, h = pil_img.size
                    c = float(w) / self.MAX_WIDTH
                    w_scaled, h_scaled = int(self.MAX_WIDTH), int(round(h / c))
                    if not self.cellh:
                        self.cellh = h_scaled
                    return self.cellw * self.NUM_COLS, self.cellh * self.NUM_ROWS
            return None
        
        #wc, hc = self.GetVirtualSize()
        w, h = compute_dc_size()
        #self.bitmapdc = wx.lib.colourchooser.canvas.BitmapBuffer(wc, hc, wx.BLACK)
        self.bitmapdc = wx.lib.colourchooser.canvas.BitmapBuffer(w, h, wx.BLACK)

        self.setup_grid()
        #self.SetupScrolling(scroll_x=True, scroll_y=True, 
        #                    rate_x=self.cellw, rate_y=self.cellh,
        #                    scrollToTop=True)
        self.SetScrollbars(self.cellw, self.cellh,
                           self.NUM_COLS - 1, self.i)
        self.Refresh()

    def _get_cur_loc(self):
        """Returns (x,y) of next cell location """
        return (self.j_cur * self.cellw, self.i_cur * self.cellh)

    def onScrollUp(self, evt):
        evt.Skip(); return
        orient = 'horizontal' if evt.GetOrientation() == wx.HORIZONTAL else 'vertical'
        if orient == 'horizontal' and self.j_cur < (self.NUM_COLS - 1):
            self.j_cur += 1
            print 'self.j_cur now is:', self.j_cur
        elif orient == 'vertical' and self.i_cur > 0:
            self.i_cur -= 1
            print 'self.i_cur now is:', self.i_cur
        else:
            print "Not doing a ScrollUp."
        self.update_cells()
        evt.Skip()

    def onScrollDown(self, evt):
        evt.Skip(); return
        orient = 'horizontal' if evt.GetOrientation() == wx.HORIZONTAL else 'vertical'
        if orient == 'horizontal' and self.j_cur > 0:
            self.j_cur -= 1
            print 'self.j_cur now is:', self.j_cur
        elif orient == 'vertical' and self.i_cur < (self.NUM_ROWS - 1):
            self.i_cur += 1
            print 'self.i_cur now is:', self.i_cur
        else:
            print "Not doing a ScrollDown."
        pos = evt.GetPosition()
        self.update_cells()
        evt.Skip()
    
    def onScrollThumbTrack(self, evt):
        evt.Skip()
        return
        orient = 'horizontal' if evt.GetOrientation() == wx.HORIZONTAL else 'vertical'
        if orient == 'horizontal':
            self.j_cur = evt.GetPosition()
            print 'self.j_cur now is:', self.j_cur
        elif orient == 'vertical' and self.i_cur > 0:
            self.i_cur = evt.GetPosition()
            print 'self.i_cur now is:', self.i_cur
        self.update_cells()
        evt.Skip()
    def onScrollThumbRelease(self, evt):
        evt.Skip()
        return
        orient = 'horizontal' if evt.GetOrientation() == wx.HORIZONTAL else 'vertical'
        if orient == 'horizontal':
            self.j_cur = evt.GetPosition() - 1
            print 'self.j_cur now is:', self.j_cur
        elif orient == 'vertical' and self.i_cur > 0:
            self.i_cur = evt.GetPosition() - 1
            print 'self.i_cur now is:', self.i_cur
        self.update_cells()
        evt.Skip()

    def onScrollTop(self, evt):
        print "ScrollTop"
        self.i_cur, self.j_cur = 0, 0
        evt.Skip()
    def onScrollBottom(self, evt):
        print "ScrollBottom"
        self.i_cur = self.i, self.j_cur = self.j
        evt.Skip()
    def onScrollAll(self, evt):
        print "ScrollAll", evt.GetOrientation(), evt.GetPosition()
        orient = evt.GetOrientation()
        pos = evt.GetPosition()
        if orient == wx.HORIZONTAL:
            self.j_cur = pos
        else:
            self.i_cur = pos
        self.update_cells()
        evt.Skip()
        self.Refresh()

    def onScrollChanged(self, evt):
        print "on scroll chaaaaaaaannnggeedd"
        orient = 'horizontal' if evt.GetOrientation() == wx.HORIZONTAL else 'vertical'
        pos = evt.GetPosition()
        self.update_cells()
        evt.Skip()

    def update_cells(self):
        """Redraws and updates self.bitmapdc. Might be called if
        the user scrolled.
        """
        def does_imgcell_exist(i, j):
            if i != self.i:
                return True
            else:
                return j < self.j
        def is_draw_blank(i,j):
            if not does_imgcell_exist(i,j):
                return True
            else:
                if i >= self.NUM_ROWS or j >= self.NUM_COLS:
                    return True
                else:
                    return False
        ct = 0
        bitmap_row = 0
        bitmap_col = 0
        i, j = 0, 0
        ii_cur, jj_cur = self.i_cur, self.j_cur
        while i < (self.NUM_ROWS):
            j = 0
            while j < self.NUM_COLS:
                if jj_cur == self.NUM_COLS or not does_imgcell_exist(ii_cur, jj_cur):
                    b = util_gui.make_blank_bitmap((self.cellh, self.cellw), 200)
                    x, y = self.cell2xy(i,j)
                    self.bitmapdc.DrawBitmap(b, x, y)
                    j += 1
                    jj_cur += 1
                    continue
                imgID = self.cell2imgID[(ii_cur, jj_cur)]
                pil_img = util_gui.open_as_grayscale(imgID)
                w, h = pil_img.size
                c = float(w) / self.MAX_WIDTH
                w_scaled, h_scaled = int(self.MAX_WIDTH), int(round(h / c))
                pil_img = pil_img.resize((w_scaled, h_scaled), resample=Image.ANTIALIAS)
                b = util_gui.PilImageToWxBitmap(pil_img)
                x,y = self.cell2xy(i, j)
                self.bitmapdc.DrawBitmap(b, x, y)
                jj_cur += 1
                j += 1
                ct += 1
            jj_cur = self.j_cur
            i += 1
            ii_cur += 1
        print "Should be {0} imgs displayed. (i,j) = {1}, (icur,jcur) = {2}".format(ct, (self.i,self.j),
                                                                                    (self.i_cur, self.j_cur))
        print "bitmapdc.size:", self.bitmapdc.GetSize()
        self.bitmapdc.GetBitmap().ConvertToImage().SaveFile('foobar.png', wx.BITMAP_TYPE_PNG)
        self.Refresh()

    def add_img(self, imgbitmap, imgID):
        """Adds a new image to this grid. """
        #(x, y) = self._get_cur_loc()
        assert imgID not in self.imgID2cell
        assert (self.i, self.j) not in self.cell2imgID
        self.imgID2cell[imgID] = (self.i, self.j)
        self.cell2imgID[(self.i, self.j)] = imgID
        #self.bitmapdc.DrawBitmap(imgbitmap, x, y)
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
        print 'num images:', len(self.imgID2cell)
        self.update_cells()
        self.Refresh()
                
    def onPaint(self, evt):
        """ Refresh screen. """
        if self.IsDoubleBuffered():
            dc = wx.PaintDC(self)
        else:
            dc = wx.BufferedPaintDC(self)
        # You must do PrepareDC in order to force the dc to account
        # for scrolling.
        #self.PrepareDC(dc)
        w, h = dc.GetSize()
        if self.i_cur == None or self.j_cur == None or self.cellw == None or self.cellh == None:
            evt.Skip(); return
        #x, y = self.cell2xy(self.i_cur % self.NUM_ROWS, self.j_cur % self.NUM_COLS)
        #j, i = self.GetViewStart()
        #x, y = self.cell2xy(i
        #print 'drawing from:', x, y
        #print 'viewstart:', self.GetViewStart()
        #dc.Blit(0, 0, w, h, self.bitmapdc, x, y)
        dc.Blit(0, 0, w, h, self.bitmapdc, 0, 0)
        self._draw_boxes(dc)
        evt.Skip()

    def _draw_boxes(self, dc):
        """ Draws boxes """
        dc.SetBrush(wx.TRANSPARENT_BRUSH)
        dc.SetPen(wx.Pen("Green", 2))
        for box in self.boxes:
            x1, y1, x2, y2 = Box.make_canonical(box)
            dc.DrawRectangle(x1, y1, box.width, box.height)
        # Draw box-in-progress
        if self._box:
            dc.SetPen(wx.Pen("Red", 2))
            x1, y1, x2, y2 = Box.make_canonical(self._box)
            dc.DrawRectangle(x1, y1, self._box.width, self._box.height)

class ThreadDoTempMatch(threading.Thread):
    def __init__(self, img1, img2_path, queue, job_id, *args, **kwargs):
        """ Search for img1 within img2. """
        threading.Thread.__init__(self, *args, **kwargs)
        self.img1 = img1
        self.img2_path = img2_path

        self.queue = queue
        self.job_id = job_id
        
    def run(self):
        h, w =  self.img1.shape
        bb = [0, h-1, 0, w-1]
        try:
            matches = shared.find_patch_matchesV1(self.img1, bb, (self.img2_path,))
        except Exception as e:
            print e
            print "ERROR"
        print "DONE with temp matching. Found: {0} matches".format(len(matches))
        self.queue.put(self.img1)
        self.queue.put(matches)
        wx.CallAfter(Publisher().sendMessage, "signals.MyGauge.tick",
                     (self.job_id,))
        wx.CallAfter(Publisher().sendMessage, "signals.MyGauge.done",
                     (self.job_id,))

    def abort(self):
        print "Sorry, abort not implemented yet. :("

class LabelDigitDialog(common.TextInputDialog):
    def __init__(self, parent, imgpath=None, *args, **kwargs):
        common.TextInputDialog.__init__(self, parent, *args, **kwargs)
        img = wx.Image(imgpath, wx.BITMAP_TYPE_PNG) # assume PNG
        staticbitmap = wx.StaticBitmap(self, bitmap=wx.BitmapFromImage(img))
        self.sizer.Insert(1, staticbitmap, proportion=0)
        self.Fit()
        
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

