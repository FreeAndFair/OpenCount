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

        self.cellw, self.cellh = DigitLabelPanel.MAX_WIDTH, None
        
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
        
        w, h = compute_dc_size()

        self.staticbitmaps = []

        self.i, self.j = 0, 0    # Keeps track of all boxes
        self.i_cur, self.j_cur = 0, 0  # Keeps track of currently
                                       # displayed boxes
        self.cellw, self.cellh = DigitLabelPanel.MAX_WIDTH, None
        self.imgID2cell = {} # Maps {str imgID: (i,j)}
        self.cell2imgID = {} # Maps {(i,j): str imgID}

        self.boxes = [] 
        self._box = None # A Box that is being created

    def start(self):
        self.setup_grid()
        self.SetupScrolling(scroll_x=True, scroll_y=True, 
                            rate_x=self.cellw, rate_y=self.cellh,
                            scrollToTop=True)
        self.Refresh()

    def _get_cur_loc(self):
        """Returns (x,y) of next cell location """
        return (self.j_cur * self.cellw, self.i_cur * self.cellh)

    def add_img(self, imgbitmap, imgID, pil_img, imgpath, rszFac):
        """Adds a new image to this grid. """
        #(x, y) = self._get_cur_loc()
        assert imgID not in self.imgID2cell
        assert (self.i, self.j) not in self.cell2imgID
        self.imgID2cell[imgID] = (self.i, self.j)
        self.cell2imgID[(self.i, self.j)] = imgID
        x = self.j * self.cellw
        y = self.i * self.cellh
        if self.j >= (DigitLabelPanel.NUM_COLS - 1):
            self.i += 1
            self.j = 0
        else:
            self.j += 1
        w, h = imgbitmap.GetSize()
        staticbitmap = MyStaticBitmap(self, self.i, self.j, imgpath, bitmap=imgbitmap, pil_img=pil_img, rszFac=rszFac)
        self.staticbitmaps.append(staticbitmap)
        self.gridsizer.Add(staticbitmap)

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
                self.add_img(b, imgpath, pil_img, imgpath, c)
        print 'num images:', len(self.imgID2cell)
        self.Refresh()
                
class MyStaticBitmap(wx.Panel):
    def __init__(self, parent, i, j, imgpath, bitmap=None, pil_img=None, rszFac=1.0, *args, **kwargs):
        wx.Panel.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.imgpath = imgpath
        self.rszFac = rszFac
        if not bitmap:
            bitmap = wx.EmptyBitmap(50, 50, -1)
        self.bitmap = bitmap
        self.pil_img = pil_img
        self.i, self.j = i, j
        self.boxes = []
        self._box = None

        # maps {str regionpath: list of (patchpath, matchID, y1,y2,x1,x2, rszFac)
        self.matches = {}   

        self.SetMinSize(bitmap.GetSize())

        self.Bind(wx.EVT_LEFT_DOWN, self.onLeftDown)
        self.Bind(wx.EVT_LEFT_UP, self.onLeftUp)
        self.Bind(wx.EVT_MOTION, self.onMotion)
        self.Bind(wx.EVT_PAINT, self.onPaint)

    def cell2xy(self, i, j):
        return (self.parent.cellw * j, self.parent.cellh * i)

    def _start_box(self, x, y):
        assert not self._box
        self._box = Box(x, y, x+1, y+1)
    def _finish_box(self, x, y):
        assert self._box
        if self._box.width < 4 or self._box.height < 4:
            self._box = None
            return
        tmp = self._box
        self._box = None
        return tmp
    def _update_box(self, x, y):
        assert self._box
        self._box.x2, self._box.y2 = x, y

    def add_box(self, box):
        assert box not in self.boxes
        self.boxes.append(box)

    def start_tempmatch(self, img, regionsdir):
        self.queue = Queue.Queue()
        t = ThreadDoTempMatch(img, regionsdir, self.queue, self.parent.DIGITTEMPMATCH_JOB_ID)

        gauge = util.MyGauge(self, 1, thread=t, ondone=self.on_tempmatchdone,
                             msg="Finding digit instances...",
                             job_id=self.parent.DIGITTEMPMATCH_JOB_ID)
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
        imgpatch = shared.standardImread(self.parent.PATCH_TMP, flatten=True)
        h, w = imgpatch.shape
        for matchID, (filename,score1,score2,Ireg,y1,y2,x1,x2,rszFac) in enumerate(matches):
            patchpath = os.path.join(tmp_dir, '{0}_match.png'.format(matchID))
            Ireg = np.nan_to_num(Ireg)
            Ireg = shared.fastResize(Ireg, 1 / rszFac)
            if Ireg.shape != (h, w):
                newIreg = np.zeros((h,w))
                newIreg[0:Ireg.shape[0], 0:Ireg.shape[1]] = Ireg
                Ireg = newIreg
            scipy.misc.imsave(patchpath, Ireg)
            examples.append((filename, (grouplabel,), patchpath))
            self.matches.setdefault(filename, []).append((patchpath, matchID, y1, y2, x1, x2, rszFac))
        group = common.GroupClass(examples)
        exemplar_paths = {grouplabel: self.parent.PATCH_TMP}

        # == Now, verify the found-matches via overlay-verification
        self.f = VerifyOverlayFrame(self, group, exemplar_paths, self.on_verifydone)
        self.f.Maximize()
        self.Disable()
        self.parent.Disable()
        self.f.Show()

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
        self.f.Close()
        self.Enable()
        self.parent.Enable()
        # Remove all matches from self.matches that the user said
        # was not relevant, during overlay verification
        for grouplabel, groups in results.iteritems():
            # groups is a list of GroupClasses
            # group[i].elements[j] = (regionpath, rankedlist, patchpath)
            if grouplabel == verify_overlays.VerifyPanel.GROUPLABEL_OTHER:
                # The user said that these elements are not relevant
                for groupclass in groups:
                    assert groupclass.getcurrentgrouplabel() == verify_overlays.VerifyPanel.GROUPLABEL_OTHER
                    for element in groupclass.elements:
                        regionpath, rankedlist, patchpath = element
                        stuff = self.matches[regionpath]
                        # stuff[i] := (patchpath, matchID, y1,y2,x1,x2, rszFac)
                        stuff = [t for t in stuff if t[0] != patchpath]
                        self.matches[regionpath] = stuff

        for regionpath, (patchpath, matchID, y1, y2, x1, x2, rszFac) in self.matches.iteritems():
            # TODO: Move logic to DigitUI, since it has to modify all
            # child MyStaticBitmaps
            x1, y1, x2, y2 = map(lambda c: int(round((c/rszFac))), (x1,y1,x2,y2))
            newbox = Box(*(x1, y1, x2, y2))
            if not dont_add(newbox, regionpath):
                self.add_box(newbox, regionpath)

    def onLeftDown(self, evt):
        x, y = evt.GetPosition()
        self._start_box(x, y)
        self.Refresh()
    def onLeftUp(self, evt):
        x, y = evt.GetPosition()
        box = self._finish_box(x, y)
        if box:
            # do template matching
            npimg = self.extract_region(box)
            scipy.misc.imsave(self.parent.PATCH_TMP, npimg)
            dlg = LabelDigitDialog(self, caption="What digit is this?",
                                   labels=("Digit?:",),
                                   imgpath=self.parent.PATCH_TMP)
            self.Disable()
            self.parent.Disable()
            retstat = dlg.ShowModal()
            self.Enable()
            self.parent.Enable()
            if retstat == wx.ID_CANCEL:
                return
            digitval = dlg.results["Digit?:"]
            self.current_digit = digitval
            # find_patch_matchesV1 currently takes in imgpaths for the
            # region: for now, just save region to a tmp img.

            #self.pil_img.save(self.parent.REGION_TMP)
            self.start_tempmatch(npimg, self.parent.extracted_dir)
        self.Refresh()
    def onMotion(self, evt):
        x, y = evt.GetPosition()
        if self._box and evt.LeftIsDown():
            self._update_box(x, y)
            self.Refresh()

    def extract_region(self, box):
        """Extracts box from the currently-displayed image. """
        coords = Box.make_canonical(box)
        #pilimg = util_gui.WxBitmapToPilImage(self.bitmap)
        npimg = np.array(Image.open(self.imgpath).convert('L'))
        x1,y1,x2,y2=map(lambda n:n*self.rszFac,coords)
        return npimg[y1:y2, x1:x2]

    def onPaint(self, evt):
        """ Refresh screen. """
        if self.IsDoubleBuffered():
            dc = wx.PaintDC(self)
        else:
            dc = wx.BufferedPaintDC(self)
        dc.DrawBitmap(self.bitmap, 0, 0)
        self._draw_boxes(dc)
        evt.Skip()
        
    def _draw_boxes(self, dc):
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
    def __init__(self, img1, regionsdir, queue, job_id, *args, **kwargs):
        """ Search for img1 within images in regionsdir. """
        threading.Thread.__init__(self, *args, **kwargs)
        self.img1 = img1
        self.regionsdir = regionsdir

        self.queue = queue
        self.job_id = job_id
        
    def run(self):
        h, w =  self.img1.shape
        bb = [0, h-1, 0, w-1]
        regions = []
        for dirpath, dirnames, filenames in os.walk(self.regionsdir):
            for imgname in [f for f in filenames if util_gui.is_image_ext(f)]:
                regions.append(pathjoin(dirpath, imgname))
        try:
            matches = shared.find_patch_matchesV1(self.img1, bb, regions, rszFac=1.0, threshold=0.6)
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

class VerifyOverlayFrame(wx.Frame):
    def __init__(self, parent, group, exemplar_paths, ondone):
        wx.Frame.__init__(self, parent)
        self.parent = parent
        self.group = group
        self.exemplar_paths = exemplar_paths
        self.ondone = ondone

        verifypanel = verify_overlays.VerifyPanel(self, verify_overlays.VerifyPanel.MODE_YESNO)
        verifypanel.start((group,), exemplar_paths, ondone=ondone)
        
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

