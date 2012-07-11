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

class DigitMainFrame(wx.Frame):
    """A frame that contains both the DigitLabelPanel, and a simple
    button toolbar.
    """
    def __init__(self, parent, extracted_dir,
                 digit_exemplars_outdir='digit_exemplars',
                 precinctnums_outpath='precinct_nums.txt',
                 ondone=None,
                 *args, **kwargs):
        wx.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        if not ondone:
            ondone = self.on_done

        self.mainpanel = DigitMainPanel(self, extracted_dir,
                                        digit_exemplars_outdir,
                                        precinctnums_outpath,
                                        ondone)
        
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.mainpanel, border=10, proportion=1, flag=wx.EXPAND | wx.ALL)
        self.SetSizer(self.sizer)

    def start(self):
        self.Maximize()
        self.Show()
        self.Fit()
        self.mainpanel.start()

    def on_done(self, results):
        self.Close()

class DigitMainPanel(wx.lib.scrolledpanel.ScrolledPanel):
    """A ScrolledPanel that contains both the DigitLabelPanel, and a
    simple button tool bar.
    """
    def __init__(self, parent, extracted_dir,
                 digit_exemplars_outdir='digit_exemplars',
                 precinctnums_outpath='precinct_nums.txt',
                 ondone=None,
                 *args, **kwargs):
        wx.lib.scrolledpanel.ScrolledPanel.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        self.button_sort = wx.Button(self, label="Sort")
        self.button_sort.Bind(wx.EVT_BUTTON, self.onButton_sort)
        self.button_done = wx.Button(self, label="I'm Done.")
        self.button_done.Bind(wx.EVT_BUTTON, self.onButton_done)

        self.digitpanel = DigitLabelPanel(self, extracted_dir,
                                          digit_exemplars_outdir,
                                          precinctnums_outpath,
                                          ondone)

        sizerbtns = wx.BoxSizer(wx.HORIZONTAL)
        sizerbtns.Add(self.button_sort)
        sizerbtns.Add((20,20))
        sizerbtns.Add(self.button_done)
        
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(sizerbtns, border=10, flag=wx.EXPAND | wx.ALL)
        self.sizer.Add(self.digitpanel, border=10, proportion=1, flag=wx.EXPAND | wx.ALL)
        self.SetSizer(self.sizer)

    def start(self):
        self.Fit()
        self.digitpanel.start()

    def onButton_sort(self, evt):
        self.digitpanel.sort_cells()

    def onButton_done(self, evt):
        self.digitpanel.on_done()

class DigitLabelPanel(wx.lib.scrolledpanel.ScrolledPanel):
    MAX_WIDTH = 200
    # Per page
    NUM_COLS = 4
    NUM_ROWS = 3
    DIGITTEMPMATCH_JOB_ID = util.GaugeID('DigitTempMatchID')
    # Temp image files that we currently use.
    PATCH_TMP = '_patch_tmp.png'
    REGION_TMP = '_region_tmp.png'

    def __init__(self, parent,
                 extracted_dir, 
                 digit_exemplars_outdir='digit_exemplars',
                 precinctnums_outpath='precinct_nums.txt',
                 ondone=None, *args, **kwargs):
        """
        str extracted_dir: Directory containing extracted patches
                           for each blank ballot.
        str digit_exemplars_outdir: Directory in which we'll save each
                                    digit exemplar patch.
        fn ondone: A callback function to call when the digit-labeling is
                   finished. It should accept the results, which is a dict:
                       {str patchpath: str precinct number}
        """
        wx.lib.scrolledpanel.ScrolledPanel.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.extracted_dir = extracted_dir
        self.digit_exemplars_outdir = digit_exemplars_outdir
        self.precinctnums_outpath = precinctnums_outpath
        self.ondone = ondone

        # Keeps track of the currently-being-labeled digit
        self.current_digit = None
        # maps {str regionpath: list of (patchpath, matchID, y1,y2,x1,x2, rszFac)
        self.matches = {}   

        # maps {str regionpath: MyStaticBitmap obj}
        self.cells = {}
        # maps {str regionpath: StaticText txt}
        self.precinct_txts = {}

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.sizer)

        self.gridsizer = wx.GridSizer(rows=DigitLabelPanel.NUM_ROWS, cols=DigitLabelPanel.NUM_COLS)
        self.sizer.Add(self.gridsizer, proportion=1, flag=wx.EXPAND)

        self.cellw, self.cellh = DigitLabelPanel.MAX_WIDTH, None

        self.rszFac = None
        
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

        self.i, self.j = 0, 0    # Keeps track of all boxes
        self.i_cur, self.j_cur = 0, 0  # Keeps track of currently
                                       # displayed boxes
        self.cellw, self.cellh = DigitLabelPanel.MAX_WIDTH, None
        self.imgID2cell = {} # Maps {str imgID: (i,j)}
        self.cell2imgID = {} # Maps {(i,j): str imgID}

        self.boxes = [] 
        self._box = None # A Box that is being created

        self.Bind(wx.EVT_CHILD_FOCUS, self.onChildFocus)

    def onChildFocus(self, evt):
        # If I don't override this child focus event, then wx will
        # reset the scrollbars at extremely annoying times. Weird.
        # For inspiration, see:
        #    http://wxpython-users.1045709.n5.nabble.com/ScrolledPanel-mouse-click-resets-scrollbars-td2335368.html
        pass

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
        s = wx.BoxSizer(wx.VERTICAL)
        txt = wx.StaticText(self, label="Precinct Number:")
        staticbitmap = MyStaticBitmap(self, self.i, self.j, imgpath, bitmap=imgbitmap, pil_img=pil_img, rszFac=rszFac)
        s.Add(txt)
        s.Add(staticbitmap, proportion=1, flag=wx.EXPAND)
        assert imgpath not in self.cells
        assert imgpath not in self.precinct_txts
        self.cells[imgpath] = staticbitmap
        self.precinct_txts[imgpath] = txt
        #self.gridsizer.Add(staticbitmap)
        self.gridsizer.Add(s, border=10, flag=wx.ALL)

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
                if self.cellh == None:
                    self.cellh = h_scaled
                if self.rszFac == None:
                    self.rszFac = c
                pil_img = pil_img.resize((w_scaled, h_scaled), resample=Image.ANTIALIAS)
                b = util_gui.PilImageToWxBitmap(pil_img)
                self.add_img(b, imgpath, pil_img, imgpath, c)
        print 'num images:', len(self.imgID2cell)
        self.Refresh()

    def update_precinct_txt(self, imgpath):
        """ Updates the 'Precinct Num:' StaticText. """
        txt = self.precinct_txts[imgpath]
        cell = self.cells[imgpath]
        txt.SetLabel("Precinct Number: {0}".format(cell.get_digits()))

    def start_tempmatch(self, imgpatch, cell):
        """ The user has selected a digit (imgpatch). Now we want to
        run template matching on all cells. """
        scipy.misc.imsave(self.PATCH_TMP, imgpatch)
        dlg = LabelDigitDialog(self, caption="What digit is this?",
                               labels=("Digit?:",),
                               imgpath=self.PATCH_TMP)
        self.Disable()
        self.disable_cells()
        retstat = dlg.ShowModal()
        self.Enable()
        self.enable_cells()
        if retstat == wx.ID_CANCEL:
            return
        digitval = dlg.results["Digit?:"]
        self.current_digit = digitval

        self.queue = Queue.Queue()
        t = ThreadDoTempMatch(imgpatch, self.extracted_dir, self.queue, self.DIGITTEMPMATCH_JOB_ID)
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

        self.overlaymaps = {} # maps {int matchID: (i,j)}
        grouplabel = common.make_grouplabel(('digit', self.current_digit))
        examples = []
        imgpatch = shared.standardImread(self.PATCH_TMP, flatten=True)
        h, w = imgpatch.shape
        for matchID, (filename,score1,score2,Ireg,y1,y2,x1,x2,rszFac) in enumerate(matches):
            rootdir = os.path.join(self.digit_exemplars_outdir, '{0}_examples'.format(self.current_digit))
            util_gui.create_dirs(rootdir)
            patchpath = os.path.join(rootdir, '{0}_match.png'.format(matchID))
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
        exemplar_paths = {grouplabel: self.PATCH_TMP}

        # == Now, verify the found-matches via overlay-verification
        self.f = VerifyOverlayFrame(self, group, exemplar_paths, self.on_verifydone)
        self.f.Maximize()
        self.Disable()
        self.disable_cells()
        self.f.Show()

    def on_verifydone(self, results):
        """Invoked once the user has finished verifying the template
        matching on the current digit. Add all 'correct' matches to
        the relevant cell's boxes.
        """
        def dont_add(newbox, regionpath):
            for box in self.cells[regionpath].boxes:
                if Box.too_close(newbox, box):
                    return True
            return False
        self.f.Close()
        self.Enable()
        self.enable_cells()
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
        added_matches = 0
        for regionpath, stuff in self.matches.iteritems():
            for (patchpath, matchID, y1, y2, x1, x2, rszFac) in stuff:
                x1, y1, x2, y2 = map(lambda c: int(round((c/rszFac))), (x1,y1,x2,y2))
                # Then, scale it by the resizing done in setup_grid
                x1, y1, x2, y2 = map(lambda c: int(round((c/self.rszFac))), (x1,y1,x2,y2))
                newbox = Box(x1, y1, x2, y2, digit=self.current_digit)
                if not dont_add(newbox, regionpath):
                    added_matches += 1
                    self.add_box(newbox, regionpath)
            self.update_precinct_txt(regionpath)
        print "Added {0} matches.".format(added_matches)

    def sort_cells(self):
        """ Sorts cells by average intensity (most-intense cells
        at the beginning of the grid). This is to allow the UI to
        present only the cells that probably still have digits-to-be-
        labeled.
        """
        pass
                
    def on_done(self):
        """When the user decides that he/she has indeed finished
        labeling all digits. Export the results, such as the
        mapping from precinct-patch to precinct number.
        """
        self.export_precinct_nums(self.precinctnums_outpath)
        result = self.get_patch2precinct()
        if self.ondone:
            self.ondone(result)
        self.Disable()

    def get_patch2precinct(self):
        """ Called by on_done. Computes the result dictionary:
            {str patchpath: str precinct number},
        where patchpath is the path to the precinct patch of
        some blank ballot.
        """
        result = {}
        for patchpath, txt in self.precinct_txts.iteritems():
            assert patchpath not in result
            result[patchpath] = txt.GetLabel()
        return result

    def export_precinct_nums(self, outpath):
        """ Export precinct nums to a specified text file. """
        f = open(outpath, 'w')
        for imgpath, txt in self.precinct_txts.iteritems():
            precinct_num = txt.GetLabel()
            print >>f, precinct_num
        f.close()

    def add_box(self, box, regionpath):
        assert regionpath in self.matches
        assert regionpath in self.cells
        self.cells[regionpath].add_box(box)

    def enable_cells(self):
        for cell in self.cells.values():
            cell.Enable()
    def disable_cells(self):
        for cell in self.cells.values():
            cell.Disable()

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

        self.SetMinSize(bitmap.GetSize())

        self.Bind(wx.EVT_LEFT_DOWN, self.onLeftDown)
        self.Bind(wx.EVT_LEFT_UP, self.onLeftUp)
        self.Bind(wx.EVT_MOTION, self.onMotion)
        self.Bind(wx.EVT_PAINT, self.onPaint)

    def cell2xy(self, i, j):
        return (self.parent.cellw * j, self.parent.cellh * i)

    def get_digits(self):
        """ Returns (in L-R order) the digits of all currently-labeled
        boxes.
        """
        sortedboxes = sorted(self.boxes, key=lambda b: b.x1)
        digits = ''
        for box in sortedboxes:
            if box.digit:
                digits += box.digit
        return digits

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
            npimg_crop = autocrop_img(npimg)
            #scipy.misc.imsave('before_crop.png', npimg)
            #scipy.misc.imsave('after_crop.png', npimg_crop)
            self.parent.start_tempmatch(npimg_crop, self)
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
        bb = [0, h, 0, w]
        regions = []
        #wx.CallAfter(Publisher().sendMessage, "signals.MyGauge.nextjob", (numticks, self.job_id))
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
    def __init__(self, x1, y1, x2, y2, digit=None):
        self.x1, self.y1 = x1, y1
        self.x2, self.y2 = x2, y2
        self.digit = digit

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
                self.x2 == o.x2 and self.y2 == o.y2 and
                self.digit == o.digit)

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
        Returns True if any part of box1 is contained within box2
        Input:
            box1, box2
        """
        def is_within_box(pt, box):
            return box[0] < pt[0] < box[2] and box[1] < pt[1] < box[3]
        x1, y1, x2, y2 = Box.make_canonical(box1)
        rect1 = Box.make_canonical(box2)
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
                Box.is_overlap(box_a, box_b) or 
                Box.is_overlap(box_b, box_a))

class VerifyOverlayFrame(wx.Frame):
    def __init__(self, parent, group, exemplar_paths, ondone):
        wx.Frame.__init__(self, parent)
        self.parent = parent
        self.group = group
        self.exemplar_paths = exemplar_paths
        self.ondone = ondone

        verifypanel = verify_overlays.VerifyPanel(self, verify_overlays.VerifyPanel.MODE_YESNO)
        verifypanel.start((group,), exemplar_paths, ondone=ondone)

def autocrop_img(img):
    """ Given an image, try to find the bounding box. """
    def new_argwhere(a):
        """ Given an array, do what argwhere does but for 255, since
        np.argwhere does it for non-zero values instead.
        """
        b = a.copy()
        for i in range(b.shape[0]):
            for j in range(b.shape[1]):
                val = a[i,j]
                if val == 255:
                    b[i,j] = 0
                else:
                    b[i,j] = 1
        return np.argwhere(b)
    thresholded = util_gui.autothreshold_numpy(img, method='otsu')
    B = new_argwhere(thresholded)
    (ystart, xstart), (ystop, xstop) = B.min(0), B.max(0) + 1
    return img[ystart:ystop, xstart:xstop]

def main():
    args = sys.argv[1:]
    if not args:
        path = 'test_imgs/extracted_precincts'
    else:
        path = args[0]
    app = wx.App(False)
    frame = DigitMainFrame(None, path)
    frame.start()
    app.MainLoop()

if __name__ == '__main__':
    main()

