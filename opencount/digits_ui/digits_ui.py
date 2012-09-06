import sys, os, pdb, Queue, threading, time, traceback
try:
    import cPickle as pickle
except ImportError as e:
    import pickle

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
from grouping import common, verify_overlays, partask

"""
Assumes extracted_dir looks like:
    <projdir>/extracted_attrs/precinct/*.png
Where each *.png is the result of encodepath'ing a blank ballot
id.
"""        

"""
Output files:

<projdir>/digitpatch2temp.p

This keeps track of the mapping from digit patch imgpaths to the 
associated blank ballot imgpath:
    {str digitpatchpath: (str templatepath, attrstr, bb, int side)}

<projdir>/digitattrvals_blanks.p

This keeps track of the precinct numbers of all blank ballots:
    {str blankimgpath: {digitattrtype: (str digitval, bb, int side)}}
Note that this is blankimgpath, not blankid.

"""

# Used to give a unique ID to all Digit-template-match matches.
matchID = 0

def get_last_matchID(imgsdir):
    """ Given the directory of saved imgmatches, return the last
    matchID. imgsdir would be like:
        <projdir>/digit_exemplars/<i>_examples/*
    Assumes that match paths are of the form:
        <matchID>_match.png
    i.e.:
        0_match.png
        1_match.png
        ...
    """
    i = 0
    for dirpath, dirnames, filenames in os.walk(imgsdir):
        for imgname in [f for f in filenames if util_gui.is_image_ext(f)]:
            curidx = int(imgname.split("_")[0])
            if curidx > i:
                i = curidx
    return i

class LabelDigitsPanel(wx.lib.scrolledpanel.ScrolledPanel):
    """ A wrapper-class of DigitMainPanel that is meant to be
    integrated into OpenCount itself.
    """
    def __init__(self, parent, *args, **kwargs):
        wx.lib.scrolledpanel.ScrolledPanel.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.project = None

    def start(self, project):
        """ First, extract all digit-patches into 
        self.project.extracted_digitpatch_dir. Then, run the 
        Digit-Labeling UI.
        """
        self.project = project
        extracted_digitpatches_fulldir = pathjoin(project.projdir_path,
                                                  project.extracted_digitpatch_dir)
        digit_ex_fulldir = pathjoin(project.projdir_path, project.digit_exemplars_outdir)
        precinctnums_fullpath = pathjoin(project.projdir_path, project.precinctnums_outpath)
        _t = time.time()
        print "Extracting Digit Patches..."
        patch2temp = do_extract_digitbased_patches(self.project)
        print "...Finished Extracting Digit Patches ({0} s).".format(time.time() - _t)
        pickle.dump(patch2temp, open(pathjoin(project.projdir_path,
                                              project.digitpatch2temp),
                                     'wb'))
        self.mainpanel = DigitMainPanel(self, extracted_digitpatches_fulldir,
                                        digit_ex_fulldir,
                                        precinctnums_fullpath,
                                        ondone=self.ondone)
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.mainpanel, proportion=1, flag=wx.EXPAND)
        self.SetSizer(self.sizer)
        #self.Fit()

        statefile = pathjoin(self.project.projdir_path,
                             self.project.labeldigitstate)
        self.mainpanel.start(statefile=statefile)
        self.project.addCloseEvent(lambda: self.mainpanel.digitpanel.save_session(statefile=statefile))
        self.Bind(wx.EVT_SIZE, self.onSize)
        self.SetupScrolling()

    def onSize(self, evt):
        self.SetupScrolling()

    def export_results(self):
        self.mainpanel.export_results()
        self.mainpanel.digitpanel.compute_and_save_digitexemplars_map()

    def ondone(self, results):
        """ Called when the user is finished labeling digit-based
        attributes.
        Input:
            dict results: maps {str patchpath: str precinct number}
        """
        print "Done Labeling Digit-Based Attributes"

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
        btn_zoomin = wx.Button(self, label="Zoom In.")
        btn_zoomin.Bind(wx.EVT_BUTTON, self.onButton_zoomin)
        btn_zoomout = wx.Button(self, label="Zoom Out.")
        btn_zoomout.Bind(wx.EVT_BUTTON, self.onButton_zoomout)

        self.digitpanel = DigitLabelPanel(self, extracted_dir,
                                          digit_exemplars_outdir,
                                          precinctnums_outpath,
                                          ondone)

        sizerbtns = wx.BoxSizer(wx.HORIZONTAL)
        sizerbtns.Add(self.button_sort)
        sizerbtns.Add((20,20))
        sizerbtns.Add(self.button_done)
        sizerbtns.Add((20,20))
        sizerbtns.Add(btn_zoomin)
        sizerbtns.Add((20,20))
        sizerbtns.Add(btn_zoomout)
        
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(sizerbtns, border=10, flag=wx.EXPAND | wx.ALL)
        self.sizer.Add(self.digitpanel, border=10, proportion=1, flag=wx.EXPAND | wx.ALL)
        self.SetSizer(self.sizer)

        self.SetClientSize(self.parent.GetClientSize())
        self.SetupScrolling()
        self.Bind(wx.EVT_SIZE, self.onSize)
    
    def onSize(self, evt):
        self.SetupScrolling()

    def start(self, statefile=None):
        if not self.digitpanel.restore_session(statefile=statefile):
            self.digitpanel.start()

    def export_results(self):
        self.digitpanel.export_results()

    def onButton_sort(self, evt):
        self.digitpanel.sort_cells()

    def onButton_done(self, evt):
        self.digitpanel.on_done()

    def onButton_zoomin(self, evt):
        dlg = wx.MessageDialog(self, message="Not implemented yet.")
        self.Disable()
        dlg.ShowModal()
        self.Enable()

    def onButton_zoomout(self, evt):
        dlg = wx.MessageDialog(self, message="Not implemented yet.")
        self.Disable()
        dlg.ShowModal()
        self.Enable()

class DigitLabelPanel(wx.lib.scrolledpanel.ScrolledPanel):
    MAX_WIDTH = 200
    # Per page
    NUM_COLS = 4
    NUM_ROWS = 3
    DIGITTEMPMATCH_JOB_ID = util.GaugeID('DigitTempMatchID')
    # Temp image files that we currently use.
    PATCH_TMP = '_patch_tmp.png'
    REGION_TMP = '_region_tmp.png'

    STATE_FILE = '_digitlabelstate.p'

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
        self.precinctnums_outpath = precinctnums_outpath  # TODO: NOT USED

        self.ondone = ondone

        # Keeps track of the currently-being-labeled digit
        self.current_digit = None
        # maps {str regionpath: list of (patchpath, matchID, digit, score, y1,y2,x1,x2, rszFac)
        self.matches = {}   

        # maps {str regionpath: MyStaticBitmap obj}
        self.cells = {}
        # maps {str regionpath: StaticText txt}
        self.precinct_txts = {}

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.sizer)

        self.gridsizer = wx.GridSizer(rows=DigitLabelPanel.NUM_ROWS, cols=DigitLabelPanel.NUM_COLS)
        #self.sizer.Add(self.gridsizer, proportion=1, flag=wx.EXPAND)
        self.sizer.Add(self.gridsizer)

        self.cellw, self.cellh = DigitLabelPanel.MAX_WIDTH, None

        self.rszFac = None
        
        self.i, self.j = 0, 0    # Keeps track of all boxes
        self.i_cur, self.j_cur = 0, 0  # Keeps track of currently
                                       # displayed boxes

        self.imgID2cell = {} # Maps {str imgID: (i,j)}
        self.cell2imgID = {} # Maps {(i,j): str imgID}

        self._box = None # A Box that is being created

        self.Bind(wx.EVT_CHILD_FOCUS, self.onChildFocus)
        self.Bind(wx.EVT_SIZE, self.onSize)

    def onSize(self, evt):
        self.SetupScrolling()

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

    def restore_session(self, statefile=None):
        """ Tries to restore the state from a previous session If it
        can't restore the state, then returns False. If this happens,
        you should call set.start(). Otherwise, returns True.
        """
        if statefile == None:
            statefile = DigitLabelPanel.STATE_FILE
        if not os.path.exists(statefile):
            return False

        state = pickle.load(open(statefile, 'rb'))
        self.matches = state['matches']
        cell_boxes = state['cell_boxes']
        digits = state['digits']
        self.start()
        for regionpath, digits_str in digits.iteritems():
            i, j = self.imgID2cell[regionpath]
            k = (self.NUM_COLS * i) + j
            self.precinct_txts[regionpath].SetLabel("{0}: Precinct Number: {1}".format(str(k),
                                                                                       digits_str))
        #for regionpath, boxes in cell_boxes.iteritems():
        #    self.cells[regionpath].boxes = boxes
        # For awhile, a bug happened where self.matches could become 
        # out-of-sync with self.cells. This code will ensure that they
        # stay synced.
        def get_digit(patchpath):
            """ patchpaths are of the form:
                <projdir>/digit_exemplars/0_examples/*.png
            """
            return os.path.split(os.path.split(patchpath)[0])[1].split("_")[0]
        for regionpath, matches in self.matches.iteritems():
            boxes = []
            for (patchpath, matchID, digit, score, y1,y2,x1,x2,rszFac) in matches:
                x1, y1, x2, y2 = map(lambda c: int(round((c/rszFac))), (x1,y1,x2,y2))
                # Then, scale it by the resizing done in setup_grid
                x1, y1, x2, y2 = map(lambda c: int(round((c/self.rszFac))), (x1,y1,x2,y2))
                digit = get_digit(patchpath)
                box = Box(x1,y1,x2,y2,digit=digit)
                boxes.append(box)
            self.cells[regionpath].boxes = boxes
        return True

    def save_session(self, statefile=None):
        """ Saves the current session state. """
        if statefile == None:
            statefile = DigitLabelPanel.STATE_FILE
        f = open(statefile, 'wb')
        state = {}
        state['matches'] = self.matches
        cell_boxes = {}
        digits = {} # maps regionpath to digits
        for regionpath, cell in self.cells.iteritems():
            cell_boxes[regionpath] = cell.boxes
            digits[regionpath] = cell.get_digits()
        state['cell_boxes'] = cell_boxes
        state['digits'] = digits
        pickle.dump(state, f)
        f.close()

    def _get_cur_loc(self):
        """Returns (x,y) of next cell location """
        return (self.j_cur * self.cellw, self.i_cur * self.cellh)

    def add_img(self, imgbitmap, imgID, pil_img, imgpath, rszFac):
        """Adds a new image to this grid. Populates the self.imgID2cell,
        self.cell2imgID, self.i, self.j, self.cells, and 
        self.precinct_txts instance variables.
        """
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
        w, h = self.GetClientSize()
        w_suggested = int(round(w / self.NUM_COLS))
        self.MAX_WIDTH = w_suggested
        self.cell_w = w_suggested

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
        i, j = self.imgID2cell[imgpath]
        k = (self.NUM_COLS * i) + j
        txt.SetLabel("{0} Precinct Number: {1}".format(str(k), cell.get_digits()))

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
        if digitval == None or digitval == '':
            d = wx.MessageDialog(self, message="You must enter in the \
digit.")
            self.Disable(); self.disable_cells()
            d.ShowModal()
            self.Enable(); self.enable_cells()
            return
        self.current_digit = digitval

        self.queue = Queue.Queue()
        t = ThreadDoTempMatch(imgpatch, self.extracted_dir, self.queue, self.DIGITTEMPMATCH_JOB_ID)
        gauge = util.MyGauge(self, 1, thread=t, ondone=self.on_tempmatchdone,
                             msg="Finding digit instances...",
                             job_id=self.DIGITTEMPMATCH_JOB_ID)
        t.start()
        gauge.Show()

    def on_tempmatchdone(self):
        """Called when the template-matching thread is finished. 
        TODO: This makes the assumption that a GroupClass representing
        a digit-based attribute will have a grouplabel whose kv-pairs
        has a key 'digit', and its value is a digit ('0','1',etc.).
        This seems to unnecessarily restrict the architecture to only
        allowing one digit-based attribute in an election.
        """
        queue = self.queue
        exemplar_img = queue.get()
        matches = queue.get()

        matches_prune = prune_matches(matches, self.matches)
        # TODO: Doesn't prune away close-matches within the results of
        #       template matching.

        print "Number of matches pruned: {0}".format(len(matches) - len(matches_prune))
        matches = matches_prune
        
        if not matches:
            print "Couldn't find any matches."
            return
        print "Num. Matches Found:", len(matches)

        self.overlaymaps = {} # maps {int matchID: (i,j)}
        grouplabel = common.make_grouplabel(('digit', self.current_digit))
        # 0.) If we are seeing this digit for the first time, this will
        # not be present in grouplabel_record, so add it in.
        grouplabel_record = common.load_grouplabel_record(self.project)
        try:
            gl_idx = grouplabel_record.index(grouplabel)
        except:
            print "Discovering digit {0} for the first time:", grouplabel
            grouplabel_record.append(grouplabel)
            common.save_grouplabel_record(self.project, grouplabel_record)
            gl_idx = len(grouplabel) - 1
        examples = []
        imgpatch = shared.standardImread(self.PATCH_TMP, flatten=True)
        h, w = imgpatch.shape
        # patchpath_scores will be used to improve 'Split' behavior
        # for digit-based attributes. TODO: NOT IN USE, replaced by kmeans
        proj = self.parent.parent.project  # TODO: breach of abstraction
        patchpath_scoresP = pathjoin(proj.projdir_path, proj.digitpatchpath_scoresBlank)
        # patchpath_scores maps {str patchpath: float score}
        if os.path.exists(patchpath_scoresP):
            patchpath_scores = pickle.load(open(patchpath_scoresP, 'rb'))
        else:
            patchpath_scores = {}
        global matchID
        matchID = get_last_matchID(self.digit_exemplars_outdir)
        # regionpath is an attrpatch, not the blank ballot itself
        for (regionpath,score1,score2,Ireg,y1,y2,x1,x2,rszFac) in matches:
            rootdir = os.path.join(self.digit_exemplars_outdir, '{0}_examples'.format(self.current_digit))
            util_gui.create_dirs(rootdir)
            patchpath = os.path.join(rootdir, '{0}_match.png'.format(matchID))
            bb = map(lambda c: int(round(c / rszFac)), (y1,y2,x1,x2))

            Ireg = np.nan_to_num(Ireg)
            Ireg = shared.fastResize(Ireg, 1 / rszFac)
            if Ireg.shape != (h, w):
                newIreg = np.zeros((h,w))
                newIreg[0:Ireg.shape[0], 0:Ireg.shape[1]] = Ireg
                Ireg = newIreg
            scipy.misc.imsave(patchpath, Ireg)
            examples.append((regionpath, (gl_idx,), patchpath))
            self.matches.setdefault(regionpath, []).append((patchpath, matchID, self.current_digit, score2, y1, y2, x1, x2, rszFac))
            matchID += 1
            patchpath_scores[patchpath] = score2
        pickle.dump(patchpath_scores, open(patchpath_scoresP, 'wb'))

        group = common.DigitGroupClass(examples, user_data=patchpath_scores)
        exemplar_paths = {grouplabel: self.PATCH_TMP}

        # == Now, verify the found-matches via overlay-verification
        self.f = VerifyOverlayFrame(self, group, exemplar_paths, self.parent.parent.project, self.on_verifydone)
        self.f.Maximize()
        self.Disable()
        self.disable_cells()
        self.f.Show()

    def on_verifydone(self, results):
        """Invoked once the user has finished verifying the template
        matching on the current digit. Add all 'correct' matches to
        the relevant cell's boxes.
        """
        self.f.Close()
        self.Enable()
        self.enable_cells()

        # 1.) Remove all matches from self.matches that the user said
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
                        os.remove(patchpath)
                        stuff = self.matches[regionpath]
                        # stuff[i] := (patchpath, matchID, digit, score, y1,y2,x1,x2, rszFac)
                        stuff = [t for t in stuff if t[0] != patchpath]
                        self.matches[regionpath] = stuff

        # 2.) Add all matches that the user said was 'Good' to the UI
        added_matches = 0

        def get_digit(patchpath):
            """ patchpaths are of the form:
                <projdir>/digit_exemplars/0_examples/*.png
            """
            return os.path.split(os.path.split(patchpath)[0])[1].split("_")[0]
        for regionpath, stuff in self.matches.iteritems():
            boxes = []
            for (patchpath, matchID, digit, score, y1, y2, x1, x2, rszFac) in stuff:
                x1, y1, x2, y2 = map(lambda c: int(round((c/rszFac))), (x1,y1,x2,y2))
                # Then, scale it by the resizing done in setup_grid
                # (these coords are only for the LabelDigits UI).
                _x1, _y1, _x2, _y2 = map(lambda c: int(round((c/self.rszFac))), (x1,y1,x2,y2))
                dig = get_digit(patchpath)
                newbox = Box(_x1, _y1, _x2, _y2, digit=dig)
                added_matches += 1
                boxes.append(newbox)
                #self.add_box(newbox, regionpath)
            self.cells[regionpath].boxes = boxes
            self.update_precinct_txt(regionpath)
        print "Added {0} matches.".format(added_matches)

    def sort_cells(self):
        """ Sorts cells by average intensity (most-intense cells
        at the beginning of the grid). This is to allow the UI to
        present only the cells that probably still have digits-to-be-
        labeled.
        """
        self.Disable()
        dlg = wx.MessageDialog(self, message="Sorry, this feature \
hasn't been implemented yet. Stay tuned!",
                               style=wx.OK)
        dlg.ShowModal()
        self.Enable()
                
    def compute_and_save_digitexemplars_map(self):
        digitexemplars_map = {} # maps {str digit: ((regionpath_i, score, bb, patchpath_i), ...)}
        for regionpath, stuff in self.matches.iteritems():
            for (patchpath, matchID, digit, score, y1, y2, x1, x2, rszFac) in stuff:
                bb = map(lambda c: int(round(c / rszFac)), (y1,y2,x1,x2))
                digitexemplars_map.setdefault(digit, []).append((regionpath, score, bb, patchpath))
        de_mapP = pathjoin(self.parent.parent.project.projdir_path,
                           self.parent.parent.project.digit_exemplars_map)
        pickle.dump(digitexemplars_map, open(de_mapP, 'wb'))

    def on_done(self):
        """When the user decides that he/she has indeed finished
        labeling all digits. Export the results, such as the
        mapping from precinct-patch to precinct number.
        """
        result = self.export_results()
        self.compute_and_save_digitexemplars_map()
        if self.ondone:
            self.ondone(result)
    
        self.Disable()

    def export_results(self):
        """ Saves out the digitattrvals_blanks.p file. """
        result = self.get_patch2precinct()
        self.export_precinct_nums(result)
        return result

    def get_patch2precinct(self):
        """ Called by on_done. Computes the result dictionary:
            {str patchpath: str precinct number},
        where patchpath is the path to the precinct patch of
        some blank ballot.
        """
        result = {}
        for patchpath, txt in self.precinct_txts.iteritems():
            assert patchpath not in result
            # txt.GetLabel() returns something like 'Precinct Number:0013038',
            # so get rid of this.
            result[patchpath] = txt.GetLabel().split(":")[1].strip()
        return result

    def export_precinct_nums(self, result):
        """ Export precinct nums to a specified outfile. Saves a data
        structure dict of the form:
            {str blankpath: {attrtype: (str digitval, (y1,y2,x1,x2), int side)}}
        Input:
            dict result: maps {str patchpath: str digitval}
            str outpath:
        """
        proj = self.parent.parent.project  # TODO: breach of abstraction
        digitpatch2temp = pickle.load(open(pathjoin(proj.projdir_path,
                                                    proj.digitpatch2temp),
                                           'rb'))
        digitattrvals_blanks = {}  # maps {str templatepath: {digitattrtype: digitval}}
        for patchpath, digitval in result.iteritems():
            if patchpath not in digitpatch2temp:
                print "Uhoh, patchpath not in digitpatch2temp:", patchpath
                pdb.set_trace()
            temppath, attrstr, bb, side = digitpatch2temp[patchpath]
            digitattrvals_blanks.setdefault(temppath, {})[attrstr] = (digitval, bb, side)
        pickle.dump(digitattrvals_blanks, open(pathjoin(proj.projdir_path,
                                                        proj.digitattrvals_blanks),
                                               'wb'))

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
            return None
        tmp = self._box
        self._box = None
        # cut off coords that are out-of-bounds
        x1,y1,x2,y2 = Box.make_canonical(tmp)
        tmp.x1 = max(0, x1)
        tmp.y1 = max(0, y1)
        tmp.x2 = min(self.pil_img.size[0]-1, x2)
        tmp.y2 = min(self.pil_img.size[1]-1, y2)
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
            if len(npimg.shape) == 1:
                print "Degenerate array returned from extract_region. \
Saving npimg as _errtmp_npimg_degenerate.png"
                scipy.misc.imsave("_errtmp_npimg_degenerate.png", npimg)
                return
            h, w = npimg.shape
            if w <= 2 or h <= 2:
                print "Extracted region was too small for template \
matching. Saving to: _errtmp_npimg.png"
                scipy.misc.imsave("_errtmp_npimg.png", npimg)
                return
            #npimg_crop = autocrop_img(npimg)
            # Let's try not cropping, since it might help with digit
            # recognition.
            npimg_crop = npimg 
            if npimg_crop == None:
                print "autocrop failed. saving to: _errtmp_npimg_failcrop.png"
                scipy.misc.imsave("_errtmp_npimg_failcrop.png", npimg)
                return
            #scipy.misc.imsave('before_crop.png', npimg)
            #scipy.misc.imsave('after_crop.png', npimg_crop)
            npimg_crop = np.float32(npimg_crop / 255.0)
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
            matches = shared.find_patch_matchesV1(self.img1, bb[:], regions, threshold=0.8)
        except Exception as e:
            scipy.misc.imsave('_err_img1.png', self.img1)
            errf = open('_err_findpatchmatches.log', 'w')
            print >>errf, bb
            print >>errf, regions
            errf.close()
            traceback.print_exc()
            raise e
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
            return (xa, ya - h, xb, yb + h)
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
    def __init__(self, parent, group, exemplar_paths, project, ondone):
        wx.Frame.__init__(self, parent)
        self.parent = parent
        self.group = group
        self.exemplar_paths = exemplar_paths
        self.ondone = ondone
        self.project = project # TODO: Breach of Abstraction

        verifypanel = verify_overlays.VerifyPanel(self, verify_overlays.VerifyPanel.MODE_YESNO)
        verifypanel.start((group,), exemplar_paths, self.project, ondone=ondone)

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

def prune_matches(matches, prev_matches):
    """ Discards matches that are already present within 'prev_matches'.
    This is if a match within matches has a bounding box that partially
    overlaps with a match in prev_matches.
    Input:
        lst matches: List of (regionpath,score1,score2,IReg,y1,y2,x1,x2,rszFac)
        dict prev_matches: maps {str regionpath: lst of (patchpath,matchID,score,y1,y2,x1,x2,rszFac)}
    Output:
        A new list of matches.
    """
    def is_overlap(bb1, bb2, c=0.50):
        """ Returns True if bb1 and bb2 share a certain amount of area.
        Input:
            bb1, bb2: tuple (y1, y2, x1, x2)
        """
        area_1 = float(abs(bb1[0]-bb1[1])*abs(bb1[2]-bb1[3]))
        common_area = float(get_common_area(bb1, bb2))
        overlap_area = common_area / area_1
        if overlap_area  >= c:
            return True
        else:
            return False
        return False
    def is_overlap_any(regionpath, bb, bb_lst, c=0.25):
        for regionpath2, bb2 in bb_lst:
            if regionpath == regionpath2 and is_overlap(bb, bb2, c=c):
                return True
        return False
    pruned_matches = []
    prev_bbs = []
    for regionpath, tuples in prev_matches.iteritems():
        for (patchpath, matchID, digit, score, y1, y2, x1, x2, rszFac) in tuples:
            prev_bbs.append((regionpath, (y1,y2,x1,x2)))
    for (regionpath,s1,s2,IReg,y1,y2,x1,x2,rszFac) in matches:
        if not is_overlap_any(regionpath, (y1,y2,x1,x2), prev_bbs):
            pruned_matches.append((regionpath,s1,s2,IReg,y1,y2,x1,x2,rszFac))
    return pruned_matches

def get_common_area(bb1, bb2):
    """ Returns common area between bb1, bb2.
    Input:
        bb1: (y1,y2,x1,x2)
        bb2: (y1,y2,x1,x2)
    Output:
        area.
    """
    def common_segment(seg1, seg2):
        """ Returns the segment common to both seg1, seg2:
        Input:
            seg1, seg2: tuples (a1, a2)
        Output:
            A tuple (b1, b2), or None if there's no intersection.
        """
        # First make seg1 to the left of seg2
        if seg2[0] < seg1[0]:
            tmp = seg1
            seg1 = seg2
            seg2 = seg1
        if seg2[0] < seg1[1]:
            outA = seg2[0]
            outB = min(seg1[1], seg2[1])
            return (outA, outB)
        else:
            return None
    if bb1[3] > bb2[3]:
        # Make bb1 to the left of bb2
        tmp = bb1
        bb1 = bb2
        bb2 = tmp
    y1a,y2a,x1a,x2a = bb1
    y1b,y2b,x1b,x2b = bb2
    w_a, h_a = abs(x1a-x2a), abs(y1a-y2a)
    w_b, h_b = abs(x1b-x2b), abs(y1b-y2b)
    segw_a = x1a, x1a+w_a
    segh_a = y1a, y1a+h_a
    segw_b = x1b, x1b+w_b
    segh_b = y1b, y1b+h_b
    cseg_w = common_segment(segw_a, segw_b)
    cseg_h = common_segment(segh_a, segh_b)
    if cseg_w == None or cseg_h == None:
        return 0.0
    else:
        return abs(cseg_w[0]-cseg_w[1]) * abs(cseg_h[0]-cseg_h[1])
    '''
    if x1b < (x1a+w_a):
        x_segment = x1a+w_a - x1b
    else:
        x_segment = 0.0
    pdb.set_trace()
    if bb1[0] > bb2[0]:
        # Make bb1 on top of bb2
        tmp = bb1
        bb1 = bb2
        bb2 = tmp
    y1a,y2a,x1a,x2a = bb1
    y1b,y2b,x1b,x2b = bb2
    w_a, h_a = abs(x1a-x2a), abs(y1a-y2a)
    w_b, h_b = abs(x1b-x2b), abs(y1b-y2b)
    pdb.set_trace()
    if y1b < (y1a+h_a):
        y_segment = y1a+h_a - y1a
    else:
        y_segment = 0.0
    return x_segment * y_segment
    '''

def _test_get_common_area():
    bb1 = [2, 0, 0, 1]
    bb2 = [1, 0, 0, 2]
    print get_common_area(bb1, bb2)

    bb1 = [3, 1, 1, 2]
    bb2 = [2, 0, 3, 5]
    print get_common_area(bb1, bb2)

    bb1 = [1, 3, 1, 3]
    bb2 = [1, 3, 2, 3]
    print get_common_area(bb1, bb2)
#_test_get_common_area()
#pdb.set_trace()

def is_overlap(rect1, rect2):
    """
    Returns True if any part of rect1 is contained within rect2.
    Input:
        rect1: Tuple of (x1,y1,x2,y2)
        rect2: Tuple of (x1,y1,x2,y2)
    """
    def is_within_box(pt, box):
        return box[0] < pt[0] < box[2] and box[1] < pt[1] < box[3]
    x1, y1, x2, y2 = rect1
    w, h = abs(x2-x1), abs(y2-y1)
    # Checks (in order): UL, UR, LR, LL corners
    return (is_within_box((x1,y1), rect2) or
            is_within_box((x1+w,y1), rect2) or 
            is_within_box((x1+w,y1+h), rect2) or 
            is_within_box((x1,y1+h), rect2))
def too_close(b1, b2):
    """
    Input:
        b1: Tuple of (x1,y1,x2,y2)
        b2: Tuple of (x1,y1,x2,y2)
    """
    dist = util_gui.dist_euclidean
    w, h = abs(b1[0]-b1[2]), abs(b1[1]-b1[3])
    return ((abs(b1[0] - b2[0]) <= w / 2.0 and
             abs(b1[1] - b2[1]) <= h / 2.0) or
            is_overlap(b1, b2) or 
            is_overlap(b2, b1))

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
    if len(B.shape) == 1:
        pdb.set_trace()
        return None
    (ystart, xstart), (ystop, xstop) = B.min(0), B.max(0) + 1
    return img[ystart:ystop, xstart:xstop]

def do_extract_digitbased_patches(proj):
    """ Extracts all digit-based attribute patches, and stores them
    in the proj.extracted_digitpatch_dir directory.
    Input:
        obj proj:
    Output:
        Returns a dict mapping {str patchpath: (templatepath, attrtype, bb, int side)}
    """

    # all_attrtypes is a list of dicts (marshall'd AttributeBoxes)
    
    all_attrtypes = pickle.load(open(proj.ballot_attributesfile, 'rb'))
    digit_attrtypes = []  # list of (attrs,x1,y1,x2,y2,side)
    for attrbox_dict in all_attrtypes:
        if attrbox_dict['is_digitbased']:
            attrs = attrbox_dict['attrs']
            x1 = attrbox_dict['x1']
            y1 = attrbox_dict['y1']
            x2 = attrbox_dict['x2']
            y2 = attrbox_dict['y2']
            side = attrbox_dict['side']
            digit_attrtypes.append((attrs,x1,y1,x2,y2,side))
    tmp2imgs = pickle.load(open(proj.template_to_images, 'rb'))
    patch2temp = {}  # maps {patchpath: templatepath}
    w_img, h_img = proj.imgsize
    tasks = [(templateid,path) for (templateid,path) in tmp2imgs.iteritems()]
    return partask.do_partask(extract_digitbased_patches,
                              tasks,
                              _args=(digit_attrtypes, (w_img,h_img), proj),
                              combfn=_my_combfn,
                              init={},
                              pass_idx=True)

def _my_combfn(results, subresults):
    return dict(results.items() + subresults.items())

def extract_digitbased_patches(tasks, (digit_attrtypes, imgsize, proj), idx):
    i = 0
    w_img, h_img = imgsize
    outdir = pathjoin(proj.projdir_path, proj.extracted_digitpatch_dir)
    patch2temp = {} # maps {str patchpath: (imgpath, attrtype, bb, str side)}
    for (attrs,x1,y1,x2,y2,side) in digit_attrtypes:
        x1, x2 = map(lambda x: int(round(x*w_img)), (x1,x2))
        y1, y2 = map(lambda y: int(round(y*h_img)), (y1,y2))
        for templateid, path in tasks:
            # Grab the correct image...
            if proj.is_multipage:
                frontpath, backpath = path
                if side == 'front':
                    imgpath = frontpath
                    img = shared.standardImread(frontpath, flatten=True)
                else:
                    imgpath = backpath
                    img = shared.standardImread(backpath, flatten=True)
            else:
                imgpath = path[0]
                img = shared.standardImread(path[0], flatten=True)
            patch = img[y1:y2, x1:x2]
            attrs_sorted = sorted(attrs.keys())
            attrs_sortedstr = '_'.join(attrs_sorted)
            util_gui.create_dirs(pathjoin(outdir,
                                          attrs_sortedstr))
            outfilename = '{0}_{1}_exemplar.png'.format(idx, i)
            outfilepath = pathjoin(outdir,
                                   attrs_sortedstr,
                                   outfilename)
            scipy.misc.imsave(outfilepath, patch)
            bb = (y1, y2, x1, x2)
            patch2temp[outfilepath] = (imgpath, attrs_sortedstr, bb, side)
            i += 1
    return patch2temp
    
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

