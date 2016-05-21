import os
import pdb
import re
from os.path import join as pathjoin
try:
    import cPickle as pickle
except ImportError:
    import pickle
import wx
try:
    from wx.lib.pubsub import pub
except:
    from wx.lib.pubsub import Publisher
    pub = Publisher()
import cv

import ffwx
from util import debug, warn, error


import util
import config
from vendors import Hart, ES_S, Sequoia, Diebold, SingleTemplate, DevVendor

BALLOT_VENDORS = (
    "Hart",
    "es_s",
    "Sequoia",
    "Diebold",
    "Single Template (generic)",
    "DevVendor",
)
VENDOR_CLASSES = {
    'hart': Hart.HartVendor,
    'es_s': ES_S.ESSVendor,
    "sequoia": Sequoia.SequoiaVendor,
    "single template (generic)": SingleTemplate.SingleTemplateVendor,
    "diebold": Diebold.DieboldVendor,
    "devvendor": DevVendor.DevVendor,
}

SEPARATE_MODE_SINGLE_SIDED = 42
SEPARATE_MODE_ALTERNATING = 43
SEPARATE_MODE_REGEX_SIMPLE = 44
SEPARATE_MODE_REGEX_CTR = 45


class ConfigPanel(ffwx.Panel):

    def __init__(self, parent, *args, **kwargs):
        wx.Panel.__init__(
            self, parent, style=wx.SIMPLE_BORDER, *args, **kwargs)

        # Instance vars
        self.parent = parent
        self.project = None
        self.voteddir = ""

        # HOOKFN: Just a callback function to pass to Project.closehooks
        self._hookfn = None

        ff = ffwx.FFBuilder(self)

        # Set up widgets
        self.box_samples = ff.static_vbox(label='Samples') \
            .add(ff.text('Please choose the directory where the '
                         'sample images reside')) \
            .add((0, 20)) \
            .add(ff.button(label='Choose voted ballot directory',
                           on_click=self.onButton_choosesamplesdir)) \
            .add(ff.text('Voted ballot directory: ')) \
            .add(ff.text('/'), name='txt_samplespath') \
            .add((0, 20))

        # self.box_samples = wx.StaticBox(self, label="Samples")
        # self.box_samples.sizer = wx.StaticBoxSizer(
        #     self.box_samples, orient=wx.VERTICAL)
        # self.box_samples.txt = wx.StaticText(
        #     self, label="Please choose the directory where the sample images reside.")
        # self.box_samples.btn = wx.Button(
        #     self, label="Choose voted ballot directory...")
        # self.box_samples.btn.Bind(
        #     wx.EVT_BUTTON, self.onButton_choosesamplesdir)
        # self.box_samples.txt2 = wx.StaticText(
        #     self, label="Voted ballot directory:")
        # self.box_samples.txt_samplespath = wx.StaticText(self)
        # # self.box_samples.sizer.Add(self.box_samples.txt)
        # # self.box_samples.sizer.Add((0, 20))
        # self.box_samples.sizer.Add(self.box_samples.btn)
        # self.box_samples.sizer.Add((0, 20))
        # self.box_samples.sizer.Add(self.box_samples.txt2)
        # self.box_samples.sizer.Add(self.box_samples.txt_samplespath)
        # self.box_samples.sizer.Add((0, 20))

        sizer0 = ff.hbox() \
            .add(self.box_samples, proportion=1, flag=wx.EXPAND) \
            .add((50, 0))

        txt_numpages = ff.text("Number of pages: ")
        self.numpages_txtctrl = ff.text_ctrl("2")
        self.varnumpages_chkbox = ff.check_box(
            label="Variable Number of Pages", default=False) \
            .on_check(self.onCheckBox_varnumpages)
        sizer_numpages = ff.hbox(
            txt_numpages,
            (10, 0),
            self.numpages_txtctrl,
            (10, 0),
            self.varnumpages_chkbox,
        )

        ssizer_ballotgroup = ff.static_vbox(
            label='Ballot Grouping/Pairing Configuration')

        txt_regex_shr = ff.text(
            'Enter a regex to match on the shared filename part.')
        self.regexShr_txtctrl = ff.text_ctrl(
            r"(.*_.*_.*_).*_.*\.[a-zA-Z]*",
            size=(300, -1))
        txt_regex_diff = ff.text(
            "Enter a regex to match on the distinguishing filename part.")
        self.regexDiff_txtctrl = ff.text_ctrl(
            r".*_.*_.*_(.*_.*)\.[a-zA-Z]*",
            size=(300, -1))
        sizer_regexShr = ff.hbox(
            txt_regex_shr,
            (10, 0),
            self.regexShr_txtctrl)
        sizer_regexDiff = ff.hbox(
            txt_regex_diff,
            (10, 0),
            self.regexDiff_txtctrl)

        self.regex_ctr_chkbox = ff.check_box(
            label=("Do the filenames end in incrementing counters? "
                   "(Typically 'Yes' for Hart ballots)"),
            default=True) \
            .on_check(self.onCheckBox_regexCtr)

        self.sizer_regex1 = ff.vbox(
            (0, 10),
            sizer_regexShr,
            (0, 10),
            sizer_regexDiff,
            (0, 10),
            self.regex_ctr_chkbox,
        )

        self.txt_or = ff.text(label="- Or -")
        self.txt_regex_shr = txt_regex_shr
        self.txt_regex_diff = txt_regex_diff

        self.alternate_chkbox = ff.check_box(
            label="Ballots alternate front and back",
            default=True) \
            .on_check(self.onCheckBox_alternate)
        self.alternate_chkbox.SetValue(True)

        ssizer_ballotgroup.add(self.alternate_chkbox, border=10, flag=wx.ALL) \
            .add((0, 10)) \
            .add(self.txt_or, 0, wx.ALIGN_CENTER, 10, wx.ALL) \
            .add((0, 10)) \
            .add(self.sizer_regex1, 0, wx.ALL, 10)
        self.txt_or.Hide()
        self.regexShr_txtctrl.Hide()
        self.regexDiff_txtctrl.Hide()
        self.regex_ctr_chkbox.Hide()
        self.txt_regex_shr.Hide()
        self.txt_regex_diff.Hide()

        self.is_straightened = ff.check_box(
            -1, label="Ballots already straightened.")
        self.is_straightened.Hide()

        txt_vendor = ff.text("What is the ballot vendor?")
        self.vendor_dropdown = ff.combo_box(style=wx.CB_READONLY,
                                            choices=BALLOT_VENDORS)
        sizer_vendor = ff.hbox(txt_vendor, self.vendor_dropdown)

        self.sizer = ff.vbox() \
            .add(sizer0, 1, wx.EXPAND) \
            .add(sizer_numpages) \
            .add(ssizer_ballotgroup) \
            .add(self.is_straightened) \
            .add(sizer_vendor)

        self.SetSizer(ff.hbox().add(self.sizer))
        self.Layout()

    def start(self, project=None, projdir=None):
        """
        Input:
            obj PROJECT:
        """
        print 'got', project
        self.project = project
        self.stateP = self.project.path('_state_config.p')
        self._hookfn = lambda: self.save_session(self.stateP)
        self.project.addCloseEvent(self._hookfn)
        if self.restore_session(stateP=self.stateP):
            return
        self.voteddir = ''

    def stop(self):
        self.save_session(stateP=self.stateP)
        self.project.removeCloseEvent(self._hookfn)
        self.export_results()

    def export_results(self):
        """ Create and store the ballot_to_images and image_to_ballot
        data structures. Also, set the proj.voteddir, proj.imgsize,
        proj.is_multipage, proj.num_pages, and proj.vendor_obj properties.
        """
        # BALLOT_TO_IMAGES: maps {int ballotID: [imgpath_side0, imgpath_side1,
        # ...]}
        ballot_to_images = {}
        image_to_ballot = {}  # maps {imgpath: int ballotID}

        def get_separate_mode():
            if self.alternate_chkbox.GetValue():
                return SEPARATE_MODE_ALTERNATING
            elif int(self.numpages_txtctrl.GetValue()) == 1:
                return SEPARATE_MODE_SINGLE_SIDED
            elif self.regex_ctr_chkbox.IsEnabled() and self.regex_ctr_chkbox.GetValue():
                return SEPARATE_MODE_REGEX_CTR
            else:
                return SEPARATE_MODE_REGEX_SIMPLE
        MODE = get_separate_mode()
        by_ballots = separate_imgs(self.voteddir, int(self.numpages_txtctrl.GetValue()),
                                   MODE,
                                   regexShr=self.regexShr_txtctrl.GetValue(),
                                   regexDiff=self.regexDiff_txtctrl.GetValue())
        curballotid = 0
        weirdballots = []
        stats_numpages = {}  # maps {int numpages: int cnt}
        for i, imgpaths in enumerate(by_ballots):
            _numpages = len(imgpaths)
            if _numpages not in stats_numpages:
                stats_numpages[_numpages] = 1
            else:
                stats_numpages[_numpages] += 1
            if not self.varnumpages_chkbox.GetValue() and len(imgpaths) != int(self.numpages_txtctrl.GetValue()):
                # Ballot has too many/few sides.
                debug("Warning -- found Ballot with {0} sides, yet project \
                      \specified {1} sides.",
                      len(imgpaths),
                      int(self.numpages_txtctrl.GetValue()))
                weirdballots.append(imgpaths)
            elif config.BALLOT_LIMIT is not None and i >= config.BALLOT_LIMIT:
                break
            else:
                ballot_to_images[curballotid] = imgpaths
                for imgpath in imgpaths:
                    image_to_ballot[imgpath] = curballotid
                curballotid += 1
        debug("Number of pages in a ballot for the election:")
        for _numpages, cnt in sorted(stats_numpages.iteritems()):
            debug("    {0} pages: {1} ballots", _numpages, cnt)
        debug("Detected {0} weird ballots with too many/few sides.",
              len(weirdballots))
        if weirdballots:
            ffwx.warn(self,
                      "Warning: OpenCount detected {0} ballots \
                    \that had too many/few sides. The project \
                    \specified that there are {1} sides for \
                    \each ballot. These ballots will be discarded \
                    \from the election, but stored in \
                    \'_config_weirdballots.p'.".format(
                          len(weirdballots),
                          self.numpages_txtctrl.GetValue()))

        pickle.dump(weirdballots, open(
            pathjoin(self.project.projdir_path, '_config_weirdballots.p'), 'wb'))
        pickle.dump(ballot_to_images, open(
            self.project.ballot_to_images, 'wb'), pickle.HIGHEST_PROTOCOL)
        pickle.dump(image_to_ballot, open(
            self.project.image_to_ballot, 'wb'), pickle.HIGHEST_PROTOCOL)
        # 2.) Set project.voteddir
        self.project.voteddir = self.voteddir
        # 3.) Set project.imgsize, assuming that all image dimensions are the
        # same
        if len(image_to_ballot) == 0:
            ffwx.warn(self,
                      "Fatal Error: OpenCount couldn't \
                    \find any valid ballots in the \
                    \directory:\n\n {0}\n\n \
                    \Are you sure this is the correct \
                    \directory containing the voted \
                    \ballots?\n\n \
                    \Alternately, is the correct vendor \
                    \specified in the configuration?\n\n\
                    \Please correct the misconfiguration \
                    \(if any) and create a new Project \
                    \with the corrections.".format(self.voteddir))
            error("Everything is going to break. OpenCount didn't find \
                  \any ballots.")
            return
        w, h = None, None
        for imgpath in image_to_ballot.keys():
            if w is not None:
                break
            try:
                I = cv.LoadImage(imgpath, cv.CV_LOAD_IMAGE_UNCHANGED)
                w, h = cv.GetSize(I)
            except IOError as e:
                pass
        if w is None:
            ffwx.warn(self, "Fatal Error: OpenCount couldn't open any of \
                          \the ballot images in {0}. Processing can not \
                          \continue. If you believe the images are in \
                          \fact not corrupt, you could try converting \
                          \all images to new PNG images, in the hopes \
                          \of OpenCV being able to read the \
                          \new images.".format(self.project.voteddir))
            error("Cannot open any ballot images.")
            exit(1)

        self.project.imgsize = (w, h)
        # 4.) Set project.is_multipage
        if int(self.numpages_txtctrl.GetValue()) >= 2:
            self.project.is_multipage = True
        else:
            self.project.is_multipage = False
        # 5.) Set project.num_pages
        if not self.varnumpages_chkbox.GetValue():
            self.project.num_pages = int(self.numpages_txtctrl.GetValue())
        else:
            self.project.num_pages = None
        # 6.) Set project.is_varnum_pages
        self.project.is_varnum_pages = self.varnumpages_chkbox.GetValue()
        # 6.) Set project.vendor_obj
        self.project.vendor_obj = VENDOR_CLASSES[
            self.vendor_dropdown.GetStringSelection().lower()](self.project)

    def restore_session(self, stateP=None):
        try:
            state = pickle.load(open(stateP, 'rb'))
            self.voteddir = state['voteddir']
            self.box_samples.txt_samplespath.SetLabel(self.voteddir)
            self.is_straightened.SetValue(state['is_straightened'])
            self.numpages_txtctrl.SetValue(str(state['num_pages']))
            self.varnumpages_chkbox.SetValue(state['varnumpages'])
            self.regexShr_txtctrl.SetValue(state['regexShr'])
            self.regexDiff_txtctrl.SetValue(state['regexDiff'])
            self.regex_ctr_chkbox.SetValue(state['is_regex_ctr'])
            self.alternate_chkbox.SetValue(state['is_alternating'])
            self.vendor_dropdown.SetStringSelection(state['vendor'])
            if self.varnumpages_chkbox.GetValue():
                self.numpages_txtctrl.Disable()
            if self.alternate_chkbox.GetValue():
                self.txt_or.Hide()
                self.txt_regex_shr.Hide()
                self.txt_regex_diff.Hide()
                self.regexShr_txtctrl.Hide()
                self.regexDiff_txtctrl.Hide()
                self.regex_ctr_chkbox.Hide()
            else:
                self.txt_or.Show()
                self.txt_regex_shr.Show()
                self.txt_regex_diff.Show()
                self.regexShr_txtctrl.Show()
                self.regexDiff_txtctrl.Show()
                self.regex_ctr_chkbox.Show()

            self.onCheckBox_regexCtr(None)
        except:
            return False
        return True

    def save_session(self, stateP=None):
        state = {'voteddir': self.voteddir,
                 'is_straightened': self.is_straightened.GetValue(),
                 'num_pages': int(self.numpages_txtctrl.GetValue()),
                 'varnumpages': self.varnumpages_chkbox.GetValue(),
                 'regexShr': self.regexShr_txtctrl.GetValue(),
                 'regexDiff': self.regexDiff_txtctrl.GetValue(),
                 'is_regex_ctr': self.regex_ctr_chkbox.GetValue(),
                 'is_alternating': self.alternate_chkbox.GetValue(),
                 'vendor': self.vendor_dropdown.GetStringSelection()}
        pickle.dump(state, open(stateP, 'wb'))

    def wrap(self, text):
        res = ""
        for i in range(0, len(text), 50):
            res += text[i:i + 50] + "\n"
        return res

    def set_samplepath(self, path):
        self.voteddir = os.path.abspath(path)
        self.box_samples.txt_samplespath.SetLabel(self.wrap(self.voteddir))
        self.project.raw_samplesdir = self.voteddir
        pub.sendMessage("processing.register", data=self.project)

    def get_samplepath(self):
        return self.box_samples.txt_samplespath.GetLabelText().replace("\n", "")

    def onSanityCheck(self, evt):
        """
        Triggered when either the templates or samples sanity check
        completes. Update the relevant ListBox widget with the results
        of a sanity check.
        """
        type, results_dict = evt.data
        listbox = self.upper_scroll if type == 'templates' else self.lower_scroll
        if len(results_dict) == 0:
            listbox.Append("All files valid")
        else:
            for imgpath, msg in results_dict.items():
                listbox.Append(imgpath + ": " + msg)
        if type == 'samples':
            # Assume that we first process the templates, then the samples last
            self.parent.Enable()

    # Event Handlers
    def onButton_choosesamplesdir(self, evt):
        dlg = wx.DirDialog(self, "Select Directory",
                           defaultPath=os.getcwd(), style=wx.DD_DEFAULT_STYLE)
        result = dlg.ShowModal()
        if result == wx.ID_OK:
            dirpath = dlg.GetPath()
            self.set_samplepath(dirpath)

    def onButton_runsanitycheck(self, evt):
        self.upper_scroll.Clear()
        self.lower_scroll.Clear()
        num_files = 0
        for dirpath, dirnames, filenames in os.walk(self.voteddir):
            num_files += len(filenames)
        self.parent.Disable()
        pgauge = util_widgets.ProgressGauge(
            self, num_files, msg="Checking files...")
        pgauge.Show()
        thread = threading.Thread(target=sanity_check.sanity_check,
                                  args=(self.voteddir, self))
        thread.start()

    def onCheckBox_regexCtr(self, evt):
        if self.regex_ctr_chkbox.GetValue():
            self.regexDiff_txtctrl.Disable()
        else:
            self.regexDiff_txtctrl.Enable()

    def onCheckBox_alternate(self, evt):
        if self.alternate_chkbox.GetValue():
            # We're going from False -> True
            self.txt_or.Hide()
            self.txt_regex_shr.Hide()
            self.txt_regex_diff.Hide()
            self.regexShr_txtctrl.Hide()
            self.regexDiff_txtctrl.Hide()
            self.regex_ctr_chkbox.Hide()
        else:
            self.txt_or.Show()
            self.txt_regex_shr.Show()
            self.txt_regex_diff.Show()
            self.regexShr_txtctrl.Show()
            self.regexDiff_txtctrl.Show()
            self.regex_ctr_chkbox.Show()
        self.Layout()

    def onCheckBox_varnumpages(self, evt):
        if self.varnumpages_chkbox.GetValue():
            self.numpages_txtctrl.Disable()
        else:
            self.numpages_txtctrl.Enable()


class DoubleSideDialog(wx.Dialog):

    def __init__(self, parent, *args, **kwargs):
        wx.Dialog.__init__(
            self, parent, title="Set Double Sided Properties", *args, **kwargs)

        self.num_pages = None
        self.regex = None
        self.is_alternating = None

        txt0 = wx.StaticText(self, label="Number of pages:")
        self.numpages_txtctrl = wx.TextCtrl(self, value="2")
        sizer0 = wx.BoxSizer(wx.HORIZONTAL)
        sizer0.AddMany([(txt0,), ((10, 0),), (self.numpages_txtctrl,)])

        txt1 = wx.StaticText(
            self, label="Enter a regex to match on the file name.")
        self.regex_txtctrl = wx.TextCtrl(self, value=r".*-(.*)")
        sizer1 = wx.BoxSizer(wx.HORIZONTAL)
        sizer1.AddMany([(txt1,), ((10, 0),), (self.regex_txtctrl,)])

        self.alternate_chkbox = wx.CheckBox(
            self, label="Ballots alternate front and back")

        btn_done = wx.Button(self, label="Ok")
        btn_done.Bind(wx.EVT_BUTTON, self.onButton_done)
        btn_cancel = wx.Button(self, label="Cancel")
        btn_cancel.Bind(wx.EVT_BUTTON, self.onButton_cancel)
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        btn_sizer.AddMany([(btn_done,), (btn_cancel,)])

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.AddMany([(sizer0,), (sizer1,), (self.alternate_chkbox,),
                       (btn_sizer, 0, wx.ALIGN_CENTER)])
        self.SetSizer(sizer)
        self.Layout()

    def onButton_done(self, evt):
        self.num_pages = int(self.numpages_txtctrl.GetValue())
        self.regex = self.regex_txtctrl.GetValue()
        self.is_alternating = self.alternate_chkbox.GetValue()
        self.EndModal(wx.ID_OK)

    def onButton_cancel(self, evt):
        self.EndModal(wx.ID_CANCEL)


def separate_imgs(voteddir, num_pages, MODE,
                  regexShr=None, regexDiff=None):
    """ Separates images into sets of Ballots.
    Input:
    str VOTEDDIR: Root directory of voted ballots.
    Output:
    list BALLOTS. [Ballot0, Ballot1, ...], where each Ballot_i
    is a list of [imgpath_side0, imgpath_side1, ...].
    """
    if MODE == SEPARATE_MODE_SINGLE_SIDED:
        return separate_singlesided(voteddir)
    elif MODE == SEPARATE_MODE_ALTERNATING:
        return separate_alternating(voteddir, num_pages)
    elif MODE == SEPARATE_MODE_REGEX_SIMPLE:
        return separate_regex_simple(voteddir, regexShr, regexDiff)
    elif MODE == SEPARATE_MODE_REGEX_CTR:
        return separate_regex_ctr(voteddir, regexShr)
    else:
        error("Fatal Error: Unrecognized separate_imgs mode: '{0}'", MODE)
        raise Exception("Bad mode: '{0}'".format(MODE))


def separate_singlesided(voteddir):
    ballots = []
    for dirpath, dirnames, filenames in os.walk(voteddir):
        imgnames = [f for f in filenames if util.is_image_ext(f)]
        for imgname in imgnames:
            imgpath = pathjoin(dirpath, imgname)
            ballots.append([imgpath])
    return ballots


def separate_alternating(voteddir, num_pages):
    ballots = []
    for dirpath, dirnames, filenames in os.walk(voteddir):
        imgnames = [f for f in filenames if util.is_image_ext(f)]
        imgnames_ordered = util.sorted_nicely(imgnames)
        if len(imgnames_ordered) % num_pages != 0:
            error("There are {0} images in directory {1}, which \
                  \isn't divisible by num_pages {2}",
                  len(imgnames_ordered),
                  dirpath,
                  num_pages)
            pdb.set_trace()
            raise RuntimeError
        i = 0
        while imgnames_ordered:
            curballot = []
            for j in xrange(num_pages):
                imgpath = pathjoin(dirpath, imgnames_ordered.pop(0))
                curballot.append(imgpath)
            ballots.append(curballot)
    return ballots


def separate_regex_simple(voteddir, regexShr, regexDiff):
    ballots = []
    for dirpath, dirnames, filenames in os.walk(voteddir):
        imgnames = [f for f in filenames if util.is_image_ext(f)]
        shrPat = re.compile(regexShr)
        diffPat = re.compile(regexDiff)
        curmats = {}  # maps {str sim_pat: [(str imgpath, str diff_pat), ...]}
        for imgname in imgnames:
            imgpath = pathjoin(dirpath, imgname)
            sim_match = shrPat.match(imgname)
            diff_match = diffPat.match(imgname)
            if sim_match is None or diff_match is None:
                warn("Ballot {0} was skipped because it didn't \
                     \match the regular expressions.",
                     imgpath)
                continue
            sim_part = sim_match.groups()[0]
            diff_part = diff_match.groups()[0]
            curmats.setdefault(sim_part, []).append((imgpath, diff_part))
        for sim_pat, tuples in curmats.iteritems():
            # sort by diffPart
            tuples_sorted = sorted(tuples, key=lambda t: t[1])
            imgpaths_sorted = [t[0] for t in tuples_sorted]
            ballots.append(imgpaths_sorted)
    return ballots


def separate_regex_ctr(voteddir, regexShr):
    """ Separates ballots whose filenames start with a shared prefix
    REGEXSHR, but then contain two incrementing counters (very-much
    Hart-specific), i.e. for the following images:
        Ballot A:
        339_1436_5_211_1.png
        339_1436_5_212_2.png
        339_1436_5_213_3.png
        Ballot B:
        339_1436_5_214_1.png
        339_1436_5_215_2.png
    """
    ballots = []
    shrPat = re.compile(regexShr)
    for dirpath, dirnames, filenames in os.walk(voteddir):
        imgnames = [f for f in filenames if util.is_image_ext(f)]
        # maps {str sim_pat: [(str imgpath, tuple ctr_vals), ...]}
        curmats = {}
        for imgname in imgnames:
            imgpath = pathjoin(dirpath, imgname)
            sim_match = shrPat.match(imgname)
            if sim_match is None:
                warn("Ballot {0} was skipped because it didn't \
                     \match the regular expressions.", imgpath)
                continue
            sim_part = sim_match.groups()[0]
            # Assumes filename is := <SIM_PART>_N1_N2.png
            ctr_vals = [int(n) for n in os.path.splitext(
                imgname)[0][len(sim_part):].split("_")]
            curmats.setdefault(sim_part, []).append((imgpath, ctr_vals))
        for sim_pat, tuples in curmats.iteritems():
            # tuple TUPLES := [(str imgpath, (int N1, int N2)), ...]
            consecs = get_consecutives(tuples)
            for imgpaths in consecs:
                ballots.append(imgpaths)
    return ballots


def get_consecutives(tuples):
    """
    Input:
        tuple TUPLES: [(str imgpath, (int N1, int N2)), ...]
    Output:
        (tuple IMGPATHS0, tuple IMGPATHS1, ...)
    """
    # Assume that the N1 ctr val increases monotonically,
    # but the N2 ctr val increases monotonically only within
    # a single ballot, and drops down for
    # sort by images with consecutive ctr_vals
    tuples_sorted = sorted(tuples, key=lambda t: t[1][0])
    imgpath_groups = []  # [tuple IMGPATHS0, ...]
    cur_group = []
    prev_N1, prev_N2 = None, None

    for (imgpath, (N1, N2)) in tuples_sorted:
        if prev_N1 is None:  # first iteration
            prev_N1, prev_N2 = N1, N2
            cur_group.append(imgpath)
        elif N1 != prev_N1 + 1:
            # Skips in N1 imply a new ballot
            imgpath_groups.append(cur_group)
            cur_group = [imgpath]
            prev_N1, prev_N2 = N1, N2
        elif N2 <= prev_N2:
            # If N2 goes down (say, from '3' to '1'), then this
            # implies a new ballot.
            imgpath_groups.append(cur_group)
            cur_group = [imgpath]
            prev_N1, prev_N2 = N1, N2
        else:
            # This image is part of the current ballot
            cur_group.append(imgpath)
            prev_N1, prev_N2 = N1, N2
    if cur_group:
        imgpath_groups.append(cur_group)
    return imgpath_groups


def test_get_consecutives():
    test0 = (('329_1447_74_5_1.png', (5, 1)),
             ('329_1447_74_6_2.png', (6, 2)),

             ('329_1447_74_7_1.png', (7, 1)),
             ('329_1447_74_8_2.png', (8, 2)),

             ('339_128_29_2_1.png', (2, 1)),

             ('2_1_1_1.png', (1, 1)),

             ('2_1_2_1.png', (2, 1)),

             ('2_1_3_1.png', (3, 1)),
             ('2_1_4_2.png', (4, 2)))
    ballots = get_consecutives(test0)
    for i, imgpaths in enumerate(ballots):
        debug("Ballot '{0}':", i)
        for imgpath in imgpaths:
            debug("    {0}", imgpath)
    pdb.set_trace()

if __name__ == '__main__':
    test_get_consecutives()
