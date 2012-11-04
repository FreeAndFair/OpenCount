import os, sys, pdb, traceback, re
from os.path import join as pathjoin
try:
    import cPickle as pickle
except:
    import pickle
import wx
from wx.lib.pubsub import Publisher
import cv

sys.path.append('..')

import util
from vendors import Hart

BALLOT_VENDORS = ("Diebold", "Hart", "Sequoia")
VENDOR_CLASSES = {'hart': Hart.HartVendor}

class ConfigPanel(wx.Panel):
    def __init__(self, parent, *args, **kwargs):
        wx.Panel.__init__(self, parent, style=wx.SIMPLE_BORDER, *args, **kwargs)
        
        # Instance vars
        self.parent = parent
        self.project = None
        self.voteddir = ""

        # HOOKFN: Just a callback function to pass to Project.closehooks
        self._hookfn = None
        
        # Set up widgets
        self.box_samples = wx.StaticBox(self, label="Samples")
        self.box_samples.sizer = wx.StaticBoxSizer(self.box_samples, orient=wx.VERTICAL)
        self.box_samples.txt = wx.StaticText(self, label="Please choose the directory where the sample images reside.")
        self.box_samples.btn = wx.Button(self, label="Choose voted ballot directory...")
        self.box_samples.btn.Bind(wx.EVT_BUTTON, self.onButton_choosesamplesdir)
        self.box_samples.txt2 = wx.StaticText(self, label="Voted ballot directory:")
        self.box_samples.txt_samplespath = wx.StaticText(self)
        self.box_samples.sizer.Add(self.box_samples.txt)
        self.box_samples.sizer.Add((0, 20))
        self.box_samples.sizer.Add(self.box_samples.btn)
        self.box_samples.sizer.Add((0, 20))
        self.box_samples.sizer.Add(self.box_samples.txt2)
        self.box_samples.sizer.Add(self.box_samples.txt_samplespath)
        self.box_samples.sizer.Add((0, 20))

        self.lower_scroll = wx.ListBox(self) # Voted Skipped ListBox
        self.lower_scroll.box = wx.StaticBox(self, label="For the voted ballots, the following files were skipped:")
        sboxsizer0 = wx.StaticBoxSizer(self.lower_scroll.box, orient=wx.VERTICAL)
        sboxsizer0.Add(self.lower_scroll, 1, flag=wx.EXPAND)

        sizer0 = wx.BoxSizer(wx.HORIZONTAL)
        sizer0.Add(self.box_samples.sizer, proportion=1, flag=wx.EXPAND)
        sizer0.Add((50, 0))
        sizer0.Add(sboxsizer0, proportion=1, flag=wx.EXPAND)
        
        txt_numpages = wx.StaticText(self, label="Number of pages: ")
        self.numpages_txtctrl = wx.TextCtrl(self, value="2")
        self.varnumpages_chkbox = wx.CheckBox(self, label="Variable Number of Pages: ")
        self.varnumpages_chkbox.Bind(wx.EVT_CHECKBOX, self.onCheckBox_varnumpages)
        sizer_numpages = wx.BoxSizer(wx.HORIZONTAL)
        sizer_numpages.AddMany([(txt_numpages,), ((10,0),), (self.numpages_txtctrl,),
                                ((10,0),), (self.varnumpages_chkbox,)])
        
        sbox_ballotgroup = wx.StaticBox(self, label="Ballot Grouping/Pairing Configuration")
        ssizer_ballotgroup = wx.StaticBoxSizer(sbox_ballotgroup, orient=wx.VERTICAL)

        txt_regex_shr = wx.StaticText(self, label="Enter a regex to match on the shared filename part.")
        self.regexShr_txtctrl = wx.TextCtrl(self, value=r"(.*_.*_.*_).*_.*\.[a-zA-Z]*", size=(300,-1))
        txt_regex_diff = wx.StaticText(self, label="Enter a regex to match on the distinguishing filename part.")
        self.regexDiff_txtctrl = wx.TextCtrl(self, value=r".*_.*_.*_(.*_.*)\.[a-zA-Z]*", size=(300,-1))
        sizer_regexShr = wx.BoxSizer(wx.HORIZONTAL)
        sizer_regexDiff = wx.BoxSizer(wx.HORIZONTAL)
        sizer_regexShr.AddMany([(txt_regex_shr,), ((10,0),), (self.regexShr_txtctrl,)])
        sizer_regexDiff.AddMany([(txt_regex_diff,), ((10,0),), (self.regexDiff_txtctrl,)])
        sizer_regex1 = wx.BoxSizer(wx.VERTICAL)
        sizer_regex1.AddMany([((0, 10),), (sizer_regexShr,), ((0,10),), (sizer_regexDiff,)])

        txt_or = wx.StaticText(self, label="- Or -")

        self.alternate_chkbox = wx.CheckBox(self, label="Ballots alternate front and back")
        self.alternate_chkbox.Bind(wx.EVT_CHECKBOX, self.onCheckBox_alternate)

        ssizer_ballotgroup.AddMany([(sizer_regex1,), ((0,10),), (txt_or,0,wx.ALIGN_CENTER), ((0,10),), (self.alternate_chkbox,)])
        
        self.is_straightened = wx.CheckBox(self, -1, label="Ballots already straightened.")
        
        txt_vendor = wx.StaticText(self, label="What is the ballot vendor?")
        self.vendor_dropdown = wx.ComboBox(self, style=wx.CB_READONLY, choices=BALLOT_VENDORS)
        sizer_vendor = wx.BoxSizer(wx.HORIZONTAL)
        sizer_vendor.AddMany([(txt_vendor,), (self.vendor_dropdown,)])

        self.btn_run = wx.Button(self, label="Run sanity check")
        self.btn_run.Bind(wx.EVT_BUTTON, self.onButton_runsanitycheck)
        self.btn_run.box = wx.StaticBox(self)
        sboxsizer1 = wx.StaticBoxSizer(self.btn_run.box, orient=wx.VERTICAL)
        sboxsizer1.Add(self.btn_run)

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(sizer0)
        self.sizer.Add((0, 25))
        self.sizer.Add(sizer_numpages)
        self.sizer.Add((0, 25))
        self.sizer.Add(ssizer_ballotgroup)
        self.sizer.Add((0, 25))
        self.sizer.Add(self.is_straightened)
        self.sizer.Add((0, 25))
        self.sizer.Add(sizer_vendor)
        self.sizer.Add((0, 25))
        self.sizer.Add(sboxsizer1)
        
        self.SetSizer(self.sizer)
        self.Layout()

    def start(self, project, stateP):
        """
        Input:
            obj PROJECT:
            str STATEP: Path of the state file.
        """
        self.project = project
        self.stateP = stateP
        self._hookfn = lambda : self.save_session(stateP)
        self.project.addCloseEvent(self._hookfn)
        if self.restore_session(stateP=stateP):
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
        def separate_imgs(voteddir, num_pages, regexShr=None, regexDiff=None,
                          is_alternating=None):
            """ Separates images into sets of Ballots.
            Input:
                str VOTEDDIR: Root directory of voted ballots.
            Output:
                list BALLOTS. [Ballot0, Ballot1, ...], where each Ballot_i
                    is a list of [imgpath_side0, imgpath_side1, ...].
            """
            ballots = []
            for dirpath, dirnames, filenames in os.walk(voteddir):
                imgnames = [f for f in filenames if util.is_image_ext(f)]
                if is_alternating:
                    imgnames_ordered = util.sorted_nicely(imgnames)
                    if len(imgnames_ordered) % num_pages != 0:
                        print "Uh oh -- there are {0} images in directory {1}, \
which isn't divisible by num_pages {2}".format(len(imgnames_ordered), dirpath, num_pages)
                        pdb.set_trace()
                        raise RuntimeError
                    i = 0
                    while imgnames_ordered:
                        curballot = []
                        for j in xrange(num_pages):
                            imgpath = pathjoin(dirpath, imgnames_ordered.pop(0))
                            curballot.append(imgpath)
                        ballots.append(curballot)
                elif num_pages == 1:
                    for imgname in imgnames:
                        imgpath = pathjoin(dirpath, imgname)
                        ballots.append([imgpath])
                else:
                    shrPat = re.compile(regexShr)
                    diffPat = re.compile(regexDiff)
                    curmats = {} # maps {str sim_pat: [(str imgpath, str diff_pat), ...]}
                    for imgname in imgnames:
                        imgpath = pathjoin(dirpath, imgname)
                        sim_part = shrPat.match(imgname).groups()[0]
                        diff_part = diffPat.match(imgname).groups()[0]
                        curmats.setdefault(sim_part, []).append((imgpath, diff_part))
                    for sim_pat, tuples in curmats.iteritems():
                        # sort by diffPart
                        tuples_sorted = sorted(tuples, key=lambda t: t[1])
                        imgpaths_sorted = [t[0] for t in tuples_sorted]
                        ballots.append(imgpaths_sorted)
            return ballots
        # BALLOT_TO_IMAGES: maps {int ballotID: [imgpath_side0, imgpath_side1, ...]}
        ballot_to_images = {}
        image_to_ballot = {} # maps {imgpath: int ballotID}
        by_ballots = separate_imgs(self.voteddir, int(self.numpages_txtctrl.GetValue()),
                                   regexShr=self.regexShr_txtctrl.GetValue(),
                                   regexDiff=self.regexDiff_txtctrl.GetValue(),
                                   is_alternating=self.alternate_chkbox.GetValue())
        for id, imgpaths in enumerate(by_ballots):
            ballot_to_images[id] = imgpaths
            for imgpath in imgpaths:
                image_to_ballot[imgpath] = id
        pickle.dump(ballot_to_images, open(self.project.ballot_to_images, 'wb'), pickle.HIGHEST_PROTOCOL)
        pickle.dump(image_to_ballot, open(self.project.image_to_ballot, 'wb'), pickle.HIGHEST_PROTOCOL)
        # 2.) Set project.voteddir
        self.project.voteddir = self.voteddir
        # 3.) Set project.imgsize, assuming that all image dimensions are the same
        I = cv.LoadImage(image_to_ballot.keys()[0], cv.CV_LOAD_IMAGE_UNCHANGED)
        w, h = cv.GetSize(I)
        self.project.imgsize = (w, h)
        # 4.) Set project.is_multipage
        if int(self.numpages_txtctrl.GetValue()) >= 2:
            self.project.is_multipage = True
        else:
            self.project.is_multipage = False
        # 5.) Set project.num_pages
        self.project.num_pages = int(self.numpages_txtctrl.GetValue())
        # 6.) Set project.vendor_obj
        self.project.vendor_obj = VENDOR_CLASSES[self.vendor_dropdown.GetStringSelection().lower()]()
        
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
            self.alternate_chkbox.SetValue(state['is_alternating'])
            self.vendor_dropdown.SetStringSelection(state['vendor'])
            if self.varnumpages_chkbox.GetValue():
                self.numpages_txtctrl.Disable()
            if self.alternate_chkbox.GetValue():
                self.regexShr_txtctrl.Disable()
                self.regexDiff_txtctrl.Disable()
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
                 'is_alternating': self.alternate_chkbox.GetValue(),
                 'vendor': self.vendor_dropdown.GetStringSelection()}
        pickle.dump(state, open(stateP, 'wb'))

    def wrap(self, text):
        res = ""
        for i in range(0,len(text),50):
            res += text[i:i+50]+"\n"
        return res

    def set_samplepath(self, path):
        self.voteddir = os.path.abspath(path)
        self.box_samples.txt_samplespath.SetLabel(self.wrap(self.voteddir))
        self.project.raw_samplesdir = self.voteddir
        Publisher().sendMessage("processing.register", data=self.project)
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
            TIMER.stop_task(('cpu', MainFrame.map_pages[MainFrame.CONFIG]['cpu']))
            TIMER.start_task(('user', MainFrame.map_pages[MainFrame.CONFIG]['user']))
            self.parent.Enable()

    #### Event Handlers
    def onButton_choosesamplesdir(self, evt):
        dlg = wx.DirDialog(self, "Select Directory", defaultPath=os.getcwd(), style=wx.DD_DEFAULT_STYLE)
        result = dlg.ShowModal()
        if result == wx.ID_OK:
            dirpath = dlg.GetPath()
            self.set_samplepath(dirpath)
                
    def onButton_runsanitycheck(self, evt):
        TIMER.stop_task(('user', MainFrame.map_pages[MainFrame.CONFIG]['user']))
        TIMER.start_task(('cpu', MainFrame.map_pages[MainFrame.CONFIG]['cpu']))
        self.upper_scroll.Clear()
        self.lower_scroll.Clear()
        num_files = 0
        for dirpath, dirnames, filenames in os.walk(self.voteddir):
            num_files += len(filenames)
        self.parent.Disable()
        pgauge = util_widgets.ProgressGauge(self, num_files, msg="Checking files...")
        pgauge.Show()
        thread = threading.Thread(target=sanity_check.sanity_check,
                                  args=(self.voteddir, self))
        thread.start()

    def onCheckBox_alternate(self, evt):
        if self.alternate_chkbox.GetValue():
            # We're going from False -> True
            self.regexShr_txtctrl.Disable()
            self.regexDiff_txtctrl.Disable()
        else:
            self.regexShr_txtctrl.Enable()
            self.regexDiff_txtctrl.Enable()

    def onCheckBox_varnumpages(self, evt):
        if self.varnumpages_chkbox.GetValue():
            self.numpages_txtctrl.Disable()
        else:
            self.numpages_txtctrl.Enable()

class DoubleSideDialog(wx.Dialog):
    def __init__(self, parent, *args, **kwargs):
        wx.Dialog.__init__(self, parent, title="Set Double Sided Properties", *args, **kwargs)
        
        self.num_pages = None
        self.regex = None
        self.is_alternating = None

        txt0 = wx.StaticText(self, label="Number of pages:")
        self.numpages_txtctrl = wx.TextCtrl(self, value="2")
        sizer0 = wx.BoxSizer(wx.HORIZONTAL)
        sizer0.AddMany([(txt0,), ((10,0),), (self.numpages_txtctrl,)])

        txt1 = wx.StaticText(self, label="Enter a regex to match on the file name.")
        self.regex_txtctrl = wx.TextCtrl(self, value=r".*-(.*)")
        sizer1 = wx.BoxSizer(wx.HORIZONTAL)
        sizer1.AddMany([(txt1,), ((10,0),), (self.regex_txtctrl,)])

        self.alternate_chkbox = wx.CheckBox(self, label="Ballots alternate front and back")

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
