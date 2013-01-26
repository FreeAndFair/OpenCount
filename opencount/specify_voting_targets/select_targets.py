import sys, os, time, math, pdb, traceback, threading, Queue, copy
import multiprocessing, csv
try:
    import cPickle as pickle
except ImportError:
    import pickle

from os.path import join as pathjoin

sys.path.append('..')

import wx, cv, numpy as np, Image, scipy, scipy.misc
from wx.lib.pubsub import Publisher
from wx.lib.scrolledpanel import ScrolledPanel

import util_gui, util
import grouping.tempmatch as tempmatch
import labelcontest.group_contests as group_contests
import pixel_reg.shared as shared
import pixel_reg.imagesAlign as imagesAlign
import global_align.global_align as global_align

class SelectTargetsMainPanel(wx.Panel):
    GLOBALALIGN_JOBID = util.GaugeID("GlobalAlignJobId")
    def __init__(self, parent, *args, **kwargs):
        wx.Panel.__init__(self, parent, *args, **kwargs)

        self.proj = None
        self.init_ui()

    def init_ui(self):
        self.seltargets_panel = SelectTargetsPanel(self)

        btn_getimgpath = wx.Button(self, label="Get Image Path...")
        btn_getimgpath.Bind(wx.EVT_BUTTON, self.onButton_getimgpath)

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.seltargets_panel, proportion=1, flag=wx.EXPAND)
        self.sizer.Add(btn_getimgpath)

        self.SetSizer(self.sizer)
        self.Layout()

    def start(self, proj, stateP, ocrtmpdir):
        self.proj = proj
        self.stateP = stateP
        # GROUP2BALLOT: dict {int groupID: [int ballotID_i, ...]}
        group2ballot = pickle.load(open(pathjoin(proj.projdir_path,
                                                 proj.group_to_ballots), 'rb'))
        group_exmpls = pickle.load(open(pathjoin(proj.projdir_path,
                                                 proj.group_exmpls), 'rb'))
        b2imgs = pickle.load(open(proj.ballot_to_images, 'rb'))
        img2page = pickle.load(open(pathjoin(proj.projdir_path,
                                             proj.image_to_page), 'rb'))
        self.img2flip = pickle.load(open(pathjoin(proj.projdir_path,
                                                  proj.image_to_flip), 'rb'))
        # 0.) Munge GROUP2BALLOT to list of lists of lists
        groups = []
        numtasks = 0
        for groupID, ballotids in sorted(group_exmpls.iteritems(), key=lambda t: t[0]):
            group = []
            for ballotid in ballotids:
                if len(group) >= 5:
                    break
                imgpaths = b2imgs[ballotid]
                imgpaths_ordered = sorted(imgpaths, key=lambda imP: img2page[imP])
                group.append(imgpaths_ordered)
            numtasks += 1
            groups.append(group)
        self.displayed_imgpaths = groups

        self.proj.addCloseEvent(self.save_session)
        self.proj.addCloseEvent(self.seltargets_panel.save_session)
        align_outdir = pathjoin(proj.projdir_path, 'groupsAlign_seltargs')

        class GlobalAlignThread(threading.Thread):
            def __init__(self, groups, img2flip, align_outdir, ocrtmpdir, 
                         manager, queue, callback, jobid, tlisten, *args, **kwargs):
                threading.Thread.__init__(self, *args, **kwargs)
                self.groups = groups
                self.img2flip = img2flip
                self.align_outdir = align_outdir
                self.ocrtmpdir = ocrtmpdir
                self.manager = manager
                self.queue = queue
                self.callback = callback
                self.jobid = jobid
                self.tlisten = tlisten
            def run(self):
                print '...Globally-aligning a subset of each partition...'
                t = time.time()
                groups_align_map = do_align_partitions(self.groups, self.img2flip,
                                                       self.align_outdir, self.manager, self.queue)
                dur = time.time() - t
                print '...Finished globally-aligning a subset of each partition ({0} s)'.format(dur)
                wx.CallAfter(Publisher().sendMessage, "signals.MyGauge.done", (self.jobid,))
                wx.CallAfter(self.callback, groups_align_map, self.ocrtmpdir)
                self.tlisten.stop()
        class ListenThread(threading.Thread):
            def __init__(self, queue, jobid, *args, **kwargs):
                threading.Thread.__init__(self, *args, **kwargs)
                self.queue = queue
                self.jobid = jobid
                self._stop = threading.Event()
            def stop(self):
                print "...ListenThread: Someone called my stop()..."
                self._stop.set()
            def is_stopped(self):
                return self._stop.isSet()
            def run(self):
                while True:
                    if self.is_stopped():
                        print "...ListenThread: Stopping."
                        return
                    try:
                        val = self.queue.get(block=True, timeout=1)
                        if val == True:
                            wx.CallAfter(Publisher().sendMessage, "signals.MyGauge.tick", (self.jobid,))
                    except Queue.Empty:
                        pass

        #if not os.path.exists(align_outdir):
        if not self.restore_session():
            manager = multiprocessing.Manager()
            queue = manager.Queue()
            tlisten = ListenThread(queue, self.GLOBALALIGN_JOBID)
            workthread = GlobalAlignThread(groups, self.img2flip, align_outdir, ocrtmpdir, 
                                           manager, queue, self.on_align_done, 
                                           self.GLOBALALIGN_JOBID, tlisten)
            workthread.start()
            tlisten.start()
            gauge = util.MyGauge(self, 1, thread=workthread, msg="Running Global Alignment...",
                                 job_id=self.GLOBALALIGN_JOBID)
            gauge.Show()
            wx.CallAfter(Publisher().sendMessage, "signals.MyGauge.nextjob", (numtasks, self.GLOBALALIGN_JOBID))
        else:
            # SelectTargets restores its self.partitions from stateP.
            seltargets_stateP = pathjoin(self.proj.projdir_path, '_state_selecttargets.p')
            self.seltargets_panel.start(None, self.img2flip, seltargets_stateP, ocrtmpdir)

    def on_align_done(self, groups_align_map, ocrtmpdir):
        groups_align = []
        for groupid in sorted(groups_align_map.keys()):
            ballots = groups_align_map[groupid]
            groups_align.append(ballots)
        # Order the displayed groups by size (smallest to largest)
        groups_sizes = map(lambda g: -len(g), groups_align)
        groups_sizes_argsort = np.argsort(groups_sizes)
        groups_align_bysize = [groups_align[i] for i in groups_sizes_argsort]
        self.i2groupid = groups_sizes_argsort
        seltargets_stateP = pathjoin(self.proj.projdir_path, '_state_selecttargets.p')
        self.seltargets_panel.start(groups_align_bysize, self.img2flip, seltargets_stateP, ocrtmpdir)

    def stop(self):
        self.proj.removeCloseEvent(self.save_session)
        self.proj.removeCloseEvent(self.seltargets_panel.save_session)
        self.save_session()
        self.seltargets_panel.save_session()
        self.export_results()

    def restore_session(self):
        try:
            state = pickle.load(open(self.stateP, 'rb'))
            self.i2groupid = state['i2groupid']
            self.displayed_imgpaths = state['displayed_imgpaths']
        except:
            return False
        return True

    def save_session(self):
        state = {'i2groupid': self.i2groupid,
                 'displayed_imgpaths': self.displayed_imgpaths}
        pickle.dump(state, open(self.stateP, 'wb'), pickle.HIGHEST_PROTOCOL)

    def export_results(self):
        """ For each group, export the locations of the voting
        targets to two locations:
            1.) A proj.target_locs pickle'd data structure
            2.) A dir of .csv files (for integration with LabelContests+
                InferContests).
        """
        try:
            os.makedirs(self.proj.target_locs_dir)
        except:
            pass
        group_targets_map = {} # maps {int groupID: [csvpath_side0, ...]}
        # TARGET_LOCS_MAP: maps {int groupID: {int page: [CONTEST_i, ...]}}, where each
        #     CONTEST_i is: [contestbox, targetbox_i, ...], where each
        #     box := [x1, y1, width, height, id, contest_id]
        target_locs_map = {}
        lonely_targets_map = {} # maps {int i: {int side: [TargetBox_i, ...]}}
        fields = ('imgpath', 'id', 'x', 'y', 'width', 'height', 'label', 'is_contest', 'contest_id')
        imgsize = None # Assumes all voted ballots are the same dimensions
        for i, boxes_sides in self.seltargets_panel.boxes.iteritems():
            group_idx = self.i2groupid[i]
            csvpaths = []
            for side, boxes in enumerate(boxes_sides):
                outpath = pathjoin(self.proj.target_locs_dir,
                                   "group_{0}_side_{1}.csv".format(group_idx, side))
                csvpaths.append(outpath)
                writer = csv.DictWriter(open(outpath, 'wb'), fields)
                # Make sure that TARGET_LOCS_MAP at least has something for this 
                # (To help out target extraction)
                target_locs_map.setdefault(group_idx, {}).setdefault(side, [])
                # BOX_ASSOCS: dict {int contest_id: [ContestBox, [TargetBox_i, ...]]}
                # LONELY_TARGETS: list [TargetBox_i, ...]
                box_assocs, lonely_targets = self.compute_box_ids(boxes)
                lonely_targets_map.setdefault(i, {}).setdefault(side, []).extend(lonely_targets)
                # For now, just grab one exemplar image from this group
                imgpath = self.seltargets_panel.partitions[i][0][side]
                if imgsize == None:
                    imgsize = cv.GetSize(cv.LoadImage(imgpath))
                rows_contests = []
                rows_targets = []
                id_c, id_t = 0, 0
                for contest_id, (contestbox, targetboxes) in box_assocs.iteritems():
                    x1_out, y1_out = contestbox.x1, contestbox.y1
                    w_out, h_out = contestbox.width, contestbox.height
                    # Make sure contest doesn't extend outside image.
                    x1_out = max(x1_out, 0)
                    y1_out = max(y1_out, 0)
                    if (x1_out + w_out) >= imgsize[0]:
                        w_out = imgsize[0] - x1_out - 1
                    if (y1_out + h_out) >= imgsize[1]:
                        h_out = imgsize[1] - y1_out - 1
                    rowC = {'imgpath': imgpath, 'id': id_c,
                            'x': x1_out, 'y': y1_out,
                            'width': w_out,
                            'height': h_out,
                            'label': '', 'is_contest': 1, 
                            'contest_id': contest_id}
                    rows_contests.append(rowC)
                    cbox = [x1_out, y1_out, w_out, h_out, id_c, contest_id]
                    curcontest = [] # list [contestbox, targetbox_i, ...]
                    curcontest.append(cbox)
                    id_c += 1
                    for box in targetboxes:
                        # Note: Ensure that all exported targets have the same dimensions,
                        # or risk breaking SetThreshold!
                        w, h = self.seltargets_panel.boxsize
                        x1_out, y1_out = box.x1, box.y1
                        # Don't let target extend outside the image
                        if (x1_out + w) >= imgsize[0]:
                            x1_out -= ((x1_out + w) - imgsize[0] + 1)
                        if (y1_out + h) >= imgsize[1]:
                            y1_out -= ((y1_out + h) - imgsize[1] + 1)
                        x1_out = max(x1_out, 0)
                        y1_out = max(y1_out, 0)
                        # Note: This doesn't necessarily guarantee that T
                        # is inside img bbox - however, since targets are
                        # small w.r.t image, this will always work.
                        rowT = {'imgpath': imgpath, 'id': id_t,
                               'x': x1_out, 'y': y1_out,
                               'width': w, 'height': h,
                               'label': '', 'is_contest': 0,
                               'contest_id': contest_id}
                        rows_targets.append(rowT)
                        tbox = [x1_out, y1_out, w, h, id_t, contest_id]
                        curcontest.append(tbox)
                        id_t += 1
                    target_locs_map.setdefault(group_idx, {}).setdefault(side, []).append(curcontest)
                writer.writerows(rows_contests + rows_targets)
            group_targets_map[group_idx] = csvpaths
        pickle.dump(group_targets_map, open(pathjoin(self.proj.projdir_path,
                                                     self.proj.group_targets_map), 'wb'),
                    pickle.HIGHEST_PROTOCOL)
        pickle.dump(target_locs_map, open(pathjoin(self.proj.projdir_path,
                                                   self.proj.target_locs_map), 'wb'),
                    pickle.HIGHEST_PROTOCOL)
        # Warn User about lonely targets.
        # TODO: Help the user out more for dealing with this case.
        _lst = []
        cnt = 0
        for i, targs_sidesMap in lonely_targets_map.iteritems():
            for side, targets in targs_sidesMap.iteritems():
                if targets:
                    print "...On Partition {0}, side {1}, there were {2} \
Lonely Targets - please check them out, or else they'll get ignored by \
LabelContests.".format(i, side, len(targets))
                    _lst.append("Partition={0} Side={1}".format(i, side))
                    cnt += len(targets)
        if _lst:
            dlg = wx.MessageDialog(self, message="Warning - there were {0} \
targets that were not enclosed in a contest. Please check them out, otherwise \
they'll get ignored by LabelContests. They are: {1}".format(cnt, str(_lst)),
                                   style=wx.OK)
            dlg.ShowModal()

    def compute_box_ids(self, boxes):
        """ Given a list of Boxes, some of which are Targets, others
        of which are Contests, geometrically compute the correct
        target->contest associations.
        Input:
            list BOXES:
        Output:
            dict ASSOCS. {int contest_id, [ContestBox, [TargetBox_i, ...]]}
        """
        def containing_box(box, boxes):
            """ Returns the box in BOXES that contains BOX. """
            w, h = box.width, box.height
            # Allow some slack when checking which targets are contained by a contest
            slack_fact = 0.1
            xEps = int(round(w*slack_fact))
            yEps = int(round(h*slack_fact))
            for i, otherbox in enumerate(boxes):
                if ((box.x1+xEps) >= otherbox.x1 and (box.y1+yEps) >= otherbox.y1
                        and (box.x2-xEps) <= otherbox.x2 and (box.y2-yEps) <= otherbox.y2):
                    return i, otherbox
            return None, None
        assocs = {}
        contests = [b for b in boxes if isinstance(b, ContestBox)]
        targets = [b for b in boxes if isinstance(b, TargetBox)]
        lonely_targets = []
        for t in targets:
            id, c = containing_box(t, contests)
            if id == None:
                print "Warning", t, "is not contained in any box."
                lonely_targets.append(t)
            elif id in assocs:
                assocs[id][1].append(t)
            else:
                assocs[id] = [c, [t]]
        return assocs, lonely_targets

    def onButton_getimgpath(self, evt):
        S = self.seltargets_panel
        cur_groupid = self.i2groupid[S.cur_i]
        imgpath = self.displayed_imgpaths[cur_groupid][S.cur_j][S.cur_page]
        print 'imgpath:', imgpath
        dlg = wx.MessageDialog(self, message="Displayed Imagepath: {0}".format(imgpath),
                               style=wx.OK)
        dlg.ShowModal()

class SelectTargetsPanel(ScrolledPanel):
    """ A widget that allows you to find voting targets on N ballot
    partitions
    """
    TEMPLATE_MATCH_JOBID = 830
    
    # TM_MODE_ALL: Run template matching on all images
    TM_MODE_ALL = 901
    # TM_MODE_POST: Run template matching only on images after (post) the
    #               currently-displayed group.
    TM_MODE_POST = 902

    def __init__(self, parent, *args, **kwargs):
        ScrolledPanel.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        # self.partitions: [[[imgpath_i0front, ...], ...], [[imgpath_i1front, ...], ...], ...]
        self.partitions = None
        # self.inv_map: {str imgpath: (int i, int j, int page)}
        self.inv_map = None
        # self.cur_i: Index of currently-displayed partition
        self.cur_i = None
        # self.cur_j: Index of currently-displayed image within partition CUR_I.
        self.cur_j = None
        # self.cur_page: Currently-displayed page
        self.cur_page = None

        # self.boxes: {int i: [[Box_iFront, ...], ...]}
        self.boxes = {}

        # BOXSIZE: (int w, int h), used to enforce that all voting targets
        # are the same size.
        self.boxsize = None

        # Sensitivity for Template Matching
        self.tm_param = 0.93
        # Window sizes for Smoothing
        self.win_ballot = (13, 13)
        self.win_target = (15, 15)
        self.tm_mode = self.TM_MODE_POST

        # STATEP: Path for state file.
        self.stateP = None

        self.toolbar = Toolbar(self)
        self.imagepanel = TargetFindPanel(self, self.do_tempmatch)

        txt = wx.StaticText(self, label="Select all Voting Targets from \
this partition.")

        btn_nextpartition = wx.Button(self, label="Next Partition...")
        btn_prevpartition = wx.Button(self, label="Previous Partition...")
        sizer_partitionbtns = wx.BoxSizer(wx.VERTICAL)
        sizer_partitionbtns.AddMany([(btn_nextpartition,), (btn_prevpartition,)])

        btn_nextimg = wx.Button(self, label="Next Ballot")
        btn_previmg = wx.Button(self, label="Previous Ballot")
        sizer_ballotbtns = wx.BoxSizer(wx.VERTICAL)
        sizer_ballotbtns.AddMany([(btn_nextimg,), (btn_previmg,)])

        btn_nextpage = wx.Button(self, label="Next Page")
        btn_prevpage = wx.Button(self, label="Previous Page")
        sizer_pagebtns = wx.BoxSizer(wx.VERTICAL)
        sizer_pagebtns.AddMany([(btn_nextpage,), (btn_prevpage,)])
        
        btn_nextpartition.Bind(wx.EVT_BUTTON, self.onButton_nextpartition)
        btn_prevpartition.Bind(wx.EVT_BUTTON, self.onButton_prevpartition)
        btn_nextimg.Bind(wx.EVT_BUTTON, self.onButton_nextimg)
        btn_previmg.Bind(wx.EVT_BUTTON, self.onButton_previmg)
        btn_nextpage.Bind(wx.EVT_BUTTON, self.onButton_nextpage)
        btn_prevpage.Bind(wx.EVT_BUTTON, self.onButton_prevpage)

        btn_jump_partition = wx.Button(self, label="Jump to Partition...")
        btn_jump_ballot = wx.Button(self, label="Jump to Ballot...")
        btn_jump_page = wx.Button(self, label="Jump to Page...")
        sizer_btn_jump = wx.BoxSizer(wx.HORIZONTAL)
        sizer_btn_jump.Add(btn_jump_partition, border=10, flag=wx.ALL)
        sizer_btn_jump.Add(btn_jump_ballot, border=10, flag=wx.ALL)
        sizer_btn_jump.Add(btn_jump_page, border=10, flag=wx.ALL)
        
        btn_jump_partition.Bind(wx.EVT_BUTTON, self.onButton_jump_partition)
        btn_jump_ballot.Bind(wx.EVT_BUTTON, self.onButton_jump_ballot)
        btn_jump_page.Bind(wx.EVT_BUTTON, self.onButton_jump_page)
        
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        btn_sizer.Add(sizer_partitionbtns, border=10, flag=wx.ALL)
        btn_sizer.Add(sizer_ballotbtns, border=10, flag=wx.ALL)
        btn_sizer.Add(sizer_pagebtns, border=10, flag=wx.ALL)
        btn_sizer.Add(sizer_btn_jump, border=10, flag=wx.ALL)

        txt1 = wx.StaticText(self, label="Partition: ")
        self.txt_curpartition = wx.StaticText(self, label="1")
        txt_slash0 = wx.StaticText(self, label=" / ")
        self.txt_totalpartitions = wx.StaticText(self, label="Foo")
        
        txt2 = wx.StaticText(self, label="Ballot: ")
        self.txt_curballot = wx.StaticText(self, label="1")
        txt_slash1 = wx.StaticText(self, label=" / ")
        self.txt_totalballots = wx.StaticText(self, label="Bar")
        
        txt3 = wx.StaticText(self, label="Page: ")
        self.txt_curpage = wx.StaticText(self, label="1")
        txt_slash2 = wx.StaticText(self, label=" / ")
        self.txt_totalpages = wx.StaticText(self, label="Baz")
        self.txt_curimgpath = wx.StaticText(self, label="")
        self.txt_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.txt_sizer.AddMany([(txt1,),
                                (self.txt_curpartition,), (txt_slash0,), (self.txt_totalpartitions,),
                                (50,0), (txt2,),
                                (self.txt_curballot,), (txt_slash1,), (self.txt_totalballots,),
                                (50,0), (txt3,),
                                (self.txt_curpage,), (txt_slash2,), (self.txt_totalpages,),
                                (50,0), (self.txt_curimgpath)])

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(txt, flag=wx.ALIGN_CENTER)
        self.sizer.Add(self.toolbar, flag=wx.EXPAND)
        self.sizer.Add(self.imagepanel, proportion=1, flag=wx.EXPAND)
        self.sizer.Add(btn_sizer, flag=wx.ALIGN_CENTER)
        self.sizer.Add(self.txt_sizer)

        self.SetSizer(self.sizer)

    def start(self, partitions, img2flip, stateP, ocrtempdir):
        """
        Input:
            list PARTITIONS: A list of lists of lists, encoding partition+ballot+side(s):
                [[[imgpath_i0_front, ...], ...], [[imgpath_i1_front, ...], ...], ...]
            dict IMG2FLIP: maps {str imgpath: bool isflipped}
            str STATEP: Path of the statefile.
            str OCRTEMPDIR: Used for InferContestRegion.
        """
        self.img2flip = img2flip
        self.stateP = stateP
        self.ocrtempdir = ocrtempdir
        if not self.restore_session():
            # 0.) Populate my self.INV_MAP
            self.partitions = partitions
            self.inv_map = {}
            self.boxes = {}
            self.boxsize = None
            for i, imgpaths in enumerate(self.partitions):
                for j, ballot in enumerate(imgpaths):
                    for page, imgpath in enumerate(ballot):
                        self.inv_map[imgpath] = i, j, page
                # (allows for variable-num pages)
                self.boxes[i] = [[] for _ in xrange(len(ballot))]
        # 1.) Update any StaticTexts in the UI.
        self.txt_totalpartitions.SetLabel(str(len(self.partitions)))
        self.txt_totalballots.SetLabel(str(len(self.partitions[0])))
        self.txt_totalpages.SetLabel(str(len(self.partitions[0][0])))
        self.txt_sizer.Layout()
        self.display_image(0, 0, 0, autofit=True)

        # 2.) Start in Target-Create mode.
        self.imagepanel.set_mode_m(BoxDrawPanel.M_CREATE)
        self.imagepanel.boxtype = TargetBox

    def restore_session(self):
        try:
            state = pickle.load(open(self.stateP, 'rb'))
            self.inv_map = state['inv_map']
            self.boxes = state['boxes']
            self.boxsize = state['boxsize']
            self.partitions = state['partitions']
        except:
            return False
        return True
            
    def save_session(self):
        state = {'inv_map': self.inv_map,
                 'boxes': self.boxes,
                 'boxsize': self.boxsize,
                 'partitions': self.partitions}
        pickle.dump(state, open(self.stateP, 'wb'), pickle.HIGHEST_PROTOCOL)

    def do_tempmatch(self, box, img):
        """ Runs template matching on all images within the current
        partition, using the BOX from IMG as the template.
        Input:
            Box BOX:
            PIL IMG:
        """
        # 1.) Do an autofit.
        patch_prefit = img.crop((box.x1, box.y1, box.x2, box.y2))
        patch = util_gui.fit_image(patch_prefit, padx=0, pady=0)
        patch_cv = pil2iplimage(patch)
        # 2.) Apply a smooth on PATCH (first adding a white border, to
        # avoid the smooth darkening PATCH, but brightening IMG).
        BRD = 20    # Amt. of white border
        patchB = cv.CreateImage((patch_cv.width+BRD, patch_cv.height+BRD), patch_cv.depth, patch_cv.channels)
        # Pass '0' as bordertype due to undocumented OpenCV flag IPL_BORDER_CONSTANT
        # being 0. Wow!
        cv.CopyMakeBorder(patch_cv, patchB, (BRD/2, BRD/2), 0, 255)
        xwin, ywin = self.win_target
        cv.Smooth(patchB, patchB, cv.CV_GAUSSIAN, param1=xwin, param2=ywin)
        # 2.a.) Copy the smooth'd PATCHB back into PATCH
        #patch_cv = cv.GetSubRect(patchB, (BRD/2, BRD/2, patch_cv.width, patch_cv.height))
        cv.SetImageROI(patchB, (BRD/2, BRD/2, patch_cv.width, patch_cv.height))
        patch = patchB
        #patch = iplimage2pil(patchB)
        #patch.save("_patch.png")        
        cv.SaveImage("_patch.png", patch)
        # 3.) Run template matching across all images in self.IMGPATHS,
        # using PATCH as the template.
        if self.tm_mode == self.TM_MODE_ALL:
            # Template match on /all/ images across all partitions, all pages
            imgpaths = sum([t for t in sum(self.partitions, [])], [])
        elif self.tm_mode == self.TM_MODE_POST:
            # Template match only on images after this partition (including 
            # this partition)
            imgpaths = sum([t for t in sum(self.partitions[self.cur_i:], [])], [])
            imgpaths = imgpaths[self.cur_page:] # Don't run on prior pages
        print "...Running template matching on {0} images...".format(len(imgpaths))
        queue = Queue.Queue()
        thread = TM_Thread(queue, self.TEMPLATE_MATCH_JOBID, patch, img,
                           imgpaths, self.tm_param, self.win_ballot, self.win_target,
                           self.on_tempmatch_done)
        thread.start()

    def on_tempmatch_done(self, results, w, h):
        """ Invoked after template matching computation is complete. 
        Input:
            dict RESULTS: maps {str imgpath: [(x1,y1,x2,y2,score_i), ...}. The matches
                that template matching discovered.
            int w: width of the patch
            int h: height of the patch
        """
        def is_overlap(rect1, rect2):
            def is_within_box(pt, box):
                return box.x1 < pt[0] < box.x2 and box.y1 < pt[1] < box.y2
            x1, y1, x2, y2 = rect1.x1, rect1.y1, rect1.x2, rect1.y2
            w, h = abs(x2-x1), abs(y2-y1)
            # Checks (in order): UL, UR, LR, LL corners
            return (is_within_box((x1,y1), rect2) or
                    is_within_box((x1+w,y1), rect2) or 
                    is_within_box((x1+w,y1+h), rect2) or 
                    is_within_box((x1,y1+h), rect2))
        def too_close(b1, b2):
            w, h = abs(b1.x1-b1.x2), abs(b1.y1-b1.y2)
            return ((abs(b1.x1 - b2.x1) <= w / 2.0 and
                     abs(b1.y1 - b2.y1) <= h / 2.0) or
                    is_overlap(b1, b2) or 
                    is_overlap(b2, b1))
        # 1.) Add the new matches to self.BOXES, but also filter out
        # any matches in RESULTS that are too close to previously-found
        # matches.
        for imgpath, matches in results.iteritems():
            partition_idx, j, page = self.inv_map[imgpath]
            for (x1, y1, x2, y2, score) in matches:
                boxB = TargetBox(x1, y1, x1+w, y1+h)
                # 1.a.) See if any already-existing box is too close
                do_add = True
                for boxA in self.boxes[partition_idx][page]:
                    if too_close(boxA, boxB):
                        do_add = False
                        break
                if do_add:
                    # 1.b.) Enforce constraint that all voting targets
                    #       are the same size.
                    if self.boxsize == None:
                        self.boxsize = (w, h)
                    else:
                        boxB.x2 = boxB.x1 + self.boxsize[0]
                        boxB.y2 = boxB.y1 + self.boxsize[1]
                    self.boxes.setdefault(partition_idx, [])[page].append(boxB)
        print 'Num boxes in current partition:', len(self.boxes[self.cur_i][self.cur_page])
        self.imagepanel.set_boxes(self.boxes[self.cur_i][self.cur_page])
        self.Refresh()
        print "...Finished adding results from tempmatch run."

    def display_image(self, i, j, page, autofit=False):
        """ Displays the J-th image in partition I. Also handles
        reading/saving in the currently-created boxes for the old/new image.
        If AUTOFIT is True, then this will auto-scale the image such that
        if fits entirely in the current client size.
        Input:
            int I: Which partition to display
            int J: Which image in partition I to display.
            int PAGE: Which page to display.
        Output:
            Returns the (I,J,PAGE) we decided to display, if successful.
        """
        if i < 0 or i >= len(self.partitions):
            print "Invalid partition idx:", i
            pdb.set_trace()
        elif j < 0 or j >= len(self.partitions[i]):
            print "Invalid image idx {0} into partition {1}".format(j, i)
            pdb.set_trace()
        # 0.) Save boxes of old image
        '''
        if self.cur_i != None:
            self.boxes.setdefault(self.cur_i, []).extend(self.imagepanel.boxes)
        '''
        self.cur_i, self.cur_j, self.cur_page = i, j, page
        imgpath = self.partitions[i][j][page]
        
        # 1.) Display New Image
        wximg = wx.Image(imgpath, wx.BITMAP_TYPE_ANY)
        if autofit:
            wP, hP = self.imagepanel.GetClientSize()
            w_img, h_img = wximg.GetWidth(), wximg.GetHeight()
            if w_img > h_img and w_img > wP:
                _c = w_img / float(wP)
                w_img_new = wP
                h_img_new = int(round(h_img / _c))
            elif w_img < h_img and h_img > hP:
                _c = h_img / float(hP)
                w_img_new = int(round(w_img / _c))
                h_img_new = hP
            self.imagepanel.set_image(wximg, size=(w_img_new, h_img_new))
        else:
            self.imagepanel.set_image(wximg)
        
        # 2.) Read in previously-created boxes for I (if exists)
        boxes = self.boxes.get(self.cur_i, [])[page]
        self.imagepanel.set_boxes(boxes)

        #self.SetupScrolling()
        # 3.) Finally, update relevant StaticText in the UI.
        self.txt_curimgpath.SetLabel(imgpath)
        self.txt_curpartition.SetLabel(str(self.cur_i+1))
        self.txt_curballot.SetLabel(str(self.cur_j+1))
        self.txt_curpage.SetLabel(str(self.cur_page+1))
        self.txt_sizer.Layout()
        self.Refresh()
        return (self.cur_i,self.cur_j,self.cur_page)

    def display_nextpartition(self):
        next_idx = self.cur_i + 1
        if next_idx >= len(self.partitions):
            return None
        self.txt_totalballots.SetLabel(str(len(self.partitions[next_idx])))
        self.txt_totalpages.SetLabel(str(len(self.partitions[next_idx][0])))
        return self.display_image(next_idx, 0, 0)
    def display_prevpartition(self):
        prev_idx = self.cur_i - 1
        if prev_idx < 0:
            return None
        self.txt_totalballots.SetLabel(str(len(self.partitions[prev_idx])))
        self.txt_totalpages.SetLabel(str(len(self.partitions[prev_idx][0])))
        return self.display_image(prev_idx, 0, 0)

    def display_nextimg(self):
        """ Displays the next image in the current partition. If the end
        of the list is reached, returns None, and does nothing. Else, 
        returns the new image index.
        """
        next_idx = self.cur_j + 1
        if next_idx >= len(self.partitions[self.cur_i]):
            return None
        self.txt_totalpages.SetLabel(str(len(self.partitions[self.cur_i][next_idx])))
        return self.display_image(self.cur_i, next_idx, self.cur_page)
    def display_previmg(self):
        prev_idx = self.cur_j - 1
        if prev_idx < 0:
            return None
        self.txt_totalpages.SetLabel(str(len(self.partitions[self.cur_i][prev_idx])))
        return self.display_image(self.cur_i, prev_idx, self.cur_page)
    def display_nextpage(self):
        next_idx = self.cur_page + 1
        if next_idx >= len(self.partitions[self.cur_i][self.cur_j]):
            return None
        return self.display_image(self.cur_i, self.cur_j, next_idx)
    def display_prevpage(self):
        prev_idx = self.cur_page - 1
        if prev_idx < 0:
            return None
        return self.display_image(self.cur_i, self.cur_j, prev_idx)
    def onButton_nextpartition(self, evt):
        self.display_nextpartition()
    def onButton_prevpartition(self, evt):
        self.display_prevpartition()
    def onButton_nextimg(self, evt):
        self.display_nextimg()
    def onButton_previmg(self, evt):
        self.display_previmg()
    def onButton_nextpage(self, evt):
        self.display_nextpage()
    def onButton_prevpage(self, evt):
        self.display_prevpage()
    def zoomin(self, amt=0.1):
        self.imagepanel.zoomin(amt=amt)
    def zoomout(self, amt=0.1):
        self.imagepanel.zoomout(amt=amt)

    def onButton_jump_partition(self, evt):
        dlg = wx.TextEntryDialog(self, "Which Group Number?", "Enter group number")
        status = dlg.ShowModal()
        if status == wx.ID_CANCEL:
            return
        try:
            idx = int(dlg.GetValue()) - 1
        except:
            print "Invalid index:", idx
            return
        if idx < 0 or idx >= len(self.partitions):
            print "Invalid group index:", idx
            return
        self.txt_totalballots.SetLabel(str(len(self.partitions[idx])))
        self.txt_totalpages.SetLabel(str(len(self.partitions[idx][0])))
        self.display_image(idx, 0, 0)

    def onButton_jump_ballot(self, evt):
        dlg = wx.TextEntryDialog(self, "Which Ballot Number?", "Enter Ballot number")
        status = dlg.ShowModal()
        if status == wx.ID_CANCEL:
            return
        try:
            idx = int(dlg.GetValue()) - 1
        except:
            print "Invalid index:", idx
            return
        if idx < 0 or idx >= len(self.partitions[self.cur_i]):
            print "Invalid ballot index:", idx
            return
        self.txt_totalpages.SetLabel(str(len(self.partitions[self.cur_i][idx])))
        self.display_image(self.cur_i, idx, 0)

    def onButton_jump_page(self, evt):
        dlg = wx.TextEntryDialog(self, "Which Page Number?", "Enter Page number")
        status = dlg.ShowModal()
        if status == wx.ID_CANCEL:
            return
        try:
            idx = int(dlg.GetValue()) - 1
        except:
            print "Invalid index:", idx
            return
        if idx < 0 or idx >= len(self.partitions[self.cur_i][self.cur_j]):
            print "Invalid page index:", idx
            return
        self.display_image(self.cur_i, self.cur_j, idx)

    def infercontests(self):
        imgpaths_exs = [] # list of [imgpath_i, ...]
        # Arbitrarily choose the first one Ballot from each partition
        for partition_idx, imgpaths_sides in enumerate(self.partitions):
            for imgpaths in imgpaths_sides:
                for side, imgpath in enumerate(imgpaths):
                    # Only add imgpaths that have boxes
                    if self.boxes[partition_idx][side]:
                        imgpaths_exs.append(imgpath)
                break
        # Since the ordering of these dataStructs encode semantic meaning,
        # and since I don't want to pass in an empty contest to InferContests
        # (it crashes), I have to manually remove all empty-pages from IMGPATHS_EXS
        # and TARGETS
        # Let i=target #, j=ballot style, k=contest idx:
        targets = [] # list of [[[box_ijk, ...], [box_ijk+1, ...], ...], ...]
        for partition_idx, boxes_sides in self.boxes.iteritems():
            for side, boxes in enumerate(boxes_sides):
                style_boxes = [] # [[contest_i, ...], ...]
                for box in boxes:
                    # InferContests throws out the pre-determined contest
                    # grouping, so just stick each target in its own
                    # 'contest'
                    if type(box) == TargetBox:
                        style_boxes.append([(box.x1, box.y1, box.x2, box.y2)])
                if style_boxes:
                    targets.append(style_boxes)

        # CONTEST_RESULTS: [[box_i, ...], ...], each subtuple_i is for imgpath_i.
        contest_results = group_contests.find_contests(self.ocrtempdir, imgpaths_exs, targets)
        # 1.) Update my self.BOXES
        for i, contests in enumerate(contest_results):
            partition_idx, j, page = self.inv_map[imgpaths_exs[i]]
            # Remove previous contest boxes
            justtargets = [b for b in self.boxes[partition_idx][page] if not isinstance(b, ContestBox)]
            contest_boxes = []
            for (x1,y1,x2,y2) in contests:
                contest_boxes.append(ContestBox(x1,y1,x2,y2))
            self.boxes[partition_idx][page] = justtargets+contest_boxes
        # 2.) Update self.IMAGEPANEL.BOXES (i.e. the UI)
        self.imagepanel.set_boxes(self.boxes[self.cur_i][self.cur_page])
        # 3.) Finally, update the self.proj.infer_bounding_boxes flag, 
        #     so that LabelContests does the right thing.
        self.GetParent().proj.infer_bounding_boxes = True
        self.Refresh()

class Toolbar(wx.Panel):
    def __init__(self, parent, *args, **kwargs):
        wx.Panel.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        
        self._setup_ui()
        self._setup_evts()
        self.Layout()

    def _setup_ui(self):
        self.btn_addtarget = wx.Button(self, label="Add Target")
        self.btn_forceaddtarget = wx.Button(self, label="Force Add Target")
        self.btn_addcontest = wx.Button(self, label="Add Contest")
        self.btn_modify = wx.Button(self, label="Modify")
        self.btn_zoomin = wx.Button(self, label="Zoom In")
        self.btn_zoomout = wx.Button(self, label="Zoom Out")
        self.btn_infercontests = wx.Button(self, label="Infer Contest Regions..")
        self.btn_opts = wx.Button(self, label="Options...")
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        btn_sizer.AddMany([(self.btn_addtarget,), (self.btn_forceaddtarget,), 
                           (self.btn_addcontest), (self.btn_modify,),
                           (self.btn_zoomin,), (self.btn_zoomout,),
                           (self.btn_infercontests,), (self.btn_opts,)])
        self.sizer.Add(btn_sizer)
        self.SetSizer(self.sizer)

    def _setup_evts(self):
        self.btn_addtarget.Bind(wx.EVT_BUTTON, self.onButton_addtarget)
        self.btn_forceaddtarget.Bind(wx.EVT_BUTTON, self.onButton_forceaddtarget)
        self.btn_addcontest.Bind(wx.EVT_BUTTON, self.onButton_addcontest)
        self.btn_modify.Bind(wx.EVT_BUTTON, lambda evt: self.setmode(BoxDrawPanel.M_IDLE))
        self.btn_zoomin.Bind(wx.EVT_BUTTON, lambda evt: self.parent.zoomin())
        self.btn_zoomout.Bind(wx.EVT_BUTTON, lambda evt: self.parent.zoomout())
        self.btn_infercontests.Bind(wx.EVT_BUTTON, lambda evt: self.parent.infercontests())
        self.btn_opts.Bind(wx.EVT_BUTTON, self.onButton_opts)
    def onButton_addtarget(self, evt):
        self.setmode(BoxDrawPanel.M_CREATE)
        self.parent.imagepanel.boxtype = TargetBox
    def onButton_forceaddtarget(self, evt):
        self.setmode(TargetFindPanel.M_FORCEADD_TARGET)
    def onButton_addcontest(self, evt):
        self.setmode(BoxDrawPanel.M_CREATE)
        self.parent.imagepanel.boxtype = ContestBox
    def setmode(self, mode_m):
        self.parent.imagepanel.set_mode_m(mode_m)
    def onButton_opts(self, evt):
        dlg = OptionsDialog(self)
        status = dlg.ShowModal()
        if status == wx.ID_CANCEL:
            return
        self.parent.tm_param = dlg.tm_param
        self.parent.win_ballot = dlg.win_ballot
        self.parent.win_target = dlg.win_target
        self.parent.tm_mode = dlg.tm_mode
        
class OptionsDialog(wx.Dialog):
    ID_APPLY = 42

    def __init__(self, parent, *args, **kwargs):
        wx.Dialog.__init__(self, parent, title="Options.", *args, **kwargs)
        self.parent = parent

        self.tm_param = None
        self.win_ballot = None
        self.win_target = None

        txt0 = wx.StaticText(self, label="Options for Template Matching.")

        tm_sizer = wx.BoxSizer(wx.HORIZONTAL)
        txt1 = wx.StaticText(self, label="Template Matching sensitivity: ")
        _val = str(self.parent.parent.tm_param)
        self.tm_param = wx.TextCtrl(self, value=_val)
        tm_sizer.AddMany([(txt1,), (self.tm_param,)])
        
        txt00 = wx.StaticText(self, label="Ballot Smoothing parameters (must be odd-integers).")
        txt01 = wx.StaticText(self, label="X-window size: ")
        txt02 = wx.StaticText(self, label="Y-window size: ")
        _val = str(self.parent.parent.win_ballot[0])
        self.xwin_ballot = wx.TextCtrl(self, value=_val)
        _val = str(self.parent.parent.win_ballot[1])        
        self.ywin_ballot = wx.TextCtrl(self, value=_val)
        sizer00 = wx.BoxSizer(wx.HORIZONTAL)
        sizer00.AddMany([(txt01,), (self.xwin_ballot,)])
        sizer01 = wx.BoxSizer(wx.HORIZONTAL)
        sizer01.AddMany([(txt02,), (self.ywin_ballot,)])
        sizer0 = wx.BoxSizer(wx.VERTICAL)
        sizer0.AddMany([(txt00,), (sizer00,), (sizer01,)])

        txt10 = wx.StaticText(self, label="Target Smoothing parameters (must be odd-integers)")
        txt11 = wx.StaticText(self, label="X-window size: ")
        txt12 = wx.StaticText(self, label="Y-window size: ")
        _val = str(self.parent.parent.win_target[0])
        self.xwin_target = wx.TextCtrl(self, value=_val)
        _val = str(self.parent.parent.win_target[1])
        self.ywin_target = wx.TextCtrl(self, value=_val)
        sizer10 = wx.BoxSizer(wx.HORIZONTAL)
        sizer10.AddMany([(txt11,), (self.xwin_target,)])
        sizer11 = wx.BoxSizer(wx.HORIZONTAL)
        sizer11.AddMany([(txt12,), (self.ywin_target,)])
        sizer1 = wx.BoxSizer(wx.VERTICAL)
        sizer1.AddMany([(txt10,), (sizer10,), (sizer11,)])

        txt_tm_mode = wx.StaticText(self, label="Template Matching Mode")
        self.radio_tm_mode_all = wx.RadioButton(self, label="Template match on all images", 
                                                style=wx.RB_GROUP)
        self.radio_tm_mode_post = wx.RadioButton(self, label="Template match only on images \
after (and including) the currently-displayed group.")
        if self.GetParent().GetParent().tm_mode == SelectTargetsPanel.TM_MODE_ALL:
            self.radio_tm_mode_all.SetValue(True)
        elif self.GetParent().GetParent().tm_mode == SelectTargetsPanel.TM_MODE_POST:
            self.radio_tm_mode_post.SetValue(True)
        sizer_tm_mode = wx.BoxSizer(wx.VERTICAL)
        sizer_tm_mode.AddMany([(txt_tm_mode,), (self.radio_tm_mode_all,),
                               (self.radio_tm_mode_post,)])

        btn_apply = wx.Button(self, label="Apply")
        btn_apply.Bind(wx.EVT_BUTTON, self.onButton_apply)
        btn_cancel = wx.Button(self, label="Cancel")
        btn_cancel.Bind(wx.EVT_BUTTON, self.onButton_cancel)
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        btn_sizer.AddMany([(btn_apply,), (btn_cancel,)])
        
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(txt0, flag=wx.ALIGN_CENTER)
        sizer.AddMany([(tm_sizer,), (sizer0,), (sizer1,), (sizer_tm_mode,), (btn_sizer, 0, wx.ALIGN_CENTER)])
        self.SetSizer(sizer)
        self.Fit()

    def onButton_apply(self, evt):
        self.tm_param = float(self.tm_param.GetValue())
        self.win_ballot = (int(self.xwin_ballot.GetValue()), int(self.ywin_ballot.GetValue()))
        self.win_target = (int(self.xwin_target.GetValue()), int(self.ywin_target.GetValue()))
        if self.radio_tm_mode_all.GetValue():
            self.tm_mode = SelectTargetsPanel.TM_MODE_ALL
        else:
            self.tm_mode = SelectTargetsPanel.TM_MODE_POST
        self.EndModal(OptionsDialog.ID_APPLY)
    def onButton_cancel(self, evt):
        self.EndModal(wx.ID_CANCEL)

class ImagePanel(ScrolledPanel):
    """ Basic widget class that display one image out of N image paths.
    Also comes with a 'Next' and 'Previous' button. Extend me to add
    more functionality (i.e. mouse-related events).
    """
    def __init__(self, parent, *args, **kwargs):
        ScrolledPanel.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        # self.img := a WxImage
        self.img = None
        # self.imgbitmap := A WxBitmap
        self.imgbitmap = None
        # self.npimg := A Numpy-version of an untarnished-version of self.imgbitmap
        self.npimg = None

        # self.scale: Scaling factor used to display self.IMGBITMAP
        self.scale = 1.0

        # If True, a signal to completely-redraw the original image
        self._imgredraw = False

        self._setup_ui()
        self._setup_evts()

    def _setup_ui(self):
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.sizer)

    def _setup_evts(self):
        self.Bind(wx.EVT_PAINT, self.onPaint)
        self.Bind(wx.EVT_LEFT_DOWN, self.onLeftDown)
        self.Bind(wx.EVT_LEFT_UP, self.onLeftUp)
        self.Bind(wx.EVT_MOTION, self.onMotion)
        self.Bind(wx.EVT_CHILD_FOCUS, self.onChildFocus)
        
    def set_image(self, img, size=None):
        """ Updates internal data-structures to allow viewing a new input
        IMG. If SIZE is given (width, height), then we will scale image
        to match SIZE, maintaining aspect ratio.
        """
        self.img = img
        
        c = size[0] / float(self.img.GetWidth()) if size else self.scale
        self.set_scale(c)
        
    def set_scale(self, scale):
        """ Changes scale, i.e. to acommodate zoom in/out. Mutates the
        self.IMGBITMAP.
        Input:
            float scale: Smaller values -> zoomed out images.
        """
        self.scale = scale
        w, h = self.img.GetWidth(), self.img.GetHeight()
        new_w, new_h = int(round(w*scale)), int(round(h*scale))
        self.imgbitmap = img_to_wxbitmap(self.img, (new_w, new_h))
        self.npimg = wxBitmap2np_v2(self.imgbitmap, is_rgb=True)

        self.sizer.Detach(0)
        self.sizer.Add(self.imgbitmap.GetSize())
        self.SetupScrolling()

        self.Refresh()

    def zoomin(self, amt=0.1):
        self.set_scale(self.scale + amt)
    def zoomout(self, amt=0.1):
        self.set_scale(self.scale - amt)

    def client_to_imgcoord(self, x, y):
        """ Transforms client (widget) coordinates to the Image
        coordinate system -- i.e. accounts for image scaling.
        Input:
            int (x,y): Client (UI) Coordinates.
        Output:
            int (X,Y), image coordinates.
        """
        return (int(round(x/self.scale)), int(round(y/self.scale)))
    def c2img(self, x, y):
        """ Convenience method to self.CLIENT_TO_IMGCOORD. """
        return self.client_to_imgcoord(x,y)
    def img_to_clientcoord(self, x, y):
        """ Transforms Image coords to client (widget) coords -- i.e.
        accounts for image scaling.
        Input:
            int (X,Y): Image coordinates.
        Output:
            int (x,y): Client (UI) coordinates.
        """
        return (int(round(x*self.scale)), int(round(y*self.scale)))
    def img2c(self, x, y):
        return self.img_to_clientcoord(x,y)

    def force_new_img_redraw(self):
        """ Forces this widget to completely-redraw self.IMG, even if
        self.imgbitmap contains modifications.
        """
        self._imgredraw = True

    def draw_image(self, dc):
        if not self.imgbitmap:
            return
        if self._imgredraw:
            # Draw the 'virgin' self.img
            self._imgredraw = False
            w, h = self.img.GetWidth(), self.img.GetHeight()
            new_w, new_h = int(round(w*self.scale)), int(round(h*self.scale))
            self.imgbitmap = img_to_wxbitmap(self.img, (new_w, new_h))

        dc.DrawBitmap(self.imgbitmap, 0, 0)
        return dc

    def onPaint(self, evt):
        if self.IsDoubleBuffered():
            dc = wx.PaintDC(self)
        else:
            dc = wx.BufferedPaintDC(self)
        self.PrepareDC(dc)
        self.draw_image(dc)

        return dc

    def onLeftDown(self, evt):
        x, y = self.CalcUnscrolledPosition(evt.GetPositionTuple())
        evt.Skip()

    def onMotion(self, evt):
        evt.Skip()

    def onChildFocus(self, evt):
        # If I don't override this child focus event, then wx will
        # reset the scrollbars at extremely annoying times. Weird.
        # For inspiration, see:
        #    http://wxpython-users.1045709.n5.nabble.com/ScrolledPanel-mouse-click-resets-scrollbars-td2335368.html
        pass

class BoxDrawPanel(ImagePanel):
    """ A widget that allows a user to draw boxes on a displayed image,
    and each image remembers its list of boxes.
    """

    """ Mouse Mouse:
        M_CREATE: Create a box on LeftDown.
        M_IDLE: Allow user to resize/move/select(multiple) boxes.
    """
    M_CREATE = 0
    M_IDLE = 1

    def __init__(self, parent, *args, **kwargs):
        ImagePanel.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        # self.boxes := [Box_i, ...]
        self.boxes = []

        # self.sel_boxes := [Box_i, ...]
        self.sel_boxes = []

        # Vars to keep track of box-being-created
        self.isCreate = False
        self.box_create = None

        # Vars for resizing behavior
        self.isResize = False
        self.box_resize = None
        self.resize_orient = None # 'N', 'NE', etc...

        # self.isDragging : Is the user moving-mouse while mouse-left-down
        # is held down?
        self.isDragging = False

        self.mode_m = BoxDrawPanel.M_CREATE

        # BOXTYPE: Class of the Box to create
        self.boxtype = Box
        
        # _x,_y keep track of last mouse position
        self._x, self._y = 0, 0

    def _setup_evts(self):
        ImagePanel._setup_evts(self)
        self.Bind(wx.EVT_KEY_DOWN, self.onKeyDown)

    def set_mode_m(self, mode):
        """ Sets my MouseMode. """
        self.mode_m = mode

    def set_boxes(self, boxes):
        self.boxes = boxes
        self.dirty_all_boxes()

    def startBox(self, x, y, boxtype=None):
        """ Starts creating a box at (x,y). """
        if boxtype == None:
            boxtype = self.boxtype
        print "...Creating Box: {0}, {1}".format((x,y), boxtype)
        self.isCreate = True
        self.box_create = boxtype(x, y, x+1, y+1)
        # Map Box coords to Image coords, not UI coords.
        self.box_create.scale(1 / self.scale)

    def finishBox(self, x, y):
        """ Finishes box creation at (x,y). """
        print "...Finished Creating Box:", (x,y)
        self.isCreate = False
        # 0.) Canonicalize box coords s.t. order is: UpperLeft, LowerRight.
        self.box_create.canonicalize()
        toreturn = self.box_create
        self.box_create = None
        self.dirty_all_boxes()
        return toreturn

    def set_scale(self, scale, *args, **kwargs):
        self.dirty_all_boxes()
        return ImagePanel.set_scale(self, scale, *args, **kwargs)

    def dirty_all_boxes(self):
        """ Signal to unconditionally-redraw all boxes. """
        for box in self.boxes:
            box._dirty = True
    
    def select_boxes(self, *boxes):
        for box in boxes:
            box.is_sel = True
        self.sel_boxes.extend(boxes)
        self.dirty_all_boxes()

    def clear_selected(self):
        """ Un-selects all currently-selected boxes, if any. """
        for box in self.sel_boxes:
            box.is_sel = False
        self.sel_boxes = []

    def delete_boxes(self, *boxes):
        """ Deletes the boxes in BOXES. """
        for box in boxes:
            self.boxes.remove(box)
            if box in self.sel_boxes:
                self.sel_boxes.remove(box)
        self.dirty_all_boxes()
        if not self.boxes:
            # Force a redraw of the image - otherwise, the last-removed
            # boxes don't go away.
            print "NO MORE BOXES, SHOULD REDRAW IMAGE"
            self.force_new_img_redraw()
            self.Refresh()

    def get_boxes_within(self, x, y, C=8.0, mode='any'):
        """ Returns a list of Boxes that are at most C units within
        the position (x,y), sorted by distance (increasing).
        Input:
            int (x,y):
        Output:
            list MATCHES, of the form:
                [(obj Box_i, float dist_i), ...]
        """

        results = []
        for box in self.boxes:
            if mode == 'N':
                x1, y1 = self.img2c((box.x1 + (box.width/2)), box.y1)
            elif mode == 'NE':
                x1, y1 = self.img2c(box.x1 + box.width, box.y1)
            elif mode == 'E':
                x1, y1 = self.img2c(box.x1 + box.width, box.y1 + (box.height/2))
            elif mode == 'SE':
                x1, y1 = self.img2c(box.x1 + box.width, box.y1 + box.height)
            elif mode == 'S':
                x1, y1 = self.img2c(box.x1 + (box.width/2), box.y1 + box.height)
            elif mode == 'SW':
                x1, y1 = self.img2c(box.x1, box.y1 + box.height)
            elif mode == 'W':
                x1, y1 = self.img2c(box.x1, box.y1 + (box.heigth/2))
            elif mode == 'NW':
                x1, y1 = self.img2c(box.x1, box.y1)
            else:
                # Default to 'any'
                x1, y1 = self.img2c(box.x1, box.y1)
                x2, y2 = self.img2c(box.x2, box.y2)
                if (x > x1 and x < x2 and
                    y > y1 and y < y2):
                    results.append((box, None))
                continue
            dist = distL2(x1, y1, x, y)
            if dist <= C:
                results.append((box, dist))
        if mode == 'any':
            return results
        results = sorted(results, key=lambda t: t[1])
        return results

    def get_box_to_resize(self, x, y, C=8.0):
        """ Returns a Box instance if the current mouse location is
        close enough to a resize location, or None o.w.
        Input:
            int X, Y: Mouse location.
        Output:
            Box or None.
        """
        results = [] # [[orient, box, dist], ...]
        for box in self.boxes:
            locs = {'N': self.img2c(box.x1 + (box.width/2), box.y1),
                    'NE': self.img2c(box.x1 + box.width, box.y1),
                    'E': self.img2c(box.x1 + box.width, box.y1 + (box.height/2)),
                    'SE': self.img2c(box.x1 + box.width, box.y1 + box.height),
                    'S': self.img2c(box.x1 + (box.width/2),box.y1 + box.height),
                    'SW': self.img2c(box.x1, box.y1 + box.height),
                    'W': self.img2c(box.x1, box.y1 + (box.height/2)),
                    'NW': self.img2c(box.x1, box.y1)}
            for (orient, (x1,y1)) in locs.iteritems():
                dist = distL2(x1,y1,x,y)
                if dist <= C:
                    results.append((orient, box, dist))
        if not results:
            return None, None
        results = sorted(results, key=lambda t: t[2])
        return results[0][1], results[0][0]

    def onLeftDown(self, evt):
        self.SetFocus()
        x, y = self.CalcUnscrolledPosition(evt.GetPositionTuple())
        
        box_resize, orient = self.get_box_to_resize(x, y)
        if self.mode_m == BoxDrawPanel.M_IDLE and box_resize:
            self.isResize = True
            self.box_resize = box_resize
            self.resize_orient = orient
            self.Refresh()
            return

        if self.mode_m == BoxDrawPanel.M_CREATE:
            print "...Creating Target box."
            self.clear_selected()
            self.startBox(x, y)
        elif self.mode_m == BoxDrawPanel.M_IDLE:
            boxes = self.get_boxes_within(x, y, mode='any')
            if boxes:
                b = boxes[0][0]
                if b not in self.sel_boxes:
                    self.clear_selected()
                    self.select_boxes(boxes[0][0])
            else:
                self.clear_selected()
                self.startBox(x, y, SelectionBox)

    def onLeftUp(self, evt):
        x, y = self.CalcUnscrolledPosition(evt.GetPositionTuple())
        self.isDragging = False
        if self.isResize:
            self.box_resize.canonicalize()
            self.box_resize = None
            self.isResize = False
            self.dirty_all_boxes()

        if self.mode_m == BoxDrawPanel.M_CREATE and self.isCreate:
            box = self.finishBox(x, y)
            self.boxes.append(box)
        elif self.mode_m == BoxDrawPanel.M_IDLE and self.isCreate:
            box = self.finishBox(x, y)
            boxes = get_boxes_within(self.boxes, box)
            print "...Selecting {0} boxes.".format(len(boxes))
            self.select_boxes(*boxes)
        self.Refresh()

    def onMotion(self, evt):
        x, y = self.CalcUnscrolledPosition(evt.GetPositionTuple())
        xdel, ydel = x - self._x, y - self._y
        self._x, self._y = x, y
        
        if self.isResize and evt.Dragging():
            xdel_img, ydel_img = self.c2img(xdel, ydel)
            if 'N' in self.resize_orient:
                self.box_resize.y1 += ydel_img
            if 'E' in self.resize_orient:
                self.box_resize.x2 += xdel_img
            if 'S' in self.resize_orient:
                self.box_resize.y2 += ydel_img
            if 'W' in self.resize_orient:
                self.box_resize.x1 += xdel_img
            self.Refresh()
            return

        if self.isCreate:
            self.box_create.x2, self.box_create.y2 = self.c2img(x, y)
            self.Refresh()
        elif self.sel_boxes and evt.Dragging():
            self.isDragging = True
            xdel_img, ydel_img = self.c2img(xdel, ydel)
            for box in self.sel_boxes:
                box.x1 += xdel_img
                box.y1 += ydel_img
                box.x2 += xdel_img
                box.y2 += ydel_img
            # Surprisingly, forcing a redraw for each mouse mvmt. is
            # a very fast operation! Very convenient.
            self.dirty_all_boxes()
            self.Refresh()

    def onKeyDown(self, evt):
        keycode = evt.GetKeyCode()
        if (keycode == wx.WXK_DELETE or keycode == wx.WXK_BACK):
            self.delete_boxes(*self.sel_boxes)
            self.Refresh()
        elif ((keycode in (wx.WXK_UP, wx.WXK_DOWN, wx.WXK_LEFT, wx.WXK_RIGHT))
              and self.sel_boxes):
            xdel, ydel = 0, 0
            if keycode == wx.WXK_UP:
                ydel -= 1
            elif keycode == wx.WXK_DOWN:
                ydel += 1
            elif keycode == wx.WXK_LEFT:
                xdel -= 1
            else:
                xdel += 1
            xdel_img, ydel_img = self.c2img(xdel, ydel)
            for box in self.sel_boxes:
                box.x1 += xdel_img
                box.y1 += ydel_img
                box.x2 += xdel_img
                box.y2 += ydel_img
            self.Refresh()

    def onPaint(self, evt):
        total_t = time.time()
        dc = ImagePanel.onPaint(self, evt)
        if self.isResize:
            dboxes = [b for b in self.boxes if b != self.box_resize]
        else:
            dboxes = self.boxes
        t = time.time()
        self.drawBoxes(dboxes, dc)
        dur = time.time() - t
        if self.isCreate:
            # Draw Box-Being-Created
            can_box = self.box_create.copy().canonicalize()
            self.drawBox(can_box, dc)
        if self.isResize:
            resize_box_can = self.box_resize.copy().canonicalize()
            self.drawBox(resize_box_can, dc)
        total_dur = time.time() - total_t
        #print "Total Time: {0:.5f}s  (drawBoxes: {1:.5f}s, {2:.4f}%)".format(total_dur, dur, 100*float(dur / total_dur))
        return dc
        
    def drawBoxes(self, boxes, dc):
        boxes_todo = [b for b in boxes if b._dirty]
        if not boxes_todo:
            return
        # First draw contests, then targets on-top.
        boxes_todo = sorted(boxes_todo, key=lambda b: 0 if type(b) == ContestBox else 1)
        npimg_cpy = self.npimg.copy()
        def draw_border(npimg, box, thickness=2, color=(0, 0, 0)):
            T = thickness
            clr = np.array(color)
            x1,y1,x2,y2 = box.x1, box.y1, box.x2, box.y2
            x1,y1 = self.img2c(x1,y1)
            x2,y2 = self.img2c(x2,y2)
            x1 = max(x1, 0)
            y1 = max(y1, 0)
            # Top
            npimg[y1:y1+T, x1:x2] *= 0.2
            npimg[y1:y1+T, x1:x2] += clr*0.8
            # Bottom
            npimg[max(0, (y2-T)):y2, x1:x2] *= 0.2
            npimg[max(0, (y2-T)):y2, x1:x2] += clr*0.8
            # Left
            npimg[y1:y2, x1:(x1+T)] *= 0.2
            npimg[y1:y2, x1:(x1+T)] += clr*0.8
            # Right
            npimg[y1:y2, max(0, (x2-T)):x2] *= 0.2
            npimg[y1:y2, max(0, (x2-T)):x2] += clr*0.8
            return npimg

        for box in boxes_todo:
            clr, thickness = box.get_draw_opts()
            draw_border(npimg_cpy, box, thickness=thickness, color=(0, 0, 0))
            if type(box) in (TargetBox, ContestBox) and box.is_sel:
                transparent_color = np.array(box.shading_selected_clr) if box.shading_selected_clr else None
            else:
                transparent_color = np.array(box.shading_clr) if box.shading_clr else None
            if transparent_color != None:
                t = time.time()
                _x1, _y1 = self.img2c(box.x1, box.y1)
                _x2, _y2 = self.img2c(box.x2, box.y2)
                np_rect = npimg_cpy[max(0, _y1):_y2, max(0, _x1):_x2]
                np_rect[:,:] *= 0.7
                np_rect[:,:] += transparent_color*0.3
                dur_wxbmp2np = time.time() - t
            
            box._dirty = False

        h, w = npimg_cpy.shape[:2]
        t = time.time()
        _image = wx.EmptyImage(w, h)
        _image.SetData(npimg_cpy.tostring())
        bitmap = _image.ConvertToBitmap()
        dur_img2bmp = time.time() - t

        self.imgbitmap = bitmap
        self.Refresh()

    def drawBox(self, box, dc):
        """ Draws BOX onto DC.
        Note: This draws directly to the PaintDC - this should only be done
        for user-driven 'dynamic' behavior (such as resizing a box), as
        drawing to the DC is much slower than just blitting everything to
        the self.imgbitmap.
        self.drawBoxes does all heavy-lifting box-related drawing in a single
        step.
        Input:
            list box: (x1, y1, x2, y2)
            wxDC DC:
        """
        total_t = time.time()

        t = time.time()
        dc.SetBrush(wx.TRANSPARENT_BRUSH)
        drawops = box.get_draw_opts()
        dc.SetPen(wx.Pen(*drawops))
        w = int(round(abs(box.x2 - box.x1) * self.scale))
        h = int(round(abs(box.y2 - box.y1) * self.scale))
        client_x, client_y = self.img2c(box.x1, box.y1)
        dc.DrawRectangle(client_x, client_y, w, h)
        dur_drawrect = time.time() - t

        transparent_color = np.array([200, 0, 0]) if isinstance(box, TargetBox) else np.array([0, 0, 200])
        if self.imgbitmap and type(box) in (TargetBox, ContestBox):
            t = time.time()
            _x1, _y1 = self.img2c(box.x1, box.y1)
            _x2, _y2 = self.img2c(box.x2, box.y2)
            _x1, _y1 = max(0, _x1), max(0, _y1)
            _x2, _y2 = min(self.imgbitmap.Width-1, _x2), min(self.imgbitmap.Height-1, _y2)
            if (_x2 - _x1) <= 1 or (_y2 - _y1) <= 1:
                return
            sub_bitmap = self.imgbitmap.GetSubBitmap((_x1, _y1, _x2-_x1, _y2-_y1))
            # I don't think I need to do a .copy() here...
            #np_rect = wxBitmap2np_v2(sub_bitmap, is_rgb=True).copy()
            np_rect = wxBitmap2np_v2(sub_bitmap, is_rgb=True)
            np_rect[:,:] *= 0.7
            np_rect[:,:] += transparent_color*0.3
            dur_wxbmp2np = time.time() - t

            _h, _w, channels = np_rect.shape

            t = time.time()
            _image = wx.EmptyImage(_w, _h)
            _image.SetData(np_rect.tostring())
            bitmap = _image.ConvertToBitmap()
            dur_img2bmp = time.time() - t

            t = time.time()
            memdc = wx.MemoryDC()
            memdc.SelectObject(bitmap)
            dc.Blit(client_x, client_y, _w, _h, memdc, 0, 0)
            memdc.SelectObject(wx.NullBitmap)
            dur_memdcBlit = time.time() - t

        t = time.time()
        if isinstance(box, TargetBox) or isinstance(box, ContestBox):
            # Draw the 'grabber' circles
            CIRCLE_RAD = 2
            dc.SetPen(wx.Pen("Black", 1))
            dc.SetBrush(wx.Brush("White"))
            dc.DrawCircle(client_x, client_y, CIRCLE_RAD)           # Upper-Left
            dc.DrawCircle(client_x+(w/2), client_y, CIRCLE_RAD)     # Top
            dc.DrawCircle(client_x+w, client_y, CIRCLE_RAD)         # Upper-Right
            dc.DrawCircle(client_x, client_y+(h/2), CIRCLE_RAD)     # Left
            dc.DrawCircle(client_x+w, client_y+(h/2), CIRCLE_RAD)   # Right
            dc.DrawCircle(client_x, client_y+h, CIRCLE_RAD)         # Lower-Left
            dc.DrawCircle(client_x+(w/2), client_y+h, CIRCLE_RAD)     # Bottom
            dc.DrawCircle(client_x+w, client_y+h, CIRCLE_RAD)           # Lower-Right
            dc.SetBrush(wx.TRANSPARENT_BRUSH)

        dur_drawgrabbers = time.time() - t

        total_dur = time.time() - total_t

        '''
        print "== drawBox Total {0:.6f}s (wxbmp2np: {1:.6f}s {2:.3f}%) \
(_img2bmp: {3:.6f}s {4:.3f}%) (memdc.blit {5:.3f}s {6:.3f}%) \
(drawrect: {7:.6f}s {8:.3f}%) (drawgrabbers {9:.6f} {10:.3f}%)".format(total_dur,
                                                                      dur_wxbmp2np,
                                                                      100*(dur_wxbmp2np / total_dur),
                                                                      dur_img2bmp,
                                                                      100*(dur_img2bmp / total_dur),
                                                                      dur_memdcBlit,
                                                                      100*(dur_memdcBlit / total_dur),
                                                                      dur_drawrect,
                                                                      100*(dur_drawrect / total_dur),
                                                                      dur_drawgrabbers,
                                                                      100*(dur_drawgrabbers / total_dur))
        '''
                                                                 
class TemplateMatchDrawPanel(BoxDrawPanel):
    """ Like a BoxDrawPanel, but when you create a Target box, it runs
    Template Matching to try to find similar instances.
    """
    def __init__(self, parent, tempmatch_fn, *args, **kwargs):
        BoxDrawPanel.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.tempmatch_fn = tempmatch_fn

    def onLeftUp(self, evt):
        x, y = self.CalcUnscrolledPosition(evt.GetPositionTuple())
        MIN_LEN = 13
        if self.mode_m == BoxDrawPanel.M_CREATE and self.isCreate:
            x_img, y_img = self.c2img(x,y)
            if (abs(self.box_create.x1 - x_img) <= MIN_LEN) or (abs(self.box_create.y1 - y_img) <= MIN_LEN):
                print "...User drew a too-small box..."
                dlg = wx.MessageDialog(self, style=wx.ID_OK,
                                       message="You drew a box that \
was too small. \n\
Either draw a bigger box, or zoom-in to better-select the targets.")
                self.Disable()
                dlg.ShowModal()
                self.Enable()
                self.isCreate = False
                self.box_create = None
                self.Refresh()
                return

            box = self.finishBox(x, y)
            if isinstance(box, TargetBox):
                imgpil = util_gui.imageToPil(self.img)
                imgpil = imgpil.convert('L')
                self.tempmatch_fn(box, imgpil)
            elif isinstance(box, ContestBox):
                self.boxes.append(box)
            self.Refresh()
        else:
            BoxDrawPanel.onLeftUp(self, evt)

class TargetFindPanel(TemplateMatchDrawPanel):
    M_FORCEADD_TARGET = 3

    def onLeftDown(self, evt):
        x, y = self.CalcUnscrolledPosition(evt.GetPositionTuple())
        x_img, y_img = self.c2img(x,y)
        w_img, h_img = self.img.GetSize()
        if x_img >= (w_img-1) or y_img >= (h_img-1):
            return
                                                        
        if self.mode_m == self.M_FORCEADD_TARGET:
            print "...Creating Forced Target."
            self.clear_selected()
            self.startBox(x, y)
            self.Refresh()
        else:
            TemplateMatchDrawPanel.onLeftDown(self, evt)

    def onLeftUp(self, evt):
        x, y = self.CalcUnscrolledPosition(evt.GetPositionTuple())
        # Restrict (x,y) to lie within the image
        w_img, h_img = self.img.GetSize()
        w_c, h_c = self.img2c(w_img-1, h_img-1)
        x = min(w_c, x)
        y = min(h_c, y)
        if self.mode_m == self.M_FORCEADD_TARGET and self.isCreate:
            # If this is the first-created box B, then make sure that 
            # subsequent-created boxes match the dimensions of B
            box = self.finishBox(x, y)
            if self.GetParent().boxsize == None:
                self.GetParent().boxsize = (box.width, box.height)
            else:
                w, h = self.GetParent().boxsize
                box.x2 = box.x1 + w
                box.y2 = box.y1 + h
            self.boxes.append(box)
            self.Refresh()
        else:
            TemplateMatchDrawPanel.onLeftUp(self, evt)        

class TM_Thread(threading.Thread):
    TEMPLATE_MATCH_JOBID = 48
    def __init__(self, queue, job_id, patch, img, imgpaths, tm_param,
                 win_ballot, win_target,
                 callback, *args, **kwargs):
        """
        Input:
            PATCH: An IplImage.
        """
        threading.Thread.__init__(self, *args, **kwargs)
        self.queue = queue
        self.job_id = job_id
        self.patch = patch
        self.img = img
        self.imgpaths = imgpaths
        self.tm_param = tm_param
        self.win_ballot = win_ballot
        self.win_target = win_target
        self.callback = callback
    def run(self):
        print "...running template matching..."
        t = time.time()
        #patch_str = self.patch.tostring()
        w, h = cv.GetSize(self.patch)
        # results: {str imgpath: [(x,y,score_i), ...]}
        #results = partask.do_partask(do_find_matches, self.imgpaths, 
        #                             _args=(patch_str, w, h, self.tm_param, self.win_ballot),
        #                             combfn='dict', singleproc=False)
        xwinB, ywinB = self.win_ballot
        xwinT, ywinT = self.win_target
        # results: {str imgpath: [(x1,y1,x2,y2,score_i), ...]}
        # Note: self.patch is already smooth'd.
        results = tempmatch.get_tempmatches_par(self.patch, self.imgpaths,
                                                do_smooth=tempmatch.SMOOTH_IMG_BRD,
                                                T=self.tm_param, xwinA=xwinT, ywinA=ywinT,
                                                xwinI=xwinB, ywinI=ywinB)
        dur = time.time() - t
        print "...finished running template matching ({0} s).".format(dur)
        self.callback(results, w, h)

class Box(object):
    # SHADING: (int R, int G, int B)
    #     (Optional) color of transparent shading for drawing
    shading_clr = None
    shading_selected_clr = None

    def __init__(self, x1, y1, x2, y2):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self._dirty = True
    @property
    def width(self):
        return abs(self.x1-self.x2)
    @property
    def height(self):
        return abs(self.y1-self.y2)
    def __str__(self):
        return "Box({0},{1},{2},{3})".format(self.x1, self.y1, self.x2, self.y2)
    def __repr__(self):
        return "Box({0},{1},{2},{3})".format(self.x1, self.y1, self.x2, self.y2)
    def __eq__(self, o):
        return (isinstance(o, Box) and self.x1 == o.x1 and self.x2 == o.x2
                and self.y1 == o.y1 and self.y2 == o.y2)
    def canonicalize(self):
        """ Re-arranges my points (x1,y1),(x2,y2) such that we get:
            (x_upperleft, y_upperleft, x_lowerright, y_lowerright)
        """
        xa, ya, xb, yb = self.x1, self.y1, self.x2, self.y2
        w, h = abs(xa - xb), abs(ya - yb)
        if xa < xb and ya < yb:
            # UpperLeft, LowerRight
            self.x1, self.y1 = xa, ya
            self.x2, self.y2 = xb, yb
        elif xa < xb and ya > yb:
            # LowerLeft, UpperRight
            self.x1, self.y1 = xa, ya - h,
            self.x2, self.y2 = xb, yb + h
        elif xa > xb and ya < yb:
            # UpperRight, LowerLeft
            self.x1, self.y1 = xa - w, ya
            self.x2, self.y2 = xb + w, yb
        else:
            # LowerRight, UpperLeft
            self.x1, self.y1 = xb, yb
            self.x2, self.y2 = xa, ya
        return self
    def scale(self, scale):
        self.x1 = int(round(self.x1*scale))
        self.y1 = int(round(self.y1*scale))
        self.x2 = int(round(self.x2*scale))
        self.y2 = int(round(self.y2*scale))
    def copy(self):
        return Box(self.x1, self.y1, self.x2, self.y2)
    def get_draw_opts(self):
        """ Given the state of me, return the color+line-width for the
        DC to use.
        """
        return ("Green", 2)
    def marshall(self):
        return {'x1': self.x1, 'y1': self.y1, 'x2': self.x2, 'y2': self.y2}

class TargetBox(Box):
    shading_clr = (0, 255, 0) # Green
    shading_selected_clr = (255, 0, 0) # Red

    def __init__(self, x1, y1, x2, y2, is_sel=False):
        Box.__init__(self, x1, y1, x2, y2)
        self.is_sel = is_sel
    def __str__(self):
        return "TargetBox({0},{1},{2},{3},is_sel={4})".format(self.x1, self.y1, self.x2, self.y2, self.is_sel)
    def __repr__(self):
        return "TargetBox({0},{1},{2},{3},is_sel={4})".format(self.x1, self.y1, self.x2, self.y2, self.is_sel)
    def get_draw_opts(self):
        """ Given the state of me, return the color+line-width for the
        DC to use.
        """
        if self.is_sel:
            return ("Yellow", 1)
        else:
            return ("Green", 1)
    def copy(self):
        return TargetBox(self.x1, self.y1, self.x2, self.y2, is_sel=self.is_sel)
class ContestBox(Box):
    shading_clr = (0, 0, 200) # Blue
    shading_selected_clr = (161, 0, 240) # Purple

    def __init__(self, x1, y1, x2, y2, is_sel=False):
        Box.__init__(self, x1, y1, x2, y2)
        self.is_sel = is_sel
    def __str__(self):
        return "ContestBox({0},{1},{2},{3},is_sel={4})".format(self.x1, self.y1, self.x2, self.y2, self.is_sel)
    def __repr__(self):
        return "ContestBox({0},{1},{2},{3},is_sel={4})".format(self.x1, self.y1, self.x2, self.y2, self.is_sel)
    def get_draw_opts(self):
        """ Given the state of me, return the color+line-width for the
        DC to use.
        """
        if self.is_sel:
            return ("Yellow", 1)
        else:
            return ("Blue", 1)
    def copy(self):
        return ContestBox(self.x1, self.y1, self.x2, self.y2, is_sel=self.is_sel)
    
class SelectionBox(Box):
    def __str__(self):
        return "SelectionBox({0},{1},{2},{3})".format(self.x1, self.y1, self.x2, self.y2)
    def __repr__(self):
        return "SelectionBox({0},{1},{2},{3})".format(self.x1, self.y1, self.x2, self.y2)
    def get_draw_opts(self):
        return ("Black", 1)
    def copy(self):
        return SelectionBox(self.x1, self.y1, self.x2, self.y2)

def canonicalize_box(box):
    """ Takes two arbitrary (x,y) points and re-arranges them
    such that we get:
        (x_upperleft, y_upperleft, x_lowerright, y_lowerright)
    """
    xa, ya, xb, yb = box
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

def get_boxes_within(boxes, box):
    """ Returns all boxes in BOXES that lie within BOX.
    Input:
        list boxes: [Box_i, ...]
        Box box: Enclosing box.
    Output:
        list outboxes.
    """
    result = []
    for boxA in boxes:
        wA, hA = int(abs(boxA.x1-boxA.x2)), int(abs(boxA.y1-boxA.y2))
        if (((boxA.x1+(wA/3)) >= box.x1) and
            ((boxA.x2-(wA/3)) <= box.x2) and
            ((boxA.y1+(hA/3)) >= box.y1) and
            ((boxA.y2-(hA/3)) <= box.y2)):
            result.append(boxA)
    return result

def expand_box(box, factor, bounds=None):
    """ Expands the Box BOX by FACTOR in each dimension. If BOUNDS is
    given, then this dictates the maximum (width,height) allowed.
    Input:
        Box BOX:
        float FACTOR: If 0.0, same size. 0.5 is 50% bigger, etc.
        list BOUNDS: (w,h)
    Output:
        Box OUTBOX.
    """
    b = box.copy()
    b.x1 = int(round(max(0, box.x1 - (box.width*factor))))
    b.y1 = int(round(max(0, box.y1 - (box.height*factor))))
    if bounds != None:
        b.x2 = int(round(min(bounds[0]-1, box.x2 + (box.width*factor))))
        b.y2 = int(round(min(bounds[1]-1, box.y2 + (box.height*factor))))
    else:
        b.x2 = int(round(box.x2 + (box.width*factor)))
        b.y2 = int(round(box.y2 + (box.height*factor)))
    return b

def img_to_wxbitmap(img, size=None):
    """ Converts IMG to a wxBitmap. """
    # TODO: Assumes that IMG is a wxImage
    if size:
        img_scaled = img.Scale(size[0], size[1], quality=wx.IMAGE_QUALITY_HIGH)
    else:
        img_scaled = img
    return wx.BitmapFromImage(img_scaled)
def pil2iplimage(img):
    """ Converts a (grayscale) PIL img IMG to a CV IplImage. """
    img_cv = cv.CreateImageHeader(map(int, img.size), cv.IPL_DEPTH_8U, 1)
    cv.SetData(img_cv, img.tostring())
    return img_cv
def iplimage2pil(img):
    """ Converts a (grayscale) CV IplImage to a PIL image. """
    return Image.fromstring("L", cv.GetSize(img), img.tostring())

def bestmatch(A, B):
    """ Tries to find the image A within the (larger) image B.
    For instance, A could be a voting target, and B could be a
    contest.
    Input:
        IplImage A: Patch to search for
        IplImage B: Image to search over
    Output:
        ((x,y), s_mat),  location on B of the best match for A.
    """
    w_A, h_A = A.width, A.height
    w_B, h_B = B.width, B.height
    s_mat = cv.CreateMat(h_B - h_A + 1, w_B - w_A + 1, cv.CV_32F)
    cv.MatchTemplate(B, A, s_mat, cv.CV_TM_CCOEFF_NORMED)
    minResp, maxResp, minLoc, maxLoc = cv.MinMaxLoc(s_mat)
    return maxLoc, s_mat

def align_partitions(partitions, (outrootdir, img2flip), queue=None, result_queue=None):
    """ 
    Input:
        list PARTITIONS: [[partitionID, [Ballot_i, ...]], [partitionID, [Ballot_i, ...]], ...]
        str OUTROOTDIR: Rootdir to save aligned images to.
    Output:
        dict PARTITIONS_ALIGN: {int partitionID: [BALLOT_i, ...]}
    """
    # Global Alignment approach: Perform alignment on a smaller patch
    # near the center, then apply the discovered transformation H to
    # the entire image. Works better than working on the entire image.
    partitions_align = {} # maps {partitionID: [[imgpath_i, ...], ...]}
    t = time.time()
    print "...this process is aligning {0} ballots...".format(sum(map(lambda t: len(t[1]), partitions), 0))
    try:
        for idx, (partitionid, ballots) in enumerate(partitions):
            outdir = pathjoin(outrootdir, 'partition_{0}'.format(partitionid))
            try: os.makedirs(outdir)
            except: pass
            ballotRef = ballots[0]
            Irefs = []
            for side, imP in enumerate(ballotRef):
                I = shared.standardImread(imP, flatten=True)
                if img2flip[imP]:
                    I = shared.fastFlip(I)
                Irefs.append((imP, I))
            # 0.) First, handle the reference Ballot
            curBallot = []
            for side, (Iref_imP, Iref) in enumerate(Irefs):
                outname = 'bal_{0}_side_{1}.png'.format(0, side)
                outpath = pathjoin(outdir, outname)
                scipy.misc.imsave(outpath, Iref)
                curBallot.append(outpath)
            partitions_align[partitionid] = [curBallot]
            # 1.) Now, align all other Ballots to BALLOTREF
            for i, ballot in enumerate(ballots[1:]):
                curBallot = []
                for side, imgpath in enumerate(ballot):
                    Iref_imgP, Iref = Irefs[side]
                    I = shared.standardImread(imgpath, flatten=True)
                    if img2flip[imgpath]:
                        I = shared.fastFlip(I)
                    H, Ireg, err = global_align.align_image(Iref, I)
                    #H, Ireg, err = global_align.align_strong(I, Iref, crop_Iref=(0.05, 0.05, 0.05, 0.05),
                    #                                         do_nan_to_num=True)
                    outname = 'bal_{0}_side_{1}.png'.format(i + 1, side)
                    outpath = pathjoin(outdir, outname)
                    scipy.misc.imsave(outpath, Ireg)
                    curBallot.append(outpath)

                partitions_align[partitionid].append(curBallot)
            if queue:
                queue.put(True)
        dur = time.time() - t
        if result_queue:
            result_queue.put(partitions_align)
        return partitions_align
    except:
        traceback.print_exc()
        if result_queue:
            result_queue.put({})
        return None

def do_align_partitions(partitions, img2flip, outrootdir, manager, queue):
    """
    Input:
        list PARTITIONS[i][j][k] := i-th partition, j-th ballot, k-th side. 
        dict IMG2FLIP: maps {str imgpath: bool isflipped}
    Output:
        dict PARTITIONS_ALIGN. maps {int partitionID: [[imgpath_i, ...], ...]}
    """
    try:
        N = min(multiprocessing.cpu_count(), len(partitions))
        # Evenly-distribute partitions by partition size.
        partitions_evenly = divy_lists(partitions, N)
        pool = multiprocessing.Pool()
        result_queue = manager.Queue()

        for i,task in enumerate(partitions_evenly):
            # TASK := [[partitionID, [Ballot_i, ...]], [partitionID, [Ballot_i, ...]], ...]
            pool.apply_async(align_partitions, args=(task, (outrootdir, img2flip), 
                                                     queue, result_queue))
        pool.close()
        pool.join()

        cnt = 0; num_tasks = len(partitions_evenly)
        partitions_align = {} 
        while cnt < num_tasks:
            subresult = result_queue.get()
            print '...got result {0}...'.format(cnt)
            partitions_align = dict(partitions_align.items() + subresult.items())
            cnt += 1
        return partitions_align
    except:
        traceback.print_exc()
        return None

def divy_lists(lst, N):
    """ Given a list of sublists (where each sublist may be of unequal
    size), output N lists of list of sublists, where the size of each 
    list of sublists is maximized (i.e. ensuring an equal distribution
    of sublist sizes).
    Input:
        list LST[i][j] := j-th element of i-th sublist
    Output:
        list OUT[i][j][k] := k-th element of j-th sublist within i-th list.
    """
    if len(lst) <= N:
        return [[[i, l]] for i, l in enumerate(lst)]
    outlst = [None]*N
    lst_np = np.array(lst)
    lstlens = map(lambda l: -len(l), lst_np)
    lstlens_argsort = np.argsort(lstlens)
    for i, lst_idx in enumerate(lstlens_argsort):
        sublist = lst[lst_idx]
        out_idx = i % N
        if outlst[out_idx] == None:
            outlst[out_idx] = [[lst_idx, sublist]]
        else:
            outlst[out_idx].append([lst_idx, sublist])
    return outlst

def wxImage2np(Iwx, is_rgb=True):
    """ Converts wxImage to numpy array """
    w, h = Iwx.GetSize()
    Inp_flat = np.frombuffer(Iwx.GetDataBuffer(), dtype='uint8')
    if is_rgb:
        Inp = Inp_flat.reshape(h,w,3)
    else:
        Inp = Inp_flat.reshape(h,w)
    return Inp
def wxBitmap2np(wxBmp, is_rgb=True):
    """ Converts wxBitmap to numpy array """
    total_t = time.time()

    t = time.time() 
    Iwx = wxBmp.ConvertToImage()
    dur_bmp2wximg = time.time() - t
    
    t = time.time()
    npimg = wxImage2np(Iwx, is_rgb=True)
    dur_wximg2np = time.time() - t
    

    total_dur = time.time() - total_t
    print "==== wxBitmap2np: {0:.6f}s (bmp2wximg: {1:.5f}s {2:.3f}%) \
(wximg2np: {3:.5f}s {4:.3f}%)".format(total_dur,
                                      dur_bmp2wximg,
                                      100*(dur_bmp2wximg / total_dur),
                                      dur_wximg2np,
                                      100*(dur_wximg2np / total_dur))
    return npimg
def wxBitmap2np_v2(wxBmp, is_rgb=True):
    """ Converts wxBitmap to numpy array """
    total_t = time.time()
    
    w, h = wxBmp.GetSize()

    npimg = np.zeros(h*w*3, dtype='uint8')
    wxBmp.CopyToBuffer(npimg, format=wx.BitmapBufferFormat_RGB)
    npimg = npimg.reshape(h,w,3)

    total_dur = time.time() - total_t
    #print "==== wxBitmap2np_v2: {0:.6f}s".format(total_dur)
    return npimg
    
def isimgext(f):
    return os.path.splitext(f)[1].lower() in ('.png', '.bmp', 'jpeg', '.jpg', '.tif')

def distL2(x1,y1,x2,y2):
    return math.sqrt((float(y1)-y2)**2.0 + (float(x1)-x2)**2.0)

def main():
    class TestFrame(wx.Frame):
        def __init__(self, parent, partitions, *args, **kwargs):
            wx.Frame.__init__(self, parent, size=(800, 900), *args, **kwargs)
            self.parent = parent
            self.partitions = partitions

            self.st_panel = SelectTargetsPanel(self)
            self.st_panel.start(partitions)

    args = sys.argv[1:]
    imgsdir = args[0]
    try:
        mode = args[1]
    except:
        mode = 'single'
    partitions = []
    for dirpath, _, filenames in os.walk(imgsdir):
        partition = []
        imgpaths = [f for f in filenames if isimgext(f)]
        if mode == 'single':
            for imgname in [f for f in filenames if isimgext(f)]:
                partition.append([os.path.join(dirpath, imgname)])
        else:
            imgpaths = util.sorted_nicely(imgpaths)
            for i, imgname in enumerate(imgpaths[:-1:2]):
                page1 = os.path.join(dirpath, imgname)
                page2 = os.path.join(dirpath, imgpaths[i+1])
                partition.append([page1, page2])
        if partition:
            partitions.append(partition)
    app = wx.App(False)
    f = TestFrame(None, partitions)
    f.Show()
    app.MainLoop()

if __name__ == '__main__':
    main()
