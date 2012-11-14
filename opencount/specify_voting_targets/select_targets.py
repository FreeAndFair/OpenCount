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
import grouping.partask as partask
import labelcontest.group_contests as group_contests
import pixel_reg.shared as shared
import pixel_reg.imagesAlign as imagesAlign

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
        # GROUP2BALLOT: dict {int groupID: [int ballotID_i, ...]}
        group2ballot = pickle.load(open(pathjoin(proj.projdir_path,
                                                 proj.group_to_ballots), 'rb'))
        group_exmpls = pickle.load(open(pathjoin(proj.projdir_path,
                                                 proj.group_exmpls), 'rb'))
        b2imgs = pickle.load(open(proj.ballot_to_images, 'rb'))
        img2page = pickle.load(open(pathjoin(proj.projdir_path,
                                             proj.image_to_page), 'rb'))
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

        self.proj.addCloseEvent(self.seltargets_panel.save_session)
        align_outdir = pathjoin(proj.projdir_path, 'groupsAlign_seltargs')

        class GlobalAlignThread(threading.Thread):
            def __init__(self, groups, align_outdir, stateP, ocrtmpdir, 
                         manager, queue, callback, jobid, tlisten, *args, **kwargs):
                threading.Thread.__init__(self, *args, **kwargs)
                self.groups = groups
                self.align_outdir = align_outdir
                self.stateP = stateP
                self.ocrtmpdir = ocrtmpdir
                self.manager = manager
                self.queue = queue
                self.callback = callback
                self.jobid = jobid
                self.tlisten = tlisten
            def run(self):
                print '...Globally-aligning a subset of each partition...'
                t = time.time()
                groups_align_map = do_align_partitions(self.groups, self.align_outdir, self.manager, self.queue)
                dur = time.time() - t
                print '...Finished globally-aligning a subset of each partition ({0} s)'.format(dur)
                wx.CallAfter(Publisher().sendMessage, "signals.MyGauge.done", (self.jobid,))
                wx.CallAfter(self.callback, groups_align_map, self.stateP, self.ocrtmpdir)
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

        if not os.path.exists(align_outdir):
            manager = multiprocessing.Manager()
            queue = manager.Queue()
            tlisten = ListenThread(queue, self.GLOBALALIGN_JOBID)
            workthread = GlobalAlignThread(groups, align_outdir, stateP, ocrtmpdir, 
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
            self.seltargets_panel.start(None, stateP, ocrtmpdir)

    def on_align_done(self, groups_align_map, stateP, ocrtmpdir):
        groups_align = []
        for groupid in sorted(groups_align_map.keys()):
            ballots = groups_align_map[groupid]
            groups_align.append(ballots)
        self.seltargets_panel.start(groups_align, stateP, ocrtmpdir)

    def stop(self):
        self.proj.removeCloseEvent(self.seltargets_panel.save_session)
        self.seltargets_panel.save_session()
        self.export_results()

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
        fields = ('imgpath', 'id', 'x', 'y', 'width', 'height', 'label', 'is_contest', 'contest_id')
        for group_idx, boxes_sides in self.seltargets_panel.boxes.iteritems():
            csvpaths = []
            for side, boxes in enumerate(boxes_sides):
                outpath = pathjoin(self.proj.target_locs_dir,
                                   "group_{0}_side_{1}.csv".format(group_idx, side))
                csvpaths.append(outpath)
                writer = csv.DictWriter(open(outpath, 'wb'), fields)

                # BOX_ASSOCS: dict {int contest_id: [ContestBox, [TargetBox_i, ...]]}
                box_assocs = self.compute_box_ids(boxes)
                # TODO: For now, just grab one exemplar image from this group
                imgpath = self.seltargets_panel.partitions[group_idx][0][side]
                rows_contests = [] 
                rows_targets = []
                id_c, id_t = 0, 0
                for contest_id, (contestbox, targetboxes) in box_assocs.iteritems():
                    rowC = {'imgpath': imgpath, 'id': id_c,
                            'x': contestbox.x1, 'y': contestbox.y1,
                            'width': contestbox.x2-contestbox.x1,
                            'height': contestbox.y2-contestbox.y1,
                            'label': '', 'is_contest': 1, 
                            'contest_id': contest_id}
                    rows_contests.append(rowC)
                    cbox = [contestbox.x1, contestbox.y1,
                            contestbox.x2 - contestbox.x1,
                            contestbox.y2 - contestbox.y1,
                            id_c, contest_id]
                    curcontest = [] # list [contestbox, targetbox_i, ...]
                    curcontest.append(cbox)
                    id_c += 1
                    for box in targetboxes:
                        rowT = {'imgpath': imgpath, 'id': id_t,
                               'x': box.x1, 'y': box.y1,
                               'width': box.x2-box.x1, 'height': box.y2-box.y1,
                               'label': '', 'is_contest': 0,
                               'contest_id': contest_id}
                        rows_targets.append(rowT)
                        tbox = [box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1,
                                id_t, contest_id]
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
            for i, otherbox in enumerate(boxes):
                if (box.x1 >= otherbox.x1 and box.y1 >= otherbox.y1
                        and box.x2 <= otherbox.x2 and box.y2 <= otherbox.y2):
                    return i, otherbox
            return None
        assocs = {}
        contests = [b for b in boxes if isinstance(b, ContestBox)]
        targets = [b for b in boxes if isinstance(b, TargetBox)]
        for t in targets:
            print t
            id, c = containing_box(t, contests)
            if id in assocs:
                assocs[id][1].append(t)
            else:
                assocs[id] = [c, [t]]
        return assocs

    def onButton_getimgpath(self, evt):
        S = self.seltargets_panel
        imgpath = self.displayed_imgpaths[S.cur_i][S.cur_j][S.cur_page]
        print 'imgpath:', imgpath
        dlg = wx.MessageDialog(self, message="Displayed Imagepath: {0}".format(imgpath),
                               style=wx.OK)
        dlg.ShowModal()

class SelectTargetsPanel(ScrolledPanel):
    """ A widget that allows you to find voting targets on N ballot
    partitions
    """
    TEMPLATE_MATCH_JOBID = 830

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

        # STATEP: Path for state file.
        self.stateP = None

        self.toolbar = Toolbar(self)
        self.imagepanel = TemplateMatchDrawPanel(self, self.do_tempmatch)

        txt = wx.StaticText(self, label="Select all Voting Targets from \
this partition.")

        btn_nextpartition = wx.Button(self, label="Next Partition...")
        btn_prevpartition = wx.Button(self, label="Previous Partition...")
        btn_nextimg = wx.Button(self, label="Next Ballot")
        btn_previmg = wx.Button(self, label="Previous Ballot")

        btn_nextpage = wx.Button(self, label="Next Page")
        btn_prevpage = wx.Button(self, label="Previous Page")
        
        btn_nextpartition.Bind(wx.EVT_BUTTON, self.onButton_nextpartition)
        btn_prevpartition.Bind(wx.EVT_BUTTON, self.onButton_prevpartition)
        btn_nextimg.Bind(wx.EVT_BUTTON, self.onButton_nextimg)
        btn_previmg.Bind(wx.EVT_BUTTON, self.onButton_previmg)
        btn_nextpage.Bind(wx.EVT_BUTTON, self.onButton_nextpage)
        btn_prevpage.Bind(wx.EVT_BUTTON, self.onButton_prevpage)

        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        btn_sizer.AddMany([(btn_prevpartition,), (btn_nextpartition,),
                           (btn_previmg,), (btn_nextimg,),
                           (btn_prevpage,), (btn_nextpage,)])
        
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

    def start(self, partitions, stateP, ocrtempdir):
        """
        Input:
            list PARTITIONS: A list of lists of lists, encoding partition+ballot+side(s):
                [[[imgpath_i0_front, ...], ...], [[imgpath_i1_front, ...], ...], ...]
            str STATEP: Path of the statefile.
            str OCRTEMPDIR: Used for InferContestRegion.
        """
        # 0.) First, align all ballots within each partition to each other.
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
        self.display_image(0, 0, 0)

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
        patch = iplimage2pil(patchB)
        patch.save("_patch.png")        
        # 3.) Run template matching across all images in self.IMGPATHS,
        # using PATCH as the template.
        
        queue = Queue.Queue()
        # Template match on /all/ images across all partitions, all pages
        imgpaths = sum([t for t in sum(self.partitions, [])], [])
        thread = TM_Thread(queue, self.TEMPLATE_MATCH_JOBID, patch, img,
                           imgpaths, self.tm_param, self.win_ballot,
                           self.on_tempmatch_done)
        thread.start()

    def on_tempmatch_done(self, results, w, h):
        """ Invoked after template matching computation is complete. 
        Input:
            dict RESULTS: maps {str imgpath: [(x,y,score_i), ...}. The matches
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
            for (x, y, score) in matches:
                boxB = TargetBox(x, y, x+w, y+h)
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

    def display_image(self, i, j, page):
        """ Displays the J-th image in partition I. Also handles
        reading/saving in the currently-created boxes for the old/new image.
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
        # 1.a.) Resize image s.t. width is equal to containing width
        wP, hP = self.parent.GetSize()
        _c = wximg.GetWidth() / float(wP)
        wimg = wP
        himg = int(round(wximg.GetHeight() / _c))
        #self.imagepanel.set_image(wximg, size=(wimg, himg))
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

    def infercontests(self):
        imgpaths_exs = [] # list of [imgpath_i, ...]
        # Arbitrarily choose the first one Ballot from each partition
        for partition_idx, imgpaths_sides in enumerate(self.partitions):
            for imgpaths in imgpaths_sides:
                imgpaths_exs.extend(imgpaths)
                break
        # Let i=target #, j=ballot style, k=contest idx:
        targets = [] # list of [[[box_ijk, ...], [box_ijk+1, ...], ...], ...]
        for partition_idx, boxes_sides in self.boxes.iteritems():
            for boxes in boxes_sides:
                style_boxes = [] # [[contest_i, ...], ...]
                for box in boxes:
                    # InferContests throws out the pre-determined contest
                    # grouping, so just stick each target in its own
                    # 'contest'
                    if type(box) == TargetBox:
                        style_boxes.append([(box.x1, box.y1, box.x2, box.y2)])
                targets.append(style_boxes)
        #bboxes = dict(zip(imgpaths, group_contests.find_contests(self.ocrtempdir, imgpaths_exs, targets)))
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
        self.btn_addcontest = wx.Button(self, label="Add Contest")
        self.btn_modify = wx.Button(self, label="Modify")
        self.btn_zoomin = wx.Button(self, label="Zoom In")
        self.btn_zoomout = wx.Button(self, label="Zoom Out")
        self.btn_infercontests = wx.Button(self, label="Infer Contest Regions..")
        self.btn_opts = wx.Button(self, label="Options...")
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        btn_sizer.AddMany([(self.btn_addtarget,), (self.btn_addcontest), (self.btn_modify,),
                           (self.btn_zoomin,), (self.btn_zoomout,),
                           (self.btn_infercontests,), (self.btn_opts,)])
        self.sizer.Add(btn_sizer)
        self.SetSizer(self.sizer)

    def _setup_evts(self):
        self.btn_addtarget.Bind(wx.EVT_BUTTON, self.onButton_addtarget)
        self.btn_addcontest.Bind(wx.EVT_BUTTON, self.onButton_addcontest)
        self.btn_modify.Bind(wx.EVT_BUTTON, lambda evt: self.setmode(BoxDrawPanel.M_IDLE))
        self.btn_zoomin.Bind(wx.EVT_BUTTON, lambda evt: self.parent.zoomin())
        self.btn_zoomout.Bind(wx.EVT_BUTTON, lambda evt: self.parent.zoomout())
        self.btn_infercontests.Bind(wx.EVT_BUTTON, lambda evt: self.parent.infercontests())
        self.btn_opts.Bind(wx.EVT_BUTTON, self.onButton_opts)
    def onButton_addtarget(self, evt):
        self.setmode(BoxDrawPanel.M_CREATE)
        self.parent.imagepanel.boxtype = TargetBox
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

        btn_apply = wx.Button(self, label="Apply")
        btn_apply.Bind(wx.EVT_BUTTON, self.onButton_apply)
        btn_cancel = wx.Button(self, label="Cancel")
        btn_cancel.Bind(wx.EVT_BUTTON, self.onButton_cancel)
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        btn_sizer.AddMany([(btn_apply,), (btn_cancel,)])
        
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(txt0, flag=wx.ALIGN_CENTER)
        sizer.AddMany([(tm_sizer,), (sizer0,), (sizer1,), (btn_sizer, 0, wx.ALIGN_CENTER)])
        self.SetSizer(sizer)
        self.Fit()

    def onButton_apply(self, evt):
        self.tm_param = float(self.tm_param.GetValue())
        self.win_ballot = (int(self.xwin_ballot.GetValue()), int(self.ywin_ballot.GetValue()))
        self.win_target = (int(self.xwin_target.GetValue()), int(self.ywin_target.GetValue()))
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

        # self.scale: Scaling factor used to display self.IMGBITMAP
        self.scale = 1.0

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

    def onPaint(self, evt):
        if self.IsDoubleBuffered():
            dc = wx.PaintDC(self)
        else:
            dc = wx.BufferedPaintDC(self)
        self.PrepareDC(dc)
        if self.imgbitmap:
            dc.DrawBitmap(self.imgbitmap, 0, 0)

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
        return toreturn
    
    def select_boxes(self, *boxes):
        for box in boxes:
            box.is_sel = True
        self.sel_boxes.extend(boxes)

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
        if self.isResize:
            self.box_resize.canonicalize()
            self.box_resize = None
            self.isResize = False

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
            xdel_img, ydel_img = self.c2img(xdel, ydel)
            for box in self.sel_boxes:
                box.x1 += xdel_img
                box.y1 += ydel_img
                box.x2 += xdel_img
                box.y2 += ydel_img
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
        dc = ImagePanel.onPaint(self, evt)
        if self.isResize:
            dboxes = [b for b in self.boxes if b != self.box_resize]
        else:
            dboxes = self.boxes
        self.drawBoxes(dboxes, dc)
        if self.isCreate:
            # Draw Box-Being-Created
            can_box = self.box_create.copy().canonicalize()
            self.drawBox(can_box, dc)
        if self.isResize:
            pass
            resize_box_can = self.box_resize.copy().canonicalize()
            self.drawBox(resize_box_can, dc)
        return dc
        
    def drawBoxes(self, boxes, dc):
        for box in boxes:
            self.drawBox(box, dc)

    def drawBox(self, box, dc):
        """ Draws BOX onto DC.
        Input:
            list box: (x1, y1, x2, y2)
            wxDC DC:
        """
        dc.SetBrush(wx.TRANSPARENT_BRUSH)
        drawops = box.get_draw_opts()
        dc.SetPen(wx.Pen(*drawops))
        w = int(round(abs(box.x2 - box.x1) * self.scale))
        h = int(round(abs(box.y2 - box.y1) * self.scale))
        client_x, client_y = self.img2c(box.x1, box.y1)
        dc.DrawRectangle(client_x, client_y, w, h)

        if isinstance(box, TargetBox) or isinstance(box, ContestBox):
            # Draw the 'grabber' circles
            CIRCLE_RAD = 3
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

class TemplateMatchDrawPanel(BoxDrawPanel):
    """ Like a BoxDrawPanel, but when you create a Target box, it runs
    Template Matching to try to find similar instances.
    """
    def __init__(self, parent, tempmatch_fn, *args, **kwargs):
        BoxDrawPanel.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.tempmatch_fn = tempmatch_fn

    def onLeftUp(self, evt):
        x, y = evt.GetPositionTuple()
        if self.mode_m == BoxDrawPanel.M_CREATE and self.isCreate:
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

class TM_Thread(threading.Thread):
    TEMPLATE_MATCH_JOBID = 48
    def __init__(self, queue, job_id, patch, img, imgpaths, tm_param,
                 win_ballot,
                 callback, *args, **kwargs):
        """
        Input:
            PATCH: A PIL Image.
        """
        threading.Thread.__init__(self, *args, **kwargs)
        self.queue = queue
        self.job_id = job_id
        self.patch = patch
        self.img = img
        self.imgpaths = imgpaths
        self.tm_param = tm_param
        self.win_ballot = win_ballot
        self.callback = callback
    def run(self):
        print "...running template matching..."
        t = time.time()
        patch_str = self.patch.tostring()
        w, h = self.patch.size
        # results: {str imgpath: [(x,y,score_i), ...]}
        results = partask.do_partask(do_find_matches, self.imgpaths, 
                                     _args=(patch_str, w, h, self.tm_param, self.win_ballot),
                                     combfn='dict', singleproc=False)
        dur = time.time() - t
        print "...finished running template matching ({0} s).".format(dur)
        self.callback(results, self.patch.size[0], self.patch.size[1])

class Box(object):
    def __init__(self, x1, y1, x2, y2):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
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
            return ("Yellow", 3)
        else:
            return ("Green", 3)
    def copy(self):
        return TargetBox(self.x1, self.y1, self.x2, self.y2, is_sel=self.is_sel)
class ContestBox(Box):
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
            return ("Yellow", 5)
        else:
            return ("Blue", 5)
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

def do_find_matches(imgpaths, (patch_str, w, h, C, win_ballot)):
    """ PATCH_STR is the string-repr of the image patch data (by calling
    img.tostring(). We do this because anything passed to multiprocessing
    methods must be pickle-able (cvMats/PIL images are not).
    """
    patch_cv = cv.CreateImageHeader((w,h), cv.IPL_DEPTH_8U, 1)
    cv.SetData(patch_cv, patch_str)
    return find_matches(imgpaths, patch_cv, C=C, xwin=win_ballot[0], ywin=win_ballot[1])
def find_matches(imgpaths, patch, C=0.8, do_smooth=True, xwin=13, ywin=13, MAX_MATS=50):
    """ Runs template matching to find PATCH in each IMGPATHS. If 
    DO_SMOOTH is True, then this will apply a gaussian blur with
    window size [XWIN,YWIN] on IMGPATHS (but not on PATCH).
    Input:
        list imgpaths: [imgpath_i, ...]
        IplImage patch: 
        float C:
    Output:
        list matches, {str imgpath: [(x,y,score_i), ...]}
    """
    matches = {} # maps {str imgpath: [(x,y,score_i), ...]}
    for imgpath in imgpaths:
        img = cv.LoadImage(imgpath, cv.CV_LOAD_IMAGE_GRAYSCALE)
        if do_smooth:
            img_smooth = cv.CreateImage((img.width, img.height), img.depth, img.channels)
            cv.Smooth(img, img_smooth, cv.CV_GAUSSIAN, param1=xwin,param2=ywin)
            img = img_smooth
        M = cv.CreateMat(img.height-patch.height+1, img.width-patch.width+1, cv.CV_32F)
        cv.MatchTemplate(img, patch, M, cv.CV_TM_CCOEFF_NORMED)
        M_np = np.array(M)
        score = np.inf
        print 'best score:', np.max(M_np)
        num_mats = 0
        while score > C and num_mats < MAX_MATS:
            M_idx = np.argmax(M_np)
            i = int(M_idx / M.cols)
            j = M_idx % M.cols
            score = M_np[i,j]
            if score < C:
                break
            matches.setdefault(imgpath, []).append((j, i, score))
            # Suppression
            M_np[i-(patch.height/3):i+(patch.height/3),
                 j-(patch.width/3):j+(patch.width/3)] = -1.0
            num_mats += 1
    return matches

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

def align_partitions(partitions, (outrootdir,), start_pid, queue=None):
    """ 
    Input:
        list PARTITIONS: list of list of lists, encodes partition+ballot+side.
        str OUTROOTDIR: Rootdir to save aligned images to.
    Output:
        dict PARTITIONS_ALIGN: {int partitionID: [BALLOT_i, ...]}
    """
    partitions_align = {} # maps {partitionID: [[imgpath_i, ...], ...]}

    t = time.time()
    for idx, partition in enumerate(partitions):
        partitionid = start_pid + idx
        outdir = pathjoin(outrootdir, 'partition_{0}'.format(partitionid))
        try:
            os.makedirs(outdir)
        except:
            pass
        ballotRef = partition[0]
        Irefs = [shared.standardImread(imP) for imP in ballotRef]
        # 0.) First, handle the reference Ballot
        curBallot = []
        for side, Iref in enumerate(Irefs):
            outname = 'bal_{0}_side_{1}.png'.format(0, side)
            outpath = pathjoin(outdir, outname)
            scipy.misc.imsave(outpath, Iref)
            curBallot.append(outpath)
        partitions_align[partitionid] = [curBallot]
        # 1.) Now, align all other Ballots to BALLOTREF
        for i, ballot in enumerate(partition[1:]):
            curBallot = []
            for side, imgpath in enumerate(ballot):
                Iref = Irefs[side]
                I = shared.standardImread(imgpath, flatten=True)
                H, Ireg, err = imagesAlign.imagesAlign(I, Iref, type='rigid', rszFac=0.25)
                Ireg = np.nan_to_num(Ireg)
                outname = 'bal_{0}_side_{1}.png'.format(i + 1, side)
                outpath = pathjoin(outdir, outname)
                scipy.misc.imsave(outpath, Ireg)
                curBallot.append(outpath)
            partitions_align[partitionid].append(curBallot)
        if queue:
            queue.put(True)
    dur = time.time() - t

    return partitions_align

def do_align_partitions(partitions, outrootdir, manager, queue):
    try:
        partitions_align = partask.do_partask(align_partitions, 
                                              partitions,
                                              _args=(outrootdir,),
                                              manager=manager,
                                              pass_queue=queue,
                                              pass_idx=True,
                                              combfn='dict',
                                              N=None)
        return partitions_align
    except:
        traceback.print_exc()
        return None

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
