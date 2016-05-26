import os
import threading
import multiprocessing
import math
import shutil
import cProfile
from os.path import join as pathjoin

import wx
import cv
try:
    from wx.lib.pubsub import pub
except:
    from wx.lib.pubsub import Publisher
    pub = Publisher()


import util
import config
import image_file as imageFile
import pixel_reg.doExtract as doExtract


class TargetExtractPanel(wx.Panel):

    def __init__(self, parent, *args, **kwargs):
        wx.Panel.__init__(self, parent, *args, **kwargs)

        self.init_ui()

    def init_ui(self):
        self.btn_run = wx.Button(self, label="Run Target Extraction...")
        self.btn_run.Bind(wx.EVT_BUTTON, self.onButton_run)
        txt = wx.StaticText(self, label="...Or, if you've already run Target \
Extraction, but you just want to create the Image File:")
        txt.Hide()
        btn_createImageFile = wx.Button(
            self, label="Advanced: Only create Image File...")
        btn_createImageFile.Bind(wx.EVT_BUTTON, self.onButton_createImageFile)
        if not config.IS_DEV:
            btn_createImageFile.Hide()
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        btn_sizer.Add(self.btn_run)

        self.txt_can_move_on = wx.StaticText(
            self, label="Target Extraction computation complete. You may move on.")
        self.txt_can_move_on.Hide()

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(btn_sizer)
        self.sizer.Add((0, 50))
        self.sizer.Add(txt)
        self.sizer.Add(btn_createImageFile)
        self.sizer.Add((0, 50))
        self.sizer.Add(self.txt_can_move_on)
        self.SetSizer(self.sizer)
        self.Layout()

    def start(self, project=None, projdir=None):
        self.proj = project

    def stop(self):
        pass

    def onButton_run(self, evt):
        self.Disable()

        # First, remove all files
        if os.path.exists(self.proj.extracted_dir):
            shutil.rmtree(self.proj.extracted_dir)
        if os.path.exists(self.proj.extracted_metadata):
            shutil.rmtree(self.proj.extracted_metadata)
        if os.path.exists(self.proj.ballot_metadata):
            shutil.rmtree(self.proj.ballot_metadata)
        if os.path.exists(pathjoin(self.proj.projdir_path, self.proj.targetextract_quarantined)):
            os.remove(pathjoin(self.proj.projdir_path,
                               self.proj.targetextract_quarantined))
        if os.path.exists(pathjoin(self.proj.projdir_path, "extracted_radix")):
            shutil.rmtree(pathjoin(self.proj.projdir_path, "extracted_radix"))
        if os.path.exists(pathjoin(self.proj.projdir_path, "extracted_diffs")):
            shutil.rmtree(pathjoin(self.proj.projdir_path, "extracted_diffs"))
        if os.path.exists(pathjoin(self.proj.projdir_path, "targetextract_avg_intensities.p")):
            os.remove(pathjoin(self.proj.projdir_path,
                               "targetextract_avg_intensities.p"))
        if os.path.exists(pathjoin(self.proj.projdir_path, self.proj.ballot_to_targets)):
            os.remove(pathjoin(self.proj.projdir_path,
                               self.proj.ballot_to_targets))
        if os.path.exists(self.proj.classified):
            os.remove(self.proj.classified)
        if os.path.exists(self.proj.extractedfile):
            os.remove(self.proj.extractedfile)
        if os.path.exists(self.proj.extractedfile + ".type"):
            os.remove(self.proj.extractedfile + ".type")
        if os.path.exists(self.proj.extractedfile + ".size"):
            os.remove(self.proj.extractedfile + ".size")
        if os.path.exists(self.proj.threshold_internal):
            os.remove(self.proj.threshold_internal)

        t = RunThread(self.proj)
        t.start()

        gauge = util.MyGauge(self,
                             5,
                             job_id=util.Gauges.extract_targets,
                             ondone=self.on_targetextract_done,
                             thread=t)
        gauge.Show()

    def onButton_createImageFile(self, evt):
        self.Disable()
        t = RunThread(self.proj, skip_extract=True)
        t.start()

        gauge = util.MyGauge(self,
                             4,
                             job_id=util.Gauges.targets_create_image,
                             ondone=self.on_targetextract_done,
                             thread=t)
        gauge.Show()

    def on_targetextract_done(self):
        print "...TargetExtraction Done!..."
        self.btn_run.Disable()
        self.txt_can_move_on.Show()
        self.Layout()
        self.Enable()


class RunThread(threading.Thread):

    def __init__(self, proj, skip_extract=False, do_profile=False, profile_out='profile_targetextract.out', *args, **kwargs):
        threading.Thread.__init__(self, *args, **kwargs)
        self.proj = proj
        self.skip_extract = skip_extract
        self.do_profile = do_profile
        self.profile_out = profile_out

    def run(self):
        if self.do_profile:
            cProfile.runctx('self.do_target_extract()', {}, {
                            'self': self}, self.profile_out)
        else:
            self.do_target_extract()

    def do_target_extract(self):
        if not self.skip_extract:
            group_to_ballots = self.proj.load_field(self.proj.group_to_ballots)
            group_exmpls = self.proj.load_field(self.proj.group_exmpls)
            b2imgs = self.proj.load_field(self.proj.ballot_to_images)
            img2b = self.proj.load_field(self.proj.image_to_ballot)
            img2page = self.proj.load_field(self.proj.image_to_page)
            img2flip = self.proj.load_field(self.proj.image_to_flip)
        target_locs_map = self.proj.load_field(self.proj.target_locs_map)

        print "...starting doExtract..."
        if not self.skip_extract:
            qballotids = self.proj.get_quarantined_ballots()
            discarded_ballotids = self.proj.get_discarded_ballots()
            ioerr_ballotids = self.proj.get_ioerr_ballots()
            bad_ballotids = list(
                set(qballotids + discarded_ballotids + ioerr_ballotids))
            nProc = 1 if self.do_profile else None
            # list AVG_INTENSITIES: [(str targetid, float avg_intensity), ...]
            # where TARGETID :=  str(balId)+"\0"+str(page)+"\0"+str(int(uid))
            avg_intensities, bal2targets = doExtract.extract_targets(
                group_to_ballots, b2imgs, img2b, img2page, img2flip,
                target_locs_map, group_exmpls,
                bad_ballotids,
                self.proj.extracted_dir,
                self.proj.extracted_metadata,
                self.proj.ballot_metadata,
                pathjoin(self.proj.projdir_path,
                         self.proj.targetextract_quarantined),
                self.proj.voteddir,
                self.proj.projdir_path,
                nProc=nProc,
                method_galign=doExtract.GALIGN_NORMAL,
                method_lalign=doExtract.LALIGN_NORMAL)
            self.proj.save_field(
                avg_intensities,
                'targetextract_avg_intensities.p')
            self.proj.save_field(
                bal2targets,
                self.proj.ballot_to_targets)
        else:
            avg_intensities = self.proj.load_field(
                'targetextract_avg_intensities.p')
            bal2targets = self.proj.load_field(self.proj.ballot_to_targets)
            print "    (skip_extract was True - not running doExtract)"

        total = len(bal2targets)

        del bal2targets  # Try to reclaim some memory

        manager = multiprocessing.Manager()

        if wx.App.IsMainLoopRunning():
            util.Gauges.extract_targets.next_job(1)
        fulllst = sorted(avg_intensities, key=lambda x: x[
                         1])  # sort by avg. intensity

        del avg_intensities  # Try to reclaim some memory

        fulllst = [(x, int(y)) for x, y in fulllst]

        out = open(self.proj.classified, "w")
        # Store imgpath \0 avg_intensity to output file OUT
        for a, b in fulllst:
            out.write(a + "\0" + str(b) + "\n")
        out.close()

        print "...Starting imageFileMake..."

        def get_target_size():
            # TARGET_LOCS_MAP: maps {int groupID: {int page: [CONTEST_i, ...]}}, where each
            #     CONTEST_i is: [contestbox, targetbox_i, ...], where each
            #     box := [x1, y1, width, height, id, contest_id]
            for groupid, pagedict in target_locs_map.iteritems():
                for page, contests in pagedict.iteritems():
                    for contest in contests:
                        targetboxes = contest[1:]
                        for (x1, y1, w, h, id, contest_id) in targetboxes:
                            return w, h
            return None, None

        w, h = get_target_size()
        if w is None:
            raise Exception("Woah, No targets in this election??")

        imageFile.makeOneFile(fulllst,
                              pathjoin(self.proj.projdir_path,
                                       'extracted_radix/'),
                              self.proj.extractedfile,
                              (w, h),
                              SORT_METHOD=imageFile.METHOD_DYN,
                              MEM_C=0.6)

        if wx.App.IsMainLoopRunning():
            wx.CallAfter(pub.sendMessage, "broadcast.rundone", msg=())
            util.MyGauge.all_done()


def doandgetAvgs(imgnames, rootdir, queue):
    for imgname in imgnames:
        imgpath = pathjoin(rootdir, imgname)
        I = cv.LoadImage(imgpath, cv.CV_LOAD_IMAGE_GRAYSCALE)
        w, h = cv.GetSize(I)
        result = cv.Sum(I)[0] / float(w * h)
        # data = shared.standardImread(pathjoin(rootdir, imgname), flatten=True)
        # result = 256 * (float(sum(map(sum, data)))) / (data.size)
        queue.put((imgname, result))
    return 0


def spawn_jobs(queue, rootdir, dirList):
    def divy_elements(lst, k):
        """ Separate lst into k chunks """
        if len(lst) <= k:
            return [lst]
        chunksize = math.floor(len(lst) / float(k))
        i = 0
        chunks = []
        curchunk = []
        while i < len(lst):
            if i != 0 and ((i % chunksize) == 0):
                chunks.append(curchunk)
                curchunk = []
            curchunk.append(lst[i])
            i += 1
        if curchunk:
            chunks.append(curchunk)
        return chunks
    pool = multiprocessing.Pool()
    n_procs = float(multiprocessing.cpu_count())
    for i, imgpaths in enumerate(divy_elements(dirList, n_procs)):
        print 'Process {0} got {1} imgs'.format(i, len(imgpaths))
        pool.apply_async(doandgetAvgs, args=(imgpaths, rootdir, queue))
    pool.close()
    pool.join()


def _run_target_extract(proj, do_profile, profile_out):
    if os.path.exists(proj.extracted_dir):
        shutil.rmtree(proj.extracted_dir)
    if os.path.exists(proj.extracted_metadata):
        shutil.rmtree(proj.extracted_metadata)
    if os.path.exists(proj.ballot_metadata):
        shutil.rmtree(proj.ballot_metadata)
    if os.path.exists(pathjoin(proj.projdir_path,
                               proj.targetextract_quarantined)):
        os.remove(pathjoin(proj.projdir_path, proj.targetextract_quarantined))
    if os.path.exists(pathjoin(proj.projdir_path, "extracted_radix")):
        shutil.rmtree(pathjoin(proj.projdir_path, "extracted_radix"))
    if os.path.exists(pathjoin(proj.projdir_path, "extractedfile")):
        os.remove(pathjoin(proj.projdir_path, "extractedfile"))
    if os.path.exists(pathjoin(proj.projdir_path, "extractedfile.size")):
        os.remove(pathjoin(proj.projdir_path, "extractedfile.size"))
    if os.path.exists(pathjoin(proj.projdir_path, "extractedfile.type")):
        os.remove(pathjoin(proj.projdir_path, "extractedfile.type"))

    t = RunThread(proj, do_profile=do_profile, profile_out=profile_out)
    t.start()
    t.join()
