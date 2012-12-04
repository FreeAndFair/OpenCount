import sys, os, traceback, pdb, threading, multiprocessing, math, array, time, shutil
try:
    import cPickle as pickle
except:
    import pickle
from os.path import join as pathjoin

import wx
import cv
from wx.lib.pubsub import Publisher

sys.path.append('..')

import util
import threshold.imageFile
import pixel_reg.doExtract as doExtract
import pixel_reg.shared as shared
import quarantine.quarantinepanel as quarantinepanel

class TargetExtractPanel(wx.Panel):
    def __init__(self, parent, *args, **kwargs):
        wx.Panel.__init__(self, parent, *args, **kwargs)
        
        self.init_ui()

    def init_ui(self):
        btn_run = wx.Button(self, label="Run Target Extraction...")
        btn_run.Bind(wx.EVT_BUTTON, self.onButton_run)
        txt = wx.StaticText(self, label="...Or, if you've already run Target \
Extraction, but you just want to create the Image File:")
        btn_createImageFile = wx.Button(self, label="Only create Image File...")
        btn_createImageFile.Bind(wx.EVT_BUTTON, self.onButton_createImageFile)
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        btn_sizer.Add(btn_run)

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(btn_sizer)
        self.sizer.Add((0, 50))
        self.sizer.Add(txt)
        self.sizer.Add(btn_createImageFile)
        self.SetSizer(self.sizer)
        self.Layout()

    def start(self, proj):
        self.proj = proj

    def stop(self):
        pass

    def onButton_run(self, evt):
        self.Disable()

        # First, remove all files
        if os.path.exists(self.proj.extracted_dir): shutil.rmtree(self.proj.extracted_dir)
        if os.path.exists(self.proj.extracted_metadata): shutil.rmtree(self.proj.extracted_metadata)
        if os.path.exists(self.proj.ballot_metadata): shutil.rmtree(self.proj.ballot_metadata)
        if os.path.exists(pathjoin(self.proj.projdir_path, self.proj.targetextract_quarantined)):
            os.remove(pathjoin(self.proj.projdir_path, self.proj.targetextract_quarantined))

        t = RunThread(self.proj)
        t.start()

        gauge = util.MyGauge(self, 5, tofile=self.proj.timing_runtarget,
                             ondone=self.on_targetextract_done, thread=t)
        gauge.Show()

    def onButton_createImageFile(self, evt):
        self.Disable()
        t = RunThread(self.proj, skip_extract=True)
        t.start()

        gauge = util.MyGauge(self, 4, tofile=self.proj.timing_runtarget,
                             ondone=self.on_targetextract_done, thread=t)
        gauge.Show()

    def on_targetextract_done(self):
        print "...TargetExtraction Done!..."
        self.Enable()

class RunThread(threading.Thread):
    def __init__(self, proj, skip_extract=False, *args, **kwargs):
        threading.Thread.__init__(self, *args, **kwargs)
        self.proj = proj
        self.skip_extract = skip_extract

    def run(self):
        group_to_ballots = pickle.load(open(pathjoin(self.proj.projdir_path,
                                                     self.proj.group_to_ballots), 'rb'))
        group_exmpls = pickle.load(open(pathjoin(self.proj.projdir_path,
                                                 self.proj.group_exmpls), 'rb'))
        b2imgs = pickle.load(open(self.proj.ballot_to_images, 'rb'))
        img2b = pickle.load(open(self.proj.image_to_ballot, 'rb'))
        img2page = pickle.load(open(pathjoin(self.proj.projdir_path,
                                             self.proj.image_to_page), 'rb'))
        img2flip = pickle.load(open(pathjoin(self.proj.projdir_path,
                                             self.proj.image_to_flip), 'rb'))
        target_locs_map = pickle.load(open(pathjoin(self.proj.projdir_path,
                                                    self.proj.target_locs_map), 'rb'))
        totalTime = time.time()
        time_doExtract = time.time()
        print "...starting doExtract..."
        if not self.skip_extract:
            qballotids = quarantinepanel.get_quarantined_ballots(self.proj)
            discarded_ballotids = quarantinepanel.get_discarded_ballots(self.proj)
            bad_ballotids = list(set(qballotids + discarded_ballotids))
            avg_intensities, bal2targets = doExtract.extract_targets(group_to_ballots, b2imgs, img2b, img2page, img2flip,
                                                                     target_locs_map, group_exmpls,
                                                                     bad_ballotids,
                                                                     self.proj.extracted_dir,
                                                                     self.proj.extracted_metadata,
                                                                     self.proj.ballot_metadata,
                                                                     pathjoin(self.proj.projdir_path,
                                                                              self.proj.targetextract_quarantined),
                                                                     self.proj.voteddir)
            pickle.dump(avg_intensities, open(pathjoin(self.proj.projdir_path,
                                                       'targetextract_avg_intensities.p'), 'wb'),
                        pickle.HIGHEST_PROTOCOL)
            pickle.dump(bal2targets, open(pathjoin(self.proj.projdir_path,
                                                   self.proj.ballot_to_targets), 'wb'),
                        pickle.HIGHEST_PROTOCOL)
        else:
            avg_intensities = pickle.load(open(pathjoin(self.proj.projdir_path,
                                                        'targetextract_avg_intensities.p'), 'rb'))
            bal2targets = pickle.load(open(pathjoin(self.proj.projdir_path,
                                                    self.proj.ballot_to_targets), 'rb'))
            print "    (skip_extract was True - not running doExtract)"
        dur_doExtract = time.time() - time_doExtract
        print "...Finished doExtract ({0} s)...".format(dur_doExtract)
        print "...Doing post-target-extraction work..."
        time_post = time.time()
        try:
            os.makedirs(self.proj.extracted_dir)
        except:
            pass

        # This will always be a common prefix. 
        # Just add it to there once. Code will be faster.
        
        #wx.CallAfter(Publisher().sendMessage, "signals.MyGauge.nextjob", len(dirList))
        print "...Doing a zip..."

        total = len(bal2targets)
        manager = multiprocessing.Manager()
        queue = manager.Queue()
        time_doandgetAvg = time.time()
        #start_doandgetAvg(queue, self.proj.extracted_dir, dirList)
        tmp = avg_intensities # TMP: [[imgpath, float avg_intensity], ...]
        wx.CallAfter(Publisher().sendMessage, "signals.MyGauge.nextjob", total)
        print "...Starting a find-longest-prefix thing..."
        time_longestPrefix = time.time()
        fulllst = sorted(tmp, key=lambda x: x[1])  # sort by avg. intensity
        fulllst = [(x,int(y)) for x,y in fulllst]
        
        #print "FULLLST:", fulllst

        # Find the longest prefix
        prefix = fulllst[0][0]
        for a,_ in fulllst:
            if not a.startswith(prefix):
                new = ""
                for x,y in zip(prefix,a):
                    if x == y:
                        new += x
                    else:
                        break
                prefix = new

        l = len(prefix)
        #prefix = pathjoin(self.proj.extracted_dir,prefix)
        
        dur_longestPrefix = time.time() - time_longestPrefix
        open(self.proj.classified+".prefix", "w").write(prefix)
        print "...Finished find-longest-prefix ({0} s).".format(dur_longestPrefix)

        print "...Starting classifiedWrite..."
        time_classifiedWrite = time.time()
        out = open(self.proj.classified, "w")
        offsets = array.array('L')
        sofar = 0
        # Store imgpath \0 avg_intensity to output file OUT
        for a,b in fulllst:
            line = a[l:] + "\0" + str(b) + "\n"
            out.write(line)
            offsets.append(sofar)
            sofar += len(line)
        out.close()

        offsets.tofile(open(self.proj.classified+".index", "w"))
        dur_classifiedWrite = time.time() - time_classifiedWrite
        print "...Finished classifiedWrite ({0} s).".format(dur_classifiedWrite)

        print "...Starting imageFileMake..."
        time_imageFileMake = time.time()

        threshold.imageFile.makeOneFile('',
                                        fulllst, self.proj.extractedfile)
        dur_imageFileMake = time.time() - time_imageFileMake
        print "...Finished imageFileMake ({0} s).".format(dur_imageFileMake)

        wx.CallAfter(Publisher().sendMessage, "broadcast.rundone")
        wx.CallAfter(Publisher().sendMessage, "signals.MyGauge.done")
        
        dur_post = time.time() - time_post
        print "...Finished post-target-extraction work ({0} s).".format(dur_post)
        dur_totalTime = time.time() - totalTime
        
        print "...Finished Target Extraction. ({0} s).".format(dur_totalTime)
        frac = (dur_doExtract / dur_totalTime) * 100
        print "    doExtract: {0:.3f}%  |  {1:.3f} s".format(frac, dur_doExtract)
        frac = (dur_post / dur_totalTime) * 100
        print "    post-work: {0:.3f}%  |  {1:.3f} s".format(frac, dur_post)
        frac = (dur_longestPrefix / dur_post) * 100
        print "        longestPrefix: {0:.3f}%  |  {1:.3f} s".format(frac, dur_longestPrefix)
        frac = (dur_classifiedWrite / dur_post) * 100
        print "        classifiedWrite: {0:.3f}%  |  {1:.3f} s".format(frac, dur_classifiedWrite)
        frac = (dur_imageFileMake / dur_post) * 100
        print "        imageFileMake: {0:.3f}%   |  {1:.3f} s".format(frac, dur_imageFileMake)

def start_doandgetAvg(queue, rootdir, dirList):
    p = multiprocessing.Process(target=spawn_jobs, args=(queue, rootdir, dirList))
    p.start()

def doandgetAvgs(imgnames, rootdir, queue):
    for imgname in imgnames:
        imgpath = pathjoin(rootdir, imgname)
        I = cv.LoadImage(imgpath, cv.CV_LOAD_IMAGE_GRAYSCALE)
        w, h = cv.GetSize(I)
        result = cv.Sum(I)[0] / float(w*h)
        #data = shared.standardImread(pathjoin(rootdir, imgname), flatten=True)
        #result = 256 * (float(sum(map(sum, data)))) / (data.size)
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
    
    
