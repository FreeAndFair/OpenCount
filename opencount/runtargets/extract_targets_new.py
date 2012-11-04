import sys, os, traceback, pdb, threading, multiprocessing, math, array
try:
    import cPickle as pickle
except:
    import pickle
from os.path import join as pathjoin

import wx
from wx.lib.pubsub import Publisher

sys.path.append('..')

import util
import threshold.imageFile
import pixel_reg.doExtract as doExtract
import pixel_reg.shared as shared

class TargetExtractPanel(wx.Panel):
    def __init__(self, parent, *args, **kwargs):
        wx.Panel.__init__(self, parent, *args, **kwargs)
        
        self.init_ui()

    def init_ui(self):
        btn_run = wx.Button(self, label="Run Target Extraction...")
        btn_run.Bind(wx.EVT_BUTTON, self.onButton_run)
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        btn_sizer.Add(btn_run)

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(btn_sizer)
        self.SetSizer(self.sizer)
        self.Layout()

    def start(self, proj):
        self.proj = proj

    def stop(self):
        pass

    def onButton_run(self, evt):
        self.Disable()

        t = RunThread(self.proj)
        t.start()

        def fn1():
            try:
                return len(os.listdir(self.proj.ballot_metadata))
            except:
                return 0
        fns = [fn1, None, None, None, None, None]

        gauge = util.MyGauge(self, 5, funs=fns, tofile=self.proj.timing_runtarget,
                             ondone=self.on_targetextract_done, thread=t)
        gauge.Show()

    def on_targetextract_done(self):
        print "...TargetExtraction Done!..."
        self.Enable()

class RunThread(threading.Thread):
    def __init__(self, proj, *args, **kwargs):
        threading.Thread.__init__(self, *args, **kwargs)
        self.proj = proj

    def run(self):
        partitions_map = pickle.load(open(pathjoin(self.proj.projdir_path,
                                                   self.proj.partitions_map), 'rb'))
        partition_exmpls = pickle.load(open(pathjoin(self.proj.projdir_path,
                                                     self.proj.partition_exmpls), 'rb'))
        b2imgs = pickle.load(open(self.proj.ballot_to_images, 'rb'))
        img2b = pickle.load(open(self.proj.image_to_ballot, 'rb'))
        img2page = pickle.load(open(pathjoin(self.proj.projdir_path,
                                             self.proj.image_to_page), 'rb'))
        target_locs_map = pickle.load(open(pathjoin(self.proj.projdir_path,
                                                    self.proj.target_locs_map), 'rb'))
        res = doExtract.extract_targets(partitions_map, b2imgs, img2b, img2page,
                                        target_locs_map, partition_exmpls,
                                        self.proj.extracted_dir,
                                        self.proj.extracted_metadata,
                                        self.proj.ballot_metadata,
                                        self.proj.quarantined)
        try:
            os.makedirs(self.proj.extracted_dir)
        except:
            pass
        dirList = os.listdir(self.proj.extracted_dir)

        dirList = [x for x in dirList if util.is_image_ext(x)]
        # This will always be a common prefix. 
        # Just add it to there once. Code will be faster.
        dirList = [x for x in dirList]
        
        #quarantined = set([util.encodepath(x[:-1]) for x in open(self.proj.quarantined)])
        quarantined = set([])

        dirList = [x for x in dirList if os.path.split(x)[1][:os.path.split(x)[1].index(".")] not in quarantined]

        wx.CallAfter(Publisher().sendMessage, "signals.MyGauge.nextjob", len(dirList))
        print "Doing a zip"

        total = len(dirList)
        manager = multiprocessing.Manager()
        queue = manager.Queue()
        start_doandgetAvg(queue, self.proj.extracted_dir, dirList)
        tmp = []  # TMP: [[imgpath, float avg_intensity], ...]
        i = 0
        while i < total:
            result = queue.get()
            tmp.append(result)
            wx.CallAfter(Publisher().sendMessage, "signals.MyGauge.tick")
            i += 1
        
        wx.CallAfter(Publisher().sendMessage, "signals.MyGauge.nextjob", len(dirList))
        print "Doing a find-longest-prefix thing"

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
        prefix = pathjoin(self.proj.extracted_dir,prefix)

        open(self.proj.classified+".prefix", "w").write(prefix)
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

        print 'done'
        
        threshold.imageFile.makeOneFile(self.proj.extracted_dir, 
                                        fulllst, self.proj.extractedfile)

        wx.CallAfter(Publisher().sendMessage, "broadcast.rundone")
        wx.CallAfter(Publisher().sendMessage, "signals.MyGauge.done")

def start_doandgetAvg(queue, rootdir, dirList):
    p = multiprocessing.Process(target=spawn_jobs, args=(queue, rootdir, dirList))
    p.start()

def doandgetAvgs(imgnames, rootdir, queue):
    for imgname in imgnames:
        data = shared.standardImread(pathjoin(rootdir, imgname), flatten=True)
        result = 256 * (float(sum(map(sum, data)))) / (data.size)
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
    
    
