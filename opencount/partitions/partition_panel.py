import sys, os, pdb, traceback, threading, Queue
try:
    import cPickle as pickle
except:
    import pickle

from os.path import join as pathjoin
import wx
from wx.lib.scrolledpanel import ScrolledPanel
from wx.lib.pubsub import Publisher

sys.path.append('..')

import util
import barcode.partition_imgs as partition_imgs

class PartitionMainPanel(wx.Panel):
    # NUM_EXMPLS: Number of exemplars to grab from each partition
    NUM_EXMPLS = 5

    def __init__(self, parent, *args, **kwargs):
        wx.Panel.__init__(self, parent, *args, **kwargs)

        self.proj = None

        self.init_ui()

    def init_ui(self):
        self.partitionpanel = PartitionPanel(self)
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.partitionpanel, proportion=1, flag=wx.EXPAND)
        self.SetSizer(self.sizer)
        self.Layout()

    def start(self, proj, stateP):
        self.proj = proj
        self.partitionpanel.start(self.proj, self.proj.voteddir, stateP)
        self.proj.addCloseEvent(self.partitionpanel.save_session)
    def stop(self):
        self.partitionpanel.save_session()
        self.proj.removeCloseEvent(self.partitionpanel.save_session)
        self.export_results()
    def export_results(self):
        """ Export the partitions_map and partitions_invmap, where
        PARTITIONS_MAP maps {partitionID: [int BallotID_i, ...]}, and
        PARTITIONS_INVMAP maps {int BallotID: partitionID}.
        Also, choose a set of exemplars for each partition and save
        them as PARTITION_EXMPLS: {partitionID: [int BallotID_i, ...]}
        """
        # partitioning: {int partitionID: [int ballotID_i, ...]}
        partitions_invmap = {}
        partition_exmpls = {}
        image_to_page = {} # maps {str imgpath: int side}
        image_to_flip = {} # maps {str imgpath: bool isflip}
        img2b = pickle.load(open(self.proj.image_to_ballot, 'rb'))
        b2imgs = pickle.load(open(self.proj.ballot_to_images, 'rb'))
        for partitionID, ballotIDs in self.partitionpanel.partitioning.iteritems():
            exmpls = set()
            for ballotID in ballotIDs:
                if ballotID in self.partitionpanel.quarantined_bals:
                    continue
                if len(exmpls) <= self.NUM_EXMPLS:
                    exmpls.add(ballotID)
                partitions_invmap[ballotID] = partitionID
                imgpaths = b2imgs[ballotID]
                for imgpath in imgpaths:
                    image_to_page[imgpath] = self.partitionpanel.imginfo[imgpath]['page']
                    image_to_flip[imgpath] = self.partitionpanel.imginfo[imgpath]['isflip']
            if exmpls:
                partition_exmpls[partitionID] = sorted(list(exmpls))
        partitions_map_outP = pathjoin(self.proj.projdir_path, self.proj.partitions_map)
        partitions_invmap_outP = pathjoin(self.proj.projdir_path, self.proj.partitions_invmap)
        decoded_map_outP = pathjoin(self.proj.projdir_path, self.proj.decoded_map)
        imginfo_map_outP = pathjoin(self.proj.projdir_path, self.proj.imginfo_map)
        bbs_map_outP = pathjoin(self.proj.projdir_path, self.proj.barcode_bbs_map)
        partition_exmpls_outP = pathjoin(self.proj.projdir_path, self.proj.partition_exmpls)
        pickle.dump(self.partitionpanel.partitioning, open(partitions_map_outP, 'wb'),
                    pickle.HIGHEST_PROTOCOL)
        pickle.dump(partitions_invmap, open(partitions_invmap_outP, 'wb'),
                    pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.partitionpanel.decoded, open(decoded_map_outP, 'wb'),
                    pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.partitionpanel.imginfo, open(imginfo_map_outP, 'wb'),
                    pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.partitionpanel.bbs_map, open(bbs_map_outP, 'wb'),
                    pickle.HIGHEST_PROTOCOL)
        pickle.dump(image_to_page, open(pathjoin(self.proj.projdir_path,
                                                 self.proj.image_to_page), 'wb'),
                    pickle.HIGHEST_PROTOCOL)
        pickle.dump(image_to_flip, open(pathjoin(self.proj.projdir_path,
                                                 self.proj.image_to_flip), 'wb'),
                    pickle.HIGHEST_PROTOCOL)
        pickle.dump(partition_exmpls, open(partition_exmpls_outP, 'wb'),
                    pickle.HIGHEST_PROTOCOL)
        
class PartitionPanel(ScrolledPanel):
    PARTITION_JOBID = util.GaugeID("PartitionJobId")

    def __init__(self, parent, *args, **kwargs):
        ScrolledPanel.__init__(self, parent, *args, **kwargs)
        
        self.voteddir = None
        # PARTITIONING: maps {int partitionID: [int ballotID_i, ...]}
        self.partitioning = None
        # DECODED: maps {int ballotID: [(str barcode_side0, ...), ...]}
        self.decoded = None
        # IMGINFO: maps {str imgpath: {str key: str val}}
        self.imginfo = None
        # BBS_MAP: maps {str imgpath: [[x1,y1,x2,y2],...]}
        self.bbs_map = None

        self.quarantined_bals = set()

        self.init_ui()

    def init_ui(self):
        self.sizer_stats = wx.BoxSizer(wx.HORIZONTAL)
        txt1 = wx.StaticText(self, label="Number of Partitions: ")
        self.num_partitions_txt = wx.StaticText(self)
        self.sizer_stats.AddMany([(txt1,), (self.num_partitions_txt,)])
        self.sizer_stats.ShowItems(False)

        btn_run = wx.Button(self, label="Run Partitioning...")
        btn_run.Bind(wx.EVT_BUTTON, self.onButton_run)
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        btn_sizer.AddMany([(btn_run,)])
        
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.AddMany([(self.sizer_stats,), (btn_sizer,)])
        self.SetSizer(self.sizer)
        self.Layout()
        self.SetupScrolling()

    def start(self, proj, voteddir, stateP='_state_partition.p'):
        """ 
        Input:
            str VOTEDDIR: Root directory of voted ballots.
        """
        self.proj = proj
        self.voteddir = voteddir
        self.stateP = stateP
        self.restore_session()
        
    def stop(self):
        self.save_session()

    def restore_session(self):
        try:
            state = pickle.load(open(self.stateP, 'rb'))
            self.voteddir = state['voteddir']
            self.partitioning = state['partitioning']
            self.decoded = state['decoded']
            self.imginfo = state['imginfo']
            self.bbs_map = state['bbs_map']
            self.quarantined_bals = state['quarantined_bals']
            if self.partitioning != None:
                self.num_partitions_txt.SetLabel(str(len(self.partitioning)))
                self.sizer_stats.ShowItems(True)
                self.Layout()
        except:
            return False
        return True
    def save_session(self):
        print "...PartitionPanel: Saving state..."
        state = {'voteddir': self.voteddir,
                 'partitioning': self.partitioning,
                 'decoded': self.decoded,
                 'imginfo': self.imginfo,
                 'bbs_map': self.bbs_map,
                 'quarantined_bals': self.quarantined_bals}
        pickle.dump(state, open(self.stateP, 'wb'))

    def onButton_run(self, evt):
        class PartitionThread(threading.Thread):
            def __init__(self, b2imgs, vendor_obj, callback, jobid, queue, tlisten, *args, **kwargs):
                threading.Thread.__init__(self, *args, **kwargs)
                self.b2imgs = b2imgs
                self.vendor_obj = vendor_obj
                self.callback = callback
                self.jobid = jobid
                self.queue = queue
                self.tlisten = tlisten
            def run(self):
                partitioning, decoded, imginfo, bbs_map, verifypatch_bbs, err_imgpaths = self.vendor_obj.partition_ballots(self.b2imgs, queue=queue)
                wx.CallAfter(Publisher().sendMessage, "signals.MyGauge.done", (self.jobid,))
                wx.CallAfter(self.callback, partitioning, decoded, imginfo, bbs_map, verifypatch_bbs, err_imgpaths)
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

        vendor_obj = self.proj.vendor_obj
        b2imgs = pickle.load(open(self.proj.ballot_to_images, 'rb'))
        queue = Queue.Queue()
        tlisten = ListenThread(queue, self.PARTITION_JOBID)
        t = PartitionThread(b2imgs, vendor_obj, self.on_partitiondone,
                            self.PARTITION_JOBID, queue, tlisten)
        numtasks = len(b2imgs)
        gauge = util.MyGauge(self, 1, thread=t, msg="Running Partitioning...",
                             job_id=self.PARTITION_JOBID)
        tlisten.start()
        t.start()
        gauge.Show()
        wx.CallAfter(Publisher().sendMessage, "signals.MyGauge.nextjob", (numtasks, self.PARTITION_JOBID))
        
    def on_partitiondone(self, partitioning, decoded, imginfo, bbs_map, verifypatch_bbs, err_imgpaths):
        """
        Input:
            dict PARTITIONING: {int partitionID: [int ballotID_i, ...]}
            dict DECODED: {int ballotID: [(str barcode_side0, ...), ...]}
            dict IMGINFO: {str imgpath: {str KEY: str VAL}}
            dict BBS_MAP: {str imgpath: [[x1,y1,x2,y2], ...]}
            dict VERIFYPATCH_BBS: {str bc_val: [(imgpath, (x1,y1,x2,y2)), ...]}
            list ERR_IMGPATHS:
        """
        print "...Partitioning Done..."
        print partitioning, '\n'
        print decoded, '\n'
        print imginfo, '\n'
        print bbs_map, '\n'
        print err_imgpaths

        # TODO: Handle ERR_IMGPATHS. Maybe have the user manually go
        # through each one, either quarantining or manually-labeling each one?
        # For now, I'm just going to quarantine all of them. 
        img2bal = pickle.load(open(self.proj.image_to_ballot, 'rb'))
        for err_imgpath in err_imgpaths:
            ballotid = img2bal[err_imgpath]
            self.quarantine_ballot(ballotid)

        # For now, just blindly accept the partitioning w/out verifying,
        # until Overlay Verification is implemented.
        self.partitioning = partitioning
        self.decoded = decoded
        self.imginfo = imginfo
        self.bbs_map = bbs_map
        self.verifypatch_bbs = verifypatch_bbs
        self.err_imgpaths = err_imgpaths
        self.num_partitions_txt.SetLabel(str(len(partitioning)))
        self.sizer_stats.ShowItems(True)
        self.Layout()

        #self.start_verify(partitioning, decoded, imginfo, bbs_map, verifypatch_bbs, err_imgpaths)

    def start_verify(self, partitioning, decoded, imginfo, bbs_map, verifypatch_bbs, err_imgpaths):
        # TODO: Verify the patches in VERIFYPATCH_BBS via overlay-verification,
        # then save the results.
        # I think we'll have 'categories' be dictated by the keys
        # of verifypatch_bbs (which will be something like:
        #     {'TIMINGMARK_ON': [(imgpath, (x1,y1,x2,y2)), ...],
        #      'TIMINGMARK_OFF': [(imgpath, (x1,y1,x2,y2)), ...]}
        # (at least for Diebold/Sequoia/ES&S). In this case, the cat_tags
        # will be 'TIMINGMARK_ON' and 'TIMINGMARK_OFF'.
        # For exmplcats (which keeps track of exemplar patches), for now 
        # you can hardcode these to whatever you like until I figure out
        # how to best handle this...maybe we just don't show exemplar_patches
        # for barcode overlay verification...? Or perhaps the partition
        # code has to provide examples of each category.
        imgcats = {} # maps {cat_tag: {grouptag: [imgpath_i, ...]}}
        exmplcats = {} # maps {cat_tag: {grouptag: [imgpath_i, ...]}}
        f = VerifyOverlaysFrame(self, imgcats, exmplcats, self.on_verify_done)
        f.Maximize()
        f.Show()

    def on_verify_done(self, verify_results):
        """ Receives the (corrected) results from VerifyOverlays.
        Input:
        dict VERIFY_RESULTS: {cat_tag: {grouptag: [imgpath_i, ...]}}
            For each category CAT_TAG, each group GROUPTAG maps to a set
            of imgpaths that the user claimed is part of GROUPTAG.
        """
        # TODO: Take the (verified) results from VerifyOverlays, and 
        # apply any fixes to the right data structures.
        pass

    def quarantine_ballot(self, ballotid):
        self.quarantined_bals.add(ballotid)

class VerifyOverlaysFrame(wx.Frame):
    def __init__(self, parent, imgcategories, exmplcategories, ondone, *args, **kwargs):
        """
        Input:
        dict IMGCATEGORIES: {cat_tag: {grouptag: [imgpath_i, ...]}}
            For each category CAT_TAG, GROUPTAG is an identifier for
            a set of imgpaths. 
        dict EXMPLCATEGORIES: {cat_tag: {grouptag: [exmplpath_i, ...]}}
            For each category CAT_TAG, GROUPTAG is an identifier for
            a set of exemplar imgpatches.
        fn ONDONE: Callback function to call after verification is done.
        """
        wx.Frame.__init__(self, parent, size=(600, 500), *args, **kwargs)

        self.verifyoverlays = verify_overlays_new.VerifyOverlaysMultCats(self)

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.verifyoverlays, proportion=1, flag=wx.EXPAND)
        self.SetSizer(self.sizer)
        self.Layout()

        self.verifyoverlays.start(imgcategories, exmplcategories, 
                                  do_align=True, ondone=self.ondone)

        self.Layout()
