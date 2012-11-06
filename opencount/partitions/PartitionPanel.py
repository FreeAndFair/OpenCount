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
                if len(exmpls) <= self.NUM_EXMPLS:
                    exmpls.add(ballotID)
                partitions_invmap[ballotID] = partitionID
                imgpaths = b2imgs[ballotID]
                for imgpath in imgpaths:
                    image_to_page[imgpath] = self.partitionpanel.imginfo[imgpath]['page']
                    image_to_flip[imgpath] = self.partitionpanel.imginfo[imgpath]['isflip']
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
                 'bbs_map': self.bbs_map}
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
                partitioning, decoded, imginfo, bbs_map = self.vendor_obj.partition_ballots(self.b2imgs, queue=queue)
                wx.CallAfter(Publisher().sendMessage, "signals.MyGauge.done", (self.jobid,))
                wx.CallAfter(self.callback, partitioning, decoded, imginfo, bbs_map)
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
        
    def on_partitiondone(self, partitioning, decoded, imginfo, bbs_map):
        """
        Input:
            dict PARTITIONING: {int partitionID: [int ballotID_i, ...]}
            dict DECODED: {int ballotID: [(str barcode_side0, ...), ...]}
            dict IMGINFO: {str imgpath: {str KEY: str VAL}}
            dict BBS_MAP: {str imgpath: [[x1,y1,x2,y2], ...]}
        """
        print "...Partitioning Done..."
        print partitioning
        print
        print decoded
        print
        print imginfo
        print
        print bbs_map
        self.partitioning = partitioning
        self.decoded = decoded
        self.imginfo = imginfo
        self.bbs_map = bbs_map
        self.num_partitions_txt.SetLabel(str(len(partitioning)))
        self.sizer_stats.ShowItems(True)
        self.Layout()

