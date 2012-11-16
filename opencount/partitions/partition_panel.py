import sys, os, pdb, traceback, threading, multiprocessing, Queue
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
import grouping.label_imgs as label_imgs

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
        partitions_map = {} 
        partitions_invmap = {}
        partition_exmpls = {}
        image_to_page = {} # maps {str imgpath: int side}
        image_to_flip = {} # maps {str imgpath: bool isflip}
        img2b = pickle.load(open(self.proj.image_to_ballot, 'rb'))
        b2imgs = pickle.load(open(self.proj.ballot_to_images, 'rb'))
        curPartID = 0
        for (partitionID, ballotIDs) in self.partitionpanel.partitioning.iteritems():
            exmpls = set()
            for ballotID in ballotIDs:
                if ballotID in self.partitionpanel.quarantined_bals:
                    continue
                if len(exmpls) <= self.NUM_EXMPLS:
                    exmpls.add(ballotID)
                partitions_map.setdefault(curPartID, []).extend(ballotIDs)
                partitions_invmap[ballotID] = curPartID
                imgpaths = b2imgs[ballotID]
                for imgpath in imgpaths:
                    image_to_page[imgpath] = self.partitionpanel.imginfo[imgpath]['page']
                    image_to_flip[imgpath] = self.partitionpanel.imginfo[imgpath]['isflip']
            if exmpls:
                partition_exmpls[curPartID] = sorted(list(exmpls))
                curPartID += 1
        partitions_map_outP = pathjoin(self.proj.projdir_path, self.proj.partitions_map)
        partitions_invmap_outP = pathjoin(self.proj.projdir_path, self.proj.partitions_invmap)
        decoded_map_outP = pathjoin(self.proj.projdir_path, self.proj.decoded_map)
        imginfo_map_outP = pathjoin(self.proj.projdir_path, self.proj.imginfo_map)
        bbs_map_outP = pathjoin(self.proj.projdir_path, self.proj.barcode_bbs_map)
        partition_exmpls_outP = pathjoin(self.proj.projdir_path, self.proj.partition_exmpls)
        # Finally, also output the quarantined/discarded ballots
        pickle.dump(tuple(self.partitionpanel.quarantined_bals), 
                    open(pathjoin(self.proj.projdir_path, self.proj.partition_quarantined), 'wb'),
                    pickle.HIGHEST_PROTOCOL)
        pickle.dump(tuple(self.partitionpanel.discarded_bals),
                    open(pathjoin(self.proj.projdir_path, self.proj.partition_discarded), 'wb'),
                    pickle.HIGHEST_PROTOCOL)
        pickle.dump(partitions_map, open(partitions_map_outP, 'wb'),
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
        self.discarded_bals = set()

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
            self.discarded_bals = state['discarded_bals']
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
                 'quarantined_bals': self.quarantined_bals,
                 'discarded_bals': self.discarded_bals}
        pickle.dump(state, open(self.stateP, 'wb'))

    def onButton_run(self, evt):
        class PartitionThread(threading.Thread):
            def __init__(self, b2imgs, vendor_obj, callback, jobid, manager, progress_queue, tlisten, *args, **kwargs):
                threading.Thread.__init__(self, *args, **kwargs)
                self.b2imgs = b2imgs
                self.vendor_obj = vendor_obj
                self.callback = callback
                self.jobid = jobid
                self.manager = manager
                self.queue = progress_queue
                self.tlisten = tlisten
            def run(self):
                partitioning, decoded, imginfo, bbs_map, verifypatch_bbs, err_imgpaths = self.vendor_obj.partition_ballots(self.b2imgs, manager=self.manager, queue=self.queue)
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
        manager = multiprocessing.Manager()
        progress_queue = manager.Queue()
        tlisten = ListenThread(progress_queue, self.PARTITION_JOBID)
        t = PartitionThread(b2imgs, vendor_obj, self.on_partitiondone,
                            self.PARTITION_JOBID, manager, progress_queue, tlisten)
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
        #print partitioning, '\n'
        #print decoded, '\n'
        #print imginfo, '\n'
        #print bbs_map, '\n'
        print 'Errors ({0} total): {1}'.format(len(err_imgpaths), err_imgpaths)

        if err_imgpaths:
            dlg = LabelDialog(self, err_imgpaths)
            status = dlg.ShowModal()
            # dict ERRS_CORRECTED: {str imgpath: str label or ID_Quarantine/ID_Discard}
            self.errs_corrected = dlg.label_res
            imgflips = dlg.imgflips
            # build a ballotid->partitionid map now, for performance
            bal2partition = {}
            for partitionid, ballotids in partitioning.iteritems():
                for bid in ballotids:
                    bal2partition[bid] = partitionid
        else:
            self.errs_corrected = {}
            imgflips = {}
            bal2partition = {} # not used if d.n.e. ERR_IMGPATHS

        # For a ballot B, if any of its sides is quarantined/discarded,
        # then don't process the rest of B.
        img2bal = pickle.load(open(self.proj.image_to_ballot, 'rb'))
        for imgpath, label in self.errs_corrected.iteritems():
            if label in (LabelDialog.ID_Quarantine, LabelDialog.ID_Discard):
                ballotid = img2bal[imgpath]
                partitionid = bal2partition[ballotid]
                try: partitioning[partitionid].remove(ballotid)
                except: pass
                if partitionid in partitioning and not partitioning[partitionid]:
                    # Minor housecleaning - pop off empty partitions as they arise
                    partitioning.pop(partitionid)
                try: decoded.pop(ballotid)
                except: pass
                try: imginfo.pop(imgpath)
                except: pass
                try: bbs_map.pop(imgpath)
                except: pass
            else:
                # TODO: For now, assume that multiple barcodes in the 
                # labeling-UI are separated by commas.
                bcs = label.split(",")
                info = self.proj.vendor_obj.get_barcode_info(bcs)
                info['isflip'] = imgflips[imgpath]
                imginfo[imgpath] = info
            if label == LabelDialog.ID_Quarantine:
                self.quarantine_ballot(ballotid)
            elif label == LabelDialog.ID_Discard:
                self.discard_ballot(ballotid)

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
    def discard_ballot(self, ballotid):
        self.discarded_bals.add(ballotid)

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

class LabelOrDiscardPanel(label_imgs.LabelPanel):
    """
    A widget that lets the user either manually label an image, quarantine,
    or discard.
    """
    def __init__(self, parent, *args, **kwargs):
        label_imgs.LabelPanel.__init__(self, parent, *args, **kwargs)

        self.quar_imgpaths = set()
        self.discard_imgpaths = set()
        self.imgflips = {}

    def _init_ui(self):
        label_imgs.LabelPanel._init_ui(self)

        self.radio_quarantine = wx.RadioButton(self, label="Quarantine (Process Later)", 
                                               style=wx.RB_GROUP)
        self.radio_discard = wx.RadioButton(self, label="Discard (Don't Process)")
        self.radio_normal = wx.RadioButton(self, label="Normal Ballot (Process Normally)")
        radiobtn_sizer = wx.BoxSizer(wx.VERTICAL)
        radiobtn_sizer.AddMany([(self.radio_quarantine,), (self.radio_discard,),
                                (self.radio_normal,)])
        self.chkbox_isflip = wx.CheckBox(self, label="Is the ballot flipped (upside down)?")
        self.btn_sizer.AddMany([(radiobtn_sizer,), (self.chkbox_isflip,)])

    def add_label(self, imgpath, label):
        curimgpath = self.imagepaths[self.cur_imgidx]
        if self.radio_quarantine.GetValue():
            self.quar_imgpaths.add(curimgpath)
        elif self.radio_discard.GetValue():
            self.discard_imgpaths.add(curimgpath)
        self.imgflips[imgpath] = self.chkbox_isflip.GetValue()
        return label_imgs.LabelPanel.add_label(self, imgpath, label)

    def display_img(self, *args, **kwargs):
        label_imgs.LabelPanel.display_img(self, *args, **kwargs)
        self.radio_quarantine.SetValue(False)
        self.radio_discard.SetValue(False)
        self.radio_normal.SetValue(False)
        self.chkbox_isflip.SetValue(False)
        curimgpath = self.imagepaths[self.cur_imgidx]
        if curimgpath in self.quar_imgpaths:
            self.radio_quarantine.SetValue(True)
        elif curimgpath in self.discard_imgpaths:
            self.radio_discard.SetValue(True)
        else:
            self.radio_normal.SetValue(True)
        if self.imgflips.get(curimgpath, False):
            self.chkbox_isflip.SetValue(True)

class LabelDialog(wx.Dialog):
    """ 
    A Modal Dialog that lets the user label a set of images.
    """
    class QuarantineID(object):
        def __str__(self):
            return "QuarantineID"
        def __repr__(self):
            return "QuarantineID()"
        def __eq__(self, o):
            return o and isinstance(o, type(self))
    class DiscardID(object):
        def __str__(self):
            return "DiscardID"
        def __repr__(self):
            return "DiscardID()"
        def __eq__(self, o):
            return o and isinstance(o, type(self))

    ID_Quarantine = QuarantineID()
    ID_Discard = DiscardID()
    def __init__(self, parent, imageslist, captions=None, possibles=None, 
                 outfile=None, *args, **kwargs):
        wx.Dialog.__init__(self, parent, title="Label These Images", 
                           size=(800, 600), style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER, 
                           *args, **kwargs)
        
        self.labelpanel = LabelOrDiscardPanel(self)

        self.sizer = wx.BoxSizer(wx.VERTICAL)

        self.sizer.Add(self.labelpanel, proportion=1, border=10, flag=wx.EXPAND | wx.ALL)
        self.SetSizer(self.sizer)
        self.Layout()

        self.labelpanel.start(imageslist, captions=captions, callback=self.on_label_done,
                              outfile=outfile, possibles=possibles)

        self.Layout()

    def on_label_done(self, label_res):
        """
        Input:
        dict LABEL_RES: {str imgpath: str label}
        """
        # Also grab the quarantined/discarded images
        label_res_cpy = label_res.copy()
        for quar_imgpath in self.labelpanel.quar_imgpaths:
            label_res_cpy[quar_imgpath] = self.ID_Quarantine
        for discard_imgpath in self.labelpanel.discard_imgpaths:
            label_res_cpy[discard_imgpath] = self.ID_Discard
        self.label_res = label_res_cpy
        self.imgflips = self.labelpanel.imgflips
        self.EndModal(wx.ID_OK)

