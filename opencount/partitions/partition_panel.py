import sys, os, pdb, traceback, threading, multiprocessing, Queue, time, textwrap
try:
    import cPickle as pickle
except:
    import pickle

from os.path import join as pathjoin
import wx
from wx.lib.scrolledpanel import ScrolledPanel
from wx.lib.pubsub import Publisher

sys.path.append('..')

import extract_patches
import util
import barcode.partition_imgs as partition_imgs
import grouping.label_imgs as label_imgs
import grouping.verify_overlays_new as verify_overlays_new

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
        # Note: IMAGE_TO_PAGE will normalize the pages s.t. they start
        # from 0, and increase by 1. 
        image_to_page = {} # maps {str imgpath: int side}
        image_to_flip = {} # maps {str imgpath: bool isflip}
        img2b = pickle.load(open(self.proj.image_to_ballot, 'rb'))
        b2imgs = pickle.load(open(self.proj.ballot_to_images, 'rb'))
        curPartID = 0

        # 0.) Record all pages outputted by the decoder, in order to
        # normalize the pages to start at 0.
        pages_counter = util.Counter()
        for imgpath, imginfo in self.partitionpanel.imginfo.iteritems():
            pages_counter[imginfo['page']] += 1
        pages_norm_map = {} # maps {int decoderPage: int normPage}
        for i, decoderPage in enumerate(sorted(pages_counter.keys())):
            pages_norm_map[decoderPage] = i

        # 1.) Perform a few sanity checks if this is a single-sided 
        #     election. 
        if not self.proj.is_varnum_pages and self.proj.num_pages == 1 and len(pages_norm_map) != self.proj.num_pages:
            print "...Uhoh, detected {0} pages, but election specifies {1} pages.".format(len(pages_norm_map), self.proj.num_pages)
            msg = "Warning: The user specified \
that this is a {0}-sided election. However, OpenCount just detected that {1} \
sides are present. \n".format(self.proj.num_pages, len(pages_norm_map))
            for decoderPage in pages_norm_map.keys():
                msg += "    Page {0}: {1} images".format(decoderPage, pages_counter[decoderPage])
                msg += "\n"
            msg += "What should OpenCount do?"
            dlg = BadPagesDialog(self, msg, pages_counter)
            status = dlg.ShowModal()
            if status == BadPagesDialog.ID_TREATNORMAL:
                # map everything to the 0 page
                for decoderPage in pages_norm_map.keys()[:]:
                    pages_norm_map[decoderPage] = 0
            else:
                keepDecoderPage = sorted(pages_norm_map.keys())[dlg.keep_page]
                pages_norm_map[keepDecoderPage] = 0
                doQuarantine = dlg.do_quarantine
                handleballot = self.partitionpanel.quarantine_ballot if doQuarantine else self.partitionpanel.discard_ballot
                for imgpath, imginfo in self.partitionpanel.imginfo.iteritems():
                    decoderPage = imginfo['page']
                    if decoderPage != keepDecoderPage:
                        if doQuarantine:
                            print "...quarantining ballot {0}".format(img2b[imgpath])
                        else:
                            print "...discarding ballot {0}".format(img2b[imgpath])
                        handleballot(img2b[imgpath])
                        
        # 1.) Build up partitions_map, partitions_invmap
        # Note: self.partitionpanel.partitioning may have partitions
        # with either no ballotids, or ballotids that are all quarantined/discarded.
        # Take care to detect these cases.
        for (partitionID, ballotIDs) in self.partitionpanel.partitioning.iteritems():
            if not ballotIDs:
                continue
            atLeastOne = False
            for ballotID in ballotIDs:
                if ballotID in self.partitionpanel.quarantined_bals or ballotID in self.partitionpanel.discarded_bals:
                    continue
                imgpaths = b2imgs[ballotID]
                for imgpath in imgpaths:
                    atLeastOne = True
                    decoderPage = self.partitionpanel.imginfo[imgpath]['page']
                    image_to_page[imgpath] = pages_norm_map[decoderPage]
                    image_to_flip[imgpath] = self.partitionpanel.flipmap[imgpath]
                partitions_map.setdefault(curPartID, []).append(ballotID)
                partitions_invmap[ballotID] = curPartID
            if atLeastOne:
                curPartID += 1
        # 2.) Grab NUM_EXMPLS number of exemplars from each partition
        for partitionID, ballotIDs in partitions_map.iteritems():
            exmpls = set()
            for ballotID in ballotIDs:
                if len(exmpls) <= self.NUM_EXMPLS:
                    exmpls.add(ballotID)
            if exmpls:
                partition_exmpls[partitionID] = sorted(list(exmpls))
        partitions_map_outP = pathjoin(self.proj.projdir_path, self.proj.partitions_map)
        partitions_invmap_outP = pathjoin(self.proj.projdir_path, self.proj.partitions_invmap)
        img2decoding_outP = pathjoin(self.proj.projdir_path, self.proj.img2decoding)
        imginfo_map_outP = pathjoin(self.proj.projdir_path, self.proj.imginfo_map)
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
        pickle.dump(self.partitionpanel.img2decoding, open(img2decoding_outP, 'wb'),
                    pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.partitionpanel.imginfo, open(imginfo_map_outP, 'wb'),
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
        # IMG2DECODING: maps {imgpath: [str bc_i, ...]}
        self.img2decoding = None
        # IMGINFO: maps {str imgpath: {str key: str val}}
        self.imginfo = None

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

        msg = textwrap.fill("Would you like to skip barcode overlay \
verification? This tends to be computationally time-consuming, not \
very helpful for certain vendors (e.g. Hart), and unnecessary to \
repeat.", 100)
        txt_skipHelp = wx.StaticText(self, label=msg)
        self.chkbox_skip_verify = wx.CheckBox(self, label="Skip Overlay Verification?")
        
        sizer_skipVerify = wx.BoxSizer(wx.VERTICAL)
        sizer_skipVerify.AddMany([(txt_skipHelp,), (self.chkbox_skip_verify,)])

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.AddMany([(self.sizer_stats,), (btn_sizer,), ((50, 50),), (sizer_skipVerify,)])
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
            self.img2decoding = state['img2decoding']
            self.imginfo = state['imginfo']
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
                 'img2decoding': self.img2decoding,
                 'imginfo': self.imginfo,
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
                t = time.time()
                print "...Running Decoding ({0} ballots)...".format(len(self.b2imgs))
                flipmap, verifypatch_bbs, err_imgpaths = self.vendor_obj.decode_ballots(self.b2imgs, manager=self.manager, queue=self.queue)
                dur = time.time() - t
                print "...Done Decoding Ballots ({0} s).".format(dur)
                print "    Avg. Time Per Ballot:", dur / float(len(self.b2imgs))
                wx.CallAfter(Publisher().sendMessage, "signals.MyGauge.done", (self.jobid,))
                wx.CallAfter(self.callback, flipmap, verifypatch_bbs, err_imgpaths)
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
        t = PartitionThread(b2imgs, vendor_obj, self.on_decodedone,
                            self.PARTITION_JOBID, manager, progress_queue, tlisten)
        numtasks = len(b2imgs)
        gauge = util.MyGauge(self, 1, thread=t, msg="Running Partitioning...",
                             job_id=self.PARTITION_JOBID)
        tlisten.start()
        t.start()
        gauge.Show()
        wx.CallAfter(Publisher().sendMessage, "signals.MyGauge.nextjob", (numtasks, self.PARTITION_JOBID))
        
    def on_decodedone(self, flipmap, verifypatch_bbs, err_imgpaths):
        """
        Input:
            dict FLIPMAP: {imgpath: bool isFlipped}
            dict VERIFYPATCH_BBS: {str bc_val: [(imgpath, (x1,y1,x2,y2), userdata), ...]}
            list ERR_IMGPATHS:
        """
        print "...Decoding Done!"
        print 'Errors ({0} total): {1}'.format(len(err_imgpaths), err_imgpaths)

        if err_imgpaths:
            dlg = LabelDialog(self, err_imgpaths)
            status = dlg.ShowModal()
            # dict ERRS_CORRECTED: {str imgpath: str label or ID_Quarantine/ID_Discard}
            self.errs_corrected = dlg.label_res
            self.errs_flipmap = dlg.imgflips
        else:
            self.errs_corrected = {}
            self.errs_flipmap = {}

        # For a ballot B in ERRS_CORRECTED, if any of its sides was
        # quarantined/discarded, then don't process the rest of B.
        img2bal = pickle.load(open(self.proj.image_to_ballot, 'rb'))
        for imgpath, label in self.errs_corrected.iteritems():
            ballotid = img2bal[imgpath]
            if label in (LabelDialog.ID_Quarantine, LabelDialog.ID_Discard):
                # Remove all mentions of this ballot from relevant data structs
                for bc_val, tups in verifypatch_bbs.iteritems():
                    i = 0
                    while i < len(tups):
                        tup_imP, tup_bb, tup_userdata = tups[i]
                        tup_balid = img2bal[tup_imP]
                        if tup_balid == ballotid:
                            tups.pop(i)
                        else:
                            i += 1
                for flipmap_imP in flipmap.keys():
                    flipmap_bid = img2bal[flipmap_imP]
                    if flipmap_bid == ballotid:
                        flipmap.pop(flipmap_imP)
            if label == LabelDialog.ID_Quarantine:
                self.quarantine_ballot(ballotid)
            elif label == LabelDialog.ID_Discard:
                self.discard_ballot(ballotid)

        self.start_verify(flipmap, verifypatch_bbs)

    def start_verify(self, flipmap, verifypatch_bbs):
        """
        Input:
            dict FLIPMAP: maps {str imgpath: bool isflip}
            dict VERIFYPATCH_BBS: maps {str bc_val: [(imgpath, (x1,y1,x2,y2), userdata), ...]}
        """
        if self.chkbox_skip_verify.GetValue():
            print "...Skipping Barcode Overlay Verification..."
            self.on_verify_done(None, None, flipmap, verifypatch_bbs, skipVerify=True)
            return
        
        # 1.) Extract all patches to an outdir
        imgpatches = {} # {imgpath: [((x1,y1,x2,y2), isflip, outpath, tag), ...]}
        outrootdir = pathjoin(self.proj.projdir_path, '_barcode_extractpats')
        bc_val_cnt = {} # maps {bc_val: int cnt}
        bc_val_dircnt = {} # maps {bc_val: int dircnt}
        img_ctr = util.Counter()
        print "...creating jobs for barcode-patch extraction..."
        for bc_val, tups in verifypatch_bbs.iteritems():
            for (imgpath, (x1,y1,x2,y2), userdata) in tups:
                i = bc_val_cnt.get(bc_val, None)
                if i == None: 
                    bc_val_cnt[bc_val] = 0
                    bc_val_dircnt[bc_val] = 0
                    i = 0
                if i != 0 and i % 750 == 0:
                    bc_val_dircnt[bc_val] += 1
                dircnt = bc_val_dircnt[bc_val]
                imgname = os.path.splitext(os.path.split(imgpath)[1])[0]
                # Recreate directory structure
                rp = os.path.splitext(os.path.relpath(os.path.abspath(imgpath), os.path.abspath(self.proj.voteddir)))[0]
                outpath = pathjoin(outrootdir, rp, "{0}_{1}.png".format(imgname, img_ctr[imgpath]))
                img_ctr[imgpath] += 1
                # use the loc as the ID, in case USERDATA isn't used
                ID = (x1,y1,x2,y2) 
                tag = (bc_val, userdata, ID)
                isflip = flipmap[imgpath]
                imgpatches.setdefault(imgpath, []).append(((x1,y1,x2,y2), isflip, outpath, tag))
                i += 1
        print '...extracting...'
        t = time.time()
        img2patch, patch2stuff = extract_patches.extract(imgpatches)
        dur = time.time() - t
        print '...done extracting ({0} s)...'.format(dur)
        print "    Avg. Time Per Image:", dur / float(len(imgpatches))
        cattag = 'BarcodeCategory'
        imgcats = {} # maps {cat_tag: {grouptag: [imgpath_i, ...]}}
        exmplcats = {} # maps {cat_tag: {grouptag: [imgpath_i, ...]}}
        for bc_val, tups in verifypatch_bbs.iteritems():
            for (imgpath, (x1,y1,x2,y2), userdata) in tups:
                id = (x1,y1,x2,y2)
                patchpath = img2patch[(imgpath, (bc_val, userdata, id))]
                imgcats.setdefault(cattag, {}).setdefault(bc_val, []).append(patchpath)
        callback = lambda verifyRes: self.on_verify_done(verifyRes, patch2stuff, flipmap, verifypatch_bbs)
        f = VerifyOverlaysFrame(self, imgcats, exmplcats, callback)
        f.Maximize()
        f.Show()

    def on_verify_done(self, verify_results, patch2stuff, flipmap, verifypatch_bbs, skipVerify=False):
        """ Receives the (corrected) results from VerifyOverlays.
        Input:
        dict VERIFY_RESULTS: {cat_tag: {grouptag: [imgpath_i, ...]}}
            For each category CAT_TAG, each group GROUPTAG maps to a set
            of imgpaths that the user claimed is part of GROUPTAG.
        """
        print "...barcode patch verification done!"
        verified_decodes = {} # maps {str bc_val: [(imgpath, (x1,y1,x2,y2), userdata), ...]}
        if skipVerify:
            verified_decodes = verifypatch_bbs
        else:
            for cat_tag, thedict in verify_results.iteritems():
                for bc_val, patchpaths in thedict.iteritems():
                    for patchpath in patchpaths:
                        imgpath, bb, (bc_val_this, userdata, id) = patch2stuff[patchpath]
                        verified_decodes.setdefault(bc_val, []).append((imgpath, bb, userdata))
        manual_labeled = {} # maps {str imgpath: str label}
        for imgpath, label in self.errs_corrected.iteritems():
            if label not in (LabelDialog.ID_Quarantine, LabelDialog.ID_Discard):
                # TODO: Officially document (or modify textentry widget) 
                # that commas separate barcode values
                decoding = tuple([s.strip() for s in label.split(",")])
                manual_labeled[imgpath] = decoding
        print "...generating partitions..."
        # dict PARTITIONING: maps {int partitionID: [int ballotID_i, ...]}
        partitioning, img2decoding, imginfo_map = self.proj.vendor_obj.partition_ballots(verified_decodes, manual_labeled)
        print "...done generating partitions..."
        # Add in manually-corrected flipped
        for imgpath, isflip in self.errs_flipmap.iteritems():
            flipmap[imgpath] = isflip
        self.partitioning = partitioning
        self.img2decoding = img2decoding
        self.imginfo = imginfo_map
        self.flipmap = flipmap

        # Finally, sanity check that, within each partition, each ballot
        # has the same number of pages.
        bal2imgs = pickle.load(open(self.proj.ballot_to_images, 'rb'))
        bad_ballotids = []
        for partitionID, ballotids in partitioning.iteritems():
            num_pages = max([len(bal2imgs[b]) for b in ballotids])
            cur_bad_ballotids = [b for b in ballotids if len(bal2imgs[b]) != num_pages]
            if cur_bad_ballotids:
                print '...REMOVING {0} ballots from partition {1}...'.format(len(cur_bad_ballotids),
                                                                             partitionID)
                print "    Should have {0} Pages".format(num_pages)
                print "    {0}".format([len(bal2imgs[b]) for b in cur_bad_ballotids])
                                                                                 
            bad_ballotids.extend(cur_bad_ballotids)
            cur_bad_ballotids = set(cur_bad_ballotids)
            ballotids[:] = [b for b in ballotids if b not in cur_bad_ballotids]
        # For each 'bad' ballotid, add them into its own new partition
        print "...There were {0} ballotids with anomalous page numbers. \
Adding to separate partitions...".format(len(bad_ballotids))
        curPartId = len(self.partitioning)
        for badballotid in bad_ballotids:
            self.partitioning[curPartId] = [badballotid]
            curPartId += 1

        # Also, for single-sided elections, quarantine any ballots which
        # has a very-rare page. NOTE: Commenting out this check, since
        # it might be best to just partition also by page for single-sided.
        '''
        if self.proj.num_pages == 1:
            page_counter = util.Counter() # keeps track of page# occurrences
            # 0.) Initialize page count PAGE_COUNTER
            for partitionID, ballotIDs in self.partitioning.iteritems():
                for ballotID in ballotIDs:
                    if ballotID in self.quarantined_bals or ballotID in self.discarded_bals:
                        continue
                    imgpaths = bal2imgs[ballotID]
                    for imgpath in imgpaths:
                        page = self.imginfo[imgpath]['page']
                        page_counter[page] += 1
            def is_anomalous_page(page, page_stats, T=0.02):
                """ Reject pages that rarely occur """
                if page not in page_stats:
                    return True
                elif page_stats[page] <= T:
                    return True
                return False
            # 0.a.) Compute page statistics
            page_stats = {} # maps {page: float percentage}
            total_count = sum(page_counter.values())
            for pagenum, count in page_counter.iteritems():
                page_stats[pagenum] = count / float(total_count)
            print page_stats
            pdb.set_trace()
            # 1.) Perform anomaly detection
            anom_cnt = 0
            for partitionid, ballotids in self.partitioning.iteritems():
                for ballotid in ballotids:
                    if ballotID in self.quarantined_bals or ballotID in self.discarded_bals:
                        continue
                    imgpaths = bal2imgs[ballotid]
                    flagit = False
                    for imgpath in imgpaths:
                        page = self.imginfo[imgpath]['page']
                        if is_anomalous_page(page, page_stats):
                            flagit = True
                            anom_cnt += 1
                            break
                    if flagit:
                        self.quarantine_ballot(ballotid)
            print "    Detected {0} anomalous ballots (weird page number)".format(anom_cnt)
        '''
        # 2.) Finally, remove all quarantined/discarded ballotids from
        # self.PARTITIONING.
        bad_ballotids = self.quarantined_bals.union(self.discarded_bals)
        for partitionid, ballotids in self.partitioning.iteritems():
            i = 0
            while i < len(ballotids):
                ballotid = ballotids[i]
                if ballotid in bad_ballotids:
                    ballotids.pop(i)
                else:
                    i += 1
        # Export results.
        self.GetParent().export_results()

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

        self.ondone = ondone

        self.verifyoverlays = verify_overlays_new.VerifyOverlaysMultCats(self)

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.verifyoverlays, proportion=1, flag=wx.EXPAND)
        self.SetSizer(self.sizer)
        self.Layout()

        self.verifyoverlays.start(imgcategories, exmplcategories, 
                                  do_align=False, ondone=self.on_verify_done)

        self.Layout()
    
    def on_verify_done(self, verify_results):
        self.Close()
        self.ondone(verify_results)

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
        self.imgflips[curimgpath] = self.chkbox_isflip.GetValue()
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

class BadPagesDialog(wx.Dialog):
    ID_TREATNORMAL = 42
    ID_KEEPONE = 43

    def __init__(self, parent, msg, page_counter, *args, **kwargs):
        wx.Dialog.__init__(self, parent, title="User action required.", size=(700, 200), *args, **kwargs)

        self.keep_page = None
        self.do_quarantine = None
        
        txt = wx.StaticText(self, label=msg)

        btn_treatNormal = wx.Button(self, label="Treat all ballots as separate pages")
        btn_treatNormal.Bind(wx.EVT_BUTTON, self.onButton_treatNormal)

        txt_choose = wx.StaticText(self, label="Or, process only one side:")
        choices = []
        for page in sorted(page_counter):
            cnt = page_counter[page]
            choices.append("Page {0} -- {1} images".format(page, cnt))
        self.cb_pages = wx.ComboBox(self, choices=choices, style=wx.CB_READONLY)

        sizer_choose = wx.BoxSizer(wx.VERTICAL)

        sizer_choose.AddMany([(txt_choose,), (self.cb_pages,)])
        
        txt_others = wx.StaticText(self, label="And do the following to the other sides:")
        self.rb_quarantine = wx.RadioButton(self, label="Quarantine the other sides", style=wx.RB_GROUP)
        self.rb_discard = wx.RadioButton(self, label="Discard the other sides")

        sizer_others = wx.BoxSizer(wx.VERTICAL)
        sizer_others.AddMany([(txt_others,), (self.rb_quarantine,), (self.rb_discard,)])

        sizer2 = wx.BoxSizer(wx.HORIZONTAL)
        sizer2.AddMany([(sizer_choose,), (sizer_others)])
        btn_ok = wx.Button(self, label="Ok")
        btn_ok.Bind(wx.EVT_BUTTON, self.onButton_ok)
        sizer2.Add(btn_ok, flag=wx.ALIGN_CENTER)

        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        btn_sizer.AddMany([(btn_treatNormal,), ((50,0),), (sizer2,)])

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.AddMany([(txt,), (btn_sizer,)])

        self.SetSizer(sizer)
        self.Fit()

    def onButton_treatNormal(self, evt):
        self.EndModal(self.ID_TREATNORMAL)
    def onButton_ok(self, evt):
        self.keep_page = self.cb_pages.GetSelection()
        self.do_quarantine = True if self.rb_quarantine.GetValue() else False
        self.EndModal(self.ID_KEEPONE)
