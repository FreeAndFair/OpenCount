import os
import multiprocessing
import Queue
import shutil
import textwrap
import threading

try:
    import cPickle as pickle
except:
    import pickle

from os.path import join as pathjoin
import wx
from wx.lib.scrolledpanel import ScrolledPanel


import util
import config
from util import debug, warn, error
import grouping.label_imgs as label_imgs
import grouping.verify_overlays_new as verify_overlays_new
import s03_partition.extract_patches as extract_patches
import ffwx

JOBID_EXPORT_RESULTS = util.GaugeID("PartitioningExportResults")


class Strings:
    BARCODE_OVERLAY_HELP = "Would you like to skip barcode overlay "\
        "verification? It tends to be computationally time-consuming, not "\
        "very helpful for certain vendors (e.g. Hart), and and typically is "\
        "unnecessary."
    UNEVEN_SIDES = "Warning: OpenCount detected an uneven number of "\
        "images per side. This violates the assumption that there are "\
        "{0} sides per ballot.\n\n "\
        "In particular, there were {1} ballots with an unusual number of "\
        "reported sides.\n\n\ "\
        "Is this perhaps an election with a variable-number of pages? \n\n "\
        "Or perhaps a few images are missing in the dataset? \n\n "\
        "These {1} ballots will be quarantined.\n\n "\
        "The page counts are:\n{2}\n "\
        "(Detailed info on these ballots have been saved to: \n "\
        "opencount/<projdir>/_ballots_unevenpages.txt"
    WRONG_NUM_PAGES = "User specified that this is an election with "\
        "{0} pages, yet partitioning only discovered {1} pages.\n "\
        "Please go back and correct the 'number of pages' option "\
        "in the previous step."


class PartitionMainPanel(ffwx.Panel):
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

    def start(self, project=None, projdir=None):
        stateP = project.path('_state_partition.p')
        self.proj = project
        self.partitionpanel.start(self.proj, self.proj.voteddir, stateP)
        self.proj.addCloseEvent(self.partitionpanel.save_session)

    def stop(self):
        self.partitionpanel.save_session()
        self.proj.removeCloseEvent(self.partitionpanel.save_session)

    def export_results(self, queue_mygauge, thread_listen):
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
        image_to_page = {}  # maps {str imgpath: int side}
        image_to_flip = {}  # maps {str imgpath: bool isflip}
        img2b = self.proj.load_field(self.proj.image_to_ballot)
        b2imgs = self.proj.load_field(self.proj.ballot_to_images)
        curPartID = 0

        # 0.) Record all pages outputted by the decoder, in order to
        # normalize the pages to start at 0.
        num_tasks = len(self.partitionpanel.imginfo) * 2
        JOBID_EXPORT_RESULTS.next_job(num_tasks)
        pages_counter = util.Counter()
        for imgpath, imginfo in self.partitionpanel.imginfo.iteritems():
            pages_counter[imginfo['page']] += 1
            JOBID_EXPORT_RESULTS.tick()
        pages_norm_map = {}  # maps {int decoderPage: int normPage}
        _prev_count = None
        flag_uneven_pages = False
        for i, decoderPage in enumerate(sorted(pages_counter.keys())):
            debug("There are {0} images from side {1}",
                  pages_counter[decoderPage],
                  decoderPage)
            if _prev_count is None:
                _prev_count = pages_counter[decoderPage]
            if pages_counter[decoderPage] != _prev_count:
                debug("...Uhoh, detected uneven pages, might be problem...")
                flag_uneven_pages = True
            _prev_count = pages_counter[decoderPage]
            pages_norm_map[decoderPage] = i
            JOBID_EXPORT_RESULTS.tick()

        # ## ## ## ## ## ## ## ## ## ## ##
        # ## # Sanity Checks # ## #
        # ## ## ## ## ## ## ## ## ## ## ##

        # 1.) Perform a few sanity checks if this is a single-sided
        #     election.
        if not self.proj.is_varnum_pages and \
           self.proj.num_pages == 1 and \
           len(pages_norm_map) != self.proj.num_pages:
            debug("...Uhoh, detected {0} pages, but election \
                  \specifies {1} pages.",
                  len(pages_norm_map),
                  self.proj.num_pages)
            msg = "Warning: The user specified that this is a "\
                  "{0}-sided election. However, OpenCount just "\
                  "detected that {1} sides are present\n".format(
                      self.proj.num_pages,
                      len(pages_norm_map))

            for decoderPage in pages_norm_map.keys():
                msg += "    Page {0}: {1} images\n".format(
                    decoderPage,
                    pages_counter[decoderPage])
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
                for decoderPage in pages_norm_map.keys():
                    if decoderPage != keepDecoderPage:
                        pages_norm_map.pop(decoderPage)
                        pages_counter.pop(decoderPage)
                doQuarantine = dlg.do_quarantine
                handleballot = self.partitionpanel.quarantine_ballot \
                    if doQuarantine else self.partitionpanel.discard_ballot
                for (imgpath, imginfo) in \
                        self.partitionpanel.imginfo.iteritems():
                    decoderPage = imginfo['page']
                    if decoderPage != keepDecoderPage:
                        if doQuarantine:
                            debug("quarantining ballot {0}", img2b[imgpath])
                        else:
                            debug("discarding ballot {0}", img2b[imgpath])
                        handleballot(img2b[imgpath])
        if not self.proj.is_varnum_pages and \
           len(set(pages_norm_map.values())) != self.proj.num_pages:
            warn("Warning: User specified this is an election with "
                 "{0} pages, yet partitioning only discovered {1} pages",
                 self.proj.num_pages,
                 len(set(pages_norm_map.values())))
            ffwx.warn(self,
                      Strings.WRONG_NUM_PAGES.format(
                          self.proj.num_pages,
                          len(set(pages_norm_map.values()))))
            return

        # END Sanity Checks

        # 1.) Build up partitions_map, partitions_invmap
        # Note: self.partitionpanel.partitioning may have partitions
        # with either no ballotids, or ballotids that are all quarantined/discarded.
        # Take care to detect these cases.
        ballots_unevenpages = []
        num_tasks = len(img2b)
        JOBID_EXPORT_RESULTS.next_job(num_tasks)

        for (partitionID, ballotIDs) in self.partitionpanel.partitioning.iteritems():
            if not ballotIDs:
                continue
            atLeastOne = False
            for ballotID in ballotIDs:
                if (ballotID in self.partitionpanel.quarantined_bals or
                        ballotID in self.partitionpanel.discarded_bals):
                    continue
                imgpaths = b2imgs[ballotID]
                pages_set = set()
                _img2page = {}
                for imgpath in imgpaths:
                    atLeastOne = True
                    decoderPage = self.partitionpanel.imginfo[imgpath]['page']
                    page_normed = pages_norm_map[decoderPage]
                    pages_set.add(page_normed)
                    _img2page[imgpath] = page_normed
                    JOBID_EXPORT_RESULTS.tick()
                if not self.proj.is_varnum_pages and len(pages_set) != self.proj.num_pages:
                    # We have a ballot with a weird number of sides
                    ballots_unevenpages.append((ballotID, _img2page))
                else:
                    for imgpath, page in _img2page.iteritems():
                        image_to_page[imgpath] = page
                        image_to_flip[
                            imgpath] = self.partitionpanel.flipmap[imgpath]
                    partitions_map.setdefault(curPartID, []).append(ballotID)
                    partitions_invmap[ballotID] = curPartID
            if atLeastOne:
                curPartID += 1
        # 2.) Grab NUM_EXMPLS number of exemplars from each partition
        num_tasks = len(partitions_map)
        JOBID_EXPORT_RESULTS.next_job(num_tasks)
        for partitionID, ballotIDs in partitions_map.iteritems():
            exmpls = set()
            for ballotID in ballotIDs:
                if len(exmpls) <= self.NUM_EXMPLS:
                    exmpls.add(ballotID)
            if exmpls:
                partition_exmpls[partitionID] = sorted(list(exmpls))
            JOBID_EXPORT_RESULTS.tick()

        if not self.proj.is_varnum_pages and ballots_unevenpages:
            _msg = 'Page Counts:\n'
            for decoderPage in sorted(pages_counter.keys()):
                _msg += '    Page {0}: {1}\n'.format(
                    decoderPage, pages_counter[decoderPage])
            warn("...Warning: Uneven page numbers detected for {0} ballots.",
                 len(ballots_unevenpages))
            warn(_msg)
            warn("Relevant Ballots:")
            _errf = open(pathjoin(self.proj.projdir_path,
                                  "_ballots_unevenpages.txt"), 'w')
            for i, (ballotid, _img2page) in enumerate(ballots_unevenpages):
                print "{0}: BallotID={1}".format(i, ballotid)
                print >>_errf, "{0}: BallotID={1}".format(i, ballotid)
                for _imP, pg in _img2page.iteritems():
                    print "    {0} -> Page {1}".format(_imP, pg)
                    print >>_errf, "    {0} -> Page {1}".format(_imP, pg)
            _errf.close()
            ffwx.warn(self, message=Strings.UNEVEN_SIDES.format(
                self.proj.num_pages,
                len(ballots_unevenpages),
                _msg))
        for (ballotid, _img2page) in ballots_unevenpages:
            self.partitionpanel.quarantined_bals.add(ballotid)

        # Finally, also output the quarantined/discarded ballots
        JOBID_EXPORT_RESULTS.next_job(8)

        self.proj.save_field(
            tuple(self.partitionpanel.quarantined_bals),
            self.proj.partition_quarantined)
        JOBID_EXPORT_RESULTS.tick()
        self.proj.save_field(
            tuple(self.partitionpanel.discarded_bals),
            self.proj.partition_discarded)
        JOBID_EXPORT_RESULTS.tick()
        self.proj.save_field(
            partitions_map, self.proj.partitions_map)
        JOBID_EXPORT_RESULTS.tick()
        self.proj.save_field(
            partitions_invmap, self.proj.partitions_invmap)
        JOBID_EXPORT_RESULTS.tick()
        self.proj.save_field(
            self.partitionpanel.img2decoding,
            self.proj.img2decoding)
        JOBID_EXPORT_RESULTS.tick()
        self.proj.save_field(
            self.partitionpanel.imginfo,
            self.proj.imginfo_map)
        JOBID_EXPORT_RESULTS.tick()
        self.proj.save_field(
            image_to_page, self.proj.image_to_page)
        JOBID_EXPORT_RESULTS.tick()
        self.proj.save_field(
            image_to_flip, self.proj.image_to_flip)
        JOBID_EXPORT_RESULTS.tick()
        self.proj.save_field(partition_exmpls,
                             self.proj.partition_exmpls)

        JOBID_EXPORT_RESULTS.done()
        thread_listen._stop.set()

        wx.CallAfter(self.partitionpanel.on_export_done)


class PartitionPanel(ScrolledPanel):
    PARTITION_JOBID = util.GaugeID("PartitionJobId")
    JOBID_EXTRACT_BARCODE_MARKS = util.GaugeID("ExtractBarcodeMarks")

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
        # statistics panel:
        self.num_partitions = ffwx.StatLabel(self, 'Number of Partitions')
        self.part_largest = ffwx.StatLabel(self, 'Largest Partition Size')
        self.part_smallest = ffwx.StatLabel(self, 'Smallest Partition Size')
        self.sizer_stats = ffwx.vbox(
            self.num_partitions,
            self.part_largest,
            self.part_smallest,
        )
        self.sizer_stats.ShowItems(False)

        # run button:
        self.btn_run = ffwx.Button(
            self,
            label='Run Partitioning',
            on_click=self.onButton_run
        )
        btn_sizer = ffwx.hbox(self.btn_run)

        # 'skip verification' checkbox:
        txt_skiphelp = ffwx.static_wrap(
            self,
            Strings.BARCODE_OVERLAY_HELP,
            100
        )
        self.chkbox_skip_verify = ffwx.CheckBox(
            self,
            label='Skip Overlay Verification (Recommended)',
            default=True)
        sizer_skipVerify = ffwx.vbox(
            txt_skiphelp,
            self.chkbox_skip_verify,
        )

        # dev button options
        btn_loadPartialDecoding = ffwx.Button(
            self,
            label="(Dev) Apply previous decoding results, decode remaining images. [only valid with Skip Overlay Verify]",
            on_click=self.onButton_loadPartialDecoding,
        )
        btn_loadDecoding = ffwx.Button(
            self,
            label="(Dev) Load complete previous decoding results.",
            on_click=self.onButton_loadDecoding,
        )

        if not config.IS_DEV:
            btn_loadPartialDecoding.Hide()
            btn_loadDecoding.Hide()
        sizer_devbuttons = ffwx.hbox(
            btn_loadPartialDecoding,
            (10, 0),
            btn_loadDecoding,
        )

        self.sizer = ffwx.vbox(
            btn_sizer,
            (50, 50),
            sizer_skipVerify,
            (50, 50),
            self.sizer_stats,
            sizer_devbuttons
        )
        self.SetSizer(self.sizer)
        self.Layout()
        self.SetupScrolling()

    def onButton_loadPartialDecoding(self, evt):
        """ Use a previous img2decoding data struct as a 'cache' when
        performing decoding.
        """
        dlg = wx.FileDialog(
            self, "Choose the _state_partition.p file.", ".", "", "*.*", wx.OPEN)
        if dlg.ShowModal() == wx.ID_CANCEL:
            return
        fpath1 = pathjoin(dlg.GetDirectory(), dlg.GetFilename())
        dlg = wx.FileDialog(self, "Choose the image_to_flip.p file.",
                            dlg.GetDirectory(), "", "*.*", wx.OPEN)
        if dlg.ShowModal() == wx.ID_CANCEL:
            return
        fpath2 = pathjoin(dlg.GetDirectory(), dlg.GetFilename())

        try:
            d = pickle.load(open(fpath1, 'rb'))
        except IOError as e:
            error("(Error) Couldn't open: {0}", fpath1)
            error("    Exception {0}", e.message)
            return
        img2decoding = d['img2decoding']
        try:
            img2flip = pickle.load(open(fpath2, 'rb'))
        except IOError as e:
            error("(Error) Couldn't open: {0}", fpath2)
            error("    Exception: {0}", e.message)

        cache = {}  # maps {str imgpath: (tuple decodings, bool isflipped)}
        cnt = 0
        for imgpath, decoding in img2decoding.iteritems():
            isflip = img2flip.get(imgpath, None)
            if isflip is not None:
                cache[imgpath] = (decoding, isflip)
                cnt += 1
        debug("(Info) Loaded {0} previous decodings into cache.", cnt)
        self.run_decoding(cache=cache)

    def onButton_loadDecoding(self, evt):
        """ A Dev feature to allow someone to load-in the previous results of
        decoding the images in this election, without having to re-decode
        the election.
        """
        dlg = wx.FileDialog(
            self, "Choose the _decoder_output.p file.", ".", "", "*.*", wx.OPEN)
        if dlg.ShowModal() == wx.ID_CANCEL:
            return
        filepath = pathjoin(dlg.GetDirectory(), dlg.GetFilename())
        try:
            # dict DECODER_OUT, keys: 'img2decoding', 'flipmap',
            # 'verifypatch_bbs', 'err_imgpaths', 'ioerr_imgpaths'
            d = pickle.load(open(filepath, 'rb'))
        except IOError as e:
            error("(Error) Couldn't open: {0}", filepath)
            error("    Exception: {0}", e.message)
            return
        img2decoding, flipmap = d['img2decoding'], d['flipmap']
        verifypatch_bbs = d['verifypatch_bbs']
        err_imPs, ioerr_imPs = d['err_imgpaths'], d['ioerr_imgpaths']
        self.on_decodedone(img2decoding, flipmap,
                           verifypatch_bbs, err_imPs, ioerr_imPs)

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
            voteddir = state['voteddir']
            partitioning = state['partitioning']
            img2decoding = state['img2decoding']
            imginfo = state['imginfo']
            quarantined_bals = state['quarantined_bals']
            discarded_bals = state['discarded_bals']
            self.voteddir = voteddir
            self.partitioning = partitioning
            self.img2decoding = img2decoding
            self.imginfo = imginfo
            self.quarantined_bals = quarantined_bals
            self.discarded_bals = discarded_bals
            if self.partitioning is not None:
                self.display_partition_stats()
        except:
            return False
        return True

    def save_session(self):
        debug("...PartitionPanel: Saving state...")
        state = {'voteddir': self.voteddir,
                 'partitioning': self.partitioning,
                 'img2decoding': self.img2decoding,
                 'imginfo': self.imginfo,
                 'quarantined_bals': self.quarantined_bals,
                 'discarded_bals': self.discarded_bals}
        pickle.dump(state, open(self.stateP, 'wb'))

    def onButton_run(self, evt):
        self.run_decoding()

    def run_decoding(self, cache=None):
        """ Runs decoding. If input dict CACHE is given, then decoding
        will first check to see if an IMGPATH has a decoding in CACHE
        before decoding.
        Input:
            dict CACHE: maps {str imgpath: (tuple decodings, bool isflipped)}
        """

        class ListenThread(threading.Thread):

            def __init__(self, queue, jobid, *args, **kwargs):
                threading.Thread.__init__(self, *args, **kwargs)
                self.queue = queue
                self.jobid = jobid
                self._stop = threading.Event()

            def stop(self):
                self._stop.set()

            def is_stopped(self):
                return self._stop.isSet()

            def run(self):
                while True:
                    if self.is_stopped():
                        return
                    try:
                        val = self.queue.get(block=True, timeout=1)
                        if val == True:
                            self.jobid.tick()
                    except Queue.Empty:
                        pass

        # First, remove all prior output files (if exists)
        for path in (pathjoin(self.proj.projdir_path,
                              self.proj.partition_quarantined),
                     pathjoin(self.proj.projdir_path,
                              self.proj.partitions_invmap),
                     pathjoin(self.proj.projdir_path, self.proj.img2decoding),
                     pathjoin(self.proj.projdir_path, self.proj.imginfo_map),
                     pathjoin(self.proj.projdir_path,
                              self.proj.partition_exmpls),
                     pathjoin(self.proj.projdir_path,
                              self.proj.partition_quarantined),
                     pathjoin(self.proj.projdir_path,
                              self.proj.partition_discarded),
                     pathjoin(self.proj.projdir_path, self.proj.image_to_page),
                     pathjoin(self.proj.projdir_path, self.proj.image_to_flip),
                     pathjoin(self.proj.projdir_path,
                              '_state_ballot_attributes.p'),
                     pathjoin(self.proj.projdir_path,
                              '_state_selecttargetsMain.p'),
                     pathjoin(self.proj.projdir_path,
                              '_state_selecttargets.p'),
                     pathjoin(self.proj.projdir_path,
                              self.proj.group_to_ballots),
                     pathjoin(self.proj.projdir_path,
                              self.proj.ballot_to_group),
                     pathjoin(self.proj.projdir_path, 'groupsAlign_seltargs')):
            try:
                if os.path.isfile(path):
                    os.remove(path)
                else:
                    shutil.rmtree(path)
            except:
                pass
        self.partitioning = None
        self.img2decoding = None
        self.imginfo = None
        self.quarantined_bals = set()
        self.discarded_bals = set()

        vendor_obj = self.proj.vendor_obj
        b2imgs = self.proj.load_field(self.proj.ballot_to_images)

        skipVerify = self.chkbox_skip_verify.GetValue()

        progress_bar = ffwx.ProgressBar(self)

        @util.as_process
        def part_thread():
            with progress_bar:
                return vendor_obj.decode_ballots(b2imgs,
                                                 skipVerify=skipVerify,
                                                 cache=cache)
        progress_bar.on_finish(lambda: self.on_decodedone(*part_thread()))

    def on_decodedone(self, img2decoding, flipmap, verifypatch_bbs, err_imgpaths, ioerr_imgpaths):
        """
        Input:
            dict IMG2DECODING: {str imgpath: (str decoding_0, ..., str decoding_N)}
            dict FLIPMAP: {imgpath: bool isFlipped}
            dict VERIFYPATCH_BBS: {str bc_val: [(imgpath, (x1,y1,x2,y2), userdata), ...]}
                May be None if skipVerify was True.
            list ERR_IMGPATHS:
                List of images that the decoder was unable to decode
                with high certainty.
            list IOERR_IMGPATHS:
                List of images that were unable to be read by OpenCount
                due to a read/load error (e.g. IOError).
        """
        debug("Decoding Done!")
        debug('Errors ({0} total): {1}', len(err_imgpaths), err_imgpaths)
        debug('IOErrors ({0} total): {1}', len(ioerr_imgpaths), ioerr_imgpaths)
        # Save the raw decoding output, in case for later usage
        self.proj.save_field({'img2decoding': img2decoding,
                              'flipmap': flipmap,
                              'verifypatch_bbs': verifypatch_bbs,
                              'err_imgpaths': err_imgpaths,
                              'ioerr_imgpaths': ioerr_imgpaths},
                             '_decoder_output.p')

        img2bal = self.proj.load_field(self.proj.image_to_ballot)
        bal2imgs = self.proj.load_field(self.proj.ballot_to_images)

        if err_imgpaths:
            dlg = LabelDialog(self, err_imgpaths)
            status = dlg.ShowModal()
            # dict ERRS_CORRECTED: {str imgpath: str label or
            # ID_Quarantine/ID_Discard}
            self.errs_corrected = dlg.label_res
            self.errs_flipmap = dlg.imgflips
        else:
            self.errs_corrected = {}
            self.errs_flipmap = {}
        if ioerr_imgpaths:
            self.ioerr_imgpaths = ioerr_imgpaths
            errpath = os.path.join(self.proj.projdir_path,
                                   'ioerr_imgpaths.txt')
            dlg = wx.MessageDialog(self, message="Warning: {0} images \
were unable to be read by OpenCount. These images (and associated \
images from that ballot) will not be processed by further steps of the \
OpenCount pipeline. \n\
The imagepaths will be written to: {1}".format(len(self.ioerr_imgpaths), errpath),
                                   style=wx.ID_OK)
            dlg.ShowModal()
            try:
                with open(errpath, 'w') as errf:
                    errf = open(errpath, 'w')
                    print >>errf, "==== Images that could not be read by OpenCount ===="
                    for ioerr_imgpath in ioerr_imgpaths:
                        print >>errf, ioerr_imgpath
            except IOError as e:
                error(
                    "Warning: Unable to write ioerr_imgpaths to {0}", errpath)
        else:
            self.ioerr_imgpaths = []
        # Output all ioerr ballot ids for future stages to be aware
        ioerr_balids = set()
        for ioerr_imgpath in self.ioerr_imgpaths:
            ioerr_balids.add(img2bal[ioerr_imgpath])
        self.proj.save_field(ioerr_balids,
                             self.proj.partition_ioerr)

        def nuke_ballots(ballotids, img2decoding, verifypatch_bbs, flipmap):
            """ Removes all references to ballotids in BALLOTIDS from
            data structs IMG2DECODING, VERIFYPATCH_BBS and FLIPMAP.
            Mutates input IMG2DECODING, VERIFYPATCH_BBS, FLIPMAP.
            """
            # If VERIFYPATCH_BBS is None, then skipVerify was True
            if verifypatch_bbs:
                for bc_val, tups in verifypatch_bbs.iteritems():
                    i = 0
                    while i < len(tups):
                        tup_imP, tup_bb, tup_userdata = tups[i]
                        tup_balid = img2bal[tup_imP]
                        if tup_balid in ballotids:
                            tups.pop(i)
                        else:
                            i += 1
            for flipmap_imP in flipmap.keys():
                flipmap_bid = img2bal[flipmap_imP]
                if flipmap_bid in ballotids:
                    flipmap.pop(flipmap_imP)
            for bad_ballotid in ballotids:
                imgpaths = bal2imgs[bad_ballotid]
                for imgpath in imgpaths:
                    if imgpath in img2decoding:
                        img2decoding.pop(imgpath)
            return img2decoding, verifypatch_bbs, flipmap

        bal2errlabel = {}  # maps {int ballotid: ID_Quarantine/ID_Discard}
        for imgpath, label in self.errs_corrected.iteritems():
            ballotid = img2bal[imgpath]
            if ballotid not in bal2errlabel:
                bal2errlabel[ballotid] = label
            elif bal2errlabel[ballotid] == LabelDialog.ID_Quarantine and label == LabelDialog.ID_Discard:
                # ID_Discard overrides ID_Quarantine
                bal2errlabel[ballotid] = label
            elif ballotid in ioerr_balids:
                continue
        # Populate the quarantined/discarded data structures
        self.quarantined_bals = set()
        self.discarded_bals = set()
        for balid, errlabel in bal2errlabel.iteritems():
            imgpaths = bal2imgs[balid]
            for imgpath in imgpaths:
                self.errs_corrected[imgpath] = errlabel
                if errlabel == LabelDialog.ID_Quarantine:
                    self.quarantine_ballot(balid)
                elif errlabel == LabelDialog.ID_Discard:
                    self.discard_ballot(balid)

        nuke_ballotids = set(tuple(bal2errlabel.keys()) +
                             tuple([img2bal[imP] for imP in self.ioerr_imgpaths]))
        # TODO: I think this call to nuke_ballots is taking quite a long time.
        #       Either speed this up, or add a progress bar?
        nuke_ballots(nuke_ballotids, img2decoding, verifypatch_bbs, flipmap)

        debug("{0} Quarantined Ballots, {1} Discarded Ballots",
              len(self.quarantined_bals),
              len(self.discarded_bals))
        self.start_verify(img2decoding, flipmap, verifypatch_bbs)

    def start_verify(self, img2decoding, flipmap, verifypatch_bbs):
        """
        Input:
            dict IMG2DECODING: {str imgpath: (str decoding_0, ...)}
            dict FLIPMAP: maps {str imgpath: bool isflip}
            dict VERIFYPATCH_BBS: maps {str bc_val: [(imgpath, (x1,y1,x2,y2), userdata), ...]}
                Will be None if skipVerify was True.
        """
        if self.chkbox_skip_verify.GetValue() or verifypatch_bbs is None:
            debug("...Skipping Barcode Overlay Verification...")
            wx.CallAfter(self.on_verify_done, None, None, img2decoding,
                         flipmap, verifypatch_bbs, skipVerify=True)
            return

        # DOGFOOD: Barcode verification has NOT been updated to keep track
        #          of the new img2decoding data structure. The following
        #          code WILL crash.

        outrootdir = pathjoin(self.proj.projdir_path, '_barcode_extractpats')

        manager = multiprocessing.Manager()
        queue_mygauge = manager.Queue()

        thread_updateMyGauge = ThreadUpdateMyGauge(queue_mygauge,
                                                   self.JOBID_EXTRACT_BARCODE_MARKS)
        thread_updateMyGauge.start()

        thread_doextract = ThreadExtractBarcodePatches(verifypatch_bbs, flipmap, img2decoding,
                                                       outrootdir, self.proj.voteddir,
                                                       manager, queue_mygauge,
                                                       thread_updateMyGauge,
                                                       callback=self.on_extract_done)
        thread_doextract.start()

        gauge = util.MyGauge(self, 1, thread=thread_doextract,
                             msg="Extracting Barcode Values...",
                             job_id=self.JOBID_EXTRACT_BARCODE_MARKS)
        gauge.Show()

        num_tasks = len(flipmap)
        self.JOBID_EXTRACT_BARCODE_MARKS.next_job(num_tasks)

    def on_extract_done(self, img2patch, patch2stuff, verifypatch_bbs, flipmap, img2decoding):
        """ Invoked once all barcode value patches have been extracted.
        Input:
            dict IMG2PATCH: {(imgpath, tag): patchpath}
            dict PATCH2STUFF: {patchpath: (imgpath, (x1,y1,x2,y2), tag)}
            dict VERIFYPATCH_BBS: {str bc_val: [(imgpath, (x1,y1,x2,y2), userdata), ...]}
            dict FLIPMAP: {imgpath: bool isFlipped}
            dict IMG2DECODING: {imgpath: (str decoding_0, ...)}
        """
        self.JOBID_EXTRACT_BARCODE_MARKS.done()
        cattag = 'BarcodeCategory'
        imgcats = {}  # maps {cat_tag: {grouptag: [imgpath_i, ...]}}
        exmplcats = {}  # maps {cat_tag: {grouptag: [imgpath_i, ...]}}
        for bc_val, tups in verifypatch_bbs.iteritems():
            for (imgpath, (x1, y1, x2, y2), userdata) in tups:
                id = (x1, y1, x2, y2)
                patchpath = img2patch[(imgpath, (bc_val, userdata, id))]
                imgcats.setdefault(cattag, {}).setdefault(
                    bc_val, []).append(patchpath)
        callback = lambda verifyRes: self.on_verify_done(
            verifyRes, patch2stuff, img2decoding, flipmap, verifypatch_bbs)
        f = VerifyOverlaysFrame(self, imgcats, exmplcats, callback)
        f.Maximize()
        f.Show()
        # Kludge: Do a 200ms delay, to allow size-related events to
        # complete. If you call f.start() immediately, then f's widgets
        # won't know its 'real' client size, causing layout issues.
        wx.FutureCall(200, f.start)

    def on_verify_done(self, verify_results, patch2stuff, img2decoding, flipmap, verifypatch_bbs, skipVerify=False):
        """ Receives the (corrected) results from VerifyOverlays.
        Input:
        dict VERIFY_RESULTS: {cat_tag: {grouptag: [imgpath_i, ...]}}
            For each category CAT_TAG, each group GROUPTAG maps to a set
            of imgpaths that the user claimed is part of GROUPTAG.
            Will be None if skipVerify was True.
        """
        debug("...barcode patch verification done!")
        # maps {str bc_val: [(imgpath, (x1,y1,x2,y2), userdata), ...]}
        verified_decodes = {}
        if skipVerify:
            verified_decodes = verifypatch_bbs
        else:
            for cat_tag, thedict in verify_results.iteritems():
                for bc_val, patchpaths in thedict.iteritems():
                    for patchpath in patchpaths:
                        imgpath, bb, (bc_val_this, userdata,
                                      id) = patch2stuff[patchpath]
                        verified_decodes.setdefault(
                            bc_val, []).append((imgpath, bb, userdata))
        # Merge in the manually-labeled images
        manual_labeled = {}  # maps {str imgpath: (str label_i, ...)}
        for imgpath, label in self.errs_corrected.iteritems():
            if label not in (LabelDialog.ID_Quarantine, LabelDialog.ID_Discard):
                # The Manual Label widget states that commas separate
                # barcode values.
                decodings = tuple([s.strip() for s in label.split(",")])
                manual_labeled[imgpath] = decodings
        debug("generating partitions")
        # dict PARTITIONING: maps {int partitionID: [int ballotID_i, ...]}
        # TODO: This call to partition_ballots takes a few minutes on large
        # elections -- add a progress bar or something, so that it doesn't
        # look like the UI is hanging.
        partitioning, img2decoding, imginfo_map = self.proj.vendor_obj.partition_ballots(
            img2decoding, verified_decodes, manual_labeled)
        debug("...done generating partitions...")
        # Add in manually-corrected flipped
        for imgpath, isflip in self.errs_flipmap.iteritems():
            flipmap[imgpath] = isflip
        self.partitioning = partitioning
        self.img2decoding = img2decoding
        self.imginfo = imginfo_map
        self.flipmap = flipmap

        # Finally, sanity check that, within each partition, each ballot
        # has the same number of pages.
        bal2imgs = self.proj.load_field(self.proj.ballot_to_images)
        bad_ballotids = []
        for partitionID, ballotids in partitioning.iteritems():
            num_pages = max([len(bal2imgs[b]) for b in ballotids])
            cur_bad_ballotids = [
                b for b in ballotids if len(bal2imgs[b]) != num_pages]
            if cur_bad_ballotids:
                debug('REMOVING {0} ballots from partition {1}...',
                      len(cur_bad_ballotids),
                      partitionID)
                debug("\tShould have {0} Pages", num_pages)
                debug("\t{0}", [len(bal2imgs[b]) for b in cur_bad_ballotids])
            bad_ballotids.extend(cur_bad_ballotids)
            cur_bad_ballotids = set(cur_bad_ballotids)
            ballotids[:] = [b for b in ballotids if b not in cur_bad_ballotids]

        # For each 'bad' ballotid, add them into its own new partition
        debug("...There were {0} ballotids with anomalous page numbers. "
              "Adding to separate partitions...",
              len(bad_ballotids))
        curPartId = len(self.partitioning)
        for badballotid in bad_ballotids:
            self.partitioning[curPartId] = [badballotid]
            curPartId += 1

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

        class ThreadExport(threading.Thread):

            def __init__(self, fn_tocall, thread_listen, queue_mygauge, *args, **kwargs):
                threading.Thread.__init__(self, *args, **kwargs)
                self.fn_tocall = fn_tocall
                self.queue_mygauge = queue_mygauge
                self.thread_listen = thread_listen

            def run(self):
                wx.CallAfter(self.fn_tocall, self.queue_mygauge,
                             self.thread_listen)

        class ThreadListen(threading.Thread):

            def __init__(self, queue_mygauge, jobid, *args, **kwargs):
                threading.Thread.__init__(self, *args, **kwargs)
                self.queue_mygauge = queue_mygauge
                self.jobid = jobid
                self._stop = threading.Event()

            def run(self):
                while not self._stop.is_set():
                    try:
                        val = self.queue_mygauge.get(block=True, timeout=1)
                        if val == True:
                            self.jobid.tick()
                    except Queue.Empty:
                        pass
        manager = multiprocessing.Manager()
        queue_mygauge = manager.Queue()
        t_listen = ThreadListen(queue_mygauge, JOBID_EXPORT_RESULTS)
        t_export = ThreadExport(
            self.GetParent().export_results, t_listen, queue_mygauge)
        mygauge = util.MyGauge(
            self, 4, thread=t_export, msg="Exporting Results...", job_id=JOBID_EXPORT_RESULTS)
        mygauge.Show()

        t_listen.start()
        t_export.start()

    def on_export_done(self):
        self.display_partition_stats()
        self.btn_run.Disable()
        self.chkbox_skip_verify.Disable()
        self.Layout()

        wx.MessageDialog(self, style=wx.OK, caption="Partitioning Done",
                         message="Partitioning is complete. \n\n\
You may move onto the next step.").ShowModal()

    def display_partition_stats(self):
        if not self.partitioning:
            return
        largest, smallest = None, None
        for partitionid, ballotids in self.partitioning.iteritems():
            n = len(ballotids)
            largest = (max(largest, n) if largest is not None else n)
            smallest = (min(smallest, n) if smallest is not None else n)

        num_partitions = len(self.partitioning)
        self.num_partitions.set_value(num_partitions)
        self.part_largest.set_value(largest)
        self.part_smallest.set_value(smallest)
        self.sizer_stats.ShowItems(True)

        self.Layout()

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
        wx.Frame.__init__(self, parent, *args, **kwargs)

        self.imgcategories = imgcategories
        self.exmplcategories = exmplcategories
        self.ondone = ondone

        self.verifyoverlays = verify_overlays_new.VerifyOverlaysMultCats(self)

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.verifyoverlays, proportion=1, flag=wx.EXPAND)
        self.SetSizer(self.sizer)
        self.Layout()

    def start(self):
        self.verifyoverlays.start(self.imgcategories, self.exmplcategories,
                                  do_align=False, ondone=self.on_verify_done)
        self.Layout()

    def on_verify_done(self, verify_results):
        self.Close()
        wx.CallAfter(self.ondone, verify_results)


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

        inst = (textwrap.fill("""Please enter the barcode(s) for each \
ballot, or mark as quarantined/discarded.""") + "\n\n" +
                textwrap.fill("""(Sequoia) -- Please enter the two \
8-bit barcode-values as comma-separated bitstrings, in Left to Right order. \
Within each barcode, interpret the bits in Top to Down order. E.g: 01000000, 00001111.""") + "\n" +
                textwrap.fill("""    Note: If the image is the backside \
of a ballot, then enter the special label: '0,' (no quotes).""") + "\n\n" +
                textwrap.fill("""(Hart) -- Please enter the decimal digits \
displayed below the upper-left barcode. Enter the digits starting from the \
bottom-most digit.""") + "\n\n" +
                textwrap.fill("""(Diebold) -- Please enter the 32-bit \
bitstring along the bottom of the ballot page, from Left to Right order. \
Do not include the left-most and right-most marks.""") + "\n\n")
        self.txt_inst.SetLabel(inst)

        self.radio_quarantine = wx.RadioButton(self, label="Quarantine (Process Later)",
                                               style=wx.RB_GROUP)
        self.radio_quarantine.Bind(wx.EVT_RADIOBUTTON, self.onRadioBtn_quar)
        self.radio_discard = wx.RadioButton(
            self, label="Discard (Don't Process)")
        self.radio_discard.Bind(wx.EVT_RADIOBUTTON, self.onRadioBtn_discard)
        self.radio_normal = wx.RadioButton(
            self, label="Normal Ballot (Process Normally)")
        self.radio_normal.Bind(wx.EVT_RADIOBUTTON, self.onRadioBtn_normal)
        radiobtn_sizer = wx.BoxSizer(wx.VERTICAL)
        radiobtn_sizer.AddMany([(self.radio_quarantine,), (self.radio_discard,),
                                (self.radio_normal,)])
        self.chkbox_isflip = wx.CheckBox(
            self, label="Is the ballot flipped (upside down)?")

        btn_quarantine_all = ffwx.Button(
            self,
            label='Quarantine REST of images',
            on_click=self.onButton_quarAll,
        )
        btn_discard_all = ffwx.Button(
            self,
            label="Discard REST of images",
            on_click=self.onButton_discardAll
        )
        self.btn_sizer.AddMany([(radiobtn_sizer,), (self.chkbox_isflip,),
                                ((0, 50),),
                                (btn_quarantine_all,), ((0, 10),),
                                (btn_discard_all,)])

    def onButton_quarAll(self, evt):
        total = len(self.imagepaths)
        num_rest = total - self.cur_imgidx
        status = wx.MessageDialog(self, style=wx.YES_NO | wx.CAPTION, caption="Are you sure?",
                                  message="Are you sure you want to Quarantine \
all {0} images starting from here?\n\n\
When a ballot B is quarantined, you will have to manually enter in all ballot \
information, such as voter marks and contest information.".format(num_rest)).ShowModal()
        if status != wx.ID_YES:
            return
        for imgpath in self.imagepaths[self.cur_imgidx:]:
            try:
                self.discard_imgpaths.remove(imgpath)
            except:
                pass
            self.quar_imgpaths.add(imgpath)
        self.callback(self.imagelabels)

    def onButton_discardAll(self, evt):
        total = len(self.imagepaths)
        num_rest = total - self.cur_imgidx
        status = wx.MessageDialog(self, style=wx.YES_NO | wx.CAPTION, caption="Are you sure?",
                                  message="Are you sure you want to Discard \
all {0} images starting from here?\n\n\
When a ballot B is Discarded, then B is omitted from ALL subsequent OpenCount \
steps, and will not be included in the final results.".format(num_rest)).ShowModal()
        if status != wx.ID_YES:
            return
        for imgpath in self.imagepaths[self.cur_imgidx:]:
            try:
                self.quar_imgpaths.remove(imgpath)
            except:
                pass
            self.discard_imgpaths.add(imgpath)
        self.callback(self.imagelabels)

    def onRadioBtn_quar(self, evt):
        evt.Skip()
        curimgpath = self.imagepaths[self.cur_imgidx]
        self.quar_imgpaths.add(curimgpath)
        try:
            self.discard_imgpaths.remove(curimgpath)
        except:
            pass
        self.onButton_next(None)

    def onRadioBtn_discard(self, evt):
        evt.Skip()
        curimgpath = self.imagepaths[self.cur_imgidx]
        self.discard_imgpaths.add(curimgpath)
        try:
            self.quar_imgpaths.remove(curimgpath)
        except:
            pass
        self.onButton_next(None)

    def onRadioBtn_normal(self, evt):
        evt.Skip()
        curimgpath = self.imagepaths[self.cur_imgidx]
        try:
            self.discard_imgpaths.remove(curimgpath)
        except:
            pass
        try:
            self.quar_imgpaths.remove(curimgpath)
        except:
            pass

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
                           size=(1000, 800), style=wx.CAPTION | wx.MAXIMIZE_BOX | wx.MINIMIZE_BOX | wx.RESIZE_BORDER,
                           *args, **kwargs)

        self.labelpanel = LabelOrDiscardPanel(self)

        self.sizer = wx.BoxSizer(wx.VERTICAL)

        self.sizer.Add(self.labelpanel, proportion=1,
                       border=10, flag=wx.EXPAND | wx.ALL)
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
        wx.Dialog.__init__(self, parent, title="User action required.", size=(
            700, 200), *args, **kwargs)

        self.keep_page = None
        self.do_quarantine = None

        txt = wx.StaticText(self, label=msg)

        btn_treatnormal = ffwx.Button(
            self,
            label='Process All',
            on_click=self.onButton_treatNormal,
        )

        _msg_treatNormal = "    In other words, treat each image as if \
it were the front-side of a ballot."
        _msg_treatNormal = textwrap.fill(_msg_treatNormal, width=300)
        txt_treatNormal = wx.StaticText(self, label=_msg_treatNormal)

        txt_choose = wx.StaticText(self, label="Or, process only one side:")
        choices = []
        for page in sorted(page_counter):
            cnt = page_counter[page]
            choices.append("Page {0} -- {1} images".format(page, cnt))
        self.cb_pages = wx.ComboBox(
            self, choices=choices, style=wx.CB_READONLY)

        sizer_choose = ffwx.vbox(txt_choose, self.cb_pages)

        txt_others = wx.StaticText(
            self, label="And do the following to the other sides:")
        self.rb_quarantine = wx.RadioButton(
            self, label="Quarantine the other sides", style=wx.RB_GROUP)
        self.rb_discard = wx.RadioButton(self, label="Discard the other sides")

        sizer_others = ffwx.vbox(
            txt_others, self.rb_quarantine, self.rb_discard)

        sizer2 = ffwx.hbox(sizer_choose, sizer_others)
        btn_ok = ffwx.Button(self, label='Ok', on_click=self.onButton_ok)
        sizer2.Add(btn_ok, flag=wx.ALIGN_CENTER)

        sizer_treatNormal = ffwx.vbox(btn_treatNormal, txt_treatNormal)
        btn_sizer = ffwx.hbox(sizer_treatNormal, (50, 0), sizer2)
        sizer = ffwx.vbox(txt, btn_sizer)

        self.SetSizer(sizer)
        self.Fit()

    def onButton_treatNormal(self, evt):
        self.EndModal(self.ID_TREATNORMAL)

    def onButton_ok(self, evt):
        self.keep_page = self.cb_pages.GetSelection()
        self.do_quarantine = True if self.rb_quarantine.GetValue() else False
        self.EndModal(self.ID_KEEPONE)


class ThreadExtractBarcodePatches(threading.Thread):
    """ A separate thread to run the extract_barcode_patches call. """

    def __init__(self, verifypatch_bbs, flipmap, img2decoding,
                 outrootdir, voteddir,
                 manager, queue_mygauge,
                 thread_updateMyGauge,
                 callback=None,
                 *args, **kwargs):
        threading.Thread.__init__(self, *args, **kwargs)
        self.verifypatch_bbs = verifypatch_bbs
        self.flipmap = flipmap
        self.img2decoding = img2decoding
        self.outrootdir = outrootdir
        self.voteddir = voteddir
        self.manager = manager
        self.queue_mygauge = queue_mygauge
        self.thread_updateMyGauge = thread_updateMyGauge
        self.callback = callback

    def run(self):
        with util.time_operation('extract_barcode_patches'):
            img2patch, patch2stuff = extract_barcode_patches(
                self.verifypatch_bbs, self.flipmap,
                self.outrootdir, self.voteddir,
                manager=self.manager,
                queue_mygauge=self.queue_mygauge)
        self.thread_updateMyGauge.stop_running()
        wx.CallAfter(self.callback, img2patch, patch2stuff,
                     self.verifypatch_bbs, self.flipmap, self.img2decoding)


class ThreadUpdateMyGauge(threading.Thread):
    """ A Thread that listens on self.queue_mygauge, and throws out
    'tick' messages to a MyGauge instance, based on self.jobid.
    """

    def __init__(self, queue_mygauge, jobid, *args, **kwargs):
        threading.Thread.__init__(self, *args, **kwargs)
        self.queue_mygauge = queue_mygauge
        self.jobid = jobid

        self._stop = threading.Event()

    def stop_running(self):
        self._stop.set()

    def i_am_running(self):
        return not self._stop.isSet()

    def run(self):
        while self.i_am_running():
            try:
                val = self.queue_mygauge.get(block=True, timeout=1)
                self.jobid.tick()
            except Queue.Empty as e:
                pass


def extract_barcode_patches(verifypatch_bbs, flipmap, outrootdir, voteddir,
                            manager=None, queue_mygauge=None):
    """ Given the results of the barcode decoder, extract each barcode
    value (given by the bounding boxes) and save them to OUTROOTDIR.
    Input:
        dict VERIFYPATCH_BBS: maps {str bc_val: [(imgpath, (x1,y1,x2,y2), userdata), ...]}
        dict FLIPMAP: maps {str imgpath: bool isflip}
        str OUTROOTDIR:
            Root directory to store extracted images
        str VOTEDDIR:
            Root directory of voted ballots directory. This is used to
            recreate the directory structure for OUTROOTDIR.
    Output:
        (dict IMG2PATCH, dict PATCH2STUFF)
    dict IMG2PATCH: maps {(imgpath, tag): patchpath}
    dict PATCH2STUFF: maps {patchpath: (imgpath, (x1,y1,x2,y2), tag)}
    """
    # 1.) Extract all patches to an outdir
    imgpatches = {}  # {imgpath: [((x1,y1,x2,y2), isflip, outpath, tag), ...]}

    bc_val_cnt = {}  # maps {bc_val: int cnt}
    bc_val_dircnt = {}  # maps {bc_val: int dircnt}
    img_ctr = util.Counter()
    debug("...creating jobs for barcode-patch extraction...")
    for bc_val, tups in verifypatch_bbs.iteritems():
        for (imgpath, (x1, y1, x2, y2), userdata) in tups:
            i = bc_val_cnt.get(bc_val, None)
            if i is None:
                bc_val_cnt[bc_val] = 0
                bc_val_dircnt[bc_val] = 0
                i = 0
            if i != 0 and i % 750 == 0:
                bc_val_dircnt[bc_val] += 1
            dircnt = bc_val_dircnt[bc_val]
            imgname = os.path.splitext(os.path.split(imgpath)[1])[0]
            # Recreate directory structure
            rp = os.path.splitext(os.path.relpath(
                os.path.abspath(imgpath), os.path.abspath(voteddir)))[0]
            outpath = pathjoin(outrootdir, rp, "{0}_{1}.png".format(
                imgname, img_ctr[imgpath]))
            img_ctr[imgpath] += 1
            # use the loc as the ID, in case USERDATA isn't used
            ID = (x1, y1, x2, y2)
            tag = (bc_val, userdata, ID)
            isflip = flipmap[imgpath]
            imgpatches.setdefault(imgpath, []).append(
                ((x1, y1, x2, y2), isflip, outpath, tag))
            i += 1
    with util.time_operation('...extracting...'):
        img2patch, patch2stuff = extract_patches.extract(
            imgpatches,
            manager=manager,
            queue_mygauge=queue_mygauge)

    return img2patch, patch2stuff
