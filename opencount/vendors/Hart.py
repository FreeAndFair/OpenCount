import os, sys, traceback
try:
    import cPickle as pickle
except ImportError:
    import pickle
from os.path import join as pathjoin

import cv

from Vendor import Vendor

sys.path.append('..')

import barcode.hart as hart
from grouping import partask

# Get this script's directory. Necessary to know this information
# since the current working directory may not be the same as where
# this script lives (which is important for loading resources like
# imgs)
try:
    MYDIR = os.path.abspath(os.path.dirname(__file__))
except NameError:
    # This script is being run directly
    MYDIR = os.path.abspath(sys.path[0])

TOP_GUARD_IMGP = pathjoin(MYDIR, 'hart_topguard.png')
BOT_GUARD_IMGP = pathjoin(MYDIR, 'hart_botguard.png')

TOP_GUARD_SKINNY_IMGP = pathjoin(MYDIR, 'hart_topguard_skinny.png')
BOT_GUARD_SKINNY_IMGP = pathjoin(MYDIR, 'hart_botguard_skinny.png')

class HartVendor(Vendor):
    def __init__(self, proj):
        self.proj = proj
    
    def decode_ballots(self, ballots, manager=None, queue=None):
        topbot_paths = [[TOP_GUARD_IMGP, BOT_GUARD_IMGP], [TOP_GUARD_SKINNY_IMGP, BOT_GUARD_SKINNY_IMGP]]
        # DECODED_RESULTS_MAP: {ballotID: [(BCS, isflip, BBS, bbstripes_dict), ...]}
        decoded_results_map = decode_ballots(ballots, topbot_paths, manager, queue)
        flipmap = {} # maps {imgpath: bool isFlipped}
        bbstripes_map = {} # maps {'wideNarrow': [(str imgpath, (x1,y1,x2,y2), int id), ...], ...}
        err_imgpaths = []
        ioerr_imgpaths = []
        for ballotid, decoded_results in decoded_results_map.iteritems():
            imgpaths = ballots[ballotid]
            for i, subtuple in enumerate(decoded_results):
                if issubclass(type(subtuple), IOError):
                    ioerr_imgpaths.append(subtuple.filename)
                    continue
                bcs, isflipped, bbs, bbstripes_dict = subtuple
                imgpath = imgpaths[i]
                flipmap[imgpath] = isflipped
                bc_ul = bcs[0]
                if not bc_ul:
                    print "..error on: {0}".format(imgpath)
                    err_imgpaths.append(imgpath)
                else:
                    # First, maintain a global ordering of each stripe, for bbstripe_idx
                    stripe_y1s = []
                    for label, bbstripes in bbstripes_dict.iteritems():
                        for (x1,y1,x2,y2) in bbstripes:
                            stripe_y1s.append(y1)
                    # sort by Y1, bottom->top
                    stripe_y1s = sorted(stripe_y1s, key=lambda t: -t)
                    for label, bbstripes in bbstripes_dict.iteritems():
                        for (x1,y1,x2,y2) in bbstripes:
                            bbstripe_idx = stripe_y1s.index(y1)
                            bbstripes_map.setdefault(label, []).append((imgpath, (x1,y1,x2,y2), bbstripe_idx))
        return flipmap, bbstripes_map, err_imgpaths, ioerr_imgpaths

    def partition_ballots(self, verified_results, manual_labeled):
        """
        Input:
            dict VERIFIED_RESULTS:
            dict MANUAL_LABELED: {str imgpath: (str bc,)}
        Output:
            (dict PARTITIONS, dict IMG2DECODING, dict IMGINFO_MAP)
        """
        partitions = {} # maps {partitionID: [int ballotID, ...]}
        img2decoding = {} # maps {imgpath: [str bc_i, ...]}
        imginfo_map = {} # maps {imgpath: {str PROPNAME: str PROPVAL}}
        img_bc_temp = {} # maps {imgpath: [(i, bcLabel_i), ...]}
        for bc_val, tups in verified_results.iteritems():
            for (imgpath, (x1,y1,x2,y2), bbstripe_i) in tups:
                img_bc_temp.setdefault(imgpath, []).append((bbstripe_i, bc_val))
        # IMG_DECODED_MAP: {str imgpath: str decoding}
        img_decoded_map = hart.interpret_labels(img_bc_temp)
        img2bal = pickle.load(open(self.proj.image_to_ballot, 'rb'))
        attrs2partitionID = {} # maps {('precinct', 'language', 'party'): int partitionID}

        def add_decoding(imgpath, decoding, curPartitionID):
            """ Returns True if a new partition is created. """
            created_new_partition = False
            img2decoding[imgpath] = (decoding,)
            imginfo = hart.get_info((decoding,))
            imginfo_map[imgpath] = imginfo
            tag = (imginfo['precinct'], imginfo['language'], imginfo['party'])
            if self.proj.num_pages == 1:
                # Additionally separate by page for single-sided, just in
                # case multiple pages are present in the ballot scans
                tag += (imginfo['page'],)
            partitionid = attrs2partitionID.get(tag, None)
            if partitionid == None:
                partitionid = curPartitionID
                attrs2partitionID[tag] = curPartitionID
                created_new_partition = True
            ballotid = img2bal[imgpath]
            partitions.setdefault(partitionid, set()).add(ballotid)
            return created_new_partition

        curPartitionID = 0
        for imgpath, decoding in img_decoded_map.iteritems():
            added_new_partition = add_decoding(imgpath, decoding, curPartitionID)
            if added_new_partition:
                curPartitionID += 1
        for imgpath, decoding_tuple in manual_labeled.iteritems():
            added_new_partition = add_decoding(imgpath, decoding_tuple[0], curPartitionID)
            if added_new_partition:
                curPartitionID += 1

        for partitionid, ballotid_set in partitions.iteritems():
            partitions[partitionid] = sorted(list(ballotid_set))
        return partitions, img2decoding, imginfo_map

    def __repr__(self):
        return 'HartVendor()'
    def __str__(self):
        return 'HartVendor()'

def _do_decode_ballots(ballots, (topbot_paths,), queue=None):
    cvread = lambda imP: cv.LoadImage(imP, cv.CV_LOAD_IMAGE_GRAYSCALE)
    topbot_pairs = [[cvread(topbot_paths[0][0]), cvread(topbot_paths[0][1])],
                    [cvread(topbot_paths[1][0]), cvread(topbot_paths[1][1])]]
    results = {} # maps {int ballotid: [(bcs_side0, isflipped_side0, bbs_side0, bbstripes_side0), ...]}
    for ballotid, imgpaths in ballots.iteritems():
        balresults = []
        for imgpath in imgpaths:
            try:
                bcs, isflipped, bbs, bbstripes_dict = hart.decode(imgpath, topbot_pairs)
                balresults.append((bcs, isflipped, bbs, bbstripes_dict))                
            except IOError as e:
                balresults.append(e)
        results[ballotid] = balresults
        if queue:
            queue.put(True)
    return results

def decode_ballots(ballots, topbot_paths, manager, queue):
    decoded_results = partask.do_partask(_do_decode_ballots,
                                         ballots,
                                         _args=(topbot_paths,),
                                         combfn='dict',
                                         manager=manager,
                                         pass_queue=queue,
                                         N=None)
    print '...finished decoding...'
    return decoded_results
