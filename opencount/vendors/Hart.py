import os, sys, traceback, pdb
from os.path import join as pathjoin

import cv
import wx
from wx.lib.pubsub import Publisher

from Vendor import Vendor

sys.path.append('..')

from barcode.hart import decode, get_info
from grouping import partask
import util

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
    def __init__(self):
        pass
    
    def decode_image(self, imgpath, topbot_pairs):
        """ 
        Input:
            str IMGPATH:
        Output:
            list OUTPUT. Stores a list of all decoded barcodes,
                a boolean saying whether or not the image is flipped, and
                the location of each barcode:
                    [[str bc_i, ...], bool is_flip, [(x1,y1,x2,y2), ...]].
        """
        decodes, isflipped, bbs, bbstripes_dict = decode(imgpath, topbot_pairs)
        return decodes, isflipped, bbs, bbstripes_dict

    def decode_ballot(self, ballot, topbot_guards):
        """
        Input:
            list BALLOT: List of image paths, together which correspond
                to one ballot. For instance, if the election is double-sided,
                then BALLOT should be a list of length two.
        Output:
            list RESULTS. RESULTS is a list of lists, where each sublist
                contains information on each image:
                    [[(str bc_i, ...), bool isflipped, [(x1,y1,x2,y2), ...]], ...]
        """
        results = []
        for imgpath in ballot:
            (bcs, isflipped, bbs, bbstripes_dict) = self.decode_image(imgpath, topbot_guards)
            results.append((bcs, isflipped, bbs, bbstripes_dict))
        return results

    def partition_ballots(self, ballots, manager=None, queue=None):
        partitions = {}
        decoded = {}
        imginfo = {}
        bbs_map = {}
        bbstripes_map = {} # maps {'wideNarrow': [(str imgpath, (x1,y1,x2,y2)), ...], ...}
        err_imgpaths = []
        bcs2partitionid = {} # maps {(str bcs, ...): int partitionID}
        cur_pid = 0
        topbot_paths = [[TOP_GUARD_IMGP, BOT_GUARD_IMGP], [TOP_GUARD_SKINNY_IMGP, BOT_GUARD_SKINNY_IMGP]]
        decoded_results_map = decode_ballots(ballots, topbot_paths, manager, queue)
        for ballotid, decoded_results in decoded_results_map.iteritems():
            imgpaths = ballots[ballotid]
            decoded_strs = []
            for i, (bcs, isflipped, bbs, bbstripes_dict) in enumerate(decoded_results):
                imgpath = imgpaths[i]
                bc_ul = bcs[0]
                if not bc_ul:
                    print "..error on: {0}".format(imgpath)
                    err_imgpaths.append(imgpath)
                else:
                    imgpath = imgpaths[i]
                    info = get_info(bcs)
                    info['isflip'] = isflipped
                    imginfo[imgpath] = info
                    bbs_map[imgpath] = bbs
                    decoded_strs.append(bcs)
                    for label, bbstripes in bbstripes_dict.iteritems():
                        bbstripes_map.setdefault(label, []).extend(bbstripes)
            decoded[ballotid] = tuple(decoded_strs)
            if tuple(decoded_strs) not in bcs2partitionid:
                bcs2partitionid[tuple(decoded_strs)] = cur_pid
                print "ADD WITH NEW CURPID", cur_pid
                partitions.setdefault(cur_pid, []).append(ballotid)
                cur_pid += 1
            else:
                pid = bcs2partitionid[tuple(decoded_strs)]
                print "old[%s]"%cur_pid
                partitions.setdefault(pid, []).append(ballotid)
        return partitions, decoded, imginfo, bbs_map, bbstripes_map, err_imgpaths

    def get_barcode_info(self, bcs):
        return get_info(bcs)

    def __repr__(self):
        return 'HartVendor()'
    def __str__(self):
        return 'HartVendor()'

def _do_decode_ballots(ballots, (topbot_paths,), queue=None):
    try:
        cvread = lambda imP: cv.LoadImage(imP, cv.CV_LOAD_IMAGE_GRAYSCALE)
        topbot_pairs = [[cvread(topbot_paths[0][0]), cvread(topbot_paths[0][1])],
                        [cvread(topbot_paths[1][0]), cvread(topbot_paths[1][1])]]
        results = {}
        for ballotid, imgpaths in ballots.iteritems():
            balresults = []
            for imgpath in imgpaths:
                bcs, isflipped, bbs, bbstripes_dict = decode(imgpath, topbot_pairs)
                balresults.append((bcs, isflipped, bbs, bbstripes_dict))
            results[ballotid] = balresults
            if queue:
                queue.put(True)
        return results
    except:
        traceback.print_exc()

def decode_ballots(ballots, topbot_paths, manager, queue):
    try:
        decoded_results = partask.do_partask(_do_decode_ballots,
                                             ballots,
                                             _args=(topbot_paths,),
                                             combfn='dict',
                                             manager=manager,
                                             pass_queue=queue,
                                             N=12)
        return decoded_results
    except:
        traceback.print_exc()
        return None

