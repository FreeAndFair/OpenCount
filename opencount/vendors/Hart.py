import os, sys
from os.path import join as pathjoin

import cv

from Vendor import Vendor

sys.path.append('..')

from barcode.hart import decode, get_info

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

class HartVendor(Vendor):
    def __init__(self):
        pass
    
    def decode_image(self, imgpath, TOP_GUARD=None, BOT_GUARD=None):
        """ 
        Input:
            str IMGPATH:
        Output:
            list OUTPUT. Stores a list of all decoded barcodes,
                a boolean saying whether or not the image is flipped, and
                the location of each barcode:
                    [[str bc_i, ...], bool is_flip, [(x1,y1,x2,y2), ...]].
        """
        decodes, isflipped, bbs = decode(imgpath, only_ul=True, TOP_GUARD=TOP_GUARD, BOT_GUARD=BOT_GUARD)
        return decodes, isflipped, bbs

    def decode_ballot(self, ballot, TOP_GUARD=None, BOT_GUARD=None):
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
            (bcs, isflipped, bbs) = self.decode_image(imgpath, TOP_GUARD=TOP_GUARD, BOT_GUARD=BOT_GUARD)
            results.append((bcs, isflipped, bbs))
        return results

    def partition_ballots(self, ballots, queue=None):
        """
        Input:
            dict BALLOTS: {int ballotID: [imgpath_side0, ...]}.
        Output:
            (dict PARTITIONS, dict DECODED, dict BALLOT_INFO, dict BBS_MAP), where PARTITIONS stores the
                partitioning as:
                    {int partitionID: [int ballotID_i, ...]}
                and DECODED stores barcode strings for each ballot as:
                    {int ballotID: [(str BC_side0i, ...), (str BC_side1i, ...)]}
                and IMAGE_INFO stores meaningful info for each image (extracted
                from the barcode):
                    {str imgpath: {str KEY: str VAL}}
                where KEY could be 'page', 'party', 'precinct', 'isflip', etc, and
                BBS_MAP stores the location of the barcodes:
                    {str imgpath: [(x1, y1, x2, y2), ...]}
        """
        partitions = {}
        decoded = {}
        imginfo = {}
        bbs_map = {}
        bcs2partitionid = {} # maps {(str bcs, ...): int partitionID}
        cur_pid = 0
        TOP_GUARD = cv.LoadImage(TOP_GUARD_IMGP, cv.CV_LOAD_IMAGE_GRAYSCALE)
        BOT_GUARD = cv.LoadImage(BOT_GUARD_IMGP, cv.CV_LOAD_IMAGE_GRAYSCALE)
        for ballotid, imgpaths in ballots.iteritems():
            decoded_results = self.decode_ballot(imgpaths, TOP_GUARD=TOP_GUARD, BOT_GUARD=BOT_GUARD)
            d_strs = []
            for i, (bcs, isflipped, bbs) in enumerate(decoded_results):
                imgpath = imgpaths[i]
                info = get_info(bcs)
                info['isflip'] = isflipped
                imginfo[imgpath] = info
                bbs_map[imgpath] = bbs
                d_strs.append(bcs)
            decoded[ballotid] = tuple(d_strs)
            if tuple(d_strs) not in bcs2partitionid:
                bcs2partitionid[tuple(d_strs)] = cur_pid
                partitions.setdefault(cur_pid, []).append(ballotid)
                cur_pid += 1
            else:
                pid = bcs2partitionid[tuple(d_strs)]
                partitions.setdefault(pid, []).append(ballotid)
            if queue:
                queue.put(True)
        return partitions, decoded, imginfo, bbs_map

    def __repr__(self):
        return 'HartVendor()'
    def __str__(self):
        return 'HartVendor()'
