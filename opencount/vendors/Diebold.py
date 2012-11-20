import os, sys, traceback, pdb, time, multiprocessing
try:
    import cPickle as pickle
except:
    import pickle
from os.path import join as pathjoin

import cv    
import diebold_raw

sys.path.append('..')
import grouping.partask as partask

from Vendor import Vendor

# Get this script's directory. Necessary to know this information
# since the current working directory may not be the same as where
# this script lives (which is important for loading resources like
# imgs)
try:
    MYDIR = os.path.abspath(os.path.dirname(__file__))
except NameError:
    # This script is being run directly
    MYDIR = os.path.abspath(sys.path[0])

TEMPLATE_PATH = pathjoin(MYDIR, 'diebold-mark.jpg')

ON = 'ON'
OFF = 'OFF'

class DieboldVendor(Vendor):
    def __init__(self, proj):
        self.proj = proj

    def decode_ballots(self, ballots, manager=None, queue=None):
        return partask.do_partask(_decode_ballots,
                                  ballots,
                                  _args=(TEMPLATE_PATH,),
                                  combfn=_combfn,
                                  init=({}, {}, []),
                                  manager=manager,
                                  pass_queue=queue,
                                  N=1)

    def partition_ballots(self, verified_results, manual_labeled):
        partitions = {}
        img2decoding = {}
        imginfo_map = {}
        
        decodings_tmp = {} # maps {imgpath: list}
        for markval, tups in verified_results.iteritems():
            digitval = '1' if markval == ON else '0'
            for (imgpath, bb, idx) in tups:
                decodings_tmp.setdefault(imgpath, []).append((bb[0], digitval))
        
        for imgpath, tups in decodings_tmp.iteritems():
            # sort by x1 coordinate
            tups_sorted = sorted(tups, key=lambda t: t[0])
            decoding = ''.join([val for (x1, val) in tups_sorted])
            img2decoding[imgpath] = (decoding,)
            imginfo = get_imginfo(decoding)
            imginfo_map[imgpath] = imginfo
        for imgpath, decoding in manual_labeled.iteritems():
            img2decoding[imgpath] = (decoding,)
            imginfo_map[imgpath] = get_imginfo(decoding)

        img2bal = pickle.load(self.proj.image_to_ballot)
        bal2imgs = pickle.load(self.proj.ballot_to_images)
        decoding2partition = {} # maps {(dec0, dec1, ...): int partitionID}
        curPartitionID = 0
        history = set()
        for imgpath, decoding in img2decoding.iteritems():
            ballotid = img2bal[imgpath]
            if ballotid in history:
                continue
            imgpaths = bal2imgs[ballotid]
            imgpaths_ordered = sorted(imgpaths, key=lambda imP: imginfo_map[imgP]['page'])
            decodings_ordered = tuple([img2decoding[imP] for imP in imgpaths_ordered])
            partitionid = decoding2partition.get(decodings_ordered, None)
            if partitionid == None:
                decoding2partition[decodings_ordered] = curPartitionID
                partitionid = curPartitionID
                curPartitionID += 1
            partitions.setdefault(partitionid, []).append(ballotid)
            history.add(ballotid)
        
        return partitions, img2decoding, imginfo_map
    
    def __repr__(self):
        return 'DieboldVendor()'
    def __str__(self):
        return 'DieboldVendor()'

def get_page(decoding):
    """ Back side always ends with 01111011110. """
    return 0 if not get_endercode(decoding) == '01111011110' else 1

def get_imginfo(decoding):
    if get_page(decoding) == 0:
        return {'checksum': get_checksum(decoding),
                'precinct': get_precinct(decoding),
                'cardnum': get_cardnum(decoding),
                'seqnum': get_seqnum(decoding),
                'startbit': get_startbit(decoding),
                'page': 0}
    else:
        return {'election_day': get_day(decoding),
                'election_month': get_month(decoding),
                'election_year': get_year(decoding),
                'election_type': get_electiontype(decoding),
                'endercode': get_endercode(decoding),
                'page': 1}

""" Information about Front side barcode """
def get_checksum(decoding):
    return decoding[0:2]
def get_precinct(decoding):
    return decoding[2:15]
def get_cardnum(decoding):
    return decoding[15:28]
def get_seqnum(decoding):
    return decoding[28:31]
def get_startbit(decoding):
    try:
        return decoding[31]
    except:
        traceback.print_exc()
        pdb.set_trace()
        return None

""" Information about Back side barcode """
def get_day(decoding):
    return decoding[0:5]
def get_month(decoding):
    return decoding[5:9]
def get_year(decoding):
    return decoding[9:16]
def get_electiontype(decoding):
    return decoding[16:21]
def get_endercode(decoding):
    return decoding[21:32]

def _combfn(a, b):
    flipmap_a, mark_bbs_map_a, errs_imgpaths_a = a
    flipmap_b, mark_bbs_map_b, errs_imgpaths_b = b
    flipmap_out = dict(flipmap_a.items() + flipmap_b.items())
    mark_bbs_map_out = mark_bbs_map_a
    for marktype, tups in mark_bbs_map_b.iteritems():
        mark_bbs_map_out.setdefault(marktype, []).extend(tups)
    errs_imgpaths_out = errs_imgpaths_a + errs_imgpaths_b
    return (flipmap_out, mark_bbs_map_out, errs_imgpaths_out)

def _decode_ballots(ballots, (template_path,), queue=None):
    """
    Input:
        dict BALLOTS: {int ballotID: [imgpath_i, ...]}
    Output:
        (dict flipmap, dict mark_bbs, list err_imgpaths)
    """
    flipmap = {}
    mark_bbs_map = {}
    err_imgpaths = []
    Itemp = cv.LoadImage(template_path, cv.CV_LOAD_IMAGE_GRAYSCALE)
    for ballotid, imgpaths in ballots.iteritems():
        for imgpath in imgpaths:
            decoding, isflip, bbs = diebold_raw.decode(imgpath, Itemp)
            if decoding == None:
                err_imgpaths.append(imgpath)
            else:
                flipmap[imgpath] = isflip
                # TODO: Diebold decoder currently only outputs bbs of
                # 'ON' timing marks.
                mark_bbs_map.setdefault(ON, []).extend([(imgpath, bb, i) for (i, bb) in enumerate(bbs)])
            if queue:
                queue.put(True)
    return flipmap, mark_bbs_map, err_imgpaths
