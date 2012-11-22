import sys

sys.path.append('..')

import barcodes.sequoia as sequoia

from Vendor import Vendor

class SequoiaVendor(Vendor):
    def __init__(self, proj):
        self.proj = proj

    def decode_ballots(self, ballots, manager=None, queue=None):
        return partask.do_partask(_decode_ballots,
                                  ballots,
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
        return 'SequoiaVendor()'
    def __str__(self):
        return 'SequoiaVendor()'

def _combfn(a, b):
    flipmap_a, mark_bbs_map_a, errs_imgpaths_a = a
    flipmap_b, mark_bbs_map_b, errs_imgpaths_b = b
    flipmap_out = dict(flipmap_a.items() + flipmap_b.items())
    mark_bbs_map_out = mark_bbs_map_a
    for marktype, tups in mark_bbs_map_b.iteritems():
        mark_bbs_map_out.setdefault(marktype, []).extend(tups)
    errs_imgpaths_out = errs_imgpaths_a + errs_imgpaths_b
    return (flipmap_out, mark_bbs_map_out, errs_imgpaths_out)

def _decode_ballots(ballots, (template_path_zero, template_path_one), queue=None):
    """
    Input:
        dict BALLOTS: {int ballotID: [imgpath_i, ...]}
    Output:
        (dict flipmap, dict mark_bbs, list err_imgpaths)
    """
    flipmap = {}
    mark_bbs_map = {} # maps {str "ON"/"OFF": [(imgpath, (x1,y1,x2,y2), userdata), ...]}
    err_imgpaths = []
    Itemp0 = cv.LoadImage(template_path_zero, cv.CV_LOAD_IMAGE_GRAYSCALE)
    Itemp1 = cv.LoadImage(template_path_one, cv.CV_LOAD_IMAGE_GRAYSCALE)
    Itemp0_smooth = tempmatch.smooth(Itemp0, 3, 3, bordertype='const', val=255.0)
    Itemp1_smooth = tempmatch.smooth(Itemp1, 3, 3, bordertype='const', val=255.0)
    for ballotid, imgpaths in ballots.iteritems():
        for imgpath in imgpaths:
            decodings, isflip, mark_locs = sequoia.decode(imgpath, Itemp0_smooth, Itemp1_smooth)
            if decodings == None:
                err_imgpaths.append(imgpath)
            else:
                flipmap[imgpath] = isflip
                for marktype, tups in mark_locs.iteritems():
                    mark_bbs_map.setdefault(marktype, []).extend(tups)
            if queue:
                queue.put(True)
    return flipmap, mark_bbs_map, err_imgpaths
