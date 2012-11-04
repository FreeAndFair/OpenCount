import sys, os, pdb, traceback

sys.path.append('..')

import pixel_reg.shared as shared
import pixel_reg.part_match as part_match

DG_PER_PARTITION = 0
DG_PER_BALLOT = 1

def do_digit_group(b2imgs, img2b, partitions_map, partitions_invmap,
                   partition_exmpls, 
                   img2page, img2flip, attrinfo, digitexemplars_map,
                   mode=DG_PER_PARTITION):
    """
    Input:
        dict B2IMGS:
        dict IMG2B:
        dict PARTITIONS_MAP:
        dict PARTITIONS_INVMAP:
        dict IMG2PAGE:
        dict IMG2FLIP: maps {str imgpath: bool isflip}
        dict ATTRINFO: [x1,y1,x2,y2,attrtype,page,numdigits,digitdist]
        dict DIGITEXEMPLARS_MAP: maps {str digitval: [[str regionP, score, (y1,y2,x1,x2), str digitpatch_i], ...]}
        int MODE:
    Output:
        dict DRESULTS. maps {int ID: [str digitstr, [[str digit_i, (x1,y1,x2,y2), score_i], ...]]}.
        where ID is partitionID if MODE is DG_PER_PARTITION, or ballotID if MODE
        is DG_PER_BALLOT.
    """
    x1, y1, x2, y2, attrtype, page, numdigits, digitdist = attrinfo
    
    # 0.) Depending on MODE, grab the image paths to work with.
    imgpaths = []
    flip_map = {} # maps {str imgpath: bool isflip}
    if mode == DG_PER_PARTITION:
        for partitionID, ballotIDs in partition_exmpls.iteritems():
            imgpaths = b2imgs[ballotIDs[0]]
            imgpaths_ordered = sorted(imgpaths, key=lambda imP: img2page[imP])
            imgpaths.append(imgpaths_ordered[page])
            for imgpath in imgpaths_ordered:
                flip_map[imgpath] = img2flip[imgpath]
    else:
        for ballotID, imgpaths in b2imgs.iteritems():
            imgpaths_ordered = sorted(imgpaths, key=lambda imP: img2page[imP])
            imgpaths.append(imgpaths_ordered[page])
            for imgpath in imgpaths_ordered:
                flip_map[imgpath] = img2flip[imgpath]

    # 1.) Load the digit exemplars
    digit_ex_imgs = {} # maps {(str digit, str meta): nparray digit_img}
    for digit, exemplars in digitexemplars_map.iteritems():
        for i, (regionP, score, bb, digitpatch) in enumerate(exemplars):
            I = shared.standardImread(digitpatch, flatten=True)
            digit_ex_imgs[(digit, i)] = I

    # 2.) Invoke digitParse
    bb = (y1, y2, x1, x2)
    rejected_hashes = {}
    accepted_hashes = {}
    # RESULTS: [(imgpath_i, ocrstr_i, imgpatches_i, patchcoords_i, scores_i), ...]
    pm_results = part_match.digitParse(digit_ex_imgs, imgpaths, bb, numdigits,
                                       flipmap=flip_map, rejected_hashes=rejected_hashes,
                                       accepted_hashes=accepted_hashes,
                                       hspace=digitdist)
    dresults = {}
    for (imgpath, ocrstr, imgpatches, patchcoords, scores) in pm_results:
        ballotid = img2b[imgpath]
        if mode == DG_PER_PARTITION:
            id = partitions_invmap[ballotid]
        else:
            id = ballotid
        entry = []
        for i,digit in enumerate(ocrstr):
            entry.append([digit, patchcoords[i], scores[i]])
        row = [ocrstr, entry]
        dresults[id] = row
    return dresults
        

