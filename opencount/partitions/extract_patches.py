import os, sys, shutil, traceback, pdb, time, multiprocessing
import cv
sys.path.append('..')
import grouping.partask as partask

def extract(imgpatches, do_threshold=None):
    """
    Input:
        dict IMGPATCHES: {imgpath: [((x1,y1,x2,y2), outpath, tag), ...]}
    Output:
        dict IMG2PATCH: {(imgpath, tag): patchpath},
        dict PATCH2STUFF. {patchpath: (imgpath, (x1,y1,x2,y2), tag)}.
    """
    return partask.do_partask(_extract_patches, imgpatches,
                              _args=(do_threshold,),
                              combfn=_combfn,
                              init=({}, {}),
                              N=None)

def _extract_patches(imgpatches, (do_threshold,)):
    img2patch = {}
    patch2stuff = {}
    for imgpath, tups in imgpatches.iteritems():
        I = cv.LoadImage(imgpath, cv.CV_LOAD_IMAGE_UNCHANGED)
        if do_threshold != None:
            cv.Threshold(I, I, do_threshold, 255.0, cv.CV_THRESH_BINARY)
        for ((x1,y1,x2,y2), outpath, tag) in tups:
            try: os.makedirs(os.path.split(outpath)[0])
            except: pass
            cv.SetImageROI(I, tuple(map(int, (x1,y1,x2-x1,y2-y1))))
            cv.SaveImage(outpath, I)
            img2patch[(imgpath, tag)] = outpath
            patch2stuff[outpath] = (imgpath, (x1,y1,x2,y2), tag)
    return img2patch, patch2stuff

def _combfn(a, b):
    img2patchA, patch2stuffA = a
    img2patchB, patch2stuffB = b
    return (dict(img2patchA.items() + img2patchB.items()),
            dict(patch2stuffA.items() + patch2stuffB.items()))
