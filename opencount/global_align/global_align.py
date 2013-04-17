import sys, os, pdb
import numpy as np
import time

from os.path import join as pathjoin

import scipy, scipy.misc, numpy as np

sys.path.append('..')
import pixel_reg.imagesAlign as imagesAlign
import pixel_reg.shared as sh

"""
Functions that globally-align ballot images together.
"""

USAGE = """Usage:
    python global_align.py [-h --help -help] [--num N] [--verbose] IMGPATHS REF_IMGPATH OUTDIR

Aligns all images in IMGPATHS to REF_IMGPATH, and stores registered images
to OUTDIR.
"""

def align_image(I, Iref, crop=True, verbose=False):
    """ Aligns I to IREF (e.g. 'global' alignment). Both IREF and I
    must be correctly 'flipped' before you pass it to this function.
    Input:
        nparray IREF
    The reference image that we are aligning against.
        nparray I
    The image that we want to align.
    Output:
        (nparray H, nparray IREG, float err)
    where H is the transformation matrix, IREG is the result of aligning
    I to IREF, and ERR is the alignment error (a float from [0.0, 1.0]).
    """
    if crop == True:
        Iref_crop = cropout_stuff(Iref, 0.07, 0.07, 0.02, 0.02)
        Icrop = cropout_stuff(I, 0.07, 0.07, 0.02, 0.02)
    H, err = imagesAlign.imagesAlign(Icrop, Iref_crop, trfm_type='rigid', rszFac=0.15, applyWarp=False)
    if verbose:
        print "Alignment Err: ", err
    Ireg = sh.imtransform(I, H)
    #Ireg = np.nan_to_num(Ireg)
    return H, Ireg, err

def cropout_stuff(I, top, bot, left, right):
    """ Crops out some percentage from each side of I. """
    h, w = I.shape
    x1 = int(round(left*w))
    y1 = int(round(top*h))
    x2 = int(round(w - (right*w)))
    y2 = int(round(h - (bot*h)))
    Inew = I[y1:y2, x1:x2]
    return np.copy(Inew)

def correctH(I, H0):
    T0=np.eye(3); T0[0,2]=I.shape[1]/2.0; T0[1,2]=I.shape[0]/2.0
    T1=np.eye(3); T1[0,2]=-I.shape[1]/2.0; T1[1,2]=-I.shape[0]/2.0
    H=np.dot(np.dot(T0,H0),T1)
    return H

def align_strong(I, Iref, scales=(0.15, 0.2, 0.25, 0.3), 
                 crop_I=(0.05, 0.05, 0.05, 0.05), 
                 crop_Iref=None, do_nan_to_num=False):
    """ Alignment strategy: First, crop out 5% from each side of I.
    Then, try a range of scales, and choose the alignment that 
    minimizes the error.
    CURRENTLY NOT USED.
    """
    if crop_I != None:
        Icrop = cropout_stuff(I, crop_I[0], crop_I[1], crop_I[2], crop_I[3])
    else:
        Icrop = I
    if crop_Iref != None:
        Iref_crop = cropout_stuff(Iref, crop_Iref[0], crop_Iref[1], crop_Iref[2], crop_Iref[3])
    else:
        Iref_crop = Iref
    H_best, Ireg_best, err_best = None, None, None
    scale_best = None
    for scale in scales:
        H, Ireg, err = imagesAlign.imagesAlign(Icrop, Iref_crop, fillval=1, trfm_type='rigid', rszFac=scale)
        if err_best == None or err < err_best:
            H_best = H
            Ireg_best = Ireg
            err_best = err
            scale_best = scale
    # Finally, apply H_BEST to I
    Ireg = sh.imtransform(I, H_best)
    if do_nan_to_num:
        Ireg = np.nan_to_num(Ireg)
    return H_best, Ireg, err_best

def main():
    args = sys.argv[1:]

    if '-h' in args or '--help' in args or '-help' in args:
        print USAGE
        exit(0)

    try:
        N = int(args[args.index('--num')+1])
    except:
        N = None

    VERBOSE = '--verbose' in args

    imgsdir = args[-3]
    ref_imgpath = args[-2]
    outdir = args[-1]

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    def get_imagepaths(imgsdir):
        imgpaths = []
        _cnt = 0
        for dirpath, dirnames, filenames in os.walk(imgsdir):
            for imgname in [f for f in filenames if f.lower().endswith('.png')]:
                if N != None and _cnt >= N:
                    return imgpaths
                imgpath = pathjoin(dirpath, imgname)
                imgpaths.append(imgpath)
                _cnt += 1
        return imgpaths

    imgpaths = get_imagepaths(imgsdir)

    Iref = sh.standardImread_v2(ref_imgpath, flatten=True)
    print "Aligning against {0} images...".format(len(imgpaths))
    t = time.time()
    errs, errs_map = [], {}
    for imgpath in imgpaths:
        imgname = os.path.split(imgpath)[1]
        I = sh.standardImread_v2(imgpath, flatten=True)
        H, Ireg, err = align_image(I, Iref, verbose=VERBOSE)
        errs_map[imgname] = err
        errs.append(err)
        outpath = pathjoin(outdir, imgname)
        scipy.misc.imsave(outpath, Ireg)

    dur = time.time() - t
    err_mean, err_std = np.mean(errs), np.std(errs)
    print "Err_Mean: {0}".format(err_mean)
    print "Err_Std : {0}".format(err_std)
    print "Done ({0:.5f}s Total, {1:.8f}s per image)".format(dur,
                                                             dur / len(imgpaths))
    pdb.set_trace()
    
if __name__ == '__main__':
    main()
