import sys
import numpy as np

sys.path.append('..')
import pixel_reg.imagesAlign as imagesAlign

"""
Functions that globally-align ballot images together.
"""

def align_image(Iref, I):
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
    # Idea: First, crop out ~20% of the image perimeter from I to get
    # a smaller image I'. Align I' to IREF to get an alignment H.
    # Apply H to I to get the final alignment IREG.
    # NOTE: The cropping-out is currently DISABLED, due to adverse effects
    #       on Yolo. 
    #Icrop = cropout_border(I, 0.2, 0.2, 0.2, 0.2)
    Icrop = I
    H, Ireg_crop, err = imagesAlign.imagesAlign(Icrop, Iref, type='rigid', rszFac=0.25)
    Ireg = imagesAlign.imtransform(I, H)
    Ireg = np.nan_to_num(Ireg)
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

def align_strong(I, Iref, scales=(0.15, 0.2, 0.25, 0.3)):
    """ Alignment strategy: First, crop out 5% from each side of I.
    Then, try a range of scales, and choose the alignment that 
    minimizes the error.
    """
    Icrop = cropout_stuff(I, 0.05, 0.05, 0.05, 0.05)
    H_best, Ireg_best, err_best = None, None, None
    scale_best = None
    for scale in scales:
        H, Ireg, err = imagesAlign.imagesAlign(Icrop, Iref, fillval=1, type='rigid', rszFac=scale)
        if err_best == None or err < err_best:
            H_best = H
            Ireg_best = Ireg
            err_best = err
            scale_best = scale
    # Finally, apply H_BEST to I
    Ireg = imagesAlign.imtransform(I, H_best)
    return H_best, Ireg, err_best
