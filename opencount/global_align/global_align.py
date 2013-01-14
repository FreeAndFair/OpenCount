import sys
import numpy as np

sys.path.append('..')
import pixel_reg.imagesAlign as imagesAlign

"""
Functions that globally-align ballot images together.
"""

def cropout_border(I, top, bot, left, right):
    """ float TOP,BOT,LEFT,RIGHT are between [0.0, 1.0]. """
    '''
    h, w = I.shape
    x1 = int(round(left*w))
    y1 = int(round(top*h))
    x2 = int(round(w - (right*w)))
    y2 = int(round(h - (bot*h)))
    Inew = I[y1:y2, x1:x2]
    return np.copy(Inew)
    '''
    # TODO: Disable the cropping, due to adverse affects on Yolo
    return I

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
    Icrop = cropout_border(I, 0.2, 0.2, 0.2, 0.2)
    H, Ireg_crop, err = imagesAlign.imagesAlign(Icrop, Iref, type='rigid', rszFac=0.25)
    Ireg = imagesAlign.imtransform(I, H)
    Ireg = np.nan_to_num(Ireg)
    return H, Ireg, err
