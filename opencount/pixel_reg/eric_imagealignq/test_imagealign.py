import sys, os, pdb, traceback, time
import cv, numpy as np, scipy.misc

import pixel_reg.shared as shared
import pixel_reg.imagesAlign as imagesAlign

def main():
    args = sys.argv[1:]
    imgA = args[0]
    imgB = args[1]

    Ia = shared.standardImread(imgA, True)
    Ib = shared.standardImread(imgB, True)

    print "Aligning {0} to {1}...".format(imgA, imgB)
    H, Ireg, err = imagesAlign.imagesAlign(Ia, Ib)
    Ireg_nonan = np.nan_to_num(Ireg)

    scipy.misc.imsave("_IregA.png", Ireg_nonan)

    print "Done, outputted result to: _IregA.png"


main()
