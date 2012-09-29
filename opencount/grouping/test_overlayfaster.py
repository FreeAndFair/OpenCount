import os, sys, time, pdb, traceback
import numpy as np, scipy.misc as misc, cv

def make_minmax_overlay(imgpaths, do_align=False, rszFac=1.0):
    overlayMin, overlayMax = None, None
    Iref = None
    for path in imgpaths:
        img = misc.imread(path, flatten=1)
        if do_align and Iref == None:
            Iref = img
        elif do_align:
            (H, img, err) = imagesAlign(img, Iref, fillval=0, rszFac=rszFac)
        if (overlayMin == None):
            overlayMin = img
        else:
            if overlayMin.shape != img.shape:
                h, w = overlayMin.shape
                img = resize_img_norescale(img, (w,h))
            overlayMin = np.fmin(overlayMin, img)
        if (overlayMax == None):
            overlayMax = img
        else:
            if overlayMax.shape != img.shape:
                h, w = overlayMax.shape
                img = resize_img_norescale(img, (w,h))
            overlayMax = np.fmax(overlayMax, img)

    #rszFac=sh.resizeOrNot(overlayMax.shape,sh.MAX_PRECINCT_PATCH_DISPLAY)
    #overlayMax = sh.fastResize(overlayMax, rszFac) #/ 255.0
    #overlayMin = sh.fastResize(overlayMin, rszFac) #/ 255.0
    return overlayMin, overlayMax

def resize_img_norescale(img, size):
    """ Resizes img to be a given size without rescaling - it only
    pads/crops if necessary.
    Input:
        obj img: a numpy array
        tuple size: (w,h)
    Output:
        A numpy array with shape (h,w)
    """
    w,h = size
    shape = (h,w)
    out = np.zeros(shape)
    i = min(img.shape[0], out.shape[0])
    j = min(img.shape[1], out.shape[1])
    out[0:i,0:j] = img[0:i, 0:j]
    return out

def overlay_cv(imgpaths):
    imgpath = imgpaths[0]
    Imin = cv.LoadImage(imgpath, cv.CV_LOAD_IMAGE_GRAYSCALE)
    Imax = cv.CloneImage(Imin)
    for imgpath in imgpaths[1:]:
        I = cv.LoadImage(imgpath, cv.CV_LOAD_IMAGE_GRAYSCALE)
        Iout = matchsize(I, Imax)
        cv.Max(Iout, Imax, Imax)
        #Iout = matchsize(I, Imin)
        cv.Min(Iout, Imin, Imin)
    return Imin, Imax

def matchsize(A, B):
    if A.width == B.width and A.height == B.height:
        return A
    wA, hA = A.width, A.height
    SetImageROI = cv.SetImageROI
    out = cv.CreateImage((B.width, B.height), A.depth, A.channels)
    wOut, hOut = out.width, out.height
    if wA < wOut and hA < hOut:
        SetImageROI(out, (0, 0, wA, hA))
    elif wA >= wOut and hA < hOut:
        SetImageROI(out, (0, 0, wOut, hA))
        SetImageROI(A, (0, 0, wOut, hA))
    elif wA < wOut and hA >= hOut:
        SetImageROI(out, (0, 0, wA, hOut))
        SetImageROI(A, (0, 0, wA, hOut))
    else: # wA >= wOut and hA >= hOut:
        SetImageROI(A, (0, 0, wOut, hOut))
    cv.Copy(A, out)
    cv.ResetImageROI(out)
    return out

def main():
    args = sys.argv[1:]
    imgsdir = args[0]
    
    imgpaths = []
    for dirpath, dirnames, filenames in os.walk(imgsdir):
        for imgname in [f for f in filenames if '.png' in f.lower()]:
            imgpaths.append(os.path.join(dirpath, imgname))
    
    t = time.time()
    print "Starting overlays..."
    minimg, maximg = make_minmax_overlay(imgpaths, do_align=False, rszFac=1.0)
    dur = time.time() - t
    print "...Finished overlays, {0} s".format(dur)

    t = time.time()
    print "Starting overlays..."
    minimg, maximg = overlay_cv(imgpaths)
    dur = time.time() - t
    print "...Finished overlays, {0} s".format(dur)


    print "done."

main()
