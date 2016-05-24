import numpy as np
import scipy.misc as misc
import cv
import random

from pixel_reg.imagesAlign import imagesAlign


def minmax_cv(imgpaths, do_align=False, rszFac=1.0, trfm_type='rigid',
              minArea=np.power(2, 16), bbs_map=None, imgCache=None):
    """ Generates min/max overlays for IMGPATHS. If DO_ALIGN is
    True, then this also aligns every image to the first image in
    IMGPATHS.
    Input:
        list IMGPATHS: [str imgpath_i, ...]
        bool DO_ALIGN:
        float RSZFAC: Resizing factor for alignment.
        dict BBS_MAP: maps {str imgpath: (x1,y1,x2,y2)}
    Output:
        cvMat minimg, cvMat maximg.
    """
    def load_image(imgpath):
        if imgCache is None:
            return cv.LoadImage(imgpath, cv.CV_LOAD_IMAGE_GRAYSCALE)
        else:
            ((img, imgpath), isHit) = imgCache.load(imgpath)
            return img

    if bbs_map is None:
        bbs_map = {}
    imgpath = imgpaths[0]
    bb0 = bbs_map.get(imgpath, None)
    Imin = load_image(imgpath)
    if bb0:
        coords = (bb0[0], bb0[1], bb0[2] - bb0[0], bb0[3] - bb0[1])
        coords = tuple(map(int, coords))
        cv.SetImageROI(Imin, coords)
    Imax = cv.CloneImage(Imin)

    # Iref = np.asarray(cv.CloneImage(Imin)) if do_align else None
    Iref = (iplimage2np(cv.CloneImage(Imin)) / 255.0) if do_align else None
    for imgpath in imgpaths[1:]:
        I = load_image(imgpath)
        bb = bbs_map.get(imgpath, None)
        if bb:
            bb = tuple(map(int, bb))
            cv.SetImageROI(I, (bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1]))
        Iout = matchsize(I, Imax)
        if do_align:
            tmp_np = iplimage2np(cv.CloneImage(Iout)) / 255.0
            H, Ireg, err = imagesAlign(
                tmp_np, Iref, trfm_type=trfm_type, fillval=0, rszFac=rszFac, minArea=minArea)
            Ireg *= 255.0
            Ireg = Ireg.astype('uint8')
            Iout = np2iplimage(Ireg)
        cv.Max(Iout, Imax, Imax)
        cv.Min(Iout, Imin, Imin)
    return Imin, Imax


def minmax_cv_V2(imgs, do_align=False, rszFac=1.0, trfm_type='rigid',
                 minArea=np.power(2, 16)):
    """ Just like minmax_cv(), but accepts a list of cvMat's instead
    of a list of imgpaths. If you're planning on generating overlays
    of tens-of-thousands of images, calling this function might result
    in a gross-amount of memory usage (since this function keeps them
    all in memory at once).
    """
    Imin = cv.CloneImage(imgs[0])
    Imax = cv.CloneImage(Imin)
    # Iref = np.asarray(cv.CloneImage(Imin)) if do_align else None
    Iref = (iplimage2np(cv.CloneImage(Imin)) / 255.0) if do_align else None
    for I in imgs[1:]:
        Iout = matchsize(I, Imax)
        if do_align:
            tmp_np = iplimage2np(cv.CloneImage(Iout)) / 255.0
            H, Ireg, err = imagesAlign(
                tmp_np, Iref, trfm_type=trfm_type, fillval=0, rszFac=rszFac, minArea=minArea)
            Ireg *= 255.0
            Ireg = Ireg.astype('uint8')
            Iout = np2iplimage(Ireg)

        cv.Max(Iout, Imax, Imax)
        cv.Min(Iout, Imin, Imin)
    return Imin, Imax


def matchsize(A, B):
    """ Given two cvMats A, B, returns a cropped/padded version of
    A that has the same dimensions as B.
    """
    wA, hA = cv.GetSize(A)
    wB, hB = cv.GetSize(B)
    if wA == wB and hA == hB:
        return A
    SetImageROI = cv.SetImageROI
    out = cv.CreateImage((wB, hB), A.depth, A.channels)
    wOut, hOut = cv.GetSize(out)
    if wA < wOut and hA < hOut:
        SetImageROI(out, (0, 0, wA, hA))
    elif wA >= wOut and hA < hOut:
        SetImageROI(out, (0, 0, wOut, hA))
        SetImageROI(A, (0, 0, wOut, hA))
    elif wA < wOut and hA >= hOut:
        SetImageROI(out, (0, 0, wA, hOut))
        SetImageROI(A, (0, 0, wA, hOut))
    else:  # wA >= wOut and hA >= hOut:
        SetImageROI(A, (0, 0, wOut, hOut))
    cv.Copy(A, out)
    cv.ResetImageROI(out)
    cv.ResetImageROI(A)
    return out


def iplimage2np(img):
    # a = np.frombuffer(img.tostring(), dtype=np.uint8)
    # a.shape = img.height, img.width
    a = np.fromstring(img.tostring(), dtype=np.uint8)
    w, h = cv.GetSize(img)
    a = a.reshape(h, w)
    return a


def np2iplimage(array):
    img = cv.CreateImageHeader(
        (array.shape[1], array.shape[0]), cv.IPL_DEPTH_8U, 1)
    cv.SetData(img, array.tostring(),
               array.dtype.itemsize * 1 * array.shape[1])
    return img


def make_minmax_overlay(imgpaths, do_align=False, rszFac=1.0, imgCache=None,
                        queue_mygauge=None,
                        bindataP=None):
    """ Generates the min/max overlays of a set of imagepaths.
    If the images in IMGPATHS are of different size, then this function
    arbitrarily chooses the first image to be the size of the output
    IMIN, IMAX.
    Input:
        list IMGPATHS:
        bool DO_ALIGN:
            If True, then this will choose an arbitrary image A as a reference
            image, and align every image in IMGPATHS to A.
        float RSZFAC:
            Which scale to perform image alignment at.
        obj IMGCACHE:
            If given, the function will use IMGCACHE, an instance of the
            ImageCache class, to load images. Otherwise, it will always
            read each image from disk.
    Output:
        (nparray Imin, nparray Imax).
    """
    # TODO: Implement with bbs_map
    def load_image(imgpath):
        if imgCache is None:
            return misc.imread(imgpath, flatten=True)
        elif bindataP is not None:
            (img, tag), isHit = imgCache.load_binarydat(imgpath, bindataP)
            return img
        else:
            (img, imgpath), isHit = imgCache.load(imgpath)
            return img
    overlayMin, overlayMax = None, None
    Iref = None
    for path in imgpaths:
        img = load_image(path)
        if do_align and Iref is None:
            Iref = img
        elif do_align:
            (H, img, err) = imagesAlign(img, Iref, fillval=0, rszFac=rszFac)
        if overlayMin is None:
            overlayMin = img
        else:
            if overlayMin.shape != img.shape:
                h, w = overlayMin.shape
                img = resize_img_norescale(img, (w, h))
            overlayMin = np.fmin(overlayMin, img)
        if overlayMax is None:
            overlayMax = img
        else:
            if overlayMax.shape != img.shape:
                h, w = overlayMax.shape
                img = resize_img_norescale(img, (w, h))
            overlayMax = np.fmax(overlayMax, img)
        if queue_mygauge is not None:
            queue_mygauge.put(True)

    # HACK: To prevent auto-dynamic-rescaling bugs, where an all-white
    # image is interpreted as all-black, artificially insert a 0.0 at (0,0).
    # See: http://stefaanlippens.net/scipy_unscaledimsave
    overlayMin[0, 0] = 0.0
    overlayMax[0, 0] = 0.0

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
    w, h = size
    shape = (h, w)
    out = np.zeros(shape, dtype=img.dtype)
    i = min(img.shape[0], out.shape[0])
    j = min(img.shape[1], out.shape[1])
    out[0:i, 0:j] = img[0:i, 0:j]
    return out

"""
Originally written by Arel Cordero. Modified by Eric Kim (3/27/2011)
in order to make overlay_im more efficient (if you want to overlay
thousands of PIL images, PIL will crash if too many Image objects
are open at once.
"""

# Colors for overlay
WHITE = (255, 255, 255)
COLOR1 = (255, 0, 0)
COLOR2 = (0, 0, 255)

# Colors represented as arrays
WHITE_ARRAY = np.array(WHITE)
COLOR1_ARRAY = np.array(COLOR1)
COLOR2_ARRAY = np.array(COLOR2)


def ave(l):
    return sum(l) / float(len(l))


def histogram_mean(l, offset=0):
    return offset + sum([l[i] * i for i in range(len(l))]) / float(sum(l))

# See: http://en.wikipedia.org/wiki/Otsu's_method


def otsu(gray_im):
    """
Computes the optimal global threshold for a gray-scale image by maximizing the
variance *between* the two classes of pixels (i.e., light and dark). Operates
efficiently by using the image histogram of the input image.

i.e., use:

threshold = otsu( im )
"""
    hist = gray_im.histogram()
    best = None
    for t in range(len(hist)):
        left = hist[:(t + 1)]
        right = hist[(t + 1):]
        if sum(left) * sum(right) == 0:
            continue  # skip degenerate cases
        inter_class_variance = sum(
            left) * sum(right) * (histogram_mean(left) - histogram_mean(right, len(left)))**2
        if best is None or inter_class_variance > best[1]:
            best = (t, inter_class_variance)
    return best[0]


def kmeans(itemlist, k=2, rounds=10, iterations=5):
    overall_best = None
    for n in range(iterations):
        means = random.sample(list(itemlist), k)  # Want any k distinct items.
        for i in range(rounds):
            counts = [0.0] * k
            newmeans = [0.0] * k
            score = 0
            for l in itemlist:
                best = None
                for n in range(k):
                    if not best or abs(l - means[n]) < best[0]:
                        best = (abs(l - means[n]), n, l)
                counts[best[1]] += 1
                score += best[0]**2
                newmeans[best[1]] = newmeans[best[1]] - \
                    (newmeans[best[1]] - best[2]) * (1 / (counts[best[1]]))
            means = newmeans
        if not overall_best or score < overall_best[0]:
            if overall_best:
                debug("K-means: Yay! Iteration did something.")
            overall_best = (score, means)
    return overall_best[1]


def autothreshold(gray_im, method="otsu"):
    """method can be either "otsu" or "kmeans"."""
    if method == "otsu":
        t = otsu(gray_im)
    elif method == "kmeans":
        t = ave(kmeans(list(gray_im.getdata())))
    return gray_im.point(lambda x: 0 if x < t else 255)  # < or <= ?


"""
Euclidean distance transform
See: http://www.logarithmic.net/pfh/blog/01185880752
"""

import numpy


def _upscan(f):
    for i, fi in enumerate(f):
        if fi == numpy.inf:
            continue
        for j in xrange(1, i + 1):
            x = fi + j * j
            if f[i - j] < x:
                break
            f[i - j] = x


def distance_transform(bitmap):
    f = numpy.where(bitmap, 0.0, numpy.inf)
    for i in xrange(f.shape[0]):
        _upscan(f[i, :])
        _upscan(f[i, ::-1])
    for i in xrange(f.shape[1]):
        _upscan(f[:, i])
        _upscan(f[::-1, i])
    numpy.sqrt(f, f)
    return f

"""
if __name__ == '__main__':
    try:
        import psyco
        psyco.full()
    except ImportError:
        pass

    import pylab
    from numpy import random
    vec = random.random((1000,1000)) < 0.0001
    vec[100:400,500:900] = 1
    vec[500:900,500:900] = 0

    pylab.imshow(distance_transform(vec))
    pylab.show()
"""
