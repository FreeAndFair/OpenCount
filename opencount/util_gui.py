import os
import math
import csv
import random
try:
    import cPickle as pickle
except ImportError:
    import pickle
import scipy
import cv2 as cv
import scipy.ndimage
import scipy.misc
import imageviewer
from os.path import join as pathjoin

import wx
import wx.animate
from PIL import Image
import numpy as np

# Threshold pixel proximity for util_gui.is_close_to(2)
CONSTANT_CLOSE_TO = 5.0


def fastResize(I, rszFac, sig=-1):
    if rszFac == 1:
        return I
    else:
        Icv = cv.fromarray(np.copy(I))
        I1cv = cv.CreateMat(int(math.floor(
            I.shape[0] * rszFac)), int(math.floor(I.shape[1] * rszFac)), Icv.type)
        cv.Resize(Icv, I1cv)
        Iout = np.asarray(I1cv)
        if sig > 0:
            Iout = gaussian_filter(Iout, sig)

        return Iout


def open_img_scipy(imgpath, flag='normal'):
    """
    Open img in scipy.
    The possible flags are:
        flag='normal' -- Open image in grayscale
        flag='original' -- Open image in its original format, i.e.
                           without flatten=True.
    """
    do_flatten = True if flag == 'normal' else False
    try:
        img = np.array(scipy.ndimage.imread(
            imgpath, flatten=do_flatten), dtype='f')
    except AttributeError:
        # scipy v0.10.0rc1 has imread in scipy.ndimage, but
        # scipy v.0.7.0 has imread in scipy.misc -- joy!
        img = np.array(scipy.misc.imread(
            imgpath, flatten=do_flatten), dtype='f')
    return img


def open_as_grayscale(filename, resize=False, HEIGHT=700):
    """
    Given a filename to an image file, return it as a PIL image in
    grayscale format.
    If the optional resize keyword is True, then this will rescale the
    img to have a height of HEIGHT.
    """
    img = Image.open(filename)
    if resize:
        new_width, new_height = img.size
        if new_height != HEIGHT:
            scale = float(HEIGHT) / new_height
            new_width = int(new_width * scale)
            new_height = int(new_height * scale)
            img = img.resize((new_width, new_height), Image.ANTIALIAS)
    if img.mode != 'L':
        img = img.convert('L')
    return img


def img_to_array(img):
    """
    Convert the input img (either a str imgpath or PIL image) to be
    a Grayscale scipy array.
    """
    try:
        # Image path
        array = open_img_scipy(img)
        return array
    except Exception as e:
        pass
    try:
        # PIL Image
        if img.mode != 'L':
            img = img.convert('L')
        array = np.array(img)
        return array
    except:
        pass
    try:
        # Numpy array
        array = np.array(img)
        return array
    except Exception as e:
        pass
    try:
        # wxBitmap
        pil = util_gui.WxBitmapToPilImage(img)
        array = img_to_array(pil)
        return array
    except:
        # wat
        print 'Unrecognized input to util_gui.img_to_array:', type(img)
        raise RuntimeError("Unrecognized input to util_gui.img_to_array")


def dist_euclidean(pos1, pos2):
    x1, y1 = pos1
    x2, y2 = pos2
    return math.sqrt((x2 - x1)**2.0 + (y2 - y1)**2.0)


def is_close_to(pos1, pos2):
    """ Return True if pos1 is 'close enough' to pos2 """
    return dist_euclidean(pos1, pos2) <= CONSTANT_CLOSE_TO


def is_on_bounding_box(mousepos, boxpos, w_box, h_box, border_size):
    x, y = mousepos
    x_box, y_box = boxpos
    if (x >= x_box and x <= (x_box + w_box) and y >= y_box and y <= (y_box + h_box)):
        # It's within the box itself -- now, is it within the border?
        if (x <= (x_box + border_size) or x >= (x_box + w_box - border_size)
                or y <= (y_box + border_size) or y >= (y_box + h_box - border_size)):
            return True
    return False


def count_images(imgsdir):
    """
    Return the number of images in imgsdir. This function memoizes
    directories too.
    """
    count = 0
    if imgsdir in count_images.cache:
        return count_images.cache[imgsdir]
    else:
        for dirpath, dirnames, filenames in os.walk(imgsdir):
            count += len([f for f in filenames if is_image_ext(f)])
        count_images.cache[imgsdir] = count
        return count
count_images.cache = {}


def create_dirs(*dirs):
    """
    For each dir in dirs, create the directory if it doesn't yet
    exist. Will work for things like:
        foo/bar/baz
    and will create foo, foo/bar, and foo/bar/baz correctly.
    """
    for dir in dirs:
        try:
            os.makedirs(dir)
        except Exception as e:
            pass


def remove_files(*filepaths):
    """ Removes all files in filepaths (only files, not dirs). """
    for filepath in filepaths:
        try:
            os.remove(filepath)
        except Exception as e:
            pass


def get_filename(filepath):
    return os.path.splitext(os.path.split(filepath)[1])[0]


def is_image_ext(filepath):
    ext = os.path.splitext(os.path.split(filepath)[1])[1]
    return ext in ['.png', '.bmp', '.jpg', '.jpeg', '.tif', '.tiff']

# Converter functions from wxPython to PIL Images
# From: http://wiki.wxpython.org/WorkingWithImages
# ## #

# Tested with wxPython 2.3.4.2 and PIL 1.1.3.


def WxBitmapToPilImage(myBitmap):
    return imageToPil(WxBitmapToWxImage(myBitmap))


def WxBitmapToWxImage(myBitmap):
    return wx.ImageFromBitmap(myBitmap)

# -----


def PilImageToWxBitmap(myPilImage):
    return WxImageToWxBitmap(PilImageToWxImage(myPilImage))

# Or, if you want to copy any alpha channel, too (available since wxPython 2.5)
# The source PIL image doesn't need to have alpha to use this routine.
# But, a PIL image with alpha is necessary to get a wx.Image with alpha.


def PilImageToWxImage(myPilImage, copyAlpha=True):

    hasAlpha = myPilImage.mode[-1] == 'A'
    if copyAlpha and hasAlpha:  # Make sure there is an alpha layer copy.

        myWxImage = wx.EmptyImage(*myPilImage.size)
        myPilImageCopyRGBA = myPilImage.copy()
        myPilImageCopyRGB = myPilImageCopyRGBA.convert('RGB')    # RGBA --> RGB
        myPilImageRgbData = myPilImageCopyRGB.tobytes()
        myWxImage.SetData(myPilImageRgbData)
        # Create layer and insert alpha values.
        myWxImage.SetAlphaData(myPilImageCopyRGBA.tobytes()[3::4])

    else:    # The resulting image will not have alpha.

        myWxImage = wx.EmptyImage(*myPilImage.size)
        myPilImageCopy = myPilImage.copy()
        # Discard any alpha from the PIL image.
        myPilImageCopyRGB = myPilImageCopy.convert('RGB')
        myPilImageRgbData = myPilImageCopyRGB.tobytes()
        myWxImage.SetData(myPilImageRgbData)

    return myWxImage

# -----


def imageToPil(myWxImage, flatten=False):
    w, h = myWxImage.GetWidth(), myWxImage.GetHeight()
    myPilImage = Image.new('RGB', (w, h))
    myPilImage.frombytes(myWxImage.GetData())
    if flatten:
        myPilImage = myPilImage.convert("L")
    return myPilImage


def WxImageToWxBitmap(myWxImage):
    return myWxImage.ConvertToBitmap()


def NumpyToWxBitmap(img):
    """
    Assumption: img represents a grayscale img [not sure if necessary]
    """
    img_pil = Image.fromarray(img)
    return PilImageToWxBitmap(img_pil)


def template_match(img, refimg, confidence=0.6, xwin=19, ywin=19):
    """
    Return all matches of refimg inside img, using Template Matching.
    (Gratefully) borrowed from:
        http://stackoverflow.com/questions/7670112/finding-a-subimage-inside-a-numpy-image/9253805# 9253805
    Input:
        obj img: A numpy array representing an image
        obj refimg: A numpy array representing the reference image
        float confidence: threshold value (from [0,1]) for template
                          matching
    Output:
        A tuple of (x,y) coodinates, w.r.t the coordinate system of
        img.
    """
    # OpenCV requires either uint8, or float, but with floats it got
    # buggy and failed badly (I think it had to do with it not
    # correctly handling when 'img' had no decimals, but 'refimg'
    # had decimal expansions, which I suppose means that internally
    # OpenCV.matchTemplate does exact integer comparisons.
    img = img.astype('uint8')
    refimg = refimg.astype('uint8')

    I = cv.fromarray(img)
    ref = cv.fromarray(refimg)
    # I = cv.fromarray(np.copy(img))
    # ref = cv.fromarray(np.copy(refimg))
    I_s = cv.CreateMat(I.rows, I.cols, I.type)
    cv.Smooth(I, I_s, cv.CV_GAUSSIAN, param1=xwin, param2=ywin)
    ref_s = cv.CreateMat(ref.rows, ref.cols, ref.type)
    cv.Smooth(ref, ref_s, cv.CV_GAUSSIAN, param1=xwin, param2=ywin)
    # img = np.array(img, dtype='uint8')
    # refimg = np.array(refimg, dtype='uint8')
    result = cv.CreateMat(I_s.rows - ref_s.rows + 1,
                          I_s.cols - ref_s.cols + 1, cv.CV_32F)
    cv.MatchTemplate(I_s, ref_s, result, cv.CV_TM_CCOEFF_NORMED)
    # result = cv2.matchTemplate(img, refimg, cv2.TM_CCOEFF_NORMED)
    # result is a 'similarity' matrix, with values from -1.0 (?) to 1.0,
    # where 1.0 is most similar to the template image.
    result_np = np.asarray(result)
    match_flatidxs = np.arange(result_np.size)[
        (result_np > confidence).flatten()]
    return [flatidx_to_pixelidx(flatidx, result_np.shape) for flatidx in match_flatidxs]

    # match_flatidxs = np.arange(result.size)[(result>confidence).flatten()]
    # return [flatidx_to_pixelidx(flatidx, result.shape) for flatidx in
    # match_flatidxs]


def flatidx_to_pixelidx(flatidx, shape):
    """
    Given a list of flat indexes, and the shape of the corresponding
    matrix, returns the list of pixel indicies (x,y).
    """
    return np.unravel_index(flatidx, shape)[::-1]


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
        # print "%s, %5.5f" % (t, inter_class_variance)
        if best is None or inter_class_variance > best[1]:
            best = (t, inter_class_variance)
    if not best:
        # This is a totally-degenerate case, i.e. an image where there
        # is only one non-zero pixel intensity value. Don't do any
        # thresholding
        return max(hist)
    return best[0]


def ave(l):
    return sum(l) / float(len(l))


def kmeans(itemlist, k=2, rounds=10, iterations=5):
    overall_best = None
    for n in range(iterations):
        means = random.sample(list(itemlist), k)  # Want any k distinct items.
        for i in range(rounds):
            # print "round", i
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
            # print score, means
        if not overall_best or score < overall_best[0]:
            # if overall_best: print "K-means: Yay! Iteration did something."
            overall_best = (score, means)
    return overall_best[1]


def autothreshold(gray_im, method="otsu"):
    """method can be either "otsu" or "kmeans"."""
    if method == "otsu":
        t = otsu(gray_im)
    elif method == "kmeans":
        t = ave(kmeans(list(gray_im.getdata())))
    return gray_im.point(lambda x: 0 if x <= t else 255)  # < or <= ?


def fit_image(img, padx=0, pady=0, BLACK=0):
    """
    Given a PIL image, cut out all whitespace around the object of
    interest (adding some optional padding too). Remember to also
    center the object.
    """
    thresholded_img = autothreshold(img, method="otsu")
    # print 'Saving thresholded version of first bounding box as: first_refimg_rect_threshold.png'
    # thresholded_img.save("first_refimg_rect_thresholded.png")
    w, h = thresholded_img.size
    pixels = list(thresholded_img.getdata())  # In rows
    # First find top-side hitting
    left, right, top, bottom = None, None, None, None
    for x in range(w):
        # Top-down
        for y in range(h):
            idx = (y * w) + x
            if pixels[idx] == BLACK and (not top or y < top):
                top = y
        # Bottom-up
        for y in range(h)[::-1]:
            idx = (y * w) + x
            if pixels[idx] == BLACK and (not bottom or y > bottom):
                bottom = y
    for y in range(h):
        # Left-right
        for x in range(w):
            idx = (y * w) + x
            if pixels[idx] == BLACK and (not left or x < left):
                left = x
        # Right-left
        for x in range(w)[::-1]:
            idx = (y * w) + x
            if pixels[idx] == BLACK and (not right or x > right):
                right = x

    left = left if left else 0
    right = right if right else w - 1
    top = top if top else 0
    bottom = bottom if bottom else h - 1
    ul_corner = (left - padx if left - padx >= 0 else 0,
                 top - pady if top - pady >= 0 else 0)
    lr_corner = (right + padx + 1 if right + padx + 1 < w else w - 1,
                 bottom + pady + 1 if bottom + pady + 1 < h else h - 1)
    box = ul_corner + lr_corner
    fitted_region = img.crop(box)
    # fitted_region.save('fitted.png')

    # Code to visually output (as a .png) the results of fitting
    '''
    orig_img = img.copy()
    for i in range(w):
        for j in range(h):
            if (i == left-padx) or (i == right+padx):
                thresholded_img.putpixel((i,j), 10)
                orig_img.putpixel((i,j), 10)
    for j in range(h):
        for i in range(w):
            if (j == top-pady) or (j == bottom+pady):
                thresholded_img.putpixel((i,j), 10)
                orig_img.putpixel((i,j), 10)
    thresholded_img.save("threshold_marks.png")
    orig_img.save("orig_marks.png")
    '''

    return fitted_region


def test_fit_image():
    imgname = 'test_fit_image_input1.png'
    try:
        img = Image.open(imgname).convert('L')
    except:
        print "Couldn't find {0}, aborting test_fit_image()".format(imgname)
        return
    fitted = fit_image(img)
    fitted.save('test_fit_image_output1a.png')
    fitted2 = fit_image(img, pady=1)
    fitted2.save('test_fit_image_output1b.png')
    fitted3 = fit_image(img, padx=5, pady=5)
    fitted3.save('test_fit_image_output1c.png')
    print 'test_fit_image(): saved output images, test_fit_image_output1a.png, test_fit_image_output1b.png, test_fit_image_output1c.png'


def get_box_corners((x1, y1), (x2, y2)):
    """
    Given a pair of points, return the same points but in the form:
        (<upper-left corner>, <lower-right corner>)
    """
    ul_x, ul_y = 0, 0
    lr_x, lr_y = 0, 0
    if x1 < x2:
        ul_x = x1
        lr_x = x2
    else:
        ul_x = x2
        lr_x = x1
    if y1 < y2:
        ul_y = y1
        lr_y = y2
    else:
        ul_y = y2
        lr_y = y1
    return ((ul_x, ul_y), (lr_x, lr_y))


def standardize_box(b):
    """
    Given a BoundingBox instance, return a new BoundingBox such
    that x1,y1,x2,y2 are such that:
        (x1,y1) := coordinates for UpperLeft corner
        (x2,y2) := coordinates for LowerRight corner
    """
    b = b.copy()
    x1, y1, x2, y2 = b.get_coords()
    (x1, y1), (x2, y2) = get_box_corners((x1, y1), (x2, y2))
    b.x1 = x1
    b.y1 = y1
    b.x2 = x2
    b.y2 = y2
    return b


def _dictwriter_writeheader(csvfile, fields):
    """ csv.DictWriter.writeheader is not in Python 2.6 or earlier """
    print >>csvfile, ','.join(fields)


def associated_targets(contest, boxes):
    """
    Given a contest and all bounding boxes, return all targets
    that are 'owned' by the contest.
    """
    result = []
    for target in [b for b in boxes if not b.is_contest]:
        if contest.contest_id == target.contest_id:
            result.append(target)
    return result


def find_assoc_contest(target, contest_boxes, debug=False):
    """
    Given a voting target and a list of all contest bounding boxes,
    return the contest bounding box that the target belongs to.
    Input:
        obj target: A BoundingBox instance
        list contest_boxes: A list of BoundingBox instances.
    Output:
        A BoundingBox instance.
    """
    x1, y1, x2, y2 = target.get_coords()
    for c in contest_boxes:
        # if (x1 >= c.x1 and y1 >= c.y1 and x2 <= c.x2 and y2 <= c.y2):
        #    return c
        if (fuzzy_gt(x1, c.x1) and fuzzy_gt(y1, c.y1) and
                fuzzy_lt(x1, c.x2) and fuzzy_lt(y1, c.y2)):
            return c
    # If we get here, then this target is not encompassed by any
    # contest bounding box, which could happen if, say, the user
    # stopped in the middle of modifications.
    return None


def fuzzy_op(x, y, fn, e=1.0e-3):
    return fn(x, y + e) or fn(x, y - e)


def fuzzy_gt(x, y, e=2.0e-3):
    """
    Is x >= y +- e?
    """
    return fuzzy_op(x, y, lambda x, y: x >= y, e=e)


def fuzzy_lt(x, y, e=2.0e-3):
    return fuzzy_op(x, y, lambda x, y: x <= y, e=e)


def get_boxes_inside(boxes, enclosing_region):
    """
    Return all boxes that are contained by enclosing_region.
    Input:
        list boxes: List of BoundingBox instances.
        tuple enclosing_region: tuple of four coordintes:
          (ul_x, ul_y, lr_x, lr_y)     [in relative coords]
    """
    x1, y1, x2, y2 = enclosing_region
    results = []
    for box in boxes:
        if (fuzzy_gt(box.x1, x1) and fuzzy_gt(box.y1, y1) and
                fuzzy_lt(box.x2, x2) and fuzzy_lt(box.y2, y2)):
            results.append(box)
    return results


def resize_boxes(boxes, size, mode='upper-left'):
    """
    Resize all boxes to have a specified size. The specified mode is
    in which direction the resize should happen:
        mode = 'upper-left', 'top', 'upper-right', 'right',
               'lower-right', 'bottom', 'lower-left', 'left'
    """
    mode = mode.lower()
    w_new, h_new = size
    for box in boxes:
        w, h = box.width, box.height
        w_delta, h_delta = w - w_new, h - h_new
        if mode in ('upper-right', 'right', 'lower-right'):
            w_delta *= -1
        if mode in ('lower-left', 'bottom', 'lower-right'):
            h_delta *= -1

        if mode == 'upper-left':
            box.x1 += w_delta
            box.y1 += h_delta
        elif mode == 'top':
            box.y1 += h_delta
        elif mode == 'upper-right':
            box.x2 += w_delta
            box.y1 += h_delta
        elif mode == 'right':
            box.x2 += w_delta
        elif mode == 'lower-right':
            box.x2 += w_delta
            box.y2 += h_delta
        elif mode == 'bottom':
            box.y2 += h_delta
        elif mode == 'lower-left':
            box.x1 += w_delta
            box.y2 += h_delta
        elif mode == 'left':
            box.x1 += w_delta
            box.y1 += h_delta
        else:
            print 'Invalid mode passed to resize_boxes:', mode
            raise RuntimeError
    return boxes

if __name__ == '__main__':
    test_fit_image()
