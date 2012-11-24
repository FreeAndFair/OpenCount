import sys, os, time, pdb
from os.path import join as pathjoin

import cv

sys.path.append('..')
import grouping.tempmatch as tempmatch

"""
A script that decodes Sequoia-style barcodes into 01 bitstrings.

Usage:
    python sequoia.py [image directory or image path]

Assumptions:
    - Intensity is are rougly the same within a ballot.
    - Two sample templates, one for 0 one for 1, have been cropped out
      manually.
    - A symbol indicating the side is present. x
"""

# Get this script's directory. Necessary to know this information
# since the current working directory may not be the same as where
# this script lives (which is important for loading resources like
# imgs)
try:
    MYDIR = os.path.abspath(os.path.dirname(__file__))
except NameError:
    # This script is being run directly
    MYDIR = os.path.abspath(sys.path[0])

# Original Image Dimensions that the timing mark and side symbol example
# patches were extracted from (used to allow rescaling of patches to
# match current image resolution).
ORIG_IMG_W = 1968
ORIG_IMG_H = 3530

ZERO_IMGPATH = pathjoin(MYDIR, "sequoia_template_zero_skinny.png")
ONE_IMGPATH = pathjoin(MYDIR, "sequoia_template_one_skinny.png")
SIDESYM_IMGPATH = pathjoin(MYDIR, "sequoia_side_symbol.png")

LEFT = "L"
RIGHT = "R"

MARK_ON = "ON"
MARK_OFF = "OFF"

def decode(imgpath, Izero, Ione, _imgpath=None):
    """ Assumes that IZERO, IONE are already smoothed.
    Input:
        str/IplImage IMGPATH: If this is passed in as an IplImage, then
            it is already smoothed, and _IMGPATH is the path to the image.
        IplImage IZERO: 
        IplImage IONE:
    Output:
        (list DECODINGS, bool ISFLIPPED, dict MARK_LOCS, bool isBack)
    DECODINGS: [str decoding_i, ...]
        If an error in decoding occured, then DECODINGS is None.
    ISFLIPPED: True if the image is flipped, False o.w.
    MARK_LOCS: maps {str ON/OFF: [(imgpath, (x1,y1,x2,y2), left/right), ...]}
    """
    if type(imgpath) in (str, unicode):
        I = cv.LoadImage(imgpath, cv.CV_LOAD_IMAGE_GRAYSCALE)
        Ismooth = tempmatch.smooth(I, 3, 3, bordertype='const', val=255.0)
    else:
        I = imgpath
        Ismooth = imgpath
        imgpath = _imgpath

    isflip = False
    decodings, mark_locs = processImg(Ismooth, Izero, Ione, imgpath)
    isbackside, flipback = is_backside(decodings, mark_locs)
    if isbackside:
        return decodings, isflip, mark_locs, True
    elif decodings == None:
        # Try flip
        isflip = True
        cv.Flip(Ismooth, Ismooth, flipMode=-1)
        decodings, mark_locs = processImg(Ismooth, Izero, Ione, imgpath)
    if decodings == None:
        # Give up.
        return None, None, None, None
    #print 'For imgpath {0}: {1}'.format(imgpath, decodings)
    return decodings, isflip, mark_locs, False

def is_backside(decodings, mark_locs):
    """ Applies Sequoia-specific knowledge. A backside ballot side has
    the following 'barcode' values (assume right-side-up):
        UpperLeft: "0"
        UpperRight: ""    (Just a black bar)
        LowerLeft: "0"
        LowerRight: "0"

    Note: This doesn't detect empty backsides.
    
    Output: 
        bool isBack, bool isFlip
    """
    if len(decodings[1]) == 0:
        # Possibly up-right backside.
        if len(decodings[0]) == 1 and decodings[0] == "0":
            return True, False
        else:
            return False, None
    elif len(decodings[0]) == 1 and len(decodings[1]) == 1:
        # Possibly upside-down back-side.
        if decodings[0] == '0' and decodings[1] == '0':
            return True, True
        else:
            return False, None
    return False, None

def compute_side(I, Isidesym):
    """ Sequoia ballots have the property that you can tell which side
    the ballot is based on the location of ISIDESYM (a unique, distinctive
    symbol). If ISIDESYM is found on the upper-right, then I is a front-side
    image. If ISIDESYM is found on the upper-left, then I is a back-side image.

    Finally, if ISIDESYM isn't on the image at all, then this is a single-sided
    ballot (weird!). However, this means that COMPUTE_SIDE won't know if this
    ballot upside-down or not.
    Output:
        (int SIDE, bool ISFLIP, flag isSingleSided). 0 if it's a front, 1 if it's a back.
    """
    w_img, h_img = cv.GetSize(I)
    x1_per = 0.0039
    y1_per = 0.0026
    x2_per = 0.9783
    y2_per = 0.1205
    x1 = int(round(x1_per * w_img))
    y1 = int(round(y1_per * h_img))
    x2 = int(round(x2_per * w_img))
    y2 = int(round(y2_per * h_img))
    w_sidesym, h_sidesym = cv.GetSize(Isidesym)
    cv.SetImageROI(I, (x1,y1,x2-x1,y2-y1))
    (x1_mat, y1_mat, score) = tempmatch.bestmatch(Isidesym, [I], do_smooth=tempmatch.SMOOTH_NONE)[0]
    if score <= 0.83:
        # It might be upside down?
        cv.SetImageROI(I, (x1, h_img - y2, x2, h_img - y1))
        (x1_mat, y1_mat, score) = tempmatch.bestmatch(Isidesym, [I], do_smooth=tempmatch.SMOOTH_NONE)[0]
        cv.ResetImageROI(I)
        x1_mat_frac = x1_mat / float(w_img)
        if score <= 0.83:
            # Couldn't find SIDESYM at all. This might be the backside
            # of a single-sided ballot?
            return None, None, True # Don't know the ISFLIP in this case.
        elif x1_mat_frac >= 0.75:
            return 1, True, False
        elif x1_mat_frac <= 0.255:
            return 0, True, False
        else:
            # Badness!
            return None, None, None
    cv.ResetImageROI(I)
    x1_mat_frac = x1_mat / float(w_img)
    if x1_mat_frac >= 0.75:
        return 0, False, False
    elif x1_mat_frac <= 0.255:
        return 1, False, False
    else:
        return None, None, None

def crop(img, left, top, new_width, new_height):
    """Crops img, returns the region defined by (left, top, new_width,
    new_height)
    """
    if left + new_width > img.width:
        new_width = img.width - left

    cropped = cv.CreateImage((new_width, new_height), cv.IPL_DEPTH_8U, img.channels)
    src_region = cv.GetSubRect(img, (left, top, new_width, new_height))

    cv.Copy(src_region, cropped)
    return cropped

def crop_rough_left(img):
    """Roughly crops the upper left barcode region."""
    w_img, h_img = cv.GetSize(img)
    x_per = 0.0
    x2_per = 0.11
    y_per = 0.02
    y2_per = 0.13542
    x1 = int(round(w_img * x_per))
    y1 = int(round(h_img * y_per))
    width = int(round(((x2_per - x_per) * w_img)))
    height = int(round(((y2_per - y_per) * h_img)))
    return crop(img, x1, y1, width, height), (x1,y1)

def crop_rough_right(img):
    """Roughly crops the upper right barcode region."""
    w_img, h_img = cv.GetSize(img)
    x_per = 0.89
    x2_per = 1.0
    y_per = 0.02
    y2_per = 0.13542
    x1 = int(round(w_img * x_per))
    y1 = int(round(h_img * y_per))
    width = int(round(((x2_per - x_per) * w_img)))
    height = int(round(((y2_per - y_per) * h_img)))
    return crop(img, x1, y1, width, height), (x1,y1)

def processImg(img, template_zero, template_one, imgpath):
    """ The pipeline for processing one image:
        1) crop out two rough barcode regions from the image
        2) run template matching against it with two templates with criteria,
           retrieving the best matches
        3) process matching result, transform into 01-bitstring
    Note: Only the front-side has a full barcodes on the UL/UR corners.
    The back-side, however, has "0", "" on the top, and "0", "0" on the
    bottom. We can leverage this information. 
    Output:
        list DECODINGS, dict MARKS_OUT.
    list DECODINGS: [str decoding_upperLeft, str decoding_upperRight]
    dict MARKS_OUT: maps {MARK_ON/MARK_OFF: [(imgpath, (x1,y1,x2,y2), LEFT/RIGHT), ...]}
    """
    rough_left_barcode, offsetLeft = crop_rough_left(img)
    rough_right_barcode, offsetRight = crop_rough_right(img)
    #cv.SaveImage("_rough_left_barcode.png", rough_left_barcode)
    #cv.SaveImage("_rough_right_barcode.png", rough_right_barcode)
    #cv.SaveImage("_template_zero.png", template_zero)
    #cv.SaveImage("_template_one.png", template_one)

    # LEFT_ZERO_BEST_LOCS: {int imgidx: [(x1,y1,x2,y2,score), ...]}
    # Note: Both IMG and the templates have already been smoothed.
    left_zero_best_locs = tempmatch.get_tempmatches(template_zero, [rough_left_barcode], 
                                           do_smooth=tempmatch.SMOOTH_NONE, T=0.9)[0]
    left_one_best_locs = tempmatch.get_tempmatches(template_one, [rough_left_barcode], 
                                           do_smooth=tempmatch.SMOOTH_NONE, T=0.9)[0]
    right_zero_best_locs = tempmatch.get_tempmatches(template_zero, [rough_right_barcode], 
                                           do_smooth=tempmatch.SMOOTH_NONE, T=0.9)[0]
    right_one_best_locs = tempmatch.get_tempmatches(template_one, [rough_right_barcode],
                                           do_smooth=tempmatch.SMOOTH_NONE, T=0.9)[0]
    
    left_locs0, left_locs1 = postprocess_locs(left_zero_best_locs, left_one_best_locs)
    right_locs0, right_locs1 = postprocess_locs(right_zero_best_locs, right_one_best_locs)

    decodings = [transformToBits((left_locs0, left_locs1), rough_left_barcode),
                 transformToBits((right_locs0, right_locs1), rough_right_barcode)]
    
    # Also correct the offsets from the crop done.
    xOffL, yOffL = offsetLeft
    xOffR, yOffR = offsetRight
    off_tups = [(imgpath, (x1+xOffL, y1+yOffL, x2+xOffL, y2+yOffL), LEFT) for (x1,y1,x2,y2,score) in left_locs0]
    off_tups.extend([(imgpath, (x1+xOffR, y1+yOffR, x2+xOffR, y2+yOffR), RIGHT) for (x1,y1,x2,y2,score) in right_locs0])

    on_tups = [(imgpath, (x1+xOffL,y1+yOffL,x2+xOffL,y2+yOffL), LEFT) for (x1,y1,x2,y2,score) in left_locs1]
    on_tups.extend([(imgpath, (x1+xOffR, y1+yOffR, x2+xOffR, y2+yOffR), RIGHT) for (x1,y1,x2,y2,score) in right_locs1])
    
    marks_out = {MARK_ON: on_tups, MARK_OFF: off_tups}

    return decodings, marks_out

def postprocess_locs(zero_locs, one_locs):
    """Post processing the locations:
        - sort them by height
        - check for a possible false positive (top bar)
    Input:
        ZERO_LOCS, ONE_LOCS: [(x1, y1, x2, y2, score), ...]
    """
    zero_locs = sorted(zero_locs, key=lambda tup: tup[1])
    one_locs = sorted(one_locs, key=lambda tup: tup[1])
    return zero_locs, one_locs

def transformToBits(best_locs, img):
    """Assumes best_locs are the correct locations. 
    Also, the BEST_LOCS are sorted by height.
    """
    # ZERO_LOCS, ONE_LOCS: [(x1,y1,x2,y2,score), ...]
    zero_locs, one_locs = best_locs

    zero_bits = [('0', y1) for (x1,y1,x2,y2,score) in zero_locs]
    one_bits = [('1', y1) for (x1,y1,x2,y2,score) in one_locs]
    # sort by y1
    bits = [val for (val, y1) in sorted(zero_bits+one_bits, key=lambda t: t[1])]
    return ''.join(bits)

def isimgext(f):
    return os.path.splitext(os.path.split(f)[1])[1].lower() in ('.png', '.jpeg', '.jpg', '.bmp')

def rescale_img(I, w0, h0, w1, h1):
    """ Rescales I from image with dimensions (w0,h0) to one with dimensions
    (w1,h1).
    """
    w, h = cv.GetSize(I)
    c = float(w0) / float(w1)
    new_w, new_h = int(round(w / c)), int(round(h / c))
    outImg = cv.CreateImage((new_w, new_h), I.depth, I.channels)
    cv.Resize(I, outImg, cv.CV_INTER_CUBIC)
    return outImg

def main():
    args = sys.argv[1:]
    arg0 = args[0]
    outdir = args[1]
    if os.path.isdir(arg0):
        imgpaths = []
        for dirpath, dirnames, filenames in os.walk(arg0):
            for imgname in [f for f in filenames if isimgext(f)]:
                imgpaths.append(os.path.join(dirpath, imgname))
    else:
        imgpaths = [arg0]

    template_zero_path = "sequoia_template_zero_skinny.png"
    template_one_path = "sequoia_template_one_skinny.png"
    sidesymbol_path = "sequoia_side_symbol.png"

    Izero = cv.LoadImage(template_zero_path, cv.CV_LOAD_IMAGE_GRAYSCALE)
    Ione = cv.LoadImage(template_one_path, cv.CV_LOAD_IMAGE_GRAYSCALE)
    Isidesym = cv.LoadImage(sidesymbol_path, cv.CV_LOAD_IMAGE_GRAYSCALE)

    # Rescale IZERO/IONE/ISIDESYM to match this dataset's image dimensions
    exmpl_imgsize = cv.GetSize(cv.LoadImage(imgpaths[0]))
    if exmpl_imgsize != (ORIG_IMG_W, ORIG_IMG_H):
        print "...rescaling images..."
        Izero = rescale_img(Izero, ORIG_IMG_W, ORIG_IMG_H, exmpl_imgsize[0], exmpl_imgsize[1])
        Ione = rescale_img(Ione, ORIG_IMG_W, ORIG_IMG_H, exmpl_imgsize[0], exmpl_imgsize[1])
        Isidesym = rescale_img(Isidesym, ORIG_IMG_W, ORIG_IMG_H, exmpl_imgsize[0], exmpl_imgsize[1])

    Izero = tempmatch.smooth(Izero, 3, 3, bordertype='const', val=255.0)
    Ione = tempmatch.smooth(Ione, 3, 3, bordertype='const', val=255.0)
    Isidesym = tempmatch.smooth(Isidesym, 3, 3, bordertype='const', val=255.0)

    t = time.time()
    err_imgpaths = []
    for imgpath in imgpaths:
        I = cv.LoadImage(imgpath, cv.CV_LOAD_IMAGE_GRAYSCALE)
        I = tempmatch.smooth(I, 3, 3, bordertype='const', val=255.0)
        print "For imgpath {0}:".format(imgpath)
        decodings, isflip, marklocs, isback = decode(I, Izero, Ione, _imgpath=imgpath)
        if isback:
            print "    Detected backside."
            continue
        if decodings == None:
            print "    ERROR"
        else:
            print "    {0} isflip={1}".format(decodings, isflip)
        
        if decodings == None: 
            err_imgpaths.append(imgpath)
            continue
        # Output colorful image with interpretation displayed nicely
        Icolor = cv.LoadImage(imgpath, cv.CV_LOAD_IMAGE_COLOR)
        if isflip:
            cv.Flip(Icolor, Icolor, flipMode=-1)
        for marktype, tups in marklocs.iteritems():
            if marktype == MARK_ON:
                color = cv.CV_RGB(0, 0, 255)
            else:
                color = cv.CV_RGB(255, 0, 0)
            for (imgpath, (x1,y1,x2,y2), userdata) in tups:
                cv.Rectangle(Icolor, (x1, y1), (x2, y2), color, thickness=2)
        imgname = os.path.split(imgpath)[1]
        outrootdir = os.path.join(outdir, imgname)
        try: os.makedirs(outrootdir)
        except: pass
        outpath = os.path.join(outrootdir, "{0}_bbs.png".format(os.path.splitext(imgname)[0]))
        cv.SaveImage(outpath, Icolor)

    dur = time.time() - t
    print "...Finished Decoding {0} images ({1} s).".format(len(imgpaths), dur)
    print "    Avg. Time per Ballot: {0} s".format(dur / float(len(imgpaths)))
    print "    Number of Errors: {0}".format(len(err_imgpaths))
    print "Done."

if __name__ == '__main__':
    main()
