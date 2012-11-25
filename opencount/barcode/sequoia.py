import sys, os, time, pdb, cProfile
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
    ISBACK: True if we detect that this is the backside of a ballot.
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
    if decodings[0] == "0" and decodings[1] == "":
        # Possibly up-right backside.
        return True, False
    elif decodings[0] == "0" and decodings[1] == "0":
        return True, True
    return False, None

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
    w_img, h_img = cv.GetSize(img)
    w_fact, h_fact = 0.15, 0.20
    w_patch, h_patch = int(round(w_img * w_fact)), int(round(h_img * h_fact))
    x_off, y_off = 0, 10
    cv.SetImageROI(img, (x_off, y_off, w_patch, h_patch))
    decodings_left, zero_locs_left, one_locs_left = decode_patch(img, template_zero, template_one, imgpath)
    cv.ResetImageROI(img)
    marks_out_left = create_marks_dict(zero_locs_left, one_locs_left, x_off, y_off, imgpath, LEFT)
    x_off = int(round(w_img - w_patch))
    cv.SetImageROI(img, (x_off, y_off, w_patch, h_patch))
    decodings_rht, zero_locs_rht, one_locs_rht = decode_patch(img, template_zero, template_one, imgpath)
    cv.ResetImageROI(img)
    marks_out_rht = create_marks_dict(zero_locs_rht, one_locs_rht, x_off, y_off, imgpath, RIGHT)

    marks_out = marks_out_left
    for markval, tups in marks_out_rht.iteritems():
        marks_out.setdefault(markval, []).extend(tups)

    return (decodings_left, decodings_rht), marks_out

def decode_patch(I, Izero, Ione, imgpath, T=0.9):
    """ Given a rough region around a barcode, return the decoding.
    Input:
        IplImage I: rough patch around one barcode.
        IplImage IZERO:
        IplImage IONE:
        str IMGPATH:
    Output:
        (str DECODING, list ZERO_LOCS, list ONE_LOCS).
    list ZERO_LOCS, ONE_LOCS: [(x1,y1,x2,y2), ...]
    """
    # LEFT_ZERO_BEST_LOCS: {int imgidx: [(x1,y1,x2,y2,score), ...]}
    # Note: Both IMG and the templates have already been smoothed.
    zero_best_locs = tempmatch.get_tempmatches(Izero, [I], 
                                          do_smooth=tempmatch.SMOOTH_NONE, T=T)[0]
    one_best_locs = tempmatch.get_tempmatches(Ione, [I], 
                                          do_smooth=tempmatch.SMOOTH_NONE, T=T)[0]
    locs0, locs1 = postprocess_locs(zero_best_locs, one_best_locs)

    return transformToBits(locs0, locs1, I), locs0, locs1

def create_marks_dict(zero_bbs, one_bbs, x_off, y_off, imgpath, side):
    """
    Input:
        list ZERO_BBS, ONE_BBS: [(x1,y1,x2,y2,score), ...]
        int X_OFF, Y_OFF:
    Output:
        dict MARKS_OUT. {MARK_ON/MARK_OFF: [(imgpath, (x1,y1,x2,y2), LEFT/RIGHT), ...]}
    """
    marks_out = {}
    for (x1,y1,x2,y2,score) in zero_bbs:
        bb_correct = (x1 + x_off, y1 + y_off, x2 + x_off, y2 + y_off)
        marks_out.setdefault(MARK_OFF, []).append((imgpath, bb_correct, side))
    for (x1,y1,x2,y2,score) in one_bbs:
        bb_correct = (x1 + x_off, y1 + y_off, x2 + x_off, y2 + y_off)
        marks_out.setdefault(MARK_ON, []).append((imgpath, bb_correct, side))
    return marks_out

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

def transformToBits(zero_locs, one_locs, img):
    """Assumes best_locs are the correct locations. 
    Also, the BEST_LOCS are sorted by height.
    """
    # ZERO_LOCS, ONE_LOCS: [(x1,y1,x2,y2,score), ...]

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
    do_display_output = True if 'display' in args else False
    if do_display_output:
        outdir = args[-1]
    else:
        outdir = None
    do_profile = True if 'profile' in args else False
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
        if do_profile:
            cProfile.runctx('decode(I, Izero, Ione, _imgpath=imgpath)', 
                            {}, {'decode': decode, 'I': I, 'Izero': Izero, 'Ione': Ione, 'imgpath': imgpath})
            return
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
        if not do_display_output:
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
    print "    Avg. Time per Image: {0} s".format(dur / float(len(imgpaths)))
    print "    Number of Errors: {0}".format(len(err_imgpaths))
    print "Done."

if __name__ == '__main__':
    main()
