import sys, os, time, pdb

import cv

sys.path.append('..')
import grouping.tempmatch as tempmatch

"""
A script that decodes Sequoia-style barcodes into 01 bitstrings.

Usage:
    python sequoia.py [List of Images]

Assumptions:
    - Intensity, sizes, and alignment of images are rougly the same within a
      ballot.
    - Two sample templates, one for 0 one for 1, have been cropped out
      manually.

TODO:
    - Overlay verification
    - Integration into Vendor interface
"""

MARK_ON = "ON"
MARK_OFF = "OFF"

def decode(imgpath, Izero, Ione):
    """ Assumes that IZERO, IONE are already smoothed.
    Input:
        str IMGPATH
        IplImage IZERO: 
        IplImage IONE: 
    Output:
        (list DECODINGS, bool ISFLIPPED, dict MARK_LOCS)
    DECODINGS: [str decoding_i, ...]
        If an error in decoding occured, then DECODINGS is None.
    ISFLIPPED: True if the image is flipped, False o.w.
    MARK_LOCS: maps {str ON/OFF: [(x1,y1,x2,y2), ...]}
    """
    I = cv.LoadImage(imgpath, cv.CV_LOAD_IMAGE_GRAYSCALE)
    Ismooth = tempmatch.smooth(I, 3, 3, bordertype='const', val=255.0)
    isflip = False
    decodings, mark_locs = processImg(Ismooth, Izero, Ione, imgpath)
    if decodings == None:
        # Try flip
        isflip = True
        cv.Flip(Ismooth, Ismooth, flipMode=-1)
        decodings, mark_locs = processImg(Ismooth, Izero, Ione, imgpath)
    if decodings == None:
        # Give up.
        return None, None, None
    return decodings, isflip, mark_locs

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

    bit_string = [transformToBits((left_locs0, left_locs1), rough_left_barcode),
                  transformToBits((right_locs0, right_locs1), rough_right_barcode)]
    
    if len(bit_string[0]) != 8 or len(bit_string[1]) != 8:
        print "...Uh oh, bad bit string lengths {0} {1}".format(len(bit_string[0]), len(bit_string[1]))
        return None, None

    # Also correct the offsets from the crop done.
    xOffL, yOffL = offsetLeft
    xOffR, yOffR = offsetRight
    off_tups = [(imgpath, (x1+xOffL, y1+yOffL, x2+xOffL, y2+yOffL), None) for (x1,y1,x2,y2,score) in left_locs0]
    off_tups.extend([(imgpath, (x1+xOffR, y1+yOffR, x2+xOffR, y2+yOffR), None) for (x1,y1,x2,y2,score) in right_locs0])

    on_tups = [(imgpath, (x1+xOffL,y1+yOffL,x2+xOffL,y2+yOffL), None) for (x1,y1,x2,y2,score) in left_locs1]
    on_tups.extend([(imgpath, (x1+xOffR, y1+yOffR, x2+xOffR, y2+yOffR), None) for (x1,y1,x2,y2,score) in right_locs1])
    
    marks_out = {MARK_ON: on_tups, MARK_OFF: off_tups}

    return bit_string, marks_out

def postprocess_locs(zero_locs, one_locs):
    """Post processing the locations:
        - sort them by height
        - check for a possible false positive (top bar)
    """
    zero_locs = sorted(zero_locs, key=lambda tup: tup[1])
    one_locs = sorted(one_locs, key=lambda tup: tup[1])
    return zero_locs, one_locs

def transformToBits(best_locs, img):
    """Assumes best_locs are the correct locations (except that in one_locs,
    the loc with smallest height is a false positive, namely the top bar).
    Also, the BEST_LOCS are sorted by height.
    """
    zero_locs = best_locs[0]
    one_locs = best_locs[1]

    zero_bits = ['0' for _ in zero_locs]
    one_bits = ['1' for _ in one_locs]
    # (Neat trick to interleave two sequences)
    bit_string = "".join(filter(lambda x: x != None, sum(map(None, zero_bits, one_bits), ())))

    return bit_string

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

    ORIG_IMG_W = 1968
    ORIG_IMG_H = 3530

    template_zero_path = "sequoia_template_zero_skinny.png"
    template_one_path = "sequoia_template_one_skinny.png"

    Izero = cv.LoadImage(template_zero_path, cv.CV_LOAD_IMAGE_GRAYSCALE)
    Ione = cv.LoadImage(template_one_path, cv.CV_LOAD_IMAGE_GRAYSCALE)

    # Rescale IZERO/IONE to match this dataset's image dimensions
    exmpl_imgsize = cv.GetSize(cv.LoadImage(imgpaths[0]))
    if exmpl_imgsize != (ORIG_IMG_W, ORIG_IMG_H):
        print "...rescaling images..."
        Izero = rescale_img(Izero, ORIG_IMG_W, ORIG_IMG_H, exmpl_imgsize[0], exmpl_imgsize[1])
        Ione = rescale_img(Ione, ORIG_IMG_W, ORIG_IMG_H, exmpl_imgsize[0], exmpl_imgsize[1])

    Izero = tempmatch.smooth(Izero, 3, 3, bordertype='const', val=255.0)
    Ione = tempmatch.smooth(Ione, 3, 3, bordertype='const', val=255.0)

    t = time.time()
    err_imgpaths = []
    for imgpath in imgpaths:
        decodings, isflip, marklocs = decode(imgpath, Izero, Ione)
        print "For imgpath {0}:".format(imgpath)
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
