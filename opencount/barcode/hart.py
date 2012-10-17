import sys, os, pickle, pdb, traceback, time
import cv
import i2of5

def decode_patch(img, n, debug=False, TOP_GUARD=None, BOT_GUARD=None):
    """ Decodes the barcode present in IMG, returns it as a string.
    Input:
        IMG: Either a string (imgpath), or an image object.
        int N: Number of decimals in the barcode.
    Output:
        (str DECODED, tuple BB), where BB is the bounding box around the
        barcode: (x1, y1, w, h)
    """
    if type(img) == str:
        I = cv.LoadImage(img, cv.CV_LOAD_IMAGE_GRAYSCALE)
    else:
        I = img
    return i2of5.decode_i2of5(I, n, debug=debug, TOP_GUARD=TOP_GUARD, BOT_GUARD=BOT_GUARD)

def decode(imgpath, only_ul=True, TOP_GUARD=None, BOT_GUARD=None):
    """ Given a Hart-style ballot, returns the barcodes in the order
    UPPERLEFT, LOWERLEFT, LOWERRIGHT. Will try to detect flipped
    ballots and correct, if need be.
    Input:
        str imgpath:
    Output:
        (list barcodes, bool isflipped, tuple BBS). BARCODES is a list of three
        strings. ISFLIPPED is True if we detected the ballot was flipped. BBS
        is a tuple of tuples: [BB_i, ...].
    """
    def check_result(decoded, type='UL'):
        """ UpperLeft has 14 digits, LowerLeft has 12 digits, and
        LowerRight has 10 digits.
        """
        if not decoded:
            return "ERR0"
        elif type == 'UL' and len(decoded) != 14:
            return "ERR1"
        elif type == 'LL' and len(decoded) != 12:
            return "ERR1"
        elif type == 'LR' and len(decoded) != 10:
            return "ERR1"
        else:
            return decoded
    def dothreshold(mat):
        newmat = cv.CreateMat(mat.rows, mat.cols, mat.type)
        cv.Threshold(mat, newmat, 0.0, 255.0, cv.CV_THRESH_BINARY | cv.CV_THRESH_OTSU)
        return newmat
        
    # UpperLeft: 15% of width, 30% of height.
    # LowerLeft: 15% of width, 30% of height.
    # LowerRight: 15% of width, 30% of height.
    if type(imgpath) == str:
        I = cv.LoadImage(imgpath, cv.CV_LOAD_IMAGE_GRAYSCALE)
    else:
        I = imgpath
    w, h = cv.GetSize(I)
    isflipped = False
    bbs = [None, None]

    # 1.) First, try to find LowerLeft first. If it fails, then we
    # guess that the ballot is flipped.
    bb_ll = (0, h-1 - int(round(h*0.3)), int(round(w * 0.15)), int(round(h*0.3)))
    cv.SetImageROI(I, bb_ll)

    if only_ul:
        dec_ll, outbb_ll, check_ll = None, [0,0,0,0], None
    else:
        dec_ll, outbb_ll = decode_patch(I, 12, TOP_GUARD=TOP_GUARD, BOT_GUARD=BOT_GUARD)
        check_ll = check_result(dec_ll, type='LL')
    if not only_ul and "ERR" in check_ll:
        # 1.a.) Flip it
        isflipped = True
        tmp = cv.CreateImage((w,h), I.depth, I.channels)
        cv.ResetImageROI(I)
        cv.Flip(I, tmp, flipMode=-1)
        I = tmp
        # 1.b.) Re-do LowerLeft
        bb_ll = (0, h-1 - int(round(h*0.3)), int(round(w * 0.15)), int(round(h*0.3)))
        #LL = cv.GetSubRect(I, bb_ll)
        cv.SetImageROI(I, bb_ll)
        dec_ll, outbb_ll = decode_patch(I, 12, TOP_GUARD=TOP_GUARD, BOT_GUARD=BOT_GUARD)
    # offset outbb_ll by the cropping we did (bb_ll).
    bbs[1] = [outbb_ll[0] + bb_ll[0],
              outbb_ll[1] + bb_ll[1],
              outbb_ll[2],
              outbb_ll[3]]
    # 2.) Decode UpperLeft
    bb_ul = (int(round(w*0.02)), int(round(h*0.03)), int(round(w * 0.13)), int(round(h * 0.23)))
    #bb_ul = (0, 0, int(round(w * 0.15)), int(round(h*0.3)))
    cv.SetImageROI(I, bb_ul)
    dec_ul, outbb_ul = decode_patch(I, 14, debug=False, TOP_GUARD=TOP_GUARD, BOT_GUARD=BOT_GUARD)
    check_ul = check_result(dec_ul, type="UL")
    if only_ul and "ERR" in check_ul:
        isflipped = True
        tmp = cv.CreateImage((w,h), I.depth, I.channels)
        cv.ResetImageROI(I)
        cv.Flip(I, tmp, flipMode=-1)
        I = tmp
        cv.SetImageROI(I, bb_ul)
        dec_ul, outbb_ul = decode_patch(I, 14, TOP_GUARD=TOP_GUARD, BOT_GUARD=BOT_GUARD)
    bbs[0] = [outbb_ul[0] + bb_ul[0],
              outbb_ul[1] + bb_ul[1],
              outbb_ul[2],
              outbb_ul[3]]
    dec_ul_res = check_result(dec_ul, type='UL')
    if only_ul:
        return ((dec_ul_res,), isflipped, [bbs[0]])
    dec_ll_res = check_result(dec_ll, type='LL')
    return ((dec_ul_res, dec_ll_res), isflipped, bbs)

def main():
    args = sys.argv[1:]
    imgpath = args[0]
    mode = args[1]
    if 'only_ul' in args:
        only_ul=True
    else:
        only_ul=False
    t = time.time()
    if mode == 'full':
        decoded = decode(imgpath, only_ul=only_ul)
    elif mode == 'patch':
        n = args[2]
        decoded = decode_patch(imgpath, n)
    else:
        print "Unrecognized mode:", mode
        return
    print decoded
    dur = time.time() - t
    print "...Time elapsed: {0} s".format(dur)
    
if __name__ == '__main__':
    main()
