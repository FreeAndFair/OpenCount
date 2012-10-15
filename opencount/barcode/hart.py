import sys, os, pickle, pdb, traceback, time
import cv
import i2of5

def decode_patch(img, n, debug=False):
    """ Decodes the barcode present in IMG, returns it as a string.
    Input:
        IMG: Either a string (imgpath), or an image object.
        int N: Number of decimals in the barcode.
    Output:
        (str DECODED, tuple BB), where BB is the bounding box around the
        barcode: (x1, y1, w, h)
    """
    if type(img) == str:
        I = cv.LoadImageM(img, cv.CV_LOAD_IMAGE_GRAYSCALE)
    else:
        I = img
    return i2of5.decode_i2of5(I, n, debug=debug)

def decode(imgpath):
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
    def makehoriz(mat):
        """ Rotates barcode to be horizontal - for some reason, zbar works
        much better on horizontal barcodes.
        """
        flat = cv.CreateMat(mat.cols, mat.rows, mat.type)
        cv.Transpose(mat, flat)
        cv.Flip(flat, flat, 1)
        return flat
    def doresize(mat, C=2.5):
        new_w, new_h = int(round(mat.cols * C)), int(round(mat.rows * C))
        newmat = cv.CreateMat(new_h, new_w, mat.type)
        cv.Resize(mat, newmat, interpolation=cv.CV_INTER_CUBIC)
        return newmat
    def dothreshold(mat):
        newmat = cv.CreateMat(mat.rows, mat.cols, mat.type)
        cv.Threshold(mat, newmat, 0.0, 255.0, cv.CV_THRESH_BINARY | cv.CV_THRESH_OTSU)
        return newmat
        
    # UpperLeft: 15% of width, 30% of height.
    # LowerLeft: 15% of width, 30% of height.
    # LowerRight: 15% of width, 30% of height.
    I = cv.LoadImageM(imgpath, cv.CV_LOAD_IMAGE_GRAYSCALE)
    w, h = I.cols, I.rows
    isflipped = False
    bbs = [None, None]

    # 1.) First, try to find LowerLeft first. If it fails, then we
    # guess that the ballot is flipped.
    bb_ll = (0, h-1 - int(round(h*0.3)), int(round(w * 0.15)), int(round(h*0.3)))
    LL = cv.GetSubRect(I, bb_ll)

    dec_ll, outbb_ll = decode_patch(LL, 12)
    check_ll = check_result(dec_ll, type='LL')
    if "ERR" in check_ll:
        # 1.a.) Flip it
        isflipped = True
        tmp = cv.CreateMat(I.rows, I.cols, I.type)
        cv.Flip(I, tmp, flipMode=-1)
        I = tmp
        # 1.b.) Re-do LowerLeft
        bb_ll = (0, h-1 - int(round(h*0.3)), int(round(w * 0.15)), int(round(h*0.3)))
        LL = cv.GetSubRect(I, bb_ll)
        dec_ll, outbb_ll = decode_patch(LL, 12)
    # offset outbb_ll by the cropping we did (bb_ll).
    bbs[1] = [outbb_ll[0] + bb_ll[0],
              outbb_ll[1] + bb_ll[1],
              outbb_ll[2],
              outbb_ll[3]]
    # 2.) Decode UpperLeft, LowerRight.
    bb_ul = (0, 0, int(round(w * 0.15)), int(round(h * 0.3)))
    UL = cv.GetSubRect(I, bb_ul)
    dec_ul, outbb_ul = decode_patch(UL, 14)
    bbs[0] = [outbb_ul[0] + bb_ul[0],
              outbb_ul[1] + bb_ul[1],
              outbb_ul[2],
              outbb_ul[3]]
    dec_ul_res = check_result(dec_ul, type='UL')
    dec_ll_res = check_result(dec_ll, type='LL')
    return ((dec_ul_res, dec_ll_res), isflipped, bbs)

def main():
    args = sys.argv[1:]
    imgpath = args[0]
    mode = args[1]
    t = time.time()
    if mode == 'full':
        decoded = decode(imgpath)
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
