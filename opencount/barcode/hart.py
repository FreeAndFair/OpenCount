import sys, os, pickle, pdb, traceback, time
import zbar, cv

def decode_patch(img):
    """ Decodes the barcode present in IMG, returns it as a string.
    Input:
        IMG: Either a string (imgpath), or an image object.
    Output:
        A tuple of strings, where each string is the decoding of some
        barcode in IMG.
    """
    if type(img) == str:
        I = cv.LoadImageM(img, cv.CV_LOAD_IMAGE_GRAYSCALE)
    else:
        I = img

    scanner = zbar.ImageScanner()
    scanner.parse_config('enable')
    
    w, h = I.cols, I.rows
    raw_img = I.tostring()
    zImg = zbar.Image(w, h, 'Y800', raw_img)
    scanner.scan(zImg)
    
    symbols = []
    for symbol in zImg:
        symbols.append(symbol.data)

    return symbols

def decode(imgpath):
    """ Given a Hart-style ballot, returns the barcodes in the order
    UPPERLEFT, LOWERLEFT, LOWERRIGHT. Will try to detect flipped
    ballots and correct, if need be.
    Input:
        str imgpath:
    Output:
        (list barcodes, bool isflipped). BARCODES is a list of three
        strings. ISFLIPPED is True if we detected the ballot was flipped.
    """
    def check_result(decs, type='UL'):
        """ UpperLeft has 14 digits, LowerLeft has 12 digits, and
        LowerRight has 10 digits.
        """
        def find_len(seq, n):
            for foo in seq:
                if len(foo) == n:
                    return foo
            return "ERROR"
        if not decs:
            return "ERROR"
        elif len(decs) > 1:
            if type == 'UL':
                return find_len(decs, 14)
            elif type == 'LL':
                return find_len(decs, 12)
            else:
                return find_len(decs, 10)
        else:
            return decs[0]
    # UpperLeft: 15% of width, 30% of height.
    # LowerLeft: 15% of width, 30% of height.
    # LowerRight: 15% of width, 30% of height.
    I = cv.LoadImageM(imgpath, cv.CV_LOAD_IMAGE_GRAYSCALE)
    w, h = I.cols, I.rows
    isflipped = False

    # 1.) First, try to find LowerLeft first. If it fails, then we
    # guess that the ballot is flipped.
    LL = cv.GetSubRect(I, (0, h-1 - int(round(h*0.3)), int(round(w * 0.15)), int(round(h*0.3))))
    dec_ll = decode_patch(LL)
    if not dec_ll:
        # 1.a.) Flip it
        isflipped = True
        tmp = cv.CreateMat(I.rows, I.cols, I.type)
        cv.Flip(I, tmp, flipMode=-1)
        I = tmp
        # 1.b.) Re-do LowerLeft
        LL = cv.GetSubRect(I, (0, h-1 - int(round(h*0.3)), int(round(w * 0.15)), int(round(h*0.3))))
        dec_ll = decode_patch(LL)
    # 2.) Decode UpperLeft, LowerRight.
    UL = cv.GetSubRect(I, (0, 0, int(round(w * 0.15)), int(round(h * 0.3))))
    LR = cv.GetSubRect(I, (w-1 - int(round(w * 0.15)), h-1 - int(round(h*0.3)),
                          int(round(w * 0.15)), int(round(h * 0.3))))
    dec_ul = decode_patch(UL)
    dec_lr = decode_patch(LR)
    return (check_result(dec_ul), check_result(dec_ll), check_result(dec_lr), isflipped)

def main():
    args = sys.argv[1:]
    imgpath = args[0]
    mode = args[1]
    t = time.time()
    if mode == 'full':
        decoded = decode(imgpath)
    elif mode == 'patch':
        decoded = decode_patch(imgpath)
    else:
        print "Unrecognized mode:", mode
        return
    print decoded
    dur = time.time() - t
    print "...Time elapsed: {0} s".format(dur)
    
if __name__ == '__main__':
    main()
