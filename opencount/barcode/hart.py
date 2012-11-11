import sys, os, pickle, pdb, traceback, time
import cv
import i2of5

def decode_patch(img, n, topbot_pairs, debug=False, imgP=None):
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
    return i2of5.decode_i2of5(I, n, topbot_pairs, debug=debug, imgP=imgP)

def decode(imgpath, topbot_pairs, only_ul=True, debug=False):
    """ Given a Hart-style ballot, returns the UPPERLEFT barcode. Will 
    try to detect flipped ballots and correct.
    Input:
        str imgpath:
        list TOPBOT_PAIRS: list of [[IplImage topguard, IplImage botguard], ...].
    Output:
        (list barcodes, bool isflipped, tuple BBS). BARCODES is a list of one
        string (UpperLeft). ISFLIPPED is True if we detected the ballot was 
        flipped. BBS is a tuple of tuples: [BB_i, ...].
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
    if type(imgpath) == str or type(imgpath) == unicode:
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
        dec_ll, outbb_ll = decode_patch(I, 12, topbot_pairs, debug=debug, imgP=imgpath)
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
        dec_ll, outbb_ll = decode_patch(I, 12, topbot_pairs, debug=debug, imgP=imgpath)
    # offset outbb_ll by the cropping we did (bb_ll).
    bbs[1] = [outbb_ll[0] + bb_ll[0],
              outbb_ll[1] + bb_ll[1],
              outbb_ll[2],
              outbb_ll[3]]
    # 2.) Decode UpperLeft
    bb_ul = (10, int(round(h*0.03)), int(round(w * 0.13)), int(round(h * 0.23)))
    #bb_ul = (0, 0, int(round(w * 0.15)), int(round(h*0.3)))
    cv.SetImageROI(I, bb_ul)
    dec_ul, outbb_ul = decode_patch(I, 14, topbot_pairs, debug=debug, imgP=imgpath)
    check_ul = check_result(dec_ul, type="UL")
    if only_ul and "ERR" in check_ul:
        isflipped = True
        tmp = cv.CreateImage((w,h), I.depth, I.channels)
        cv.ResetImageROI(I)
        cv.Flip(I, tmp, flipMode=-1)
        I = tmp
        cv.SetImageROI(I, bb_ul)
        print '...checking FLIP...'
        print dec_ul, outbb_ul
        dec_ul, outbb_ul = decode_patch(I, 14, topbot_pairs, debug=debug, imgP=imgpath)
    bbs[0] = [outbb_ul[0] + bb_ul[0],
              outbb_ul[1] + bb_ul[1],
              outbb_ul[2],
              outbb_ul[3]]
    dec_ul_res = check_result(dec_ul, type='UL')
    if only_ul:
        return ((dec_ul_res,), isflipped, [bbs[0]])
    dec_ll_res = check_result(dec_ll, type='LL')
    return ((dec_ul_res, dec_ll_res), isflipped, bbs)

def get_sheet(bc):
    return bc[0]
def get_precinct(bc):
    return bc[1:7]
def get_page(bc):
    return int(bc[8])
def get_language(bc):
    '''
    MAP = {'0': 'en',
           '1': 'span',
           '2': 'viet',
           '3': 'cn',
           '4': 'kor'}
    k = bc[9]
    return MAP.get(k, 'lang{0}'.format(k))
    '''
    return bc[9]
def get_party(bc):
    '''
    MAP = {'0': 'nonpartisan',
           '1': 'dem',
           '2': 'lib',
           '3': 'american_indep',
           '4': 'green',
           '5': 'peace',
           '6': 'rep',
           '7': 'americans_elect',
           '8': 'demV2',
           '9': 'american_indepV2'}
    return MAP[bc[11]]
    '''
    return bc[10:12]
def get_checksum(bc):
    return bc[-2:]

def get_info(barcodes):
    """ Extracts various semantic meaning(s) from the decoded
    barcodes.
    Input:
        list BARCODES. [bc_i, ...].
    Output:
        dict INFO. Maps {'page': int page, 'party': party_idx, 'sheet': sheet,
                         'language': lang_idx, 'precinct': precinct_idx}
    """
    ul = barcodes[0]
    info = {'sheet': get_sheet(ul), 'precinct': get_precinct(ul),
            'page': get_page(ul), 'language': get_language(ul),
            'party': get_party(ul)}
    return info

def isimgext(f):
    return os.path.splitext(os.path.split(f)[1])[1].lower() in ('.png', '.bmp', '.jpg')

def main():
    args = sys.argv[1:]
    arg0 = args[0]
    mode = args[1]
    if 'only_ul' in args:
        only_ul=True
    else:
        only_ul=False
    imgpaths = []
    if isimgext(arg0):
        imgpaths.append(arg0)
    else:
        for dirpath, dirnames, filenames in os.walk(arg0):
            for imgname in [f for f in filenames if isimgext(f)]:
                imgpaths.append(os.path.join(dirpath, imgname))
    # 1.) Load in top/bot guard pairs.
    topbot_pairs = [[cv.LoadImage(i2of5.TOP_GUARD_IMGP, cv.CV_LOAD_IMAGE_GRAYSCALE),
                     cv.LoadImage(i2of5.BOT_GUARD_IMGP, cv.CV_LOAD_IMAGE_GRAYSCALE)],
                    [cv.LoadImage(i2of5.TOP_GUARD_SKINNY_IMGP, cv.CV_LOAD_IMAGE_GRAYSCALE),
                     cv.LoadImage(i2of5.BOT_GUARD_SKINNY_IMGP, cv.CV_LOAD_IMAGE_GRAYSCALE)]]
    errs = []
    t = time.time()
    if mode == 'full':
        for imgpath in imgpaths:
            decoded = decode(imgpath, topbot_pairs, debug=True, only_ul=only_ul)
            print '{0}: '.format(imgpath), decoded
            if 'ERR' in decoded[0][0]:
                errs.append(imgpath)
    elif mode == 'patch':
        n = args[2]
        decoded = decode_patch(imgpath, n)
    else:
        print "Unrecognized mode:", mode
        return
    dur = time.time() - t
    print "...Time elapsed: {0} s".format(dur)
    avg_time = dur / len(imgpaths)
    print "Avg. Time per ballot ({0} ballots): {1} s".format(len(imgpaths), avg_time)
    print "    Number of Errors: {0}".format(len(errs))
    print errs
    
if __name__ == '__main__':
    main()
