import os, sys, time, pdb, traceback, math, pickle
import cv, numpy as np, scipy.stats

import scan_bars
sys.path.append('..')
import grouping.tempmatch as tempmatch
from grouping.verify_overlays_new import iplimage2np

"""
TODO: 
a.) Set GAP parameter programatically somehow.
b.) The MARKFULL_PATH and COLMARK_PATH exemplar images were extracted
    from Marin 2012 images w/ imagesize 1280x2104. Other elections will
    not have the same image size (and possibly even different shape?), so
    we need to account for this.
"""

MARKFULL_PATH = 'diebold_mark.png'
COLMARK_PATH = 'diebold_colmark.png'

DEBUG = False
DEBUG_SAVEIMGS = False

DEBUG_SKIP_FLIP = False

def print_dbg(*args):
    if DEBUG == True:
        for x in args:
            print x,
        print

def decode(imgpath, markpath, colpath):
    return decode_robust_v2(imgpath, markpath, colpath)

def decode_robust_v2(imgpath, markpath, colpath):
    if type(markpath) in (str, unicode):
        markfull = cv.LoadImage(markpath, cv.CV_LOAD_IMAGE_GRAYSCALE)
    else:
        markfull = markpath
    if type(colpath) in (str, unicode):
        Icol = cv.LoadImage(colpath, cv.CV_LOAD_IMAGE_GRAYSCALE)
    else:
        Icol = colpath

    decoding, isflip, bbs = decode_v2_wrapper(imgpath, markfull, Icol)
    return decoding, isflip, bbs

def compute_border(A):
    """ Determines if the image contains rows/cols that are all black. """
    h, w = A.shape
    for i1 in xrange(h):
        thesum = np.sum(A[i1])
        if thesum != 0:
            break
    for i2 in xrange(h-1, -1, -1):
        thesum = np.sum(A[i2])
        if thesum != 0:
            break
    for j1 in xrange(w):
        thesum = np.sum(A[:,j1])
        if thesum != 0:
            break
    for j2 in xrange(w-1, -1, -1):
        thesum = np.sum(A[:,j2])
        if thesum != 0:
            break
    return i1, h - i2, j1, w - j2

def compute_border_leftright(A):
    """ Determines if the left/right contains columns that are all black. """
    h, w = A.shape
    for j1 in xrange(w):
        thesum = np.sum(A[:,j1])
        if thesum != 0:
            break
    for j2 in xrange(w-1, -1, -1):
        thesum = np.sum(A[:,j2])
        if thesum != 0:
            break
    return j1, w - j2

def decode_v2_wrapper(imgpath, markpath, Icol):
    I = cv.LoadImage(imgpath, cv.CV_LOAD_IMAGE_GRAYSCALE)
    result = decode_v2(I, markpath, Icol, False, _imgpath=imgpath)
    if result == None and not DEBUG_SKIP_FLIP:
        print_dbg("...Trying FLIP...")
        cv.ResetImageROI(I)
        result = decode_v2(I, markpath, Icol, True, _imgpath=imgpath)
    
    if result == None:
        return None, None, None
    else:
        decoding, isflip, bbs_out = result
        return (decoding, isflip, bbs_out)

def decode_v2(imgpath, markpath, Icol, isflip, _imgpath=None):
    if type(imgpath) in (str, unicode):
        I = cv.LoadImage(imgpath, cv.CV_LOAD_IMAGE_GRAYSCALE)
    else:
        I = imgpath
    if type(markpath) in (str, unicode):
        Imark = cv.LoadImage(markpath, cv.CV_LOAD_IMAGE_GRAYSCALE)
    else:
        Imark = markpath

    GAP = 7 # TODO: Estimate from current img. resolution.

    if isflip:
        cv.Flip(I, I, -1)

    # Remove the black border from the ballot
    i1_blk, i2_blk, j1_blk, j2_blk = compute_border(iplimage2np(I))
    if DEBUG_SAVEIMGS:
        print_dbg("<><><><> Saving '_I_noblkborder_pre.png' <><><><>")
        cv.SaveImage("_I_noblkborder_pre.png", I)
    shift_roi(I, j1_blk, i1_blk, cv.GetSize(I)[0]-(j1_blk+j2_blk), cv.GetSize(I)[1]-(i1_blk+i2_blk))

    if DEBUG_SAVEIMGS:
        print_dbg("<><><><> Saving '_I_noblkborder.png' <><><><>")
        cv.SaveImage("_I_noblkborder.png", I)
        pdb.set_trace()

    w, h = cv.GetSize(I)
    w_markfull, h_markfull = cv.GetSize(Imark)

    bbs_middle = ((w * 0.2, 0.947 * h,
                   (w-1) - (w*0.4),
                   (0.97 * h)),
                  (w * 0.2, 0.945 * h,
                   (w-1) - (w*0.4),
                   (0.995 *h)),
                  (w * 0.1, 0.93 * h,
                   (w-1) - (w*0.1),
                   (0.995 * h)))

    theta = estimate_ballot_rot(I, Imark, bbs_middle)
    if theta == None:
        print_dbg("Warning: THETA was None.")
        return None
    else:
        print_dbg("==== Theta={0}".format(theta))
    
    bbs_rough = ((0, 0.95 * h,
                  (w-1), (0.98 * h)),
                 (0, 0.945 * h,
                  (w-1), (0.995 * h)))
    result = decoder_v2_helper(I, Icol, bbs_rough, w_markfull, h_markfull, isflip,
                               GAP, theta, 
                               i1_blk, i2_blk, j1_blk, j2_blk,
                               imgpath=_imgpath)
    return result

def find_col_x1(I, Icol, bb, K=3, AX=0.2, AY=0.2, T=0.9):
    """ Tries to find the column of marks on I, using ICOL as a ref.
    image in template matching.
    """
    roi_prev = cv.GetImageROI(I)
    shift_roi(I, bb[0], bb[1], bb[2]-bb[0], bb[3]-bb[1])

    w_A, h_A = cv.GetSize(Icol)
    w_I, h_I = cv.GetSize(I)
    M = cv.CreateMat(h_I - h_A + 1, w_I - w_A + 1, cv.CV_32F)
    cv.MatchTemplate(I, Icol, M, cv.CV_TM_CCOEFF_NORMED)
    if DEBUG_SAVEIMGS:
        M_np = np.array(M)
        import scipy.misc
        print_dbg("<><><><> Saving '_Mbase.png' <><><><>"); cv.SaveImage("_Mbase.png", I)
        print_dbg("<><><><> Saving '_M.png' <><><><>"); scipy.misc.imsave("_M.png", M)
        pdb.set_trace()
    cv.SetImageROI(I, roi_prev)
    i = 0
    xs = []
    _xamt, _yamt = int(round(AX * w_A)), int(round(AY * h_A))
    while i < K:
        minResp, maxResp, minLoc, maxLoc = cv.MinMaxLoc(M)
        if maxResp < T:
            break
        x, y = maxLoc
        # Find the /leftmost/ match: don't find a match in the middle
        # of a column.
        while M[y,x] >= T:
            x -= 1
        xs.append((x + bb[0]))
        _x1 = max(1, x - _xamt)
        _x2 = max(1, x + _xamt)
        _y1 = max(1, y - _yamt)
        _y2 = max(1, y + _yamt)
        M[_y1:_y2, _x1:_x2] = -1.0
        i += 1
    if not xs:
        return None
    elif len(xs) == 1:
        return xs[0]
    return np.median(xs)

def decoder_v2_helper(I, Icol, bbs_rough, w_markfull, h_markfull, isflip, H_GAP, theta, 
                      i1_blk, i2_blk, j1_blk, j2_blk,
                      imgpath=None, find_col=True):
    result = None
    roi_prev = cv.GetImageROI(I)
    w, h = cv.GetSize(I)

    if find_col:

        w, h = cv.GetSize(I)
        bb_left = (0.0, 0.86 * h, 0.06 * w, h-1)
        bb_right = ((w-1) - (0.06*w), 0.86 * h, w-1, h-1)
        if DEBUG_SAVEIMGS:
            print_dbg("<><><><> Saving '_Inocolflush.png' <><><><>"); cv.SaveImage("_Inocolflush.png", I)
        x1_left = find_col_x1(I, Icol, bb_left)
        x1_right = find_col_x1(I, Icol, bb_right)
        print_dbg("== x1_left={0} x1_right={1}".format(x1_left, x1_right))
        x1_left = x1_left if x1_left != None else 0
        x1_right = (min(h-1, x1_right+w_markfull) if x1_right != None else w-1)
    else:
        x1_left, x1_right = 0, w-1
    bb_thecols = (x1_left, 0, x1_right, h-1)
    shift_roi(I, bb_thecols[0], bb_thecols[1], bb_thecols[2] - bb_thecols[0], bb_thecols[3] - bb_thecols[1])
    roi_flushcol = cv.GetImageROI(I)
    if DEBUG_SAVEIMGS:
        print_dbg("<><><><> Saving '_Iflushcols.png' <><><><>"); cv.SaveImage("_Iflushcols.png", I)
        pdb.set_trace()
    for bb_rough in bbs_rough:
        cv.SetImageROI(I, roi_flushcol)
        shift_roi(I, bb_rough[0], bb_rough[1], bb_rough[2] - bb_rough[0], bb_rough[3] - bb_rough[1])
        w_foo, h_foo = cv.GetSize(I)
        Icor, H = rotate_img(I, -theta)

        Hfoo = np.eye(3)
        Hfoo[0:2, 0:3] = np.array(H)
        H_inv = np.linalg.inv(Hfoo)
        H_inv = H_inv[0:2]

        if DEBUG_SAVEIMGS:
            print_dbg("<><><><> Saving '_Icor.png' <><><><>")
            cv.SaveImage("_Icor.png", Icor)
        i1_blk_cor, i2_blk_cor, j1_blk_cor, j2_blk_cor = compute_border(iplimage2np(Icor))
        shift_roi(Icor, j1_blk_cor, i1_blk_cor, cv.GetSize(Icor)[0]-(j1_blk_cor+j2_blk_cor), cv.GetSize(Icor)[1]-(i1_blk_cor+i2_blk_cor))

        if DEBUG_SAVEIMGS:
            print_dbg("<><><><> Saving '_Icor_noblkborder.png' <><><><>")
            cv.SaveImage("_Icor_noblkborder.png", Icor)

        def to_orig_coords(bb, H_inv):
            """ Returns BB (in corrected coord. system) to the original
            coordinate system.
            """
            x1, y1 = transform_pt((bb[0], bb[1]), H_inv, (w_foo, h_foo))
            x1 += bb_rough[0] + bb_thecols[0]
            y1 += bb_rough[1]
            x2, y2 = transform_pt((bb[2], bb[3]), H_inv, (w_foo, h_foo))
            x2 += bb_rough[0] + bb_thecols[0]
            y2 += bb_rough[1]
            return tuple(map(lambda x: int(round(x)), (x1,y1,x2,y2)))

        w_cor, h_cor = cv.GetSize(Icor)
        candidates = []
        y1_step = int(h_markfull / 2.0)
        for step in xrange(int((h_cor-1) / y1_step)):
            shift_roi(Icor, 0, y1_step, w_cor, h_markfull)

            syms, params_ = scan_bars.parse_patch(Icor, (w_markfull, h_markfull), gap=H_GAP, 
                                                  LEN=34,
                                                  orient=scan_bars.HORIZONTAL, MARKTOL=0.7,
                                                  BEGIN_TOL=0.3, END_TOL=0.3)
            decoding = ''.join([t[0] for t in syms])
            # Correct for current-offset in sweep
            bbs_out = [(t[1], y1_step*step, t[1] + w_markfull, y1_step*step + h_markfull) for t in syms]
            # Undo rotation correction
            bbs_out = [to_orig_coords(bb, H_inv) for bb in bbs_out]
            # Add the compute_border offsets (part1)
            bbs_out = [(x1+j1_blk_cor, y1+i1_blk_cor, x2+j1_blk_cor, y2+i2_blk_cor) for (x1,y1,x2,y2) in bbs_out]            
            # Add the compute_border_leftright offsets
            bbs_out = [(x1+j1_blk, y1+i1_blk, x2+j1_blk, y2+i2_blk) for (x1,y1,x2,y2) in bbs_out]

            if DEBUG_SAVEIMGS:
                print_dbg("<><><><> Saving '_Icor_strip.ong' <><><><>")
                cv.SaveImage("_Icor_strip.png", Icor)
                print_dbg("==== decoding ({0}): {1}".format(len(decoding), decoding))
                Icolor = draw_bbs(imgpath, decoding, bbs_out, isflip)
                cv.SaveImage("_dbg_showit.png", Icolor)
                print "<><><><> Saving '_dbg_showit.png' <><><><>"
                pdb.set_trace()

            if sanitycheck_decoding_v2(decoding):
                candidates.append((decoding, isflip, bbs_out))
        if candidates:
            result = most_popular(candidates)
        if result != None:
            break
        print_dbg("==== Trying another bb_rough")
    cv.SetImageROI(I, roi_prev)
    return result

def most_popular(candidates):
    votes = {}
    outputs = {} # maps {str decoding: [(isflip_i, bbs_i), ...]}
    for decoding, isflip, bbs_out in candidates:
        outputs.setdefault(decoding, []).append((isflip, bbs_out))
        if decoding not in votes:
            votes[decoding] = 1
        else:
            votes[decoding] += 1
    best_decoding, best_isflip, best_bbs_out, best_votes = None, None, None, None
    for decoding, vote_cnt in votes.iteritems():
        if best_decoding == None or vote_cnt > best_votes:
            best_decoding = decoding
            best_votes = vote_cnt
    best_outputs = outputs[best_decoding]
    # Return the 'middle' bounding boxes. Other options include
    # returning the median/mean x1s/y1s for each bbox...
    best_isflip, best_bbs_out = best_outputs[int(len(best_outputs) / 2)]
    return best_decoding, best_isflip, best_bbs_out

def sanitycheck_decoding_v2(decoding):
    ALL_ONES = '1'*34
    return (decoding and len(decoding) == 34 and decoding[0] == '1'
            and decoding[-1] == '1'
            and decoding != ALL_ONES)

def estimate_ballot_rot(I, Imarkfull, bbs, MAX_THETA=2.0, K=5):
    roi_prev = cv.GetImageROI(I)
    w_markfull, h_markfull = cv.GetSize(Imarkfull)
    theta_tm = None
    for bb in bbs:
        roi_cur = tuple(map(lambda x: int(round(x)),
                            (bb[0], bb[1],
                             bb[2] - bb[0],
                             bb[3] - bb[1])))
        cv.SetImageROI(I, roi_cur)
        w_cur, h_cur = cv.GetSize(I)

        if DEBUG_SAVEIMGS:
            print_dbg("<><><><> Saving '_Imiddle.png' <><><><>")
            cv.SaveImage("_Imiddle.png", I)
            pdb.set_trace()

        matches = tempmatch.get_tempmatches(Imarkfull, [I], T=0.9, do_smooth=tempmatch.SMOOTH_BOTH_BRD,
                                            xwinI=5, ywinI=5, xwinA=5, ywinA=5)[0]
        matches = sorted(matches, key=lambda t: t[0])
        if matches:
            xs = np.array([t[0] for t in matches])
            ys = np.array([cv.GetSize(I)[1] - t[1] for t in matches])
            if len(xs) <= 1:
                print_dbg("==== Couldn't find enough marks in '_Imiddle.png'.")
                continue
            # Filter out any obvious outliers
            lonely_idxs = detect_lonely_vals(ys, h_markfull)
            xs = np.delete(xs, lonely_idxs)
            ys = np.delete(ys, lonely_idxs)
            if len(xs) <= 1:
                print_dbg("==== Couldn't find enough marks in '_Imiddle.png'.")
                continue
            # Discovered marks must take up at least K*w_markfull space.
            x_area = max(xs) - min(xs)
            if x_area < (K * w_markfull):
                print_dbg("==== Marks only took up {0}, too small space.".format(x_area))
            else:
                theta_tm_ = estimate_rotation(xs, ys)
                if abs(theta_tm_) > MAX_THETA:
                    print_dbg("==== Theta was too large: {0}".format(theta_tm_))
                else:
                    theta_tm = theta_tm_
                    break
        else:
            print_dbg("==== Couldn't find any marks in '_Imiddle.png'.")

    cv.SetImageROI(I, roi_prev)

    return theta_tm

def detect_lonely_vals(vals, h_mark, C=2.0):
    i = 0
    lonely_idxs = []
    while i < len(vals):
        has_friend = False
        val_i = vals[i]
        j = 0
        while j < len(vals):
            if i == j:
                j += 1
                continue
            val_j = vals[j]
            if abs(val_i - val_j) <= (h_mark * C):
                has_friend = True
                break
            j += 1
        if not has_friend:
            lonely_idxs.append(i)
        i += 1
    return lonely_idxs

def transform_pt(pt, H0, Isize):
    x,y = pt
    w, h = Isize
    H = np.eye(3)
    H[0:2, 0:3] = np.array(H0)
    out = np.dot(H, [x,y,1])
    return int(round(out[0])), int(round(out[1]))

def get_rotmat(I, degrees):
    w, h = cv.GetSize(I)
    rotmat = cv.CreateMat(2, 3, cv.CV_32F)
    cv.GetRotationMatrix2D((w/2, h/2), degrees, 1.0, rotmat)
    return rotmat
def apply_rot(I, H):
    Idst = cv.CreateImage(cv.GetSize(I), I.depth, I.channels)
    cv.WarpAffine(I, Idst, H)
    return Idst
def rotate_img(I, degrees):
    H = get_rotmat(I, degrees)
    Idst = apply_rot(I, H)
    return Idst, H

def shift_roi(I, x, y, w, h):
    roi_prev = cv.GetImageROI(I)
    new_roi = tuple(map(int, 
                        (roi_prev[0] + x, roi_prev[1] + y, w, h)))
    cv.SetImageROI(I, new_roi)
    return I

def drawit(I, bbs, imgpath, isflip=False):
    Icolor = cv.LoadImage(imgpath, cv.CV_LOAD_IMAGE_COLOR)
    if isflip:
        cv.Flip(Icolor, Icolor, -1)
    cv.SetImageROI(Icolor, cv.GetImageROI(I))
    for (x1,y1,x2,y2) in bbs:
        cv.Rectangle(Icolor, (x1,y1), (x2,y2), cv.CV_RGB(255, 0, 0))
    print "<><><><> Saving '_Icolor.png' <><><><>"
    cv.SaveImage("_Icolor.png", Icolor)
    pdb.set_trace()

def estimate_rotation(xs, ys):
    # Assumption: len(xs) >= 2
    if not np.any(ys != ys[0]):
        # All the same YS - explicitly return 0.0 to avoid linregress error
        return 0.0
    elif len(xs) == 2:
        # scipy.stats.linregress errors on two points. 
        slope = (ys[1] - ys[0]) / float(xs[1] - xs[0])
        return math.degrees(math.atan2(slope, 1.0))
    slope, intercept, rval, pval, std_err = scipy.stats.linregress(xs, ys)
    degrees = math.degrees(math.atan2((slope*1.0 + intercept) - intercept, 1.0 - 0.0))
    return degrees

def isimgext(f):
    return os.path.splitext(f)[1].lower() in ('.png', '.jpg', '.jpeg', '.bmp')

def draw_bbs(imgpath, decoding, bbs, isflip=False):
    Icolor = cv.LoadImage(imgpath, cv.CV_LOAD_IMAGE_COLOR)
    if isflip:
        cv.Flip(Icolor, Icolor, -1)
    for i, (x1,y1,x2,y2) in enumerate(bbs):
        color = cv.CV_RGB(255, 0, 0) if decoding[i] == '0' else cv.CV_RGB(0, 0, 255)
        cv.Rectangle(Icolor, (x1,y1), (x2,y2), color, thickness=2)
    return Icolor

def main():
    args = sys.argv[1:]
    arg0 = args[-1]
    do_show = '-show' in args
    try: N = int(args[args.index('-n')+1])
    except: N = None
    try: outpath = args[args.index('-o')+1]
    except: outpath = None
    try: erroutpath = args[args.index('--erroutpath')+1]
    except: erroutpath = 'errs_diebold_raw.txt'
    global DEBUG, DEBUG_SAVEIMGS, DEBUG_SKIP_FLIP
    DEBUG = '--debug' in args
    DEBUG_SAVEIMGS = '--saveimgs' in args
    DEBUG_SKIP_FLIP = '--skipflip' in args
    
    if isimgext(arg0):
        imgpaths = [arg0]
    else:
        imgpaths = []
        for dirpath, dirnames, filenames in os.walk(arg0):
            for imgname in [f for f in filenames if isimgext(f)]:
                imgpaths.append(os.path.join(dirpath, imgname))

    t = time.time()
    cnt = 0
    decoding2imgs = {} # maps {str decoding: [imgpath_i, ...]}
    img2decoding = {} # maps {imgpath: str decoding}
    flipmap = {} # maps {str imgpath: bool isflip}
    img2bbs = {} # maps {str imgpath: [bb_i, ...]}
    errs = []
    Imarkfull = cv.LoadImage(MARKFULL_PATH, cv.CV_LOAD_IMAGE_GRAYSCALE)
    Icol = cv.LoadImage(COLMARK_PATH, cv.CV_LOAD_IMAGE_GRAYSCALE)
    for imgpath in imgpaths:
        if N != None and cnt >= N:
            break
        try:
            decoding, isflip, bbs = decode_robust_v2(imgpath, Imarkfull, Icol)
        except Exception as e:
            if type(e) == KeyboardInterrupt:
                raise e
            traceback.print_exc()
            decoding = None

        if decoding == None:
            print 'Error:', imgpath
            errs.append(imgpath)
        else:
            print "{0}: {1} ({2})".format(os.path.split(imgpath)[1], decoding, len(decoding))
            print "    isflip={0}  {1}".format(isflip, imgpath)
            if do_show:
                Icolor = draw_bbs(imgpath, decoding, bbs, isflip=isflip)
                cv.SaveImage("_showit.png", Icolor)
                print "<><><><> Saved '_showit.png' <><><><>"
                pdb.set_trace()
            decoding2imgs.setdefault(decoding, []).append(imgpath)
            img2decoding[imgpath] = decoding
            flipmap[imgpath] = isflip
            img2bbs[imgpath] = bbs
        cnt += 1

    total_dur = time.time() - t
    print "...Done ({0:.6f} s).".format(total_dur)
    if N == None:
        print "    Average Time Per Image: {0:.6f} s".format(total_dur / float(len(imgpaths)))
    else:
        print "    Average Time Per Image: {0:.6f} s".format(total_dur / float(N))
    print "    Number of Errors: {0}".format(len(errs))

    if outpath:
        outfile = open(outpath, 'wb')
        pickle.dump((decoding2imgs, img2decoding, flipmap, img2bbs, errs), outfile)
        print "...Saved pickle'd files to: {0}...".format(outpath)
    
    if errs:
        do_write = raw_input("Would you like to write out all err imgpaths to '{0}' (Y/N)?".format(erroutpath))
        if do_write and do_write.strip().lower() in ('y', 'yes'):
            print "...Writing all err imgpaths to '{0}'...".format(erroutpath)
            f = open(erroutpath, 'w')
            for errpath in errs:
                print >>f, errpath

    print "Done."

if __name__ == '__main__':
    main()
