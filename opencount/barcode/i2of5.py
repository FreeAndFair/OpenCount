import os, sys, time, pdb
import numpy as np
import cv

TOP_GUARD_IMGP = 'hart_topguard.png'
BOT_GUARD_IMGP = 'hart_botguard.png'

BC_14_IMGP = 'hart_bc_14.png'
BC_12_IMGP = 'hart_bc_12.png'
BC_10_IMGP = 'hart_bc_10.png'

BC_14_HEIGHT = 489
BC_12_HEIGHT = 450

VERTICAL = 1
HORIZONTAL = 2
WIDE = 3
NARROW = 4

def decode_i2of5(img, n, orient=VERTICAL, debug=False, TOP_GUARD=None, BOT_GUARD=None):
    """ Decodes the interleaved two-of-five barcode. Returns a string.
    Input:
        IplImage img:
        int n: Number of digits in the barcode.
    Output:
        (str decoded, tuple bb), where BB is the bounding box of the
        barcode within IMG: (x, y, w, h)
    """
    # For now, assume that the barcode is bottom-to-top.
    if TOP_GUARD == None:
        TOP_GUARD = cv.LoadImage(TOP_GUARD_IMGP, cv.CV_LOAD_IMAGE_GRAYSCALE)
    if BOT_GUARD == None:
        BOT_GUARD = cv.LoadImage(BOT_GUARD_IMGP, cv.CV_LOAD_IMAGE_GRAYSCALE)

    TOP_GUARD = smooth_constborder(TOP_GUARD, xwin=3, ywin=3, val=255)
    BOT_GUARD = smooth_constborder(BOT_GUARD, xwin=3, ywin=3, val=255)

    TOP_WHITE_PAD = 24  # Num. pixs from white->top guard barcode
    BOT_WHITE_PAD = 31
    
    # 1.) Find location of top/bottom guards, to find location of barcode
    w,h = cv.GetSize(img)
    _ROI = cv.GetImageROI(img)
    cv.SetImageROI(img, (_ROI[0], _ROI[1], w, h/2))
    top_mats = get_tempmatches(TOP_GUARD, img, T=0.86, do_smooth=True, xwin=3, ywin=3)
    cv.SetImageROI(img, _ROI)
    _ROI = cv.GetImageROI(img)
    cv.SetImageROI(img, (_ROI[0], _ROI[1]+h / 2, w, h / 2))
    bot_mats = get_tempmatches(BOT_GUARD, img, T=0.86, do_smooth=True, xwin=3, ywin=3)
    cv.SetImageROI(img, _ROI)
    # 1.a.) Get top-most/bottom-most match.
    top_sorted = sorted(top_mats, key=lambda t: t[1])
    bot_sorted = sorted(bot_mats, key=lambda t: -t[1])
    if not top_sorted and not bot_sorted:
        #print "...couldn't find either TOP/BOT guard..."
        return None, [0,0,1,1]
    elif top_sorted and not bot_sorted:
        (xtop, ytop, sctop) = top_sorted[0]
        (xbot, ybot, scbot) = xtop, ytop + (BC_14_HEIGHT if n==14 else BC_12_HEIGHT), 1.0
    elif bot_sorted and not top_sorted:
        (xbot, ybot, scbot) = bot_sorted[0]
        ybot += h / 2
        (xtop, ytop, sctop) = xbot, ybot - (BC_14_HEIGHT if n==14 else BC_12_HEIGHT), 1.0
    else:
        (xtop, ytop, sctop) = top_sorted[0]
        (xbot, ybot, scbot) = bot_sorted[0]
        ybot += h / 2

    '''
    cv.SetImageROI(img, (_ROI[0]+min(xtop, xbot), 
                         _ROI[1]+ytop,
                         TOP_GUARD.width,
                         ybot-ytop + BOT_GUARD.height - TOP_WHITE_PAD))
    cv.SaveImage("_tightbc.png", img)
    '''
    cv.SetImageROI(img, _ROI)

    out_bb = [min(xtop, xbot),
              ytop + TOP_WHITE_PAD,
              TOP_GUARD.width, 
              (ybot - ytop) + BOT_GUARD.height - TOP_WHITE_PAD]

    # 2.) Crop+Straighten the barcode, so that it's totally horiz/vertical.
    if (ytop >= ybot):
        # Badness - TOP_GUARD needs to be on top of BOT_GUARD
        print "Error - TOP_GUARD not on top of BOT_GUARD:", (xtop,ytop),(xbot,ybot)
        return None, out_bb
        
    cv.SetImageROI(img, shiftROI(cv.GetImageROI(img),
                                 (xtop, ytop, max(TOP_GUARD.width, BOT_GUARD.width),
                                  ybot - ytop + BOT_GUARD.height)))
    img_post = dothreshold(img)

    w_imgpost, h_imgpost = cv.GetSize(img_post)
    # TODO: Implement Me.

    # 3.) Collapse the barcode to be 1D (by summing pix intensity values
    # to the parallel axis). 
    flat = cv.CreateMat(h_imgpost, 1, cv.CV_32S)
    cv.Reduce(img_post, flat, dim=1, op=cv.CV_REDUCE_SUM)
    flat_np = np.asarray(flat)[::-1][:,0]
    flat_tpl = tuple(flat_np)

    # 4.) Now we have something like [100 100 100 0 0 0 100 100 100 0 0 0 ...].
    # Compute the PIX_ON/PIX_OFF, which tells us what intensity
    # value is 'black' or 'white' by computing the histogram of FLAT.

    pix_on, pix_off = 0.0, 0.0
    bins, binsizes = np.histogram(flat_np)
    bins_asort = np.argsort(bins)[::-1]
    a_idx, b_idx = bins_asort[0], bins_asort[1]
    a_val = (binsizes[a_idx] + binsizes[a_idx+1]) / 2.0
    b_val = (binsizes[b_idx] + binsizes[b_idx+1]) / 2.0
    if a_val < b_val:
        pix_on = a_val
        pix_off = b_val
    else:
        pix_on = b_val
        pix_off = a_val

    # 4.a.) Advance to start of barcode
    i, foundbegin = 0, False
    while i < len(flat_tpl) and not foundbegin:
        val = flat_tpl[i]
        if is_pix_on(val, pix_on, pix_off):
            foundbegin = True
        else:
            i += 1
    if not foundbegin: 
        print "Uhoh, couldn't find start of barcode?"
        if debug:
            pdb.set_trace()
        return None, out_bb
    start_idx = i
    out_bb[3] -= i    # skip to start of barcode

    # 4.b.) Find W_NARROW, W_WIDE, B_NARROW, B_WIDE
    # Due to image artifacts, wide/narrow may differ for black/white.
    w_narrow, w_wide = 0.0, 0.0
    b_narrow, b_wide = 0.0, 0.0

    whts, blks = [], []
    curlen = 0
    isblack = True
    for idx, val in enumerate(flat_np[start_idx:]):
        if abs(val - pix_on) <= (0.4 * pix_on):
            if not isblack:
                # Entering Black
                whts.append(curlen)
                curlen = 1
            else:
                curlen += 1
            isblack = True
        elif abs(val - pix_off) <= (0.4 * pix_off):
            if isblack:
                # Entering White
                blks.append(curlen)
                curlen = 1
            else:
                curlen += 1
            isblack = False
        else:
            curlen += 1
    
    bins_whts, binedges_whts = np.histogram(whts)
    bins_blks, binedges_blks = np.histogram(blks)
    # we get somethign like:
    # bins_whts: [16 8 0 0 0 0 5 4 3 2]
    # binsizes_whts: [4, 4.9, 5.8, 6.7, 7.6, 8.5, 9.4, 10.3, 11.2, 12.1, 13.]
    # so, first separate by (16, 8), (5, 4, 3, 2), and then get the 
    # most-populated bucket, and set the parameter to be the coresponding
    # bin value. 
    _idxs0 = np.where(bins_whts == 0)[0]
    _idx0 = _idxs0[int(len(_idxs0)/2)]
    _idxs1 = np.where(bins_blks == 0)[0]
    _idx1 = _idxs1[int(len(_idxs1)/2)]
    _bins0_wht = bins_whts[:_idx0]
    _bins1_wht = bins_whts[_idx0:]
    w_narrow = binedges_whts[np.argmax(_bins0_wht)]
    w_wide = binedges_whts[np.argmax(_bins1_wht)+_idx0]
    _bins0_blk = bins_blks[:_idx1]
    _bins1_blk = bins_blks[_idx1:]
    b_narrow = binedges_blks[np.argmax(_bins0_blk)]
    b_wide = binedges_blks[np.argmax(_bins1_blk)+_idx1]

    #print 'wht_narrow, wht_wide:', int(round(w_narrow)), int(round(w_wide))
    #print 'blk_narrow, blk_wide:', int(round(b_narrow)), int(round(b_wide))
    #cv.SaveImage("_img_post.png", img_post)
    #pdb.set_trace()
    if w_narrow == 0 or w_wide == 0 or b_narrow == 0 or b_wide == 0:
        # Default to sensible values if badness happens. 
        w_narrow = 2.0
        w_wide = 8.0
        b_narrow = 3.0
        b_wide = 10.0
    w_wide = min(w_wide, 12)
    b_wide = min(b_wide, 12)
    
    # 5.) Convert the FLAT_NP to something like [Nblk, Nwht, Wblk, Wwht],
    # i.e. 'Narrow black', 'Wide white'. 
    bars = [] # [barstr_i, ...]
    i = 0
    # 5.b.) Do Convert.
    bars = _convert_flat(flat_tpl, start_idx, pix_on, pix_off, w_narrow, w_wide, b_narrow, b_wide)
    # I2OF5 always starts and ends with (N,N,N,N) and (W,N,N).
    test1 = bars[:4] == [NARROW, NARROW, NARROW, NARROW]
    if not test1:
        '''
        print "Warning: Begin-guard not found. Continuing \
to try decoding anyways."
        '''
        if debug:
            pdb.set_trace()
    test2 = bars[-3:] == [WIDE, NARROW, NARROW]
    if not test2:
        '''
        print "Warning: End-guard not found. Continuing to try \
decoding anyways."
        '''
        if debug:
            pdb.set_trace()
    bars = bars[4:]
    bars = bars[:-3]
    # 6.) Interpret BARS.
    bars_blk, bars_wht = bars[::2], bars[1::2]

    decs_blk, decs_wht = [], []
    for i_blk, bars_sym in enumerate(gen_by_n(bars_blk, 5)):
        sym = get_i2of5_val(bars_sym)
        if sym == None:
            print "...Invalid symbol:", bars_sym
            if debug:
                pdb.set_trace()
            return None, out_bb
        decs_blk.append(sym)
    for i_wht, bars_sym in enumerate(gen_by_n(bars_wht, 5)):
        sym = get_i2of5_val(bars_sym)
        if sym == None:
            print "...Invalid symbol:", bars_sym
            if debug:
                pdb.set_trace()
            return None, out_bb
        decs_wht.append(sym)
    decoded = ''.join(sum(map(None, decs_blk, decs_wht), ()))
    return decoded, out_bb

def is_pix_on(val, pix_on, pix_off):
    return abs(pix_on - val) < abs(pix_off - val)
def w_or_n(cnt, w_narrow, w_wide, step=1):
    return NARROW if (abs((cnt+((step-1)*cnt)) - w_narrow) < abs((cnt+((step-1)*cnt)) - w_wide)) else WIDE

def _convert_flat(flat_np, start_i, pix_on, pix_off, w_narrow, w_wide, b_narrow, b_wide):
    """ Walks through FLAT_NP, turning the 1D-array into a series of
    [NARROW, WIDE, ...]. Note that it alternates from black->white, and
    the first bar is always black.
    TODO: This is currently the most expensive operation. Perhaps 
    doing this in OpenCV (say, computing the derivative?) would be the
    best thing to do.
    """
    bars = [] # i.e. ['NB', 'NW', 'WB', 'WW']
    i = start_i
    cnt = 0
    is_on = False
    n_step = int(round(w_narrow / 2.0))
    w_step = int(round(w_wide / 2.0))
    step = 1
    # Start forward once
    prev_val = flat_np[i]
    i += 1
    cnt += 1
    is_on = True if is_pix_on(prev_val, pix_on, pix_off) else False

    while i < len(flat_np):
        val = flat_np[i]
        ispixon = is_pix_on(val, pix_on, pix_off)
        if ispixon == is_on:
            cnt += 1
        elif is_on:
            bars.append(w_or_n(cnt, b_narrow, b_wide, step=step))
            cnt = 1
            is_on = False
        else:
            bars.append(w_or_n(cnt, w_narrow, w_wide, step=step))
            cnt = 1
            is_on = True
        # Optimization: Step-size larger than 1 (BUGGY)
        i += step
        prev_val = val
    return bars

def get_i2of5_val(bars):
    """ Given a sequence of narrow/wide, returns the value of the
    sequence, as dictated by Interleaved-2-of-5.
    Input:
        list bars: List of ints, i.e.:
            [NARROW, NARROW, WIDE, WIDE, NARROW]
    Output:
        str decimal value.
    """
    N = NARROW; W = WIDE
    mapping = {(N, N, W, W, N): '0',
               (W, N, N, N, W): '1',
               (N, W, N, N, W): '2',
               (W, W, N, N, N): '3',
               (N, N, W, N, W): '4',
               (W, N, W, N, N): '5',
               (N, W, W, N, N): '6',
               (N, N, N, W, W): '7',
               (W, N, N, W, N): '8',
               (N, W, N, W, N): '9'}
    return mapping.get(tuple(bars), None)
        
def gen_by_n(seq, n):
    """ Outputs elements from seq in N-sized chunks. """
    out, cnt = [], 0
    for i, el in enumerate(seq):
        if cnt == 0:
            out.append(el)
            cnt += 1
        elif cnt % n == 0:
            yield out
            out = [el]
            cnt = 1
        else:
            out.append(el)
            cnt += 1
    if out:
        yield out
        
def bestmatch(A, B):
    """ Tries to find the image A within the (larger) image B.
    For instance, A could be a voting target, and B could be a
    contest.
    Input:
        cvMat A: Patch to search for
        cvMat B: Image to search over
    Output:
        ((x,y), s_mat),  location on B of the best match for A.
    """
    w_A, h_A = A.cols, A.rows
    w_B, h_B = B.cols, B.rows
    s_mat = cv.CreateMat(h_B - h_A + 1, w_B - w_A + 1, cv.CV_32F)
    cv.MatchTemplate(B, A, s_mat, cv.CV_TM_CCOEFF_NORMED)
    minResp, maxResp, minLoc, maxLoc = cv.MinMaxLoc(s_mat)
    return maxLoc, s_mat
def iplimage2cvmat(I):
    w, h = cv.GetSize(I)
    if I.depth == cv.IPL_DEPTH_8U and I.channels == 1:
        cvmat = cv.CreateMat(h, w, cv.CV_8UC1)
    elif I.depth == cv.IPL_DEPTH_32F and I.channels == 1:
        cvmat = cv.CreateMat(h, w, cv.CV_32FC1)
    else:
        cvmat = cv.CreateMat(h, w, cv.CV_8UC1)
    cv.Copy(I, cvmat)
    return cvmat

def dothreshold(I):
    # bins := [ 9358    83    67   119   991  2183   153    64   141 12377]
    # binsizes: [   0.    25.5   51.    76.5  102.   127.5  153.   178.5  204.   229.5  255. ]
    newI = cv.CreateImage(cv.GetSize(I), I.depth, I.channels)
    #I_mat = iplimage2cvmat(I)
    #I_np = np.asarray(I_mat)
    #bins, binsizes = np.histogram(I_np)
    cv.Threshold(I, newI, 75, 255.0, cv.CV_THRESH_BINARY)
    return newI

def get_tempmatches(A, B, T=0.8, do_smooth=True, xwin=13, ywin=13, MAX_MATS=50):
    """ Runs template matching, trying to find image A within image
    B. Returns location (and responses) of all matches greater than
    some threshold T.
    Input:
        IplImage A:
        IplImage B:
        float T:
    Output:
        list matches, i.e. [(x1, y1, float resp), ...]
    """
    if do_smooth:
        B_smooth = cv.CreateImage(cv.GetSize(B), B.depth, B.channels)
        cv.Smooth(B, B_smooth, cv.CV_GAUSSIAN, param1=xwin,param2=ywin)
        B = B_smooth
    wA, hA = cv.GetSize(A)
    wB, hB = cv.GetSize(B)
    M = cv.CreateMat(hB-hA+1, wB-wA+1, cv.CV_32F)
    cv.MatchTemplate(B, A, M, cv.CV_TM_CCOEFF_NORMED)
    M_np = np.array(M)
    score = np.inf
    #print 'best score:', np.max(M_np)
    num_mats = 0
    matches = []
    while score > T and num_mats < MAX_MATS:
        M_idx = np.argmax(M_np)
        i = int(M_idx / M.cols)
        j = M_idx % M.cols
        score = M_np[i,j]
        if score < T:
            break
        matches.append((j, i, score))
        # Suppression
        M_np[i-(hA/3):i+(hA/3),
             j-(wA/3):j+(wA/3)] = -1.0
        num_mats += 1
    return matches

def smooth_constborder(A, xwin=5, ywin=5, val=0):
    """ Smooths A with a Gaussian kernel (with window size [XWIN,YWIN]),
    handling the borders of A by using VAL as the intensity value used
    for pixels outside of A.
    Input:
        IplImage A:
    Output:
        IplImage A_smoothed. 
    """
    wA, hA = cv.GetSize(A)
    A_big = cv.CreateImage((wA+2*xwin, hA+2*ywin), A.depth, A.channels)
    # Pass '0' as bordertype due to undocumented OpenCV flag IPL_BORDER_CONSTANT
    # being 0. Wow!
    cv.CopyMakeBorder(A, A_big, (xwin, ywin), 0, val)
    cv.Smooth(A_big, A_big, cv.CV_GAUSSIAN, param1=xwin, param2=ywin)
    cv.SetImageROI(A_big, (xwin, ywin, wA, hA))
    return A_big
def shiftROI(roi, bb):
    """ Returns a new ROI that is the result of shifting ROI by BB. """
    return (roi[0]+bb[0], roi[1]+bb[1], bb[2], bb[3])

def main():
    args = sys.argv[1:]
    imgpath = args[0]
    n = int(args[1])
    img = cv.LoadImage(imgpath, cv.CV_LOAD_IMAGE_GRAYSCALE)
    t = time.time()
    print "Starting decode_i2of5..."
    decoded = decode_i2of5(img, n)
    dur = time.time() - t
    print "...Finished decode_i2of5 ({0} s)".format(dur)
    print "Decoded was:", decoded
    
if __name__ == '__main__':
    main()
