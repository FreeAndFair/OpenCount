import os, sys, time, pdb
import numpy as np
import cv

TOP_GUARD_IMGP = 'hart_topguard.png'
BOT_GUARD_IMGP = 'hart_botguard.png'

BC_14_IMGP = 'hart_bc_14.png'
BC_12_IMGP = 'hart_bc_12.png'
BC_10_IMGP = 'hart_bc_10.png'

VERTICAL = 1
HORIZONTAL = 2

def decode_i2of5(img, n, orient=VERTICAL):
    """ Decodes the interleaved two-of-five barcode. Returns a string.
    Input:
        cvMat img:
        int n: Number of digits in the barcode.
    Output:
        str decoded.
    """
    # For now, assume that the barcode is bottom-to-top.
    #TOP_GUARD = cv.LoadImageM(TOP_GUARD_IMGP, cv.CV_LOAD_IMAGE_GRAYSCALE)
    #BOT_GUARD = cv.LoadImageM(BOT_GUARD_IMGP, cv.CV_LOAD_IMAGE_GRAYSCALE)
    
    # 1.) Find location of top/bottom guards, to find location of barcode
    #matches_top = get_tempmatches(TOP_GUARD, img)
    #matches_bot = get_tempmatches(BOT_GUARD, img)
    # 1.a.) Choose top-most match for TOP, bottom-most match for BOT.
    #(itop, jtop) = 0, 0
    #(ibot, jbot) = 0, 0

    # 1.a.) Search for the barcode.
    if n == 14:
        bcP = BC_14_IMGP
    elif n == 12:
        bcP = BC_12_IMGP
    elif n == 10:
        bcP = BC_10_IMGP
    else:
        print "{0}-digit not supported:", n
        return None
    
    BC = cv.LoadImageM(bcP, cv.CV_LOAD_IMAGE_GRAYSCALE)
    (i, j) = bestmatch(BC, img)

    '''
    imgcpy = cv.CloneMat(img)
    cv.Circle(imgcpy, (i, j), 10, (0, 0, 0, 0), thickness=4)
    cv.SaveImage("imgcpy_bc.png", imgcpy)
    exit(1)
    '''

    # 2.) Crop+Straighten the barcode, so that it's totally horiz/vertical.
    img_post = cv.CloneMat(img)
    # TODO: Implement Me.

    # 3.) Collapse the barcode to be 1D (by summing pix intensity values
    # to the parallel axis). 
    flat = cv.CreateMat(img_post.rows, 1, cv.CV_32S)
    cv.Reduce(img_post, flat, dim=1, op=cv.CV_REDUCE_SUM)

    # 4.) Now we have something like [100 100 100 0 0 0 100 100 100 0 0 0 ...].
    # Compute the PIX_ON/PIX_OFF, which tells us what intensity
    # value is 'black' or 'white'. Could either hardcode, i.e. 0.0 and
    # 255.0, or histogram the FLAT intensity values into two buckets.
    # Find value for W_NARROW (length of first 'cluster'). This will tell
    # us the length of narrow bars, and wide bars (>= 2.0*W_NARROW).
    # Or, we could compute the length of each 'group', then histogram the
    # group lengths into two bins - the median value from each bin will
    # tell us 'narrow' and 'wide'.
    flat_np = np.asarray(flat)
    pix_on, pix_off = 0.0, 0.0

    # 4.a.) Find PIX_ON, PIX_OFF
    bins, binsizes = np.histogram(flat_np)
    bins_asort = np.argsort(bins)
    a_idx, b_idx = bins_asort[0], bins_asort[1]
    a_val = binsizes[a_idx]
    b_val = binsizes[b_idx]
    if a_val < b_val:
        pix_on = a_val
        pix_off = b_val
    else:
        pix_on = b_val
        pix_off = a_val

    # 4.b.) Find W_NARROW, W_WIDE
    w_narrow, w_wide = 0.0, 0.0
    # TODO: For now, hardcode w_narrow, w_wide
    w_narrow = 5.0
    w_wide = 10.0
    
    # 5.) Convert the FLAT_NP to something like [Nblk, Nwht, Wblk, Wwht],
    # i.e. 'Narrow black', 'Wide white'. 
    bars = [] # [barstr_i, ...]
    i = 0
    # 5.a.) Advance to start of barcode
    foundbegin = False
    while i < len(flat_np) and not foundbegin:
        val = flat_np[i][0] # annoying: flat_np[i] = [1984.0]
        if is_pix_on(val, pix_on, pix_off):
            foundbegin = True
        else:
            i += 1
    if not foundbegin: 
        print "Uhoh, couldn't find start of barcode?"
        pdb.set_trace()
    # 5.b.) Do Convert.
    bars = _convert_flat(flat_np, i, pix_on, pix_off, w_narrow, w_wide)
    # 6.) Interpret BARS.
    bars_blk, bars_wht = bars[::2], bars[1::2]

    decs_blk, decs_wht = [], []
    for bars in gen_by_n(bars_blk, 5):
        decs_blk.append(get_i2of5_val(bars))
    for bars in gen_by_n(bars_wht, 5):
        decs_wht.append(get_i2of5_val(bars))
    decoded = zip(decs_blk, decs_wht)
    return decoded

def is_pix_on(val, pix_on, pix_off):
    return abs(pix_on - val) < abs(pix_off - val)
def w_or_n(cnt, w_narrow, w_wide):
    return 'N' if (abs(cnt - w_narrow) < abs(cnt - w_wide)) else 'W'

def _convert_flat(flat_np, start_i, pix_on, pix_off, w_narrow, w_wide):
    """ Walks through FLAT_NP, turning the 1D-array into a series of
    ['NB', 'NW', 'WB', 'WW']. 
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
            bars.append(w_or_n(cnt, w_narrow, w_wide)+'B')
            cnt = 0
            is_on = False
        else:
            bars.append(w_or_n(cnt, w_narrow, w_wide)+'W')
            cnt = 0
            is_on = True
        # Optimization: Step-size larger than 1
        i += n_step
        prev_val = val
    return bars

def get_i2of5_val(bars):
    """ Given a sequence of narrow/wide, returns the value of the
    sequence, as dictated by Interleaved-2-of-5.
    Input:
        list bars: List of strings, i.e.:
            ['NB', 'NW', 'WB', 'WW', 'NW']
    Output:
        str decimal value.
    """
    return '0'
        
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
        
def fuzzy_eq(a, b, C=10e-04):
    return abs(a-b) <= C

def bestmatch(A, B):
    """ Tries to find the image A within the (larger) image B.
    For instance, A could be a voting target, and B could be a
    contest.
    Input:
        cvMat A: Patch to search for
        cvMat B: Image to search over
    Output:
        (x,y) location on B of the best match for A.
    """
    w_A, h_A = A.cols, A.rows
    w_B, h_B = B.cols, B.rows
    s_mat = cv.CreateMat(h_B - h_A + 1, w_B - w_A + 1, cv.CV_32F)
    cv.MatchTemplate(A, B, s_mat, cv.CV_TM_CCOEFF_NORMED)
    minResp, maxResp, minLoc, maxLoc = cv.MinMaxLoc(s_mat)
    return maxLoc

def get_tempmatches(A, B, T=0.85):
    """ Runs template matching, trying to find image A within image
    B. Returns location (and responses) of all matches greater than
    some threshold T.
    Input:
        cvMat A:
        cvMat B:
        float T:
    Output:
        list matches, i.e. [(i1, j1, float resp), ...]
    """
    w_A, h_A = A.cols, A.rows
    w_B, h_B = B.cols, B.rows
    s_mat = cv.CreateMat(h_B - h_A + 1, w_B - w_A + 1, cv.CV_32F)
    cv.MatchTemplate(A, B, s_mat, cv.CV_TM_CCOEFF_NORMED)
    # TODO: IMPLEMENT ME

def main():
    args = sys.argv[1:]
    imgpath = args[0]
    n = int(args[1])
    img = cv.LoadImageM(imgpath, cv.CV_LOAD_IMAGE_GRAYSCALE)
    t = time.time()
    print "Starting decode_i2of5..."
    decoded = decode_i2of5(img, n)
    dur = time.time() - t
    print "...Finished decode_i2of5 ({0} s)".format(dur)
    print "Decoded was:", decoded
    
if __name__ == '__main__':
    main()
