import os, sys, time, pdb
import numpy as np
import cv

TOP_GUARD_IMGP = 'hart_topguard.png'
BOT_GUARD_IMGP = 'hart_botguard.png'

def decode_i2of5(img):
    """ Decodes the interleaved two-of-five barcode. Returns a string.
    Input:
        cvMat img:
    Output:
        str decoded.
    """
    # For now, assume that the barcode is bottom-to-top.
    TOP_GUARD = cv.LoadImageM(TOP_GUARD_IMGP, cv.CV_LOAD_IMAGE_GRAYSCALE)
    BOT_GUARD = cv.LoadImageM(BOT_GUARD_IMGP, cv.CV_LOAD_IMAGE_GRAYSCALE)
    
    # 1.) Find location of top/bottom guards, to find location of barcode
    (itop, jtop) = bestmatch(TOP_GUARD, img)
    (ibot, jbot) = bestmatch(BOT_GUARD, img)
    
    # 2.) Crop+Straighten the barcode, so that it's totally horiz/vertical.

    # 3.) Collapse the barcode to be 1D (by summing pix intensity values
    # to the parallel axis). 
    flat = None

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
    w_narrow, w_wide = 0.0, 0.0
    
    # 5.) Convert the FLAT_NP to something like [Nblk, Nwht, Wblk, Wwht],
    # i.e. 'Narrow black', 'Wide white'. 
    bars = [] # [barstr_i, ...]
    i = 0
    # 5.a.) Advance to start of barcode
    foundbegin = False
    while i < len(flat_np) and not foundbegin:
        val = flat_np[i]
        if fuzzy_eq(val, pix_on):
            foundbegin = True
        else:
            i += 1
    if not foundbegin: 
        print "Uhoh, couldn't find start of barcode?"
        pdb.set_trace()
    # 6.) Interpret BARS.
    bars_blk, bars_wht = zip(*bars)
    decs_blk, decs_wht = [], []
    for bars in gen_by_n(blacks, 5):
        decs_blk.append(get_i2of5_val(bars))
    for bars in gen_by_n(whites, 5):
        decs_wht.append(get_i2of5_val(bars))
    decoded = zip(decs_blk, decs_wht)
    return decoded

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
