import sys, os, time, pdb, traceback
import cv

from wx.lib.pubsub import Publisher
import wx

import partask

SMOOTH_NONE = 0
SMOOTH_IMG = 1
SMOOTH_A = 2
SMOOTH_BOTH = 3

SMOOTH_IMG_BRD = 4
SMOOTH_A_BRD = 5
SMOOTH_BOTH_BRD = 6

def bestmatch(A, imgpaths, do_smooth=0, xwinA=3, ywinA=3, 
              xwinI=3, ywinI=3, prevmatches=None, jobid=None):
    """ Runs template matching on IMGPATHS, searching for best match
    for A. 
    Input:
        A: Either a string (path), or an IplImage.
        list IMGPATHS: List of imgpaths to search over
        int DO_SMOOTH:
        dict  PREVMATCHES: {imgpath: [(x_i, y_i), ...]}. Matches to ignore.
    Output:
        dict {str IMGPATH: (x1, y1, float score)}.
    """
    if type(A) == str:
        A_im = cv.LoadImage(A, cv.CV_LOAD_IMAGE_GRAYSCALE)
    else:
        A_im = A
    if do_smooth == SMOOTH_BOTH_BRD or do_smooth == SMOOTH_A_BRD:
        A_im = smooth(A_im, xwinA, ywinA, bordertype='const', val=255)
    elif do_smooth in (SMOOTH_BOTH, SMOOTH_A):
        A_im = smooth(A_im, xwinA, ywinA)
    w_A, h_A = cv.GetSize(A_im)
    results = {}
    for imgpath in imgpaths:
        I = cv.LoadImage(imgpath, cv.CV_LOAD_IMAGE_GRAYSCALE)
        if do_smooth in (SMOOTH_BOTH_BRD, SMOOTH_IMG_BRD):
            I = smooth(I, xwinI, ywinI, bordertype='const', val=255)
        elif do_smooth in (SMOOTH_BOTH, SMOOTH_IMG):
            I = smooth(I, xwinI, ywinI)
        w_I, h_I = cv.GetSize(I)
        matchmat = cv.CreateMat(h_I-h_A+1, w_I-w_A+1, cv.CV_32F)
        cv.MatchTemplate(I, A_im, matchmat, cv.CV_TM_CCOEFF_NORMED)
        # 0.) Suppress previously-found matches, if any
        prevmats = prevmatches.get(imgpath, []) if prevmatches else []
        for (x,y) in prevmats:
            print 'suppressing: {0} at {1}'.format(imgpath, (x, y))
            _x1 = max(0, int(x - (w_A / 3)))
            _y1 = max(0, int(y - (h_A / 3)))
            _x2 = min(matchmat.cols, int(x + (w_A / 3)))
            _y2 = min(matchmat.rows, int(y + (h_A / 3)))
            matchmat[_y1:_y2, _x1:_x2] = -1.0
        minResp, maxResp, minLoc, maxLoc = cv.MinMaxLoc(matchmat)
        results[imgpath] = (maxLoc[0], maxLoc[1], maxResp)
        if jobid and wx.App.IsMainLoopRunning():    
            wx.CallAfter(Publisher().sendMessage, "signals.MyGauge.tick", (jobid,))

    print 'results:', results
    return results

def _do_bestmatch(imgpaths, (A_str, do_smooth, xwinA, ywinA, xwinI, ywinI, w, h, prevmatches, jobid)):
    A_cv = cv.CreateImageHeader((w,h), cv.IPL_DEPTH_8U, 1)
    cv.SetData(A_cv, A_str)
    try:
        result = bestmatch(A_cv, imgpaths, do_smooth=do_smooth, xwinA=xwinA,
                           ywinA=ywinA, xwinI=xwinI, ywinI=ywinI, prevmatches=prevmatches,
                           jobid=jobid)
        return result
    except:
        traceback.print_exc()
        return {}

def bestmatch_par(A, imgpaths, NP=None, do_smooth=0, xwinA=3, ywinA=3,
                  xwinI=3, ywinI=3, prevmatches=None, jobid=None):
    """
    Input:
        IplImage A:
        list IMGPATHS:
        int NP: Number of processors to use.
        ...
        list PREVMATCHES: [(x_i, y_i), ...]. Previously-found matches
            that bestmatch should ignore.
    Output:
        List MATCHES. [(x_i, y_i, float score_i), ...]
    """
    A_str = A.tostring()
    w, h = cv.GetSize(A)
    result = partask.do_partask(_do_bestmatch, imgpaths,
                                _args=(A_str, do_smooth, xwinA, ywinA,
                                       xwinI, ywinI, w, h, prevmatches, jobid),
                                combfn='dict', singleproc=True)
    if jobid and wx.App.IsMainLoopRunning():
        wx.CallAfter(Publisher().sendMessage, "signals.MyGauge.done", (jobid,))
    return result

def smooth(I, xwin, ywin, bordertype=None, val=255.0):
    """ Apply a gaussian blur to I, with window size [XWIN,YWIN].
    If BORDERTYPE is 'const', then treat pixels that lie outside of I as
    VAL (rather than what OpenCV defaults to).
    """
    w, h = cv.GetSize(I)
    if bordertype == 'const':
        Ibig = cv.CreateImage((w+2*xwin, h+2*ywin), I.depth, I.channels)
        cv.CopyMakeBorder(I, Ibig, (xwin, ywin), 0, value=val)
        cv.SetImageROI(Ibig, (xwin, ywin, w, h))
    else:
        Ibig = I
    Iout = cv.CreateImage((w,h), I.depth, I.channels)
    cv.Smooth(Ibig, Iout, cv.CV_GAUSSIAN, param1=xwin, param2=ywin)
    return Iout

              
