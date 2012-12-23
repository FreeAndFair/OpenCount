import traceback
import cv, numpy as np

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

def bestmatch(A, imgpaths, img2flip=None, do_smooth=0, xwinA=3, ywinA=3, 
              xwinI=3, ywinI=3, prevmatches=None, jobid=None):
    """ Runs template matching on IMGPATHS, searching for best match
    for A. 
    Input:
        A: Either a string (path), or an IplImage.
        list IMGPATHS: List of imgpaths to search over
        dict IMG2FLIP: maps {str imgpath: bool isflipped}
        int DO_SMOOTH:
        dict  PREVMATCHES: {imgpath: [(x_i, y_i), ...]}. Matches to ignore.
    Output:
        dict {str IMGPATH: (x1, y1, float score)}.
    """
    if type(A) in (str, unicode):
        A_im = cv.LoadImage(A, cv.CV_LOAD_IMAGE_GRAYSCALE)
    else:
        A_im = A
    if do_smooth == SMOOTH_BOTH_BRD or do_smooth == SMOOTH_A_BRD:
        A_im = smooth(A_im, xwinA, ywinA, bordertype='const', val=255)
    elif do_smooth in (SMOOTH_BOTH, SMOOTH_A):
        A_im = smooth(A_im, xwinA, ywinA)
    w_A, h_A = cv.GetSize(A_im)
    results = {}
    for i, imgpath in enumerate(imgpaths):
        if type(imgpath) in (str, unicode):
            I = cv.LoadImage(imgpath, cv.CV_LOAD_IMAGE_GRAYSCALE)
        else:
            I = imgpath
            imgpath = i
        if do_smooth in (SMOOTH_BOTH_BRD, SMOOTH_IMG_BRD):
            I = smooth(I, xwinI, ywinI, bordertype='const', val=255)
        elif do_smooth in (SMOOTH_BOTH, SMOOTH_IMG):
            I = smooth(I, xwinI, ywinI)
        if img2flip and img2flip[imgpath]:
            cv.Flip(I, I, flipMode=-1)
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

    #print 'results:', results
    return results

def _do_bestmatch(imgpaths, (A_str, img2flip, do_smooth, xwinA, ywinA, xwinI, ywinI, w, h, prevmatches, jobid)):
    A_cv = cv.CreateImageHeader((w,h), cv.IPL_DEPTH_8U, 1)
    cv.SetData(A_cv, A_str)
    try:
        result = bestmatch(A_cv, imgpaths, img2flip=img2flip, do_smooth=do_smooth, xwinA=xwinA,
                           ywinA=ywinA, xwinI=xwinI, ywinI=ywinI, prevmatches=prevmatches,
                           jobid=jobid)
        return result
    except:
        traceback.print_exc()
        return {}

def bestmatch_par(A, imgpaths, img2flip=None, NP=None, do_smooth=0, xwinA=3, ywinA=3,
                  xwinI=3, ywinI=3, prevmatches=None, jobid=None):
    """ Find the best match for A in each image in IMGPATHS, using NP
    processes. A multiprocessing-wrapper for bestmatch (see doc for
    bestmatch for more details).
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
                                _args=(A_str, img2flip, do_smooth, xwinA, ywinA,
                                       xwinI, ywinI, w, h, prevmatches, jobid),
                                combfn='dict', N=None)
    if jobid and wx.App.IsMainLoopRunning():
        wx.CallAfter(Publisher().sendMessage, "signals.MyGauge.done", (jobid,))
    return result

def get_tempmatches(A, imgpaths, img2flip=None, T=0.8, do_smooth=0, xwinA=13, ywinA=13,
                    xwinI=13, ywinI=13, MAX_MATS=50, prevmatches=None,
                    atleastone=False, jobid=None):
    """ Runs template matching, trying to find image A within each image
    in IMGPATHS. Returns location (and responses) of all matches greater than
    some threshold T.
    Input:
        IplImage A:
        list IMGPATHS:
        dict IMG2FLIP: maps {str imgpath: bool isflipped}
        float T:
        dict PREVMATCHES: maps {str imgpath: [(x1,y1,x2,y2), ...]}
    Output:
        dict MATCHES, of the form {str imgpath: [(x1, y1, x2, y2, float score), ...]}
    """
    if do_smooth == SMOOTH_BOTH_BRD or do_smooth == SMOOTH_A_BRD:
        A_im = smooth(A, xwinA, ywinA, bordertype='const', val=255)
    elif do_smooth in (SMOOTH_BOTH, SMOOTH_A):
        A_im = smooth(A, xwinA, ywinA)
    else:
        A_im = A
    wA, hA = cv.GetSize(A_im)
    results = {} # {str imgpath: [(x1,y1,x2,y2,score),...]}
    for i,imgpath in enumerate(imgpaths):
        if isinstance(imgpath, str) or isinstance(imgpath, unicode):
            I = cv.LoadImage(imgpath, cv.CV_LOAD_IMAGE_GRAYSCALE)
        else:
            I = imgpath
            imgpath = i
        if do_smooth in (SMOOTH_BOTH_BRD, SMOOTH_IMG_BRD):
            I = smooth(I, xwinI, ywinI, bordertype='const', val=255)
        elif do_smooth in (SMOOTH_BOTH, SMOOTH_IMG):
            I = smooth(I, xwinI, ywinI)
        if img2flip and img2flip[imgpath]:
            cv.Flip(I, I, flipMode=-1)
        wI, hI = cv.GetSize(I)
        M = cv.CreateMat(hI-hA+1, wI-wA+1, cv.CV_32F)
        cv.MatchTemplate(I, A_im, M, cv.CV_TM_CCOEFF_NORMED)
        M_np = np.array(M)
        # 0.) Suppress previously-found matches, if any
        prevmats = prevmatches.get(imgpath, []) if prevmatches else []
        for (x1,y1,x2,y2) in prevmats:
            #print 'suppressing: {0} at {1}'.format(imgpath, (x1, y1))
            _x1 = max(0, int(x1 - (wA / 2)))
            _y1 = max(0, int(y1 - (hA / 2)))
            _x2 = min(M_np.shape[1], int(x1 + (wA / 2)))
            _y2 = min(M_np.shape[0], int(y1 + (hA / 2)))
            M_np[_y1:_y2, _x1:_x2] = -1.0
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
            matches.append((j, i, j+wA, i+hA, score))
            # Suppression
            _x1 = max(0, int(j - (wA / 2)))
            _y1 = max(0, int(i - (hA / 2)))
            _x2 = min(M_np.shape[1], int(j + (wA / 2)))
            _y2 = min(M_np.shape[0], int(i + (hA / 2)))
            M_np[_y1:_y2, _x1:_x2] = -1.0
            #M_np[i-(hA/2):i+(hA/2),
            #     j-(wA/2):j+(wA/2)] = -1.0
            num_mats += 1
        if not matches and atleastone:
            print 'DOO DOO DOO'
            M_idx = np.argmax(M_np)
            i = int(M_idx / M.cols)
            j = M_idx % M.cols
            score = M_np[i,j]
            matches.append((j, i, j + wA, i + hA, score))
        results[imgpath] = matches
        if jobid and wx.App.IsMainLoopRunning():
            wx.CallAfter(Publisher().sendMessage, "signals.MyGauge.tick", (jobid,))
    return results

def _do_get_tempmatches(imgpaths, (A_str, img2flip, T, do_smooth, xwinA, ywinA,
                                   xwinI, ywinI, w, h, MAX_MATS, prevmatches, 
                                   atleastone, jobid)):
    result = {}
    A = cv.CreateImageHeader((w,h), cv.IPL_DEPTH_8U, 1)
    cv.SetData(A, A_str)
    try:
        results = get_tempmatches(A, imgpaths, img2flip=img2flip, T=T, do_smooth=do_smooth, xwinA=xwinA, 
                                  ywinA=ywinA, xwinI=xwinI, ywinI=ywinI, MAX_MATS=MAX_MATS,
                                  prevmatches=prevmatches,
                                  jobid=jobid)
    except:
        traceback.print_exc()
        return {}
    return results

def get_tempmatches_par(A, imgpaths, img2flip=None, T=0.8, do_smooth=0, 
                        xwinA=13, ywinA=13, xwinI=13, ywinI=13,
                        MAX_MATS=50, prevmatches=None,
                        atleastone=False, NP=None, jobid=None):
    """ For each img in IMGPATHS, template match for A, using NP processes.
    A multiprocessing wrapper for get_tempmatches (see doc for get_tempmatches
    for more details).
    Input:
        IplImage A:
        lst IMGPATHS:
        int NP: Number of processes, or None for auto.
        
    Output:
        dict MATCHES, of the form {str imgpath: [(x1, y1, x2, y2 float resp), ...]}
    """
    A_str = A.tostring()
    w, h = cv.GetSize(A)
    try:
        result = partask.do_partask(_do_get_tempmatches, imgpaths,
                                    _args=(A_str, img2flip, T, do_smooth, xwinA, ywinA,
                                           xwinI, ywinI, w, h, MAX_MATS, prevmatches,
                                           atleastone, jobid),
                                    combfn='dict', singleproc=False)
    except Exception as e:
        traceback.print_exc()
        return {}
    if jobid and wx.App.IsMainLoopRunning():
        wx.CallAfter(Publisher().sendMessage, "signals.MyGauge.done", (jobid,))
    return result    

def smooth(I, xwin, ywin, bordertype=None, val=255.0):
    """ Apply a gaussian blur to I, with window size [XWIN,YWIN].
    If BORDERTYPE is 'const', then treat pixels that lie outside of I as
    VAL (rather than what OpenCV defaults to).
    Input:
        IplImage I:
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

def smooth_mat(Imat, xwin, ywin, bordertype=None, val=255.0):
    """ Apply a gaussian blur to IMAT, with window size [XWIN,YWIN].
    If BORDERTYPE is 'const', then treat pixels that lie outside of I as
    VAL (rather than what OpenCV defaults to).
    Input:
        cvMat IMAT:
    """
    '''
    w, h = Imat.cols, Imat.rows
    if bordertype == 'const':
        #Ibig = cv.CreateImage((w+2*xwin, h+2*ywin), I.depth, I.channels)
        Ibig = cv.CreateMat(h+2*ywin, w+2*xwin, Imat.type)
        
        cv.CopyMakeBorder(Imat, Ibig, (xwin, ywin), 0, value=val)
        cv.SetImageROI(Ibig, (xwin, ywin, w, h))
    else:
        Ibig = Imat
    #Iout = cv.CreateImage((w,h), I.depth, I.channels)
    Iout = cv.CreateMat(h, w, I.type)
    cv.Smooth(Ibig, Iout, cv.CV_GAUSSIAN, param1=xwin, param2=ywin)
    return Iout
    '''
    return cv.GetMat(smooth(cv.GetImage(Imat), xwin, ywin, bordertype=bordertype, val=val))
    
