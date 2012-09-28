import os, sys, time, pdb
import cv

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

def main():
    targetpath = 'target_oc.png'
    contestpath = 'contest_oc.png'

    # 1.) Load images via OpenCV. I use cv.LoadImageM instead of
    # cv.LoadImage, because I personally find it more convenient to
    # work with cvMat's (which cv.LoadImageM returns). cv.LoadImage,
    # on the other hand, returns IplImage instances.
    Itarget = cv.LoadImageM(targetpath, cv.CV_LOAD_IMAGE_GRAYSCALE)
    Icontest = cv.LoadImageM(contestpath, cv.CV_LOAD_IMAGE_GRAYSCALE)
    
    # 2.) Run template matching
    (x, y) = bestmatch(Itarget, Icontest)
    
    # 3.) Mark on the image where template matching found the target,
    # and save it to the current directory.
    Icolor = cv.LoadImageM(contestpath, cv.CV_LOAD_IMAGE_COLOR)
    cv.Circle(Icolor, (x, y), 15, cv.RGB(0, 60, 255), thickness=4)
    cv.SaveImage("bestmatch_result.png", Icolor)

    print "Done. Saved graphical output of template matching to: bestmatch_result.png."

if __name__ == '__main__':
    main()

    
