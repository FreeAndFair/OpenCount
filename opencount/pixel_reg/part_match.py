import multiprocessing as mp
import pdb, os
import numpy as np
import cv, cv2
import csv
import string
import math
import imagesAlign as lk
import shared as sh
from scipy import misc
from matplotlib.pyplot import show, imshow, figure, title, colorbar, savefig, annotate

def dt1(f):
    n = f.size
    D = np.zeros(n)
    R = np.zeros(n)

    k = 0
    v = np.ones(n+1)
    z = np.ones(n+1)

    z[0] = -np.inf
    z[1] = np.inf

    for q in range(1,n):
        s1 = ((f[q] + pow(q,2)) - (f[v[k]] + pow(v[k],2)))/(2*q - 2*v[k])
        while s1 <= z[k]:
            k -= 1
            s1 = ((f[q] + pow(q,2)) - (f[v[k]] + pow(v[k],2)))/(2*q - 2*v[k])
        
        k += 1
        v[k] = q
        z[k] = s1
        z[k+1] = np.inf

    k = 1

    for q in range(n):
        while z[k+1] < q:
            k += 1
        D[q] = pow((q - v[k]),2) + f[v[k]]
        R[q] = v[k]
        
    return (D,R)

def dt2(I):
    res = np.zeros(I.shape)
    Rx = np.zeros(I.shape)
    Ry = np.zeros(I.shape)
    for i in range(I.shape[0]):
        (D,x) = dt1(I[i,:])
        res[i,:] = D
        Rx[i,:] = x

    for i in range(I.shape[1]):
        (D,y) = dt1(res[:,i])
        res[:,i] = D
        Ry[:,i] = y

    return (res,Rx,Ry)

# partmatch
def pm(digit_hash,I,nDigits,hspace, hackConstant=250):
    # TODO: check if user has accepted/rejected any positions
 
    # either load previously computed results or compute new
    
    matchMat = []
    count = 0;
    keys = digit_hash.keys()
    for key in keys:
        patch = sh.prepOpenCV(digit_hash[key]);
        patchCv=cv.fromarray(np.copy(patch))
        ICv=cv.fromarray(np.copy(I))
        outCv=cv.CreateMat(I.shape[0]-patch.shape[0]+1,I.shape[1]-patch.shape[1]+1,patchCv.type)
        cv.MatchTemplate(ICv,patchCv,outCv,cv.CV_TM_CCOEFF_NORMED)
        Iout=np.asarray(outCv)
        Iout[Iout==1.0]=0; # opencv bug
        if len(matchMat) == 0:
            matchMat = np.zeros((Iout.shape[0],Iout.shape[1],len(keys)))

        matchMat[:,:,count] = Iout;
        count += 1

    maxResp = np.amax(matchMat,axis=2)
    maxObj = np.argmax(matchMat,axis=2)

    # re-scale resp
    unary = hackConstant*np.power(2-(maxResp+1),2)
    res = dt2(unary)
    # cache bottom up
    M = [[]]*nDigits; 
    Mx = [[]]*nDigits; 
    My = [[]]*nDigits; 
    M[0] = res[0]
    Mx[0] = res[1]
    My[0] = res[2]

    for i in range(1,nDigits-1):
        prev = M[i-1]
        shiftH = np.eye(3)
        shiftH[0,2] = hspace
        prevT = lk.imtransform(prev,shiftH,fillval=prev.max());
        # shift
        res = dt2(prevT+unary)
        M[i] = res[0]
        Mx[i] = res[1]
        My[i] = res[2]

    prev = M[nDigits-2]
    shiftH = np.eye(3)
    shiftH[0,2] = hspace
    prevT = lk.imtransform(prev,shiftH,fillval=prev.max());
    M[nDigits-1] = prevT+unary
    # get best root position
    rootM = M[nDigits-1]
    YX=np.unravel_index(rootM.argmin(),rootM.shape)
    miny=YX[0]; minx=YX[1];

    # store top down
    optYX = [[]]*nDigits; 
    optYX[nDigits-1] = (miny,minx)
    
    for i in reversed(range(0,nDigits-1)):
        prevMiny = optYX[i+1][0]
        prevMinx = optYX[i+1][1]
        curMx = Mx[i]
        curMy = My[i]
        optYX[i] = (round(curMy[prevMiny,prevMinx-hspace]),
                    round(curMx[prevMiny,prevMinx-hspace]))
        
    patches = []
    bbs = []
    scores = []
    ocr_str = ''
    for i in range(len(optYX)):
        (i1,j1)=optYX[i]

        key = keys[maxObj[(i1,j1)]]
        ocr_str += key
        i2=i1+digit_hash[key].shape[0]
        j2=j1+digit_hash[key].shape[1]
        P = I[i1:i2,j1:j2]
        bbs.append((i1,i2,j1,j2))
        patches.append(P)
        scores.append(maxResp[(i1,j1)])

    return (ocr_str,patches,bbs,scores)

def stackMax1(result_hash):
    pdb.set_trace()
    maxSurf=np.zeros(1); symmax=-1;
    for key in result_hash.keys():
        out=result_hash[key]
        if out.max() > maxSurf.max():
            maxSurf = out
            symmax = key
            
    return (maxSurf,symmax)

def digitParse(digit_hash,imList,bbSearch,nDigits, do_flip=False, hspace=20):
    """Runs NCC-based OCR on the images on imList.
    Input:
        dict digit_hash: maps {str digit: img digit_exemplar}
        lst imList: list of imagepaths to search over
        bbSearch: [y1,y2,x1,x2] coords to search on
        nDigits: an integer that specifies how many digits there are.
        do_flip: If True, then flip the image.
    Output:
        A list of results of the form:
            [(imgpath_i, ocr_str_i, res_meta_i), ...)
        where res_meta_i is a tuple of len nDigits, containing:
            (y1,y2,x1,x2, str digit, obj digitimg, float score)
    """
    digitList = digit_hash.values();
    patchExample = digitList[0]

    results = []

    for imP in imList:
        I1 = sh.standardImread(imP,flatten=True)
        if do_flip == True:
            I1 = fastFlip(I1)
        I1=sh.prepOpenCV(I1)
        I1=I1[bbSearch[0]:bbSearch[1],bbSearch[2]:bbSearch[3]]
        # perform matching for all digits
        # return best matching digit
        # mask out 
        res = pm(digit_hash,I1,nDigits,hspace)
        results.append((imP,res[0],res[1],res[2]))

    return results
