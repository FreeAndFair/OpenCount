import multiprocessing as mp
import pdb, os
import numpy as np
import cv, cv2
import csv
import string
import math
import imagesAlign as lk
import shared as sh
import time
import distance_transform
from scipy import misc
from matplotlib.pyplot import show, imshow, figure, title, colorbar, savefig, annotate

def dt1(f):
    n = f.size
    D = np.zeros(n)
    R = np.zeros(n)

    k = 0
    v = np.zeros(n)
    z = np.zeros(n+1)

    z[0] = -np.inf
    z[1] = np.inf

    for q in range(1,n-1):
        s1 = ((f[q] + pow(q,2)) - (f[v[k]] + pow(v[k],2)))/(2*q - 2*v[k])
        while s1 <= z[k]:
            k -= 1
            s1 = ((f[q] + pow(q,2)) - (f[v[k]] + pow(v[k],2)))/(2*q - 2*v[k])
        
        k += 1
        v[k] = q
        z[k] = s1
        z[k+1] = np.inf

    k = 1

    for q in range(n-1):
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
def pm1(digit_hash,I,nDigits,hspace,hackConstant=250,rejected_hash=None,accepted_hash=None):
    """
    Applies digit-OCR to an image.
    Input:
        dict digit_hash: maps {str digit: img digit_exemplar}
        obj I: image to search over (i.e. voted ballot)
        int nDigits: number of digits to find
        hspace: 
        hackConstant:
        dict rejected_hash: maps {str digit: [((y1,y2,x1,x2), str side_i, bool isflip_i), ...]}
        dict accepted_hash: maps {str digit: [((y1,y2,x1,x2), str side_i, bool isflip_i), ...]}
    """
    # either load previously computed results or compute new
    reject_penalty = .2
    accept_bonus = .2
    matchMat = []
    count = 0;
    keys = digit_hash.keys()
    t0=time.clock()    
    for key in keys:
        Iout = sh.NCC(I,digit_hash[key])
        #misc.imsave("_Iout_{0}.png".format(key), Iout)
        # mask out any part if given by param
        if rejected_hash and rejected_hash.has_key(key):
            for (bbMask, side, isflip) in rejected_hash[key]:
                # TODO: I don't ever use the 'side'. Is it worth removing it
                #       from rejected_hashes, or will it be used downstream?
                h = bbMask[1] - bbMask[0]
                w = bbMask[3] - bbMask[2]
                # Expand the mask-region a little bit
                i1 = max(0,bbMask[0]-(h/4))
                i2 = min(Iout.shape[0], bbMask[0]+(h/4))
                j1 = max(0,bbMask[2]-(w/4))
                j2 = min(Iout.shape[1],bbMask[2]+(w/4))
                Iout[i1:i2,j1:j2]=Iout[i1:i2,j1:j2]-reject_penalty
            #misc.imsave("_Iout_{0}_postmask.png".format(key), Iout)

        if accepted_hash and accepted_hash.has_key(key):
            for (bbMask, side, isflip) in accepted_hash[key]:
                # TODO: I don't ever use the 'side'. Is it worth removing it
                #       from rejected_hashes, or will it be used downstream?
                h = bbMask[1] - bbMask[0]
                w = bbMask[3] - bbMask[2]
                # Expand the mask-region a little bit
                i1 = max(0,bbMask[0]-(h/4))
                i2 = min(Iout.shape[0], bbMask[0]+(h/4))
                j1 = max(0,bbMask[2]-(w/4))
                j2 = min(Iout.shape[1],bbMask[2]+(w/4))
                Iout[i1:i2,j1:j2]=Iout[i1:i2,j1:j2]+accept_bonus
            #misc.imsave("_Iout_{0}_postmask.png".format(key), Iout)

        if len(matchMat) == 0:
            matchMat = np.zeros((Iout.shape[0],Iout.shape[1],len(keys)))

        matchMat[:,:,count] = Iout;
        count += 1

    print 'match time:',time.clock()-t0,'(s)'
    maxResp = np.amax(matchMat,axis=2)
    maxObj = np.argmax(matchMat,axis=2)

    tDP=time.clock()    
    # re-scale resp
    unary = hackConstant*np.power(2-(maxResp+1),2)

    #res = dt2(unary)
    res = distance_transform.dt2(unary)

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
        #t1=time.clock()    
        # old
        #res0 = dt2(prevT+unary) 
        #print 'old DP time:',time.clock()-t1,'(s)'

        # new cython implementation
        res = distance_transform.dt2(prevT+unary)
        #res = dt2(prevT+unary)
        #print 'diff = ', np.sum(np.abs(res0[0] - res[0]))
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
        #patches.append(P)
        patches.append(None)
        scores.append(maxResp[(i1,j1)])

    print 'DP time:',time.clock()-tDP,'(s)'
    return (ocr_str,patches,bbs,scores)

# partmatch
def pm2(digit_hash,I,nDigits,hspace,hackConstant=250,rejected_hash=None,accepted_hash=None):
    """
    Applies digit-OCR to an image.
    Input:
        dict digit_hash: maps {str digit: img digit_exemplar}
        obj I: image to search over (i.e. voted ballot)
        int nDigits: number of digits to find
        hspace: 
        hackConstant:
        dict rejected_hash: maps {str digit: [((y1,y2,x1,x2), str side_i, bool isflip_i), ...]}
        dict accepted_hash: maps {str digit: [((y1,y2,x1,x2), str side_i, bool isflip_i), ...]}
    """
    # either load previously computed results or compute new
    reject_penalty = .2
    accept_bonus = .2
    matchMat = []
    count = 0;
    keys = digit_hash.keys()
    t0=time.clock()    
    for key in keys:
        Iout = sh.NCC(I,digit_hash[key])
        symbol = key[0]
        meta = key[1]
        #misc.imsave("_Iout_{0}.png".format(key), Iout)
        # mask out any part if given by param
        if rejected_hash and rejected_hash.has_key(symbol):
            for (bbMask, side, isflip) in rejected_hash[symbol]:
                # TODO: I don't ever use the 'side'. Is it worth removing it
                #       from rejected_hashes, or will it be used downstream?
                h = bbMask[1] - bbMask[0]
                w = bbMask[3] - bbMask[2]
                # Expand the mask-region a little bit
                i1 = max(0,bbMask[0]-(h/4))
                i2 = min(Iout.shape[0], bbMask[0]+(h/4))
                j1 = max(0,bbMask[2]-(w/4))
                j2 = min(Iout.shape[1],bbMask[2]+(w/4))
                Iout[i1:i2,j1:j2]=Iout[i1:i2,j1:j2]-reject_penalty
            #misc.imsave("_Iout_{0}_postmask.png".format(key), Iout)

        if accepted_hash and accepted_hash.has_key(symbol):
            for (bbMask, side, isflip) in accepted_hash[symbol]:
                # TODO: I don't ever use the 'side'. Is it worth removing it
                #       from rejected_hashes, or will it be used downstream?
                h = bbMask[1] - bbMask[0]
                w = bbMask[3] - bbMask[2]
                # Expand the mask-region a little bit
                i1 = max(0,bbMask[0]-(h/4))
                i2 = min(Iout.shape[0], bbMask[0]+(h/4))
                j1 = max(0,bbMask[2]-(w/4))
                j2 = min(Iout.shape[1],bbMask[2]+(w/4))
                Iout[i1:i2,j1:j2]=Iout[i1:i2,j1:j2]+accept_bonus
            #misc.imsave("_Iout_{0}_postmask.png".format(key), Iout)

        if len(matchMat) == 0:
            matchMat = np.zeros((Iout.shape[0],Iout.shape[1],len(keys)))

        matchMat[:,:,count] = Iout;
        count += 1

    print 'match time:',time.clock()-t0,'(s)'
    maxResp = np.amax(matchMat,axis=2)
    maxObj = np.argmax(matchMat,axis=2)

    tDP=time.clock()    
    # re-scale resp
    unary = hackConstant*np.power(2-(maxResp+1),2)

    #res = dt2(unary)
    res = distance_transform.dt2(unary)

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
        #t1=time.clock()    
        # old
        #res0 = dt2(prevT+unary) 
        #print 'old DP time:',time.clock()-t1,'(s)'

        # new cython implementation
        res = distance_transform.dt2(prevT+unary)
        #res = dt2(prevT+unary)
        #print 'diff = ', np.sum(np.abs(res0[0] - res[0]))
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
    matched_keys = []
    for i in range(len(optYX)):
        (i1,j1)=optYX[i]

        key = keys[maxObj[(i1,j1)]]
        ocr_str += key[0]
        i2=i1+digit_hash[key].shape[0]
        j2=j1+digit_hash[key].shape[1]
        P = I[i1:i2,j1:j2]
        bbs.append((i1,i2,j1,j2))
        #patches.append(P)
        patches.append(None)
        scores.append(maxResp[(i1,j1)])
        matched_keys.append(key)

    print 'DP time:',time.clock()-tDP,'(s)'
    return (ocr_str,patches,bbs,scores,matched_keys)

def stackMax1(result_hash):
    maxSurf=np.zeros(1); symmax=-1;
    for key in result_hash.keys():
        out=result_hash[key]
        if out.max() > maxSurf.max():
            maxSurf = out
            symmax = key
            
    return (maxSurf,symmax)

def process_one(args):
    imP, digit_hash,imList,bbSearch,nDigits, do_flip, hspace, rejected_hashes,accepted_hashes = args
    I1 = sh.standardImread(imP,flatten=True)
    if do_flip == True:
        I1 = sh.fastFlip(I1)
    #I1=sh.prepOpenCV(I1)
    I1=I1[bbSearch[0]:bbSearch[1],bbSearch[2]:bbSearch[3]]
    rejected_hash = rejected_hashes.get(imP, None) if rejected_hashes else None
    accepted_hash = accepted_hashes.get(imP, None) if accepted_hashes else None
    # perform matching for all digits
    # return best matching digit
    # mask out 
    res = pm2(digit_hash,I1,nDigits,hspace,rejected_hash=rejected_hash,accepted_hash=accepted_hash)
    #res = pm1(digit_hash,I1,nDigits,hspace,rejected_hash=rejected_hash,accepted_hash=accepted_hash)

    return (imP,res[0],res[1],res[2],res[3])

def digitParse(digit_hash,imList,bbSearch,nDigits, do_flip=False, hspace=20,
               rejected_hashes=None, accepted_hashes=None):
    """Runs NCC-based OCR on the images on imList.
    Input:
        dict digit_hash: maps {(str digit, str meta): img digit_exemplar}
        lst imList: list of imagepaths to search over
        bbSearch: [y1,y2,x1,x2] coords to search on
        nDigits: an integer that specifies how many digits there are.
        do_flip: If True, then flip the image.
        dict rejected_hashes: Contains all user rejections for each image,
                              maps:
                                {imgpath: {str digit: [((y1,y2,x1,x2),str side_i,bool isflip_i), ...]}}
        dict accepted_hashes: Contains all user accepts for each image,
                              maps:
                                {imgpath: {str digit: [((y1,y2,x1,x2),str side_i,bool isflip_i), ...]}}
    Output:
        A list of results of the form:
            [(imgpath_i, ocr_str_i, imgpatches_i, patchcoords_i, scores_i), ... ]
    """
    digitList = digit_hash.values();
    patchExample = digitList[0]

    nProc=sh.numProcs()

    if nProc < 2:
        results = []
        for x in imList:
            results.append(process_one((x,digit_hash,imList,bbSearch,nDigits,do_flip,hspace,rejected_hashes,accepted_hashes)))
    else:
        pool = mp.Pool(processes=nProc)
        results = pool.map(process_one, [(x,digit_hash,imList,bbSearch,nDigits, do_flip, hspace, rejected_hashes,accepted_hashes) for x in  imList])
        pool.close()
        pool.join()

    return results


