from os.path import join as pathjoin
from PIL import Image
from scipy import misc,ndimage
import shared as sh
from imagesAlign import *
import cProfile
import cv
import csv
import traceback
import os
import string
import sys, shutil, traceback
import multiprocessing as mp
from wx.lib.pubsub import Publisher
import json
import wx
import pickle
import time
import random
import grouping.tempmatch as tempmatch
from util import get_filename, encodepath


def doWrite(finalOrder, Ip, err, attrName, patchDir, metaDir, origfullpath):
    fullpath = encodepath(origfullpath)

    # - patchDir: write out Ip into patchDir
    Ip[np.isnan(Ip)]=1
    to = os.path.join(patchDir, fullpath + '.png')
    sh.imsave(to,Ip)

    # - metaDir:
    # loop over finalOrder and compile array of group IDs and flip/nonflip
    attrOrder=[]; flipOrder=[];
    for pt in finalOrder:
        # pt[2] := (str temppath, str attrval)
        attrOrder.append(pt[2][1])
        flipOrder.append(pt[3])
    
    to = os.path.join(metaDir, fullpath)
    toWrite={"attrOrder": attrOrder, "flipOrder":flipOrder,"err":err}
    file = open(to, "wb")
    pickle.dump(toWrite, file)
    file.close()

def doWriteMAP(finalOrder, Ip, err, attrName, patchDir, metaDir, balKey, exemplar_idx):
    """
    Input:
        ...
        int exemplar_idx: Which exemplar attrpatch did this match to?
    """
    # TODO: For now, just use the int ballotID itself, in a flat
    # dirstruct manner.
    #fullpath = encodepath(balKey)
    fullpath = str(balKey)

    # - patchDir: write out Ip into patchDir
    Ip[np.isnan(Ip)]=1
    to = os.path.join(patchDir, fullpath + '.png')
    misc.imsave(to,Ip)

    # - metaDir:
    # loop over finalOrder and compile array of group IDs and flip/nonflip
    attrOrder=[]; imageOrder=[]; flipOrder=[];
    for pt in finalOrder:
        # pt[2] := (str temppath, str attrval)
        attrOrder.append(pt[2])
        imageOrder.append(pt[3])
        flipOrder.append(pt[4])
    
    to = os.path.join(metaDir, fullpath)
    toWrite={"attrOrder": attrOrder, "flipOrder":flipOrder,"err":err,"imageOrder":imageOrder,
             "exemplar_idx": exemplar_idx}
    file = open(to, "wb")
    pickle.dump(toWrite, file)
    file.close()

def evalPatchSimilarity(I,patch, debug=False):
    # perform template matching and return the best match in expanded region
    I_in = np.copy(I)
    patch_in = np.copy(patch)

    if debug == True:
        print "...stepping into evalPatchSimilarity."
        pdb.set_trace()

    I=sh.prepOpenCV(I)
    patch=sh.prepOpenCV(patch)
    # See pixel_reg/eric_np2cv/demo.py for why I scale by 255.0 when 
    # converting NP -> OpenCV.
    patchCv=cv.fromarray(np.copy(patch) * 255.0)
    ICv=cv.fromarray(np.copy(I) * 255.0)
    #patchCv=cv.fromarray(np.copy(patch))  
    #ICv=cv.fromarray(np.copy(I))
    
    # call template match
    outCv=cv.CreateMat(I.shape[0]-patch.shape[0]+1,I.shape[1]-patch.shape[1]+1,patchCv.type)
    cv.MatchTemplate(ICv,patchCv,outCv,cv.CV_TM_CCOEFF_NORMED)
    Iout=np.asarray(outCv) / 255.0
    #Iout=np.asarray(outCv)
    Iout[Iout==1.0]=0;
    YX=np.unravel_index(Iout.argmax(),Iout.shape)

    # local alignment: expand a little, then local align
    i1=YX[0]; i2=YX[0]+patch.shape[0]
    j1=YX[1]; j2=YX[1]+patch.shape[1]
    I1c=I[i1:i2,j1:j2]
    IO=imagesAlign(I1c,patch,type='rigid', minArea=np.power(2, 20))

    Ireg=IO[1]
    #Ireg = np.nan_to_num(Ireg)
    # TODO: Ireg is frequently just a competely-black image (due to
    # presence of Nan's?). By inserting the line:
    #     Ireg = np.nan_to_num(Ireg)
    # This stopped an apparent bug in Marin, where various attribute
    # patches would erroneously be matched to the wrong side of the
    # ballot.

    # C := num pixels to discard around border. This used to be C=5,
    #      but this caused issues if one of the 'patch' dimensions was
    #      <= 10, causing an ill-formed image patch.
    AMT = 0.2
    C_i = int(round(Ireg.shape[0] * AMT))
    C_j = int(round(Ireg.shape[1] * AMT))
    D_i = int(round(patch.shape[0] * AMT))
    D_j = int(round(patch.shape[1] * AMT))
    bb_1 = [C_i, 
            min(Ireg.shape[0]-1, Ireg.shape[0]-C_i),
            C_j, 
            min(Ireg.shape[1]-1, Ireg.shape[1]-C_j)]
    bb_2 = [D_i,
            min(patch.shape[0]-1, patch.shape[0]-D_i),
            D_j,
            min(patch.shape[1]-1, patch.shape[1]-D_j)]
    if bb_1[0] >= bb_1[1]:
        bb_1[0] = 0
    if bb_1[1] <= 2:
        bb_1[1] = Ireg.shape[0]-1
    if bb_1[2] >= bb_1[3]:
        bb_1[2] = 0
    if bb_1[3] <= 2:
        bb_1[3] = Ireg.shape[1]-1

    if bb_2[0] >= bb_2[1]:
        bb_2[0] = 0
    if bb_2[1] <= 2:
        bb_2[1] = patch.shape[0]-1
    if bb_2[2] >= bb_2[3]:
        bb_2[2] = 0
    if bb_2[3] <= 2:
        bb_2[3] = patch.shape[1]-1
    #Ireg1=Ireg[C:Ireg.shape[0]-C,C:Ireg.shape[1]-C]
    #patch1=patch[C:patch.shape[0]-C,C:patch.shape[1]-C]
    Ireg1 = Ireg[bb_1[0]:bb_1[1], bb_1[2]:bb_1[3]]
    patch1 = patch[bb_2[0]:bb_2[1], bb_2[2]:bb_2[3]]

    if 0 in Ireg1.shape or 0 in patch1.shape:
        print "==== Uhoh, a crash is about to happen."
        print "Ireg.shape: {0}  patch.shape: {1}".format(Ireg.shape, patch.shape)
        print "Ireg1.shape: {0}  patch1.shape: {1}".format(Ireg1.shape, patch1.shape)
        print "bb_1:", bb_1
        print "bb_2:", bb_2
        misc.imsave("_evalpatchsim_ireg.png", Ireg)
        misc.imsave("_evalpatchsim_patch.png", patch)
        misc.imsave("_evalpatchsim_I1c.png", I1c)
        misc.imsave("_evalpatchsim_I.png", I)

    err = sh.variableDiffThr(Ireg1,patch1)
    diff=np.abs(Ireg1-patch1);
    # #estimate threshold for comparison: 

    return (-err,YX,diff)
    
def evalPatchSimilarity2(I,patch, debug=False):
    # perform template matching and return the best match in expanded region
    I_in = np.copy(I)
    patch_in = np.copy(patch)

    I=sh.prepOpenCV(I)
    patch=sh.prepOpenCV(patch)
    # See pixel_reg/eric_np2cv/demo.py for why I scale by 255.0 when 
    # converting NP -> OpenCV.
    patchCv=cv.fromarray(np.copy(patch) * 255.0)
    ICv=cv.fromarray(np.copy(I) * 255.0)
    #cv.SaveImage("_patchCv.png", patchCv)
    #cv.SaveImage("_ICv.png", ICv)
    #patchCv = tempmatch.smooth_mat(patchCv, 5, 5, bordertype='const', val=255)
    #ICv = tempmatch.smooth_mat(ICv, 5, 5, bordertype='const', val=255)
    #cv.SaveImage("_patchCv_smooth.png", patchCv)
    #cv.SaveImage("_ICv_smooth.png", ICv)
    #pdb.set_trace()
    # call template match
    outCv=cv.CreateMat(I.shape[0]-patch.shape[0]+1,
                       I.shape[1]-patch.shape[1]+1,patchCv.type)
    cv.MatchTemplate(ICv,patchCv,outCv,cv.CV_TM_CCOEFF_NORMED)
    #Iout=np.asarray(outCv) / 255.0
    Iout = np.asarray(outCv)
    YX=np.unravel_index(Iout.argmax(),Iout.shape)

    # take result and expand in each dimension by pixPad
    i1=YX[0]; i2=YX[0]+patch.shape[0]
    j1=YX[1]; j2=YX[1]+patch.shape[1]

    # [new] patch pad 25%
    # if the result has any nans then we'll call that bad, 
    # undo the transformation and compute error wrt original 
    # inputs.

    pixPad = round(.25 * min(patch.shape));
    patchPad = np.empty((patch.shape[0]+2*pixPad,
                         patch.shape[1]+2*pixPad))
    patchPad[:] = np.nan
    patchPad[pixPad:patch.shape[0]+pixPad,
             pixPad:patch.shape[1]+pixPad] = patch

    # check how much we need to pad in each dimension
    # if any values are positive, that means we need to pad
    # the difference
    i1pad = i1 - pixPad;
    i1exp = max(0,-i1pad);
    i1pad = (i1 - pixPad) + i1exp;

    i2pad = i2 + pixPad;
    i2exp = max(0,i2pad-I.shape[0]);
    i2pad = (i2 + pixPad) - i2exp;

    j1pad = j1 - pixPad;
    j1exp = max(0,-j1pad);
    j1pad = (j1 - pixPad) + j1exp;

    j2pad = j2 + pixPad;
    j2exp = max(0,j2pad-I.shape[1]);
    j2pad = (j2 + pixPad) - j2exp;

    # crop out padded patch
    I1c=I[i1pad:i2pad,j1pad:j2pad]

    # check for padding
    if (i1exp+i2exp+j1exp+j2exp) > 0:
        I1c = sh.padWithBorderHandling(I1c,i1exp,i2exp,j1exp,j2exp)

    # expand if necessary
    #hCells = max(int(round(I1c.shape[1] / 200)), 1)
    #vCells = max(int(round(I1c.shape[0] / 200)), 1)
    hCells, vCells = 1, 1
    
    IO=imagesAlign(I1c,patchPad,type='rigid',hCells=hCells, vCells=vCells, minArea=np.power(2, 15))
    if debug:
        pdb.set_trace()
    Ireg=IO[1]
    Ireg = Ireg[pixPad:patch.shape[0]+pixPad,
                pixPad:patch.shape[1]+pixPad]

    if np.sum(np.isnan(Ireg)) > 0:
        # if there are nan values [brought in from the border]
        # then that means the alignment result was pretty extreme
        err = np.inf
        diff = patch
    else:
        err = sh.variableDiffThr(Ireg, patch)
        diff = np.abs(Ireg-patch);

    return (-err,YX,diff)

def dist2patches(patchTuples,scale,debug=False):
    """
    Input:
        list patchTuples: EITHER (!) of the form:
              ((imgpatch_i, attrpatch_i, str attval_i, isflip_i), ...)
            or
              ((imgpatch_i, [attrpatch_i, ...], str attrval_i, int page_i, isflip_i), ...)
            I'm not entirely sure when it's a 4-tuple or a 5-tuple...but beware.
        float scale: Current scale factor.
    Output:
        (scores, locs, exemplar_idxs)
    """
    # patchTuples ((K img super regions),(K template patches))
    # for each pair, compute avg distance at scale sc
    scores=np.zeros(len(patchTuples))
    idx=0;
    locs=[]
    exemplar_idxs = []  # Keeps track of which exemplar patch was the best for a given voted ballot

    '''
    if debug:
        print "....dist 2 patches ...."
        pdb.set_trace()
    '''

    for idx in range(len(patchTuples)):
        # pt is either 4-tuple:
        #     ((imgpatch_i,[attrpatch_i, ...],,attrval_i,isflip_i), ...)
        # or a 5-tuple:
        #     ((imgpatch_i,[attrpatch_i, ...],attrval_i,page_i,isflip_i), ...)
        pt=patchTuples[idx]
        imgpatch = pt[0]
        attrpatches = pt[1]
        attrval = pt[2]
        flag = False
        if attrval == 'mail-pct' and debug==True:
            flag = True
        # A fix for a very bizarre openCv bug follows..... [check pixel_reg/opencv_bug_repo.py]
        I=np.round(sh.fastResize(imgpatch,scale)*255.)/255.
        # opencv appears to not like pure 1.0 and 0.0 values.
        #I[I==1.0]=.999; I[I==0.0]=.001
        #patchScale = sh.resizeOrNot(attrpatch.shape, int(round(max(attrpatch.shape)*scale)))
        bestscore = None
        bestloc = None
        best_idx_ex = None # Index of the best exemplar
        # Get the best score, over all possible exemplars (this is to
        # account for background variation). 
        for idx_ex, attrpatch in enumerate(attrpatches):
            patch=np.round(sh.fastResize(attrpatch,scale)*255.)/255.
            #patch[patch==1.0]=.999; patch[patch==0.0]=.001
            try:
                res=evalPatchSimilarity2(I,patch, debug=flag)
            except Exception as e:
                traceback.print_exc()
                print "CRASHED AT IDX:", idx
                print "    Scale was: {0}".format(scale)
                print "    I.shape: {0} patch.shape: {1}".format(I.shape, patch.shape)
                print "    imgpatch: {0} attrpatch: {1}".format(imgpatch.shape, attrpatch.shape)
                raise e
            # TODO: Do I want to maximize, or minimize 'score'?
            score = res[0] # I'm pretty sure we want to maximize.
            score = res[0] / (patch.shape[0]*patch.shape[1])
            if debug and attrval in ('american independent', 'libertarian'):
                print 'attrval: {0}  score: {1}'.format(attrval, score)
                pdb.set_trace()
            if bestscore == None or score > bestscore:
                bestscore = score
                best_idx_ex = idx_ex
                bestloc = (res[1][0]/scale,res[1][1]/scale)
            #scores[idx]=res[0]
            #locs.append((res[1][0]/scale,res[1][1]/scale))
        scores[idx] = bestscore
        locs.append(bestloc)
        exemplar_idxs.append(best_idx_ex)
    return (scores,locs, exemplar_idxs)

# input: image, patch images, super-region
# output: tuples of cropped image, patch image, template index, and flipped bit
#    ((I_i, attrexemplarpatch_i, str attrval, isflip_i), ...)
def createPatchTuples(I,attr2pat,R,flip=False):
    """
    Only used in templateSSWorker. Note: This is quite different from 
    createPatchTuplesMAP, despite the very similar name.
    Input:
        nparray I:
        dict ATTR2PAT: maps {str attrval: [[str imgpath_i, nparray patch_i], ...]}
        tuple R: SuperRegion.
    Output:
        tuple PATCHTUPLES. [[I_i, attrexemplarpatch_i, str attrval, bool isflip_i], ...]
    """
    pFac=1;
    (rOut,rOff)=sh.expand(R[0],R[1],R[2],R[3],I.shape[0],I.shape[1],pFac)
    I1=I[rOut[0]:rOut[1],rOut[2]:rOut[3]]
 
    patchTuples=[];
    for attrval, exmpl_pairs in attr2pat.keys():
        for (imgpath, patch) in exmpl_pairs:
            patchTuples.append((I1,patch,attrval,0))

    if not(flip):
        return patchTuples

    Ifl=sh.fastFlip(I)
    Ifl1=Ifl[rOut[0]:rOut[1],rOut[2]:rOut[3]]

    for key in attr2pat.keys():
        # key : (str temppath, str attrval)
        patchTuples.append((Ifl1,attr2pat[key],key,1))

    return patchTuples

def createPatchTuplesMAP(balL,attr2pat,R,flip=False):
    """
    Sort of creates 'tasks' for groupImagesWorkerMAP, where each task
    is a tuple of the form:
        (imgpatch_i, [attrpatch_i, ...], attrval_i, side_i, isflip_i)
    And you create one task for each side of a voted ballot (i.e. one
    for side0, another for side1, ...), to figure out the imgorder.
    Note that there can be multiple exemplar attrpatches.
    Input:
        tuple balL: (sidepath_i, ...)
        dict attr2pat: maps {str attrval: [obj imgpatch_i, ...]}
        tuple R: (y1, y2, x1, x2). A 'super' region.
    Output:
        ((obj imgpatch_i, [obj attrpatch_i0, ...], str attrval_i, int side_i, int isflip_i), ...)
    """
    pFac=1;
    patchTuples=[];

    for idx in range(len(balL)):
        balP=balL[idx]
        I=sh.standardImread(balP,flatten=True)
        (rOut,rOff)=sh.expand(R[0],R[1],R[2],R[3],I.shape[0],I.shape[1],pFac)
        I1=I[rOut[0]:rOut[1],rOut[2]:rOut[3]]
        for attrval, exemplars in attr2pat.iteritems():
            patchTuples.append((I1, exemplars, attrval, idx, 0))

        if flip:
            Ifl=sh.fastFlip(I)
            Ifl1=Ifl[rOut[0]:rOut[1],rOut[2]:rOut[3]]
            for key, exemplars in attr2pat.iteritems():
                # key := str attrval
                patchTuples.append((Ifl1,exemplars,key,idx,1))

    return patchTuples


def templateSSWorker(job):
    # dict attr2pat: maps {str attrval: [[str imgpath_i, obj patch_i], ...]}
    (attr2pat, key, superRegion, sStep, fOut) = job
    
    # 'key' is str attrval
    attr2pat1=attr2pat.copy()
    # TODO: Arbitrarily chooses a patch_i
    (imgpath, patch) = attr2pat1[key]
    I=sh.standardImread(imgpath,flatten=True)
    
    superRegionNp=np.array(superRegion)
    patchTuples=createPatchTuples(I,attr2pat1,superRegionNp,flip=True)

    sc0=sh.resizeOrNot(patch.shape,sh.MAX_PRECINCT_PATCH_DIM)
    minSc=sh.resizeOrNot(patch.shape,sh.MIN_PRECINCT_PATCH_DIM)

    (scores0,locs,exemplar_idxs)=dist2patches(patchTuples,sc0)

    sidx=np.argsort(scores0)
    sidx=sidx[::-1]
    trackIdx=sidx[0]

    # sc1 is the 'scale' that we're currently working with.
    sc1=sc0-sStep  # Starting scale.

    while sc1>minSc:
        try:
            (scores,locs, exemplar_idxs)=dist2patches(patchTuples,sc1)
        except Exception as e:
            d = {'patchTuples': patchTuples, 'sc1': sc1}
            path = '_errdict_0'
            '''
            while os.path.exists(path):
                new_i = int(path.split("_")[-1]) + 1
                path = '_errdict_{0}'.format(str(new_i))
            print '...outputting debug info to:', path
            pickle.dump(d, open(path, 'wb'))
            '''
            print "Exiting."
            exit(1)

        sidx=np.argsort(scores)
        sidx=sidx[::-1]
        mid=np.ceil(len(sidx)/2.0)
        dumpIdx=sidx[mid:len(sidx)]
        if sum(0+(dumpIdx==trackIdx))>0:
            break
        else:
            # decrease the scale
            sc1=sc1-sStep

    # write scale to file
    toWrite={"scale": min(sc1+sStep,sc0)}
    file = open(fOut, "wb")
    pickle.dump(toWrite, file)
    file.close()

def groupImagesWorkerMAP(job):
    try:
        #(ballotid, imgpaths, attrName, bb, attrinfo, attr2pat,scale) = job
        (ballotid, imgpaths, attrtype, bb, attr2pat, isflip, scale) = job

        # ((obj imgpatch_i, [obj attrpatch_i, ...], str attrval_i, int side_i, int isflip_i), ...)
        patchTuples = createPatchTuplesMAP(imgpaths,attr2pat,bb,flip=isflip)
        # Remember, attr2pat contains multiple exemplars
        firstPat=attr2pat.values()[0][0] # TODO: arbitrary choice, is this ok?
        rszFac = sh.resizeOrNot(firstPat.shape,sh.MAX_PRECINCT_PATCH_DIM);
        sweep=np.linspace(scale,rszFac,num=np.ceil(np.log2(len(attr2pat)))+2)
        sweep = sweep[sweep <= 1.0]

        finalOrder = [] # [(imgpatch_i, attrpatch_i, str attrval_i, int side_i, int isflip_i), ...]
        debug = False
        #if attrName == 'realmode':
        #    debug = True

        # 2. process
        #    Workers:
        #      - align with pyramid + prune
        #      - fine-alignment on best result
        #      - store precinct patch in grouping result folder
        #      - store list in grouping meta result file
        for sc in sweep:
            if len(patchTuples)<2:
                break
            # TODO: handle flipped and unflipped versions differently to save computation
            (scores,locs, exemplar_idxs)=dist2patches(patchTuples,sc,debug=debug)
            sidx=np.argsort(scores)
            # reverse for descend
            sidx=sidx[::-1]
            mid=np.ceil(len(sidx)/2.0)
            bestScore=scores[sidx[0]];
            bestLoc=locs[sidx[0]];
            bestExemplarIdx = exemplar_idxs[sidx[0]]
            keepIdx=sidx[0:mid]
            dumpIdx=sidx[mid:len(sidx)]
            dumped=sh.arraySlice(patchTuples,dumpIdx)        
            finalOrder.extend(dumped)
            patchTuples=sh.arraySlice(patchTuples,keepIdx)

        # align patch to top patch
        # patchTuples[0]: Best patch
        # I1: region around the attribute patch
        # P1: an exemplar attribute patch to compare against
        I1=patchTuples[0][0]
        P1=patchTuples[0][1][bestExemplarIdx]
        # finalOrder is of the form:
        #   ((obj imgpatch_i, obj attrpatch_i, str attrval_i, int imgorder_i, int isflip_i), ...)
        finalOrder.extend(patchTuples)
        finalOrder.reverse()

        bestLocG=[round(bestLoc[0]),round(bestLoc[1])]
        # I1c is the purported attrpatch on I1 (voted ballot)
        I1c=I1[bestLocG[0]:bestLocG[0]+P1.shape[0],bestLocG[1]:bestLocG[1]+P1.shape[1]]
        rszFac=sh.resizeOrNot(I1c.shape,sh.MAX_PRECINCT_PATCH_DIM)
        # IO := [transmatrix (?), img, err]
        if P1.shape[0] > I1c.shape[0] or P1.shape[1] > I1c.shape[1]:
            w1 = min(P1.shape[0], I1c.shape[0])
            h1 = min(P1.shape[1], I1c.shape[1])
            P1_a = np.zeros((w1,h1))
            P1_a[0:w1, 0:h1] = P1[0:w1, 0:h1]
            P1 = P1_a
            print 'WEIRD CASE:', P1.shape, I1c.shape
            misc.imsave("_weird_{0}.png".format(str(P1.shape)), P1_a)

        IO=imagesAlign(I1c,P1,type='rigid',rszFac=rszFac, minArea=np.power(2, 15))
        Ireg = np.nan_to_num(IO[1])
        doWriteMAP(finalOrder, Ireg, IO[2], attrtype, destDir, metaDir, balKey, bestExemplarIdx)
    except Exception as e:
        traceback.print_exc()
        raise e

def listAttributes(patchesH):
    # tuple ((key=attrType, patchesH tuple))

    attrL = set()
    for val in patchesH.values():
        for (regioncoords, attrtype, attrval, side) in val:
            attrL.add(attrtype)
    
    return list(attrL)

def listAttributesNEW(patchesH):
    """
    Input:
        dict patchesH:
    Output:
        A dict mapping {str attrtype: {str attrval: (bb, int side, k)}}
    """
    # tuple ((key=attrType, patchesH tuple))
    attrMap = {}
    for k in patchesH.keys():
        val=patchesH[k]
        for (bb,attrName,attrVal,side,is_digitbased,is_tabulationonly) in val:
            # check if type is in attrMap, if not, create
            
            # [kai] temporary hack for testing
            # attrVal = attrVal + '--' + os.path.basename(k)
            if attrMap.has_key(attrName):
                attrMap[attrName][attrVal]=(bb,side,k)
            else:
                attrMap[attrName]={}
                attrMap[attrName][attrVal]=(bb,side,k)                

    return attrMap

def estimateScale(attr2pat,superRegion,initDir,rszFac,stopped):
    """
    Input:
        dict attr2pat: maps {str attrval: [(str imgpath_i, nparray imgpatch_i), ...]}
        tuple superRegion: 
        str initDir:
        float rszFac:
        fn stopped:
    Output:
        A scale.
    """
    print 'estimating scale.'
    jobs=[]
    sStep=.05
    sList=[]
    nProc=sh.numProcs()

    for key in attr2pat.keys():
        # key := str attrval
        jobs.append((attr2pat,key,superRegion,sStep,pathjoin(initDir,key+'.png')))

    if nProc < 2:
        # default behavior for non multiproc machines
        for job in jobs:
            if stopped():
                return False
            templateSSWorker(job)
    else:
        print 'using ', nProc, ' processes'
        pool=mp.Pool(processes=nProc)

        it = [False]
        def imdone(x):
            it[0] = True
            print "I AM DONE NOW!"
        pool.map_async(templateSSWorker,jobs, callback=lambda x: imdone(it))

        while not it[0]:
            if stopped():
                pool.terminate()
                return False
            time.sleep(.1)

        pool.close()
        pool.join()
    # collect results
    for job in jobs:
        f1=job[5]
        s=pickle.load(open(f1))['scale']
        sList.append(s)

    print sList
    scale=min(max(sList)+2*sStep,rszFac)
    #scale = 0.95
    return scale

def groupByAttr(bal2imgs, attrName, attrMap, destDir, metaDir, stopped, proj, verbose=False, deleteall=True):
    """
    Input:
        dict bal2imgs: maps {str ballotid: (sidepath_i, ...)}
        str attrName: the current attribute type
        dict attrMap: maps {str attrtype: {str attrval: (bb, str side, blankpath)}}
        str destDir: A directory, i.e. 'extracted_precincts-ballottype'
        str metaDir: A directory, i.e. 'ballot_grouping_metadata-ballottype'
        fn stopped:
        obj proj:
    options:
        bool deleteall: if True, this will first remove all output files
                         before computing.
    """                       
    destDir=destDir+'-'+attrName
    metaDir=metaDir+'-'+attrName

    initDir=metaDir+'_init'
    exmDir=metaDir+'_exemplars'

    if deleteall:
        if os.path.exists(initDir): shutil.rmtree(initDir)
        if os.path.exists(exmDir): shutil.rmtree(exmDir)
        if os.path.exists(destDir): shutil.rmtree(destDir)
        if os.path.exists(metaDir): shutil.rmtree(metaDir)

    create_dirs(destDir)
    create_dirs(metaDir)
    create_dirs(initDir)
    create_dirs(exmDir)

    # maps {(str temppath, str attrval): obj imagepatch}
    attr2pat={}
    # maps {(str temppath, str attrval): str temppath}
    attr2tem={}
    superRegion=(float('inf'),0,float('inf'),0)
    # attrValMap: {str attrval: (bb, str side, blankpath)}
    attrValMap=attrMap[attrName]
    # 0.) First, grab an exemplar patch for each attrval. Add them to
    #     attr2pat, and save them to directories like:
    #         ballot_grouping_metadata-ballottype_exemplars/013.png
    # multexemplars_map: maps {attrtype: {attrval: ((str patchpath_i, str blankpath_i, bb_i), ...)}}
    multexemplars_map = pickle.load(open(pathjoin(proj.projdir_path,
                                                  proj.multexemplars_map),
                                         'rb'))
    exemplar_dict = multexemplars_map[attrName]
    for attrval, exemplars in exemplar_dict.iteritems():
        # Sort, in order to provide a canonical ordering
        exemplars_sorted = sorted(exemplars, key=lambda t: t[0])
        for (patchpath, blankpath, bb) in exemplars_sorted:
            P = sh.standardImread(patchpath, flatten=True)

            # TODO: I noticed that:
            # fooI = scipy.misc.imread(blankpath, flatten=True)
            # fooI = fooI[bb[0]:bb[1], bb[2]:bb[3]]
            # fooI.shape
            # fooI has a height that is 1 pixel greater (or smaller?)
            # than P, even though they should both be the same size.
            # Is this bad? Will it break things in strange and mysterious
            # ways?
            #attr2pat[attrval] = P
            attr2pat.setdefault(attrval, []).append(P)
            attr2tem[attrval] = blankpath
            superRegion = sh.bbUnion(superRegion, bb)
            # TODO: Don't re-save the patch to a separate directory.
            # I'm doing this just to ease integration with the existing
            # code base.
            sh.imsave(pathjoin(exmDir, attrval + '.png'), P)
    for _attr, patches in attr2pat.iteritems():
        print 'for attr {0}, there are {1} exemplars'.format(_attr, len(patches))
    '''
    for attrVal in attrValMap.keys():
        attrTuple=attrValMap[attrVal]
        bb = attrTuple[0]
        Iref=sh.standardImread(attrTuple[2],flatten=True)
        P=Iref[bb[0]:bb[1],bb[2]:bb[3]]
        attr2pat[attrVal]=P
        attr2tem[attrVal]=attrTuple[2]
        superRegion=sh.bbUnion(superRegion,bb)
        # store exemplar patch
        sh.imsave(pathjoin(exmDir,attrVal+'.png'),P);
    '''
    # 1.) Estimate smallest viable scale (for performance)
    if len(attr2pat)>2:
        scale = estimateScale(attr2pat,attr2tem,superRegion,initDir,sh.MAX_PRECINCT_PATCH_DIM,stopped)
    else:
        scale = sh.resizeOrNot(P.shape,sh.MAX_PRECINCT_PATCH_DIM);
    #scale = 1.0
    print 'ATTR: ', attrName,': using starting scale:',scale

    # 2.) Generate jobs for the multiprocessing
    jobs=[]
    nProc=sh.numProcs()

    #if attrName == 'realmode':
    #    nProc = 1

    for balKey in bal2imgs.keys():
        balL=bal2imgs[balKey]
        #if '339_122_106_128_2.png' in balL[0]:
        jobs.append([attr2pat, superRegion, balKey, balL, scale,
                     destDir, metaDir, attrName])

    # 3.) Perform jobs.
    if nProc < 2:
        # default behavior for non multiproc machines
        for job in jobs:
            if stopped():
                return False
            groupImagesWorkerMAP(job)

    else:
        print 'using ', nProc, ' processes'
        pool=mp.Pool(processes=nProc)

        it = [False]
        def imdone(x):
            it[0] = True
            print "I AM DONE NOW!"
        pool.map_async(groupImagesWorkerMAP,jobs, callback=lambda x: imdone(it))

        while not it[0]:
            if stopped():
                pool.terminate()
                return False
            time.sleep(.1)

        pool.close()
        pool.join()
        
    # TODO: quarantine on grouping errors. For now, just let alignment check handle it
    print 'ATTR: ', attrName, ': done'
    return True

def groupImagesMAP(bal2imgs, patchesH, destDir, metaDir, stopped, proj, verbose=False, deleteall=True):
    """
    Input:
      patchesH: A dict mapping:
                  {str imgpath: List of [(y1,y2,x1,x2), str attrtype, str attrval, str side, is_digit, is_tabulationonly]},
                where 'side' is either 'front' or 'back'.
      ballotD:
      destDir:
      metaDir:
      stopped:
      obj proj: 
    """
    # NOTE: assuming each ballot has same number of attributes

    # 1. loop over each attribute
    # 2. perform grouping using unique examples of attribute
    # 3. store in metadata folder
    # 4. [verification] look at each attr separately

    # 1. pre-load all template regions
    # Note: because multi-page elections will have different
    # attribute types on the front and back sides, we will have
    # to modify the grouping to accomodate multi-page.
    attrMap=listAttributesNEW(patchesH)

    for attrName in attrMap.keys():
        groupByAttr(bal2imgs,attrName,attrMap,destDir,metaDir,stopped, proj, verbose=verbose,deleteall=deleteall)

def is_image_ext(filename):
    IMG_EXTS = ('.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff')
    return os.path.splitext(filename)[1].lower() in IMG_EXTS

def create_dirs(*dirs):
    """
    For each dir in dirs, create the directory if it doesn't yet
    exist. Will work for things like:
        foo/bar/baz
    and will create foo, foo/bar, and foo/bar/baz correctly.
    """
    for dir in dirs:
        try:
            os.makedirs(dir)
        except Exception as e:
            pass

def extract_attrvals(bal2imgs, attrtype, bbSuper, attrexemplars, imginfo_map, img2page, page):
    """
    Input:
        dict BAL2IMGS: 
        str ATTRTYPE:
        tuple BBSUPER: Super region of ATTRTYPE (x1,y1,x2,y2). This is the
            original bounding box that the user made in 'Define Attributes'.
        dict ATTRINFO: [_,x,y,w,h,attrtype,attrval,page,isdigitbased,istabonly]
        dict ATTREXEMPLARS: maps {str attrval: [(str subpatchP_i, str imgpathP, (x1,y1,x2,y2), ...]}. Stores 
            all exemplar images for ATTRVAL, including the original image path
            that the patch was extracted from.
        dict IMGINFO_MAP: 
        int PAGE:
    Output:
        dict RESULT. maps {int ballotid: {'attrOrder': attrorder, 'err': err,
                                          'exemplaridx': exemplaridx, 'bb': (x1,y1,x2,y2)}}
    """
    # 0.) Create the ATTR2PAT map
    attr2pat = {} # maps {str attrval: [obj imgpatch_i, ...]}
    for attrval, exemplars in attrexemplars.iteritems():
        for (subpatchpath, imgpath, (x1,y1,x2,y2)) in exemplars:
            exmpl = sh.standardImread(subpatchpath, flatten=True)
            attr2pat.setdefault(attrval, []).append(exmpl)

    # 1.) Estimate smallest viable scale (for performance)
    '''
    if len(attr2pat)>2:
        scale = estimateScale(attr2pat,attr2tem,superRegion,initDir,sh.MAX_PRECINCT_PATCH_DIM,stopped)
    else:
        scale = sh.resizeOrNot(P.shape,sh.MAX_PRECINCT_PATCH_DIM);
    '''
    scale = 0.85
    print 'ATTR: ', attrName,': using starting scale:',scale


    # 2.) Generate jobs for the multiprocessing
    jobs=[]
    nProc=sh.numProcs()

    for ballotid, imgpaths in b2imgs.iteritems():
        # B2IMGS has been sorted by page by the caller
        imgpath = imgpaths[page]
        isflip = imginfo_map[imgpath]['isflip']
        # BBSUPER_INPUT := (y1,y2,x1,x2)
        bbSuper_in = bbSuper[1], bbSuper[3], bbSuper[0], bbSuper[2]
        jobs.append([ballotid, [imgpath], attrType, bbSuper_in, attr2pat, isflip, scale])

    # 3.) Perform jobs.
    if nProc < 2:
        # default behavior for non multiproc machines
        for job in jobs:
            if stopped():
                return False
            groupImagesWorkerMAP(job)

    else:
        print 'using ', nProc, ' processes'
        pool=mp.Pool(processes=nProc)

        it = [False]
        def imdone(x):
            it[0] = True
            print "I AM DONE NOW!"
        pool.map_async(groupImagesWorkerMAP,jobs, callback=lambda x: imdone(it))

        while not it[0]:
            if stopped():
                pool.terminate()
                return False
            time.sleep(.1)

        pool.close()
        pool.join()
        
    # TODO: quarantine on grouping errors. For now, just let alignment check handle it
    print 'ATTR: ', attrName, ': done'
    return True

def do_extract_attrvals(partitions_map, partition_attrmap, partition_exmpls, b2imgs, img2bal,
                        multexemplars_map, img2page, imginfo_map, attrs):
    """ Extracts the values of image-based Attributes for each voted
    ballot. 
    Input:
        dict PARTITIONS_MAP: maps {int partitionID: [int ballotID_i, ...]}
        dict PARTITION_ATTRMAP: maps {int partitionID: [[imP,x,y,w,h,attrtype,attrval,
                                                         page,isdigitbased,istabonly],...]}
        dict B2IMGS: maps {int ballotID: [imP_side0, imP_side1, ...]}
        dict IMG2BAL: maps {str imgpath: int ballotID}
        dict MULTEXEMPLARS_MAP: maps {str attrtype: {str attrval: [(subpatchP_i, imgpath_i, (x1,y1,x2,y2)), ...]}}
        dict IMG2PAGE: maps {str imgpath: int page}
        dict IMGINFO_MAP: maps {str imgpath: {str key: str val}}. Used for
            the 'isflip' mapping.
        list ATTRS: [dict attr_i, ...]
    Output:
        dict RESULTS. maps {int ballotID: {attrtype: {'attrOrder': attrorder, 'err': err,
                                                      'exemplar_idx': exemplar_idx,
                                                      'bb': (x1,y1,x2,y2)}}}
    """
    def get_grpmodes(attrs):
        grpmodes_map = {} # maps {str attrtype: bool is_grp_per_partition}
        for attr in attrs:
            attrtype = '_'.join(sorted(attr['attrs']))
            grpmodes_map[attrtype] = attr['grp_per_partition']
        return grpmodes_map
    results = {}
    attrmap = {} # maps {str attrtype: {str attrval: (bb, int side, blankpath)}}
    grpmodes_map = get_grpmodes(attrs)
    for partitionID, attrslst in partition_attrmap.iteritems():
        for attr in attrslst:
            attrtype = attr[5]
            page = attr[7]
            in_bal2imgs = {}
            if grpmodes_map[attrtype]:
                # ATTRTYPE is group by partition
                for partitionid, ballotids in partition_exmpls.iteritems():
                    exmpl_ballotid = ballotids[0]
                    imgpaths = b2imgs[exmpl_ballotid]
                    imgpaths_ordered = sorted(imgpaths, key=lambda imP: img2page[imP])
                    in_bal2imgs[partitionid] = imgpaths_ordered
            else:
                # ATTRTYPE is group by ballot
                for ballotid, imgpaths in b2imgs.iteritems():
                    imgpaths = b2imgs[exmpl_ballotid]
                    imgpaths_ordered = sorted(imgpaths, key=lambda imP: img2page[imP])
                    in_bal2imgs[ballotid] = imgpaths_ordered

            bbSuper = None
            for attrdict in attrs:
                if attrtype == '_'.join(sorted(attrdict['attrs'])):
                    bbSuper = attrdict['x1'], attrdict['y1'], attrdict['x2'], attrdict['y2']
                    break
            # dict ATTREXEMPLARS: maps {attrval: [(subpatchP, imgpath, (x1,y1,x2,y2)), ...]}
            attrexemplars = multexemplars_map[attrtype]
            result = extract_attrvals(in_bal2imgs, attrtype, bbSuper, attrexemplars, imginfo_map, page)
            results.setdefault(ID, {})[attrtype] = result
    return results
