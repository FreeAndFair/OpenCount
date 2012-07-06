import multiprocessing as mp
import pdb
import numpy as np
import cv
import csv
import string
import math
import imagesAlign as lk
from scipy import misc
from matplotlib.pyplot import show, imshow, figure, title, colorbar, savefig, annotate

MAX_PRECINCT_PATCH_DIM=800
MAX_PRECINCT_PATCH_DISPLAY=800
FLIP_CHECK_HEIGHT=200
COARSE_BALLOT_REG_HEIGHT=500
LOCAL_PATCH_REG_HEIGHT=250
MAX_DIFF_HEIGHT=10

def prepOpenCV(I):
    I[I>.99]=.99
    I[I<.01]=.01

def fastResize(I,rszFac,sig=-1):
    if rszFac==1:
        return I
    else:
        Icv=cv.fromarray(np.copy(I))
        I1cv=cv.CreateMat(int(math.floor(I.shape[0]*rszFac)),int(math.floor(I.shape[1]*rszFac)),Icv.type)
        cv.Resize(Icv,I1cv)
        Iout=np.asarray(I1cv)
        if sig>0:
            Iout=gaussian_filter(Iout,sig);

        return Iout

def fastFlip(I):
    Icv=cv.fromarray(np.copy(I))
    I1cv=cv.CreateMat(I.shape[0],I.shape[1],Icv.type)
    cv.Flip(Icv,I1cv,-1)
    Iout=np.asarray(I1cv)
    return Iout


''' 
Input: 
  I0: full image
  bb: bounding box of patch
  imList: list of full filenames for images to search
  threshold: only return matches above this value
  rszFac: downsampling factor for speed
  region: bounding box to limit search for speed (TODO)

Output:
  list of tuples, one for every match
  ((filename, score1, score2, patch, i1, i2, j1, j2, resize factor), (...) )

  score1: result from NCC
  score2: produced from local alignment. this score is much more reliable.

  Example:
  I1cropped=I1[i1:i2,j1:j2]

TODOS(kai)
  - return multiple matches on same image
  - seems to be weird behavior when rszFac is .75
'''
def find_patch_matchesV1(I,bb,imList,threshold=.8,rszFac=.75,bbSearch=[],padding=.75):
    matchList = [] # (filename, left,right,up,down)
    prepOpenCV(I);
    I = np.round(fastResize(I,rszFac)*255.)/255;

    bb[0] = bb[0]*rszFac
    bb[1] = bb[1]*rszFac
    bb[2] = bb[2]*rszFac
    bb[3] = bb[3]*rszFac
    [bbOut,bbOff]=expand(bb[0],bb[1],bb[2],bb[3],I.shape[0],I.shape[1],.25)
    patch = I[bbOut[0]:bbOut[1],bbOut[2]:bbOut[3]]
    patch0 = I[bb[0]:bb[1],bb[2]:bb[3]]

    if len(bbSearch)>0:
        bbSearch[0] = bbSearch[0]*rszFac
        bbSearch[1] = bbSearch[1]*rszFac
        bbSearch[2] = bbSearch[2]*rszFac
        bbSearch[3] = bbSearch[3]*rszFac

    for imP in imList:
        I1 = standardImread(imP,flatten=True)
        prepOpenCV(I1)
        I1 = np.round(fastResize(I1,rszFac)*255.)/255.

        # crop to region if specified
        if len(bbSearch)>0:
            [bbOut1,bbOff1]=expand(bbSearch[0],bbSearch[1],
                                   bbSearch[2],bbSearch[3],
                                   I1.shape[0],I1.shape[1],padding)
            I1=I1[bbOut1[0]:bbOut1[1],bbOut1[2]:bbOut1[3]]

        patchCv=cv.fromarray(np.copy(patch))
        ICv=cv.fromarray(np.copy(I1))
        outCv=cv.CreateMat(I1.shape[0]-patch.shape[0]+1,I1.shape[1]-patch.shape[1]+1,patchCv.type)
        cv.MatchTemplate(ICv,patchCv,outCv,cv.CV_TM_CCOEFF_NORMED)
        Iout=np.asarray(outCv)

        Iout[Iout==1.0]=0;

        if Iout.max() < threshold:
            continue

        score1 = Iout.max()

        YX=np.unravel_index(Iout.argmax(),Iout.shape)
        i1=YX[0]; i2=YX[0]+patch.shape[0]
        j1=YX[1]; j2=YX[1]+patch.shape[1]
        I1c = I1[i1:i2,j1:j2]
        IO=lk.imagesAlign(I1c,patch,type='rigid')
        Ireg = IO[1]
        Ireg = Ireg[bbOff[0]:bbOff[1],bbOff[2]:bbOff[3]]

        diff=np.abs(Ireg-patch0);
        err=np.sum(diff[np.nonzero(diff>.25)])

        score2 = err / diff.size

        matchList.append((imP,score1,score2,Ireg,i1,i2,j1,j2,rszFac))

    return matchList


''' 
Input: 
  patch: image patch to find
  imList: list of full filenames for images to search
  threshold: only return matches above this value
  rszFac: downsampling factor for speed
  region: bounding box to limit search for speed (TODO)

Output:
  list of tuples, one for every match
  ((filename, score, rszFac, left, right, up, down), (...) )

  Example:
  I1cropped=I1[i1:i2,j1:j2]

TODOS(kai)
  - implement the region input
  - return multiple matches on same image
  - seems to be weird behavior when rszFac is .75
'''
def find_patch_matches(patch,imList,threshold=.8,rszFac=.75,region=[],padding=.75):
    matchList = [] # (filename, left,right,up,down)

    patch1 = np.round(fastResize(patch,rszFac)*255.)/255;
    prepOpenCV(patch)
    #patch[patch>.99]=.99; patch[patch<.01]=.01
    for imP in imList:
        I = standardImread(imP,flatten=True)
        prepOpenCV(I)
        #I[I>.99]=.99; I[I==0.0]=.001
        I = np.round(fastResize(I,rszFac)*255.)/255.

        # crop to region if specified
        if len(region)>0:
            i1 = region[0]*rszFac
            i2 = region[1]*rszFac
            j1 = region[2]*rszFac
            j2 = region[3]*rszFac
            [bbOut,bbOff]=expand(i1,i2,j1,j2,I.shape[0],I.shape[1],padding)
            I1=I[bbOut[0]:bbOut[1],bbOut[2]:bbOut[3]]
        else:
            I1=I;

        patchCv=cv.fromarray(np.copy(patch1))
        ICv=cv.fromarray(np.copy(I1))
        outCv=cv.CreateMat(I1.shape[0]-patch1.shape[0]+1,I1.shape[1]-patch1.shape[1]+1,patchCv.type)
        cv.MatchTemplate(ICv,patchCv,outCv,cv.CV_TM_CCOEFF_NORMED)
        Iout=np.asarray(outCv)
        Iout[Iout==1.0]=0;
        # TODO: nonmax suppression

        if Iout.max() < threshold:
            continue

        YX=np.unravel_index(Iout.argmax(),Iout.shape)
        i1=YX[0]; i2=YX[0]+patch1.shape[0]
        j1=YX[1]; j2=YX[1]+patch1.shape[1]
        if len(region)>0:
            i1 = i1 + bbOut[0]
            i2 = i2 + bbOut[0]
            j1 = j1 + bbOut[2]
            j2 = j2 + bbOut[2]
        matchList.append((imP,Iout.max(),rszFac,i1,i2,j1,j2))

    return matchList

def numProcs():
    nProc=mp.cpu_count() 
    return nProc

def cropBb(I,bb):
    return I[bb[0]:bb[1],bb[2]:bb[3]]

def expand(i1,i2,j1,j2,iMx,jMx,pFac):
    # TODO: fix all refs to this function to use the Bbs version
    ''' Pad the rectangle from the input by factor pd. Return both the new rectangle and coordinates for extracting the unpadded rectangle from new padded, cropped one'''
    # don't allow the resulting image to take up more than 33% of the width
#    maxPd=round((min(iMx,jMx)/3.0)/2.0)
    pd=min(i2-i1,j2-j1)*pFac

    i1o=i1-pd; i1o=max(i1o,0);
    i2o=i2+pd; i2o=min(i2o,iMx)

    j1o=j1-pd; j1o=max(j1o,0);
    j2o=j2+pd; j2o=min(j2o,jMx)

    iOff=i1-i1o; jOff=j1-j1o;
    rOut=np.array([i1o,i2o,j1o,j2o])
    rOff=np.array([iOff,iOff+i2-i1,jOff,jOff+j2-j1])

    return (rOut,rOff)

def imsave(path,I):
    # assumed image is between 0 and 1
    I[0,0]=.01; I[1,0]=.99
    misc.imsave(path,I)

def safeExpand(bb,pd,iMx,jMx):
    i1=bb[0]-pd; i1=max(i1,0);
    i2=bb[1]+pd; i2=min(i2,iMx)
    j1=bb[2]-pd; j1=max(j1,0);
    j2=bb[3]+pd; j2=min(j2,jMx)

    return np.array([i1,i2,j1,j2])
    
def expandBbsSingle(bb,iMx,jMx,pFac):

    iLen=bb[1]-bb[0]
    jLen=bb[3]-bb[2]
    # compute pad factor to smaller dimension
    pd=min(iLen,jLen)*pFac

    # compute new cropped region
    bbOut=safeExpand(bb,pd,iMx,jMx)

    # correct offsets of bb
    i1=bb[0]-bbOut[0]
    i2=i1+iLen
    j1=bb[2]-bbOut[2]
    j2=j1+jLen

    bbOff=np.array([i1,i2,j1,j2,bb[4]])
    return (bbOut,bbOff)
    

def expandBbsMulti(bbs,iMx,jMx,pFac):

    # compute pad factor to smaller dimension
    bb1=bbs[0,:]
    iLen=bb1[1]-bb1[0]
    jLen=bb1[3]-bb1[2]
    # compute pad factor to smaller dimension
    pd=min(iLen,jLen)*pFac

    bbSuper=bbUnion1(bbs)

    bbOut=safeExpand(bbSuper,pd,iMx,jMx)

    # compute super region
    # then pad super region
    # then compute offset to that

    bbsOff=np.zeros((0,5))
    for i in range(bbs.shape[0]):
        bb=bbs[i,:]
        i1=bb[0]-bbOut[0]
        i2=i1+iLen
        j1=bb[2]-bbOut[2]
        j2=j1+jLen
        bbsOff=np.vstack((bbsOff,np.array([i1,i2,j1,j2,bb[4]])))

    return (bbOut,bbsOff)


def expandBbs(bbs,iMx,jMx,pFac):
    ''' Assumed that bbs are all the same dim. Padding is relative to the size 
        one bb.
    '''
    # don't allow the resulting image to take up more than 33% of the width
    if len(bbs.shape)==1:
        return expandBbsSingle(bbs,iMx,jMx,pFac)
    else:
        return expandBbsMulti(bbs,iMx,jMx,pFac)


# extend bbUnion to take in a list of BBS
def bbUnion1(bbs):
    bbOut=[float('inf'),0,float('inf'),0]
    for i in range(bbs.shape[0]):
        bb1=bbs[i,:]
        bbOut[0]=min(bbOut[0],bb1[0])
        bbOut[1]=max(bbOut[1],bb1[1])
        bbOut[2]=min(bbOut[2],bb1[2])
        bbOut[3]=max(bbOut[3],bb1[3])
        
    return bbOut

def bbUnion(bb1,bb2):
    if len(bb1)==0:
        return bb2
    if len(bb2)==0:
        return bb1
        
    bbOut=[0,0,0,0]
    bbOut[0]=min(bb1[0],bb2[0])
    bbOut[1]=max(bb1[1],bb2[1])
    bbOut[2]=min(bb1[2],bb2[2])
    bbOut[3]=max(bb1[3],bb2[3])
    return bbOut

def csv2bbs(csvP):
    bdReader=csv.reader(open(csvP,'r'))
    isFirst=True; pFac=.1;
    bbs=np.empty((0,5))
    # loop over every target location
    for row in bdReader:
        if isFirst:
            isFirst=False;
            continue
        isContest=string.atoi(row[7]);
        if isContest==1:
            continue
        x1=string.atoi(row[2]); x2=x1+string.atoi(row[4]);
        y1=string.atoi(row[3]); y2=y1+string.atoi(row[5]);
        idx=string.atoi(row[1]);
        # expand region around target
        row1=np.array([y1,y2,x1,x2,idx])
        bbs=np.vstack((bbs,row1));
        
    return bbs;

def maskBordersTargets(I,bbs,pf=.05):
    return maskBorders(maskTargets(I,bbs,pf=pf),pf=pf);

def maskBorders(I,pf=.05):
    # to account for poorly scanned borders, create nan-padded Irefs
    IM=np.copy(I);
    rPd=pf*I.shape[0]; cPd=pf*I.shape[1];
    IM[1:rPd,:]=np.nan; IM[-rPd:,:]=np.nan;
    IM[:,1:cPd]=np.nan; IM[:,-cPd:]=np.nan;
    return IM

def maskTargets(I,bbs,pf=.05):
    ''' Set regions of targets to be NANs '''
    IM=np.copy(I);
    for i in range(bbs.shape[0]):
        y1=bbs[i,0]; y2=bbs[i,1]
        x1=bbs[i,2]; x2=bbs[i,3]        
        IM[y1:y2,x1:x2]=np.nan;        

    return IM

def resizeOrNot(shape,c):
    largestDim=max(shape[0:2])
    if largestDim<c:
        return 1
    else:
        return c/(largestDim+0.0)

def rgb2gray(I):
    if len(I.shape)<3:
        return I
    else:
        Ichn1=I[:,:,0]
        Icv=cv.fromarray(np.copy(I))
        Ichn1cv=cv.fromarray(np.copy(Ichn1))
        cv.CvtColor(Icv,Ichn1cv,cv.CV_RGB2GRAY)
        Iout=np.asarray(Ichn1cv)
        return Iout

def arraySlice(A,inds):
    out=[]
    for i in inds:
        out.append(A[i])

    return out

def standardImread(fNm,flatten=False):
    I=np.float32(misc.imread(fNm)/255.0)
    if flatten:
        I=rgb2gray(I)
    return I
