import multiprocessing as mp
import pdb
import numpy as np
import cv
import csv
import string
from scipy import misc

MAX_PRECINCT_PATCH_DIM=800
MAX_PRECINCT_PATCH_DISPLAY=800
FLIP_CHECK_HEIGHT=200
COARSE_BALLOT_REG_HEIGHT=500
LOCAL_PATCH_REG_HEIGHT=250
MAX_DIFF_HEIGHT=10

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
