import numpy as np
import math, pickle, os, traceback
import pdb
import time
import cv
import shared as sh
from scipy.ndimage import gaussian_filter
from matplotlib.pyplot import show, imshow, figure, title, colorbar, savefig, annotate

def imagesAlign(I,Iref,fillval=np.nan,type='similarity',vCells=1,hCells=1,rszFac=1,verbose=False):

    if len(I.shape)==3:
        I1=sh.rgb2gray(I)
    else:
        I1=I
        
    if len(Iref.shape)==3:
        Iref1=sh.rgb2gray(Iref)
    else:
        Iref1=Iref

    # check if more than one vertical or horizontal cell
    if (vCells>1) or (hCells>1):
        I2=imagesAlign(I1,Iref1,type=type)[1];
        Iout=np.copy(Iref1);
        pFac=.25;
        vStep=math.ceil(I1.shape[0]/vCells); vPad=pFac*vStep;
        hStep=math.ceil(I1.shape[1]/hCells); hPad=pFac*vStep;
        for i in range(vCells):
            for j in range(hCells):
                # 2. chop + pad each cell then align
                # 3. stitch back together
                i1=i*vStep; i1=max(i1,0);
                i2=(i+1)*vStep; i2=min(i2,I1.shape[0]-1);
                j1=j*hStep; j1=max(j1,0);
                j2=(j+1)*hStep; j2=min(j2,I1.shape[1]-1);

                i1p=i1-vPad; i1p=max(i1p,0);
                i2p=i2+vPad; i2p=min(i2p,I1.shape[0]-1);
                j1p=j1-hPad; j1p=max(j1p,0);
                j2p=j2+hPad; j2p=min(j2p,I1.shape[1]-1);
                
                Ic=I2[i1p:i2p,j1p:j2p]
                Irefc=Iref1[i1p:i2p,j1p:j2p]
                (H,err)=imagesAlign1(Ic,Irefc,type=type,verbose=verbose)
                IcT=imtransform(np.copy(Ic),H)
                Iout[i1:i2,j1:j2]=IcT[i1-i1p:(i1-i1p)+(i2-i1),j1-j1p:(j1-j1p)+(j2-j1)]

        return (np.eye(3),Iout,-1)

    if rszFac==1:
        t0=time.clock()
        (H,err)=imagesAlign1(I1,Iref1,type=type,verbose=verbose)
        if verbose:
            print 'alignment time:',time.clock()-t0,'(s)'
    else:
        I1=sh.fastResize(I1,rszFac)
        Iref1=sh.fastResize(Iref1,rszFac)
        S=np.eye(3); S[0,0]=1/rszFac; S[1,1]=1/rszFac;
        H0=np.eye(3)
        H0=np.dot(np.dot(np.linalg.inv(S),H0),S)
        t0=time.clock()
        (H,err)=imagesAlign1(I1,Iref1,H0=H0,type=type,verbose=verbose)
        if verbose:
            print 'alignment time:',time.clock()-t0,'(s)'
        H=np.dot(S,np.dot(H,np.linalg.inv(S)))
    
    return (H,imtransform(np.copy(I),H,fillval=fillval),err);

def imagesAlign1(I,Iref,H0=np.eye(3),type='similarity',verbose=False):

    minArea=np.power(2,15)
    lbda=1e-6
    wh=Iref.shape
    eps=1e-3
    sig=2

    # recursive check
    if np.prod(wh)<minArea:
        H=H0
    else:
        I1=sh.fastResize(I,.5)
        Iref1=sh.fastResize(Iref,.5)
        S=np.eye(3); S[0,0]=2; S[1,1]=2;
        H0=np.dot(np.dot(np.linalg.inv(S),H0),S)
        (H,errx)=imagesAlign1(I1,Iref1,H0=H0,type=type,verbose=verbose)
        H=np.dot(S,np.dot(H,np.linalg.inv(S)))


    # smooth images
    Iref=gaussian_filter(Iref,sig); I=gaussian_filter(I,sig);

    # pad image with NaNs
    ws=np.concatenate(([0],[0],range(wh[0]),[wh[0]-1],[wh[0]-1]))
    hs=np.concatenate(([0],[0],range(wh[1]),[wh[1]-1],[wh[1]-1]))

    Iref=Iref[np.ix_(ws,hs)]; I=I[np.ix_(ws,hs)]
    hs=np.array([0,1,wh[1]+2,wh[1]+3])
    ws=np.array([0,1,wh[0]+2,wh[0]+3])

    Iref[ws,:]=np.nan; I[ws,:]=np.nan;
    Iref[:,hs]=np.nan; I[:,hs]=np.nan;
    
    wts=np.array([1,1,1.0204,.03125,1.0313,.0204,.000555,.000555]);
    s=math.sqrt(Iref.size)/128.0
    wts[2]=math.pow(wts[2],1/s)
    wts[3]=wts[3]/s
    wts[4]=math.pow(wts[4],1/s)
    wts[5]=wts[5]/s
    wts[6]=wts[6]/(s*s)
    wts[7]=wts[7]/(s*s)

    # compute differences

    if type=='translation':
        keep=[0,1];
    elif type=='rigid':
        keep=[0,1,5];
    elif type=='similarity':
        keep=[0,1,2,5];
    elif type=='affine':
        keep=[0,1,2,3,4,5];
    elif type=='projective':
        keep=[0,1,2,3,4,5,6,7];
                
    # compute transformations
    HH=ds2H(-1*np.ones(8),wts)
    Hs=HH[1][keep,:]

    # apply transformations
    Ts=np.zeros([Hs.shape[0],Iref.shape[0],Iref.shape[1]])
    Ms=np.ones([Iref.shape[0],Iref.shape[1]],dtype='bool')
    for i in range(Hs.shape[0]):
        Ts[i,:,:]=imtransform(Iref,Hs[i,:,:])
        Ms=Ms & ~np.isnan(Ts[i,:,:])
    
    Ds=Ts-np.tile(Iref,[Hs.shape[0],1,1])
    D=Ds.reshape(Ds.shape[0],np.prod(Iref.shape))
    Lbda=lbda*np.prod(Iref.shape)*np.eye(Ds.shape[0])
    err=np.Inf
    ds=np.zeros([8,1])
    for i in range(100):
        # warp image with current esimate
        Ip=imtransform(I,H)
        M=(Ms & ~np.isnan(Ip)) & ~np.isnan(Iref)
        Mf=1*M.reshape(np.prod(I.shape),1)
        dI=Ip-Iref; dIf=dI.reshape(np.prod(I.shape),1)
        idx=np.nonzero(np.squeeze(Mf))
        D0=np.squeeze(D[:,idx]); dI1=dIf[idx]

        # check if > half of pixels turn to NAN
        # subtract new nans from old nans, divide by old valids
        origValidPixels=np.sum(1-(np.isnan(I)+0))
        newValidPixels=np.sum(1-(np.isnan(Ip+I)+0))
        if newValidPixels<(origValidPixels/3.):
            return (np.eye(3),np.inf)

        _A = np.dot(D0, D0.T)
        _B = np.linalg.inv(_A + Lbda)
        try:
            _C = np.dot(D0, dI1)
        except Exception as e:
            print e
            print "D0.shape:", D0.shape
            print "dI1.shape:", dI1.shape
            print "_B shape:", _B.shape
            print "_C shape:", _C.shape
            traceback.print_exc()
            raise Exception("_C computation failed")
        ds1 = np.dot(_B, _C)
        #ds1=np.dot(np.linalg.inv(np.dot(D0,D0.T)+Lbda),np.dot(D0,dI1))
        ds[keep]=ds1;
        HH=ds2H(ds,wts); H=np.dot(H,HH[0]); H=H/H[2,2]
        err0=err; err=np.abs(dI1); err=np.mean(err); delta=err0-err;
        if verbose:
            print I.shape," i=",i," err=",err," del=",delta
        if delta<eps:
            break

    return (H,err)

def ds2H(ds,wts):

    Hs=np.eye(3)
    Hs=np.tile(Hs,[8,1,1])
    # 1. x translation
    Hs[0,0,2]=wts[0]*ds[0]
    # 2. y translation
    Hs[1,1,2]=wts[1]*ds[1]
    # 3. scale
    Hs[2,:2,:2]=np.eye(2)*math.pow(wts[2],ds[2])
    # # # 4. shear
    Hs[3,0,1]=wts[3]*ds[3]    
    # # 5. scale non-uniform
    Hs[4,1,1]=math.pow(wts[4],ds[4])
    # # 6. rotation z
    ct=math.cos(wts[5]*ds[5]); st=math.sin(wts[5]*ds[5])
    Hs[5,:2,:2]=np.array([[ct,st],[-st,ct]])
    # # 7. rotation x
    ct=math.cos(wts[6]*ds[6]); st=math.sin(wts[6]*ds[6])    
    Hs[6,1,1]=ct; Hs[6,1,2]=-st;
    Hs[6,2,1]=st; Hs[6,2,2]=ct;
    # # 8. rotation y
    ct=math.cos(wts[7]*ds[7]); st=math.sin(wts[7]*ds[7])
    Hs[7,0,0]=ct; Hs[7,0,2]=-st;
    Hs[7,2,0]=st; Hs[7,2,2]=ct;    
    
    # collect into H
    H=np.eye(3)
    for i in range(Hs.shape[0]):
        H=np.dot(Hs[i,:,:],H)

    return (H,Hs)

def imtransform(I,H0,fillval=np.nan):
    # transform image using center as origin
    if len(I.shape)==3:
        Iout=np.copy(I)
        Iout[:,:,0]=imtransform(I[:,:,0],H0,fillval=fillval)
        Iout[:,:,1]=imtransform(I[:,:,1],H0,fillval=fillval)
        Iout[:,:,2]=imtransform(I[:,:,2],H0,fillval=fillval)
        return Iout
    else:
        T0=np.eye(3); T0[0,2]=I.shape[1]/2.0; T0[1,2]=I.shape[0]/2.0
        T1=np.eye(3); T1[0,2]=-I.shape[1]/2.0; T1[1,2]=-I.shape[0]/2.0
        H=np.dot(np.dot(T0,H0),T1)

        # transform each channel separately
        Icv=cv.fromarray(np.copy(I))
        I1cv=cv.CreateMat(I.shape[0],I.shape[1],Icv.type)

        cv.WarpPerspective(Icv,I1cv,cv.fromarray(np.copy(H)),fillval=-1);
        I1=np.asarray(I1cv)
        I1[np.nonzero(I1<0)]=fillval
        return I1

def imtransform2(I,H0,fillval=3.0):
    # transform image using center as origin
    if len(I.shape)==3:
        Iout=np.copy(I)
        Iout[:,:,0]=imtransform2(I[:,:,0],H0,fillval=fillval)
        Iout[:,:,1]=imtransform2(I[:,:,1],H0,fillval=fillval)
        Iout[:,:,2]=imtransform2(I[:,:,2],H0,fillval=fillval)
        return Iout
    else:
        pdb.set_trace()
        T0=np.eye(3); T0[0,2]=I.shape[1]/2.0; T0[1,2]=I.shape[0]/2.0
        T1=np.eye(3); T1[0,2]=-I.shape[1]/2.0; T1[1,2]=-I.shape[0]/2.0
        H=np.dot(np.dot(T0,H0),T1)

        # transform each channel separately
        Icv=cv.fromarray(np.copy(I))
        I1cv=cv.CreateMat(I.shape[0],I.shape[1],Icv.type)

        cv.WarpPerspective(Icv,I1cv,cv.fromarray(np.copy(H)),fillval=1.0);
        I1=np.asarray(I1cv)
        I1[np.nonzero(I1<0)]=fillval
        return I1

def pttransform(I,H0,pt0):
    # transform point using center as origin
    T0=np.eye(3); T0[0,2]=I.shape[1]/2.0; T0[1,2]=I.shape[0]/2.0
    T1=np.eye(3); T1[0,2]=-I.shape[1]/2.0; T1[1,2]=-I.shape[0]/2.0
    H=np.dot(np.dot(T0,H0),T1)

    pt1=np.dot(H,pt0)
    pt1[0]=pt1[0]/pt1[2]; pt1[1]=pt1[1]/pt1[2]; pt1[2]=1

    return pt1


def associateTwoPage(tplImL, balImL):
    # return permuted balImL list
    # check first template against both ballots
    # Assumes that tplImL, balImL are ordered by imageorder, meaning:
    #   tplImL := [frontpath, backpath]
    #   balImL := [frontback, backpath]
    tpl0=tplImL[0]; tpl1=tplImL[1]
    bal0=balImL[0]; bal1=balImL[1]

    res0=checkBallotFlipped(bal0,tpl0)
    res1=checkBallotFlipped(bal1,tpl0)
    if res0[2]<res1[2]:
        return (res0,checkBallotFlipped(bal1,tpl1),(0,1))
    else:
        return (res1,checkBallotFlipped(bal0,tpl1),(1,0))

def checkBallotFlipped(I,Iref,verbose=False):
    rszFac=sh.resizeOrNot(I.shape,sh.FLIP_CHECK_HEIGHT)
    Iref1=sh.fastResize(Iref,rszFac)
    I1=sh.fastResize(I,rszFac)
    IR=sh.fastFlip(I1)
    (H,Io,err)=imagesAlign(I1,Iref1,type='translation')
    (HR,IoR,errR)=imagesAlign(IR,Iref1,type='translation')
    
    if(verbose):
        print 'flip margin: ', err, errR
        
    if err>errR:
        return (True, sh.fastFlip(I),errR);
    else:
        return (False, I, err);

