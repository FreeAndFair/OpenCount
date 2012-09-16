from os.path import join as pathjoin
from PIL import Image
from scipy import misc,ndimage
from scipy.stats import chi2
import shared as sh
from imagesAlign import *
import cProfile
import csv
import os
import string
import sys
import multiprocessing as mp
from wx.lib.pubsub import Publisher
import wx
from util import get_filename, create_dirs, is_image_ext, encodepath
import pickle
import fnmatch
import shutil
from random import random

def extractTargets(I,Iref,bbs,verbose=False):

    ''' Perform local alignment around each target, then crop out target  '''
    rszFac=sh.resizeOrNot(I.shape,sh.COARSE_BALLOT_REG_HEIGHT)
    IrefM=sh.maskBordersTargets(Iref,bbs);
    t0=time.clock();
    IO=imagesAlign(I,IrefM,fillval=1,type='translation',rszFac=rszFac)
    if(verbose):
        print 'coarse align time = ',time.clock()-t0,'(s)'

    H1=IO[0]; I1=IO[1]
    result = []
    pFac=7;

    for i in range(bbs.shape[0]):
        bb1=bbs[i,:]
        iLen=bb1[1]-bb1[0]
        jLen=bb1[3]-bb1[2]
        (bbOut,bbOff)=sh.expandBbs(bb1,I.shape[0],I.shape[1],pFac)

        Ic=sh.cropBb(I1,bbOut)
        IrefcNOMASK=sh.cropBb(Iref,bbOut)
        Irefc=sh.cropBb(IrefM,bbOut)

        rszFac=sh.resizeOrNot(Ic.shape,sh.LOCAL_PATCH_REG_HEIGHT)
        IO=imagesAlign(Ic,Irefc,fillval=1,rszFac=rszFac,type='rigid')

        Hc1=IO[0]; Ic1=IO[1]; err=IO[2]
        targ=np.copy(sh.cropBb(Ic1,bbOff))

        # unwind the transformations to get the global location of the target
        rOut_tr=pttransform(I,np.linalg.inv(H1),np.array([bbOut[2],bbOut[0],1]))
        rOff_tr=pttransform(Ic,np.linalg.inv(Hc1),np.array([bbOff[2],bbOff[0],1]))
        targLocGl=np.zeros(4)
        targLocGl[0]=round(rOut_tr[1]+rOff_tr[1])
        targLocGl[1]=round(rOut_tr[1]+rOff_tr[1]+iLen)
        targLocGl[2]=round(rOut_tr[0]+rOff_tr[0])
        targLocGl[3]=round(rOut_tr[0]+rOff_tr[0]+jLen)

        # weird bug in imsave where if the matrix is all ones, it saves as pure black
        targ[0,0]=.99
        result.append((bbs[i,4],targ,map(int,tuple(targLocGl)),err))
        #result.append((bbs[i,4],targ,map(int,tuple(targLocGl)),Idiff,Ic1))
        
    if(verbose):
        print 'total extract time = ',time.clock()-t0,'(s)'

    return result

def bbsInCell(bbs,i1,i2,j1,j2):
    bbOut=np.zeros((0,5))
    for bb in bbs:
        iCtr=(bb[0]+bb[1])/2.
        jCtr=(bb[2]+bb[3])/2.        
        if (iCtr>i1) & (iCtr<i2) & (jCtr>j1) & (jCtr<j2):
            bbOut=np.vstack((bbOut,bb))

    return bbOut

def extractTargetsRegions(I,Iref,bbs,vCells=4,hCells=4,verbose=False,balP=None):

    # parameter specified number of cells
    # for each cell, grab the targets that fall in the center
    #   compute super-region and pad

    ''' Perform local alignment around each target, then crop out target  '''
    '''
    if balP != None and '329_672_157_3_3' in balP:
        print "SPECIAL ONE"
        #pdb.set_trace()
    else:
        print "NORMAL"
        #pdb.set_trace()
    '''
    rszFac=sh.resizeOrNot(I.shape,sh.COARSE_BALLOT_REG_HEIGHT)
    IrefM=sh.maskBordersTargets(Iref,bbs,pf=0.05);
    t0=time.clock();
    IO=imagesAlign(I,IrefM,fillval=1,type='translation', rszFac=rszFac)
    #IO = (np.eye(3), I, 0.0)
    if(verbose):
        print 'coarse align time = ',time.clock()-t0,'(s)'
    #misc.imsave('_Ireg.png', IO[1])
    H1=IO[0]; I1=IO[1]
    result = []
    pFac=7;

    vStep=math.ceil(Iref.shape[0]/vCells);
    hStep=math.ceil(Iref.shape[1]/hCells);
    for i in range(vCells):
        for j in range(hCells):
            i1=i*vStep; i1=max(i1,0);
            i2=(i+1)*vStep; i2=min(i2,I1.shape[0]-1);
            j1=j*hStep; j1=max(j1,0);
            j2=(j+1)*hStep; j2=min(j2,I1.shape[1]-1);
            # grab all targets within this range
            bbs1=bbsInCell(bbs,i1,i2,j1,j2)
            if bbs1.size == 0:
                continue

            (bbOut,bbsOff)=sh.expandBbs(bbs1,I.shape[0],I.shape[1],pFac)

            Ic=sh.cropBb(I1,bbOut)
            IrefcNOMASK=sh.cropBb(Iref,bbOut)
            Irefc=sh.cropBb(IrefM,bbOut)
            #misc.imsave('_Ic.png', Ic)
            #misc.imsave('_IrefcNOMASK.png', IrefcNOMASK)
            #misc.imsave('_Irefc.png', Irefc)
            rszFac=sh.resizeOrNot(Ic.shape,sh.LOCAL_PATCH_REG_HEIGHT)
            IO=imagesAlign(Ic,Irefc,fillval=1,rszFac=rszFac,type='rigid')
            Hc1=IO[0]; Ic1=IO[1]; err=IO[2]
            #misc.imsave('_Ic1.png', Ic1)
            #if balP != None and '329_672_157_3_3' in balP:
            #    pdb.set_trace()

            for k in range(bbsOff.shape[0]):
                bbOff1=bbsOff[k,:]
                iLen=bbOff1[1]-bbOff1[0]
                jLen=bbOff1[3]-bbOff1[2]
                targ=np.copy(sh.cropBb(Ic1,bbOff1))
                # unwind the transformations to get the global location of the target
                rOut_tr=pttransform(I,np.linalg.inv(H1),np.array([bbOut[2],bbOut[0],1]))
                rOff_tr=pttransform(Ic,np.linalg.inv(Hc1),np.array([bbOff1[2],bbOff1[0],1]))
                targLocGl=np.zeros(4)
                targLocGl[0]=round(rOut_tr[1]+rOff_tr[1])
                targLocGl[1]=round(rOut_tr[1]+rOff_tr[1]+iLen)
                targLocGl[2]=round(rOut_tr[0]+rOff_tr[0])
                targLocGl[3]=round(rOut_tr[0]+rOff_tr[0]+jLen)

                # weird bug in imsave where if the matrix is all ones, it saves as pure black
                result.append((bbs1[k,4],targ,map(int,tuple(targLocGl)),err))
        
    if(verbose):
        print 'total extract time = ',time.clock()-t0,'(s)'

    return result


def writeMAP(imgs, targetDir, targetDiffDir, targetMetaDir, imageMetaDir, balP, tplP, flipped):
    fullpath = encodepath(balP)
    targs = [fullpath+"."+str(uid)+".png" for uid,_,_,_ in imgs]
    to = os.path.join(imageMetaDir, fullpath)
    toWrite={"flipped": flipped, "targets":targs, "ballot": balP, "template": tplP}

    # store the grouping and local alignment error with the ballot metadata.
    # use this for quarintine purposes.
    #bbox_errs=[];
    #for uid,img,bbox,err in imgs:
    #    bbox_errs.append(err);

    #toWrite["bbox_errs"]=str(bbox_errs);
    pickle.dump(toWrite, open(to, "w"))

    for uid,img,bbox,Idiff in imgs:
        #misc.imsave(pathjoin(targetDir, fullpath+"."+str(uid)+'.png'),img*255.)
        sh.imsave(pathjoin(targetDir, fullpath+"."+str(uid)+'.png'),img)
        #Ic1[np.isnan(Ic1)]=0
        #misc.imsave(pathjoin(diffDir, fullpath+"."+str(int(uid))+'.png'),Ic1*255.)
        np.save(pathjoin(targetDiffDir, fullpath+"."+str(int(uid))+'.npy'),Idiff)
        metafile = pathjoin(targetMetaDir, fullpath+"."+str(uid))
        pickle.dump({'bbox':bbox}, open(metafile, "w"))

def debugWorker(job):
    (Iref, bbs, fIn, destDir, extractedMeta, contestMeta, f1) = job
    
    I=sh.standardImread(fIn)
    # trim down or expand I to fit Iref
    I1=np.ones([Iref.shape[0],Iref.shape[1]],dtype='float32')
    h=min(Iref.shape[0],I.shape[0])-1; w=min(Iref.shape[1],I.shape[1])-1;
    I1[1:h,1:w]=I[1:h,1:w]

    # check if ballot is flipped
    res=checkBallotFlipped(I1,Iref)
    Ichk=res[1]; flipped=res[0]
    # extract target and write result
    return (Ichk,extractTargets(Ichk,Iref,bbs))

def findOutliers(Errs,thr,N):

    sidx=np.argsort(Errs)
    sidx=sidx[::-1]

    qFlag=[]
    for i in range(len(sidx)):
        j = i+1
        found=False
        while ((j-i)<(N+1)) & (j < len(sidx)):

            dif=Errs[sidx[i]]-Errs[sidx[j]]
            if dif>thr:
                qFlag.append(sidx[i])
                found=True
                break
            else:
                j+=1

        if ~found:
            break
                
    return qFlag


def quarantineCheckMAP(jobs, targetDiffDir, quarantineCvr, project, imageMetaDir=[] ):

    # Algorithm.
    # Pick percentage p,e.g., .1%
    # Set D to range(x)
    # - Sort 
    # - from largest x, read in nearest neighbors
    # - if any neighbor is more than D distance away, flag outlier
    # - else: break

    print 'done. now computing quarantine info.'
    # identify outliers and quarantine if necessary
        
    diffList=os.listdir(targetDiffDir)
    ballot2targets={}
    # create Hash
    wx.CallAfter(Publisher().sendMessage, "signals.MyGauge.nextjob", len(diffList))

    for f1 in diffList:
        if fnmatch.fnmatch(f1,'*npy'):
            (f2,ext2)=os.path.splitext(f1)
            (f3,ext3)=os.path.splitext(f2)
            if ballot2targets.has_key(f3):
                ballot2targets[f3].append(f1)
            else:
                ballot2targets[f3]=[]
                ballot2targets[f3].append(f1)
        
        wx.CallAfter(Publisher().sendMessage, "signals.MyGauge.tick")

    # Voted ballots with no contests/voting targets will not be
    # found within diffList - thus, we have to add them in and
    # add dummy values
    img2bal = pickle.load(open(project.image_to_ballot, 'rb'))
    for votedpath in img2bal:
        voted_abspath = os.path.abspath(votedpath)
        enc_path = encodepath(voted_abspath)
        if enc_path not in ballot2targets:
            ballot2targets[enc_path] = []

    print 'Done w/ hash.'
    wx.CallAfter(Publisher().sendMessage, "signals.MyGauge.nextjob", len(jobs))

    # STEP 1: load in all error values
    ErrHash={}
    JobHash={}
    Errs=[];K=0

    for K in range(len(jobs)):
        # job := [tplL_i, bbsL_i, balL_i, targetDiri, targetDiffDir_i, targetMetaDir_i, imageMetaDira_i]
        job=jobs[K]
        balL=job[2]
        for balP in balL:
            # loop over jobs
            M1=[]; IDX1=np.empty(0);
            try:
                targList=ballot2targets[encodepath(balP)]
            except Exception as e:
                print e
                pdb.set_trace()
            for f1 in targList:
                (f2,npext)=os.path.splitext(f1)
                (foo,idx)=os.path.splitext(f2)
                idx=eval(idx[1:])
                Idiff=np.load(os.path.join(targetDiffDir,f1))

                M1.append(Idiff)
                IDX1=np.append(IDX1,idx)
                
            # sort idx to straighten out order
            sidx=np.argsort(IDX1)

            # index into hash based on template
            if len(imageMetaDir)==0:
                k1='default'
            else:
                # TODO: MUST READ IN TEMPLATE SIDE FROM RESULT FILE                
                meta=pickle.load(open(pathjoin(imageMetaDir,encodepath(balP)),'rb'))
                k1=str(meta['template'])
    
            if ErrHash.has_key(k1):
                Errs=ErrHash[k1]
                jList=JobHash[k1]
            else:
                Errs=np.zeros((0,len(sidx)))
                jList=[]

            Errs=np.vstack((Errs,np.squeeze(M1)[sidx]))
            jList.append(balP)
            ErrHash[k1]=Errs
            JobHash[k1]=jList
            K+=1
            wx.CallAfter(Publisher().sendMessage, "signals.MyGauge.tick")

    print 'Done reading in errs.'

    qFiles=[]
    # perform outlier check for each template (key)
    for key in ErrHash.keys():
        qFlag=[]
        Errs=ErrHash[key]
        jList=JobHash[key]
        thr=(np.max(Errs,0)-np.min(Errs,0))/2
        
        #NP=max(round(len(jList)*.001),0)
        NP=max(round(len(jList)*.01),0)
        # compute D for each dimension
        # for each dimension
        #   sort
        #   for worst one, compare to nearest k
        #     if any are outside d, flag, and move to next point
        #     else: break to next dim
        for i in range(Errs.shape[1]):
            Errs1=Errs[:,i]
            qFlag+=findOutliers(Errs1,thr[i],NP)

        for i in range(len(qFlag)):
            #print qFlag
            #print jList
            #print i
            qFiles.append(jList[qFlag[i]])

    qFiles=list(set(qFiles))

    print qFiles
    print quarantineCvr
    f = open(quarantineCvr, "a")
    for each in qFiles:
        f.write(str(each)+"\n")
    

def convertImagesWorkerMAP(job):
    # match to front-back
    # (list of template images, target bbs for each template, filepath for image,
    #  output for targets, output for quarantine info, output for extracted
    #(tplL, bbsL, balL, targetDir, targetDiffDir, targetMetaDir, imageMetaDir) = job
    print "START"
    (tplL, bbsL, balL, targetDir, targetDiffDir, targetMetaDir, imageMetaDir) = job
    t0=time.clock();

    # need to load the template images
    tplImL=[]
    for tplP in tplL:
        tplImL.append(sh.standardImread(tplP, flatten=True));

    # load images
    balImL=[]

    for b in balL:
        #if '329_672_157_3_3' in b:
        #    pdb.set_trace()
        balImL.append(sh.standardImread(b, flatten=True));

    print 'load bal: ', time.clock()-t0
    # check if ballot is flipped
    t0=time.clock();

    # TODO: extend to more than two pages at some point
    
    if len(tplImL)==1:
        balPerm=[checkBallotFlipped(balImL[0],tplImL[0])]
        order=[0]
    else:
        # tplImL, balImL is already in [front, back] order
        balPerm=associateTwoPage(tplImL,balImL)
        order=balPerm[2]

    print 'assoc bal: ', time.clock()-t0
    for idx in range(len(tplL)):
        print "tick", idx
        res=balPerm[idx]
        tpl=tplImL[idx]
        bbs=bbsL[idx]
        bal=res[1]; flipped=res[0]
        writeMAP(extractTargetsRegions(bal,tpl,bbs, balP=b), targetDir, targetDiffDir, 
                 targetMetaDir, imageMetaDir, balL[order[idx]], tplL[idx], flipped)
    print "DONE"

def convertImagesMasterMAP(targetDir, targetMetaDir, imageMetaDir, jobs, stopped, verbose=False):
    """ Called by both single and multi-page elections. Performs
    Target Extraction.
    Input:
        targetDir:
        targetMetaDir:
        imageMetaDir:
        list jobs: [[tmppaths_i, bbs_i, imgpaths_i, targetDir_i, targetDiffDir_i, imageMetaDir_i], ...]
        stopped:
    """
    targetDiffDir=targetDir+'_diffs'

    if os.path.exists(targetDir): shutil.rmtree(targetDir)
    if os.path.exists(targetDiffDir): shutil.rmtree(targetDiffDir)
    if os.path.exists(targetMetaDir): shutil.rmtree(targetMetaDir)
    if os.path.exists(imageMetaDir): shutil.rmtree(imageMetaDir)

    create_dirs(targetDir)
    create_dirs(targetDiffDir)
    create_dirs(targetMetaDir)
    create_dirs(imageMetaDir)

    nProc=sh.numProcs()
    #nProc = 1
    
    if nProc < 2:
        print 'using only 1 processes'
        # default behavior for non multiproc machines
        for job in jobs:
            if stopped():
                return False
            t0=time.clock();
            convertImagesWorkerMAP(job)
            print time.clock()-t0
    else:
        print 'using ', nProc, ' processes'
        pool=mp.Pool(processes=nProc)

        it = [False]
        def imdone(x):
            it[0] = True
            print "I AM DONE NOW!"
            
        pool.map_async(convertImagesWorkerMAP,jobs,callback=lambda x: imdone(it))
        while not it[0]:
            if stopped():
                pool.terminate()
                return False
            time.sleep(.1)

        pool.close()
        pool.join()

    print 'done.'
    return True

def convertImagesSingleMAP(bal2imgs, tpl2imgs, bal2tpl, csvPattern, targetDir, targetMetaDir, imageMetaDir, quarantineCvr, stopped, project, verbose=False):
    """ Starts the target-extraction routines. Preps some input, and then
    calls convertImagesMasterMAP. For single-page elections.
    """
    targetDiffDir=targetDir+'_diffs'

    qfile = open(quarantineCvr, 'r')
    qfiles = set([f.strip() for f in qfile.readlines()])

    print "...Prepping jobs"
    t = time.time()
    jobs = []
    for k in bal2imgs.keys():
        if k not in qfiles:
            tplP = bal2tpl[k]
            csvP = csvPattern % get_filename(tplP, NO_EXT=True)
            bbsL = [sh.csv2bbs(csvP)]
            tplL = [tplP]
            balL = bal2imgs[k]
            jobs.append([tplL, bbsL, balL, targetDir, targetDiffDir, targetMetaDir, imageMetaDir])
    dur = time.time() - t
    print "...finished Prepping jobs. ({0} s).".format(dur)
    worked = convertImagesMasterMAP(targetDir, targetMetaDir, imageMetaDir, jobs, stopped, verbose=verbose)
    if worked:
        quarantineCheckMAP(jobs,targetDiffDir,quarantineCvr,project,imageMetaDir=imageMetaDir)
    return worked

def fix_ballot_order(balL, proj):
    """ Using ballot_to_page, correct the image ordering (i.e. 'side0',
    'side1', ...
    Input:
        lst balL: [imgpath_i, ...]
        obj proj
    Output:
        [side0_path, side1_path, ...]
    """
    assert issubclass(type(balL), list)
    bal2page = pickle.load(open(os.path.join(proj.projdir_path,
                                             proj.ballot_to_page),
                           'rb'))
    out = [None]*len(balL)
    for imgpath in balL:
        if imgpath not in bal2page:
            print "Uh oh, imgpath not in bal2page."
            pdb.set_trace()
        assert imgpath in bal2page
        side = bal2page[imgpath]
        assert issubclass(type(side), int)
        out[side] = imgpath
    assert None not in out
    return out

def convertImagesMultiMAP_makejobs(args):
    (keyset, bal2imgs, tpl2imgs, bal2tpl, csvPattern, targetDir, targetMetaDir, imageMetaDir, quarantineCvr, stopped, project, qfiles, verbose) = args
    #print map(type, args)
    targetDiffDir=targetDir+'_diffs'
    jobs = []
    print 'going up to', len(keyset)
    for i, k in enumerate(keyset):
        if i%100 == 0: 
            print 'on image', i
        if k not in qfiles:
            baltpl = bal2tpl[k]
            #try:
            tplL=tpl2imgs[baltpl]
            #except Exception as e:
            #    print e
            #    pdb.set_trace()
            balL=bal2imgs[k]
            # correct ordering in balL, via ballot_to_page
            balL = fix_ballot_order(balL, project)
            bbsL=[]
            for tplP in tplL:
                csvP=csvPattern % get_filename(tplP, NO_EXT=True)
                bbsL.append(sh.csv2bbs(csvP));

            jobs.append([tplL, bbsL, balL, targetDir, targetDiffDir, targetMetaDir, imageMetaDir])
    return jobs

def hack_stopped():
    return False
    
def convertImagesMultiMAP(bal2imgs, tpl2imgs, bal2tpl, csvPattern, targetDir, targetMetaDir, imageMetaDir, quarantineCvr, stopped, project,verbose=False):
    targetDiffDir=targetDir+'_diffs'

    qfile = open(quarantineCvr, 'r')
    qfiles = set([f.strip() for f in qfile.readlines()])

    num = sh.numProcs()
    alljobs = bal2imgs.keys()

    tomap = [[] for _ in range(num)]
    i = 0
    for each in alljobs:
        tomap[i].append(each)
        i += 1
        i %= num
    
    print 'have tomap', map(len, tomap)
    jobs = [(keyset, bal2imgs, tpl2imgs, bal2tpl, csvPattern, targetDir, targetMetaDir, imageMetaDir, quarantineCvr, hack_stopped, project, qfiles, verbose) for keyset in tomap]
    print 'have jobs'

    pool = mp.Pool(processes=num)
    print 'start map to generate jobs'
    jobs = pool.map(convertImagesMultiMAP_makejobs, jobs)
    jobs = [x for y in jobs for x in y]
    print 'start real convert map'

    worked = convertImagesMasterMAP(targetDir, targetMetaDir, imageMetaDir, jobs, stopped, verbose=verbose)
    if worked:
        quarantineCheckMAP(jobs,targetDiffDir,quarantineCvr,project,imageMetaDir=imageMetaDir)
    return worked

