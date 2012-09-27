import sys, os, pdb, time, traceback, csv
import cPickle as pickle

from os.path import join as pathjoin

import pixel_reg.shared as shared
import pixel_reg.doExtract as doExtract

"""
Yolo bad target extract paths:

yolo_s2_074/yolo_s2_074-020.png
yolo_s2_074/yolo_s2_074-044.png
yolo_s3_086/yolo_s3_086-032.png
yolo_s2_074/yolo_s2_074-083.png
yolo_s2_074/yolo_s2_074-007.png
yolo_s2_074/yolo_s2_074-027.png
yolo_s2_074/yolo_s2_074-012.png
yolo_s3_086/yolo_s3_086-035.png
yolo_s3_071/yolo_s3_071-099.png
yolo_s2_074/yolo_s2_074-046.png

yolo_s4_006/yolo_s4_006-219.png
    Only the state senator contest targets were misaligned

Usage:
    To reproduce Yolo bad target extraction:

    python do_single_textract.py /home/arya/opencount/opencount/projects/Yolo_2012 \
/media/data1/audits2012_straight/yolo/votedballots/yolo_s2_074/yolo_s2_074-020.png bad_out

This will do target extraction on yolo_s2_074-020.png, and dump it to bad_out/.
"""

def main():
    args = sys.argv[1:]
    projdir = args[0]
    votedpath = args[1]
    outdir = args[2]
    
    bal2imgs = pickle.load(open(pathjoin(projdir, 'ballot_to_images.p'), 'rb'))
    tpl2imgs = pickle.load(open(pathjoin(projdir, 'template_to_images.p'), 'rb'))
    img2bal = pickle.load(open(pathjoin(projdir, 'image_to_ballot.p'), 'rb'))

    fh=open(pathjoin(projdir, 'grouping_results.csv'))
    dreader=csv.DictReader(fh)
    bal2tpl={}
    print "Now load quarantined data"
    if os.path.exists(pathjoin(projdir, 'quarantined.csv')):
        qfile = open(pathjoin(projdir, 'quarantined.csv'), 'r')
        qfiles = set([f.strip() for f in qfile.readlines()])
        qfile.close()
    else:
        qfiles = set()
    print "Now process them all"
    for row in dreader:
        sample = os.path.abspath(row['samplepath'])
        if sample not in qfiles:
            bal2tpl[sample]=row['templatepath']
    fh.close()

    csvPattern = pathjoin(projdir, 'target_locations', '%s_targetlocs.csv')

    t_imgs = pathjoin(outdir, 'extracted')
    t_diff = pathjoin(outdir, 'extracted_diff')
    t_meta = pathjoin(outdir, 'extracted_metadata')
    b_meta = pathjoin(outdir, 'ballot_metadata')

    try:
        os.makedirs(t_imgs)
    except:
        pass
    try:
        os.makedirs(t_diff)
    except:
        pass
    try:
        os.makedirs(t_meta)
    except:
        pass
    try:
        os.makedirs(b_meta)
    except:
        pass
    
    # 0.) Set up job
    jobs = []
    tplP = bal2tpl[os.path.abspath(votedpath)]
    csvP = csvPattern % os.path.splitext(os.path.split(tplP)[1])[0]
    bbsL = [shared.csv2bbs(csvP)]
    tplL = [tplP]
    balL = bal2imgs[os.path.abspath(votedpath)]
    voteddir_root = '/media/data1/audits_2012/yolo/votedballots/'
    jobs.append([tplL, bbsL, balL, t_imgs, t_imgs+"_diff", t_meta, b_meta])
    
    '''
    res = doExtract.convertImagesSingleMAP(bal2imgs, tpl2imgs, bal2tpl, img2bal,
                                           csvPattern, 
                                           t_imgs, t_meta, b_meta,
                                           pathjoin(projdir, 'quarantined.csv'),
                                           lambda: False,
                                           None)
    '''
    doExtract.convertImagesWorkerMAP(jobs[0])
    
if __name__ == '__main__':
    main()
