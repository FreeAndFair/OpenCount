import os, sys, pdb, traceback, time, shutil, cProfile
import cPickle as pickle    
import cv

import hart, diebold, sequoia

sys.path.append('..')
import grouping.partask as partask
import grouping.make_overlays as make_overlays
import pixel_reg.imagesAlign as imagesAlign

decode_fns = {'hart': hart.decode}

# 5000 imgs (/media/data1/audits_2012/orange/votedballots): 
#     Single Image Read at a time (12 procs):
#         251.42 s    (0.05 s per ballot)
#     
# 500 imgs
#     Single Image Read at a time (12 procs):
#         5.647 s     (0.011 s per ballot)
#     Single Img (single proc):
#         39.34 s     (0.078 s per ballot)
#     50-imgs (12 proc):
#         5.845 s
#     50-imgs (1 proc):
#         39.162 s    (0.078 s per ballot)
#
# 100 imgs
#     Single img (12 proc):
#         1.243 s     (0.0124 s per ballot)
#     Single img (1 proc):
#         7.899 s     (0.078 s per ballot)
#     50-imgs (12 proc):
#         1.334 s     (0.0133 s per ballot)
#     50-imgs (1 proc):
#         8.007 s     (0.08 s per ballot)

def partition_imgs(imgpaths, vendor="hart"):
    """ Partition the images in IMGPATHS, assuming that the images
    are from the VENDOR.
    Input:
        list imgpaths: 
        str vendor: One of 'hart', 'diebold', 'sequoia'.
    Output:
        dict grouping. GROUPING is a dict of the following form:
            {(barcode_i, ...): [(imgpath_i, isflip_i, bbs_i), ...]}
    """
    grouping = {} 
    decode = decode_fns.get(vendor, None)
    if not decode:
        print "Unrecognized vendor:", vendor
        return None
    kwargs = {}
    if vendor == 'hart':
        kwargs['TOP_GUARD'] = cv.LoadImage('hart_topguard.png', cv.CV_LOAD_IMAGE_GRAYSCALE)
        kwargs['BOT_GUARD'] = cv.LoadImage('hart_botguard.png', cv.CV_LOAD_IMAGE_GRAYSCALE)
    
    for imgpath in imgpaths:
        barcodes, isflip, bbs = decode(imgpath, **kwargs)
        grouping.setdefault(barcodes, []).append((imgpath, isflip, bbs))
        #grouping.setdefault((barcodes[0],), []).append((imgpath, isflip, [bbs[0]]))
        
    return grouping

def _do_partition_imgs(imgpaths, (vendor,)):
    try:
        return partition_imgs(imgpaths, vendor=vendor)
    except Exception as e:
        traceback.print_exc()
        raise e

def main():
    def isimgext(f):
        return os.path.splitext(f)[1].lower() in ('.png', '.tif', '.tiff', '.jpg', '.jpeg')
    args = sys.argv[1:]
    imgsdir = args[0]
    vendor = args[1]
    outdir = args[2]
    try:
        N = int(args[3])
    except:
        N = -1
    if 'align' in args:
        # Align the barcodes when computing Min/Max overlays
        do_align = True
    else:
        do_align = False
    if 'do_cpyimg' in args:
        # Copy the entire images to OUTDIR (don't do this for large N!)
        do_cpyimg = True
    else:
        do_cpyimg = False
    if 'just_grouping' in args:
        # Just compute the barcodes + group, don't compute overlays
        just_grouping = True
    else:
        just_grouping = False
    if args[-2] == 'load':
        grouping = pickle.load(open(args[-1], 'rb'))
    else:
        grouping = None
    do_profile = True if 'profile' in args else False

    imgpaths = []
    cnt = 0
    for dirpath, dirnames, filenames in os.walk(imgsdir):
        for imgname in [f for f in filenames if isimgext(f)]:
            if N > 0 and cnt >= N:
                break
            imgpath = os.path.join(dirpath, imgname)
            imgpaths.append(imgpath)
            cnt += 1
        if N > 0 and cnt >= N:
            break
    print "Starting partition_imgs..."
    t = time.time()
    if do_profile:
        cProfile.runctx('partition_imgs(imgpaths, vendor=vendor)',
                        {}, {'imgpaths': imgpaths, 'vendor': vendor,
                             'partition_imgs': partition_imgs})
        return
    if grouping == None:
        grouping = partask.do_partask(_do_partition_imgs, 
                                      imgpaths,
                                      _args=(vendor,),
                                      combfn="dict", 
                                      N=None)
        try:
            os.makedirs(outdir)
        except:
            pass
        pickle.dump(grouping, open(os.path.join(outdir, 'grouping.p'), 'wb'))

    dur = time.time() - t
    print "...Finished partition_imgs ({0} s).".format(dur)

    print "Copying groups to outdir {0}...".format(outdir)
    t = time.time()
    errcount = 0
    for barcodes, group in grouping.iteritems():
        if len(group) == 1:
            errcount += 1 if ("ERR0" in barcodes or "ERR1" in barcodes) else 0
            continue
        elif "ERR0" in barcodes or "ERR1" in barcodes:
            #continue
            errcount += len(group)
            pass
        if just_grouping:
            continue
        bcs = '_'.join([thing for thing in barcodes if type(thing) == str])
        rootdir = os.path.join(outdir, bcs)
        try:
            os.makedirs(rootdir)
        except:
            pass
        Imins = [None for _ in barcodes]
        Imaxes = [None for _ in barcodes]
        Irefs = [None for _ in barcodes]

        for i, (imgpath, isflip, bbs) in enumerate(group):
            if do_cpyimg:
                imgname = os.path.split(imgpath)[1]
                outpath_foo = os.path.join(rootdir, imgname)
                shutil.copy(imgpath, outpath_foo)
            img = cv.LoadImage(imgpath, cv.CV_LOAD_IMAGE_GRAYSCALE)
            if isflip:
                cv.Flip(img, img, flipMode=-1)
            for j, bb in enumerate(bbs):
                outpath = os.path.join(rootdir, str(j), "{0}_{1}.png".format(i, j))
                try:
                    os.makedirs(os.path.split(outpath)[0])
                except:
                    pass
                x, y, w, h = bb
                cv.SetImageROI(img, (x, y, w, h))
                wbig, hbig = int(round(w*2.0)), int(round(h*2.0))
                bcBig = cv.CreateImage((wbig, hbig), img.depth, img.channels)
                cv.Resize(img, bcBig, interpolation=cv.CV_INTER_CUBIC)
                cv.SaveImage(outpath, bcBig)

                if Imins[j] == None:
                    Imins[j] = cv.CloneImage(bcBig)
                    Imaxes[j] = cv.CloneImage(bcBig)
                    if do_align:
                        Irefs[j] = make_overlays.iplimage2np(cv.CloneImage(bcBig)) / 255.0
                else:
                    bcBig_sized = make_overlays.matchsize(bcBig, Imins[j])
                    if do_align:
                        tmp_np = make_overlays.iplimage2np(cv.CloneImage(bcBig_sized)) / 255.0
                        H, Ireg, err = imagesAlign.imagesAlign(tmp_np, Irefs[j], fillval=0.2, rszFac=0.75)
                        Ireg *= 255.0
                        Ireg = Ireg.astype('uint8')
                        bcBig_sized = make_overlays.np2iplimage(Ireg)
                    cv.Min(bcBig_sized, Imins[j], Imins[j])
                    cv.Max(bcBig_sized, Imaxes[j], Imaxes[j])
        for idx, Imin in enumerate(Imins):
            Imax = Imaxes[idx]
            cv.SaveImage(os.path.join(rootdir, "_{0}_minimg.png".format(idx)), Imin)
            cv.SaveImage(os.path.join(rootdir, "_{0}_maximg.png".format(idx)), Imax)
            
    dur = time.time() - t
    print "...Finished Copying groups to outdir {0} ({1} s).".format(outdir, dur)
    print "Number of error ballots:", errcount
    print "Done."

if __name__ == '__main__':
    main()

