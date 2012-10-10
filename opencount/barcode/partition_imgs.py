import os, sys, pdb, traceback, time, shutil
import cPickle as pickle    
import cv

import hart, diebold, sequoia

sys.path.append('..')
import grouping.partask as partask
import grouping.make_overlays as make_overlays
import pixel_reg.imagesAlign as imagesAlign

decode_fns = {'hart': hart.decode}

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
    
    for imgpath in imgpaths:
        barcodes, isflip, bbs = decode(imgpath)
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
        do_align = True
    else:
        do_align = False
    if 'do_cpyimg' in args:
        do_cpyimg = True
    else:
        do_cpyimg = False
    if 'just_grouping' in args:
        just_grouping = True
    else:
        just_grouping = False
    if args[-2] == 'load':
        grouping = pickle.load(open(args[-1], 'rb'))
    else:
        grouping = None

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
    #grouping = partition_imgs(imgpaths, vendor=vendor)
    if grouping == None:
        grouping = partask.do_partask(_do_partition_imgs, 
                                      imgpaths,
                                      _args=(vendor,),
                                      combfn="dict")
        try:
            os.makedirs(outdir)
        except:
            pass
        pickle.dump(grouping, open(os.path.join(outdir, 'grouping.p'), 'wb'))

    dur = time.time() - t
    print "...Finished partition_imgs ({0} s).".format(dur)

    if just_grouping:
        return

    print "Copying groups to outdir {0}...".format(outdir)
    t = time.time()
    for barcodes, group in grouping.iteritems():
        if len(group) == 1:
            continue
        elif "ERR0" in barcodes or "ERR1" in barcodes:
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
    print "Done."

if __name__ == '__main__':
    main()

