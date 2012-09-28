import os, sys, pdb, traceback, time, shutil
import hart, diebold, sequoia

decode_fns = {'hart': hart.decode}

def partition_imgs(imgpaths, vendor="hart"):
    """ Partition the images in IMGPATHS, assuming that the images
    are from the VENDOR.
    Input:
        list imgpaths: 
        str vendor: One of 'hart', 'diebold', 'sequoia'.
    Output:
        dict grouping. GROUPING is a dict of the following form:
            {(barcode_i, ...): [imgpath_i, ...]}
    """
    grouping = {} 
    decode = decode_fns.get(vendor, None)
    if not decode:
        print "Unrecognized vendor:", vendor
        return None
    
    for imgpath in imgpaths:
        barcodes = decode(imgpath)
        grouping.setdefault(barcodes, []).append(imgpath)
        
    return grouping

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
    grouping = partition_imgs(imgpaths, vendor=vendor)
    dur = time.time() - t
    print "...Finished partition_imgs ({0} s).".format(dur)

    print "Copying groups to outdir {0}...".format(outdir)
    t = time.time()
    for barcodes, group in grouping.iteritems():
        bcs = '_'.join(barcodes[:3])
        rootdir = os.path.join(outdir, bcs)
        try:
            os.makedirs(rootdir)
        except:
            pass
        for imgpath in group:
            imgname = os.path.split(imgpath)[1]
            outpath = os.path.join(rootdir, imgname)
            shutil.copy(imgpath, outpath)
    dur = time.time() - t
    print "...Finished Copying groups to outdir {0} ({1} s).".format(outdir, dur)
    print "Done."

if __name__ == '__main__':
    main()

