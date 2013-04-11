import sys, os, pdb, time, cPickle as pickle
import numpy as np, scipy.misc, scipy.ndimage

"""
A script that generates a synthetic testset to evaluate alignment
algorithms.
For each input voted ballot image, varying amounts of translation,
rotation, noise, and illumination differences are added to the image
and saved to disk.
"""

USAGE = """Usage:
    python create_testset.py [-h --help -help] 
                             [--trans X Y STEP] [--rot THETA STEP]
                             [--brightness AMT STEP]
                             [--noise MEAN STD]
                             IMGSDIR OUTDIR
--trans X Y STEP
    Adds horizontal/vertical amount of translation in the ranges
    [-X, X], [-Y, Y] with a step size given by STEP.
--rot THETA STEP
    Adds an amount of rotation in the range [-THETA, THETA] with step
    size given by STEP.
--brightness AMT STEP (default NONE)
    Adds a constant amount to each pixel intensity value (saturating
    at 0 and 255) in the range [-AMT, AMT] with stepsize STEP.
--noise MEAN STD (default NONE. Sensible values: MEAN=0, STD=10).
    Adds noise to the image by sampling from a gaussian distribution
    with mean MEAN, std deviation STD.
"""

def create_testset(imgpaths, outdir, 
                   X, Y, STEP_TRANS,
                   THETA, STEP_ROT,
                   MEAN, STD, BRIGHT_AMT, BRIGHT_STEP):
    """ Synthetically-perturbs each image I in IMGPATHS, then stores 
    it to OUTDIR.
    """
    def saveimg(I, idx, x, y, theta, bright_amt):
        rootdir = os.path.join(outdir, str(idx))
        if not os.path.exists(rootdir):
            os.makedirs(rootdir)
        outname = "img{0}_{1}_{2}_{3}.png".format(x, y, theta, bright_amt)
        outpath = os.path.join(rootdir, outname)
        scipy.misc.imsave(outpath, I)
        return outpath
    cnt = 0
    src2dsts = {} # maps {str src_imP: [(str dst_imP, x, y, theta, bright_amt), ...]
    dst2src = {} # maps {str dst_imP: str src_imP}
    for idx, imgpath in enumerate(sorted(imgpaths)):
        I = scipy.misc.imread(imgpath, flatten=True)
        for x in range(-X, X, STEP_TRANS):
            Ix = np.zeros(I.shape, dtype=I.dtype)
            if x == 0:
                Ix[:,:] = I
            elif x < 0:
                Ix[:,:x] = I[:,-x:]
            else:
                Ix[:,x:] = I[:,:-x]
            for y in range(-Y, Y, STEP_TRANS):
                Ixy = np.zeros(I.shape, dtype=I.dtype)
                if y == 0:
                    Ixy[:,:] = Ix
                elif y < 0:
                    Ixy[:y,:] = Ix[-y:,:]
                else:
                    Ixy[y:,:] = Ix[:-y,:]
                for theta in np.linspace(-THETA, THETA, num=(2*THETA / STEP_ROT), endpoint=True):
                    Ixyt = scipy.ndimage.rotate(Ixy, theta, reshape=False)
                    brightamt_iter = np.linspace(-BRIGHT_AMT, BRIGHT_AMT, num=(2*BRIGHT_AMT / BRIGHT_STEP), endpoint=True) if BRIGHT_AMT != None else [0]
                    for bright_amt in brightamt_iter:
                        Ixytb = Ixyt + bright_amt
                        if MEAN != None and STD != None:
                            noise = np.random.normal(MEAN, STD, size=(Ixytb.shape))
                            Ixytb += noise
                        Ixytb[np.where(Ixytb < 0)] = 0
                        Ixytb[np.where(Ixytb > 255)] = 255
                        outpath = saveimg(Ixytb, idx, x, y, theta, bright_amt)
                        src2dsts.setdefault(imgpath, []).append((outpath, x, y, theta, bright_amt))
                        dst2src[outpath] = imgpath
                        cnt += 1
    return cnt, src2dsts, dst2src

def parse_args(args):
    try:
        i = args.index('--trans')
        X, Y, STEP_TRANS = int(args[i+1]), int(args[i+2]), int(args[i+3])
    except:
        X, Y, STEP_TRANS = 12, 12, 4
    try:
        i = args.index('--rot')
        THETA, STEP_ROT = float(args[i+1]), float(args[i+2])
    except:
        THETA, STEP_ROT = 1.2, 0.4
    try:
        i = args.index('--noise')
        MEAN, STD = float(args[i+1]), float(args[i+2])
    except:
        MEAN, STD = None, None
    try:
        i = args.index('--brightness')
        BRIGHT_AMT, BRIGHT_STEP = int(args[i+1]), int(args[i+2])
    except:
        BRIGHT_AMT, BRIGHT_STEP = None, None

    imgsdir, outdir = args[-2], args[-1]
    return (X, Y, STEP_TRANS, THETA, STEP_ROT, MEAN, STD,
            BRIGHT_AMT, BRIGHT_STEP,
            imgsdir, outdir)

def main():
    args = sys.argv[1:]
    if '-h' in args or '--help' in args or '-help' in args:
        print USAGE
        exit(0)
    
    X, Y, STEP_TRANS, THETA, STEP_ROT, MEAN, STD, BRIGHT_AMT, BRIGHT_STEP, imgsdir, outdir = parse_args(args)
    
    imgpaths = []
    if not os.path.isdir(imgsdir):
        imgpaths = [imgsdir]
    else:
        for dirpath, dirnames, filenames in os.walk(imgsdir):
            for imgname in [f for f in filenames if f.lower().endswith('.png')]:
                imgpath = os.path.join(dirpath, imgname)
                imgpaths.append(imgpath)
    
    print "...Creating testset from {0} to {1}...".format(imgsdir, outdir)
    t = time.time()
    cnt, src2dsts, dst2src,  = create_testset(imgpaths, outdir, 
                                              X, Y, STEP_TRANS, 
                                              THETA, STEP_ROT,
                                              MEAN, STD, 
                                              BRIGHT_AMT, BRIGHT_STEP)
    dur = time.time() - t
    print "...Done ({0}s). Saved {1} images to {2}.".format(dur, cnt, outdir)
    pickle.dump(src2dsts, open(os.path.join(outdir, 'src2dsts.p'), 'wb'))
    pickle.dump(dst2src, open(os.path.join(outdir, 'dst2src.p'), 'wb'))
        
if __name__ == '__main__':
    main()
