import os, sys, time

import numpy as np, scipy, scipy.misc

sys.path.append('..')

from pixel_reg.imagesAlign import imagesAlign, imtransform
import pixel_reg.shared as shared

"""
A test script to globally-align an input directoy of ballots, and output a 
basic overlay for each alignment. 

Usage:
    python test_globalalign.py INPUT_DIR OUTPUT_DIR [IDX]

where INPUT_DIR is a directory containing subdirectories D_i, where
each D_i should contain ballots with the same layout. (Alternately,
INPUT_DIR can point to a directory containing only images). 
For instance, an example such directory is at:

    /home/ekim/opencount/test-ballots-ek/oc_badglobalalign/

For instance, oc_badglobalalign/10002940100056/ has three ballots A, B,
and C. If I run:

    python test_globalalign.py oc_badglobalalign/10002940100056 outdir 

The script will do the following for every pair of images I0, I1:

    Align I0 to I1 to get I0'. Save overlay(I0', I1) to outdir. 

Finally, to run a large-scale 'unit test' (say, every single partition in 
oc_badglobalalign), you can do:

    python test_globalalign.py oc_badglobalalign/ unittest_out
"""

# NOTE: Currently, the global alignment procedure is the 'global_align'
#       below, which crops 20% of the image on every side before running
#       Kai's imagesAlign function.
#       This improved things for Orange County, but made things worse for 
#       Yolo.
#       One idea is to reduce the cropping (say, ~2-5% or so).
#       Another idea is to try running alignment a few times on smaller
#       patches of the ballot to get a few candidate alignments H_i, and
#       choosing the H_i that minimizes the alignment error (or some other
#       metric).
#
#       If you make your changes to 'global_align' when running your experiments,
#       you won't have to change any of the other code. 

SAMPLE_SIZE = 5

def global_align(Iref, imgpaths, rsz):
    """ Using IREF as a reference, aligns every image in IMGPATHS to IREF.
    Input:
        IplImage IREF: An OpenCV IplImage instance, i.e. the reference
            image we will align against.
        list IMGPATHS: A list of image paths.
    Output:
        list IOUTS. [(str imgpath, nparray H, IplImage Ireg, float err), ...].
    IOUTS:
        A list of tuples containing the aligned image Ireg, along with
        the discovered transformation matrix H, alignment error ERR, and
        the path to the ballot image IMGPATH.
    """
    # TODO: Take the alignment errors to be normally distributed.
    # If an error output is atleast two standard devs away from the mean,
    # then recompute the alignment(different resize factors, sliding scale)
    #
    # Keep a running estimator for mean and standard deviations.
    # Possible scenario: The first alignment produces an error(or first couple)
    # at such a large scale it shifts the distribution mean. Perhaps consider,
    # a first pass alignment, and store a mapping from image to alignment error,
    # second pass we can determine which alignments need to be recomputed.
	
    Iouts = [] # [(imgpath, H, Ireg, err), ...]
    sample_outs = {}
    counter = 0    
    mu = None
    sigma = None
    accum = 0
    flag = False
   
    def align_simple(imgpath):
	t1 = time.time()
	I = shared.standardImread(imgpath, flatten=True)
        Icrop = cropout_stuff(I, 0.02, 0.02, 0.02, 0.02)
        H, Ireg, err = imagesAlign(Icrop, Iref, type='rigid', rszFac=rsz)

        Ireg = np.nan_to_num(Ireg)
        Iouts.append((imgpath, H, Ireg, err, rsz))
	t2 = time.time()

    def align_thorough(imgpath):
	t1 = time.time()
	I = shared.standardImread(imgpath, flatten=True)
	Icrop = cropout_stuff(I, 0.02, 0.02, 0.02, 0.02)
	H, Iref, err = imagesAlign(Icrop, Iref, type='rigid', rszFac = 1.0)
	Ireg = np.nan_to_num(Ireg)
	Iouts.append((imgpath, H, Ireg, err, rsz))
	t2 = time.time()
    

    for imgpath in imgpaths:
	if counter < SAMPLE_SIZE:
		t1 = time.time()
		I = shared.standardImread(imgpath, flatten=True)
		Icrop = cropout_stuff(I, 0.02, 0.02, 0.02, 0.02)
		H, Ireg, err = imagesAlign(Icrop, Iref, type='rigid', rszFac=rsz)
	        accum += err
                sample_outs[imgpath] = err
		counter += 1
		print "On counter=", counter
	else:
		if flag == False:
			mu = accum / float(SAMPLE_SIZE)
			print "mu:", mu
			accum2 = 0
			errs = sample_outs.values()
			print errs
			for i in xrange(SAMPLE_SIZE):
				accum2 += (errs[i] - mu)**2
				sigma = accum2 * (1 / (SAMPLE_SIZE - 1))
				print "sigma:", sigma
		
			lb = mu - 2*sigma
			ub = mu + 2*sigma
		
			print "lb: ", lb
			print "ub: ", ub	
		
			for x in sample_outs.iteritems():
				if x.value() > ub or x.value() < lb:
					align_thorough(x.key())
				else:
					align_simple(x.key())
			flag = True
		align_simple(imgpath)
			
    return Iouts

def cropout_stuff(I, top, bot, left, right):
    h, w = I.shape
    x1 = int(round(left*w))
    y1 = int(round(top*h))
    x2 = int(round(w - (right*w)))
    y2 = int(round(h - (bot*h)))
    Inew = I[y1:y2, x1:x2]
    return np.copy(Inew)

def do_aligning(imgpaths, outdir, idx, rsz):
    Iref_imgP = imgpaths.pop(idx)
    Iref_np = scipy.misc.imread(Iref_imgP, flatten=True)
    Iref = shared.standardImread(Iref_imgP, flatten=True)

    Iref_crop = cropout_stuff(Iref, 0.02, 0.02, 0.02, 0.02)

    Iouts = global_align(Iref_crop, imgpaths, rsz)

    ref_dir = os.path.join(outdir, 'ref')

    try:
        os.makedirs(ref_dir)
    except: pass

    Iref_imgname = os.path.split(Iref_imgP)[1]
    scipy.misc.imsave(os.path.join(ref_dir, Iref_imgname), Iref)

    for imgpath, H, Ireg_crop, err, rsz in Iouts:
        print "For imgpath {0}, err={1:.4f}, rsz={2}, H:".format(imgpath, err, rsz)
        imgname = os.path.splitext(os.path.split(imgpath)[1])[0]
        I = scipy.misc.imread(imgpath, flatten=True)
        Hc = correctH(Ireg_crop, H)
        print Hc
        rot_H = Hc[:2, :2]
        trans_H = Hc[:2,2]
        """
        Note: imagesAlign's H is of the form:
            [a b X]
            [c d Y]
        where positive Y means go down, and positive X means to right (as usual).
        scipy's affine_transform's offset should be offset=(y,x), but where
            positive y is go UP, and positive x is to the LEFT.
        """
        #Itrans = scipy.ndimage.interpolation.affine_transform(I, rot_H, 
        #                                                      offset=(-trans_H[1], -trans_H[0]),
        #                                                      output_shape=I.shape)
        #Icv = cv.LoadImageM(imgpath, cv.CV_LOAD_IMAGE_GRAYSCALE)
        #Icv_trans = cv.CreateMat(Icv.rows, Icv.cols, Icv.type)
        #H_cv = cv.fromarray(Hc.astype(np.float32)[:2,:])
        #cv.WarpAffine(Icv, Icv_trans, H_cv)
        #Itrans = np.asarray(Icv)
        Itrans = imtransform(I, H)
        Itrans = np.nan_to_num(Itrans)
        
        outP = os.path.join(outdir, "{0}_err{1:.4f}.png".format(imgname, err))
        Ioverlay = np.zeros(Itrans.shape)
        Ioverlay[:,:] = Itrans
        Ioverlay[:,:] += Iref_np
        scipy.misc.imsave(outP, Ioverlay)
    
def correctH(I, H0):
    T0=np.eye(3); T0[0,2]=I.shape[1]/2.0; T0[1,2]=I.shape[0]/2.0
    T1=np.eye(3); T1[0,2]=-I.shape[1]/2.0; T1[1,2]=-I.shape[0]/2.0
    H=np.dot(np.dot(T0,H0),T1)
    return H

def start(args, rsz):
    #args = sys.argv[1:]
    imgsdir = args[0]
    outdir = args[1]
    try:
        idx = int(args[2])
        do_all_idxs = False
    except:
        do_all_idxs = True
    imgpaths_per_dir = []
    for dirpath, dirnames, filenames in os.walk(imgsdir):
        imgpaths = []
        for imgname in filenames:
            imgpaths.append(os.path.join(dirpath, imgname))
        if imgpaths:
            imgpaths_per_dir.append(imgpaths)
    if not do_all_idxs:
        for imgpaths in imgpaths_per_dir:
            parentdir = os.path.split(os.path.split(imgpaths[0])[0])[1]
            outdir_sub = os.path.join(outdir, parentdir, 'idx_{0}'.format(idx))
            do_aligning(imgpaths_per_dir[:], outdir_sub, idx, rsz)
    else:
        for imgpaths in imgpaths_per_dir:
            parentdir = os.path.split(os.path.split(imgpaths[0])[0])[1]
            outdir_sub = os.path.join(outdir, parentdir)
            for idx in xrange(len(imgpaths)):
                print '...doing idx {0}...'.format(idx)
                outdir_2 = os.path.join(outdir_sub, 'idx_{0}'.format(idx))
                do_aligning(imgpaths[:], outdir_2, idx, rsz)

if __name__ == '__main__':
    start(sys.argv[1:],0.15) # default rsz
