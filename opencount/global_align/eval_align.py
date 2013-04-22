import sys, os, pdb, time, cPickle as pickle, math, traceback
import numpy as np, scipy.misc, cv2
import global_align
"""
A script that evaluates alignment methods against a synthetic test set
(generated from eval_create_testset.py).
"""

USAGE = """Usage:
    python eval_align.py [-h --help -help] [--debug] [--cache]
                         [--align STRAT] TESTSET
--align STRAT
    Which alignment strategy to use. One of:
        align_cv, align_ncc
--cache
    Cache the reference images in-memory (uses more memory).
"""

STRAT_CV = 'align_cv'
STRAT_NCC = 'align_ncc'

def eval_testset(testsetdir, align_strat=STRAT_CV,
                 debug=False, CACHE_IMG=None):
    # dict SRC2DSTS: {str srcimgP: tuple (dstimgP, x, y, theta, bright_amt)}
    src2dsts = pickle.load(open(os.path.join(testsetdir, 'src2dsts.p'), 'rb'))
    # dict DST2SRC: {str dstimgP: str srcimgP}
    dst2src = pickle.load(open(os.path.join(testsetdir, 'dst2src.p'), 'rb'))
    errs = []
    errs_x, errs_y, errs_theta = [], [], []
    errs_map = {} # maps {(str srcimgP, str dstimgP): (tuple P_EXPECTED, tuple P_ESTIMATED)}
    t_start = time.time()
    def load_img(imP):
        I = CACHE_IMG.get(imP, None) if CACHE_IMG else None
        if I == None:
            I = scipy.misc.imread(imP, flatten=True)
            if CACHE_IMG != None:
                CACHE_IMG[imP] = I
        return I
    N = len(src2dsts) * len(src2dsts[src2dsts.keys()[0]])
    i = 0
    t_prev = time.time()
    step = N / 10.0 # Print out at each 10% interval
    def update_status():
        t_cur = time.time()
        dur_step = t_cur - t_prev
        n_remain = N - i
        est = ((dur_step / float(step)) * n_remain) / 60.0 # est. time left in minutes
        if i % step == 0:
            print "...{0:.2f}% complete... ({1} left, {2:.4f} min. left)".format(100.0 * (i / float(N)), n_remain, est)

    for refimgpath, dst_tpls in src2dsts.iteritems():
        Iref = load_img(refimgpath)
        for (dstimgpath, x, y, theta, bright_amt) in dst_tpls:
            update_status()
            t_prev = time.time()
            I = scipy.misc.imread(dstimgpath, flatten=True)
            if align_strat == STRAT_CV:
                I = I.astype('uint8')
                Iref = Iref.astype('uint8') if Iref.dtype != 'uint8' else Iref
                H, Ireg, err_ = global_align.align_cv(I, Iref)
                x_ = -H[0,2]
                y_ = -H[1,2]
                theta_ = -math.degrees(H[0,1])
                x_err = x_ - x
                y_err = y_ - y
                theta_err = theta_ - theta
            elif align_strat == STRAT_NCC:
                H, Ireg, err = global_align.align_image(I, Iref)
                # Note: This is borrowed from shared.imtransform. Is this
                #       doing the right thing?
                T0=np.eye(3); T0[0,2]=Ireg.shape[1]/2.0; T0[1,2]=Ireg.shape[0]/2.0
                T1=np.eye(3); T1[0,2]=-Ireg.shape[1]/2.0; T1[1,2]=-Ireg.shape[0]/2.0
                H_new = np.dot(np.dot(T0,H),T1)
                x_ = -H_new[0,2] # TODO: Verify this!
                y_ = -H_new[1,2] # TODO: Verify this!
                H00 = min(max(H_new[0,0], -1.0), 1.0) # clamp to [-1.0, 1.0] to avoid numerical instability
                theta_= -math.degrees(math.acos(H00)) # TODO: Verify this!
                x_err = x_ - x
                y_err = y_ - y
                theta_err = theta_ - theta
            else:
                raise Exception("Unrecognized alignment strategy: {0}".format(align_strat))
            err = np.mean(np.abs(Ireg - Iref)) # L1 error of pixel intensity values
            errs_x.append(x_err)
            errs_y.append(y_err)
            errs_theta.append(theta_err)
            P_expected = (x, y, theta, bright_amt)
            P_estimate = (x_, y_, theta_, None)
            errs_map.setdefault((refimgpath, dstimgpath), []).append((P_expected, P_estimate))
            errs.append(err)
            if debug:
                print H
                print "(Found): x={0} y={1} theta={2}".format(x_, y_, theta_)
                print "(Expect): x={0}, y={1}, theta={2}".format(x,y,theta)
                print "Err:", err
                print "x_err={0} y_err={1} theta_err={2}".format(x_err, y_err, theta_err)
                A = Iref + Ireg
                scipy.misc.imsave("overlay.png", A)
                pdb.set_trace()
            i += 1

    dur_total = time.time() - t_start
    return errs_map, errs, errs_x, errs_y, errs_theta, dur_total

def main():
    args = sys.argv[1:]
    if '-h' in args or '--help' in args or '-help' in args:
        print USAGE
        exit(0)
    try:
        align_strat = args[args.index('--align')+1]
    except:
        align_strat = STRAT_CV
    debug = '--debug' in args
    do_cache = '--cache' in args
    testsetdir = args[-1]

    CACHE_IMG = {} if do_cache else None
    print "...Evaluating the testset at: {0} (with align_strat={1})".format(testsetdir, align_strat)
    errs_map, errs, errs_x, errs_y, errs_theta, dur = eval_testset(testsetdir, align_strat=align_strat, debug=debug, CACHE_IMG=CACHE_IMG)
    print "...Done Evaluating ({0} secs)".format(dur)
    print "    Average per Alignment: {0} secs".format(dur / float(len(errs)))
    print
    print "(Pixel Error) Mean={0}    Std={1}    ({2} total alignments)".format(np.mean(errs), np.std(errs), len(errs))
    print "(X Error) Mean={0}    Std={1}".format(np.mean(errs_x), np.std(errs_x))
    print "(Y Error) Mean={0}    Std={1}".format(np.mean(errs_y), np.std(errs_y))
    print "(Theta Error) Mean={0}    Std={1}".format(np.mean(errs_theta), np.std(errs_theta))
    
    pdb.set_trace()
if __name__ == '__main__':
    main()
