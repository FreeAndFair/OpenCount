import sys, os, pdb, time, cPickle as pickle, math, traceback
import numpy as np, scipy.misc, cv2
import global_align
"""
A script that evaluates alignment methods against a synthetic test set
(generated from eval_create_testset.py).
"""

USAGE = """Usage:
    python eval_align.py [-h --help -help] [--debug]
                         [--align STRAT] TESTSET
--align STRAT
    Which alignment strategy to use. One of:
        align_cv, align_ncc
"""

STRAT_CV = 'align_cv'
STRAT_NCC = 'align_ncc'

def eval_testset(testsetdir, align_strat=STRAT_CV, debug=False):
    src2dsts = pickle.load(open(os.path.join(testsetdir, 'src2dsts.p'), 'rb'))
    dst2src = pickle.load(open(os.path.join(testsetdir, 'dst2src.p'), 'rb'))
    errs = []
    errs_x, errs_y, errs_theta = [], [], []
    t = time.time()
    for refimgpath, dst_tpls in src2dsts.iteritems():
        Iref = scipy.misc.imread(refimgpath, flatten=True)
        for (dstimgpath, x, y, theta, bright_amt) in dst_tpls:
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
                theta_ = -math.degrees(math.acos(H_new[0,0])) # TODO: Verify this!
                x_err = x_ - x
                y_err = y_ - y
                theta_err = theta_ - theta
            else:
                raise Exception("Unrecognized alignment strategy: {0}".format(align_strat))
            try:
                err = np.sum(np.abs(Ireg - Iref)) / float(Ireg.shape[0] * Ireg.shape[1])
            except:
                traceback.print_exc()
                pdb.set_trace()
            errs_x.append(x_err)
            errs_y.append(y_err)
            errs_theta.append(theta_err)
            if debug:
                print H
                print "(Found): x={0} y={1} theta={2}".format(x_, y_, theta_)
                print "(Expect): x={0}, y={1}, theta={2}".format(x,y,theta)
                print "Err:", err
                print "x_err={0} y_err={1} theta_err={2}".format(x_err, y_err, theta_err)
                A = Iref + Ireg
                scipy.misc.imsave("overlay.png", A)
                pdb.set_trace()
            errs.append(err)
    dur = time.time() - t
    return errs, errs_x, errs_y, errs_theta, dur

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
    testsetdir = args[-1]
    
    print "...Evaluating the testset at: {0} (with align_strat={1})".format(testsetdir, align_strat)
    errs, errs_x, errs_y, errs_theta, dur = eval_testset(testsetdir, align_strat=align_strat, debug=debug)
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
