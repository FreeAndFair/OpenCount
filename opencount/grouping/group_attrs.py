import pdb
import copy
import numpy as np
import cv


import common
from pixel_reg import shared

_i = 0


def temp_match(I, bb, imList, bbSearch=None, bbSearches=None, rszFac=0.75,
               padSearch=0.75, padPatch=0.0):
    bb = list(bb)
    if bbSearch is not None:
        bbSearch = list(bbSearch)
    if bbSearches is not None:
        bbSearches = list(bbSearches)
    matchList = []  # (filename, left,right,up,down)

    I = np.round(shared.fastResize(I, rszFac) * 255.) / 255

    bb[0] = bb[0] * rszFac
    bb[1] = bb[1] * rszFac
    bb[2] = bb[2] * rszFac
    bb[3] = bb[3] * rszFac
    [bbOut, bbOff] = shared.expand(bb[0], bb[1], bb[2], bb[3], I.shape[
                                   0], I.shape[1], padPatch)
    patchFoo = I[bbOut[0]:bbOut[1], bbOut[2]:bbOut[3]]

    patch = patchFoo[bbOff[0]:bbOff[1], bbOff[2]:bbOff[3]]

    if bbSearch is not None:
        bbSearch[0] = bbSearch[0] * rszFac
        bbSearch[1] = bbSearch[1] * rszFac
        bbSearch[2] = bbSearch[2] * rszFac
        bbSearch[3] = bbSearch[3] * rszFac

    for cur_i, imP in enumerate(imList):
        if bbSearches is not None:
            bbSearch = map(lambda c: c * rszFac, bbSearches[cur_i])
        I1 = shared.standardImread(imP, flatten=True)
        I1 = np.round(shared.fastResize(I1, rszFac) * 255.) / 255.
        # crop to region if specified
        if bbSearch is not None:
            [bbOut1, bbOff1] = shared.expand(bbSearch[0], bbSearch[1],
                                             bbSearch[2], bbSearch[3],
                                             I1.shape[0], I1.shape[1], padSearch)
            I1 = I1[bbOut1[0]:bbOut1[1], bbOut1[2]:bbOut1[3]]
        if I1.shape[0] < patch.shape[0] or I1.shape[1] < patch.shape[1]:
            w_big = max(I1.shape[0], patch.shape[0])
            h_big = max(I1.shape[1], patch.shape[1])
            I1_big = np.zeros((w_big, h_big)).astype('float32')
            I1_big[0:I1.shape[0], 0:I1.shape[1]] = I1
            I1 = I1_big

        patchCv = cv.fromarray(np.copy(patch))
        ICv = cv.fromarray(np.copy(I1))
        outCv = cv.CreateMat(abs(I1.shape[
                             0] - patch.shape[0]) + 1, abs(I1.shape[1] - patch.shape[1]) + 1, cv.CV_32F)

        cv.MatchTemplate(ICv, patchCv, outCv, cv.CV_TM_CCOEFF_NORMED)
        Iout = np.asarray(outCv)

        Iout[Iout == 1.0] = 0.995  # opencv bug

        score1 = Iout.max()  # NCC score
        YX = np.unravel_index(Iout.argmax(), Iout.shape)
        i1 = YX[0]
        i2 = YX[0] + patch.shape[0]
        j1 = YX[1]
        j2 = YX[1] + patch.shape[1]
        (err, diff, Ireg) = shared.lkSmallLarge(
            patch, I1, i1, i2, j1, j2, minArea=np.power(2, 17))
        score2 = err / diff.size  # pixel reg score
        if bbSearch is not None:
            matchList.append((imP, score1, score2, Ireg,
                              i1 + bbOut1[0], i2 + bbOut1[0],
                              j1 + bbOut1[2], j2 + bbOut1[2], rszFac))
        else:
            matchList.append((imP, score1, score2, Ireg,
                              i1, i2, j1, j2, rszFac))

    return matchList


def compute_exemplars_fullimg(mapping, MAXCAP=None):
    """ Given a mapping {str label: ((imgpath_i, bb_i),...)}, extracts a subset
    of the imgpaths {str label: (imgpath_i, ...)} such that these
    imgpaths are the best-describing 'exemplars' of the entire input
    mapping.
    NOTE: bb's here are in [y1,y2,x1,x2] format.
    Input:
        dict mapping: {label: ((imgpath_i, bb_i), ...)}
        int MAXCAP: Maximum number of exemplars per label (optional).
    Output:
        A (hopefully smaller) dict mapping {label: ((imgpath_i, bb_i), ...)}
    """
    def get_closest_ncclk(imgpath, img, bb, imgpaths2, bbs2):
        if bb is None:
            bb = [0, img.shape[0] - 1, 0, img.shape[1] - 1]
            bbs2 = None

        matches = temp_match(img, bb, imgpaths2, bbSearches=bbs2)

        if not matches:
            print "Uhoh, no matches found for imgpath {0}.".format(imgpath)
            pdb.set_trace()
            return 9999, bb

        matches = sorted(matches, key=lambda t: t[2])
        imgpath, bb, rszFac = (matches[0][0], matches[0][4:8], matches[0][8])
        bb = map(lambda c: int(round(c / rszFac)), bb)
        return (matches[0][2], bb)

    def closest_label(imgpath, bb, exemplars):
        bestlabel = None
        mindist = None
        bbBest = None
        img = shared.standardImread(imgpath, flatten=True)
        for label, tuples in exemplars.iteritems():
            imgpaths2, bbs2 = [], []
            for imgpath2, bb2 in tuples:
                imgpaths2.append(imgpath2)
                bbs2.append(bb2)
            closestdist, bbOut = get_closest_ncclk(
                imgpath, img, bb, imgpaths2, bbs2)
            if bestlabel is None or closestdist < mindist:
                bestlabel = label
                mindist = closestdist
                bbBest = bbOut
        return bestlabel, mindist, bbBest
    mapping = copy.deepcopy(mapping)
    exemplars = {}  # maps {str label: ((imgpath_i, bb_i), ...)}
    for label, tuples in mapping.iteritems():
        imgpaths = [t[0] for t in tuples]
        pathL, scoreL, idxL = common.get_avglightest_img(imgpaths)
        print "Chose starting exemplar {0}, with a score of {1}".format(pathL, scoreL)
        imgpath, bb = tuples.pop(idxL)
        exemplars[label] = [(imgpath, bb)]
    print "Making tasks..."
    tasks = make_tasks(mapping)
    init_len_tasks = len(tasks)

    # tasks = make_interleave_gen(*[(imgpath, bb) for (imgpath, bb) in
    # itertools.izip(imgpath, bbs)
    counter = {}  # maps {str label: int count}
    is_done = False
    while not is_done:
        is_done = True
        taskidx = 0
        while taskidx < len(tasks):
            if (init_len_tasks > 10) and taskidx % (init_len_tasks / 10) == 0:
                print "."
            label, (imgpath, bb) = tasks[taskidx]
            if MAXCAP is not None:
                cur = counter.get(label, 0)
                if cur >= MAXCAP:
                    taskidx += 1
                    continue

            bestlabel, mindist, bbOut = closest_label(imgpath, bb, exemplars)
            if label != bestlabel:
                print "...for label {0}, found new exemplar {1}.".format(label, imgpath)
                tasks.pop(taskidx)
                exemplars[label].append((imgpath, bb))
                if label not in counter:
                    counter[label] = 1
                else:
                    counter[label] += 1

                is_done = False
            else:
                taskidx += 1
    return exemplars


def make_tasks(mapping):
    """ Returns a series of tasks, where each task alternates by label,
    so that we try, say, '1', then '2', then '3', instead of trying
    all the '1's first, followed by all the '2's, etc. Helps to keep
    the running time down.
    Input:
        dict mapping: maps {str label: ((imgpath_i, bb_i), ...)}
    """
    label_tasks = []  # [[...], [...], ...]
    for label, tuples in mapping.iteritems():
        tups = []
        for (imgpath, bb) in tuples:
            tups.append((label, (imgpath, bb)))
        label_tasks.append(tups)
    return interleave(*label_tasks)


def interleave(*lsts):
    result = []
    for idx in range(0, max(len(lst) for lst in lsts)):
        for lst in lsts:
            try:
                result.append(lst[idx])
            except IndexError:
                continue
    return result
