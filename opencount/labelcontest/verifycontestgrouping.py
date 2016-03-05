import wx
import sys
import os
import pdb

sys.path.append("..")
from util import pdb_on_crash
import grouping.common
import pixel_reg.imagesAlign as imagesAlign
import pixel_reg.shared as sh
import numpy as np
import scipy.misc
import multiprocessing as mp

import grouping.verify_overlays_new

TMP = "tmp/"


def merge_and_align(dat):
    i, group = dat
    print i
    # [merge(align(i, group[x:x+10])) for x in range(0,len(group),100)]
    return [merge(align(i, group))]


def translate(name):
    return os.path.join(TMP, os.path.abspath(name).replace("/", "~"))


@pdb_on_crash
def merge(args):
    all_files, all_images = args
    res = []
    for contest_paths, contest_images in zip(zip(*all_files), zip(*all_images)):
        name = os.path.commonprefix(contest_paths) + ".png"
        M = np.zeros((sum(x.shape[0] for x in contest_images),
                      max(x.shape[1] for x in contest_images)))
        pos = 0
        for img in contest_images:
            M[pos:pos + img.shape[0], 0:img.shape[1]] = img
            pos += img.shape[0]
        scipy.misc.imsave(name, M)
        res.append(name)
    return res


def make_norm(I, Iref):
    Inorm = np.zeros(Iref.shape, dtype=Iref.dtype)
    # make patch the same size as Iref

    min0 = min(I.shape[0], Iref.shape[0])
    min1 = min(I.shape[1], Iref.shape[1])
    Inorm[0:min0, 0:min1] = I[0:min0, 0:min1]

    diff0 = Iref.shape[0] - I.shape[0]
    diff1 = Iref.shape[1] - I.shape[1]

    if diff0 > 0:
        Inorm[I.shape[0]:I.shape[0] + diff0, :] = 1
    if diff1 > 0:
        Inorm[:, I.shape[1]:I.shape[1] + diff1] = 1

    return Inorm


@pdb_on_crash
def align(groupid, dat):
    res = []
    translations = []

    for i, group in enumerate(zip(*dat)):

        if i == 0:
            # We want to look at everything for the title
            left_border = 0
        else:
            # Skip the voting target for the rest
            #left_border = 80
            left_border = 0

        Iref_orig = sh.standardImread(group[0], flatten=True)
        Iref = Iref_orig[:, left_border:]
        r = []
        r_img = []

        for i in range(len(group)):
            I_orig = sh.standardImread(group[i], flatten=True)
            I = I_orig[:, left_border:]
            Inorm = make_norm(I, Iref)

            (H, imres, err) = imagesAlign.imagesAlign(
                Inorm, Iref, trfm_type='translation')

            r_img.append((make_norm(I_orig, Iref_orig), H))
            r.append(translate(group[i]))

        translations.append(r_img)
        res.append(r)

    translated_images = []
    for contest in zip(*translations):
        c_res = []
        """
        print 'new'
        arr = np.zeros((3,3))
        for _,H in contest:
            arr += H
        arr /= len(contest)
        print arr
        """
        for img, H in contest:
            translated = sh.imtransform(np.copy(img), H, fillval=np.nan)
            align_res = np.nan_to_num(translated)
            c_res.append(align_res)
        translated_images.append(c_res)
    translated_images = zip(*translated_images)

    return res, translated_images


class VerifyContestGrouping:

    def __init__(self, ocrdir, dirList, equivs, reorder, reorder_inverse, mapping, mapping_inverse, multiboxcontests, callback, NPROC=None):
        # print "ARGS", (ocrdir, dirList, equivs, reorder, reorder_inverse,
        # mapping, mapping_inverse, multiboxcontests)
        global TMP
        self.callback = callback
        TMP = ocrdir

        self.ocrdir = ocrdir
        self.dirList = dirList
        self.equivs = equivs
        self.reorder = reorder
        self.reorder_inverse = reorder_inverse
        self.mapping = mapping
        self.mapping_inverse = mapping_inverse
        self.multiboxcontests = multiboxcontests
        self.processgroups = [
            i for i, x in enumerate(self.equivs) if len(x) > 1]

        # print self.equivs
        # print self.processgroups
        res = []
        if NPROC == None:
            NPROC = mp.cpu_count()
        print "(Info) Using {0} processes for merge_and_align".format(NPROC)
        if NPROC == 1:
            res = []
            args = enumerate(
                map(self.generate_one, range(len(self.processgroups))))
            for arg in args:
                res.append(merge_and_align(arg))

        else:
            pool = mp.Pool(mp.cpu_count())
            print "Go up to", len(self.processgroups)
            res = pool.map(merge_and_align, enumerate(
                map(self.generate_one, range(len(self.processgroups)))))
        res = [x for y in res for x in y]
        # print len(res), map(len,res)

        # TODO: Provide the 'realign_callback' function for realigning a
        # set of overlay'd contest patches. See the docstring for
        #     verify_overlays_new.SeparateImages.do_realign
        realign_callback = None
        frame = grouping.verify_overlays_new.SeparateImagesFrame(None, res, self.on_verify_done,
                                                                 realign_callback=realign_callback)
        frame.Maximize()
        frame.Show()

    @pdb_on_crash
    def on_verify_done(self, results):
        """ Called when user finishes verifying the grouping.
        Input:
            list results: List of lists, where each sublist is considered
                one 'group': [[imgpath_0i, ...], [imgpath_1i, ...], ...]
        """
        mapping = {}
        for i, group in enumerate(results):
            for j, path in enumerate(group):
                print path
                mapping[path] = i
        sets = {}
        for groupid in self.processgroups:
            group = self.equivs[groupid]
            print "NEXT GROUP"
            for ballot, contest in group:
                print "NEW", ballot, contest
                print self.get_files(ballot, contest)
                print self.translate(os.path.commonprefix(self.get_files(ballot, contest))) + "~.png"
                ids = mapping[self.translate(os.path.commonprefix(
                    map(os.path.abspath, self.get_files(ballot, contest)))) + "~.png"]
                print ids

                if ids not in sets:
                    sets[ids] = []
                sets[ids].append((ballot, contest))
            print
        print sets
        self.callback(sets.values())

    @pdb_on_crash
    def get_files(self, ballot, contest):
        ballotname = os.path.split(self.dirList[ballot])[1].split('.')[0]
        boundingbox = (ballot, contest)
        if any(boundingbox in x for x in self.multiboxcontests):
            boundingboxes = [
                x for x in self.multiboxcontests if boundingbox in x][0]
            boundingbox = [
                x for x in boundingboxes if x in self.mapping_inverse][0]
            boundingboxes = [k[1]
                             for k, v in self.mapping.items() if v == boundingbox]
        else:
            boundingboxes = [self.mapping_inverse[boundingbox][1]]

        boundingboxes = sorted(boundingboxes)

        ballotdir = os.path.join(self.ocrdir, ballotname + "-dir")
        boundingboxdirs = [os.path.join(
            ballotdir, '-'.join(map(str, bb))) for bb in boundingboxes]
        order = dict(self.reorder[self.reorder_inverse[
                     ballot, contest]][ballot, contest])
        images = [[img for img in os.listdir(
            bbdir) if img[-3:] != 'txt'] for bbdir in boundingboxdirs]

        images = [sorted(imgs, key=lambda x: int(x.split('.')[0]))
                  for imgs in images]
        title = images[0][0]
        images = [(i, y) for i, x in enumerate(images) for y in x[1:]]
        orderedimages = [None] * (len(images) + 1)
        orderedimages[0] = (0, title)
        for i in range(len(images)):
            orderedimages[i + 1] = images[order[i]]
        paths = [os.path.join(boundingboxdirs[i], img)
                 for i, img in orderedimages]
        return paths

    def generate_one(self, which):
        orderedpaths = []
        # print "STARTING", self.equivs[self.processgroups[which]]
        print self.equivs
        for ballot, contest in self.equivs[self.processgroups[which]]:
            orderedpaths.append((self.get_files(ballot, contest)))
        return orderedpaths

    def translate(self, name):
        return os.path.join(TMP, os.path.abspath(name).replace("/", "~"))


if __name__ == '__main__':
    app = wx.App(False)
    VerifyContestGrouping(u'../projects_new/bug/ocr_tmp_dir', [u'/media/data1/ekim_misc/test-ballots-ek/orange_badmultibox_short/A_00_side0.png', u'/media/data1/ekim_misc/test-ballots-ek/orange_badmultibox_short/B_00_side1.png'], [[(0, 0), (0, 0)]], {(1, 0): {(1, 0): [(0, 0), (1, 1), (2, 2)]}, (0, 0): {(0, 0): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]}, (1, 1): {(
        1, 1): [(0, 0), (1, 1), (2, 2)]}}, {(1, 0): (1, 0), (0, 0): (0, 0), (1, 1): (1, 1)}, {(0, (1117, 280, 1610, 599)): (0, 0), (1, (613, 280, 1112, 599)): (1, 0), (0, (625, 280, 1123, 686)): (0, 0), (1, (125, 280, 619, 688)): (1, 1)}, {(1, 0): (1, (613, 280, 1112, 599)), (0, 0): (0, (625, 280, 1123, 686)), (1, 1): (1, (125, 280, 619, 688))}, [[(0, 1), (0, 0)]], lambda x: x)
    app.MainLoop()
