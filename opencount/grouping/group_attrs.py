import sys, os, pickle, pdb, wx, time, shutil, copy
from os.path import join as pathjoin
import scipy, scipy.misc
import numpy as np

sys.path.append('../')
import common, util
import verify_overlays
from specify_voting_targets import util_gui
from pixel_reg import shared

_i = 0

def group_attributes(attrdata, imgsize, projdir_path, tmp2imgs_path, job_id=None):
    """ Using NCC, group all attributes to try to reduce operator
    effort.
      attrdata: A list of dicts (marshall'd AttributeBoxes)
      project: A Project.
    """
    OUTDIR_ATTRPATCHES = pathjoin(projdir_path, 'extract_attrs_templates')
    def munge_matches(matches, grouplabel, attrtype, outdir='extract_attrs_templates'):
        """ Given matches from find_patch_matches, convert it to
        a format that GroupClass likes:
          (sampleid, rankedlist, patchpath)
        Also saves the registered patches to output dir.
        """
        results = []
        for (filename, score1, score2, Ireg, y1,y2,x1,x2, rszFac) in matches:
            relpath = os.path.relpath(filename)

            # Save registered patch
            relpath = os.path.relpath(filename)
            attrtype_str = '-'.join(attrtype)
            outdir_full = os.path.join(projdir_path, outdir, attrtype_str)
            outpath_full = os.path.join(outdir_full, util.encodepath(relpath)+'.png')
            results.append((relpath, (grouplabel,), outpath_full))
            if not os.path.exists(outpath_full):
                Ireg[0,0] = 0.99
                Ireg[0,1] = 0.01
                Ireg = np.nan_to_num(Ireg)
                util_gui.create_dirs(outdir_full)
                scipy.misc.imsave(outpath_full, Ireg)
        return results
    def get_temp_patches(d, temppaths, outdir='extract_attrs_templates'):
        """Return a mapping that does:
          {str filenname: str patchpath}
        """
        result = {}
        for temppath in temppaths:
            attrtype = tuple(sorted(d['attrs'].keys()))
            relpath = os.path.relpath(temppath)
            attrtype_str = '-'.join(attrtype)
            outdir_full = os.path.join(projdir_path, outdir, attrtype_str)
            outpath_full = os.path.join(outdir_full, util.encodepath(relpath)+'.png')
            result[relpath] = outpath_full
        return result
    def missing_templates(matches, temppaths):
        """ Returns templatepaths missing from matches. """
        history = {}
        result = []
        for (filename, _, _, _, _, _, _) in matches:
            history[filename] = True
        for temppath in temppaths:
            if temppath not in history:
                result.append(temppath)
        return result

    if os.path.isdir(OUTDIR_ATTRPATCHES):
        shutil.rmtree(OUTDIR_ATTRPATCHES)
        
    tmp2imgs = pickle.load(open(tmp2imgs_path, 'rb'))
    groups = []
    """
    Fixed-point iteration.
    """
    no_change = False
    history = set()
    attrtype_ctr = {}
    w_img, h_img = imgsize
    n_iters = 0
    _starttime = time.time()
    THRESHOLD = 0.7
    unlabeled_blanks = tmp2imgs.copy()
    def choose_blank(unlabeled_blanks, side):
        for tempid, path in unlabeled_blanks.iteritems():
            if side == 'front':
                return path[0]
            else:
                return path[1]
        print "wat"
        pdb.set_trace()
    def get_unlabeled_paths(unlabeled_blanks, side):
        for tempid, path in unlabeled_blanks.iteritems():
            if side == 'front':
                yield path[0]
            else:
                yield path[1]
        raise StopIteration
    for attrdict in attrdata:
        x1, y1 = int(round(attrdict['x1']*w_img)), int(round(attrdict['y1']*h_img))
        x2, y2 = int(round(attrdict['x2']*w_img)), int(round(attrdict['y2']*h_img))
        side = attrdict['side']
        attrtype = tuple(sorted(attrdict['attrs'].keys()))
        blankpath = choose_blank(unlabeled_blanks, side)
        img = shared.standardImread(blankpath, flatten=True)
        patch = img[y1:y2, x1:x2]
        h, w = patch.shape
        bb = [0, h, 0, w]
        temppaths = get_unlabeled_paths(unlabeled_blanks, side)
        _t = time.time()
        matches = shared.find_patch_matchesV1(patch, bb, 
                                              temppaths,
                                              bbSearch=[y1, y2, x1, x2],
                                              threshold=THRESHOLD)
        _endt = time.time() - _t
        print "len(matches): {0}  time: {1}".format(len(matches),_endt)
        if matches:
            flag = True
            # Discard worst-scoring duplicates
            bestmatches = {} # maps {(attrtype, filename): (filename,sc1,sc2,Ireg,x1,y1,x2,y2,rszFac)}
            for (filename,sc1,sc2,Ireg,x1,y1,x2,y2,rszFac) in matches:
                key = (attrtype, filename)
                if key not in bestmatches:
                    bestmatches[key] = (filename,sc1,sc2,Ireg,x1,y1,x2,y2,rszFac)
                else:
                    tup = bestmatches[key]
                    sc = tup[2]
                    if sc and sc2 < sc:
                        bestmatches[key] = (filename,sc1,sc2,Ireg,x1,y1,x2,y2,rszFac)
            bestmatches_lst = bestmatches.values()
            # Now handle 'found' templates
            for (filename, _, _, _,  _, _, _, _, rszFac) in bestmatches_lst:
                history.add((attrtype, filename))
            inc_counter(attrtype_ctr, attrtype)
            grouplabel = common.make_grouplabel((attrtype, attrtype_ctr[attrtype]))
            elements = munge_matches(bestmatches_lst, grouplabel, attrtype)
            in_group = common.GroupClass(elements)
            groups.append(in_group)
        if not flag:
            # Something bad happened, if we still have blank
            # ballots left to work over.
            print "UH OH BAD."
            THRESHOLD -= 0.1
            #no_change = True
        n_iters += 1
    print "== Total Time:", time.time() - _starttime
    return groups

def group_attributes_V2(attrdata, imgsize, projdirpath, tmp2imgs_path, job_id=None):
    """ Alternative grouping algorithm. """
    pass

def cluster_patches(imgpaths):
    """ Given a list of imgpaths, where the number of groups are not
    known ahead of time, cluster them into groups. """
    def make_cluster(imgpath):
        return [imgpath]
    def cluster_add(cluster, imgpath):
        assert imgpath not in cluster
        cluster.append(imgpath)
    def get_closest_cluster(clusters, imgpath, distfn):
        mindist = None
        bestcluster = None
        for cluster in clusters:
            dist = distfn(cluster, imgpath)
            if mindist == None or dist < mindist:
                mindist = dist
                bestcluster = cluster
        return bestcluster
    def distmetric(cluster, imgpath):
        patch = shared.standardImread(imgpath, flatten=True)
        h, w = patch.shape
        bb = [0, h, 0, w]
        matches = shared.find_patch_matchesV1(img, bb, cluster, threshold=0.0)
        return sum([t[2] for t in matches]) / len(matches)
    clusters = [make_cluster(imgpaths[0])]
    if type(imgpaths) != list:
        imgpaths = list(imgpaths)
    else:
        imgpaths = imgpaths[:]
    while len(imgpaths) >= 0:
        imgpath = imgpaths.pop()
        cluster = get_closest_cluster(clusters, imgpath, distmetric)
        if cluster == None:
            clusters.append(make_cluster(imgpath))
        else:
            cluster_add(cluster, imgpath)
    return clusters

def cluster_attributes(blankpatches):
    """ Given a map that, for each attribute type, maps an attribute 
    value to a list of atttribute patches (with the given type->val),
    return a new mapping that maps each type->val to a set of exemplars.
    Input:
        dict blankpatches: maps {attrtype: {attrval: list of imgpatches}}
        obj project
    Output:
        A dict mapping {attrtype: {attrval: list of imgpatches}}, which
        is a subset of the input blankpatches.
    """
    all_exemplars = {}
    for attrtype in blankpatches:
        attrval_exemplars = compute_exemplars2(blankpatches[attrtype])
        all_exemplars[attrtype] = attrval_exemplars
    return all_exemplars

def compute_exemplars(mapping):
    """ Given a mapping {str label: list of imgpaths}, extracts a subset
    of the imgpaths {str label: list of imgpaths} such that these
    imgpaths are the best-describing 'exemplars' of the entire input
    mapping.
    Input:
        dict mapping: {label: list of imgpaths}
    Output:
        A (hopefully smaller) dict mapping {label: list of exemplar
        imgpaths}.
    """
    def distance(img, imgpath2):
        h, w = img.shape
        bb = [0, h, 0, w]
        matches = shared.find_patch_matchesV1(img, bb, (imgpath2,), threshold=0.1)
        matches = sorted(matches, key=lambda t: t[2])
        return matches[0][2]
    def closest_label(imgpath, exemplars):
        mindist = None
        bestmatch = None
        img = shared.standardImread(imgpath, flatten=True)
        for label, imgpaths in exemplars.iteritems():
            for imgpathB in imgpaths:
                dist = distance(img, imgpathB)
                if mindist == None or dist < mindist:
                    bestmatch = label
                    mindist = dist
        return bestmatch
    mapping = copy.deepcopy(mapping)
    exemplars = {}
    for label, imgpaths in mapping.iteritems():
        exemplars[label] = [imgpaths.pop()]
    is_done = False
    while not is_done:
        is_done = True
        for label, imgpaths in mapping.iteritems():
            for imgpath in imgpaths:
                j = closest_label(imgpath, exemplars)
                print 'label should be {0} closest was {1}'.format(label, j)
                if label != j:
                    exemplars[label].append(imgpath)
                    is_done = False
    return exemplars

def compute_exemplars2(mapping):
    """
    Given a mapping {str label: list of imgpaths}, extracts a subset
    of the imgpaths {str label: list of imgpaths} such that these
    imgpaths are the best-describing 'exemplars' of the entire input
    mapping.
    Input:
        dict mapping: {label: list of imgpaths}
    Output:
        A (hopefully smaller) dict mapping {label: list of exemplar
        imgpaths}.
    """
    def make_inverse_mapping(mapping):
        inverse = {}
        for label, imgpaths in mapping.iteritems():
            for imgpath in imgpaths:
                assert imgpath not in inverse
                inverse[imgpath] = label
        return inverse
    def get_incorrect_matches(bestmatches):
        """Returns all matches that are incorrect, as a list of tuples. """
        result = []
        for imgpath, (bestscore, correctlabel, label) in bestmatches.iteritems():
            if correctlabel != label:
                result.append((imgpath, (bestscore, correctlabel, label)))
        return result
    mapping = copy.deepcopy(mapping)
    inv_mapping = make_inverse_mapping(mapping)
    exemplars = {} # maps {label: list of imgpaths}
    bestmatches = {}  # maps {imgpath: (float best-score, str correctlabel, str computedlabel)}
    all_imgpaths = reduce(lambda x, y: x + y, mapping.values())
    # First, start off with only one exemplar for each attr value
    for label, imgpaths in mapping.iteritems():
        imgpathA = imgpaths[0]
        exemplars[label] = [imgpathA]
        patch = shared.standardImread(imgpathA, flatten=True)
        h, w = patch.shape
        bb = [0, h, 0, w]
        matches = shared.find_patch_matchesV1(patch, bb, all_imgpaths, threshold=0.0)
        all_s2 = True
        for (filename,s1,s2,I,i1,i2,j1,j2,rszFac) in matches:
            if s2 != 0.0:
                all_s2 = False
            if s2 == 0.0:
                print "sc2 was 0.0. correctlabel: {0}  label: {1}".format(inv_mapping[filename], label, filename)
                print "comparing patch {0} with image {1}".format(imgpathA, filename)
                #pdb.set_trace()
            if filename not in bestmatches or s2 < bestmatches[filename][0]:
                bestmatches[filename] = s2, inv_mapping[filename], label
        if all_s2:
            pass
            #pdb.set_trace()
    is_done = False
    i = 0
    last_len = 0
    while not is_done:
        incorrect_matches = get_incorrect_matches(bestmatches)
        print "i={0}, len(incorrect_matches)={1}".format(i, len(incorrect_matches))
        if not incorrect_matches:
            is_done = True
            continue
        if len(incorrect_matches) == last_len:
            pdb.set_trace()
        last_len = len(incorrect_matches)
        imgpath, (bestscore, correctlabel, computedlabel) = incorrect_matches[0]
        exemplars[correctlabel].append(imgpath)
        patch = shared.standardImread(imgpath, flatten=True)
        h, w = patch.shape
        bb = [0, h, 0, w]
        matches = shared.find_patch_matchesV1(patch, bb, all_imgpaths, threshold=0.0)
        for (filename,s1,s2,I,i1,i2,j1,j2,rszFac) in matches:
            if filename not in bestmatches or s2 < bestmatches[filename][0]:
                bestmatches[filename] = s2, inv_mapping[filename], inv_mapping[imgpath]
        i += 1
    return exemplars


def inc_counter(ctr, k):
    if k not in ctr:
        ctr[k] = 1
    else:
        ctr[k] = ctr[k] + 1

class TestFrame(wx.Frame):
    def __init__(self, parent, groups, *args, **kwargs):
        wx.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.groups = groups

        self.panel = verify_overlays.VerifyPanel(self, verifymode=verify_overlays.VerifyPanel.MODE_YESNO2)
        self.Show()
        self.Maximize()
        self.Fit()
        
        self.panel.start(groups, None, ondone=self.verify_done)
        
    def verify_done(self, results):
        """ Called when the user finished verifying the auto-grouping of
        attribute patches (for blank ballots).
        results is a dict mapping:
            {grouplabel: list of GroupClass objects}
        The idea is that for each key-value pairing (k,v) in results, the
        user has said that: "These blank ballots in 'v' all have the same
        attribute value, since their overlays looked the same."
        The next step is for the user to actually label these groups (instead
        of labeling each individual blank ballot). The hope is that the
        number of groups is much less than the number of blank ballots.
        """
        print "Verifying done."
        num_elements = 0
        for grouplabel,groups in results.iteritems():
            cts = 0
            for group in groups:
                cts += len(group.elements)
            print "grouplabel {0}, {1} elements, is_manual: {2}".format(grouplabel, cts, group.is_manual)
            num_elements += cts
        print "The idea: Each group contains images whose overlays \
are the same. It might be the case that two groups could be merged \
together."
        if num_elements == 0:
            reduction = 0
        else:
            reduction = len(sum(results.values(), [])) / float(num_elements)
        
        print "== Reduction in effort: ({0} / {1}) = {2}".format(len(sum(results.values(), [])),
                                                                 num_elements,
                                                                 reduction)
def main():
    args = sys.argv[1:]
    rootdir = args[0]
    attrdata = pickle.load(open(pathjoin(rootdir, 'ballot_attributes.p'), 'rb'))
    #imgsize = (1460, 2100)
    #imgsize = (1715, 2847)
    #imgsize = (1459, 2099)    # alameda
    #imgsize = (1968, 3530)    # napa
    imgsize = (1280, 2104)    # marin
    projdir_path = rootdir
    tmp2imgs_path = pathjoin(rootdir, 'template_to_images.p')
    groups = group_attributes(attrdata, imgsize, projdir_path, tmp2imgs_path)
    for group in groups:
        print group
    
    # Visualize results
    app = wx.App(False)
    frame = TestFrame(None, groups)
    app.MainLoop()

if __name__ == '__main__':
    main()

