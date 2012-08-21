import sys, os, pickle, pdb, wx, time, shutil, copy, random
from os.path import join as pathjoin
import scipy, scipy.misc
import numpy as np

sys.path.append('../')
import common, util
import verify_overlays
import partask
from specify_voting_targets import util_gui
from pixel_reg import shared

_i = 0

def cluster_imgpatches(imgpaths, bb_map, init_clusters=None):
    """ Given a list of imgpaths, and bounding boxes for each image,
    cluster the bounding boxes from each image.
    Input:
        list imgpaths:
        dict bb_map: maps {str imgpath: (y1,y2,x1,x2)}
        list init_clusters: An initial set of cluster centers, of the form:
            {imgpath_i: (imgpath_i, bb_i)}
    Output:
        A dict of the form:
            {c_imgpath: [(imgpath_i, bb_i), ...}
        where each c_imgpath is the 'center' of a given cluster C.
    """
    clusters = {}
    if init_clusters == None:
        # Randomly select one image as the first cluster center
        _imgpath = random.choice(imgpaths)
        clusters[_imgpath] = [(_imgpath, bb_map[_imgpath])]
    else:
        clusters = dict(init_clusters)
    THRESHOLD = 0.7
    C_NEW_CLUSTER = 0.1  # sc2 ranges from 0.0 - 1.0, where 0.0 is 'best'
    no_matches = False
    unclustered_imgpaths = [p for p in imgpaths if p not in clusters]
    while unclustered_imgpaths:
        no_matches = True
        for c_imgpath in dict(clusters):
            bb_c = bb_map[c_imgpath]
            img = shared.standardImread(c_imgpath, flatten=True)
            patch = img[bb_c[0]:bb_c[1], bb_c[2]:bb_c[3]]
            #scipy.misc.imsave("patch_{0}_{1}.png".format(os.path.split(c_imgpath)[1], time.time()),
            #                  patch)
            _bb = [0, patch.shape[0], 0, patch.shape[1]]
            _t = time.time()
            print "...calling find_patch_matchesV1..."
            matches = shared.find_patch_matchesV1(patch, _bb,
                                                  unclustered_imgpaths,
                                                  bbSearch=bb_c,
                                                  threshold=THRESHOLD)
            print "...finished find_patch_matchesV1 ({0} s)".format(time.time() - _t)
            if matches:
                # 0.) Retrieve best matches from matches (may have multiple
                # matches for the same imagepath)
                print "...number of pre-filtered matches: {0}".format(len(matches))
                no_matches = False
                bestmatches = {} # maps {imgpath: (imgpath,sc1,sc2,Ireg,y1,y2,x1,x2,rszFac)}
                for (filename,sc1,sc2,Ireg,y1,y2,x1,x2,rszFac) in matches:
                    y1, y2, x1, x2 = map(lambda c: int(round(c / rszFac)), (y1,y2,x1,x2))
                    if filename not in bestmatches:
                        bestmatches[filename] = (filename,sc1,sc2,Ireg,y1,y2,x1,x2,rszFac)
                    else:
                        old_sc2 = bestmatches[filename][2]
                        if sc2 < old_sc2:
                            bestmatches[filename] = (filename,sc1,sc2,Ireg,y1,y2,x1,x2,rszFac)
                # 1.) Decide whether to create a new cluster, or not
                print "...found {0} matches".format(len(bestmatches))
                for _, (filename,sc1,sc2,Ireg,y1,y2,x1,x2,rszFac) in bestmatches.iteritems():
                    unclustered_imgpaths.remove(filename)
                    print 'sc2 is:', sc2
                    if sc2 >= C_NEW_CLUSTER:
                        print "...created new cluster. num_clusters: {0}".format(len(clusters))
                        clusters[filename] = [(filename, (y1,y2,x1,x2), sc2)]
                    else:
                        print "...added element to a cluster C."
                        clusters[c_imgpath].append((filename, (y1,y2,x1,x2), sc2))
            else:
                print "...no matches found."
        if no_matches == True:
            new_k = THRESHOLD - 0.1
            print "... Uh oh, never found any matches. We could fall \
into an infinite loop. Decreasing THRESHOLD from {0} to {1}".format(THRESHOLD, new_k)
            print "... Trying another iteration."
            THRESHOLD = new_k
    print "...Completed clustering. We found {0} clusters.".format(len(clusters))
    return clusters
                
def cluster_imgpatchesV2(imgpaths, bb_map, init_clusters=None):
    """ Given a list of imgpaths, and bounding boxes for each image,
    cluster the bounding boxes from each image.
    Input:
        list imgpaths:
        dict bb_map: maps {str imgpath: (y1,y2,x1,x2)}
        list init_clusters: An initial set of cluster centers, of the form:
            {imgpath_i: (imgpath_i, bb_i)}
    Output:
        A dict of the form:
            {c_imgpath: [(imgpath_i, bb_i, score), ...}
        where each c_imgpath is the 'center' of a given cluster C.
    """
    clusters = {}
    unlabeled_imgpaths = list(imgpaths)
    THRESHOLD = 0.7
    while unlabeled_imgpaths:
        curimgpath = unlabeled_imgpaths[0]
        bb = bb_map[curimgpath]
        I = shared.standardImread(curimgpath, flatten=True)
        patch = I[bb[0]:bb[1], bb[2]:bb[3]]
        _t = time.time()
        print "...calling find_patch_matchesV1..."
        matches = partask.do_partask(findpatchmatches,
                                     unlabeled_imgpaths,
                                     _args=(patch,
                                            (0, patch.shape[0], 0, patch.shape[1]),
                                            bb, THRESHOLD))
        print "...finished find_patch_matchesV1 ({0} s)".format(time.time() - _t)
        if matches:
            # 0.) Retrieve best matches from matches (may have multiple
            # matches for the same imagepath)
            print "...number of pre-filtered matches: {0}".format(len(matches))
            bestmatches = {} # maps {imgpath: (imgpath,sc1,sc2,Ireg,y1,y2,x1,x2,rszFac)}
            for (filename,sc1,sc2,Ireg,y1,y2,x1,x2,rszFac) in matches:
                y1, y2, x1, x2 = map(lambda c: int(round(c/rszFac)), (y1, y2, x1, x2))
                if filename not in bestmatches:
                    bestmatches[filename] = (filename,sc1,sc2,Ireg,y1,y2,x1,x2,rszFac)
                else:
                    old_sc2 = bestmatches[filename][2]
                    if sc2 < old_sc2:
                        bestmatches[filename] = (filename,sc1,sc2,Ireg,y1,y2,x1,x2,rszFac)
            print "...found {0} matches".format(len(bestmatches))
            # 1.) Handle the best matches
            for _, (filename,sc1,sc2,Ireg,y1,y2,x1,x2,rszFac) in bestmatches.iteritems():
                unlabeled_imgpaths.remove(filename)
                clusters.setdefault(curimgpath, []).append((filename, (y1,y2,x1,x2), sc2))
        else:
            print "...Uh oh, no matches found. This shouldnt' have \
happened."
            pdb.set_trace()
    print "...Completed clustering. Found {0} clusters.".format(len(clusters))
    return clusters

def findpatchmatches(imlist, (patch, bb, bbsearch, threshold)):
    return shared.find_patch_matchesV1(patch, bb, imlist, bbSearch=bbsearch,
                                       threshold=threshold)

def group_attributes_V2(project, job_id=None):
    """ Try to cluster all attribute patches from blank ballots into
    groups, in order to reduce operator effort during 'Label Ballot
    Attributes.'
    Output:
        dict mapping {str attrtype: {str c_imgpath: [(imgpath_i, bb_i), ...]}}
    """
    ballot_attrs = pickle.load(open(project.ballot_attributesfile, 'rb'))
    w_img, h_img = project.imgsize
    tmp2imgs = pickle.load(open(project.template_to_images, 'rb'))
    attr_clusters = {} # maps {str attrtype: {str c_imgpath: [(imgpath_i, bb_i, score_i), ...]}}
    bb_mapAll = {} # maps {attrtype: {imgpath: bb}}
    for attr in ballot_attrs:
        x1,y1,x2,y2 = attr['x1'],attr['y1'],attr['x2'],attr['y2']
        x1 = int(round(x1*w_img))
        y1 = int(round(y1*h_img))
        x2 = int(round(x2*w_img))
        y2 = int(round(y2*h_img))
        side = attr['side']
        attrtype = common.get_attrtype_str(attr['attrs'])
        # TODO: Generalize to N-sided elections
        if side == 'front': 
            blank_imgpaths = [paths[0] for paths in tmp2imgs.values()]
        else:
            blank_imgpaths = [paths[1] for paths in tmp2imgs.values()]
        bb_map = {} # maps {imgpath: (y1,y2,x1,x2)}
        for imgpath in blank_imgpaths:
            bb_map[imgpath] = (y1,y2,x1,x2)
        bb_mapAll[attrtype] = bb_map
        clusters = cluster_imgpatchesV2(blank_imgpaths, bb_map)
        attr_clusters[attrtype] = clusters
    return attr_clusters

def group_attributes(attrdata, imgsize, projdir_path, tmp2imgs_path, project, job_id=None):
    """ Using NCC, group all attributes to try to reduce operator
    effort.
      attrdata: A list of dicts (marshall'd AttributeBoxes)
      project: A Project.
    """
    OUTDIR_ATTRPATCHES = pathjoin(projdir_path, 'extract_attrs_templates')
    def munge_matches(matches, grouplabel, attrtype_str, outdir='extract_attrs_templates'):
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
    def init_unlabeled_blanks(tmp2imgs, attrdicts, is_multipage):
        result = {}
        for attrdict in attrdicts:
            side = attrdict['side']
            attrtypestr = common.get_attrtype_str(attrdict['attrs'])
            if not is_multipage or side == 'front':
                result[attrtypestr] = list([paths[0] for paths in tmp2imgs.values()])
            else:
                result[attrtypestr] = list([paths[1] for paths in tmp2imgs.values()])
        return result
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
    #unlabeled_blanks = tmp2imgs.copy()
    # maps {attrtype: list of imgpaths}
    unlabeled_blanks = init_unlabeled_blanks(tmp2imgs, attrdata, project.is_multipage)
    labeled_blanks = {}  # maps {attrtype: list of imgpaths}
    was_change = True
    while was_change:
        amidone = True
        for attrtype, blankpaths in unlabeled_blanks.iteritems():
            if blankpaths:
                amidone=False
                break
        if amidone == True:
            break
        was_change = False
        for attrdict in attrdata:
            x1, y1 = int(round(attrdict['x1']*w_img)), int(round(attrdict['y1']*h_img))
            x2, y2 = int(round(attrdict['x2']*w_img)), int(round(attrdict['y2']*h_img))
            side = attrdict['side']
            #attrtype = tuple(sorted(attrdict['attrs'].keys()))
            #blankpath = choose_blank(unlabeled_blanks, side)
            attrtype = common.get_attrtype_str(attrdict['attrs'])
            # Grab an unclustered blank ballot for attrtype
            ungrouped_blanks = unlabeled_blanks[attrtype]
            if not ungrouped_blanks:
                # we've grouped all blank ballots for this attrtype
                continue
            blankpath = ungrouped_blanks[0]
            img = shared.standardImread(blankpath, flatten=True)
            patch = img[y1:y2, x1:x2]
            h, w = patch.shape
            bb = [0, h, 0, w]
            #temppaths = get_unlabeled_paths(unlabeled_blanks, side)
            temppaths = ungrouped_blanks
            _t = time.time()
            matches = shared.find_patch_matchesV1(patch, bb, 
                                                  temppaths,
                                                  bbSearch=[y1, y2, x1, x2],
                                                  threshold=THRESHOLD)
            _endt = time.time() - _t
            print "len(matches): {0}  time: {1}".format(len(matches),_endt)
            if matches:
                was_change = True
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
                    unlabeled_blanks[attrtype].remove(filename)
                inc_counter(attrtype_ctr, attrtype)
                grouplabel = common.make_grouplabel((attrtype, attrtype_ctr[attrtype]))
                elements = munge_matches(bestmatches_lst, grouplabel, attrtype)
                in_group = common.GroupClass(elements)
                groups.append(in_group)
            else:
                # This is weird. We should have at least clustered
                # one blank ballot (the one pointed by blankpath).
                print "UH OH BAD. Lowering threshold..."
                THRESHOLD -= 0.1
                #no_change = True
        n_iters += 1
    for attrtype in unlabeled_blanks:
        if len(unlabeled_blanks[attrtype]) != 0:
            print "WAT. Unlabeled blanks still exist."
            pdb.set_trace()
    print "== Total Time:", time.time() - _starttime
    return groups

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

def cluster_attributesV2(blanklabels, project):
    """ For all attribute types, try to find a set of exempalrs for
    each type.
    Input:
        dict blanklabels: maps {str attrtype: {str attrval: list of blank paths}}
    Output:
        A dict mapping {attrtype: {attrval: list of blank paths}}
    """
    def pixcoords(coords, w, h):
        return (int(round(coords[0]*w)), int(round(coords[1]*h)),
                int(round(coords[2]*w)), int(round(coords[3]*h)))
    attr_dicts = pickle.load(open(project.ballot_attributesfile, 'rb'))
    # Grab the blank ballot displayed in 'DefineBallotAttributes', since
    # this is guaranteed to 'look good' in terms of registration
    blankballotpath = None
    for dirpath, dirnames, filenames in sorted(os.walk(project.templatesdir)):
        for imgname in sorted([f for f in filenames if util_gui.is_image_ext(f)]):
            blankballotpath = os.path.join(dirpath, imgname)
            break
    assert blankballotpath != None
    blankballot = shared.standardImread(blankballotpath, flatten=True)
    tmp2imgs = pickle.load(open(project.template_to_images, 'rb'))
    h_img, w_img = blankballot.shape
    all_exemplars = {}  # maps {attrtype: {attrval: ((tpath_i,y1,y2,x1,x2), ...)}}
    for attrdict in attr_dicts:
        attrtypestr = common.get_attrtype_str(attrdict['attrs'])
        x1, y1, x2, y2 = pixcoords((attrdict['x1'], attrdict['y1'],
                                   attrdict['x2'], attrdict['y2']),
                                   w_img, h_img)
        bbsearch = (y1,y2,x1,x2)
        # grab the label from the blank ballot
        startlabel = None
        for attrval, blankpaths in blanklabels[attrtypestr].iteritems():
            if blankballotpath in blankpaths:
                startlabel = attrval
                break
        if startlabel == None:
            print "Uh oh, startlabel was None."
            pdb.set_trace()
        assert startlabel != None
        initials = {startlabel: (blankballotpath, y1, y2, x1, x2)}
        # maps {attrval: ((templatepath_i, y1_i, y2_i, x1_i, x2_i), ...)}
        exemplars = find_min_exemplars(initials, bbsearch, blanklabels[attrtypestr])
        all_exemplars[attrtypestr] = exemplars
    return all_exemplars

def find_min_exemplars(initials, bbsearch, samplelabels):
    """
    Input:
        dict initials: maps {label: (imgpath, y1, y2, x1, x2)}
        bbsearch: A tuple (y1,y2,x1,x2)
        dict samplelabels: maps {label: imgpath}
    Output:
        dict {label: ((imgpath_i, y1,y2,x1,x2), ...)}
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
        for imgpath, (bestscore, correctlabel, label, y1,y2,x1,x2,rszFac) in bestmatches.iteritems():
            if label != correctlabel:
                result.append((imgpath, (bestscore, correctlabel, label,y1,y2,x1,x2,rszFac)))
        return result
    correctlabels = make_inverse_mapping(samplelabels)
    all_exemplars = {} # maps {label: ((imgpath_i,y1,y2,x1,x2, rszFac), ...)}
    bestmatches = {} # maps {imgpath: (score, correctlabel, computedlabel, y1,y2,x1,x2, rszFac)}
    all_imgpaths = list(reduce(lambda x,y: x+y, samplelabels.values()))
    for label, (imgpath, y1, y2, x1, x2) in initials.iteritems():
        all_exemplars[label] = [(imgpath,y1,y2,x1,x2,1.0),]
    # 1.) Apply initial exemplar guess
    print "DOING LABEL: ", label
    for label, exemplars in all_exemplars.iteritems():
        for (imgpath, y1, y2, x1, x2,rszFac) in exemplars:
            img = shared.standardImread(imgpath, flatten=True)
            bb = tuple(map(lambda c: int(round(c / rszFac)), (y1,y2,x1,x2)))
            matches = shared.find_patch_matchesV1(img, bb, all_imgpaths, threshold=0.0,
                                                  bbSearch=bb)
            for (filename,sc1,sc2,Ireg,y1,y2,x1,x2,rszFac) in matches:
                print 'sc1: {0} sc2: {1}  region: {2}'.format(sc1, sc2, (y1,y2,x1,x2))
                if filename not in bestmatches or sc2 < bestmatches[filename][0]:
                    bestmatches[filename] = (sc2, correctlabels[filename], label, y1, y2, x1, x2, rszFac)
    # 2.) Fixed point iteration
    is_done = False
    while not is_done:
        incorrectmatches = get_incorrect_matches(bestmatches)
        print "Num incorrectmatches:", len(incorrectmatches)
        if not incorrectmatches:
            is_done = True
            continue
        # Add the worst-scoring mismatch as a 'new' exemplar
        sorted_mismatches = sorted(incorrectmatches, key=lambda t: t[1][0])
        imgpath,(score,correctlabel,label,y1,y2,x1,x2,rszFac) = sorted_mismatches[-1]
        all_exemplars.setdefault(correctlabel, []).append((imgpath, y1,y2,x1,x2,rszFac))
        # Re-run classification
        for label, exemplars in all_exemplars.iteritems():
            for (imgpath, y1,y2,x1,x2,rszFac) in exemplars:
                img = shared.standardImread(imgpath, flatten=True)
                bb = tuple(map(lambda c: c / rszFac, (y1,y2,x1,x2)))
                matches = shared.find_patch_matchesV1(img, bb, all_imgpaths, threshold=0.0,
                                                      bbSearch=bb)
                for (filename, sc1, sc2, Ireg, y1, y2, x1, x2, rszFac) in matches:
                    if filename not in bestmatches or sc2 < bestmatches[filename][0]:
                        bestmatches[filename] = (sc2, correctlabels[filename], label, y1, y2, x1, x2, rszFac)
    #pdb.set_trace()
    return all_exemplars

def compute_exemplarsV2(blankballot, bbsearch, mapping):
    """
    Input:
        obj blankballot: Blank ballot image to extract patch from
        bbsearch: A tuple (y1,y2,x1,x2)
        list tpaths: A list of blank ballot paths.
        dict mapping: Maps {attrval: list of blank ballot paths}
    Output:
        A dict {attrval: ((tpath_i, y1_i, y2_i, x1_i, x2_i), ...)}
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
        matches = shared.find_patch_matchesV1(blankballot, bb, all_imgpaths, threshold=0.0)
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

