import sys, os, pdb, time, shutil, traceback

try:
    import cPickle as pickle
except ImportError as e:
    print "Can't import cPickle, falling back to pickle."
    import pickle

import scipy, scipy.misc
import cv
from os.path import join as pathjoin
sys.path.append('..')
import specify_voting_targets.util_gui as util_gui
import pixel_reg.shared as sh
import util, common, group_attrs
from PIL import Image
import grouping.partask as partask

def do_digitocr_patches(bal2imgs, digitattrs, project, ignorelist=None,
                        rejected_hashes=None,
                        accepted_hashes=None):
    """ For each digitbased attribute, run our NCC-OCR on the patch
    (using our digit exemplars).
    Input:
        dict bal2imgs
        dict digitattrs: maps {attrtype: ((y1,y2,x1,x2), side)}
        obj project
        dict rejected_hashes: maps {imgpath: {digit: [((y1,y2,x1,x2),side_i,isflip_i), ...]}}
        dict accepted_hashes: maps {imgpath: {digit: [((y1,y2,x1,x2),side_i,isflip_i), ...]}}
        lst ignorelist: List of ballotid's to ignore. UNUSED.
    Output:
        A dict that maps:
          {ballotid: ((attrtype_i, ocrresult_i, meta_i, isflip_i, side_i), ...)
        where meta_i is a tuple containing numDigits tuples:
          (y1_i,y2_i,x1_i,x2_i, str digit_i, str digitimgpath_i, score)
    """
    def make_digithashmap(project, exemplars):
        """ exemplars is a dict {str digit: ((blankpath_i, bb_i, exemplarP_i), ...)}.
        Returns a dict {(str digit, str meta): obj digitpatch}.
        """
        digit_hash = {}
        for digit, tuples in exemplars.iteritems():
            for i, (blankpath, bb, exemplarP) in enumerate(tuples):
                digit_hash[(digit, i)] = sh.standardImread(exemplarP, flatten=True)
        return digit_hash

        #digitmap = {} # maps {(str digit, str meta): obj img}
        #digit_exemplarsdir = os.path.join(project.projdir_path,
        #                                  project.digit_exemplars_outdir)
        #digitdirs = os.listdir(digit_exemplarsdir)
        #for digitdir in digitdirs:
        #    # Assumes this has directories of the form:
        #    #    0_examples/*.png
        #    #    1_examples/*.png
        #    #    ...
        #    fullpath = os.path.join(digit_exemplarsdir, digitdir)
        #    digit = digitdir.split('_')[0]
        #    for dirpath, dirnames, filenames in os.walk(fullpath):
        #        for imgname in [f for f in filenames if util_gui.is_image_ext(f)]:
        #            # This currently scans through all images, unnecessary.
        #            # We only need just one.
        #            imgpath = os.path.join(dirpath, imgname)
        #            img = sh.standardImread(imgpath, flatten=True)
        #            digitmap[digit] = img
        #return digitmap
    def all_ballotimgs(bal2imgs, side, ignorelist=None):
        """ Generate all ballot images for either side 0 or 1. """
        # TODO: Generalize to N-sided ballots
        paths = []
        ignorelist = ignorelist if ignorelist != None else []
        for ballotid, path in bal2imgs.iteritems():
            if ballotid in ignorelist:
                continue
            elif side == 0:
                paths.append(path[0])
            else:
                paths.append(path[1])
        return paths
    def get_best_side(results_side0, results_side1):
        """ Given the result of digitParse for both side0 and side1,
        return the 'best' results where, for each imgpath, we return
        the match for whose side's score is maximized.
        Input:
            list results_side0: [(imgpath0_i, ocr_str_i, meta_i, isflip_i), ...]
            list results_side1: [(imgpath1_i, ocr_str_i, meta_i, isflip_i), ...]
        Output:
            list of [(imgpath_i, ocr_str_i, meta_i, isflip_i, side_i), ...]
        """
        assert len(results_side0) == len(results_side1)
        results_side0 = sorted(results_side0, key=lambda tup: tup[0])
        results_side1 = sorted(results_side1, key=lambda tup: tup[0])
        results = []
        for idx, (path_side0, ocrstr_side0, meta_side0, isflip_side0) in enumerate(results_side0):
            path_side1, ocrstr_side1, meta_side1, isflip_side1 = results_side1[idx]
            assert len(meta_side0) == len(meta_side1)
            avg_score_side0 = sum([tup[6] for tup in meta_side0]) / float(len(meta_side0))
            avg_score_side1 = sum([tup[6] for tup in meta_side1]) / float(len(meta_side1))
            if avg_score_side0 > avg_score_side1:
                results.append((path_side0, ocrstr_side0, meta_side0, isflip_side0, 0))
            else:
                results.append((path_side1, ocrstr_side1, meta_side1, isflip_side1, 1))
        assert (len(results) == len(results_side0)) and (len(results) == len(results_side1))
        return results
    if ignorelist == None:
        ignorelist = []
    result = {} # maps {ballotid: ((attrtype_i, ocrresult_i, meta_i, isflip_i, side_i), ...)}
    if os.path.exists(os.path.join(project.projdir_path,
                                   project.digitmultexemplars)):
        print "Loading previously-computed Digit Exemplars."
        exemplars = pickle.load(open(os.path.join(project.projdir_path,
                                                  project.digitmultexemplars_map),
                                     'rb'))
    else:
        # 0.) Construct digit exemplars
        # exemplars := maps {str digit: ((temppath_i, bb_i, exemplarP_i), ...)}
        t = time.time()
        print "Computing Digit Exemplars..."
        exemplars = compute_digit_exemplars(project)
        dur = time.time() - t
        print "...Finished Computing Digit Exemplars ({0} s)".format(dur)

    digit_exs = make_digithashmap(project, exemplars)
    numdigitsmap = pickle.load(open(os.path.join(project.projdir_path, 
                                                 project.num_digitsmap),
                                    'rb'))
    voteddigits_dir = os.path.join(project.projdir_path,
                                     project.voteddigits_dir)
    img2bal = pickle.load(open(project.image_to_ballot, 'rb'))
    if os.path.exists(voteddigits_dir):
        # {str ballotid: ((attrtype_i,ocrstr_i,meta_i,flip_i,side_i), ...)}
        digitgroup_results = load_digitgroup_results(project)
        if digitgroup_results == None:
            shutil.rmtree(voteddigits_dir)
        else:
            for ballotid, tuples in digitgroup_results.iteritems():
                # This is indeed 'ballotid', not 'votedpath'.
                if ballotid in bal2imgs:
                    for (digitattr,ocrstr,meta,isflip,side) in tuples:
                        for (y1,y2,x1,x2,digit,digitimgpath,score) in meta:
                            try:
                                os.remove(digitimgpath)
                            except OSError as e:
                                pass
    digitmatch_info = {}  # maps {str patchpath: ((y1,y2,x1,x2), side, isflip, ballotid)}

    for digitattr, ((y1,y2,x1,x2),side) in digitattrs.iteritems():
        num_digits = numdigitsmap[digitattr]
        # add some border, for good measure
        w, h = abs(x1-x2), abs(y1-y2)
        c = 0.0    # On second thought, Don't add any Border...
        bb = [max(0, y1-int(round(h*c))),
              y2+int(round(h*c)),
              max(0, x1-int(round(w*c))),
              x2+int(round(w*c))]
        median_dist = compute_median_dist(project, digitattr)
        # a list of results [(imgpath_i, ocr_str_i, meta_i, isflip_i, side_i), ...]
        if not util.is_multipage(project):
            digitparse_results = common.do_digitocr(all_ballotimgs(bal2imgs, 0, ignorelist=ignorelist),
                                                    digit_exs,
                                                    num_digits,
                                                    bb=bb, rejected_hashes=rejected_hashes,
                                                    accepted_hashes=accepted_hashes,
                                                    digitdist=median_dist)
            digitparse_results = [tuple(thing)+(0,) for thing in digitparse_results]
        else:
            results_side0 = common.do_digitocr(all_ballotimgs(bal2imgs, 0, ignorelist=ignorelist),
                                               digit_exs,
                                               num_digits,
                                               bb=bb, rejected_hashes=rejected_hashes,
                                               accepted_hashes=accepted_hashes,
                                               digitdist=median_dist)
            results_side1 = common.do_digitocr(all_ballotimgs(bal2imgs, 1, ignorelist=ignorelist),
                                               digit_exs,
                                               num_digits,
                                               bb=bb, rejected_hashes=rejected_hashes,
                                               accepted_hashes=accepted_hashes,
                                               digitdist=median_dist)
            digitparse_results = get_best_side(results_side0, results_side1)
        # With PIL Crop: 3.63 s
        # With scipy sh.standardImread: 19.174 s
        # With scipy misc.imread: 6.59 s
        _t = time.time()
        print "== Extracting Voted Digit Patches..."
        # Note: I use pass_idx=True because I need to assign a unique
        #       ID to the {0}_votedextract.png patch paths.

        r, d = partask.do_partask(extract_voted_digitpatches,
                                  digitparse_results,
                                  _args=(bb, digitattr, voteddigits_dir, img2bal, project.samplesdir),
                                  combfn=my_combfn,
                                  init=({},{}),
                                  pass_idx=True)

        for ballotid, lsts in r.iteritems():
            result.setdefault(ballotid,[]).extend(lsts)
        for patchpath, (bb, side, isflip, ballotid) in d.iteritems():
            digitmatch_info[patchpath] = (bb, side, isflip, ballotid)
        print "== Finished Extracting Voted Digit Patches. Took {0}s.".format(time.time()-_t)
            
    return result, digitmatch_info

def my_combfn(result, subresult):
    """ The combfn used for partask.do_partask. """
    comb_result, comb_digitmatch_info = dict(result[0]), dict(result[1])
    sub_result, sub_digitmatch_info = subresult
    for ballotid, lsts in sub_result.iteritems():
        comb_result.setdefault(ballotid, []).extend(lsts)
    for patchpath, (bb, side, isflip, ballotid) in sub_digitmatch_info.iteritems():
        comb_digitmatch_info[patchpath] = (bb, side, isflip, ballotid)
    return (comb_result, comb_digitmatch_info)

def extract_voted_digitpatches(stuff, (bb, digitattr, voteddigits_dir, img2bal, voteddir), idx):
    """ Extracts each digit from each voted ballot, saves them to an
    output directory, and stores some meta-data.
    Input:
        list stuff: ((imgpath_i, ocr_str_i, meta_i, isflip_i, side_i), ...)
    Output:
        (dict result, digt digitmatch_info), where result is:
            {str ballotid: ((digitattr_i, ocrstr_i, meta_i, flip_i, side_i), ...)}
        and digitmatch_info is:
            {str patchpath: ((y1,y2,x1,x2), side, bool isflip, ballotid)}
    """
    result = {}  # maps {str ballotid: [(digitattr_i, ocrstr_I, meta_i, flip_i, side_i), ...]}
    digitmatch_info = {}  # maps {str patchpath: ((y1,y2,x1,x2), side, bool is_flip, str ballotid)}
    ctr = 0
    for (imgpath, ocr_str, meta, isflip, side) in stuff:
        ballotid = img2bal[imgpath]
        meta_out = []
        for (y1,y2,x1,x2, digit, digitimg, score) in meta:
            rootdir = os.path.join(voteddigits_dir, digit)
            # Recreate directory structure
            voteddir_abs = os.path.abspath(voteddir)
            if voteddir_abs[-1] != '/':
                voteddir_abs += '/'
            ballotid_abs = os.path.abspath(ballotid)
            rootdir = os.path.join(rootdir, ballotid_abs[len(voteddir_abs):])
            util.create_dirs(rootdir)
            outpath = os.path.join(rootdir, '{0}_{1}_votedextract.png'.format(idx, ctr))
            digitmatch_info[outpath] = ((y1,y2,x1,x2), side, isflip, ballotid)
            assert isinstance(isflip, bool)
            #img = cv.LoadImage(imgpath, False)
            #if isflip == True:
            #    cv.Flip(img, flipMode=-1)
            img = scipy.misc.imread(imgpath, flatten=True)
            if isflip == True:
                img = sh.fastFlip(img)
            img = sh.remove_border_topleft(img)
            # _y1, etc. are coordinates of digit patch w.r.t image coords.
            # Expand by E pixels, for user benefit.
            E = 3
            _y1 = int(bb[0]+y1 - E)
            _y2 = int(bb[0]+y2 + E)
            _x1 = int(bb[2]+x1 - E)
            _x2 = int(bb[2]+x2 + E)
            outdigitpatch = img[_y1:_y2, _x1:_x2]
            #cv.SaveImage(outpath, outdigitpatch)
            scipy.misc.imsave(outpath, outdigitpatch)
            meta_out.append((y1,y2,x1,x2, digit, outpath, score))
            ctr += 1
        result.setdefault(ballotid, []).append((digitattr, ocr_str, meta_out, isflip, side))
    return result, digitmatch_info

def compute_digit_exemplars(proj):
    """ Computes multiple digit exemplars, in order to enhance the
    digit grouping.
    Input:
        obj proj
    Output:
        A dict, mapping {str digit: (str exemplarpath_i, ...)}
    """
    digit_exemplars_mapP = pathjoin(proj.projdir_path,
                                    proj.digit_exemplars_map)
    # maps {str digit: ((regionpath_i, score_i, bb_i, digitpatchpath_i), ...)}
    digit_exemplars_map = pickle.load(open(digit_exemplars_mapP, 'rb'))

    # 0.) Munge digit_exemplars_map into compatible-format
    mapping = {} # maps {str digit: ((regionpath_i, bb_i), ...)}
    for digit, tuples in digit_exemplars_map.iteritems():
        thing = []
        for (regionpath, score, bb, patchpath) in tuples:
            thing.append((regionpath, bb))
        mapping[digit] = thing

    # exemplars := maps {str digit: ((regionpath_i, bb_i), ...)}
    exemplars = group_attrs.compute_exemplars_fullimg(mapping, MAXCAP=10)
    digitmultexemplars_map = {} # maps {str digit: ((regionpath_i, bb_i, patchpath_i), ...)}
    for digit, tuples in exemplars.iteritems():
        for i, (regionpath, bb) in enumerate(tuples):
            regionimg = scipy.misc.imread(regionpath) # don't open a grayscale img twice, tends to lighten it
            patch = regionimg[bb[0]:bb[1], bb[2]:bb[3]]
            rootdir = os.path.join(proj.projdir_path,
                                   proj.digitmultexemplars,
                                   digit)
            util_gui.create_dirs(rootdir)
            exemplarP = pathjoin(rootdir, '{0}.png'.format(i))
            scipy.misc.imsave(exemplarP, patch)
            digitmultexemplars_map.setdefault(digit, []).append((regionpath, bb, exemplarP))
    pickle.dump(digitmultexemplars_map,
                open(os.path.join(proj.projdir_path, proj.digitmultexemplars_map), 'wb'))
    return digitmultexemplars_map

def max_bb_dims(bb1, bb2):
    return (min(bb1[0], bb2[0]), max(bb1[1], bb2[1]),
            min(bb1[2], bb2[2]), max(bb1[3], bb2[3]))

def compute_median_dist(proj, digitattr):
    """ Computes the median (horiz) distance between adjacent digits,
    based off of the digits from the blank ballots.
    Input:
        obj proj:
        str digitattr: Which digit-based attribute to compute the
                       median-distance for.
    Output:
        int distance, in pixels.
    """
    digit_med_dists = load_digit_median_dists(proj)
    if digit_med_dists == None:
        digit_med_dists = {}
    dist = digit_med_dists.get(digitattr, None)
    if dist != None:
        return dist
    # Bit hacky - peer into LabelDigit's 'matches' internal state
    labeldigits_stateP = pathjoin(proj.projdir_path, proj.labeldigitstate)
    # matches maps {str regionpath: ((patchpath_i,matchID_i,digit,score,y1,y2,x1,x2,rszFac_i), ...)
    matches = pickle.load(open(labeldigits_stateP, 'rb'))['matches']
    dists = [] # stores adjacent distances
    for regionpath, tuples in matches.iteritems():
        x1_all = []
        for (patchpath, matchID, digit, score, y1, y2, x1, x2, rszFac) in tuples:
            x1_all.append(int(round(x1 / rszFac)))
        x1_all = sorted(x1_all)
        for i, x1 in enumerate(x1_all[:-1]):
            x1_i = x1_all[i+1]
            dists.append(int(round(abs(x1 - x1_i))))
    dists = sorted(dists)
    if len(dists) <= 2:
        median_dist = min(dists)
    else:
        median_dist = dists[int(len(dists) / 2)]
    print '=== median_dist is:', median_dist
    digit_med_dists[digitattr] = median_dist
    save_digit_median_dists(proj, digit_med_dists)
    return median_dist

def load_digit_median_dists(proj):
    """ Returns the digit median distances if it exists, or None o.w.
    Input:
        obj proj:
    Output:
        A dict {str digittype: int median_distance}
    """
    digit_med_distsP = pathjoin(proj.projdir_path,
                                proj.digit_median_dists)
    if not os.path.exists(digit_med_distsP):
        return None
    digit_med_dists = pickle.load(open(digit_med_distsP, 'rb'))
    return digit_med_dists

def save_digit_median_dists(proj, digit_median_dists):
    """ Stores the digit_median_dists dict to an output file.
    Input:
        obj proj:
        dict digit_median_dists: maps {str digitattr: int median_distance}
    """
    digit_med_distsP = pathjoin(proj.projdir_path, proj.digit_median_dists)
    pickle.dump(digit_median_dists, open(digit_med_distsP, 'wb'))

def get_digitmatch_info(proj):
    """ Loads the digitmatch_info data structure, which is of the form:
        {str patchpath: ((y1,y2,x1,x2), str side, bool isflip, str ballotid)
    """
    digitmatch_infoP = pathjoin(proj.projdir_path, proj.digitmatch_info)
    digitmatch_info = pickle.load(open(digitmatch_infoP, 'rb'))
    return digitmatch_info

def get_digitpatch_info(proj, patchpath, digitmatch_info=None):
    """ Given the path to a digit-patch (from a votedballot), return
    the ((y1,y2,x1,x2), side, bool isflip, ballotid) region from the votedballot it was extracted
    from.
    Input:
        obj proj:
        str patchpath: Path of a digitpatch from some image
        dict digitmatch_info: 
    Output:
        ((y1,y2,x1,x2), str side, bool isflip, str ballotid)
    """
    if digitmatch_info == None:
        digitmatch_info = get_digitmatch_info(proj)
    if patchpath not in digitmatch_info:
        print "Uhoh, couldn't find patchpath."
        pdb.set_trace()
    assert patchpath in digitmatch_info
    return digitmatch_info[patchpath]

def load_digitgroup_results(proj):
    """ Loads the results of doing grouping-by-digits.
    Input:
        obj proj
    Output:
        {str ballotid: ((digitattr_i, ocrstr_i, meta_i, isflip_i, side_i), ...])} if
        it exists, or None o.w.
    """
    path = pathjoin(proj.projdir_path, proj.digitgroup_results)
    if os.path.exists(path):
        return pickle.load(open(path, 'rb'))
    else:
        return None

def save_digitgroup_results(proj, digitgroup_results):
    """ Saves the results of doing grouping-by-digits.
    Input:
        obj proj:
        dict digitgroup_results: maps
          {ballotid: [(digitattr_i, ocrstr_i, meta_i, isflip_i, side_i),...]}
        where meta_i is numDigits-tuples of the form:
          [(y1,y2,x1,x2,digit_i, outpath_i, score_i), ...]
    """
    outpath = os.path.join(proj.projdir_path, proj.digitgroup_results)
    pickle.dump(digitgroup_results, open(outpath, 'wb'))

def save_digitmatch_info(proj, digitmatch_info):
    """ Saves the digitmatch_info dictionary, which contains information
    about all extracted digit patches from voted ballots.
    Input:
        obj proj:
        dict digitmatch_info: maps {str patchpath: ((y1,y2,x1,x2), str side, bool isflip, str ballotid)}
    """
    outpath = pathjoin(proj.projdir_path, proj.digitmatch_info)
    pickle.dump(digitmatch_info, open(outpath, 'wb'))

def to_groupclasses_digits(proj, digitgroup_results, ignorelist=None, grouplabel_record=None):
    """ Converts the result of do_digitocr_patches to a list of 
    GroupClass instances.
    Input:
        obj proj:
        dict digitgroup_results: maps 
            {str ballotid: ((attrtype_i, ocrstr_i, meta_i, isflip_i, side_i),...)}
          where meta_i is numDigits-tuples of the form:
            (y1,y2,x1,x2,digit_i, digitimgpath_i, score)
        list ignorelist: List of sampleids to ignore.
        list grouplabel_record: The canonical ordering of attributes for
            this election. If not given, then this will load it from disk.
    Output:
        List of GroupClass instances.
    """
    if ignorelist == None:
        ignorelist = []
    if grouplabel_record == None:
        grouplabel_record = common.load_grouplabel_record(proj)
    bal2imgs=pickle.load(open(proj.ballot_to_images,'rb'))
    digitpatchpath_scoresP = pathjoin(proj.projdir_path,
                                      proj.digitpatchpath_scoresVoted)
    # maps {str patchpath: float score}, used for 'Split'
    digitpatchpath_scores = {}

    digits_results = {} # maps {str digit: list of [ballotid, patchpath, isflip_i, side_i]}
    attr_types = common.get_attrtypes(proj)
    def removedups_rankedlist(rlist):
        """ Remove duplicates from the input ranked list, starting
        from L-R. This is needed because Kais' NCC-OCR code can,
        for a given digit patch, signal that a given digit D
        occurs in the patch multiple times.
        """
        result = []
        for i, grouplabel in enumerate(rlist):
            if i >= len(rlist)-1:
                break
            if grouplabel not in rlist[i+1:]:
                result.append(grouplabel)
        return result
    def sanity_check_rankedlist(rlist):
        history = {}
        for grouplabel in rlist:
            if grouplabel in history:
                print 'woah, this grouplabel was already here:', grouplabel
                pdb.set_trace()
                return False
            else:
                history[grouplabel] = True
        return True
    # Munge the grouping results into grouping_results, digit_results
    ## Note: the isflip_i/side_i info from digitgroup_results gets
    ## thrown out after these blocks. Do I actually need them?
    digitattrtype = None
    if not util.is_multipage(proj):
        for attr_type in attr_types:
            for ballotid in bal2imgs:
                if ballotid in ignorelist:
                    continue
                if common.is_digitbased(proj, attr_type):
                    if digitattrtype == None:
                        digitattrtype = attr_type
                    if ballotid not in digitgroup_results:
                        # This is OK, it means that we did not have to
                        # run digitocr on ballotid.
                        continue
                    for (attrtype_i, ocr_str_i, meta_i, isflip_i, side_i) in digitgroup_results[ballotid]:
                        if attrtype_i == attr_type:
                            for (y1,y2,x1,x2, digit, digitpatchpath, score) in meta_i:
                                digits_results.setdefault(digit, []).append((ballotid, digitpatchpath))
                                digitpatchpath_scores[digitpatchpath] = score
                            break
    else:
        # Multipage
        for attr_type in attr_types:
            for ballotid, (frontpath, backpath) in bal2imgs.iteritems():
                if ballotid in ignorelist:
                    continue
                if common.is_digitbased(proj, attr_type):
                    if digitattrtype == None:
                        digitattrtype = attr_type
                    # Note: digitgroup_results has correct side info
                    sidepath = frontpath if frontpath in digitgroup_results else backpath
                    if sidepath not in digitgroup_results:
                        # This is OK, it means that we didn't have to
                        # run digitocr on ballotid.
                        continue
                    for (attrtype_i, ocr_str_i, meta_i, isflip_i, side_i) in digitgroup_results[sidepath]:
                        if attrtype_i == attr_type:
                            for (y1,y2,x1,x2,digit,digitpatchpath,score) in meta_i:
                                digits_results.setdefault(digit, []).append((sidepath, digitpatchpath))
                                digitpatchpath_scores[digitpatchpath] = score
                            break

    groups = []
    # Seed initial set of digit-based groups
    #alldigits = digits_results.keys()
    alldigits = common.get_attrtype_possiblevals(proj, digitattrtype)
    for digit, lst in digits_results.iteritems():
        elements = []
        rankedlist = make_digits_rankedlist(digit, alldigits, grouplabel_record)
        for (ballotid, patchpath) in lst:
            elements.append((ballotid, rankedlist, patchpath))
        group = common.DigitGroupClass(elements,
                                       user_data=digitpatchpath_scores)
        groups.append(group)
    return groups

def make_digits_rankedlist(d, digits, gl_record):
    #intuition = {'0': ('8', '9', ''),
    #             '1': '7',
    #             '2': '0',
    #             '3': '8',
    #             '4': '5',
    #             '5': '4',
    #             '6': 
    cpy = list(digits)[:]
    cpy.remove(d)
    cpy.insert(0, d)
    result = []
    for digit in cpy:
        grouplabel = common.make_grouplabel(('digit', digit))
        gl_idx = gl_record.index(grouplabel)
        result.append(gl_idx)
    return result
