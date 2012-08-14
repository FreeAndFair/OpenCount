import sys, os, pdb, time, pickle, shutil
import scipy, scipy.misc
from os.path import join as pathjoin
sys.path.append('..')
import specify_voting_targets.util_gui as util_gui
import pixel_reg.shared as sh
import util, common
from PIL import Image

def do_digitocr_patches(bal2imgs, digitattrs, project,
                        rejected_hashes=None, ignorelist=None):
    """ For each digitbased attribute, run our NCC-OCR on the patch
    (using our digit exemplars).
    Input:
        dict bal2imgs
        dict digitattrs: maps {attrtype: ((y1,y2,x1,x2), side)}
        obj project
        dict rejected_hashes: maps {imgpath: {digit: [((y1,y2,x1,x2),side_i), ...]}}
        list ignorelist: List of imagepaths to not process, i.e. if imgs
                         have already been verified by the user. Should
                         be sampleids.
    Output:
        A dict that maps:
          {ballotid: ((attrtype_i, ocrresult_i, meta_i, isflip_i, side_i), ...)
        where meta_i is a tuple containing numDigits tuples:
          (y1_i,y2_i,x1_i,x2_i, str digit_i, str digitimgpath_i, score)
    """
    def make_digithashmap(project):
        digitmap = {} # maps {str digit: obj img}
        digit_exemplarsdir = os.path.join(project.projdir_path,
                                          project.digit_exemplars_outdir)
        digitdirs = os.listdir(digit_exemplarsdir)
        for digitdir in digitdirs:
            # Assumes this has directories of the form:
            #    0_examples/*.png
            #    1_examples/*.png
            #    ...
            fullpath = os.path.join(digit_exemplarsdir, digitdir)
            digit = digitdir.split('_')[0]
            for dirpath, dirnames, filenames in os.walk(fullpath):
                for imgname in [f for f in filenames if util_gui.is_image_ext(f)]:
                    # This currently scans through all images, unnecessary.
                    # We only need just one.
                    imgpath = os.path.join(dirpath, imgname)
                    img = sh.standardImread(imgpath, flatten=True)
                    digitmap[digit] = img
        return digitmap
    def all_ballotimgs(bal2imgs, side, ignorelist=None):
        """ Generate all ballot images for either side 0 or 1. """
        # TODO: Generalize to N-sided ballots
        paths = []
        for ballotid, path in bal2imgs.iteritems():
            if ignorelist != None and ballotid in ignorelist:
                continue
            if side == 0:
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
    result = {}
    digit_exs = make_digithashmap(project)
    numdigitsmap = pickle.load(open(os.path.join(project.projdir_path, 
                                                 project.num_digitsmap),
                                    'rb'))
    voteddigits_dir = os.path.join(project.projdir_path,
                                     project.voteddigits_dir)
    img2bal = pickle.load(open(project.image_to_ballot, 'rb'))
    if os.path.exists(voteddigits_dir):
        print "Removing everything in:", voteddigits_dir
        shutil.rmtree(voteddigits_dir)
    ctr = 0
    for digitattr, ((y1,y2,x1,x2),side) in digitattrs.iteritems():
        num_digits = numdigitsmap[digitattr]
        # add some border, for good measure
        w, h = abs(x1-x2), abs(y1-y2)
        c = 0.0    # NO BORDER
        bb = [max(0, y1-int(round(h*c))),
              y2+int(round(h*c)),
              max(0, x1-int(round(w*c))),
              x2+int(round(w*c))]
        # a list of results [(imgpath_i, ocr_str_i, meta_i, isflip_i, side_i), ...]
        if not util.is_multipage(project):
            digitparse_results = common.do_digitocr(all_ballotimgs(bal2imgs, 0, ignorelist=ignorelist),
                                                    digit_exs,
                                                    num_digits,
                                                    bb=bb, rejected_hashes=rejected_hashes)
            digitparse_results = [tuple(thing)+(0,) for thing in digitparse_results]
        else:
            results_side0 = common.do_digitocr(all_ballotimgs(bal2imgs, 0, ignorelist=ignorelist),
                                               digit_exs,
                                               num_digits,
                                               bb=bb, rejected_hashes=rejected_hashes)
            results_side1 = common.do_digitocr(all_ballotimgs(bal2imgs, 1, ignorelist=ignorelist),
                                               digit_exs,
                                               num_digits,
                                               bb=bb, rejected_hashes=rejected_hashes)
            digitparse_results = get_best_side(results_side0, results_side1)
        for (imgpath, ocr_str, meta, isflip, side) in digitparse_results:
            meta_out = []
            for (y1,y2,x1,x2, digit, digitimg, score) in meta:
                rootdir = os.path.join(voteddigits_dir, digit)
                util.create_dirs(rootdir)
                outpath = os.path.join(rootdir, '{0}_votedextract.png'.format(ctr))
                #scipy.misc.imsave(outpath, digitimg)
                Image.open(imgpath).crop((int(bb[2]+x1),int(bb[0]+y1),int(bb[2]+x2),int(bb[0]+y2))).save(outpath)
                meta_out.append((y1,y2,x1,x2, digit, outpath, score))
                ctr += 1
            ballotid = img2bal[imgpath]
            result.setdefault(ballotid, []).append((digitattr, ocr_str, meta_out, isflip, side))
    return result

def get_digitmatch_info(proj, patchpath):
    """ Given the path to a digit-patch (from a votedballot), return
    the (y1,y2,x1,x2) region from the votedballot it was extracted
    from.
    """
    res_path = os.path.join(proj.projdir_path, proj.digitgroup_results)
    digitgroup_results = pickle.load(open(res_path, 'rb'))
    for ballotid, matches in digitgroup_results.iteritems():
        for (digitattr, ocrstr, meta, isflip, side) in matches:
            for (y1,y2,x1,x2,digit,outpath,score) in meta:
                if patchpath == outpath:
                    return (y1,y2,x1,x2), side
    print "Uhoh, couldn't find patchpath."
    pdb.set_trace()
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

def to_groupclasses_digits(proj, digitgroup_results, ignorelist=None):
    """ Converts the result of do_digitocr_patches to a list of 
    GroupClass instances.
    Input:
        obj proj:
        dict digitgroup_results: maps 
            {str ballotid: ((attrtype_i, ocrstr_i, meta_i, isflip_i, side_i),...)}
          where meta_i is numDigits-tuples of the form:
            (y1,y2,x1,x2,digit_i, digitimgpath_i, score)
        list ignorelist: List of sampleids to ignore.
    Output:
        List of GroupClass instances.
    """
    if ignorelist == None:
        ignorelist = []
    bal2imgs=pickle.load(open(proj.ballot_to_images,'rb'))

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
    if not util.is_multipage(proj):
        for attr_type in attr_types:
            for ballotid in bal2imgs:
                if ballotid in ignorelist:
                    continue
                if common.is_digitbased(proj, attr_type):
                    for (attrtype_i, ocr_str_i, meta_i, isflip_i, side_i) in digitgroup_results[ballotid]:
                        if attrtype_i == attr_type:
                            for (y1,y2,x1,x2, digit, digitpatchpath, score) in meta_i:
                                digits_results.setdefault(digit, []).append((ballotid, digitpatchpath))
                            break
    else:
        # Multipage
        for attr_type in attr_types:
            for ballotid, (frontpath, backpath) in bal2imgs.iteritems():
                if ballotid in ignorelist:
                    continue
                if common.is_digitbased(proj, attr_type):
                    # Note: digitgroup_results has correct side info
                    sidepath = frontpath if frontpath in digitgroup_results else backpath
                    if sidepath not in digitgroup_results:
                        print "Uhoh, sidepath not in digitgroup_results"
                        pdb.set_trace()
                    for (attrtype_i, ocr_str_i, meta_i, isflip_i, side_i) in digitgroup_results[sidepath]:
                        if attrtype_i == attr_type:
                            for (y1,y2,x1,x2,digit,digitpatchpath,score) in meta_i:
                                digits_results.setdefault(digit, []).append((sidepath, digitpatchpath))
                            break

    groups = []
    # Seed initial set of digit-based groups
    alldigits = digits_results.keys()
    for digit, lst in digits_results.iteritems():
        elements = []
        rankedlist = make_digits_rankedlist(digit, alldigits)
        for (ballotid, patchpath) in lst:
            elements.append((ballotid, rankedlist, patchpath))
        group = common.GroupClass(elements)
        groups.append(group)
    return groups

def make_digits_rankedlist(d, digits):
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
        result.append(grouplabel)
    return result
