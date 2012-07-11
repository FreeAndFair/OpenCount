import sys, os, pickle, pdb, wx, time
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
    def munge_matches(matches, grouplabel, patchpaths, d, outdir='extract_attrs_templates'):
        """ Given matches from find_patch_matches, convert it to
        a format that GroupClass likes:
          (sampleid, rankedlist, patchpath)
        Also saves the registered patches to output dir.
        """
        results = []
        for (filename, score1, score2, Ireg, y1,y2,x1,x2, rszFac) in matches:
            relpath = os.path.relpath(filename)
            patchpath = patchpaths[relpath]
            results.append((relpath, (grouplabel,), patchpath))

            # Save registered patch
            attrtype = tuple(sorted(d['attrs'].keys()))
            relpath = os.path.relpath(filename)
            attrtype_str = '-'.join(attrtype)
            outdir_full = os.path.join(projdir_path, outdir, attrtype_str)
            outpath_full = os.path.join(outdir_full, util.encodepath(relpath)+'.png')
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
    while tmp2imgs and not no_change:
        print '== Running iteration:', n_iters
        flag = False
        for d in attrdata:
            print 'Trying to group attribute:', d['attrs']
            x1, y1 = int(round(d['x1']*w_img)), int(round(d['y1']*h_img))
            x2, y2 = int(round(d['x2']*w_img)), int(round(d['y2']*h_img))
            side = d['side']
            attrtype = tuple(sorted(d['attrs'].keys()))
            temppaths = []
            patch = None   # Current patch we're examining
            for tempid, paths in tmp2imgs.iteritems():
                if side == 'front':
                    path = paths[0]
                else:
                    path = paths[1]
                if (attrtype, path) in history:
                    continue
                temppaths.append(path)
                if patch == None:
                    tempimg = shared.standardImread(path,flatten=True)
                    patch = tempimg[y1:y2, x1:x2]
                    patch[0,0] = 0.99
                                      
            if patch == None:
                # If we get here, then for the given attribute d,
                # we've grouped every blank ballot.
                continue
            global _i
            #scipy.misc.imsave("{0}_{1}.png".format(str(attrtype), _i), patch)

            _i += 1
            #patchpaths = extract_temp_patches(d, temppaths)
            patchpaths = get_temp_patches(d, temppaths)
            _t = time.time()
            h, w = patch.shape
            x1, y1 = 0, 0
            x2, y2 = w-3, h-3
            bb = [y1, y2, x1, x2]
            matches = shared.find_patch_matchesV1(patch, bb, temppaths, threshold=0.6)
            
            _endt = time.time() - _t
            print "len(matches): {0}  time: {1} avgtime per template: {2}".format(len(matches),
                                                                     _endt,
                                                                     _endt / len(temppaths))
            if matches:
                flag = True
                # First handle 'found' templates
                for (filename, _, _, _,  _, _, _, _, rszFac) in matches:
                    history.add((attrtype, filename))
                inc_counter(attrtype_ctr, attrtype)
                grouplabel = common.make_grouplabel((attrtype, attrtype_ctr[attrtype]))
                elements = munge_matches(matches, grouplabel, patchpaths, d)
                in_group = common.GroupClass(elements)
                groups.append(in_group)
        if not flag:
            # Convergence achieved, stop iterating
            no_change = True
        n_iters += 1
    print "== Total Time:", time.time() - _starttime
    return groups
    
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
    imgsize = (1715, 2847)
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

