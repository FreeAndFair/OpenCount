import os, sys, pdb, wx
from labelcontest.labelcontest import LabelContest
from PIL import Image
import scipy
import scipy.misc
import pickle
import csv

sys.path.append('../')
import pixel_reg.shared as shared
from specify_voting_targets import util_gui
import common
import util
import verify_overlays

DUMMY_ROW_ID = -42

class TestFrame(wx.Frame):
    def __init__(self, parent, project, *args, **kwargs):
        wx.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.project = project

        self.panel = verify_overlays.VerifyPanel(self, verifymode=verify_overlays.VerifyPanel.MODE_YESNO2)
        self.Show()
        self.Maximize()
        self.Fit()
        
        attrdata = pickle.load(open(self.project.ballot_attributesfile, 'rb'))
        groups = group_attributes(attrdata, self.project)

        self.panel.start(groups, None, ondone=self.verify_done)
        
    def verify_done(self, results):
        """ Called when the user finished verifying the auto-grouping of
        attribute patches (for blank ballots).
        results is a dict mapping:
            {grouplabel: list of (str temppath, rankedlist, str patchpath)}
        The idea is that for each key-value pairing (k,v) in results, the
        user has said that: "These blank ballots in 'v' all have the same
        attribute value, since their overlays looked the same."
        The next step is for the user to actually label these groups (instead
        of labeling each individual blank ballot). The hope is that the
        number of groups is much less than the number of blank ballots.
        """
        print "Verifying done."
        for grouplabel,elements in results.iteritems():
            print "For grouplabel {0}, there were {1} elements:".format(grouplabel, len(elements))
        print "The idea: Each group contains images whose overlays \
are the same. It might be the case that two groups could be merged \
together."

class LabelAttributesPanel(LabelContest):
    """
    This class is one big giant hack of a monkey-patch.
    """
    def gatherData(self):
        self.groupedtargets = []
        self.dirList = []
    
        # attrdata is a list of dicts (marshall'd AttributeBoxes)
        attrdata = pickle.load(open(self.proj.ballot_attributesfile))

        frame = TestFrame(self, self.proj)

        self.sides = [x['side'] for x in attrdata]
        self.types = [x['attrs'].keys()[0] for x in attrdata]

        print "LOAD", attrdata

        width, height = self.proj.imgsize
        self.dirList = [os.path.join(self.proj.blankballots_straightdir,x) for x in os.listdir(self.proj.blankballots_straightdir)]
        for f in self.dirList:
            thisballot = [[(at['id'], 0,
                          int(at['x1']*width), int(at['y1']*height), 
                          int(at['x2']*width), int(at['y2']*height))] for at in attrdata]
    
            self.groupedtargets.append(thisballot)
        self.groupedtargets_back = self.groupedtargets

        self.template_width, self.template_height = Image.open(self.dirList[0]).size

    def unsubscribe_pubsubs(self):
        pass

    def setupBoxes(self):
        self.boxes = []
        for each in self.groupedtargets:
            bb = []
            for contest in each:
                id,_,l,u,r,d = contest[0]
                bb.append((id,l,u,r,d))
            self.boxes.append(bb)
        self.groupedtargets = [[[]]*len(x) for x in self.groupedtargets]

        # EVEN BIGGER HACK!!!
        self.text = {}
        for i,each in enumerate(self.boxes):
            for x in each:
                self.text[i,x[0]] = []
                self.voteupto[i,x[0]] = 1
        if os.path.exists(self.proj.attr_internal):
            self.text = pickle.load(open(self.proj.attr_internal))
        # </hack>

    def addText(self):
        LabelContest.addText(self)
        name = self.types[self.count]
        self.contesttitle.SetLabel("Attribute Value (%s)"%name)

    def save(self):
        self.saveText(removeit=False)
        print "TEXT", self.text
        if not os.path.exists(self.proj.patch_loc_dir):
            os.mkdir(self.proj.patch_loc_dir)
        pickle.dump(self.text, open(self.proj.attr_internal, "w"))
        for ballot in range(len(self.dirList)):
            vals = sorted([(k,v) for k,v in self.text.items() if k[0] == ballot])
            name = os.path.splitext(os.path.split(self.dirList[ballot])[-1])[0]+"_patchlocs.csv"
            name = os.path.join(self.proj.patch_loc_dir, name)
            print "MAKING", name
            out = csv.writer(open(name, "w"))
            out.writerow(["imgpath","id","x","y","width",
                          "height","attr_type","attr_val","side"])
            out.writerow([os.path.abspath(self.dirList[ballot]), DUMMY_ROW_ID,0,0,0,0,"_dummy_","_dummy_","_dummy_"])
            for uid,each in enumerate(vals):
                pos = self.groupedtargets_back[ballot][uid][0]
                print "POS IS", pos, "EACH", each
                value = "_none_" if each[1] == [] else each[1][0]
                out.writerow([os.path.abspath(self.dirList[ballot]),
                              uid, pos[2], pos[3],
                              pos[4]-pos[2], pos[5]-pos[3],
                              self.types[uid], value, self.sides[uid]])
    def validate_outputs(self):
        return True
    def stop(sefl):
        pass
    def export_bounding_boxes(sefl):
        pass
    def checkCanMoveOn(self):
        return True

def group_attributes(attrdata, project):
    """ Using NCC, group all attributes to try to reduce operator
    effort.
      attrdata: A list of dicts (marshall'd AttributeBoxes)
      project: A Project.
    """
    def munge_matches(matches, grouplabel, patchpaths):
        """ Given matches from find_patch_matches, convert it to
        a format that GroupClass likes:
          (sampleid, rankedlist, patchpath)
        """
        results = []
        for (filename, score, rszFac, l,r,u,d) in matches:
            relpath = os.path.relpath(filename)
            patchpath = patchpaths[relpath]
            results.append((relpath, (grouplabel,), patchpath))
        return results
    def extract_temp_patches(d, temppaths, outdir='extract_attrs_templates'):
        """ Save the attribute patch pointed to by d, and return a
        mapping that does:
          {str filenname: str patchpath}
        """
        result = {}
        img_w, img_h = project.imgsize
        x1, y1 = int(round(d['x1']*img_w)), int(round(d['y1']*img_h))
        x2, y2 = int(round(d['x2']*img_w)), int(round(d['y2']*img_h))
        for temppath in temppaths:
            attrtype = tuple(sorted(d['attrs'].keys()))
            relpath = os.path.relpath(temppath)
            attrtype_str = '-'.join(attrtype)
            outdir_full = os.path.join(project.projdir_path, outdir, attrtype_str)
            outpath_full = os.path.join(outdir_full, util.encodepath(relpath)+'.png')
            if not os.path.exists(outpath_full):
                util_gui.create_dirs(outdir_full)
                im = shared.standardImread(temppath, flatten=True)
                patch = im[y1:y2, x1:x2]
                scipy.misc.imsave(outpath_full, patch)
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
        
    tmp2imgs = pickle.load(open(project.template_to_images, 'rb'))
    groups = []
    """
    Fixed-point iteration.
    """
    #all_templatepaths = common.get_imagepaths(project.templatesdir)
    no_change = False
    history = set()
    attrtype_ctr = {}
    w_img, h_img = project.imgsize
    while tmp2imgs and not no_change:
        flag = False
        for d in attrdata:
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
            if patch == None:
                # If we get here, then for the given attribute d,
                # we've grouped every blank ballot.
                continue
            patchpaths = extract_temp_patches(d, temppaths)
            matches = shared.find_patch_matches(patch, temppaths)
            print "len(matches):", len(matches)
            if matches:
                flag = True
                # First handle 'found' templates
                for (filename, _, rscFac, _, _, _, _) in matches:
                    history.add((attrtype, filename))
                inc_counter(attrtype_ctr, attrtype)
                grouplabel = common.make_grouplabel((attrtype, attrtype_ctr[attrtype]))
                elements = munge_matches(matches, grouplabel, patchpaths)
                in_group = common.GroupClass(elements)
                groups.append(in_group)
        if not flag:
            # Convergence achieved, stop iterating
            no_change = True
    return groups
    
def inc_counter(ctr, k):
    if k not in ctr:
        ctr[k] = 1
    else:
        ctr[k] = ctr[k] + 1
