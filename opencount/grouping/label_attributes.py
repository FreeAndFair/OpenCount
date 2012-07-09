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
import group_attrs

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
        groups = group_attrs.group_attributes(attrdata, self.project.imgsize,
                                              self.project.projdir_path,
                                              self.project.template_to_images)

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
        # If a group has group.is_manual set to True, then, every
        # element in the group should be manually labeled by the
        # user (this is activated when, say, the overlays are so
        # terrible that the user just wants to label them all
        # one by one)
        if num_elements == 0:
            reduction = 0
        else:
            reduction = len(sum(results.values(), [])) / float(num_elements)
        
        print "== Reduction in effort: ({0} / {1}) = {2}".format(len(sum(results.values(), [])),
                                                                 num_elements,
                                                                 reduction)

class LabelAttributesPanel(LabelContest):
    """
    This class is one big giant hack of a monkey-patch.
    """
    def gatherData(self):
        self.groupedtargets = []
        self.dirList = []
    
        # attrdata is a list of dicts (marshall'd AttributeBoxes)
        attrdata = pickle.load(open(self.proj.ballot_attributesfile))
        frontback = pickle.load(open(self.proj.frontback_map))

        #frame = TestFrame(self, self.proj)

        self.sides = [x['side'] for x in attrdata]
        self.types = [x['attrs'].keys()[0] for x in attrdata]
        self.is_digitbased = [x['is_digitbased'] for x in attrdata]
        self.is_tabulationonly = [x['is_tabulationonly'] for x in attrdata]

        width, height = self.proj.imgsize
        self.dirList = [os.path.join(self.proj.blankballots_straightdir,x) for x in os.listdir(self.proj.blankballots_straightdir)]
        for i,f in enumerate(self.dirList):
            thisballot = [[(at['id'], 0,
                          int(at['x1']*width), int(at['y1']*height), 
                          int(at['x2']*width), int(at['y2']*height))] for at in attrdata if at['side'] == frontback[os.path.abspath(f)]]
    
            self.groupedtargets.append(thisballot)
        self.groupedtargets_back = self.groupedtargets

        #self.template_width, self.template_height = Image.open(self.dirList[0]).size
        self.template_width, self.template_height = self.proj.imgsize

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
                          "height","attr_type","attr_val","side","is_digitbased","is_tabulationonly"])
            out.writerow([os.path.abspath(self.dirList[ballot]), DUMMY_ROW_ID,0,0,0,0,"_dummy_","_dummy_","_dummy_","_dummy_"])
            for uid,each in enumerate(vals):
                pos = self.groupedtargets_back[ballot][uid][0]
                print "POS IS", pos, "EACH", each
                value = "_none_" if each[1] == [] else each[1][0]
                out.writerow([os.path.abspath(self.dirList[ballot]),
                              uid, pos[2], pos[3],
                              pos[4]-pos[2], pos[5]-pos[3],
                              self.types[uid], value, self.sides[uid], self.is_digitbased[uid], self.is_tabulationbased[uid]])
    def validate_outputs(self):
        return True
    def stop(sefl):
        pass
    def export_bounding_boxes(sefl):
        pass
    def checkCanMoveOn(self):
        return True

