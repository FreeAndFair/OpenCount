import os, sys, pdb, wx, threading, Queue
from wx.lib.pubsub import Publisher
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

class GroupAttributesThread(threading.Thread):
    def __init__(self, attrdata, project, job_id, queue):
        threading.Thread.__init__(self)
        self.attrdata = attrdata
        self.project = project
        self.job_id = job_id
        self.queue = queue

    def run(self):
        groups = group_attrs.group_attributes(self.attrdata, self.project.imgsize,
                                              self.project.projdir_path,
                                              self.project.template_to_images,
                                              job_id=self.job_id)
        self.queue.put(groups)
        wx.CallAfter(Publisher().sendMessage, "signals.MyGauge.done", (self.job_id,))

class GroupAttrsFrame(wx.Frame):
    """ Frame that both groups attribute patches, and allows the
    user to verify the grouping.
    """

    GROUP_ATTRS_JOB_ID = util.GaugeID("Group_Attrs_Job_ID")

    def __init__(self, parent, project, ondone,*args, **kwargs):
        wx.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.project = project
        self.ondone = ondone
        self.sizer = wx.BoxSizer(wx.VERTICAL)

        self.panel = verify_overlays.VerifyPanel(self, verifymode=verify_overlays.VerifyPanel.MODE_YESNO2)
        self.sizer.Add(self.panel, proportion=1, flag=wx.EXPAND)
        
        attrdata = pickle.load(open(self.project.ballot_attributesfile, 'rb'))
        self.queue = Queue.Queue()
        t = GroupAttributesThread(attrdata, self.project, self.GROUP_ATTRS_JOB_ID, self.queue)
        gauge = util.MyGauge(self, 1, thread=t, ondone=self.on_groupattrs_done,
                             msg="Grouping Attribute Patches...",
                             job_id=self.GROUP_ATTRS_JOB_ID)
        t.start()
        gauge.Show()

    def on_groupattrs_done(self):
        groups = self.queue.get()
        self.panel.start(groups, None, ondone=self.verify_done)
        self.Fit()
        
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
        self.Close()
        self.ondone(results)

class LabelAttributesPanel(LabelContest):
    """
    This class is one big giant hack of a monkey-patch.
    """
    def set_attrgroup_results(self, groupresults):
        """ Given the result of grouping the attribute patches, update
        my self.equivs data structures. groupresults is a dict:
            {grouplabel: list of GroupClass objects}
        """
        def get_cid(blankpath, patchpath):
            """Determine the c_id of this attribute patch. """
            # Assumes that patchpath is of the form:
            #    <projdir_path>/extract_attrs_templates/<ATTR_STR>/*.png
            attrs_str = os.path.basename(os.path.split(patchpath)[0])
            return self.cid_map[(blankpath, attrs_str)]
        maps = {} # maps {grouplabel: list of (bid, cid)}
        # First, merge each group in groupresults into maps
        for grouplabel, groups in groupresults.iteritems():
            for group in groups:
                for (blankpath, rankedlist, patchpath) in group.elements:
                    bid = self.bid_map[blankpath]
                    cid = get_cid(blankpath, patchpath)
                    maps.setdefault(grouplabel, []).append((bid, cid))
        for grouplabel, tups in maps.iteritems():
            self.equivs.append(tups)
        self.multibox_contests = []
        self.has_equiv_classes = True

    def gatherData(self):
 
        self.groupedtargets = []
    
        # attrdata is a list of dicts (marshall'd AttributeBoxes)
        attrdata = pickle.load(open(self.proj.ballot_attributesfile))
        frontback = pickle.load(open(self.proj.frontback_map))

        self.sides = [x['side'] for x in attrdata]
        self.types = [x['attrs'].keys()[0] for x in attrdata]
        self.is_digitbased = [x['is_digitbased'] for x in attrdata]
        self.is_tabulationonly = [x['is_tabulationonly'] for x in attrdata]

        width, height = self.proj.imgsize
        self.dirList = []
        curbid = 0
        # bid -> ballot id
        bid_map = {}  # maps {str ballotpath: int b_id}
        for dirpath, dirnames, filenames in os.walk(self.proj.blankballots_straightdir):
            for imgname in [f for f in filenames if util_gui.is_image_ext(f)]:
                ballotpath = os.path.join(dirpath, imgname)
                self.dirList.append(ballotpath)
                assert ballotpath not in bid_map
                bid_map[ballotpath] = curbid
                curbid += 1
        self.bid_map = bid_map
        # cid -> contest id
        cid_map = {} # maps {(str ballotpath, str attrs): int c_id}
        curcid = 0
        for i,f in enumerate(self.dirList):
            thisballot = []
            for at in attrdata:
                if at['side'] == frontback[os.path.abspath(f)]:
                    assert f not in cid_map
                    attrs = tuple(sorted(at['attrs'].keys()))
                    attrs_str = '_'.join(attrs)
                    cid_map[(f, attrs_str)] = curcid
                    thisballot.append([(curcid, 0,
                                        int(round(at['x1']*width)),
                                        int(round(at['y1']*height)),
                                        int(round(at['x2']*width)),
                                        int(round(at['y2']*height)))])
                    curcid += 1
                    #cid_map[(f, attrs_str)] = at['id']
                    #thisballot.append([(at['id'], 0,
                    #                    int(round(at['x1']*width)),
                    #                    int(round(at['y1']*height)),
                    #                    int(round(at['x2']*width)),
                    #                    int(round(at['y2']*height)))])
            #thisballot = [[(at['id'], 0,
            #              int(at['x1']*width), int(at['y1']*height), 
            #              int(at['x2']*width), int(at['y2']*height))] for at in attrdata if at['side'] == frontback[os.path.abspath(f)]]
            self.groupedtargets.append(thisballot)
        self.groupedtargets_back = self.groupedtargets
        self.cid_map = cid_map

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
            out.writerow(["imgpath","id","x","y","width","height","attr_type",
                          "attr_val","side","is_digitbased","is_tabulationonly"])
            out.writerow([os.path.abspath(self.dirList[ballot]), DUMMY_ROW_ID,0,0,0,0,"_dummy_","_dummy_","_dummy_","_dummy_"])
            for uid,each in enumerate(vals):
                pos = self.groupedtargets_back[ballot][uid][0]
                print "POS IS", pos, "EACH", each
                value = "_none_" if each[1] == [] else each[1][0]
                out.writerow([os.path.abspath(self.dirList[ballot]),
                              uid, pos[2], pos[3],
                              pos[4]-pos[2], pos[5]-pos[3],
                              self.types[uid], value, self.sides[uid], self.is_digitbased[uid], self.is_tabulationonly[uid]])
    def validate_outputs(self):
        return True
    def stop(sefl):
        pass
    def export_bounding_boxes(sefl):
        pass
    def checkCanMoveOn(self):
        return True

