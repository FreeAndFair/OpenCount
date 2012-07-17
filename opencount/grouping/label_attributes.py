import os, sys, pdb, wx, threading, Queue
from os.path import join as pathjoin
from wx.lib.pubsub import Publisher
import wx.lib.scrolledpanel
from PIL import Image
import scipy
import scipy.misc
import pickle
import csv

sys.path.append('..')
from labelcontest.labelcontest import LabelContest
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

def no_digitattrs(attrdata):
    res = []
    for attrdict in attrdata:
        if not attrdict['is_digitbased']:
            res.append(attrdict)
    return res

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
        attrdata_nodigits = no_digitattrs(attrdata)
        self.queue = Queue.Queue()
        t = GroupAttributesThread(attrdata_nodigits, self.project, self.GROUP_ATTRS_JOB_ID, self.queue)
        gauge = util.MyGauge(self, 1, thread=t, ondone=self.on_groupattrs_done,
                             msg="Grouping Attribute Patches...",
                             job_id=self.GROUP_ATTRS_JOB_ID)
        t.start()
        gauge.Show()

    def on_groupattrs_done(self):
        groups = self.queue.get()
        self.Maximize()
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

class LabelPanel(wx.lib.scrolledpanel.ScrolledPanel):
    """
    A panel that allows you to, given a set of images I, give a text
    label to each image. Outputs to an output file.
    """
    def __init__(self, parent, *args, **kwargs):
        wx.lib.scrolledpanel.ScrolledPanel.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        
        self.imagelabels = {}   # maps {imagepath: str label}

        self.imagepaths = []  # ordered list of imagepaths
        self.cur_imgidx = 0  # which image we're currently at

        self.outpath = 'labelpanelout.csv'

        self._init_ui()
        self.Fit()

    def _init_ui(self):
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.sizer)

        self.sizer2 = wx.BoxSizer(wx.HORIZONTAL)
        self.imgpatch = wx.StaticBitmap(self)

        labeltxt = wx.StaticText(self, label='Label:')
        self.inputctrl = wx.TextCtrl(self, style=wx.TE_PROCESS_ENTER)
        self.inputctrl.Bind(wx.EVT_TEXT_ENTER, self.onInputEnter, self.inputctrl)
        nextbtn = wx.Button(self, label="Next")
        prevbtn = wx.Button(self, label="Previous")
        nextbtn.Bind(wx.EVT_BUTTON, self.onButton_next)
        prevbtn.Bind(wx.EVT_BUTTON, self.onButton_prev)
        inputsizer = wx.BoxSizer(wx.HORIZONTAL)
        inputsizer.Add(labeltxt)
        inputsizer.Add(self.inputctrl)

        sizer3 = wx.BoxSizer(wx.VERTICAL)
        sizer3.Add(inputsizer)
        sizer3.Add(nextbtn)
        sizer3.Add(prevbtn)

        self.sizer2.Add(self.imgpatch, proportion=0)
        self.sizer2.Add(sizer3, proportion=0)
        
        self.sizer.Add(self.sizer2, proportion=1, flag=wx.EXPAND)

    def onInputEnter(self, evt):
        """ Triggered when the user hits 'enter' when inputting text
        """
        curimgpath = self.imagepaths[self.cur_imgidx]
        cur_val = self.inputctrl.GetValue()
        self.imagelabels[curimgpath] = cur_val
        if (self.cur_imgidx+1) >= len(self.imagepaths):
            return
        self.display_img(self.cur_imgidx + 1)

    def onButton_next(self, evt):
        if self.cur_imgidx >= len(self.imagepaths):
            return
        else:
            self.display_img(self.cur_imgidx + 1)
            
    def onButton_prev(self, evt):
        if self.cur_imgidx <= 0:
            return
        else:
            self.display_img(self.cur_imgidx - 1)

    def start(self, imageslist, outfile='labelpanelout.csv'):
        """Given a dict of imagepaths to label, set up the UI, and
        allow the user to start labeling things.
        Input:
            lst imageslist: list of image paths
        """
        for imgpath in imageslist:
            assert imgpath not in self.imagelabels
            assert imgpath not in self.imagepaths
            self.imagelabels[imgpath] = ''
            self.imagepaths.append(imgpath)

        self.cur_imgidx = 0
        self.display_img(self.cur_imgidx)
        self.Fit()

    def display_img(self, idx):
        """Displays the image at idx, and allow the user to start labeling
        it.
        """
        if not (idx < len(self.imagepaths)):
            pdb.set_trace()
        assert idx < len(self.imagepaths)
        # First, store current input into our dict
        old_imgpath = self.imagepaths[self.cur_imgidx]
        cur_input = self.inputctrl.GetValue()
        self.imagelabels[old_imgpath] = cur_input

        self.cur_imgidx = idx
        imgpath = self.imagepaths[self.cur_imgidx]
        bitmap = wx.Bitmap(imgpath, type=wx.BITMAP_TYPE_PNG)
        self.imgpatch.SetBitmap(bitmap)
        self.inputctrl.SetValue(self.imagelabels[imgpath])
        
    def export_labels(self):
        """ Exports all labels to an output csvfile. """
        
        f = open(self.outpath, 'w')
        header = ('imgpath', 'label')
        dictwriter = csv.DictWriter(f, header)
        util_gui._dictwriter_writeheader(f, header)
        for imgpath, label in self.imagelabels.iteritems():
            row = {'imgpath': imgpath, 'label': label}
            dictwriter.write_row(row)
        f.close()



class LabelAttributesPanel(wx.lib.scrolledpanel.ScrolledPanel):
    """ A panel that will be integrated directly into OpenCount. """
    def __init__(self, parent, *args, **kwargs):
        wx.lib.scrolledpanel.ScrolledPanel.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.project = None

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.sizer)

        self.labelpanel = LabelPanel(self)
        self.sizer.Add(self.labelpanel, proportion=1, flag=wx.EXPAND)

    def start(self, project):
        def extract_attr_patches(project, outdir):
            """Extract all attribute patches from all blank ballots into
            the specified outdir.
            """
            templatesdir = project.templatesdir
            w_img, h_img = project.imgsize
            # list of marshall'd attrboxes (i.e. dicts)
            ballot_attributes = pickle.load(open(self.project.ballot_attributesfile, 'rb'))
            frontback_map = pickle.load(open(self.project.frontback_map, 'rb'))
            mapping = {} # maps {imgpath: {str attrtypestr: str patchPath}}
            patchpaths = set()
            for dirpath, dirnames, filenames in os.walk(templatesdir):
                for imgname in [f for f in filenames if util_gui.is_image_ext(f)]:
                    imgpath = pathjoin(dirpath, imgname)
                    imgpath_abs = os.path.abspath(imgpath)
                    for attrdict in ballot_attributes:
                        side = attrdict['side']
                        x1 = int(round(attrdict['x1']*w_img))
                        y1 = int(round(attrdict['y1']*h_img))
                        x2 = int(round(attrdict['x2']*w_img))
                        y2 = int(round(attrdict['y2']*h_img))
                        attrtype_str = common.get_attrtype_str(attrdict['attrs'])
                        if imgpath_abs not in frontback_map:
                            print "Uhoh, {0} not in frontback_map".format(imgpath_abs)
                            pdb.set_trace()
                        blankballot_side = frontback_map[imgpath_abs]
                        assert type(side) == str
                        assert type(blankballot_side) == str
                        assert side in ('front', 'back')
                        assert blankballot_side in ('front', 'back')
                        if frontback_map[imgpath_abs] == side:
                            img = shared.standardImread(imgpath, flatten=True)
                            patch = img[y1:y2,x1:x2]
                            # patchP: if outdir is: 'labelattrs_patchesdir',
                            # imgpath is: '/media/data1/election/blanks/foo/1.png',
                            # project.templatesdir is: '/media/data1/election/blanks/
                            tmp = self.project.templatesdir
                            if not tmp.endswith('/'):
                                tmp += '/'
                            partdir = os.path.split(imgpath[len(tmp):])[0] # foo/
                            patchrootDir = pathjoin(project.projdir_path,
                                                    outdir,
                                                    partdir,
                                                    os.path.splitext(imgname)[0])
                            # patchrootDir: labelattrs_patchesdir/foo/1/
                            util_gui.create_dirs(patchrootDir)
                            patchoutP = pathjoin(patchrootDir, "{0}_{1}.png".format(os.path.splitext(imgname)[0],
                                                                                    attrtype_str))
                            scipy.misc.imsave(patchoutP, patch)
                            mapping.setdefault(imgpath, {})[attrtype_str] = patchoutP
                            assert patchoutP not in patchpaths
                            patchpaths.add(patchoutP)
            return mapping, tuple(patchpaths)
        self.project = project
        mapping, patchpaths = extract_attr_patches(self.project, self.project.labelattrs_patchesdir)
        outfilepath = pathjoin(self.project.projdir_path,
                               self.project.labelattrs_out)
        self.labelpanel.start(patchpaths, outfile=outfilepath)
        self.Fit()
        self.SetupScrolling()

    def stop(self):
        pass

    def validate_outputs(self):
        """ Check to see if all outputs are complete -- issue warnings
        to the user if otherwise, and return False.
        """
        return True

    def export_results(self):
        """ Instead of using LabelPanel's export_labels, which saves
        all patchpath->label mappings to one .csv file, we want to save
        the blankballotpath->(attr labels) to multiple .csv files.
        """
        patchlabels = self.labelpanel.imagelabels
        
    
    def checkCanMoveOn(self):
        """ Return True if the user can move on, False otherwise. """
        return True
        
'''
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
            if common.is_digit_grouplabel(grouplabel, self.proj):
                continue
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
        attrdata = no_digitattrs(attrdata)
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
        for i,f in enumerate(self.dirList):
            thisballot = []
            for curcid,at in enumerate(attrdata):
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
            self.groupedtargets.append(thisballot)
        print "CID_MAP", cid_map
        print "GRTARGS", self.groupedtargets
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
                print "SETTING TEXT", i, x[0]
                self.text[i,x[0]] = []
                self.voteupto[i,x[0]] = 1
        print self.text
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
'''
