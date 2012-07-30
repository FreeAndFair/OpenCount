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

class LabelAttributesPanel(wx.lib.scrolledpanel.ScrolledPanel):
    """ A panel that will be integrated directly into OpenCount. """
    def __init__(self, parent, *args, **kwargs):
        wx.lib.scrolledpanel.ScrolledPanel.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.project = None

        self.mapping = None # maps {imgpath: {str attrtypestr: str patchPath}}
        self.inv_mapping = None # maps {str patchPath: (imgpath, attrtypestr)}

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
            inv_mapping = {} # maps {str patchPath: (imgpath, attrtypestr)}
            patchpaths = set()
            for dirpath, dirnames, filenames in os.walk(templatesdir):
                for imgname in [f for f in filenames if util_gui.is_image_ext(f)]:
                    imgpath = pathjoin(dirpath, imgname)
                    imgpath_abs = os.path.abspath(imgpath)
                    for attrdict in ballot_attributes:
                        if attrdict['is_digitbased']:
                            continue
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
                            if not os.path.exists(patchoutP):
                            #if True:
                                # TODO: Only extract+save the imge patches
                                # when you /have/ to.
                                img = shared.standardImread(imgpath, flatten=True)
                                patch = img[y1:y2,x1:x2]
                                scipy.misc.imsave(patchoutP, patch)
                            mapping.setdefault(imgpath, {})[attrtype_str] = patchoutP
                            inv_mapping[patchoutP] = (imgpath, attrtype_str)
                            assert patchoutP not in patchpaths
                            patchpaths.add(patchoutP)
            return mapping, inv_mapping, tuple(patchpaths)
        self.project = project
        mapping, inv_mapping, patchpaths = extract_attr_patches(self.project, self.project.labelattrs_patchesdir)
        self.mapping = mapping
        self.inv_mapping = inv_mapping
        # outfilepath isn't used at the moment.
        outfilepath = pathjoin(self.project.projdir_path,
                               self.project.labelattrs_out)
        statefilepath = pathjoin(self.project.projdir_path,
                                 LabelPanel.STATE_FILE)
        if not self.labelpanel.restore_session(statefile=statefilepath):
            self.labelpanel.start(patchpaths, outfile=outfilepath)
        self.Fit()
        self.SetupScrolling()
        self.project.addCloseEvent(self.stop)

    def set_attrgroup_results(self, groups):
        """ Takes the results of autogrouping attribute patches, and
        updates me. """
        pass

    def stop(self):
        """ Saves some state. """
        self.labelpanel.save_session(statefile=pathjoin(self.project.projdir_path,
                                                        LabelPanel.STATE_FILE))

    def cluster_attr_patches(self, outdir):
        """ After the user has manually labeled every attribute patch
        from all blank ballots, we will try to discover clusters
        within a particular attribute value. For instance, if the
        attribute type is 'language', and the attribute values are
        'eng' and 'span', and some language patches have a white or
        dark gray background, then this algorithm should discover two
        clusters within 'eng' (white backs, gray backs) and within 'span'
        (white backs, gray backs).
        """
        blankpatches = {} # maps {attrtype: {attrval: list of blank paths}}
        patchlabels = self.labelpanel.imagelabels
        for patchPath, label in patchlabels.iteritems():
            imgpath, attrtypestr = self.inv_mapping[patchPath]
            blankpatches.setdefault(attrtypestr, {}).setdefault(label, []).append(imgpath)
        # maps {attrtype: {attrval: ((imgpath_i,y1,y2,x1,x2,rszFac), ...)}}
        exemplars = group_attrs.cluster_attributesV2(blankpatches, self.project)
        for attrtype, thedict in exemplars.iteritems():
            for attrval, exemplars in thedict.iteritems():
                rootdir = os.path.join(outdir, attrtype)
                util_gui.create_dirs(rootdir)
                for i, (imgpath,y1,y2,x1,x2,rszFac) in enumerate(exemplars):
                    img = scipy.misc.imread(imgpath, flatten=True)
                    y1,y2,x1,x2 = map(lambda c: c / rszFac, (y1,y2,x1,x2))
                    patch = img[y1:y2,x1:x2]
                    outfilename = "{0}_{1}.png".format(attrval, i)
                    scipy.misc.imsave(os.path.join(rootdir, outfilename),
                                      patch)
        print "Done saving exemplar patches."
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
        print "Exporting results."
        patchlabels = self.labelpanel.imagelabels
        ballot_attr_labels = {} # maps {imgpath: {attrstr: label}}
        for patchPath, label in patchlabels.iteritems():
            imgpath, attrtypestr = self.inv_mapping[patchPath]
            ballot_attr_labels.setdefault(imgpath, {})[attrtypestr] = label
        util_gui.create_dirs(self.project.patch_loc_dir)
        header = ("imgpath", "id", "x", "y", "width", "height", "attr_type",
                  "attr_val", "side", "is_digitbased", "is_tabulationonly")
        ballot_attrs = pickle.load(open(self.project.ballot_attributesfile, 'rb'))
        w_img, h_img = self.project.imgsize
        uid = 0
        for imgpath, attrlabels in ballot_attr_labels.iteritems():
            imgname = os.path.splitext(os.path.split(imgpath)[1])[0]
            csvoutpath = pathjoin(self.project.patch_loc_dir,
                                  "{0}_patchlocs.csv".format(imgname))
            f = open(csvoutpath, 'w')
            writer = csv.DictWriter(f, header)
            util_gui._dictwriter_writeheader(f, header)
            for attrtype, label in attrlabels.iteritems():
                row = {}
                row['imgpath'] = imgpath; row['id'] = uid
                x1 = int(round(w_img*common.get_attr_prop(self.project,
                                                          attrtype, 'x1')))
                y1 = int(round(h_img*common.get_attr_prop(self.project,
                                                          attrtype, 'y1')))
                x2 = int(round(w_img*common.get_attr_prop(self.project,
                                                          attrtype, 'x2')))
                y2 = int(round(h_img*common.get_attr_prop(self.project,
                                                          attrtype, 'y2')))
                row['x'] = x1; row['y'] = y1
                row['width'] = int(abs(x1-x2))
                row['height'] = int(abs(y1-y2))
                row['attr_type'] = attrtype
                row['attr_val'] = label
                row['side'] = common.get_attr_prop(self.project,
                                                   attrtype, 'side')
                row['is_digitbased'] = common.get_attr_prop(self.project,
                                                            attrtype, 'is_digitbased')
                row['is_tabulationonly'] = common.get_attr_prop(self.project,
                                                                attrtype, 'is_tabulationonly')
                writer.writerow(row)
                uid += 1
            f.close()
        print "Done writing out LabelBallotAttributes stuff."
        
    def checkCanMoveOn(self):
        """ Return True if the user can move on, False otherwise. """
        return True

class LabelPanel(wx.lib.scrolledpanel.ScrolledPanel):
    """
    A panel that allows you to, given a set of images I, give a text
    label to each image. Outputs to an output file.
    """
    STATE_FILE = '_labelpanelstate.p'

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
        self.progress_txt = wx.StaticText(self, label='')
        sizer3 = wx.BoxSizer(wx.VERTICAL)
        sizer3.Add(inputsizer)
        sizer3.Add(nextbtn)
        sizer3.Add(prevbtn)
        sizer3.Add(self.progress_txt)

        self.sizer2.Add(self.imgpatch, proportion=0)
        self.sizer2.Add((40, 40))
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
        if (self.cur_imgidx+1) >= len(self.imagepaths):
            curimgpath = self.imagepaths[self.cur_imgidx]
            cur_val = self.inputctrl.GetValue()
            self.imagelabels[curimgpath] = cur_val
            return
        else:
            self.display_img(self.cur_imgidx + 1)
            
    def onButton_prev(self, evt):
        if self.cur_imgidx <= 0:
            curimgpath = self.imagepaths[self.cur_imgidx]
            cur_val = self.inputctrl.GetValue()
            self.imagelabels[curimgpath] = cur_val
            return
        else:
            self.display_img(self.cur_imgidx - 1)

    def start(self, imageslist, outfile='labelpanelout.csv'):
        """Given a dict of imagepaths to label, set up the UI, and
        allow the user to start labeling things.
        Input:
            lst imageslist: list of image paths
            outfile: Output file to write results to.
        """
        for imgpath in imageslist:
            assert imgpath not in self.imagelabels
            assert imgpath not in self.imagepaths
            self.imagelabels[imgpath] = ''
            self.imagepaths.append(imgpath)

        self.cur_imgidx = 0
        self.display_img(self.cur_imgidx)
        self.SetClientSize(self.parent.GetClientSize())
        self.SetupScrolling()

    def restore_session(self, statefile=None):
        """ Tries to restore the state of a previous session. If this
        fails (say, the internal state file was deleted), then this
        will return False. If this happens, then you should just call
        self.start().
        """
        if statefile == None:
            statefile = LabelPanel.STATE_FILE
        if not os.path.exists(statefile):
            return False
        state = pickle.load(open(statefile, 'rb'))
        imagelabels = state['imagelabels']
        imagepaths = state['imagepaths']
        self.imagelabels = imagelabels
        self.imagepaths = imagepaths
        self.cur_imgidx = 0
        self.display_img(self.cur_imgidx, no_overwrite=True)
        self.Fit()
        return True

    def save_session(self, statefile=None):
        """ Saves the current state of the current session. """
        if statefile == None:
            statefile = LabelPanel.STATE_FILE
        # Remember to store the currently-displayed label
        curimgpath = self.imagepaths[self.cur_imgidx]
        cur_label = self.inputctrl.GetValue()
        self.imagelabels[curimgpath] = cur_label
        state = {}
        state['imagelabels'] = self.imagelabels
        state['imagepaths'] = self.imagepaths
        f = open(statefile, 'wb')
        pickle.dump(state, f)
        f.close()

    def display_img(self, idx, no_overwrite=False):
        """Displays the image at idx, and allow the user to start labeling
        it. Also updates the progress_txt.
        """
        if not (idx < len(self.imagepaths)):
            pdb.set_trace()
        assert idx < len(self.imagepaths)
        if not no_overwrite:
            # First, store current input into our dict
            old_imgpath = self.imagepaths[self.cur_imgidx]
            cur_input = self.inputctrl.GetValue()
            self.imagelabels[old_imgpath] = cur_input

        self.cur_imgidx = idx
        imgpath = self.imagepaths[self.cur_imgidx]
        bitmap = wx.Bitmap(imgpath, type=wx.BITMAP_TYPE_PNG)
        self.imgpatch.SetBitmap(bitmap)
        self.progress_txt.SetLabel("Currently viewing: Patch {0}/{1}".format(self.cur_imgidx+1,
                                                                             len(self.imagepaths)))
        self.inputctrl.SetValue(self.imagelabels[imgpath])
        #self.Fit()
        self.SetupScrolling()
        
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
