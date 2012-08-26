import os, sys, pdb, wx, threading, Queue, time
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
import partask

DUMMY_ROW_ID = -42

class GroupAttributesThread(threading.Thread):
    def __init__(self, attrdata, project, job_id, queue):
        threading.Thread.__init__(self)
        self.attrdata = attrdata
        self.project = project
        self.job_id = job_id
        self.queue = queue

    def run(self):
        # attrgroups is dict {str attrtype: {c_imgpath: [(imgpath_i, bb_i), ...]}}
        print "Grouping Ballot Attributes..."
        _t0 = time.time()
        attrgroups = group_attrs.group_attributes_V2(self.project)
        print "...Finished Grouping Ballot Attributes ({0} s).".format(time.time() - _t0)
        #groups = group_attrs.group_attributes(self.attrdata, self.project.imgsize,
        #                                      self.project.projdir_path,
        #                                      self.project.template_to_images,
        #                                      self.project,
        #                                      job_id=self.job_id)
        print "...converting attrgroups to groupclasses."
        _t = time.time()
        groups = clusters_to_groupclasses(self.project, attrgroups)
        print "...finished converting attrgroups to groupclasses ({0} s).".format(time.time() - _t)
        # Convert 'groups' to a list of GroupClass instances
        self.queue.put(groups)
        wx.CallAfter(Publisher().sendMessage, "signals.MyGauge.done", (self.job_id,))

def clusters_to_groupclasses(proj, attrgroups):
    """ Converts the result of attribute clustering into a list of
    GroupClass instances. As a side effect, this also extracts all
    attribute patches, and saves them in proj.extract_attrs_templates.
    They are saved in the following directory structure:
        projdir/extract_attrs_templates/ATTRTYPE/TEMPDIRSTRUCTURE/*.png
    Input:
        obj proj:
        dict attrgroups: {attrtype: {clusterID: [(imgpath_i, bb_i, score_i), ...]}}
    Output:
        A list of GroupClass instances.
    """
    groups = []
    extract_tasks = []
    for attrtype, cluster in attrgroups.iteritems():
        for i, (clusterID, c_elements) in enumerate(cluster.iteritems()):
            elements = [] # [(imgpath_i, rlist_i, patchpath_i), ...]
            for (imgpath, bb, score) in c_elements:
                imgname = os.path.split(imgpath)[1]
                imgname_noext = os.path.splitext(imgname)[0]
                relpath = os.path.relpath(imgpath)
                basedir = os.path.join(proj.projdir_path,
                                       proj.extract_attrs_templates,
                                       attrtype)
                # Recreate directory structure
                tmp = proj.templatesdir
                if not tmp.endswith('/'):
                    tmp += '/'
                partdir = os.path.split(imgpath[len(tmp):])[0]
                patchrootDir = pathjoin(basedir,
                                        partdir,
                                        os.path.splitext(os.path.split(imgpath)[1])[0])
                util_gui.create_dirs(patchrootDir)
                patchpath = pathjoin(patchrootDir,
                                     "{0}_{1}.png".format(imgname_noext, attrtype))
                #patchpath = os.path.join(patchrootDir,
                #                         util.encodepath(relpath)) + '.png'
                extract_tasks.append((imgpath, patchpath, bb))
                grouplabel = common.make_grouplabel((attrtype, i))
                elements.append((imgpath, (grouplabel,), patchpath))
            groups.append(common.GroupClass(elements))
    print "...saving attribute patches to {0}".format(proj.extract_attrs_templates)
    _t = time.time()
    partask.do_partask(extract_attrpatches, extract_tasks)
    print "...finished saving attributes patches to {0} ({1} s).".format(proj.extract_attrs_templates,
                                                                         time.time() - _t)
    return groups
    
def extract_attrpatches(tasks):
    for (imgpath, outpath, bb) in tasks:
        bb = map(lambda c: int(round(c)), bb)
        img = Image.open(imgpath)
        img = img.convert('L')
        img = img.crop((bb[2], bb[0], bb[3], bb[1]))
        img.save(outpath)
    return []

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

    def __init__(self, parent, project, ondone, *args, **kwargs):
        """
        obj parent:
        obj project:
        fn ondone: A callback function that is to be called after the
                   grouping+verify step of Attrs has been completed.
                   ondone should accept one argument, 'results', which
                   is a dict mapping: {grouplabel: list GroupClasses}
        """
        wx.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.project = project
        self.ondone = ondone
        self.sizer = wx.BoxSizer(wx.VERTICAL)

        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        btn_rungroup = wx.Button(self, label="Run Attribute Grouping...")
        btn_rungroup.Bind(wx.EVT_BUTTON, self.onButton_rungroup)
        btn_skip = wx.Button(self, label="Skip Attribute Grouping.")
        btn_skip.Bind(wx.EVT_BUTTON, self.onButton_skipgroup)
        btn_sizer.AddMany([(btn_rungroup,), (btn_skip,)])
        self.btn_rungroup = btn_rungroup
        self.btn_skip = btn_skip

        self.sizer.Add(btn_sizer, border=5, flag=wx.ALL | wx.ALIGN_CENTER)

        self.panel = verify_overlays.VerifyPanel(self, verifymode=verify_overlays.VerifyPanel.MODE_YESNO2)
        self.sizer.Add(self.panel, proportion=1, flag=wx.EXPAND)
        self.panel.Hide()
        self.SetSizer(self.sizer)

    def start_grouping(self):
        print "== Starting Attribute Grouping..."
        self.panel.Show()
        attrdata = pickle.load(open(self.project.ballot_attributesfile, 'rb'))
        attrdata_nodigits = no_digitattrs(attrdata)
        self.queue = Queue.Queue()
        t = GroupAttributesThread(attrdata_nodigits, self.project, self.GROUP_ATTRS_JOB_ID, self.queue)
        gauge = util.MyGauge(self, 1, thread=t, ondone=self.on_groupattrs_done,
                             msg="Grouping Attribute Patches...",
                             job_id=self.GROUP_ATTRS_JOB_ID)
        t.start()
        gauge.Show()
        
    def skip_grouping(self):
        print "== Skipping Attribute Grouping."
        self.ondone(None)
        self.Close()

    def onButton_rungroup(self, evt):
        self.btn_rungroup.Hide()
        self.btn_skip.Hide()
        self.start_grouping()
    def onButton_skipgroup(self, evt):
        self.skip_grouping()

    def on_groupattrs_done(self):
        groups = self.queue.get()
        self.Maximize()
        self.panel.start(groups, None, self.project, ondone=self.verify_done)
        self.project.addCloseEvent(self.panel.dump_state)
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
        self.project.removeCloseEvent(self.panel.dump_state)
        self.Close()
        self.ondone(results)

class LabelAttributesPanel(wx.lib.scrolledpanel.ScrolledPanel):
    """ A panel that will be integrated directly into OpenCount. """
    def __init__(self, parent, *args, **kwargs):
        wx.lib.scrolledpanel.ScrolledPanel.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.project = None

        # mapping, inv_mapping contain information about every image
        # patch that the user will manually label.
        self.mapping = None # maps {imgpath: {str attrtypestr: str patchPath}}
        self.inv_mapping = None # maps {str patchPath: (imgpath, attrtypestr)}

        # This keeps track of all patches that the attribute-grouping
        # code found to be 'equal'
        self.patch_groups = {} # maps {patchpath: list of imgpaths}

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.sizer)

        self.labelpanel = LabelPanel(self)
        self.sizer.Add(self.labelpanel, proportion=1, flag=wx.EXPAND)

    def start(self, project, groupresults=None):
        """ Start the Labeling widget. If groups is given, then this
        means that only a subset of the patches need to be labeled.
        Input:
            dict groupresults: maps {grouplabel: List of GroupClass objects}
        """
        self.project = project
        if groupresults == None:
            # We are manually labeling everything
            self.mapping, self.inv_mapping = do_extract_attr_patches(self.project)
        else:
            self.mapping, self.inv_mapping = self.handle_grouping_results(groupresults)
        # outfilepath isn't used at the moment.
        outfilepath = pathjoin(self.project.projdir_path,
                               self.project.labelattrs_out)
        statefilepath = pathjoin(self.project.projdir_path,
                                 LabelPanel.STATE_FILE)
        # 'sort' patchpaths by attribute type
        patchpaths = get_ordered_patchpaths(self.inv_mapping)
        if not self.labelpanel.restore_session(statefile=statefilepath):
            imagecaptions = {} # maps {str patchpath: str attrtype}
            for patchpath, (imgpath, attrtype) in self.inv_mapping.iteritems():
                imagecaptions[patchpath] = attrtype
            self.labelpanel.start(patchpaths, captions=imagecaptions,
                                  outfile=outfilepath)
        self.check_state()
        self.Fit()
        self.SetupScrolling()
        self.project.addCloseEvent(self.stop)

    def check_state(self):
        """ Makes sure that the LabelPanel's internal data structures
        are in-sync with my own data structures. They might become out
        of sync if, say, in session A, I save state. Then in session B,
        I re-run grouping, make a change, and then restore the (now
        invalid) state.
        """
        # The below is the 'idea', yet this throws an image-not-found
        # error.
        pass
        '''
        for patchpath in self.labelpanel.imagepaths[:]:
            if patchpath not in self.inv_mapping:
                print "Removing {0} from LabelPanel.".format(patchpath)
                self.labelpanel.imagepaths.remove(patchpath)
                self.labelpanel.imagelabels.pop(patchpath)
        '''

    def handle_grouping_results(self, groupresults):
        """ Takes the results of autogrouping attribute patches, and
        updates my internal data structures. Importantly, this updates
        the self.patch_groups data structure, which allows me to know
        that labeling a patch P implies the labeling of all votedpaths
        given by self.patch_groups[patchpathP].
        Input:
            dict groupresults: maps {grouplabel: list of GroupClass instances}
        Output:
            dict mapping, dict inv_mapping, but only for
            one exemplar from each group, where mapping is:
              {str votedpath: {attrtype: patchpath}}
            inv_mapping is:
              {str patchpath: (imgpath, attrtype)}
        """
        mapping = {}  # maps {imgpath: {attrtypestr: patchpath}}
        inv_mapping = {}  # maps {patchpath: (imgpath, attrtypestr)}
        for grouplabel, groups in groupresults.iteritems():
            attrtypestr = tuple(grouplabel)[0][0] # why do i do this?!
            flag = True
            for group in groups:
                if group.is_manual:
                    for votedpath, rankedlist, patchpath in group.elements:
                        mapping.setdefault(votedpath, {})[attrtypestr] = patchpath
                        inv_mapping[patchpath] = (votedpath, attrtypestr)
                elif flag:
                    # grab one exemplar
                    votedpath_exemplar, _, patchpath_exemplar = groups[0].elements[0]
                    mapping.setdefault(votedpath_exemplar, {})[attrtypestr] = patchpath_exemplar
                    inv_mapping[patchpath_exemplar] = (votedpath_exemplar, attrtypestr)
                    flag = False
                    # update my self.patch_groups map
                    for votedpath, rankedlist, patchpath in group.elements[1:]:
                        self.patch_groups.setdefault(patchpath_exemplar, []).append(votedpath)
                else:
                    for votedpath, rankedlist, patchpath in group.elements:
                        self.patch_groups.setdefault(patchpath_exemplar, []).append(votedpath)

        return mapping, inv_mapping

    def stop(self):
        """ Saves some state. """
        if self.project == None:
            # We never called 'self.start()', so don't proceed. This
            # happens if, say, there are no Img-Based attrs in the
            # election.
            return
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
        blankpatches = {} # maps {attrtype: {attrval: (patchpath_i, ...)}}
        patchlabels = self.labelpanel.imagelabels
        attrs = pickle.load(open(self.project.ballot_attributesfile, 'rb'))
        w_img, h_img = self.project.imgsize

        for patchPath, label in patchlabels.iteritems():
            imgpath, attrtypestr = self.inv_mapping[patchPath]
            blankpatches.setdefault(attrtypestr, {}).setdefault(label, []).append(patchPath)
            
        # TODO: Call group_attrs.cluster_bkgd on /all/ blank ballots,
        #       not just attribute exemplars.

        # TODO: This code below is garbage! I've already extracted all
        # blank ballot attr patches to:
        #    projdir/extract_attrs_templates/<ATTRTYPE>
        # I should just those. Find where it's saving those, and also
        # keep track of templatepath -> FILENAME.png.
        # Then, pass those in as input to group_attrs.cluster_bkgd.
        # But I'll also have to use the internal mappings (since attribute
        # grouping has been run, probably.), to get attribute values.
        
        # 0. Save all template patches to a directory
        tmp2imgs = pickle.load(open(self.project.template_to_images, 'rb'))
        tasks = []
        for attr in attrs:
            attrtype = common.get_attrtype_str(attr)
            x1 = int(round(attr['x1'] * w_img))
            y1 = int(round(attr['y1'] * h_img))
            x2 = int(round(attr['x2'] * w_img))
            y2 = int(round(attr['y2'] * h_img))
            side = attr['side']
            # TODO: Generalize to N-sides
            assert side in ('front', 'back')
            for i, (templateid, paths) in enumerate(tmp2imgs.iteritems()):
                if side == 'front':
                    temppath = paths[0]
                else:
                    temppath = paths[1]
                outpath = os.path.join(proj.projdir_path, 
                                       '_all_blank_attrpatches',
                                       attrtype,
                                       '{0}_{1}.png'.format(i, os.path.splitext(os.path.split(templateid)[1])[0]))
                tasks.append((temppath, (y1,y2,x1,x2), outpath))
        print "...Extracting all attribute patches to '_all_blank_attrpatches'..."
        _t = time.time()
        common.par_extract_patches(tasks)
        print "...Finished extracting all attribute patches to '_all_blank_attrpatches' ({0} s).".format(time.time() - _t)
            
        attrtype_exemplars = {}  # maps {attrtype: {attrval: (patchpath_i, ...)}}
        for attrtype, attrval_map in blankpatches.iteritems():
            exemplars = group_attrs.cluster_bkgd(attrval_map)
            print "==== For Attribute {0}, {1} exemplars were found.".format(attrtype,
                                                                             len(exemplars))
            attrtype_exemplars[attrtype] = exemplars

        # Save the patches to outdir
        outfile_map = {} # maps {attrtype: {attrval: ((str patchpath_i, str blankpath_i, bb_i), ...)}}

        for attrtype, thedict in attrtype_exemplars.iteritems():
            attr = None
            for attrdict in attrs:
                if common.get_attrtype_str(attrdict['attrs']) == attrtype:
                    attr = attrdict
                    break
            if attr == None:
                print "Uhoh, couldn't find attrtype {0}.".format(attrtype)
                pdb.set_trace()

            x1 = int(round(attr['x1'] * w_img))
            y1 = int(round(attr['y1'] * h_img))
            x2 = int(round(attr['x2'] * w_img))
            y2 = int(round(attr['y2'] * h_img))
            bb = (y1, y2, x1, x2)
            for attrval, exemplars in thedict.iteritems():
                rootdir = os.path.join(outdir, attrtype)
                util_gui.create_dirs(rootdir)
                for i, patchpath in enumerate(exemplars):
                    img = scipy.misc.imread(patchpath, flatten=True)
                    outfilename = "{0}_{1}.png".format(attrval, i)
                    fulloutpath = os.path.join(rootdir, outfilename)
                    scipy.misc.imsave(fulloutpath, img)
                    blankpath, _ = self.inv_mapping[patchpath]
                    outfile_map.setdefault(attrtype, {}).setdefault(attrval, []).append((fulloutpath, blankpath, bb))
        # Also save out the outfile_map
        pickle.dump(outfile_map, open(pathjoin(self.project.projdir_path,
                                               self.project.multexemplars_map),
                                      'wb'))
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
        if self.project == None:
            # self.start was never called, so don't proceed. This could
            # happen if, say, this election has no Img-based attrs.
            return
        print "Exporting results."
        patchlabels = self.labelpanel.imagelabels
        ballot_attr_labels = {} # maps {imgpath: {attrstr: label}}
        for patchPath, label in patchlabels.iteritems():
            imgpath, attrtypestr = self.inv_mapping[patchPath]
            ballot_attr_labels.setdefault(imgpath, {})[attrtypestr] = label
            # Finally, also add this labeling for all blank ballots
            # that were grouped together by attr-grouping
            for siblingpath in self.patch_groups.get(patchPath, []):
                ballot_attr_labels.setdefault(siblingpath, {})[attrtypestr] = label
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
        
        # self.imagelabels keeps track of the labels that the user has
        # given to each image.
        self.imagelabels = {}   # maps {imagepath: str label}

        # self.imagecaptions keeps track of the caption that the UI 
        # should display to the user.
        self.imagecaptions = {} # maps {imagepath: str caption}

        # self.captionlabels keeps track of all labels given to a
        # specific caption.
        self.captionlabels = {} # maps {str caption: (str label_i, ...)}

        self.imagepaths = []  # ordered list of imagepaths
        self.cur_imgidx = 0  # which image we're currently at

        self.outpath = 'labelpanelout.csv'

        self._init_ui()

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

        sizer_img = wx.BoxSizer(wx.VERTICAL)
        sizer_img.Add(self.imgpatch, proportion=0)
        self.caption_txt = wx.StaticText(self, label="Foo.")
        sizer_img.Add(self.caption_txt)

        sizer_lstbox = wx.BoxSizer(wx.VERTICAL)
        txt_listbox = wx.StaticText(self, label="Previously entered \
values for this caption...")
        self.listbox = wx.ListBox(self, choices=[])
        sizer_lstbox.Add(txt_listbox)
        sizer_lstbox.Add(self.listbox)

        self.sizer2.Add(sizer_img)
        self.sizer2.Add((40, 40))
        self.sizer2.Add(sizer3, proportion=0)
        self.sizer2.Add((40, 40))
        self.sizer2.Add(sizer_lstbox)
        
        self.sizer.Add(self.sizer2, proportion=1, flag=wx.EXPAND)

    def update_caption_txt(self, imgidx):
        """ Updates the self.caption_txt StaticText widget. """
        if imgidx >= len(self.imagepaths):
            print "Uh oh, imgidx out of range."
            print "== Press 'c' to continue."
            pdb.set_trace()
            return
        imgpath = self.imagepaths[imgidx]
        if imgpath in self.imagecaptions:
            caption = self.imagecaptions[imgpath]
            self.caption_txt.SetLabel("Caption: {0}.".format(caption))

    def update_listbox(self, imgidx):
        """ Updates the self.listbox ListBox widget, and displays all
        previously-entered labels that have the same caption.
        """
        if imgidx >= len(self.imagepaths):
            print "Uh oh, imgidx out of range."
            print "== Press 'c' to continue."
            pdb.set_trace()
            return
        imgpath = self.imagepaths[imgidx]
        if imgpath in self.imagecaptions:
            caption = self.imagecaptions[imgpath]
            labels = self.captionlabels.get(caption, None)
            if labels == None:
                self.listbox.SetItems([])
                return
            self.listbox.SetItems(list(labels))
        else:
            self.listbox.SetItems([])

    def add_label(self, imgpath, label):
        """ Adds the 'label' for the given image by updating internal
        data structures. 
        """
        self.imagelabels[imgpath] = label
        caption = self.imagecaptions.get(imgpath, None)
        if caption != None:
            self.captionlabels.setdefault(caption, set()).add(label)

    def onInputEnter(self, evt):
        """ Triggered when the user hits 'enter' when inputting text.
        """
        curimgpath = self.imagepaths[self.cur_imgidx]
        cur_val = self.inputctrl.GetValue()
        self.add_label(curimgpath, cur_val)
        if (self.cur_imgidx+1) >= len(self.imagepaths):
            return
        self.display_img(self.cur_imgidx + 1)

    def onButton_next(self, evt):
        if (self.cur_imgidx+1) >= len(self.imagepaths):
            curimgpath = self.imagepaths[self.cur_imgidx]
            cur_val = self.inputctrl.GetValue()
            self.add_label(curimgpath, cur_val)
            return
        else:
            self.display_img(self.cur_imgidx + 1)
            
    def onButton_prev(self, evt):
        if self.cur_imgidx <= 0:
            curimgpath = self.imagepaths[self.cur_imgidx]
            cur_val = self.inputctrl.GetValue()
            self.add_label(curimgpath, cur_val)
            return
        else:
            self.display_img(self.cur_imgidx - 1)

    def reset(self):
        """ Resets my state to a 'clean' slate. """
        # TODO: Reset UI also.
        self.imagelabels = {}
        self.imagepaths = []
        self.imagecaptions = {}
        self.captionlabels = {}
        self.cur_imgidx = 0

    def start(self, imageslist, captions=None, outfile='labelpanelout.csv'):
        """Given a dict of imagepaths to label, set up the UI, and
        allow the user to start labeling things. This will reset all
        state to be a 'clean slate'.
        Input:
            lst imageslist: list of image paths
            dict captions: A dict mapping {str imgpath: str caption}.
                           If you want to display some caption for an
                           image to the user, provide it here.
            outfile: Output file to write results to.
        """
        captions = captions if captions != None else {}
        self.reset()
        for imgpath in imageslist:
            if imgpath in self.imagelabels:
                print "Uhoh, imgpath was already in self.imagelabels. \
Implies that imgpath is present in imageslist more than once."
                print "    imgpath is:", imgpath
                print "    self.imagelabels[imgpath] is:", self.imagelabels[imgpath]
                pdb.set_trace()
            if imgpath in self.imagepaths:
                print "Uhoh, imgpath was already self.imagepaths. \
Implies that imgpath is present in imageslist more than once."
                print "    imgpath is:", imgpath
                pdb.set_trace()
            assert imgpath not in self.imagelabels
            assert imgpath not in self.imagepaths
            self.imagelabels[imgpath] = ''
            self.imagepaths.append(imgpath)
            caption = captions.get(imgpath, "An Image.")
            self.imagecaptions[imgpath] = caption

        self.cur_imgidx = 0
        self.display_img(self.cur_imgidx)

        #self.SetClientSize(self.parent.GetClientSize())
        self.SetupScrolling()
        self.SendSizeEvent()

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

        ## TODO: Legacy-handling code follows.
        if 'imagecaptions' not in state:
            state['imagecaptions'] = {}
        if 'captionlabels' not in state:
            state['captionlabels'] = {}
        # END Legacy-handling code.

        self.imagelabels = imagelabels
        self.imagepaths = imagepaths
        self.imagecaptions = state['imagecaptions']
        self.captionlabels = state['captionlabels']
        self.cur_imgidx = 0
        self.display_img(self.cur_imgidx, no_overwrite=True)
        #self.SetClientSize(self.parent.GetClientSize())
        self.SetupScrolling()
        self.SendSizeEvent()
        #self.Fit()
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
        state['imagecaptions'] = self.imagecaptions
        state['captionlabels'] = self.captionlabels
        f = open(statefile, 'wb')
        pickle.dump(state, f)
        f.close()

    def display_img(self, idx, no_overwrite=False):
        """Displays the image at idx, and allow the user to start labeling
        it. Also updates the progress_txt.
        Input:
            int idx: Idx into self.imagepaths of the image to display.
            bool no_overwrite: If True, then this will /not/ store the
                current text in the text input (self.inputctrl) into
                our internal data structures (self.imagelabels). This
                is useful (albeit a bit of a hack) within
                self.restore_session().
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
        self.update_caption_txt(self.cur_imgidx)
        self.update_listbox(self.cur_imgidx)
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

def do_extract_attr_patches(proj):
    """Extract all attribute patches from all blank ballots into
    the specified outdir. Saves them to:
        <projdir>/extract_attrs_templates/ATTRTYPE/*.png
    Output:
        (dict mapping, dict inv_mapping, where:
          mapping is {imgpath: {str attrtype: str patchpath}}
          inv_mapping is {str patchpath: (imgpath, attrtype)}
    """
    tmp2imgs = pickle.load(open(proj.template_to_images, 'rb'))
    blanks = tmp2imgs.values() # list of ((pathside0, pathside1,...), ...)
    return partask.do_partask(extract_attr_patches,
                              blanks,
                              _args=(proj,),
                              combfn=_extract_combfn,
                              init=({}, {}))
    
def _extract_combfn(result, subresult):
    """ Aux. function used for the partask.do_partask interface.
    Input:
        result: (dict mapping_0, dict invmapping_0)
        subresult: (dict mapping_1, dict invmapping_1)
    Output:
        The result of 'combining' result and subresult:
            (dict mapping*, dict invmapping*)
    """
    mapping, invmapping = result
    mapping_sub, invmapping_sub = subresult
    new_mapping = dict(mapping.items() + mapping_sub.items())
    new_invmapping = dict(invmapping.items() + invmapping_sub.items())
    return (new_mapping, new_invmapping)

def extract_attr_patches(blanks, (proj,)):
    """
    Extracts all image-based attributes from blank ballots, and saves
    the patches to the outdir proj.labelattrs_patchesdir.
    Input:
        list blanks: Of the form ((frontpath_i, backpath_i), ...)
        obj proj:
    Output:
        (dict mapping, dict inv_mapping)
    """
    try:
        outdir = os.path.join(proj.projdir_path,
                              proj.extract_attrs_templates)
        w_img, h_img = proj.imgsize
        # list of marshall'd attrboxes (i.e. dicts)
        ballot_attributes = pickle.load(open(proj.ballot_attributesfile, 'rb'))
        mapping = {} # maps {imgpath: {str attrtypestr: str patchPath}}
        inv_mapping = {} # maps {str patchPath: (imgpath, attrtypestr)}
        for blankpaths in blanks:
            for blankside, imgpath in enumerate(blankpaths):
                for attr in ballot_attributes:
                    if attr['is_digitbased']:
                        continue
                    imgname =  os.path.split(imgpath)[1]
                    attrside = attr['side']
                    assert type(attrside) == str # 'front' or 'back'
                    attrside = 0 if attrside == 'front' else 1
                    x1 = int(round(attr['x1']*w_img))
                    y1 = int(round(attr['y1']*h_img))
                    x2 = int(round(attr['x2']*w_img))
                    y2 = int(round(attr['y2']*h_img))
                    attrtype = common.get_attrtype_str(attr['attrs'])
                    if blankside == attrside:
                        # patchP: if outdir is: 'labelattrs_patchesdir',
                        # imgpath is: '/media/data1/election/blanks/foo/1.png',
                        # proj.templatesdir is: '/media/data1/election/blanks/
                        tmp = proj.templatesdir
                        if not tmp.endswith('/'):
                            tmp += '/'
                        partdir = os.path.split(imgpath[len(tmp):])[0] # foo/
                        patchrootDir = pathjoin(outdir,
                                                partdir,
                                                os.path.splitext(imgname)[0])
                        # patchrootDir: labelattrs_patchesdir/foo/1/
                        util_gui.create_dirs(patchrootDir)
                        patchoutP = pathjoin(patchrootDir, "{0}_{1}.png".format(os.path.splitext(imgname)[0],
                                                                                attrtype))
                        if not os.path.exists(patchoutP):
                        #if True:
                            # TODO: Only extract+save the imge patches
                            # when you /have/ to.
                            img = shared.standardImread(imgpath, flatten=True)
                            if abs(y1-y2) == 0 or abs(x1-x2) == 0:
                                print "Uh oh, about to crash. Why is this happening?"
                                print "    proj.imgsize:", proj.imgsize
                                print "    (y1,y2,x1,x2):", (y1,y2,x1,x2)
                                pdb.set_trace()
                            patch = img[y1:y2,x1:x2]
                            scipy.misc.imsave(patchoutP, patch)
                        mapping.setdefault(imgpath, {})[attrtype] = patchoutP
                        inv_mapping[patchoutP] = (imgpath, attrtype)
    except Exception as e:
        import traceback
        traceback.print_exc()
    return mapping, inv_mapping

def get_ordered_patchpaths(inv_mapping):
    """ Given an input 'inv_mapping', output a list of patchpaths such
    that the patchpaths is sorted by attribute type.
    Input:
        dict inv_mapping: maps {str patchpath: (imgpath, attrtype)}
    Output:
        A list of patchpaths.
    """
    attrtypes = {} # maps {str attrtype: (patchpath_i, ...)}
    for patchpath, (imgpath, attrtype) in inv_mapping.iteritems():
        attrtypes.setdefault(attrtype, []).append(patchpath)
    patchpaths = []
    for attrtype, paths in attrtypes.iteritems():
        patchpaths.extend(paths)
    return patchpaths

