import sys, os, re, pdb, traceback
from os.path import join as pathjoin
try:
    import cPickle as pickle
except:
    import pickle
import wx

PROJ_FNAME = 'proj.p'

class ProjectPanel(wx.Panel):
    def __init__(self, parent, *args, **kwargs):
        wx.Panel.__init__(self, parent, *args, **kwargs)

        # PROJDIR: Root directory of all projects
        self.projdir = None
        # PROJECTS: List of Project instances
        self.projects = []

        self.init_ui()

    def init_ui(self):
        box0 = wx.StaticBox(self, label="Select Election Project")
        ssizer0 = wx.StaticBoxSizer(box0, wx.VERTICAL)

        txt0 = wx.StaticText(self, label="Select the election project you'd like to work on.")
        box1 = wx.StaticBox(self, label="Election Projects")
        ssizer1 = wx.StaticBoxSizer(box1, wx.VERTICAL)

        self.listbox_projs = wx.ListBox(self, choices=(), size=(500, 400))
        
        btn_create = wx.Button(self, label="Create New Project...")
        btn_create.Bind(wx.EVT_BUTTON, self.onButton_create)
        btn_remove = wx.Button(self, label="Delete Selected Project...")
        btn_remove.Bind(wx.EVT_BUTTON, self.onButton_remove)
        btnsizer = wx.BoxSizer(wx.HORIZONTAL)
        btnsizer.AddMany([(btn_create,), (btn_remove,)])
        
        ssizer1.AddMany([(self.listbox_projs,), (btnsizer,)])

        ssizer0.AddMany([(txt0,), (ssizer1,)])
        
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(ssizer0)
        self.SetSizer(self.sizer)
        self.Layout()

    def start(self, projdir):
        """
        Input:
            str PROJDIR: Root directory where all projects reside.
        """
        self.projdir = projdir
        projects = load_projects(projdir)
        for proj in projects:
            self.add_project(proj)

    def get_project(self):
        """ Returns the Project instance of the selected project. """
        idx = self.listbox_projs.GetSelection()
        if idx == wx.NOT_FOUND:
            print "NONE SELECTED"
            return None
        return self.projects[idx]

    def add_project(self, proj):
        self.projects.append(proj)
        self.listbox_projs.Append(proj.name)
    def contains_project(self, projname):
        return projname in [proj.name for proj in self.projects]
    def create_new_project(self, name):
        proj = create_project(name, pathjoin(self.projdir, name))
        self.add_project(proj)

    def onButton_create(self, evt):
        dlg = wx.TextEntryDialog(self, message="New Project Name:", caption="New Project", defaultValue="ProjectNameHere")
        val = dlg.ShowModal()
        if val == wx.ID_OK:
            project_name = dlg.GetValue().strip()
            if self.contains_project(project_name):
                dlg = wx.MessageDialog(self, 
                                       message="{0} already exists as a project.".format(project_name),
                                       style=wx.OK)
                dlg.ShowModal()
                return
            else:
                if not is_valid_projectname(project_name):
                    warn = wx.MessageDialog(self,
                                            message='{0} is not a valid \
project name. Please only use letters, numbers, and punctuation.'.format(project_name),
                                            style=wx.OK)
                    warn.ShowModal()
                    return
                self.create_new_project(project_name)
                self.listbox_projs.SetStringSelection(project_name)

    def onButton_remove(self, evt):
        """
        Removes project from the ListBox, internal data structures,
        and from the projects/ directory.
        """
        self.projects.remove(project)
        idx = self.listbox_projs.FindString(project.name)
        self.listbox_projs.Delete(idx)
        projdir = project.projdir_path
        print 'removing everything in:', projdir
        #shutil.rmtree(projdir)

def is_valid_projectname(name):
    """
    Only allow letters, numbers, and [', ", _, (, )].
    """
    pattern = r'(\w|\d|[_\'"()])+'
    return not re.match(pattern, name) == None

class Project(object):
    """
    A Project is represented in the filesystem as a folder in the
    projects/ directory, where the name of the folder denotes the
    project name.
    """
    closehook = []

    def __init__(self, name='', projdir_path=''):
        self.vals = {'name': name,
                     'projdir_path': projdir_path,
                     'voteddir': '',
                     'is_multipage': False,
                     'num_pages': None,
                     'vendor_obj': None,
                     'partition_exmpls': 'partition_exmpls.p',
                     'partitions_map': 'partitions_map.p',
                     'partitions_invmap': 'partitions_invmap.p',
                     'decoded_map': 'decoded_map.p',
                     'imginfo_map': 'imginfo_map.p',
                     'bbs_map': 'bbs_map.p',
                     'partition_attrmap': 'partition_attrmap.p',
                     'partition_targets_map': 'partitions_targets_map.p',
                     'target_locs_map': 'target_locs_map.p',
                     'extract_results': 'extract_results.p',
                     'infer_bounding_boxes': False,
                     'ocr_tmp_dir': pathjoin(projdir_path, 'ocr_tmp_dir'),
                     'contest_id': pathjoin(projdir_path, 'contest_id.csv'),
                     'contest_text': pathjoin(projdir_path, 'contest_text.csv'),
                     'contest_internal': pathjoin(projdir_path, 'contest_internal.p'),
                     'contest_grouping_data': pathjoin(projdir_path, 'contest_grouping_data.p'),
                     'target_locs_dir': pathjoin(projdir_path, 'target_locations'),
                     'tmp': pathjoin(projdir_path, 'tmp'),
                     'extracted_dir': pathjoin(projdir_path, 'extracted'),
                     'extracted_metadata': pathjoin(projdir_path, 'extracted_metadata'),
                     'ballot_metadata': pathjoin(projdir_path, 'ballot_metadata'),
                     'classified': pathjoin(projdir_path, 'classified'),
                     'timing_runtarget': pathjoin(projdir_path, 'timing_runtarget'),
                     'threshold_internal': pathjoin(projdir_path, 'threshold_internal.p'),
                     'sample_flipped': pathjoin(projdir_path, 'sample_flipped'),
                     'extractedfile': pathjoin(projdir_path, 'extractedfile'),
                     'targets_result': pathjoin(projdir_path, 'targets_result.csv'),
                     'ballot_to_images': pathjoin(projdir_path, 'ballot_to_images.p'),
                     'image_to_ballot': pathjoin(projdir_path, 'image_to_ballot.p'),
                     'election_results': pathjoin(projdir_path, 'election_results.txt'),
                     'election_results_batches': pathjoin(projdir_path, 'election_results_batches.txt'),
                     'cvr_csv': pathjoin(projdir_path, 'cvr.csv'),
                     'cvr_dir': pathjoin(projdir_path, 'cvr'),
                     'quarantined': pathjoin(projdir_path, 'quarantined.csv'),
                     'quarantined_manual': pathjoin(projdir_path, 'quarantined_manual.csv'),
                     'quarantine_res': pathjoin(projdir_path, 'quarantine_res.csv'),
                     'quarantine_attributes': pathjoin(projdir_path, 'quarantine_attributes.csv'),
                     'quarantine_internal': pathjoin(projdir_path, 'quarantine_internal.p'),
                     'extracted_precinct_dir': pathjoin(projdir_path, 'extracted_precincts'),
                     'ballot_grouping_metadata': pathjoin(projdir_path, 'ballot_grouping_metadata'),
                     'patch_loc_dir': pathjoin(projdir_path, 'precinct_locations'),
                     'attr_internal': pathjoin(projdir_path, 'attr_internal.p'),
                     'grouping_results': pathjoin(projdir_path, 'grouping_results.csv'),
                     'ballot_attributesfile': pathjoin(projdir_path, 'ballot_attributes.p'),
                     'imgsize': (0,0),
                     'frontback_map': pathjoin(projdir_path, 'frontback_map.p'),
                     'extracted_digitpatch_dir': 'extracted_digitpatches',
                     'digit_exemplars_outdir': 'digit_exemplars',
                     'digit_exemplars_map': 'digit_exemplars_map.p',
                     'precinctnums_outpath': 'precinctnums.txt',
                     'num_digitsmap': 'num_digitsmap.p',
                     'digitgroup_results': 'digitgroup_results.p',
                     'labeldigitstate': '_labeldigitstate.p',
                     'voteddigits_dir': 'voteddigits_dir',
                     'attrgroup_results': 'attrgroup_results.p',
                     'labelpanel_state': 'labelpanel_state.p',
                     'labelattrs_out': 'labelattrs_out.csv',
                     'labelattrs_patchesdir': 'labelattrs_patchesdir',
                     'attrexemplars_dir': 'attrexemplars_dir',
                     'multexemplars_map': 'multexemplars_map.p',
                     'image_to_page': 'image_to_page.p',
                     'image_to_flip': 'image_to_flip.p',
                     'rejected_hashes': 'rejected_hashes.p',
                     'accepted_hashes': 'accepted_hashes.p',
                     'custom_attrs': 'custom_attrs.p',
                     'digitpatch2temp': 'digitpatch2temp.p',
                     'digitattrvals_blanks': 'digitattrvals_blanks.p',
                     'digitpatchpath_scoresBlank': 'digitpatchpath_scoresBlank.p',
                     'digitpatchpath_scoresVoted': 'digitpatchpath_scoresVoted.p',
                     'digitmatch_info': 'digitmatch_info.p',
                     'extract_attrs_templates': 'extract_attrs_templates',
                     'digit_median_dists': 'digit_median_dists.p',
                     'blank2attrpatch': 'blank2attrpatch.p',
                     'invblank2attrpatch': 'invblank2attrpatch.p',
                     'digitmultexemplars': 'digitmultexemplars',
                     'digitmultexemplars_map': 'digitmultexemplars_map.p',
                     'grouplabels_record': 'grouplabels_record.p',
                     'devmode': True}
        self.createFields()

    def addCloseEvent(self, func):
        Project.closehook.append(func)

    def removeCloseEvent(self, func):
        Project.closehook = [x for x in Project.closehook if x != func]

    def createFields(self):
        for k,v in self.vals.items():
            setattr(self, k, v)

    def save(self):
        write_project(self)

    def __repr__(self):
        return 'Project({0})'.format(self.name)

def load_projects(projdir):
    """ Returns a list of all Project instances contained in PROJDIR.
    Input:
        str PROJDIR:
    Output:
        list PROJECTS.
    """
    projects = []
    for dirpath, dirnames, filenames in os.walk(projdir):
        for f in filenames:
            if f == PROJ_FNAME:
                fullpath = pathjoin(dirpath, f)
                try:
                    proj = pickle.load(open(fullpath, 'rb'))
                    projects.append(proj)
                except:
                    pass
    return projects

def create_project(name, projrootdir):
    proj = Project(name, projrootdir)
    projoutpath = pathjoin(projrootdir, PROJ_FNAME)
    try: os.makedirs(projrootdir)
    except: pass
    pickle.dump(proj, open(projoutpath, 'wb'))
    return proj

def write_project(project):
    projoutpath = pathjoin(project.projdir_path, PROJ_FNAME)
    pickle.dump(project, open(projoutpath, 'wb'))
    return project

