import os, sys, csv, time, optparse, threading, shutil, re, traceback, pdb, multiprocessing, logging
from xml.etree.ElementTree import Element, ElementTree
from os.path import join as pathjoin

import util
import specify_voting_targets.find_targets_wizard as find_targets_wizard
import specify_voting_targets.util_gui as util_gui
import specify_voting_targets.util_widgets as util_widgets
import specify_voting_targets.sanity_check as sanity_check
import pre_processing.straighten_ballots as straighten_ballots
import wx, Image, cv
from wx.lib.pubsub import Publisher
import pickle

import sanitycheck
from tab_wrap import tab_wrap
from threshold.threshold import ThresholdPanel
from labelcontest.labelcontest import LabelContest
from runtargets.runtargets import RunTargets
from grouping.define_attributes import DefineAttributesPanel
from grouping.define_attributes import AttributeBox
from grouping.label_attributes import LabelAttributesPanel, GroupAttrsFrame
from digits_ui.digits_ui import LabelDigitsPanel
from grouping.verify_grouping import GroupingMasterPanel
from post_processing.postprocess import ResultsPanel
from quarantine.quarantinepanel import QuarantinePanel

TIMER = None
TIMING_FILENAME = 'timings.log'

class BlankPanel(wx.Panel):
    def __init__(self, parent, *args, **kwargs):
        wx.Panel.__init__(self, parent, *args, **kwargs) 
        wx.StaticText(self, -1, "TODO "*100)

class ProjectPanel(wx.Panel):
    def __init__(self, parent, *args, **kwargs):
        wx.Panel.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        # List of Project instances
        self.projects = []
        self.project = None

        panel1 = wx.Panel(self)
        panel1.sizer = wx.BoxSizer(wx.VERTICAL)

        box = wx.StaticBox(panel1, label="Setup Project Configuration")

        box2 = wx.StaticBox(panel1, label="Project Configurations")
        self.listbox_projects = wx.ListBox(panel1, choices=(), size=(500, 400))
        self.listbox_projects.Bind(wx.EVT_LISTBOX, self.on_selected)
        
        self.panel_btn = wx.Panel(panel1)
        self.btn_create = wx.Button(self.panel_btn, label="Create New Project...")
        self.btn_create.Bind(wx.EVT_BUTTON, self.on_button)
        self.btn_remove = wx.Button(self.panel_btn, label="Delete Selected Project")
        self.btn_remove.Bind(wx.EVT_BUTTON, self.on_button_remove)

        self.panel_btn.sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.panel_btn.sizer.Add(self.btn_create)
        self.panel_btn.sizer.Add(self.btn_remove)
        self.panel_btn.SetSizer(self.panel_btn.sizer)
        self.panel_btn.Fit()

        boxsizer = wx.StaticBoxSizer(box, orient=wx.VERTICAL)
        boxsizer.Add((0, 25))
        boxsizer2 = wx.StaticBoxSizer(box2, orient=wx.VERTICAL)
        boxsizer2.Add(self.listbox_projects)
        boxsizer.Add(boxsizer2)
        boxsizer.Add(self.panel_btn)

        panel1.sizer.Add(boxsizer)
        panel1.SetSizer(panel1.sizer)

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(panel1)
        
        self.SetSizer(self.sizer)

    def start(self):
        self.detect_projects()

    def detect_projects(self):
        """
        Look in directory MainFrame.DIR_PROJECTS, and populate the
        ListBox.
        """
        projects = get_projects(MainFrame.DIR_PROJECTS)
        for project in projects:
            self.add_project(project)
    def add_project(self, project):
        self.projects.append(project)
        self.listbox_projects.Append(project.name)
    def remove_project(self, project):
        """
        Removes project from the ListBox, internal data structures,
        and from the projects/ directory.
        """
        self.projects.remove(project)
        idx = self.listbox_projects.FindString(project.name)
        self.listbox_projects.Delete(idx)
        projdir = project.projdir_path
        print 'removing everything in:', projdir
        shutil.rmtree(projdir)
        self.project = None
        Publisher().sendMessage("broadcast.cant_proceed")
                         
    def create_new_project(self, name):
        util_gui.create_dirs(pathjoin(MainFrame.DIR_PROJECTS, name))
        create_projconfig(name, pathjoin(MainFrame.DIR_PROJECTS, name))
        self.add_project(Project(name, pathjoin(MainFrame.DIR_PROJECTS, name)))
    def contains_project(self, name):
        return name in [proj.name for proj in self.projects]

    def get_selected_project(self):
        return self.project
    def get_selected_projectname(self):
        return self.project.name
    def get_selected_projectdir(self):
        return self.project.projdir_path

    #### Event handlers

    def on_button(self, evt):
        """
        Triggered when the user clicks 'Create New Project' button.
        """
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
                self.listbox_projects.SetStringSelection(project_name)
                self.project = [proj for proj in self.projects if proj.name == project_name][0]
                Publisher().sendMessage("broadcast.can_proceed")
        
    def on_button_remove(self, evt):
        selected_project = self.project
        if not selected_project:
            dlg = wx.MessageDialog(self, message="No project selected. \
Please select a project in order to remove it.", style=wx.OK)
            dlg.ShowModal()
        else:
            dlg = wx.MessageDialog(self, message="Are you sure you want \
to delete the '{0}' project? This will also delete all associated files \
in the projects/ directory.".format(selected_project.name), style=wx.YES | wx.NO)
            val = dlg.ShowModal()
            if val == wx.ID_YES:
                self.remove_project(selected_project)
                
    def on_selected(self, evt):
        """
        Triggered when the user selects an entry in the listbox
        """
        projname = evt.GetEventObject().GetStringSelection()
        try:
            self.project = [proj for proj in self.projects if proj.name == projname][0]
            Publisher().sendMessage("broadcast.can_proceed")
        except Exception as e:
            # For some reason, this callback is invoked on UI exit, so catch
            # the inevitable error. 
            pass

class ConfigPanel(wx.Panel):
    def __init__(self, parent, *args, **kwargs):
        wx.Panel.__init__(self, parent, style=wx.SIMPLE_BORDER, *args, **kwargs)
        
        # Instance vars
        self.parent = parent
        self.project = None
        self.templatesdir = ""
        self.samplesdir = ""
        
        # Set up widgets
        # StaticBoxes are weird, in that the contents of the StaticBox
        # are siblings of the StaticBox, and you add the StaticBoxSizer
        # to the parent widget. API, why u inconsistent?
        self.box_temps = wx.StaticBox(self, label="Templates")
        self.box_temps.sizer = wx.StaticBoxSizer(self.box_temps, orient=wx.VERTICAL)
        self.box_temps.txt = wx.StaticText(self, label="Please choose the directory where the blank ballot images reside.")
        self.box_temps.btn_opendir = wx.Button(self, label="Choose templates directory...")
        self.box_temps.btn_opendir.Bind(wx.EVT_BUTTON, self.onButton_choosetempdir)
        self.box_temps.txt2 = wx.StaticText(self, label="Blank ballot directory: ")
        self.box_temps.txt_templatepath = wx.StaticText(self)
        #self.box_temps.txt_templatepath.SetMinSize((250, 21))
        self.box_temps.sizer.Add(self.box_temps.txt)
        self.box_temps.sizer.Add(self.box_temps.btn_opendir)
        self.box_temps.sizer.Add(self.box_temps.txt2)
        self.box_temps.sizer.Add(self.box_temps.txt_templatepath)
        self.box_temps.Fit()
        
        self.box_samples = wx.StaticBox(self, label="Samples")
        self.box_samples.sizer = wx.StaticBoxSizer(self.box_samples, orient=wx.VERTICAL)
        self.box_samples.txt = wx.StaticText(self, label="Please choose the directory where the sample images reside.")
        self.box_samples.btn = wx.Button(self, label="Choose voted ballot directory...")
        self.box_samples.btn.Bind(wx.EVT_BUTTON, self.onButton_choosesamplesdir)
        self.box_samples.txt2 = wx.StaticText(self, label="Voted ballot directory:")
        self.box_samples.txt_samplespath = wx.StaticText(self)
        #self.box_samples.txt_samplespath.SetMinSize((250, 21))
        self.box_samples.sizer.Add(self.box_samples.txt)
        self.box_samples.sizer.Add(self.box_samples.btn)
        self.box_samples.sizer.Add(self.box_samples.txt2)
        self.box_samples.sizer.Add(self.box_samples.txt_samplespath)
        self.box_samples.Fit()
        
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        
        self.top_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        self.upper_left_sizer = wx.BoxSizer(wx.VERTICAL)
        self.upper_left_sizer.Add(self.box_temps.sizer, flag=wx.EXPAND)
        self.lower_left_sizer = wx.BoxSizer(wx.VERTICAL)
        self.lower_left_sizer.Add(self.box_samples.sizer, flag=wx.EXPAND)
        self.left_sizer = wx.GridSizer(2, 1, 10, 10)
        self.left_sizer.Add(self.upper_left_sizer, flag=wx.EXPAND)
        self.left_sizer.Add(self.lower_left_sizer, flag=wx.EXPAND)
        self.left_sizer.Add((0,100))
        self.is_double_sided = wx.CheckBox(self, -1, label="Double sided ballots.")
        self.is_double_sided.Bind(wx.EVT_CHECKBOX, self.changeDoubleSided, self.is_double_sided)
        self.left_sizer.Add(self.is_double_sided)
        
        self.is_straightened = wx.CheckBox(self, -1, label="Ballots already straightened.")
        self.is_straightened.Bind(wx.EVT_CHECKBOX, self.changeStraightened, self.is_straightened)
        self.left_sizer.Add(self.is_straightened)
        
        self.upper_scroll = wx.ListBox(self)
        self.upper_scroll.box = wx.StaticBox(self, label="For the blank ballots, the following files were skipped:")
        self.upper_scroll.sizer = wx.StaticBoxSizer(self.upper_scroll.box, orient=wx.VERTICAL)
        self.upper_scroll.sizer.Add(self.upper_scroll, 1, flag=wx.EXPAND)
        self.lower_scroll = wx.ListBox(self)
        self.lower_scroll.box = wx.StaticBox(self, label="For the voted ballots, the following files were skipped:")
        self.lower_scroll.sizer = wx.StaticBoxSizer(self.lower_scroll.box, orient=wx.VERTICAL)
        self.lower_scroll.sizer.Add(self.lower_scroll, 1, flag=wx.EXPAND)
        self.right_sizer = wx.GridSizer(2, 1, 10, 10)
        self.right_sizer.Add(self.upper_scroll.sizer, flag=wx.EXPAND)
        self.right_sizer.Add(self.lower_scroll.sizer, flag=wx.EXPAND)
        
        self.top_sizer.Add(self.left_sizer, flag=wx.EXPAND)
        self.top_sizer.Add((10, 10))
        self.top_sizer.Add(self.right_sizer, 1, flag=wx.EXPAND)
        
        self.btn_run = wx.Button(self, label="Run sanity check")
        self.btn_run.Bind(wx.EVT_BUTTON, self.onButton_runsanitycheck)
        self.btn_run.box = wx.StaticBox(self)
        self.btn_run.sizer = wx.StaticBoxSizer(self.btn_run.box, orient=wx.VERTICAL)
        self.btn_run.sizer.Add(self.btn_run)
        
        self.sizer.Add(self.top_sizer, 1, flag=wx.EXPAND)
        self.sizer.Add(self.btn_run.sizer, flag=wx.EXPAND)
        
        self.SetSizer(self.sizer)
        self.Fit()
        sanity_check.EVT_SANITYCHECK(self, self.onSanityCheck)

        Publisher().subscribe(self._pubsub_project, "broadcast.project")

    def initDoubleSided(self):
        ds = DoubleSided(self, -1)
        ds.regex.SetValue("(.*)")
        ds.finished()
        ds.Destroy()

    def changeStraightened(self, x):
        if not self.project.raw_templatesdir or not self.project.raw_samplesdir:
            dlg = wx.MessageDialog(self, message="Please select the \
blank ballots and voted ballots directories first.")
            self.Disable()
            dlg.ShowModal()
            self.Enable()
            return
        val = self.is_straightened.GetValue()
        if val:
            self.project.templatesdir = self.project.raw_templatesdir
            self.project.samplesdir = self.project.raw_samplesdir
            self.project.blankballots_straightdir = self.project.raw_templatesdir
            self.project.votedballots_straightdir = self.project.raw_samplesdir
        else:
            self.project.templatesdir = ''
            self.project.samplesdir = ''
            self.blankballots_straightdir = pathjoin(self.project.projdir_path, 'blankballots_straight')
            self.votedballots_straightdir = pathjoin(self.project.projdir_path, 'votedballots_straight')
        self.project.are_votedballots_straightened = val
        self.project.are_blankballots_straightened = val
            
    def changeDoubleSided(self, x):
        val = self.is_double_sided.GetValue()
        self.project.is_multipage = val
        if val:
            ds = DoubleSided(self, -1)
            ds.Show()

    def wrap(self, text):
        res = ""
        for i in range(0,len(text),50):
            res += text[i:i+50]+"\n"
        return res

    def set_templatepath(self, path):
        self.templatesdir = os.path.abspath(path)
        self.box_temps.txt_templatepath.SetLabel(self.wrap(self.templatesdir))
        self.project.raw_templatesdir = self.templatesdir
        Publisher().sendMessage("broadcast.projupdate")
        Publisher().sendMessage("processing.register", data=self.project)
    def set_samplepath(self, path):
        self.samplesdir = os.path.abspath(path)
        self.box_samples.txt_samplespath.SetLabel(self.wrap(self.samplesdir))
        self.project.raw_samplesdir = self.samplesdir
        Publisher().sendMessage("broadcast.projupdate")
        Publisher().sendMessage("processing.register", data=self.project)
    def get_templatepath(self):
        return self.box_temps.txt_templatepath.GetLabelText().replace("\n", "")
    def get_samplepath(self):
        return self.box_samples.txt_samplespath.GetLabelText().replace("\n", "")
        
    def onSanityCheck(self, evt):
        """
        Triggered when either the templates or samples sanity check
        completes. Update the relevant ListBox widget with the results
        of a sanity check.
        """
        type, results_dict = evt.data
        listbox = self.upper_scroll if type == 'templates' else self.lower_scroll
        if len(results_dict) == 0:
            listbox.Append("All files valid")
        else:
            for imgpath, msg in results_dict.items():
                listbox.Append(imgpath + ": " + msg)
        if type == 'samples':
            # Assume that we first process the templates, then the samples last
            TIMER.stop_task(('cpu', MainFrame.map_pages[MainFrame.CONFIG]['cpu']))
            TIMER.start_task(('user', MainFrame.map_pages[MainFrame.CONFIG]['user']))
            self.parent.Enable()

    def _pubsub_project(self, msg):
        """
        Triggered whenever the current project changes.
        """
        project = msg.data
        self.project = project
        templatesdir, samplesdir = project.raw_templatesdir, project.raw_samplesdir
        if os.path.exists(templatesdir):
            self.set_templatepath(templatesdir)
        else:
            self.box_temps.txt_templatepath.SetLabel(templatesdir)
        if os.path.exists(samplesdir):
            self.set_samplepath(samplesdir)
        else:
            self.box_samples.txt_samplespath.SetLabel(samplesdir)
        if self.project.is_multipage:
            self.is_double_sided.SetValue(True)

        if os.path.exists(templatesdir) and os.path.exists(samplesdir):
            Publisher().sendMessage("broadcast.can_proceed")

    #### Event Handlers
    def onButton_choosetempdir(self, evt):
        dlg = wx.DirDialog(self, "Select Directory", defaultPath=os.getcwd(), style=wx.DD_DEFAULT_STYLE)
        result = dlg.ShowModal()
        if result == wx.ID_OK:
            dirpath = dlg.GetPath()
            self.set_templatepath(dirpath)
            if self.get_samplepath() != '' and os.path.exists(self.get_samplepath()):
                Publisher().sendMessage("broadcast.can_proceed")

    def onButton_choosesamplesdir(self, evt):
        dlg = wx.DirDialog(self, "Select Directory", defaultPath=os.getcwd(), style=wx.DD_DEFAULT_STYLE)
        result = dlg.ShowModal()
        if result == wx.ID_OK:
            dirpath = dlg.GetPath()
            self.set_samplepath(dirpath)
            if self.get_templatepath() != '' and os.path.exists(self.get_samplepath()):
                Publisher().sendMessage("broadcast.can_proceed")
                
    def onButton_runsanitycheck(self, evt):
        TIMER.stop_task(('user', MainFrame.map_pages[MainFrame.CONFIG]['user']))
        TIMER.start_task(('cpu', MainFrame.map_pages[MainFrame.CONFIG]['cpu']))
        self.upper_scroll.Clear()
        self.lower_scroll.Clear()
        num_files = 0
        for dirpath, dirnames, filenames in os.walk(self.templatesdir):
            num_files += len(filenames)
        for dirpath, dirnames, filenames in os.walk(self.samplesdir):
            num_files += len(filenames)
        self.parent.Disable()
        pgauge = util_widgets.ProgressGauge(self, num_files, msg="Checking files...")
        pgauge.Show()
        thread = threading.Thread(target=sanity_check.sanity_check,
                                  args=(self.templatesdir, self.samplesdir, self))
        thread.start()

class DoubleSided(wx.Frame):
    def __init__(self, parent, id):
        self.parent = parent
        wx.Frame.__init__(self, parent, id, "Set Double Sided Properties")
        sizer = wx.BoxSizer(wx.VERTICAL)
        t = wx.StaticText(self, -1, "Enter a regex to match on the file name")
        self.regex = wx.TextCtrl(self, -1)
        self.regex.SetValue(r"(.*)-(.*)")
        sizer.Add(t)
        sizer.Add(self.regex)
        t = wx.StaticText(self, -1, "How to construct the similar portion")
        self.part = wx.TextCtrl(self, -1)
        self.part.SetValue(r"\1")
        sizer.Add(t)
        sizer.Add(self.part)
        self.check = wx.CheckBox(self, -1, "Ballots alternate front and back.")
        self.check.Bind(wx.EVT_CHECKBOX, self.togglefrontback)
        sizer.Add(self.check)
        d = wx.Button(self, -1, label="Done")
        d.Bind(wx.EVT_BUTTON, self.finished)
        sizer.Add(d)
        self.SetSizer(sizer)
        self.isalternating = False
    
    def togglefrontback(self, evt):
        self.isalternating = self.check.GetValue()
        if self.isalternating:
            self.regex.Disable()
            self.part.Disable()
        else:
            self.regex.Enable()
            self.part.Enable()

    def finished(self, x=None):
        voteddir_raw = os.path.abspath(self.parent.project.raw_samplesdir)
        blankdir_raw = os.path.abspath(self.parent.project.raw_templatesdir)
        voteddir = os.path.abspath(self.parent.project.votedballots_straightdir)
        blankdir = os.path.abspath(self.parent.project.blankballots_straightdir)
        def get(from_dir, to_dir, load_dir):
            res = []
            for root,_,files in os.walk(load_dir):
                files = [x for x in files if util.is_image_ext(x)]
                # Straightener converts all images to .png
                files = util.replace_exts(files, '.png')
                if len(files)%2 != 0 and self.isalternating:
                    raise Exception("OH NO! Odd number of files in directory %s"%root)
                root = os.path.abspath(root)
                res += [util.to_straightened_path(pathjoin(root, x), from_dir, to_dir) for x in files]
            return res

        images = get(voteddir_raw, voteddir, self.parent.samplesdir)
        templates = get(blankdir_raw, blankdir, self.parent.templatesdir)

        if self.isalternating:
            images = sorted(images)
            templates = sorted(templates)
            images = dict(zip(images[::2], map(list,zip(images,images[1:]))[::2]))
            templates = dict(zip(templates[::2], map(list,zip(templates,templates[1:]))[::2]))
        else:
            split = self.regex.GetValue()
            join = self.part.GetValue()
            def group(it):
                it = [(re.sub(split, join, x), x) for x in it]
                ht = {}
                for a,b in it:
                    if a not in ht:
                        ht[a] = []
                    ht[a].append(b)
                return ht
            images = group(images)
            templates = group(templates)
        
        pickle.dump(images, open(self.parent.project.ballot_to_images, "w"))
        pickle.dump(templates, open(self.parent.project.template_to_images, "w"))

        rev_images = {}
        for k,v in images.items():
            for vv in v:
                rev_images[vv] = k
        rev_temp = {}
        for k,v in templates.items():
            for vv in v:
                rev_temp[vv] = k

        pickle.dump(rev_images, open(self.parent.project.image_to_ballot, "w"))
        pickle.dump(rev_temp, open(self.parent.project.image_to_template, "w"))

        self.Destroy()
        
class MainFrame(wx.Frame):
    """
    Frame that holds all other widgets
    """
    DIR_PROJECTS = 'projects'

    PROJECTS = 0
    CONFIG = 1
    SELECT_TARGETS = 2
    DEFINE_ATTRIBUTES = 3
    LABEL_ATTRS = 4
    LABEL_DIGIT_ATTRS = 5
    LABEL_CONTESTS = 6
    CORRECT_GROUPING = 7
    RUN = 8
    SET_THRESHOLD = 9
    QUARANTINE = 10
    PROCESS = 11

    map_pages = {}

    #----------------------------------------------------------------------
    def __init__(self, options=None, *args, **kwargs):
        wx.Frame.__init__(self, None, wx.ID_ANY, "OpenCount", *args, **kwargs)
        self.options = options

        MainFrame.map_pages = {MainFrame.PROJECTS: {'user': 'Set-up Project',
                                                    'cpu' : 'Set-up Project'},
                               MainFrame.CONFIG: {'user': 'Set Templates/Samples Dir',
                                                  'cpu' : 'Sanity-check Images Computation'},
                               MainFrame.SELECT_TARGETS: {'user': 'Select/Group Voting Targets',
                                                          'cpu' : 'TemplateMatch Targets Computation'},
                               MainFrame.DEFINE_ATTRIBUTES: {'user': 'Label Ballot Attributes',
                                                           'cpu' : 'Label Ballot Attributes'},
                               MainFrame.LABEL_ATTRS: {'user': 'Label Ballot Attributes',
                                                           'cpu' : 'Label Ballot Attributes'},
                               MainFrame.LABEL_DIGIT_ATTRS: {'user': 'Label Digit-Attributes',
                                                             'cpu': 'Label Digit-Attributes'},
                               MainFrame.LABEL_CONTESTS: {'user': 'Label Contests data entry',
                                                          'cpu' : 'Label Contests data entry'},
                               MainFrame.CORRECT_GROUPING: {'user': 'Verify Ballot Grouping',
                                                            'cpu' : 'Group Ballots Computation'},
                               MainFrame.RUN: {'user': 'Target Extraction',
                                               'cpu' : 'Target Extraction Computation'},
                               MainFrame.SET_THRESHOLD: {'user': 'Set Voter-Mark Threshold',
                                                         'cpu' : 'Set Voter-Mark Threshold'},
                               MainFrame.PROCESS: {'user': 'Process/Create CVRs',
                                                   'cpu' : 'Process/Create CVRs'},
                               MainFrame.DEFINE_ATTRIBUTES: {'user': 'Define Ballot Attributes',
                                                             'cpu' : 'Define Ballot Attributes'}}
        mainpanel = wx.Panel(self)
        util_gui.create_dirs(MainFrame.DIR_PROJECTS)
        sizer = wx.BoxSizer(wx.VERTICAL)

        self.project = None
        self.num_templates = -1
        self.num_samples = -1
        self.waiting_for_votedstraights = False

        menubar = wx.MenuBar()
        optionsmenu = wx.Menu()
        advoptsitem = optionsmenu.Append(wx.ID_ANY, 'Advanced Options...', 'View/Set Advanced Options')
        
        helpmenu = wx.Menu()
        helpitem = helpmenu.Append(wx.ID_ABOUT, 'About...', 'About this application')
        menubar.Append(optionsmenu, '&Options')
        menubar.Append(helpmenu, '&Help')
        self.SetMenuBar(menubar)
        self.Bind(wx.EVT_MENU, self.onMenu_about, helpitem)
        self.Bind(wx.EVT_MENU, self.onMenu_advopts, advoptsitem)

        notebook = wx.Notebook(mainpanel)
        self.notebook = notebook

        # Instance vars/setup to add tab animations
        self.timer = wx.Timer(self)
        # colors to cycle through (beta)
        self._colors = ["White", "Green"]
        # Will only animate if this is True
        self._running = False
        # Current 'frame' number
        self._coloridx = 0
        # Current page index that we're animating
        self._pageidx = 0
        self.Bind(wx.EVT_TIMER, self.on_timer, self.timer)
        
        sizer.Add(notebook, 1, wx.ALL|wx.EXPAND, 5)

        mainpanel.SetSizer(sizer)

        self.Layout()
        self.Maximize()
        self.Show()
        
        self.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.onPageChange)
        self.Bind(wx.EVT_CLOSE, self.onClose)

        Publisher().subscribe(self._pubsub_can_proceed, "broadcast.can_proceed")
        Publisher().subscribe(self._pubsub_cant_proceed, "broadcast.cant_proceed")
        Publisher().subscribe(self._pubsub_update_tab_icon, "signals.mainframe.update_tab_icon")
        Publisher().subscribe(self._pubsub_rundone, "signals.run_done")
        Publisher().subscribe(self._pubsub_blankballot_done, "straighten_blankballot_done")
        Publisher().subscribe(self._pubsub_votedballot_done, "straighten_votedballot_done")

        id1 = wx.NewId()
        self.Bind(wx.EVT_MENU, self.emit_undo, id=id1)
        accel_tbl = wx.AcceleratorTable([(wx.ACCEL_CTRL, ord('Z'), id1)])
        self.SetAcceleratorTable(accel_tbl)

        self.panel_projects = ProjectPanel(notebook)
        notebook.AddPage(self.panel_projects, "Projects")
        wx.CallAfter(self.setup_panels)

    def setup_panels(self):
        notebook = self.notebook
        self.panel_config = ConfigPanel(notebook)
        self.panel_specify_voting_targets = find_targets_wizard.SpecifyTargetsPanel(notebook)
        self.panel_specify_voting_targets.unsubscribe_pubsubs()
        self.panel_label_contests = tab_wrap(LabelContest)(notebook)
        self.panel_define_attrs = DefineAttributesPanel(notebook)
        self.panel_define_attrs.unsubscribe_pubsubs()
        self.panel_label_attrs = tab_wrap(LabelAttributesPanel)(notebook)
        self.panel_label_attrs.unsubscribe_pubsubs()
        self.panel_label_digitattrs = tab_wrap(LabelDigitsPanel)(notebook)
        self.panel_correct_grouping = GroupingMasterPanel(notebook)
        self.panel_run = RunTargets(notebook)
        self.panel_set_threshold = tab_wrap(ThresholdPanel)(notebook)
        self.panel_quarantine = QuarantinePanel(notebook)
        self.panel_process = ResultsPanel(notebook)
        self.pages = [(self.panel_projects, "Projects"),
                      (self.panel_config, "Import Files"), 
                      (self.panel_specify_voting_targets, "Select and Group Targets"), 
                      (self.panel_define_attrs, "Define Ballot Attributes"),
                      (self.panel_label_attrs, "Label Ballot Attributes"),
                      (self.panel_label_digitattrs, "Label Digit-Based Attributes"),
                      (self.panel_label_contests, "Label Contests"),
                      (self.panel_correct_grouping, "Correct Grouping"),
                      (self.panel_run, "Run"),
                      (self.panel_set_threshold, "Set Threshold"),
                      (self.panel_quarantine, "Process Quarantine"),
                      (self.panel_process, "Process")]

        for panel, text in self.pages[1:]:
            notebook.AddPage(panel, text)

        imglist = wx.ImageList(16, 16)
        self.notebook.imglist = imglist
        self.notebook.AssignImageList(imglist)

        # tabidx_map maps page index to the page's index into the
        # ImageList. They should be the same anyways, but let's generalize
        self.notebook.tabidx_map = {}
        for i in range(self.notebook.GetPageCount()):
            blankbmp = wx.EmptyBitmap(16, 16)
            dc = wx.MemoryDC(blankbmp)
            dc.SetBackground(wx.Brush(self.notebook.GetBackgroundColour(), style=wx.TRANSPARENT))
            dc.Clear()
            dc.SelectObject(wx.NullBitmap)
            idx = imglist.Add(blankbmp)
            self.notebook.tabidx_map[i] = idx
            self.notebook.SetPageImage(i, idx)

        self.panel_projects.start()

    def _pubsub_rundone(self, msg):
        TIMER.stop_task(('cpu', self.map_pages[self.RUN]['cpu']))

    def _pubsub_blankballot_done(self, msg):
        """
        Triggered when the background process is done straightening all
        blank ballots. 
        """
        print 'Blank ballots have been straightened.'
        self.project.are_blankballots_straightened = True
        self.Enable()
        self.panel_specify_voting_targets.start()
        self.panel_specify_voting_targets.SendSizeEvent()
        self.SendSizeEvent()

    def _pubsub_votedballot_done(self, msg):
        print "Voted ballots have been straightened."
        self.project.are_votedballots_straightened = True
        if self.waiting_for_votedstraights:
            print "== Since UI was waiting for voted ballots to be \
straightened, I'm now unlocking the UI."
            self.Enable()
            self.waiting_for_votedstraights = False
            if self.get_num_template_ballots() != 1:
                self.panel_correct_grouping.start()
                self.panel_correct_grouping.SendSizeEvent()
            self.SendSizeEvent()
        
    def spawn_straightener_jobs(self):
        """
        Spawn jobs that straighten blank/voted ballots if necessary.
        """
        # First update self.project.imgsize, which contains the 
        # 'max' dimensions of all blank ballots
        if not self.project.imgsize:
            self.project.imgsize, num = get_max_dimensions(self.project.raw_templatesdir)
            Publisher().sendMessage("broadcast.projupdate")
        else:
            imgsize, num = get_max_dimensions(self.project.raw_templatesdir)
            if (self.project.imgsize != imgsize):
                print 'Project imgsize was {0}, but newly computed \
imgsize was {1}. Oh well, using new imgsize'.format(self.project.imgsize,
                                                imgsize)
            self.project.imgsize = imgsize
            Publisher().sendMessage("broadcast.projupdate")

        if not self.project.are_blankballots_straightened:
            self.num_templates = num
            outdir = pathjoin(self.project.projdir_path, 'blankballots_straight')
            self.project.templatesdir = outdir
            Publisher().sendMessage("broadcast.projupdate")
            thread = straighten_ballots.StraightenThread(self.project.raw_templatesdir, 
                                                         self.project,
                                                         self.num_templates,
                                                         outdir=outdir,
                                                         job_id=straighten_ballots.BLANKBALLOT_JOB_ID)
            def on_done():
                wx.CallAfter(Publisher().sendMessage, "straighten_blankballot_done")
            gauge = util.MyGauge(self, 1, thread=thread, ondone=on_done,
                                 msg="Straightening blank ballots...",
                                 job_id=straighten_ballots.BLANKBALLOT_JOB_ID)
            self.Disable()
            thread.start()
            gauge.Show()        
        if not self.project.are_votedballots_straightened:
            self.num_samples = util_gui.count_images(self.project.raw_samplesdir)
            outdir = pathjoin(self.project.projdir_path, 'votedballots_straight')
            self.project.samplesdir = outdir
            Publisher().sendMessage("broadcast.projupdate")
            def on_done():
                wx.CallAfter(Publisher().sendMessage, "straighten_votedballot_done")
            thread = straighten_ballots.StraightenThread(self.project.raw_samplesdir,
                                                         self.project,
                                                         self.num_samples,
                                                         outdir=outdir,
                                                         job_id=straighten_ballots.VOTEDBALLOT_JOB_ID)
            gauge = util.MyGauge(self, 1, thread=thread, ondone=on_done,
                                 msg="Straightening voted ballots...",
                                 job_id=straighten_ballots.VOTEDBALLOT_JOB_ID)
            thread.start()
            gauge.Show()
            
    def enable_tab_flare(self, tab_idx):
        """
        Add some 'flare' to a given tab, to indicate to the user to
        proceed to this tab.
        """
        self._pageidx = tab_idx
        self._running = True
        self.on_timer(None)

    def disable_tab_flare(self, tab_idx):
        """
        Remove some 'flare' for a given tab.
        """
        self._running = False
        bmp = wx.EmptyBitmap(16, 16)
        dc = wx.MemoryDC(bmp)
        dc.SetBackground(wx.Brush(self.notebook.GetBackgroundColour()))
        dc.Clear()
        dc.SelectObject(wx.NullBitmap)
        self.notebook.imglist.Replace(self.notebook.tabidx_map[tab_idx], bmp)
        self.notebook.SetPageImage(tab_idx, self.notebook.tabidx_map[tab_idx])

    def checkmark_tab(self, tab_idx):
        """
        Change a tab's icon to be a checkmark (i.e. 'completed')
        """
        self._running = False
        bmp = wx.EmptyBitmap(16, 16)
        dc = wx.MemoryDC(bmp)
        dc.SetBrush(wx.Brush("Green"))
        dc.SetPen(wx.Pen("Green", 1))
        dc.SetBackground(wx.Brush(self.notebook.GetBackgroundColour(), style=wx.TRANSPARENT))
        dc.Clear()
        dc.DrawCheckMark(0, 0, 14, 14)
        dc.SelectObject(wx.NullBitmap)
        idx = self.notebook.imglist.Add(bmp)
        self.notebook.SetPageImage(tab_idx, idx)

    def get_current_tab(self):
        """
        Returns which tab is currently displayed.
        Output:
            An int (0 indexed), whose value represents which tab is
            currently displayed. A tab's value is decided by the order
            in MainFrame.__init__
        """
        return self.notebook.GetSelection()
        
    def get_num_templates(self):
        """
        Return number of templates in this election. Assumes that the
        current project has a templates directory set already. If not,
        then this returns -1.
        Note that this returns the number of template images, instead of
        template Ballots, i.e. for a multipage election with only one
        template, this will still return 2.
        """
        if self.num_templates == -1:
            self.num_templates = 0
            if self.project.raw_templatesdir == '':
                return -1
            for dirpath, dirnames, filenames in os.walk(self.project.raw_templatesdir):
                self.num_templates += len([f for f in filenames if util_gui.is_image_ext(f)])
        return self.num_templates

    def get_num_template_ballots(self):
        """
        Return number of template Ballots. If this isn't a multipage
        election, this should return the same thing as get_num_templates.
        """
        if not util.is_multipage(self.project):
            return self.get_num_templates()
        else:
            template_to_images = pickle.load(open(self.project.template_to_images, 'rb'))
            return len(template_to_images)

    def _pubsub_can_proceed(self, msg):
        """
        Triggered when some component realizes that the user can
        proceed to the next step.
        """
        if self.get_current_tab() + 1 < len(self.pages):  
            self.enable_tab_flare(self.get_current_tab() + 1)

    def _pubsub_cant_proceed(self, msg):
        """
        Triggered when some component realizes that the user
        can't proceed anymore.
        """
        if self.get_current_tab() + 1 < len(self.pages):  
            self.disable_tab_flare(self.get_current_tab() + 1)

    def _pubsub_update_tab_icon(self, msg):
        tab_idx, bmp = msg.data
        idx = self.notebook.imglist.Add(bmp)
        self.notebook.SetPageImage(tab_idx, idx)

    #### Event Handlers
    def on_timer(self, evt):
        """
        Triggered when the timer goes off, signals the UI to
        update the 'glowing' icon on the notebook tab.
        """
        if self._running:
            if self._coloridx >= len(self._colors):
                self._coloridx = 0
            bmp = wx.EmptyBitmap(16, 16)
            dc = wx.MemoryDC(bmp)
            dc.SetBackground(wx.Brush(self.notebook.GetBackgroundColour()))
            dc.Clear()
            dc.SetPen(wx.Pen("Black", 1))
            dc.SetBrush(wx.Brush(self._colors[self._coloridx]))
            self._coloridx += 1
            dc.DrawCircle(8, 8, 6)
            dc.SelectObject(wx.NullBitmap)
            self.notebook.imglist.Replace(self._pageidx, bmp)
            self.notebook.SetPageImage(self._pageidx, self.notebook.tabidx_map[self._pageidx])
            self.timer.Start(400, oneShot=True)

    def onClose(self, evt):
        """
        Triggered when the user/program exits/closes the MainFrame.
        """
        if self.project:
            # If self.project is None, then this means that the user
            # hasn't selected a project yet, which is OK.
            write_projconfig(self.project, close=True)
        try:
            TIMER.stop_task('Total Time')
            TIMER.dump()
        except:
            pass
        if self.notebook.GetCurrentPage() == self.panel_define_attrs:
            self.panel_define_attrs.stop()
        self.panel_correct_grouping.dump_state()
        wx.Frame.Destroy(self)

    def onMenu_about(self, evt):
        print "ABOUT"

    def onMenu_advopts(self, evt):
        if not self.project:
            dlg = wx.MessageDialog(self, message="A project must be open \
and active in order to access this.",
                                style=wx.OK)
            self.Disable()
            dlg.ShowModal()
            self.Enable()
        else:
            dlg = AdvancedOptionsDialog(self, self.project)
            self.Disable()
            dlg.ShowModal()
            self.Enable()

    def emit_undo(self, evt):
        """
        Triggered when the user hits 'Ctrl-z' - emit an 'undo' signal
        that 'Select and Group Targets' will use.
        """
        Publisher().sendMessage("broadcast.undo", self.get_current_tab())
    
    def onPageChange(self, evt):
        map_pages = MainFrame.map_pages
        # Save the project data in case something happens.

        old = evt.GetOldSelection()
        new = evt.GetSelection()

        if self.project:
            write_projconfig(self.project, old=old, new=new)

        if old == -1:
            # Skip the initial setup
            return
        
        if new > old+1:
            if not self.options.devmode:
                dlg = wx.MessageDialog(self, message="You can't skip past sections in the workflow.", style=wx.OK)
                dlg.ShowModal()
                self.notebook.ChangeSelection(old)                

        if new < old:
            for i in range(new+1, old+1):
                try:
                    # HACK SPECIAL CASE
                    if i == self.LABEL_CONTESTS:
                        if new == self.SELECT_TARGETS:
                            self.pages[i][0].reset_data()
                    else:
                        self.pages[i][0].reset_data()
                except:
                    pass
                try:
                    self.pages[i][0].reset_panel()
                except:
                    pass
            if self.project:
                Publisher().sendMessage("broadcast.project", self.project)

        if old == self.CONFIG and new >= self.SELECT_TARGETS:
            # Make sure the user entered in the templates/samples dir
            templatesdir = self.panel_config.get_templatepath()
            samplesdir = self.panel_config.get_samplepath()
            if not templatesdir.strip() and not samplesdir.strip():
                msg = "Please enter the templates and samples directories."
            elif not templatesdir.strip():
                msg = "Please enter the templates directory."
            elif not samplesdir.strip():
                msg = "Please enter the samples directory"
            elif not os.path.exists(templatesdir) and not os.path.exists(samplesdir):
                msg = "Folders {0} and {1} weren't found.".format(templatesdir, samplesdir)
            elif not os.path.exists(templatesdir):
                msg = "Templates folder {0} wasn't found.".format(templatesdir)
            elif not os.path.exists(samplesdir):
                msg = "Samples folder {0} wasn't found.".format(samplesdir)
            else:
                msg = ''
            if msg:
                dlg = wx.MessageDialog(self, 
                                       message=msg, 
                                       style=wx.OK)
                dlg.ShowModal()
                self.notebook.ChangeSelection(self.CONFIG)
                return
            else:
                TIMER.stop_task(('user', map_pages[self.CONFIG]['user']))
            # Start background processes for blank ballot and voted
            # ballot straightening
            if not os.path.exists(self.project.ballot_to_images):
                self.panel_config.initDoubleSided()
            self.spawn_straightener_jobs()
            
        if old == self.PROJECTS:
            # Leaving the Projects tab. Announce the selected Project.
            project = self.panel_projects.get_selected_project()
            if not project:
                dlg = wx.MessageDialog(self, message="Please select a Project.", style=wx.OK)
                dlg.ShowModal()
                self.notebook.ChangeSelection(self.PROJECTS)
                return
            else:
                self.project = project
                self.project.options = self.options
                Publisher().sendMessage("broadcast.project", project)
            # Also inform Timer object
            set_timer(project)
            self.panel_specify_voting_targets.set_timer(TIMER)
            self.panel_correct_grouping.set_timer(TIMER)
            self.panel_run.set_timer(TIMER)
            TIMER.start_task('Total Time')
            TIMER.write_msg("Project name is: {0}".format(project.name))
        elif old == self.SELECT_TARGETS:
            # Write out voting/contest locations to a .csv file. Also
            # broadcast it to anybody that's listening.
            if not self.panel_specify_voting_targets.validate_outputs():
                self.notebook.ChangeSelection(self.SELECT_TARGETS)
                return
            TIMER.stop_task(('user', map_pages[self.SELECT_TARGETS]['user']))
            self.panel_specify_voting_targets.stop()
            box_locations = self.panel_specify_voting_targets.export_bounding_boxes()
            Publisher().sendMessage("broadcast.box_locations", box_locations)
            Publisher().sendMessage("broadcast.tempmatchdone", None)
            if util.is_multipage(self.project):
                # Also, if this is multipage, update the template_to_images
                # ordering to reflect {str ballot_id: [str frontpath, str backpath]}
                frontback_map = self.panel_specify_voting_targets.frontback_map
                image_to_template = pickle.load(open(self.project.image_to_template, 'rb'))
                template_to_images = pickle.load(open(self.project.template_to_images, 'rb'))
                for temppath, side in frontback_map.iteritems():
                    if temppath not in image_to_template:
                        print "Couldn't find temppath in template_to_images:", temppath
                        pdb.set_trace()
                    id = image_to_template[temppath]
                    path1, path2 = template_to_images[id]
                    # Absolute-Normalize all paths to avoid comparison 
                    # hiccups
                    path1 = os.path.abspath(path1)
                    path2 = os.path.abspath(path2)
                    temppath = os.path.abspath(temppath)
                    new_tuple = []
                    if side == 'front':
                        if temppath == path1:
                            new_tuple = [path1, path2]
                        else:
                            new_tuple = [path2, path1]
                    else:
                        if temppath == path1:
                            new_tuple = [path2, path1]
                        else:
                            new_tuple = [path1, path2]
                    template_to_images[id] = new_tuple
                pickle.dump(template_to_images, open(self.project.template_to_images, 'wb'))
                
        elif old == self.LABEL_CONTESTS:
            TIMER.stop_task(('user', map_pages[self.LABEL_CONTESTS]['user']))
            if (not self.panel_label_contests.canMoveOn) and new > old:
                dlg = wx.MessageDialog(self,
                                       message="Can't move along yet.", 
                                       style=wx.OK)
                dlg.ShowModal()
                self.notebook.ChangeSelection(self.LABEL_CONTESTS)
                return
        elif old == self.DEFINE_ATTRIBUTES:
            self.panel_define_attrs.stop()
            self.panel_define_attrs.export_attribute_patches()
        elif old == self.LABEL_ATTRS:
            if not self.panel_label_attrs.validate_outputs():
                self.notebook.ChangeSelection(self.LABEL_ATTRS)
                return
            TIMER.stop_task(('user', map_pages[self.LABEL_ATTRS]['user']))
            self.panel_label_attrs.stop()
            self.panel_label_attrs.export_bounding_boxes()
            if (not self.panel_label_attrs.checkCanMoveOn()) and new > old:
                dlg = wx.MessageDialog(self,
                                       message="Can't move along yet.", 
                                       style=wx.OK)
                dlg.ShowModal()
                self.notebook.ChangeSelection(self.LABEL_ATTRS)
                return            
        elif old == self.LABEL_DIGIT_ATTRS:
            # Should sanity-check the results
            pass
        elif old == self.CORRECT_GROUPING:
            TIMER.stop_task(('user', map_pages[self.CORRECT_GROUPING]['user']))
            self.panel_correct_grouping.exportResults()
            if (not self.panel_correct_grouping.checkCanMoveOn()) and new > old:
                dlg = wx.MessageDialog(self,
                                       message="Can't move along yet.", 
                                       style=wx.OK)
                dlg.ShowModal()
                self.notebook.ChangeSelection(self.CORRECT_GROUPING)
                return
        elif old == self.SET_THRESHOLD:
            TIMER.stop_task(('user', map_pages[self.SET_THRESHOLD]['user']))
        elif old == self.QUARANTINE:
            pass

        if new == self.PROJECTS:
            self.panel_label_contests.reset_panel()
        elif new == self.CONFIG:
            TIMER.start_task(('user', map_pages[self.CONFIG]['user']))
        elif new == self.SELECT_TARGETS:
            # Select and Group Targets
            # The below block is in the on_blankstraighten_done function
            #self.panel_specify_voting_targets.start()
            #self.panel_specify_voting_targets.SendSizeEvent()
            #self.SendSizeEvent()
            #TIMER.start_task(('user', map_pages[self.SELECT_TARGETS]['user']))
            if self.project.are_blankballots_straightened:
                self.panel_specify_voting_targets.start()
                self.panel_specify_voting_targets.SendSizeEvent()
                TIMER.start_task(('user', map_pages[self.SELECT_TARGETS]['user']))
            else:
                print "== Must wait for blank ballots to be finished straightening."
        elif new == self.LABEL_CONTESTS:
            # Label Contests
            self.panel_label_contests.start(self.GetSize())
            self.SendSizeEvent()
            TIMER.start_task(('user', map_pages[self.LABEL_CONTESTS]['user']))
        elif new == self.PROCESS:
            self.panel_process.set_results()
            self.SendSizeEvent()
        elif new == self.DEFINE_ATTRIBUTES:
            if self.get_num_template_ballots() == 1:
                msg = "The step 'Define Ballot Attributes' is unnecessary \
because the current election has only one template. Skipping ahead to 'Run'."
                dlg = wx.MessageDialog(self, message=msg, style=wx.OK)
                self.Disable()
                dlg.ShowModal()
                self.Enable()
                old_page = self.notebook.GetSelection()
                self.notebook.ChangeSelection(self.RUN)
                self.notebook.SendPageChangedEvent(old_page, self.RUN)
            else:
                self.panel_define_attrs.start()
                self.panel_define_attrs.Show()
                self.panel_define_attrs.SendSizeEvent()
                self.SendSizeEvent()
        elif new == self.LABEL_ATTRS:
            def start_labelattrs(groupresults):
                """ Invoked when user has finished verifying the
                attr grouping. groupresults is a dict:
                    {grouplabel: list of GroupClass objects}
                """
                self.panel_label_attrs.start(self.GetSize())
                self.panel_label_attrs.start()
                self.panel_label_attrs.set_attrgroup_results(groupresults)
                self.panel_label_attrs.SendSizeEvent()
                self.SendSizeEvent()
                TIMER.start_task(('user', map_pages[self.LABEL_ATTRS]['user']))

            if self.get_num_template_ballots() == 1:
                msg = "The step 'Specify Precinct Paches' (along with \
'Correct Grouping') is unnecessary because the current election only has \
one template. \nSkipping ahead to 'Run'."
                dlg = wx.MessageDialog(self, message=msg, style=wx.OK)
                dlg.ShowModal()
                self.notebook.ChangeSelection(self.RUN)
                self.notebook.SendPageChangedEvent(self.LABEL_ATTRS, self.RUN)
                return
            elif True:
                f = GroupAttrsFrame(self, self.project, start_labelattrs)
                f.Show()
                f.Maximize()
            else:
                start_labelattrs(None)
        elif new == self.LABEL_DIGIT_ATTRS:
            def is_any_digitspatches(project):
                all_attrtypes = pickle.load(open(self.project.ballot_attributesfile, 'rb'))
                for attrbox_dict in all_attrtypes:
                    if attrbox_dict['is_digitbased']:
                        return True
                return False
            if is_any_digitspatches(self.project):
                self.panel_label_digitattrs.start(self.project)
            else:
                msg = "There are no digit-based attribute patches, \
so this step is unnecessary. Skipping ahead."
                dlg = wx.MessageDialog(self, message=msg, style=wx.OK)
                dlg.ShowModal()
                self.notebook.ChangeSelection(self.LABEL_CONTESTS)
                self.notebook.SendPageChangedEvent(self.LABEL_DIGIT_ATTRS, self.LABEL_CONTESTS)
                self.SendSizeEvent()
                return
            self.SendSizeEvent()
        elif new == self.CORRECT_GROUPING:
            # Note: This panel includes both the 'Run Grouping'
            # CPU computation, in addition to the 'Verify Grouping'
            # UI.
            if self.get_num_template_ballots() == 1:
                msg = "The step 'Correct Grouping' (along with \
'Specify Precinct Patches') is unnecessary because the current election \
only has one template. \nSkipping ahead to 'Run'."
                dlg = wx.MessageDialog(self, message=msg, style=wx.OK)
                dlg.ShowModal()
                self.notebook.ChangeSelection(self.RUN)
                self.notebook.SendPageChangedEvent(self.CORRECT_GROUPING, self.RUN)
                return
            else:
                # Wait until voted ballots have been straightened
                if not self.project.are_votedballots_straightened:
                    print "== Voted ballots aren't straightened yet, must \
wait."
                    self.waiting_for_votedstraights = True
                    dlg = wx.MessageDialog(self, message="Please wait for \
voted ballots to be straightened.", style=wx.OK)
                    self.Disable()
                    dlg.ShowModal()
                else:
                    print "== Voted ballots are already straightened, can \
proceed normally."
                    self.panel_correct_grouping.start()
                    self.panel_correct_grouping.SendSizeEvent()
                    self.SendSizeEvent()
        elif new == self.RUN:
            # Run
            if self.get_num_template_ballots() == 1:
                if self.project.are_votedballots_straightened:
                    print "== Voted ballots are already straightened, can \
proceed normally."
                else:
                    print "== Voted ballots aren't straightened yet, must \
wait."
                    self.waiting_for_votedstraights = True
                    dlg = wx.MessageDialog(self, message="Please wait for \
voted ballots to be straightened.", style=wx.OK)
                    self.Disable()
                    dlg.ShowModal()
            self.SendSizeEvent()
        elif new == self.SET_THRESHOLD:
            # Set Threshold
            self.panel_set_threshold.start(self.GetSize())
            self.SendSizeEvent()
            TIMER.start_task(('user', map_pages[self.SET_THRESHOLD]['user']))
        elif new == self.QUARANTINE:
            self.panel_quarantine.start()
        
        # If we get here, the event is valid to go through.
        self.disable_tab_flare(new)
        for x in range(new):
            self.checkmark_tab(x)
        
class AdvancedOptionsDialog(wx.Dialog):
    def __init__(self, parent, project, *args, **kwargs):
        wx.Dialog.__init__(self, parent, title='Advanced Options', *args, **kwargs)
        self.parent = parent
        self.project = project
        # A list of validator functions, no args
        self.validators = self._populate_validators()
        box1 = wx.StaticBox(self, label="Select and Group Targets", 
                            size=(self.GetClientSize()[0]-10, -1))
        box1.sizer = wx.StaticBoxSizer(box1, orient=wx.VERTICAL)
        box1.txt = wx.StaticText(self, label="Sensitivity for \
auto-detecting voting targets.")
        panel = wx.Panel(self)
        panel.sizer = wx.BoxSizer(wx.HORIZONTAL)
        tempmatch_param = int(round(float(self.project.tempmatch_param) * 100))
        self.slider = wx.Slider(panel, size=(200, 50), value=tempmatch_param)
        self.slider.Bind(wx.EVT_SCROLL, self.sliderscroll, self.slider)
        self.slidertxtctrl = wx.TextCtrl(panel, value=str(tempmatch_param))
        self.slidertxtctrl.Bind(wx.EVT_TEXT, self.slidertxtctrl_change, self.slidertxtctrl)
        panel.sizer.Add(self.slider)
        panel.sizer.Add((10,10))
        panel.sizer.Add(self.slidertxtctrl)
        panel.SetSizer(panel.sizer)

        box1.sizer.Add((10,10))
        box1.sizer.Add(box1.txt)
        box1.sizer.Add(panel)

        panelbtns = wx.Panel(self)
        panelbtns.sizer = wx.BoxSizer(wx.HORIZONTAL)
        panelbtns.SetSizer(panelbtns.sizer)
        btn_ok = wx.Button(panelbtns, label="Ok", id=wx.ID_OK)
        btn_ok.Bind(wx.EVT_BUTTON, self.onbutton_ok)
        btn_cancel = wx.Button(panelbtns, id=wx.ID_CANCEL, label="Cancel")
        panelbtns.sizer.Add(btn_ok)
        panelbtns.sizer.Add((20,10))
        panelbtns.sizer.Add(btn_cancel)

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add((10,10))
        self.sizer.Add(box1.sizer)
        self.sizer.Add((10,40))
        self.sizer.Add(panelbtns)
        self.sizer.Add((10,10))
        self.SetSizer(self.sizer)

    def _populate_validators(self):
        """
        A validator function is a function of no arguments that sanity
        checks necessary user input. If the sanity check passes, 
        return True. 
        If it doesn't pass, then do any necessary UI changes (say,
        SetFocus() to the wrong widget in question) if desired, and 
        then return False.
        """
        return (self._validate_sliderval,)

    def sliderscroll(self, evt):
        slider = evt.GetEventObject()
        self.slidertxtctrl.ChangeValue(str(slider.GetValue()))
    def slidertxtctrl_change(self, evt):
        if not self.slidertxtctrl.GetValue():
            return
        if self._validate_sliderval():
            self.slider.SetValue(int(self.slidertxtctrl.GetValue()))

    def _validate_sliderval(self):
        """
        Ensures that the value of the confidence parameter (given
        by the slider/slidertextctrl value) is an integer in between
        0 and 100.
        """
        def isnum(val):
            try:
                return 0 <= int(val) <= 100
            except:
                return False
        val = self.slidertxtctrl.GetValue()
        if not val or not isnum(val):
            dlg = wx.MessageDialog(self, message="Please use an \
integer in between 1 and 100.",
                                   style=wx.OK)
            self.Disable()
            dlg.ShowModal()
            self.Enable()
            self.slidertxtctrl.SetFocus()
            return False
        else:
            return True

    def onbutton_ok(self, evt):
        """
        Re-apply validators to appropriate widgets, then if all passes,
        write out changes to project as necessary.
        """
        canexit = True
        for validator in self.validators:
            if not validator():
                canexit = False
        if canexit:
            self.project.tempmatch_param = self.slider.GetValue() / 100.0
            Publisher().sendMessage("broadcast.projupdate")
            self.Close()

class Project(object):
    """
    A Project is represented in the filesystem as a folder in the
    projects/ directory, where the name of the folder denotes the
    project name. The directory structrue is:
        projects/ProjectName/project.config
            An XML-like file used to store textual information.
        projects/ProjectName/target_locations/*
            Contains the .csv files for each template image, which
            contains information on voting target locations.
        Insert your own stuff here...

    """
    closehook = []

    def __init__(self, name='', projdir_path=''):
        self.vals = {'name': name,
                     'projdir_path': projdir_path,
                     'config_path': pathjoin(projdir_path, 'project.config'),
                     'is_multipage': False,
                     'ocr_tmp_dir': pathjoin(projdir_path, 'ocr_tmp_dir'),
                     'contest_id': pathjoin(projdir_path, 'contest_id.csv'),
                     'contest_text': pathjoin(projdir_path, 'contest_text.csv'),
                     'contest_internal': pathjoin(projdir_path, 'contest_internal.p'),
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
                     'templatesdir': '',
                     'samplesdir': '',
                     'raw_templatesdir': '',
                     'raw_samplesdir': '',                     
                     'ballot_to_images': pathjoin(projdir_path, 'ballot_to_images.p'),
                     'template_to_images': pathjoin(projdir_path, 'template_to_images.p'),
                     'image_to_ballot': pathjoin(projdir_path, 'image_to_ballot.p'),
                     'image_to_template': pathjoin(projdir_path, 'image_to_template.p'),
                     'election_results': pathjoin(projdir_path, 'election_results.txt'),
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
                     'tempmatch_param': str(find_targets_wizard.SpecifyTargetsPanel.TEMPMATCH_DEFAULT_PARAM),
                     'ballot_attributesfile': pathjoin(projdir_path, 'ballot_attributes.p'),
                     'imgsize': (0,0),
                     'votedballots_straightdir': pathjoin(projdir_path, 'votedballots_straight'),
                     'blankballots_straightdir': pathjoin(projdir_path, 'blankballots_straight'),
                     'are_blankballots_straightened': False,
                     'are_votedballots_straightened': False,
                     'frontback_map': pathjoin(projdir_path, 'frontback_map.p'),
                     'extracted_digitpatch_dir': 'extracted_digitpatches',
                     'digit_exemplars_outdir': 'digit_exemplars',
                     'precinctnums_outpath': 'precinctnums.txt',
                     'num_digitsmap': 'num_digitsmap.p',
                     'digitgroup_results': 'digitgroup_results.p',
                     'voteddigits_dir': 'voteddigits_dir',
                     'tmp2digitpatch': 'tmp2digitpatch.p'}
        self.createFields()

    def addCloseEvent(self, func):
        Project.closehook.append(func)

    def removeCloseEvent(self, func):
        Project.closehook = [x for x in Project.closehook if x != func]

    def createFields(self):
        for k,v in self.vals.items():
            setattr(self, k, v)

    def __repr__(self):
        return 'Project({0})'.format(self.name)

def get_projects(dir_projects):
    """
    Given the base directory of all stored project configurations,
    return a list of all projects.
    Assumes the following directory structure:
    dir_projects/leon_election/project.config
    dir_projects/leon_election/<step>/*
    """
    projects = []

    dirpath, dirnames, filenames = next(os.walk(dir_projects))
    for proj_name in dirnames:
        projdir_path = pathjoin(dirpath, proj_name)
        proj_config = pathjoin(projdir_path, 'project.config')
        if os.path.exists(proj_config):
            project = read_projconfig(proj_config)
            if project:
                projects.append(project)
            else:
                print "Warning: Project {0} was unable to be read.".format(proj_name)
    return projects

def create_projconfig(projname, projpath):
    """
    Create a blank config xml file for a new project.
    """
    root = Element("config")
    dummyProject = Project(projname, projpath)
    for each in dummyProject.vals.keys():
        obj = Element(each)
        # Then fill in with default value.
        obj.text = str(getattr(dummyProject, each))
        root.append(obj)
    
    tree = ElementTree(root)
    tree.write(pathjoin(projpath, 'project.config'))

def read_projconfig(configpath):
    """
    Given the path to a project config file, 'parse' it and return a
    Project object that represents the config.
    """
    tree = ElementTree()
    try:
        tree.parse(configpath)
    except:
        # If the parse fails (i.e. if the config file is an empty file),
        # handle it, or face a segfault.
        return None

    project = Project()
    
    for each in project.vals.keys():
        obj = tree.find(each)
        if obj != None and obj.text:
            if type(getattr(project, each)) == type(0):
                val = int(obj.text)
            elif type(getattr(project, each)) == type(True):
                val = obj.text == 'True'
            elif type(getattr(project, each)) == type(tuple()):
                val = eval(obj.text) # HACK FIXME
            else:
                val = obj.text
            setattr(project, each, val)
    
    return project

def write_projconfig(project, close=False, old=None, new=None):
    """
    Write out the current state of the project into the xml file
    project.config
    """
    for each in Project.closehook:
        each()

    #if old != None:
    #    sanitycheck.run(project, old)

    tree = ElementTree()
    tree.parse(project.config_path)

    for each in project.vals.keys():
        obj = tree.find(each)
        if obj == None:
            #print "ERROR:", each, "does not correspond to a node."
            #print "But we'll add it anyway, since each comes from project.vals.keys()"
            print "Adding new attribute to this project config:", each
            obj = Element(each)
            obj.text = getattr(project, each)
            tree.getroot().append(obj)
        else:
            obj.text = str(getattr(project, each))

    tree.write(project.config_path)

def is_step_finished(project, stepnum):
    """
    Return True iff the project state is 'done' with stepnum.
    """
    if stepnum == MainFrame.PROJECTS:
        # Must select a project. Trivially True, if we assume
        # that the caller has already set project.
        return True
    elif stepnum == MainFrame.CONFIG:
        # Must have valid templatedir/sampledir
        return (project.raw_templatesdir != '' 
                and project.raw_samplesdir != ''
                and os.path.exists(project.raw_templatesdir)
                and os.path.exists(project.raw_samplesdir))
    elif stepnum == MainFrame.SELECT_TARGETS:
        # Must have at least one voting target selected, and
        # all contests must have more than one targyet.
        pass

def get_max_dimensions(imgsdir):
    """
    Given a directory of images, return the biggest width and the
    biggest height.
    Also return the number of images.
    """
    w, h = 0, 0
    counter = 0
    for dirpath, dirnames, filenames in os.walk(imgsdir):
        for imgname in [f for f in filenames if util_gui.is_image_ext(f)]:
            imgpath = pathjoin(dirpath, imgname)
            img = cv.LoadImage(imgpath)
            imgsize = img.width, img.height
            w_img, h_img = imgsize[0], imgsize[1]

            w = max(w, w_img)
            h = max(h, h_img)
            counter += 1
    return (w, h), counter

def is_any_digitattrs(project):
    """ Returns True if any attribute is a digits patch """
    for dirpath, dirnames, filenames in os.walk(project.patch_loc_dir):
        for filename in [f for f in filenames if f.lower().endswith('.csv')]:
            csvfile = open(pathjoin(dirpath, filename), 'r')
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['is_digitbased'] == 'True':
                    return True
            csvfile.close()
    return False

def is_valid_projectname(name):
    """
    Only allow letters, numbers, and [', ", _, (, )].
    """
    pattern = r'(\w|\d|[_\'"()])+'
    return not re.match(pattern, name) == None

def set_timer(project):
    global TIMER
    TIMER = util.MyTimer(pathjoin(project.projdir_path, TIMING_FILENAME))
    TIMER.prelude()

def make_optionparser():
    usage = "usage: python maingui.py [--dev]"
    parser = optparse.OptionParser(usage=usage)
    parser.add_option("--dev", dest='devmode', action="store_true",
                      help="Enter developer mode. Enables several features which may break projects if not used correctly.")
    return parser

def main():
    logger = multiprocessing.log_to_stderr()
    logger.setLevel(logging.INFO)

    parser = make_optionparser()
    options, args = parser.parse_args()
    options.devmode = True

    app = wx.App(False)
    frame = MainFrame(options=options)
    #wx.lib.inspection.InspectionTool().Show()
    app.MainLoop()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        try:
            TIMER.dump()
        except:
            pass
        
