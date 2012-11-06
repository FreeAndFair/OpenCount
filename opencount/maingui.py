import os, sys, time, pdb, traceback
try:
    import cPickle as pickle
except ImportError as e:
    import pickle

from os.path import join as pathjoin

import wx
from wx.lib.scrolledpanel import ScrolledPanel

sys.path.append('..')
from tab_wrap import tab_wrap
from projconfig_new.ProjectPanel import ProjectPanel, Project
from projconfig_new.ConfigPanel import ConfigPanel
from partitions.PartitionPanel import PartitionMainPanel
from specify_voting_targets.select_targets import SelectTargetsMainPanel
from labelcontest.labelcontest import LabelContest
from grouping.define_attributes_new import DefineAttributesMainPanel
from grouping.select_attributes import SelectAttributesMasterPanel
from digits_ui.digits_ui import LabelDigitsPanel
from grouping.RunGrouping import RunGroupingMainPanel
from grouping.VerifyGroupingPanel import VerifyGroupingMainPanel
from runtargets.extract_targets_new import TargetExtractPanel
from threshold.threshold import ThresholdPanel
from quarantine.quarantinepanel import QuarantinePanel
from post_processing.postprocess import ResultsPanel

PROJROOTDIR = 'projects_new'

class MainFrame(wx.Frame):
    PROJECT = 0
    CONFIG = 1
    PARTITION = 2
    DEFINE_ATTRS = 3
    LABEL_ATTRS = 4
    LABEL_DIGATTRS = 5
    RUN_GROUPING = 6
    CORRECT_GROUPING = 7
    SELTARGETS = 8
    LABEL_CONTESTS = 9
    TARGET_EXTRACT = 10
    SET_THRESHOLD = 11
    QUARANTINE = 12
    PROCESS = 13

    def __init__(self, parent, *args, **kwargs):
        wx.Frame.__init__(self, parent, title="OpenCount", *args, **kwargs)
        
        # PROJECT: Current Project being worked on.
        self.project = None

        self.init_ui()
        
        self.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.onPageChange)
        self.Bind(wx.EVT_CLOSE, self.onClose)

        self.notebook.ChangeSelection(0)
        self.notebook.SendPageChangedEvent(-1, 0)

    def init_ui(self):
        self.notebook = wx.Notebook(self)
        self.setup_pages()

    def setup_pages(self):
        self.panel_projects = ProjectPanel(self.notebook)
        self.panel_config = ConfigPanel(self.notebook)
        self.panel_partition = PartitionMainPanel(self.notebook)
        self.panel_define_attrs = DefineAttributesMainPanel(self.notebook)
        self.panel_label_attrs = SelectAttributesMasterPanel(self.notebook)
        self.panel_label_digitattrs = LabelDigitsPanel(self.notebook)
        self.panel_run_grouping = RunGroupingMainPanel(self.notebook)
        self.panel_correct_grouping = VerifyGroupingMainPanel(self.notebook)
        self.panel_seltargets = SelectTargetsMainPanel(self.notebook)
        self.panel_label_contests = LabelContest(self.notebook, self.GetSize())
        self.panel_target_extract = TargetExtractPanel(self.notebook)
        self.panel_set_threshold = ThresholdPanel(self.notebook, self.GetSize())
        self.panel_quarantine = QuarantinePanel(self.notebook)
        self.panel_process = ResultsPanel(self.notebook)
        self.pages = [(self.panel_projects, "Projects"),
                      (self.panel_config, "Import Files"), 
                      (self.panel_partition, "Partition ballots"),
                      (self.panel_define_attrs, "Define Ballot Attributes"),
                      (self.panel_label_attrs, "Label Ballot Attributes"),
                      (self.panel_label_digitattrs, "Label Digit-Based Attributes"),
                      (self.panel_run_grouping, "Run Grouping"),
                      (self.panel_correct_grouping, "Correct Grouping"),
                      (self.panel_seltargets, "Select Voting Targets"),
                      (self.panel_label_contests, "Label Contests"),
                      (self.panel_target_extract, "Extract Targets"),
                      (self.panel_set_threshold, "Set Threshold"),
                      (self.panel_quarantine, "Process Quarantine"),
                      (self.panel_process, "Results")]
        for panel, text in self.pages:
            self.notebook.AddPage(panel, text)
        
    def onPageChange(self, evt):
        old = evt.GetOldSelection()
        new = evt.GetSelection()

        if old == MainFrame.PROJECT:
            self.project = self.panel_projects.get_project()
            self.SetTitle("OpenCount -- Project {0}".format(self.project.name))
        elif old == MainFrame.CONFIG:
            self.panel_config.stop()
        elif old == MainFrame.PARTITION:
            self.panel_partition.stop()
        elif old == MainFrame.DEFINE_ATTRS:
            self.panel_define_attrs.stop()
        elif old == MainFrame.LABEL_ATTRS:
            self.panel_label_attrs.stop()
        elif old == MainFrame.LABEL_DIGATTRS:
            self.panel_label_digitattrs.stop()
        elif old == MainFrame.RUN_GROUPING:
            self.panel_run_grouping.stop()
        elif old == MainFrame.CORRECT_GROUPING:
            self.panel_correct_grouping.stop()
        elif old == MainFrame.SELTARGETS:
            self.panel_seltargets.stop()
        elif old == MainFrame.LABEL_CONTESTS:
            pass
        elif old == MainFrame.TARGET_EXTRACT:
            self.panel_target_extract.stop()
        elif old == MainFrame.SET_THRESHOLD:
            self.panel_set_threshold.stop()
        elif old == MainFrame.QUARANTINE:
            pass
        elif old == MainFrame.PROCESS:
            pass

        if new == MainFrame.PROJECT:
            self.panel_projects.start(PROJROOTDIR)
        elif new == MainFrame.CONFIG:
            self.panel_config.start(self.project, pathjoin(self.project.projdir_path,
                                                           '_state_config.p'))
        elif new == MainFrame.PARTITION:
            self.panel_partition.start(self.project, pathjoin(self.project.projdir_path,
                                                              '_state_partition.p'))
        elif new == MainFrame.DEFINE_ATTRS:
            self.panel_define_attrs.start(self.project, pathjoin(self.project.projdir_path,
                                                                 '_state_defineattrs.p'))
        elif new == MainFrame.LABEL_ATTRS:
            # Skip if there are no defined attributes
            if not exists_attrs(self.project):
                dlg = wx.MessageDialog(self, message="There are no Attributes defined \
in this election -- skipping to the next relevant task.", style=wx.OK)
                dlg.ShowModal()
                self.notebook.ChangeSelection(self.SELTARGETS)
                self.notebook.SendPageChangedEvent(self.LABEL_ATTRS, self.SELTARGETS)
            else:
                self.panel_label_attrs.start(self.project)
        elif new == MainFrame.LABEL_DIGATTRS:
            # Skip if there are no digit-based attributes
            if not exists_digitbasedattr(self.project):
                dlg = wx.MessageDialog(self, message="There are no Digit-Based \
Attributes in this election -- skipping to the next page.", style=wx.OK)
                dlg.ShowModal()
                self.notebook.ChangeSelection(self.RUN_GROUPING)
                self.notebook.SendPageChangedEvent(self.LABEL_DIGATTRS, self.RUN_GROUPING)
            else:
                self.panel_label_digitattrs.start(self.project)
        elif new == MainFrame.RUN_GROUPING:
            self.panel_run_grouping.start(self.project, pathjoin(self.project.projdir_path,
                                                                 '_state_run_grouping.p'))
        elif new == MainFrame.CORRECT_GROUPING:
            self.panel_correct_grouping.start(self.project, pathjoin(self.project.projdir_path,
                                                                     '_state_correct_grouping.p'))
        elif new == MainFrame.SELTARGETS:
            self.panel_seltargets.start(self.project, pathjoin(self.project.projdir_path,
                                                               '_state_seltargets.p'),
                                        self.project.ocr_tmp_dir)
        elif new == MainFrame.LABEL_CONTESTS:
            """ Requires:
                proj.target_locs_dir -- Location of targets
                proj.patch_loc_dir -- For language, and *something* else.
            """
            self.panel_label_contests.proj = self.project
            sz = self.GetSize()
            self.panel_label_contests.start(sz)
            self.SendSizeEvent()
        elif new == MainFrame.TARGET_EXTRACT:
            self.panel_target_extract.start(self.project)
        elif new == MainFrame.SET_THRESHOLD:
            sz = self.GetSize()
            self.panel_set_threshold.start(self.project, size=sz)
            self.SendSizeEvent()
        elif new == MainFrame.QUARANTINE:
            pass
        elif new == MainFrame.PROCESS:
            self.panel_process.start(self.project)

    def onClose(self, evt):
        """
        Triggered when the user/program exits/closes the MainFrame.
        """
        if self.project:
            self.project.save()
        if self.notebook.GetCurrentPage() == self.panel_define_attrs:
            self.panel_define_attrs.stop()
        for fn in Project.closehook:
            fn()
        evt.Skip()

def exists_digitbasedattr(proj):
    attrs = pickle.load(open(proj.ballot_attributesfile, 'rb'))
    for attr in attrs:
        if attr['is_digitbased']:
            return True
    return False

def exists_attrs(proj):
    if not os.path.exists(proj.ballot_attributesfile):
        return False
    ballot_attributesfile = pickle.load(open(proj.ballot_attributesfile, 'rb'))
    if not ballot_attributesfile:
        return False
    else:
        return True

def main():
    app = wx.App(False)
    f = MainFrame(None)
    f.Show()
    f.Maximize()
    app.MainLoop()

if __name__ == '__main__':
    main()
