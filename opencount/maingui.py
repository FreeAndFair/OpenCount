import os, sys, csv
try:
    import cPickle as pickle
except ImportError as e:
    import pickle

from os.path import join as pathjoin

import wx

sys.path.append('..')
from projconfig_new.project_panel import ProjectPanel, Project
from projconfig_new.config_panel import ConfigPanel
from partitions.partition_panel import PartitionMainPanel
from specify_voting_targets.select_targets import SelectTargetsMainPanel
from labelcontest.labelcontest import LabelContest
from grouping.define_attributes_new import DefineAttributesMainPanel
from grouping.select_attributes import SelectAttributesMasterPanel
from digits_ui.digits_ui import LabelDigitsPanel
from grouping.run_grouping import RunGroupingMainPanel
from grouping.verify_grouping_panel import VerifyGroupingMainPanel
from runtargets.extract_targets_new import TargetExtractPanel
from threshold.threshold import ThresholdPanel
from quarantine.quarantinepanel import QuarantinePanel
from post_processing.postprocess import ResultsPanel

import specify_voting_targets.util_gui as util_gui

"""
The main module for OpenCount.

Usage:
    $ python maingui.py [-h --help -help]
"""

USAGE = """Usage:

    $ python maingui.py [-h --help -help]

"""

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
        self.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGING, self.onPageChanging)
        self.Bind(wx.EVT_CLOSE, self.onClose)

        self.notebook.ChangeSelection(0)
        self.notebook.SendPageChangedEvent(-1, 0)

    def init_ui(self):
        self.notebook = wx.Notebook(self)
        self.setup_pages()
        self.Maximize()

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
        self.pages = [(self.panel_projects, "Projects", "Projects"),
                      (self.panel_config, "Import Files", "Import"),
                      (self.panel_partition, "Partition Ballots", "Partition"),
                      (self.panel_define_attrs, "Define Ballot Attributes", "Define Attrs"),
                      (self.panel_label_attrs, "Label Ballot Attributes", "Label Attrs"),
                      (self.panel_label_digitattrs, "Label Digit-Based Attributes", "Label Digit Attrs"),
                      (self.panel_run_grouping, "Run Grouping", "Group"),
                      (self.panel_correct_grouping, "Correct Grouping", "Correct Grouping"),
                      (self.panel_seltargets, "Select Voting Targets", "Targets"),
                      (self.panel_label_contests, "Label Contests", "Contests"),
                      (self.panel_target_extract, "Extract Targets", "Extract"),
                      (self.panel_set_threshold, "Set Threshold", "Threshold"),
                      (self.panel_quarantine, "Process Quarantine", "Quarantine"),
                      (self.panel_process, "Results", "Results")]
        self.titles = {}
        for panel, fullname, shortname in self.pages:
            self.notebook.AddPage(panel, shortname)
            self.titles[panel] = (fullname, shortname)
    def onPageChanging(self, evt):
        old = evt.GetOldSelection()
        new = evt.GetSelection()
        if old == -1:
            # Don't know why these events are sometimes triggered...
            return

        if old == MainFrame.PROJECT:
            status, msg = self.panel_projects.can_move_on()
            if status:
                self.project = self.panel_projects.get_project()
                self.SetTitle("OpenCount -- Project {0}".format(self.project.name))
            else:
                dlg = wx.MessageDialog(self, message=msg, style=wx.ID_OK)
                dlg.ShowModal()
                evt.Veto()
            return

        curpanel = self.notebook.GetPage(old)
        if hasattr(curpanel, 'can_move_on'):
            if not curpanel.can_move_on():
                wx.MessageDialog(self, message="Error: You can not \
proceed. Please address the prior warnings first.",
                                 caption="OpenCount: Can't go on",
                                 style=wx.OK).ShowModal()
                evt.Veto()
                return
        else:
            print "...Warning: Class {0} has no can_move_on method.".format(curpanel)

        if old == MainFrame.CONFIG:
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
            self.panel_quarantine.stop()
        elif old == MainFrame.PROCESS:
            pass

    def onPageChange(self, evt):
        old = evt.GetOldSelection()
        new = evt.GetSelection()

        if self.project:
            self.project.save()

        if old != -1:
            curpanel = self.notebook.GetPage(old)
            self.notebook.SetPageText(old, self.titles[curpanel][1])
        newpanel = self.notebook.GetPage(new)
        self.notebook.SetPageText(new, self.titles[newpanel][0])


        if new >= MainFrame.SELTARGETS:
            if not os.path.exists(pathjoin(self.project.projdir_path,
                                           self.project.group_to_ballots)):
                # Grouping wasn't performed, which means that we should
                # simply use the partitions as the groups, since the user
                # 'knows' that the partitions also specify a grouping.
                partitions_map = pickle.load(open(pathjoin(self.project.projdir_path,
                                                           self.project.partitions_map), 'rb'))
                partitions_invmap = pickle.load(open(pathjoin(self.project.projdir_path,
                                                              self.project.partitions_invmap), 'rb'))
                partition_exmpls = pickle.load(open(pathjoin(self.project.projdir_path,
                                                             self.project.partition_exmpls), 'rb'))

                # The GRP_INFOMAP should just contain partitionid info.
                grp_infomap = {} # maps {int groupID: {str prop: str val}}
                grp2bals = {}
                bal2grp = {}
                grpexmpls = {}
                curgroupid = 0
                for (partitionid, ballotids) in sorted(partitions_map.iteritems()):
                    if not ballotids:
                        continue
                    propdict = {'pid': partitionid}
                    grp_infomap[curgroupid] = propdict
                    print 'extend', curgroupid, 'by', ballotids
                    grp2bals.setdefault(curgroupid, []).extend(ballotids)
                    for ballotid in ballotids:
                        bal2grp[ballotid] = curgroupid
                    curgroupid += 1
                curgroupid = 0
                for (partitionid, ballotids) in sorted(partition_exmpls.iteritems()):
                    if not ballotids:
                        continue
                    grpexmpls[curgroupid] = ballotids
                    curgroupid += 1

                # Also, export to proj.group_results.csv, for integration with
                # quarantine/post-processing panels.
                print "SET TO", grp2bals
                fields = ('ballotid', 'groupid')
                csvfile = open(self.project.grouping_results, 'wb')
                dictwriter = csv.DictWriter(csvfile, fieldnames=fields)
                try:
                    dictwriter.writeheader()
                except:
                    util_gui._dictwriter_writeheader(csvfile, fields)
                rows = []
                for ballotid, groupid in bal2grp.iteritems():
                    rows.append({'ballotid': ballotid, 'groupid': groupid})
                dictwriter.writerows(rows)
                csvfile.close()

                pickle.dump(grp2bals, open(pathjoin(self.project.projdir_path,
                                                    self.project.group_to_ballots), 'wb'),
                            pickle.HIGHEST_PROTOCOL)
                pickle.dump(bal2grp, open(pathjoin(self.project.projdir_path,
                                                   self.project.ballot_to_group), 'wb'),
                            pickle.HIGHEST_PROTOCOL)
                pickle.dump(grpexmpls, open(pathjoin(self.project.projdir_path,
                                                     self.project.group_exmpls), 'wb'),
                            pickle.HIGHEST_PROTOCOL)
                pickle.dump(grp_infomap, open(pathjoin(self.project.projdir_path,
                                                       self.project.group_infomap), 'wb'),
                            pickle.HIGHEST_PROTOCOL)

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
            if not exists_imgattr(self.project):
                dlg = wx.MessageDialog(self, message="There are no Image-based Attributes defined \
in this election -- skipping to the next relevant task.", style=wx.OK)
                dlg.ShowModal()
                self.notebook.ChangeSelection(self.LABEL_DIGATTRS)
                self.notebook.SendPageChangedEvent(self.LABEL_ATTRS, self.LABEL_DIGATTRS)
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
            if not exists_imgattr(self.project) and not exists_digitbasedattr(self.project):
                dlg = wx.MessageDialog(self, message="There are no attributes \
to group in this election -- skipping to the next page.", style=wx.OK)
                dlg.ShowModal()
                self.notebook.ChangeSelection(self.SELTARGETS)
                self.notebook.SendPageChangedEvent(self.RUN_GROUPING, self.SELTARGETS)
            else:
                self.panel_run_grouping.start(self.project, pathjoin(self.project.projdir_path,
                                                                     '_state_run_grouping.p'))
        elif new == MainFrame.CORRECT_GROUPING:
            if not exists_imgattr(self.project) and not exists_digitbasedattr(self.project):
                dlg = wx.MessageDialog(self, message="There are no attributes \
to verify grouping for in this election -- skipping to the next page.", style=wx.OK)
                dlg.ShowModal()
                self.notebook.ChangeSelection(self.SELTARGETS)
                self.notebook.SendPageChangedEvent(self.CORRECT_GROUPING, self.SELTARGETS)
            else:
                self.panel_correct_grouping.start(self.project, pathjoin(self.project.projdir_path,
                                                                         '_state_correct_grouping.p'))
        elif new == MainFrame.SELTARGETS:
            self.panel_seltargets.start(self.project, pathjoin(self.project.projdir_path,
                                                               '_state_selecttargetsMain.p'),
                                        self.project.ocr_tmp_dir)
        elif new == MainFrame.LABEL_CONTESTS:
            """ Requires:
                proj.target_locs_dir -- Location of targets
                proj.patch_loc_dir -- For language, and *something* else.
            """
            self.panel_label_contests.proj = self.project
            img2flip = pickle.load(open(pathjoin(self.project.projdir_path,
                                                 self.project.image_to_flip), 'rb'))
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
            self.panel_quarantine.start(self.project)
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
    if not exists_attrs(proj):
        return False
    attrs = pickle.load(open(proj.ballot_attributesfile, 'rb'))
    for attr in attrs:
        if attr['is_digitbased']:
            return True
    return False
def exists_imgattr(proj):
    if not exists_attrs(proj):
        return False
    attrs = pickle.load(open(proj.ballot_attributesfile, 'rb'))
    for attr in attrs:
        if not attr['is_digitbased']:
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
    args = sys.argv[1:]
    if '-h' in args or '--help' in args or '-help' in args:
        print USAGE
        return 0

    app = wx.App(False)
    f = MainFrame(None, size=wx.GetDisplaySize())
    f.Show()
    app.MainLoop()

if __name__ == '__main__':
    main()
