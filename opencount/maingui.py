'''
The main entry function for OpenCount, including the main panel which
contains all the other panels.
'''

import argparse
import csv
import datetime
import os
import sys

try:
    import cPickle as pickle
except ImportError:
    import pickle

import matplotlib
matplotlib.use('Agg')

from os.path import join as pathjoin

import wx

from project import Project
import panels

import ffwx
from opencount.util import debug, warn, error

import config
import opencount.util as util

PROJROOTDIR = 'projects_new'

class Steps:
    PROJECT = 0
    CONFIG = 1
    PARTITION = 2
    BALLOT_ATTRIBUTES = 3
    LABEL_DIGATTRS = 4
    RUN_GROUPING = 5
    CORRECT_GROUPING = 6
    SELTARGETS = 7
    LABEL_CONTESTS = 8
    TARGET_EXTRACT = 9
    SET_THRESHOLD = 10
    QUARANTINE = 11
    PROCESS = 12

    _PATH_NAME = [
        None,
        '_state_config.p',
        '_state_partition.p',
        '_state_ballot_attributes.p',
        None,
        '_state_run_grouping.p',
        '_state_correct_grouping.p',
        '_state_selecttargetsMain.p',
        None,
        None,
        None,
    ]

    @classmethod
    def path_for(proj, step):
        name = Steps._PATH_NAME[step]
        if name:
            return [pathjoin(proj.projdir_path, name)]
        else:
            return []

class MainFrame(wx.Frame):
    '''
    The main frame, which contains the relevant project data.
    '''
    PROJECT = 0
    CONFIG = 1
    PARTITION = 2
    BALLOT_ATTRIBUTES = 3
    LABEL_DIGATTRS = 4
    RUN_GROUPING = 5
    CORRECT_GROUPING = 6
    SELTARGETS = 7
    LABEL_CONTESTS = 8
    TARGET_EXTRACT = 9
    SET_THRESHOLD = 10
    QUARANTINE = 11
    PROCESS = 12

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
        self.pages = [
            (panels.ProjectPanel(self.notebook),
             "Projects",
             "Projects"),
            (panels.ConfigPanel(self.notebook),
             "Import Files",
             "Import"),
            (panels.PartitionMainPanel(self.notebook),
             "Partition Ballots",
             "Partition"),
            (panels.BallotAttributesPanel(self.notebook),
             "Ballot Attributes",
             "Attrs"),
            (panels.LabelDigitsPanel(self.notebook),
             "Label Digit-Based Attributes",
             "Label Digit Attrs"),
            (panels.RunGroupingMainPanel(self.notebook),
             "Run Grouping",
             "Group"),
            (panels.VerifyGroupingMainPanel(self.notebook),
             "Correct Grouping",
             "Correct Grouping"),
            (panels.SelectTargetsMainPanel(self.notebook),
             "Select Voting Targets",
             "Targets"),
            (panels.LabelContest(self.notebook, self.GetSize()),
             "Label Contests",
             "Contests"),
            (panels.TargetExtractPanel(self.notebook),
             "Extract Targets",
             "Extract"),
            (panels.ThresholdPanel(self.notebook, self.GetSize()),
             "Set Threshold",
             "Threshold"),
            (panels.QuarantinePanel(self.notebook),
             "Process Quarantine",
             "Quarantine"),
            (panels.ResultsPanel(self.notebook),
             "Results",
             "Results"),
        ]
        self.panels = list((panel for (panel, _, _) in self.pages))

        self.titles = {}
        for panel, fullname, shortname in self.pages:
            self.notebook.AddPage(panel, shortname)
            self.titles[panel] = (fullname, shortname)

    def set_project(self, proj):
        self.project = proj
        self.SetTitle('OpenCount: "{0}"'.format(proj.name))

    def onPageChanging(self, evt):
        old = evt.GetOldSelection()
        new = evt.GetSelection()
        if old == -1:
            # Don't know why these events are sometimes triggered...
            return

        status, msg = self.notebook.GetPage(old).can_move_on()
        if not status:
            ffwx.modal(self, msg)
            evt.Veto()

        if old == Steps.PROJECT:
            self.set_project(self.notebook.GetPage(old).get_project())

        if old >= 1:
            self.panels[old].stop()

    def switch_to(self, tgt):
        self.notebook.ChangeSelection(tgt)
        self.notebook.SendPageChangedEvent(self.BALLOT_ATTRIBUTES, tgt)

    def onPageChange(self, evt):
        old = evt.GetOldSelection()
        new = evt.GetSelection()

        if self.project:
            self.project.save()
        if config.TIMER:
            config.TIMER.dump()

        if old != -1:
            curpanel = self.notebook.GetPage(old)
            self.notebook.SetPageText(old, self.titles[curpanel][1])
        newpanel = self.notebook.GetPage(new)
        self.notebook.SetPageText(new, self.titles[newpanel][0])

        if new >= MainFrame.SELTARGETS:
            if not self.project.is_grouped():
                # Grouping wasn't performed, which means that we should
                # simply use the partitions as the groups, since the user
                # 'knows' that the partitions also specify a grouping.
                if self.project.is_partitioned():
                    debug("No Attributes Exists, so, using Partitioning as the Grouping.")
                    self.project.use_partitions_as_grouping()
                else:
                    debug("Couldn't find {0}. Was decoding not performed?",
                          self.project.partition_path())
                    ffwx.warn(self, 'You must first run decoding (partitioning) '
                              'before proceeding to this step. OpenCount will '
                              'take you there now.')
                    self.switch_to(self.PARTITION)
                    return

        if new == MainFrame.PROJECT:
            self.panels[new].start(projdir=PROJROOTDIR)
        elif new == MainFrame.CONFIG:
            self.panels[new].start(project=self.project)
        elif new == MainFrame.PARTITION:
            self.panels[new].start(project=self.project)
        elif new == MainFrame.BALLOT_ATTRIBUTES:
            # A (crude) way of detecting if the user started working on
            # ballot attributes already
            if not self.project.has_attribute_data():
                resp = ffwx.yesno(self,
                                'Do you wish to define any Ballot Attributes? \n\n'
                                'If you intend to define ballot attributes '
                                '(precinct number, tally group, etc.), then '
                                'click \'Yes\'.\n\nOtherwise, click \'No\'.')
                if resp == wx.ID_NO:
                    ffwx.modal(self,
                             'You indicated that you do not want to define '
                             'any ballot attributes.\n\n '
                             'You will now be taken to the ballot annotation '
                             'stage.')
                    self.switch_to(self.SELTARGETS)
                    return

            self.panels[new].start(project=self.project)
        elif new == MainFrame.LABEL_DIGATTRS:
            # Skip if there are no digit-based attributes
            if not self.project.has_digitbasedattr():
                ffwx.modal(self,
                         'There are no Digit-Based Attributes in this '
                         'election -- skipping to the next page.')
                self.switch_to(self.RUN_GROUPING)
                self.notebook.ChangeSelection(self.RUN_GROUPING)
                self.notebook.SendPageChangedEvent(
                    self.LABEL_DIGATTRS, self.RUN_GROUPING)
            else:
                self.panels[new].start(self.project)
        elif new == MainFrame.RUN_GROUPING:
            if not self.project.has_imgattr() \
               and not self.project.has_digitbasedattr():
                ffwx.modal(self,
                         'There are no attributes to group in this election '
                         '-- skipping to the next page.')
                if self.project.has_custattr():
                    dst_page = self.CORRECT_GROUPING
                else:
                    dst_page = self.SELTARGETS
                self.notebook.ChangeSelection(dst_page)
                self.notebook.SendPageChangedEvent(self.RUN_GROUPING, dst_page)
            else:
                self.panels[new].start(project=self.project)
        elif new == MainFrame.CORRECT_GROUPING:
            if not exists_imgattr(self.project) and \
               not exists_digitbasedattr(self.project) and \
               not exists_custattr(self.project):
                ffwx.modal(self,
                         'There are no attributes to verify grouping for '
                         'in this election -- skipping to the next page.')
                self.notebook.ChangeSelection(self.SELTARGETS)
                self.notebook.SendPageChangedEvent(
                    self.CORRECT_GROUPING, self.SELTARGETS)
            else:
                self.panels[new].start(
                    self.project,
                    pathjoin(self.project.projdir_path, '_state_correct_grouping.p'))
        elif new == MainFrame.SELTARGETS:
            self.panels[new].start(self.project,
                                   self.project.path('_state_selecttargetsMain.p'),
                                   self.project.ocr_tmp_dir)
        elif new == MainFrame.LABEL_CONTESTS:
            """ Requires:
                proj.target_locs_dir -- Location of targets
                proj.patch_loc_dir -- For language, and *something* else.
            """
            self.panels[new].proj = self.project
            self.panels[new].start(self.GetSize())
            self.SendSizeEvent()
        elif new == MainFrame.TARGET_EXTRACT:
            self.panels[new].start(self.project)
        elif new == MainFrame.SET_THRESHOLD:
            sz = self.GetSize()
            self.panels[new].start(self.project, size=sz)
            self.SendSizeEvent()
        elif new == MainFrame.QUARANTINE:
            self.panels[new].start(self.project, self.GetSize())
        elif new == MainFrame.PROCESS:
            self.panels[new].start(self.project)

    def onClose(self, evt):
        """
        Triggered when the user/program exits/closes the MainFrame.
        """
        if self.project:
            self.project.save()
        if config.TIMER:
            config.TIMER.stop_task("TOTALTIME")
            config.TIMER.dump()
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


def exists_custattr(proj):
    if not exists_attrs(proj):
        return False
    attrprops = pickle.load(
        open(pathjoin(proj.projdir_path, proj.attrprops), 'rb'))
    return len(attrprops[ATTRMODE_CUSTOM]) != 0


def exists_attrs(proj):
    """ Doesn't account for custom attrs. """
    if not os.path.exists(proj.ballot_attributesfile):
        return False
    ballot_attributesfile = pickle.load(open(proj.ballot_attributesfile, 'rb'))
    if not ballot_attributesfile:
        return False
    else:
        return True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num", type=int, dest='n',
                        help="Only process the first N ballots.")
    parser.add_argument("--time", metavar="PREFIX",
                        help='OpenCount will output timing statistics '
                        'to a logfile. If PREFIX is given as \'foo\', '
                        'then the output filename is: \n'
                        'foo_YEAR_MONTH_DAY_HOUR_MINUTE.log')
    parser.add_argument("--dev", action='store_true',
                        help='Run OpenCount in Development mode. This '
                        'enables a few dev-specific buttons in the UI '
                        'which are useful when debugging projects.')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.n:
        debug("Processing first {0} ballots [user passed in -n,--num]", args.n)
        config.BALLOT_LIMIT = args.n
    config.IS_DEV = args.dev
    if config.IS_DEV:
        debug("Running in dev-mode")
    if args.time:
        prefix = args.time
        now = datetime.datetime.now()
        date_suffix = "{0}_{1}_{2}_{3}_{4}".format(
            now.year, now.month, now.day, now.hour, now.minute)
        # "PREFIX_YEAR_MONTH_DAY_HOUR_MINUTE.log"
        timing_filepath = "{0}_{1}.log".format(prefix, date_suffix)
        config.TIMER = util.MyTimer(timing_filepath)
        debug("User passed in '--time': Saving timing statistics to {0}",
              timing_filepath)

    app = wx.App(False)
    f = MainFrame(None, size=wx.GetDisplaySize())
    f.Show()
    app.MainLoop()

if __name__ == '__main__':
    main()
