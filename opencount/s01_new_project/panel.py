import os
from os import path
import re
import shutil
import wx

import ffwx
from util import debug, warn, error
from project import load_projects


class ProjectPanel(ffwx.Panel):

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

        txt0 = wx.StaticText(
            self, label="Select the election project you'd like to work on.")
        box1 = wx.StaticBox(self, label="Election Projects")
        ssizer1 = wx.StaticBoxSizer(box1, wx.VERTICAL)

        self.listbox_projs = wx.ListBox(self, choices=(), size=(500, 400))
        btn_create = ffwx.Button(
            self, label='Create New Project', on_click=self.onButton_create
        )
        btn_remove = ffwx.Button(
            self, label="Delete Selected Project", on_click=self.onButton_remove
        )
        btnsizer = ffwx.hbox(btn_create, btn_remove)

        ssizer1.AddMany([(self.listbox_projs,), (btnsizer,)])

        ssizer0.AddMany([(txt0,), (ssizer1,)])

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(ssizer0)
        self.SetSizer(self.sizer)
        self.Layout()

    def start(self, project=None, projdir=''):
        """
        Input:
            str PROJDIR: Root directory where all projects reside.
        """
        self.projdir = projdir
        projects = sorted(load_projects(projdir), key=lambda proj: proj.name)
        for proj in projects:
            self.add_project(proj)

    def can_move_on(self):
        if not self.get_project():
            msg = "Please select a project before moving on."
            return False, msg
        return True, None

    def get_project(self):
        """ Returns the Project instance of the selected project. """
        idx = self.listbox_projs.GetSelection()
        if idx == wx.NOT_FOUND:
            error("NONE SELECTED")
            return None
        return self.projects[idx]

    def add_project(self, proj):
        self.projects.append(proj)
        self.listbox_projs.Append(proj.name)

    def remove_project(self, proj):
        self.projects.remove(proj)
        idx = self.listbox_projs.FindString(proj.name)
        self.listbox_projs.Delete(idx)

    def contains_project(self, projname):
        return projname in [proj.name for proj in self.projects]

    def create_new_project(self, name):
        proj = create_project(name, path.join(self.projdir, name))
        self.add_project(proj)

    def onButton_create(self, evt):
        dlg = wx.TextEntryDialog(self, message="New Project Name:",
                                 caption="New Project", defaultValue="ProjectNameHere")
        val = dlg.ShowModal()
        if val == wx.ID_OK:
            project_name = dlg.GetValue().strip()
            if self.contains_project(project_name):
                dlg = wx.MessageDialog(self,
                                       message="{0} already exists as a project.".format(
                                           project_name),
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
        idx = self.listbox_projs.FindString(
            self.listbox_projs.GetStringSelection())
        proj = self.projects[idx]
        projdir = proj.projdir_path
        dlg = wx.MessageDialog(self, message="Are you sure you want to delete \
project {0}, as well as all of its files within {1}?".format(proj.name, projdir),
                               style=wx.YES_NO)
        status = dlg.ShowModal()
        if status == wx.ID_NO:
            return
        self.remove_project(proj)
        shutil.rmtree(projdir)

