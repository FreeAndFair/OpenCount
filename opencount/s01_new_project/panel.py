'''
The panel which lists all available projects, allows the user to
create new ones, and to delete older ones.
'''

from os import path
import wx

import ffwx
import util
from util import warn, error
from project import Project


class ProjectPanel(ffwx.Panel):
    '''
    The project-picker panel.
    '''
    def __init__(self, parent, *args, **kwargs):
        ffwx.Panel.__init__(self, parent, *args, **kwargs)
        # PROJDIR: Root directory of all projects
        self.projdir = None
        self.init_ui()

    def init_ui(self):
        '''
        Create all the necessary widgets.
        '''

        ff = ffwx.FFBuilder(self)
        self.listbox_projs = ff.list_box(choices=(), size=(500, 400))

        self.sizer = ff.vbox(
            ff.text('Select an election project to work on:'),
            ff.static_vbox(
                self.listbox_projs,
                ff.hbox(
                    ff.button(label='Create New Project',
                              on_click=self.on_button_create),
                    ff.button(label='Delete Selected Project',
                              on_click=self.on_button_remove)),
                label='Election Projects'))

        self.SetSizer(self.sizer)
        self.Layout()

    def start(self, projdir='', **kwargs):
        """
        Input:
            str PROJDIR: Root directory where all projects reside.
        """
        self.projdir = projdir
        projects = sorted(Project.load_projects(projdir),
                          key=lambda proj: proj.name)
        self.listbox_projs.Clear()
        for proj in projects:
            self.listbox_projs.Append(proj.name)

    def can_move_on(self):
        '''
        Allow the user to move on only if a project is selected.
        '''
        if not self.get_project():
            return False, "Please select a project before moving on."
        else:
            return True, None

    @util.show_exception_as_modal
    def get_project(self):
        '''
        Returns the Project instance of the selected project.
        '''
        idx = self.listbox_projs.get_selected()
        if idx:
            return Project.load_project(self.projdir, idx)
        else:
            error("NONE SELECTED")
            return None

    @util.show_exception_as_modal
    def on_button_create(self, evt):
        '''
        Create a new project with the supplied project name.
        '''
        name = ffwx.text_entry(self, "New Project Name:", "New Project")
        if name is not None:
            Project.create_project(name, path.join(self.projdir, name))
            self.listbox_projs.add_focused(name)

    @util.show_exception_as_modal
    def on_button_remove(self, evt):
        '''
        Removes project from the ListBox, internal data structures,
        and from the projects/ directory.
        '''
        name = self.listbox_projs.get_selected()
        if name is None:
            raise util.InformativeException("No project selected.")
        status = ffwx.yesno(self,
                            "Are you sure you want to delete project "
                            "'{0}' as well as all of its files?".format(
                                name))
        if status:
            warn("Deleting project '{0}'".format(name))
            self.listbox_projs.Delete(self.listbox_projs.FindString(name))
            Project.delete_project(self.projdir, name)
