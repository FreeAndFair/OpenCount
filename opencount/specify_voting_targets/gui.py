import os
import util_gui
from imageviewer import BallotScreen, Autodetect_Confirm, Autodetect_Panel, WorldState

"""
UI intended for a user to denote all locations of voting targets, and
export these locations to a .csv file.
This UI currently is suited to process individual template images on 
a 'fine-grained' level - it'd be nice to have a UI that can perform
the auto-detection on many templates at once. find_targets_wizard.py
accomplishes this.
"""

'''
Bug list:
1.) hand-tool/resizing not working when zoomed in
2.) When Autodetect frame is killed by an unhandled exception, unable 
    to re-use it without restarting application.
3.) When in 'hand' mode, resizing another target while having a selected
    target results on bad behavior.
4.) While creating a target, click-dragging to the upper-left causes 
    weird behavior.
'''

####
## Import 3rd party libraries
####
'''
wget http://sourceforge.net/projects/opencvlibrary/files/opencv-unix/2.3.1/OpenCV-2.3.1a.tar.bz2/download
tar -xvf download
'''

try:
    import wx
    import wx.animate
except ImportError:
    print """Error importing wxPython (wx) -- to install wxPython (a Python GUI \
library), do (if you're on Linux):
    sudo apt-get install python-wxgtk2.8
Or go to: 
    http://www.wxpython.org/download.php
For OS-specific installation instructions."""
    exit(1)
try:
    import Image
except ImportError:
    print """Error importing Python Imaging Library (Image) -- to install \
PIL (a Python image-processing library), go to: 
    http://www.pythonware.com/products/pil/"""
    exit(1)
try:
    import cv2
except ImportError:
    print """Error importing OpenCV w/ Python bindings (cv2) -- to install \
OpenCV w/ Python bindings (a Python computer vision library), go to:
    http://opencv.willowgarage.com/wiki/
Note that documentation for installing OpenCV is pretty shaky in my \
experience. A README section on installing OpenCV will be created soon.
On Windows, to get the Python bindings, copy/paste the contents of:
    opencv/build/python/2.7 (or 2.6)
to the site-packages directory of your Python installation, i.e.:
    C:/Python27/Lib/site-packages/
For me, this means that you'll be adding two new files to that directory:
    C:/Python27/Lib/site-packages/cv.py
    C:/Python27/Lib/site-packages/cv2.pyd"""
    exit(1)
try:
    import numpy as np
except ImportError:
    print """Error importing Numpy (numpy) -- to install Numpy, go to:
    http://numpy.scipy.org/
You'll probably want to install both scipy and numpy."""
    exit(1)
import wx.lib.inspection
from wx.lib.pubsub import Publisher
    
_helptext = """
This user interface is primarily designed to specify the location of \
all voting targets on each template style. Once target locations have \
been specified, the user can export this work into an outfile for other \
programs to use. \


To read in and display an image, go to 'File->Open Image...', and \
select an image. To begin specifying voting target locations, enter \
'creation' mode by clicking the 'Add Target' button. \


To modify previously created voting target locations, enter 'modify' \
mode by clicking the 'Modify/Move' button. You can select a target by \
clicking the upper-left corner. \


To resize a selected voting target, change the width/height values in \
the 'Dimensions' bar, and hit 'enter'. Or, you can use the 'Global Resize' \
button to resize all voting targets to a certain dimension. \


Once you are finished, you can export the locations to a csv (Comma \
Separated Values) file by going to 'File->Export Locations...'. You may \
specify a custom filename if you wish. Now, other programs can use this \
csv file. \


If you want to re-check a previous session, you can import a previously-saved \
csvfile by going to 'File->Import Locations', and selecting the csvfile. \
"""

class MainFrame(wx.Frame):
    """
    Main top-level frame window for the application.
    """
    def __init__(self, parent):
        wx.Frame.__init__(self, parent, title="Ballot viewer")
        self.CreateStatusBar()
        
        # Setting up the menu.
        filemenu = wx.Menu()
        menu_open = filemenu.Append(wx.ID_OPEN, "&Open Image...", " Open a ballot image")
        menu_import = filemenu.Append(wx.ID_ANY, "&Import Target Locations...", " Import Target locations from a csv file")
        menu_export = filemenu.Append(wx.ID_SAVEAS, "&Export Target Locations...", " Export Target locations to file")
        filemenu.AppendSeparator()
        menu_exit = filemenu.Append(wx.ID_EXIT, "E&xit", " Terminate the program.")
        menubar = wx.MenuBar()
        menubar.Append(filemenu, "&File")
        helpmenu = wx.Menu()
        menu_howtouse = helpmenu.Append(wx.ID_ANY, "&How to use", " How to use this program")
        menu_about = helpmenu.Append(wx.ID_ABOUT, "&About", " Information about this program")
        menubar.Append(helpmenu, "&Help")
        self.SetMenuBar(menubar)
        
        # Events.
        self.Bind(wx.EVT_MENU, self.OnOpen, menu_open)
        self.Bind(wx.EVT_MENU, self.onImport, menu_import)
        self.Bind(wx.EVT_MENU, self.onExport, menu_export)
        self.Bind(wx.EVT_MENU, self.onHelp, menu_howtouse)
        self.Bind(wx.EVT_MENU, self.OnAbout, menu_about)
        self.Bind(wx.EVT_MENU, self.OnExit, menu_exit)
        
        # Pubsub Subscribing
        Publisher().subscribe(self._pushStatusBarMsg, "signals.StatusBar.push")
        Publisher().subscribe(self._popStatusBarMsg, "signals.StatusBar.pop")
        
        # Panel
        self.world = WorldState()
        self.panel = MainPanel(self, self.world)
        
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.panel, 1, wx.EXPAND)
        
        # Layout sizers
        self.SetSizer(self.sizer)
        self.sizer.Fit(self)
        
        
    #### PubSub Calbacks
    
    def _pushStatusBarMsg(self, msg):
        self.GetStatusBar().PushStatusText(msg.data)
    def _popStatusBarMsg(self, msg):
        self.GetStatusBar().PopStatusText()
        
    #### Event Handlers
    
    def OnOpen(self, event):
        self.dirname = ''
        dlg = wx.FileDialog(self, "Choose a file", self.dirname, "", "*.*", style=wx.OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            self.cur_imgname = dlg.GetFilename()
            self.dirname = dlg.GetDirectory()
            filepath = os.path.join(self.dirname, self.cur_imgname)
            Publisher().sendMessage("signals.BallotScreen.new_image", filepath)
            Publisher().sendMessage("signals.Toolbar.clear_buttons", None)
        dlg.Destroy()

    def onImport(self, event):
        dirname = ''
        cur_imgname = util_gui.get_filename(self.panel.ballot_screen.current_imgpath)
        dlg = wx.FileDialog(self, "Import target locations", dirname, "{0}_targetlocs.csv".format(cur_imgname), "*.*", style=wx.OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            filename, dirname = dlg.GetFilename(), dlg.GetDirectory()
            filepath = os.path.join(dirname, filename)
            self.panel.ballot_screen.import_locations(filepath)
        dlg.Destroy()

    def onExport(self, event):
        dirname = ''
        self.cur_imgname = util_gui.get_filename(self.panel.ballot_screen.current_imgpath)
        dlg = wx.FileDialog(self, "Save to file", dirname, "{0}_targetlocs.csv".format(self.cur_imgname), "*.*", style=wx.SAVE)
        if dlg.ShowModal() == wx.ID_OK:
            filename, dirname = dlg.GetFilename(), dlg.GetDirectory()
            filepath = os.path.join(dirname, filename)
            self.panel.ballot_screen.export_locations(filepath)
        dlg.Destroy()
        
    def onHelp(self, event):
        dlg = wx.MessageDialog(self, message=_helptext, style=wx.OK)
        dlg.ShowModal()
        
    def OnAbout(self, event):
        dlg = wx.MessageDialog(self, message="This is being made in 2012.", style=wx.OK)
        dlg.ShowModal()
        
    def OnExit(self, event):
        self.Destroy()

class MainPanel(wx.Panel):
    """
    Contains the target dimensions, ballot image, and controls.
    """
    def __init__(self, parent, world, *args, **kwargs):
        wx.Panel.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.world = world
        # Target dimensions
        self.target_dim_panel = _Target_Dim_Panel(self)
        # Image
        self.img_panel = wx.Panel(self)
        self.img_panel.sizer = wx.BoxSizer(wx.VERTICAL)
        self.imgpath_txt = wx.StaticText(self.img_panel, label="Currently opened imagepath.")
        self.ballot_screen = BallotScreen(self.img_panel, world)
        self.img_panel.sizer.Add(self.imgpath_txt, flag=wx.ALIGN_LEFT)
        self.img_panel.sizer.Add(self.ballot_screen)
        self.img_panel.SetSizer(self.img_panel.sizer)
        self.img_panel.Fit()
        # Instructions
        self.inst_panel = _Control_Panel(self)
        # Toolbar
        self.toolbar = ToolBar(self)
        # Autodetect panel (not always visible)
        self.autodetect_panel = None
        self.autodetect_verify_panel = None
        
        self.sizer = wx.FlexGridSizer(rows=2, cols=2, hgap=5, vgap=5)
        self.sizer.AddGrowableRow(1)
        self.sizer.AddGrowableRow(2)
        self.sizer.AddGrowableCol(0)

        self.sizer.Add(self.toolbar, 0, flag=wx.ALIGN_LEFT | wx.EXPAND)
        self.sizer.Add(self.target_dim_panel, 0, wx.ALIGN_LEFT)
        #self.sizer.Add(self.ballot_screen, proportion=2, border=2, flag=wx.ALIGN_LEFT | wx.EXPAND | wx.ALL)
        self.sizer.Add(self.img_panel, proportion=2, flag=wx.ALIGN_LEFT | wx.EXPAND | wx.ALL)
        self.sizer.Add(self.inst_panel, 0, flag=wx.ALIGN_LEFT)
        
        self.Bind(wx.EVT_SIZE, self.onSize)
        self.SetSizer(self.sizer)
        self.SetAutoLayout(1)
        self.sizer.Fit(self)
        
        Publisher().subscribe(self._new_image, "signals.BallotScreen.new_image")
        Publisher().subscribe(self._enter_autodetect, "signals.enter_autodetect")
        Publisher().subscribe(self._cancel_autodetect, "signals.cancel_autodetect")
        Publisher().subscribe(self._enter_autodetect_verify, "signals.enter_autodetect_verify")
        Publisher().subscribe(self._leave_autodetect_verify, "signals.leave_autodetect_verify")        
    
    def onSize(self, event):
        self.Refresh()
        event.Skip()
        
    # Pubsub callbacks
    def _new_image(self, msg):
        filepath = msg.data
        self.imgpath_txt.SetLabel(filepath)
        self.Refresh()
    def _enter_autodetect(self, msg):
        """
        Add the auto-detect panel to the lower-right area
        """
        self.autodetect_panel = Autodetect_Panel(self)
        #self.sizer.Remove(3)    # remove inst panel
        #self.sizer.Insert(3, self.autodetect_panel)
        self.sizer.Replace(self.inst_panel, self.autodetect_panel)
        self.inst_panel.Hide()
        
        #self.parent.Layout()
        self.parent.Fit()
        #self.parent.Refresh()
        self.Refresh()
    def _cancel_autodetect(self, msg):
        """
        Replace the auto-detect panel with the text instructions
        """
        #self.sizer.Remove(3)
        #self.sizer.Insert(3, self.inst_panel, proportion=0, flag=wx.ALIGN_LEFT)
        self.sizer.Replace(self.autodetect_panel, self.inst_panel)
        self.autodetect_panel.Hide()
        self.autodetect_panel = None
        self.inst_panel.Show()
        self.parent.Fit()
        self.Refresh()
        
    def _enter_autodetect_verify(self, msg):
        self.autodetect_verify_panel = Autodetect_Confirm(self, self.ballot_screen.candidate_targets)
        self.sizer.Replace(self.autodetect_panel, self.autodetect_verify_panel)
        self.autodetect_panel.Hide()
        self.autodetect_panel = None
        self.parent.Fit()
        self.Refresh()
    def _leave_autodetect_verify(self, msg):
        self.sizer.Replace(self.autodetect_verify_panel, self.inst_panel)
        self.autodetect_verify_panel.Hide()
        self.autodetect_verify_panel = None
        self.inst_panel.Show()
        self.parent.Fit()
        self.Refresh()
        
        
class _Target_Dim_Panel(wx.Panel):
    """
    Panel to display the current voting target dimensions.
    """
    def __init__(self, *args, **kwargs):
        wx.Panel.__init__(self, *args, **kwargs)
        
        self.label = wx.StaticText(self, label="Target dimensions:")
        panel_1 = wx.Panel(self)
        panel_1.sizer = wx.BoxSizer(wx.VERTICAL)
        panel_width = wx.Panel(panel_1)
        panel_height = wx.Panel(panel_1)
        panel_width.sizer = wx.BoxSizer(wx.HORIZONTAL)
        panel_height.sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        self.label_w = wx.StaticText(panel_width, label="Width: ")
        self.width_txt = wx.TextCtrl(panel_width, style=wx.TE_PROCESS_ENTER)
        self.label_h = wx.StaticText(panel_height, label="Height: ")
        self.height_txt = wx.TextCtrl(panel_height, style=wx.TE_PROCESS_ENTER)
        self.width_txt.SetValue("40")
        self.height_txt.SetValue("30")
        
        self.globalresize_btn = wx.Button(panel_1, label="Global Resize...")
       
        panel_width.sizer.Add(self.label_w)
        panel_width.sizer.Add(self.width_txt)
        panel_height.sizer.Add(self.label_h)
        panel_height.sizer.Add(self.height_txt)
        panel_width.SetSizer(panel_width.sizer)
        panel_height.SetSizer(panel_height.sizer)
        panel_width.Fit()
        panel_height.Fit()
        panel_1.sizer.Add(panel_width)
        panel_1.sizer.Add(panel_height)
        panel_1.sizer.Add(self.globalresize_btn)
        panel_1.SetSizer(panel_1.sizer)
        panel_1.Fit()
        
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.label, 1, wx.EXPAND)
        self.sizer.Add(panel_1)
        #self.sizer.Add(Dummy(self, size=(20, 0)), 0, wx.EXPAND)
        #self.sizer.Add(self.label_w, 0, wx.EXPAND)
        #self.sizer.Add(self.width_txt, 0, wx.EXPAND)
        #self.sizer.Add(Dummy(self, size=(10, 0)), 0, wx.EXPAND)
        #self.sizer.Add(self.label_h, 0, wx.EXPAND)
        #self.sizer.Add(self.height_txt, 0, wx.EXPAND)
        #self.sizer.Add(self.globalresize_btn, 0, wx.EXPAND)
        
        self.SetSizer(self.sizer)
        self.SetAutoLayout(1)
        self.sizer.Fit(self)
        
        # Pubsub subscribing (for State changes)
        Publisher().subscribe(self.update_dim_text, "signals.TargetDimPanel")
        
        self._bind_events()
        
    def _bind_events(self):
        self.globalresize_btn.Bind(wx.EVT_BUTTON, self.onGlobalResize)
        self.width_txt.Bind(wx.EVT_TEXT_ENTER, self.onTextEnter)
        self.width_txt.Bind(wx.EVT_CHAR, self.onChar)
        self.width_txt.Bind(wx.EVT_KILL_FOCUS, self.txtLoseFocus)
        self.height_txt.Bind(wx.EVT_TEXT_ENTER, self.onTextEnter)
        self.height_txt.Bind(wx.EVT_KILL_FOCUS, self.txtLoseFocus)
        self.height_txt.Bind(wx.EVT_CHAR, self.onChar)
        
    #### Event handling
    
    def onGlobalResize(self, event):
        """ Open a dialogue box that confirms global resize. """
        new_w, new_h = int(self.width_txt.GetValue()), int(self.height_txt.GetValue())
        dlg = wx.MessageDialog(self, message="Are you sure you want to \
resize all target boxes to new dimensions: {0}, {1}?".format(new_w, new_h), 
                             caption="Global Resize?",
                             style=wx.YES_NO)
        response = dlg.ShowModal()
        if response == wx.ID_YES:
            Publisher().sendMessage("signals.BallotScreen.global_resize", (new_w, new_h))
        event.Skip()
        
    def onTextEnter(self, event):
        w, h = int(self.width_txt.GetValue()), int(self.height_txt.GetValue())
        Publisher().sendMessage("signals.BallotScreen.set_sel_target_size", (w,h))
        event.Skip()
        
    def txtLoseFocus(self, event):
        w, h = int(self.width_txt.GetValue()), int(self.height_txt.GetValue())
        Publisher().sendMessage("signals.BallotScreen.set_sel_target_size", (w,h))
        event.Skip()
        
    def onChar(self, event):
        event.Skip()
        
    def update_dim_text(self, msg):
        """ Method for PubSub Interface """
        w, h = msg.data
        self.width_txt.SetValue(str(w))
        self.height_txt.SetValue(str(h))
        
class _Control_Panel(wx.Panel):
    """
    Panel that displays instructions on what to do, in addition to
    providing control buttons.
    """
    def __init__(self, *args, **kwargs):
        wx.Panel.__init__(self, *args, **kwargs)
        # TODO: Figure out how to do text line-wrapping
        lbl1 = wx.StaticText(self, label="Instructions:")
        lbl2 = wx.StaticText(self, label="Add target(s): click on location(s) on ballot")
        lbl3 = wx.StaticText(self, label='''Move target: click on the upper-left corner of the target, then move \n\
it either with the mouse (click/drag), or with the arrow keys''')
        lbl4 = wx.StaticText(self, label="Label target: click on target, press 'L'")
        lbl5 = wx.StaticText(self, label="Delete target: click target, then press 'Delete' or 'Backspace'")
        
        self.btn_panel = wx.Panel(self)
        self.sizer2 = wx.BoxSizer(wx.HORIZONTAL)
        self.btn_panel.SetSizer(self.sizer2)
        self.btn_panel.SetAutoLayout(1)
        self.sizer2.Fit(self.btn_panel)
        
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(lbl1, 1, wx.EXPAND)
        self.sizer.Add(lbl2, 1, wx.EXPAND)
        self.sizer.Add(lbl3, 1, wx.EXPAND)
        self.sizer.Add(lbl4, 1, wx.EXPAND)
        self.sizer.Add(lbl5, 1, wx.EXPAND)
        self.sizer.Add(Dummy(self), 1, wx.EXPAND)
        self.sizer.Add(self.btn_panel, 1)
        
        self.SetSizer(self.sizer)
        self.SetAutoLayout(1)
        self.sizer.Fit(self)
        
class ToolBar(wx.Panel):
    """
    Panel that displays all available tools (like create target,
    resize, zoom in, etc)
    """
    # Restrict size of icons to 50 pixels height
    SIZE_ICON = 50.0
    
    def __init__(self, *args, **kwargs):
        wx.Panel.__init__(self, *args, **kwargs)

        # Instance vars
        self.images = []
        self.iconsdir = os.path.join('imgs','icons')
        
        # vars for button states
        self.state_zoomin = False
        self.state_zoomout = False
        self.state_addtarget = False
        self.state_select = False
        self.state_autodetect = False
        
        self._populate_icons(self.iconsdir)
        self._bind_events()
        
        self.sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.sizer.Add(self.btn_zoomin)
        self.sizer.Add(self.btn_zoomout)
        self.sizer.Add(self.btn_addtarget)
        self.sizer.Add(self.btn_select)
        self.sizer.Add(self.btn_autodetect)
        self.SetSizer(self.sizer)
        self.SetAutoLayout(1)
        
        ### PubSub Subscribing
        Publisher().subscribe(self._enter_autodetect, "signals.enter_autodetect")
        Publisher().subscribe(self._cancel_autodetect, "signals.cancel_autodetect")
        Publisher().subscribe(self._leave_autodetect_verify, "signals.leave_autodetect_verify")
        Publisher().subscribe(self._clear_buttons, "signals.Toolbar.clear_buttons")

    def _bind_events(self):
        self.btn_zoomin.Bind(wx.EVT_BUTTON, self.onButton_zoomin)
        self.btn_zoomin.Bind(wx.EVT_ENTER_WINDOW, self.onEnter_zoomin)
        self.btn_zoomin.Bind(wx.EVT_LEAVE_WINDOW, self.onLeave_zoomin)
        self.btn_zoomout.Bind(wx.EVT_BUTTON, self.onButton_zoomout)
        self.btn_zoomout.Bind(wx.EVT_ENTER_WINDOW, self.onEnter_zoomout)
        self.btn_zoomout.Bind(wx.EVT_LEAVE_WINDOW, self.onLeave_zoomout)
        self.btn_addtarget.Bind(wx.EVT_BUTTON, self.onButton_addtarget)
        self.btn_addtarget.Bind(wx.EVT_ENTER_WINDOW, self.onEnter_addtarget)
        self.btn_addtarget.Bind(wx.EVT_LEAVE_WINDOW, self.onLeave_addtarget)
        self.btn_select.Bind(wx.EVT_BUTTON, self.onButton_select)
        self.btn_select.Bind(wx.EVT_ENTER_WINDOW, self.onEnter_select)
        self.btn_select.Bind(wx.EVT_LEAVE_WINDOW, self.onLeave_select)
        self.btn_autodetect.Bind(wx.EVT_BUTTON, self.onButton_autodetect)
        self.btn_autodetect.Bind(wx.EVT_ENTER_WINDOW, self.onEnter_autodetect)
        self.btn_autodetect.Bind(wx.EVT_LEAVE_WINDOW, self.onLeave_autodetect)
        
    def _resize_icons(self, iconpaths):
        """ Rescale all icon images to have height Toolbar.SIZE_ICON """
        bitmaps = {}
        for dirpath, dirnames, filenames in os.walk(iconpaths):
            for imgfile in [x for x in filenames if util_gui.is_image_ext(x)]:
                imgpath = os.path.join(dirpath, imgfile)
                wx_img = wx.Image(imgpath, wx.BITMAP_TYPE_ANY)
                c = wx_img.GetHeight() / ToolBar.SIZE_ICON
                wx_img = wx_img.Scale(wx_img.GetWidth() / c, wx_img.GetHeight() / c, wx.IMAGE_QUALITY_HIGH)
                bitmaps[util_gui.get_filename(imgpath)] = wx_img.ConvertToBitmap()
        return bitmaps
       
    def _populate_icons(self, iconsdir):
        bitmaps = self._resize_icons(iconsdir)
        self.bitmaps = bitmaps
        
        zoomin_unsel = bitmaps['zoomin_unsel']
        zoomin_sel = bitmaps['zoomin_sel']
        zoomout_unsel = bitmaps['zoomout_unsel']
        zoomout_sel = bitmaps['zoomout_sel']
        addtarget_unsel = bitmaps['addtarget_unsel']
        addtarget_sel = bitmaps['addtarget_sel']
        select_sel = bitmaps['select_sel']
        select_unsel = bitmaps['select_unsel']
        autodetect_sel = bitmaps['autodetect_sel']
        autodetect_unsel = bitmaps['autodetect_unsel']
        self.btn_zoomin = wx.BitmapButton(self, bitmap=zoomin_unsel,
                                           id=wx.ID_ZOOM_IN,
                                           size=(zoomin_unsel.GetWidth()+8,
                                                 zoomin_unsel.GetHeight()+8),
                                          name='btn_zoomin')
        self.btn_zoomout = wx.BitmapButton(self, bitmap=zoomout_unsel,
                                            id=wx.ID_ZOOM_OUT,
                                            size=(zoomout_unsel.GetWidth()+8,
                                                  zoomout_unsel.GetHeight()+8),
                                           name='btn_zoomout')
        self.btn_addtarget = wx.BitmapButton(self, bitmap=addtarget_unsel,
                                       id=wx.ID_ANY,
                                       size=(addtarget_unsel.GetWidth()+8,
                                             addtarget_unsel.GetHeight()+8),
                                       name='btn_addtarget')
        self.btn_select = wx.BitmapButton(self, bitmap=select_unsel,
                                          id=wx.ID_ANY,
                                          size=(select_unsel.GetWidth()+8,
                                                select_unsel.GetHeight()+8),
                                          name='btn_select')
        self.btn_autodetect = wx.BitmapButton(self, bitmap=autodetect_unsel,
                                              id=wx.ID_ANY,
                                              size=(autodetect_unsel.GetWidth()+8,
                                                    autodetect_unsel.GetHeight()+8),
                                              name='btn_autodetect')

    def clear_btns(self):
        self.state_zoomin = False
        self.state_zoomout = False
        self.state_addtarget = False
        self.state_select = False
        self.state_autodetect = False
        self.btn_zoomin.SetBitmapLabel(self.bitmaps['zoomin_unsel'])
        self.btn_zoomout.SetBitmapLabel(self.bitmaps['zoomout_unsel'])
        self.btn_addtarget.SetBitmapLabel(self.bitmaps['addtarget_unsel'])
        self.btn_select.SetBitmapLabel(self.bitmaps['select_unsel'])
        self.btn_autodetect.SetBitmapLabel(self.bitmaps['autodetect_unsel'])
        
    #### PubSub Callbacks
    def _enter_autodetect(self, msg):
        self.disable_buttons()
    def _cancel_autodetect(self, msg):
        self.enable_buttons()
    def _leave_autodetect_verify(self, msg):
        self.enable_buttons()
    def _clear_buttons(self, msg):
        self.clear_btns()
        
    def select_button(self, name):
        mapping = {'addtarget': self.btn_addtarget,
                   'select': self.btn_select,
                   'autodetect': self.btn_autodetect,
                   'zoomin': self.btn_zoomin,
                   'zoomout': self.btn_zoomout}
        sel_bitmap = self.bitmaps[name+"_sel"]
        mapping[name].SetBitmapLabel(sel_bitmap)
        
    def enable_buttons(self):
        self.btn_zoomin.Enable()
        self.btn_zoomout.Enable()
        self.btn_addtarget.Enable()
        self.btn_select.Enable()
        self.btn_autodetect.Enable()
        self.clear_btns()
    def disable_buttons(self, flag=None):
        if flag != "allow_zoom":
            self.btn_zoomin.Disable()
            self.btn_zoomout.Disable()
        self.btn_addtarget.Disable()
        self.btn_select.Disable()
        self.btn_autodetect.Disable()        
        
    #### Event handling
    def onButton_zoomin(self, event):
        if self.state_zoomin:
            self.state_zoomin = False
            self.btn_zoomin.SetBitmapLabel(self.bitmaps['zoomin_unsel'])
            Publisher().sendMessage("signals.BallotScreen.update_state", BallotScreen.STATE_IDLE)
        else:
            self.clear_btns()
            self.state_zoomin = True
            self.btn_zoomin.SetBitmapLabel(self.bitmaps['zoomin_sel'])
            Publisher().sendMessage("signals.BallotScreen.update_state", BallotScreen.STATE_ZOOM_IN)
        event.Skip()
    def onButton_zoomout(self, event):
        if self.state_zoomout:
            self.state_zoomout = False
            self.btn_zoomout.SetBitmapLabel(self.bitmaps['zoomout_unsel'])
            Publisher().sendMessage("signals.BallotScreen.update_state", BallotScreen.STATE_IDLE)
        else:
            self.clear_btns()
            self.state_zoomout = True
            self.btn_zoomout.SetBitmapLabel(self.bitmaps['zoomout_sel'])
            Publisher().sendMessage("signals.BallotScreen.update_state", BallotScreen.STATE_ZOOM_OUT)
        event.Skip()
    def onButton_addtarget(self, event):
        if self.state_addtarget:
            self.state_addtarget = False
            self.btn_addtarget.SetBitmapLabel(self.bitmaps['addtarget_unsel'])
            Publisher().sendMessage("signals.BallotScreen.update_state", BallotScreen.STATE_IDLE)
        else:
            self.clear_btns()
            self.state_addtarget = True
            self.btn_addtarget.SetBitmapLabel(self.bitmaps['addtarget_sel'])
            Publisher.sendMessage("signals.BallotScreen.update_state", BallotScreen.STATE_ADD_TARGET)
        event.Skip()
    def onButton_select(self, event):
        if self.state_select:
            self.state_select = False
            self.btn_select.SetBitmapLabel(self.bitmaps['select_unsel'])
            Publisher().sendMessage("signals.BallotScreen.update_state", BallotScreen.STATE_IDLE)
        else:
            self.clear_btns()
            self.state_select = True
            self.btn_select.SetBitmapLabel(self.bitmaps['select_sel'])
            Publisher().sendMessage("signals.BallotScreen.update_state", BallotScreen.STATE_MODIFY)
        event.Skip()
    def onButton_autodetect(self, event):
        if self.state_autodetect:
            self.state_autodetect = False
            self.btn_autodetect.SetBitmapLabel(self.bitmaps['autodetect_unsel'])
        else:
            self.clear_btns()
            self.state_autodetect = True
            self.btn_autodetect.SetBitmapLabel(self.bitmaps['autodetect_sel'])
        #auto_panel = Autodetect_Panel(self)
        Publisher().sendMessage("signals.enter_autodetect", None)
        #f.Show()
        event.Skip()
        
    def onEnter_zoomin(self, event):
        Publisher().sendMessage('signals.StatusBar.push', "Zoom into the opened image.")
        event.Skip()
    def onLeave_zoomin(self, event):
        Publisher().sendMessage('signals.StatusBar.pop', None)
        event.Skip()
    def onEnter_zoomout(self, event):
        Publisher().sendMessage('signals.StatusBar.push', "Zoom out of the opened image.")
        event.Skip()
    def onLeave_zoomout(self, event):
        Publisher().sendMessage('signals.StatusBar.pop', None)
        event.Skip()
    def onEnter_addtarget(self, event):
        Publisher().sendMessage('signals.StatusBar.push', "Create new voting targets.")
        event.Skip()
    def onLeave_addtarget(self, event):
        Publisher().sendMessage('signals.StatusBar.pop', None)
        event.Skip()
    def onEnter_select(self, event):
        Publisher().sendMessage('signals.StatusBar.push', 'Select voting targets.')
        event.Skip()
    def onLeave_select(self, event):
        Publisher().sendMessage('signals.StatusBar.pop', None)
        event.Skip()
    def onEnter_autodetect(self, event):
        Publisher().sendMessage('signals.StatusBar.push', 'Autodetect Voting Targets.')
        event.Skip()
    def onLeave_autodetect(self, event):
        Publisher().sendMessage('signals.StatusBar.pop', None)
        event.Skip()
        
class Dummy(wx.StaticText):
    """
    A dummy widget, used to fill in gaps in Grid sizer
    """
    def __init__(self, parent, *args, **kwargs):
        wx.StaticText.__init__(self, parent, label='', *args, **kwargs)
        

def main():
    app = wx.App(False)
    frame = MainFrame(None)
    frame.Show()
    #wx.lib.inspection.InspectionTool().Show()
    app.MainLoop()
    
if __name__ == '__main__':
    main()
