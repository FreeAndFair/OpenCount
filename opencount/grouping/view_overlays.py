import sys, os, pdb, pickle, time, traceback
import wx
from wx.lib.scrolledpanel import ScrolledPanel as wxScrolledPanel

import common
from verify_overlays import VerifyPanel

"""
A quick-and-dirty widget/command-line tool to visualize a min/max
overlay of a set of (grouped) images.

Usage:
    $ python view_overlays IMGSDIR_0 IMGSDIR_1 ... IMGSDIR_N

where each IMGSDIR_i is a separate 'group'.
"""

class ViewOverlays(wxScrolledPanel):
    def __init__(self, parent, *args, **kwargs):
        print parent
        wxScrolledPanel.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        sizer = wx.BoxSizer(wx.VERTICAL)
        self.verifypanel = VerifyPanel(self, verifymode=VerifyPanel.MODE_YESNO2)
        sizer.Add(self.verifypanel, proportion=1, flag=wx.EXPAND)
        self.SetSizer(sizer)
        self.Fit()
        
    def start(self, imggroups):
        """ Displays the images in IMGGROUPS. 
        Input:
            list IMGGROUPS: A list of lists of imagepaths, where each 
                sublist is considered one 'group':
                    [[imgpath_0i, ...], [imgpath_1i, ...], ...]
        """
        gl_record = []
        groups = []
        elements_map = {} # maps {int gl_idx: elements}
        for i, imgpaths in enumerate(imggroups):
            elements = [[imgpath, None, imgpath] for imgpath in imgpaths]
            elements_map[i] = elements
            gl_record.append(common.make_grouplabel(('group', i)))
        groups = []
        for gl_idx, elements in elements_map.iteritems():
            other_gl_idxs = list(set(elements_map.keys()))
            other_gl_idxs.remove(gl_idx)
            rlist = [gl_idxA for gl_idxA in other_gl_idxs]
            rlist = tuple([gl_idx] + rlist)
            for element in elements:
                element[1] = rlist
            groups.append(common.GroupClass(elements))

        self.verifypanel.start(groups, None, None, grouplabel_record=gl_record)

class ViewOverlaysFrame(wx.Frame):
    def __init__(self, parent, imgpaths, *args, **kwargs):
        wx.Frame.__init__(self, parent, *args, **kwargs)
        self.viewoverlays = ViewOverlays(self)
        self.imgpaths = imgpaths
        self.viewoverlays.start(imgpaths)

def is_img_ext(p):
    return os.path.splitext(p)[1].lower() in ('.png', '.jpg', '.jpeg', '.bmp')

def main():
    args = sys.argv[1:]
    
    imggroups = []
    for imgsdir_i in args:
        imggroup = []
        for dirpath, dirnames, filenames in os.walk(imgsdir_i):
            for imgname in [f for f in filenames if is_img_ext(f)]:
                imggroup.append(os.path.join(dirpath, imgname))
        if imggroup:
            imggroups.append(imggroup)
    app = wx.App(False)
    frame = ViewOverlaysFrame(None, imggroups)
    frame.Maximize()
    frame.Show()
    app.MainLoop()

if __name__ == '__main__':
    main()

