import os, sys, traceback, time, pdb
try:
    import cPickle as pickle
except:
    import pickle

from os.path import join as pathjoin

import wx
from wx.lib.scrolledpanel import ScrolledPanel

import cv, numpy as np, scipy, scipy.misc, Image
import make_overlays
import cluster_imgs

sys.path.append('..')
import util

class ViewOverlaysPanel(ScrolledPanel):
    """ Class that contains both a Header, a ViewOverlays, and a Footer. """
    def __init__(self, parent, *args, **kwargs):
        ScrolledPanel.__init__(self, parent, *args, **kwargs)
        self.set_classes()
        self.init_ui()

    def set_classes(self):
        self.headerCls = wx.Panel
        self.overlaypanelCls = ViewOverlays
        self.footerCls = wx.Panel

    def init_ui(self):
        self.header = self.headerCls(self)
        self.overlaypanel = self.overlaypanelCls(self)
        self.footer = self.footerCls(self)

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.header, border=5, flag=wx.ALL)
        self.sizer.Add(self.overlaypanel, border=5, proportion=1, flag=wx.EXPAND | wx.ALL)
        self.sizer.Add(self.footer, border=5, flag=wx.ALL)

        self.SetSizer(self.sizer)
        self.Layout()
        self.SetupScrolling()

    def start(self, *args, **kwargs):
        self.overlaypanel.start(*args, **kwargs)
        self.Layout()
        self.SetupScrolling()

    def save_session(self, *args, **kwargs):
        self.overlaypanel.save_session(*args, **kwargs)
    def export_results(self, *args, **kwargs):
        self.overlaypanel.export_results(*args, **kwargs)

class ViewOverlays(ScrolledPanel):
    def __init__(self, parent, *args, **kwargs):
        ScrolledPanel.__init__(self, parent, *args, **kwargs)

        # list GROUPS: [obj GROUP_i, ...]
        self.groups = None

        # dict BBS_MAP: maps {(tag, str imgpath): (x1,y1,x2,y2)}
        self.bbs_map = None
        
        # IDX: Current idx into self.GROUPS that we are displaying
        self.idx = None

        # float RSZFAC: Current Image scale of displayed images
        self.rszfac = 1.0

        # Numpy versions of the original-sized min/max images (useful
        # to keep track of to scale larger/smaller)
        self.minimg_np_orig = None
        self.maximg_np_orig = None

        self.init_ui()

    def overlays_layout_vert(self):
        """ Layout the images in a single vertical column. """
        self.sizer_overlays.SetOrientation(wx.VERTICAL)
        self.sizer_overlays_voted.SetOrientation(wx.VERTICAL)
    def overlays_layout_horiz(self):
        """ Layout the images in a single horizontal row. """
        self.sizer_overlays.SetOrientation(wx.HORIZONTAL)
        self.sizer_overlays_voted.SetOrientation(wx.HORIZONTAL)
    def set_patch_layout(self, orient='horizontal'):
        """ Change the orientation of the images. Either arrange 
        'horizontal', or stack 'vertical'.
        """
        if orient == 'horizontal':
            sizer = self.overlays_layout_horiz()
        else:
            sizer = self.overlays_layout_vert()
        self.Layout()
        self.Refresh()

    def init_ui(self):
        stxt_grplabel = wx.StaticText(self, label="Current Group Label: ")
        self.txt_grplabel = wx.StaticText(self, label='')
        txt_0 = wx.StaticText(self, label="Number of images in group: ")
        self.txtctrl_num_elements = wx.TextCtrl(self, value='0')
        self.listbox_groups = wx.ListBox(self, size=(200, 300))
        self.listbox_groups.Hide()
        self.listbox_groups.Bind(wx.EVT_LISTBOX, self.onListBox_groups)
        self.btn_showhidelistbox = wx.Button(self, label="Show List")
        self.btn_showhidelistbox.Bind(wx.EVT_BUTTON, self.onButton_showhidelistbox)
        sizer_grplabel = wx.BoxSizer(wx.HORIZONTAL)
        self.sizer_grplabel = sizer_grplabel
        # Align self.txt_grplabel, self.txtctrl_num_elements as a column
        _maxtxtlen = max(self.GetTextExtent(stxt_grplabel.GetLabel())[0],
                         self.GetTextExtent(txt_0.GetLabel())[0])
        _pad = 30

        sizer_grplabel.AddMany([(stxt_grplabel,), 
                                (_maxtxtlen - self.GetTextExtent(stxt_grplabel.GetLabel())[0] + _pad, 0),
                                (self.txt_grplabel,)])
        sizer_numimgs = wx.BoxSizer(wx.HORIZONTAL)
        sizer_numimgs.AddMany([(txt_0,), 
                               (_maxtxtlen - self.GetTextExtent(txt_0.GetLabel())[0] + _pad, 0),
                               (self.txtctrl_num_elements,)])
        sizer_groups = wx.BoxSizer(wx.VERTICAL)
        self.sizer_groups = sizer_groups
        sizer_groups.AddMany([(sizer_grplabel,), (sizer_numimgs,), (self.listbox_groups,), (self.btn_showhidelistbox,)])

        st1 = wx.StaticText(self, -1, "min: ")
        st2 = wx.StaticText(self, -1, "max: ")
        st3 = wx.StaticText(self, -1, "Looks like? ")
        self.st1, self.st2, self.st3 = st1, st2, st3

        self.minOverlayImg = wx.StaticBitmap(self, bitmap=wx.EmptyBitmap(1, 1))
        self.maxOverlayImg = wx.StaticBitmap(self, bitmap=wx.EmptyBitmap(1, 1))
        self.txt_exemplarTag = wx.StaticText(self, label='')
        self.exemplarImg = wx.StaticBitmap(self, bitmap=wx.EmptyBitmap(1, 1))

        maxTxtW = max([txt.GetSize()[0] for txt in (st1, st2, st3)]) + 20

        sizer_overlays = wx.BoxSizer(wx.HORIZONTAL)
        self.sizer_overlays = sizer_overlays
        self.sizer_overlays_voted = wx.BoxSizer(wx.HORIZONTAL)

        self.sizer_min = wx.BoxSizer(wx.VERTICAL)
        self.sizer_min.AddMany([(st1,), (self.minOverlayImg,)])
        self.sizer_max = wx.BoxSizer(wx.VERTICAL)
        self.sizer_max.AddMany([(st2,), (self.maxOverlayImg,)])

        self.sizer_innerattrpatch = wx.BoxSizer(wx.VERTICAL)
        self.sizer_innerattrpatch.AddMany([(self.txt_exemplarTag,), (self.exemplarImg,)])

        self.sizer_attrpatch = wx.BoxSizer(wx.VERTICAL)
        self.sizer_attrpatch.AddMany([(st3,), (self.sizer_innerattrpatch,)])

        self.sizer_overlays_voted.AddMany([(self.sizer_min,), ((50, 50),), (self.sizer_max,)])
        self.sizer_overlays.AddMany([(self.sizer_overlays_voted,), ((50, 50),),
                                     (self.sizer_attrpatch, 0, wx.ALIGN_CENTER)])
        self.set_patch_layout('horizontal')

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(sizer_groups)
        self.sizer.Add(sizer_overlays, flag=wx.ALIGN_CENTER)

        self.SetSizer(self.sizer)
        self.Layout()
        self.SetupScrolling()
        
    def select_group(self, idx):
        if idx < 0 or idx >= len(self.groups):
            return None
        self.idx = idx
        self.listbox_groups.SetSelection(self.idx)
        group = self.groups[idx]
        self.txt_grplabel.SetLabel(group.tag)
        self.txtctrl_num_elements.SetValue(str(len(group.imgpaths)))

        # OVERLAY_MIN, OVERLAY_MAX are IplImages
        if self.bbs_map:
            curtag = self.get_current_group().tag
            bbs_map_v2 = {}
            for (tag, imgpath), (x1,y1,x2,y2) in self.bbs_map.iteritems():
                if curtag == tag:
                    bbs_map_v2[imgpath] = (x1,y1,x2,y2)
        else:
            bbs_map_v2 = {}
        overlay_min, overlay_max = group.get_overlays(bbs_map=bbs_map_v2)

        minimg_np = iplimage2np(overlay_min)
        maximg_np = iplimage2np(overlay_max)
        
        self.minimg_np_orig = gray2rgb_np(minimg_np)
        self.maximg_np_orig = gray2rgb_np(maximg_np)

        min_bitmap = NumpyToWxBitmap(minimg_np)
        max_bitmap = NumpyToWxBitmap(maximg_np)

        self.minOverlayImg.SetBitmap(min_bitmap)
        self.maxOverlayImg.SetBitmap(max_bitmap)

        orientation, rszfac = self.compute_img_layout(overlay_min, overlay_max)
        self.apply_img_layout(orientation, rszfac)
        
        self.Layout()
        self.SetupScrolling()

        return self.idx

    def compute_img_layout(self, minimg, maximg):
        """ Computes the best layout considering the min and max
        images. 
        Returns (str orientation, float rescale_factor).
        """
        # Determine the best layout+autoscaling for these images.
        w_space, h_space = map(float, self.GetClientSize())
        h_space -= max(1, self.sizer_groups.GetSize()[1])

        w_min, h_min = cv.GetSize(minimg)
        w_max, h_max = cv.GetSize(maximg)

        # E.g If C_HORIZ = 0.5, then we downscale by 2.0 (rescale by 0.5)
        MIN_RSZ_FAC = 0.5
        MAX_RSZ_FAC = 10.0
        c_horiz = w_space / (w_min+w_max)
        if (h_min * c_horiz) >= h_space:
            c_horiz = h_space / h_min
        c_vert = h_space / (h_min+h_max)
        if (w_min * c_vert) >= w_space:
            c_vert = w_space / w_min

        _w0, _h0 = self.st1.GetSize()
        _w1, _h1 = self.st2.GetSize()

        if c_horiz > c_vert:
            c_out = c_horiz
            return 'horizontal', min(max(c_out, MIN_RSZ_FAC), MAX_RSZ_FAC)
        else:
            _added_amt = (_h0 + _h1)
            _added_frac = _added_amt / float(h_space)
            c_out = c_vert - _added_frac
            return 'vertical', min(max(c_out, MIN_RSZ_FAC), MAX_RSZ_FAC)

    def apply_img_layout(self, orientation, rszfac):
        if orientation != None:
            self.set_patch_layout(orientation)
        if rszfac != None:
            self.rescale_images(rszfac)
    
    def show_larger(self, amt=0.2):
        self.rescale_images(self.rszfac + amt)
        self.Layout()
    def show_smaller(self, amt=0.2):
        self.rescale_images(self.rszfac - amt)
        self.Layout()

    def rescale_images(self, rszfac):
        """ Rescales the images by RSZFAC. """
        if rszfac <= 0.01:
            return self.rszfac
        min_resized = scipy.misc.imresize(self.minimg_np_orig, rszfac, interp='bicubic')
        max_resized = scipy.misc.imresize(self.maximg_np_orig, rszfac, interp='bicubic')

        self.minOverlayImg.SetBitmap(util.np2wxBitmap(min_resized))
        self.maxOverlayImg.SetBitmap(util.np2wxBitmap(max_resized))

        self.rszfac = rszfac
        self.Layout()
        self.SetupScrolling()
        return rszfac

    def get_current_group(self):
        return self.groups[self.idx]
        
    def add_group(self, group):
        self.groups.insert(0, group)
        label = "{0} -> {1} elements".format(group.tag, len(group.imgpaths))
        self.listbox_groups.Insert(label, 0)
    def remove_group(self, group):
        idx = self.groups.index(group)
        self.groups.pop(idx)
        self.listbox_groups.Delete(idx)
        if self.groups:
            self.select_group(0)
        else:
            # No more groups to display, so do some cleanup
            self.handle_nomoregroups()

    def update_listbox(self):
        """ Updates the entries in the self.listbox_groups widget,
        recomputing group sizes. In particular, removes empty groups.
        """
        # 1.) First, remove all empty groups
        remove_groups = []
        for i, group in enumerate(self.groups):
            if not group.imgpaths:
                remove_groups.append(group)
        for group in remove_groups:
            self.remove_group(group)
        # 2.) Now, update all entries
        newlabels = []
        for i, group in enumerate(self.groups):
            label = "{0} -> {1} elements".format(group.tag, len(group.imgpaths))
            newlabels.append(label)
        self.listbox_groups.SetItems(newlabels)

    def handle_nomoregroups(self):
        """ Called when there are no more groups in the queue. """
        self.Disable()

    def start(self, imgpath_groups, do_align=False, bbs_map=None, stateP=None):
        """
        Input:
            dict IMGPATH_GROUPS: {str grouptag: [imgpath_i, ...]}
            dict BBS_MAP: maps {(tag, str imgpath): (x1,y1,x2,y2)}. Used to optionally
                overlay subregions of images in IMGPATH_GROUPS, rather than
                extracting+saving each subregion.
        """
        self.stateP = stateP
        if not self.restore_session():
            self.groups = []
            self.bbs_map = bbs_map if bbs_map != None else {}
            for (tag, imgpaths) in imgpath_groups.iteritems():
                group = Group(imgpaths, tag=tag, do_align=do_align)
                self.add_group(group)
        self.select_group(0)

    def restore_session(self):
        try:
            print 'trying to load:', self.stateP
            state = pickle.load(open(self.stateP, 'rb'))
            groups = state['groups']
            self.groups = []
            for group_dict in groups:
                self.add_group(Group.unmarshall(group_dict))
            self.bbs_map = state['bbs_map']
            return state
        except:
            traceback.print_exc()
            return False
    def create_state_dict(self):
        state = {'groups': [g.marshall() for g in self.groups], 
                 'bbs_map': self.bbs_map}
        return state
    def save_session(self):
        try:
            state = self.create_state_dict()
            pickle.dump(state, open(self.stateP, 'wb'))
            return state
        except:
            return False
        
    def onListBox_groups(self, evt):
        if evt.Selection == -1:
            # Some ListBox events fire when nothing is selected (i.e. -1)
            return
        idx = self.listbox_groups.GetSelection()
        if self.groups:
            self.select_group(idx)

    def onButton_showhidelistbox(self, evt):
        if self.listbox_groups.IsShown():
            self.listbox_groups.Hide()
            self.btn_showhidelistbox.SetLabel("Show List")
        else:
            self.listbox_groups.Show()
            self.btn_showhidelistbox.SetLabel("Hide List")
        self.Layout()
        self.SetupScrolling()

class SplitOverlaysPanel(ViewOverlaysPanel):
    """ Class that contains both a Header, a SplitOverlays, and a Button
    Toolbar Footer. """
    def set_classes(self):
        ViewOverlaysPanel.set_classes(self)
        self.headerCls = wx.Panel
        self.overlaypanelCls = SplitOverlays
        self.footerCls = SplitOverlaysFooter

class SplitOverlaysFooter(wx.Panel):
    def __init__(self, parent, *args, **kwargs):
        wx.Panel.__init__(self, parent, *args, **kwargs)

        self.init_ui()
    def init_ui(self):
        btn_split = wx.Button(self, label="Split...")
        btn_split.Bind(wx.EVT_BUTTON, self.onButton_split)
        btn_setsplitmode = wx.Button(self, label="Set Split Mode...")
        btn_setsplitmode.Bind(wx.EVT_BUTTON, self.onButton_setsplitmode)
        sizer_split = wx.BoxSizer(wx.VERTICAL)
        sizer_split.AddMany([(btn_split,0,wx.ALIGN_CENTER), (btn_setsplitmode,0,wx.ALIGN_CENTER)])

        btn_larger = wx.Button(self, label="Show Larger")
        btn_larger.Bind(wx.EVT_BUTTON, self.onButton_showlarger)
        btn_smaller = wx.Button(self, label="Show Smaller")
        btn_smaller.Bind(wx.EVT_BUTTON, self.onButton_showsmaller)
        sizer_scaling = wx.BoxSizer(wx.VERTICAL)
        sizer_scaling.AddMany([(btn_larger,0,wx.ALIGN_CENTER), ((10,10),),(btn_smaller,0,wx.ALIGN_CENTER)])

        self.btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.btn_sizer.Add(sizer_split)
        self.btn_sizer.Add(sizer_scaling)

        self.sizer = wx.BoxSizer(wx.VERTICAL)

        self.sizer.Add(self.btn_sizer, proportion=0, border=10, flag=wx.ALL)

        self.SetSizer(self.sizer)
        self.Layout()

    def onButton_split(self, evt):
        self.GetParent().overlaypanel.do_split()
    def onButton_setsplitmode(self, evt):
        self.GetParent().overlaypanel.do_setsplitmode()
    def onButton_showlarger(self, evt):
        self.GetParent().overlaypanel.show_larger()
    def onButton_showsmaller(self, evt):
        self.GetParent().overlaypanel.show_smaller()

class SplitOverlays(ViewOverlays):
    def __init__(self, parent, *args, **kwargs):
        ViewOverlays.__init__(self, parent, *args, **kwargs)

        self.splitmode = 'kmeans'

    def start(self, imgpath_groups, do_align=False, bbs_map=None, stateP=None):
        """
        Input:
            dict IMGPATH_GROUPS: {str grouptag: [imgpath_i, ...]}
            dict BBS_MAP: maps {(tag, str imgpath): (x1,y1,x2,y2)}. Used to optionally
                overlay subregions of images in IMGPATH_GROUPS, rather than
                extracting+saving each subregion.
        """
        self.stateP = stateP
        if not self.restore_session():
            self.groups = []
            self.bbs_map = bbs_map if bbs_map != None else {}
            for (tag, imgpaths) in imgpath_groups.iteritems():
                group = SplitGroup(imgpaths, tag=tag, do_align=do_align)
                self.add_group(group)
        self.select_group(0)

    def do_split(self):
        curgroup = self.get_current_group()
        groups = curgroup.split(mode=self.splitmode)
        for group in groups:
            self.add_group(group)
        self.remove_group(curgroup)

    def do_setsplitmode(self):
        if not isinstance(self.get_current_group(), VerifyGroup):
            disabled = [ChooseSplitModeDialog.ID_RANKEDLIST]
        else:
            disabled = None
        dlg = ChooseSplitModeDialog(self, disable=disabled)
        status = dlg.ShowModal()
        if status == wx.ID_CANCEL:
            return
        splitmode = 'kmeans'
        if status == ChooseSplitModeDialog.ID_MIDSPLIT:
            splitmode = 'midsplit'
        elif status == ChooseSplitModeDialog.ID_RANKEDLIST:
            splitmode = 'rankedlist'
        elif status == ChooseSplitModeDialog.ID_KMEANS:
            splitmode = 'kmeans'
        elif status == ChooseSplitModeDialog.ID_PCA_KMEANS:
            splitmode = 'pca_kmeans'
        elif status == ChooseSplitModeDialog.ID_KMEANS2:
            splitmode = 'kmeans2'
        elif status == ChooseSplitModeDialog.ID_KMEDIODS:
            splitmode = 'kmediods'
        self.splitmode = splitmode

class VerifyOverlaysPanel(SplitOverlaysPanel):
    """ Class that contains both a Header, a VerifyOverlays, and a Button
    Toolbar Footer. """
    def set_classes(self):
        SplitOverlaysPanel.set_classes(self)
        self.overlaypanelCls = VerifyOverlays
        self.footerCls = VerifyOverlaysFooter

class VerifyOverlaysFooter(SplitOverlaysFooter):
    def init_ui(self):
        SplitOverlaysFooter.init_ui(self)
        btn_matches = wx.Button(self, label="Matches")
        btn_matches.Bind(wx.EVT_BUTTON, self.onButton_matches)
        self.btn_manual_relabel = wx.Button(self, label="Manually Relabel...")
        self.btn_manual_relabel.Bind(wx.EVT_BUTTON, self.onButton_manual_relabel)

        btn_nextexmpl = wx.Button(self, label="Next Exemplar Patch")
        btn_nextexmpl.Bind(wx.EVT_BUTTON, self.onButton_nextexmpl)
        btn_prevexmpl = wx.Button(self, label="Previous Exemplar Patch")
        btn_prevexmpl.Bind(wx.EVT_BUTTON, self.onButton_prevexmpl)
        txt0 = wx.StaticText(self, label="Current Exemplar: ")
        self.txt_curexmplidx = wx.StaticText(self, label='')
        txt1 = wx.StaticText(self, label=" / ")
        self.txt_totalexmplidxs = wx.StaticText(self, label='')
        sizer_txtexmpls = wx.BoxSizer(wx.HORIZONTAL)
        sizer_txtexmpls.AddMany([(txt0,), (self.txt_curexmplidx,), (txt1,),
                                 (self.txt_totalexmplidxs,)])
        self.sizer_exmpls = wx.BoxSizer(wx.VERTICAL)
        self.sizer_exmpls.AddMany([(sizer_txtexmpls,), (btn_nextexmpl,), (btn_prevexmpl,)])

        self.btn_sizer.AddMany([(btn_matches,), (self.btn_manual_relabel,), (self.sizer_exmpls,)])

        txt_curlabel0 = wx.StaticText(self, label="Current guess: ")
        self.txt_curlabel = wx.StaticText(self, label="")
        self.sizer_curlabel = wx.BoxSizer(wx.HORIZONTAL)
        self.sizer_curlabel.AddMany([(txt_curlabel0,), (self.txt_curlabel,)])
        self.sizer.Add(self.sizer_curlabel, proportion=0, flag=wx.ALIGN_CENTER)

        self.Layout()

    def onButton_matches(self, evt):
        self.GetParent().overlaypanel.do_matches()
    def onButton_manual_relabel(self, evt):
        self.GetParent().overlaypanel.do_manual_relabel()
    def onButton_nextexmpl(self, evt):
        self.GetParent().overlaypanel.do_nextexmpl()
    def onButton_prevexmpl(self, evt):
        self.GetParent().overlaypanel.do_prevexmpl()

class VerifyOverlays(SplitOverlays):
    def __init__(self, parent, *args, **kwargs):
        SplitOverlays.__init__(self, parent, *args, **kwargs)

        # dict self.EXEMPLAR_IMGPATHS: {str grouptag: [str exmpl_imgpath_i, ...]}
        self.exemplar_imgpaths = {}
        # self.RANKEDLIST_MAP: maps {str imgpath: (groupID_0, groupID_1, ...)}
        self.rankedlist_map = {}
        # self.FINISHED_GROUPS: maps {tag: [obj group_i, ...]}, where
        # tag is the group that the user finalized on.
        self.finished_groups = {}

        # self.EXMPLIDX_SEL: The exemplaridx that the user has currently selected
        self.exmplidx_sel = None

        # A numpy-version of the original-sized image (useful to keep
        # track of to scale larger/smaller)
        self.exemplarimg_np_orig = None

        # self.ONDONE: A callback function to call when verifying is done.
        self.ondone = None

    def start(self, imgpath_groups, group_exemplars, rlist_map, 
              do_align=False, bbs_map=None, ondone=None, auto_ondone=False,
              stateP=None):
        """
        Input:
            dict IMGPATH_GROUPS: {grouptag: [imgpath_i, ...]}
            dict GROUP_EXEMPLARS: maps {grouptag: [exmpl_imgpath_i, ...]}
            dict RLIST_MAP: maps {str imgpath: (groupID_0, ...)}
            dict BBS_MAP: maps {(grouptag, imgpath): (x1,y1,x2,y2)}
            fn ONDONE: Function that accepts one argument:
                dict {grouptag: [obj group_i, ...]}
            bool AUTO_ONDONE: If True, then when all groups are gone,
                this will immediately call the ondone function.
        """
        self.stateP = stateP
        self.auto_ondone = auto_ondone
        self.ondone = ondone
        if not self.restore_session():
            self.exemplar_imgpaths = group_exemplars
            self.groups = []
            self.bbs_map = bbs_map if bbs_map != None else {}
            self.possible_tags = set()
            self.rankedlist_map = rlist_map
            self.finished_groups = {}
            self.exmplidx_sel = 0
            for (tag, imgpaths) in imgpath_groups.iteritems():
                group = VerifyGroup(imgpaths, tag=tag, do_align=do_align)
                self.possible_tags.add(group.tag)
                if imgpaths:
                    self.add_group(group)
            self.possible_tags = tuple(self.possible_tags)
        if len(self.groups) == 0:
            self.handle_nomoregroups()
        else:
            self.select_group(0)

    def stop(self):
        # Do an export.
        self.save_session()
        self.export_results()

    def export_results(self):
        """ Calls the callback function and passes the verify_results
        off.
        """
        if self.ondone:
            verify_results = {} # maps {tag: [imgpath_i, ...]}
            for tag, groups in self.finished_groups.iteritems():
                for group in groups:
                    verify_results.setdefault(tag, []).extend(group.imgpaths)
            #self.ondone(verify_results)
            wx.CallAfter(self.ondone, verify_results)

    def restore_session(self):
        try:
            state = pickle.load(open(self.stateP, 'rb'))
            groups = state['groups']
            self.groups = []
            for group_dict in groups:
                self.add_group(VerifyGroup.unmarshall(group_dict))

            self.bbs_map = state['bbs_map']
            self.exemplar_imgpaths = state['exemplar_imgpaths']
            self.rankedlist_map = state['rankedlist_map']
            fingroups_in = state['finished_groups']
            fingroups_new = {}
            for tag, groups_marsh in fingroups_in.iteritems():
                fingroups_new[tag] = [VerifyGroup.unmarshall(gdict) for gdict in groups_marsh]
            self.finished_groups = fingroups_new
            self.quarantined_groups = [VerifyGroup.unmarshall(gdict) for gdict in state['quarantined_groups']]
            print '...Successfully loaded VerifyOverlays state...'
            return state
        except Exception as e:
            print '...Failed to load VerifyOverlays state...'
            return False
    def create_state_dict(self):
        state = SplitOverlays.create_state_dict(self)
        state['exemplar_imgpaths'] = self.exemplar_imgpaths
        state['rankedlist_map'] = self.rankedlist_map
        fingroups_out = {}
        for tag, groups in self.finished_groups.iteritems():
            fingroups_out[tag] = [g.marshall() for g in groups]
        state['finished_groups'] = fingroups_out
        state['quarantined_groups'] = [g.marshall() for g in self.quarantined_groups]
        return state

    def select_group(self, idx):
        if idx < 0 or idx >= len(self.groups):
            return None
        group = self.groups[idx]
        self.select_exmpl_group(group.tag, group.exmpl_idx)
        
        curidx = SplitOverlays.select_group(self, idx)
        if curidx == None:
            # Say, if IDX is invalid (maybe no more groups?)
            return None

        self.Layout()

    def compute_img_layout(self, minimg, maximg):
        """ Computes the best layout considering the min, max, and the
        exemplar image patches.
        Returns (str orientation, float rescale_factor).
        """
        # TODO: This method is a copy+paste of ViewOverlays.compute_img_layout,
        # which is frustrating, but, I couldn't see another way to do this
        # without a tricky refactor...

        # Assumes that select_exmpl_group() has already been called
        # with the exemplar-image that we are about to display
        w_space, h_space = map(float, self.GetClientSize())
        h_space -= max(1, self.sizer_groups.GetSize()[1])

        w_min, h_min = cv.GetSize(minimg)
        w_max, h_max = cv.GetSize(maximg)
        w_exm, h_exm = self.exemplarImg.GetSize()

        # E.g If C_HORIZ = 0.5, then we downscale by 2.0 (rescale by 0.5)
        MIN_RSZ_FAC = 0.5
        MAX_RSZ_FAC = 10.0
        c_horiz = w_space / (w_min+w_max+w_exm)
        if (h_min * c_horiz) >= h_space:
            c_horiz = h_space / h_min
        c_vert = h_space / (h_min+h_max+h_exm)
        if (w_min * c_vert) >= w_space:
            c_vert = w_space / w_min

        _w0, _h0 = self.st1.GetSize()
        _w1, _h1 = self.st2.GetSize()
        _w2, _h2 = self.st3.GetSize()
        _w3, _h3 = self.txt_exemplarTag.GetSize()

        if c_horiz > c_vert:
            _added_amt = 50 + 50 # Padding added between images
            _added_frac = _added_amt / float(w_space)
            c_out = c_horiz - _added_frac
            print "Chose Horizontal, c_out={0:.5f}".format(c_out)
            return 'horizontal', min(max(c_out, MIN_RSZ_FAC), MAX_RSZ_FAC)
        else:
            _added_amt = max(_h0, _h1, _h2+_h3) + 50 + 50 # Padding added btwn imgs
            _added_frac = _added_amt / float(h_space)
            c_out = c_vert - _added_frac
            print "Chose Vertical, c_out={0:.5f}".format(c_out)
            return 'vertical', min(max(c_out, MIN_RSZ_FAC), MAX_RSZ_FAC)

    def rescale_images(self, rszfac):
        if rszfac <= 0.01:
            return self.rszfac
        rszfac_out = SplitOverlays.rescale_images(self, rszfac)
        if not self.exemplarImg.IsShown():
            return rszfac_out
        exemplar_resized = scipy.misc.imresize(self.exemplarimg_np_orig, rszfac, interp='bicubic')
        self.exemplarImg.SetBitmap(util.np2wxBitmap(exemplar_resized))
        self.Layout()
        self.SetupScrolling()
        return rszfac_out

    def select_exmpl_group(self, grouptag, exmpl_idx):
        """ Displays the correct exemplar img patch on the screen. """
        if grouptag not in self.exemplar_imgpaths:
            print "...Invalid GroupTAG: {0}. Hiding exemplar-related UI components...".format(grouptag)
            self.sizer_attrpatch.ShowItems(False)
            self.Layout()
            self.SetupScrolling()
            return
        if not self.exemplarImg.IsShown():
            self.sizer_attrpatch.ShowItems(True)
        exemplar_paths = self.exemplar_imgpaths[grouptag]
        if exmpl_idx < 0 or exmpl_idx >= len(exemplar_paths):
            print "...Invalid exmpl_idx: {0}...".format(exmpl_idx)
            return
        exemplar_npimg = scipy.misc.imread(exemplar_paths[exmpl_idx], flatten=True)
        self.exemplarimg_np_orig = gray2rgb_np(exemplar_npimg)

        exemplarImg_bitmap = NumpyToWxBitmap(exemplar_npimg)
        self.exmplidx_sel = exmpl_idx
        self.exemplarImg.SetBitmap(exemplarImg_bitmap)
        self.txt_exemplarTag.SetLabel(str(grouptag))
        self.GetParent().footer.txt_curexmplidx.SetLabel(str(exmpl_idx+1))
        self.GetParent().footer.txt_totalexmplidxs.SetLabel(str(len(exemplar_paths)))
        self.GetParent().footer.txt_curlabel.SetLabel(str(grouptag))
        self.Layout()
        self.SetupScrolling()

    def handle_nomoregroups(self):
        SplitOverlays.handle_nomoregroups(self)
        if self.auto_ondone:
            self.stop()

    def do_matches(self):
        """ Handles the scenario where the user says, "The current overlays
        indeed match and look fine."
        """
        curgroup = self.groups[self.idx]
        curtag = curgroup.tag
        self.finished_groups.setdefault(curtag, []).append(curgroup)
        self.remove_group(curgroup)
        self.Layout()
        self.SetupScrolling()

    def do_manual_relabel(self):
        """ The user clicked the 'Manual Relabel this Group' """
        dlg = ManualRelabelDialog(self, self.possible_tags)
        status = dlg.ShowModal()
        if status == wx.CANCEL:
            return
        sel_tag = dlg.tag
        self.select_exmpl_group(sel_tag, self.get_current_group().exmpl_idx)
    def do_nextexmpl(self):
        """ The user wants to see the next exemplar patch. """
        curtag = self.get_current_group().tag
        nextidx = self.exmplidx_sel + 1
        self.select_exmpl_group(curtag, nextidx)
    def do_prevexmpl(self):
        """ The user wants to see the previous exemplar patch. """
        curtag = self.get_current_group().tag
        previdx = self.exmplidx_sel - 1
        self.select_exmpl_group(curtag, previdx)

class VerifyOrFlagOverlaysPanel(VerifyOverlaysPanel):
    """ Class that contains both a Header, a VerifyOrFlagOverlays, and a Button
    Toolbar Footer. """
    def set_classes(self):
        VerifyOverlaysPanel.set_classes(self)
        self.overlaypanelCls = VerifyOrFlagOverlays
        self.footerCls = VerifyOrFlagOverlaysFooter

class VerifyOrFlagOverlaysFooter(VerifyOverlaysFooter):
    def init_ui(self):
        VerifyOverlaysFooter.init_ui(self)
        self.btn_quarantine = wx.Button(self, label="Quarantine")
        self.btn_quarantine.Bind(wx.EVT_BUTTON, self.onButton_quarantine)
        self.btn_sizer.Add(self.btn_quarantine)
        self.Layout()
    def onButton_quarantine(self, evt):
        self.GetParent().overlaypanel.do_quarantine()

class VerifyOrFlagOverlays(VerifyOverlays):
    """ A widget that lets you either verify overlays, or flag a group
    to set aside (Quarantine). 
    """
    def __init__(self, parent, *args, **kwargs):
        VerifyOverlays.__init__(self, parent, *args, **kwargs)

        # list self.QUARANTINED_GROUPS: List of [obj group_i, ...]
        self.quarantined_groups = None

    def start(self, *args, **kwargs):
        self.quarantined_groups = []
        VerifyOverlays.start(self, *args, **kwargs)

    def export_results(self):
        if self.ondone:
            verify_results = {} # maps {tag: [imgpath_i, ...]}
            for tag, groups in self.finished_groups.iteritems():
                for group in groups:
                    verify_results.setdefault(tag, []).extend(group.imgpaths)
            quarantined_results = []
            for group in self.quarantined_groups:
                quarantined_results.extend(group.imgpaths)
            #self.ondone(verify_results, quarantined_results)
            wx.CallAfter(self.ondone, verify_results, quarantined_results)

    def restore_session(self):
        try:
            state = pickle.load(open(self.stateP, 'rb'))
            groups = state['groups']
            self.groups = []
            for group_dict in groups:
                self.add_group(VerifyGroup.unmarshall(group_dict))

            self.bbs_map = state['bbs_map']
            self.exemplar_imgpaths = state['exemplar_imgpaths']
            self.rankedlist_map = state['rankedlist_map']
            fingroups_in = state['finished_groups']
            fingroups_new = {}
            for tag, groups_marsh in fingroups_in.iteritems():
                fingroups_new[tag] = [VerifyGroup.unmarshall(gdict) for gdict in groups_marsh]
            self.finished_groups = fingroups_new
            self.quarantined_groups = [VerifyGroup.unmarshall(gdict) for gdict in state['quarantined_groups']]
            print '...Successfully loaded VerifyOverlays state...'
            return state
        except Exception as e:
            print '...Failed to load VerifyOverlays state...'
            return False
    def create_state_dict(self):
        state = VerifyOverlays.create_state_dict(self)
        state['quarantined_groups'] = [g.marshall() for g in self.quarantined_groups]
        return state

    def do_quarantine(self):
        """ The user wants to quarantine the currently-displayed group. """
        curgroup = self.get_current_group()
        self.quarantined_groups.append(curgroup)
        self.remove_group(curgroup)

class VerifyOverlaysMultCats(wx.Panel):
    """ A widget that lets the user verify the overlays of N different
    categories (i.e. 'party', 'language', 'precinct').
    """
    def __init__(self, parent, *args, **kwargs):
        wx.Panel.__init__(self, parent, *args, **kwargs)

        self.verify_results_cat = {} # maps {cat_tag: {grouptag: [imgpath_i, ...]}}
        
        self.init_ui()

    def init_ui(self):
        self.nb = wx.Notebook(self)
        self.nb.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.onPageChange)
        
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.nb, proportion=1, border=10, flag=wx.EXPAND | wx.ALL)
        self.SetSizer(self.sizer)
        self.Layout()

    def start(self, imgpath_cats, cat_exemplars, do_align=False,
              bbs_map_cats=None, ondone=None, verifypanelClass=None):
        """
        Input:
        dict IMGPATH_CATS: {cat_tag: {grouptag: [imgpath_i, ...]}}
            For each category CAT_TAG, GROUPTAG is an identifier for
            a set of imgpaths. 
        dict CAT_EXEMPLARS: {cat_tag: {grouptag: [exmplpath_i, ...]}}
            For each category CAT_TAG, GROUPTAG is an identifier for
            a set of exemplar imgpatches.
        bool DO_ALIGN:
            If True, then this will align all imgpatches when overlaying.
        """
        if verifypanelClass == None:
            verifypanelClass = VerifyOverlaysPanel
        categories = tuple(set(imgpath_cats.keys()))
        self.imgpath_cats = imgpath_cats
        self.cat_exemplars = cat_exemplars
        self.do_align = do_align
        self.bbs_map_cats = bbs_map_cats if bbs_map_cats else {}
        self.ondone = ondone
        self.cat2page = {} # maps {cat_tag: int pageidx}
        self.page2cat = {} # maps {int pageidx: cat_tag}
        self.started_pages = {} # maps {int pageidx: bool hasStarted}
        pages = []
        for i, category in enumerate(categories):
            verifyoverlays = verifypanelClass(self.nb)
            pages.append((verifyoverlays, category))
            self.cat2page[category] = i
            self.page2cat[i] = category
        for i, (page, category) in enumerate(pages):
            self.nb.AddPage(page, str(category))
        self.nb.ChangeSelection(0)
        self.nb.SendPageChangedEvent(-1, 0)
        self.Layout()

    def save_session(self):
        pages = range(self.nb.GetPageCount())
        for i in pages:
            verifypanel = self.nb.GetPage(i)
            verifypanel.save_session()
            
    def onPageChange(self, evt):
        old = evt.GetOldSelection()
        new = evt.GetSelection()
        if new in self.started_pages:
            return
        curcat = self.page2cat[new]
        imgpath_groups = self.imgpath_cats[curcat]
        #exemplar_groups = self.cat_exemplars[curcat]
        exemplar_groups = self.cat_exemplars.get(curcat, {})
        bbs_map = self.bbs_map_cats.get(curcat, None)
        verifyoverlays = self.nb.GetPage(new)
        verifyoverlays.start(imgpath_groups, exemplar_groups, None, 
                             bbs_map=bbs_map, do_align=self.do_align, 
                             ondone=self.on_cat_done, auto_ondone=True)
        self.started_pages[new] = True
        
    def on_cat_done(self, verify_results):
        """ Called when a category is finished overlay verification.
        Input:
            dict VERIFY_RESULTS: {grouptag: [imgpath_i, ...]}
        """
        print "...In on_cat_done..."
        curcat = self.page2cat[self.nb.GetSelection()]
        self.verify_results_cat[curcat] = verify_results

        if len(self.verify_results_cat) == len(self.cat2page):
            print "We're done verifying all categories!"
            self.Disable()
            if self.ondone:
                #self.ondone(self.verify_results_cat)
                wx.CallAfter(self.ondone, self.verify_results_cat)

class CheckImageEqualsPanel(VerifyOverlaysPanel):
    """ Class that contains both a Header, a CheckImageEquals, and a Button
    Toolbar Footer. """
    def set_classes(self):
        VerifyOverlaysPanel.set_classes(self)
        self.overlaypanelCls = CheckImageEquals
        self.footerCls = CheckImageEqualsFooter

class CheckImageEqualsFooter(VerifyOverlaysFooter):
    def init_ui(self):
        VerifyOverlaysFooter.init_ui(self)
        btn_no = wx.Button(self, label="Doesn't Match")
        btn_no.Bind(wx.EVT_BUTTON, self.onButton_no)
        
        self.btn_sizer.Add(btn_no)

        self.btn_manual_relabel.Hide()
        self.sizer_exmpls.ShowItems(False)
        self.sizer_curlabel.ShowItems(False)

        self.Layout()

    def onButton_no(self, evt):
        self.GetParent().overlaypanel.do_no()

class CheckImageEquals(VerifyOverlays):
    """ A widget that lets the user separate a set of images into two
    categories:
        A.) These images match category A
        B.) These images do /not/ match category A.
    """
    TAG_YES = "YES_TAG"
    TAG_NO = "NO_TAG"
    def __init__(self, parent, *args, **kwargs):
        VerifyOverlays.__init__(self, parent, *args, **kwargs)

        self.cat_imgpath = None
        
    def start(self, imgpaths, cat_imgpath, do_align=False, bbs_map=None,
              ondone=None, stateP=None):
        """
        Input:
            list IMGPATHS: [imgpath_i, ...]
            str CAT_IMGPATH: Imagepath of the category.
            dict BBS_MAP: maps {str imgpath: (x1,y1,x2,y2}
            fn ONDONE: Function that accepts one argument:
                dict {str tag: [obj group_i, ...]}
                
        """
        self.stateP = stateP
        if not self.restore_session():
            # 0.) Munge IMGPATHS, BBS_MAP into VerifyOverlay-friendly versions
            imgpath_groups = {} # maps {str tag: [imgpath_i, ...]}
            bbs_map_v2 = {} # maps {(str tag, imgpath): (x1,y1,x2,y2)}
            for imgpath in imgpaths:
                imgpath_groups.setdefault(self.TAG_YES, []).append(imgpath)
                if bbs_map:
                    bbs_map_v2[(self.TAG_YES, imgpath)] = bbs_map[imgpath]
            imgpath_groups[self.TAG_NO] = []
            group_exemplars = {self.TAG_YES: [cat_imgpath]}
            rlist_map = {} # Don't care
            VerifyOverlays.start(self, imgpath_groups, group_exemplars, rlist_map, 
                                 do_align=do_align, bbs_map=bbs_map_v2, ondone=ondone)
        self.cat_imgpath = cat_imgpath
        I = scipy.misc.imread(cat_imgpath, flatten=True)
        bitmap = NumpyToWxBitmap(I)
        self.exemplarImg.SetBitmap(bitmap)
        self.Layout()

    def do_no(self):
        """ The user says that the current group does NOT match category A. """
        curgroup = self.get_current_group()
        self.finished_groups.setdefault(self.TAG_NO, []).append(curgroup)
        self.remove_group(curgroup)
    def handle_nomoregroups(self):
        self.export_results()
        self.Close()

class CheckImageEqualsFrame(wx.Frame):
    def __init__(self, parent, imgpaths, exemplar_imgpath, ondone):
        """
        Input:
            list IMGPATHS: List of image paths
            str EXEMPLAR_IMGPATH:
            fn ONDONE:
        """
        wx.Frame.__init__(self, parent)

        verifypanel = CheckImageEqualsPanel(self)
        verifypanel.start(imgpaths, exemplar_imgpath, ondone=ondone, do_align=True)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(verifypanel, proportion=1, flag=wx.EXPAND)
        self.SetSizer(sizer)
        self.Layout()

class SeparateImagesPanel(VerifyOverlaysPanel):
    """ Class that contains both a Header, a CheckImageEquals, and a Button
    Toolbar Footer. """
    def set_classes(self):
        VerifyOverlaysPanel.set_classes(self)
        self.overlaypanelCls = SeparateImages
        self.footerCls = SeparateImagesFooter

class SeparateImagesFooter(VerifyOverlaysFooter):
    def init_ui(self):
        VerifyOverlaysFooter.init_ui(self)
        self.btn_explode_group = wx.Button(self, label="Explode this group.")
        self.btn_explode_group.Bind(wx.EVT_BUTTON, self.onButton_explode)
        self.btn_realign_imgs = wx.Button(self, label="Re-align images...")
        self.btn_realign_imgs.Bind(wx.EVT_BUTTON, self.onButton_realign)

        self.btn_sizer.AddMany([(self.btn_explode_group,), (self.btn_realign_imgs,)])

        self.sizer_exmpls.ShowItems(False)
        self.sizer_curlabel.ShowItems(False)
        self.btn_manual_relabel.Hide()
        self.btn_realign_imgs.Hide()

        self.Layout()
    def onButton_explode(self, evt):
        self.GetParent().overlaypanel.do_explode()
    def onButton_realign(self, evt):
        self.GetParent().overlaypanel.do_realign()

class SeparateImages(VerifyOverlays):
    """ A widget that lets a user separate a set of images into different
    categories, where the number of categories isn't known in advance.
    """
    TAG_UNIVERSAL = "TAG"

    def init_ui(self):
        VerifyOverlays.init_ui(self)
        self.sizer_attrpatch.ShowItems(False)

    def start(self, imggroups, do_align=False, bbs_map=None,
              ondone=None, stateP=None, auto_ondone=False,
              realign_callback=None):
        """
        Input:
            dict IMGGROUPS: {tag: [imgpath_i, ...]}
            dict BBS_MAP: maps {(tag, imgpath): (x1,y1,x2,y2)}
            fn REALIGN_CALLBACK: If given, this should be given the list
                of imgpaths to re-align. It will return either:
                    'okay' -- reload from the old imgepaths
                    lst imgpaths: -- imgpaths to use in place of the
                                     old imgpaths.
        """
        self.stateP = stateP
        self.realign_callback = realign_callback
        if self.realign_callback:
            self.btn_realign_imgs.Show()
        if not self.restore_session():
            exemplars = [] # No need for exemplars
            VerifyOverlays.start(self, imggroups, exemplars, None, 
                                 bbs_map=bbs_map, ondone=ondone, stateP=stateP,
                                 auto_ondone=auto_ondone)
            
    def export_results(self):
        if self.ondone:
            verify_results = {} # maps {int id: [imgpath_i, ...]}
            idx = 0
            for (tag, groups) in self.finished_groups.iteritems():
                for group in groups:
                    verify_results[idx] = group.imgpaths
                    idx += 1
            #self.ondone(verify_results)
            wx.CallAfter(self.ondone, verify_results)

    def do_explode(self):
        """ Add each individual element of the current group into
        self.finished_groups in their own groups. Used if the user just
        'gives up' on a particularly bad group overlays.
        """
        curgroup = self.get_current_group()
        newgroups = []
        for imgpath in curgroup.imgpaths:
            newgroups.append(Group([imgpath]))
        self.finished_groups.setdefault(self.TAG_UNIVERSAL, []).extend(newgroups)
        self.remove_group(curgroup)

    def do_realign(self):
        """ Calls the self.REALIGN_CALLBACK function with the (ordered) list
        of imgpaths of the currently-displayed group, and expects one
        of the following return values:
            'okay' -- The overlays should re-compute the min/max overlays
                      of the current group with the old imgpaths.
            lst IMGPATHS -- Replace the group's list of imagepaths with
                            the new IMGPATHS, and re-compute the min/max
                            overlays with this.
        """
        curgroup = self.get_current_group()
        result = self.realign_callback(curgroup.imgpaths)
        if result == 'okay':
            pass
        else:
            for i, new_imgpath in enumerate(result):
                curgroup.imgpaths[i] = new_imgpath
        # Now, re-compute the overlays for this group
        overlay_min, overlay_max = curgroup.get_overlays(force=True)
        minimg_np = iplimage2np(overlay_min)
        maximg_np = iplimage2np(overlay_max)

        min_bitmap = NumpyToWxBitmap(minimg_np)
        max_bitmap = NumpyToWxBitmap(maximg_np)

        self.minOverlayImg.SetBitmap(min_bitmap)
        self.maxOverlayImg.SetBitmap(max_bitmap)
        
        self.Layout()
        self.Refresh()

class SeparateImagesFrame(wx.Frame):
    def __init__(self, parent, imgpath_groups, callback,
                 realign_callback=None, *args, **kwargs):
        """
        Input:
            list IMGPATH_GROUPS: List of lists of imgpaths
            fn CALLBACK: Callback function to call with final results. 
                Should accept one argument: List of lists of imgpaths.
            fn REALIGN_CALLBACK: Function to call when the user wishes
                to realign the images in the current group. Should
                accept a list of imgpaths, and return either 'okay' or
                a new list of imgpaths.
        """
        wx.Frame.__init__(self, parent, *args, **kwargs)
        self.imgpath_groups = imgpath_groups
        self.callback = callback
        self.realign_callback = realign_callback
        
        self.separatepanel = SeparateImagesPanel(self)

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.separatepanel, border=10, proportion=1, flag=wx.EXPAND | wx.ALL)
        
        self.SetSizer(self.sizer)
        
        imggroups = {} # maps {tag: [imgpath_i, ...]}
        for tagid, imgpaths in enumerate(imgpath_groups):
            imggroups[tagid] = imgpaths

        self.separatepanel.start(imggroups, do_align=False, ondone=self.ondone_verify,
                                 auto_ondone=True, realign_callback=self.realign_callback)

        self.Layout()

    def ondone_verify(self, results):
        """ 
        Input: 
            dict RESULTS: maps {tag: [imgpath_i, ...]}
        """
        out = []
        for tag, imgpaths in results.iteritems():
            out.append(imgpaths)
        wx.CallAfter(self.callback, out)
        self.Close()

class Group(object):
    def __init__(self, imgpaths, tag=None, do_align=False):
        self.tag = tag
        self.imgpaths = imgpaths
    
        # self.OVERLAY_MIN, self.OVERLAY_MAX: IplImage overlays.
        self.overlay_min = None
        self.overlay_max = None
        self.do_align = do_align
    def get_overlays(self, bbs_map=None, force=False):
        """
        Input:
            dict BBS_MAP: maps {str imgpath: (x1,y1,x2,y2)}
            bool FORCE: If True, this will re-compute the overlays.
        Output:
            IplImage minimg, IplImage maximg.
        """
        if not self.overlay_min or force:
            print "...Computing Min/Max Overlays for {0} images...".format(len(self.imgpaths))
            t = time.time()
            #minimg, maximg = make_overlays.minmax_cv(self.imgpaths, do_align=self.do_align,
            #                                         rszFac=0.75, bbs_map=bbs_map)
            # NOTE: calling with numProcs>1 actually slows things down. 
            #       The slowdown might be due to all the .tostring()'ing
            #       I have to do to deal with IplImages not being
            #       pickle'able. Might just have to use nparrays (which
            #       are pickle'able.
            minimg, maximg = make_overlays.minmax_cv_par(self.imgpaths, do_align=self.do_align,
                                                         rszFac=0.75, bbs_map=bbs_map, numProcs=1)
            dur = time.time() - t
            print "...Finished Computing Min/Max Overlays ({0} s).".format(dur)
            self.overlay_min = minimg
            self.overlay_max = maximg
        return self.overlay_min, self.overlay_max
    def marshall(self):
        """ Returns a dict-rep of myself. In particular, you can't pickle
        IplImages, so don't include them.
        """
        me = {'tag': self.tag,
              'imgpaths': self.imgpaths, 'do_align': self.do_align}
        return me

    @staticmethod
    def unmarshall(d):
        return Group(d['imgpaths'], tag=d['tag'], do_align=d['do_align'])

    def __eq__(self, o):
        return (isinstance(o, Group) and self.imgpaths == o.imgpaths)
    def __repr__(self):
        return "Group({0},numimgs={1})".format(self.tag,
                                               len(self.imgpaths))
    def __str__(self):
        return "Group({0},numimgs={1})".format(self.tag,
                                               len(self.imgpaths))

class SplitGroup(Group):
    def midsplit(self):
        """ Laziest split method: Split down the middle. """
        mid = len(self.imgpaths) / 2
        imgsA, imgsB = self.imgpaths[:mid], self.imgpaths[mid:]
        return [type(self)(imgsA, tag=self.tag, do_align=self.do_align),
                type(self)(imgsB, tag=self.tag, do_align=self.do_align)]

    def split_kmeans(self, K=2):
        t = time.time()
        print "...running k-means..."
        clusters = cluster_imgs.cluster_imgs_kmeans(self.imgpaths, k=K, do_downsize=True,
                                                    do_align=True)
        dur = time.time() - t
        print "...Completed k-means ({0} s)".format(dur)
        if len(clusters) != K:
            print "...Warning: Kmeans only found {0} clusters, yet user \
specified K={1}. Falling back to simple split-down-the-middle.".format(len(clusters), K)
            return self.midsplit()
        groups = []
        for clusterid, imgpaths in clusters.iteritems():
            groups.append(type(self)(imgpaths, tag=self.tag, do_align=self.do_align))
        assert len(groups) == K
        return groups

    def split_pca_kmeans(self, K=2, N=3):
        t = time.time()
        print "...running PCA+k-means..."
        clusters = cluster_imgs.cluster_imgs_pca_kmeans(self.imgpaths, k=K, do_align=True)
        dur = time.time() - t
        print "...Completed PCA+k-means ({0} s)".format(dur)
        if len(clusters) != K:
            print "...Warning: PCA+Kmeans only found {0} clusters, yet user \
specified K={1}. Falling back to simple split-down-the-middle.".format(len(clusters), K)
            return self.midsplit()
        groups = []
        for clusterid, imgpaths in clusters.iteritems():
            groups.append(type(self)(imgpaths, tag=self.tag, do_align=self.do_align))
        assert len(groups) == K
        return groups
        
    def split_kmeans2(self, K=2):
        t = time.time()
        print "...running k-meansV2..."
        clusters = cluster_imgs.kmeans_2D(self.imgpaths, k=K, distfn_method='vardiff',
                                          do_align=True)
        dur = time.time() - t
        print "...Completed k-meansV2 ({0} s)".format(dur)
        if len(clusters) != K:
            print "...Warning: Kmeans2 only found {0} clusters, yet user \
specified K={1}. Falling back to simple split-down-the-middle.".format(len(clusters), K)
            return self.midsplit()
        groups = []
        for clusterid, imgpaths in clusters.iteritems():
            groups.append(type(self)(imgpaths, tag=self.tag, do_align=self.do_align))
        assert len(groups) == K
        return groups

    def split_kmediods(self, K=2):
        t = time.time()
        print "...running k-mediods..."
        clusters = cluster_imgs.kmediods_2D(self.imgpaths, k=K, distfn_method='vardiff',
                                            do_align=True)
        dur = time.time() - t
        print "...Completed k-mediods ({0} s)".format(dur)
        if len(clusters) != K:
            print "...Warning: Kmediods only found {0} clusters, yet user \
specified K={1}. Falling back to simple split-down-the-middle.".format(len(clusters), K)
            return self.midsplit()
        groups = []
        for clusterid, imgpaths in clusters.iteritems():
            groups.append(type(self)(imgpaths, tag=self.tag, do_align=self.do_align))
        assert len(groups) == K
        return groups

    def split(self, mode=None):
        if mode == None:
            mode == 'kmeans'
        if len(self.imgpaths) == 1:
            return [self]
        elif len(self.imgpaths) == 2:
            return [type(self)([self.imgpaths[0]], tag=self.tag, do_align=self.do_align),
                    type(self)([self.imgpaths[1]], tag=self.tag, do_align=self.do_align)]
        if mode == 'midsplit':
            return self.midsplit()
        elif mode == 'kmeans':
            return self.split_kmeans(K=2)
        elif mode == 'pca_kmeans':
            return self.split_pca_kmeans(K=2, N=3)
        elif mode == 'kmeans2':
            return self.split_kmeans2(K=2)
        elif mode == 'kmediods':
            return self.split_kmediods(K=2)
        else:
            return self.split_kmeans(K=2)

    @staticmethod
    def unmarshall(d):
        return SplitGroup(d['imgpaths'], tag=d['tag'], do_align=d['do_align'])
    def __eq__(self, o):
        return (isinstance(o, SplitGroup) and self.imgpaths == o.imgpaths)
    def __repr__(self):
        return "SplitGroup({0},numimgs={1})".format(self.tag,
                                                    len(self.imgpaths))
    def __str__(self):
        return "SplitGroup({0},numimgs={1})".format(self.tag,
                                                    len(self.imgpaths))
    
class VerifyGroup(SplitGroup):
    def __init__(self, imgpaths, rlist_idx=0, exmpl_idx=0, *args, **kwargs):
        SplitGroup.__init__(self, imgpaths, *args, **kwargs)
        self.rlist_idx = rlist_idx
        self.exmpl_idx = exmpl_idx
    def split(self, mode=None):
        if mode == None:
            mode = 'rankedlist'
        if mode == 'rankedlist':
            return [self]
        else:
            return SplitGroup.split(self, mode=mode)
    def marshall(self):
        me = SplitGroup.marshall(self)
        me['rlist_idx'] = self.rlist_idx
        me['exmpl_idx'] = self.exmpl_idx
        return me
    @staticmethod
    def unmarshall(d):
        return VerifyGroup(d['imgpaths'], rlist_idx=d['rlist_idx'],
                           exmpl_idx=d['exmpl_idx'], tag=d['tag'], do_align=d['do_align'])
    def __eq__(self, o):
        return (isinstance(o, VerifyGroup) and self.imgpaths == o.imgpaths)
    def __repr__(self):
        return "VerifyGroup({0},rlidx={1},exidx={2},numimgs={3})".format(self.tag,
                                                                         self.rlist_idx,
                                                                         self.exmpl_idx,
                                                                         len(self.imgpaths))
    def __str__(self):
        return "VerifyGroup({0},rlidx={1},exidx={2},numimgs={3})".format(self.tag,
                                                                         self.rlist_idx,
                                                                         self.exmpl_idx,
                                                                         len(self.imgpaths))

class DigitGroup(VerifyGroup):
    @staticmethod
    def unmarshall(d):
        return DigitGroup(d['imgpaths'], rlist_idx=d['rlist_idx'],
                          exmpl_idx=d['exmpl_idx'], tag=d['tag'], do_align=d['do_align'])
    def __eq__(self, o):
        return (isinstance(o, VerifyGroup) and self.imgpaths == o.imgpaths)
    def __repr__(self):
        return "DigitGroup({0},rlidx={2},exidx={3},numimgs={4})".format(self.tag,
                                                                        self.rlist_idx,
                                                                        self.exmpl_idx,
                                                                        len(self.imgpaths))
    def __str__(self):
        return "DigitGroup({0},rlidx={2},exidx={3},numimgs={4})".format(self.tag,
                                                                        self.rlist_idx,
                                                                        self.exmpl_idx,
                                                                        len(self.imgpaths))

class ManualRelabelDialog(wx.Dialog):
    def __init__(self, parent, tags, *args, **kwargs):
        """
        Input:
            list TAGS: list [tag_i, ...]
        """
        wx.Dialog.__init__(self, parent, *args, **kwargs)
        
        self.tag = None

        self.tags = tags

        txt0 = wx.StaticText(self, label="What is the correct tag?")
        self.combobox_tags = wx.ComboBox(self, choices=map(str, tags), 
                                         style=wx.CB_READONLY | wx.CB_SORT, size=(200, -1))
        cbox_sizer = wx.BoxSizer(wx.HORIZONTAL)
        cbox_sizer.AddMany([(txt0,), (self.combobox_tags,)])
        
        btn_ok = wx.Button(self, label="Ok")
        btn_ok.Bind(wx.EVT_BUTTON, self.onButton_ok)
        btn_cancel = wx.Button(self, label="Cancel")
        btn_cancel.Bind(wx.EVT_BUTTON, self.onButton_cancel)
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        btn_sizer.AddMany([(btn_ok,), (btn_cancel,)])

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(cbox_sizer)
        self.sizer.Add(btn_sizer, flag=wx.ALIGN_CENTER)

        self.SetSizer(self.sizer)
        self.Layout()

    def onButton_ok(self, evt):
        self.tag = self.tags[self.combobox_tags.GetSelection()]
        self.EndModal(wx.OK)
    def onButton_cancel(self, evt):
        self.EndModal(wx.CANCEL)

class ChooseSplitModeDialog(wx.Dialog):
    ID_MIDSPLIT = 41
    ID_RANKEDLIST = 42
    ID_KMEANS = 43
    ID_PCA_KMEANS = 44
    ID_KMEANS2 = 45
    ID_KMEDIODS = 46

    def __init__(self, parent, disable=None, *args, **kwargs):
        """ disable is a list of ID's (ID_RANKEDLIST, etc.) to disable. """
        wx.Dialog.__init__(self, parent, *args, **kwargs)
        if disable == None:
            disable = []
        sizer = wx.BoxSizer(wx.VERTICAL)
        txt = wx.StaticText(self, label="Please choose the desired 'Split' method.")

        self.midsplit_rbtn = wx.RadioButton(self, label="Split in the middle (fast, but not good)", style=wx.RB_GROUP)
        self.rankedlist_rbtn = wx.RadioButton(self, label='Ranked-List (fast)')
        self.kmeans_rbtn = wx.RadioButton(self, label='K-means (not-as-fast)')
        self.pca_kmeans_rbtn = wx.RadioButton(self, label='PCA+K-means (not-as-fast)')
        self.kmeans2_rbtn = wx.RadioButton(self, label="K-means V2 (not-as-fast)")
        self.kmediods_rbtn = wx.RadioButton(self, label="K-Mediods")
        
        if parent.splitmode == 'midsplit':
            self.midsplit_rbtn.SetValue(1)
        elif parent.splitmode == 'rankedlist':
            self.rankedlist_rbtn.SetValue(1)
        elif parent.splitmode == 'kmeans':
            self.kmeans_rbtn.SetValue(1)
        elif parent.splitmode == 'pca_kmeans':
            self.pca_kmeans_rbtn.SetValue(1)
        elif parent.splitmode == 'kmeans2':
            self.kmeans2_rbtn.SetValue(1)
        elif parent.splitmode == 'kmediods':
            self.kmediods_rbtn.SetValue(1)
        else:
            print "Unrecognized parent.splitmode: {0}. Defaulting to kmeans.".format(parent.splitmode)
            self.kmeans_rbtn.SetValue(1)

        if self.ID_MIDSPLIT in disable:
            self.midsplit_rbtn.Disable()
        if ChooseSplitModeDialog.ID_RANKEDLIST in disable:
            self.rankedlist_rbtn.Disable()
        if ChooseSplitModeDialog.ID_KMEANS in disable:
            self.kmeans_rbtn.Disable()
        if ChooseSplitModeDialog.ID_PCA_KMEANS in disable:
            self.pca_kmeans_rbtn.Disable()
        if ChooseSplitModeDialog.ID_KMEANS2 in disable:
            self.kmeans2_rbtn.Disable()
        if ChooseSplitModeDialog.ID_KMEDIODS in disable:
            self.kmediods_rbtn.Disable()
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        btn_ok = wx.Button(self, label="Ok")
        btn_cancel = wx.Button(self, label="Cancel")
        btn_ok.Bind(wx.EVT_BUTTON, self.onButton_ok)
        btn_cancel.Bind(wx.EVT_BUTTON, lambda evt: self.EndModal(wx.ID_CANCEL))
        
        btn_sizer.AddMany([(btn_ok,), (btn_cancel,)])

        sizer.AddMany([(txt,), ((20,20),), (self.midsplit_rbtn,), (self.rankedlist_rbtn,),
                       (self.kmeans_rbtn,), (self.pca_kmeans_rbtn,),
                       (self.kmeans2_rbtn,), (self.kmediods_rbtn),])
        sizer.Add(btn_sizer, flag=wx.ALIGN_CENTER)

        self.SetSizer(sizer)
        self.Fit()

    def onButton_ok(self, evt):
        if self.midsplit_rbtn.GetValue():
            self.EndModal(self.ID_MIDSPLIT)
        elif self.rankedlist_rbtn.GetValue():
            self.EndModal(self.ID_RANKEDLIST)
        elif self.kmeans_rbtn.GetValue():
            self.EndModal(self.ID_KMEANS)
        elif self.pca_kmeans_rbtn.GetValue():
            self.EndModal(self.ID_PCA_KMEANS)
        elif self.kmeans2_rbtn.GetValue():
            self.EndModal(self.ID_KMEANS2)
        elif self.kmediods_rbtn.GetValue():
            self.EndModal(self.ID_KMEDIODS)
        else:
            print "Unrecognized split mode. Defaulting to K-means."
            self.EndModal(self.ID_KMEANS)

def PilImageToWxBitmap( myPilImage ) :
    return WxImageToWxBitmap( PilImageToWxImage( myPilImage ) )
def PilImageToWxImage( myPilImage ):
    myWxImage = wx.EmptyImage( myPilImage.size[0], myPilImage.size[1] )
    myWxImage.SetData( myPilImage.convert( 'RGB' ).tostring() )
    return myWxImage
def WxImageToWxBitmap( myWxImage ) :
    return myWxImage.ConvertToBitmap()
def NumpyToWxBitmap(img):
    """
    Assumption: img represents a grayscale img [not sure if necessary]
    """
    img_pil = Image.fromarray(img)
    return PilImageToWxBitmap(img_pil)

def iplimage2np(iplimage):
    """ Assumes IPLIMAGE has depth cv.CV_8U. """
    w, h = cv.GetSize(iplimage)
    img_np = np.fromstring(iplimage.tostring(), dtype='uint8')
    img_np = img_np.reshape(h, w)
    
    return img_np

def is_img_ext(p):
    return os.path.splitext(p)[1].lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tif')

def gray2rgb_np(nparray):
    if len(nparray.shape) == 2:
        npout = np.zeros((nparray.shape[0],nparray.shape[1], 3), dtype=nparray.dtype)
        npout[:,:,0] = nparray
        npout[:,:,1] = nparray
        npout[:,:,2] = nparray
    else:
        npout = nparray.copy()
    return npout

def test_verifyoverlays():
    class TestFrame(wx.Frame):
        def __init__(self, parent, imggroups, exemplars, *args, **kwargs):
            wx.Frame.__init__(self, parent, size=(600, 500), *args, **kwargs)

            self.imggroups = imggroups

            self.viewoverlays = VerifyOverlaysPanel(self)

            self.sizer = wx.BoxSizer(wx.VERTICAL)
            self.sizer.Add(self.viewoverlays, proportion=1, flag=wx.EXPAND)
            self.SetSizer(self.sizer)
            self.Layout()

            self.viewoverlays.start(self.imggroups, exemplars, {}, do_align=True, ondone=self.ondone)

        def ondone(self, verify_results):
            print '...In ondone...'
            print 'verify_results:', verify_results
    args = sys.argv[1:]
    imgsdir = args[0]
    exmpls_dir = args[1]
    
    imggroups = {} # maps {str groupname: [imgpath_i, ...]}
    for dirpath, dirnames, filenames in os.walk(imgsdir):
        imggroup = []
        groupname = os.path.split(dirpath)[1]
        print filenames, groupname
        for imgname in [f for f in filenames if is_img_ext(f)]:
            imggroup.append(os.path.join(dirpath, imgname))
        if imggroup:
            imggroups[groupname] = imggroup

    exmpl_paths = {}
    for dirpath, dirnames, filenames in os.walk(exmpls_dir):
        exmpls = []
        groupname = os.path.split(dirpath)[1]
        for imgname in [f for f in filenames if is_img_ext(f)]:
            exmpls.append(os.path.join(dirpath, imgname))
        if exmpls:
            exmpl_paths[groupname] = exmpls

    app = wx.App(False)
    f = TestFrame(None, imggroups, exmpl_paths)
    f.Show()
    app.MainLoop()

def test_checkimgequal():
    class TestFrame(wx.Frame):
        def __init__(self, parent, imgpaths, catimgpath, *args, **kwargs):
            wx.Frame.__init__(self, parent, size=(600, 500), *args, **kwargs)

            self.chkimgequals = CheckImageEqualsPanel(self)

            self.sizer = wx.BoxSizer(wx.VERTICAL)
            self.sizer.Add(self.chkimgequals, proportion=1, flag=wx.EXPAND)
            self.SetSizer(self.sizer)
            self.Layout()

            self.chkimgequals.start(imgpaths, catimgpath, do_align=True, ondone=self.ondone)

        def ondone(self, verify_results):
            print '...In TestFrame.ondone...'
            print 'verify_results:', verify_results
    args = sys.argv[1:]
    imgsdir = args[0]
    catimgpath = args[1]
    
    imgpaths = []
    for dirpath, dirnames, filenames in os.walk(imgsdir):
        for imgname in [f for f in filenames if is_img_ext(f)]:
            imgpaths.append(os.path.join(dirpath, imgname))

    app = wx.App(False)
    f = TestFrame(None, imgpaths, catimgpath)
    f.Show()
    app.MainLoop()

def test_verifycategories():
    class TestFrame(wx.Frame):
        def __init__(self, parent, imgcategories, exmplcategories, *args, **kwargs):
            wx.Frame.__init__(self, parent, size=(600, 500), *args, **kwargs)

            self.viewoverlays = VerifyOverlaysMultCats(self)

            self.sizer = wx.BoxSizer(wx.VERTICAL)
            self.sizer.Add(self.viewoverlays, proportion=1, flag=wx.EXPAND)
            self.SetSizer(self.sizer)
            self.Layout()

            self.viewoverlays.start(imgcategories, exmplcategories, 
                                    do_align=True, ondone=self.ondone)

        def ondone(self, verify_results):
            print 'verify_results:', verify_results
    args = sys.argv[1:]
    imgsdir = args[0]
    exmpls_dir = args[1]
    
    imgcats = {} # maps {cat_tag: {str groupname: [imgpath_i, ...]}}
    for catdir in os.listdir(imgsdir):
        cat_fulldir = pathjoin(imgsdir, catdir)
        for groupdir in os.listdir(cat_fulldir):
            group_fulldir = pathjoin(cat_fulldir, groupdir)
            for imgname in os.listdir(group_fulldir):
                imgpath = pathjoin(group_fulldir, imgname)
                imgcats.setdefault(catdir, {}).setdefault(groupdir, []).append(imgpath)

    exmplscats = {} # maps {cat_tag: {groupname: [exmplpath_i, ...]}}
    for catdir in os.listdir(exmpls_dir):
        cat_fulldir = pathjoin(exmpls_dir, catdir)
        for groupdir in os.listdir(cat_fulldir):
            group_fulldir = pathjoin(cat_fulldir, groupdir)
            for imgname in os.listdir(group_fulldir):
                imgpath = pathjoin(group_fulldir, imgname)
                exmplscats.setdefault(catdir, {}).setdefault(groupdir, []).append(imgpath)

    app = wx.App(False)
    f = TestFrame(None, imgcats, exmplscats)
    f.Show()
    app.MainLoop()

def test_separateimages():
    class TestFrame(wx.Frame):
        def __init__(self, parent, imggroups, altimg, *args, **kwargs):
            wx.Frame.__init__(self, parent, size=(600, 500), *args, **kwargs)

            self.altimg = altimg

            self.separateimages = SeparateImagesPanel(self)

            self.sizer = wx.BoxSizer(wx.VERTICAL)
            self.sizer.Add(self.separateimages, proportion=1, flag=wx.EXPAND)
            self.SetSizer(self.sizer)
            self.Layout()
            realign_callback = self.realign if altimg != None else None
            self.separateimages.start(imggroups, ondone=self.ondone, auto_ondone=True,
                                      realign_callback=realign_callback)

        def ondone(self, verify_results):
            print "Number of groups:", len(verify_results)

        def realign(self, imgpaths):
            out = []
            for imgpath in imgpaths:
                out.append(self.altimg)
            return out

    args = sys.argv[1:]
    imgsdir = args[0]
    try:
        altimg = args[1]
    except:
        altimg = None
    
    imggroups = {} # maps {tag: [imgpath_i, ...]}
    for catdir in os.listdir(imgsdir):
        cat_fulldir = pathjoin(imgsdir, catdir)
        for imgname in os.listdir(cat_fulldir):
            imgpath = pathjoin(cat_fulldir, imgname)
            imggroups.setdefault('tag', []).append(imgpath)

    app = wx.App(False)
    f = TestFrame(None, imggroups, altimg)
    f.Show()
    app.MainLoop()

def main():
    test_verifyoverlays()
    #test_checkimgequal()
    #test_verifycategories()
    #test_separateimages()

if __name__ == '__main__':
    main()
