import os, sys, pdb, traceback, time
try:
    import cPickle as pickle
except:
    import pickle

from os.path import join as pathjoin

import wx
from wx.lib.scrolledpanel import ScrolledPanel
from wx.lib.pubsub import Publisher

import cv, numpy as np, scipy, scipy.misc, Image
import make_overlays
import util
import cluster_imgs

class VerifyOverlaysMainPanel(wx.Panel):
    def __init__(self, parent, *args, **kwargs):
        wx.Panel.__init__(self, parent, *args, **kwargs)

        self.proj = None
        self.stateP = None

    def start(self, proj, stateP):
        self.proj = proj
        self.stateP = stateP

        self.verifyoverlays.start()

    def stop(self):
        self.export_results()

    def export_results(self):
        pass

class ViewOverlays(ScrolledPanel):
    def __init__(self, parent, *args, **kwargs):
        ScrolledPanel.__init__(self, parent, *args, **kwargs)

        # list GROUPS: [obj GROUP_i, ...]
        self.groups = None

        # dict BBS_MAP: maps {(tag, str imgpath): (x1,y1,x2,y2)}
        self.bbs_map = None
        
        # IDX: Current idx into self.GROUPS that we are displaying
        self.idx = None

        self.init_ui()

    def overlays_layout_vert(self):
        """ Layout the overlay patches s.t. there is one row of N columns. 
        Typically called when the patch height > patch width.
        """
        self.sizer_overlays.SetOrientation(wx.VERTICAL)
        self.sizer_overlays_voted.SetOrientation(wx.HORIZONTAL)
        self.sizer_min.SetOrientation(wx.VERTICAL)
        self.sizer_max.SetOrientation(wx.VERTICAL)
        self.sizer_attrpatch.SetOrientation(wx.VERTICAL)
        self.sizer_diff.SetOrientation(wx.VERTICAL)
    def overlays_layout_horiz(self):
        """ Layout the overlay patches s.t. there are N rows of 1 column.
        Typically called when the patch width > patch height.
        """
        self.sizer_overlays.SetOrientation(wx.HORIZONTAL)
        self.sizer_overlays_voted.SetOrientation(wx.VERTICAL)
        self.sizer_min.SetOrientation(wx.HORIZONTAL)
        self.sizer_max.SetOrientation(wx.HORIZONTAL)
        self.sizer_attrpatch.SetOrientation(wx.HORIZONTAL)
        self.sizer_diff.SetOrientation(wx.HORIZONTAL)
    def set_patch_layout(self, orient='horizontal'):
        """ Change the orientation of the overlay patch images. Either
        arrange 'horizontal', or stack 'vertical'.
        """
        if orient == 'horizontal':
            sizer = self.overlays_layout_horiz()
        else:
            sizer = self.overlays_layout_vert()
        self.Layout()
        self.Refresh()

    def init_ui(self):
        txt_0 = wx.StaticText(self, label="Number of images in group: ")
        self.txtctrl_num_elements = wx.TextCtrl(self, value='0')
        self.listbox_groups = wx.ListBox(self, size=(200, 300))
        self.listbox_groups.Bind(wx.EVT_LISTBOX, self.onListBox_groups)
        sizer_numimgs = wx.BoxSizer(wx.HORIZONTAL)
        sizer_numimgs.AddMany([(txt_0,), (self.txtctrl_num_elements,)])
        sizer_groups = wx.BoxSizer(wx.VERTICAL)
        sizer_groups.AddMany([(sizer_numimgs,), (self.listbox_groups,)])

        st1 = wx.StaticText(self, -1, "min: ")
        st2 = wx.StaticText(self, -1, "max: ")
        st3 = wx.StaticText(self, -1, "Looks like? ")
        st4 = wx.StaticText(self, -1, "diff: ")
        self.st1, self.st2, self.st3, self.st4 = st1, st2, st3, st4

        self.minOverlayImg = wx.StaticBitmap(self, bitmap=wx.EmptyBitmap(1, 1))
        self.maxOverlayImg = wx.StaticBitmap(self, bitmap=wx.EmptyBitmap(1, 1))
        self.txt_exemplarTag = wx.StaticText(self, label='')
        self.exemplarImg = wx.StaticBitmap(self, bitmap=wx.EmptyBitmap(1, 1))
        self.diffImg = wx.StaticBitmap(self, bitmap=wx.EmptyBitmap(1, 1))

        maxTxtW = max([txt.GetSize()[0] for txt in (st1, st2, st3, st4)]) + 20

        sizer_overlays = wx.BoxSizer(wx.HORIZONTAL)
        self.sizer_overlays = sizer_overlays
        self.sizer_overlays_voted = wx.BoxSizer(wx.VERTICAL)
        self.sizer_min = wx.BoxSizer(wx.HORIZONTAL)
        self.sizer_min.AddMany([(st1,), ((maxTxtW-st1.GetSize()[0],0),), (self.minOverlayImg,)])
        self.sizer_max = wx.BoxSizer(wx.HORIZONTAL)
        self.sizer_max.AddMany([(st2,), ((maxTxtW-st2.GetSize()[0],0),), (self.maxOverlayImg,)])
        self.sizer_innerattrpatch = wx.BoxSizer(wx.VERTICAL)
        self.sizer_innerattrpatch.AddMany([(self.txt_exemplarTag,), (self.exemplarImg,)])
        self.sizer_attrpatch = wx.BoxSizer(wx.HORIZONTAL)
        self.sizer_attrpatch.AddMany([(st3,), ((maxTxtW-st3.GetSize()[0],0),), (self.sizer_innerattrpatch,)])
        self.sizer_diff = wx.BoxSizer(wx.HORIZONTAL)
        self.sizer_diff.AddMany([(st4,), ((maxTxtW-st4.GetSize()[0],0),), (self.diffImg,)])
        self.sizer_overlays_voted.AddMany([(self.sizer_min,), ((50, 50),), (self.sizer_max,), ((50, 50),),
                                           (self.sizer_diff,)])
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

        min_bitmap = NumpyToWxBitmap(minimg_np)
        max_bitmap = NumpyToWxBitmap(maximg_np)

        self.minOverlayImg.SetBitmap(min_bitmap)
        self.maxOverlayImg.SetBitmap(max_bitmap)
        
        self.Layout()

        return self.idx

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
            newidx = min(len(self.groups)-1, idx)
            self.select_group(newidx)
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

class SplitOverlays(ViewOverlays):
    def __init__(self, parent, *args, **kwargs):
        ViewOverlays.__init__(self, parent, *args, **kwargs)

        self.splitmode = 'kmeans'
        
    def init_ui(self):
        ViewOverlays.init_ui(self)
        
        btn_split = wx.Button(self, label="Split...")
        btn_split.Bind(wx.EVT_BUTTON, self.onButton_split)
        btn_setsplitmode = wx.Button(self, label="Set Split Mode...")
        btn_setsplitmode.Bind(wx.EVT_BUTTON, self.onButton_setsplitmode)
        sizer_split = wx.BoxSizer(wx.VERTICAL)
        sizer_split.AddMany([(btn_split,0,wx.ALIGN_CENTER), (btn_setsplitmode,0,wx.ALIGN_CENTER)])

        self.btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.btn_sizer.Add(sizer_split)

        self.sizer.Add(self.btn_sizer, proportion=0, border=10, flag=wx.ALL)
        self.Layout()

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

    def onButton_split(self, evt):
        curgroup = self.get_current_group()
        groups = curgroup.split(mode=self.splitmode)
        for group in groups:
            self.add_group(group)
        self.remove_group(curgroup)

    def onButton_setsplitmode(self, evt):
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

        # self.ONDONE: A callback function to call when verifying is done.
        self.ondone = None

    def init_ui(self):
        SplitOverlays.init_ui(self)
        
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
            self.ondone(verify_results)

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
        curidx = SplitOverlays.select_group(self, idx)
        if curidx == None:
            # Say, if IDX is invalid (maybe no more groups?)
            return
        group = self.groups[curidx]
        self.select_exmpl_group(group.tag, group.exmpl_idx)

        self.Layout()

    def select_exmpl_group(self, grouptag, exmpl_idx):
        """ Displays the correct exemplar img patch on the screen. """
        if grouptag not in self.exemplar_imgpaths:
            print "...Invalid GroupTAG: {0}...".format(grouptag)
            return
        exemplar_paths = self.exemplar_imgpaths[grouptag]
        if exmpl_idx < 0 or exmpl_idx >= len(exemplar_paths):
            print "...Invalid exmpl_idx: {0}...".format(exmpl_idx)
            return
        exemplar_npimg = scipy.misc.imread(exemplar_paths[exmpl_idx])
        exemplarImg_bitmap = NumpyToWxBitmap(exemplar_npimg)
        self.exmplidx_sel = exmpl_idx
        self.exemplarImg.SetBitmap(exemplarImg_bitmap)
        self.txt_exemplarTag.SetLabel(str(grouptag))
        self.txt_curexmplidx.SetLabel(str(exmpl_idx+1))
        self.txt_totalexmplidxs.SetLabel(str(len(exemplar_paths)))
        self.txt_curlabel.SetLabel(str(grouptag))
        self.Layout()

    def handle_nomoregroups(self):
        SplitOverlays.handle_nomoregroups(self)
        if self.auto_ondone:
            self.stop()

    def onButton_matches(self, evt):
        curgroup = self.groups[self.idx]
        curtag = curgroup.tag
        self.finished_groups.setdefault(curtag, []).append(curgroup)
        self.remove_group(curgroup)
        print "FinishedGroups:", self.finished_groups

    def onButton_manual_relabel(self, evt):
        dlg = ManualRelabelDialog(self, self.possible_tags)
        status = dlg.ShowModal()
        if status == wx.CANCEL:
            return
        sel_tag = dlg.tag
        self.select_exmpl_group(sel_tag, self.get_current_group().exmpl_idx)
    def onButton_nextexmpl(self, evt):
        curtag = self.get_current_group().tag
        nextidx = self.exmplidx_sel + 1
        self.select_exmpl_group(curtag, nextidx)
    def onButton_prevexmpl(self, evt):
        curtag = self.get_current_group().tag
        previdx = self.exmplidx_sel - 1
        self.select_exmpl_group(curtag, previdx)

class VerifyOrFlagOverlays(VerifyOverlays):
    """ A widget that lets you either verify overlays, or flag a group
    to set aside (Quarantine). 
    """
    def __init__(self, parent, *args, **kwargs):
        VerifyOverlays.__init__(self, parent, *args, **kwargs)

        # list self.QUARANTINED_GROUPS: List of [obj group_i, ...]
        self.quarantined_groups = None

    def init_ui(self):
        VerifyOverlays.init_ui(self)
        
        self.btn_quarantine = wx.Button(self, label="Quarantine")
        self.btn_quarantine.Bind(wx.EVT_BUTTON, self.onButton_quarantine)
        self.btn_sizer.Add(self.btn_quarantine)

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
            self.ondone(verify_results, quarantined_results)

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

    def onButton_quarantine(self, evt):
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
            verifypanelClass = VerifyOverlays
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
                self.ondone(self.verify_results_cat)

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
        
    def init_ui(self):
        VerifyOverlays.init_ui(self)

        btn_no = wx.Button(self, label="Doesn't Match")
        btn_no.Bind(wx.EVT_BUTTON, self.onButton_no)
        
        self.btn_sizer.Add(btn_no)

        self.btn_manual_relabel.Hide()
        self.sizer_exmpls.ShowItems(False)
        self.sizer_curlabel.ShowItems(False)

        self.Layout()

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

    def onButton_no(self, evt):
        curgroup = self.get_current_group()
        self.finished_groups.setdefault(self.TAG_NO, []).append(curgroup)
        self.remove_group(curgroup)
    def handle_nomoregroups(self):
        self.export_results()
        self.Close()

class SeparateImages(VerifyOverlays):
    """ A widget that lets a user separate a set of images into different
    categories, where the number of categories isn't known in advance.
    """
    TAG_UNIVERSAL = "TAG"

    def __init__(self, parent, *args, **kwargs):
        VerifyOverlays.__init__(self, parent, *args, **kwargs)
        
    def init_ui(self):
        VerifyOverlays.init_ui(self)

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
            self.ondone(verify_results)

    def onButton_explode(self, evt):
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

    def onButton_realign(self, evt):
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
            minimg, maximg = make_overlays.minmax_cv(self.imgpaths, do_align=self.do_align,
                                                     rszFac=0.75, bbs_map=bbs_map)
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

def test_verifyoverlays():
    class TestFrame(wx.Frame):
        def __init__(self, parent, imggroups, exemplars, *args, **kwargs):
            wx.Frame.__init__(self, parent, size=(600, 500), *args, **kwargs)

            self.imggroups = imggroups

            self.viewoverlays = VerifyOverlays(self)#ViewOverlays(self)

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

            self.chkimgequals = CheckImageEquals(self)

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

            self.separateimages = SeparateImages(self)

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
    #test_verifyoverlays()
    #test_checkimgequal()
    #test_verifycategories()
    test_separateimages()

if __name__ == '__main__':
    main()
