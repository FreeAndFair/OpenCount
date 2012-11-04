import os, sys, traceback, pdb
try:
    import cPickle as pickle
except:
    import pickle

from os.path import join as pathjoin

import wx
from wx.lib.scrolledpanel import ScrolledPanel

class GroupingMainPanel(wx.Panel):
    def __init__(self, parent, *args, **kwargs):
        wx.Panel.__init__(self, parent, *args, **kwargs)
        
        self.proj = None

        self.init_ui()

    def init_ui(self):
        self.btn_rungrouping = wx.Button(self, label="Run Grouping.")
        self.btn_rungrouping.Bind(wx.EVT_BUTTON, self.onButton_rungrouping)
        
        self.btn_continueverify = wx.Button(self, label="Continue Verifying Grouping.")
        self.btn_continueverify.Bind(wx.EVT_BUTTON, self.onButton_continueverify)

        self.btn_rerun_imggroup = wx.Button(self, label="Re-run Image-Based Grouping.")
        self.btn_rerun_imggroup.Bind(wx.EVT_BUTTON, self.onButton_rerun_imggroup)
        self.btn_rerun_digitgroup = wx.Button(self, label="Re-run Digit-Based Grouping.")
        self.btn_rerun_digitgroup.Bind(wx.EVT_BUTTON, self.onButton_rerun_digitgroup)
        txt_rerunOr = wx.StaticText(self, label="- Or -")
        self.btn_rerun_grouping = wx.Button(self, label="Re-run All Grouping.")
        rerun_btnsizer0 = wx.BoxSizer(wx.VERTICAL)
        rerun_btnsizer0.AddMany([(self.btn_rerun_imggroup,0,wx.ALL,10), (self.btn_rerun_digitgroup,0,wx.ALL,10)])
        rerun_btnsizer = wx.BoxSizer(wx.HORIZONTAL)
        rerun_btnsizer.AddMany([(rerun_btnsizer0,0,wx.ALL,10), (txt_rerunOr, 0, wx.ALIGN_CENTER | wx.ALL, 10),
                                (self.btn_rerun_grouping,0,wx.ALIGN_CENTER | wx.ALL, 10)])

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.AddMany([(self.btn_rungrouping,0,wx.ALL,10),
                            (self.btn_continueverify,0,wx.ALL,10),
                            (rerun_btnsizer,0,wx.ALL,10)])
        self.SetSizer(self.sizer)
        self.Layout()

    def start(self, proj):
        self.proj = proj

    def onButton_rungrouping(self, evt):
        pass
    def onButton_continueverify(self, evt):
        pass
    def onButton_rerun_imggroup(self, evt):
        pass
    def onButton_rerun_digitgroup(self, evt):
        pass
