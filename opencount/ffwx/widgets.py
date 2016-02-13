'''
A set of wrapper classes designed to make commonly-used UI idioms
smaller and clearer.
'''

import textwrap
import wx

class FFStatLabel(wx.BoxSizer):
    '''
    A set of labels designed to show key-value pairs. By
    default, the value is empty, but it can be supplied
    with the value keyword, or set with the set_value
    method.
    '''
    def __init__(self, parent, name, value=None):
        wx.BoxSizer.__init__(self, wx.HORIZONTAL)
        self._name  = wx.StaticText(parent, label=name + ": ")
        self._value = wx.StaticText(parent)
        self.AddMany([(self._name,), (self._value,)])
        if value:
            self.set_value(value)

    def set_value(self, val):
        self._value.SetLabel(str(val))
        return self

def ff_vbox(*contents, **kwargs):
    '''
    A wrapper function for creating and populating a
    vertical BoxSizer.
    '''
    sizer = wx.BoxSizer(wx.VERTICAL, **kwargs)
    sizer.AddMany([(x,) for x in contents])
    return sizer

def ff_hbox(*contents, **kwargs):
    '''
    A wrapper function for creating and populating a
    horizontal BoxSizer.
    '''
    sizer = wx.BoxSizer(wx.HORIZONTAL, **kwargs)
    sizer.AddMany([(x,) for x in contents])
    return sizer

def ff_static_wrap(parent, msg, ln, *args, **kwargs):
    st = wx.StaticText(parent,
                       *args,
                       label=textwrap.fill(msg, ln),
                       **kwargs)
    return st

class FFButton(wx.Button):
    '''
    A wrapper for the WXWidgets button that makes supplying button
    actions simpler.
    '''
    def __init__(self, *args, **kwargs):
        on_click = kwargs.pop('on_click', None)
        wx.Button.__init__(self, *args, **kwargs)
        if on_click:
            self.Bind(wx.EVT_BUTTON, on_click)

    def on_click(self, action):
        self.Bind(wx.EVT_BUTTON, action)
        return self

class FFCheckBox(wx.CheckBox):
    def __init__(self, *args, **kwargs):
        default = kwargs.pop('default', False)
        wx.CheckBox.__init__(self, *args, **kwargs)
        self.SetValue(default)
