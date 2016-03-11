'''
A set of wrapper classes designed to make commonly-used UI idioms
smaller and clearer.
'''

import ffwx.widgets as ff
import wx

class FFBuilder(object):
    def __init__(self, parent):
        self.parent = parent


    def text(self, label, **kwargs):
        '''
        A wrapper function for creating text labels.
        '''
        return ff.text(self.parent, label, **kwargs)


    def vbox(self, *contents, **kwargs):
        '''
        A wrapper function for creating and populating a
        vertical BoxSizer.
        '''
        return ff.vbox(*contents, **kwargs)

    def hbox(self, *contents, **kwargs):
        '''
        A wrapper function for creating and populating a
        horizontal BoxSizer.
        '''
        return ff.hbox(*contents, **kwargs)

    def static_hbox(self, *contents, **kwargs):
        '''
        A wrapper function for creating and populating a horizontal
        StaticBoxSizer (which contains a text label)
        '''
        return ff.static_hbox(self.parent, *contents, **kwargs)

    def static_vbox(self, *contents, **kwargs):
        '''
        A wrapper function for creating and populating a vertical
        StaticBoxSizer (which contains a text label)
        '''
        return ff.static_vbox(self.parent, *contents, **kwargs)

    def static_wrap(self, msg, length, *args, **kwargs):
        '''
        A wrapper function for creating a simple text-wrapped
        label.
        '''
        return ff.static_wrap(self.parent,
                              msg,
                              length,
                              *args,
                              **kwargs)


    def button(self, *args, **kwargs):
        '''
        A wrapper function for creating an extended ffwx Button.
        '''
        return ff.Button(self.parent, *args, **kwargs)

    def stat_label(self, name, value=None):
        '''
        Create a label for showing key/value pairs.
        '''
        return ff.StatLabel(self.parent, name, value)

    def check_box(self, *args, **kwargs):
        '''
        Create a checkbox with a specified default value.
        '''
        kwargs['default'] = kwargs.get('default', False)
        return ff.CheckBox(self.parent, *args, **kwargs)

    def list_box(self, *args, **kwargs):
        '''
        Create an extended ffwx ListBox
        '''
        return ff.ListBox(self.parent, *args, **kwargs)

    def text_ctrl(self, value='', *args, **kwargs):
        '''
        Create a user-editable text control
        '''
        kwargs['value'] = value
        return wx.TextCtrl(self.parent, *args, **kwargs)

    def combo_box(self, *args, **kwargs):
        '''
        Create a drop-down list of options
        '''
        return wx.ComboBox(self.parent, *args, **kwargs)
