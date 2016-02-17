'''
A set of wrapper classes designed to make commonly-used UI idioms
smaller and clearer.
'''

import multiprocessing
import textwrap
import wx
from wx.lib.pubsub import pub

class FFProgressBar(wx.Dialog):
    '''
    This is a disgusting hack.

    The problem here is that
    1. OpenCount uses multiprocessing
    2. wxWidgets progress bars need sporadic nudging in order to
       move forward.
    3. This means a thread of execution needs to sporadaically
       nudge the progress bar.
    4. It is not safe to use multiprocessing and threading
       simultaneously, for fork/lock reasons.
    5. We have to use the wxwidgets eventing system to avoid

    So, here's what's going on here:
    - The progress bar has a 'mailbox' which is shared between
      processes. This can be written to by several processes,
      but will only be read from the process that created the
      progress bar
    - Various methods that interact with the progress bar will,
      underneath the surface, actually deliver messages to that
      mailbox.
    - A wxTimer in the main process will read from that mailbox
      regularly, forwarding those messages to the wx pub/sub
      system
    - Those will, in turn, ACTUALLY update the object.
    '''
    def __init__(self,
                 parent,
                 num_tasks=None,
                 msg='Please wait...',
                 size=(400, 300),
                 *args,
                 **kwargs):
        wx.Dialog.__init__(self, parent, size=size, *args, **kwargs)

        self.signal = 'signals.progress.{id}'.format(id=id(self))

        panel = wx.Panel(self)
        self.t_msg   = wx.StaticText(panel, label=msg)
        self.t_task  = wx.StaticText(panel, label='')
        self.gauge   = wx.Gauge(panel, size=(200, 25))

        panel.sizer = ff_vbox(self.t_msg,
                              self.t_task,
                              self.gauge)
        panel.SetSizer(panel.sizer)
        panel.Fit()
        self.Fit()

        self.alive = True
        self.mailbox = multiprocessing.Queue()
        self.finish_cb = None

        pub.subscribe(self._start_event, self.signal + '.start')
        pub.subscribe(self._tick_event, self.signal + '.tick')
        pub.subscribe(self._done_event, self.signal + '.done')

        self.num_tasks = num_tasks
        if num_tasks is not None:
            self.val = 0
            self.gauge.SetRange(num_tasks)
            self.timer = None
        else:
            self.val = None
            self.gauge.Pulse()
            self.timer = wx.Timer(self)
            self.Bind(wx.EVT_TIMER,
                      lambda _: self.deliver())
            self.timer.Start(50)

        self.Show()

    def deliver(self):
        try:
            msg = self.mailbox.get(False)
            wx.CallAfter(pub.sendMessage, self.signal + msg)
        except:
            if self.num_tasks is None:
                wx.CallAfter(pub.sendMessage, self.signal + '.tick')

    def on_finish(self, callback):
        if not self.alive:
            self.finish_cb = callback
        else:
            wx.CallAfter(callback)
        return self

    def update(self):
        if self.num_tasks:
            self.t_task.SetLabel('On task {0} of {1}'.format(
                self.val,
                self.num_tasks))
            self.gauge.SetValue(self.val)
        else:
            self.t_task.SetLabel('In progress...')
            self.gauge.Pulse()

    def __enter__(self):
        self.mailbox.put('.start')
        return self

    def __exit__(self, exn, val, tb):
        self.done()
        return exn

    def _start_event(self):
        self.Show()

    def _tick_event(self, event=None):
        if self.num_tasks:
            self.val += 1
        self.update()

    def _done_event(self):
        self.alive = False
        pub.unsubscribe(self._tick_event, self.signal + '.start')
        pub.unsubscribe(self._tick_event, self.signal + '.tick')
        pub.unsubscribe(self._done_event, self.signal + '.done')
        if self.timer: self.timer.Stop()
        if self.finish_cb: self.finish_cb()
        self.Destroy()

    def tick(self):
        self.mailbox.put('.tick')
        return self

    def done(self):
        self.mailbox.put('.done')
        return self

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
        on_focus = kwargs.pop('on_focus', None)
        wx.Button.__init__(self, *args, **kwargs)
        if on_click:
            self.Bind(wx.EVT_BUTTON, on_click)
        if on_focus:
            self.Bind(wx.EVT_SET_FOCUS, on_focus)

    def on_click(self, action):
        self.Bind(wx.EVT_BUTTON, action)
        return self

    def on_focus(self, action):
        self.Bind(wx.EVT_SET_FOCUS, action)
        return self

class FFCheckBox(wx.CheckBox):
    def __init__(self, *args, **kwargs):
        default = kwargs.pop('default')
        wx.CheckBox.__init__(self, *args, **kwargs)
        self.SetValue(default)

def ff_warn(parent, message, show=True):
    dialog = wx.MessageDialog(
        parent,
        style=wx.ID_OK,
        message='Warning: ' + message,
    )
    if show:
        dialog.ShowModal()
    return dialog
