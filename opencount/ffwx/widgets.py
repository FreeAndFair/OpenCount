'''
A set of wrapper classes designed to make commonly-used UI idioms
smaller and clearer.
'''

import multiprocessing
import Queue
import textwrap

import wx
from wx.lib.pubsub import pub


class ProgressBar(wx.Dialog):
    '''
    This is a disgusting hack.

    The problem here is that
    1. OpenCount uses multiprocessing
    2. wxWidgets progress bars need sporadic nudging in order to
       move forward.
    3. This means a thread of execution needs to sporadically
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
        t_msg = wx.StaticText(panel, label=msg)
        self.t_task = wx.StaticText(panel, label='')
        self.gauge = wx.Gauge(panel, size=(200, 25))

        panel.sizer = vbox(t_msg,
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
        '''
        Try to fetch a message from the mailbox. This allows copies
        of this object in a subprocess to interact with the 'original'
        version of the object.
        '''
        try:
            msg = self.mailbox.get(False)
            wx.CallAfter(pub.sendMessage, self.signal + msg)
        except Queue.Empty:
            if self.num_tasks is None:
                wx.CallAfter(pub.sendMessage, self.signal + '.tick')

    def on_finish(self, callback):
        '''
        Set an action that should happen once the progress bar
        has been destroyed. If that has already happened, schedule
        it to be called as soon as possible.
        '''
        if not self.alive:
            self.finish_cb = callback
        else:
            wx.CallAfter(callback)
        return self

    def update(self):
        '''
        Update the GUI to correspond to the internal state.
        '''
        if self.num_tasks:
            self.t_task.SetLabel('On task {0} of {1}'.format(
                self.val,
                self.num_tasks))
            self.gauge.SetValue(self.val)
        else:
            self.t_task.SetLabel('In progress...')
            self.gauge.Pulse()

    def __enter__(self):
        '''
        Show the widget.
        '''
        self.mailbox.put('.start')
        return self

    def __exit__(self, exn, val, tb):
        '''
        Destroy the widget.
        '''
        self.done()
        return exn

    def _start_event(self):
        '''
        Internal callback: show the widget.

        Must be called in the original process, ideally via
        the pub-sub system.
        '''
        self.Show()

    def _tick_event(self, _=None):
        '''
        Internal callback: update the view.

        Must be called in the original process, ideally via
        the pub-sub system.
        '''
        if self.num_tasks:
            self.val += 1
        self.update()

    def _done_event(self):
        '''
        Internal callback: destroy the loading bar, closing the
        relevant events, stopping the timer, and detroying the
        pop-up.

        Must be called in the original process, ideally via
        the pub-sub system.
        '''
        self.alive = False
        pub.unsubscribe(self._tick_event, self.signal + '.start')
        pub.unsubscribe(self._tick_event, self.signal + '.tick')
        pub.unsubscribe(self._done_event, self.signal + '.done')
        if self.timer:
            self.timer.Stop()
        if self.finish_cb:
            self.finish_cb()
        self.Destroy()

    def tick(self):
        '''
        Tick the progress of the loading bar.
        '''
        self.mailbox.put('.tick')
        return self

    def done(self):
        '''
        Finish showing the loading bar and destroy it.
        '''
        self.mailbox.put('.done')
        return self


class StatLabel(wx.BoxSizer):
    '''
    A set of labels designed to show key-value pairs. By
    default, the value is empty, but it can be supplied
    with the value keyword, or set with the set_value
    method.
    '''

    def __init__(self, parent, name, value=None):
        wx.BoxSizer.__init__(self, wx.HORIZONTAL)
        self._name = wx.StaticText(parent, label=name + ": ")
        self._value = wx.StaticText(parent)
        self.AddMany([(self._name,), (self._value,)])
        if value:
            self.set_value(value)

    def set_value(self, val):
        '''
        Wrapper over label updating.
        '''
        self._value.SetLabel(str(val))
        return self


class BoxSizer(wx.BoxSizer):
    '''
    A wrapper over wx.BoxSizer that makes it easier to build up
    boxes of widgets.
    '''

    def __init__(self, *args, **kwargs):
        wx.BoxSizer.__init__(self, *args, **kwargs)

    def add(self,
            item,
            proportion=1,
            flag=wx.ALL | wx.EXPAND,
            border=8,
            userData=None,
            name=None):
        '''
        A chaining wrapper of the Add method
        '''
        if 'name' is not None:
            self.__dict__[name] = item
        self.Add(item, proportion, flag, border, userData)
        return self

    def add_sizer(self, width, height):
        '''
        A chaining wrapper for adding sizers
        '''
        self.Add((width, height))
        return self


class StaticBoxSizer(wx.StaticBoxSizer):
    '''
    A wrapper over wx.StaticBoxSizer that makes it easier to build up
    boxes of widgets.
    '''

    def __init__(self, *args, **kwargs):
        wx.StaticBoxSizer.__init__(self, *args, **kwargs)

    def add(self,
            item,
            proportion=1,
            flag=wx.ALL | wx.EXPAND,
            border=8,
            userData=None,
            name=None):
        '''
        A chaining wrapper of the Add method
        '''
        if 'name' is not None:
            self.__dict__[name] = item
        self.Add(item, proportion, flag, border, userData)
        return self

    def add_sizer(self, width, height):
        '''
        A chaining wrapper for adding sizers
        '''
        self.Add((width, height))
        return self


def text(parent, label, **kwargs):
    '''
    A wrapper over text labels
    '''
    return wx.StaticText(parent, label=label, **kwargs)


def vbox(*contents, **kwargs):
    '''
    A wrapper function for creating and populating a
    vertical BoxSizer.
    '''
    sizer = BoxSizer(wx.VERTICAL, **kwargs)
    sizer.AddMany([(x, 0, wx.ALL, 8) for x in contents])
    return sizer


def hbox(*contents, **kwargs):
    '''
    A wrapper function for creating and populating a
    horizontal BoxSizer.
    '''
    sizer = BoxSizer(wx.HORIZONTAL, **kwargs)
    sizer.AddMany([(x, 0, wx.ALL, 8) for x in contents])
    return sizer


def static_hbox(parent, *contents, **kwargs):
    '''
    A wrapper function for creating and populating a horizontal
    StaticBoxSizer (which contains a text label)
    '''
    sizer = StaticBoxSizer(wx.StaticBox(parent, label=kwargs['label']),
                           wx.HORIZONTAL)
    sizer.AddMany((x, 0, wx.ALL, 8) for x in contents)
    return sizer


def static_vbox(parent, *contents, **kwargs):
    '''
    A wrapper function for creating and populating a vertical
    StaticBoxSizer (which contains a text label)
    '''
    sizer = StaticBoxSizer(wx.StaticBox(parent, label=kwargs['label']),
                           wx.VERTICAL)
    sizer.AddMany((x, 0, wx.ALL, 8) for x in contents)
    return sizer


def label(parent, msg, *args, **kwargs):
    return wx.StaticText(parent, *args, label=str(msg), **kwargs)


def static_wrap(parent, msg, length, *args, **kwargs):
    '''
    A wrapper function for creating a simple text-wrapped
    label.
    '''
    text = wx.StaticText(parent,
                         *args,
                         label=textwrap.fill(msg, length),
                         **kwargs)
    return text


class ListBox(wx.ListBox):
    '''
    A wrapper for the WXWidgets ListBox that makes adding, removing,
    and setting the set of options easier.
    '''

    def __init__(self, *args, **kwargs):
        wx.ListBox.__init__(self, *args, **kwargs)

    def set_options(self, all_items):
        '''
        Given an iterable of things, clear the list box and
        refresh it with that list of things.
        '''
        self.Clear()
        for item in all_items:
            self.Append(item)

    def get_selected(self):
        '''
        Get the currently selected string, or None if nothing
        is currently selected.
        '''
        if self.GetSelection() == wx.NOT_FOUND:
            return None
        else:
            return self.GetStringSelection()

    def add(self, item):
        '''
        Chaining alias for wx.ListBox.Append
        '''
        self.Append(item)
        return self

    def add_focused(self, item):
        '''
        Chaining alias for wx.ListBox.Append, focusing on the
        newly created element
        '''
        self.Append(item)
        self.SetStringSelection(item)
        return self


class Button(wx.Button):
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
        '''
        Bind a callback for the EVT_BUTTON event.
        '''
        self.Bind(wx.EVT_BUTTON, action)
        return self

    def on_focus(self, action):
        '''
        Bind a callback for the EVT_SET_FOCUS event.
        '''
        self.Bind(wx.EVT_SET_FOCUS, action)
        return self


class CheckBox(wx.CheckBox):
    '''
    A wx.CheckBox wrapper.
    '''

    def __init__(self, *args, **kwargs):
        default = kwargs.pop('default')
        wx.CheckBox.__init__(self, *args, **kwargs)
        self.SetValue(default)

    def on_check(self, action):
        '''
        Bind a callback for the EVT_CHECKBOX event.
        '''
        self.Bind(wx.EVT_CHECKBOX, action)
        return self


def modal(parent, message, show=True, prefix=''):
    '''
    Create a new warning dialog and show it. Wrapper over
    commonly-used wx.MessageDialog code.
    '''
    if prefix:
        msg = prefix + ': ' + message
    else:
        msg = message
    dialog = wx.MessageDialog(
        parent,
        style=wx.OK,
        message=msg,
    )
    if show:
        dialog.ShowModal()
    return dialog


def warn(parent, message, show=True):
    '''
    Toss up a warning dialogue.
    '''
    return modal(parent, message, show=show, prefix='Warning')


def error(parent, message, show=True):
    '''
    Toss up an error dialogue
    '''
    return modal(parent, message, show=show, prefix='Error')


def yesno(parent, message):
    '''
    Toss up a yes/no dialogue, returning the corresponding
    boolean value.
    '''
    dialog = wx.MessageDialog(parent,
                              style=wx.YES_NO,
                              message=message)
    return dialog.ShowModal() == wx.ID_YES


def text_entry(parent, message="", caption="", default=""):
    '''
    Show a text entry field and return the value of the entered
    text, or None if no text was entered.
    '''
    dlg = wx.TextEntryDialog(
        parent,
        message=message,
        caption=caption,
        defaultValue=default)
    val = dlg.ShowModal()
    if val == wx.ID_OK:
        return dlg.GetValue().strip()
    else:
        return None
