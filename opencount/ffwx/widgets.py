'''
A set of wrapper classes designed to make commonly-used UI idioms
smaller and clearer.
'''

import multiprocessing
import Queue
import textwrap
import wx
from wx.lib.pubsub import pub


class Panel(wx.Panel):
    '''
    A wrapper that all visible panels should inherit from.
    '''

    def start(self, project=None, projdir=None, size=None):
        '''
        Set up the correct state for the tab
        '''
        raise NotImplentedError()

    def stop(self):
        '''
        Leave the relevant tab
        '''
        raise NotImplentedError()

    def can_move_on(self):
        '''
        Returns a boolean indicating whether the tasks associated
        with this panel have been completed, as well as a message
        to be used in the case that something is left to do.
        '''
        return self.run_sanity_checks(), '[ATTENTION NEEDED]'

    def run_sanity_checks(self):
        '''
        Return True if everything is fine
        '''
        return True


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


def vbox(*contents, **kwargs):
    '''
    A wrapper function for creating and populating a
    vertical BoxSizer.
    '''
    sizer = wx.BoxSizer(wx.VERTICAL, **kwargs)
    sizer.AddMany([(x,) for x in contents])
    return sizer


def hbox(*contents, **kwargs):
    '''
    A wrapper function for creating and populating a
    horizontal BoxSizer.
    '''
    sizer = wx.BoxSizer(wx.HORIZONTAL, **kwargs)
    sizer.AddMany([(x,) for x in contents])
    return sizer


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
    return dialog.ShowModal()
