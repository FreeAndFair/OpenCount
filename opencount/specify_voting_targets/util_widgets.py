import time, threading
import wx
from wx.lib.pubsub import Publisher

"""
A module to store widgets that might be useful in several
places.
"""

class ProgressGauge(wx.Frame):
    """
    A dialog that pops up to display a progress gauge when some
    long-running process is running.
    """
    def __init__(self, parent, numjobs, msg="Please wait...", *args, **kwargs):
        wx.Frame.__init__(self, parent, size=(400, 300), 
                          style=wx.DEFAULT_FRAME_STYLE | wx.FRAME_FLOAT_ON_PARENT, 
                          *args, **kwargs)
        self.parent = parent
        panel = wx.Panel(self)
        
        self.val = 0        
        self.numjobs = numjobs
        
        txt1 = wx.StaticText(panel, label=msg)
        self.gauge = wx.Gauge(panel, range=numjobs, size=(200, 25))
        self.btn_abort = wx.Button(panel, label="Abort")
        self.btn_abort.Bind(wx.EVT_BUTTON, self.onbutton_abort)
        
        panel.sizer = wx.BoxSizer(wx.VERTICAL)
        panel.sizer.Add(txt1)
        panel.sizer.Add(self.gauge)
        panel.sizer.Add(self.btn_abort)
        panel.SetSizer(panel.sizer)
        panel.Fit()
        self.Fit()
        
        Publisher().subscribe(self._pubsub_done, "signals.ProgressGauge.done")
        Publisher().subscribe(self._pubsub_tick, "signals.ProgressGauge.tick")
        
    def _pubsub_done(self, msg):
        self.Destroy()
    def _pubsub_tick(self, msg):
        self.val += 1
        self.gauge.SetValue(self.val)
    
    def onbutton_abort(self, evt):
        print "Abort not implemented yet. Maybe never."
        #self.Destroy()

class _MainFrame(wx.Frame):
    """
    Frame to demo the ProgressGauge
    """
    def __init__(self, parent, *args, **kwargs):
        wx.Frame.__init__(self, parent, wx.ID_ANY, "title", size=(400, 400), *args, **kwargs)

        btn = wx.Button(self, label="Click to start progress bar demo")
        btn.Bind(wx.EVT_BUTTON, self.onbutton)

    def onbutton(self, evt):
        num_tasks = 10
        progressgauge = ProgressGauge(self, num_tasks, msg="Doing work...")
        progressgauge.Show()
        workthread = _WorkThread(num_tasks)
        workthread.start()
class _WorkThread(threading.Thread):
    def __init__(self, num_tasks, *args, **kwargs):
        threading.Thread.__init__(self, *args, **kwargs)
        self.num_tasks = num_tasks
    def run(self):
        for i in range(self.num_tasks):
            # Do 'work', sending a tick message after every step
            #time.sleep(1.0)
            sum(range(5000000))
            print 'a'
            #Publisher().sendMessage("signals.ProgressGauge.tick")
            wx.CallAfter(Publisher().sendMessage, "signals.ProgressGauge.tick")

        # Notify ProgressGauge that the work is done
        #Publisher().sendMessage("signals.ProgressGauge.done")        
        wx.CallAfter(Publisher().sendMessage, "signals.ProgressGauge.done")

def demo_progressgauge():
    app = wx.App(False)
    frame = _MainFrame(None)
    frame.Show()
    app.MainLoop()

if __name__ == '__main__':
    demo_progressgauge()
