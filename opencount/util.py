import contextlib
import itertools
import math
import multiprocessing
import os
import re
import sys
import time

import cv
import numpy as np
import wx
from wx.lib.pubsub import pub
try:
    import Image
except:
    from PIL import Image

import ffwx
import util_gui


class Steps:
    PROJECT = 0
    CONFIG = 1
    PARTITION = 2
    BALLOT_ATTRIBUTES = 3
    LABEL_DIGATTRS = 4
    RUN_GROUPING = 5
    CORRECT_GROUPING = 6
    SELTARGETS = 7
    LABEL_CONTESTS = 8
    TARGET_EXTRACT = 9
    SET_THRESHOLD = 10
    QUARANTINE = 11
    PROCESS = 12

if sys.stderr.isatty():
    SIGIL_DICT = dict(
        warn='\x1b[93m?',
        error='\x1b[91m!',
        info='@',
    )
    CODA = '\x1b[39m'
else:
    SIGIL_DICT = dict(warn='?', error='!', debug='@')
    CODA = ''


def debug(msg, *args, **kwargs):
    dbg_level = kwargs.pop('debug_level', 'info')
    sigil = SIGIL_DICT.get(dbg_level, '@')
    stamped = '{sigil}{time}: {msg}{coda}\n'.format(
        time=time.time(),
        msg=msg.format(*args, **kwargs),
        sigil=sigil,
        coda=CODA)
    sys.stderr.write(stamped)


def warn(str, *args, **kwargs):
    debug(str, *args, debug_level='warn', **kwargs)


def error(str, *args, **kwargs):
    debug(str, *args, debug_level='error', **kwargs)


class MyGauge(wx.Dialog):
    """
    A dialog that pops up to display a progress gauge when some
    long-running process is running.
    """

    def __init__(self, parent, numtasks, funs=None,
                 msg="Please wait...", ondone=None, ispanel=None,
                 destroyondone=True, thread=None, size=(400, 300),
                 job_id=None, *args, **kwargs):
        """
        obj parent: Parent widget
        int numtasks:
        lst funs:
        str msg: Message displayed to user.
        fn ondone:
        bool ispanel:
        bool destroyondone:
        obj thread:
        obj job_id: If given, this MyGauge will only listen to events
                    that have a matching 'job_id'. This is to allow
                    multiple MyGauge instances to co-exist peacefully.
                    If no job_id is given, then this MyGauge instance
                    will react to /all/ PubSub events that don't have
                    a job_id in the msg.data. This is an instance of
                    GaugeID.
        """
        self.job_id = job_id
        if not self.job_id:
            pass
            # raise Exception("All Gauges now must have a Job ID.")

        self.destroyondone = destroyondone
        wx.Dialog.__init__(self, parent, size=size, *args, **kwargs)

        self.thethread = thread
        self.ondone = ondone
        if funs is not None:
            self.funs = funs
        else:
            self.funs = [None] * numtasks

        self.parent = parent
        panel = wx.Panel(self)

        self.val = 0
        self.numtasks = numtasks
        self.upto = 0
        self.ontask = 0

        self.txt1 = wx.StaticText(panel, label=msg)
        self.txt2 = wx.StaticText(panel, label="")
        self.txt3 = wx.StaticText(panel, label="")
        self.txt4 = wx.StaticText(panel, label="")
        self.gauge = wx.Gauge(panel, size=(200, 25))
        self.btn_abort = wx.Button(panel, label="Abort")
        self.btn_abort.Bind(wx.EVT_BUTTON, self.onbutton_abort)

        panel.sizer = wx.BoxSizer(wx.VERTICAL)
        panel.sizer.Add(self.txt1)
        panel.sizer.Add(self.txt2)
        panel.sizer.Add(self.txt3)
        panel.sizer.Add(self.txt4)
        panel.sizer.Add(self.gauge)
        panel.sizer.Add(self.btn_abort)
        panel.SetSizer(panel.sizer)
        panel.Fit()
        self.Fit()

        signal = 'signals.MyGauge'
        pub.subscribe(self._pubsub_done, signal + '.done')
        pub.subscribe(self._pubsub_nextjob, signal + '.nextjob')
        pub.subscribe(self._pubsub_tick, signal + '.tick')

        self.inittime = self.startTime = time.time()
        self.updater = wx.CallLater(1000, self.update)
        self.finishedon = -1
        self.dead = False

    def my_id(self):
        return self.job_id

    def is_event_relevant(self, msg):
        """
        Return True iff I should react to this event.
        If I have no job_id:
            True iff msg has no job_id
        If I have a job_id:
            True iff msg has a job_id, and it matches self.my_id()
        """
        if isinstance(msg, list) or isinstance(msg, tuple):
            last_item = msg[-1]
            if isinstance(last_item, GaugeID):
                return self.my_id() == last_item
            else:
                return False if self.my_id() else True
        else:
            return False if self.my_id() else True

    def _munge_msg(self, msg):
        """
        For backwards compatibility, remove the 'job_id' argument
        in msg.data, returning a stripped msg.data.
        Also, if after stripping job_id, the resulting data is a
        tuple with one element, this method returns just the
        element.
        """
        if isinstance(msg, list) or isinstance(msg, tuple):
            last_item = msg[-1]
            if isinstance(last_item, GaugeID):
                msg = msg[:-1]
                if len(msg) == 1:
                    msg = msg[0]
        return msg

    def _pubsub_done(self, msg):
        if msg[0] is not None and not self.is_event_relevant(msg):
            return
        pub.unsubscribe(self._pubsub_done, "signals.MyGauge.done")
        pub.unsubscribe(self._pubsub_tick, "signals.MyGauge.tick")
        pub.unsubscribe(self._pubsub_nextjob, "signals.MyGauge.nextjob")
        if self.ondone:
            self.ondone()
        if self.destroyondone:
            self.Destroy()
        else:
            self.dead = True
            self.txt1.SetLabel("Computation finished.")
            t = time.time() - self.inittime
            self.txt3.SetLabel("Time taken: " + str(self.totime(t)))
            self.txt4.SetLabel("")

    def totime(self, it):
        r = []

        it = int(it)

        kw = [(60, 's'), (60, 'm'), (24, 'h'), (365, 'd'), (1 << 30, 'y')]
        for t, word in kw:
            if it > 0:
                v = it % t
                r.append(str(v) + word)
                it /= t
        if len(r) == 0:
            return "0s"
        return ", ".join(r[::-1])

    def update(self):
        if self.dead:
            return

        self.updater.Restart()
        t = time.time() - self.startTime

        text = "On task %d of %d" % (self.ontask, self.numtasks)
        self.txt2.SetLabel(text)
        self.txt3.SetLabel("Elapsed Time: " + self.totime(t))
        if self.val == 0:
            self.txt4.SetLabel("Expected Completion: unknown seconds")
        else:
            if self.finishedon != -1:
                guess = int((float(self.upto + 1) / self.val) *
                            (self.finishedon - self.startTime) - t)
            else:
                guess = int((float(self.upto + 1) / self.val) *
                            (time.time() - self.startTime) - t)
            guess = max(guess, 0)
            self.txt4.SetLabel("Completion In: " + self.totime(guess))
        if self.ontask - 1 >= len(self.funs):
            print "Tried to access array out of bounds in MyGaguge"
            print "Likely that there were more tasks than functions given."
            return
        fnc = self.funs[self.ontask - 1]
        if fnc is not None:
            self.val = fnc()
            self.gauge.SetValue(self.val)

    def _pubsub_nextjob(self, msg):
        if not self.is_event_relevant(msg):
            return
        data = self._munge_msg(msg)
        self.gauge.SetRange(data)
        self.upto = data
        self.gauge.SetValue(0)
        self.ontask += 1
        self.val = 0
        self.startTime = time.time()
        self.update()

    def _pubsub_tick(self, msg):
        if not self.is_event_relevant(msg):
            return
        self.finishedon = time.time()
        self.val += 1
        self.gauge.SetValue(self.val)

    def onbutton_abort(self, evt):
        if self.thethread:
            self.thethread.abort()
            self.dead = True
            self.txt1.SetLabel("Computation aborted.")
            t = time.time() - self.inittime
            self.txt3.SetLabel("Time taken: " + str(self.totime(t)))
            self.txt4.SetLabel("")

    def next_job(self, num_tasks):
        wx.CallAfter(pub.sendMessage,
                     "signals.MyGauge.nextjob",
                     msg=(num_tasks, self.job_id))

    def tick(self):
        wx.CallAfter(pub.sendMessage,
                     "signals.MyGauge.tick",
                     msg=(self.job_id,))

    def done(self):
        wx.CallAfter(pub.sendMessage,
                     "signals.MyGauge.done",
                     msg=(self.job_id,))

    @staticmethod
    def all_next_job(num_tasks):
        pass
        # raise Exception('REPLACE ME')

    @staticmethod
    def all_tick():
        pass
        # raise Exception('REPLACE ME')

    @staticmethod
    def all_done():
        print 'ALL DONE'
        wx.CallAfter(pub.sendMessage,
                     "signals.MyGauge.done",
                     msg=(None,))


class GaugeID(object):
    new_id = 0
    """
    A class to facilitate JOB_IDS for MyGauge. This also
    provides abstractions for manipulating timers themselves.
    """

    def __init__(self, job_id=None):
        self.job_id = self.__class__.new_id
        self.__class__.new_id += 1

    def __eq__(self, other):
        try:
            return self.job_id == other.job_id
        except AttributeError as e:
            return False

    def __repr__(self):
        return 'GaugeID({0})'.format(self.job_id)

    def next_job(self, num_tasks):
        wx.CallAfter(pub.sendMessage,
                     "signals.MyGauge.nextjob",
                     msg=(num_tasks, self))

    def tick(self):
        wx.CallAfter(pub.sendMessage,
                     "signals.MyGauge.tick",
                     msg=(self,))

    def done(self):
        wx.CallAfter(pub.sendMessage,
                     "signals.MyGauge.done",
                     msg=(self,))


class Gauges:
    '''
    An object for holding all the possible GaugeIds for
    all the gauges used throughout the application.
    '''
    infer_contests = GaugeID()
    label_digit_match = GaugeID()
    digit_match = GaugeID()
    partitioning_export_results = GaugeID()
    partitioning = GaugeID()
    extract_barcode_marks = GaugeID()
    template_match_targets = GaugeID()
    global_align = GaugeID()
    blank_straighten_ballots = GaugeID()
    voted_straighten_ballots = GaugeID()
    grouping_imagebased = GaugeID()
    grouping_digitbased = GaugeID()
    find_attr_matches = GaugeID()
    compute_mult_exemplars = GaugeID()
    generate_mix_max_overlays = GaugeID()
    select_attrs = GaugeID()
    extract_targets = GaugeID()
    targets_create_image = GaugeID()


class MyTimer(object):

    def __init__(self, filepath):
        self.filepath = filepath
        self.start_times = {}  # {str task: int starttime}
        self.tasks_ordered = []  # [str task0, ..., str taskN]
        self.total_times = {}  # {str task: int dur}

    def prelude(self, f):
        """
        Include a separator indicating that we're starting to log
        to the logfile for this session.
        """
        import datetime
        print >>f, "=" * 16
        print >>f, "Beginning new log session, on:"
        print >>f, datetime.datetime.now()
        print >>f, "=" * 16

    def start_task(self, task):
        self.start_times[task] = time.time()
        if task not in self.tasks_ordered:
            self.tasks_ordered.append(task)

    def stop_task(self, task):
        if task not in self.start_times:
            print "Warning -- task {0} was not found in MyTimer.".format(task)
        else:
            dur = time.time() - self.start_times.pop(task)
            if task not in self.total_times:
                self.total_times[task] = dur
            else:
                self.total_times[task] += dur
        self.dump()

    def dump(self):
        """
        Outputs all timing information to the logfile.
        """
        f = open(self.filepath, 'w')
        self.prelude(f)
        # First, sum up all times within each task
        for task in self.tasks_ordered:
            dur = self.total_times.get(task, "UNKNOWN")
            print >>f, "Task '{0}':".format(task)

            if dur == "UNKNOWN":
                # Record all currently-running tasks
                dur = time.time() - self.start_times[task]

            print >>f, "    {0:.8f} seconds".format(dur)
        print "(MyTimer) Writing timing statistics to: {0}".format(os.path.abspath(self.filepath))
        f.flush()
        f.close()


class WarningDialog(wx.Dialog):
    """
    A custom dialog to display a warning message to the user, in
    addition to displaying custom button labels.
    """

    def __init__(self, parent, warn_msg, btn_labels, status_vals, *args, **kwargs):
        wx.Dialog.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        txt = wx.StaticText(self, label=warn_msg)
        txt.Wrap(600)
        btns = []
        sizer_btns = wx.BoxSizer(wx.HORIZONTAL)
        for i, btn_label in enumerate(btn_labels):
            btn = wx.Button(self, label=btn_label)
            btn.Bind(wx.EVT_BUTTON, self.onButton)
            btns.append(btn)
            btn_statusval = status_vals[i]
            btn._mystatusval = btn_statusval
            sizer_btns.Add(btn, border=5, flag=wx.ALL)
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(txt, border=20, flag=wx.ALL)
        sizer.Add(sizer_btns, border=10, flag=wx.BOTTOM | wx.ALIGN_CENTER)
        self.SetSizer(sizer)
        self.Fit()

    def onButton(self, evt):
        btn = evt.GetEventObject()
        self.EndModal(btn._mystatusval)


def is_image_ext(filename):
    IMG_EXTS = ('.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff')
    return os.path.splitext(filename)[1].lower() in IMG_EXTS


def get_filename(filepath, NO_EXT=False):
    """
    Given the path to a file, return the filename. If NO_EXT is True,
    then strip out the extension.
    Ex:
    >>> get_filename('foo/bar/baz.jpg')
    baz.jpg
    >>> get_filename('baz.jpg')
    baz.jpg
    >>> get_filename('foo/bar/garply.png', NO_EXT=True)
    garply
    """
    filename = os.path.split(filepath)[1]
    return os.path.splitext(filename)[0] if NO_EXT else filename


def create_dirs(*dirs):
    """
    For each dir in dirs, create the directory if it doesn't yet
    exist. Will work for things like:
        foo/bar/baz
    and will create foo, foo/bar, and foo/bar/baz correctly.
    """
    for dir in dirs:
        try:
            os.makedirs(dir)
        except Exception as e:
            pass


# http://wiki.wxpython.org/WorkingWithImages

# Tested with wxPython 2.3.4.2 and PIL 1.1.3.

def pil2wxb(myPilImage):
    return WxImageToWxBitmap(PilImageToWxImage(myPilImage))

# Or, if you want to copy any alpha channel, too (available since wxPython 2.5)
# The source PIL image doesn't need to have alpha to use this routine.
# But, a PIL image with alpha is necessary to get a wx.Image with alpha.


def PilImageToWxImage(myPilImage, copyAlpha=True):

    hasAlpha = myPilImage.mode[-1] == 'A'
    if copyAlpha and hasAlpha:  # Make sure there is an alpha layer copy.

        myWxImage = wx.EmptyImage(*myPilImage.size)
        myPilImageCopyRGBA = myPilImage.copy()
        myPilImageCopyRGB = myPilImageCopyRGBA.convert('RGB')    # RGBA --> RGB
        myPilImageRgbData = myPilImageCopyRGB.tobytes()
        myWxImage.SetData(myPilImageRgbData)
        # Create layer and insert alpha values.
        myWxImage.SetAlphaData(myPilImageCopyRGBA.tobytes()[3::4])

    else:    # The resulting image will not have alpha.

        myWxImage = wx.EmptyImage(*myPilImage.size)
        myPilImageCopy = myPilImage.copy()
        # Discard any alpha from the PIL image.
        myPilImageCopyRGB = myPilImageCopy.convert('RGB')
        myPilImageRgbData = myPilImageCopyRGB.tobytes()
        myWxImage.SetData(myPilImageRgbData)

    return myWxImage

# -----


def imageToPil(myWxImage):
    myPilImage = Image.new(
        'RGB', (myWxImage.GetWidth(), myWxImage.GetHeight()))
    myPilImage.fromstring(myWxImage.GetData())
    return myPilImage


def WxImageToWxBitmap(myWxImage):
    return myWxImage.ConvertToBitmap()

# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## #
# # Additional methods to quickly convert between wx* and numpy/cv # #
# ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## #


def wxImage2np(Iwx, is_rgb=True):
    """ Converts wxImage to numpy array """
    w, h = Iwx.GetSize()
    Inp_flat = np.frombuffer(Iwx.GetDataBuffer(), dtype='uint8')
    if is_rgb:
        Inp = Inp_flat.reshape(h, w, 3)
    else:
        Inp = Inp_flat.reshape(h, w)
    return Inp


def np2wxImage(nparray):
    """ Converts a numpy array to a wxImage. Note that wxImages are
    always RGB (3-channeled) - if a single-channel (grayscale) nparray
    image is passed, then it will be silently-converted to a three-channel
    image.
    """
    if len(nparray.shape) == 2:
        Inp = np.zeros((nparray.shape[0], nparray.shape[
                       1], 3), dtype=nparray.dtype)
        Inp[:, :, 0] = nparray
        Inp[:, :, 1] = nparray
        Inp[:, :, 2] = nparray
    else:
        Inp = nparray
    h, w, channels = Inp.shape
    image = wx.EmptyImage(w, h)
    image.SetData(Inp.tobytes())
    return image


def np2wxBitmap(nparray):
    """ Converts a numpy array to a wxBitmap. Single-channel (grayscale)
    nparray images will be 'treated' as a three-channel image (see the
    docstring for np2wxImage).
    """
    wximg = np2wxImage(nparray)
    return wximg.ConvertToBitmap()


class MainFrame(wx.Frame):

    def __init__(self, parent, *args, **kwargs):
        wx.Frame.__init__(self, parent, size=(800, 800),
                          title='Image Manipulate', *args, **kwargs)
        self.img_manipulate = ImageManipulate(self, size=(500, 600))
        self.img_manipulate.set_image(Image.open('a_election/colored_test/20111129143441041_0001-a.jpg'),
                                      scale=2.0)
        self._contest = 0

        self.btn_zoomin = wx.Button(self, label="Zoom in")
        self.btn_zoomin.Bind(wx.EVT_BUTTON, self.onButton_zoomin)
        self.btn_zoomout = wx.Button(self, label="Zoom out")
        self.btn_zoomout.Bind(wx.EVT_BUTTON, self.onButton_zoomout)

        self.btn_resize = wx.Button(self, label="Resize")
        self.btn_resize.Bind(wx.EVT_BUTTON, self.onButton_resize)

        self.btn_focus = wx.Button(self, label="Focus on a contest")
        self.btn_focus.Bind(wx.EVT_BUTTON, self.onButton_focus)

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        # self.sizer.Add(self.img_manipulate, proportion=1, flag=wx.SHAPED)
        self.sizer.Add(self.img_manipulate, proportion=0)
        self.sizer.Add(self.btn_zoomin)
        self.sizer.Add(self.btn_zoomout)
        self.sizer.Add(self.btn_resize)
        self.sizer.Add(self.btn_focus)
        self.SetSizer(self.sizer)

    def onButton_zoomin(self, evt):
        self.img_manipulate.zoomin(0.5)

    def onButton_zoomout(self, evt):
        self.img_manipulate.zoomout(0.5)

    def onButton_resize(self, evt):
        self.img_manipulate.set_size((200, 200))
        self.Layout()

    def onButton_focus(self, evt):
        contests = ((6, 154, 54, 366), (6, 566, 54, 67),
                    (143, 328, 53, 54), (143, 578, 53, 55))
        contest = contests[self._contest % len(contests)]
        focus_on_contest(self.img_manipulate, contest)
        self._contest += 1


class ImageManipulate(wx.Panel):
    """
    A widget that both displays an image, and allows you to pan
    around the image by clicking and dragging. Also lets you
    zoom in and out.

    Relevant Methods:
        set_image(img, scale=1.0, center=None)
            Sets the image to display.

        set_center(center)
            If you want a region of interest to be displayed in the
            center of the displayed image, then do:
                imgmanipulate.set_center((x,y))

        set_size(size)
            If you want to explicitly set the size of the viewport,
            call this, i.e.:
                imgman.set_size((300, 400))

        zoomin(amt)
        zoomout(amt)
            I.e., zoomin(0.5), zoomout(0.3)
    """

    STATE_IDLE = 0
    STATE_ZOOMIN = 1
    STATE_ZOOMOUT = 2

    def __init__(self, parent, size=(300, 300), *args, **kwargs):
        wx.Panel.__init__(self, parent, size=size, *args, **kwargs)

        # Instance vars
        self.parent = parent
        self.img = None
        self.center = (size[0] / 2, size[1] / 2)
        self.scale = 1.0
        self.state = ImageManipulate.STATE_IDLE

        # set dummy image at first
        dummy_img = np.ones((300, 300))
        dummy_img *= 200
        self.img = dummy_img

        # Tells us which part of the image is currently displayed.
        # i.e. if region=((30,50),(200,200)), this tell us that the
        # pixels in the bounding box [30,50,200,200] are currently
        # visible.
        self.region = ((0, 0), size)

        self._buffer = wx.EmptyBitmap(size[0], size[1])
        self._dirty = True

        self._oldx = 0
        self._oldy = 0

        self.sizer = wx.BoxSizer(wx.VERTICAL)

        self.SetSizer(self.sizer)
        self.SetMinSize(size)

        self.Bind(wx.EVT_LEFT_DOWN, self.onLeftDown)
        self.Bind(wx.EVT_MOTION, self.onMotion)
        self.Bind(wx.EVT_PAINT, self.onPaint)
        self.Bind(wx.EVT_SIZE, self.onSize)

    def set_image(self, img, scale=1.0, center=None):
        """
        Display 'img' with a certain scale factor. If 'center' is given,
        then the view will be shifted such that the point demarcated by
        'center' is at the center of the screen.
        Input:
            obj img: Either a scipy or PIL image.
            float scale: A float from >=0.0, i.e. scale=2.0 means zoom in
                         200%, scale=0.5 means zoom out.
            tuple center: A tuple (x,y). If not given, then default to
                          the center of the img.
        """
        if img is None:
            img = np.ones((300, 300))
            img *= 200
        if type(img) != type(np.array(1)):
            img = np.array(img)
        self.scale = scale
        self.img = img
        self._dirty = True
        if not center:
            center = (int(round(self.img.shape[1] / 2.0)),
                      int(round(self.img.shape[0] / 2.0)))
        self.center = center
        self.Refresh()

    def set_size(self, size):
        """
        Set the size of the viewport.
        Input:
            tuple size: (w,h)
        """
        self.SetMinSize(size)
        self.SetClientSize(size)
        self._dirty = True
        self.Refresh()

    def set_scale(self, scale):
        """
        Either zoom in/out on the currently displayed image.
        Input:
            float scale: A number >0.0. 1.0 is the 'native' resolution,
                         2.0 is zoomed in, 0.5 is zoomed out, etc.
        """
        self.scale = scale
        self._dirty = True
        self.Refresh()

    def set_center(self, center):
        """
        Set the focus to a point given by 'center'.
        Input:
            tuple center: (x,y)
        """
        self.center = center
        self._dirty = True
        self.Refresh()

    def zoomin(self, amt=0.5):
        self.set_scale(self.scale * 2)

    def zoomout(self, amt=0.5):
        self.set_scale(self.scale / 2)

    def move_image(self, xamt, yamt):
        """
        Shift the currently displayed image by xamt, yamt. This won't
        let you move outside the bounds of the image.
        """
        width, height = self.GetClientSize()
        x, y = self.center
        x -= xamt
        y -= yamt

        def compute():
            w, h = width / 2 / self.scale, height / 2 / self.scale
            return x - w, y - h, x + w, y + h
        ul_x, ul_y, dr_x, dr_y = compute()
        if ul_x < 0:
            x += -ul_x
            ul_x, ul_y, dr_x, dr_y = compute()
        if dr_x > self.img.shape[1]:
            x -= dr_x - self.img.shape[1]
            ul_x, ul_y, dr_x, dr_y = compute()
        if ul_y < 0:
            y += -ul_y
            ul_x, ul_y, dr_x, dr_y = compute()
        if dr_y > self.img.shape[0]:
            y -= dr_y - self.img.shape[0]
            ul_x, ul_y, dr_x, dr_y = compute()
        self.set_center((x, y))

    def _update_buffer(self):
        """
        Update self._buffer.
        """
        self._dirty = False

        view = self.GetClientSize()
        view = (view[0] / self.scale, view[1] / self.scale)
        imgview, region = crop_img(self.img, view, self.center)
        imgview = fastResize(imgview, self.scale)

        h = imgview.shape[0]
        w = imgview.shape[1]
        if w != self._buffer.GetWidth() or h != self._buffer.GetHeight():
            self._buffer = wx.EmptyBitmap(w, h)
        dc = wx.MemoryDC(self._buffer)
        dc.SetBackground(wx.Brush("White"))
        dc.Clear()
        dc.DrawBitmap(util_gui.NumpyToWxBitmap(imgview), 0, 0)
        dc.SelectObject(wx.NullBitmap)

    # Event Handlers
    def onLeftDown(self, evt):
        # self.GetFocus()
        pass

    def onLeftUp(self, evt):
        pass

    def onMotion(self, evt):
        x, y = evt.GetPosition()
        if self.state == ImageManipulate.STATE_IDLE:
            if evt.LeftIsDown():
                xdelta = x - self._oldx
                ydelta = y - self._oldy
                self.move_image(xdelta, ydelta)

        self._oldx = x
        self._oldy = y

    def onPaint(self, event):
        """ Refresh screen. """
        if self.IsDoubleBuffered():
            dc = wx.PaintDC(self)
        else:
            dc = wx.BufferedPaintDC(self)
        if self._dirty:
            self._update_buffer()
        dc.DrawBitmap(self._buffer, 0, 0)
        event.Skip()

    def onSize(self, event):
        self._dirty = True
        self.Refresh()


def crop_img(img, size, center=(0, 0)):
    """
    Given a numpy array, return a size=(w,h) crop, centered at the
    given 'center'.
    """
    w, h = size
    h_img = img.shape[0]
    w_img = img.shape[1]
    xc, yc = center
    ul_x = int(round(xc - (w / 2.0)))
    ul_y = int(round(yc - (h / 2.0)))
    lr_x = int(round(xc + (w / 2.0)))
    lr_y = int(round(yc + (h / 2.0)))
    if ul_x < 0:
        lr_x += abs(ul_x)
    elif lr_x > w_img - 1:
        ul_x -= (lr_x - (w_img - 1))
    if ul_y < 0:
        lr_y += abs(ul_y)
    elif lr_y > h_img - 1:
        ul_y -= (lr_y - (h_img - 1))
    ul_x = max(0, ul_x)
    ul_y = max(0, ul_y)
    lr_x = min(w_img - 1, lr_x)
    lr_y = min(h_img - 1, lr_y)

    return img[ul_y:lr_y, ul_x:lr_x], (ul_x, ul_y, lr_x, lr_y)


def fastResize(I, rszFac):
    Icv = cv.fromarray(np.copy(I))
    I1cv = cv.CreateMat(int(math.floor(I.shape[0] * rszFac)),
                        int(math.floor(I.shape[1] * rszFac)), Icv.type)
    cv.Resize(Icv, I1cv)
    Iout = np.asarray(I1cv)
    return Iout


def focus_on_contest(imgman, contest_bbox):
    """
    Given an ImageManipulate and a bounding box for a contest, set up
    the ImageManipulate such that the contest is in focus. This also
    resizes the ImageManipulate to 'fit' to the contest_bbox.
    Input:
        obj imgnam: An ImageManipulate instance
        tuple contest_bbox: A tuple of the form (x, y, width, height)
    """
    x, y, w, h = contest_bbox
    xc, yc = (int(round(x + (w / 2.0))),
              int(round(y + (h / 2.0))))
    imgman.set_center((xc, yc))
    # Now, resize the imgman
    w_parent, h_parent = imgman.parent.GetClientSize()
    new_h = min(h_parent - 20, int(round(imgman.scale * h)))
    imgman.set_size((int(round(imgman.scale * w)),
                     new_h))
    imgman.parent.Layout()

"""
For compatibility with Python 2.6, including the Counter class in this
file.
Counter class is pulled from:
    http://code.activestate.com/recipes/576611/
"""


class Counter(dict):
    '''Dict subclass for counting hashable objects.  Sometimes called a bag
    or multiset.  Elements are stored as dictionary keys and their counts
    are stored as dictionary values.

    >>> Counter('zyzygy')
    Counter({'y': 3, 'z': 2, 'g': 1})

    '''

    def __init__(self, iterable=None, **kwds):
        '''Create a new, empty Counter object.  And if given, count elements
        from an input iterable.  Or, initialize the count from another mapping
        of elements to their counts.

        >>> c = Counter()                           # a new, empty counter
        # a new counter from an iterable
        >>> c = Counter('gallahad')
        # a new counter from a mapping
        >>> c = Counter({'a': 4, 'b': 2})
        # a new counter from keyword args
        >>> c = Counter(a=4, b=2)

        '''
        self.update(iterable, **kwds)

    def __missing__(self, key):
        return 0

    def most_common(self, n=None):
        '''List the n most common elements and their counts from the most
        common to the least.  If n is None, then list all element counts.

        >>> Counter('abracadabra').most_common(3)
        [('a', 5), ('r', 2), ('b', 2)]

        '''
        import heapq
        if n is None:
            return sorted(self.iteritems(), key=lambda l: l[1], reverse=True)
        return heapq.nlargest(n, self.iteritems(), key=lambda l: l[1])

    def elements(self):
        '''Iterator over elements repeating each as many times as its count.

        >>> c = Counter('ABCABC')
        >>> sorted(c.elements())
        ['A', 'A', 'B', 'B', 'C', 'C']

        If an element's count has been set to zero or is a negative number,
        elements() will ignore it.

        '''
        for elem, count in self.iteritems():
            for _ in itertools.repeat(None, count):
                yield elem

    def update(self, iterable=None, **kwds):
        '''Like dict.update() but add counts instead of replacing them.

        Source can be an iterable, a dictionary, or another Counter instance.

        >>> c = Counter('which')
        >>> c.update('witch')           # add elements from another iterable
        >>> d = Counter('watch')
        >>> c.update(d)                 # add elements from another counter
        >>> c['h']                      # four 'h' in which, witch, and watch
        4

        '''
        if iterable is not None:
            if hasattr(iterable, 'iteritems'):
                if self:
                    self_get = self.get
                    for elem, count in iterable.iteritems():
                        self[elem] = self_get(elem, 0) + count
                else:
                    # fast path when counter is empty
                    dict.update(self, iterable)
            else:
                self_get = self.get
                for elem in iterable:
                    self[elem] = self_get(elem, 0) + 1
        if kwds:
            self.update(kwds)

    def copy(self):
        'Like dict.copy() but returns a Counter instance instead of a dict.'
        return Counter(self)

    def __delitem__(self, elem):
        'Like dict.__delitem__() but does not raise KeyError for missing values.'
        if elem in self:
            dict.__delitem__(self, elem)

    def __repr__(self):
        if not self:
            return '%s()' % self.__class__.__name__
        items = ', '.join(map('%r: %r'.__mod__, self.most_common()))
        return '%s({%s})' % (self.__class__.__name__, items)

    # Multiset-style mathematical operations discussed in:
    #       Knuth TAOCP Volume II section 4.6.3 exercise 19
    #       and at http://en.wikipedia.org/wiki/Multiset
    #
    # Outputs guaranteed to only include positive counts.
    #
    # To strip negative and zero counts, add-in an empty counter:
    #       c += Counter()

    def __add__(self, other):
        '''Add counts from two counters.

        >>> Counter('abbb') + Counter('bcc')
        Counter({'b': 4, 'c': 2, 'a': 1})


        '''
        if not isinstance(other, Counter):
            return NotImplemented
        result = Counter()
        for elem in set(self) | set(other):
            newcount = self[elem] + other[elem]
            if newcount > 0:
                result[elem] = newcount
        return result

    def __sub__(self, other):
        ''' Subtract count, but keep only results with positive counts.

        >>> Counter('abbbc') - Counter('bccd')
        Counter({'b': 2, 'a': 1})

        '''
        if not isinstance(other, Counter):
            return NotImplemented
        result = Counter()
        for elem in set(self) | set(other):
            newcount = self[elem] - other[elem]
            if newcount > 0:
                result[elem] = newcount
        return result

    def __or__(self, other):
        '''Union is the maximum of value in either of the input counters.

        >>> Counter('abbb') | Counter('bcc')
        Counter({'b': 3, 'c': 2, 'a': 1})

        '''
        if not isinstance(other, Counter):
            return NotImplemented
        _max = max
        result = Counter()
        for elem in set(self) | set(other):
            newcount = _max(self[elem], other[elem])
            if newcount > 0:
                result[elem] = newcount
        return result

    def __and__(self, other):
        ''' Intersection is the minimum of corresponding counts.

        >>> Counter('abbb') & Counter('bcc')
        Counter({'b': 1})

        '''
        if not isinstance(other, Counter):
            return NotImplemented
        _min = min
        result = Counter()
        if len(self) < len(other):
            self, other = other, self
        for elem in itertools.ifilter(self.__contains__, other):
            newcount = _min(self[elem], other[elem])
            if newcount > 0:
                result[elem] = newcount
        return result


def is_multipage(project):
    """
    Currently an ad-hoc method of determining if the current election
    is multipage or not.
    """
    return project.is_multipage


def sort_nicely(l):
    """ Sort the given list in the way that humans expect. Does an inplace sort.
    From:
        http://www.codinghorror.com/blog/2007/12/sorting-for-humans-natural-sort-order.html
    """
    def convert(text):
        return int(text) if text.isdigit() else text

    def alphanum_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key)]

    l.sort(key=alphanum_key)


def sorted_nicely(lst):
    """ Sort the given list in the way that humans expect. Returns a new
    list.
    From:
        http://www.codinghorror.com/blog/2007/12/sorting-for-humans-natural-sort-order.html
    """
    lst_copy = lst[:]
    sort_nicely(lst_copy)
    return lst_copy


def pdb_on_crash(f):
    """
    Decorator to run PDB on crashing
    """
    def res(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            import pdb as err_pdb
            err_pdb.post_mortem()
    return res


def get_memory_stats():
    """ Returns statistics on system memory, in bytes. Requires the
    psutil python module to be installed - if psutil is not found, then
    this raises an Exception.
    Output:
        (int available, int total)
    """
    try:
        import psutil
        virtmem = psutil.virtual_memory()
        return virtmem.available, virtmem.total
    except ImportError:
        raise Exception("Module psutil not detected.")


def as_process(func):
    '''
    This decorator allows us to wrap up a single lexically scoped
    block of code as a thread. Eventually, this helper should ideally
    be phased out by eliminating the need for threaded code in
    general , but it makes a lot of code shorter right now.
    '''
    queue = multiprocessing.Queue()

    class WrappedFunction(multiprocessing.Process):

        def run(self):
            queue.put(func())

    WrappedFunction.__name__ = func.__name__
    w = WrappedFunction()
    w.start()
    return queue.get


@contextlib.contextmanager
def time_operation(op):
    '''
    Time an operation, taking a timing measurement at the beginning
    and at the end and printing them accordingly.
    '''
    t = time.time()
    debug('starting: {0}', op)
    yield
    dur = time.time() - t
    debug('done {0} in {1}s', op, dur)


class InformativeException(Exception):
    '''
    An exception for problems that convey a message to be used in
    some kind of modal dialogue.
    '''

    def __init__(self, message):
        self.message = message

    def __repr__(self):
        return '{0}({1})'.format(
            self.__class__.__name__,
            ','.join('{0}={1}'.format(k, repr(v))
                     for (k, v) in self.__dict__.items()))


def show_exception_as_modal(func):
    '''
    If any InformativeExceptions appear in the context of the
    wrapped function, they will be exposed to the user in a
    modal dialogue, and the function itself will return None.

    This must be used on methods of a WxWidgets widget, because
    the modal dialogue needs a parent.
    '''

    def wrapped_function(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except InformativeException as exn:
            ffwx.modal(self, exn.message)
            return None

    wrapped_function.__name__ = func.__name__
    return wrapped_function


def main():
    app = wx.App(False)
    frame = MainFrame(None)
    frame.Show()
    app.MainLoop()

if __name__ == '__main__':
    main()
