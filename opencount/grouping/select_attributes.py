import os
import sys
import pdb
import threading
import random
import cv
import Image
import wx
import ffwx

try:
    import cPickle as pickle
except ImportError:
    import pickle

from os.path import join as pathjoin


import tempmatch
import util
import common
import specify_voting_targets.util_widgets as util_widgets
import specify_voting_targets.util_gui as util_gui
import verify_overlays_new
import grouping.group_attrs as group_attrs
import grouping.partask as partask

# Global CTR used to ensure unique filenames for outpatches.
CTR = 0


class SelectAttributesPanel(wx.Panel):

    def __init__(self, parent, *args, **kwargs):
        wx.Panel.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        self.patchpaths = None
        self.attr = None
        # maps {str patchpath: [((x1, y1, x2, y2), str label, str subpatchP) ...]}
        # Note: These x1,y1 coords are wrt the PATCH image, not the entire
        #       ballot image.
        self.boxes = {}

        # OUTDIR0: Stores the selected regions that the user made, on
        # each attribute patch.
        self.outdir0 = 'select_attr_outdir0'

        # self.mosaicpanel = util_widgets.MosaicPanel(self, imgmosaicpanel=AttrMosaicPanel,
        # CellClass=PatchPanel, CellBitmapClass=PatchBitmap)
        self.mosaicpanel = MosaicPanel_sub(self)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.mosaicpanel, proportion=1, flag=wx.EXPAND)
        self.SetSizer(sizer)
        self.Fit()

    def start(self, patchpaths, img2flip, attr, boxes=None, outdir0=None):
        self.patchpaths = patchpaths
        self.img2flip = img2flip
        self.attr = attr
        self.boxes = boxes if boxes is not None else {}
        if outdir0 is not None:
            self.outdir0 = outdir0
        self.mosaicpanel.set_images(patchpaths)
        self.Fit()


class MosaicPanel_sub(util_widgets.MosaicPanel):

    def __init__(self, parent, *args, **kwargs):
        util_widgets.MosaicPanel.__init__(self, parent, imgmosaicpanel=AttrMosaicPanel,
                                          CellClass=PatchPanel, CellBitmapClass=PatchBitmap,
                                          *args, **kwargs)

    def init_ui(self):
        util_widgets.MosaicPanel.init_ui(self)
        self.btn_sizer.Add((50, 0))
        self.btn_nextattr = ffwx.Button(
            self,
            label='Next Attribute',
            on_click=lambda evt: self.GetParent().GetParent().next_attribute(),
        )
        self.btn_prevtattr = ffwx.Button(
            self,
            label='Previous Attribute',
            on_click=lambda evt: self.GetParent().GetParent().prev_attribute(),
        )
        self.attrtype = ffwx.StatLabel(self, 'Current Attribute', '(0/0)')
        # txt0 = wx.StaticText(self, label="Current Attribute: ")
        # self.txt_attrtype = wx.StaticText(self, label="Foo (0/0).")
        btn_opts = ffwx.Button(self, label="Options...",
                               on_click=self.onButton_opts)
        btn_hide = ffwx.Button(
            self, label="Hide Labeled Patches", on_click=self.onButton_hide
        )
        btn_show = ffwx.Button(
            self, label="Show Labeled Patches", on_click=self.onButton_show
        )
        self.btn_sizer.AddMany([(self.btn_nextattr,), (self.btn_prevattr,),
                                ((30, 0),),
                                (txt0,), (self.txt_attrtype,),
                                ((50, 0),),
                                (btn_opts,),
                                (btn_hide,), (btn_show,)])
        self.Layout()

    def onButton_hide(self, evt):
        """ Only display patches that don't have a bounding box. """
        patchpaths = []
        for patchpath in self.GetParent().patchpaths:
            if not self.GetParent().boxes.get(patchpath, None):
                patchpaths.append(patchpath)
        self.set_images(patchpaths)

    def onButton_show(self, evt):
        """ Show all patches. """
        self.set_images(self.GetParent().patchpaths)

    def onButton_opts(self, evt):
        class OptsDialog(wx.Dialog):

            def __init__(self, parent, cur_thresh, *args, **kwargs):
                wx.Dialog.__init__(
                    self, parent, title="Select Attributes Options", *args, **kwargs)

                self.thresh_out = cur_thresh

                txt0 = wx.StaticText(
                    self, label="Template Matching Sensitivity: ")
                self.txtin_tmthresh = wx.TextCtrl(
                    self, value=str(self.thresh_out))
                sizer0 = wx.BoxSizer(wx.HORIZONTAL)
                sizer0.AddMany([(txt0,), (self.txtin_tmthresh,)])

                btn_ok = ffwx.Button(
                    self, label="Ok", on_click=self.onButton_ok
                )
                btn_cancel = ffwx.Button(
                    self, label="Cancel", on_click=self.onButton_cancel
                )
                btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
                btn_sizer.AddMany([(btn_ok,), (btn_cancel,)])

                sizer = wx.BoxSizer(wx.VERTICAL)
                sizer.Add(sizer0)
                sizer.Add(btn_sizer, flag=wx.ALIGN_CENTER)

                self.SetSizer(sizer)
                self.Layout()

            def onButton_ok(self, evt):
                try:
                    self.thresh_out = float(self.txtin_tmthresh.GetValue())
                except:
                    error("Invalid input: {0}", self.txtin_tmthresh.GetValue())
                self.EndModal(wx.ID_OK)

            def onButton_cancel(self, evt):
                self.EndModal(wx.ID_CANCEL)

        dlg = OptsDialog(self, self.imagemosaic.TM_THRESHOLD)
        status = dlg.ShowModal()
        if status == wx.ID_CANCEL:
            return
        self.imagemosaic.TM_THRESHOLD = dlg.thresh_out


class AttrMosaicPanel(util_widgets.ImageMosaicPanel):
    TEMPMATCH_JOB_ID = util.GaugeID("SelectAttrsTempMatchJobId")

    def __init__(self, parent, *args, **kwargs):
        util_widgets.ImageMosaicPanel.__init__(
            self, parent, cellheight=100, *args, **kwargs)

        # Threshold value to use for template matching
        self.TM_THRESHOLD = 0.85

        self._verifyframe = None

    def get_boxes(self, imgpath):
        '''
        boxes = self.GetParent().GetParent().boxes.get(imgpath, [])
        return boxes
        '''
        return []

    def start_tempmatch(self, patch, bb, patchpath):
        """ Run template matching on all unlabeled self.IMGPATHS,
        searching for PATCH.
        Input:
            IplImage PATCH:
            tuple BB: (x1,y1,x2,y2), wrt the patch.
            str PATCHPATH: Path of the patch where PATCH comes from.
        """
        # 1.) Ask user what attr value this is
        dlg = AttrValueDialog(self, patch)
        status = dlg.ShowModal()
        if status == wx.ID_CANCEL:
            return
        attrval = dlg.attrval

        # Save PATCH to temp file for VerifyPanel to use later.
        attrtype = self.GetParent().GetParent().attr
        outrootdir = pathjoin(self.GetParent().GetParent().outdir0,
                              'user_selected',
                              attrtype)
        try:
            os.makedirs(outrootdir)
        except:
            pass
        i = len(os.listdir(outrootdir))
        outfilepath = pathjoin(outrootdir, "{0}_{1}.png".format(attrval, i))
        cv.SaveImage(outfilepath, patch)
        cv.SaveImage("_selectattr_patch.png", patch)
        x = common.get_attr_prop(
            self.GetParent().GetParent().GetParent().project, attrtype, 'x1')
        y = common.get_attr_prop(
            self.GetParent().GetParent().GetParent().project, attrtype, 'y1')
        bb_off = bb[0] + x, bb[1] + y, bb[2] + x, bb[3] + y
        blankpath, _ = self.GetParent().GetParent(
        ).GetParent().inv_mapping[patchpath]
        self.GetParent().GetParent().GetParent().usersel_exs.setdefault(
            attrtype, []).append((attrval, i, outfilepath, blankpath, bb_off))

        # 2.) Only run template matching on un-labeled imgpaths
        unlabeled_imgpaths = []
        for imgpath in self.imgpaths:
            if imgpath not in self.GetParent().GetParent().boxes:
                unlabeled_imgpaths.append(imgpath)

        t = TM_Thread(patch, unlabeled_imgpaths, attrval, self.TEMPMATCH_JOB_ID,
                      callback=self.on_tempmatchdone)
        debug("starting TM Thread")
        numtasks = len(unlabeled_imgpaths)
        gauge = util.MyGauge(self, numtasks, thread=t, msg="Running template matching...",
                             job_id=self.TEMPMATCH_JOB_ID)
        t.start()
        gauge.Show()

    def on_tempmatchdone(self, results, w, h, attrval):
        """
        Input:
            dict RESULTS: {str regionpath: (x, y, float score)}
            int W, H: Width/Height of patch being searched for.
        """
        debug("TempMatching done!")
        # 1.) Extract+Save imgs, so that ViewOverlays can access them.
        outpaths = []
        patch2img = {}  # maps {str patchpath: str imgpath}
        # scores = []
        # for _, (_, _, score) in results.iteritems():
        #    scores.append(score)
        # hist, edges = np.histogram(scores)
        global CTR
        for regionpath, (x, y, score) in results.iteritems():
            if score < self.TM_THRESHOLD:
                continue
            I = cv.LoadImage(regionpath, cv.CV_LOAD_IMAGE_UNCHANGED)
            cv.SetImageROI(I, (x, y, w, h))
            imgname = os.path.split(regionpath)[1]
            imgname_noext = os.path.splitext(imgname)[0]
            outname = "{0}_{1}_{2}.png".format(
                imgname_noext, self.GetParent().GetParent().attr, CTR)
            # Use a global incrementing CTR to avoid problems if two different
            # ballots have the same imagename.
            CTR += 1
            outrootdir = os.path.join(self.GetParent().GetParent(
            ).outdir0, self.GetParent().GetParent().attr)
            try:
                os.makedirs(outrootdir)
            except:
                pass
            outpath = os.path.join(outrootdir, outname)
            cv.SaveImage(outpath, I)
            outpaths.append(outpath)
            patch2img[outpath] = regionpath
        initgroups = [outpaths]
        exemplar_imP = '_selectattr_patch.png'
        self._verifyframe = verify_overlays_new.CheckImageEqualsFrame(None, outpaths,
                                                                      exemplar_imP,
                                                                      ondone=lambda verify_results: self.on_verifydone(verify_results, attrval, results, w, h, patch2img))
        self._verifyframe.Show()
        self._verifyframe.Maximize()

    def on_verifydone(self, verify_results, attrval, results, w, h, patch2img):
        """
        Input:
            list VERIFY_RESULTS: maps {tag: [imgpath_i, ...]}
            str ATTRVAL:
            dict RESULTS: maps {str imgpath: (x, y, float score)}
            int W, H: Size of patch
            dict PATCH2IMG: maps {str patchpath: str imgpath}
        """
        self._verifyframe.Close()
        self._verifyframe = None
        # Add all TAG_YES groups
        subpatchpaths = verify_results[
            verify_overlays_new.CheckImageEquals.TAG_YES]
        for subpatchpath in subpatchpaths:
            imgpath = patch2img[subpatchpath]
            (x, y, score) = results[imgpath]
            self.GetParent().GetParent().boxes.setdefault(imgpath, []).append(
                ((x, y, x + w, y + h), attrval, subpatchpath))

        self.Refresh()


class TM_Thread(threading.Thread):

    def __init__(self, patch, regionpaths, attrval, jobid, callback=None, *args, **kwargs):
        threading.Thread.__init__(self, *args, **kwargs)
        self.patch = patch
        self.regionpaths = regionpaths
        self.callback = callback
        self.attrval = attrval
        self.jobid = jobid

    def run(self):
        with util.time_operation("calling template matching"):
            results = tempmatch.bestmatch_par(self.patch,
                                              self.regionpaths,
                                              do_smooth=tempmatch.SMOOTH_BOTH,
                                              xwinA=3,
                                              ywinA=3,
                                              xwinI=3,
                                              ywinI=3,
                                              NP=12,
                                              jobid=self.jobid)
        if self.callback:
            w, h = cv.GetSize(self.patch)
            wx.CallAfter(self.callback, results, w, h, self.attrval)


class AttrValueDialog(wx.Dialog):

    def __init__(self, parent, patch, *args, **kwargs):
        """
        Input:
            IplImage PATCH:
        """
        wx.Dialog.__init__(
            self, parent, title="What Attribute Value is this?", *args, **kwargs)
        self.patch = patch

        self.attrval = None

        patchpil = Image.fromstring("L", cv.GetSize(patch), patch.tostring())
        patchpil = patchpil.convert("RGB")
        wxbmp = util.pil2wxb(patchpil)

        self.sbitmap = wx.StaticBitmap(self, bitmap=wxbmp)

        txt0 = wx.StaticText(self, label="What is the Attribute Value? ")
        self.attrval_in = wx.TextCtrl(self, style=wx.TE_PROCESS_ENTER)
        self.attrval_in.Bind(wx.EVT_TEXT_ENTER, self.onPressEnter)
        sizer_in = wx.BoxSizer(wx.HORIZONTAL)
        sizer_in.Add(txt0)
        sizer_in.Add(self.attrval_in)

        btn_ok = ffwx.Button(self, label="Ok", on_click=self.onButton_ok)
        btn_cancel = ffwx.Button(
            self, label="Cancel", on_click=lambda e: self.EndModal(wx.ID_CANCEL)
        )
        btn_sizer = ffwx.hbox(btn_ok, btn_cancel)

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.sbitmap, flag=wx.ALIGN_CENTER)
        self.sizer.Add(sizer_in)
        self.sizer.Add(btn_sizer, flag=wx.ALIGN_CENTER)

        self.SetSizer(self.sizer)
        self.Fit()
        self.attrval_in.SetFocus()

    def onPressEnter(self, evt):
        self.onButton_ok(None)

    def onButton_ok(self, evt):
        self.attrval = self.attrval_in.GetValue()
        self.EndModal(wx.ID_OK)


class PatchPanel(util_widgets.CellPanel):

    def __init__(self, *args, **kwargs):
        util_widgets.CellPanel.__init__(self, *args, **kwargs)

    def onLeftDown(self, evt):
        pass


class PatchBitmap(util_widgets.CellBitmap):

    def __init__(self, *args, **kwargs):
        util_widgets.CellBitmap.__init__(self, *args, **kwargs)
        self.Bind(wx.EVT_LEFT_DOWN, self.onLeftDown)
        self.Bind(wx.EVT_LEFT_UP, self.onLeftUp)
        self.Bind(wx.EVT_MOTION, self.onMotion)

        self.newbox = None

    def onLeftDown(self, evt):
        if self.GetParent().is_dummy:
            return
        x, y = evt.GetPositionTuple()
        w_bmp, h_bmp = self.bitmap.GetSize()
        if x >= w_bmp or y >= h_bmp:
            return
        if not self.GetParent().GetParent().GetParent().GetParent().boxes.get(self.GetParent().imgpath, []):
            # Can only have one box at a time
            self.newbox = [None, None, None, None]
            self.newbox[:2] = self.c2img(x, y)
            self.newbox[2] = self.newbox[0] + 1
            self.newbox[3] = self.newbox[1] + 1
        self.Refresh()

    def onLeftUp(self, evt):
        if self.GetParent().is_dummy:
            return
        x, y = evt.GetPositionTuple()
        w_bmp, h_bmp = self.bitmap.GetSize()
        x = min(x, w_bmp - 1)
        y = min(y, h_bmp - 1)
        if self.newbox:
            self.newbox[2:] = self.c2img(x, y)
            box = normbox(self.newbox)
            self.newbox = None
            I = cv.LoadImage(self.GetParent().imgpath,
                             cv.CV_LOAD_IMAGE_GRAYSCALE)
            x1, y1, x2, y2 = box
            cv.SetImageROI(I, (x1, y1, x2 - x1, y2 - y1))
            self.GetParent().GetParent().start_tempmatch(
                I, (x1, y1, x2, y2), self.GetParent().imgpath)
        self.Refresh()

    def onMotion(self, evt):
        if self.GetParent().is_dummy:
            return
        x, y = evt.GetPositionTuple()
        w_bmp, h_bmp = self.bitmap.GetSize()
        x = min(x, w_bmp - 1)
        y = min(y, h_bmp - 1)
        if self.newbox:
            self.newbox[2:] = self.c2img(x, y)
            self.Refresh()

    def c2img(self, x, y):
        return map(lambda n: int(round(n)), (x * self.scale, y * self.scale))

    def img2c(self, x, y):
        return map(lambda n: int(round(n)), (x / self.scale, y / self.scale))

    def onPaint(self, evt):
        dc = util_widgets.CellBitmap.onPaint(self, evt)
        if self.GetParent().is_dummy:
            return
        boxes = self.GetParent().GetParent().GetParent(
        ).GetParent().boxes.get(self.GetParent().imgpath, [])
        if boxes:
            txtfont = wx.Font(14, wx.FONTFAMILY_DEFAULT,
                              wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
            dc.SetTextForeground("Blue")
            dc.SetFont(txtfont)
            for (box, label, subpatchP) in boxes:
                x1, y1 = self.img2c(*box[:2])
                x2, y2 = self.img2c(*box[2:])
                w, h = x2 - x1, y2 - y1
                dc.SetPen(wx.Pen("Green", 4))
                dc.DrawRectangle(x1, y1, w, h)
                w_txt, h_txt = dc.GetTextExtent(label)
                x_txt, y_txt = x1, y1 - h_txt
                if y_txt < 0:
                    y_txt = y1 + h
                dc.DrawText(label, x_txt, y_txt)
        if self.newbox:
            x1, y1, x2, y2 = normbox(self.newbox)
            x1, y1 = self.img2c(x1, y1)
            x2, y2 = self.img2c(x2, y2)
            dc.SetPen(wx.Pen("Red", 2))
            dc.DrawRectangle(x1, y1, x2 - x1, y2 - y1)


def normbox((x1, y1, x2, y2)):
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def main():
    class TestFrame(wx.Frame):

        def __init__(self, parent, patchpaths, *args, **kwargs):
            wx.Frame.__init__(self, parent, title="Select Attributes Test Frame",
                              size=(800, 800), *args, **kwargs)
            self.patchpaths = patchpaths

            self.mainpanel = SelectAttributesPanel(self)
            self.mainpanel.start(patchpaths, "AttrName")

    args = sys.argv[1:]
    patchdir = args[0]
    patchpaths = []
    for dirpath, dirnames, filenames in os.walk(patchdir):
        for imgname in [f for f in filenames if f.endswith('.png')]:
            patchpaths.append(os.path.join(dirpath, imgname))
    app = wx.App(False)
    f = TestFrame(None, patchpaths)
    f.Show()
    app.MainLoop()

if __name__ == '__main__':
    main()
