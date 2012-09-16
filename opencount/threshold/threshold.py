import pdb
import sys
import random
import math
import cStringIO
import wx
import os
from os.path import join as pathjoin
from util import get_filename, create_dirs, is_image_ext, encodepath
from PIL import Image, ImageDraw
from util import pil2wxb, wxb2pil, MyGauge
from time import time
import imageFile
from wx.lib.pubsub import Publisher
import array
import pickle

sys.path.append('..')
import ViewOverlays

class GridShow(wx.ScrolledWindow):
    """
    Class that displays voting targets
    """
    threshold = None
    wrong = {}
    jpgs = {}
    basejpgs = {}
    images = {}
    numcols = 20

    def lookupFullList(self, i):
        return self.classified_file[i]

        if self.classifiedindex:
            prefix = open(self.proj.classified+".prefix").read()
            fin = open(self.proj.classified)
            fin.seek(self.classifiedindex[i])
            sofar = ""
            while True:
                nxt = fin.read(1000)
                if '\n' in nxt:
                    sofar += nxt[:nxt.index('\n')]
                    break
                else:
                    sofar += nxt
            res = sofar.split('\0')
            res[0] = prefix+res[0]
            return tuple(res)
            
        else:
            for j,each in enumerate(self.enumerateOverFullList()):
                if i == j: 
                    return each
        raise RuntimeError("NOT FOUND INDEX " + str(i))

    def enumerateOverFullList(self):
        prefix = open(self.proj.classified+".prefix").read()
        for line in open(self.proj.classified):
            d = line[:-1].split('\0') 
            d[0] = prefix+d[0]
            yield tuple(d)
    
    def target_to_sample(self, target):
        ballot = target.split(".")[0]
        return pickle.load(open(pathjoin(self.proj.ballot_metadata, ballot)))['ballot']

    def sample_to_targets(self, ballot):
        return pickle.load(open(pathjoin(self.proj.ballot_metadata, ballot)))['targets']

    def lightBox(self, i, evt=None):
        # Which target we clicked on
        i = i+evt.GetPositionTuple()[0]/self.targetw

        pan = wx.Panel(self.parent, size=self.parent.thesize, pos=(0,0))
        targetpath = self.lookupFullList(i)[0]
        ballotpath = self.target_to_sample(os.path.split(targetpath)[-1][:-4])

        before = Image.open(ballotpath).convert("RGB")

        f = pathjoin(self.proj.ballot_metadata, encodepath(ballotpath))
        dat = pickle.load(open(f))
        doflip = dat['flipped']
        if doflip:
            before = before.rotate(180)

        if before.size[1] > self.parent.thesize[1]:
            fact = float(before.size[1])/self.parent.thesize[1]
            before = before.resize((int(before.size[0]/fact), 
                                    int(before.size[1]/fact)))
        else:
            fact = 1

        temp = before.copy()
        draw = ImageDraw.Draw(temp)

        targetname = os.path.split(targetpath)[-1]

        for each in self.sample_to_targets(encodepath(ballotpath)):
            n = os.path.split(each)[-1][:-4]
            dat = pickle.load(open(os.path.join(self.proj.extracted_metadata,n)))
            locs = dat['bbox']
            color = (200,0,0) if each == targetname else (0, 0, 200)
            draw.rectangle(((locs[2])/fact-1, (locs[0])/fact-1, 
                            (locs[3])/fact+1, (locs[1])/fact+1),
                           outline=color)
            draw.rectangle(((locs[2])/fact-2, (locs[0])/fact-2, 
                            (locs[3])/fact+2, (locs[1])/fact+2),
                           outline=color)

        img = wx.StaticBitmap(pan, -1, pil2wxb(Image.blend(before, temp, .5)))
        def remove(x):
            pan.Destroy()

        ifflipped = "\n\n\n(auto-flipped)" if doflip else ""
        def lines(x):
            if len(x) < 60: return x
            return x[:60]+"\n"+lines(x[60:])
            
        wx.StaticText(pan, label="Ballot image:\n"+lines(ballotpath)+"\n\nTarget image:\n"+lines(targetpath)+ifflipped, 
                      pos=(before.size[0],before.size[1]/3))

        b = wx.Button(pan, label="Back", pos=(before.size[0], 2*before.size[1]/3))
        b.Bind(wx.EVT_BUTTON, remove)

        q = wx.Button(pan, label="Quarantine", pos=(before.size[0], 2*before.size[1]/3+50))
        q.Bind(wx.EVT_BUTTON, lambda x: self.markQuarantine(i))
        
    def addimg(self, i):
        pilimg = self.jpgs[i]
        offset = self.CalcScrolledPosition((0,0))[1]
        pos = (0, i*self.targeth/self.numcols+offset)

        #print "DRAW IMG", pilimg, pos, i
        img = wx.StaticBitmap(self, -1, pil2wxb(pilimg), pos=pos)
        #img.SetToolTip(wx.ToolTip(str(weight)))
        def call(im, jp, ii):
            def domark(x): 
                self.markWrong(ii, evt=x)
            img.Bind(wx.EVT_LEFT_DOWN, domark)
            
            def menu(event1):
                m = wx.Menu()
                def decide(event2):
                    item = m.FindItemById(event2.GetId())
                    text = item.GetText()
                    if text == "Set Threshold":
                        self.setLine(ii, evt=event1)
                    if text == "Open Ballot":
                        self.lightBox(ii, evt=event1)
                    if text == "Mark Row Wrong":
                        for ct in range(self.numcols):
                            self.markWrong(i+ct)
                    if text == "Generate Overlays Starting Here...":
                        self.show_overlays(i)

                a = m.Append(-1, "Set Threshold")
                self.Bind(wx.EVT_MENU, decide, a)
                b = m.Append(-1, "Open Ballot")
                self.Bind(wx.EVT_MENU, decide, b)
                c = m.Append(-1, "Mark Row Wrong")
                self.Bind(wx.EVT_MENU, decide, c)
                d = m.Append(-1, "Generate Overlays Starting Here...")
                self.Bind(wx.EVT_MENU, decide, d)
                pos = event1.GetPosition()
                pos = self.ScreenToClient(pos)
                m.Bind(wx.EVT_CONTEXT_MENU, decide)
                q = self.PopupMenu(m)
            img.Bind(wx.EVT_RIGHT_DOWN, menu)
        call(img, pilimg, i)
        self.images[i] = img
        return img

    def markQuarantine(self, i):
        targetpath = self.lookupFullList(i)[0]
        ballotpath = self.target_to_sample(os.path.split(targetpath)[-1][:-4])
        if ballotpath not in self.quarantined:
            self.quarantined.append(ballotpath)
        for each in self.sample_to_targets(encodepath(ballotpath)):
            if each in self.classified_lookup:
                #print 'A'
                self.markQuarantineSingle(self.classified_lookup[each])
            #for j,line in enumerate(open(self.proj.classified)):
            #    if each == line.split('\0')[0]:
            #        self.markQuarantineSingle(j)

    def markQuarantineSingle(self, i):
        self.quarantined_targets.append(i)
        rdown = i/self.numcols*self.numcols
        if rdown not in self.jpgs: return
        offset = self.CalcScrolledPosition((0,0))[1]
        
        self.somethingHasChanged = True

        jpg = self.jpgs[rdown]
        imd = ImageDraw.Draw(jpg)
        imd.rectangle((self.targetw*(i%self.numcols), 0,
                       self.targetw*(i%self.numcols)-1+self.targetw, self.targeth-1),
                      fill=(120,0,0))
        self.jpgs[rdown] = jpg

        self.images[rdown].Destroy()
        self.addimg(rdown)

        jpg = self.basejpgs[rdown]
        imd = ImageDraw.Draw(jpg)
        imd.rectangle((self.targetw*(i%self.numcols), 0,
                       self.targetw*(i%self.numcols)-1+self.targetw, self.targeth-1),
                      fill=(120,0,0))
        self.basejpgs[rdown] = jpg

        self.Refresh()

    def drawWrongMark(self, i):
        imgIdx = i-i%self.numcols
        jpg = self.jpgs[imgIdx]
        imd = ImageDraw.Draw(jpg)
        imd.rectangle((self.targetw*(i-imgIdx), 0,
                       self.targetw*(i-imgIdx)+self.targetw-1, self.targeth-1),
                      outline=(255,0,0))
        self.images[imgIdx].Destroy()
        self.addimg(imgIdx)

    def markWrong(self, which, evt=None):
        if evt == None:
            imgIdx = which - (which%self.numcols)
        else:
            imgIdx = which
            which = which+evt.GetPositionTuple()[0]/self.targetw
            
        if imgIdx not in self.jpgs: 
            print "BAD!"
            return

        self.somethingHasChanged = True

        offset = self.CalcScrolledPosition((0,0))[1]

        if which not in self.wrong:
            self.wrong[which] = True
            self.drawWrongMark(which)
        else:
            jpg = self.basejpgs[imgIdx].copy()
            self.jpgs[imgIdx] = jpg
            self.images[imgIdx].Destroy()
            self.addimg(imgIdx)
            del self.wrong[which]
            for each in self.wrong:
                if each/self.numcols == which/self.numcols:
                    self.drawWrongMark(each)
            if self.threshold != None:
                self.drawThreshold()
        self.Refresh()

    def drawThreshold(self):
        imgIdx = self.threshold - (self.threshold%self.numcols)
        if not (imgIdx in self.jpgs and imgIdx in self.images): return

        take = self.jpgs[imgIdx]
        dr = ImageDraw.Draw(take)
        dr.rectangle((0, self.targeth-1, 
                      (self.threshold-imgIdx)*self.targetw, self.targeth-1),
                     fill=(0,255,0))

        dr.rectangle(((self.threshold-imgIdx)*self.targetw, 0, 
                      (self.threshold-imgIdx)*self.targetw, self.targeth-1),
                     fill=(0,255,0))

        dr.rectangle(((self.threshold-imgIdx)*self.targetw, 0, 
                      self.targetw*self.numcols, 0),
                     fill=(0,255,0))

        self.images[imgIdx].Destroy()
        self.addimg(imgIdx)
        
        self.Refresh()

    def setLine(self, which, evt=None):
        if evt == None:
            imgIdx = which - (which%self.numcols)
        else:
            imgIdx = which
            which = which+int(round(float(evt.GetPositionTuple()[0])/self.targetw))
            print 'click line ', evt.GetPositionTuple()[0]/self.targetw
            
        if imgIdx not in self.jpgs: 
            print "BAD LINE!"
            return

        self.somethingHasChanged = True

        if self.threshold != None:
            lastIdx = self.threshold - self.threshold%self.numcols
            if lastIdx in self.images:
                self.jpgs[lastIdx] = self.basejpgs[lastIdx].copy()
                self.images[lastIdx].Destroy()
                self.addimg(lastIdx)
            for each in self.wrong:
                if each/self.numcols == self.threshold/self.numcols:
                    self.drawWrongMark(each)
            
        self.threshold = which

        self.drawThreshold()

    def show_overlays(self, start_idx):
        """ Starting at START_IDX, generate min/max overlays from all
        voting targets, up to the last image.
        """
        imgpaths = []

        # ...magic...

        frame = ViewOverlays.SimpleOverlayFrame(self, imgpaths)
        frame.Show()

    def __init__(self, parent, proj):
        """
        Set things up! Yay!
        """
        wx.ScrolledWindow.__init__(self, parent, -1)

        self.proj = proj
        self.proj.addCloseEvent(self.dosave)
        self.parent = parent

        self.quarantined = []
        self.quarantined_targets = []

    def reset_panel(self):
        self.proj.removeCloseEvent(self.dosave)

    def getImageList(self):
        """
        Get a list of all the images.
        """

        path = self.proj.extracted_dir
        for each in os.listdir(path):
            if is_image_ext(each):
                self.basetargetw, self.basetargeth = Image.open(pathjoin(path,each)).size
                break

        self.targetResize = 1

        self.changeSize(1, False)


    def findBoundry(self):
        hist = [0]*256
        gaguge = MyGauge(self, 1)
        #wx.CallAfter(Publisher().sendMessage, 
        #             "signals.MyGauge.nextjob", 
        #             len(self.classifiedindex)/1000)
        #gauge.Show()
        for i,(_,v) in enumerate(self.enumerateOverFullList()):
            #if i%1000 == 0:
                # Don't want to slow it down too much
                #wx.CallAfter(Publisher().sendMessage, "signals.MyGauge.tick")
            hist[int(v)] += 1
        #wx.CallAfter(Publisher().sendMessage, "signals.MyGauge.done")
        #print list(enumerate(hist))

        # I'm going to assume there are two normal dist. variables 
        #  which combine to make the histogram.

        # I'll go through all possible thresholds, and find the one which
        #  allows two normal dist. to match the best.

        def matchnormal(data):

            if sum(data) == 0: return 0
            if sum([1 if x != 0 else 0 for x in data]) == 1: return 0

            mean = float(sum([i*x for i,x in enumerate(data)]))/sum(data)
            total = sum(data)
            
            stdev = 0
            for i in range(len(data)):
                dist = abs(mean-i)
                sq = dist*dist
                stdev += sq*(float(data[i])/total)
            stdev = stdev**.5

            def pdf(m, s, x):
                return ((2*3.14*s*s)**-.5)*2.71**(-(x-m)**2/(2*s*s));

            err = 0
            for i in range(len(data)):
                err += abs((float(data[i])/total-pdf(mean,stdev,i)))
            return err

        best = None
        mval = 1<<30
        for line in range(1,255):
            lower = hist[:line]
            m1 = matchnormal(lower)
            upper = hist[line:]
            m2 = matchnormal(upper)
            if m1+m2 < mval:
                best = line
                mval = m1+m2
        print "BEST LINE", best

        count = sum(hist[:best])
        return count, (count-(self.numcols*self.numrows)/2)

    def changeSize(self, num, redraw=True):
        self.targetResize *= num

        self.targeth = int(self.basetargeth*self.targetResize)
        self.targetw = int(self.basetargetw*self.targetResize)

        self.numcols = int((self.parent.thesize[0]-30)/self.targetw)
        self.numrows = int(self.parent.thesize[1]/self.targeth)
        if redraw:
            self.setupScrollBars()

            startat = self.lastpos
            startat = startat - startat%self.numcols
            print "START AT ", startat
            print 'other', self.numcols, self.targeth

            self.onScroll(startat)

            if self.threshold != None:
                self.drawThreshold()
    
    def setupScrollBars(self):
        height = (self.numberOfTargets/self.numcols)*self.targeth

        print "Height", height
        self.SetScrollbars(1, 1, 1, int(height*1.05), 0, 0, True)
        self.Scroll(0, 0)
        self.Show(True)
        self.Centre()

        self.Layout()
        self.parent.Fit()
        self.Fit()

        for each in self.images:
            self.images[each].Destroy()

        self.jpgs = {}
        self.basejpgs = {}
        self.images = {}

    def setup(self):
        self.somethingHasChanged = False
        i = 0

        self.Bind(wx.EVT_SCROLLWIN_THUMBTRACK, 
                  lambda x: self.onScroll(evtpos=x.GetPosition()))
        self.Bind(wx.EVT_SCROLLWIN_THUMBRELEASE, 
                  lambda x: self.onScroll(evtpos=x.GetPosition()))
        self.Bind(wx.EVT_SCROLLWIN_LINEUP,
                  lambda x: self.onScroll(self.lastpos-self.numcols))
        self.Bind(wx.EVT_SCROLLWIN_LINEDOWN, 
                  lambda x: self.onScroll(self.lastpos+self.numcols))
        self.Bind(wx.EVT_SCROLLWIN_PAGEUP,
                  lambda x: self.onScroll(self.lastpos-self.numcols*self.numrows))
        self.Bind(wx.EVT_SCROLLWIN_PAGEDOWN, 
                  lambda x: self.onScroll(self.lastpos+self.numcols*self.numrows))


        self.numberOfTargets = 0
        for _ in self.enumerateOverFullList():
            self.numberOfTargets += 1

        self.classified_file = [l.split("\0") for l in open(self.proj.classified)]
        self.classified_lookup = dict([(l.split("\0")[0], i) for i,l in enumerate(open(self.proj.classified))])
        if os.path.exists(self.proj.classified+".index"):
            try:
                is64bit = (sys.maxsize > (2**32))
                size = 8 if is64bit else 4
                arr = array.array("L")
                arr.fromfile(open(self.proj.classified+".index"), 
                             os.path.getsize(self.proj.classified+".index")/size)
                self.classifiedindex = arr
            except Exception as e:
                print e
                print "Could not load index file. Doing it the slow way. err1"
                self.classifiedindex = None
        else:
            print "Could not load index file. Doing it the slow way. err2"
            self.classifiedindex = None


        self.getImageList()

        self.imagefile = imageFile.ImageFile(self.proj.extractedfile)

        self.setupScrollBars()
        
        self.quarantined_targets = []
        if os.path.exists(self.proj.threshold_internal):
            dat = open(self.proj.threshold_internal).read()
            if dat:
                data = pickle.load(open(self.proj.threshold_internal))
                if len(data) == 4:
                    self.threshold, self.wrong, self.quarantined, self.quarantined_targets = data
                    self.onScroll(self.threshold-(self.numcols*self.numrows)/2)
                else:
                    self.threshold, self.wrong, self.quarantined, self.quarantined_targets, pos = data
                    self.onScroll(pos)
        else:
            newthresh, bound = self.findBoundry()
            self.onScroll(bound)
            self.setLine(newthresh)
                    

    def onScroll(self, pos=None, evtpos=None):
        if evtpos != None:
            print "SET FROM", evtpos
            pos = int(evtpos/self.targeth*self.numcols)
        else:
            pos = pos - pos%self.numcols
        if pos < 0: pos = 0
        GAP = self.numcols*4

        self.lastpos = pos
        low = max(0,pos-GAP)
        high = min(pos+self.numcols*self.numrows+GAP,self.numberOfTargets)

        # Draw the images from low to high.
        print "Drawing from", low, "to", high
        for i in range(low,high,self.numcols):
            #print i
            if i in self.jpgs:
                # If we've drawn it before, then it's still there, skip over it
                continue
            # Open and draw it.

            jpg = self.imagefile.readManyImages(i, self.numcols, 
                                                self.basetargetw, self.basetargeth,
                                                self.targetw, self.targeth)

            # TODO this could use a lot of memory 
            #    if the person scrolls through all the images.
            # Keep around only the "important" one 
            #    which have been marked by the user as wrong.

             # Want to be able to modify this but not the base
            self.jpgs[i] = jpg.copy()
            self.basejpgs[i] = jpg

            self.addimg(i)

            for j in range(self.numcols):
                if i+j in self.quarantined_targets:
                    self.markQuarantine(i+j)

        # This could get very slow on big elections with lots of wrong marks
        for each in self.wrong:
            if (each/self.numcols)*self.numcols in self.jpgs:
                self.drawWrongMark(each)

        if self.threshold != None:
            if self.threshold/self.numcols*self.numcols in self.jpgs:
                self.drawThreshold()

        # Scroll us to the right place.
        if evtpos != None:
            print 'scroll to right pos'
            self.Scroll(0, evtpos)
        else:
            print "SCROLL TO", pos*self.targeth/self.numcols
            self.Scroll(0, pos*self.targeth/self.numcols)
        # Record where we were last time.
        wx.CallAfter(self.Refresh)

    def dosave(self):
        """
        Save all the data.
        """

        if not self.somethingHasChanged: return
        self.somethingHasChanged = False

        print "SAVING!!!!"
        filled = {}
        unfilled = {}
        for i in range(self.numberOfTargets):
            if i not in self.wrong:
                if i < self.threshold:
                    filled[i] = True
                else:
                    unfilled[i] = True
            else:
                if i < self.threshold:
                    unfilled[i] = True
                else:
                    filled[i] = True
        f = open(self.proj.targets_result, "w")

        for i,(t,_) in enumerate(self.enumerateOverFullList()):
            if i in filled:
                f.write(os.path.split(t)[1]+", 1\n")
        for i,(t,_) in enumerate(self.enumerateOverFullList()):
            if i in unfilled:
                f.write(os.path.split(t)[1]+", 0\n")
        f.close()

        pickle.dump((self.threshold, self.wrong, self.quarantined, self.quarantined_targets, self.lastpos), open(self.proj.threshold_internal, "w"))
            
        out = open(self.proj.quarantined_manual, "w")
        for each in self.quarantined:
            if type(each) == type(0):
                targetpath = self.lookupFullList(each)[0]
                ballotpath = self.target_to_sample(os.path.split(targetpath)[-1][:-4])
                out.write(ballotpath+"\n")
            else:
                out.write(each+"\n")
        out.close()


class ThresholdPanel(wx.Panel):
    def __init__(self, parent, size):
        wx.Panel.__init__(self, parent, id=-1, size=size) 
        #print "AND SIZE", parent.GetSize()
        self.parent = parent
        self.parent.Fit()

        Publisher().subscribe(self.getproj, "broadcast.project")
    
    def getproj(self, msg):
        self.proj = msg.data

    def reset_panel(self):
        self.tabOne.reset_panel()

    first = True
    def start(self, size=None):

        if not self.first: return
        self.first = False

        self.thesize = size

        sizer = wx.BoxSizer(wx.VERTICAL)

        tabOne = GridShow(self, self.proj)

        top = wx.BoxSizer(wx.HORIZONTAL)
        button1 = wx.Button(self, label="Increase Size")
        button1.Bind(wx.EVT_BUTTON, lambda x: tabOne.changeSize(2))
        button2 = wx.Button(self, label="Decrease Size")
        button2.Bind(wx.EVT_BUTTON, lambda x: tabOne.changeSize(0.5))
        button3 = wx.Button(self, label="Scroll Up")
        button3.Bind(wx.EVT_BUTTON, lambda x: tabOne.onScroll(tabOne.lastpos-tabOne.numcols*(tabOne.numrows-5)))
        button4 = wx.Button(self, label="Scroll Down")
        button4.Bind(wx.EVT_BUTTON, lambda x: tabOne.onScroll(tabOne.lastpos+tabOne.numcols*(tabOne.numrows-5)))
        top.Add(button1)
        top.Add(button2)
        top.Add(button3)
        top.Add(button4)

        sizer.Add(top)
        tabOne.setup()
        self.Refresh()

        sizer.Add(tabOne, proportion=10, flag=wx.ALL|wx.EXPAND, border=5)

        self.SetSizer(sizer)
        self.Fit()
        self.Refresh()
