import pdb, traceback
import sys
import wx
import os
from os.path import join as pathjoin
from util import is_image_ext, pil2wxb, MyGauge
from PIL import Image, ImageDraw
import util
from time import time
import imageFile
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
    preloaded_fulllist = None
    inverse_fulllist = None

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

    def enumerateOverFullList(self, force=False):
        if self.preloaded_fulllist:
            print 'case a'
            for each in self.preloaded_fulllist:
                yield each
        elif not force:
            print 'case b'
            it = self.enumerateOverFullList(True)
            first = next(it)
            if len(str(first))*self.numberOfTargets < (1<<30):
                self.preloaded_fulllist = [first]
                self.preloaded_fulllist.extend(it)
                for each in self.preloaded_fulllist:
                    yield each
        else:
            print 'case c'
            prefix = open(self.proj.classified+".prefix").read()
            for line in open(self.proj.classified):
                d = line[:-1].split('\0') 
                d[0] = prefix+d[0]
                yield tuple(d)

    def target_to_sample(self, targetpath):
        # Has to be a bit hackier, since I don't want to construct in-memory
        # data-structs linear in the # of voting targets...
        # Recall: targetname is {imgname}.{uid}.png
        # Note: To save space, TARGETPATH must be joined with self.PREFIX
        targetpath_full = self.prefix + targetpath
        targetname = os.path.splitext(os.path.split(targetpath_full)[1])[0]
        imgname = targetname.split('.')[0]
        # (also removes the 'pageN/' off of the targetpath_full)
        rootdir = os.path.split(os.path.split(os.path.split(targetpath_full)[0])[0])[0]
        relpath = os.path.normpath(os.path.relpath(os.path.abspath(rootdir),
                                                   os.path.abspath(self.proj.extracted_dir)))
        ballotpath = os.path.normpath(pathjoin(self.proj.voteddir, relpath, imgname+".png"))
        page = self.img2page[ballotpath]
        ballotid = self.img2bal[ballotpath]

        z=self.get_meta(ballotid, page)['ballot']
        return z

    def get_meta(self, ballotid, page):
        print ballotid #REMOVE
        if page not in self.bal2targets[ballotid]:
            _,_,_, imgmeta_path = self.bal2targets[ballotid].values()[0]
        else:
            _,_,_, imgmeta_path = self.bal2targets[ballotid][page]
        print imgmeta_path #REMOVE
        return pickle.load(open(imgmeta_path, 'rb'))

    def sample_to_targets(self, ballotpath):
        page = self.img2page[ballotpath]
        ballotid = self.img2bal[ballotpath]
        return self.get_meta(ballotid, page)['targets']

    @util.pdb_on_crash
    def lightBox(self, i, evt=None):
        # Which target we clicked on
        _t = time()
        print "...Starting LightBox..."

        i = i+evt.GetPositionTuple()[0]/self.targetw

        targetpath = self.lookupFullList(i)[0]
        try:
            ballotpath = self.target_to_sample(targetpath)
        except:
            dlg = wx.MessageDialog(self, message="Oh no. We couldn't open this ballot for some reason ...", style=wx.ID_OK)
            dlg.ShowModal()

            return


        pan = wx.Panel(self.parent, size=self.parent.thesize, pos=(0,0))

        before = Image.open(ballotpath).convert("RGB")

        dur = time() - _t
        print "    Phase 1: {0} s".format(dur)
        _t = time()

        doflip = self.img2flip[ballotpath]
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

        targetname = os.path.split(self.prefix+targetpath)[-1]
        ballotid = self.img2bal[ballotpath]

        dur = time() - _t
        print "    Phase 2: {0} s".format(dur)
        _t = time()

        indexs = []
        other_stuff = [] 
        
        targetpaths = self.sample_to_targets(ballotpath)
        page = self.img2page[ballotpath]
        if page in self.bal2targets[ballotid]:
            _,targetmeta_dir,_,_ = self.bal2targets[ballotid][page]
        else:
            _,targetmeta_dir,_,_ = self.bal2targets[ballotid].values()[0]

        if self.inverse_fulllist == None:
            self.inverse_fulllist = {}
            for ind, (p, _) in enumerate(self.enumerateOverFullList()):
                self.inverse_fulllist[p] = ind
        #for ind, (p, _) in enumerate(self.enumerateOverFullList()):
        for p in targetpaths:
            # P is path to a target image
            # Note to self:
            # when adding target-adjustment from here, you need to some how map
            # targetID name -> index in the list to find if it is 'wrong' or not.
            ind = self.inverse_fulllist[p]
            pname = os.path.split(p)[-1]
            #if pname in targetpaths:
            if p in targetpaths:
                # Recall: targetname is {imgname}.{uid}.png
                #         metaname is {imgname}.{uid}
                imgname, uid, ext = pname.split(".")
                metaname = "{0}.{1}".format(imgname, uid)
                targetmeta_path = pathjoin(targetmeta_dir, metaname)
                dat = pickle.load(open(targetmeta_path, 'rb'))
                locs = dat['bbox']
                indexs.append(([a / fact for a in locs], ind))
                other_stuff.append((ind, locs, pname))

        print "    Phase 3: {0} s".format(time() - _t)
        _t = time()

        #for each in self.sample_to_targets(encodepath(ballotpath)):
        for (ind, locs, pname) in other_stuff:
            # Note to self:
            # when adding target-adjustment from here, you need to some how map
            # targetID name -> index in the list to find if it is 'wrong' or not.
            #ind = next(i for i,(p,_) in enumerate(self.enumerateOverFullList()) if each in p)
            #n = os.path.split(each)[-1][:-4]
            #dat = pickle.load(open(os.path.join(self.proj.extracted_metadata,n)))
            #locs = dat['bbox']
            #indexs.append(([a/fact for a in locs], ind))
            color = (0,255,0) if pname == targetname else (0, 0, 200)
            draw.rectangle(((locs[2])/fact-1, (locs[0])/fact-1, 
                            (locs[3])/fact+1, (locs[1])/fact+1),
                           outline=color)
            draw.rectangle(((locs[2])/fact-2, (locs[0])/fact-2, 
                            (locs[3])/fact+2, (locs[1])/fact+2),
                           outline=color)
            isfilled = (ind < self.threshold)^(ind in self.wrong)
            if isfilled:
                draw.rectangle(((locs[2])/fact, (locs[0])/fact, 
                                (locs[3])/fact, (locs[1])/fact),
                               fill=(255, 0, 0))
                
        print "    Phase 4: {0} s".format(time() - _t)
        _t = time()

        img = wx.StaticBitmap(pan, -1, pil2wxb(Image.blend(before, temp, .5)))

        def markwrong(evt):
            x,y = evt.GetPositionTuple()
            for (u,d,l,r),index in indexs:
                if l <= x <= r and u <= y <= d:
                    print "IS", index
                    self.markWrong(index)

        img.Bind(wx.EVT_LEFT_DOWN, markwrong)
        def remove(x):
            pan.Destroy()

        ifflipped = "\n\n\n(auto-flipped)" if doflip else ""
        def lines(x):
            if len(x) < 60: return x
            return x[:60]+"\n"+lines(x[60:])

        
        templatepath = self.get_meta(self.img2bal[ballotpath], page)['template']
        wx.StaticText(pan, label="Ballot image:\n"+lines(ballotpath)+"\n\nTemplate image:\n"+lines(templatepath)+"\n\nTarget image:\n"+lines(targetpath)+ifflipped, 
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
        try:
            img = wx.StaticBitmap(self, -1, pil2wxb(pilimg), pos=pos)
        except:
            traceback.print_exc()
            pdb.set_trace()
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
                        self.show_overlays(ii, event1)

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

    def markQuarantine(self, i, ballotpath=None):
        if ballotpath == None:
            targetpath = self.lookupFullList(i)[0]
            ballotpath = self.target_to_sample(targetpath)
        if ballotpath not in self.quarantined:
            self.quarantined.append(ballotpath)
        #for each in self.sample_to_targets(encodepath(ballotpath)):
        for each in self.sample_to_targets(ballotpath):
            each_minusprefix = each[len(self.prefix):]
            if each_minusprefix in self.classified_lookup:
                #print 'A'
                self.markQuarantineSingle(self.classified_lookup[each_minusprefix])
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
        try:
            jpg = self.jpgs[imgIdx]
        except:
            traceback.print_exc()
            pdb.set_trace()
        copy = jpg.copy()
        imd = ImageDraw.Draw(copy)
        imd.rectangle((self.targetw*(i-imgIdx), 0,
                       self.targetw*(i-imgIdx)+self.targetw-1, self.targeth-1),
                      fill=(255,0,0))
        self.jpgs[imgIdx] = Image.blend(jpg, copy, .3)
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

    def show_overlays(self, ii, evt):
        """ Starting at the place where the user right-clicked, generate 
        min/max overlays from all voting targets, up to the last target.
        """
        start_idx = ii+int(round(float(evt.GetPositionTuple()[0])/self.targetw))

        print 'start_idx:', start_idx
        imgpaths = []
        for idx, (target_imgpath, id) in enumerate(self.enumerateOverFullList()):
            if idx >= start_idx:
                imgpaths.append(target_imgpath)

        frame = ViewOverlays.SimpleOverlayFrame(self, imgpaths)
        frame.Show()

    def __init__(self, parent, proj):
        """
        Set things up! Yay!
        """
        wx.ScrolledWindow.__init__(self, parent, -1)

        self.proj = proj
        self.proj.addCloseEvent(self.dosave)
        self.img2bal = pickle.load(open(proj.image_to_ballot, 'rb'))
        self.img2page = pickle.load(open(pathjoin(proj.projdir_path, proj.image_to_page), 'rb'))
        self.img2flip = pickle.load(open(pathjoin(proj.projdir_path, proj.image_to_flip), 'rb'))
        self.bal2targets = pickle.load(open(pathjoin(proj.projdir_path, proj.ballot_to_targets), 'rb'))

        self.somethingHasChanged = False

        self.prefix = open(self.proj.classified+".prefix").read()
        
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
        for dirpath, dirnames, filenames in os.walk(path):
            done = False
            for imgname in (f for f in filenames if is_image_ext(f)):
                imgpath = pathjoin(dirpath, imgname)
                self.basetargetw, self.basetargeth = Image.open(imgpath).size
                done = True
                break
            if done: break

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


    def clear_unused(self, low, high):
        print "Del",
        keys = self.jpgs.keys()
        for each in keys:
            if low <= each <= high: continue
            print each,
            del self.jpgs[each]
            del self.basejpgs[each]
            self.images[each].Destroy()
            del self.images[each]
        print

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

        self.clear_unused(low, high)

        qt = set(self.quarantined_targets)

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

            # Want to be able to modify this but not the base
            self.jpgs[i] = jpg.copy()
            self.basejpgs[i] = jpg
            self.addimg(i)
            for j in range(self.numcols):
                if i+j in qt:
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
                f.write(t+", 1\n")
        for i,(t,_) in enumerate(self.enumerateOverFullList()):
            if i in unfilled:
                f.write(t+", 0\n")
        f.close()

        pickle.dump((self.threshold, self.wrong, self.quarantined, self.quarantined_targets, self.lastpos), open(self.proj.threshold_internal, "w"))
        img2bal = pickle.load(open(self.proj.image_to_ballot, 'rb'))
            
        out = open(self.proj.quarantined_manual, "w")
        for each in self.quarantined:
            if type(each) == type(0):
                targetpath = self.lookupFullList(each)[0]
                ballotpath = self.target_to_sample(targetpath)
                ballotid = img2bal[ballotpath]
                out.write(str(ballotid)+"\n")
            else:
                # EACH is ballotpath
                ballotid = img2bal[each]
                out.write(str(ballotid)+"\n")
        out.close()

class ThresholdPanel(wx.Panel):
    def __init__(self, parent, size):
        wx.Panel.__init__(self, parent, id=-1, size=size) 
        #print "AND SIZE", parent.GetSize()
        self.parent = parent

        self.tabOne = None

        self.parent.Fit()
    
    def reset_panel(self):
        self.tabOne.reset_panel()

    first = True
    def start(self, proj, size=None):
        self.proj = proj

        if not self.first: return
        self.first = False

        self.thesize = size

        sizer = wx.BoxSizer(wx.VERTICAL)

        tabOne = GridShow(self, self.proj)

        self.tabOne = tabOne

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
        
    def stop(self):
        self.tabOne.dosave()
