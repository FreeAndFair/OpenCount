import random
import math
import cStringIO
import wx, wx.lib.scrolledpanel, wx.lib.intctrl
import os, sys
from sets import Set
from PIL import Image, ImageDraw
import csv
import pickle

sys.path.append('..')
from util import ImageManipulate, pil2wxb

from wx.lib.pubsub import Publisher

class LabelContest(wx.Panel):
    def __init__(self, parent, size):
        wx.Panel.__init__(self, parent, id=-1, size=size)

        self.parent = parent
        self.canMoveOn = False
        
        Publisher().subscribe(self.getproj, "broadcast.project")
    
    def getproj(self, msg):
        self.proj = msg.data

    def gatherData(self):
        """
        Get the data from the files.
        This includes the location of the template files, as well as
        how the targets are grouped together.
        """
        self.dirList = []

        thewidth = None

        # groupedtargets is [[[(targetid,contestid,left,up,right,down)]]]
        # where:
        #   groupedtargets[a][b][c] is the ID and location of target C in contest B of template A
        #   len(groupedtargets) is the number of templates
        #   len(groupedtargets[a]) is the number of contests in template A
        #   len(groupedtargets[a][b]) is the number of targets in contest B of template A
        self.groupedtargets = []
        # os.listdir is okay here -- it's flat.
        for each in os.listdir(self.proj.target_locs_dir):
            if each[-4:] != '.csv': continue
            gr = {}
            name = os.path.join(self.proj.target_locs_dir, each)
            for i, row in enumerate(csv.reader(open(name))):
                if i == 0:
                    # skip the header row, to avoid adding header
                    # information to our data structures
                    continue
                # If this one is a target, not a contest
                if row[7] == '0':
                    if row[8] not in gr:
                        gr[row[8]] = []
                    # 2,3,4,5 are left,up,width,height but need left,up,right,down
                    gr[row[8]].append((int(row[1]), int(row[8]),
                                       int(row[2]), int(row[3]), 
                                       int(row[2])+int(row[4]), 
                                       int(row[3])+int(row[5])))
                # Only add the row's imgpath once
                if row[0] not in self.dirList:
                    self.dirList.append(row[0])
                    if thewidth == None:
                        thewidth = Image.open(row[0]).size[0]
            lst = gr.values()

            # Figure out where the columns are.
            # We want to sort each group going left->right top->down
            #   but only go left->right if we're on a new column,
            #   not if we're only off by a few pixels to the left.
            errorby = thewidth/100

            cols = {}
            for _,_,x,_,_,_ in sum(lst, []):
                found = False
                for c in cols:
                    if abs(c-x) < errorby:
                        found = True
                        cols[x] = cols[c]
                        break
                if not found:
                    cols[x] = x

            # And sort by columns within each contest
            lst = [sorted(x, key=lambda x: (cols[x[2]], x[3])) for x in lst]
            # And then sort each contest in the same way, globally
            slist = sorted(lst, key=lambda x: (cols[x[0][2]], x[0][3]))
            
            self.groupedtargets.append(slist)

    def reset_panel(self):
        self.proj.removeCloseEvent(self.save)
    
    def reset_data(self):
        print "DO IT"
        for f in [self.proj.contest_id, 
                  self.proj.contest_internal, 
                  self.proj.contest_text]:
            print "ON", f
            if os.path.exists(f):
                print "UNLINK", f
                os.unlink(f)

    firstTime = True

    def start(self, sz=None):
        """
        Set everything up to display.
        """

        if not self.firstTime: return

        self.firstTime = False

        print "SET UP", sz
        self.thesize = sz

        sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.sizer = sizer

        sz2 = wx.BoxSizer(wx.VERTICAL)
        self.imagebox = ImageManipulate(self, size=(500,600))

        sz3 = wx.BoxSizer(wx.HORIZONTAL)
        zoomin = wx.Button(self, label="Zoom In")
        zoomout = wx.Button(self, label="Zoom Out")
        zoomin.Bind(wx.EVT_BUTTON, lambda x: self.imagebox.zoomin())
        zoomout.Bind(wx.EVT_BUTTON, lambda x: self.imagebox.zoomout())

        sz3.Add(zoomin)
        sz3.Add(zoomout)
        sz2.Add(self.imagebox)
        sz2.Add(sz3)
        sizer.Add(sz2)

        rightside = wx.BoxSizer(wx.HORIZONTAL)

        self.textarea = wx.Panel(self)

        self.proj.addCloseEvent(self.save)

        self.count = 0
        self.templatenum = 0
        
        self.gatherData()

        self.getValues()

        self.text_targets = []
        self.addText()

        self.canMoveOn = False

        textbox = wx.BoxSizer(wx.VERTICAL)

        button1 = wx.Button(self, label='Previous Contest')
        button2 = wx.Button(self, label='Next Contest')
        button1.Bind(wx.EVT_BUTTON, lambda x: self.doadd(-1))
        button2.Bind(wx.EVT_BUTTON, lambda x: self.doadd(1))

        textbox.Add(self.textarea)
        textbox.Add(button1)
        textbox.Add(button2)

        self.remainingText = wx.StaticText(self, style=wx.TE_READONLY, size=(150,30))
        textbox.Add(self.remainingText)

        template = wx.BoxSizer(wx.VERTICAL)
        button3 = wx.Button(self, label='Previous Ballot')
        button4 = wx.Button(self, label='Next Ballot')
        button3.Bind(wx.EVT_BUTTON, lambda x: self.nexttemplate(-1))
        button4.Bind(wx.EVT_BUTTON, lambda x: self.nexttemplate(1))
        # How much to scale the template by.
        self.templatebox = wx.Panel(self, size=(303,500))
        self.templatebox.img = wx.StaticBitmap(self.templatebox)
        
        template.Add(self.templatebox)
        template.Add(button3)
        template.Add(button4)

        if self.proj.options.devmode:
            button5 = wx.Button(self, label="Magic \"I'm Done\" Button")
            def declareReady(x):
                self.save()
                print "GOT", self.contest_order
                for ct,cid_lst in enumerate(self.contest_order):
                    for cid in cid_lst:
                        print "WORKING ON", ct, cid
                        if (ct,cid) not in self.text or self.text[ct,cid] == []:
                            numt = len(self.groupedtargets[self.templatenum][self.count])
                            title = ":".join(["title", str(ct), str(cid)])
                            contests = [":".join(["contest", str(ct), str(cid), str(targ)]
    ) for targ in range(numt)]
                            self.text[ct,cid] = [title]+contests
                        if (ct,cid) not in self.voteupto: 
                            self.voteupto[ct, cid] = 1
                        
                self.canMoveOn = True
                Publisher().sendMessage("broadcast.can_proceed")
            button5.Bind(wx.EVT_BUTTON, declareReady)
            template.Add(button5)

        rightside.Add(textbox)
        rightside.Add((20,-1))
        rightside.Add(template)

        # If set to true, don't restore text.
        self.doNotClear = False

        self.nexttemplate(0)

        sizer.Add(rightside, wx.ALL|wx.EXPAND, 5)
        self.SetSizer(sizer)

        sizer.Fit(self)
        self.Layout()
        self.Fit()
 
        self.Show()

    def save(self):
        self.saveText(removeit=False)

        # We want to figure out which contests are "equal"
        #  so that when we tally votes we report them together.
        # Equality is defined as having all the same text.

        # (bid,cid) tuples
        equal = []
        used = {}

        for k1,v1 in self.text.items():
            if k1 in used: continue
            eq = []
            for k2,v2 in self.text.items():
                if k2 in used: continue
                if v1 == v2:
                    it = self.contestID[k2]
                    eq.append((it[0], it[1]))
                    used[k2] = True
            equal.append(eq)
            used[k1] = True

        c_id = csv.writer(open(self.proj.contest_id, "w"))

        mapping = {}
        for num,group in enumerate(equal):
            print "-----ON", num
            for item in group:
                mapping[item] = num
                # We need to get the contest ID in the new list
                print "Know that targets are at", self.groupedtargets[item[0]]
                targets = [x for x in self.groupedtargets[item[0]] if x[0][1] == item[1]][0]
                #print "The id is", item[1]
                #targets = self.groupedtargets[item[0]][cont]
                print "so targets is", targets
                ids = [str(x[0]) for x in targets]
                print "ids", ids
                c_id.writerow([self.dirList[item[0]],item[1],num]+ids)

        # We write out the result as a mapping from Contest ID to text
        id_to_text = {}
        for k,v in self.text.items():
            bid, cid = self.contestID[k]
            id_to_text[(bid, cid)] = [str(self.voteupto[k])]+v

        fout = csv.writer(open(self.proj.contest_text, "w"))

        did = {}
        for k,v in id_to_text.items():
            if mapping[k] not in did:
                # ContestID, upto, title, (label)*
                fout.writerow([mapping[k]]+v)
                did[mapping[k]] = True

        pickle.dump((self.text, self.voteupto), open(self.proj.contest_internal, "w"))
                    

    def setupBoxes(self):
        self.boxes = []
        def bound(targetlst, goleft, width):
            l = u = 2**32
            r = d = 0
            for _,ID,ll,uu,rr,dd in targetlst:
                l = min(ll,l)
                u = min(uu,u)
                d = max(dd,d)
                r = max(rr,r)
            return ID, l-goleft,u-int(.06*self.templateimg[i].size[1]), l+width,d+int(.03*self.templateimg[i].size[1])
        
        # The (id,l,u,r,d) tuple for each contest of each template

        for i,ballot in enumerate(self.groupedtargets):
            ballotw = self.templateimg[i].size[0]
            columns = {}
            for group in ballot:
                for target in group:
                    if not any([target[2] + x in columns for x in range(-5,5)]):
                        columns[target[2]] = True
            columns = sorted(columns.keys())
            if len(columns) == 0:
                continue
            leftmargin = min(columns)
            # Interior space available
            remaining = max(ballotw-(2*leftmargin), 0.10*self.templateimg[i].size[1])
            # The size of one box
            boxwidth = remaining/len(columns)
            boxwidth = min(boxwidth, self.templateimg[i].size[0]/2)
            goleft = 0
            if len(columns) >= 2:
                goleft = max(0,columns[1]-(columns[0]+boxwidth))

            bboxes = [bound(x,goleft,boxwidth) for x in ballot]
            print "I THINK BBOX", bboxes
            # TODO shouldn't extend in to next column

            self.boxes.append(bboxes)


    def getValues(self):
        """
        Set up all the variables
        """

        # The text that the user enters.
        # Keys are of the form templateid:(l,u,r,d)
        self.text = {}
        self.voteupto = {}

        restored = False
        if os.path.exists(self.proj.contest_internal):
            d = open(self.proj.contest_internal).read()
            if d:
                restored = True
                self.text, self.voteupto = pickle.load(open(self.proj.contest_internal))

        # The PIL image for the contest.
        # Keys are of the form templateid:(l,u,r,d)
        self.crop = {}
        self.resize = {}

        self.templateimg = []
        for each in self.dirList:
            self.templateimg.append(Image.open(each).convert("RGB"))

        self.groups = []

        self.setupBoxes()

        # Convert the ballot:boundingbox -> (ballotid, contestid)
        self.contestID = {}

        maxw,maxh = self.thesize

        for i,each in enumerate(self.boxes):
            for x in each:
                if not restored:
                    self.text[i,x[0]] = []
                    self.voteupto[i,x[0]] = 1
                factor = 1
                self.crop[i,x[0]] = (self.templateimg[i],
                                     (x[1], x[2], 
                                      int((x[3]-x[1])*factor+x[1]),
                                      int((x[4]-x[2])*factor+x[2])))
                self.contestID[i,x[0]] = (i, x[0])

        self.contest_order = [[y[0] for y in x] for x in self.boxes]
        self.boxes = [[y[1:] for y in x] for x in self.boxes]

        self.currentcontests = []

    
    def nexttemplate(self, ct):
        """
        Load up the next template.
        Make sure to save everything, then clear all the data.
        """

        if self.templatenum+ct >= len(self.dirList) or self.templatenum+ct < 0:
            self.templatenum = max(self.templatenum, 0)
            self.templatenum = min(self.templatenum, len(self.dirList)-1)
            # Don't do anything bad.
            return

        if self.currentcontests != []:
            self.saveText()

        self.currentcontests = []
        self.templatenum += ct

        # Save the image corresponding to this template
        self.imgo = self.templateimg[self.templatenum]

        # The text that we're going to guess goes in these boxes.
        # This is used when we pattern match contests against each other.
        self.guesstext = []
        
        for ct,cid in enumerate(self.contest_order[self.templatenum]):
            # For now just punt on it.
            self.guesstext.append("")
            # But actually do fill in the current contest keys to use to index in the hashtables.
            self.currentcontests.append((self.templatenum,cid))

        # Which contest we're on.
        self.count = 0

        # It's okay to clear things now.
        self.doNotClear = False

        # Fill in any text we might have entered so far.
        self.restoreText()

        # Show everything.
        self.doadd(0)

    def updateTemplate(self):
        """
        Make the template image correspond to how it should.
        Color the current selected contest, as well as the ones we've seen so far.
        """
        img = Image.open(self.dirList[self.templatenum]).convert("RGB")

        dr = ImageDraw.Draw(img)

        c = 0
        for box in self.boxes[self.templatenum]:
            if c == self.count:
                #dr.rectangle(box, fill=(200,200,0))
                pass
            elif self.text[self.currentcontests[c]] != []:
                dr.rectangle(box, fill=(0,200,0))
            elif self.guesstext[c] != "":
                # For when we think we know what it should be but are'nt sure.
                dr.rectangle(box, fill=(0,0,200))
            else:
                dr.rectangle(box, fill=(200,0,0))

            c += 1
        # Redraw the yellow on the current so it goes on top of everything else
        dr.rectangle(self.boxes[self.templatenum][self.count], fill=(200,200,0))

        new_template = pil2wxb(Image.blend(img,self.imgo,.5).resize((303, 500)))
        self.templatebox.img.SetBitmap(new_template)
        
        SCALE = float(self.imgo.size[1])/500
        # Switch to selected contest.
        def foo(x):
            for i,(l,u,r,d) in enumerate(self.boxes[self.templatenum]):
                if l <= x.X*SCALE <= r and u <= x.Y*SCALE <= d:

                    i = self.boxes[self.templatenum].index((l,u,r,d))

                    self.doadd(i-self.count)
                    break
        self.templatebox.img.Bind(wx.EVT_LEFT_DOWN, lambda x: foo(x))

    def restoreText(self):
        arr = self.text[self.currentcontests[self.count]]
        self.text_upto.SetValue(int(self.voteupto[self.currentcontests[self.count]]))
        # First check if we've filled in text here before.
        if len(arr) == len(self.text_targets)+1:
            # Yep, we have. Restore it.
            self.text_title.SetValue(arr[0])
            for i,each in enumerate(self.text_targets):
                # NO OFF BY ONE ERROR FOR YOU!
                each.SetValue(arr[i+1])
        else:
            # None. Set the title to be what we guess it might be.
            self.text_title.SetValue(self.guesstext[self.count])
        self.text_title.SetMark(0,0)
        self.text_title.SetInsertionPointEnd()

    def saveText(self, removeit=True):
        """
        I hope I don't have to explain what this does.
        """
        #print "SAVING", self.count
        try:
            self.text_title.GetValue()
        except:
            # We haven't filled anything in yet. Just abort.
            return
        v = [self.text_title.GetValue()]+[x.GetValue() for x in self.text_targets]
        if not all(x == '' for x in v):
            # We have entered at least something ... save it
            self.text[self.currentcontests[self.count]] = v
        else:
            self.text[self.currentcontests[self.count]] = []
        self.voteupto[self.currentcontests[self.count]] = self.text_upto.GetValue()
        if removeit:
            for each in self.text_targets:
                each.SetValue("")

    def changeCompleted(self, addone=0):
        didsofar = sum([x != [] for x in self.text.values()])+addone
        num = len(self.text.values())
        didsofar = min(didsofar, num)
        self.canMoveOn = didsofar == num
        if self.canMoveOn:
            Publisher().sendMessage("broadcast.can_proceed")
        
        self.remainingText.SetLabel("Completed %d of %d."%(didsofar, num) )


    def addText(self):
        """
        Add the text to the dropdown menus.
        """
        self.text_targets = []

        def changeOptions(x, override=False):
            """
            We run this whenever we've changed the title so that we can autopopulate the rest.
            """

            # If override is set to true, it means we should clear it anyways.
            if self.doNotClear and not override:
                return
            
            self.changeCompleted(addone=1)

            v = self.text_title.GetValue()

            for k,vv in self.text.items():
                if len(vv) > 0 and v.lower() == vv[0].lower() and len(vv)-1 == len(self.text_targets):
                    # Found it. And there was text corresponding to it.
                    break
            else:
                # This title name hasn't occurred before.
                return

            for i,each in enumerate(self.text_targets):
                each.Clear()
                each.SetValue("")

            # Fill in the possible options.
            for i,each in enumerate(self.text_targets):
                # Let them reorder if need be.
                each.AppendItems(vv[1:])
                # And set to the default.
                each.SetValue(vv[1+i])
            print "SET TO", k, self.text_upto
            self.text_upto.SetValue(self.voteupto[k])

        #print "AND", self.text.values()
        print map(len,self.text.values())
        print len(self.text_targets)
        wx.StaticText(self.textarea, label="Contest Title", pos=(0,0))
        number_targets = len(self.groupedtargets[self.templatenum][self.count])
        self.text_title = wx.ComboBox(self.textarea, -1,
                                      choices=list(Set([x[0] for x in self.text.values() if x and len(x)-1 == number_targets])),
                                      style=wx.CB_DROPDOWN, pos=(0,25))
        self.text_title.Bind(wx.EVT_COMBOBOX, lambda x: changeOptions(x, override=True))
        self.text_title.Bind(wx.EVT_TEXT, changeOptions)

        self.focusIsOn = -2
        def showFocus(where, i=-1):
            # Put a little blue box over where we're entering text
            self.focusIsOn = i
            def doDraw(img):
                mine = img.copy()
                size = img.size
                dr = ImageDraw.Draw(mine)
                                      
                box = self.crop[self.currentcontests[self.count]][1]
                print box
                dr.rectangle(box, fill=(0,250,0))
                if where != None:
                    # Extract the coords, ignore the IDs
                    todraw = where[2:]
                    print todraw
                    dr.rectangle(todraw, fill=(0,0,250))
                return Image.blend(mine, img, .85)
            self.changeFocusImage(applyfn=doDraw)

        def enterPushed(it):
            print self.focusIsOn
            if self.focusIsOn == -1:
                # Focus is on the title
                if all([x.GetValue() != '' for x in self.text_targets]):
                    wx.CallAfter(self.doadd, 1)
                else:
                    self.text_targets[0].SetFocus()
            elif self.focusIsOn < len(self.text_targets)-1:
                # Focus is on a target
                self.text_targets[self.focusIsOn+1].SetFocus()
            else:
                wx.CallAfter(self.doadd, 1)

        self.text_title.Bind(wx.EVT_SET_FOCUS, lambda x: showFocus(None, -1))

        self.text_title.Bind(wx.EVT_TEXT_ENTER, enterPushed)
        
        wx.StaticText(self.textarea, label="Candidates", pos=(0,70))
        for i in range(number_targets):
            tt = wx.ComboBox(self.textarea, -1,
                             style=wx.CB_DROPDOWN, pos=(0,95+i*25))
            def c(j):
                tt.Bind(wx.EVT_SET_FOCUS, 
                        lambda x: showFocus(self.groupedtargets[self.templatenum][self.count][j], i=j))
            c(i)

            tt.Bind(wx.EVT_TEXT_ENTER, enterPushed)

            # Typing in the top box usually edits the lower stuff
            # We don't want typing to do that if we've modified the text.
            def dontrestore(x): 
                self.doNotClear = True
            tt.Bind(wx.EVT_TEXT, dontrestore)

            self.text_targets.append(tt)

        wx.StaticText(self.textarea, label="Vote for up to", pos=(0,25+95+(1+i)*25))

        self.text_upto = wx.lib.intctrl.IntCtrl(self.textarea, -1,
                                                pos=(0,50+95+(i+1)*25), value=1,
                                                min = 1, max=len(self.text_targets))
        self.text_upto.Bind(wx.EVT_SET_FOCUS, lambda x: showFocus(None))
        def enter_upto(x):
            print 'aaa'
            self.focusIsOn = -2
            enterPushed(x)
        self.text_upto.Bind(wx.EVT_TEXT_ENTER, enterPushed)


    def changeFocusImage(self, move=False, applyfn=None):
        it = self.crop[self.currentcontests[self.count]][0]
        if applyfn != None:
            it = applyfn(it)
        if not move:
            restore = self.imagebox.center, self.imagebox.scale
        self.imagebox.set_image(it)

        coords = self.crop[self.currentcontests[self.count]][1]
        center = ((coords[2]+coords[0])/2, (coords[3]+coords[1])/2)

        percentage_w = float(coords[2]-coords[0])/(500)
        percentage_h = float(coords[3]-coords[1])/(600)
        scale = min(1/percentage_w, 1/percentage_h)
        if not move:
            center, scale = restore
            print "RESTORE TO", center
        self.imagebox.set_center(center)
        self.imagebox.set_scale(scale)
        self.imagebox.Refresh()
        

    def doadd(self, ctby):
        """
        Set up everything for a given contest.
        ctby is how many contests to skip by.
        It's usually 1 or -1 (forward/backward).
        Sometimes it's 0 to show the contest for the first time.
        """

        if self.count+ctby >= len(self.currentcontests):
            self.nexttemplate(1)
            return
        if self.count+ctby < 0:
            self.nexttemplate(-1)
            return

        self.doNotClear = False

        self.saveText()

        self.count += ctby

        self.changeCompleted()

        self.textarea.DestroyChildren()

        self.updateTemplate()

        self.changeFocusImage(move=True)
        
        self.addText()

        self.restoreText()
        
        self.text_title.SetFocus()

        self.Fit()
