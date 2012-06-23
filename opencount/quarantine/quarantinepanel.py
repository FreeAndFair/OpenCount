import wx, pdb
from wx.lib.pubsub import Publisher
from util import ImageManipulate
import PIL
from PIL import Image
import pickle
import csv
import os

class QuarantinePanel(wx.Panel):
    def __init__(self, parent, *args, **kwargs):
        wx.Panel.__init__(self, parent, -1, *args, **kwargs)
        self.parent = parent

        Publisher().subscribe(self.getproj, "broadcast.project")
    
    def getproj(self, msg):
        self.proj = msg.data
        
    firstTime = True

    def start(self):
        if not self.firstTime: return
        self.firstTime = False

        sizer = wx.BoxSizer(wx.VERTICAL)

        top = TopPanel(self)
        main = MainPanel(self, self.proj)
        top.start(main)

        sizer.Add(top, proportion=1, flag=wx.ALL|wx.EXPAND, border=5)
 
        sizer.Add(main, proportion=10, flag=wx.ALL|wx.EXPAND, border=5)

        self.SetSizer(sizer)
        self.Fit()
        self.Refresh()

class MainPanel(wx.Panel):
    def __init__(self, parent, proj, *args, **kwargs):
        wx.Panel.__init__(self, parent, -1, *args, **kwargs)
        self.parent = parent
        
        self.proj = proj
        lines1 = open(self.proj.quarantined).read().split("\n")
        # Catches case when self.proj.quarantined is the empty file
        lines1 = [line for line in lines1 if line != '']
        lines2 = open(self.proj.quarantined_manual).read().split("\n")
        # Catches case when self.proj.quarantined is the empty file
        lines2 = [line for line in lines2 if line != '']
        self.qfiles = lines1+lines2
        self.count = 0
        self.number_of_contests = 0
        self.label_index = 0

        self.labeltext = {}

        for line in csv.reader(open(self.proj.contest_text)):
            if len(line) < 2: break
            self.labeltext[int(line[0])] = line[2:]
        self.labeltext = [x[1] for x in sorted(self.labeltext.items())]

        self.ballot_attributes = self.load_grouping()
        print "I GET", self.ballot_attributes

        if os.path.exists(self.proj.quarantine_internal):
            print "EXISTS"
            self.data, self.discardlist, self.attributes = pickle.load(open(self.proj.quarantine_internal))
            print self.data
            print self.discardlist
        else:
            self.data = []
            self.discardlist = []
            self.attributes = []
        
        self.candidates = []
        self.curballot = 0

        sizer = wx.BoxSizer(wx.HORIZONTAL)

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
        
        self.middle_col = wx.BoxSizer(wx.VERTICAL)

        self.contest_title = wx.ListBox(self, -1)
        for each in self.labeltext:
            self.contest_title.Append(each[0])
        self.contest_title.Select(0)

        def clicked1(x):
            v = self.contest_title.HitTest(x.GetPosition())
            self.contest_title.Select(v)
            self.show_contest(v)
        self.contest_title.Bind(wx.EVT_LEFT_DOWN, clicked1)

        self.middle_col.Add(self.contest_title)

        t = wx.StaticText(self, -1, label="Enter the information about one contest below")
        self.input_area = wx.BoxSizer(wx.VERTICAL)

        self.middle_col.Add(t)
        self.middle_col.Add(self.input_area)

        self.discard = wx.CheckBox(self, -1, label="Discard image.")
        self.middle_col.Add((1,20))
        
        self.middle_col.Add(self.discard)
        self.middle_col.Add((1,50))

        self.attribute_area = wx.BoxSizer(wx.VERTICAL)
        self.attributes_text = []

        if self.ballot_attributes:
            self.middle_col.Add(wx.StaticText(self, -1, label="Enter the ballot attributes below."))
            for title in self.ballot_attributes['header'][1:-2]:
                sz = wx.BoxSizer(wx.HORIZONTAL)
                attr = wx.TextCtrl(self, -1)
                sz.Add(wx.StaticText(self, -1, label=title))
                sz.Add(attr)
                self.attributes_text.append(attr)
                self.attribute_area.Add(sz)

        self.middle_col.Add(self.attribute_area)
        
        sizer.Add(self.middle_col)
        sizer.Add((50,1))

        contest_box = wx.BoxSizer(wx.VERTICAL)
        self.contests = wx.ListBox(self, -1, choices=[])
        def clicked2(x):
            v = self.contests.HitTest(x.GetPosition())
            self.contests.Select(v)
            self.set_contest(v)
            
        self.contests.Bind(wx.EVT_LEFT_DOWN, clicked2)

        add_new_contest = wx.Button(self, -1, label="New Contest")
        add_new_contest.Bind(wx.EVT_BUTTON, self.add_new_contest)

        remove_contest = wx.Button(self, -1, label="Remove Contest")
        remove_contest.Bind(wx.EVT_BUTTON, self.remove_contest)

        contest_box.Add(self.contests)
        contest_box.Add(add_new_contest)
        contest_box.Add(remove_contest)
        
        sizer.Add(contest_box)
        
        self.SetSizer(sizer)

        self.proj.addCloseEvent(self.save)

        if not self.qfiles:
            dlg = wx.MessageDialog(self, message="OpenCount did not \
quarantine any voted ballot images - in fact, it seemed to be able \
to process all voted ballots just fine. Press 'Ok' to proceed to \
the next step.",
                                   style=wx.OK)
            self.Disable()
            dlg.ShowModal()
            self.Enable()
            # Change pages to the next step
            notebook = self.parent.parent
            oldpage = notebook.GetSelection()
            notebook.ChangeSelection(oldpage+1)
            notebook.SendPageChangedEvent(oldpage,oldpage+1)
        else:
            self.show_ballot(0, True)
            if os.path.exists(self.proj.quarantine_internal):
                print "RESTORE"
                self.restore_contest_data()

    def load_grouping(self):
        if not os.path.exists(self.proj.grouping_results):
            print "RET NONE"
            return None

        c_t = {}
        for line in csv.reader(open(self.proj.grouping_results)):
            if len(line) < 2: continue
            if line[0] == 'samplepath':
                c_t['header'] = line[1:]
            elif os.path.abspath(line[0]) in self.qfiles:
                c_t[os.path.abspath(line[0])] = line[1:]

        return c_t

    def show_ballot(self, n, firsttime=False):
        if not self.qfiles:
            print "== No quarantined ballots exist, returning."
            return
        if self.curballot < len(self.data) and not firsttime:
            self.save_contest_data()
            self.attributes[self.curballot] = [x.GetValue() for x in self.attributes_text]

        self.reset_contests()
        first = False
        while n >= len(self.data):
            first = True
            self.data.append([])
            self.discardlist.append(False)
            self.attributes.append([])
        self.contests.Clear()


        self.curballot = n
        self.number_of_contests = len(self.data[n])
        self.count = 0 
        self.candidates = []

        print "SETTING TO", self.discardlist, self.discardlist[self.curballot]
        self.discard.SetValue(self.discardlist[self.curballot])

        print "AND ATTRS IS", self.attributes
        if self.attributes[self.curballot] != []:
            print 'a'
            data = self.attributes[self.curballot]
        elif self.qfiles[self.curballot] in self.ballot_attributes:
            print 'b'
            data = self.ballot_attributes[self.qfiles[self.curballot]][1:-2]
        else:
            print 'c'
            data = ['']*(len(self.ballot_attributes['header'])-3)
        print 'SO GET', data
        for inp,dat in zip(self.attributes_text, data):
            inp.SetValue(dat)

        for each in self.data[n]:
            self.contests.Append(self.labeltext[each[0]][0])

        if first:
            self.contest_title.SetSelection(0)
            self.add_new_contest()
        else:
            self.set_contest(0, False)
        self.contests.SetSelection(0)
        self.show_image()
        

    def add_new_target(self, x=None):
        s = wx.BoxSizer(wx.HORIZONTAL)
        name = wx.StaticText(self, -1)
        check = wx.CheckBox(self, -1, label="Voted?")
        self.candidates.append((name, check))
        s.Add(name)
        s.Add(check)
        self.input_area.Add(s)
        self.Fit()
        self.Refresh()

    def save_contest_data(self):
        if not self.qfiles:
            print "== quarantine save_contest_data: No quarantined \
ballot images exist, so, no need to save contest data."
            return
        collected = []
        collected.append(self.contest_title.GetSelection())
        for a,b in self.candidates:
            collected.append(b.GetValue())
        self.data[self.curballot][self.count] = collected
        self.discardlist[self.curballot] = self.discard.GetValue()

    def add_new_contest(self, x=None):
        self.contests.Append("choose contest")
        self.data[self.curballot].append([])
        self.set_contest(self.number_of_contests)
        self.contests.SetSelection(self.number_of_contests)
        self.number_of_contests += 1

    def reset_contests(self):
        self.input_area.Clear()
        #self.discard.SetValue(False)

        for a,b in self.candidates:
            a.Destroy()
            b.Destroy()

        self.candidates = []

    def restore_contest_data(self):
        dat = self.data[self.curballot][self.count]
        if dat == None: return
        for i,yesno in enumerate(dat[1:]):
            cs = self.candidates[i]
            cs[1].SetValue(yesno)
        self.discard.SetValue(self.discardlist[self.curballot])

    def show_contest(self, which):
        """
        Show a contest that we've decided to look at
        by clicking on the contest_title ListBox
        """
        self.contest_title.SetSelection(which)
        self.label_index = which
        self.reset_contests()
        text = self.labeltext[self.label_index]
        self.contests.SetString(self.count, text[0])
        for name in text[1:]:
            self.add_new_target()
            cs = self.candidates[-1]
            cs[0].SetLabel(name)
        self.data[self.curballot][self.count]
        self.Fit()
        self.Refresh()

    def set_contest(self, which, save=True):
        """
        Set the contest we're currently looking at to one of the
        contests we've defined from the contest ListBox.
        """
        if save:
            self.save_contest_data()
        self.contests.SetSelection(which)

        self.reset_contests()
        self.count = which

        if self.data[self.curballot][self.count] != []:
            self.show_contest(self.data[self.curballot][self.count][0])
        else:
            self.show_contest(0)

        self.restore_contest_data()
        self.Fit()
        self.Refresh()
        

    def remove_contest(self, x=None):
        if len(self.data[self.curballot]) == 1:
            dlg = wx.MessageDialog(self, message="You can not remove the only contest from a ballot.", style=wx.OK)
            dlg.ShowModal()
            return

        self.reset_contests()
        bdata = self.data[self.curballot]
        bdata = bdata[:self.count] + bdata[self.count+1:]
        self.number_of_contests -= 1
        print 'setting', bdata
        self.data[self.curballot] = bdata 
        self.contests.Delete(self.count)
        if self.count == len(bdata):
            self.set_contest(self.count-1, False)
        else:
            self.set_contest(self.count, False)
        #print 'current contest', self.count
        #print 'data', self.data
        #print 'list', self.contests

    def show_image(self):
        f = Image.open(self.qfiles[self.curballot])
        self.imagebox.set_image(f)
        self.imagebox.Refresh()
    
    def save(self):
        self.save_contest_data()

        print "WRITING ON", self.proj.quarantine_internal
        pickle.dump((self.data, self.discardlist, self.attributes), 
                    open(self.proj.quarantine_internal, "w"))

        print "SAVING DATA"
        out = open(self.proj.quarantine_res, "w")
        outattr = csv.writer(open(self.proj.quarantine_attributes, "w"))
        for bpath,ballot,drop,attr in zip(self.qfiles,self.data,self.discardlist,self.attributes):
            if drop: continue
            out.write(bpath+",")
            outattr.writerow([bpath]+attr)
            for contest in ballot:
                out.write(str(contest[0])+",")
                for each in contest[1:]:
                    # Write T/F for this contest
                    out.write(str(each)[0])
                out.write(",")
            out.write("\n")
        out.close()

class TopPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent, -1)
        
    def start(self, main):

        curnum = wx.StaticText(self, -1)
        curnum.SetLabel("1 of " + str(len(main.qfiles)))

        path = wx.StaticText(self, -1)
        if len(main.qfiles) > 0:
            path.SetLabel("Path: " + main.qfiles[0])

        def do(x):
            v = int(curnum.GetLabel().split(" ")[0])+x
            if 0 < v <= len(main.qfiles):
                curnum.SetLabel(str(v) + " of " + str(len(main.qfiles)))
                path.SetLabel("Path: " + str(main.qfiles[v-1]))
                main.show_ballot(v-1)

        prev = wx.Button(self, -1, label="Previous Ballot")
        prev.Bind(wx.EVT_BUTTON, lambda x: do(-1))
        nxt = wx.Button(self, -1, label="Next Ballot")
        nxt.Bind(wx.EVT_BUTTON, lambda x: do(1))

        mainsizer = wx.BoxSizer(wx.VERTICAL)
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(prev)
        sizer.Add(curnum)
        sizer.Add(nxt)
        mainsizer.Add(sizer)
        mainsizer.Add(path)

        self.SetSizer(mainsizer)

        self.Fit()
        self.Refresh()
