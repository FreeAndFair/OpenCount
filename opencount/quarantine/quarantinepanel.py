import wx, pdb
import wx.lib.scrolledpanel
from wx.lib.pubsub import Publisher
from util import ImageManipulate
import PIL
from PIL import Image
import csv
import os

try:
    import cPickle as pickle
except:
    import pickle

from os.path import join as pathjoin

class QuarantinePanel(wx.Panel):
    def __init__(self, parent, *args, **kwargs):
        wx.Panel.__init__(self, parent, -1, *args, **kwargs)
        self.parent = parent
        self.quarantinepanel = None

    firstTime = True

    def start(self, proj):
        self.proj = proj
        if not self.firstTime: return
        self.firstTime = False

        # 0.) Grab all quarantined ballots.
        qballotids = get_quarantined_ballots(proj)

        sizer = wx.BoxSizer(wx.VERTICAL)

        top = TopPanel(self)
        self.quarantinepanel = MainPanel(self, qballotids, self.proj)
        top.start(self.quarantinepanel)

        sizer.Add(top, proportion=1, flag=wx.ALL|wx.EXPAND, border=5)
 
        sizer.Add(self.quarantinepanel, proportion=10, flag=wx.ALL|wx.EXPAND, border=5)

        self.SetSizer(sizer)
        self.Fit()
        self.Refresh()

    def stop(self):
        if self.quarantinepanel:
            # Funny thing: If there are no quarantined attributes, then
            # the page change can happen before self.quarantinepanel gets set.
            self.quarantinepanel.save()
            self.proj.removeCloseEvent(self.quarantinepanel.save)

class MainPanel(wx.Panel):
    def __init__(self, parent, qballotids, proj, *args, **kwargs):
        wx.Panel.__init__(self, parent, -1, *args, **kwargs)
        self.parent = parent
        self.qballotids = qballotids
        
        self.proj = proj

        image_to_ballot = pickle.load(open(self.proj.image_to_ballot, 'rb'))
        ballot_to_images = pickle.load(open(self.proj.ballot_to_images, 'rb'))
        img2page = pickle.load(open(pathjoin(self.proj.projdir_path, self.proj.image_to_page), 'rb'))

        self.qfiles = []
        for ballotid in qballotids:
            imgpaths = ballot_to_images[ballotid]
            self.qfiles.extend(imgpaths)
        self.qfiles = sorted(list(set(self.qfiles)))

        self.count = 0
        self.number_of_contests = 0
        self.label_index = 0

        self.labeltext = {}

        for line in csv.reader(open(self.proj.contest_text)):
            if len(line) < 2: break
            self.labeltext[int(line[0])] = line[2:]
        self.labeltext = [x[1] for x in sorted(self.labeltext.items())]

        self.ballot_attributes = self.load_grouping()
        #print "I GET", self.ballot_attributes

        if os.path.exists(self.proj.quarantine_internal):
            print "EXISTS"
            self.data, self.discardlist, self.attributes = pickle.load(open(self.proj.quarantine_internal))
            #print self.data
            #print self.discardlist
        else:
            self.data = []
            self.discardlist = []
            self.attributes = []
        
        self.candidates = []
        self.curballot = 0

        sizer = wx.BoxSizer(wx.HORIZONTAL)

        sz2 = wx.BoxSizer(wx.VERTICAL)
        self.imagebox = ImageManipulate(self, size=(700,600))

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

        self.title_order = title_order = sorted(self.labeltext)

        title_names = [x[0] for x in title_order]
        def change(x):
            val = self.contest_title_box.GetValue()
            if val in title_names:
                v1 = title_names.index(val)
                v = self.labeltext.index(title_order[v1])
                #print "Which corresponds to", v, self.labeltext[v]
                okay = self.show_contest(v)
                if okay:
                    self.contest_title.Select(v1)
                
        self.contest_title_box = wx.ComboBox(self, -1, size=(300, 30))
        self.contest_title_box.Bind(wx.EVT_TEXT, change)

        self.contest_title = wx.ListBox(self, -1, size=(300, 100))
        for each in title_order:
            self.contest_title.Append(each[0])
        self.contest_title.Select(0)
        
        def clicked1(x):
            v1 = self.contest_title.HitTest(x.GetPosition())
            #print "And index", v1, title_order[v1]
            v = self.labeltext.index(title_order[v1])
            #print "Which corresponds to", v, self.labeltext[v]
            okay = self.show_contest(v)
            if okay:
                self.contest_title.Select(v1)
        self.contest_title.Bind(wx.EVT_LEFT_DOWN, clicked1)

        self.middle_col.Add(self.contest_title_box)
        self.middle_col.Add(self.contest_title)

        #t = wx.StaticText(self, -1, label="Enter the information about one contest below")

        self.input_area = wx.lib.scrolledpanel.ScrolledPanel(self, size=(300, 300))
        self.input_area.SetAutoLayout(True)
        self.input_area.SetupScrolling(False, True)
        self.input_area_sizer = wx.BoxSizer(wx.VERTICAL)
        self.input_area.SetSizer(self.input_area_sizer)

        #self.middle_col.Add(t)
        self.middle_col.Add(self.input_area)

        self.discard = wx.CheckBox(self, -1, label="Discard image.")
        self.middle_col.Add((1,20))
        
        self.middle_col.Add(self.discard)
        self.middle_col.Add((1,50))

        self.attribute_area = wx.BoxSizer(wx.VERTICAL)
        self.attributes_text = []

        if self.ballot_attributes:
            self.middle_col.Add(wx.StaticText(self, -1, label="Enter the ballot attributes below."))
            for title in self.ballot_attributes['header'][1:]:
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
            self.save()
            self.proj.removeCloseEvent(self.save)
            # Change pages to the next step
            notebook = self.parent.parent
            oldpage = notebook.GetSelection()
            notebook.ChangeSelection(oldpage+1)
            notebook.SendPageChangedEvent(oldpage,oldpage+1)
        else:
            self.show_ballot(0, True)
            if os.path.exists(self.proj.quarantine_internal):
                #print "RESTORE"
                self.restore_contest_data()

    def load_grouping(self):
        if not os.path.exists(self.proj.grouping_results):
            #print "RET NONE"
            return None

        bal2imgs = pickle.load(open(self.proj.ballot_to_images, 'rb'))
        img2page = pickle.load(open(pathjoin(self.proj.projdir_path,
                                             self.proj.image_to_page), 'rb'))
        c_t = {}
        for line in csv.reader(open(self.proj.grouping_results)):
            if len(line) < 2: continue
            if line[0] == 'ballotid':
                attrtypes = line[1:]
                attrtypes = attrtypes[:-1] # ignore partitionID (always at end)
                c_t['header'] = attrtypes
            elif line[0] in self.qfiles:
                ballotid = int(line[0])
                imgpaths = bal2imgs[ballotid]
                imgpaths_ordered = sorted(imgpaths, key=lambda imP: img2page[imP])
                attrvals = line[1:]
                attrvals = attrvals[:-1] # ignore partitionID (always at end)
                c_t[imgpaths_ordered[0]] = attrvals
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

        #print "SETTING TO", self.discardlist, self.discardlist[self.curballot]
        basedir, imgname = os.path.split(self.qfiles[self.curballot])
        self.discard.SetValue(self.discardlist[self.curballot])

        #print "AND ATTRS IS", self.attributes
        #print "qf", self.qfiles
        #print 'cur', self.curballot
        #print self.ballot_attributes
        if self.attributes[self.curballot] != []:
            #print 'a'
            data = self.attributes[self.curballot]
        elif self.ballot_attributes:
            if self.qfiles[self.curballot] in self.ballot_attributes:
                #print 'b'
                data = self.ballot_attributes[self.qfiles[self.curballot]][1:]
            else:
                #print 'c'
                #data = ['']*(len(self.ballot_attributes['header'])-3)
                data = ['']*(len(self.ballot_attributes['header'])-1)
        else:
            #print 'd'
            data = []
        #print 'SO GET', data
        for inp,dat in zip(self.attributes_text, data):
            inp.SetValue(dat)

        for each in self.data[n]:
            #print "ADD", each
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
        name = wx.StaticText(self.input_area, -1)
        check = wx.CheckBox(self.input_area, -1, label="Voted?")
        #print 'add ', name
        self.candidates.append((name, check))
        s.Add(name)
        s.Add(check)
        self.input_area_sizer.Add(s)
        self.Fit()
        self.Refresh()

    def save_contest_data(self):
        if not self.qfiles:
            print "== quarantine save_contest_data: No quarantined \
ballot images exist, so, no need to save contest data."
            return
        collected = []
        collected.append(self.labeltext.index(self.title_order[self.contest_title.GetSelection()]))
        for a,b in self.candidates:
            collected.append(b.GetValue())
        self.data[self.curballot][self.count] = collected
        self.discardlist[self.curballot] = self.discard.GetValue()
        for i, txtinput in enumerate(self.attributes_text):
            if i < len(self.attributes[self.curballot]):
                self.attributes[self.curballot][i] = txtinput.GetValue()
            else:
                self.attributes[self.curballot].append(txtinput.GetValue())

    def add_new_contest(self, x=None):
        self.contest_title_box.SetValue("")
        self.contest_title_box.SetFocus()
        self.contests.Append("choose contest")
        self.data[self.curballot].append([])
        self.set_contest(self.number_of_contests)
        self.contests.SetSelection(self.number_of_contests)
        self.number_of_contests += 1

    def reset_contests(self):
        self.input_area_sizer.Clear()
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

        for idx, txtinput in enumerate(self.attributes_text):
            txtinput.SetValue(self.attributes[self.curballot][idx])

    def show_contest(self, which, force=False):
        """
        Show a contest that we've decided to look at
        by clicking on the contest_title ListBox
        """
        # You can't have the same contest twice on a ballot
        if which in [x[0] if x else None for x in self.data[self.curballot]]: 
            #print which, self.count, self.data
            if [x[0] if x else None for x in self.data[self.curballot]].index(which) != self.count: return False

        self.contest_title.SetSelection(self.title_order.index(self.labeltext[which]))
        self.label_index = which
        self.reset_contests()
        text = self.labeltext[self.label_index]
        self.contests.SetString(self.count, text[0])
        for name in text[1:]:
            self.add_new_target()
            cs = self.candidates[-1]
            cs[0].SetLabel(name)
        self.Fit()
        self.Refresh()

        return True # A-OK!

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
            self.show_contest(self.data[self.curballot][self.count][0], force=True)
        else:
            self.show_contest(0, force=True)

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
        self.data[self.curballot] = bdata 

        self.contests.Delete(self.count)

        self.number_of_contests -= 1
        if self.count == len(bdata):
            self.set_contest(self.count-1, False)
        else:
            self.set_contest(self.count, False)

    def show_image(self):
        f = Image.open(self.qfiles[self.curballot])
        self.imagebox.set_image(f)
        self.imagebox.Refresh()
    
    def save(self):
        self.save_contest_data()

        #print "DATA", self.data
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

def get_quarantined_ballots(proj):
    qballotids = []
    if os.path.exists(pathjoin(proj.projdir_path, proj.partition_quarantined)):
        partition_quarantined = pickle.load(open(pathjoin(proj.projdir_path,
                                                          proj.partition_quarantined), 'rb'))
        qballotids.extend(partition_quarantined)
    if os.path.exists(pathjoin(proj.projdir_path, proj.grouping_quarantined)):
        # list GROUPING_QUARANTINED: [int ballotID_i, ...]
        grouping_quarantined = pickle.load(open(pathjoin(proj.projdir_path,
                                                         proj.grouping_quarantined), 'rb'))
        qballotids.extend(grouping_quarantined)
    if os.path.exists(proj.quarantined):
        lines = open(proj.quarantined, 'r').read().split("\n")
        lines = [int(l) for l in lines if l != '']
        qballotids.extend(lines1)
    if os.path.exists(proj.quarantined_manual):
        lines = open(proj.quarantined_manual, 'r').read().split("\n")
        lines = [int(l) for l in lines if l != '']
        qballotids.extend(lines)
    return list(set(qballotids))

def get_discarded_ballots(proj):
    discarded_balids = []
    if os.path.exists(pathjoin(proj.projdir_path, proj.partition_discarded)):
        discarded_balids.extend(pickle.load(open(pathjoin(proj.projdir_path,
                                                          proj.partition_discarded), 'rb')))
    if os.path.exists(proj.quarantine_internal):
        # Bit hacky: Peer into QuarantinePanel's internal state
        bal2imgs = pickle.load(open(proj.ballot_to_images, 'rb'))
        img2bal = pickle.load(open(proj.image_to_ballot, 'rb'))
        # Recreate the qfiles data structure...
        qballotids = list(sorted(get_quarantined_ballots(proj)))
        qfiles  = []
        for qballotid in qballotids:
            qfiles.extend(bal2imgs[qballotid])
        qfiles = sorted(list(set(qfiles)))
        data, discardlist, attributes = pickle.load(open(proj.quarantine_internal, 'rb'))
        for i, isDiscard in enumerate(discardlist):
            if isDiscard:
                imgpath = qfiles[i]
                discarded_balids.append(img2bal[imgpath])
        
    return list(set(discarded_balids))
