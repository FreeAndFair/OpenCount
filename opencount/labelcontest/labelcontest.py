import random, pdb
import math
import cStringIO
import wx, wx.lib.scrolledpanel, wx.lib.intctrl
import os, sys
from os.path import join as pathjoin
from sets import Set
from PIL import Image, ImageDraw
import csv
import pickle

from group_contests import do_grouping, final_grouping, extend_multibox, intersect, group_given_contests

sys.path.append('..')
from util import ImageManipulate, pil2wxb

from wx.lib.pubsub import Publisher

class LabelContest(wx.Panel):
    def __init__(self, parent, size):
        wx.Panel.__init__(self, parent, id=-1, size=size)

        self.parent = parent
        self.canMoveOn = False

        # a dict mapping:
        #  {(ballotid, contestid): 
        self.text = {}
        
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
        # Maps {csvfilepath: str template_imgpath}
        csvpath_map = pickle.load(open(pathjoin(self.proj.target_locs_dir,
                                                'csvpath_map.p'),
                                       'rb'))
        thewidth = theheight = None

        # groupedtargets is [[[(targetid,contestid,left,up,right,down)]]]
        # where:
        #   groupedtargets[a][b][c] is the ID and location of target C in contest B of template A
        #   len(groupedtargets) is the number of templates
        #   len(groupedtargets[a]) is the number of contests in template A
        #   len(groupedtargets[a][b]) is the number of targets in contest B of template A
        self.groupedtargets = []
        for root,dirs,files in os.walk(self.proj.target_locs_dir):
            for each in files:
                if each[-4:] != '.csv': continue
                gr = {}
                name = os.path.join(root, each)
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
                            thewidth, theheight = Image.open(row[0]).size
                lst = gr.values()
                if not lst:
                    # Means this file had no contests, so, add dummy 
                    # values to my data structures
                    self.dirList.append(csvpath_map[pathjoin(root, each)])
                    self.groupedtargets.append([])
                    continue
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
        self.template_width, self.template_height = thewidth, theheight

    def reset_panel(self):
        self.proj.removeCloseEvent(self.save)
    
    def reset_data(self):
        for f in [self.proj.contest_id, 
                  self.proj.contest_internal, 
                  self.proj.contest_text]:
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

        self.grouping_cached = None
        self.getValues()

        self.text_targets = []

        self.canMoveOn = False

        textbox = wx.BoxSizer(wx.VERTICAL)

        button1 = wx.Button(self, label='Previous Contest')
        button2 = wx.Button(self, label='Next Contest')
        button2.Bind(wx.EVT_BUTTON, lambda x: self.doadd(1))
        button22 = wx.Button(self, label='Next Unfilled Contest')
        button1.Bind(wx.EVT_BUTTON, lambda x: self.doadd(-1))
        def nextunfilled(x):
            # Get the unfilled contests.
            aftertext = [x[0] for x in self.text.items() if x[1] == []]
            # Get their actual order on the screen, not the internal order.
            aftertext = [(t,self.contest_order[t].index(c)) for t,c in aftertext]
            # Remove the ones before the current target.
            aftertext = [x for x in aftertext if x > (self.templatenum, self.count)]
            # Pick the first.
            temp,cont = min(aftertext)
            if temp != self.templatenum:
                self.nexttemplate(temp-self.templatenum)
            self.doadd(cont)
        button22.Bind(wx.EVT_BUTTON, nextunfilled)

        textbox.Add(self.textarea)
        textbox.Add(button1)
        textbox.Add(button2)
        textbox.Add(button22)

        self.remainingText = wx.StaticText(self, style=wx.TE_READONLY, size=(150,30))
        textbox.Add(self.remainingText)

        template = wx.BoxSizer(wx.VERTICAL)
        button3 = wx.Button(self, label='Previous Ballot')
        button4 = wx.Button(self, label='Next Ballot')
        button3.Bind(wx.EVT_BUTTON, lambda x: self.nexttemplate(-1))
        button4.Bind(wx.EVT_BUTTON, lambda x: self.nexttemplate(1))

        self.templatebox = wx.Panel(self, size=(303,500))
        self.templatebox.img = wx.StaticBitmap(self.templatebox)
        
        template.Add(self.templatebox)
        template.Add(button3)
        template.Add(button4)
                
        self.equivs = []
        self.has_equiv_classes = False
        self.multiboxcontests = []
        self.multiboxcontests_enter = []

        button6 = wx.Button(self, label="Compute Equiv Classes")
        button6.Bind(wx.EVT_BUTTON, self.compute_equivs)
        template.Add(button6)
        
        def addmultibox(x):
            orders = []
            for bid in range(len(self.grouping_cached)):
                order = []
                for cid in range(len(self.grouping_cached[bid])-1):
                    m1 = self.inverse_mapping[(bid,self.contest_order[bid][cid])]
                    m2 = self.inverse_mapping[(bid,self.contest_order[bid][cid+1])]
                    order.append((m1,m2))
                orders.append(order)
            print "ORDS", orders
            extension = extend_multibox(self.grouping_cached,
                                        self.inverse_mapping[(self.templatenum, self.contest_order[self.templatenum][self.count])],
                                        self.inverse_mapping[(self.templatenum, self.contest_order[self.templatenum][self.count+1])],
                                        orders)
            print "EXTENSION", extension
            self.multiboxcontests_enter += [tuple([(self.mapping[x][0], self.contest_order[self.mapping[x][0]].index(self.mapping[x][1])) for x in pair]) for pair in extension]
            
            print "MULTIBOX"
            print self.multiboxcontests_enter
            self.compute_equivs(None)
        button6 = wx.Button(self, label="Mark as Multi-Box")
        button6.Bind(wx.EVT_BUTTON, addmultibox)
        template.Add(button6)

        if self.proj.options.devmode:
            button5 = wx.Button(self, label="Magic \"I'm Done\" Button")
            def declareReady(x):
                self.save()
                for ct,cid_lst in enumerate(self.contest_order):
                    for cid in cid_lst:
                        if (ct,cid) not in self.text or self.text[ct,cid] == []:
                            numt = len(self.groupedtargets[ct][cid])
                            title = ":".join(["title", str(ct), str(cid)])
                            contests = [":".join(["contest", str(ct), str(cid), str(targ)]) for targ in range(numt)]
                            self.text[ct,cid] = [title]+contests
                        if (ct,cid) not in self.voteupto: 
                            self.voteupto[ct, cid] = 1
                print "TEXT NOW", self.text
                self.restoreText()
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


    def load_languages(self):
        if not os.path.exists(self.proj.patch_loc_dir): return {}

        result = {}
        for f in os.listdir(self.proj.patch_loc_dir):
            print "AND I GET", f, f[-4:]
            if f[-4:] == '.csv':
                print 'test', f
                take = 0
                for i,row in enumerate(csv.reader(open(os.path.join(self.proj.patch_loc_dir, f)))):
                    if i == 0:
                        if 'attr_type' in row:
                            take = row.index('attr_type')
                        else:
                            break
                    else:
                        if row[take] == 'language':
                            print 'found with lang', row[take+1]
                            result[row[0]] = row[take+1]
                            break
        return result
        

    def compute_equivs(self, x):
        self.has_equiv_classes = True
        self.equivs = [[(16, 1)], [(11, 0)], [(16, 2)], [(16, 3)], [(16, 6)], [(16, 5)], [(16, 4)], [(0, 5), (5, 6), (8, 4), (9, 5), (10, 4), (12, 5), (13, 5), (21, 5)], [(0, 6), (5, 8), (8, 5), (9, 6), (10, 5), (12, 6), (13, 6), (21, 6)], [(4, 4), (10, 6)], [(5, 7)], [(11, 1)], [(21, 4)], [(0, 1), (1, 2), (4, 1), (5, 0), (8, 1), (9, 0), (10, 1), (12, 0), (13, 0), (17, 2), (20, 1), (21, 0)], [(0, 0), (4, 2), (8, 0), (10, 2), (13, 1), (16, 0)], [(5, 2)], [(1, 3), (5, 1), (9, 1), (21, 1)], [(1, 4), (4, 5)], [(2, 0)], [(6, 0)], [(7, 0)], [(11, 2)], [(12, 1)], [(14, 0)], [(18, 0)], [(0, 2), (5, 5), (9, 2), (12, 2), (13, 2)], [(3, 0), (19, 0), (22, 0)], [(0, 3), (12, 3)], [(1, 0), (4, 3), (10, 3)], [(5, 3)], [(8, 2)], [(9, 3), (21, 2)], [(13, 3)], [(17, 3)], [(20, 2)], [(15, 0)], [(22, 1)], [(0, 4)], [(1, 1), (4, 0), (10, 0)], [(5, 4)], [(8, 3)], [(9, 4)], [(12, 4)], [(13, 4)], [(17, 1)], [(20, 3)], [(21, 3)], [(2, 1)], [(3, 1)], [(6, 1), (22, 2)], [(7, 1)], [(11, 3)], [(14, 1)], [(15, 1)], [(17, 0)], [(18, 1)], [(19, 1)], [(20, 0)]]
        self.mapping = {(11, (67, 1006, 632, 1105)): (11, 1), (13, (48, 1407, 633, 1600)): (13, 1), (3, (642, 804, 1233, 1825)): (3, 1), (10, (648, 774, 1233, 1071)): (10, 6), (1, (51, 1407, 639, 1640)): (1, 3), (17, (648, 1635, 1243, 1828)): (17, 2), (13, (48, 1632, 633, 1898)): (13, 2), (0, (54, 1416, 633, 1602)): (0, 0), (4, (51, 774, 648, 1152)): (4, 3), (6, (645, 804, 1233, 1826)): (6, 1), (15, (42, 735, 641, 1190)): (15, 0), (10, (51, 1185, 648, 1378)): (10, 1), (4, (51, 1407, 648, 1603)): (4, 2), (12, (46, 768, 633, 1150)): (12, 3), (0, (54, 771, 633, 1149)): (0, 3), (8, (48, 138, 645, 737)): (8, 3), (3, (39, 735, 638, 1077)): (3, 0), (2, (39, 735, 639, 967)): (2, 0), (21, (48, 1182, 633, 1376)): (21, 0), (14, (39, 735, 640, 965)): (14, 0), (12, (646, 399, 1243, 737)): (12, 6), (15, (646, 810, 1233, 1826)): (15, 1), (0, (654, 177, 1243, 405)): (0, 5), (4, (633, 165, 1233, 733)): (4, 5), (11, (67, 1115, 630, 1186)): (11, 0), (8, (648, 174, 1233, 401)): (8, 4), (5, (51, 132, 633, 739)): (5, 4), (12, (46, 132, 633, 737)): (12, 4), (9, (47, 138, 653, 740)): (9, 4), (8, (48, 1182, 645, 1376)): (8, 1), (22, (39, 735, 639, 1075)): (22, 0), (5, (51, 1668, 633, 1938)): (5, 2), (10, (639, 399, 1233, 734)): (10, 5), (1, (51, 1182, 639, 1377)): (1, 2), (5, (51, 1182, 633, 1372)): (5, 0), (21, (48, 771, 646, 1152)): (21, 2), (6, (39, 735, 640, 964)): (6, 0), (7, (670, 810, 1234, 1823)): (7, 1), (21, (48, 127, 646, 739)): (21, 3), (16, (847, 768, 1233, 1075)): (16, 4), (18, (70, 810, 640, 963)): (18, 0), (20, (645, 1218, 1241, 1597)): (20, 2), (5, (654, 1071, 1243, 1372)): (5, 7), (4, (42, 135, 648, 733)): (4, 0), (4, (51, 1182, 648, 1377)): (4, 1), (16, (857, 167, 1243, 393)): (16, 6), (7, (69, 774, 638, 964)): (7, 0), (12, (46, 1407, 633, 1634)): (12, 1), (21, (48, 1407, 633, 1639)): (21, 1), (1, (42, 774, 639, 1152)): (1, 0), (4, (633, 774, 1233, 1071)): (4, 4), (17, (637, 603, 1243, 1190)): (17, 1), (5, (654, 697, 1243, 1041)): (5, 8), (9, (47, 1671, 618, 1937)): (9, 2), (21, (651, 397, 1233, 739)): (21, 6), (16, (654, 768, 857, 1075)): (16, 3), (5, (51, 1413, 633, 1636)): (5, 1), (21, (651, 174, 1233, 403)): (21, 5), (22, (653, 804, 1233, 1824)): (22, 2), (0, (54, 132, 633, 740)): (0, 4), (18, (673, 810, 1233, 1824)): (18, 1), (10, (42, 138, 639, 744)): (10, 0), (2, (664, 807, 1233, 1829)): (2, 1), (8, (48, 768, 645, 1152)): (8, 2), (1, (642, 165, 1233, 734)): (1, 4), (9, (647, 400, 1243, 740)): (9, 6), (9, (647, 177, 1243, 406)): (9, 5), (10, (639, 165, 1233, 395)): (10, 4), (16, (639, 167, 851, 393)): (16, 1), (9, (47, 1185, 618, 1376)): (9, 0), (8, (648, 395, 1233, 737)): (8, 5), (22, (39, 1107, 623, 1599)): (22, 1), (16, (45, 141, 641, 327)): (16, 0), (17, (39, 846, 575, 1867)): (17, 0), (12, (46, 1674, 633, 1936)): (12, 2), (13, (48, 132, 654, 737)): (13, 4), (0, (654, 399, 1243, 740)): (0, 6), (16, (847, 394, 1233, 737)): (16, 5), (19, (42, 735, 639, 1077)): (19, 0), (5, (51, 771, 633, 1152)): (5, 3), (20, (39, 846, 603, 1861)): (20, 0), (0, (54, 1641, 633, 1904)): (0, 2), (13, (48, 1182, 633, 1375)): (13, 0), (9, (47, 1407, 618, 1639)): (9, 1), (5, (643, 132, 1243, 404)): (5, 5), (10, (51, 774, 648, 1153)): (10, 3), (21, (640, 771, 1233, 1038)): (21, 4), (13, (48, 774, 654, 1150)): (13, 3), (10, (51, 1410, 648, 1604)): (10, 2), (12, (646, 177, 1243, 405)): (12, 5), (13, (666, 399, 1243, 737)): (13, 6), (16, (645, 394, 857, 737)): (16, 2), (17, (648, 1221, 1243, 1603)): (17, 3), (5, (654, 474, 1243, 703)): (5, 6), (9, (47, 771, 633, 1151)): (9, 3), (1, (42, 138, 639, 734)): (1, 1), (19, (642, 813, 1233, 1824)): (19, 1), (0, (54, 1188, 633, 1375)): (0, 1), (20, (645, 1629, 1241, 1822)): (20, 1), (20, (645, 588, 1241, 1186)): (20, 3), (11, (67, 819, 623, 992)): (11, 2), (8, (48, 1407, 645, 1601)): (8, 0), (11, (657, 813, 1233, 1828)): (11, 3), (12, (46, 1182, 633, 1375)): (12, 0), (13, (666, 177, 1243, 405)): (13, 5), (14, (677, 810, 1233, 1825)): (14, 1)}
        self.inverse_mapping = dict((v,k) for k,v in mapping.items())
        self.reorder = {(16, 6): {(16, 6): [(0, 0), (1, 1)]}, (13, 4): {(13, 4): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13)]}, (1, 3): {(5, 1): [(3, 3), (2, 2), (1, 1), (0, 0)], (1, 3): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)], (9, 1): [(3, 3), (0, 0), (1, 1), (2, 2)], (21, 1): [(3, 3), (2, 2), (0, 0), (1, 1)]}, (12, 1): {(12, 1): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]}, (20, 3): {(20, 3): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13)]}, (18, 0): {(18, 0): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]}, (3, 0): {(3, 0): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)], (19, 0): [(6, 6), (2, 2), (4, 4), (5, 5), (3, 3), (1, 1), (0, 0)], (22, 0): [(0, 5), (6, 6), (3, 3), (2, 2), (4, 4), (1, 1), (5, 0)]}, (11, 2): {(11, 2): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]}, (16, 2): {(16, 2): [(0, 0), (1, 1)]}, (2, 1): {(2, 1): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23), (24, 24), (25, 25)]}, (15, 1): {(15, 1): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23), (24, 24), (25, 25)]}, (17, 3): {(17, 3): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)]}, (14, 0): {(14, 0): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]}, (9, 4): {(9, 4): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13)]}, (0, 3): {(0, 3): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)], (12, 3): [(7, 7), (0, 0), (4, 4), (1, 1), (5, 5), (3, 3), (6, 6), (2, 2)]}, (18, 1): {(18, 1): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23), (24, 24), (25, 25)]}, (3, 1): {(3, 1): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23), (24, 24), (25, 25)]}, (2, 0): {(2, 0): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]}, (16, 3): {(16, 3): [(0, 0), (1, 1)]}, (4, 4): {(4, 4): [(0, 0), (1, 1), (2, 2)], (10, 6): [(0, 0), (1, 1)]}, (15, 0): {(15, 0): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10)]}, (17, 0): {(17, 0): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23), (24, 24), (25, 25)]}, (20, 0): {(20, 0): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23), (24, 24), (25, 25)]}, (14, 1): {(14, 1): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23), (24, 24), (25, 25)]}, (11, 1): {(11, 1): [(0, 0), (1, 1), (2, 2)]}, (0, 4): {(0, 4): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13)]}, (1, 1): {(10, 0): [(12, 12), (4, 4), (9, 9), (6, 6), (7, 7), (1, 1), (2, 2), (3, 3), (0, 0), (11, 11), (8, 8), (5, 5), (10, 10)], (1, 1): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13)], (4, 0): [(12, 12), (4, 4), (1, 1), (3, 3), (9, 9), (2, 2), (11, 11), (6, 6), (5, 5), (7, 7), (8, 8), (0, 0), (10, 10)]}, (5, 4): {(5, 4): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13)]}, (0, 0): {(13, 1): [(2, 2), (0, 0), (1, 1)], (0, 0): [(0, 0), (1, 1), (2, 2), (3, 3)], (8, 0): [(2, 2), (0, 0), (1, 1)], (16, 0): [(2, 2), (0, 0), (1, 1)], (4, 2): [(2, 2), (0, 0), (1, 1)], (10, 2): [(2, 2), (0, 0), (1, 1)]}, (8, 2): {(8, 2): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)]}, (16, 4): {(16, 4): [(0, 0), (1, 1)]}, (9, 3): {(9, 3): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)], (21, 2): [(7, 7), (4, 4), (5, 5), (6, 6), (1, 1), (0, 0), (3, 3), (2, 2)]}, (6, 0): {(6, 0): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]}, (1, 4): {(4, 5): [(0, 0), (3, 3), (1, 1)], (1, 4): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]}, (17, 1): {(17, 1): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13)]}, (11, 0): {(11, 0): [(0, 0), (1, 1)]}, (21, 4): {(21, 4): [(0, 0), (1, 1), (2, 2)]}, (7, 1): {(7, 1): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23), (24, 24), (25, 25)]}, (0, 5): {(10, 4): [(0, 0), (1, 1)], (5, 6): [(0, 0), (1, 1)], (13, 5): [(0, 0), (1, 1)], (12, 5): [(0, 0), (1, 1)], (0, 5): [(0, 0), (1, 1), (2, 2)], (21, 5): [(0, 0), (1, 1)], (9, 5): [(0, 0), (1, 1)], (8, 4): [(0, 0), (1, 1)]}, (16, 5): {(16, 5): [(0, 0), (1, 1)]}, (1, 0): {(1, 0): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)], (10, 3): [(7, 7), (6, 6), (0, 0), (4, 4), (1, 1), (3, 3), (5, 5), (2, 2)], (4, 3): [(7, 7), (4, 4), (6, 6), (2, 2), (0, 0), (1, 1), (5, 5), (3, 3)]}, (13, 3): {(13, 3): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)]}, (5, 3): {(5, 3): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)]}, (0, 1): {(0, 1): [(0, 0), (1, 1), (2, 2), (3, 3)], (1, 2): [(2, 2), (0, 0), (1, 1)], (9, 0): [(2, 2), (0, 0), (1, 1)], (10, 1): [(2, 2), (0, 0), (1, 1)], (21, 0): [(2, 2), (1, 1), (0, 0)], (12, 0): [(2, 2), (0, 0), (1, 1)], (8, 1): [(2, 2), (0, 0), (1, 1)], (20, 1): [(2, 2), (0, 0), (1, 1)], (17, 2): [(2, 2), (0, 0), (1, 1)], (5, 0): [(2, 2), (1, 1), (0, 0)], (4, 1): [(2, 2), (0, 0), (1, 1)], (13, 0): [(2, 2), (1, 1), (0, 0)]}, (8, 3): {(8, 3): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13)]}, (7, 0): {(7, 0): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]}, (20, 2): {(20, 2): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)]}, (6, 1): {(6, 1): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23), (24, 24), (25, 25)], (22, 2): [(24, 24), (18, 18), (23, 23), (21, 21), (6, 6), (19, 19), (11, 11), (9, 9), (15, 15), (20, 20), (5, 5), (10, 10), (0, 0), (1, 1), (22, 22), (16, 16), (7, 7), (17, 17), (12, 12), (4, 4), (2, 2), (8, 8), (3, 3), (13, 13), (14, 14)]}, (5, 7): {(5, 7): [(0, 0), (1, 1), (2, 2)]}, (11, 3): {(11, 3): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23), (24, 24), (25, 25)]}, (21, 3): {(21, 3): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13)]}, (16, 1): {(16, 1): [(0, 0)]}, (0, 6): {(10, 5): [(0, 0), (1, 1)], (8, 5): [(0, 0), (1, 1)], (12, 6): [(0, 0), (1, 1)], (13, 6): [(0, 0), (1, 1)], (0, 6): [(0, 0), (1, 1), (2, 2)], (21, 6): [(0, 0), (1, 1)], (9, 6): [(0, 0), (1, 1)], (5, 8): [(0, 0), (1, 1)]}, (12, 4): {(12, 4): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13)]}, (22, 1): {(22, 1): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10)]}, (19, 1): {(19, 1): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23), (24, 24), (25, 25)]}, (5, 2): {(5, 2): [(0, 0), (1, 1), (2, 2), (3, 3)]}, (0, 2): {(9, 2): [(3, 3), (1, 1), (0, 0), (2, 2)], (12, 2): [(3, 3), (1, 1), (2, 2), (0, 0)], (13, 2): [(3, 3), (1, 1), (0, 0), (2, 2)], (0, 2): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)], (5, 5): [(3, 3), (1, 1), (0, 0), (2, 2)]}}
        self.reorder_inverse = {(16, 6): (16, 6), (12, 1): (12, 1), (13, 4): (13, 4), (15, 1): (15, 1), (21, 6): (0, 6), (8, 5): (0, 6), (5, 8): (0, 6), (4, 0): (1, 1), (9, 0): (0, 1), (5, 5): (0, 2), (16, 3): (16, 3), (12, 6): (0, 6), (17, 2): (0, 1), (14, 1): (14, 1), (0, 4): (0, 4), (1, 1): (1, 1), (8, 2): (8, 2), (4, 5): (1, 4), (9, 3): (9, 3), (6, 0): (6, 0), (11, 0): (11, 0), (16, 0): (0, 0), (16, 5): (16, 5), (0, 1): (0, 1), (3, 1): (3, 1), (13, 0): (0, 1), (18, 0): (18, 0), (20, 3): (20, 3), (21, 2): (9, 3), (2, 1): (2, 1), (9, 4): (9, 4), (5, 1): (1, 3), (10, 3): (1, 0), (12, 2): (0, 2), (13, 3): (13, 3), (7, 1): (7, 1), (15, 0): (15, 0), (20, 0): (20, 0), (21, 5): (0, 5), (4, 1): (0, 1), (5, 4): (5, 4), (10, 4): (0, 5), (16, 4): (16, 4), (17, 1): (17, 1), (13, 6): (0, 6), (0, 5): (0, 5), (1, 0): (1, 0), (22, 0): (3, 0), (8, 3): (8, 3), (9, 2): (0, 2), (6, 1): (6, 1), (5, 7): (5, 7), (11, 3): (11, 3), (16, 1): (16, 1), (12, 4): (12, 4), (19, 1): (19, 1), (0, 2): (0, 2), (1, 3): (1, 3), (3, 0): (3, 0), (8, 0): (0, 0), (18, 1): (18, 1), (21, 1): (1, 3), (5, 0): (0, 1), (10, 0): (1, 1), (12, 3): (0, 3), (13, 2): (0, 2), (20, 1): (0, 1), (1, 4): (1, 4), (21, 4): (21, 4), (4, 2): (0, 0), (9, 6): (0, 6), (5, 3): (5, 3), (10, 5): (0, 6), (7, 0): (7, 0), (12, 0): (0, 1), (17, 0): (17, 0), (13, 5): (0, 5), (0, 6): (0, 6), (22, 1): (22, 1), (8, 4): (0, 5), (9, 1): (1, 3), (5, 6): (0, 5), (11, 2): (11, 2), (10, 6): (4, 4), (16, 2): (16, 2), (12, 5): (0, 5), (17, 3): (17, 3), (14, 0): (14, 0), (19, 0): (3, 0), (0, 3): (0, 3), (1, 2): (0, 1), (22, 2): (6, 1), (8, 1): (0, 1), (4, 4): (4, 4), (11, 1): (11, 1), (0, 0): (0, 0), (21, 0): (0, 1), (10, 1): (0, 1), (13, 1): (0, 0), (20, 2): (20, 2), (21, 3): (21, 3), (2, 0): (2, 0), (4, 3): (1, 0), (9, 5): (0, 5), (5, 2): (5, 2), (10, 2): (0, 0)}
        self.multiboxcontests = []
        
        return
        languages = self.load_languages()

        # Regroup the targets so that equal contests are merged.
        targets = []
        did = {}
        for bid,ballot in enumerate(self.groupedtargets):
            ballotlist = []
            for gid,targlist in enumerate(ballot):
                if (bid, gid) in did: continue
                if any((bid, gid) in x for x in self.multiboxcontests_enter):
                    # These are the ones we will merge
                    use = [x for x in self.multiboxcontests_enter if (bid, gid) in x][0]
                    tmp = []
                    for b,g in use:
                        tmp += self.groupedtargets[b][g]
                        did[b,g] = True
                    ballotlist.append([x[2:] for x in tmp])
                else:
                    ballotlist.append([x[2:] for x in targlist])
                    did[bid,gid] = True
            targets.append(ballotlist)

        #print "ALL", targets
        if self.grouping_cached:
            groups = final_grouping(self.grouping_cached, targets)
        else:
            if not self.proj.infer_bounding_boxes:
                dlg = wx.MessageDialog(self, message="You must auto-detect bounding boxes in select-and-group-targets to run the inference.", style=wx.OK)
                dlg.ShowModal()
                return

            ballots, groups = group_given_contests(self.proj.ocr_tmp_dir, 
                                                   self.dirList, targets, 
                                                   self.boxes, languages)
            self.grouping_cached = ballots
            print "CACHED", ballots

        gr = [[(b[0][0],b[0][1]) for b in group] for group in groups]

        print gr
        mapping = {}
        for ballot_count, ballot in enumerate(self.groupedtargets):
            print 'on ballot', ballot_count
            # All of the bounding boxes for a ballot.
            contestbboxes = [x[1] for x in sum(gr, []) if x[0] == ballot_count]
            print 'bboxes', contestbboxes
            for targetlist in ballot:
                print 'picking rep ', targetlist[0][2:]
                w = [i for i,bblist in enumerate(contestbboxes) if any(intersect(targetlist[0][2:], x) for x in bblist)]
                if len(w) != 1:
                    print 'I got', w, 'of them'
                    print [bblist for i,bblist in enumerate(contestbboxes) if any(intersect(targetlist[0][2:], x) for x in bblist)]
                    print 'gr', gr
                    print 'mapping', mapping
                    print 'contest bboxes', contestbboxes
                    print 'targetlist', targetlist
                    raise Exception("OH NO SOMETHING WENT WRONG")
                print 'w', w
                print 'contest', targetlist[0][1], 'corresponds to', contestbboxes[w[0]]
                for bbox in contestbboxes[w[0]]:
                    mapping[ballot_count, bbox] = (ballot_count, targetlist[0][1])
        print 
        print mapping
        self.mapping = mapping
        self.inverse_mapping = dict((v,k) for k,v in mapping.items())

        reorder = {}
        reorder_inverse = {}
        for group in [[(x[0][0],x[0][1], x[1]) for x in g] for g in groups]:
            first = (group[0][0], group[0][1][0])
            reorder[mapping[first]] = {}
            for ballotid, contests, order in group:
                contest = contests[0]
                reorder[mapping[first]][mapping[ballotid,contest]] = order
                reorder_inverse[mapping[ballotid, contest]] = mapping[first]
        self.reorder = reorder
        self.reorder_inverse = reorder_inverse
        print "BOTH"
        print reorder
        print reorder_inverse

        self.multiboxcontests = [[(c,self.contest_order[c][b]) for c,b in y] for y in self.multiboxcontests_enter]
        print "MUL", self.multiboxcontests

        self.equivs = [[mapping[bid,bboxes[0]] for bid,bboxes in group] for group in gr]
        print "EQUIVS", self.equivs


    def save(self):
        self.saveText(removeit=False)


        did_multibox = {}
        groupedtext = {}
        for k in self.text.keys():
            if k in did_multibox: continue
            t = []
            did_multibox[k] = []
            for each in self.continued_contest(k):
                did_multibox[k].append(each)
                t += self.text[each][1:]
            if self.text[k]:
                groupedtext[k] = [self.text[k][0]]+t
            
        print did_multibox

        # We want to figure out which contests are "equal"
        #  so that when we tally votes we report them together.
        # Equality is defined as having all the same text.


        # (bid,cid) tuples
        equal = []
        used = {}

        for k1,v1 in groupedtext.items():
            if k1 in used: continue
            eq = []
            for k2,v2 in groupedtext.items():
                if k2 in used: continue
                if v1 == v2:
                    it = self.contestID[k2]
                    eq.append((it[0], it[1]))
                    used[k2] = True
            equal.append(eq)
            used[k1] = True

        c_id = csv.writer(open(self.proj.contest_id, "w"))
        # path, ID in select-and-group-targets file, new ID, ordering

        mapping = {}
        for num,group in enumerate(equal):
            for item in group:
                print "ITEM", item
                mapping[item] = num
                ids = []
                for each in self.continued_contest(item):
                    # We need to get the contest ID in the new list
                    targets = [x for x in self.groupedtargets[each[0]] if x[0][1] == each[1]][0]
                    ids += [str(x[0]) for x in targets]
                for cont in did_multibox[each]:
                    # Save it for all boxes in the contest.
                    c_id.writerow([self.dirList[cont[0]],cont[1],num]+ids)

        # We write out the result as a mapping from Contest ID to text
        id_to_text = {}
        for k,v in groupedtext.items():
            bid, cid = self.contestID[k]
            id_to_text[(bid, cid)] = [str(self.voteupto[k])]+v

        fout = csv.writer(open(self.proj.contest_text, "w"))

        did = {}
        for k,v in id_to_text.items():
            if mapping[k] not in did:
                # ContestID, upto, title, (label)*
                fout.writerow([mapping[k]]+v)
                did[mapping[k]] = True

        pickle.dump((self.text, self.voteupto, self.grouping_cached), open(self.proj.contest_internal, "w"))
                    
    def setupBoxes(self):
        if self.proj.infer_bounding_boxes:
            res = []
            for root,dirs,files in os.walk(self.proj.target_locs_dir):
                for each in files:
                    if each[-4:] != '.csv': continue
                    name = os.path.join(root, each)
                    ballot = []
                    for i, row in enumerate(csv.reader(open(name))):
                        if i == 0:
                            continue
                        if row[7] == '1':
                            ballot.append((int(row[1]),
                                           int(row[2]), int(row[3]), 
                                           int(row[2])+int(row[4]), 
                                           int(row[3])+int(row[5])))
                    res.append(ballot)
            print "LOADING", res
            # When we load from select-and-group-targets, the order we
            # get isn't the adjusted ordering. We need to correct the
            # order so that we can work with it.
            correctorder = [[y[0][1] for y in x] for x in self.groupedtargets]

            self.boxes = []
            for ballot,order in zip(res,correctorder):
                boxes = []
                for o in order:
                    boxes.append([x for x in ballot if x[0] == o][0])
                self.boxes.append(boxes)
            return

        self.boxes = []
        def bound(targetlst, goleft, width):
            l = u = 2**32
            r = d = 0
            for _,ID,ll,uu,rr,dd in targetlst:
                l = min(ll,l)
                u = min(uu,u)
                d = max(dd,d)
                r = max(rr,r)
            return ID, l-goleft,u-int(.06*self.template_height), l+width,d+int(.03*self.template_height)
        
        # The (id,l,u,r,d) tuple for each contest of each template

        for i,ballot in enumerate(self.groupedtargets):
            ballotw = self.template_width
            columns = {}
            for group in ballot:
                for target in group:
                    if not any([target[2] + x in columns for x in range(-5,5)]):
                        columns[target[2]] = True
            columns = sorted(columns.keys())
            if len(columns) == 0:
                # We have a blank ballot with no contests
                self.boxes.append([])
                continue
            leftmargin = min(columns)
            # Interior space available
            remaining = max(ballotw-(2*leftmargin), 0.10*self.template_height)
            # The size of one box
            boxwidth = remaining/len(columns)
            boxwidth = min(boxwidth, self.template_width/2)
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
                self.text, self.voteupto, self.grouping_cached = pickle.load(open(self.proj.contest_internal))

        # The PIL image for the contest.
        # Keys are of the form templateid:(l,u,r,d)
        self.crop = {}
        self.resize = {}

        self.groups = []

        self.setupBoxes()

        # Convert the ballot:boundingbox -> (ballotid, contestid)
        self.contestID = {}

        maxw,maxh = self.thesize
        # self.boxes is
        for i,each in enumerate(self.boxes):
            for x in each:
                if not restored:
                    self.text[i,x[0]] = []
                    self.voteupto[i,x[0]] = 1
                factor = 1
                try:
                    self.crop[i,x[0]] = (self.dirList[i],
                                         (x[1], x[2], 
                                          int((x[3]-x[1])*factor+x[1]),
                                          int((x[4]-x[2])*factor+x[2])))
                except Exception as e:
                    print e
                    pdb.set_trace()
                self.contestID[i,x[0]] = (i, x[0])

        self.contest_order = [[y[0] for y in x] for x in self.boxes]
        self.boxes = [[y[1:] for y in x] for x in self.boxes]
        print "CROP", self.crop

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

        if len(self.groupedtargets[self.templatenum]) == 0:
            # No contests, so skip it
            print "Skipping the empty blank template:", self.templatenum
            self.nexttemplate(1)
            return

        # Save the image corresponding to this template
        self.imgo = Image.open(self.dirList[self.templatenum]).convert("RGB")
        
        for cid in self.contest_order[self.templatenum]:
            # Fill in the current contest keys to use to index in the hashtables.
            self.currentcontests.append((self.templatenum,cid))

        # Which contest we're on.
        self.count = 0

        # It's okay to clear things now.
        self.doNotClear = False

        print 'ballot switch restore'
        # Fill in any text we might have entered so far.
        #self.restoreText()
        print 'now do add'

        # Show everything.
        self.doadd(0, dosave=False)

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
        if len(self.groupedtargets[self.templatenum]) == 0:
            return
        arr = self.text[self.currentcontests[self.count]]
        print "RESTORE", self.currentcontests[self.count], arr
        self.text_upto.SetValue(int(self.voteupto[self.currentcontests[self.count]]))
        # First check if we've filled in text here before.
        if len(arr) == len(self.text_targets)+1:
            # Yep, we have. Restore it.
            self.text_title.SetValue(arr[0])
            for i,each in enumerate(self.text_targets):
                # NO OFF BY ONE ERROR FOR YOU!
                each.SetValue(arr[i+1])
            print 'all is well'
        self.text_title.SetMark(0,0)
        self.text_title.SetInsertionPointEnd()


    def continued_contest(self, item):
        if any(item in x for x in self.multiboxcontests):
            return [x for x in self.multiboxcontests if item in x][0]
        else:
            return [item]


    def saveText(self, removeit=True):
        """
        Save the text associated with the current contest.

        We also look to see if this contest is in an equiv-class, and, if it is,
        then go ahead and automatically enter the text in the other contests.
        We may need to update the candidate order, since the contest might
        have randomized candidate ordering.
        """
        print "SAVING", self.templatenum, self.count
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

        if not self.has_equiv_classes:
            return

        print "EQUAL ARE", self.equivs

        cur = self.currentcontests[self.count]
        print 'This contest is', cur
        cur = self.continued_contest(cur)

        # TODO: SORT ME
        print 'and now it is', cur
        
        print 'txt', self.text

        if self.text[cur[0]] == []: return

        title = self.text[cur[0]][0]
        
        # This is a temporary hack.
        try:
            # Test if it's defined
            x = self.reorder_inverse
        except:
            equclass = [x for x in self.equivs if cur[0] in x][0]
            for each in equclass:
                self.text[each] = [title]
            return

        text = [self.text[x][1:] if self.text[x] != [] else ['']*len(self.groupedtargets[x[0]][self.contest_order[x[0]].index(x[1])]) for x in cur]
        text = sum(text, [])
        print 'this', text

        cur_in_dict = [x for x in cur if x in self.reorder_inverse][0]
        set_repr = self.reorder_inverse[cur_in_dict]
        print 'set repr', set_repr
        reorder = self.reorder[set_repr][cur_in_dict]
        print "reorder", reorder
        adjusted = {}
        for i,t in enumerate(text):
            print 'sending', i, 'to', [x for x,y in reorder if y == i][0]
            adjusted[[x for x,y in reorder if y == i][0]] = t
        print adjusted
        print 'compare'
        print text
        text = [x[1] for x in sorted(adjusted.items())]
        print text
        
        if any(x in y for x in cur for y in self.equivs):
            print 'yes'
            # Find the equivilance class
            for each in self.equivs:
                if any(x in cur for x in each):
                    print 'found'
                    eqclass = each
                    break
            print 'diff', eqclass
            # Get the different one
            for continuation in eqclass:
                print 'WORKING ON CONT', continuation
                continuation = self.continued_contest(continuation)
                
                print 'assign to', continuation
                index = 0
                each_in_dict = [x for x in continuation if x in self.reorder[set_repr]][0]
                reorder = self.reorder[set_repr][each_in_dict]
                print 'new reorder', reorder
                twiceadjusted = {}
                for i,t in enumerate(adjusted):
                    print i, [y for x,y in reorder if x == i]
                    twiceadjusted[[y for x,y in reorder if x == i][0]] = t
                print 'setting', each, twiceadjusted
                newtext = [text[x[1]] for x in sorted(twiceadjusted.items())]
                print 'is now', newtext


                for each in sorted(continuation):
                    size = len(self.groupedtargets[each[0]][self.contest_order[each[0]].index(each[1])])
                    print 'assign', size, 'of them starting from', index, ':', [title]+newtext[index:index+size]
                    self.text[each] = [title]+newtext[index:index+size]
                    self.voteupto[each] = self.voteupto[self.currentcontests[self.count]]

                    index += size
            
        print 'txtnow', self.text
        return

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


        if len(self.groupedtargets[self.templatenum]) == 0:
            # There are no contests on this ballot.
            print "A"*500
            print self.templatenum, self.count
            self.contesttitle = wx.StaticText(self.textarea, label="Contest Title", pos=(0,0))
            self.text_title = wx.ComboBox(self.textarea, -1,
                                          choices=[],
                                          style=wx.CB_DROPDOWN, pos=(0,25))
            self.text_upto = wx.lib.intctrl.IntCtrl(self.textarea, pos=(0,-10000))
            return
        
        #print "AND", self.text.values()
        print map(len,self.text.values())
        print len(self.text_targets)
        self.contesttitle = wx.StaticText(self.textarea, label="Contest Title", pos=(0,0))
        print "---------", self.currentcontests, self.currentcontests[self.count], self.templatenum, self.count
        print self.contest_order
        number_targets = len(self.groupedtargets[self.templatenum][self.count])
        print self.groupedtargets[self.templatenum]
        print map(len,self.groupedtargets[self.templatenum])
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
        
        if number_targets == 0: 
            self.text_upto = wx.lib.intctrl.IntCtrl(self.textarea, pos=(0,-10000))
            return

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
            self.focusIsOn = -2
            enterPushed(x)
        self.text_upto.Bind(wx.EVT_TEXT_ENTER, enterPushed)


    def changeFocusImage(self, move=False, applyfn=None):
        it = self.imgo#self.crop[self.currentcontests[self.count]][0]
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
        self.imagebox.set_center(center)
        self.imagebox.set_scale(scale)
        self.imagebox.Refresh()
        

    def doadd(self, ctby, dosave=True):
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

        if dosave:
            # We don't want to save when we're moving to a different
            #  ballot image. We've already done that in nexttemplate.
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

