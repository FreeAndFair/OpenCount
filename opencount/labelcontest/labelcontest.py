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
from verifycontestgrouping import VerifyContestGrouping

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
        print "dirList", self.dirList

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
                    m1 = self.mapping_inverse[(bid,self.contest_order[bid][cid])]
                    m2 = self.mapping_inverse[(bid,self.contest_order[bid][cid+1])]
                    order.append((m1,m2))
                orders.append(order)
            print "ORDS", orders
            extension = extend_multibox(self.grouping_cached,
                                        self.mapping_inverse[(self.templatenum, self.contest_order[self.templatenum][self.count])],
                                        self.mapping_inverse[(self.templatenum, self.contest_order[self.templatenum][self.count+1])],
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

        if False:
            groups = [[((16, [(639, 167, 851, 393)], [(False, u'PROPOSITION 28:\nOFFICE. INITIATIVI\nReduces total amount m\nlegislature from 14 yean\nhouse. Applies only lo\npassed, Fiscal Impact:\ngovemmenls.\n  `\n\n')]), [(0, 0)])], [((11, [(67, 1115, 630, 1186)], [(False, u'\n'), (True, u'"j \n\n')]), [(0, 0), (1, 1)])], [((16, [(645, 394, 857, 737)], [(False, u"\u2019ROPOSITION 29: I\nSIGARETTES FOR (\n'STATUTE. lmposesa\nan equivaleni tax increa\nesearch for cancer and\nQet increase in cigarsm\nannually by 2013-14 for\nzessalkan programs. O1\namounting to tens of mi`\n\n"), (True, u"<'>1\n\n")]), [(0, 0), (1, 1)])], [((16, [(654, 768, 857, 1075)], [(False, u'MUIR BEACH (\nhall Muir Beach CSD 1\nnnual special lax cl tw\nxur (4) years, FY 2012-\n015-2015, on each pai\nrctsdion, inducing ius\npprcved and shall the\nmessed by the amour\n\n'), (True, u'(_)\\\n\n')]), [(0, 0), (1, 1)])], [((16, [(857, 167, 1243, 393)], [(False, u"IITS ON LEG|SLATOR$\u2019 TERMS IN\n\xa70NSTITUTIONAL AMENDMENT.\nme a person may serve in the state\n0 12 years Allows 12 years' service in one\nislalors lirst eleqed after measure is\no direct Gscal ellect on state or local\n\n"), (True, u'E KTNH\n\n')]), [(0, 0), (1, 1)])], [((16, [(847, 394, 1233, 737)], [(False, u"IPOSES ADDITIONAL TAX ON\nANCER RESEARCH. |NI'I'IATIVE\nlditiunal $1.00 per pack tax on cigarettes anr\ne on other tobawo products. Revenues fun\n:cbecc0-related diseases. Fiscal Impact:\nexcise lax revenues of about $735 million\nzenain research and tobacco prevention and\nser state and local revenue increases\nons of dollars annually.\n\n"), (True, u'ES C) NO\n\n')]), [(0, 0), (1, 1)])], [((16, [(847, 768, 1233, 1075)], [(False, u'OMMUNITY SERVICES DISTRICT\nMEASURE E\nrdinsnoe Nc. 2012\xbb1, which imposes an\n-hundred ($200.00) dollars for \xa4 period of\nZ013, FY 2013-2014 and FY 2014-2015 and\n>sI within the Muir Beach CSD Ior tire\nabatement and emergency preparedness, b\nluir Beach CSD appropriations limit be\nof lhis voter-approved tax?\n\n'), (True, u"ES (Q') N0\n\n")]), [(0, 0), (1, 1)])], [((0, [(654, 177, 1243, 405)], [(False, u'\u2019ROPOSITION 28: LIMITS ON LEGlSLATORS\u2019 TERMS IN\nZIFFICE. INTTIATNE CONSTITUTIONAL AMENDMENT.\nleduoes talal amount of time a person may serve in the slate\nagislalure from 14 years to 12 years. Allows 12 years\u2018 service in one\nnouse Applies only lo legislators hrst elected alter measure is\nrassed. Fiscal Impact: No direct Hscal effect on state or local\nuwemments.\n\n'), (True, u''), (True, u'C) YES C) NO\n\n')]), [(0, 0), (1, 1), (2, 2)]), ((5, [(654, 474, 1243, 703)], [(False, u"'ROPOSITION 28: LIMITS ON LEGlSLATORS' TERMS IN\n)FFlCE. INITIATIVE CONSTITUTIONAL AMENDMENT.\nleduces total amount of time a person may serve in the state\nagisleture from 14 years to 12 years. Allows 12 years\u2019 service in one\niouse. Applies only lo legislators lirst elected eller measure is\niassed. Fiscal Impact: No direct liscal eflect on slate or local\ninvemments.\n\n"), (True, ''), (True, u'Q YES (D N0\n\n')]), [(0, 0), (1, 1)]), ((8, [(648, 174, 1233, 401)], [(False, u"\u2019ROPOSITlON 28: LIMITS ON LEGlSLATORS' TERMS IN\nDFFICE. INITIATIVE CONSTITUTIONAL AMENDMENT.\nQeduees total amount of time e person mey serve in the state\negisleture from 14 years to 12 years. Allows 12 ye\xa4rs\u2019 service in one\nmuse. Applies only to legislators Hrst elected after measure is\nyassed, Fiscal Impact: No direct Gscel effect on state or local\nwvernments.\n\n"), (True, ''), (True, u'Q YES Q NO\n\n')]), [(0, 0), (1, 1)]), ((9, [(647, 177, 1243, 406)], [(False, u'PROPOSITION 28: LIMITS ON LEGISLATORS\u2019 TERMS IN\nOFFICE. INITIATIVE CONSTITUTIONAL AMENDMENT.\nReduces total amount of time a person may serve in the state\nlegislature from 14 years to 12 years. Allows 12 years\u2018 service in one\nhouse. Applies only to legislators tirst elected alter measure is\npassed. Fiscal Impact; No direct liscal effect on state or local\ngovemments.\n\n'), (True, u''), (True, u'C) YES C) N0\n\n')]), [(0, 0), (1, 1)]), ((10, [(639, 165, 1233, 395)], [(False, u'PROPOSITION 28: LIMITS ON LEGISLATORS\u2019 TERMS IN\nOFFICE. INITIATIVE CONSTITUTIONAL AMENDMENT.\nReduces total amount of time a person may serve in the state\nlegislature from 14 years to 12 years Allows 12 years service in one\nhouse. Applies only to legislators frrst elected alter measure is\npassed. Fiscal Impact: No direct tiscal effect on state or local\ngovemments.\n\n'), (True, u''), (True, u'FW YES (_) NO\n\n')]), [(0, 0), (1, 1)]), ((12, [(646, 177, 1243, 405)], [(False, u'PROPOSITION 28: LIMITS ON LEGISLATORS\u2019 TERMS IN\nOFFICE. INITIATNE CONSTITUTIONAL AMENDMENT.\nReduces tolsl amount ol time a person may serve in the state\nlegislature from 14 years to 12 years. Allows 12 years\u2019 service in one\nhouse. Applies only to legislators lirst eleded after measure is\npassed. Fiscal Impact: No direct fiscal eflsct on state or local\ngovemments.\n\n'), (True, u''), (True, u'C) YES Q NO\n\n')]), [(0, 0), (1, 1)]), ((13, [(666, 177, 1243, 405)], [(False, u"ROPOSITION 28: LIMITS ON LEGlSLATORS' TERMS IN\nFFICE. INITIATIVE CONSTITUTIONAL AMENDMENT.\nsduces total amount of time a person may serve in the state\ngislature from 14 years to 12 years. Allows 12 years' service in one\n>use. Applies only to legislators Grsl elected after measure is\nassed. Fiscal Impact: No direct Escal elieot on slate or local\nyvemments.\n\n"), (True, u''), (True, u'C) YES C-) N0\n\n')]), [(0, 0), (1, 1)]), ((21, [(651, 174, 1233, 403)], [(False, u"\u2019ROPOSITION 28: LIMITS ON LEGISLATORS\u2019 TERMS IN\n)FFICE. INITIATIVE CONSTITUTIONAL AMENDMENT.\nieduoes total amount oi time a person may serve in the state\nagislature from 14 years to 12 years. Allows 12 yeers' service in cnr\nnouse. Applies only to legislators iirst elected alter measure as\nrassed. Fiscal Impact: No direct tiscal elfect on state or local\nlovemments.\n\n"), (True, ''), (True, u'C) YES C) NO\n\n')]), [(0, 0), (1, 1)])], [((0, [(654, 399, 1243, 740)], [(False, u"'ROPOSITION 29: IMPOSES ADDITIONAL TAX ON\nIIGARETTES FOR CANCER RESEARCH. INITIATIVE\nSTATUTE. imposes additional $1,00 per pack tax on cigarettes and\nln equivalent tax increase on other tobacco products. Revenues func\nesearch for cancer and tobacco-related diseases. Fiscal Impact:\nlet increase in cigarette excise tax revenues of about $735 million\nnnually by 2013-14 Ior certain research and tobacco prevention and\nessation programs. Other state and local revenue increases\nmounting to tens of millions of dollars annually.\n\n"), (True, u''), (True, u"('U YES ('> NO\n\n")]), [(0, 0), (1, 1), (2, 2)]), ((5, [(654, 697, 1243, 1041)], [(False, u"'ROPOSITION 29: IMPOSES ADDITIONAL TAX ON\nHGARETTES FOR CANCER RESEARCH. INITIATNE\nITATUTE. imposes additional $1.00 per pack tax on cigarettes and\nn equivalent tax increase on other tobacco products. Revenues func\nesearch for cancer and tobaccckrelated diseases. Fiscal Impact:\nlet increase in cigarette excise tax revenues of about $735 million\nnnually by 201314 for certain research and tobacco prevention and\nessation programs. Other state and local revenue increases\nmounting to tens of millions of dollars annually.\n\n"), (True, u''), (True, u'Q YES Q NO\n\n')]), [(0, 0), (1, 1)]), ((8, [(648, 395, 1233, 737)], [(False, u'\u2019ROPOSITl0N 29: IMPOSES ADDITIONAL TAX ON\nDIGARETTES FOR CANCER RESEARCH. INITIATIVE\nSTATUTE. Impcses additional $1.00 per pack tax on cigarettes ant\n1n equivalent tax increase on other tobaoco products. Revenues lun\nesearch for cenoer and tohacotrreleted diseases. Fiscal Impact:\nlet increase in cigarette excise tax revenues oi about $735 million\nannually by 2013-14 lor certain research and tobacco prevention and\n>essation programs. Other state and local revenue increases\namounting to tens of millions oi dollars annually.\n\n'), (True, ''), (True, u'C) YES Q N0\n\n')]), [(0, 0), (1, 1)]), ((9, [(647, 400, 1243, 740)], [(False, u'PROPOSITION 29: IMPOSES ADDITIONAL TAX ON\nCIGARETTES FOR CANCER RESEARCH. INITIATIVE\nSTATUTE. imposes additional $1.00 per pack tax on cigarettes anc\nan equivalent tax increase on other tobacco products. Revenues funn\nresearch tor cancer and tobacco-related diseases. Fiscal Impact:\nNet increase in cigarette excise tax revenues of about $735 million\nannually hy 2013-14 tor certain research and tobacco prevention and\ncessation programs. Other state and local revenue increases\namounting to tens ol millions of dollars annually.\n\n'), (True, u''), (True, u'(7 YES (_`\xbb NO\n\n')]), [(0, 0), (1, 1)]), ((10, [(639, 399, 1233, 734)], [(False, u'PROPOSITION Z9: IMPOSES ADDITIONAL TAX ON\nCIGARETTES FOR CANCER RESEARCH. INITIATIVE\nSTATUTE. imposes additional $1.00 per pack lax on cigarettes anc\nan equivalent lax increase on other tobacco products. Revenues furv\nresearch lor cancer and tobacco-related diseases. Fiscal Impact:\nNet increase in cigarette excise tax revenues ol about $735 million\nannually by 2013\u201414 for certain research and tobacco prevention and\ncessation programs. Other state and local revenue increases\namounting to tens of millions ol dollars annually.\n\n'), (True, u''), (True, u'(W YES L) Nu\n\n')]), [(0, 0), (1, 1)]), ((12, [(646, 399, 1243, 737)], [(False, u"PROPOSITION 29: IMPOSES ADDITIONAL TAX ON\nClGARE`I'TES FOR CANCER RESEARCH. INITIATIVE\nSTATUTE. imposes additional $1.00 per pack lax on cigarettes and\nan equivalent lax increase on other tobacco products. Revenues luni\nresearch for cancer and tobacco-related diseases. Fisml Impact:\nNet increase in cigarette exdse tax revenues of about $735 million\nannually by 2013-14 for certain research and tobacco prevention and\ncessation progmms. Other slate and local revenue increases\namounting to tens ol millions of dollars annually.\n\n"), (True, u''), (True, u'(7 YES (_1 NO\n\n')]), [(0, 0), (1, 1)]), ((13, [(666, 399, 1243, 737)], [(False, u'ROPOSITION 29: IMPOSES ADDITIONAL TAX ON\nIGARETTES FOR CANCER RESEARCH. INITIATIVE\nTATUTE. imposes additional $1.00 per pack tax on cigarettes anc\ni equivalent tax increase on other tobacco products. Revenues funn\nsearch for cancer and tobacco-related diseases. Fiscal Impact:\nat increase in cigarelte exdse lax revenuu cf about $735 million\ninually hy 2013-14 for certain research and tobacco prevention and\nlssaticn programs. Other state and local revenue increases\nnounting to tens of millions 01 dollars annually.\n\n'), (True, u''), (True, u'(-3 YES f? ND\n\n')]), [(0, 0), (1, 1)]), ((21, [(651, 397, 1233, 739)], [(False, u'\u2019ROPOSITl0N 29: IMPOSES ADDITIONAL TAX ON\nZIGARETTES FOR CANCER RESEARCH. INITIATIVE\nBTATUTE. imposes additional $1.00 per pack tax on cigarettes ani\nin equivalent tax increase on other tobacco products. Revenues fun\nesearch tor cancer and tobacco-related diseases. Fiscal Impact:\nJet inuease in cigarette excise tax revenues of about $735 million\ninnually by 2013-14 lor certain research and tobaooo prevention and\nzessation programs. Other state and locel revenue increases\nimounting to tens ol millions ol dollars annually.\n\n'), (True, ''), (True, u'C) YES Q NO\n\n')]), [(0, 0), (1, 1)])], [((4, [(633, 774, 1233, 1071)], [(False, u'MUIR BEACH COMMUNITY SERVICES DISTRICT\nMEASURE E\nShall Muir Beach CSD Ordinance No. 2012-1, which imposes an\nannual spedal tax of two\xb7huodrsd ($200.00) dollars for a period of\nfour (4) years, FV 2012-2013, FY 2013-2014 and FY 2014-2015 and\n2015-2016, on each parcel within the Muir Beach CSD lor fire\nprotection, including iuel abatement and emergency preparedness, lz\napproved and shall the Muir Beach CSD appropriations limit be\nincreased by the amount of this voter-approved tax?\n\n'), (True, u''), (True, u'I (7 YES C) N0\n\n')]), [(0, 0), (1, 1), (2, 2)]), ((10, [(648, 774, 1233, 1071)], [(False, u'MUIR BEACH COMMUNITY SERVICES DISTRICT\nMEASURE E\nShall Muir Beach CSD Ordinance N0. 2012-1, which imposes an\nannual special tax ol two-hundred ($200.00) dollars lor a period ul\nbur (4) years, FY 2012-2013, FY 2013-2014 and FY 2014-2015 and\nE015-2016, on each parcel within the Muir Beech CSD lor tire\narotection. including fuel abatement end emergency preparedness, lz\napproved and shall the Muir Beach CSD appropriations limit be\nhcreased by the amount ol this vetenapprcved tax?\n\n'), (True, u''), (True, u'(W YES C) NU\n\n')]), [(0, 0), (1, 1)])], [((5, [(654, 1071, 1243, 1372)], [(False, u"TOWN OF ROSS MEASURE C\nhall the voters of the Town of Ross adopt an ordinance authorizing\nom July 1, 2012 through June 30. 2016, the levy ofa special lax for\nublic safety services in an amount not to exceed $1000 per dwelling\nnil for single family residential uses and not lo exceed $1000 per\narcel for multi-family. commercial or other non\xb7residential uses, and\nicreesing the Tovm's epproprialions limit by the amount of the\npecial lax proceeds?\n\n"), (True, u''), (True, u'(7 YES {-5 NO\n\n')]), [(0, 0), (1, 1), (2, 2)])], [((11, [(67, 1006, 632, 1105)], [(False, u'\n'), (True, u'D GARY JOHNSON LIE\n\n'), (True, u'T R. J. HARRIS LIE\n\n')]), [(0, 0), (1, 1), (2, 2)])], [((21, [(640, 771, 1233, 1038)], [(False, u'ROSS VALLEY SCHOOL DISTRICT MEASURE A\nTo provide local funding the State cannot take away, preserve high\nquality education in reading, writing, math and science, educationally\nsound class sizes. school libraries, and art and music instruction, an:\ntc help attract and retain highly\xbbqua|ified teachers, shall the Ross\nValley School Distriu renew its existing parcel tax for another eight\nyears, increasing it by $149 per year, with no lunds used for\nadministrators salaries and an exemption for seniors, and requiring\nannual audits?\n\n'), (True, ''), (True, u'Q YES Q NO\n\n')]), [(0, 0), (1, 1), (2, 2)])], [((0, [(54, 1188, 633, 1375)], [(False, u'Judge of the Superior Court, Ofilcs N0. 3\nVote for One (1)\n\n'), (True, u'Mnrln County Superior Court Ju\xa4g\u2022\n\n'), (True, u'\xa9 RUSSELL K. MARNE\nAl\\\xa4m|Y AI Law\n\n'), (True, u' \n\n')]), [(0, 0), (1, 1), (2, 2), (3, 3)]), ((1, [(51, 1182, 639, 1377)], [(False, u'Judge of the Superior Court, Offlce No. 3\nVote lor One (1)\n\n'), (True, u'Mnrln County Supirlur C\xa4u\u2022\u2018\\ Judy!\n\n'), (True, u'Atlurnuy AI Law\n\n'), (True, u' \n\n')]), [(2, 2), (0, 0), (1, 1)]), ((4, [(51, 1182, 648, 1377)], [(False, u'Judge or the Superior Court, Ofllce N0. 3\nVote for One (1)\n\n'), (True, u'Marin Ccuuly Sup\u2022rl\xa4r C\xa4uI\\ Judy!\n\n'), (True, u'Attornny Al Liv:\n\n'), (True, u' \n\n')]), [(2, 2), (0, 0), (1, 1)]), ((5, [(51, 1182, 633, 1372)], [(False, u'Judge of the Superior Court, Oflico N0. 3\nVote for Ons {1)\n\n'), (True, u'O JAMES CHOU\nMnrin Couniy Supirlcr Chun Judgt\n\n'), (True, u'O RUSSELL K. MARNE\nA\\1\xa4rv\\\u2022Y At Lkw\n\n'), (True, u'( \n\n')]), [(2, 2), (1, 1), (0, 0)]), ((8, [(48, 1182, 645, 1376)], [(False, u'Judge of the Superior Court, Ofiice N0. 3\nVots for One (1)\n\n'), (True, u'Marin County Su9\u2022r|\xa4r Cuurl Judg\u2022\n\n'), (True, u'Attcrvli At uw\n\n'), (True, u' \n\n')]), [(2, 2), (0, 0), (1, 1)]), ((9, [(47, 1185, 618, 1376)], [(False, u'Judge of the Superior Court, Office N0. 3\nVote for One (1)\n\n'), (True, u'Marin County Sup\u2022rlor Coun Judg\u2022\n\n'), (True, u'O RUSSELL K. MARNE\nAttorney A! Law\n\n'), (True, u' \n\n')]), [(2, 2), (0, 0), (1, 1)]), ((10, [(51, 1185, 648, 1378)], [(False, u'Judge of the Superior Court, Offlce N0. 3\nVote Iur One (1)\n\n'), (True, u'Mlrln County Sup\u2022r\\\xa4r Coun Judgn\n\n'), (True, u'A\\t\xa4m\u2022y AI \\;w\n\n'), (True, u' \n\n')]), [(2, 2), (0, 0), (1, 1)]), ((12, [(46, 1182, 633, 1375)], [(False, u'Judge of the Suporlcr Court, Office No. 3\nVuto 1\xa4r Ons (1)\n\n'), (True, u'Mnrln Ccunty Sup\u2022rI\xa4r Court Judga\n\n'), (True, u'O RUSSELL K. IIARNE\nAk\\\xa4rI\\\u2022y Al Liv\n\n'), (True, u' \n\n')]), [(2, 2), (0, 0), (1, 1)]), ((13, [(48, 1182, 633, 1375)], [(False, u'Judge of the Supodcr Court, Ofilcs Nc. 3\nVote for Ona (1)\n\n'), (True, u'O JAMES CHOU\nMarin Counly $up\u2022fI0f Court Judga\n\n'), (True, u'G RUSSELL K, MARNE\nAltorniy Al LIU\n\n'), (True, u' \n\n')]), [(2, 2), (1, 1), (0, 0)]), ((17, [(648, 1635, 1243, 1828)], [(False, u'Judge of the Supsrlor Court, Oftica Nc. 3\nVote lor Ona (1)\n\n'), (True, u'Mlrln C\xa4un|Y Sup\u2022r|cr Gout! .|ud\xa7\u2022\n\n'), (True, u'Anamay A! Lnw\n\n'), (True, u' \n\n')]), [(2, 2), (0, 0), (1, 1)]), ((20, [(645, 1629, 1241, 1822)], [(False, u'Judge of the Superior Court, Office Nc. 3\nVote for Ons (1)\n\n'), (True, u'Mlrln County Sup\u2022rlor C\xa4ur\\ Judgt\n\n'), (True, u'Atlornty Al Llw\n\n'), (True, u' \n\n')]), [(2, 2), (0, 0), (1, 1)]), ((21, [(48, 1182, 633, 1376)], [(False, u'Judge of the Superior Court, Office No. 3\nVote for Ona (1)\n\n'), (True, u'llarin County Suparlcr Coun Ju\xa4\xa4\u2022\n\n'), (True, u"O RUSSELL K. MARNE\nAX\\\xa4l'l\\\u2022y AI Llw\n\n"), (True, u' \n\n')]), [(2, 2), (1, 1), (0, 0)])], [((0, [(54, 1416, 633, 1602)], [(False, u'County Supervisor, District 4\nVote for One (1)\n\n'), (True, u'Incumb\u2022M\n\n'), (True, u'C\xa4r\\\u2022 Mauna Vlc\u2022 Mnyor\n\n'), (True, u' \n\n')]), [(0, 0), (1, 1), (2, 2), (3, 3)]), ((4, [(51, 1407, 648, 1603)], [(False, u'\xe9nunty Supervlsor, Dlstrlct 4\nVote lor Ons (1)\n\n'), (True, u'Incumhlrll\n\n'), (True, u'Curl! Iladtrt V|\xa4\u2022 Mayor\n\n'), (True, u' \n\n')]), [(2, 2), (0, 0), (1, 1)]), ((8, [(48, 1407, 645, 1601)], [(False, u'County Suporvlsor, District 4\nVote for Ons (1)\n\n'), (True, u'Incumhint\n\n'), (True, u'C\xa4n\u2022 Madnra VI\xa4\u2022 Mlyor\n\n'), (True, u' \n\n')]), [(2, 2), (0, 0), (1, 1)]), ((10, [(51, 1410, 648, 1604)], [(False, u'County Supsrvlsor, Distrlct 4\nVote lor One (1)\n\n'), (True, u'Incumb\u2022m\n\n'), (True, u'Corn Mldara Vlct Mnyol\n\n'), (True, u' \n\n')]), [(2, 2), (0, 0), (1, 1)]), ((13, [(48, 1407, 633, 1600)], [(False, u'County Supervisor, District 4\nVote for One (1)\n\n'), (True, u'Incumb\u2022nt\n\n'), (True, u'O DIANE FURST\nCNM |I|d\u2022\xa4 Vlc\u2022 Iliyor\n\n'), (True, u' \n\n')]), [(2, 2), (0, 0), (1, 1)]), ((16, [(45, 141, 641, 327)], [(False, u'County Supervlsor, D|strict 4\nVote lor Ona (1)\n\n'), (True, u'Incumblllt\n\n'), (True, u'Cana Msden VI\xa4\u2022 Mlyu\n\n'), (True, u' \n\n')]), [(2, 2), (0, 0), (1, 1)])], [((5, [(51, 1668, 633, 1938)], [(False, u'TOWN OF ROSS\nMambor, Town Council\nVoto for no m\xa4r\u2022 than Thr\u2022\u2022 (3)\nNo Candldate Has Fllsd\n\n'), (True, u'\n'), (True, u'\n'), (True, u'\n')]), [(0, 0), (1, 1), (2, 2), (3, 3)])], [((1, [(51, 1407, 639, 1640)], [(False, u'County Supervisor, Dlstrlct 2\nVote for Ons (1)\n\n'), (True, u'Fllriix \xa20ur\\c|Im\u2022mb\u2022rIAI1\xa4m\u2022y\n\n'), (True, u'Educalurlldmlnlttrltor\n\n'), (True, u'Marin County $up\u2022rvI1\xa4r\n\n'), (True, u' \n\n')]), [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]), ((5, [(51, 1413, 633, 1636)], [(False, u'County Supervisor, District 2\nVote for One (1)\n\n'), (True, u'Fnlrhx C\xa4unclIrn\u2022mb\u2022rlA\\t\xa4rI\u2022\u2022y\n\n'), (True, u'EducIt\xa4rIAdmInIIIrn\\\xa4r\n\n'), (True, u'Marin County 8up\u2022rvI|or\n\n'), (True, u' \n\n')]), [(3, 3), (2, 2), (1, 1), (0, 0)]), ((9, [(47, 1407, 618, 1639)], [(False, u'County Suparvlsor, District Z\nVots for Ons (1)\n\n'), (True, u'Fnltflx C0unc|Im|mb|rIA\\t\xa4rn\u2022y\n\n'), (True, u'B EVA LONG\nEducalcdhdmlnlstmtcr\n\n'), (True, u'O KATIE RICE\nMlrln Cuunly Sup\u2022rvI\xa4\xa4r\n\n'), (True, u' \n\n')]), [(3, 3), (0, 0), (1, 1), (2, 2)]), ((21, [(48, 1407, 633, 1639)], [(False, u'County Sup\u2022rv|\u20220r. District 2\nVote for Ona (1)\n\n'), (True, u'Fnlrhx CounclIm\u2022mh\u2022rIA|1\xa4rn\u2022y\n\n'), (True, u'Eduo|l\xa4rIA\xa4mIn(|\\r\xa4I\xa4r\n\n'), (True, u'Marin County Suparvlsnr\n\n'), (True, u'C)- \n\n')]), [(3, 3), (2, 2), (0, 0), (1, 1)])], [((1, [(642, 165, 1233, 734)], [(False, u'\u2019ROPO$ITION 28: LIMITS ON LEGISLATORS\u2019 TERMS IN\nDFFICE. INITIATIVE CONSTITUTIONAL AMENDMENT.\nReduces tolal amount ol time a person may serve in the state\negislalure from 14 years to 12 years. Allows 12 years\u2018 service in one\nmuse. Applies only to legislators lirst elected aller measure is\n>assed. Fiscal Impact: No direct Gscal ellect on slate or local\n;0vemments,\n\n'), (True, u''), (True, u'Q) YES C) N0\n \nPROPOSITION 29: IMPOSES ADDITIONAL TAX ON\nCIGARETTES FOR CANCER RESEARCH. INITIATIVE\nSTATUTE. imposes additional $1.00 per pack tax on cigarettes anc\nan equivalent tax increase on other tobacco products. Revenues fum\n\xb7esearch for cancer and tobacoerelated diseases. Fiscal Impact:\nNet increase in cigarette excise tax revenues of about $735 million\nannually by 2013-14 for oenain research and tobacco prevention and\ncessation programs. Other state and local revenue increases\namounting to lens ol millions of dollars annually.\n\n'), (True, u''), (True, u"f') YES (D N0\n\n")]), [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]), ((4, [(633, 165, 1233, 733)], [(False, u"PROPOSITION 28: LIMITS ON LEGISLATORS\u2019 TERMS IN\nOFFICE. INITIATNE CONSTITUTIONAL AMENDMENT.\nReduces total amount of time a person may serve in the state\nlegislature from 14 years lo 12 years. Allows 12 years' service in one\nhouse. Applies only to legislators first elected alter measure is\npassed. Fiscal Impact: No direct fiscal eflect un slate ur local\ngovemmanls.\n\n"), (True, u''), (True, u'Q YES Q NU\n \nPROPOSITION 29: IMPOSES ADDITIONAL TAX ON\nCIGARETTES FOR CANCER RESEARCH. INITIATIVE\nSTATUTE. imposes additional $1.00 per pack tax on cigarettes ant\nan equivalent tax increase on other tobacco products. Revenues funt\nresearch lor cancer and tobacco-related diseases. Fiscal Impact:\nNet increase in cigarette excise tax revenues ol about $735 million\nannually by 2013-14 for certain research and tobaccxz prevention end\ncessation programs. Other state and local revenue increases\namounting to tens of millions ol dollars annually.\n\n'), (True, u''), (True, u"('3 YES Q) Nu\n\n")]), [(0, 0), (2, 2), (3, 3), (1, 1)])], [((2, [(39, 735, 639, 967)], [(False, u'American Independent Party Candidates\nVote for One (1)\n\n'), (True, u'O EDWARD C. NUUNAN AI\n\n'), (True, u'O MAD MAX RIEKSE AI\n\n'), (True, u'O LAURIE RUTH AI\n\n'), (True, u'@ \n\n')]), [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)])], [((6, [(39, 735, 640, 964)], [(False, u'Green Party Candidates\nVote for One (1)\n\n'), (True, u'O .II|.L 51:IN GRN\n\n'), (True, u'O KENT NIESFLAY GRN\n\n'), (True, u'O ROSEANNE BARR GRN\n\n'), (True, u' \n\n')]), [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)])], [((7, [(69, 774, 638, 964)], [(False, u'\n'), (True, u'D :U\\NARD C. NUUNAN AI\n\n'), (True, u'\n'), (True, u'D LAURIE RUTH AI\n\n'), (True, u';) \n\n')]), [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)])], [((11, [(67, 819, 623, 992)], [(False, u'D LEE WRIGHTS L\nD ROGER GARY L\nD JAMES OGLE L\n\n'), (True, u'\n'), (True, u'\n'), (True, u'D JAMES OGLE L\n\n'), (True, u'\u2019-\\ SCOTT KELLER \\\n\n')]), [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)])], [((12, [(46, 1407, 633, 1634)], [(False, u'County Supervisor, District 2\nVots For One (1)\n\n'), (True, u'G DAVID VIIEINSOFF\nFnlrhx C0un\xa4||n\u2022\u2022mb\u2022rIAk!0rnOy\n\n'), (True, u'3 EVA LONG\nEducatcrihdmlnlllrllvf\n\n'), (True, u'Harln Caonty Sup\u2022rvl\u2022\xa4r\n\n'), (True, u'( \n\n')]), [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)])], [((14, [(39, 735, 640, 965)], [(False, u'Peace and Freedom Party Candidates\nVote for One (1)\n\n'), (True, u'O ROSS C. "ROCKY" ANDERSON PF\n\n'), (True, u'O STEWART AL|:XAN|J\\:R PF\n\n'), (True, u'Q STEPHEN DURHAM Pk\n\n'), (True, u' \n\n')]), [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)])], [((18, [(70, 810, 640, 963)], [(False, u''), (True, u'D EDWARD C. NUUNAN AI\n\n'), (True, u'D MAD MAX RIEKSE AI\n\n'), (True, u'D LAURIE ROTH AI\n\n'), (True, u';) \n\n')]), [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)])], [((0, [(54, 1641, 633, 1904)], [(False, u'Ross Valley Sanhary Dlstrict No. 1, Diractor\nVote for no more than Two (2)\n\n'), (True, u'O MARY SVLLA\nNonprofll Dlr\u2022c\\0rIAIi\xa4n\u2018\\\u2022y\n\n'), (True, u'Q NIARCIA A. JOHNSON\nlm:umb\u2022nl\n\n'), (True, u'O FRANK EGGER\nDlr\u2022c1or, Rus! VaI|\u2022y Sanlhry Dlrlrid\n\n'), (True, u' \n\n'), (True, u' \n\n')]), [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]), ((5, [(643, 132, 1243, 404)], [(False, u'Ross Valley Sanitary Dlstrlct N0. 1, Directtgr\nVote for nc mom than Two (2)\n\n'), (True, u'Nonprvfll DIr\u2022ci\xa4rIAkt\xa4rn\u2022y\n\n'), (True, u'O MARCIA A, JOHNSON\nIncumhlnl\n\n'), (True, u'Dlriciuf. Rell: Vnllry Sanitary Dlnlrllzt\n\n'), (True, u' \n\n'), (True, u' \n\n')]), [(3, 3), (4, 4), (1, 1), (0, 0), (2, 2)]), ((9, [(47, 1671, 618, 1937)], [(False, u'Russ Valley Sanitary District No. 1, Director\nVote for no more than Two (2)\n\n'), (True, u'Q MARY SVLLA\nNcnprwfll Dlr\u2022c1\xa4rIAt\\\xa4rn\u2022y\n\n'), (True, u'O NIARCIA A, JOHNSON\nIncumbam\n\n'), (True, u'O FRANK EGGER\nDlmcinr, R0\u2022| VIII\u2022y Slnlury Dlnrict\n\n'), (True, u' \n\n'), (True, u' \n\n')]), [(3, 3), (4, 4), (1, 1), (0, 0), (2, 2)]), ((12, [(46, 1674, 633, 1936)], [(False, u'Ross Valley Sanitary District No. 1, Director\nVote for no mnro than Two (2)\n\n'), (True, u'O MARY SYLLA\nNonpfuflt Dlnctorlhllorndy\n\n'), (True, u"O MARCIA A. JOHNSON\nIncul'Ilb\u2022rvt\n\n"), (True, u'G FRANK EGGER\nDlmdor. Ross Vlllty Sinllnry Dlslrid\n\n'), (True, u' \n\n'), (True, u'( \n\n')]), [(3, 3), (4, 4), (1, 1), (2, 2), (0, 0)]), ((13, [(48, 1632, 633, 1898)], [(False, u'Ross Valley Sanltary Dlstrlct N0. 1, Dlrector\nVote for no more than Two (Z)\n\n'), (True, u'O MARY SYLLA\nNonprvfh D|r\u2022d\xa4rIAtt\xa4rn\u2022y\n\n'), (True, u'O IIARCIA A. JOHNSON\nIncumbtnl\n\n'), (True, u'O FRANK EGGER\nDlnctuv. Rott Vallty Snnlhry Dlslrlct\n\n'), (True, u' \n\n'), (True, u' \n\n')]), [(3, 3), (4, 4), (1, 1), (0, 0), (2, 2)])], [((3, [(39, 735, 638, 1077)], [(False, u'Republican Party Candidates\nVute for Ona (1)\n\n'), (True, u'\n'), (True, u'G RICK SAN IUKUNI HEP\n\n'), (True, u'O RUN PAUL NEP\n\n'), (True, u'O NEVTT GINGRICH REP\n\n'), (True, u'Q MITT RUMNEY REP\n\n'), (True, u'O CHARLES E, "BUDDY" ROENIER, III REP\n\n'), (True, u' \u2014\n\n')]), [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)]), ((19, [(42, 735, 639, 1077)], [(False, u'Republican Party Candidates\nVols for One (1)\n\n'), (True, u'O FRED KARGER REP\n\n'), (True, u'O RICK SANIURUM REP\n\n'), (True, u'O RON PAUL REP\n\n'), (True, u'O NEWT GINGRICI-I REP\n\n'), (True, u'\xa9 MITT ROMNEY REP\n\n'), (True, u'O CHARLES E. "BUDDY" ROEMER. III REP\n\n'), (True, u'@ \n\n')]), [(6, 6), (2, 2), (4, 4), (5, 5), (3, 3), (1, 1), (0, 0)]), ((22, [(39, 735, 639, 1075)], [(False, u'Republican Party Candidates\nVute fur One (1)\n\n'), (True, u'G I-Htl) KAKti|:R HEP\n\n'), (True, u'C) RICK SANIURUIVI REF\n\n'), (True, u'\xa9 RON PAUL NEP\n\n'), (True, u'O NEWT GINGRICH REP\n\n'), (True, u'Q MII I RUMN|:Y REP\n\n'), (True, u'\n'), (True, u' \n\n')]), [(0, 5), (6, 6), (3, 3), (2, 2), (4, 4), (1, 1), (5, 0)])], [((0, [(54, 771, 633, 1149)], [(False, u'Candidates to tho Assembly, 10th District\nVots for Ons (1)\n\n'), (True, u'C) H. CHRISTIAN GUNDERSON Par\\yPreference; DE\nChlmpmctlc D0c\\\xa4rIEntr\u2022pnn\u2022ur\n\n'), (True, u'O ALEX EASTON-BROWN Pam Pygleyemg; BE\nPtnslnn Rlkwm Coofdlnltbr\n\n'), (True, u'O unc:-\u2022AEL ALLEN\n{\u2022ym\xa4\u2022!\xa4n\u2022m\xa4\u2022nAn\xa4n\u2022\u2022y *\u2019\xa4\xa4y Prelemncez DE\n\n'), (True, u'O CONNIE VVONG pany Pygjgmme; DE\nM\xa2*lh\u2022rllIIIIt|ry Offlur\n\n'), (True, u'G PETER J. MANCUS PBR)! PIBFBIBVICS. RE\nSmall Eusinors Ownnr\n\n'), (True, u'O MARC LEVINE Pam PIGWSVSHCEZ DE\nC\xa4uncIIm\u2022mb\u2022V. Clty 01San Rnfnl\n\n'), (True, u'JOE BOSWELL P PM; ;\nO Small Buslnns Parson any rm N0,\n\n'), (True, u' \n\n')]), [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)]), ((12, [(46, 768, 633, 1150)], [(False, u'Candidates to the Assembly, 10th Dlstrlct\nVols for Ons (1)\n\n'), (True, u'O H. CHRISTIAN GUNDERSON Pany Pyqfgygngg; DE\nChlmpmcllc D0\xa2torIEn!r\u2022prIn\u2022ur\n\n'), (True, u'O ALEX EASTON-BROWN Pzliy PVBWMEIMJ DE\nPtnllon Rnforln Coordlnltbr\n\n'), (True, u'O MICHAEL ALLEN Party Pfeklnmxi DE\nA||\u2022mb|ym\u2022mb\u2022rIAt\\0rn\u2022ry\n\n'), (True, u'O CONNIE WONG Plny Pf0bfBI\\0e: DE\nMmhsrllllllhry Offlclt\n\n'), (True, u'O PETER J. MANCUS Party Prufuvnllcsi RE\nSmall Busln\u2022s\u2022 Ownar\n\n'), (True, u'O MARC LEVINE Party PVMSFSDCQI DE\nCcuncllrntmbtr. Clty cl Sin Rafatl\n\n'), (True, u'O JOE BOSWELL Party Pm1\xa4r\xa2n0\xa4:N\xa4<\nSmall Buslnux F\u2022non\n\n'), (True, u' \n\n')]), [(7, 7), (0, 0), (4, 4), (1, 1), (5, 5), (3, 3), (6, 6), (2, 2)])], [((1, [(42, 774, 639, 1152)], [(False, u'Candidates to the Assembly, 10th District\nVote for Ona (1)\n\n'), (True, u'Chlmpmctic DOG\\\xa4rIEn!r\u2022pr\u2022n\u2022ur\n\n'), (True, u'P\u2022n|I\xa4n Rifcfm Coordlnalcr\n\n'), (True, u'As|\u2022mh|ym\u2022mb\u2022r1At1\xa4n\xbb\u2022y\n\n'), (True, u'M\xa4!h\u2022rIMllItIry OHIc\u2022r\n\n'), (True, u'Smnll Businttr Ownnr\n\n'), (True, u'Cauncllmtmbar, Clty M San R1|n\u2022I\n\n'), (True, u'Small Busintin P\u2022\xa4on\n\n'), (True, u'( \n\n')]), [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)]), ((4, [(51, 774, 648, 1152)], [(False, u'Candidatsn to the Assombly, 10th District\nVote for Ons (1)\n\n'), (True, u'Chlropractlc D\u2022d\xa4rIEnlr\u2022pnn\u2022ur\n\n'), (True, u'Puulun Rdcrm Cocrdlrlltcr\n\n'), (True, u'A||\u2022mbIyn\\\u2022mb\u2022rIAl(\xa4n\\\u2022y\n\n'), (True, u'Md\\h\u2022flHl|"l|\u2019Y Offlatr\n\n'), (True, u'Small Buslntu Ownlr\n\n'), (True, u'C\xa4um:IIm\u2022mb\u2022r. Clty of San Rahul\n\n'), (True, u'Small Bunlmn P\u2022n\xa4n\n\n'), (True, u' \n\n')]), [(7, 7), (4, 4), (6, 6), (2, 2), (0, 0), (1, 1), (5, 5), (3, 3)]), ((10, [(51, 774, 648, 1153)], [(False, u'Candidates tothe Assembly, 10th District\nVcts for One (1)\n\n'), (True, u'Chlrnnrldlc D\xa4\xa21\xa4rIEn!r\u2022pr\u2022n\u2022ur\n\n'), (True, u'Panslon Rnlnrm Coordinator\n\n'), (True, u'A||OfI\\hIyrI\\\u2022mb\u2022rIAtl\xa4rv\\|y\n\n'), (True, u'Ikthtrhllllury Oflclr\n\n'), (True, u'Small Bullrhll Owrlir\n\n'), (True, u'C\xa4um:lIm\u2022mb\u2022r, Clly of San Rafal\n\n'), (True, u'Small Butlmsi P\u2022n0n\n\n'), (True, u' \n\n')]), [(7, 7), (6, 6), (0, 0), (4, 4), (1, 1), (3, 3), (5, 5), (2, 2)])], [((5, [(51, 771, 633, 1152)], [(False, u'Candldatss to the Assembly, 10th Dlstrlct\nVote for One (1)\n\n'), (True, u"O H. CHRISTIAN GUNDERSON Party Pygiggrggg; DE\nCh|l\u2019\xa4pfI\xa4\xa2l\xa4 D0ct\xa4fIErlIl'|pr\u2022n\u2022uf\n\n"), (True, u'O ALEX EASTON-BROWN Pgny Pggh\xa2pm;g\xb7 DE\nPnnslon Rhfuim Coofdlnltor\n\n'), (True, u'Q MICHAEL ALLEN Pgny Pmmywe, DE\nA.$|4f!\\b|ylI\\\u2022rIlb\u2022rlAI(0m|y\n\n'), (True, u'C) CONNIE WONG Pam Pygmgmg; DE\nMothtrllllllary 0*NIc\u2022r\n\n'), (True, u'O PETER J. NIANCUS Party Prshroncoi RE\nSmall Bu\u2022In\u2022\xa4s Owner\n\n'), (True, u'O MARC LEVINE Paw Ppygygme DE\nC\xa4uncIlm\u2022mb\u2022l. Clly of San Rafatl\n\n'), (True, u'G Small Buslrwn Parson any\n\n'), (True, u' \n\n')]), [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)])], [((8, [(48, 768, 645, 1152)], [(False, u'Cnndldatns to the Assembly, 10th District\nVon for Ons (1)\n\n'), (True, u'O H, CHRISTIAN GUNDERSUN Parly PYQISFSDOBI UCM\nChlrnprlctlc D\xa4c\\\xa4rIEntr\u2022pr\u2022n\u2022ur\n\n'), (True, u'P\u2022n|I\xa4n Rdofm Caordlnltnr\n\n'), (True, u'O MICHAEL ALLEN Pafly PIBBNMQ2 ULM\nA|I\u2022mbIym\u2022mb\u2022r1AI\\\xa4rI\\Iy\n\n'), (True, u'Moihnrlllllllary 0fI\u2018\\c|r\n\n'), (True, u'Small Bu|In\u2022s| Ownlr\n\n'), (True, u'S MARC LEVINE Party Plevsrencei Ut:N\nC\xa4uncIlm\u2022mb\u2022r. Clty 01 San Rlfnl\n\n'), (True, u'Small Buslrnn Parson\n\n'), (True, u' \n\n')]), [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)])], [((9, [(47, 771, 633, 1151)], [(False, u'Candidates to tha Assembly, 10th District\nVote for Ono (1)\n\n'), (True, u'Q H. CHRISTIAN GUNDERSON Pafly PISYBVBIHDE DK\nChlrcpricllc D0cl\xa4rIEnIr\xa4pi\u2022rl\u2022ur\n\n'), (True, u'\xa9 ALEX EASTON-BROWN ParYyPref\xa41&0c2t DE\nPanslon R\u2022fMm Coordinator\n\n'), (True, u'O mcn-1AEL ALLEN Pam; Pmmenoez Dt\nAu\u2022mn\u2022ym\u2022m\xa4\u2022nAn\xa4\u2022n\u2022y\n\n'), (True, u'O CONNIE VIIONB Pafly PIGBVQHCEZ DE\nM\xa4ih\u2022rIMlIII.Iry OfYIc\u2022r\n\n'), (True, u'O PETER J. MANCUS Pa\xa4y Preference: Rl\nSmall Buslnun Owrmr\n\n'), (True, u'O MARC LEVINE Par\\yPre1e1ence: DE\nCoumzllmsmlur. Clty cl Sun Ra|\xa4\u2022I\n\n'), (True, u'Q JUE BOSWELL Pam Prefsrencei ND\nSmall Buslnass P\u2022m\xa4n\n\n'), (True, u' \n\n')]), [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)]), ((21, [(48, 771, 646, 1152)], [(False, u'Candldatn to the Asumbly, 10th District\nVote for Ono (1)\n\n'), (True, u'G H. CHRISTIAN GUNDERSON Par\\yPvE1etem>2; DEM\nChlrupmciic DodorIEn\\r\u2022pr\u2022n\u2022ur\n\n'), (True, u'O ALEX EASTONBROWN PaRyPm1\xa4f\xa4I\\\xa4\xa4Z DEM\nPinllcrl Rhfcrm Cootdlnltur\n\n'), (True, u'3 MICHAEL ALLEN Paky PlB4sfBf\\08Z DEM\nA|\u2022\u2022mbIym\u2022mb\u2022rIAI\\\xa4m\u2022y\n\n'), (True, u'\xa9 CONNIE WONG PaIYy Pmumcei DEM\nMmhsrlllllhry Offlur\n\n'), (True, u'O PETER J. MANCUS Parly Pleinmlliei REP\nSmlll Bullrwtl Ownar\n\n'), (True, u'C) MARC LEVINE Pany Prslelencet DEM\nC\xa4um:IIm\u2022mb\u2022r, Clty af San R|f\xa4\u2022I\n\n'), (True, u'O JOE BOSWELL Parly PfG\xa4eI\u20acI\\G8ZN\xa4I\\e\nSmall Buslnan Futon\n\n'), (True, u' \n\n')]), [(7, 7), (4, 4), (5, 5), (6, 6), (1, 1), (0, 0), (3, 3), (2, 2)])], [((13, [(48, 774, 654, 1150)], [(False, u'Candidates to \u2022h\u2022 Assembly, 10th Dlstrlct\nvm lor One (1)\n\n'), (True, u'O H. CHRISTIAN GUNDERSON Pany Pmlgygqgg; DEM\nChlmpructlc D0ct0rIEMr1pr\u2022n\u2022ur\n\n'), (True, u'ALEX EASTON-BROWN P Pyghgg ; xy\nO Ptntlon Rnform Ccovdlnltar any me\n\n'), (True, u'O "\u201cc*V\\E|-   HWY Praismncus DEM\n\n'), (True, u'CONNIE VVONG hmma-\nO H\xa4lh\u2022rIMIl||\xa4N OM\xa4\u2022r pany Pm I xm\n\n'), (True, u'weren .\xa4. mucus p p,. 1\nO .......\xa4......... 0..... "" \u2019\xb0\'\xb0"\xb0\xb0 REP\n\n'), (True, u'B MARC LEVINE PGM PY6f8f8l\u2018\\0\xa2Z DEM\nCbuncllnwmbtr, Clty of Bin Rlhll\n\n'), (True, u'JOE BOSVVELL {gymn-\nO SINIII luslntss Ptnon Pimp INN!\n\n'), (True, u' \n\n')]), [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)])], [((17, [(648, 1221, 1243, 1603)], [(False, u'Candidates to tha Assembly, 10th Dlstrlct\nVote for Ono (1)\n\n'), (True, u'Chlmpmcilc Dod0rIEn!r\\pr\\n\u2022ur\n\n'), (True, u'P\u2022nsI\xa4n R\u2022|\xa4rm Coordinator\n\n'), (True, u'A.|\u2022\u2022mbIyn\u2022\u2022mh\u2022rIAt\xa20m\u2022y\n\n'), (True, u'O Mmmrmnnmgggu, Pmy Pruluence DEM\n\n'), (True, u'(D "\xa5\nSmlll Bullnttl Ovlntr\n\n'), (True, u'C) MARC I-EVINE Party Pmluvncei DEM\nCouncllmambu. Clty uf San R\u2022|\u2022\u2022l\n\n'), (True, u'JOE BOSWELL P P in :N0v\\\xa4\nO SmnIIEunln\u2022n Parson any m mm,\n\n'), (True, u' \n\n')]), [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)])], [((20, [(645, 1218, 1241, 1597)], [(False, u'Candidates to the Assembly, 10th District\nVote ior One (1)\n\n'), (True, u'O H. CHRISTIAN GUNDERSON Par\\yPm|\u20acfBI\\c\xa2: DEM\nGhlmprncilc D0c\\\xa4rIEn|r\u2022p\u2022\u2022lwur\n\n'), (True, u'P\u2022n\u2022Ion R\u2022fonn Coordinator\n\n'), (True, u'G MICHAEL ALLEN Party Preluencei DEM\nA1s\u2022vnb|yn\u2022\u2022mb\u2022v1Att\xa4n\u2022Oy\n\n'), (True, u'G CDNNIE WONG Party Prekmncei DEM\nMo(h\u2022rIMIIIIary Omar\n\n'), (True, u'O PETER J. IIIANCUS P3rlyPr!fB1Er\\oe: REP\nSmill Butlntss Owntr\n\n'), (True, u'G MARC LEVINE PGN! Plelefellcei DEM\nC\xa4uncIIm\u2022mb\u2022r. Elly of Sun Rnfnnl\n\n'), (True, u'O JOE BOSWELL Pariy Pmferer\\0!ZNOn9\nSmall Bunlmsn Parson\n\n'), (True, u' \n\n')]), [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)])], [((15, [(42, 735, 641, 1190)], [(False, u'Libertarian Party Candidates\nVcto for Ons (1)\n\n'), (True, u'O BARBARA JOY WAYNIIRE LIB\n\n'), (True, u'O LEE WRIGHTS LIB\n\n'), (True, u'O ROGER GARY LIE\n\n'), (True, u'O JAMES OGLE LIB\n\n'), (True, u'O SCOTT KELLER LIB\n\n'), (True, u'\n'), (True, u'O GARY JOHNSON UB\n\n'), (True, u'Q R. .1. HARRIS LIB\n\n'), (True, u'O CARL PERSON LIB\n\n'), (True, u'@ \n\n')]), [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10)])], [((22, [(39, 1107, 623, 1599)], [(False, u'Member, County Central Committee\nDistrict 2\nVote for no more than Four (4)\n\n'), (True, u'O K|:VIN KKIUK Pafly PVBIEIEOCE2 PU\nlncumbtnl\n\n'), (True, u'O MORGAN I\\:|.|.:V Pafly PVBISIBMDB: IU\n|ncumb\u2022n\\\n\n'), (True, u'O KAI<:N MuLU\\:N Pavly PVEIEIEMJE2 K\\\nlncumbinl\n\n'), (True, u'O ILENE MEYERS PaNy Prelemflcei R\\\nRatlrcd Educator\n\n'), (True, u'G VIC CANBY Parly Prelelencei R\\\nIm:umb\u2022nl\n\n'), (True, u'O G:URG|: H. BUUKLI: Paw Prslererwei R\\\nlncumbtnl\n\n'), (True, u'(;)(-\u2014 \n\n'), (True, u' \n\n'), (True, u'(j \n\n'), (True, u' \n\n')]), [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10)])], [((0, [(54, 132, 633, 740)], [(False, u'Candidates to the U.S. House of Representatlves\nZnd District\nVote for One (1)\n\n'), (True, u'O STACEV LAWSON pam/pygjemfycy Dg\nEdu\u2022:|l\xa4dSmaII Bullnitswunun\n\n'), (True, u'.\xbb\xa4\u2022-an LEvwu.n.su p p mm;\nO \xa4......... 0..... \u2018"\u2019 \u201d"\u2018 "\xb0\n\n'), (True, u'G DANIEL W. ROBERTS Paw Pftfelerwei RI\nStcumlll B1\xa4k\u2022r D\u2022|I\u2022r\n\n'), (True, u'C) pmnum vnu mi; Fw Prefevencs: ns\n\n'), (True, u"O   ?rg:.\xa4N|ON PaIYy PVE'\xa2r9r\\C:Z DE\n \n\n"), (True, u'O JARED HUFFMAN piyprpygeyeme; gg\nEnvlrunmtnial Ak\\\xa4rniyIA.|\xbb|\u2022mb|yIn\u2022mb\u2022r\n\n'), (True, u'Q MIKE HALLIWELL PaIYy PIEMEN2, RI\nColby! Pr\xa4f\u2022s|\xa4r\n\n'), (True, u'O Gn\u2022n Cnnvinlcn Ccmlullnnl ny me\n\n'), (True, u'O WILLIAM L. COURTNEY Party PIEYQKGVICB2 DE\nPhyIlcianllnv\u2022nI\xa4rIR\u2022|\u2022|mhtr\n\n'), (True, u'G BROOKE CLARKE PartyPre1srsnce:Nu\xa4\nSmnll Bu|ln\u2022|| Own\u2022r\n\n'), (True, u"O LARRY FRHZLAN Paw Prgfgygmgg, DE\nPtych\xa4(!\\\u2022r'apIsIlln\\\u2022rv\u2022n\\|01\\l$tlBusiruIspirtcn\n\n"), (True, u'O SUSAN L. ADAMS pam pmjemma; DE\nNu\xa4\u2022IC\xa4unty Suptrvlnnr\n\n'), (True, u' \n\n')]), [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13)])], [((1, [(42, 138, 639, 734)], [(False, u'Candidates to the U.S. House of Representatives\n2nd District\nVote tor One (1)\n\n'), (True, u'Educat\xa4fISm|Il Bullrllsswonun\n\n'), (True, u'Eu\xa4In\u2022Is Owrwr\n\n'), (True, u'S\u2022cur1tI\u2022s Bmknr D\u2022nI\u2022r\n\n'), (True, u'P\u2022taIuma Vlct Hlyar\n\n'), (True, u'Educ|tor1Aulh\xa4r\n\n'), (True, u'Env|r\xa4nm\u2022nlII A\xa2l\xa4rn\u2022yIAs1\u2022mbIym\u2022mb\u2022r\n\n'), (True, u'C0ll\u2022g\u2022 P1\xa4|\u2022\u2022|\xa4r\n\n'), (True, u'Gr\u2022\u2022n Convtrllon Consultant\n\n'), (True, u'Physlclln/|nv\u2022I1t\xa4rII\\\u2022$\u2022lr\\:h\u2022r\n\n'), (True, u'Smnll Buslnitl Ownnr\n\n'), (True, u'Psynhuihimpillllnhrv\u2022ntl\xa4n||lIBuIIn\u2022stp\u2022rI\xa4n\n\n'), (True, u'Nur!\u2022IC\xa4urlty Suplrvlsct\n\n'), (True, u' \n\n')]), [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13)]), ((4, [(42, 135, 648, 733)], [(False, u'Candidates to the U.S. Hcuso cl Representatives\n2nd Dlstrlct\nVote for Ons (1)\n\n'), (True, u'Educator/Sn\\III BusIn\u2022s|w\xa4man\n\n'), (True, u'Buslrnss Ownlr\n\n'), (True, u'S\u2022CurItl\u2022i Br\xa4k\u2022r D\u2022II\u2022r\n\n'), (True, u'Puinluml Vlci Mtyar\n\n'), (True, u'Educat0flAuIh0r\n\n'), (True, u'EnvIr\xa4nm\u2022r\\II| At|\xa4m|yIA|I|mbIym\u2022mb\u2022r\n\n'), (True, u'G0|I\u2022q\u2022 Prdnmr\n\n'), (True, u'Gran C\xa4\u2022\xb7vv\u2022\xa4l\xa4n Ccnsuttam\n\n'), (True, u'Phy!lcIanllnv\u2022II\\0rIR\u2022s\u2022irch\u2022r\n\n'), (True, u'Smnll Bu!|n\u2022\u2022| Ommlr\n\n'), (True, u'Piych\xa41h\u2022rIp|||lIr\u2022\\\u2022rv\u2022n\\I0nI\xa4|IBusln\u2022nnp\u2022m\xa4n\n\n'), (True, u'Nun\u2022ICounty Sup\u2022rvlnor\n\n'), (True, u'{T  \n\n')]), [(12, 12), (4, 4), (1, 1), (3, 3), (9, 9), (2, 2), (11, 11), (6, 6), (5, 5), (7, 7), (8, 8), (0, 0), (10, 10)]), ((10, [(42, 138, 639, 744)], [(False, u'Candidates to the U.S. House of Representatives\n2nd Dlstrict\nVote lor One (1)\n\n'), (True, u'Educ|t\xa4l\u2019ISmIII Burlrwllwumln\n\n'), (True, u'Buslrnn Ownar\n\n'), (True, u'S\u2022cur|\\|\u2022| lmku D\u2022|I0r\n\n'), (True, u'Pnlilumn Vlu Mayof\n\n'), (True, u'Educa\\orIAuth\xa4r\n\n'), (True, u'Envlrunmtnlnl Ahon\u2022\u2022yIA|s\u2022mhIylr\\\u2022mb\u2022r\n\n'), (True, u'C\xa4II\u2022q\u2022 Pr\xa4f\u2022$\\\xa4r\n\n'), (True, u'Grlin Cnrwirslun Consultant\n\n'), (True, u'Phy|IcIlnIInv\u2022r\u2022t\xa4rlR\u2022|\u2022|rcI\\\u2022r\n\n'), (True, u'Slnlll Bunlntsl Ownar\n\n'), (True, u'Pty:hMh\u2022ripl|(lIr\\l\u2022rv\u2022ntI\xa4nIs|lBu|ln\u2022||p\u2022n\xa4n\n\n'), (True, u'Nursalbaunty Sup\u2022rvI1or\n\n'), (True, u' \n\n')]), [(12, 12), (4, 4), (9, 9), (6, 6), (7, 7), (1, 1), (2, 2), (3, 3), (0, 0), (11, 11), (8, 8), (5, 5), (10, 10)])], [((5, [(51, 132, 633, 739)], [(False, u'Candldatns to the U.S. House of Roprosontatlvos\n2nd District\nVote for Ons (1)\n\n'), (True, u'O STACEY LAWSON Paw Pmhmrm; p[\nEducatcrlsmall Eullniuwcmln\n\n'), (True, u'C) :0*;*** I-%\u2018LVA\\-I-EN Pany PreIemn0e:N\xa4\n\n'), (True, u'O mums:. w. n\xa4sERTs Parry Pmfmncnz RI\ns\u2022wrm\u2022\xa4 Br\xa4k\u2022r \xa4\u2022\u2022\xa4\u2022r\n\n'), (True, u'Q TIFFANY RENEE pany pmkygnm; DE\nPdnlumn Vlct Mayor\n\n'), (True, u'C) NORMAN SOLOMON Pgny Pmyyycy DE\nEducIt\xa4dAUIh\xa4r\n\n'), (True, u'O JARED HUFFMAN Pgny Pyglqqngg; Di\nEnvlr\xa4nm\u2022nIII AI10r1\\\u2022yIA\xa4s\u2022mhlym\u2022mb\u2022r\n\n'), (True, u'O MIKE HALLIVVELL pany pmhmme; Rl\nC\xa4II\u20229\u2022 F!\xa4f\u2022I|\xa4r\n\n'), (True, u'Gr\u2022\u2022n Cnmvtrslnn Cunsulunl\n\n'), (True, u'O WILLIAM L. COURTNEY Par\\yPI\xa4I\xa4Itm6\u2018 DE\nPhysIcInnIInv\u2022n\\\xa4r/Rlsalrchtr\n\n'), (True, u'G BROOKE CLARKE Pyyy pyeqcmnw; No\nSmall Bullnitl Ovlntr\n\n'), (True, u'Psy\xa2h\xa4th\u2022mp|sIlIm\u2022rv\u2022mI0nI|VBu1ln\u2022np\u2022non\n\n'), (True, u'G SUSAN L. ADAMS pany Pmbwqceg DE\nNurs\u2022lC\xa4unty Supsrvlsor\n\n'), (True, u' \n\n')]), [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13)])], [((8, [(48, 138, 645, 737)], [(False, u'Candldatos to the U.S. House of Reprsacntatlves\n2nd District\nVote for Ons (1)\n\n'), (True, u'C) SYACEY LAWSON Pany PIBIGTBHOB2 DEM\nEduc|t\xa4rISm|II Bu|In\u2022||w\xa4m\xa4n\n\n'), (True, u'O JOHN LEWALLEN Pa|1yPr\xa4I\xa4f0I\\0e2 NON\nBun|n\u2022|\u2022 Ovmnr\n\n'), (True, u'B\u2022cuM|\xa4! Bmknr D\u2022nI\u2022r\n\n'), (True, u'O TIFFANY RENEE Par\\yPI\\!I$f0fl0c2 UtM\nPtuluma V\\\xa4\u2022 lllyor\n\n'), (True, u'G NORMAN SOLOMON Pariy PIGIHBDDQZ DEM\nEducaI\xa4rIAu\\h\xa4r\n\n'), (True, u'O JARED HUFFIIIAN Pariy PI\u2019BY8VSf\\C\u20aci UtM\nEnvlrcnmnntal Ali\xa4rn\u2022yIA.ls\u2022mb|ym\u2022mb\u2022r\n\n'), (True, u'C\xa4II\u2022@ PrM\u2022sso\xa2\n\n'), (True, u'Gr\u2022\u2022n Ccnvsnkan Consultant\n\n'), (True, u'S WILLIAM L. COURTNEY PIM PVBIGIMKZBJ DHA\nPhyI|cl|nIIrlv\u2022rII\xa4fIR\u2022s\u2022|rcO\u2022\u2022r\n\n'), (True, u'Small Bu|In\u2022|| 0\u2022m\u2022r\n\n'), (True, u'O LARRY FRITZLAN Pal1yPV8IB1GV\\CBZ DtM\nPlychcthtmplnllnliwinllcnlll/Bu1|n\u2022I|p\u2022rs\xa4n\n\n'), (True, u'Nur\\\u2022IC\xa4uMy $uP\u2022rvI|0r\n\n'), (True, u' \n\n')]), [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13)])], [((9, [(47, 138, 653, 740)], [(False, u'Candidates to the U.S. House of Representatives\n2nd District\nVots for One (1)\n\n'), (True, u'O STACEY LAWSON Pam pmbmme; DEM\nEdu\xa4n\\0rISma|I Bu|In\u2022\u2022|v1\u2022>man\n\n'), (True, u'O Buslntss Owmr aw re mms Nom\n\n'), (True, u'G DANIEL W. ROBERTS PBM PIMBVBDCBI REP\nSccuritlas Brokur Dllltf\n\n'), (True, u'O TIFFANY RENEE Par1yPrs|erence: DEM\nPnaluma Vlc\u2022 Mayor\n\n'), (True, u'O NORMAN SOLOMON Par\xa5yPrB|eren\xa4e DEM\nE\xa4uc\u2022t\xa4rVAu\\h\xa4r\n\n'), (True, u'O JARED HUFFMAN Party Pruinmmsi DEM\nErwlmmmnnul Alt\xa4m\u2022yIA||\u2022mbIym\u2022mh\u2022r\n\n'), (True, u'COII\u2022g\u2022 Pr\xa4|\u2022\u2022\u2022\xa4r\n\n'), (True, u'G ANDY CAFFREY PiVYy Pmlelwcet DEM\nGran Ccnvnrslcn Ccmtulunt\n\n'), (True, u'O wn.uAm L. c\xa4uR\xb7rNEv Pam Pveterence DEM\n\xbb=ny\xa4s\xa4a\xa4nn\xa4v\u2022m\xa4\xa41n\u2022\u2022\u2022\u2022v\xa4n\u2022r\n\n'), (True, u'O BROOKE CLARKE PBITY Pfefefenbei None\nSmall Buslnirt Owrwr\n\n'), (True, u'O LARRY FRITZLAN PartyP1e1erence. DEM\nPsychothtmpnsIIIn|\u2022rv\u2022Ivtl\xa4nis|IBusin\u2022ssp\u2022r1A:n\n\n'), (True, u'O SUSAN L. ADAMS PEN)! PIEIGIQNCEZ DEM\nNum\xbblC\xa4unty $up\u2022rv||\xa4r\n\n'), (True, u' \n\n')]), [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13)])], [((12, [(46, 132, 633, 737)], [(False, u'Candidates to tha U.S. House of Representatives\nZnd District\nVoto for Ons (1)\n\n'), (True, u'O STACEY LAWSON Party Pmfennoe: DI\nEdv1=|t\xa4r1SmalI Bu.||n\u2022|\u2022w0mnn\n\n'), (True, u'JOHN LEWALLEN Meanw-\nO Bullnwss Owntr pany P jk\n\n'), (True, u'DANIEL W. ROBERTS P Pwfsfencei R\nG ""\u2019\ns\u2022\u2022:urm\u2022s Bmknr Dnlsr\n\n'), (True, u'rnsnwv RENEE -\nO p...\xa4...... vm. my., my """\'\xb0\'\xb0"\xb0\xb0\u2018 \xb0\'\n\n'), (True, u'O NORMAN SOLOMON PaVYy Pmfeferbei DI\nEduc|t\xa4rIAulh\xa4r\n\n'), (True, u'O JARED HUFFMAN paw pwbmrmz Dj\nEnvlrbnmintll At\\0!r\\\u2022yIA||\u2022mb|yII\xbbmb\u2022r\n\n'), (True, u'G \u2018?;`I;\xa7\u2022HALLNVELL P3yPfM0ftfbe\u2018 Rl\n\n'), (True, u'O ANDY CAFFREY pany pmkmme; Dy\nGlitn C\xa4nv\u2022ml0n Cnruultlm\n\n'), (True, u'O WILLIAM L. COURTNEY Party Pvnlnmmgi DI\nPhy|Ic|anI|nv\u2022n\\\xa4fIR\u2022\xa2\u2022ar\xa4I\\\u2022l\n\n'), (True, u'O raucous cumxs Parry PMe1en\xa4e:No\nsmlu summa own-:\n\n'), (True, u'O LARRY FRITZLAN pany pmgmmo; D;\nPnychuthsruplslllntirvsntlonlsl/Buslntssparson\n\n'), (True, u'O SUSAN L. ADAMS Pal1yPre|\xa4mnce: Di\nNun\u2022ICoumy Sup\u2022rvI\u2022or\n\n'), (True, u' \n\n')]), [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13)])], [((13, [(48, 132, 654, 737)], [(False, u'Candldatss to the U.S. Hausa \xa4| Representatives\n2nd District\nV\xa4t\u2022 for Ono (1)\n\n'), (True, u'O STACEY LAWSON Party Plthmlwel DEM\nEducatarlsmnll Buslnnuwonun\n\n'), (True, u'JOHN LEWALLEN .\nO Bud Party PrI|\xa4rM\\0e.N0m\n\n'), (True, u'O DANIEL W. ROBERTS Pany Pllkmllxi REP\n8\u2022cur|\\I\u2022\u2022 Blulwr D\u2022nI\u20221\n\n'), (True, u'O nmumn v:\u2022E  Pam Putnam; DEM\n\n'), (True, u'NORMAN SOLOMON -\nO EducaI\xa4rIAu\\h\xa4r pany PMMMWI DEM\n\n'), (True, u'Q JARED HUFFMAN Party Pfufhf\xe9hcei DEM\nEnvlr\xa4nm\u2022rI\\AI At\\\xa4rI\\\u2022yIA||\u2022mhIylr\\\u2022mb\u2022r\n\n'), (True, u'G g&I;$\xb7HALLIWELL Par1yPr\xa4fer\xa4r\\\xa4sZ REP\n\n'), (True, u'ANDY CAFFREY pa pmhuu; M\nO Gntn Comnnloll Oansulhnl ny DE\n\n'), (True, u'wn.u\xbbun L. counrnsv mym-\nO \xbb\u2022\xb7y\xb7\xb7.....,..v.......\xb7\xbb.....\xa2\xbb.., """ \xb7 \xb0""\n\n'), (True, u'O BROOKE CLARKE Party Pyy1;m~\xbb;Mq|;g\nSmall Buslrnls Ownar\n\n'), (True, u'O LARRY FRITZLAN Pam Pyyggmyg DEM\nPIycholhlrlplslllnhrvll1\\l\xa4rIlt|lBu||I1\u2022||p\u2022l1\xa4\xa4\n\n'), (True, u'susna \u2022.. Anna; pa I\nO ~.....\xbb\xa4......, s.....~..... \'\u201d"\'\xb0\'\xb0"\u201c\xb0 \xb0\u2018\u201c\n\n'), (True, u' \n\n')]), [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13)])], [((17, [(637, 603, 1243, 1190)], [(False, u'Vote for Ona (1)\n\n'), (True, u'Edu\xa4I\\\xa4r1SmlII Butlntliwomln\n\n'), (True, u'C) \xa4\xa4\u2022\xa4\xb7\u2022\u2022\u2022\u2022 o\xbb:\u2022`FLEN PW *\u2019*=*\xa2*=\xa4\xa4=\xa4 Nw\n\n'), (True, u'O DANIEL IN. RUBtRI3 Pany Prelntemei REP\nSlcuhtln Brvlmr Dnalnr\n\n'), (True, u'Fthluinl Vl\xa4\u2022 Mlyw\n\n'), (True, u'Q NURMAN SOLOMON Par\xa5yPfB9efBI\\O!Z DEM\nEduc|l\xa4rVAulh0r\n\n'), (True, u'ErwIr\xa4nm\u2022nI|I At|\xa4m\u2022yIA1i\xbb\u2022mbly|n|mb\u2022r\n\n'), (True, u'C\xa4II\u2022g\u2022 Prchstar\n\n'), (True, u'O ANDY CAFFREY Pariy PvB9eIeI\\OBZ DEM\nGr\u2022\u2022n Cunvanlon Cannullnnl\n\n'), (True, u'Physl\xa4I\u2022nlInv\u2022nt0rIR\u2022|\u2022nrch\u2022r\n\n'), (True, u'O BROOKE CLARKE PBM PMBNHCBZ NODE\nSmall Bu\xa4In\u2022n \xa4wn\u2022r\n\n'), (True, u'P|ych0th\u2022r|pI\u2022tI|nt\u2022rv\u2022n\\I\xa4nI\u2022\\lBu\u2022In\u2022|\u2022p\u2022r|on\n\n'), (True, u'Q SUSAN L. ADAMS Par\\yPM\u20acIer\\0e: DEM\nNum\u2022IC\xa4unty Sup\u2022rv||\xa4r\n\n'), (True, u'@ \n\n')]), [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13)])], [((20, [(645, 588, 1241, 1186)], [(False, u'Candidates to the U.S. House of Reprasentatlves\n2nd District\nV\xa4ta for Ono (1)\n\n'), (True, u'O STACEV LAWSON Pany Prsiemmsi DEM\nEducI\\orI\xa7mIII Butlnatnwoman\n\n'), (True, u'O Eusllwtl Gina M `\n\n'), (True, u'S\u2022cur|tI\u2022s Brcknr D\u2022II\u2022r\n\n'), (True, u'O TIFFANY RENEE Par\\yPI!721ErI;\xa2Z DEM\nPnruluma Vlcu Mnyur\n\n'), (True, u'Educ||0rIAuIh\xa4v\n\n'), (True, u'Env|r\xa4nm\u2022n\\|l Atl\xa4n\xb7\\\u2022ylAn|\u2022mhIyn$\u2022mh\u2022r\n\n'), (True, u'O MIKE HALLIWELL Party Preference: REP\nCoII\u2022p\u2022 Pruhssur\n\n'), (True, u'O ANDY CAFFREY Party Prufsrsncoi DEM\nGnsn C\xa4nv\u2022r|I\xa4n Ccruullnnl\n\n'), (True, u'O WILLIAM L. CDLIRTNEY Pany Preference: DEM\nPhysI\xa4I\xa4nIInv\u2022nt\xa4rIR\u2022|\u2022|rch\u2022r\n\n'), (True, u'O BROOKE CLARKE Pa\xa4yPrv1ar\xa4ncs: None\nSmall Bu\u2022In\u2022n\u2022 Ownsr\n\n'), (True, u'Fsychothlmpllt/Int\u2022rv\u2022mI\xa4nIsUBusIn\u2022|sp\u2022r|\xa4n\n\n'), (True, u'Q SUSAN L. ADAMS Pany Preference: DEM\nNum\u2022IC\xa4un|y Suparvlncr\n\n'), (True, u'@ \n\n')]), [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13)])], [((21, [(48, 127, 646, 739)], [(False, u'Candldatn to th\u2022 U.S. House of Representatives\n2nd Dlntrlct\nVote for Ona (1)\n\n'), (True, u'O STACEV LAWSON Parly Preftmncei DEM\nEduc|t\xa4rI8m|II Bu|ln\u2022\u2022|w0m\u2022n\n\n'), (True, u'JOHN LEWALLEN Pg Pygigrgnwg Nom\nO Bunlntit Owntr ny\n\n'), (True, u'S DANIEL W. ROBERTS Party Preference: REP\nS\u2022\xa4ur|lI\u2022| Br\xa4k\u2022r D\u2022|I\u2022r\n\n'), (True, u'TIFFANY RENEE P P fs i DEM\nC) Palalumn Vlcn Mayor any .8 [mm\n\n'), (True, u'Q NORMAN SCLCIIIUN Party Prefemncei DEM\nE\xa4u\xa4It\xa4rIAuth\xa4r\n\n'), (True, u'O JARED HUFFIIIAN Parly Pl\xe9iefemei DEM\nEnvlmnmamnl Atl0rn\u2022ylA.|\xbb\u2022\u2022mbIym\u2022mb\u2022r\n\n'), (True, u'Q MIKE HALLIWELL Party Preiarencei REP\nC\xa4II\u20229\u2022 Pr\xa4f\u2022|\u2022\xa4r\n\n'), (True, u'O ANDY CAFFREY Patty Pmfnmmei DEM\nGr\u2022\u2022n Ccnvsrulan Cansultnnt\n\n'), (True, u'O WILLIAM L. COURTNEY Paty Pvvsfunmi DEM\nPhy|Icl\u2022nIInv\u2022m\xa4\u20221R\u2022r|\u2022\u2022rcI\u2022\u2022r\n\n'), (True, u'O BROOKE CLARKE Parly Prthvemei Nom\nSmall Bu|ln\u2022\u2022\u2022 Own\u2022r\n\n'), (True, u'Plychtlhlflplitllllhwtmlvnlillluslnasnpamon\n\n'), (True, u'SUSAN L. ADAMS Pvafammei DEM\nO Nur\\\u2022lCounty Sup\u2022rvI\u2022\xa4r Pm,\n\n'), (True, u'@ \n\n')]), [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13)])], [((2, [(664, 807, 1233, 1829)], [(False, u'Candidates to the United States Senate\nVote tor Ona (1)\n\n'), (True, u'O DIRK ALLEN KONUFIK PsRy PIBVQVEHOQZ RE\nMEA Sludtnl\n\n'), (True, u'Q DONALD KRANIFE Ps|1yPrn9nI\\!r\\C\xa4Z RE\nR\u2022tImd Admlnlitmtlun Dlrlclnr\n\n'), (True, u'\xa9 MIK: S I RINILING Parly PVSIMBHCE Uk\nCcnsumnr Rights Attcmty\n\n'), (True, u'O DIANE STEWART Pariy PI07\xa41\xa4I\\CB\u2018 DE\nSunIn\u2022nsw\xa4m|nIFln\xa4nc\u2022 Mnnngsr\n\n'), (True, u'C) NAK SHAH PGVTY PVENMBHCEI Ut\nEnv|r\xa4nm\u2022nt\xa4I Htulth Cchrultlnl\n\n'), (True, u'O NACHUM SHIFREN Parly PIGOBIBHCBZ RE\nEducat\xa4rIAulh\xa4rIButIn\u2022||mIn\n\n'), (True, u'O DENNIS JACKSON FarwFIeIeref\\c81 RE\nAarospaca Ganarnl Manaqar\n\n'), (True, u'C) DAN HUG!-1l:S Farly FKGIBIBOCEZ Kt\nSmlll Eullrwsl Owrvtr\n\n'), (True, u'O GREG CONLON Par\xa5YPm|sI!v\\C\xa4Z RE\nBusln\u2022|sm.|nICPA\n\n'), (True, u"O JOHN BORUFF ParNF'rB|erev\\C2; RE\nBusIn\u2022umnn\n\n"), (True, u'G OSCAR ALEJANDRD BRAUN PAr\xa5yPr%|BrBv\\CnZ RE\nBuslnsssmnnlliamshnr\n\n'), (True, u"O MARSHA FEINLANU PEM F'r\xa4I!r!rI(2I P\nRillnd T\u2022ach\u2022r\n\n"), (True, u'O DIANNE FEINSTEIN Party Preiefencei DE\nUnlhd S\\It\u2022s Stnaicr\n\n'), (True, u'M\xa4lh\u2022rIC\xa4nnuIIar\u2022YJAr!l$!\n\n'), (True, u'Busln\u2022|Iw\xa4nmnIN\xa4npr\xa4f|\\ Ex\u2022l:u||v\u2022\n\n'), (True, u'O KABIRUDDIN KARIM ALI Pariy Preference. F\nBuxlrnsnman\n\n'), (True, u'BusIn\u2022sI Alturnsy\n\n'), (True, u'O ROGELIO T. GLORIA Paw Pfelsfencez RE\nGr\xa4\xa4u\u2022t\u2022 $lu\xa4\u2022n\\IBu|In\u20221|mnn\n\n'), (True, u'O DUN J. GRUNIJIVIANN FarYy PVQIBBDCQ r\nDonor M Chllbbrictlc\n\n'), (True, u'O ROBERT LAUTEN PaNyPMM01u2 RE\n\n'), (True, u'O GAIL K. LIGHTFDDT PaRyFMtN1l\xa4 L\\\nR\u2022llN\xa4 Nur\\\u2022\n\n'), (True, u'C) DAVID ALEX LEVITT Plny FNVBIBHOQZ DE\nCompmar ScI\u2022ntIsIIEng|n\u2022\u2022r\n\n'), (True, u'Q URLY IAIIL Fariy PV!I\xa2f\xa2VlCeI RE\nDoctor/At\\or1$\u2022ylBu$ln\u2022xsw\xa44nan\n\n'), (True, u'O AI. RAMIK=L PGPWPTUIQYQTIN1 Ht\nBunlnasnman\n\n'), (True, u' \n\n')]), [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23), (24, 24), (25, 25)])], [((3, [(642, 804, 1233, 1825)], [(False, u'Candldates tc the Unlted States Senate\nVote for One (1)\n\n'), (True, u'\xa9 DIKK A|.|..:N KUNUPIR Pafly FIBISVSHOGI Kt\nMBA S\\ud\u2022n|\n\n'), (True, u'R\u2022!Ir\u2022d Admlnlnlnllun Dlncinr\n\n'), (True, u'G MIKE STRIMLING PlVYy PIEIBVBDCE DH\nCnnuumtr Rlghtl Att0m\u2022y\n\n'), (True, u'O DIANE STEWART Paliy Prslulnms. DE\\\nBusln\u2022n.wom|nIFIn|n0\u2022 Mnnagtr\n\n'), (True, u'S NAR SHAH Farly PIEISISDOQZ DU\nEnvlrcnrhtnhl Nlillh Consultant\n\n'), (True, u'G NACHUNI SHIFREN Paw Prnfamnca: RE\nEducal0rIAu!h\xa4rIBu.lIn\u2022i\xbb.Iman\n\n'), (True, u'O DENNIS JACKSON Pally PVEYQIQVICEZ RE\nAnroipict G\u2022n\u2022rII M|n|\xa7\u2022r\n\n'), (True, u'G DAN HUGHES Paw Plslsmllcsi RE\nSmall Bunlnnn Owmr\n\n'), (True, u'O GREG CUNLUN PBNYPVGIEIEMGZ RE\nBusIn\u2022|\xa4m\xa4nI\xa4FA\n\n'), (True, u'G JOHN BORUFF PartyPm|amr\\00$ RE\nBusIn\u2022ssmnn\n\n'), (True, u'B OSCAR ALEJANDRU BRAUN Par\xa5yPf2|efBY\\wZ RE\nBu|In\u2022|sm\xa4nIRan:h\u2022r\n\n'), (True, u'G MARSHA FEINLAND PGIYYPIIVBVQDOQI F\nR\u2022tI1\u2022d T\u2022ach\u2022r\n\n'), (True, u'Q DIANNE FEINSTEIN PBf\xa5yPrB|\xa4fM\\O\u20acZ DE\\\nl1nII\u2022d Slahn S\u2022nnt\xa4r\n\n'), (True, u'Mmh\u2022rIC\xa4n|uII1nIIAr!Iit\n\n'), (True, u'C) ELIZABETH EMKEN Parly Preference: RE\nBu1In\xa2ssw\xa4marVN\xa4npr\xa4fI\\ Ex\u2022cutIv\u2022\n\n'), (True, u'O :ABIRUDDIN KARINI ALI Pany Pmfnmncei P\nuxlrnssmln\n\n'), (True, u'O RICK WILLIAMS Party Prelemllcei RE\nBu\u2022In\u2022\u2022\u2022 Altonuy\n\n'), (True, u'O ROGELI0 T. GLDRIA Party Pmhmru: RE\nGradu\u2022t\u2022 Slud\u2022ntIBu|In\u2022\u2022\u2022m|n\n\n'), (True, u'DON J. GRUNDIIANN Party Prebvsmei A\nO Douur cf Chlroprsctlc\n\n'), (True, u"8 ROBERT LAIJTEN Pa\u2022tyFm'IuMl\xa40i RE\n\n"), (True, u'GAIL K. LIGHTFODT Party Preftmncet Ll\nC) R\u2022tlr\u2022d Nurs\u2022\n\n'), (True, u'G DAVID ALEX LEVITT Pariy Pmhlthcti DE\\\nCampmor SoI\u2022ntI|tIEngIn\u2022\u2022r\n\n'), (True, u'S ORLV TAI12 Party Pvdlefencei RE\nD\xa4\xa4\xa4rIAtt\xa4n\u2022\u2022ylEus|r\u2022\u2022||\xa4\xbbo\u2022nan\n\n'), (True, u'AL RANIIREZ Pa Fldbmboi RE\nO Buslnassman ny\n\n'), (True, u'(D \n\n')]), [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23), (24, 24), (25, 25)])], [((6, [(645, 804, 1233, 1826)], [(False, u'Candldates to the United States Senate\nVote for One (1)\n\n'), (True, u'O UIRK A|.L.|:N KUNUPIK Parly PVSIGYBVICEZ RI:\nMEA Studlnt\n\n'), (True, u'CD DONALD KRAMFE Parly Prelelencei RE\nR\u2022tIr\u2022d Admlnlshatlon Dlrtdor\n\n'), (True, u'O MIK: SIKIMLING Far!)/Frelelerloei DE\nOunsumtr Rights Atlurnty\n\n'), (True, u'Q DIANE STEWART Parly Pmlsrencei DE\nBu$ln\xa4$tw\xa4m|nIFlnam:\u2022 Managar\n\n'), (True, u'O NAK SHAH Farly Ffefelencei DE\nEnvlronmullal Htnlth Ccmsultint\n\n'), (True, u'Q NACHUM SHIFREN Parfy Plefershcei RE\nEducal\xa4rIAulh\xa4rIBu\xa4lncsrman\n\n'), (True, u'C) DENNIS JACKSON Party Prelsrencei RE\nA\u2022r0spnc\u2022 G\u2022n\u2022raI M|nag\u2022r\n\n'), (True, u'C) DAN HUGHES Paffy Prefershcei RE\nSmall Buslrnu 0wn\u2022r\n\n'), (True, u'O LSREG CONl6g;1 Party Plek\xe9rencei RE\nuslntssmanl\n\n'), (True, u'O JOHN BORUFF ParYy Prelererwei RE\nBusinessman\n\n'), (True, u'\xa9 OSCAR ALEJANDRU ERAUN PaHyPI\u20ac!eVBI1C92 RE\nBusln\u2022ssm|nIRnm:h\u2022r\n\n'), (True, u'Q NIARSHA FEINLAND Par\xa5yPre7eren\xa4eZ F\nRetlrud T\u2022a|:h\u2022r\n\n'), (True, u'O DIANN: I-EINSI :IN Parly Prelcrencei DE\nUnlhd Sinks S\u2022nIt\xa4r\n\n'), (True, u'O CCLLEEN SHEA FERNALD Party Preference: DE\nM\xa4lh\u2022rlCon|uIIanIIAr\\IsI\n\n'), (True, u'O ELIZABETH EMKEN Party Preference; RE\nBusIn\u2022ssw\xa4m\xa4nINcnprcIIt ExucutIv\u2022\n\n'), (True, u'O KAEIRUDDIN KARINI ALI Par\\yPre|erem:e2 F\nEuslmssman\n\n'), (True, u'Q RICK WILLIAMS PaKy Preference RE\nBuslnnsu Atlomay\n\n'), (True, u'O ROGELIO T. GLDRIA Pany Preference RE\nGmduah $tud\u2022mIBu1ln\u2022s1man\n\n'), (True, u'O DUN J. GRUNDMANN Party Pleierenoei .\nDoctor of Chiropractic\n\n'), (True, u'O ROBERT LAUTEN Party Preference RE\n\n'), (True, u'Q GAII. K. LIGHTFOOT Par\\yPre1er2n0e$ L\\\nRallmd Nurse\n\n'), (True, u'C) DAVID ALEX LEVITT Party Preference: DE\nCcmpuhr Sclantlst/Er1gIn\u2022\u2022r\n\n'), (True, u'O ORLY TAITZ Par\\y Prefererwe RE\nD\xa4c1\xa4rIAll\xa4m\xa4yIBuuInusswcman\n\n'), (True, u'O AL RAM|I\xb7u:L Pafly Plewehcei RE\nEuxlrwssmln\n\n'), (True, u'@ \n\n')]), [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23), (24, 24), (25, 25)]), ((22, [(653, 804, 1233, 1824)], [(False, u'Candidates to the United States Senate\nVote for One (1)\n\n'), (True, u'Q DIRK ALL:N KUNUPIK Parfy PISYEVEHCSZ Ht\nMBA Sludtnl\n\n'), (True, u'O DONALD KRANIPE Pal1yPIe7eVBDCeZ RE\nR\u2022tIr\u2022d Admlnlstratlun Dlr\u2022c1\xa4r\n\n'), (True, u'O MIKI: S I RIMLING Parly PIEIEIEHCE: lJI:\\\nC\xa4nsum\u2022r Rlghls Attorn\u2022y\n\n'), (True, u'\xa9 DIANI: S I :vlvlAKI Parly PIEIQIEHCEZ UI:\\\nEusIn\u2022sswcm|nIFlr1\xa4nca Manager\n\n'), (True, u'O NAK SHAH Parly PVEOEIEHCEZ DE\nEnvlranmsnul H\u2022aIth Consulunl\n\n'), (True, u'O NACHUM SHIFREN Pa|1yPre|erer\\0eZ RE\nEduc\u2022t\xa4rIAuth\xa4rIEusInsssman\n\n'), (True, u'O DENNIS JACKSON Pa|1yPre9erer\\0eZ RE\nA\u2022r0spa0\u2022 G\u2022n\u2022\xa4I M|n|\xa4\u2022r\n\n'), (True, u'O DAN HUGHES PBl1yPIeIeI\u20acl\u2019\\0e2 RE\nSmnll Business Ownar\n\n'), (True, u'O GI(:G UUNLUN Pafly FIBIEVEDWI Kt\nBu!|r\\l$|m|nlCPA\n\n'), (True, u'G JOHN EDRUFF PBl1yPfe|eI\u20acr\u2022C\xa4Z RE\nBuslnansman\n\n'), (True, u'Q DSCARALEJANDRU BRAUN FarlyPre9erence: RE\nBusIn\u2022ssmnnIRam:h\u2022r\n\n'), (True, u'O MARSHA FEINLAND Pariy Plelerencei F\nRnllmd Tnachar\n\n'), (True, u'O DIANNI: I-tIN$II:IN Parly PIQVBIBIWOQZ DE\nUnlt\u2022d Shit! Sanator\n\n'), (True, u"O UULL\\:|:N 5H:A I\xb7|:I<NALU F3|1yF|'!VEf\u20acf\\0eZ DE\nM0th\u2022rIC\xa4nsuItantlArtlst\n\n"), (True, u'BusIn\u2022|sw\xa4manlN\xa4npr\xa4fl! Exmcutlvu\n\n'), (True, u'O KABIRUIJDIN KARIM ALI Fa|1yFre7Bren0eZ F\nBus|n\u2022s1man\n\n'), (True, u'O RICK WILLIAMS Paw PVBIBIQVIDBZ RE\nBuslntis Attcrnty\n\n'), (True, u'O ROGELI0 T. GLORIA Party Prefelerviei RE\nGraduata Studanllhuslnnmman\n\n'), (True, u'O DUN J. GRUNDMANN Pariy Plehremiei y\nDoctor nl Chlroprtctht\n\n'), (True, u'O I(UB:RI LAuI:N Par\xa5yPm|er\xa2nC2Z RE\n\n'), (True, u'O GAIL K. LIGHTFUUT Pany Plelerencei U\nRullmd Num!\n\n'), (True, u'O DAVID ALEX LEVITT Party Freiereme: DE\nOcmputar ScI\u2022ntIstIEngln\u2022ar\n\n'), (True, u'O ORLY TAITZ Pariy PVEIEIBVICBZ RE\nD\xa4c1\xa4rIAIi\xa4m\u2022yIBu||n\u2022s|w0rnan\n\n'), (True, u'O AL HAMII\xb7(|:L Pavly Pr\xe9lefent\xe9i RE\nBusirmssman\n\n'), (True, u' \n\n')]), [(24, 24), (18, 18), (23, 23), (21, 21), (6, 6), (19, 19), (11, 11), (9, 9), (15, 15), (20, 20), (5, 5), (10, 10), (0, 0), (1, 1), (22, 22), (16, 16), (7, 7), (17, 17), (12, 12), (4, 4), (2, 2), (8, 8), (3, 3), (13, 13), (14, 14)])], [((7, [(670, 810, 1234, 1823)], [(False, u'Candidates to the United States Senate\nVote for One (1)\n\n'), (True, u'D DIRK ALL:N KUNUPIK Parly PIEIHBHDE. RE\nMBA Sludtnl\n\n'), (True, u'D DONALD KRAMPE Pzriy Pralnfenuei RE\\\nR\u2022tIr\u2022d Admlnlnrltlcn DIr\u2022ct\xa4r\n\n'), (True, u'D MIKE STRIMLING Pal1yPIe|eIer\\Ce2 DEI\nCnnsumar Rights Atl\xa4m\u2022y\n\n'), (True, u'Bu|In\u2022\u2022|w\xa4m|nlFIn|r1c\u2022 M|n1\xa4\u2022r\n\n'), (True, u'D NAK SHAH Fafly Freielencei DEI\nEnvlrcnmsnlal Haaith Consultant\n\n'), (True, u'D NACHUM SHIFREN PaNyPr\xae12vBn\xa2e1 RE\nEduc|t0rlAu\\h\xa4rIBusln\u2022s|man\n\n'), (True, u'D DENNIS JACKSON PaHy PIQYHBHDGZ RE\\\nAirnspacl B|n\u2022ra| Manag\u2022r\n\n'), (True, u'D UAN huts!-uzs Fariy Preh`:\xa2Bnc\u20ac\u2018 RH\nSmill Eutlrmst Own\u2022r\n\n'), (True, u'D GREG CONLON Par\\yPre1erence: RE\\\nEusIn\u2022ssmanICPA\n\n'), (True, u'3 JOHN EURur\xb7F PaKyPIeIefnIIDB: RE\\\nBuslnassman\n\n'), (True, u'D USUAK AL\xa4.IANUI\xb7(U BRAUN Parly PIEIHBHCE2 RE\\\nBuslna\xa4|.manlR1nch\u2022r\n\n'), (True, u'D MAHSHA I-I:IN|.ANU Pa|1yFreYBI\u20ac\xa4\xaeZ F\nRttlrid Tllchtf\n\n'), (True, u'C) DIANNE FEINSTEIN Par\xa5yP\xa5B|t!fBf\\Ce\u2019 DEI\nUnltnd Subs Senator\n\n'), (True, u'3 UULL|:|:N SH|:A I\xb7:I<NALU Parly Preference: Uh!\nM\xa4th\u2022rIC\xa4n:ulhr1IIAr\\|sl\n\n'), (True, u'D :LILAI$:IH l:NIK|:N Pany PVEYBVEHLEI RU\nBu$ln\u2022\xa4$w\xa4manINO\xa4pr0/Ht Ex\u2022cutlv\u2022\n\n'), (True, u'D KABIRUDDIN KARIIVI ALI Par\\yPre92Ien0\u20acZ P\nBuslnansmnn\n\n'), (True, u'O RICK WILLIAMS ParYy Prefsfsntei RE\nBunlnnnn Att\xa4m\u2022y\n\n'), (True, u'O RUGELIU I. GLUHIA Pafly PIBVHHIUEZ RE\\\nGrnduih S\\ud|ntIBu\xa4|n\u2022a\xa4mar|\n\n'), (True, u'O DON J. GRUNDMANN Pany Plefelehot l\nD\xa4ct\xa4r\xa4fChIroprac1I|:\n\n'), (True, u'D RUB|:RI LAuI\\:N ParYyFIEQef!rlL\u2018EZ RE\n\n'), (True, u'C) GAIL K, LIGHTFUUT Pafly Pfehlehmi LI\\\nRnllmd Nunn\n\n'), (True, u'D DAVID ALI:x LI:v|I I Pa|1yFreIeIence: DEI\nComputtr ScI\u2022rIIl:|IEngIn\u2022\u2022r\n\n'), (True, u'D ORLY TAITZ PaIYyPIeVBFsh061 RE\nD\xa4ct\xa4rIA!t\xa4m\u2022yIBusIn\u2022|sw\xa4man\n\n'), (True, u'Z) Bulmnlmn Pan; Pmrmnm: Raw\n\n'), (True, u' \n\n')]), [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23), (24, 24), (25, 25)])], [((11, [(657, 813, 1233, 1828)], [(False, u'Candidates to the United States Senate\nVols for One (1)\n\n'), (True, u'O 3:}I;AL.LEN KONOFIK Par1yPreIerem:e: RE\\\n\n'), (True, u'O UUNAI-D KRAMFE Par\\y Pmfelentei RE\\\nRullrvd Admlnlstrnllon Dlnntcr\n\n'), (True, u'O MIKE STRIMLING Pany Prefelencet DEI\nCnnsumar Rlghu Attorney\n\n'), (True, u'C) UIANE 5 V BWAR Y Party Preference? DEI\nBusIn\u2022ssw\xa4mnnlFIn\xa4nc\u2022 Managnr\n\n'), (True, u'O NAK 5HAH Pany Preference; DEI\nEnvlronmantal Health Consultant\n\n'), (True, u'O NACHUM SHIFREN Party Preference: RE\\\nEduc\u2022t0rIAulhorIBusIn\u2022rsman\n\n'), (True, u'O DENNIS JACKSON Parly Preference! RE\\\nAsmspau G\xa4n\u2022raI Manager\n\n'), (True, u'O DAN HUGHES Parly Preference: RE\\\nSmnll Buslnuss 0wn\u2022r\n\n'), (True, u'\xa9 GREG CONLON Pal1yFre|9I\u20acv\\CeZ RE\\\nBusIncssmanICPA\n\n'), (True, u'O JOHN BORUFF Party Plsiersncsi RE\\\nBuslrwssrnan\n\n'), (True, u'G OSCAR ALEJANDRO BRAUN Party Preierencei RE\\\nBusin\u2022ssm\xa4nIR\xa4nchar\n\n'), (True, u'Rstlrtd T\u2022ach\u2022r\n\n'), (True, u'Q DIAN NE FEINSTEIN Farly Preference: DEI\nUnihd Sinus Sanatnr\n\n'), (True, u'Muthsr/C\xa4nsul!|n\xa2IArtIst\n\n'), (True, u'Busin\xa4ssw0manIN\xa4npr\xa4fil Exuculivn\n\n'), (True, u'Busln\u2022:sm|n\n\n'), (True, u'O RICK WILLIAMS Party Prsfemncei REI\nBusinass Attorney\n\n'), (True, u'O ROGELIO T. GLORIA Party Preference: REI\nGradunn Stud\u2022ntlEusln\u2022\u2022smsn\n\n'), (True, u'C) DUN J. GRUNDMANN Pariy Preference: F\nDuctur \xa4f Chiropractic\n\n'), (True, u'Q ROBERT LAUTEN FariyFr\xa4|erence: REI\n\n'), (True, u'Rltimd Nurse\n\n'), (True, u'Ccmbullr Scl\u2022r\\II$IIEngln\u2022\u2022r\n\n'), (True, u'O ORLY TAITZ Party Preference: REI\nD\xa4ct\xa4rIAn\xa4rn\xa4ylEuslnnnswcman\n\n'), (True, u'\xa9 AI. RAMIREZ Party Preference: RE!\nBunlnsslman\n\n'), (True, u' \n\n')]), [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23), (24, 24), (25, 25)])], [((14, [(677, 810, 1233, 1825)], [(False, u'Candidates to the United States Senate\nVote f\xa4r One (1)\n\n'), (True, u'D DIRK ALL|:N KUNUFIK Farly PIBYSVEDCBI NI:\nMBA Studtnt\n\n'), (True, u'D DONALD KRAMFE Parly Plelelencni RE\nRallnd Admlnlslrnllon Dlruciur\n\n'), (True, u'D MIKE STRINILING Pariy Pleierehce DE\nC\xa4nsum\u2022r RI\xa4htL Allcmay\n\n'), (True, u'D DIANE STEWART Pal1yFIef2re\xa4C\u20ac DE\nBuslrusswuman/Financt Ilanaihr\n\n'), (True, u'D NAK SHAH Pafly Pfeielemei DE\nEm1Ir0nm\u2022n!nI Hnallh Ccnsultnnt\n\n'), (True, u'D NAUHUM SHIP K|:N Parly PVEOBIBHCBZ KI:\nEdu\u2022:al\xa4rIAulh0rIBus|r\\0ssman\n\n'), (True, u'D DENNIS JACKSON Pam PVEIGIGMZB2 RE\nA\u2022r\xa4|p||:\u2022 G\u2022n\u2022mI Illnnngur\n\n'), (True, u'D DAN HUGH|:S Paf\\yPV\u20ac|\u20ac1GV\\C8Z KI:\nSmall Eullnlss Ovmtr\n\n'), (True, u'D GREG CONLON Party Prelerencei RE\nEunln\xa4.|nmnnICPA\n\n'), (True, u'D JOHN Bukurr Pafly PVSIBNMB2 Kt:\nBusinttsmln\n\n'), (True, u'BusIn\u2022nsmnnIR1nch\u2022r\n\n'), (True, u'D MARSHA r:INLANL> Fafly PIBYETENOQZ P\nRnlrod Tanchur\n\n'), (True, u'D DIANNI: I\xb7|:INS I |:IN Farly PYEIEIENCEZ Ut\nUnltad Still! $\u2022r\\at\xa4r\n\n'), (True, u'M\xa4\\h\u2022rIC\xa4n|uI\\an\\lAnI|l\n\n'), (True, u'D E|.|LAB:|H |:I\\M\\:N ParYy PVSIEVEHCS2 Ht\nBusIn\u2022ssw\xa4mnnIN\xa4r\\|\xa4P\xa4I|t Exncutlva\n\n'), (True, u'3 KABIRUDDIN KARINI ALI FBr\xa5yFr\u20acVeI\u20acf\\0\u20acI P\nBusIn\u2022i|man\n\n'), (True, u'D Klux WILLIAMS Party Plelerehcs. Kt:\nBuslnsss Attnmay\n\n'), (True, u'D ROGELIU T. GLUKIA PBM PIEYEVEHCSZ NI:\nGmduah SKud\u2022r\\l/Eulmtslman\n\n'), (True, u'Dccicr ul Chlropracllu:\n\n'), (True, u'D ROBERT LAUTEN Fll1yFIeYBVEf\\(\xa3 Kt\n\n'), (True, u'Rallnd N umn\n\n'), (True, u'D DAVID ALEX LI:VI I I Parly PVEIGYEMEZ Uh\nCompmir Scl\u2022nIlsIIEngIn\u2022\u2022r\n\n'), (True, u'D ORLY TAITZ PEITYPIGOBIBDCQZ RE\nD\xa4ct\xa4rIA\\\\\xa4m\u2022yIBunlnasnwnman\n\n'), (True, u'D AL RAMIREZ Pa|1yFr\u20ac|EIEf\\\xa4\u20acZ KI:\nEuilnsssman\n\n'), (True, u'j -\xb7\u2014\u2014\n\n')]), [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23), (24, 24), (25, 25)])], [((15, [(646, 810, 1233, 1826)], [(False, u'Candldatss to the Unltod Status Senate\nVote for Ons (1)\n\n'), (True, u'O DIRK ALLEN KDNOFIK PartyPre|\xa4r\xa4nc\xa4: RE\nMBA Student\n\n'), (True, u'C) DONALD KRANIFE Par\xa5yPrcfsIeM:9\u2018 RE\nRullrnd Admlnlslrnllon Dlrvctor\n\n'), (True, u'G MIKE STRINILING Parly Pruinrnncu DE\nCcnnumu Rlgmn Altcrmy\n\n'), (True, u'O DIANE S TEVIIARI PaIYy Prefsmma, DE\nBus|n\u2022rlw\xa4minIFInIn0\u2022 Mlrmqir\n\n'), (True, u'O NAK SHAH Pany Preiumnce. DE\nEnvlmnmontnl H\u2022\xa4IIh Cunsulunt\n\n'), (True, u'O NACHUM SHIFREN Par\xa5yPIO|nInr\\0nZ RE\nEducat\xa4rlAuthorIBu\u2022ln\u2022s\u2022m\u2022n\n\n'), (True, u'O DENNIS JACKSON Party Prelarsncai RE\nA\u2022r\xa41pa0\u2022 G\u2022n\u2022r\xa4I Man\u2022g\u2022r\n\n'), (True, u'O DAN HUGHES Parly PI8|efer\\O9\u2019 RE\nSmall Buslnssr Owner\n\n'), (True, u'O Gkcu UUNLUN Pal1yPIa|0I!r\\0\xa4Z RE\nBusIn\u2022\xa4smanICPA\n\n'), (True, u'JOHN BDRUFF Pa Prafevemei RE\n8 Buslnsnamnn ny\n\n'), (True, u'8 OSCAR ALEJANDRO BRAUN ParYy PrG7M\xa4r\\O0\u2018 RE\nBusIn\u2022|sm\u2022nlRam:h\u2022r\n\n'), (True, u'G NIARSHA FEINLAND PaMPr01erer\\0e\u2018 F\nRallmd T|ach\u2022r\n\n'), (True, u'O DIANN: rzllls I zu! Party Pmhmncn: DE\nUnIt\u2022\xa4 $\\II\u2022I $\u2022n|I\xa4r\n\n'), (True, u'O CDLLEEN SHEA FERNALD PaMPr\xa4|21\xa4nc\xa4: DE\nM\xa4th\u2022rI00\u20221|uR1MIAnI||\n\n'), (True, u'O ELIZABETH EMKEN Party Prelstsnce. RE\nBuslnuswunanmcnprvhl Ex\u2022cu!|v\u2022\n\n'), (True, u'O KABIRUDDIN KARIIII ALI Par\xa5yPrBfsrm\xa4\xa4: F\nBunlmnunan\n\n'), (True, u'Bu!|n\u2022Is Atltwnty\n\n'), (True, u'O ROGELIO T, GLORIA Parfy Pf\xa4Ue|\\0\xa4` RE\nGmdunn Stud\u2022nIIBu\xa4In\u2022umnn\n\n'), (True, u'G DON J, GRUNDMANN P\xa4f\\y Prtieiehce, r\nDonor \xa4fCI\u2022Ir\xa4pmc1I1:\n\n'), (True, u'C) ROBERT LAUTEN Party PmIamnc\xa4\u2018 RE\n\n'), (True, u'R\u2022\\Ir\u2022d Nur\\\u2022\n\n'), (True, u'O DAVID ALEX LEVITT Party Pmfsmmei DE\nC\xa4mP\xa4\xa5\u2022r Sc|\u2022n1l\u2022IIE\u2022\\\xa4|n\u2022\u2022r\n\n'), (True, u'D0\xa2l\xa4rIAt|\xa4rr\u2022OyIBu1|n\u2022\xa2sw\xa41v\\|n\n\n'), (True, u'O AL RANNREZ P3fYYPIB7\xa4I\xa4t\\\xa2GZ RE\nBuslmsunan\n\n'), (True, u'(:) -\n\n')]), [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23), (24, 24), (25, 25)])], [((17, [(39, 846, 575, 1867)], [(False, u'Candidates tn the United States Senate\nVote for One (1)\n\n'), (True, u'O DIRK ALLEN KONOFIK Paky PVQYBVU\nMEA Student\n\n'), (True, u'O DONALD RRAMPI: Flny PIEYMH\nRMING Admlnlllrlllun Dlrtctnr\n\n'), (True, u'G MIKE STRIMLING Pam PVBOEIB\nConsumar Rlghu Alturnny\n\n'), (True, u'O DIAN: S I :wAI( I Parly Pmlere\nBullnstlwomnn/Flninm M\u2022na\xa4\u2022r\n\n'), (True, u'O NAK SHAH Paw PNOBN\nEnvlronmsnul H\u2022\u2022I!h Connullanl\n\n'), (True, u'O NACHUNI SHIFREN Party Prehm\nEducator/Auth\xa4rIBu\u2022In\u2022n\u2022msn\n\n'), (True, u'G DENNIS JACKSON Paky Prefsw\nAumspacu Gansml Managar\n\n'), (True, u'DAN HUGHES Party Prefnm\nC) Small Bu|ln\u2022\u2022| Own\u2022r\n\n'), (True, u'GREG CDNLDN Paw PMBM\nO BusIn\u2022r\u2022mnnICPA\n\n'), (True, u'O JOHN BURUFF Paky PIMBM\nBuslnntmnn\n\n'), (True, u'O OSCAR ALEJANDRO BRAUN Pany Pmfefm\nBu\u2022ln\u2022s\u2022m\u2022nIRancI\u2022\u2022r\n\n'), (True, u'C) Rtllrld T\u2022\u2022ch\u2022r ny\n\n'), (True, u'O DIANNE FEINSTEIN Pafiy PTBUBVB\nUnlud Subs Senator\n\n'), (True, u'M\xa4thQrI\xa220v\\IuII1nIlAf\\|I\\\n\n'), (True, u'O ELIZABETH EIUIKEN PaI\xa5yP1E|Bf\xa4l\nBu|In\u2022ssw\xa4m|nINOnpr\xa4fII Ex\u20221:mIv\u2022\n\n'), (True, u'O \xa4.....\u2018.......n "\u2019 \u201d\u201d\n\n'), (True, u'O RICK WILLIAMS Paw Prtfeml\nEunlnnsn Aliornay\n\n'), (True, u'Grlduhlt Studtmllullntlimln\n\n'), (True, u'O DON J. GRUNDMANN Party Pmferm\nDoctor of Chiropractic\n\n'), (True, u'O ROBERT LAUTEN Party Preferen\n\n'), (True, u'R\u2022tlr\u2022d Nuns\n\n'), (True, u'O DAVID ALEX LEVITT Party Prelere\nCampular Sc|\u2022ntI\u2022tIEnq|n\u2022\u2022r\n\n'), (True, u'O ORLY TAITZ Party Preisrsl\nD\xa4|:I\xa4rIAh0rn\u2022yIBunlnasswnman\n\n'), (True, u'AL mummez va mem.\nO \xa4..............\xbb "Y\n\n'), (True, u' \n\n')]), [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23), (24, 24), (25, 25)])], [((18, [(673, 810, 1233, 1824)], [(False, u'Candidates to the Unlted States Senate\nVote for One (1)\n\n'), (True, u'D DIRK ALLEN KONOFIK Pam Preielenoei RE\nMBA Studant\n\n'), (True, u'D DONALD KRAMPE F8f\\yF1\u20ac79fer\\<;9; RE\nRntlnd Admlnlnmtlon Dlmciur\n\n'), (True, u'D MIKE STRINILING Paw PVSVBVSHLG1 DE\nC\xa4nsum\u2022r Rllhtt Atlcmty\n\n'), (True, u'D DIANE STEWART Palfy Pleievehce. DE\nBunIr\u2022\u2022nnw\xa4manIFIn|nc\xa4 Mannqnr\n\n'), (True, u'D NAK SHAH Pany PreVBVBnw` DE\nEnvlmnmnmal Haalth Consultant\n\n'), (True, u'D NACHUM SHIFRLN Parly Pveiertncei RE\nEduc|l\xa4rIAulh\xa4rIBu||n\u2022:smln\n\n'), (True, u'D DENNIS JACKSON PBM PYQUBIBMB: RE\nAnmspacn Gnnnml Mnnngar\n\n'), (True, u'D DAN HI.IGHI:S Pariy PVQIGIGMEZ RE\nSmnll Buslnosl 0wn\u2022r\n\n'), (True, u'D GREG CONLON Paf\xa5yPv8|BI01\\\u20acs2 RE\nEusIn\u2022\u2022|manlCPA\n\n'), (True, u'D JOHN EORUFF PBNYPVSIQIQHCE2 RE\nBuslnsnsmnn\n\n'), (True, u'D OSCAR ALEJANDR0 BRAUN Paw Prsfsfewsi RE\nBuslnassnunlkanchar\n\n'), (True, u'D MARSHA FEINLAND Pa|1yFr\xa29EI!mcZ F\nR\u2022!Ir\u2022d T|I|:h\u2022r\n\n'), (True, u'D DIANNE FEINSTEIN PSIYYPYBOBNNCB2 DE\nUnlt\u2022\xa4 Stalu S\u2022n.It\xa4r\n\n'), (True, u'D COLLEEN SHEA FERNALD FBIIYPVBOBIBHOQZ DE\nM\xa4th\u2022rIC\xa4n|uIlanIlAnIrI\n\n'), (True, u'D E|.IZAB\xa4IH l:InI<|:N Parly PIEIQWMEZ Rl:\nBu|In\u2022!|wovn|nlN\xa4npr0f|l Ex\u2022cu\\|v\u2022\n\n'), (True, u'D KAHIKUUUIN KANIM AI.! ParIyPrBYer8meZ P\nBuslnsstmnn\n\n'), (True, u'D RICK WILLIAMS PzrYy Pmlcmllcei RE\nButlniil Atlcmty\n\n'), (True, u'D RUu:\\.Iu I. GLUNIA Paliy Pre|2f8I\\0e\u2018 Kt\nGrnduah Stu\xa4\u2022nIIBu\u2022ln\u2022s\xbbman\n\n'), (True, u'D DON J. GRUNDMANN Paliy Plclnmhcl. 1\nDoctoral Chlroprlctlt\n\n'), (True, u'3 ROBERT LAUTEN Paw Pralsfsrlcs RE\n\n'), (True, u'3 GAIL K. LIGH I ruul Pifly PIMBVQOOBI L\\\nRnllnd Nur\\\u2022\n\n'), (True, u'D DAVID ALEX LEVITT PBM Prulvamncei DE\nC\xa4mput\u2022r Scl\u2022nIInIIEn\xa4In\u2022\u2022r\n\n'), (True, u'D ONLY IAIIL PZVIyPIBYEVEm2, Ht\nD0c10\xa2IAIt\xa4n1\u2022yIBu!In\u2022||w0min\n\n'), (True, u'D su\u2022m\u2022\u2022\u2022m\u2022n Paw Preiererw RE\n\n'), (True, u' \n\n')]), [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23), (24, 24), (25, 25)])], [((19, [(642, 813, 1233, 1824)], [(False, u'Candidates to the Unltsd Stats: Sonata\nVute for Ons (1)\n\n'), (True, u'C) IJIKK A|.L|:N NUNUFIK Fa|1YPf\xa4V\xa4IEl\u2019\\\xa4!I RE\nMBA Studint\n\n'), (True, u'8 DONALD KRAMFE PBRYPMWSVSIRGZ RE\nRnllnd Admlnlnmllon Dlmctcr\n\n'), (True, u'C) MIK: S I KIIIILING PBVYy FIBIMQIICD2 DEI\nC\xa4nsum\u2022r Rlghtl Attarnay\n\n'), (True, u'G DIANE STEWART ParYy PreVef2nC9` DEV\nBusln\u2022ssw\xa4manIFInanc\xa4 Manngar\n\n'), (True, u'Q MAK SHAH PirYy Prslofnncei DE\\\nEnvlrcnmantnl H\u2022|IIh Coniulunt\n\n'), (True, u'O NACHUNI SHIFREN FarIyFr\xa4|6fer\\0\xa4I RE\nEducatur/Auth\xa4rIBusIn\u2022|sm.|n\n\n'), (True, u'A\u2022r\xa41pac\u2022 G\u2022n\u2022raI Mar1|g\u2022r\n\n'), (True, u'O DAN HUGHES Parly Frslclsncsi RE\nSmall Busln\u2022\xa4s Owntr\n\n'), (True, u'GREG CDNLON Pa Preference: RE\nO BusIn\u2022s\u2022m\u2022nI\xa2PA ny\n\n'), (True, u'Joan \xa4\xa4uu\xbb=\xbb= my mmm; Ra\nO \xa4...\xbb...........\n\n'), (True, u'G OSCAR ALEJANDRU BRAUN Paffy PIBIMBIIBBZ RE\nBu||n\u2022t|m|nIRnnch\u2022r\n\n'), (True, u"C) MARSHA FEINLAND ParIyPre|erence\u2018 F\nRatlrtd 'I\u2019\u2022\u2022ch\u2022r\n\n"), (True, u'O DIANNE FEINSTEIN ParNPMel!r\\fAZ DEI\nl.|nlt\u2022d Stnn S\u2022ru\\\xa4r\n\n'), (True, u'O COLLEEN SHEA FERNALD Par\xa5yPfe|erer\\\xa2\xa4Z DEI\nM0th\u2022rIC\xa4n|uIhnUAn|||\n\n'), (True, u'Q ELIZABETH EIIIKEH Pa1P{Pr!!9mI\\(x1 RE\nBu|In\u2022:|w\xa4rnnnINOnpr\xa4|I\\ Ex\u2022\xa2u!Iv\u2022\n\n'), (True, u'O KABIRUDDIN KARIIII ALI PaHyPIE|e\xa2e\u2022\\0e\u2018 P\nBuslnssnman\n\n'), (True, u'RICK WILLIAMS PartyPm1\xa4mno\xa4: RE\nO Eunlrnsn Aliorndy\n\n'), (True, u'O ROGELI0 T, GLURIA PBIIYPMQYBHOBZ RE\nGraduam Slud\u2022mlBusIn\u2022s\u2022m|n\n\n'), (True, u'O DON J. GRUNDIIANN PartyPv\xa41\xa4mn0\xa4: I\nDoctor of Chlr\xa49ractlc\n\n'), (True, u"O ROBERT LAIJTEN Par\xa5yPMer!f\\\xa4\xa4' REI\n\n"), (True, u'GAIL K. LIGHTFOOT Pa\u2022\xa5yPMe!!m!: LI\nO Rdlnd Nuru\n\n'), (True, u'3 DAVID ALEX LEVITT Paliy Pveltflncei DEI\nCompuhr S\u2022:|\u2022ntIsIIEngIn\u2022\u2022r\n\n'), (True, u'G DRLV TAITZ ParIyPr\xa4i\xa4\xa4r\\\xa4eI RE\nD\xa4d\xa4rlAk\\\xa4n\\\u2022yIBunIr\\\u2022||w0rn|n\n\n'), (True, u'Au. \xbbummEz pa mmm; an\nO \xa4............... "\u2019\n\n'), (True, u'(_) \n\n')]), [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23), (24, 24), (25, 25)])], [((20, [(39, 846, 603, 1861)], [(False, u'Candidates in Qhe United States Senate\nV\xa4t\xa4 for One (1)\n\n'), (True, u'O DIRK ALLEN KONOFIK Parly Preference:\nMBA Student\n\n'), (True, u'\xa9 DUNALU KRAMPE Pany Prelevehcsi\nRltlrld Admlnlstrlllnn DIr\u2022|:t\xa4r\n\n'), (True, u'O MIKE STRINILING Paw Preference:\nConsumar Rlqhu Alicrnay\n\n'), (True, u'O DIANE STEWART Pavly Prelerencei\nBunln\u2022snw\xa4rnanIFInnnc\u2022 Mana\xa4\u2022r\n\n'), (True, u'\xa9 NAR SHA!-I Pavly Prelemrviei\nEnvlr0nm\u2022nI|I Htillh Canlultant\n\n'), (True, u'Q NACHUNI SHIFREN PBr\\y Preference\nEducaI\xa4rlAu\\h\xa4\u20221Bu\xa4|n\u2022||m.|n\n\n'), (True, u'A\u2022r0|p\u2022\u2022:\u2022 G\u2022n\u2022r|I Managor\n\n'), (True, u'C) DAN HUGHES Party PIeIel!r\\\xa2e`\nSmall Buslrnsx Dwntr\n\n'), (True, u'Bu!ln\u2022L\xa4ni\u2022\\lCPA\n\n'), (True, u'O \xa4............... "\u2019\n\n'), (True, u'O OSCAR ALEJANDR0 BRAUN Party Pmleience.\nBuslntsimanlknnchtr\n\n'), (True, u'S NIARSHA FEINLAND Parly Pmhrenoei\nRdlmd Y\u2022ach\u2022r\n\n'), (True, u'G DIANNE FEINSTEIN Parly Pmfemmei\nUnlhd Shih: Senator\n\n'), (True, u'O COLLEEN SHEA FERNALD Parly Prefnrumei\nM\xa4tn\u2022rlC\xa4n\u2022uItantIArlI|I\n\n'), (True, u'O ELIZABETH EMKEN PartyPve1er\xa4n0e\nBuslnnsswcmanhlcnprdh Ex\u2022\u2022:u(|v\u2022\n\n'), (True, u'Bullnltsman\n\n'), (True, u'Q RICK WILLIAMS Paw Pvefeiencei\nBuslnan Attcrvwy\n\n'), (True, u'Gmduah Sludtnl/Bunlnasnmnn\n\n'), (True, u'O DON J. GRUNDMANN Party Pmfemncei\nDoctor \xa4f Chlropmnlc\n\n'), (True, u'\n'), (True, u'O GAIL K, LIGHTFOOT ParIyPMemncei\nRnlhnd Nuru\n\n'), (True, u'O DAVID ALEX LEVITT Party Preierencej\nCompuhr Sc|\u2022n\\||IIEngln\u2022\u2022r\n\n'), (True, u'O ORLY TAI11 Party Pvsferencet\nD\xa4ci\xa4rIA!t\xa4n\u2022\u2022yIBu|Ir\u2022\u2022||w\xa4m|n\n\n'), (True, u'O \xa4............... "*\n\n'), (True, u'( \n\n')]), [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23), (24, 24), (25, 25)])]]
        elif self.grouping_cached:
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

        print groups, "GROUPS IS", groups
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
        self.mapping = mapping
        self.mapping_inverse = dict((v,k) for k,v in mapping.items())

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

        self.multiboxcontests = [[(c,self.contest_order[c][b]) for c,b in y] for y in self.multiboxcontests_enter]

        self.equivs = [[mapping[bid,bboxes[0]] for bid,bboxes in group] for group in gr]
        
        print "MAPPING", self.mapping
        print "MAPPING_INVERSE", self.mapping_inverse
        print "REORDER", self.reorder
        print "REORDER_INVERSE", self.reorder_inverse
        print "EQUIVS", self.equivs

        def putresults(data):
            print "I get the data", data
            self.validequivs = dict(data)

        if any(len(x) > 1 for x in self.equivs):
            frame = wx.Frame (None, -1, 'Verify Contest Grouping', size=(1024, 768))
            VerifyContestGrouping(frame, self.proj.ocr_tmp_dir, self.dirList, self.equivs, self.reorder, self.reorder_inverse, self.mapping, self.mapping_inverse, putresults)
            frame.Show()

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
            for i,each in enumerate(self.equivs):
                if any(x in cur for x in each):
                    print 'found', each
                    if i in self.validequivs:
                        eqclass = [x for x,y in zip(each, self.validequivs[i]) if y]
                    else:
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
                print 'lookup', set_repr, each_in_dict
                print self.reorder[set_repr]
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

