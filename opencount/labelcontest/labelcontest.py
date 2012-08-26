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
        self.validequivs = {}


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
            extension, newgroup = extend_multibox(self.grouping_cached,
                                        self.mapping_inverse[(self.templatenum, self.contest_order[self.templatenum][self.count])],
                                        self.mapping_inverse[(self.templatenum, self.contest_order[self.templatenum][self.count+1])],
                                        orders)
            print "EXTENSION", extension
            print "NEWGROUP", newgroup
            self.multiboxcontests_enter += [tuple([(self.mapping[x][0], self.contest_order[self.mapping[x][0]].index(self.mapping[x][1])) for x in pair]) for pair in extension]
            
            print "MULTIBOX"
            print self.multiboxcontests_enter
            boxes_in_new_group = [bid_cid for pair in extension for bid_cid in pair]
            print boxes_in_new_group
            cleared = []
            for group in self.groups_saved:
                new_group = []
                for ((bid,boxes,text),order) in group:
                    if (bid,boxes[0]) in boxes_in_new_group:
                        print (bid,boxes[0]), 'skip'
                    else:
                        new_group.append(((bid,boxes,text),order))
                if new_group != []:
                    cleared.append(new_group)
            print "CLEARED", [[(bid,boxes,order) for ((bid,boxes,text),order) in group] for group in cleared]
            fixed = cleared + [newgroup]
            print "FIXED", [[(bid,boxes,order) for ((bid,boxes,text),order) in group] for group in fixed]
            #print "BEFORE", [[(bid,boxes,order) for ((bid,boxes,text),order) in group] for group in self.groups_saved]
            #self.compute_equivs(None)
            self.groups_saved = fixed
            self.compute_equivs_2()
            #print "AFTER", [[(bid,boxes,order) for ((bid,boxes,text),order) in group] for group in self.groups_saved]

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
        mapping = {'english': 'eng', 'spanish': 'esp', 'korean': 'kor', 'chinese': 'chi_sim', 'vietnamese': 'vie'}
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
                            language = row[take+1].lower()
                            print 'found with lang', language
                            if language in mapping:
                                language = mapping[language] 
                            result[row[0]] = language
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
            groups = [[((0, [(1109, 1085, 1598, 1829)], [(False, u'MEMBER OF THE STATE ASSEMBLY\n72nd District\nVoin kx Ons\n\n'), (True, u'E \xb7rRAvls ALLEN\nParty Preference: Republican\nSmall Business Owner\n\n'), (True, u'm ALBERT AYALA\nParty Preference: Democratic\nRetired Police Commander\n\n'), (True, u'E JOE \xa40v1NH\nParty Preierence: Democratic\nCity Commissioner/Businessperson\n\n'), (True, u'E LONc PHAM\nParty Prelerence: Republican\nMember, Orange County Board oi\nEducation\n\n'), (True, u'\xa4 mov EDGAR\nParty Preference: Republican\nBusinessman/Mayor\n\n'), (True, u'E\n\n')]), [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]), ((2, [(1110, 1089, 1601, 1832)], [(False, u'MEMBER OF THE STATE ASSEMBLY\n72nd Disfrict\nVote for Ons\n\n'), (True, u'E TRAv\xa4s ALLEN\nParty Prelerervce: Republi n\nSmall Business Owner\n\n'), (True, u'\xa4 ALBERT AYALA\nParty Preference; Democratic\nRetired Police Commander\n\n'), (True, u'E Joe 00vmH\nParty Preference: Democratic\nCity Commissioner/Businessperson\n\n'), (True, u'\xa4 Lows PHAM\nParty Preferencet Republican\nMember, Orange County Board or\nEducation\n\n'), (True, u'\xa4 mov sucm\nParly Preference: Republican\nBusinessman/Mayor\n\n'), (True, u'EI\n\n')]), [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)])], [((0, [(1109, 280, 1598, 590)], [(False, u'\n'), (True, u'Z DIANNE Ferwsrerw\nParty Preference: Democratic\nUnited States Senator\n\n'), (True, u'E c0Lu;&N SHEA FERNALD\nParty Preference: DemOcfa\\ic\nMother/Consultant/Artist\n\n'), (True, u'EI\n\n')]), [(0, 0), (1, 1), (2, 2)]), ((2, [(1110, 281, 1601, 593)], [(False, u'\n'), (True, u'\xa4 DIANNE Fenwsrsm\nParty Prelerencez Democratic\nUnited States Senator\n\n'), (True, u'Q c0LLEEN SHEA FERNALD\nPany Preference: Democratic\nMother/Consultant/Artist\n\n'), (True, u'Z\n\n')]), [(0, 0), (1, 1), (2, 2)])], [((0, [(1109, 2127, 1598, 2471)], [(False, u'Judge 0Hhs Superior Court\nOffice N0.1\nV0te|0rOne\n\n'), (True, u'E Eucseue .uz\u2022-\u2022A\xbb<\nGeneral Practice Attomey\n\n'), (True, u'm DEBORAH J, cHuAN<;\nJudge ul the Superior Court\n\n'), (True, u'III\n\n')]), [(0, 0), (1, 1), (2, 2)]), ((2, [(1110, 2131, 1601, 2474)], [(False, u'Judgaoimcsenpovicrccnxt\nOffIt\xbbN0.1\nV\xa4m|or0ne\n\n'), (True, u'\xa4 EUGENE JIZHAK\nGeneral Practnce Attorney\n\n'), (True, u'2 \xa4za0RAH J. cnumcs\nJudge ol the Superior Court\n\n'), (True, u'E\n\n')]), [(0, 0), (1, 1), (2, 2)]), ((3, [(1107, 2187, 1598, 2530)], [(False, u'Judgs0Hh\xa4Sup\u2022ri0rCo\u2022n\nO|1lcoN0.1\nv\xa4n\xa4brOne\n\n'), (True, u'D EUGENE JIZHAK\nGeneral Practice Attorney\n\n'), (True, u'\xa4 DEBORAH J. cuumc\nJudge 0I the Superior Court\n\n'), (True, u'I2\n\n')]), [(0, 0), (1, 1), (2, 2)]), ((4, [(1107, 2188, 1598, 2532)], [(False, u'JudgaoI\u20221*\xa4aS\u2022.;>ori01C\xa4\xa4rt\nOfhcaN0.1\nV\xa4ts|0fOnc\n\n'), (True, u'E Euseme .uzHA\xbb<\nGeneral Practice Attorney\n\n'), (True, u'E mzaomxu .1.CHUANG\nJudge ol the Superior Court\n\n'), (True, u'III\n\n')]), [(0, 0), (1, 1), (2, 2)]), ((5, [(1114, 2188, 1604, 2532)], [(False, u'Judge ofthe Superior Court\nOfhce N0.1\nV\xa4tel\xa4rOne\n\n'), (True, u'\xa4 EUGENE Jnzwxx\nGeneral Practice Atiomey\n\n'), (True, u'E oEB0RAH J. cHuANG\nJudge of the Superior Court\n\n'), (True, u'III\n\n')]), [(0, 0), (1, 1), (2, 2)])], [((1, [(1103, 280, 1592, 593)], [(False, u'\n'), (True, u'\xa4 Romam |.AuTEN\nParty Prelercncez Repuxlican\n\n'), (True, u'\xa4 GAIL K u<;mi=00T\nParty Preterencei Libertarian\nRetired Nurse\n\n'), (True, u'III\n\n')]), [(0, 0), (1, 1), (2, 2)]), ((3, [(1107, 278, 1598, 595)], [(False, u'\n'), (True, u'\xa4 Roszm murem\nParty Preference. Republican\n\n'), (True, u'D Gm K. ucurroor\nParty Preference: Libertarian\nRetired Nurse\n\n'), (True, u'EI\n\n')]), [(0, 0), (1, 1), (2, 2)]), ((4, [(1107, 285, 1598, 597)], [(False, u'\n'), (True, u'E ROBERT LAUTEN\nParty Preference: Republican\n\n'), (True, u'\xa4 Gm K, uGHn=00T\nParty Prererenw. Ubedarian\nRetired Nurse\n\n'), (True, u'Z\n\n')]), [(0, 0), (1, 1), (2, 2)])], [((1, [(1103, 1088, 1592, 1491)], [(False, u'STATE SENATOR\n29m District\nVob br Ona\n\n'), (True, u"E GREG DIAMOND\nParty Preterence: Democratic\nW0rkers' Rights Attorney\n\n"), (True, u'\xa4 ROBERT \xb7\xb7BOB\xb7 \u2022-\u2022ur=r=\nParty Preference: Republican\nLawmaker/Business Owner\n\n'), (True, u'Z\n\n')]), [(0, 0), (1, 1), (2, 2)]), ((3, [(1107, 1089, 1598, 1492)], [(False, u'STATE SENATOR\n29Ih District\nVote Iur Ons\n\n'), (True, u"Q case DIAMOND\nParty Preference: Democratic\nW0rkers' Rights Attorney\n\n"), (True, u'D Rosaar \xb7\xa40\xa4\xb7 Hur:\nParty Preference: Republican\nLawmaksr/Business Ounev\n\n'), (True, u'Z\n\n')]), [(0, 0), (1, 1), (2, 2)]), ((4, [(1107, 1090, 1598, 1493)], [(False, u'STATE SENATOR\n29th Disfrict\nVote for Ons\n\n'), (True, u'\xa4 GREG DIAMOND\nParty Prelerencez Democratic\nWort<ers\u2018 Rights Attorney\n\n'), (True, u'E Roaenr "BOB" Hur=F\nParty Preference: Republican\nLawmakerlBusiness Owner\n\n'), (True, u'III\n\n')]), [(0, 0), (1, 1), (2, 2)])], [((1, [(1103, 1485, 1592, 1889)], [(False, u'MEMBER OF THE STATE ASSEMBLY\n55\\h Distrid\nVous for One\n\n'), (True, u'\xa4 cum HAGrv1AN\nParty Preference: Repubhcan\nBusiness Owner/Assemblyman\n\n'), (True, u'E GREGG D. Fnncr-me\nParly Preference: Democratic\nSocial Worker\n\n'), (True, u'E\n\n')]), [(0, 0), (1, 1), (2, 2)]), ((3, [(1107, 1486, 1598, 1890)], [(False, u'MEMBER OF THE STATE ASSEMBLY\n551h District\n[gm kx Ons\n\n'), (True, u'\xa4 cum HAGMAN\nParty Preference: Republi  n\nBusiness Oomev/Assemblyman\n\n'), (True, u'\xa4 emacs D \xbb=\xa4nc\xbb\u2014sLE\nParty Pretevence; Democratic\nSoda! Worker\n\n'), (True, u'IZ\n\n')]), [(0, 0), (1, 1), (2, 2)]), ((4, [(1107, 1487, 1598, 1891)], [(False, u'MEMBER OF THE STATE ASSEMBLY\n55\\h District\nVous for Ons\n\n'), (True, u'E cum HAGMAN\nParty Preference: Republican\nBusiness Owner/Assemblyman\n\n'), (True, u'Q GREGG D. \xbb=Rnc\xbb\u2014n.E\nParty Preference; Democratic\nSocial Worker\n\n'), (True, u'lj\n\n')]), [(0, 0), (1, 1), (2, 2)])], [((1, [(1103, 2185, 1592, 2529)], [(False, u'JlK$\xb0O0\u2018IYBS|.Q0|i\xa2\xa5C(Xl\u2018{\n0||i0\xa4N0.1\nV0hf\xa4fOna\n\n'), (True, u'E EUGENE .nzm\xbb<\nGeneral Practace Attorney\n\n'), (True, u'E 0r;B0RAH J. cHuANG\nJudge ol me Superior Coun\n\n'), (True, u'EI\n\n')]), [(0, 0), (1, 1), (2, 2)])], [((5, [(1114, 284, 1604, 596)], [(False, u'\n'), (True, u'\xa4 GREG commu\nParty Preference: Republican\nBusmessman/CPA\n\n'), (True, u'\xa4 Jon-in \xa40Ru\xbb=\u2022=\nParty Preference: Republican\nBusinessman\n\n'), (True, u'IZ\n\n')]), [(0, 0), (1, 1), (2, 2)])], [((5, [(1114, 1092, 1604, 1494)], [(False, u'STATE SENATOR\n3701 District\nVots for Ons\n\n'), (True, u'E Mwu wALTERs\nParty Preference: Repubhcan\nBusinesssucrnan/Senator\n\n'), (True, u'E s\xb7rEvE voumc\nParty Preference: Democratic\nCivil Justice Attorney\n\n'), (True, u'EI\n\n')]), [(0, 0), (1, 1), (2, 2)])], [((5, [(1114, 1488, 1604, 1890)], [(False, u'MEMBER OF THE STATE ASSEMBLY\n68th Distric!\n[ow br One\n\n'), (True, u'\xa4 cunnsnm AVALOS\nParty Prelerenoez Democratic\n\n'), (True, u'Q DONALD P. (Dom WAGNER\nParty Prelerence: Republican\nAssembly Member\n\n'), (True, u'III\n\n')]), [(0, 0), (1, 1), (2, 2)])], [((0, [(1109, 584, 1598, 1091)], [(False, u'UNITED STATES REPRESENTATIVE\n48th District\nVote lor One\n\n'), (True, u'\xa4 DANA R0:-aRABAcHeR\nPany Preference: Republican\nU.S. Representative\n\n'), (True, u'\xa4 ALAN SCHLAR\nParty Preference: None\nMarketing Sales Executive\n\n'), (True, u'D RON vARAs1\xb7EH\nPariy Preference: Democratic\nEngineer/Small Businessman\n\n'), (True, u'IZ]\n\n')]), [(0, 0), (1, 1), (2, 2), (3, 3)]), ((2, [(1110, 587, 1601, 1095)], [(False, u'UNITED STATES REPRESENTATIVE\n48Ih District\nVon Inr Ons\n\n'), (True, u'Q mm ROHRABACHER\nParty Preterencet Republican\nUS. Representative\n\n'), (True, u'\xa4 ALAN scH1.AR\nParty Prelerence None\nMarketing Sales Executive\n\n'), (True, u'E Rom vARAsm\xb71\nParty Preference: Democratic\nEngineer/Small Businessman\n\n'), (True, u'EI\n\n')]), [(0, 0), (1, 1), (2, 2), (3, 3)])], [((1, [(1103, 587, 1592, 1094)], [(False, u'UNITED STATES REPRESENTATIVE\n39th Dlstdci\nVqts Icr Ona\n\n'), (True, u'\xa4 \xa4\xb7MARra Mumwnenn\nParty Preference: None\nCommunaty Volunteer\n\n'), (True, u'E JAY cum\nParty Preference: Demouatic\nBusinessman/School Boardmember\n\n'), (True, u'\xa4 ED Rovcra\nParty Preference: Repubhmn\nU.5, Representative\n\n'), (True, u'[II\n\n')]), [(0, 0), (1, 1), (2, 2), (3, 3)]), ((3, [(1107, 589, 1598, 1095)], [(False, u'UNITED STATES REPRESENTATIVE\n39th District\nVote Ior Ono\n\n'), (True, u'Q D\xb7MAR\xa4E MuLAmERn\nParty Preference: None\nCommunity Vulunteer\n\n'), (True, u'\xa4 Jn cwan\nParty Preisrencex Democratic\nBusinessman/Scmol Bnammember\n\n'), (True, u'E so Royce\nPady Preference: Republican\nU.S. Representative\n\n'), (True, u'IZ\n\n')]), [(0, 0), (1, 1), (2, 2), (3, 3)]), ((4, [(1107, 591, 1598, 1096)], [(False, u'UNITED STATES REPRESENTATIVE\n3901 District\nVote lor One\n\n'), (True, u'\xa4 cmmne Mun.AmER\xa4\nParty Prelerence; None\nCommunity Volunteer\n\n'), (True, u'E .uAv cum\nParty Preference: Democratic\nBusinessman/School Bcardmember\n\n'), (True, u'D ED Rovcz\nParty Preference: Repubhcan\nU.S. Representative\n\n'), (True, u'Z\n\n')]), [(0, 0), (1, 1), (2, 2), (3, 3)])], [((5, [(139, 1152, 630, 1451)], [(False, u'PRESIDENT OF THE UNITED STATES\nVote lor Ons\n\n'), (True, u'E sTEwART ALEXANDER\n\n'), (True, u'E STEPHEN DURHAM\n\n'), (True, u'E ROSS c. \xb7R0c\u2022<Y\xb7 ANDERSON\n\n'), (True, u'EI\n\n')]), [(0, 0), (1, 1), (2, 2), (3, 3)])], [((5, [(1114, 590, 1604, 1098)], [(False, u'UNITED STATES REPRESENTATIVE\n45th Distric!\nVon lor Ons\n\n'), (True, u'D .10HN wana\nParty Preference: Republican\nSmall Business Owner\n\n'), (True, u'\xa4 sum-use KANG\nParty Preference: Democratic\nMayor ol Irvine\n\n'), (True, u'E .100-1N CAMPBELL\nParty Preference: Republican\nBusinessmanIU.S, Representative\n\n'), (True, u'EZI\n\n')]), [(0, 0), (1, 1), (2, 2), (3, 3)])], [((0, [(620, 280, 1115, 2642)], [(False, u'UNITED STATES SENATOR\nVote for One\n\n'), (True, u'D ELIZABETH EMKEN\nParty Preierence Republican\nBusinessw0rnanINcnpr\xa4Ht Executive\n\n'), (True, u'E KABIRUDDIN \u2022<ARuv\xa4 ALI\nParty Preference: Peace and Freedom\nBusinessman\n\n'), (True, u'E Rncx w|LuAMs\nParty Prelerencer Republican\nBusiness Aticmey\n\n'), (True, u'E R0<sEu0 T. G\xa4.0R\xa4A\nPany Preierence: Republican\nGraduate Student/Businessman\n\n'), (True, u'E 00N J. GRUNDMANN\nParty Preference: American lndependen\nDoctor 01 Chiropractic\n\n'), (True, u'D Roazm LAUTEN\nParty Preference; Republican\n\n'), (True, u'E ami. K. ucv-m=00T\nParty Preference: Libertarian\nRetired Nurse\n\n'), (True, u'E \xa4Avn\xa4 max Lsvnrr\nParty Prelerence; Democratic\nComputer S<:ientistlEngineer\n\n'), (True, u'\xa4 0RLY mrz\nParty Preference: Republi \xbb n\nDoctor/Attorney/Businessworrran\n\n'), (True, u'E AL RAMrREz\nParty Preference; Republican\nBusinessman\n\n'), (True, u'D 01RK ALLEN \u2022<0NOPn<\nParty Preference: Republican\nMBA Student\n\n'), (True, u'E DONALD KRAMPE\nParty Preference: Republican\nRetired Administration Director\n\n'), (True, u'Q MIKE smnmuwc;\nParty Preference; Demouatic\nConsumer Rights Attomey\n\n'), (True, u'Q 0:ANE srew/mr\nParty Preference; Democratic\nBusinesswoman/Finance Manager\n\n'), (True, u'Q NAK sHA\xbb-1\nParty Prelevenoe; Democratic\nEnvironmental Health Consultant\n\n'), (True, u'\xa4 mcnum sunmzm\nParty Preierence: Republican\nEducator/Author/Businessrnan\n\n'), (True, u'E DENNIS JAc\u2022<s0N\nParty Preference Republican\nAerospace General Manager\n\n'), (True, u'\xa4 DAN HUGHES\nParty Pralerence: Republican\nSmall Business Owner\n\n'), (True, u'E GREG <:0r~1L0N\nParty Preference: Republican\nBusmessman/CPA\n\n'), (True, u'E Jo}-iN BORUFF\nPariy Preference, Republican\nBusinessman\n\n'), (True, u'D OSCAR ALEJANDRO BRAUN\nParty Preference: Republican\nBusinessman/Rancher\n\n'), (True, u'\xa4 MARSHA Fznwumn\nParty Prelerence: Peace and Freedom\nRehred Teacher\n\n')]), [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21)]), ((1, [(614, 280, 1109, 2645)], [(False, u'UNITED STATES SENATOR\nVote for Ono\n\n'), (True, u'E \xa4Avn0 Auax uavrrr\nParty Preference: Democraiic\nComputer Scientist/Engineer\n\n'), (True, u'\xa4 0RLv TA|rz\nParty Preference: Repdzlium\nDoctor/Attorney/Bsnsinesswunarn\n\n'), (True, u'E AL Rmvnasz\nParty Preference: Republican\nBusinessman\n\n'), (True, u'E DIRK ALLEN K0N0n\xa4n<\nParty Preference: Republican\nMBA Student\n\n'), (True, u'D DONALD KRAMPE\nParty Preterence: Republican\nRetired Administration Director\n\n'), (True, u'E Mme smnmuwc\nParty Prelerencez Democratic\nConsumer Rights Attorney\n\n'), (True, u'E DIANE STEWART\nParty Preference; Democratic\nBusinesswoman/Finance Manager\n\n'), (True, u'Z NAK slum\nParty Prelerence: Democratic\nEnvironmental Health Consultant\n\n'), (True, u'E NAc+\u2022uM sHnr=REN\nParty Preference: Republican\nEducator/Autfror/Businessrnan\n\n'), (True, u'E DENNIS JAcr<s0N\nParty Prelerence: Republican\nAerospace General Manager\n\n'), (True, u'E DAN HUGHES\nParty Preference: Republican\nSmall Business Owner\n\n'), (True, u'E GREG comow\nParty Preference: Republican\nBusinessman/CPA\n\n'), (True, u'E Jon-aw a0Ru\xbb=\xbb=\nPany Prelerenoe: Republi n\nBusinessman\n\n'), (True, u'Z 0scAR ALEJANDRO BRAuN\nParty Prelerencez Repubhcan\nBusinessman/Rancher\n\n'), (True, u'E MARsuA FEINLAND\nParty Preference; Peace and Freedom\nRetired Teacher\n\n'), (True, u'E \xa4nANNE Franwsrenw\nParty Preference: Democratic\nUnited States Senator\n\n'), (True, u'\xa4 c0|.LEEr~1 SHEA FERNALD\nParty Preference: Democratic\nMother/Consultantlmtist\n\n'), (True, u'E EUZABETH EMKEN\nParty Preference: Republican\nBusinesswcerman/Nor1pr0Ii\\ Execume\n\n'), (True, u'E \u2022<Aa1Ru\xa4mN Kmnm ALI\nParty Preference: Peace and Freedom\nBusinessman\n\n'), (True, u'D Rncx wu.uAMs\nParty Prdecenoex Republican\nBusiness Attorney\n\n'), (True, u'E R0GEu0 T. c;|.0R|A\nParty Prelerencez Republican\nGraduate StudentIBusinessman\n\n'), (True, u'E DON J. GRUNDMANN\nParty Preference: American lndependem\nDoctor ol Chnropractic\n\n')]), [(0, 15), (1, 16), (2, 17), (3, 18), (4, 19), (5, 20), (6, 21), (7, 0), (8, 1), (9, 2), (10, 3), (11, 4), (12, 5), (13, 6), (14, 7), (15, 8), (16, 9), (17, 10), (18, 11), (19, 12), (20, 13), (21, 14)]), ((2, [(621, 281, 1116, 2645)], [(False, u'UNITED STATES SENATOR\nVote lor One\n\n'), (True, u'Q Er.|zABErH EMKEN\nParty Preference Republican\nBusinesswoman/Nonproht Executive\n\n'), (True, u'E r<ABrRu\xa4\xa4\xa4N r<ARrM ALI\nParty Preference: Peace and Freedom\nBusinessman\n\n'), (True, u'Q Rncx w\xbbLuAMs\nParty Preference: Republican\nBusiness Allcmey\n\n'), (True, u'E ROGELIO T. G|.0R|A\nParty Preference; Republican\nGraduate Studen|IBusin$sma\\\n\n'), (True, u'D DON .1. <sRuN\xa4MANN\nParty Preierence: American Independen\nDodor 01 Chiropraciic\n\n'), (True, u'E Roarzm murem\nParty Preference; Republican\n\n'), (True, u'E GAIL K. uGHTF00T\nParry Preference: Libertarian\nRetired Nurse\n\n'), (True, u'D \xa4Avn0 Auax Levm\nParty Preference: Democratic\nComputer Scientist/Engineer\n\n'), (True, u'Z 0Ra.v mrz\nParty Preference: Repubhcan\nDoctorlAttomey/Businesswcrnan\n\n'), (True, u'E AL RAMnRE2\nParty Preference; Republican\nBusinessman\n\n'), (True, u'E max ALLEN \u2022<0N0Pn<\nParty Preference; Republican\nMBA Student\n\n'), (True, u'E DONALD KRAMPE\nParty Preference: Republican\nRetired Adminnstration Director\n\n'), (True, u'E mma STRIMLING\nParty Preference: Democratic\nConsumer Rights Attorney\n\n'), (True, u'E name sTEwARr\nParty Preference: Democratic\nBusinessw0manlFinanoe Manager\n\n'), (True, u'E NAK si-im\nParty Preference: Democratic\nEnvironmental Health Consultant\n\n'), (True, u'\xa4 NAc\u2022-cum snnmew\nParty Preference: Republican\nEducator/Autfnor/Businessunan\n\n'), (True, u'E neuuns mcxm\nParty Prelerenuez Republican\nAerospace General Manager\n\n'), (True, u'E DAN Hucr-ues\nParty Prekzrenuez Republican\nSmall Business Owner\n\n'), (True, u'Z GREG c0N\xa4.0N\nParty Preierencex Republican\nBusinessman/CPA\n\n'), (True, u'\xa4 .10HN B0Rui=r\nPany Preference: Republican\nBusinessman\n\n'), (True, u'E 0scAR ALEJANDR0 BRAUN\nParty Preference: Republican\nBusinessmanIRancher\n\n'), (True, u'E MARSHA FEINLAND\nParty Preference: Peace and Freedom\nRetired Teacher\n\n')]), [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21)]), ((3, [(619, 278, 1113, 2647)], [(False, u'UNITED STATES SENATOR\nVote for One\n\n'), (True, u'\xa4 \xa4Av\u2022c> ALEX Lzzvm\nParty Preference: Democratic\nComputer Scieniist/Engineer\n\n'), (True, u'E 0RLv Tmz\nParty Preference; Republican\nDoctor/Att0rneylBusinessw\xa4\u2022\u2018nan\n\n'), (True, u'Q AL RAMrR&z\nParty Preference; Republican\nBusinessman\n\n'), (True, u'E \xa4u=<\xbb< ALLEN \u2022<0r~10\u2022>n<\nPany Preference: Republican\nMBA Student\n\n'), (True, u'E DONALD KRAMPE\nParty Preference; Republican\nRetired Adminisvation Direcwr\n\n'), (True, u'D MIKE sTRrMuNG\nParty Preference: Democratic\nConsumer Rights Attorney\n\n'), (True, u'E muws STEWART\nParty Prelerencez Demcualic\nBusinesswcman/Finance Manager\n\n'), (True, u'E MAK sum\nParty Preterenoe; Democratic\nEnvironmental Health Consultant\n\n'), (True, u'D mcnum SHIFREN\nParty Prelerencez Republican\nEducator/Auttwor/Busirmessrnan\n\n'), (True, u'E DENNIS JACKSON\nParty Preference: Republican\nAerospace General Manager\n\n'), (True, u'E DAN Hucsuss\nParty Preference; Republkzan\nSmall Busmess Owner\n\n'), (True, u'Z GREG c0NL0N\nParty Prelerenoe: Republican\nBusinessman/CPA\n\n'), (True, u'\xa4 J0HN Bc>Rur=F\nParty Preference: Republican\nBusinessman\n\n'), (True, u'E OSCAR ALEJANDRO awww\nParty Preference: Republican\nBus6nessmanIRa1cher\n\n'), (True, u'D Mmsm Famumu\nPa\u20221y Preference: Peace and Freedom\nRetired Teacher\n\n'), (True, u'Q DIANNE Fenwsmm\nParty Pretsrence: Democratic\nUnited States Senator\n\n'), (True, u'\xa4 c0LL&EN sun FERNAL0\nParty Preference; Democratic\nMother/Consultant/Artist\n\n'), (True, u'E E\xa4.nzABETH EMKEN\nParty Preierenoe: Republican\nBusinessw0manIN0npr0Ht Executive\n\n'), (True, u'E KABIRUDDIN KARIM ALI\nParty Preference: Peace and Freedom\nBusinessman\n\n'), (True, u'E RICK wu.uAMs\nParty Preference: Republican\nBusiness Attorney\n\n'), (True, u'E Rocssuo T. cLc>RnA\nParty Preference: Repubhcan\nGraduate S!uden\\IBusinessman\n\n'), (True, u'D mom J. GRUNDMANN\nParty Preference: American lndepcnden\nDoctor of Chircpradic\n\n')]), [(0, 15), (1, 16), (2, 17), (3, 18), (4, 19), (5, 20), (6, 21), (7, 0), (8, 1), (9, 2), (10, 3), (11, 4), (12, 5), (13, 6), (14, 7), (15, 8), (16, 9), (17, 10), (18, 11), (19, 12), (20, 13), (21, 14)]), ((4, [(619, 285, 1113, 2647)], [(False, u'UNITED STATES SENATOR\nVote for One\n\n'), (True, u'E \xa4Av\xa4\xa4 ALEX Lsvnrr\nParty Prelerence; Democratic\nComputer Scientist/Engineer\n\n'), (True, u'E 0\u2022=<n.v mrz\nPany Preference; Republican\nDoctor/Atiorney/Busirmessxuornan\n\n'), (True, u'Z AL Rnuunzz\nParty Prelerenuez Republimn\nBusinessman\n\n'), (True, u'E DIRK ALl.EN Kouovnx\nParty Preference: Republi \xa4 n\nMBA Student\n\n'), (True, u'E DONALD KRAMPE\nParty Preference: Republi n\nRetired Administralion Direcfor\n\n'), (True, u'E MIKE sTR\xa4M|.1N<;\nParty Preference; Democratic\nConsumer Rights Attomey\n\n'), (True, u'\xa4 DIANE srEwARr\nParty Preference: Democratic\nBusinesswoman/Finance Manager\n\n'), (True, u'E NAK SHAH\nParty Preference: Democratic\nEnvironmental Health Consultant\n\n'), (True, u'\xa4 NAcHuM SHIFREN\nPar1y Preference: Republican\nEducator/Author/Businessman\n\n'), (True, u'E DENNIS JACKSON\nParty Preference; Republican\nAerospace General Manager\n\n'), (True, u'E DAN Hucuzs\nParty Prelerence: Republican\nSmall Business Owner\n\n'), (True, u'\xa4 GREG c0NL0r~1\nParty Prelerencez Republican\nBusinessman/CPA\n\n'), (True, u'E JOHN B0RuFr\nParty Prelerencez Republican\nBusinessman\n\n'), (True, u'2 oscm ALEJANDR0 sRAuN\nParty Prslerenoe: Republican\nBusinessman/Rancher\n\n'), (True, u'\xa4 MARSHA Ferwumu\nParty Preference; Peace and Freedom\nRetired Teacher\n\n'), (True, u'D DIANNE \u2022=E1NsrE1N\nParty Preference: Democratic\nUnited States Senator\n\n'), (True, u'E c0u.EEN SHEA FERNALD\nParty Preference: Democratic\nMoiher/C0nsu|tanIIAr1isl\n\n'), (True, u'E EuzAsErH EMKEN\nParty Preference: Republican\nBusiness\xbbumnanIN<x1pro|it Executive\n\n'), (True, u'E KABIRUDDIN \xbb<AR\xa4M ALI\nPany Preference: Peace and Freedom\nBusinessman\n\n'), (True, u'E RICK wnnuxxms\nParty Preference: Republican\nBusiness Atiomey\n\n'), (True, u'E R0<;Eu0 T. cn.0RaA\nParty Preference; Republican\nGraduate Student/Businessman\n\n'), (True, u'\xa4 \xa40N Jr cnuwummu\nParty Preference. American Independen\nDoctor of Chiropractic\n\n')]), [(0, 15), (1, 16), (2, 17), (3, 18), (4, 19), (5, 20), (6, 21), (7, 0), (8, 1), (9, 2), (10, 3), (11, 4), (12, 5), (13, 6), (14, 7), (15, 8), (16, 9), (17, 10), (18, 11), (19, 12), (20, 13), (21, 14)]), ((5, [(624, 284, 1120, 2649)], [(False, u'UNITED STATES SENATOR\nVote for One\n\n'), (True, u'Q oscm ALEJANDRO BRAUN\nParty Preference: Republican\nBusinessman/Rancher\n\n'), (True, u'E MARs\xbb\u20141A FEnr~u.AN0\nParty Preference: Peace and Freedom\nRetired Teacher\n\n'), (True, u'\xa4 Dimmu; r=ErNsTr;n~1\nParty Preference: Democratic\nUnited States Senator\n\n'), (True, u'E c0LLsEN sa-on FERNALD\nParty Prelerenoez Democratic\nM0lherIC\xa4nsuItant/Artist\n\n'), (True, u'D ei.nzABem EMKEN\nParty Preference: Republican\nBusinesswornan/Nonprofit Executive\n\n'), (True, u'\xa4 KABIRUDDIN KARIM ALI\nParty Prelerencez Peace and Freedom\nBusinessman\n\n'), (True, u'E RICK wu.uAMs\nParty Preference: Republican\nBusiness Attomey\n\n'), (True, u'Z Roceuo TA c\xa4.0RnA\nParty Preference: Republican\nGraduate Student/Businessman\n\n'), (True, u'E DON J. GRUNDMANN\nParty Preference: American Independcn\nDoctor oi Chiropractic\n\n'), (True, u'Z ROBERT LAUTEN\nParty Prelerencez Republican\n\n'), (True, u'E GAnL K. uc}-nFOOT\nParty Preference; Libedarian\nRetired Nurse\n\n'), (True, u'E DAVID Auax Lzvnrr\nParty Preference: Democratic\nComputer SdentisvJEngineer\n\n'), (True, u'E 0RLY mrrz\nParty Prekzrencez Republican\nD0ct0rIAtt0rney/Busir\\essw0\u2022nan\n\n'), (True, u'Z AL RAMIREZ\nParty Preference: Republican\nBusinessman\n\n'), (True, u'E DIRK ALLEN K0N0Pn<\nParty Preterence: Republi \xbb\xa2 n\nMBA Student\n\n'), (True, u'Z DONALD KRAMPE\nParty Preference: Republican\nRetired Administration Director\n\n'), (True, u'E Mme sTRnMuNG\nParty Preference: Democratic\nConsumer Rights Attorney\n\n'), (True, u'Z umm; STEWART\nParty Prelevence: Democratic\nBusmesswuman/Finance Managa\n\n'), (True, u'\xa4 MAK sl-1AH\nParty Preterence: Democratic\nEnvironmental Health Consultant\n\n'), (True, u'\xa4 NACHUM so-uFREN\nParty Preference: Republican\nEducator/Autr\u2022orIBusinessman\n\n'), (True, u'D DENNIS JACKSON\nParty Prelerence: Republican\nAerospace General Manager\n\n'), (True, u'D DAN Hucnss\nParty Prelerence: Republican\nSmall Business Owner\n\n')]), [(0, 4), (1, 5), (2, 6), (3, 7), (4, 8), (5, 9), (6, 10), (7, 11), (8, 12), (9, 13), (10, 14), (11, 15), (12, 16), (13, 17), (14, 18), (15, 19), (16, 20), (17, 21), (18, 0), (19, 1), (20, 2), (21, 3)])]]
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

        self.groups_saved = groups
        self.compute_equivs_2()
    
    def compute_equivs_2(self):
        groups = self.groups_saved
        print "GROUPS IS", groups

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
            VerifyContestGrouping(frame, self.proj.ocr_tmp_dir, self.dirList, self.equivs, self.reorder, self.reorder_inverse, self.mapping, self.mapping_inverse, self.multiboxcontests, putresults)
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

