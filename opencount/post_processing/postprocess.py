import wx, pdb
from wx.lib.pubsub import Publisher
import os
import pickle
import csv
from util import encodepath

class ResultsPanel(wx.Panel):
    def __init__(self, parent, *args, **kwargs):
        wx.Panel.__init__(self, parent, style=wx.SIMPLE_BORDER, *args, **kwargs)
        
        self.sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.results = wx.StaticText(self, label="")
        self.sizer.Add(self.results, flag=wx.EXPAND)
        
        self.SetSizer(self.sizer)
        self.Fit()

        Publisher().subscribe(self._pubsub_project, "broadcast.project")
    
    def _pubsub_project(self, msg):
        self.proj = msg.data

    def set_results(self):
        cvr = self.process()
        self.human_readable_cvr(cvr)

        res = self.tally_by_precinct_and_mode(cvr)
        self.results.SetLabel(res)
        open(self.proj.election_results, "w").write(res)

    def load_grouping(self):
        if not os.path.exists(self.proj.grouping_results):
            return None

        c_t = {}
        for line in csv.reader(open(self.proj.grouping_results)):
            if len(line) < 2: continue
            if line[0] == 'samplepath':
                c_t['header'] = line[1:]
            else:
                c_t[os.path.abspath(line[0])] = line[1:]

        for line in csv.reader(open(self.proj.quarantine_attributes)):
            c_t[os.path.abspath(line[0])] = [None]+line[1:]

        if len(c_t) < 2: return None

        return c_t

    def get_templatemap(self):
        localid_to_globalid = {}
        for line in csv.reader(open(self.proj.contest_id)):
            if len(line) < 3: continue
            localid_to_globalid[(line[0],int(line[1]))] = int(line[2])

        # template -> target id -> contest
        templatemap = {}
        for template in os.listdir(self.proj.target_locs_dir):
            if os.path.splitext(template)[1].lower() != '.csv': continue
            thismap = {}
            for line in open(os.path.join(self.proj.target_locs_dir,template)):
                if line[0] == '#': continue
                row = line.split(",")
                if row[7] == '0':
                    # only map the targets -> contests
                    glob = localid_to_globalid[(row[0],int(row[8]))]
                    thismap[int(row[1])] = glob
            templatemap[row[0]] = thismap
        return templatemap

    def get_text(self):
        # global cid -> text
        text = {}
        # template -> global cid -> order
        order = {}

        for row in csv.reader(open(self.proj.contest_text)):
            text[int(row[0])] = [int(row[1])]+row[2:]
        for row in csv.reader(open(self.proj.contest_id)):
            order[row[0],int(row[2])] = map(int,row[3:])

        return text, order

    def process(self):
        # target -> voted yes/no
        isvoted = open(self.proj.targets_result).read().split("\n")[:-1]
        isvoted = set([x.split(",")[0] for x in isvoted if x[-1] == '1'])

        quarantined = set(open(self.proj.quarantined).read().split("\n") +
                          open(self.proj.quarantined_manual).read().split("\n"))

        frontback_map = pickle.load(open(self.proj.frontback_map))
        #print 'frontback', frontback_map

        templatemap = self.get_templatemap()

        text, order = self.get_text()

        ballot_to_images = pickle.load(open(self.proj.ballot_to_images))
        

        #print "ORDER", order
        #print "TEXT", text
        #print "tempmap", templatemap
        #print 'isvoted', isvoted


        def processContest(template, cid, votedlist):
            #print 'process', cid, votedlist
            upto = text[cid][0]
            truefalse = [x[1] for x in votedlist]
            if sum(truefalse) > int(upto):
                return ["0"]*len(votedlist)+["OVERVOTED"]
            elif sum(truefalse) == 0:
                return ["0"]*len(votedlist)+["UNDERVOTED"]
            voted = dict(votedlist)
            #print 'voted', voted
            #print 'in', order[template,cid]
            return ['01'[voted[x]] for x in order[template,cid]]+['OK']

        def noexist(cid):
            # When a contest doesn't appear on a ballot, write this
            return ["0"]*(len(text[cid])-2)+["ABSENT"]

        # Hold the CVR results for a single image.
        image_cvr = {}

        for ballot in os.listdir(self.proj.ballot_metadata):
            meta = pickle.load(open(os.path.join(self.proj.ballot_metadata,ballot)))
            if meta['ballot'] not in quarantined:
                #print 'bal', meta['ballot']
                template = meta['template']
                #print 'template', template
                targets = meta['targets']
                voted = {}
                for target in targets:
                    targetid = int(target.split(".")[1])
                    #print "t", target
                    contest = templatemap[template][targetid]
                    #print 'c', contest, targetid
                    if contest not in voted: voted[contest] = []
                    voted[contest].append((targetid, target in isvoted))
    
                #print 'voted a', voted
                voted = dict((a,sorted(b)) for a,b in voted.items())
                #for k,v in voted.items():
                    #print k, v
                    #print k, order[template,k]
                    #pass
                    
                voted = dict([(id,processContest(template,id,lst)) for id,lst in voted.items()])
                #print 'voted b', voted
                image_cvr[meta['ballot']] = voted

        # Now process the quarantined files
        def processContestQuar(cid, voted):
            upto = text[cid][0]
            if sum(voted) > int(upto):
                return ["0"]*len(voted)+["OVERVOTED"]
            elif sum(voted) == 0:
                return ["0"]*len(voted)+["UNDERVOTED"]
            return ['01'[x] for x in voted]+['OK']

        if os.path.exists(self.proj.quarantine_res):
            for data in csv.reader(open(self.proj.quarantine_res)):
                bpath, data = data[0], data[1:]
    
                # Group by contest, so we have (CID, votecast)
                data = zip(data,data[1:])[::2]
                #print "data", data
                lst = []
                for cid, voted in data:
                    cid = int(cid)
                    voted = [x == 'T' for x in voted]
                    #print "CID,voted", cid, voted
                    lst.append((cid,processContestQuar(cid, voted)))
                lst = dict(lst)
                image_cvr[bpath] = lst

        #print "BCVR", image_cvr
        
        cvr = csv.writer(open(self.proj.cvr_csv, "w"))
        headerstr = ['#path']+sum([[b[1]+":"+c for c in b[2:]]+[b[1]] for _,b in text.items()], [])
        cvr.writerow(headerstr)

        full_cvr = []
        for ballot,images in ballot_to_images.items():
            #print "----"
            #print ballot
            #print images
            #print "----"
            # Store the cvr for the full ballot
            ballot_cvr = {}
            for image in images:
                if image in image_cvr:
                    #print 'imcvr', image_cvr[image]
                    for cid, filled in image_cvr[image].items():
                        ballot_cvr[cid] = filled

            full_cvr.append((ballot,ballot_cvr))

            ballot_cvr = [x[1] for x in sorted(ballot_cvr.items())]
            cvr.writerow([ballot]+sum(ballot_cvr,[]))
        #print 'end', full_cvr
        return full_cvr

    def human_readable_cvr(self, cvr):
        text, order = self.get_text()
        #print text
        # Group things together.

        for path, ballot_cvr in cvr:
            path = os.path.relpath(path, self.proj.votedballots_straightdir)
            path = os.path.join(self.proj.cvr_dir, path)
            if not os.path.exists(os.path.split(path)[0]):
                os.makedirs(os.path.split(path)[0])
            #print ballot_cvr
            out = open(path, "w")
            for cid,votes in ballot_cvr.items():
                processed_votes = [n for n,vote in zip(text[cid][2:], votes[:-1]) if vote == '1']
                #print "VOTES", processed_votes
                out.write("\n\t".join([text[cid][1]]+processed_votes))
                if votes[-1] != 'OK':
                    out.write("\n\t"+votes[-1])
                out.write("\n")
            out.close()
            
    def final_tally(self, cvr, name=None):
        text, order = self.get_text()

        res = {}
        overunder = {}
        total = {}
        for path,ballot_cvr in cvr:
            for cid in ballot_cvr:
                if cid not in res: 
                    res[cid] = [0]*(len(text[cid])-2)
                    overunder[cid] = [0,0]
                    total[cid] = 0
                for i,each in enumerate(ballot_cvr[cid][:-1]):
                    res[cid][i] += int(each)
                if ballot_cvr[cid][-1] == 'OVERVOTED':
                    overunder[cid][0] += 1
                elif ballot_cvr[cid][-1] == 'UNDERVOTED':
                    overunder[cid][1] += 1
                total[cid] += 1
        s = ""
        if name != None:
            s += "------------------\n"+name+"\n------------------\n"
        for cid,votes in res.items():
            votelist = [a+": "+str(b) for a,b in zip(text[cid][2:],votes)]
            show_overunder = ['Over Votes: ' + str(overunder[cid][0]),
                              'Under Votes: ' + str(overunder[cid][1]),
                              'Total Ballots: ' + str(total[cid])]
            s += "\n\t".join([text[cid][1]]+votelist+show_overunder)
            s += "\n"
        #print s
        return s+"\n"

    def tally_by_precinct_and_mode(self, cvr):
        attributes = self.load_grouping()
        quar = set(x[0] for x in csv.reader(open(self.proj.quarantine_res)))
        print attributes

        result = ""
        result += self.final_tally(cvr, name="TOTAL")

        if attributes != None: 
            def groupby(lst, attr):
                attr = attributes['header'].index(attr)
                res = {}
                for a in lst:
                    if os.path.abspath(a[0]) not in attributes:
                        raise Exception("Oh no. " + os.path.abspath(a[0]) + " was not in the grouping results.")
                    else:
                        thisattr = attributes[os.path.abspath(a[0])][attr]
                        if thisattr not in res: res[thisattr] = []
                        res[thisattr].append(a)
                return res
    
            if 'precinct' in attributes['header']:
                ht = groupby(cvr, 'precinct')
                for k,v in ht.items():
                    result += self.final_tally(v, name="Precinct: "+k)
                    if 'mode' in attributes['header']:
                        ht2 = groupby(v, 'mode')
                        for k2,v2 in ht2.items():
                            name = "Precinct, Mode: "+k+", "+k2
                            result += self.final_tally(v2, name)
        return result
