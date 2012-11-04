import wx, pdb, traceback
from wx.lib.pubsub import Publisher
from wx.lib.scrolledpanel import ScrolledPanel
import os
try:
    import cPickle as pickle
except:
    import pickle
import csv
from os.path import join as pathjoin
from util import encodepath
import util

class ResultsPanel(ScrolledPanel):
    def __init__(self, parent, *args, **kwargs):
        ScrolledPanel.__init__(self, parent, style=wx.SIMPLE_BORDER, *args, **kwargs)
        
        self.sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.results = wx.StaticText(self, label="")
        self.sizer.Add(self.results, flag=wx.EXPAND)
        
        self.SetSizer(self.sizer)
        self.Fit()

    def start(self, proj):
        self.proj = proj
        self.set_results()

    def set_results(self):
        """Processes cvr file, outputs results files."""
        print 'First creating the CVR'
        cvr = self.process()

        print 'Now generate the human CVR'

        self.human_readable_cvr(cvr)
        print 'And now the precinct and mode tally'
        res = self.tally_by_precinct_and_mode(cvr)
        self.results.SetLabel(res)
        self.SetupScrolling()
        open(self.proj.election_results, "w").write(res)
        
        # If there are batches
        if len([x[0] for x in os.walk(self.proj.voteddir)]) > 1:
            print 'Tally by batch finally'
            batches_res = self.tally_by_batch(cvr)
            open(self.proj.election_results_batches, "w").write(batches_res)

    def load_grouping(self):
        """Processes grouping_results.
        
        Returns dict with key-val pairs:
        'header' -> header line
        samplepath -> [templatepath,precinct,flipped_front,flipped_back]
        
        """
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
        b2imgs = pickle.load(open(self.proj.ballot_to_images, 'rb'))
        partition_exmpls = pickle.load(open(pathjoin(self.proj.projdir_path,
                                                     self.proj.partition_exmpls), 'rb'))
        img2page = pickle.load(open(pathjoin(self.proj.projdir_path, 
                                             self.proj.image_to_page), 'rb'))
        target_locs_map = pickle.load(open(pathjoin(self.proj.projdir_path,
                                                    self.proj.target_locs_map), 'rb'))
        for partitionID, contests_sides in target_locs_map.iteritems():
            
            exmpl_id = partition_exmpls[partitionID][0]
            imgpaths = b2imgs[exmpl_id]
            imgpaths_ordered = sorted(imgpaths, key=lambda imP: img2page[imP])
            for side, contests in sorted(contests_sides.iteritems(), key=lambda t: t[0]):
                exmpl_imgP = imgpaths_ordered[side]
                thismap = {}
                for contest in contests:
                    # BOX := [x1, y1, x2, y2, id, contest_id]
                    contestbox, targetboxes = contest[0], contest[1:]
                    for tbox in targetboxes:
                        if (exmpl_imgP, tbox[5]) in localid_to_globalid:
                            # only do it if it's in the map
                            # it might not be in the map if it's a multi-box
                            # contest, and the other box is in the map.
                            glob = localid_to_globalid[(exmpl_imgP, tbox[5])]
                            thismap[tbox[4]] = glob
                if thismap == {}:
                    # Means that 'template' has no contests/targets on it
                    # (i.e. it's a totally-blank page), so, skip it.
                    continue
                else:
                    templatemap[exmpl_imgP] = thismap
        
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

        quarantined = set()
        if os.path.exists(self.proj.quarantined):
            qfiles = open(self.proj.quarantined).read().split("\n")
            for qfile in qfiles:
                quarantined.add(qfile)
        if os.path.exists(self.proj.quarantined_manual):
            qfiles = open(self.proj.quarantined_manual).read().split("\n")
            for qfile in qfiles:
                quarantined.add(qfile)

        templatemap = self.get_templatemap()

        text, order = self.get_text()

        ballot_to_images = pickle.load(open(self.proj.ballot_to_images))
        image_to_ballot = pickle.load(open(self.proj.image_to_ballot, 'rb'))
        img2page = pickle.load(open(pathjoin(self.proj.projdir_path,
                                             self.proj.image_to_page), 'rb'))
        print 'Loaded all the information'
        
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
            # TODO: crashes at voted[x]
            retval = ['01'[voted[x]] for x in order[template,cid]]+['OK']
            return retval
            #return ['01'[voted[x]] for x in order[template,cid]]+['OK']

        def noexist(cid):
            # When a contest doesn't appear on a ballot, write this
            return ["0"]*(len(text[cid])-2)+["ABSENT"]

        # Hold the CVR results for a single image.
        image_cvr = {}

        print 'Counting up to', len(os.listdir(self.proj.ballot_metadata))

        for i,ballot in enumerate(os.listdir(self.proj.ballot_metadata)):
            if i%1000 == 0:
                print 'On ballot', i
            meta = pickle.load(open(os.path.join(self.proj.ballot_metadata,ballot)))
            # meta['ballot'] is a voted imgpath, not an int ballotID
            ballotid = image_to_ballot[meta['ballot']]
            votedpaths = ballot_to_images[ballotid]
            votedpaths = sorted(votedpaths, key=lambda imP: img2page[imP])
            
            #votedpaths = ballot_to_images[image_to_ballot[meta['ballot']]]
            bools = [votedpath in quarantined for votedpath in votedpaths]
            # If any of the sides is quarantined, skip it
            if True in bools: continue

            print 'bal', meta['ballot']
            template = meta['template']
            print 'template', template
            targets = meta['targets']
            print targets
            voted = {}
            for target in targets:
                targetid = int(target.split(".")[1])
                print 'targetid', targetid
                try:
                    contest = templatemap[template][targetid]
                except:
                    traceback.print_exc()
                    pdb.set_trace()
                print 'contest', contest
                if contest not in voted: voted[contest] = []
                voted[contest].append((targetid, target in isvoted))
                print 'so', target in isvoted
                print 'total', voted
                

            #print 'voted a', voted
            voted = dict((a,sorted(b)) for a,b in voted.items())
            #for k,v in voted.items():
            #    print k, v
            #    print k, order[template,k]
    
            voted = dict([(id,processContest(template,id,lst)) for id,lst in voted.items()])
            print "Results for ballot", meta['ballot']
            print voted
            image_cvr[meta['ballot']] = voted
    
        print 'Now going through the ballots'
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
        print 'And now the quarantine ones'
        
        cvr = csv.writer(open(self.proj.cvr_csv, "w"))
        headerstr = ['#path']+sum([[b[1]+":"+c for c in b[2:]]+[b[1]] for _,b in text.items()], [])
        cvr.writerow(headerstr)

        full_cvr = []
        print 'And now going up to', len(ballot_to_images)
        for i,(ballotid,images) in enumerate(ballot_to_images.items()):
            if i%1000 == 0: print 'on', i
            #print "----"
            #print ballot
            #print images
            #print "----"
            # Store the cvr for the full ballot
            ballot_cvr = {}
            images = sorted(images, key=lambda imP: img2page[imP])
            for image in images:
                if image in image_cvr:
                    #print 'imcvr', image_cvr[image]
                    for cid, filled in image_cvr[image].items():
                        ballot_cvr[cid] = filled

            full_cvr.append((images[0],ballot_cvr))

            ballot_cvr = [x[1] for x in sorted(ballot_cvr.items())]
            cvr.writerow([images[0]]+sum(ballot_cvr,[]))
#        print 'end', full_cvr

        print 'And ending'
       
        return full_cvr

    def human_readable_cvr(self, cvr):
        text, order = self.get_text()
        #print text
        # Group things together.

        for path, ballot_cvr in cvr:
            path = os.path.relpath(path, self.proj.voteddir)
            path = os.path.join(self.proj.cvr_dir, path)
            if not os.path.exists(os.path.split(path)[0]):
                os.makedirs(os.path.split(path)[0])
            #print ballot_cvr
                   
            path = os.path.splitext(path)[0] + '.txt'

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
        """Aggregrate tallies to form a final tally.
        
        Keyword args:
        name -- Title of grouping, e.g. 'TOTAL', 'Precinct', 'Dirname'

        """
        text, order = self.get_text()

        res = {}
        overunder = {}
        total = {}
        for path,ballot_cvr in cvr:
            for ocid in ballot_cvr:
                cid = (tuple(text[ocid][:2]),tuple(sorted(text[ocid][2:])))
                if cid not in res:
                    res[cid] = dict((candidate, 0) for candidate in text[ocid][2:])
                    overunder[cid] = [0,0]
                    total[cid] = 0
                for i,each in enumerate(ballot_cvr[ocid][:-1]):
                    try:
                        res[cid][text[ocid][i+2]] += int(each)
                    except Exception as e:
                        print e
                        pdb.set_trace()
                if ballot_cvr[ocid][-1] == 'OVERVOTED':
                    overunder[cid][0] += 1
                elif ballot_cvr[ocid][-1] == 'UNDERVOTED':
                    overunder[cid][1] += 1
                total[cid] += 1

        s = ""
        if name != None:
            s += "------------------\n"+name+"\n------------------\n"
        for cid,votes in res.items():
            votelist = [a+": "+str(b) for a,b in votes.items()]
            show_overunder = ['Over Votes: ' + str(overunder[cid][0]),
                              'Under Votes: ' + str(overunder[cid][1]),
                              'Total Ballots: ' + str(total[cid])]
            s += "\n\t".join([cid[0][1]]+votelist+show_overunder)
            s += "\n"
        #print s
        return s+"\n"

    def tally_by_precinct_and_mode(self, cvr):
        """ Tallies by groupings of precinct and mode
        
        Returns: dict containing key-value pairs of 
        attribute -> cvr item
        e.g. 'precinct 1' : cvr item
        """
        attributes = self.load_grouping()
        if os.path.exists(self.proj.quarantined):
            quar1 = set(x[0] for x in csv.reader(open(self.proj.quarantined)) if x)
        else:
            quar1 = set()
        if os.path.exists(self.proj.quarantined_manual):
            quar2 = set(x[0] for x in csv.reader(open(self.proj.quarantined_manual)) if x)
        else:
            quar2 = set()
        quar = quar1.union(quar2)

        result = ""
        result += self.final_tally(cvr, name="TOTAL")

        if attributes != None: 
            def groupby(lst, attr):
                attr = attributes['header'].index(attr)
                res = {}
                for a in lst:
                    if os.path.abspath(a[0]) not in attributes:
                        if os.path.abspath(a[0]) in quar:
                            print "A-OKAY! It was quarantined"
                        else:
                            pdb.set_trace()
                            raise Exception("Oh no. " + os.path.abspath(a[0]) + " was not in the grouping results.")
                    else:
                        try:
                            thisattr = attributes[os.path.abspath(a[0])][attr]
                        except Exception as e:
                            print e
                            pdb.set_trace()
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

    def tally_by_precinct_and_mode_hack(self, cvr):
        quar1 = set(x[0] for x in csv.reader(open(self.proj.quarantined)))
        quar2 = set(x[0] for x in csv.reader(open(self.proj.quarantined_manual)))
        quar = quar1.union(quar2)

        result = ""
        result += self.final_tally(cvr, name="TOTAL")

        if True:
            def groupby(lst, attr):
                res = {}
                for a in lst:
                    if attr == 'precinct':
                        thisattr = a[0].split("/")[-1].split("_")[1]
                    elif attr == 'mode':
                        thisattr = a[0].split("/")[-1].split("_")[0]
                    if thisattr not in res: res[thisattr] = []
                    res[thisattr].append(a)
                return res
    
            if True:
                ht = groupby(cvr, 'precinct')
                for k,v in ht.items():
                    result += self.final_tally(v, name="Precinct: "+k)
                    if True:
                        ht2 = groupby(v, 'mode')
                        for k2,v2 in ht2.items():
                            name = "Precinct, Mode: "+k+", "+k2
                            result += self.final_tally(v2, name)
        return result


    def tally_by_batch(self, cvr):
        """Tallies by batches rooted at voted/ directory.
        e.g. /000, /000/Absentee, etc.
        
        """
        
        result = ""
        result += self.final_tally(cvr, name="TOTAL")
               
        sampledirs_lvl1 = [x[0] for x in os.walk(self.proj.voteddir)]
        util.sort_nicely(sampledirs_lvl1)
        
        batch_paths = sampledirs_lvl1
        batch_paths = batch_paths[1:]

        def dircontains(parent, path):
            """Returns true if the file is a subdirectory of parent"""
            path = os.path.normpath(os.path.abspath(path))
            parent = os.path.normpath(os.path.abspath(parent)) + os.sep
            return parent in path
  

        for batch in batch_paths:
            matchingcvrs = []
            for entry in cvr:
                if dircontains(batch, entry[0]):
                     matchingcvrs.append(entry)

            name = batch.replace(self.proj.voteddir + os.sep, '')
            result += self.final_tally(matchingcvrs,name)

        return result            
    
        
            




    

                

            

        


