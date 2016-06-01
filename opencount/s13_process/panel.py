import wx
import pdb
import traceback
from wx.lib.scrolledpanel import ScrolledPanel
import os
try:
    import cPickle as pickle
except:
    import pickle
import json
import subprocess

import util
from util import debug, error


def external_vote_count(choices, candidates):
    '''
    Use the external `plurality_election` vote-counting utility to
    count votes. Accepts a list of choices (names) and a list of
    possible candidate names.
    '''
    # Right now, the subprocess expects numbers for candidates.
    # These tables let us convert back and forth:
    index_to_candidate = dict(enumerate(candidates))
    candidate_to_index = dict((v, k) for (k, v)
                              in index_to_candidate.items())

    # Massage the output here:
    choices = [candidate_to_index[c] for c in choices]
    votes = {'votes': [{'id': i, 'selection': r}
                       for (i, r) in enumerate(choices)]}

    # Open the subprocess:
    process = subprocess.Popen(['plurality_election'],
                               stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE)
    # and write the JSON to it
    json.dump(votes, process.stdin)
    process.stdin.close()
    process.wait()

    # pull it back out
    raw = process.stdout.read()
    debug('{0}', raw)
    result = json.loads(raw)
    # and present the winner in proper form
    if result:
        return index_to_candidate[result['winner']], raw
    else:
        return 'Tie', raw


class ResultsPanel(ScrolledPanel):

    def __init__(self, parent, *args, **kwargs):
        ScrolledPanel.__init__(
            self, parent, style=wx.SIMPLE_BORDER, *args, **kwargs)

        self.sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.results = wx.StaticText(self, label="")
        self.sizer.Add(self.results, flag=wx.EXPAND)

        self.SetSizer(self.sizer)
        self.Fit()

    def start(self, project=None, projdir=None):
        self.proj = project
        # 0.) Grab all quarantined ballots
        self.qballotids = project.get_quarantined_ballots()
        bal2imgs = project.load_field(project.ballot_to_images)

        self.qvotedpaths = []
        for ballotid in self.qballotids:
            votedpaths = bal2imgs[ballotid]
            self.qvotedpaths.extend(votedpaths)
        self.qvotedpaths = set(self.qvotedpaths)

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
        samplepath -> [templatepath,attrvals,flipped_front,flipped_back]
        """
        if not os.path.exists(self.proj.grouping_results):
            return None
        b2imgs = self.proj.load_field(self.proj.ballot_to_images)
        img2page = self.proj.load_field(self.proj.image_to_page)

        c_t = {}
        for line in self.proj.read_csv(self.proj.grouping_results):
            if len(line) < 2:
                continue
            if line[0] == 'ballotid':
                attrtypes = line[1:]
                # ignore partitionID (always at end)
                attrtypes = attrtypes[:-1]
                c_t['header'] = line[1:]
            else:
                ballotid = int(line[0])
                imgpaths = sorted(b2imgs[ballotid],
                                  key=lambda imP: img2page[imP])
                # c_t[os.path.abspath(line[0])] = line[1:]
                attrvals = line[1:]
                attrvals = attrvals[:-1]  # ignore partitionID (always at end)
                c_t[imgpaths[0]] = attrvals

        for line in self.proj.read_csv(self.proj.quarantine_attributes):
            c_t[line[0]] = [None] + line[1:]
        if len(c_t) < 2:
            return None

        return c_t

    def get_templatemap(self):
        localid_to_globalid = {}
        for line in self.proj.read_csv(self.proj.contest_id):
            if len(line) < 3:
                continue
            localid_to_globalid[(line[0], int(line[1]))] = int(line[2])
        # template -> target id -> contest
        templatemap = {}

        b2imgs = self.proj.load_field(self.proj.ballot_to_images)
        group_exmpls = self.proj.load_field(self.proj.group_exmpls)
        img2page = self.proj.load_field(self.proj.image_to_page)
        target_locs_map = self.proj.load_field(self.proj.target_locs_map)

        for groupID, contests_sides in target_locs_map.iteritems():
            exmpl_id = group_exmpls[groupID][0]
            imgpaths = b2imgs[exmpl_id]
            imgpaths_ordered = sorted(imgpaths, key=lambda imP: img2page[imP])
            for side, contests in sorted(contests_sides.iteritems(),
                                         key=lambda t: t[0]):
                exmpl_imgP = imgpaths_ordered[side]
                thismap = {}
                for contest in contests:
                    # BOX := [x1, y1, x2, y2, id, contest_id]
                    targetboxes = contest[1:]
                    for tbox in targetboxes:
                        if (exmpl_imgP, tbox[5]) in localid_to_globalid:
                            # only do it if it's in the map
                            # it might not be in the map if it's a multi-box
                            # contest, and the other box is in the map.
                            glob = localid_to_globalid[(exmpl_imgP, tbox[5])]
                            thismap[tbox[4]] = glob
                        else:
                            error('{0} not in ID map', (exmpl_imgP, tbox[5]))
                            pdb.set_trace()
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

        for row in self.proj.read_csv(self.proj.contest_text):
            text[int(row[0])] = [int(row[1])] + row[2:]
        for row in self.proj.read_csv(self.proj.contest_id):
            order[row[0], int(row[2])] = map(int, row[3:])

        return text, order

    def process(self):
        # target -> voted yes/no
        isvoted = open(self.proj.targets_result).read().split("\n")[:-1]
        isvoted = set([tuple(map(int, x.split(",")[:-1]))
                       for x in isvoted if x[-1] == '1'])

        templatemap = self.get_templatemap()

        text, order = self.get_text()

        ballot_to_images = self.proj.load_field(self.proj.ballot_to_images)
        image_to_ballot = self.proj.load_field(self.proj.image_to_ballot)
        img2page = self.proj.load_field(self.proj.image_to_page)
        group_exmpls = self.proj.load_field(self.proj.group_exmpls)
        bal2group = self.proj.load_field(self.proj.ballot_to_group)

        if self.proj.path_exists('group_to_Iref.p'):
            group2refidx = self.proj.load_field('group_to_Iref.p')
        else:
            group2refidx = None
        print 'Loaded all the information'

        def processContest(template, cid, votedlist):
            upto = text[cid][0]
            truefalse = [x[1] for x in votedlist]
            if sum(truefalse) > int(upto):
                return ["0"] * len(votedlist) + ["OVERVOTED"]
            elif sum(truefalse) == 0:
                return ["0"] * len(votedlist) + ["UNDERVOTED"]
            voted = dict(votedlist)
            retval = ['01'[voted[x]] for x in order[template, cid]] + ['OK']
            return retval

        # Hold the CVR results for a single image.
        image_cvr = {}

        i = 0
        for dirpath, dirnames, filenames in os.walk(self.proj.ballot_metadata):
            for f in filenames:
                meta = pickle.load(open(os.path.join(dirpath, f), 'rb'))
                if i % 1000 == 0:
                    print 'On ballot', i

                # meta['ballot'] is a voted imgpath, not an int ballotID
                ballotid = image_to_ballot[meta['ballot']]
                groupid = bal2group[ballotid]
                votedpaths = ballot_to_images[ballotid]
                votedpaths = sorted(votedpaths, key=lambda imP: img2page[imP])
                page = img2page[meta['ballot']]

                # votedpaths =
                # ballot_to_images[image_to_ballot[meta['ballot']]]
                bools = [votedpath in self.qvotedpaths
                         for votedpath in votedpaths]
                # If any of the sides is quarantined, skip it
                if True in bools:
                    continue

                # Annoying detail: META will have the Iref path used, which
                # may not match the keys in TEMPLATEMAP. TEMPLATEMAP has
                # imagepaths from the first group exemplar, whereas TEMPLATE
                # could be any of the GROUP_EXMPLS, if the user chooses a
                # Representative Ballot in 'Select Targets'.
                if group2refidx is None:
                    # Easy case: the TEMPLATE path here will always match
                    # TEMPLATEMAP
                    template = meta['template']
                else:
                    # Annoying case: meta['template'] holds the Irefpath
                    # used during target extraction, which may NOT be the
                    # templatepath that TEMPLATEMAP is expecting. So, we
                    # explicitly grab the correct templatepath. Sigh.
                    template = sorted(
                        ballot_to_images[group_exmpls[groupid][0]],
                        key=lambda imP: img2page[imP])[page]

                targets = meta['targets']
                voted = {}

                for target in targets:
                    targetid = target[2]
                    try:
                        contest = templatemap[template][targetid]
                    except:
                        traceback.print_exc()
                        pdb.set_trace()
                    if contest not in voted:
                        voted[contest] = []
                    voted[contest].append((targetid, target in isvoted))

                voted = dict((a, sorted(b)) for a, b in voted.items())
                # for k,v in voted.items():
                #    print k, v
                #    print k, order[template,k]

                voted = dict([(id, processContest(template, id, lst))
                              for id, lst in voted.items()])
                # if 'yolo_s4_062-072' in meta['ballot']:
                #    print "Results for ballot", meta['ballot']
                #    print voted
                image_cvr[meta['ballot']] = voted
                i += 1

        print 'Now going through the ballots'
        # Now process the quarantined files

        def processContestQuar(cid, voted):
            upto = text[cid][0]
            if sum(voted) > int(upto):
                return ["0"] * len(voted) + ["OVERVOTED"]
            elif sum(voted) == 0:
                return ["0"] * len(voted) + ["UNDERVOTED"]
            return ['01'[x] for x in voted] + ['OK']

        if os.path.exists(self.proj.quarantine_res):
            for data in self.proj.read_csv(self.proj.quarantine_res):
                bpath, data = data[0], data[1:]

                # Group by contest, so we have (CID, votecast)
                data = zip(data, data[1:])[::2]
                # print "data", data
                lst = []
                for cid, voted in data:
                    cid = int(cid)
                    voted = [x == 'T' for x in voted]
                    # print "CID,voted", cid, voted
                    lst.append((cid, processContestQuar(cid, voted)))
                lst = dict(lst)
                image_cvr[bpath] = lst

        # print "BCVR", image_cvr
        print 'And now the quarantine ones'

        with self.proj.write_csv(self.proj.cvr_csv) as cvr:
            # cvr = csv.writer(open(self.proj.cvr_csv, "w"))
            headerstr = ['# path'] + sum(
                [[b[1] + ":" + c for c in b[2:]] + [b[1]]
                 for _, b in text.items()], [])
            cvr.writerow(headerstr)

            full_cvr = []
            discarded_balids = self.proj.get_discarded_ballots()
            ioerr_balids = self.proj.get_ioerr_ballots()
            print 'And now going up to', len(ballot_to_images)
            for i, (ballotid, images) in enumerate(ballot_to_images.items()):
                # Ignore discarded ballots
                if ballotid in discarded_balids or ballotid in ioerr_balids:
                    continue
                if i % 1000 == 0:
                    print 'on', i
                # print "----"
                # print ballotid
                # print images
                # print "----"
                # Store the cvr for the full ballot
                ballot_cvr = {}
                for image in images:
                    if image in image_cvr:
                        # print 'imcvr', image_cvr[image]
                        for cid, filled in image_cvr[image].items():
                            ballot_cvr[cid] = filled

                full_cvr.append((images[0], ballot_cvr))

                ballot_cvr = [x[1] for x in sorted(ballot_cvr.items())]
                cvr.writerow([images[0]] + sum(ballot_cvr, []))

        print 'And ending', full_cvr

        return full_cvr

    def human_readable_cvr(self, cvr):
        text, order = self.get_text()
        # print text
        # Group things together.

        for path, ballot_cvr in cvr:
            path = os.path.relpath(path, self.proj.voteddir)
            path = os.path.join(self.proj.cvr_dir, path)
            if not os.path.exists(os.path.split(path)[0]):
                os.makedirs(os.path.split(path)[0])
            # print ballot_cvr

            path = os.path.splitext(path)[0] + '.txt'

            out = open(path, "w")
            for cid, votes in ballot_cvr.items():
                processed_votes = [n for n, vote in zip(
                    text[cid][2:], votes[:-1]) if vote == '1']
                # print "VOTES", processed_votes
                out.write("\n\t".join([text[cid][1]] + processed_votes))
                if votes[-1] != 'OK':
                    out.write("\n\t" + votes[-1])
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
        for path, ballot_cvr in cvr:
            for ocid in ballot_cvr:
                cid = (tuple(text[ocid][:2]), tuple(sorted(text[ocid][2:])))
                if cid not in res:
                    res[cid] = dict((candidate, 0)
                                    for candidate in text[ocid][2:])
                    overunder[cid] = [0, 0]
                    total[cid] = 0
                for i, each in enumerate(ballot_cvr[ocid][:-1]):
                    res[cid][text[ocid][i + 2]] += int(each)
                if ballot_cvr[ocid][-1] == 'OVERVOTED':
                    overunder[cid][0] += 1
                elif ballot_cvr[ocid][-1] == 'UNDERVOTED':
                    overunder[cid][1] += 1
                total[cid] += 1

        winner = {}
        output = {}
        for contest, votes in res.items():
            debug('Selecting winner for contest {0}', contest[0][1])
            ranks = []
            cname = contest[0][1]
            choices = contest[1]

            for v, k in votes.items():
                ranks.extend([v] * k)

            if ranks:
                won, raw = external_vote_count(ranks, choices)
                winner[cname] = won
                output[cname] = raw
            else:
                winner[cname] = 'N/A'
                output[cname] = 'null'

        s = ""
        if name is not None:
            s += "------------------\n" + name + "\n------------------\n"
        for cid, votes in res.items():
            votelist = [a + ": " + str(b) for a, b in votes.items()]
            show_overunder = ['Over Votes: ' + str(overunder[cid][0]),
                              'Under Votes: ' + str(overunder[cid][1]),
                              'Total Ballots: ' + str(total[cid]),
                              'Winner: ' + winner[cid[0][1]]]
            s += "\n\t".join([cid[0][1]] + votelist + show_overunder)
            s += '\n  [raw output: {0}]'.format(output[cid[0][1]])
            s += "\n\n"
        # print s
        return s + "\n"

    def tally_by_precinct_and_mode(self, cvr):
        """ Tallies by groupings of precinct and mode
        Input:
            list CVR: List of [[imgpath_i, ballot_cvr_i, ...], ...], where each
                ballot_cvr_i is a dict {int targetid: [bool isfilled_i, ..., 'OK'/'UNDERVOTE'/etc]}
                for each contest.
        Returns: dict containing key-value pairs of
        attribute -> cvr item
        e.g. 'precinct 1' : cvr item
        """
        def is_attrtype_exists(attrtype, proj):
            if not os.path.exists(proj.ballot_attributesfile):
                return False
            attrs = proj.load_field(proj.ballot_attributesfile)
            for attr in attrs:
                attrtypestr = '_'.join(sorted(attr['attrs'])).lower()
                if attrtypestr == attrtype:
                    return True
            return False
        # dict BALLOT_ATTRIBUTES: {'header': header_stuff, str imgpath: imgpath
        # attr info}
        ballot_attributes = self.load_grouping()

        b2grp = self.proj.load_field(self.proj.ballot_to_group)
        img2bal = self.proj.load_field(self.proj.image_to_ballot)

        quar = self.qvotedpaths
        result = ""
        result += self.final_tally(cvr, name="TOTAL")
        if ballot_attributes is not None:
            def groupby(lst, attrtype, ballot_attributes, img2bal, bal2grp, quar):
                """
                Input:
                    lst LST:
                    str ATTRTYPE: Attrtype to group by
                    list QUAR: [imgpath_i, ...], list of quarantined voted imgpaths.
                Output:
                    dict RES. maps {attrval: [[imgpath_i, ballot_cvr_i, ...], ...]}
                """
                res = {}
                # ROW := [imgpath, ...]
                for a in lst:
                    imgpath = a[0]
                    ballotid = img2bal[imgpath]
                    if imgpath not in ballot_attributes:
                        if imgpath in quar:
                            # IMGPATH will be processed later as a quarantined
                            # ballot.
                            print "OK, it's quarantined."
                        else:
                            print "Uhoh, couldn't find imgpath {0}, with ballot id {1}".format(imgpath, ballotid)
                            pdb.set_trace()
                    else:
                        attrtype_idx = ballot_attributes[
                            'header'].index(attrtype)
                        attrval = ballot_attributes[imgpath][attrtype_idx]
                        res.setdefault(attrval, []).append(a)
                return res

            # if 'precinct' in attributes['header']:
            if is_attrtype_exists('precinct', self.proj):
                ht = groupby(cvr, 'precinct', ballot_attributes,
                             img2bal, b2grp, quar)
                for k, v in ht.items():
                    result += self.final_tally(v, name="Precinct: " + k)
                    if is_attrtype_exists('mode', self.proj):
                        ht2 = groupby(v, 'mode', ballot_attributes,
                                      img2bal, b2grp, quar)
                        for k2, v2 in ht2.items():
                            name = "Precinct, Mode: " + k + ", " + k2
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
            result += self.final_tally(matchingcvrs, name)

        return result
