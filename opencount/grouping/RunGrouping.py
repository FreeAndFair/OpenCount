import os, sys, traceback, pdb, time
try:
    import cPickle as pickle
except:
    import pickle

from os.path import join as pathjoin

import wx
from wx.lib.scrolledpanel import ScrolledPanel

sys.path.append('..')

import pixel_reg.doGrouping as doGrouping
import grouping.digit_group_new as digit_group_new

GRP_PER_BALLOT = 0
GRP_PER_PARTITION = 1 

class RunGroupingMainPanel(wx.Panel):
    def __init__(self, parent, *args, **kwargs):
        wx.Panel.__init__(self, parent, *args, **kwargs)
        
        self.proj = None

        # dict EXTRACT_RESULTS: maps {int ballotID: {attrtype: {'attrOrder': attrorder, 'err': err,
        #                                            'exemplar_idx': exemplar_idx,
        #                                            'bb': (x1,y1,x2,y2)}}
        self.extract_results = None

        # dict DIGITGROUP_RESULTS: maps {str attrtype: {int ID: ...}}
        self.digitgroup_results = None

        # float DIGITDIST: Est. distance between each digit.
        self.digitdist = None

        self.init_ui()

    def init_ui(self):
        self.btn_rungrouping = wx.Button(self, label="Run Grouping.")
        self.btn_rungrouping.Bind(wx.EVT_BUTTON, self.onButton_rungrouping)
        
        self.btn_rerun_imggroup = wx.Button(self, label="Re-run Image-Based Grouping.")
        self.btn_rerun_imggroup.Bind(wx.EVT_BUTTON, self.onButton_rerun_imggroup)
        self.btn_rerun_digitgroup = wx.Button(self, label="Re-run Digit-Based Grouping.")
        self.btn_rerun_digitgroup.Bind(wx.EVT_BUTTON, self.onButton_rerun_digitgroup)
        txt_rerunOr = wx.StaticText(self, label="- Or -")
        self.btn_rerun_grouping = wx.Button(self, label="Re-run All Grouping.")
        rerun_btnsizer0 = wx.BoxSizer(wx.VERTICAL)
        rerun_btnsizer0.AddMany([(self.btn_rerun_imggroup,0,wx.ALL,10), (self.btn_rerun_digitgroup,0,wx.ALL,10)])
        rerun_btnsizer = wx.BoxSizer(wx.HORIZONTAL)
        rerun_btnsizer.AddMany([(rerun_btnsizer0,0,wx.ALL,10), (txt_rerunOr, 0, wx.ALIGN_CENTER | wx.ALL, 10),
                                (self.btn_rerun_grouping,0,wx.ALIGN_CENTER | wx.ALL, 10)])
        self.btn_rerun_imggroup.Disable()
        self.btn_rerun_digitgroup.Disable()
        self.btn_rerun_grouping.Disable()

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.AddMany([(self.btn_rungrouping,0,wx.ALL,10),
                            (rerun_btnsizer,0,wx.ALL,10)])
        self.SetSizer(self.sizer)
        self.Layout()

    def start(self, proj, stateP):
        self.proj = proj
        self.stateP = stateP
        self.proj.addCloseEvent(self.save_session)
        if not self.restore_session():
            pass

    def stop(self):
        if not self.proj:
            return
        self.save_session()
        self.proj.removeCloseEvent(self.save_session)
        self.export_results()

    def export_results(self):
        """ Saves the results of imgbased/digitbased grouping.
        """
        pickle.dump(self.extract_results, open(pathjoin(self.proj.projdir_path,
                                                        self.proj.extract_results), 'wb'),
                    pickle.HIGHEST_PROTOCOL)
        pickle.dump(self.digitgroup_results, open(pathjoin(self.proj.projdir_path,
                                                           self.proj.digitgroup_results), 'wb'),
                    pickle.HIGHEST_PROTOCOL)

    def restore_session(self):
        try:
            state = pickle.load(open(self.stateP, 'rb'))
            self.digitdist = state['digitdist']
            self.extract_results = pickle.load(open(pathjoin(self.proj.projdir_path,
                                                             self.proj.extract_results), 'rb'))
            self.digitgroup_results = pickle.load(open(pathjoin(self.proj.projdir_path,
                                                                self.proj.digitgroup_results), 'rb'))
        except:
            return False
        if self.extract_results:
            self.btn_rungrouping.Disable()
            self.btn_rerun_imggroup.Enable()
            self.btn_rerun_digitgroup.Enable()
            self.btn_rerun_grouping.Enable()
        return True

    def save_session(self):
        state = {}
        state['digitdist'] = self.digitdist
        pickle.dump(state, open(self.stateP, 'wb'), pickle.HIGHEST_PROTOCOL)

    def run_imgbased_grouping(self):
        '''
        partitions_map = pickle.load(open(pathjoin(self.proj.projdir_path,
                                                   self.proj.partitions_map), 'rb'))
        partition_attrmap = pickle.load(open(pathjoin(self.proj.projdir_path,
                                                      self.proj.partition_attrmap), 'rb'))
        partition_exmpls = pickle.load(open(pathjoin(self.proj.projdir_path,
                                                     self.proj.partition_exmpls), 'rb'))
        b2imgs = pickle.load(open(self.proj.ballot_to_images, 'rb'))
        img2b = pickle.load(open(self.proj.image_to_ballot, 'rb'))
        # dict MULTEXEMPLARS_MAP: maps {attrtype: {attrval: [(subpatchP, blankpathP, (x1,y1,x2,y2)), ...]}}
        multexemplars_map = pickle.load(open(pathjoin(self.proj.projdir_path,
                                                      self.proj.multexemplars_map), 'rb'))
        img2page = pickle.load(open(pathjoin(self.proj.projdir_path,
                                             self.proj.image_to_page), 'rb'))
        imginfo_map = pickle.load(open(pathjoin(self.proj.projdir_path,
                                                self.proj.imginfo_map), 'rb'))
        attrs = pickle.load(open(self.proj.ballot_attributesfile, 'rb'))
        # 0.) Behavior depends on the Grouping Mode (PER_PARTITION or PER_BALLOT).
        grp_mode = get_imggroup_modes

        print "...Running Extract Attrvals..."
        t = time.time()
        extract_results = doGrouping.extract_attrvals(partitions_map, partition_attrmap,
                                                      partition_exmpls,
                                                      b2imgs, img2b, multexemplars_map,
                                                      img2page, imginfo_map, attrs)
        dur = time.time() - t
        print "...Finished Running Extract Attrvals ({0} s).".format(dur)
        self.extract_results = extract_results
        '''
        return {}

    def run_digitbased_grouping(self):
        partitions_map = pickle.load(open(pathjoin(self.proj.projdir_path,
                                                   self.proj.partitions_map), 'rb'))
        partitions_invmap = pickle.load(open(pathjoin(self.proj.projdir_path,
                                                      self.proj.partitions_invmap), 'rb'))
        partition_exmpls = pickle.load(open(pathjoin(self.proj.projdir_path,
                                                     self.proj.partition_exmpls), 'rb'))
        b2imgs = pickle.load(open(self.proj.ballot_to_images, 'rb'))
        img2b = pickle.load(open(self.proj.image_to_ballot, 'rb'))
        attrs = pickle.load(open(self.proj.ballot_attributesfile, 'rb'))
        img2page = pickle.load(open(pathjoin(self.proj.projdir_path,
                                             self.proj.image_to_page), 'rb'))
        img2flip = pickle.load(open(pathjoin(self.proj.projdir_path,
                                             self.proj.image_to_flip), 'rb'))
        digitexemplars_map = pickle.load(open(pathjoin(self.proj.projdir_path,
                                                       self.proj.digit_exemplars_map), 'rb'))
        all_results = {} # maps {str attrtype: dict results}
        MODE = get_digitgroup_mode(self.proj)
        digitpatch_dir = pathjoin(self.proj.projdir_path, self.proj.digitpatch_dir)
        digpatch2imgpath_outP = pathjoin(self.proj.projdir_path, self.proj.digpatch2imgpath)
        print "...DigitGroup Mode: {0}...".format({GRP_PER_PARTITION: 'GRP_PER_PARTITION', 
                                                   GRP_PER_BALLOT: 'GRP_PER_BALLOT'}[MODE])
        voteddir_root = self.proj.voteddir
        print '...voteddir_root is:', voteddir_root
        if self.digitdist == None:
            self.digitdist = compute_median_dist(self.proj)
        for attr in attrs:
            if attr['is_digitbased']:
                attrtypestr = '_'.join(sorted(attr['attrs']))
                attrinfo = [attr['x1'], attr['y1'], attr['x2'], attr['y2'],
                            attrtypestr, attr['side'], attr['num_digits'], self.digitdist]
                results = digit_group_new.do_digit_group(b2imgs, img2b, partitions_map,
                                                         partitions_invmap, partition_exmpls,
                                                         img2page, img2flip, attrinfo,
                                                         digitexemplars_map, digitpatch_dir,
                                                         voteddir_root,
                                                         digpatch2imgpath_outP,
                                                         mode=MODE)
                all_results[attrtypestr] = results
        self.digitgroup_results = all_results

        print '...DigitGrouping Done.'

    def onButton_rungrouping(self, evt):
        """ Runs both Image-based and Digit-based Attrval extraction. """
        self.Disable()
        if exists_imgattr(self.proj):
            self.run_imgbased_grouping()
        if exists_digattr(self.proj):
            self.run_digitbased_grouping()

        self.btn_rungrouping.Disable()
        self.Enable()
        
    def onButton_continueverify(self, evt):
        pass
    def onButton_rerun_imggroup(self, evt):
        pass
    def onButton_rerun_digitgroup(self, evt):
        pass

def compute_median_dist(proj):
    """ Computes the median (horiz) distance between adjacent digits,
    based off of the digits from the blank ballots.
    Input:
        obj proj:
    Output:
        int distance, in pixels.
    """
    # Bit hacky - peer into LabelDigit's 'matches' internal state
    labeldigits_stateP = pathjoin(proj.projdir_path, proj.labeldigitstate)
    # matches maps {str regionpath: ((patchpath_i,matchID_i,digit,score,y1,y2,x1,x2,rszFac_i), ...)
    matches = pickle.load(open(labeldigits_stateP, 'rb'))['matches']
    dists = [] # stores adjacent distances
    for regionpath, tuples in matches.iteritems():
        x1_all = []
        for (patchpath, matchID, digit, score, y1, y2, x1, x2, rszFac) in tuples:
            x1_all.append(int(round(x1 / rszFac)))
        x1_all = sorted(x1_all)
        for i, x1 in enumerate(x1_all[:-1]):
            x1_i = x1_all[i+1]
            dists.append(int(round(abs(x1 - x1_i))))
    dists = sorted(dists)
    if len(dists) <= 2:
        median_dist = min(dists)
    else:
        median_dist = dists[int(len(dists) / 2)]
    print '=== median_dist is:', median_dist
    return median_dist

def exists_digattr(proj):
    attrs = pickle.load(open(proj.ballot_attributesfile, 'rb'))
    for attr in attrs:
        if attr['is_digitbased']:
            return True
    return False
def exists_imgattr(proj):
    attrs = pickle.load(open(proj.ballot_attributesfile, 'rb'))
    for attr in attrs:
        if not attr['is_digitbased']:
            return True
    return False

def get_digitgroup_mode(proj):
    """ Determines the digitgroup mode, either DG_PER_BALLOT or DG_PER_PARTITION.
    Input:
        obj PROJ:
    Output:
        int MODE.
    """
    # If the 'partition' key exists in the IMGINFO_MAP, then this implies
    # that the partitions are separated by precinct number.
    # IMGINFO_MAP: maps {str imgpath: {str key: val}}
    '''
    imginfo_map = pickle.load(open(pathjoin(proj.projdir_path,
                                            proj.imginfo_map), 'rb'))
    if 'precinct' in imginfo_map[imginfo_map.keys()[0]]:
        return digit_group_new.DG_PER_PARTITION
    else:
        return digit_group_new.DG_PER_BALLOT
    '''
    attrs = pickle.load(open(proj.ballot_attributesfile, 'rb'))
    for attr in attrs:
        if attr['is_digitbased']:
            return GRP_PER_PARTITION if attr['grp_per_partition'] else GRP_PER_BALLOT
    print "uhoh, shouldn't get here."
    raise Exception
