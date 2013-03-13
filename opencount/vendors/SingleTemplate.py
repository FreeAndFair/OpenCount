import sys, pdb, traceback

from Vendor import Vendor

class SingleTemplateVendor(Vendor):
    def __init__(self, proj):
        self.proj = proj

    def decode_ballots(self, ballots, manager=None, queue=None):
        self.bals = ballots.keys()
        paths = [x for y in ballots.values() for x in y]
        print 'a'
        return dict((x,False) for x in paths), {'None':[(x,(0,0,1,1),None) for x in paths]},[],[]

    def partition_ballots(self, verified_results, manual_labeled):
        ballots = [x[0] for x in verified_results.values()[0]]
        return {0: self.bals}, dict((x,('',)) for x in ballots), dict((x,{'page': 0}) for x in ballots)

    def __repr__(self):
        return 'SingleTemplateVendor()'
    def __str__(self):
        return 'SingleTemplateVendor()'
