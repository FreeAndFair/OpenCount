import sys, os, pdb, pickle
sys.path.append('../')

import pixel_reg.part_match as part_match

"""
A series of helper functions in order to integrate the digit 'reject'
UI step, and the partmatch* function feedback loop.
"""

"""
Output:
This script keeps track of all partmatch* metadata (mainly the
'rejected_hash' dictionary).

We mirror the directory structure of the voted
ballots directory looks like:
    napa_straight/votedballots/lib1/lib1_0.png
    napa_straight/votedballots/lib1/lib1_1.png
    napa_straight/votedballots/non1/non1_0.png
    napa_straight/votedballots/non1/non1_1.png

Then the partmatch metadata directoy would be:
    napa_project/pm_metadata/lib1/lib1_0/precinct.p
    napa_project/pm_metadata/lib1/lib1_1/precinct.p
    napa_straight/votedballots/non1/non1_0/precinct.p
    napa_straight/votedballots/non1/non1_1/precinct.p

(or, rather, since only front-sides have a precinct patch):
    napa_project/pm_metadata/lib1/lib1_0/precinct.p
    napa_straight/votedballots/non1/non1_0/precinct.p

Assuming that we were working with the 'precinct' digit-based
attribute. Each '*.p' file could be a pickle'd object containing
all the metadata that partmatch might need (like the rejected
information, DP tables, etc).
"""

def reject_match(imgpath, proj):
    """
    Re-run partmatch*, but with the new knowledge that the digit patch
    'imgpath' is incorrect.
    """
    pass

