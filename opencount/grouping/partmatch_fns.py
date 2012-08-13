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

def get_rejected_hashes(project):
    """ Returns the rejected_hashes for the entire data set.
    Returns:
        {str imgpath: {str digit: ((y1,y2,x1,x2,side),...)}}
        or None if rejected_hashes.p doesn't exist yet.
    """
    rej_path = os.path.join(project.projdir_path, project.rejected_hashes)
    if not os.path.exists(rej_path):
        return None
    return pickle.load(open(rej_path, 'rb'))
    
def save_rejected_hashes(project, rejected_hashes):
    """ Saves the newer-version of rejected_hashes. """
    rej_path = os.path.join(project.projdir_path, project.rejected_hashes)
    pickle.dump(rejected_hashes, open(rej_path, 'wb'))

def reject_match(imgpath, digit, bbBox, proj):
    """
    Re-run partmatch*, but with the new knowledge that the digit patch
    'imgpath' is incorrect.
    Input:
        str imgpath:
        str digit: 
        tuple bb: (y1, y2, x1, x2)
        obj proj:
    Output:
        New digit results.
    """
    pass

