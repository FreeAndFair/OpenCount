"""
A series of helper functions in order to integrate the digit 'reject'
UI step, and the partmatch* function feedback loop.
"""

"""
Output:
<projdir>/rejected_hashes.p
    {str imgpath: {str digit: [((y1,y2,x1,x2), str side_i), ...]}}

<projdir>/accepted_hashes.p
    {str imgpath: {str digit: [((y1,y2,x1,x2), str side_i), ...]}}

"""


def get_rejected_hashes(project):
    """ Returns the rejected_hashes for the entire data set.
    Returns:
        {str imgpath: {str digit: [((y1,y2,x1,x2),side_i,isflip_i), ...]}}
        or None if rejected_hashes.p doesn't exist yet.
    """
    return project.load_field(project.rejected_hashes)

def save_rejected_hashes(project, rejected_hashes):
    """ Saves the newer-version of rejected_hashes. """
    project.save_field(rejected_hashes, project.rejected_hashes)

def get_accepted_hashes(project):
    """ Returns the accepted_hashes for the entire data set.
    Returns:
        {str imgpath: {str digit: [((y1,y2,x1,x2),side_i,isflip_i), ...]}}
        or None if accepted_hashes.p doesn't exist yet.
    """
    return project.load_field(project.rejected_hashes)

def save_accepted_hashes(project, accepted_hashes):
    """ Saves the newer-version of accepted_hashes. """
    project.save_field(accepted_hashes, project.accepted_hashes
