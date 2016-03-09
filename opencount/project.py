'''
This contains the project class and various helper functions.
In the future, all loading and saving logic should exist here,
instead of scattered throughout the codebase.
'''

import csv
import os
from os import path

PROJ_FNAME = 'proj.p'

from util import debug, warn, error, pickle

def is_valid_projectname(name):
    """
    Only allow letters, numbers, and [_, (, )].
    """
    pattern = r'(\w|\d|[_\()])+'
    return ' ' not in name and (not re.match(pattern, name) == None)

class Project(object):
    """
    A Project is represented in the filesystem as a folder in the
    projects/ directory, where the name of the folder denotes the
    project name.
    """
    closehook = []

    def __init__(self, name='', projdir_path=''):
        self.vals = {'name': name,
                     'projdir_path': projdir_path,
                     'voteddir': '',
                     'is_multipage': False,
                     'num_pages': None,
                     'is_varnum_pages': None,
                     'vendor_obj': None,
                     'partition_exmpls': 'partition_exmpls.p',
                     'partitions_map': 'partitions_map.p',
                     'partitions_invmap': 'partitions_invmap.p',
                     'img2decoding': 'img2decoding.p',
                     'imginfo_map': 'imginfo_map.p',
                     'barcode_bbs_map': 'barcode_bbs_map.p',
                     'partition_quarantined': 'partition_quarantined.p',
                     'partition_discarded': 'partition_discarded.p',
                     'partition_ioerr': 'partition_ioerr.p',
                     'partition_attrmap': 'partition_attrmap.p',
                     'attrprops': 'attrprops.p',
                     'target_locs_map': 'target_locs_map.p',
                     'extract_results': 'extract_results.p',
                     'digitpatch_dir': 'digitpatch_dir',
                     'imgpatch2imgpath': 'imgpatch2imgpath.p',
                     'digpatch2imgpath': 'digpatch2imgpath.p',
                     'ballot_to_group': 'ballot_to_group.p',
                     'grouping_quarantined': 'grouping_quarantined.p',
                     'group_to_ballots': 'group_to_ballots.p',
                     'group_infomap': 'group_infomap.p',
                     'group_exmpls': 'group_exmpls.p',
                     'group_targets_map': 'group_targets_map.p',
                     'infer_bounding_boxes': False,
                     'targetextract_quarantined': 'targetextract_quarantined.p',
                     'ocr_tmp_dir': path.join(projdir_path, 'ocr_tmp_dir'),
                     'contest_id': path.join(projdir_path, 'contest_id.csv'),
                     'contest_text': path.join(projdir_path, 'contest_text.csv'),
                     'contest_internal': path.join(projdir_path, 'contest_internal.p'),
                     'contest_grouping_data': path.join(projdir_path, 'contest_grouping_data.p'),
                     'target_locs_dir': path.join(projdir_path, 'target_locations'),
                     'tmp': path.join(projdir_path, 'tmp'),
                     'extracted_dir': path.join(projdir_path, 'extracted'),
                     'extracted_metadata': path.join(projdir_path, 'extracted_metadata'),
                     'ballot_metadata': path.join(projdir_path, 'ballot_metadata'),
                     'classified': path.join(projdir_path, 'classified'),
                     'timing_runtarget': path.join(projdir_path, 'timing_runtarget'),
                     'threshold_internal': path.join(projdir_path, 'threshold_internal.p'),
                     'sample_flipped': path.join(projdir_path, 'sample_flipped'),
                     'extractedfile': path.join(projdir_path, 'extractedfile'),
                     'ballot_to_targets': 'ballot_to_targets.p',
                     'targets_result': path.join(projdir_path, 'targets_result.csv'),
                     'ballot_to_images': path.join(projdir_path, 'ballot_to_images.p'),
                     'image_to_ballot': path.join(projdir_path, 'image_to_ballot.p'),
                     'election_results': path.join(projdir_path, 'election_results.txt'),
                     'election_results_batches': path.join(projdir_path, 'election_results_batches.txt'),
                     'cvr_csv': path.join(projdir_path, 'cvr.csv'),
                     'cvr_dir': path.join(projdir_path, 'cvr'),
                     'quarantined': path.join(projdir_path, 'quarantined.csv'),
                     'quarantined_manual': path.join(projdir_path, 'quarantined_manual.csv'),
                     'quarantine_res': path.join(projdir_path, 'quarantine_res.csv'),
                     'quarantine_attributes': path.join(projdir_path, 'quarantine_attributes.csv'),
                     'quarantine_internal': path.join(projdir_path, 'quarantine_internal.p'),
                     'extracted_precinct_dir': path.join(projdir_path, 'extracted_precincts'),
                     'ballot_grouping_metadata': path.join(projdir_path, 'ballot_grouping_metadata'),
                     'patch_loc_dir': path.join(projdir_path, 'precinct_locations'),
                     'attr_internal': path.join(projdir_path, 'attr_internal.p'),
                     'grouping_results': path.join(projdir_path, 'grouping_results.csv'),
                     'ballot_attributesfile': path.join(projdir_path, 'ballot_attributes.p'),
                     'imgsize': (0, 0),
                     'frontback_map': path.join(projdir_path, 'frontback_map.p'),
                     'extracted_digitpatch_dir': 'extracted_digitpatches',
                     'digit_exemplars_outdir': 'digit_exemplars',
                     'digit_exemplars_map': 'digit_exemplars_map.p',
                     'precinctnums_outpath': 'precinctnums.txt',
                     'num_digitsmap': 'num_digitsmap.p',
                     'digitgroup_results': 'digitgroup_results.p',
                     'labeldigitstate': '_labeldigitstate.p',
                     'voteddigits_dir': 'voteddigits_dir',
                     'attrgroup_results': 'attrgroup_results.p',
                     'labelpanel_state': 'labelpanel_state.p',
                     'labelattrs_out': 'labelattrs_out.csv',
                     'labelattrs_patchesdir': 'labelattrs_patchesdir',
                     'attrexemplars_dir': 'attrexemplars_dir',
                     'multexemplars_map': 'multexemplars_map.p',
                     'image_to_page': 'image_to_page.p',
                     'image_to_flip': 'image_to_flip.p',
                     'rejected_hashes': 'rejected_hashes.p',
                     'accepted_hashes': 'accepted_hashes.p',
                     'custom_attrs': 'custom_attrs.p',
                     'digitpatch2temp': 'digitpatch2temp.p',
                     'digitattrvals_blanks': 'digitattrvals_blanks.p',
                     'digitpatchpath_scoresBlank': 'digitpatchpath_scoresBlank.p',
                     'digitpatchpath_scoresVoted': 'digitpatchpath_scoresVoted.p',
                     'digitmatch_info': 'digitmatch_info.p',
                     'extract_attrs_templates': 'extract_attrs_templates',
                     'digit_median_dists': 'digit_median_dists.p',
                     'blank2attrpatch': 'blank2attrpatch.p',
                     'invblank2attrpatch': 'invblank2attrpatch.p',
                     'digitmultexemplars': 'digitmultexemplars',
                     'digitmultexemplars_map': 'digitmultexemplars_map.p',
                     'grouplabels_record': 'grouplabels_record.p',
                     'devmode': True}
        self.createFields()

    def addCloseEvent(self, func):
        Project.closehook.append(func)

    def removeCloseEvent(self, func):
        Project.closehook = [x for x in Project.closehook if x != func]

    def createFields(self):
        for k, v in self.vals.items():
            setattr(self, k, v)

    def path(self, field):
        '''
        Find the path to the saved part of this project
        '''
        return path.join(self.projdir_path, field)

    def path_exists(self, field):
        '''
        Find out whether a part of the project has yet been saved
        '''
        return path.exists(path.join(self.projdir_path, field))

    # ----

    def is_grouped(self):
        '''
        Returns True if the ballots have been sorted into groups
        '''
        return self.path_exists(self.group_to_ballots)

    def is_partitioned(self):
        '''
        Returns True if the ballots have been properly partitioned
        '''
        return self.path_exists(self.partitions_map)

    def has_attribute_data(self):
        return self.path_exists('_state_ballot_attributes.p')

    def use_partitions_as_grouping(self):
        '''
        In the case that we have partitions but no groups, we can use
        the ballot partitioning as our grouping.
        '''
        partitions_map = self.load_field(self.partitions_map)
        partitions_invmap = self.load_field(self.partitions_invmap)
        partitions_exmpls = self.load_field(self.partition_exmpls)

        group_infomap = {}
        group_to_ballots = {}
        ballot_to_group = {}
        group_examples = {}

        for (group_id, (part_id, ballots)) in \
            enumerate(sorted(partitions_map.items())):
            if not ballots:
                continue

            group_infomap[group_id] = { 'pid': part_id }
            group_to_ballots.setdefault(group_id, []).extend(ballots)

            for b_id in ballots:
                ballot_to_group[b_id] = group_id

        for (group_id, (part_id, ballots)) in \
            enumerate(sorted(partitions_exmpls.items())):
            if not ballots:
                continue

            group_examples[group_id] = ballots

        with open(self.grouping_results, 'wb') as csvfile:
            dictwriter = csv.DictWriter(csvfile,
                                        fieldnames=('ballotid', 'groupid'))
            dictwriter.writeheader()
            dictwriter.writerows((
                {'ballotid': b_id, 'groupid': g_id }
                for (b_id, g_id) in ballot_to_group.items()
            ))

        self.save_field(group_to_ballots, self.group_to_ballots)
        self.save_field(ballot_to_group, self.ballot_to_group)
        self.save_field(group_examples, self.group_exmpls)
        self.save_field(group_infomap, self.group_infomap)

    # ----

    def load_field(self, field):
        '''
        Load the named field from the disk. This assumes it exists,
        and if it does not, it will raise a FileNotFound exception.
        '''
        with open(self.path(field), 'rb') as f:
            return pickle.load(f)

    def save_field(self, value, field):
        '''
        Save the value to disk as the named field.
        '''
        with open(self.path(field), 'wb') as f:
            pickle.dump(value, f, pickle.HIGHEST_PROTOCOL)

    def save(self):
        '''
        Save the entire project to disk.
        '''
        debug('saving project: {0}', self)
        write_project(self)

    def __repr__(self):
        return 'Project({0})'.format(self.name)


def load_projects(projdir):
    """ Returns a list of all Project instances contained in PROJDIR.
    Input:
        str PROJDIR:
    Output:
        list PROJECTS.
    """
    projects = []
    dummy_proj = Project()
    # for dirpath, dirnames, filenames in os.walk(projdir):
    #    for f in filenames:
    try:
        os.makedirs(projdir)
    except:
        pass

    for subfolder in os.listdir(projdir):
        if os.path.isdir(path.join(projdir, subfolder)):
            for f in os.listdir(path.join(projdir, subfolder)):
                if f == PROJ_FNAME:
                    fullpath = path.join(projdir, path.join(subfolder, f))
                    try:
                        proj = pickle.load(open(fullpath, 'rb'))
                        # Add in any new Project properties to PROJ
                        for prop, propval_default in dummy_proj.vals.iteritems():
                            if not hasattr(proj, prop):
                                debug('adding property {0}->{1} to project',
                                      prop,
                                      propval_default)
                                setattr(proj, prop, propval_default)
                        projects.append(proj)
                    except:
                        pass
    return projects


def create_project(name, projrootdir):
    proj = Project(name, projrootdir)
    projoutpath = path.join(projrootdir, PROJ_FNAME)
    try:
        os.makedirs(projrootdir)
    except:
        pass
    pickle.dump(proj, open(projoutpath, 'wb'))
    return proj


def write_project(project):
    projoutpath = path.join(project.projdir_path, PROJ_FNAME)
    pickle.dump(project, open(projoutpath, 'wb'))
    return project
