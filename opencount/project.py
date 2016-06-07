'''
This contains the project class and various helper functions.
In the future, all loading and saving logic should exist here,
instead of scattered throughout the codebase.
'''

import contextlib
import csv
import os
from os import path
import re
import shutil
try:
    import cPickle as pickle
except ImportError:
    import pickle

import util
from util import debug

PROJ_FNAME = 'proj.p'


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
                     'ocr_tmp_dir': 'ocr_tmp_dir',
                     'contest_id': 'contest_id.csv',
                     'contest_text': 'contest_text.csv',
                     'contest_internal': 'contest_internal.p',
                     'contest_grouping_data': 'contest_grouping_data.p',
                     'target_locs_dir': 'target_locations',
                     'tmp': 'tmp',
                     'extracted_dir': 'extracted',
                     'extracted_metadata': 'extracted_metadata',
                     'ballot_metadata': 'ballot_metadata',
                     'classified': 'classified',
                     'timing_runtarget': 'timing_runtarget',
                     'threshold_internal': 'threshold_internal.p',
                     'sample_flipped': 'sample_flipped',
                     'extractedfile': 'extractedfile',
                     'ballot_to_targets': 'ballot_to_targets.p',
                     'targets_result': 'targets_result.csv',
                     'ballot_to_images': 'ballot_to_images.p',
                     'image_to_ballot': 'image_to_ballot.p',
                     'election_results': 'election_results.txt',
                     'election_results_batches':
                     'election_results_batches.txt',
                     'cvr_csv': 'cvr.csv',
                     'cvr_dir': 'cvr',
                     'quarantined': 'quarantined.csv',
                     'quarantined_manual': 'quarantined_manual.csv',
                     'quarantine_res': 'quarantine_res.csv',
                     'quarantine_attributes': 'quarantine_attributes.csv',
                     'quarantine_internal': 'quarantine_internal.p',
                     'extracted_precinct_dir': 'extracted_precincts',
                     'ballot_grouping_metadata': 'ballot_grouping_metadata',
                     'patch_loc_dir': 'precinct_locations',
                     'attr_internal': 'attr_internal.p',
                     'grouping_results': 'grouping_results.csv',
                     'ballot_attributesfile': 'ballot_attributes.p',
                     'imgsize': (0, 0),
                     'frontback_map': 'frontback_map.p',
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
        partitions_exmpls = self.load_field(self.partition_exmpls)

        group_infomap = {}
        group_to_ballots = {}
        ballot_to_group = {}
        group_examples = {}

        for (group_id, (part_id, ballots)) in \
                enumerate(sorted(partitions_map.items())):
            if not ballots:
                continue

            group_infomap[group_id] = {'pid': part_id}
            group_to_ballots.setdefault(group_id, []).extend(ballots)

            for b_id in ballots:
                ballot_to_group[b_id] = group_id

        for (group_id, (part_id, ballots)) in \
                enumerate(sorted(partitions_exmpls.items())):
            if not ballots:
                continue

            group_examples[group_id] = ballots

        with open(self.path(self.grouping_results), 'wb') as csvfile:
            dictwriter = csv.DictWriter(csvfile,
                                        fieldnames=('ballotid', 'groupid'))
            dictwriter.writeheader()
            dictwriter.writerows((
                {'ballotid': b_id, 'groupid': g_id}
                for (b_id, g_id) in ballot_to_group.items()
            ))

        self.save_field(group_to_ballots, self.group_to_ballots)
        self.save_field(ballot_to_group, self.ballot_to_group)
        self.save_field(group_examples, self.group_exmpls)
        self.save_field(group_infomap, self.group_infomap)

    # ----

    def load_field(self, field, **kwargs):
        '''
        Load the named field from the disk. This assumes it exists,
        and if it does not, it will raise a FileNotFound exception.
        '''
        try:
            with open(self.path(field), 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            if 'default' in kwargs:
                return kwargs['default']
            else:
                raise e

    def load_field_default(self, field, default=None):
        '''
        Load the named field from the disk, returning a specified
        default value or None if the field does not exist.
        '''
        try:
            self.load_field(field)
        except:
            return default

    def save_field(self, value, field):
        '''
        Save the value to disk as the named field.
        '''
        with open(self.path(field), 'wb') as f:
            pickle.dump(value, f, pickle.HIGHEST_PROTOCOL)

    def read_csv(self, field):
        '''
        Open the given field as a CSV file and produce the rows
        in order.
        '''
        with open(self.path(field)) as f:
            for line in csv.reader(f):
                yield line

    @contextlib.contextmanager
    def write_csv(self, field):
        '''
        Create a context manager that exposes access to a CSV writer.
        '''
        with open(self.path(field), 'w') as f:
            yield csv.writer(f)

    @contextlib.contextmanager
    def open_field(self, field, mode='r'):
        '''
        Create a context manager to access the raw file handler for
        a given field.
        '''
        with open(self.path(field), mode) as f:
            yield f

    def load_raw_field(self, field):
        '''
        Load a field as a string.
        '''
        with self.open_field(field) as f:
            return f.read()

    def save_raw_field(self, value, field):
        '''
        Save a field as a string.
        '''
        with self.open_field(field, 'w') as f:
            f.write(value)

    def save(self):
        '''
        Save the entire project to disk.
        '''
        debug('saving project: {0}', self)
        self.write_project()

    def __repr__(self):
        return 'Project({0})'.format(self.name)

    def exists_attrs(self):
        '''
        Returns True if the project has any attributes. Does not
        take into account custom attributes.
        '''
        return (self.path_exists(self.ballot_attributesfile) and
                self.load_field(self.ballot_attributesfile))

    def has_digitbasedattr(self):
        '''
        Returns True if any of the project's attributes are digit-based.
        '''
        if not self.exists_attrs():
            return False
        attrs = self.load_field(self.ballot_attributesfile)
        return any(a['is_digitbased'] for a in attrs)

    def has_imgattr(self):
        '''
        Returns True if any of the project's attributes are image-based.
        '''
        if not self.exists_attrs():
            return False
        attrs = self.load_field(self.ballot_attributesfile)
        return any(not a['is_digitbased'] for a in attrs)

    def has_custattr(self):
        '''
        Returns True if the project has any custom attributes.
        '''
        if not self.exists_attrs():
            return False
        return len(self.load_field(self.attrprops)['CUSTATTR'])

    def get_ioerr_ballots(self):
        '''
        Returns a list of all ballotids that had some image that
        was unable to be read by OpenCount during Partitioning.
        '''
        return list(set(self.load_field(self.partition_ioerr)))

    def get_discarded_ballots(self):
        '''
        Returns a list of all ballotids discarded prior to grouping.
        '''
        return list(set(self.load_field(self.partition_discarded)))

    def get_quarantined_ballots(self):
        '''
        Returns a list of all ballotids quarantined prior to grouping.
        '''
        return list(set(self.load_field(self.partition_quarantined)))

    @staticmethod
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

    @staticmethod
    def load_project(projdir, name):
        proj_path = path.join(projdir, name, PROJ_FNAME)
        if path.exists(proj_path):
            with open(proj_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ProjectLoadException(
                "The files for project '{0}' cannot be found.".format(name))

    @staticmethod
    def create_project(name, basedir):
        if not Project.is_valid_projectname(name):
            raise ProjectCreationException(
                "'{0}' is not a valid project name. "
                "Please use only letters, numbers, and "
                "punctuation.".format(name))
        projrootdir = path.join(basedir, name)
        if path.exists(path.join(projrootdir, PROJ_FNAME)):
            raise ProjectCreationException(
                "The project '{0}' already exists. ".format(name))
        proj = Project(name, projrootdir)
        projoutpath = path.join(projrootdir, PROJ_FNAME)
        try:
            os.makedirs(projrootdir)
        except:
            pass
        pickle.dump(proj, open(projoutpath, 'wb'))
        return proj

    @staticmethod
    def is_valid_projectname(name):
        """
        Only allow letters, numbers, and [_, (, )].
        """
        pattern = r'(\w|\d|[_\()])+'
        return ' ' not in name and (not re.match(pattern, name) is None)

    @staticmethod
    def delete_project(projdir, name):
        '''
        Delete all the files for a project from the file system.
        '''
        shutil.rmtree(path.join(projdir, name))

    def write_project(self):
        # path.join(project.projdir_path, PROJ_FNAME)
        projoutpath = self.path(PROJ_FNAME)
        pickle.dump(self, open(projoutpath, 'wb'))
        return self


class ProjectCreationException(util.InformativeException):
    '''
    An informative exception that arises in the course of creating
    a new project.
    '''
    pass


class ProjectLoadException(util.InformativeException):
    '''
    An informative exception that arises in the course of loading
    an existing project's data.
    '''
    pass
