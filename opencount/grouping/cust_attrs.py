import sys, os, pdb, pickle, csv
from os.path import join as pathjoin
sys.path.append('..')

"""
Functions that handle the Custom Attributes extension. 

Assumes that the input spreadsheetpaths are .csv-like files of the form:

in, out
in_0, out_0
in_1, out_1
...
in_N, out_N

"""

"""
Output files:

<projdir>/custom_attrs.p
  
A list of marshall'd custom_attributes (i.e. dictionaries):
  [m_custattr_i, ... ]
"""

class CustomAttribute:
    """
    Custom Attribute Modalities:
    """
    M_SPREADSHEET = 0
    M_FILENAME = 1
    
    def __init__(self, attrname, mode=0, sspath=None, attrin=None, filename_regex=None):
        self.attrname = attrname
        self.mode = mode

        """ M_SPREADSHEET """
        self.sspath = sspath
        self.attrin = attrin

        """ M_FILENAME """
        self.filename_regex = filename_regex

def marshall_cust_attr(custattr):
    marsh = {}
    marsh['attrname'] = custattr.attrname
    marsh['mode'] = custattr.mode
    marsh['attrin'] = custattr.attrin
    marsh['sspath'] = custattr.sspath
    marsh['filename_regex'] = custattr.filename_regex
    return marsh

def unmarshall_cust_attr(d):
    return CustomAttribute(d['attrname'], mode=d['mode'], sspath=d['sspath'],
                           attrin=d['attrin'],
                           filename_regex=d['filename_regex'])

def add_custom_attr_ss(proj, attrname, sspath, attrin):
    """ Adds a new SpreadSheet-based Custom Attribute """
    custom_attrs = load_custom_attrs(proj)
    if custom_attrs == None:
        custom_attrs = []
    cattr = CustomAttribute(attrname, mode=CustomAttribute.M_SPREADSHEET,
                            sspath=sspath, attrin=attrin)
    custom_attrs.append(cattr)
    path = pathjoin(proj.projdir_path, proj.custom_attrs)
    dump_custom_attrs(proj, custom_attrs)

def add_custom_attr_filename(proj, attrname, regex):
    """ Adds a new Filename-based Custom Attribute. """
    custom_attrs = load_custom_attrs(proj)
    if custom_attrs == None:
        custom_attrs = []
    cattr = CustomAttribute(attrname, mode=CustomAttribute.M_FILENAME,
                            filename_regex=regex)
    custom_attrs.append(cattr)
    path = pathjoin(proj.projdir_path, proj.custom_attrs)
    dump_custom_attrs(proj, custom_attrs)

def dump_custom_attrs(proj, custattrs=None):
    """ Stores the custom_attributes into the correct output location. """
    if custattrs == None:
        custattrs = load_custom_attrs(proj)
    if custattrs == None:
        custattrs = []
    marshalled = [marshall_cust_attr(cattr) for cattr in custattrs]
    pickle.dump(marshalled, open(pathjoin(proj.projdir_path, proj.custom_attrs), 'wb'))

def load_custom_attrs(proj):
    """ Returns the custom_attrs data structure if it exists, or None
    if it doesn't exist yet.
    Input:
      obj project
    Output:
      list of CustomAttribute instances
    """
    path = pathjoin(proj.projdir_path, proj.custom_attrs)
    if not os.path.exists(path):
        return None
    marshalled = pickle.load(open(path, 'rb'))
    return [unmarshall_cust_attr(m) for m in marshalled]

def custattr_get(custom_attrs, attrname):
    """ Returns the CustomAttribute if it exists in custom_attrs,
    or None otherwise.
    """
    if custom_attrs == None:
        return None
    for cattr in custom_attrs:
        if cattr.attrname == attrname_i:
            return cattr
    return None

def custattr_exists(proj, attrname):
    """ Returns True if attrname is a custom_attribute. """
    custom_attrs = load_custom_attrs(proj)
    if custom_attrs != None:
        return custattr_get(custom_attrs, attrname) != None
    return False

def custattr_map_inval_ss(proj, attrname, attr_inval):
    """ Maps the attr_inval through the custom_attrs. Assumes that
    attr_inval is a string. """
    custom_attrs = load_custom_attrs(proj)
    ss_attrs = [cattr for cattr in custom_attrs if cattr.mode == CustomAttribute.M_SPREADSHEET]
    for attrname_i, sspath, attrin in ss_attrs:
        if attrname_i == attrname:
            csvf = open(sspath, 'rb')
            reader = csv.DictReader(csvf)
            for row in reader:
                if row['in'] == attr_inval:
                    return row['out']
            print "Uhoh, attr_inval wasn't ever found:", attr_inval
            pdb.set_trace()
            assert False
    print "Uhoh, attrname wasn't found in custom_attrs:", attrname
    pdb.set_trace()
    assert False

