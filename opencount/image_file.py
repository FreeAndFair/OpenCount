import os
import pickle
import pdb
import shutil
import tempfile
import time
import traceback

from PIL import Image
try:
    from wx.lib.pubsub import pub
except:
    from wx.lib.pubsub import Publisher
    pub = Publisher()
import wx
import numpy as np

import extsort
import util


def is_image_ext(filename):
    IMG_EXTS = ('.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff')
    return os.path.splitext(filename)[1].lower() in IMG_EXTS

METHOD_DYN = 0    # Dynamically determine sort method
METHOD_MEM_SORT = 1    # In-memory sorting
METHOD_EXT_SORT = 2    # External merge-sort


def makeOneFile(src, radix, dst, targetdims, MEM_C=0.8, SORT_METHOD=METHOD_DYN):
    """ Creates an ImageFile, which is a single binary file containing
    raw image pixel data.
    Writes three output files:
        DST       -- The imagefile binary data
        DST.type  -- A file signifying the 'type' (gray/RGB) of the images
                     in DST. "A" -> grayscale, "B" -> RGB.
        DST.size  -- Contains the image dimensions (width, height).
    Input:
        tuple SRC: [(str targetid, float avg_intensity), ...]
            This is sorted by average intensity (increasing). The
            targetid follows the format:
                <int ballotid>\0<int page>\0<int targuid>
        str RADIX:
        str DST:
        tuple TARGETDIMS: (int w, int h)
        float MEM_C: float in [0.0, 1.0]
            Limits memory usage of this script s.t. if MEM_C * <avail_mem>
            is exceeded, then an external merge-sort is run. Otherwise,
            radix files are sorted in memory.
        int SORT_METHOD:
            Determines which sorting method to use.
    """
    # NOTE: If a ballot B is quarantined after target extraction (i.e.
    # during the quarantine-check), then that ballot will NOT appear
    # in the SRC input, but it WILL appear in the radix buckets.
    # Unchecked, this WILL cause crashes to happen down the line!
    reverse_mapping = {}  # maps {str targetid: int idx}
    for i, x in enumerate(src):
        reverse_mapping[x[0]] = i

    out = open(dst, "wb")
    tout = open(dst + ".type", "wb")

    imgSize = targetdims[0] * targetdims[1]

    for index in range(256):
        which = "%02x" % index
        data = []
        names = []
        n = time.time()

        filepairs = []  # [(str filepath, str indexpath, int size), ...]
        for directory in os.listdir(radix):
            dired = os.path.join(radix, directory)
            fullpath = os.path.join(dired, which)
            if os.path.exists(fullpath):
                filepairs.append(
                    (fullpath, fullpath + ".index", os.path.getsize(fullpath)))

        total_size = sum([tup[2] for tup in filepairs])
        try:
            mem_avail, mem_total = util.get_memory_stats()
        except:
            mem_avail, mem_total = np.inf, np.inf
        # print "(Info) {0} Radix files, {1}MB.   Avail: {2}MB".format(len(filepairs),
        #                                                             total_size / 1e6,
        # mem_avail / 1e6)
        if SORT_METHOD == METHOD_EXT_SORT:
            mem_avail = -1  # Force the external-sort method
        if (total_size < mem_avail) or (SORT_METHOD == METHOD_MEM_SORT):
            # Sort in-memory as normal
            for filepath, indexpath, size in filepairs:
                content = open(filepath).read()
                data.extend([content[i * imgSize:(i + 1) * imgSize]
                             for i in range(len(content) / imgSize)])
                names.extend(open(indexpath).read().split("\n")[:-1])

            sort_order = sorted([x for x in range(len(data)) if names[
                                x] in reverse_mapping], key=lambda x: reverse_mapping[names[x]])
            sorted_data = [data[i] for i in sort_order]

            out.write("".join(sorted_data))
            tout.write("A" * len(sorted_data))
        else:
            outpath_extsort = os.path.join(os.path.split(dst)[0],
                                           '_extsort.dat')
            # Files won't fit in memory - do external merge-sort instead
            # FILEPATH_SORTED is the path to the sorted file output by
            # the external mergesort.
            num_entries = ext_mergesort([t[0] for t in filepairs],
                                        [t[1] for t in filepairs],
                                        reverse_mapping,
                                        outpath_extsort,
                                        index, imgSize)
            if num_entries == -1:
                print "(makeOneFile) Error: ext_merge_sort returned FAILURE for radix index {0}. Oh no!".format(index)
                pdb.set_trace()
            # Concatenate the sorted radixfile to OUT.
            f_extsort = open(outpath_extsort, 'rb')
            shutil.copyfileobj(f_extsort, out)
            tout.write("A" * num_entries)

    if src:
        open(dst + ".size", "w").write(str(targetdims))

    out.close()
    tout.close()


def ext_mergesort(datafpaths, idxfpaths, reverse_mapping, outpath, radixdigit, imgSize,
                  tempdir=None):
    """ Sorts the data in DATAFPATHS, and outputs the sorted data to
    a single output file OUTPATH. Utilizes the external mergesort
    to allow sorting of data that can't fit in main memory.
    Input:
        tuple DATAFPATHS: [str datapath, ...]
        tuple IDXFPATHS: [str indexpath, ...]
            Used for the sorting criterion.
        dict REVERSE_MAPPING:
            Used for the sorting criterion.
        str OUTPATH:
            Path to save the output sorted data.
        str TEMPDIR:
            Directory to store any intermediate data files.
        int RADIXDIGIT: in [0, 255]
            Which radix digit we are currently sorting.
    Output:
        int NUM_ENTRIES.
    The number of sorted entries. If this fails, returns -1.
    """
    if tempdir is None:
        tempdir = tempfile.gettempdir()
    num_entries = 0
    idxfpath2names = {}  # maps {str idxfpath: str names}
    intervals = []  # [(int tidx_low, int tidx_high, str idxfpath), ...]
    cur_high = 0
    total_targets = 0
    for i, idxfpath in enumerate(idxfpaths):
        names = open(idxfpath).read().split("\n")[:-1]
        idxfpath2names[idxfpath] = names
        ntargets = len(names)
        intervals.append((cur_high, cur_high + ntargets, idxfpath))
        cur_high += ntargets
        total_targets += len(names)
    # 1.) Concatenate all files in DATAFPATHS into one mega-file
    if not os.path.exists(tempdir):
        os.makedirs(tempdir)
    fpath_concat = os.path.join(tempdir, 'radix_{0}.tmp'.format(radixdigit))
    fconcat = open(fpath_concat, 'wb')
    for i, fpath in enumerate(datafpaths):
        shutil.copyfileobj(open(fpath, 'rb'), fconcat)
    fconcat.flush()
    fconcat.close()

    def get_sort_val(targetidx, tdata=None):
        """
        Input:
            int TARGETIDX:
                The target index we are at within FCONCAT.
        Output:
            int SORTVAL. Or None if this should be ignored.
        """
        try:
            for (low, high, idxfpath) in intervals:
                if (targetidx >= low) and (targetidx < high):
                    idx = targetidx - low
                    targetid = idxfpath2names[idxfpath][idx]
                    return reverse_mapping.get(targetid, None)
            print "Woah -- couldn't find interval for targetidx:", targetidx
            print total_targets
            pdb.set_trace()
        except Exception as e:
            traceback.print_exc()
            return -1

    num_written = extsort.batch_sort_mod(
        fpath_concat, outpath, imgSize, key=get_sort_val, DELETE_TEMPFILES=True)
    if os.path.exists(fpath_concat):
        os.remove(fpath_concat)
    return num_written


class ImageFile:

    def __init__(self, inp):
        self.infile = open(inp, "rb")
        self.dims = map(int, open(inp + ".size").read().strip()
                        [1:-1].split(","))
        self.imtype = open(inp + ".type").read()
        self.size = self.dims[0] * self.dims[1]
        self.offsets = []
        off = 0
        for each in self.imtype:
            self.offsets.append(off)
            off += 3 if each == 'B' else 1

    def readRawBytes(self, imagenum, count):
        self.infile.seek(self.size * imagenum)
        return np.fromstring(self.infile.read(self.size * count), dtype='uint8')
        # return [ord(x) for x in self.infile.read(self.size*count)]

    @util.pdb_on_crash
    def readManyImages_filter(self, imagenums, numcols, width, height, curwidth, curheight, returnnumpy=False):
        imagenums = [x for x in imagenums if x < len(self.imtype)]
        imagetypes = [self.imtype[x] for x in imagenums]
        # Three bytes for colored images, one byte otherwise
        types = [3 if x == "B" else 1 for x in imagetypes]
        if not all(x == types[0] for x in types):
            raise Exception("All the images must be L or RGB, not both.")
        toread = sum(types)

        data = [self.readRawBytes(self.offsets[x], y)
                for x, y in zip(imagenums, types)]
        if returnnumpy:
            return data, imagetypes[0] == 'A'

        data = np.concatenate(data)

        if imagetypes[0] == 'A':  # single chanel
            fixed = np.concatenate([data[j:j + self.size].reshape((height, width))
                                    for j in range(0, data.shape[0], self.size)], axis=1)
            jpg = Image.fromarray(fixed)
            tomerge = jpg, jpg, jpg
        else:
            tomerge = []
            for start in range(3):
                fixed = np.concatenate([data[j + start:j + self.size * 3:3].reshape(
                    (height, width)) for j in range(0, data.shape[0], self.size * 3)], axis=1)
                jpg = Image.fromarray(fixed)
                tomerge.append(jpg)
            tomerge = tuple(tomerge)

        realnumcols = (fixed.shape[0] * fixed.shape[1]) / (width * height)
        jpg = Image.merge('RGB', tomerge)
        jpg = jpg.resize((curwidth * realnumcols, curheight))
        return jpg
