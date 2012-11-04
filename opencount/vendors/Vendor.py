"""
An abstract base class to be used for vendor-specific functions, such
as barcode decoding.
"""

class Vendor(object):
    def __init__(self):
        raise NotImplementedError("Can't instantiate abstract Vendor class.")

    def decode_image(self, imgpath):
        """ 
        Input:
            str IMGPATH:
        Output:
            list OUTPUT. Stores a list of all decoded barcodes,
                a boolean saying whether or not the image is flipped, and
                the location of each barcode:
                    [[str bc_i, ...], bool is_flip, [(x1,y1,x2,y2), ...]].
        """
        raise NotImplementedError

    def decode_ballot(self, ballot):
        """ Tries to decode the barcodes/timing marks present on BALLOT.
        If an error occurs, then decode_barcodes is expected to raise
        an exception with an appropriate error message.
        Input:
            list BALLOT: List of image paths, together which correspond
                to one ballot. For instance, if the election is double-sided,
                then BALLOT should be a list of length two.
        Output:
            list RESULTS. RESULTS is a list of lists, where each sublist
                contains information on each image:
                    [[(str bc_i, ...), bool isflipped, [(x1,y1,x2,y2), ...]], ...]
        """
        # Here's an example body -- feel free to override me
        decodeds = []
        isflips = []
        bbs_all = []
        for imgpath in ballot:
            bcs, isflip, bbs = self.decode_image(imgpath, *_args, **_kwargs)
            decoded.append(bcs)
            isflip.append(isflip)
            bbs_all.append(bbs)
        return decodeds, isflips, bbs_all

    def partition_ballots(self, ballots):
        """
        Input:
            dict BALLOTS: {int ballotID: [imgpath_side0, ...]}.
        Output:
            (dict PARTITIONS, dict DECODED, dict BALLOT_INFO, dict BBS_MAP), where PARTITIONS stores the
                partitioning as:
                    {int partitionID: [int ballotID_i, ...]}
                and DECODED stores barcode strings for each ballot as:
                    {int ballotID: [(str BC_side0i, ...), (str BC_side1i, ...)]}
                and IMAGE_INFO stores meaningful info for each image (extracted
                from the barcode):
                    {str imgpath: {str KEY: str VAL}}
                where KEY could be 'page', 'party', 'precinct', 'isflip', etc, and
                BBS_MAP stores the location of the barcodes:
                    {str imgpath: [(x1, y1, x2, y2), ...]}
        """
        raise NotImplementedError("Implement your own partition_ballots.")

    def split_contest_to_targets(self, ballot_image, contest, targets):
        """
        Given an image of a contest, extract 
            (a) the tile and 
            (b) each of the voting targets
        
        Input:
            PIL Image: ballot_image
            (int left, int up, int right, int down) contest
            targets: [(int left, int up, int right, int down),...]
        
        Output:
            [(int upper, int lower),...], the upper and lower coords of each thing to extract
        """
        
        l,u,r,d = contest
        tops = sorted([a[1]-u-10 for a in targets])+[d]
        if tops[0] > 0:
            tops = [0]+tops
        else:
            tops = [0,0]+tops[1:] # In case the top is negative.

        blocks = []
        for upper,lower in enumerate(zip(tops, tops[1:])):
            blocks.append((upper, lower))
        
        

    def __repr__(self):
        return 'Vendor()'
    def __str__(self):
        return 'Vendor()'
