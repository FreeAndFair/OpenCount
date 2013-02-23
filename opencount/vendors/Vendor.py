"""
An abstract base class to be used for vendor-specific functions, such
as barcode decoding.
"""

class Vendor(object):
    def __init__(self, proj):
        pass
        #raise NotImplementedError("Can't instantiate abstract Vendor class.")

    def decode_ballots(self, ballots, *args, **kwargs):
        """
        Use automated image-processing algorithms to output an initial
        decoding for each voted ballot. The user will then verify the
        output using overlay-verification.

        Input:
            dict BALLOTS: {int ballotID: [imgpath_side0, ...]}.
        Output:
            (dict FLIP_MAP,
             dict VERIFYPATCH_BBS,
             list ERR_IMGPATHS,
             list IOERR_IMGPATHS)

        FLIP_MAP: stores whether an image is flipped or not:
            {str imgpath: bool isFlipped}
        VERIFYPATCHES_BBS: stores the locations of barcode patches that
            need to be verified by the user:
                {str bc_val: [(str imgpath, (x1,y1,x2,y2), userinfo), ...]}
            For instance, the keys to this dict could be '0' and '1',
            indicating absence/presence of marks.
            USERINFO can be any additional information that your decoder
            needs, or simply None. In many cases you won't need USERINFO,
            but this is present to allow meta-data to flow from
            Vendor.decode_ballots to Vendor.partition_ballots if need be.
        ERR_IMGPATHS: List of voted imgpaths that were unable to be 
            successfully decoded. These will be handled specially, by
            having the user manually enter the barcode values.
        IOERR_IMGPATHS: List of voted imgpaths that were unable to be
            read/loaded (i.e. IOError). OpenCount will set these aside
            separately (along with its associated images, if any).
        """
        raise NotImplementedError("Implement your own decode_ballots.")

    def partition_ballots(self, verified_results, manual_labeled):
        """
        Given the user-verified, corrected results of Vendor.decode_ballots,
        output the partitioning.

        Input:
            dict VERIFIED_RESULTS: {str bc_val: [(str imgpath, (x1,y1,x2,y2), userinfo), ...]}
            dict MANUAL_LABELED: {str imgpath: (str label_i, ...)}
                Stores the decoded barcode(s) of any images that the user
                manually entered.
                This maps to a /tuple/ of strings, because there may be
                multiple barcodes on a given image.
        Output:
            (dict PARTITIONS, 
             dict IMG2DECODING,
             dict IMAGE_INFO)

        PARTITIONS: stores the partitioning as:
            {int partitionID: [int ballotID_i, ...]}
        IMG2DECODING: stores barcode strings for each image as:
            {str imgpath: [str bc_i, ...]}
            Note: IMG2DECODING maps each imgpath to a /tuple/ of 
                  strings. This is to account for ballots where multiple
                  barcodes may be present in a single image.
        IMAGE_INFO: stores meaningful info for each image:
                {str imgpath: {str PROPNAME: str PROPVAL}}
            where PROPNAME could be 'page', 'party', 'precinct', etc.
            Note: The key 'page' /must/ be present, and should
                  map to an integer. i.e. 0/1 is Front/Back
                  respectively.
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
            [(int count, (int upper, int lower)),...], the upper and lower coords of each thing to extract
        """

        target_x_pos = [x[0] for x in targets]
        target_x_range = max(target_x_pos)-min(target_x_pos)
        target_y_pos = [x[1] for x in targets]
        target_y_range = max(target_y_pos)-min(target_y_pos)
        l,u,r,d = contest
        tops = sorted([a[1]-u-10 for a in targets])+[d]
        if tops[0] > 0:
            tops = [0]+tops
        else:
            print "it was negative", tops
            tops = [0,0]+tops[1:] # In case the top is negative.
        print "now tops", tops
        if target_y_range > target_x_range:
            # it is a vertical contest
            blocks = []
            for count,upperlower in enumerate(zip(tops, tops[1:])):
                blocks.append((count, upperlower))
            
            return blocks
        else:
            print "AAAAA", list(enumerate(zip(tops, tops[1:])))
            return [(0,(tops[0],tops[1])), #header
                    (1,(tops[1], d)),
                    (2,(d, d))]
        

    def __repr__(self):
        return 'Vendor()'
    def __str__(self):
        return 'Vendor()'
