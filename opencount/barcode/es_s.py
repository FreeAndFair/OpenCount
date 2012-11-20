import sys, os, time
import cv

def decode_patch(original_image, original_mark, expected_bits):
    """
    Given a ES&S-style ballot, returns the LHS barcode as a bitstring 
    if one is found, along with bounding box of each digit in the barcode.
    The algorithm works by finding finding the column of timing marks on 
    the left side of the ballot and looking at the intensity of pixels
    just to the right of each of them to detect "on" or "off" bits.
    Input:
        original_image : cv image of ballot
        original_mark  : cv image of mark
        expected_bits  : number of bits expected in barcode
    Output:
        bitstring : string representation of barcode (ex: "100110...")
        locations : {bit_value: [(x1,y1,x2,y2), ...]}
    """

    resized_mark_height = 20  # cannot be too low or will not match
    portion = 5               # cannot be too high or will error

    mark_w,mark_h = cv.GetSize(original_mark)
    scaling = float(resized_mark_height)/mark_h
    w = int(round(mark_w * scaling))
    h = int(round(mark_h * scaling))
    resized_mark = cv.CreateImage((w, h), 8, 1)
    cv.Resize(original_mark, resized_mark)

    image_W, image_H = cv.GetSize(original_image)
    cv.SetImageROI(original_image, (0, 0, int(round(image_W/portion)), image_H))
    W = int(round(image_W / portion * scaling))
    H = int(round(image_H * scaling))
    resized_image = cv.CreateImage((W, H), 8, 1)
    cv.Resize(original_image, resized_image)
    width = W - w + 1
    height = H - h + 1
    match_mat = cv.CreateImage((width, height), 32, 1)
    cv.MatchTemplate(resized_image, resized_mark, match_mat, cv.CV_TM_CCOEFF_NORMED)
    cv.ResetImageROI(original_image)

    best_column = 0
    most_matches = 0
    for x in range(width):
        column_matches = 0
        for y in range(height):
            if match_mat[y,x] > 0.7:
                column_matches += 1
        if column_matches > most_matches:
            most_matches = column_matches
            best_column = x
    
    last_max = 0.5
    last_y = 0
    is_possible = False
    y_locations = []
    for y in range(height):
        p = match_mat[y,best_column]
        if p > last_max:
            last_max = p
            last_y = y
            is_possible = True
        if is_possible and p < 0.4:
            y_locations.append(last_y)
            last_max = 0.5
            is_possible = False
    
    y_locations = y_locations[1:-1]
    error_value = (None, None)
    if len(y_locations) != expected_bits:
        return error_value
    dist = y_locations[1] - y_locations[0]
    for i in range(2, len(y_locations)):
        diff = y_locations[i] - y_locations[i-1]
        if abs(diff-dist) > dist*0.1:
            return error_value

    x_start = best_column + (2*w)
    x_end = x_start + w
    threshold = 0.7 * 255 * w
    bitstring = ''
    bit_locations = {}
    for y in y_locations:
        intensity = 0
        digit = ''
        for x in range(x_start, x_end):
            intensity += resized_image[y+(h/2),x]
        digit = '1' if (intensity < threshold) else '0'
        bitstring += digit
        resized_locations = [x_start, y, x_end, y+h]
        mark_location = tuple([int(round(z/scaling)) for z in resized_locations])
        bit_locations.setdefault(digit, []).append(mark_location)
    return bitstring, bit_locations

def decode(imgpath, mark, bits):
    """
    Given a ES&S-style ballot, returns the LHS barcode as a bitstring. 
    Will try to detect and report flipped ballots.
    Input:
        imgpath : path to ballot image
        mark    : image of mark
        bits    : number of bits expected in barcode
    Output:
        bitstring     : string for detected barcode
        is_flipped    : boolean indicating whether ballot was flipped
        bit_locations : {str bit_value: [(x1,y1,x2,y2), ...]}
    """

    is_flipped = False
    img = cv.LoadImage(imgpath, cv.CV_LOAD_IMAGE_GRAYSCALE)
    bitstring, bit_locations = decode_patch(img, mark, bits)
    if not bitstring:
        is_flipped = True
        w, h = cv.GetSize(img)
        tmp = cv.CreateImage((w,h), img.depth, img.channels)
        cv.Flip(img, tmp, flipMode=-1)
        img = tmp
        bitstring, bit_locations = decode_patch(img, mark, bits)
    return bitstring, is_flipped, bit_locations

def build_bitstrings(img_bit_locations, expected_bits):
    """
    For each ballot, build bitstring from the locations of the barcode digits.
    Input:
        img_bit_locations : {imgpath: [(bit_value, y), ...]}
        expected_bits     : number of bits expected in the bitstring
    Output:
        img_decoded_map : {imgpath: bitstring}
    """

    img_decoded_map = {}
    for imgpath, tups in img_bit_locations.iteritems():
        tups_sorted = sorted(tups, key=lambda x: x[1])
        decoding = ''.join([str(tup[0]) for tup in tups_sorted])
        assert len(decoding) == expected_bits
        img_decoded_map[imgpath] = decoding
    return img_decoded_map

def get_info(bitstring):
    """
    Converts the barcode bitstring to dictionary with info about the ballot.
    Input:
        bitstring : string representation of barcode on ballot
    Output:
        info : {infotype: value}
    """

    info = {'page': 0}  # TODO: change once barcode meaning is understood
    return info

def main():
    imgpath = "110204-General-Sacramento-County.png" 
    mark_path = "ess_mark.png"
    bits = 41
    trials = 10

    start = time.time()
    mark = cv.LoadImage(mark_path, cv.CV_LOAD_IMAGE_GRAYSCALE)
    for i in range(trials):
        bitstring, is_flipped, bit_locations = decode(imgpath, mark, bits)
        print "%s\t%s\t%s" % (imgpath, is_flipped, bitstring)
    print "Time/ballot: %s" % str((time.time() - start)/trials)

    print "\nTesting build_bitstrings():"
    img_bc_temp = {}
    for bit_value, tups in bit_locations.iteritems():
        for (x1, y1, x2, y2) in tups:
            img_bc_temp.setdefault(imgpath, []).append((bit_value, y1))
    img_decoded_map = build_bitstrings(img_bc_temp, bits)
    print img_decoded_map


if __name__ == '__main__':
    main()

