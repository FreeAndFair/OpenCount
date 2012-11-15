import cv
import sys

"""
A script that decodes Sequoia-style barcodes into 01 bitstrings.

Usage:
    python sequoia.py [List of Images]

Assumptions:
    - Intensity, sizes, and alignment of images are rougly the same within a
      ballot.
    - Two sample templates, one for 0 one for 1, have been cropped out
      manually.

TODO:
    - Overlay verification
    - Integration into Vendor interface
"""

def crop(img, left, top, new_width, new_height):
    """Crops img, returns the region defined by (left, top, new_width,
    new_height)
    """
    if left + new_width > img.width:
        new_width = img.width - left

    cropped = cv.CreateImage((new_width, new_height), cv.IPL_DEPTH_8U, img.channels)
    src_region = cv.GetSubRect(img, (left, top, new_width, new_height))

    cv.Copy(src_region, cropped)
    return cropped

def crop_rough_left(img):
    """Roughly crops the upper left barcode region."""
    x_per = 0.0
    #x2_per = 0.07917
    x2_per = 0.11
    #y_per = 0.02604
    y_per = 0.02
    y2_per = 0.13542
    width = int((x2_per - x_per) * img.width)
    height = int((y2_per - y_per) * img.height)
    return crop(img, int(x_per * img.width), int(y_per * img.height), width, height)

def crop_rough_right(img):
    """Roughly crops the upper right barcode region."""
    #x_per = 0.89167
    x_per = 0.89
    x2_per = 1.0
    y_per = 0.02
    y2_per = 0.13542
    width = int((x2_per - x_per) * img.width)
    height = int((y2_per - y_per) * img.height)
    return crop(img, int(x_per * img.width), int(y_per * img.height), width, height)

def processImg(img, template_zero, template_one):
    """ The pipeline for processing one image:
        1) crop out two rough barcode regions from the image
        2) run template matching against it with two templates with criteria,
           retrieving the best matches
        3) process matching result, transform into 01-bitstring
    """
    rough_left_barcode = crop_rough_left(img)
    rough_right_barcode = crop_rough_right(img)
    cv.SaveImage("rough_left_barcode.png", rough_left_barcode)
    cv.SaveImage("rough_right_barcode.png", rough_right_barcode)


    left_zero_best_locs = bestMatchedLocs(rough_left_barcode, template_zero)
    left_one_best_locs = bestMatchedLocs(rough_left_barcode, template_one)
    right_zero_best_locs = bestMatchedLocs(rough_right_barcode, template_zero)
    right_one_best_locs = bestMatchedLocs(rough_right_barcode, template_one)

    left_best_locs = postprocess_locs(left_zero_best_locs, left_one_best_locs)
    right_best_locs = postprocess_locs(right_zero_best_locs, right_one_best_locs)

    bit_string = [transformToBits(left_best_locs, rough_left_barcode),
            transformToBits(right_best_locs, rough_right_barcode)]

    return bit_string

def postprocess_locs(zero_locs, one_locs):
    """Post processing the locations:
        - sort them by height
        - delete the potential false positive for the top bar
        - ensure every loc's width is roughly the same
    """
    zero_locs = sorted(zero_locs, key=lambda tup: tup[1])
    one_locs = sorted(one_locs, key=lambda tup: tup[1])

    del(one_locs[0])

    zero_locs = iterative_validate_width(zero_locs)
    one_locs = iterative_validate_width(one_locs)

    return [zero_locs, one_locs]

def iterative_validate_width(locs):
    for i in range(1, len(locs) - 1):
        if abs(locs[i][0] - locs[i - 1][0]) > 10 and abs(locs[i][0] - locs[i + 1][0]) > 10:
            del(locs[i])
            return iterative_validate_width(locs)
            
    if len(locs) > 1 and abs(locs[0][0] - locs[1][0]) > 10:
        del(locs[0])
    if len(locs) > 1 and abs(locs[len(locs) - 1][0] - locs[len(locs) - 2][0]) > 10:
        del(locs[len(locs) - 1])
    return locs


def transformToBits(best_locs, img):
    """Assumes best_locs are the correct locations (except that in one_locs,
    the loc with smallest height is a false positive, namely the top bar).
    """
    zero_locs = best_locs[0]
    one_locs = best_locs[1]

    i = 0
    j = 0
    bit_string = ""

    while i < len(zero_locs) and j < len(one_locs):
        if zero_locs[i][1] < one_locs[j][1]:
            bit_string = bit_string + '0'
            i += 1
        else:
            bit_string = bit_string + '1'
            j += 1

    while i < len(zero_locs):
        bit_string = bit_string + '0'
        i += 1
    while j < len(one_locs):
        bit_string = bit_string + '1'
        j += 1

    return bit_string

def bestMatchedLocs(src_img, template):
    """After finding a best location, marks the region which has the best
    location as its upper left and is covered by a region of size of template
    as absolutely irrelevant (-1). In addition, prevent locations with similar
    height. Returns a list of best matched locations.
    """
    h_src = src_img.height
    w_src = src_img.width
    h_temp = template.height
    w_temp = template.width
    lastMaxVal = -1.0
    threshold = 0.2
    best_locs = []
    threshold_reached = False
    res_mat = cv.CreateMat(h_src - h_temp + 1, w_src - w_temp + 1, cv.CV_32F)
    cv.MatchTemplate(src_img, template, res_mat, cv.CV_TM_CCOEFF_NORMED)

    while len(best_locs) < 8 and not threshold_reached:
        minVal, maxVal, minLoc, maxLoc = cv.MinMaxLoc(res_mat)
        if len(best_locs) == 0 or abs(maxVal - lastMaxVal) < threshold:
            if possible_duplicate(best_locs, maxLoc[1], 10):
                fill(res_mat, maxLoc[0], w_temp, maxLoc[1], h_temp, -1.0)
                continue
            best_locs.append(maxLoc)
            if len(best_locs) == 1:
                lastMaxVal = maxVal
        elif abs(maxVal - lastMaxVal) >= threshold:
            threshold_reached = True
            continue

        fill(res_mat, maxLoc[0], w_temp, maxLoc[1], h_temp, -1.0)
    return best_locs


def possible_duplicate(locs, loc_h, min_dis=5):
    """Judge if loc_h is within min_dis of any loc inside locs."""
    for loc in locs:
        if abs(loc[1] - loc_h) <= min_dis:
            return True
    return False

def fill(res_mat, w, w_len, h, h_len, val):
    reached = False
    for x in range(1, w_len):
        for y in range(1, h_len):
            if w + x > res_mat.cols or h + y > res_mat.height:
                continue
            else:
                if res_mat[h + y - 1, w + x - 1] == val:
                    reached = True
                    break
                else:
                    res_mat[h + y - 1, w + x - 1] = val
        if reached:
            break


def main():
    template_one_path = "template_one.png"
    template_zero_path = "template_zero.png"

    if len(sys.argv) > 1:
        img_paths = sys.argv[1:]
        for img_path in img_paths:
            print processImg(cv.LoadImageM(img_path,
                cv.CV_LOAD_IMAGE_GRAYSCALE), cv.LoadImageM(template_zero_path,
                    cv.CV_LOAD_IMAGE_GRAYSCALE),
                cv.LoadImageM(template_one_path, cv.CV_LOAD_IMAGE_GRAYSCALE))

if __name__ == '__main__':
    main()
