import os, sys, time, pdb
import cv

def decode(imgpath, template="diebold-mark.jpg"):
    """
    Input:
        str IMGPATH: Path to voted ballot image.
    Output:
        (str decoding, bool isflip, list BBS)
    """
    if isinstance(template, str) or isinstance(template, unicode):
        Itemp = cv.LoadImage(template, cv.CV_LOAD_IMAGE_GRAYSCALE)
    else:
        Itemp = template
    I = cv.LoadImage(imgpath, cv.CV_LOAD_IMAGE_GRAYSCALE)
    w, h = cv.GetSize(I)
    isflip = False
    # C_LOW, C_HIGH are lower/upper percentage to crop image by
    c_low = 0.9268
    c_high = 0.9909
    y1 = int(round(c_low * h))
    y2 = int(round(c_high * h))
    cv.SetImageROI(I, (0, y1, w, int(y2-y1)))
    decoding, bbs = decode_patch(I, Itemp)
    if decoding in (None, '1'*32):
        # Try Flipping the image, and try again.
        c_low = 1.0 - c_high
        c_high = 1.0 - c_low
        y1 = int(round(c_low * h))
        y2 = int(round(c_high*h))
        cv.SetImageROI(I, (0, y1, w, int(y2-y1)))
        cv.Flip(I, I, flipMode=-1)
        isflip = True
        decoding, bbs = decode_patch(I, Itemp)

    if decoding == None:
        return None, None, None
    else:
        return decoding, isflip, bbs

def decode_patch(patch, template):
    """
    input: path to image, path to template
    output: decoded string, coordinates of black markings in form(x_topleft, y_topleft, x_bottomright, y_bottomright)
    """
    w,h = cv.GetSize(template)
    W,H = cv.GetSize(patch)
    width = W - w + 1
    height = H - h + 1
    result = cv.CreateImage((width, height), 32, 1)
    cv.MatchTemplate(patch, template, result, cv.CV_TM_CCOEFF_NORMED)
    coords = []
    for x in range(width):
        for y in range(height):
            if result[y,x] > .7:
                coords.append((x,y))
    min_distance = 2*w
    delete = []
    for index in range (1, len(coords)):
        distance = coords[index][0] - coords[index-1][0]
        if distance < w:
            delete.append(coords[index])
        if distance > w and distance < min_distance:
           min_distance = distance
    for d in delete:
        coords.remove(d)
    if len(coords) == 0:
        return None, None
    first = coords[0][0]
    coords = [(x - first,y) for x,y in coords]

    last = coords[-1][0]
    if last == 0.0:
        return None, None
    spacing = last / 33.0
    # 33 spots between first and last
    string = ["0" for k in range(0,34)]

    for x,y in coords:
        string[int((x+spacing*.2) // spacing)] = "1"
    out_str = ''.join(string)

    coords = [(x,y,x+w,y+h) for x,y in coords]
    
    return out_str, coords

def isimgext(f):
    return os.path.splitext(f)[1].lower() in ('.png', '.jpg', '.jpeg', '.bmp')

def main():
    args = sys.argv[1:]
    arg0 = args[0]
    if isimgext(arg0):
        imgpaths = [arg0]
    else:
        imgpaths = []
        for dirpath, dirnames, filenames in os.walk(arg0):
            for imgname in [f for f in filenames if isimgext(f)]:
                imgpaths.append(os.path.join(dirpath, imgname))

    t = time.time()
    Itemp = cv.LoadImage("diebold-mark.jpg", cv.CV_LOAD_IMAGE_GRAYSCALE)
    for imgpath in imgpaths:
        decoding, isflip, bbs = decode(imgpath, Itemp)
        if decoding == None:
            print 'Error:', imgpath
        else:
            print "{0}: {1}".format(os.path.split(imgpath)[1], decoding)
            print "    isflip={0}".format(isflip)

    total_dur = time.time() - t
    print "...Done ({0} s).".format(total_dur)
    print "    Average Time Per Image: {0} s".format(total_dur / float(len(imgpaths)))

if __name__ == '__main__':
    main()
