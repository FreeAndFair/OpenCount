import sys, os, pickle, pdb, traceback, time
import zbar, Image

def decode(img):
    """ Decodes the barcode present in IMG, returns it as a string.
    Input:
        IMG: Either a string (imgpath), or an image object.
    Output:
        A tuple of strings, where each string is the decoding of some
        barcode in IMG.
    """
    if type(img) == str:
        I = Image.open(img).convert("L")
    else:
        I = img

    scanner = zbar.ImageScanner()
    scanner.parse_config('enable')
    
    if I.mode != "L":
        I = I.convert("L")
    w, h = I.size
    raw_img = I.tostring()
    zImg = zbar.Image(w, h, 'Y800', raw_img)
    scanner.scan(zImg)
    
    symbols = []
    for symbol in zImg:
        print 'decoded', symbol.type, 'symbol: {0}'.format(symbol.data)
        symbols.append(symbol.data)

    if not symbols:
        print "Uhoh, couldn't find anything."
        pdb.set_trace()
    return symbols

def main():
    args = sys.argv[1:]
    imgpath = args[0]
    t = time.time()
    decoded = decode(imgpath)
    dur = time.time() - t
    print "...Time elapsed: {0} s".format(dur)
    
if __name__ == '__main__':
    main()
