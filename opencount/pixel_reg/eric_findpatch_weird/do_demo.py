import os, sys, scipy, scipy.misc
sys.path.append('..')

import shared

bb = [3333, 3452, 1512, 1639]
imgpath = 'npp1.png'
imgpaths = ('npp1.png', 'lib1.png')

def main():
    img = shared.standardImread(imgpath, flatten=True)
    matches = shared.find_patch_matchesV1(img, bb[:], imgpaths, threshold=0.0,
                                          bbSearch=bb[:])
    for i, (filename, sc1, sc2, Ireg, y1, y2, x1, x2, rszFac) in enumerate(matches):
        print "sc1: {0} sc2: {1} (y1,y2,x1,x2): {2}".format(sc1, sc2, (y1,y2,x1,x2))
        #scipy.misc.imsave('matchscore_{0}_{1}.png'.format(sc2, i), Ireg)

    sortedmatches = sorted(matches, key=lambda t: t[2])
    bestmatch = sortedmatches[0]
    (filename, sc1, sc2, Ireg, y1, y2, x1, x2, rszFac) = bestmatch
    print "Saving best Ireg as bestIreg.png..."
    scipy.misc.imsave('bestIreg.png', Ireg)

    ## Now, let's try to extract the best patch using only (y1,y2,x1,x2)
    ## and rszFac.

    imgA = shared.standardImread(filename, flatten=True)

    # account for resizing (?)
    (y1,y2,x1,x2) = map(lambda c: int(round(c / rszFac)), (y1,y2,x1,x2)) 

    img = imgA[y1:y2, x1:x2]

    scipy.misc.imsave('extract1.png', img)  # super off? 

    # Do I need to offset (y1,y2,x1,x2) by bbSearch...?

    img2 = imgA[bb[0]+y1:bb[0]+y2, bb[2]+x1:bb[2]+x2]

    scipy.misc.imsave('extract2.png', img2)  # also off?

if __name__ == '__main__':
    main()
