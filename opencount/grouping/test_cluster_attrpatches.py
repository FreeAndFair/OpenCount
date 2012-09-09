import sys, os, time, pdb, random

import scipy.misc
from matplotlib.pyplot import show, imshow, figure
import cluster_imgs, make_overlays

def is_img_ext(p):
    return os.path.splitext(p.lower())[1] in ('.png', '.jpg', '.jpeg')

def main():
    args = sys.argv[1:]
    imgsdir = args[0]
    outdir = args[1]
    imgpaths = []
    for dirpath, dirnames, filenames in os.walk(imgsdir):
        for imgname in [f for f in filenames if is_img_ext(f)]:
            imgpaths.append(os.path.join(dirpath, imgname))
    random.shuffle(imgpaths)
    #clusters = cluster_imgs.cluster_imgs_kmeans_alignerr(imgpaths)
    #clusters = cluster_imgs.cluster_imgs_kmeans_mine(imgpaths)
    clusters = cluster_imgs.cluster_imgs_kmeans(imgpaths)
    #clusters = cluster_imgs.cluster_imgs_pca_kmeans(imgpaths)

    for cluster, imgpaths in clusters.iteritems():
        #overlay, minimg, maximg = make_overlays.overlay_im(imgpaths, include_min_max=True)
        minimg, maximg = make_overlays.make_minmax_overlay(imgpaths)
        
        outrootdir = os.path.join(outdir, str(cluster))
        try:
            os.makedirs(outrootdir)
        except:
            pass
        for imgpath in imgpaths:
            img = scipy.misc.imread(imgpath, flatten=True)
            scipy.misc.imsave(os.path.join(outrootdir, os.path.split(imgpath)[1]), img)

        scipy.misc.imsave(os.path.join(outrootdir, 'min.png'), minimg)
        scipy.misc.imsave(os.path.join(outrootdir, 'max.png'), maximg)

if __name__ == '__main__':
    main()

