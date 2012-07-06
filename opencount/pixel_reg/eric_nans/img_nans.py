import sys, os, pickle, pdb, wx, time
from os.path import join as pathjoin
import scipy, scipy.misc
import numpy as np
from matplotlib.pyplot import show, imshow, figure, title, colorbar, savefig, annotate

sys.path.append('../')

import shared

"""
Script to show a case where find_patch_matchesV1 returns an
Ireg with a lot of NaN's.
"""

patch = shared.standardImread('lang_en.png', flatten=True)
img = shared.standardImread('bad_english_nans.png', flatten=True)

h, w = patch.shape
x1, y1 = 0, 0
x2, y2 = w-1, h-1
bb = [y1, y2, x1, x2]
matches = shared.find_patch_matchesV1(patch, bb, ('bad_english_nans.png',), threshold=0.6)
(path, score1, score2, Ireg, y1, y2, x1, x2, rszFac) = matches[0]

print "Ireg is:"
print Ireg

imshow(Ireg);show()
