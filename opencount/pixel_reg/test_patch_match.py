import shared as sh
import os
from matplotlib.pyplot import show, imshow, figure, title, colorbar, savefig, annotate

# load patch
ballotDir = '../../test-ballots/1side_Ntemplates/cedar2008primary_full/blank'
I=sh.standardImread(os.path.join(ballotDir,'20120608170512502_0001.jpg'),flatten=True)

patch = I[320:400,500:950]
#imshow(patch); show()

# generate image list
imList = []
for root, dirs, files in os.walk(ballotDir):
    for f in files:
        (f0,ext)=os.path.splitext(f)
        if ext != '.jpg':
            continue
        p1=os.path.join(root,f)
        imList.append(p1)
        

out = sh.find_patch_matches(patch,imList)
print out
