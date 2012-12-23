from scipy import misc
from imagesAlign import *

# convert to double
I=misc.imread('lena_t.jpg')/255.0;
Iref=misc.imread('lena.jpg')/255.0;
IO=imagesAlign(I,Iref)
H=IO[0]
viz=1
if viz:
    # figure(0)
    # imshow(np.abs(Iref-I),cmap='gray');

    # figure(1)
    # imshow(np.abs(Iref-IO[1]),cmap='gray');

    pt0=np.array([155,65,1])
    figure(2)
    imshow(I,cmap='gray');
    annotate('x',[pt0[0],pt0[1]])

    figure(3)
    imshow(IO[1],cmap='gray');
    annotate('x',[pt0[0],pt0[1]])

    figure(4)
    imshow(I,cmap='gray');

    pt1=pttransform(I,np.linalg.inv(H),pt0);
    annotate('x',[pt1[0],pt1[1]])

    show();
