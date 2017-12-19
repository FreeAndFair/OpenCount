Warning: This is not complete; it is only intended for our internal use.

Basically, you will install OpenCV, then install other prerequisites.  See below for specific instructions for several different distributions.  We use Ubuntu in our testing, so we recommend using OpenCount on Ubuntu.  OpenCount has not been tested on other systems (e.g., Mac, Windows).

# Instructions for Ubuntu #

First, install OpenCV, including Python wrappers.  We require OpenCV version 2.4 or later.
On Ubuntu, you will need to build from source.  (Make sure you get a 2.4 version, not a 2.3 version.)

First, install prerequisites for OpenCV and other packages needed for OpenCount:

```
apt-get install python-wxgtk2.8 python-numpy python-scipy libpython2.7 python-dev python2.7-dev cmake python-matplotlib cmake g++ python-scipy python-wxgtk2.8
apt-get install libtbb-dev libtbb-doc tbb-examples libtbb2-dbg libtbb2
apt-get install python-matplotlib cython tesseract-ocr tesseract-ocr-eng tesseract-ocr-chi-sim tesseract-ocr-chi-tra tesseract-ocr-spa tesseract-ocr-kor tesseract-ocr-vie
```

Next, [install OpenCV from source](http://opencv.willowgarage.com/wiki/InstallGuide).
(Here's [a tutorial on installing OpenCV from source](http://www.samontab.com/web/2011/06/installing-opencv-2-2-in-ubuntu-11-04/) that you may find helpful.)

The short version is, after downloading and extracting the OpenCV 2.4 source code, do this:

```
cd OpenCV-2.4.3
mkdir release
cd release
cmake -D WITH_TBB=ON -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D BUILD_PYTHON_SUPPORT=ON -D BUILD_EXAMPLES=ON ..
make
sudo make install
```

(The `make install` saves you from having to set the `PYTHONPATH` environment variable to point to your local OpenCV installation.)

Next, install python-Levenshtein from source, as follows:

```
apt-get install python-setuptools
wget http://pypi.python.org/packages/source/p/python-Levenshtein/python-Levenshtein-0.10.2.tar.gz
tar -xzf python-Levenshtein-0.10.2.tar.gz
cd python-Levenshtein-0.10.2/
python setup.py build
sudo python setup.py install
```

Next, compile our special high-performance pixel-alignment code.  After downloading the OpenCount source from the git repository (git clone ...), do the following:

```
cd opencount/opencount/pixel_reg
./build_all.sh
```

You'll also need to get and install our special straightener.
Instructions are
[available separately](http://code.google.com/p/straightener/wiki/InstallationAndUsage);
follow them.

An optional Python module to install is psutil, which is used
by OpenCount to avoid consuming too much memory during
memory-intensive computation:
> http://code.google.com/p/psutil/

# Instructions for Fedora #

On Fedora, you can do

```
yum -y install tbb tbb-devel cmake cmake-gui numpy numpy-devel scipy scipy-devel gtk+ gtk+-devel glib libjpeg libjpeg-devel libtiff libtiff-devel jasper jasper-devel libpng libpng-devel zlib zlib-devel openexpr openexpr-devel blas python-sphinx
yum -y install numpy python-matplotlib scipy python-imaging wxPython python-sqlite2 python-Levenshtein cython
```

then install OpenCV from source following Ubuntu instructions above.

(I think during the cmake process, after setting `WITH_TBB` I had to subsequently fix `TBB_LIB_DIR` to be `/usr/lib64` or whatever is appropriate.  This may not apply to you.)

Finally compile the high-performance pixel-alignment code and install the special straightener.

Optionally, install the `psutil` Python module.

# Other Linux distributions #

Make sure you install `numpy`, `python-matplotlib`, `scipy`, `python-imaging` (PIL), `wxPython`, `python-sqlite2` (Python bindings for sqlite3), `python-Levenshtein`.

Optionally, install `psutil`.

# Mac OS X #

Sorry, we haven't tested OpenCount on Mac OS X yet, so no instructions.

# Windows #

Sorry, we haven't tested OpenCount on Windows yet, so no instructions.

Here are some instructions for install OpenCV on Windows.
To get the Python bindings, copy/paste the contents of `opencv/build/python/2.7` (or 2.6)
to the site-packages directory of your Python installation, i.e. `C:/Python27/Lib/site-packages/`
For me, this means that you'll be adding two new files to that directory:
```
    C:/Python27/Lib/site-packages/cv.py
    C:/Python27/Lib/site-packages/cv2.pyd
```

# Trying it out #

If you want to try out OpenCount quickly, we've provided a small sample election for you in `test-ballots/tutorial_dataset`.  You can try processing the scanned ballots in that directory.  For an illustrated, step-by-step tutorial to the OpenCount system, check out the [Tutorial](Tutorial.md) in the wiki.