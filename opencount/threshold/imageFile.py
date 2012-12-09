import os
from PIL import Image
from wx.lib.pubsub import Publisher
import wx
import numpy as np
import util

def is_image_ext(filename):
    IMG_EXTS = ('.bmp', '.png', '.jpg', '.jpeg', '.tif', '.tiff')
    return os.path.splitext(filename)[1].lower() in IMG_EXTS

def makeOneFile(prefix, src, dst):
    out = open(dst, "wb")
    tout = open(dst+".type", "wb")
    for each,score in src:
        wx.CallAfter(Publisher().sendMessage, "signals.MyGauge.tick")
        #if i%100 == 0:
        #    print i
        img = Image.open(os.path.join(prefix, each))
        dat = img.getdata()
        if type(dat[0]) != type(tuple()):
            s = "".join((chr(x) for x in dat))
            tout.write("A")
        else:
            s = "".join((chr(x[0])+chr(x[1])+chr(x[2]) for x in dat))
            tout.write("B")

        out.write(s)

    if src:
        open(dst+".size", "w").write(str(img.size))
                
    out.close()
    tout.close()

class ImageFile:
    def __init__(self, inp):
        self.infile = open(inp, "rb")
        self.dims = map(int,open(inp+".size").read().strip()[1:-1].split(","))
        self.imtype = open(inp+".type").read()
        self.size = self.dims[0]*self.dims[1]
        self.offsets = []
        off = 0
        for each in self.imtype:
            self.offsets.append(off)
            off += 3 if each == 'B' else 1

    def readRawBytes(self, imagenum, count):
        self.infile.seek(self.size*imagenum)
        return np.fromstring(self.infile.read(self.size*count), dtype='uint8')
        #return [ord(x) for x in self.infile.read(self.size*count)]

    @util.pdb_on_crash
    def readManyImages(self, imagenum, numcols, width, height, curwidth, curheight):
        imagetypes = self.imtype[imagenum:imagenum+numcols]
        # Three bytes for colored images, one byte otherwise
        types = [3 if x == "B" else 1 for x in imagetypes]
        if not all(x == types[0] for x in types):
            return readManyImages2(imagenum, numcols, width, height, curwidth, curheight)
        toread = sum(types)

        data = self.readRawBytes(self.offsets[imagenum], toread)

        
        if imagetypes[0] == 'A': # single chanel
            fixed = np.concatenate([data[j:j+self.size].reshape((height,width)) for j in range(0,data.shape[0],self.size)], axis=1)
            jpg = Image.fromarray(fixed)
            tomerge = jpg,jpg,jpg
        else:
            tomerge = []
            for start in range(3):
                fixed = np.concatenate([data[j+start:j+self.size*3:3].reshape((height,width)) for j in range(0,data.shape[0],self.size*3)], axis=1)
                jpg = Image.fromarray(fixed)
                tomerge.append(jpg)
            tomerge = tuple(tomerge)

        realnumcols = (fixed.shape[0]*fixed.shape[1])/(width*height)
        jpg = Image.merge('RGB', tomerge)
        #print jpg
        jpg = jpg.resize((curwidth*realnumcols, curheight))
        #print jpg
        return jpg

    def readManyImages2(self, imagenum, numcols, width, height, curwidth, curheight):
        imagetypes = self.imtype[imagenum:imagenum+numcols]
        # Three bytes for colored images, one byte otherwise
        toread = sum([3 if x == "B" else 1 for x in imagetypes])
        data = self.readRawBytes(self.offsets[imagenum], toread)

        expanded = []
        j = 0
        i = 0
        while i < len(data):
            if imagetypes[j/self.size] == 'A':
                expanded.append((data[i],data[i],data[i]))
                i += 1
            else:
                expanded.append((data[i],data[i+1],data[i+2]))
                i += 3
            j += 1
        
        size = width*height
        fixed = []
        for row in range(height):
            for q in range(numcols):
                fixed += expanded[q*size+row*width:q*size+row*width+width]

        #print fixed
        realnumcols = len(fixed)/(width*height)
        jpg = Image.new("RGBA", (width*realnumcols,height))
        print jpg
        jpg.putdata(fixed)
        jpg = jpg.resize((curwidth*realnumcols, curheight))
        return jpg

    def readManyImagesO(self, imagenum, numcols, width, height, curwidth, curheight):
        data = self.readRawBytes(imagenum, numcols)
        size = width*height
        fixed = []
        for row in range(height):
            for q in range(numcols):
                fixed += data[q*size+row*width:q*size+row*width+width]
        fixed = [(x,x,x) for x in fixed]
        #print fixed
        realnumcols = len(fixed)/(width*height)
        jpg = Image.new("RGBA", (width*realnumcols,height))
        jpg.putdata(fixed)
        jpg = jpg.resize((curwidth*realnumcols, curheight))
        return jpg
        

    def readImage(self, num):
        self.infile.seek(self.size*num)
        data = self.infile.read(self.size)
        return Image.frombuffer("L", (self.dims[0],self.dims[1]), data)

    # def readImageCount(self, start, count):
    #     self.infile.seek(self.size*start)
    #     fulldata = self.infile.read((self.size+4)*count)
    #     imgs = []
    #     for i in range(count):
    #         offset = i*(self.size+4)
    #         size = [ord(x) for x in fulldata[offset:offset+4]]
    #         x,y = size[0]+(size[1]<<8), size[2]+(size[3]<<8)
    #         imgs.append(Image.frombuffer("L", (x,y), 
    #                                      fulldata[offset:offset+x*y]))
    #     return imgs
    
if __name__ == "__main__":
    x = input("ARE YOU SURE?!")
    makeOneFile("fake/", "bigfile")
