import wx
import sys
from PIL import Image
import os

sys.path.append("..")
from util import pil2wxb


class VerifyContestGrouping(wx.Panel):
    def __init__(self, parent, ocrdir, dirList, equivs, reorder, reorder_inverse, mapping, mapping_inverse, multiboxcontests, callback):
        print "ARGS", (ocrdir, dirList, equivs, reorder, reorder_inverse, mapping, mapping_inverse, multiboxcontests, callback)
        wx.Panel.__init__(self, parent, wx.ID_ANY)
        self.frame = parent
        self.callback = callback

        self.ocrdir = ocrdir
        self.dirList = dirList
        self.equivs = equivs
        self.reorder = reorder
        self.reorder_inverse = reorder_inverse
        self.mapping = mapping
        self.mapping_inverse = mapping_inverse
        self.multiboxcontests = multiboxcontests
        
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.imagearea = wx.Panel(self, size=(1000, 700))
        datasizer =  wx.BoxSizer(wx.HORIZONTAL)
        self.sofar = wx.StaticText(self, -1, label="")
        datasizer.Add(self.sofar)
        isokaysizer = wx.BoxSizer(wx.HORIZONTAL)
        back = wx.Button(self, -1, label="Back")
        allow = wx.Button(self, -1, label="Yes")
        deny = wx.Button(self, -1, label="No")
        denyall = wx.Button(self, -1, label="Deny All")
        allowall = wx.Button(self, -1, label="Accpet All (dangerous)")
        back.Bind(wx.EVT_BUTTON, self.back)
        allow.Bind(wx.EVT_BUTTON, self.allow)
        deny.Bind(wx.EVT_BUTTON, self.deny)
        denyall.Bind(wx.EVT_BUTTON, self.deny_all)
        allowall.Bind(wx.EVT_BUTTON, self.allow_all)
        isokaysizer.Add(back)
        isokaysizer.Add(allow)
        isokaysizer.Add(deny)
        isokaysizer.Add((100, 0))
        isokaysizer.Add(denyall)
        isokaysizer.Add(allowall)
        self.sizer.Add(isokaysizer)
        self.sizer.Add(datasizer)
        self.sizer.Add((0,20))
        self.sizer.Add(self.imagearea)

        self.SetSizer(self.sizer)
        self.Layout()
        self.Fit()

        self.group_index = 0
        self.is_valid = {}
        self.processgroups = [i for i,x in enumerate(self.equivs) if len(x) > 1]
        self.compareimage = None
        self.testimage = None

        self.load_next_group(0)
        self.show()


    def load_next_group(self, inc=1):
        self.index = 0
        self.orderedpaths = []

        self.group_index += inc

        if self.group_index < 0:
            return
        if self.group_index >= len(self.processgroups):
            print "DONE"
            self.group_index = 0
            self.callback([(self.processgroups[k],v) for k,v in self.is_valid.items()])
            self.frame.Close(True)

        for ballot,contest in self.equivs[self.processgroups[self.group_index]]:
            #print ballot, contest
            ballotname = os.path.split(self.dirList[ballot])[1].split('.')[0]
            boundingbox = (ballot, contest)
            print 'mbc', self.multiboxcontests, boundingbox
            if any(boundingbox in x for x in self.multiboxcontests):
                boundingboxes = [x for x in self.multiboxcontests if boundingbox in x][0]
                print boundingboxes
                boundingbox = [x for x in boundingboxes if x in self.mapping_inverse][0]
                print boundingbox
                boundingboxes = [k[1] for k,v in self.mapping.items() if v == boundingbox]
            else:
                boundingboxes = [self.mapping_inverse[boundingbox][1]]
                
            boundingboxes = sorted(boundingboxes)
            print 'bb', boundingboxes

            ballotdir = os.path.join(self.ocrdir,ballotname+"-dir")
            boundingboxdirs = [os.path.join(ballotdir, '-'.join(map(str,bb))) for bb in boundingboxes]
            print 'bbdir', boundingboxdirs
            order = dict(self.reorder[self.reorder_inverse[ballot,contest]][ballot,contest])
            images = [[img for img in os.listdir(bbdir) if img[-3:] != 'txt'] for bbdir in boundingboxdirs]
            # TODO: Figure out if I need to update this code.
            print "Do I need a sort_nicely call?"
            pdb.set_trace()
            print 'im1', images
            images = [sorted(imgs, key=lambda x: int(x.split('.')[0])) for imgs in images]
            print 'im2', images
            title = images[0][0]
            images = [(i,y) for i,x in enumerate(images) for y in x[1:]]
            orderedimages = [None]*(len(images)+1)
            orderedimages[0] = (0, title)
            print 'im3', images
            print 'ord', order
            print 'szs', len(order), len(images)
            for i in range(len(images)):
                orderedimages[i+1] = images[order[i]]
            print orderedimages
            paths = [os.path.join(boundingboxdirs[i],img) for i,img in orderedimages]
            self.orderedpaths.append(paths)

        if self.group_index not in self.is_valid:
            self.is_valid[self.group_index] = [None]*len(self.orderedpaths)

    def back(self, x=None):
        self.index -= 1
        if self.index < 0:
            self.load_next_group(-1)
            self.show()
            self.index = len(self.orderedpaths)-1
        self.show()
        
    def allow_all(self, x=None):
        self.is_valid[self.group_index] = [True]*len(self.orderedpaths)
        self.load_next_group()
        self.show()

    def deny_all(self, x=None):
        self.is_valid[self.group_index] = [False]*len(self.orderedpaths)
        self.load_next_group()
        self.show()

    def allow(self, x=None):
        self.is_valid[self.group_index][self.index] = True
        self.inc_and_show()

    def deny(self, x=None):
        self.is_valid[self.group_index][self.index] = False
        self.inc_and_show()
    
    def inc_and_show(self):
        self.index += 1
        if self.index >= len(self.orderedpaths):
            self.load_next_group()
        self.show()

    def show(self):
        print "SHOWING", self.index, self.group_index
        self.sofar.SetLabel("On item %d of %d in group %d of %d."%(self.index+1,len(self.orderedpaths),self.group_index+1,len(self.processgroups)))
        curpaths = self.orderedpaths[self.index]
        imgs = map(Image.open, curpaths)
        height = sum(x.size[1] for x in imgs)
        width = max(x.size[0] for x in imgs)
        showimg = Image.new("RGB", (width, height))
        pos = 0
        for img in imgs:
            showimg.paste(img, (0, pos))
            pos += img.size[1]
        print showimg.size
        x,y = showimg.size
        if y > 700:
            factor = float(y)/700
            print factor
            showimg = showimg.resize((int(x/factor), int(y/factor)))

        if self.testimage != None:
            self.testimage.Destroy()

        if self.index == 0:
            if self.compareimage != None:
                self.compareimage.Destroy()
            self.compareimage = wx.StaticBitmap(self.imagearea, -1, 
                                                pil2wxb(showimg), pos=(0, 0))

        self.testimage = wx.StaticBitmap(self.imagearea, -1, pil2wxb(showimg),
                                         pos=(512, 0))


if __name__ == "__main__":
    app = wx.App()
    frame = wx.Frame (None, -1, 'Test', size=(1024, 768))
    VerifyContestGrouping(frame, '/home/nicholas/opencount/opencount/projects/orange/ocr_tmp_dir', ['/home/nicholas/opencount/opencount/projects/orange/blankballots_straight/339_3116_1_36_1.png', '/home/nicholas/opencount/opencount/projects/orange/blankballots_straight/339_3114_1_32_1.png', '/home/nicholas/opencount/opencount/projects/orange/blankballots_straight/339_3117_1_38_1.png', '/home/nicholas/opencount/opencount/projects/orange/blankballots_straight/339_3115_1_34_1.png', '/home/nicholas/opencount/opencount/projects/orange/blankballots_straight/339_3113_1_30_1.png', '/home/nicholas/opencount/opencount/projects/orange/blankballots_straight/339_3136_1_31_1.png'], [[(0, 4), (2, 4)], [(0, 2), (2, 2), (3, 2), (4, 2), (5, 3)], [(1, 3), (3, 3), (4, 3)], [(1, 4), (3, 4), (4, 4)], [(1, 2)], [(5, 4)], [(5, 5)], [(0, 3), (2, 3)], [(1, 5), (3, 5), (4, 5)], [(5, 0)], [(5, 6)], [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 2)]], {(1, 2): {(1, 2): [(0, 0), (1, 1), (2, 2)]}, (0, 1): {(0, 1): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23), (24, 24)], (4, 1): [(0, 18), (1, 19), (2, 20), (3, 21), (4, 22), (5, 23), (6, 24), (7, 0), (8, 1), (9, 2), (10, 3), (11, 4), (12, 5), (13, 6), (14, 7), (15, 8), (16, 9), (17, 10), (18, 11), (19, 12), (20, 13), (21, 14), (22, 15), (23, 16), (24, 17)], (3, 1): [(0, 18), (1, 19), (2, 20), (3, 21), (4, 22), (5, 23), (6, 24), (7, 0), (8, 1), (9, 2), (10, 3), (11, 4), (12, 5), (13, 6), (14, 7), (15, 8), (16, 9), (17, 10), (18, 11), (19, 12), (20, 13), (21, 14), (22, 15), (23, 16), (24, 17)], (2, 1): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15), (16, 16), (17, 17), (18, 18), (19, 19), (20, 20), (21, 21), (22, 22), (23, 23), (24, 24)], (5, 2): [(0, 4), (1, 5), (2, 6), (3, 7), (4, 8), (5, 9), (6, 10), (7, 11), (8, 12), (9, 13), (10, 14), (11, 15), (12, 16), (13, 17), (14, 18), (15, 19), (16, 20), (17, 21), (18, 22), (19, 23), (20, 24), (21, 0), (22, 1), (23, 2), (24, 3)], (1, 1): [(0, 18), (1, 19), (2, 20), (3, 21), (4, 22), (5, 23), (6, 24), (7, 0), (8, 1), (9, 2), (10, 3), (11, 4), (12, 5), (13, 6), (14, 7), (15, 8), (16, 9), (17, 10), (18, 11), (19, 12), (20, 13), (21, 14), (22, 15), (23, 16), (24, 17)]}, (5, 4): {(5, 4): [(0, 0), (1, 1), (2, 2)]}, (1, 3): {(1, 3): [(0, 0), (1, 1), (2, 2)], (3, 3): [(0, 0), (1, 1), (2, 2)], (4, 3): [(0, 0), (1, 1), (2, 2)]}, (5, 5): {(5, 5): [(0, 0), (1, 1), (2, 2)]}, (5, 6): {(5, 6): [(0, 0), (1, 1), (2, 2), (3, 3)]}, (1, 4): {(4, 4): [(0, 0), (1, 1), (2, 2)], (3, 4): [(0, 0), (1, 1), (2, 2)], (1, 4): [(0, 0), (1, 1), (2, 2)]}, (1, 5): {(4, 5): [(0, 0), (1, 1), (2, 2), (3, 3)], (1, 5): [(0, 0), (1, 1), (2, 2), (3, 3)], (3, 5): [(0, 0), (1, 1), (2, 2), (3, 3)]}, (5, 0): {(5, 0): [(0, 0), (1, 1), (2, 2), (3, 3)]}, (0, 4): {(2, 4): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)], (0, 4): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]}, (0, 3): {(0, 3): [(0, 0), (1, 1), (2, 2), (3, 3)], (2, 3): [(0, 0), (1, 1), (2, 2), (3, 3)]}, (0, 2): {(4, 2): [(0, 0), (1, 1), (2, 2)], (3, 2): [(0, 0), (1, 1), (2, 2)], (5, 3): [(0, 0), (1, 1), (2, 2)], (0, 2): [(0, 0), (1, 1), (2, 2)], (2, 2): [(0, 0), (1, 1), (2, 2)]}}, {(1, 3): (1, 3), (5, 6): (5, 6), (5, 4): (5, 4), (2, 1): (0, 1), (0, 3): (0, 3), (1, 2): (1, 2), (3, 3): (1, 3), (4, 4): (1, 4), (1, 5): (1, 5), (5, 0): (5, 0), (0, 4): (0, 4), (3, 5): (1, 5), (4, 1): (0, 1), (1, 1): (0, 1), (3, 2): (0, 2), (4, 5): (1, 5), (2, 2): (0, 2), (1, 4): (1, 4), (2, 3): (0, 3), (4, 2): (0, 2), (5, 5): (5, 5), (0, 1): (0, 1), (5, 3): (0, 2), (5, 2): (0, 1), (3, 1): (0, 1), (0, 2): (0, 2), (4, 3): (1, 3), (3, 4): (1, 4), (2, 4): (0, 4)}, {(1, (1103, 1485, 1592, 1889)): (1, 4), (1, (1103, 587, 1592, 1094)): (1, 5), (2, (1110, 587, 1601, 1095)): (2, 3), (2, (1110, 281, 1601, 593)): (2, 1), (3, (619, 278, 1113, 2647)): (3, 1), (3, (1107, 1486, 1598, 1890)): (3, 4), (1, (1103, 1088, 1592, 1491)): (1, 3), (2, (1110, 1089, 1601, 1832)): (2, 4), (4, (619, 285, 1113, 2647)): (4, 1), (3, (1107, 1089, 1598, 1492)): (3, 3), (1, (614, 280, 1109, 2645)): (1, 1), (5, (1114, 1488, 1604, 1890)): (5, 5), (5, (1114, 2188, 1604, 2532)): (5, 3), (5, (624, 284, 1120, 2649)): (5, 2), (4, (1107, 1487, 1598, 1891)): (4, 4), (4, (1107, 591, 1598, 1096)): (4, 5), (5, (139, 1152, 630, 1451)): (5, 0), (1, (1103, 280, 1592, 593)): (1, 1), (3, (1107, 278, 1598, 595)): (3, 1), (0, (1109, 280, 1598, 590)): (0, 1), (5, (1114, 590, 1604, 1098)): (5, 6), (2, (1110, 2131, 1601, 2474)): (2, 2), (4, (1107, 1090, 1598, 1493)): (4, 3), (4, (1107, 285, 1598, 597)): (4, 1), (0, (1109, 584, 1598, 1091)): (0, 3), (3, (1107, 2187, 1598, 2530)): (3, 2), (0, (1109, 2127, 1598, 2471)): (0, 2), (2, (621, 281, 1116, 2645)): (2, 1), (1, (1103, 2185, 1592, 2529)): (1, 2), (0, (620, 280, 1115, 2642)): (0, 1), (5, (1114, 284, 1604, 596)): (5, 2), (3, (1107, 589, 1598, 1095)): (3, 5), (5, (1114, 1092, 1604, 1494)): (5, 4), (0, (1109, 1085, 1598, 1829)): (0, 4), (4, (1107, 2188, 1598, 2532)): (4, 2)}, {(1, 3): (1, (1103, 1088, 1592, 1491)), (5, 6): (5, (1114, 590, 1604, 1098)), (5, 4): (5, (1114, 1092, 1604, 1494)), (2, 1): (2, (621, 281, 1116, 2645)), (0, 3): (0, (1109, 584, 1598, 1091)), (1, 2): (1, (1103, 2185, 1592, 2529)), (3, 3): (3, (1107, 1089, 1598, 1492)), (4, 4): (4, (1107, 1487, 1598, 1891)), (1, 5): (1, (1103, 587, 1592, 1094)), (5, 0): (5, (139, 1152, 630, 1451)), (2, 2): (2, (1110, 2131, 1601, 2474)), (3, 5): (3, (1107, 589, 1598, 1095)), (4, 1): (4, (1107, 285, 1598, 597)), (1, 1): (1, (1103, 280, 1592, 593)), (3, 2): (3, (1107, 2187, 1598, 2530)), (4, 5): (4, (1107, 591, 1598, 1096)), (0, 4): (0, (1109, 1085, 1598, 1829)), (5, 5): (5, (1114, 1488, 1604, 1890)), (1, 4): (1, (1103, 1485, 1592, 1889)), (2, 3): (2, (1110, 587, 1601, 1095)), (4, 2): (4, (1107, 2188, 1598, 2532)), (5, 3): (5, (1114, 2188, 1604, 2532)), (0, 1): (0, (620, 280, 1115, 2642)), (3, 4): (3, (1107, 1486, 1598, 1890)), (3, 1): (3, (1107, 278, 1598, 595)), (0, 2): (0, (1109, 2127, 1598, 2471)), (4, 3): (4, (1107, 1090, 1598, 1493)), (5, 2): (5, (1114, 284, 1604, 596)), (2, 4): (2, (1110, 1089, 1601, 1832))}, [[(0, 0), (0, 1)], [(1, 0), (1, 1)], [(2, 0), (2, 1)], [(3, 0), (3, 1)], [(4, 0), (4, 1)], [(5, 1), (5, 2)]], lambda x: x)
    frame.Show()
    app.MainLoop()
