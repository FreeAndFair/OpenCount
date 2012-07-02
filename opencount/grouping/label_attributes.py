import os, sys
from labelcontest.labelcontest import LabelContest
import pickle
import csv

DUMMY_ROW_ID = -42
class LabelAttributesPanel(LabelContest):
    """
    This class is one big giant hack of a monkey-patch.
    """
    def gatherData(self):
        self.groupedtargets = []
        self.dirList = []
    
        attrdata = pickle.load(open(self.proj.ballot_attributesfile))
        self.sides = [x['side'] for x in attrdata]
        self.types = [x['attrs'].keys()[0] for x in attrdata]

        print "LOAD", attrdata
        print "IMSIZE", self.proj.imgsize

        width, height = self.proj.imgsize
        self.dirList = [os.path.join(self.proj.blankballots_straightdir,x) for x in os.listdir(self.proj.blankballots_straightdir)]
        for f in self.dirList:
            thisballot = [[(at['id'], 0,
                          int(at['x1']*width), int(at['y1']*height), 
                          int(at['x2']*width), int(at['y2']*height))] for at in attrdata]
    
            self.groupedtargets.append(thisballot)
        self.groupedtargets_back = self.groupedtargets
    def unsubscribe_pubsubs(self):
        pass

    def setupBoxes(self):
        self.boxes = []
        for each in self.groupedtargets:
            bb = []
            for contest in each:
                id,_,l,u,r,d = contest[0]
                bb.append((id,l,u,r,d))
            self.boxes.append(bb)
        self.groupedtargets = [[[]]*len(x) for x in self.groupedtargets]

        # EVEN BIGGER HACK!!!
        self.text = {}
        for i,each in enumerate(self.boxes):
            for x in each:
                self.text[i,x[0]] = []
                self.voteupto[i,x[0]] = 1
        if os.path.exists(self.proj.attr_internal):
            self.text = pickle.load(open(self.proj.attr_internal))
        # </hack>

    def addText(self):
        LabelContest.addText(self)
        name = self.types[self.count]
        self.contesttitle.SetLabel("Attribute Value (%s)"%name)

    def save(self):
        self.saveText(removeit=False)
        print "TEXT", self.text
        if not os.path.exists(self.proj.patch_loc_dir):
            os.mkdir(self.proj.patch_loc_dir)
        pickle.dump(self.text, open(self.proj.attr_internal, "w"))
        for ballot in range(len(self.dirList)):
            vals = sorted([(k,v) for k,v in self.text.items() if k[0] == ballot])
            name = os.path.splitext(os.path.split(self.dirList[ballot])[-1])[0]+"_patchlocs.csv"
            name = os.path.join(self.proj.patch_loc_dir, name)
            print "MAKING", name
            out = csv.writer(open(name, "w"))
            out.writerow(["imgpath","id","x","y","width",
                          "height","attr_type","attr_val","side"])
            out.writerow([os.path.abspath(self.dirList[ballot]), DUMMY_ROW_ID,0,0,0,0,"_dummy_","_dummy_","_dummy_"])
            for uid,each in enumerate(vals):
                pos = self.groupedtargets_back[ballot][uid][0]
                print "POS IS", pos, "EACH", each
                value = "_none_" if each[1] == [] else each[1][0]
                out.writerow([os.path.abspath(self.dirList[ballot]),
                              uid, pos[2], pos[3],
                              pos[4]-pos[2], pos[5]-pos[3],
                              self.types[uid], value, self.sides[uid]])
    def validate_outputs(self):
        return True
    def stop(sefl):
        pass
    def export_bounding_boxes(sefl):
        pass
    def checkCanMoveOn(self):
        return True
