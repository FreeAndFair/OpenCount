import wx, os, pdb
from verify_overlays import VerifyPanel
from common import GroupClass
import common

"""
WHICH_DEMO := 0 -- Demo Normal N-class Verify Overlay UI
WHICH_DEMO := 1 -- Demo Yes/No Verify Overlay UI
"""

WHICH_DEMO = 1

class MainFrame(wx.Frame):
    def __init__(self, parent, *args, **kwargs):
        wx.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        if WHICH_DEMO == 0:
            mode = VerifyPanel.MODE_YESNO
            groupclasses, patches, exemplar_paths = self.get_data()
        else:
            mode = VerifyPanel.MODE_NORMAL
            groupclasses, patches, exemplar_paths = self.get_data2()
        self.verifypanel = VerifyPanel(self, verifymode = mode)
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.verifypanel, proportion=1, flag=wx.EXPAND)
        self.SetSizer(sizer)
        self.verifypanel.start(groupclasses, patches, exemplar_paths, 'groupout.p', ondone=self.on_verify_done)
        self.Layout()

    def on_verify_done(self, results):
        print "Verifying is finished."
        for grouplabel in results:
            print "In Group {0}, there were: {1} elements".format(grouplabel, len(results[grouplabel]))

    def get_data(self):
        groups1_dir = 'test_imgs/groups/ones'
        groups2_dir = 'test_imgs/groups/twos'
        groups1, groups2 = [], []
        g_one = common.make_grouplabel(('digit', 'one'))
        for imgname in os.listdir(groups1_dir):
            imgpath = os.path.join(groups1_dir, imgname)
            patchpath = os.path.join('test_imgs/extracted_patches/ones', 'patch_{0}'.format(imgname))
            groups1.append((imgpath, (g_one,), patchpath))
        for imgname in os.listdir(groups2_dir):
            imgpath = os.path.join(groups2_dir, imgname)
            patchpath = os.path.join('test_imgs/extracted_patches/twos', 'patch_{0}'.format(imgname))
            groups2.append((imgpath, (g_one,), patchpath))
        allexamples = groups1 + groups2
        groupclass = GroupClass(allexamples)
        patches = {'test_imgs/template/template.png': ((300,30, 400,130), g_one)}
        exemplar_paths = {g_one: 'test_imgs/extracted_patches/patch_template_one.png'}
        return (groupclass,), patches, exemplar_paths

    def get_data2(self):
        groups1_dir = 'test_imgs2/groups/ones'
        groups2_dir = 'test_imgs2/groups/twos'
        groups1, groups2 = [], []
        g_one = common.make_grouplabel(('digit', 'one'))
        g_two = common.make_grouplabel(('digit', 'two'))
        for imgname in os.listdir(groups1_dir):
            imgpath = os.path.join(groups1_dir, imgname)
            patchpath = os.path.join('test_imgs2/extracted_patches/ones', 'patch_{0}'.format(imgname))
            groups1.append((imgpath, (g_one, g_two), patchpath))
        for imgname in os.listdir(groups2_dir):
            imgpath = os.path.join(groups2_dir, imgname)
            patchpath = os.path.join('test_imgs2/extracted_patches/twos', 'patch_{0}'.format(imgname))
            groups2.append((imgpath, (g_two, g_one), patchpath))
        g1 = GroupClass(groups1)
        g2 = GroupClass(groups2)
        patches = {'test_imgs2/template/template_one.png': ((300,30, 400,130), g_one),
                   'test_imgs2/template/template_two.png': ((300,30, 400,130), g_two)}
        exemplar_paths = {g_one: 'test_imgs2/extracted_patches/patch_template_one.png',
                          g_two: 'test_imgs2/extracted_patches/patch_template_two.png'}
        return (g1,g2), patches, exemplar_paths
        

if __name__ == '__main__':
    app = wx.App(False)
    frame = MainFrame(None)
    frame.Show()
    app.MainLoop()
