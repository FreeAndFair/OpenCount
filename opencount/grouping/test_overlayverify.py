import wx, os, pdb
from verify_overlays import VerifyPanel
from common import GroupClass
import common

"""
WHICH_DEMO := 0 -- Demo Yes/No Verify Overlay UI
WHICH_DEMO := 1 -- Demo Normal N-class Verify Overlay UI
"""

WHICH_DEMO = 1      # <-- Change this to switch modes

class MainFrame(wx.Frame):
    def __init__(self, parent, *args, **kwargs):
        wx.Frame.__init__(self, parent, *args, **kwargs)
        # 1.) Set up data
        if WHICH_DEMO == 0:
            mode = VerifyPanel.MODE_YESNO
            groupclasses, exemplar_paths = self.get_data()
        else:
            mode = VerifyPanel.MODE_NORMAL
            groupclasses, exemplar_paths = self.get_data2()

        # 2.) Create VerifyPanel widget
        self.verifypanel = VerifyPanel(self, verifymode = mode)
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.verifypanel, proportion=1, flag=wx.EXPAND)
        self.SetSizer(sizer)

        # 3.) Start
        self.verifypanel.start(groupclasses, exemplar_paths, ondone=self.on_verify_done)
        self.Maximize()
        self.Layout()

    def on_verify_done(self, results):
        """ results is a dict mapping:
            {grouplabel: list of GroupClass instances}
        """
        print "Verifying is finished."
        for grouplabel in results:
            groups = results[grouplabel]
            num_elements = sum([len(g.elements) for g in groups])
            print "In Group {0}, there were: {1} elements".format(common.str_grouplabel(grouplabel), num_elements)

    def get_data(self):
        """ Mode Yes/No. This is asking: please separate these overlayed
        images into only the images that display a 'ones' digit.
        """
        groups1_dir = 'test_imgs/groups/ones'
        groups2_dir = 'test_imgs/groups/twos'
        all_overlays = []
        g_one = common.make_grouplabel(('digit', 'one'))
        for imgname in os.listdir(groups1_dir):
            imgpath = os.path.join(groups1_dir, imgname)
            patchpath = os.path.join('test_imgs/extracted_patches/ones', 'patch_{0}'.format(imgname))
            all_overlays.append((imgpath, (g_one,), patchpath))
        for imgname in os.listdir(groups2_dir):
            imgpath = os.path.join(groups2_dir, imgname)
            patchpath = os.path.join('test_imgs/extracted_patches/twos', 'patch_{0}'.format(imgname))
            all_overlays.append((imgpath, (g_one,), patchpath))
        groupclass = GroupClass(all_overlays)
        exemplar_paths = {g_one: 'test_imgs/extracted_patches/patch_template_one.png'}
        return (groupclass,), exemplar_paths

    def get_data2(self):
        """ Mode 'Normal'. This is asking: we have attempted to group
        these overlays into N number of groups. Please verify that
        this grouping is correct, and if there are any errors, to
        correct them.
        """
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
        exemplar_paths = {g_one: 'test_imgs2/extracted_patches/patch_template_one.png',
                          g_two: 'test_imgs2/extracted_patches/patch_template_two.png'}
        return (g1,g2), exemplar_paths
        

if __name__ == '__main__':
    app = wx.App(False)
    frame = MainFrame(None)
    frame.Show()
    app.MainLoop()
