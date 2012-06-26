import sys
sys.path.append('../')

import wx, pickle
from specify_voting_targets.imageviewer import WorldState as WorldState
from specify_voting_targets.imageviewer import BoundingBox as BoundingBox

class AttributeBox(BoundingBox):
    """
    Represents a bounding box around a ballot attribute.
    """
    def __init__(self, x1, y1, x2, y2, label='', color=None, 
                 id=None, is_contest=False, contest_id=None, 
                 target_id=None,
                 line_width=None, children=None,
                 is_new=False, is_selected=False,
                 attrs=None, side='front'):
        """
        attrs is a dict mapping:
            {str attrtype: str attrval}
        """
        BoundingBox.__init__(self, x1, y1, x2, y2, label=label, color=color,
                             id=id, is_contest=is_contest, contest_id=contest_id,
                             target_id=target_id,
                             line_width=line_width, children=children,
                             is_new=is_new, is_selected=is_selected)
        self.attrs = attrs if attrs else {}
        self.side = side

    def has_attrtype(self, attrtype):
        return attrtype in self.attrs

    def get_attrval(self, attrtype):
        return self.attrs.get(attrtype, None)

    def get_attrtypes(self):
        return self.attrs.keys()

    def add_attrtypes(self, attrtypes, attrvals=None):
        if not attrvals:
            attrvals = (None,)*len(attrtypes)
        assert len(attrtypes) == len(attrvals)
        for i, attrtype in enumerate(attrtypes):
            self.set_attrtype(attrtype, attrvals[i])

    def add_attrtype(self, attrtype, attrval=None):
        self.attrs[attrtype] = attrval

    def set_attrtype(self, attrtype, attrval=None):
        self.attrs[attrtype] = attrval

    def remove_attrtype(self, attrtype):
        assert attrtype in self.attrs, "Error - {0} was not found in \
self.attrs: {1}".format(attrtype, self.attrs)
        self.attrs.pop(attrtype)

    def relabel_attrtype(self, oldname, newname):
        assert oldname in self.attrs, "Error - {0} was not found in \
self.attrs: {1}".format(oldname, self.attrs)
        oldval = self.get_attrval(oldname)
        self.remove_attrtype(oldname)
        self.set_attrtype(newname, oldval)

    def copy(self):
        """ Return a copy of myself """
        return AttributeBox(self.x1, self.y1, 
                           self.x2, self.y2, label=self.label,
                           color=self.color, id=self.id, is_contest=self.is_contest,
                           contest_id=self.contest_id, is_new=self.is_new,
                           is_selected=self.is_selected,
                           target_id=self.target_id,
                            attrs=self.attrs,
                            side=self.side)

    def marshall(self):
        """
        Return a dict-equivalent of myself.
        """
        data = BoundingBox.marshall(self)
        data['attrs'] = self.attrs
        data['side'] = self.side
        return data

    @staticmethod
    def unmarshall(data):
        box = AttributeBox(0,0,0,0)
        for (propname, propval) in data.iteritems():
            setattr(box, propname, propval)
        return box

    def __eq__(self, a):
        return (a and self.x1 == a.x1 and self.y1 == a.y1 and self.x2 == a.x2
                and self.y2 == a.y2 and self.is_contest == a.is_contest
                and self.label == a.label and self.attrs == a.attrs 
                and self.side == a.side)
    def __repr__(self):
        return "AttributeBox({0},{1},{2},{3},attrs={4},side={5})".format(self.x1, self.y1, self.x2, self.y2, self.attrs, self.side)
    def __str___(self):
        return "AttributeBox({0},{1},{2},{3},attrs={4},side={5})".format(self.x1, self.y1, self.x2, self.y2, self.attrs, self.side)

class IWorldState(WorldState):
    def __init__(self, box_locations=None):
        WorldState.__init__(self, box_locations=box_locations)

    def get_attrpage(self, attrtype):
        return self.get_attrbox(attrtype).side

    def get_attrtypes(self):
        """
        Return a list of all attribute types.
        """
        result = set()
        for b in self.get_attrboxes():
            for attrtype in b.get_attrtypes():
                result.add(attrtype)
        return list(result)

    def get_attrbox(self, attrtype):
        """
        Return the AttributeBox with given attrtype.
        """
        for b in self.get_boxes_all_list():
            if b.has_attrtype(attrtype):
                return b
        print "== Error: In IWorldState.get_attrbox, no AttributeBox \
with type {0} was found."
        return None

    def get_attrboxes(self):
        """
        Return all AttributeBoxes in a flat list.
        """
        return self.get_boxes_all_list()

    def remove_attrtype(self, attrtype):
        """
        Removes the attrtype from all instances of AttributeBoxes.
        """
        for temppath, boxes in self.box_locations.iteritems():
            for b in boxes:
                if attrtype in b.get_attrtypes():
                    b.remove_attrtype(attrtype)
        self._remove_empties()

    def _remove_empties(self):
        newboxlocs = {}
        for temppath, boxes in self.box_locations.iteritems():
            newboxlocs[temppath] = [b for b in boxes if b.get_attrtypes()]
        self.box_locations = newboxlocs

    def remove_attrbox(self, box):
        for temppath in self.box_locations:
            if box in self.get_boxes(temppath):
                self.remove_box(temppath, box)

    def mutate(self, iworld):
        WorldState.mutate(self, iworld)

def dump_attrboxes(attrboxes, filepath):
    listdata = [b.marshall() for b in attrboxes]
    f = open(filepath, 'wb')
    pickle.dump(listdata, f)
    f.close()
def load_attrboxes(filepath):
    f = open(filepath, 'rb')
    listdata = pickle.load(f)
    f.close()
    return [AttributeBox.unmarshall(b) for b in listdata]

def marshall_iworldstate(world):
    """
    Marshall world.box_locations such that it's 
    possible to pickle them.
    """
    boxlocs = {}
    for temppath, boxes in world.box_locations.iteritems():
        for b in boxes:
            b_data = b.marshall()
            boxlocs.setdefault(temppath, []).append(b_data)
    return boxlocs

def unmarshall_iworldstate(data):
    """
    Unmarshall data, which is of the form:
        <boxlocations data>
    to return a new IWorldState.
    """
    iworld = IWorldState()
    new_boxlocations = {}
    boxlocsdata = data
    for temppath, boxesdata in boxlocsdata:
        for b_data in boxesdata:
            box = AttributeBox.unmarshall(b_data)
            new_boxlocations.setdefault(temppath, []).append(box)
    iworld.box_locations = new_boxlocations
    return iworld

def load_iworldstate(filepath):
    f = open(filepath, 'rb')
    data = pickle.load(filepath)
    return unmarshall_iworldstate(data)

def dump_iworldstate(iworld, filepath):
    f = open(filepath, 'wb')
    pickle.dump(marshall_iworldstate(iworld), f)
    f.close()

class TextInputDialog(wx.Dialog):
    """
    A dialog to accept N user inputs.
    """
    def __init__(self, parent, caption="Please enter your input(s).", 
                 labels=('Input 1:',), 
                 vals=('',),
                 *args, **kwargs):
        """
        labels: A list of strings. The number of strings determines the
              number of inputs desired.
        vals: An optional list of values to pre-populate the inputs.
        """
        wx.Dialog.__init__(self, parent, title='Input required', *args, **kwargs)
        self.parent = parent
        self.results = {}

        self.input_pairs = []
        for idx, label in enumerate(labels):
            txt = wx.StaticText(self, label=label)
            input_ctrl = wx.TextCtrl(self, style=wx.TE_PROCESS_ENTER)
            if idx == len(labels) - 1:
                input_ctrl.Bind(wx.EVT_TEXT_ENTER, self.onButton_ok)
            try:
                input_ctrl.SetValue(vals[idx])
            except:
                pass
            self.input_pairs.append((txt, input_ctrl))
        panel_btn = wx.Panel(self)
        btn_ok = wx.Button(panel_btn, id=wx.ID_OK)
        btn_ok.Bind(wx.EVT_BUTTON, self.onButton_ok)
        btn_cancel = wx.Button(panel_btn, id=wx.ID_CANCEL)
        btn_cancel.Bind(wx.EVT_BUTTON, self.onButton_cancel)
        panel_btn.sizer = wx.BoxSizer(wx.HORIZONTAL)
        panel_btn.sizer.Add(btn_ok, border=10, flag=wx.RIGHT)
        panel_btn.sizer.Add(btn_cancel, border=10, flag=wx.LEFT)
        panel_btn.SetSizer(panel_btn.sizer)

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        caption_txt = wx.StaticText(self, label=caption)
        self.sizer.Add(caption_txt, border=10, flag=wx.ALL)
        gridsizer = wx.GridSizer(rows=0, cols=2, hgap=5, vgap=3)
        gridsizer.Add(self.input_pairs[0][0])
        gridsizer.Add(self.input_pairs[0][1])
        for txt, input_ctrl in self.input_pairs[1:]:
            gridsizer.Add(txt, border=10, flag=wx.ALL)
            gridsizer.Add(input_ctrl, border=10, flag=wx.ALL)
        self.gridsizer = gridsizer
        self.sizer.Add(gridsizer)
        self.sizer.Add(panel_btn, border=10, flag=wx.ALL | wx.ALIGN_CENTER)
        self.SetSizer(self.sizer)

        self.Fit()

        self.input_pairs[0][1].SetFocus()

    def onButton_ok(self, evt):
        for txt, input_ctrl in self.input_pairs:
            self.results[txt.GetLabel()] = input_ctrl.GetValue()
        self.EndModal(wx.ID_OK)
    def onButton_cancel(self, evt):
        self.EndModal(wx.ID_CANCEL)

if __name__ == '__main__':
    class MyFrame(wx.Frame):
        def __init__(self, parent, *args, **kwargs):
            wx.Frame.__init__(self, parent, *args, **kwargs)
            btn = wx.Button(self, label="Click me")
            btn.Bind(wx.EVT_BUTTON, self.onButton)
            
        def onButton(self, evt):
            dlg = TextInputDialog(self, labels=("Input 1:", "Input 2:", "Input 3:"))
            dlg.ShowModal()
            print dlg.results
    app = wx.App(False)
    frame = MyFrame(None)
    frame.Show()
    app.MainLoop()
