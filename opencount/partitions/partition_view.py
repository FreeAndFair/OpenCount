import wx


class PartitionStrings:
    BARCODE_MSG = '''
Would you like to skip barcode overlay verification? It tends
to be computationally time-consuming, not very helpful for
certain vendors (e.g. Hart), and typically is unnecessary.
'''


class StatLabel(wx.BoxSizer):

    def __init__(self, parent, name):
        wx.BoxSizer.__init__(self, wx.HORIZONTAL)
        self._name = wx.StaticText(parent, label=name + ": ")
        self._value = wx.StaticText(parent)
        self.AddMany([(self._name,), (self._value,)])

    def set_value(self, val):
        self._value.SetLabel(str(val))


def VBox(contents, *args, **kwargs):
    sizer = wx.BoxSizer(wx.VERTICAL, *args, **kwargs)
    sizer.AddMany([(x,) for x in contents])
    return sizer


def HBox(stuff, *args, **kwargs):
    sizer = wx.BoxSizer(wx.HORIZONTAL, *args, **kwargs)
    sizer.AddMany([(x,) for x in contents])
    return sizer


class PartitionMainView(wx.Panel):

    def __init__(self, parent, model, *args, **kwargs):
        wx.Panel.__init__(self, parent, *args, **kwargs)
        self.model = model
        self.init_ui()

    def init_ui(self):
        self.panel = PartitionPanel(self, self.model)
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.panel, proprtion=1, flag=wx.EXPAND)
        self.Layout()


class PartitionPanel(wx.lib.scrolledpanel.ScrolledPanel):

    def __init__(self, parent, model, *args, **kwargs):
        wx.Panel.__init__(self, parent, *args, **kwargs)
        self.model = model
        self.init_ui()

    def init_ui(self):
        # statistics panel:
        self.num_partitions = ffwx.StatLabel(self, 'Number of Partitions')
        self.part_largest = ffwx.StatLabel(self, 'Largest Partition Size')
        self.part_smallest = ffwx.StatLabel(self, 'Smallest Partition Size')
        self.sizer_stats = ffwx.vbox(
            self.num_partitions,
            self.part_largest,
            self.part_smallest,
        )
        self.sizer_stats.ShowItems(False)

        # run button:
        self.btn_run = ffwx.Button(
            self, label='Run Partitioning', on_click=self.onButton_run
        )
        btn_sizer = ffwx.hbox(self.btn_run)

        # 'skip verification' checkbox:
        txt_skiphelp = ffwx.static_wrap(
            self, Strings.BARCODE_OVERLAY_HELP, 100
        )
        self.chkbox_skip_verify = ffwx.CheckBox(
            self, 'Skip Overlay Verification (Recommended)', True
        )
        sizer_skipVerify = ffwx.vbox(
            txt_skiphelp,
            self.chkbox_skip_verify,
        )

        # dev button options
        btn_loadPartialDecoding = ffwx.Button(
            self,
            label="(Dev) Apply previous decoding results, decode remaining images. [only valid with Skip Overlay Verify]",
            on_click=self.onButton_loadPartialDecoding,
        )
        btn_loadDecoding = ffwx.Button(
            self,
            label="(Dev) Load complete previous decoding results.",
            on_click=self.onButton_loadDecoding,
        )

        if not config.IS_DEV:
            btn_loadPartialDecoding.Hide()
            btn_loadDecoding.Hide()

        sizer_devbuttons = ffwx.hbox(
            btn_loadPartialDecoding,
            (10, 0),
            btn_loadDecoding,
        )

        self.sizer = ffwx.vbox(
            btn_sizer,
            (50, 50),
            sizer_skipVerify,
            (50, 50),
            self.sizer_stats,
            sizer_devbuttons
        )
        self.SetSizer(self.sizer)
        self.Layout()
        self.SetupScrolling()

    def button_load_partial_decoding(self, event):
        self.model.load_partial_decoding()

    def button_load_decoding(self, event):
        self.model.load_decoding()

    def button_run_partitioning(self, event):
        self.model.run_partitioning()
