'''
Re-exported module containing all the relevant panels
'''
from s01_new_project.panel import ProjectPanel
from s02_configure.panel import ConfigPanel
from s03_partition.panel import PartitionMainPanel
from s04_attributes.panel import BallotAttributesPanel
from s05_label_digits.panel import LabelDigitsPanel
from s06_run_grouping.panel import RunGroupingMainPanel
from s07_verify_grouping.panel import VerifyGroupingMainPanel
from s08_select_targets.panel import SelectTargetsMainPanel
from s09_label_contests.panel import LabelContest
from s10_extract_targets.panel import TargetExtractPanel
from s11_set_threshold.panel import ThresholdPanel
from s12_quarantine.panel import QuarantinePanel
from s13_process.panel import ResultsPanel

PANEL_CLASSES = [
    ProjectPanel,
    ConfigPanel,
    PartitionMainPanel,
    BallotAttributesPanel,
    LabelDigitsPanel,
    RunGroupingMainPanel,
    VerifyGroupingMainPanel,
    SelectTargetsMainPanel,
    LabelContest,
    TargetExtractPanel,
    ThresholdPanel,
    QuarantinePanel,
    ResultsPanel
]
