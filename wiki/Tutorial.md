# Under Construction #

Mind the dust - this is still being written.

# Introduction #

In this step-by-step illustrated tutorial, you'll learn the basics of
OpenCount:
  * Creating and configuring projects
  * Grouping ballots by style
  * Annotating each ballot style
  * Interpreting ballots
  * Generating Cast Vote Records (CVRs)

First, you will need to install OpenCount onto your system. Please
visit InstallNotes for
instructions on how to do so.

# The Dataset #

The election dataset that we will be processing for the tutorials are
a set of Hart ballots that have been artificially marked using an
image editing software. They can be found in the repository at:

> opencount/test-ballots/tutorial\_dataset/votedballots/

This dataset happens to consist of double-sided ballots, and two
different ballot styles are present (groupA and groupB).

# Basic OpenCount Interface #

The overarching OpenCount interface is quite simple. The most important
component is the top menu bar, outlined below:

<img src='http://i.imgur.com/ShGUOsX.png' alt='overviewui' title='Hosted by imgur.com' width='90%' />

Here are all of the steps of the OpenCount pipeline, laid out
in order. To move to a step, simply click on the corresponding
tab.

Note: Currently, OpenCount may not work correctly if you go back to a
previous step, makes some changes, and move forward again. To be safe,
if you want to make changes to a previous step, do the following:
  * Exit OpenCount
  * Re-open the project, and skip to the step you want to modify
  * Make the desired changes, and then proceed as normal.

# Creating the Project #

First, run OpenCount from the terminal by moving to the
`opencount/opencount/` directory, and entering:

` $ python maingui.py `

Assuming that all dependencies have been correctly installed, the user
interface should be displayed.

First, you will want to create and name the project for this election.
Click the `Create New Project...` button, and enter a name (for
instance, `TutorialElection`):

<img src='http://i.imgur.com/jrFFaAo.png' alt='0a_createproj' title='Hosted by imgur.com' width='90%' />

Once you have created the project, it will show up on the list of
projects. You may create and delete projects, but beware: deleting a
project is final and irreversible. Don't worry, deleting a project
will **not** delete the voted ballots.

Select your newly-created project from the list, and continue to the
next step by clicking the `Import` tab from the top menu.

<img src='http://i.imgur.com/RfqS3qn.png' alt='0b_createproj' title='Hosted by imgur.com' width='90%' />

## Configuring the Project ##

Next, we will tell OpenCount the characteristics of the election,
including:
  * Where the voted ballot images reside
  * How many sides are on each ballot (i.e., single-sided, double-sided, etc.)
  * Vendor

First, select the `Choose voted ballot directory...' button, and show
OpenCount the directory of voted ballots.

Next, this election consists of double-sided ballots. So, we can
leave the `Number of pages` as `2`.

In this next component `Ballot Grouping/Pairing Configuration`, we
need to tell OpenCount how to pair up the individual images in the
voted ballot directory into ballots. Recall that each side of the
ballot consists of separate images. For instance, if we take a look at
the groupA/ directory, the image filenames are:

  * A\_00\_side0.png
  * A\_00\_side1.png
  * A\_01\_side0.png
  * A\_01\_side1.png
  * A\_02\_side0.png
  * A\_02\_side1.png

In this case, the ballot pairing is determined by simply pairing
consecutive ballots. Note that OpenCount does not require that the
order of the filenames dictate "front" and "back".

Thus, under the `Ballot Grouping/Pairing Configuration`, you will
check the `Ballots alternate front and back` box.

Finally, the Vendor in this election is Hart. So, choose the Hart
entry in the dropdown menu.

Your screen should now look like this:

<img src='http://i.imgur.com/O4MpF3D.png' alt='1b_configproj' title='Hosted by imgur.com' width='90%' />

Move onto the next step by clicking the `Partition` tab at the top
menu.

# Decoding Ballots #

In this step, we will decode the barcodes present on each ballot, and
group the ballots by style.

To start this process, simply press the `Run Partitioning...` button.
Depending on how large the election is, this may take several hours.
But on this election dataset, it should complete in a few seconds.

For this tutorial, we will skip the Move onto the next step by clicking the `Select Voting Targets` tab
at the top menu. For this tutorial, we will skip the steps concerning
**Ballot Attributes**.

# Ballot Attributes #

In this first tutorial, we will skip ballot attributes to keep things
simple. After you have completed this tutorial, feel free to check
out the tutorials covering attributes here:
  * Precinct-stamp decoding
  * Generic ballot attributes (tally group, etc.)



# Ballot Annotation #

In this step, we will annotate each ballot style with the following
information:
  * Location of voting targets and contests
  * Contest text information

In other words, the end result of this step will look something like
this:

<img src='http://i.imgur.com/2t1xEyJ.png' alt='show_annotate' title='Hosted by imgur.com' width='90%' />

## Style Annotation ##

First, we will annotate the location of voting targets and contests.

Click the `Add Target` button on the top button bar to begin marking
the location of voting targets. The mouse cursor, when hovered over
the ballot, should change to a cross-like cursor when you are in the
`Add` mode.

To mark your first voting target, click and drag a box around one of
the empty voting targets, then release the mouse:

<img src='http://i.imgur.com/lh3HSzB.png' alt='1a_firsttarget' title='Hosted by imgur.com' width='90%' />

For the first voting target that you mark, a pop-up will come up,
asking you to indicate where the voter is expected to fill in the
voting target. This information is used in future steps to help
determine which voting targets are filled in. Draw a box that encloses the entire interior of the voting target, and click `Use this region` to accept:

<img src='http://i.imgur.com/OVF4R4s.png' alt='targetroi_big' title='Hosted by imgur.com' width='65%' />

You may also change this Mark Region by clicking the "Set Mark Region" button on the top button menu bar.

Note: it is actually recommended to select a smaller Mark Region whenever
possible. For instance, this is a Mark Region that will improve OpenCount's
ability to discern filled-in from unfilled-in voting targets:

<img src='http://i.imgur.com/xhsTBhP.png' alt='1b_targetroi' title='Hosted by imgur.com' width='65%' />

However, for the purposes of this tutorial use the larger Mark Region area.

Now, OpenCount will run an automated search to find all instances of
voting targets. Once it completes, the user interface will be updated
accordingly:

<img src='http://i.imgur.com/B4EAsRc.png' alt='1c_tempmatch' title='Hosted by imgur.com' width='60%' />

We will need to examine all images to verify that every voting target
was indeed detected correctly. We also need to make sure that false
matches are discarded. To advance through the images, use the
`Next Image...` button on the bottom button bar. Click this button
once.

On this ballot side, one of the voting targets was not correctly
detected on the lower right hand side.

<img src='http://i.imgur.com/OonG6tm.png' alt='2a_targetmiss' title='Hosted by imgur.com' width='60%' />

Draw another box around the voting target. Note that this will trigger
another automated search:

<img src='http://i.imgur.com/NpcQOH1.png' alt='mark_missing_target' title='Hosted by imgur.com' width='90%' />

Advance to the next image (by clicking the `Next Image...` button).
Here, we see an instance where a voting target was not found because
the voting target is filled in:

<img src='http://i.imgur.com/fXoVWi9.png' alt='marked_target' title='Hosted by imgur.com' width='60%' />

You have two options to mark this voting target:
  * Draw another box around this filled in voting target with the `Add Target` button. This will trigger an automated search, which may take a few minutes depending on the election size.
  * Use the `Force Add Target` button (highlighted in blue above) to mark the filled-in voting target without triggering an automated search. This is handy if you wish to mark a voting target without waiting for an automated search to complete.

Whatever option you choose, you will still draw a box around the
voting target:

<img src='http://i.imgur.com/aASL2Mr.png' alt='verify_marked_target' title='Hosted by imgur.com' width='90%' />

Step through the rest of the images to make sure that all voting
targets have been detected.

Once you have verified that all voting targets have been detected,
the next step is to provide the bounding boxes for each contest.
The typical way to do this is to first run the automated contest box
detection procedure, and then verify/correct the results.

To run the automated contest detection, click the `Infer Contest Regions...`
button on the top button bar. This will trigger some computation that
may take a few minutes, depending on the election size:

<img src='http://i.imgur.com/if8wMI4.png' alt='4a_runinfercontests' title='Hosted by imgur.com' width='90%' />

Once the computation completes, step through the images and verify that
all contests have been correctly detected. Common errors to look out
for are:
  * Contests that are incorrectly split into 2+ contests
  * Contests that have been incorrectly merged
  * Missing contests

### Tips ###

You can add new contest bounding boxes by clicking the `Add Contest`
button on the top button bar. To resize, move, or delete a contest bounding
box, click the `Modify` button on the top button bar. You can resize
a box by clicking+dragging any of the corners of the box. To move a
box, simply click on the interior of a box, and click+drag to move it.
Finally, to delete a box, select the box, and press the `DELETE` or
`BACKSPACE` key on your keyboard.

Finally, you can select multiple boxes at once by clicking on an
uninhabited area of the ballot, and dragging a selection box around
multiple boxes. You can then move/delete multiple boxes at once.

## Contest Data Entry ##

In this step, you will enter the title and candidate names for each
contest. To reduce the required operator effort, OpenCount will
employ an automated contest duplicate detection technique.

First, click the `Compute Equiv Classes` button to run the contest
duplicate detection routine:

<img src='http://i.imgur.com/Nhxd5t7.png' alt='labelcontest_detectduplicates' title='Hosted by imgur.com' width='90%' />

When the computation is complete, a pop-up
window will come up displaying the detected contest duplicates:

<img src='http://i.imgur.com/znYn0vB.png' alt='contestoverlay_clean' title='Hosted by imgur.com' width='90%' />

In this window, OpenCount displays all contests that it claims is the
same all at once in the form of a **min** and **max** overlay. For instance,
in the above picture there are **3** contests displayed at once.

In this instance, we can visually confirm that these all indeed refer
to the contest **"Judge of the Superior Court (Office No. 1)"**, so click the `Accept` button to confirm that these
are indeed all the same contest.

However, here is an instance where the overlay is too noisy to
definitively make a judgement:

<img src='http://i.imgur.com/pnBE0KS.png' alt='contestoverlay_messy_0' title='Hosted by imgur.com' width='90%' />

For these cases, you can either:
  * Reject the entire group of contests, and move on, or:
  * Split the current group of contests into two different groups to try to "clean" up the overlays.

When the **"Split and Continue..."** action is selected, the current group
is split into two smaller groups:

<img src='http://i.imgur.com/kDxy5ZM.png' alt='contestoverlay_messy_1_goodsplit_judge' title='Hosted by imgur.com' width='90%' />

<img src='http://i.imgur.com/bt0TIGQ.png' alt='contestoverlay_messy_2_goodsplit_county' title='Hosted by imgur.com' width='90%' />

As you can see, in this case the **"Split"** action successfully separated
the two contests into separate groups. We can then choose the **"Accept"**
action for both groups.

Tip: Try to choose the `Split` option over the `Reject` option when possible,
especially with large groups. The goal of this step is to identify as
many contest duplicates as possible, so that the manual data entry
step is minimized.

If instead you reject every contest group, then you will have to
manually enter every single contest.

## Data Entry ##

Once the contest duplicate detection process is complete, begin filling
in the contest title and candidate names. You can use the 

&lt;TAB&gt;

**and**

&lt;SHIFT&gt;

+

&lt;TAB&gt;

**to move forward/backwards among the text fields.
Pressing**

&lt;ENTER&gt;

**will advance the cursor to the next candidate name,
or to the next contest if you've reached the end of the contest.**

<img src='http://i.imgur.com/54xX4Yy.png' alt='labelcontest_dataentry' title='Hosted by imgur.com' width='90%' />

For write-in candidates, simply enter **"Writein"** as the candidate name.
If there are multiple write-in candidates, add a number to the end, such
as: "Writein1", "Writein2", etc:

<img src='http://i.imgur.com/loOLLxA.png' alt='labelcontest_writeins' title='Hosted by imgur.com' width='90%' />

For contests that allow more than one vote, be sure to update the
**"Vote for up to"** field (pointed to by the blue arrow above).

### Contests that span multiple columns (Multi-box) ###

Sometimes, a single contest spans multiple columns. This typically
happens if there are so many candidates in a single contest that they
all can't fit within a single column. For instance, in this tutorial
dataset the "United States Senators" contest spans two columns:

INSERT PICTURE SHOWING THE US SENATOR CONTEST WITH "MARK AS MULTIBOX" HIGHLIGHTED

When this occurs, you will need to inform OpenCount that this contest
spans two columns by clicking the `Mark as Multibox` button when the
**beginning** of the contest is displayed.

This will trigger some additional computation, and the overlay
verification window will show up again.

Move onto the next step by clicking the `Extract` tab on the top menu
bar.

# Ballot Interpretation #

This stage consists of two main steps:
  * OpenCount finds all voting targets on every voted ballot
  * The operator classifies the voting targets as **filled** or **empty**.

## Extract Voting Targets ##

To start the first step, click the `Run Target Extract...` button.
Depending on the size of the election, this can take some time.

Once the computation is complete, proceed to the next step by clicking
the `Threshold` tab on the top menu bar.

## Classify Voting Targets ##

The objective of this step is to classify all voting targets as
**filled** and **empty**. To do this, you will define a separating
line on the grid of voting targets that separates the **filled** targets
from the **empty** targets.

Because the voting targets are sorted by how dark the target patch is,
the filled-in targets should all reside at the top of the grid, whereas
the empty targets will be at the bottom of the grid.

<img src='http://i.imgur.com/JWrVoCk.png' alt='threshold_overview' title='Hosted by imgur.com' width='90%' />

To place the separating line, right click on the location, and choose
the **Set Line** option from the context menu. A green line will be
displayed: all targets above the line will be treated as **filled**,
and all targets below the line will be treated as **empty**.

<img src='http://i.imgur.com/2i4LKsu.png' alt='threshold_setline_0' title='Hosted by imgur.com' width='90%' />

<img src='http://i.imgur.com/fmapQbW.png' alt='threshold_setline_1' title='Hosted by imgur.com' width='90%' />

### Correcting Mis-classified Targets ###

Sometimes, a perfect separating line is not possible. For instance,
faint voter marks may lie below the separating line, and hesitation
marks may lie above the separating line.

For instance, in this tutorial dataset you will notice that a few
marked voting targets are mixed in with a few empty voting targets.


<img src='http://i.imgur.com/OKsmhBU.png' alt='threshold_misclassify' title='Hosted by imgur.com' width='90%' />

Because these filled-in targets reside below the separating line, they
would be interpreted as **empty** unless we flag them as mis-classified.
To do so, simply left-click the voting targets in question: flagged
targets will have a pink transparent background.

Similarly, if any empty voting targets lie above the separating line,
flag them by left-clicking them to indicate that they are actually
empty:

<img src='http://i.imgur.com/mRdtpVH.png' alt='threshold_mark_misclass' title='Hosted by imgur.com' />

#### Ambiguous Marks ####

Sometimes, it is difficult to tell determine the voter intent merely
from the voting target itself. For instance, in the voting target
enclosed in the cyan box above, the mark may be a **hesitation**
mark, rather than an actual mark.

OpenCount provides a way to view the entire ballot, so that you can
view the proper context and make an informed judgement. To do so,
**right click** on the voting target, and click the **View Ballot...**
option:

<img src='http://i.imgur.com/il3zogz.png' alt='threshold_mark_hesitate_0' title='Hosted by imgur.com' width='90%' />

<img src='http://i.imgur.com/RSOpxEx.png' alt='threshold_mark_hesitate_1' title='Hosted by imgur.com' width='90%' />

From here, we can see that the mark is a hesitation mark after all.
Click the **Back** button to return to the grid screen, and then click
the voting target in question to mark it as being **empty**:

<img src='http://i.imgur.com/uddx8is.png' alt='threshold_mark_hesitate_2' title='Hosted by imgur.com' width='90%' />

### Quarantining Ballots ###

Sometimes, the target extraction routine will not output good results
on a ballot. You can see an example here, where a "voting target" is
in fact a region of contest text:

<img src='http://i.imgur.com/hc3vUTO.png' alt='threshold_badextract' title='Hosted by imgur.com' width='90%' />

To handle these cases, right click the offending "voting target",
click the **Show Ballot** button, then click the **Quarantine Ballot**
button.
All voting targets from this ballot will be flagged as quarantined,
and is visually indicated by an opaque dark-red mask.

These ballots will be manually interpreted in the next step.

# Process Quarantined Ballots #

Throughout the OpenCount pipeline, ballots may be flagged as "problematic"
for many reasons. This can happen via manual user intervention (such as
during "Ballot Interpretation") or during automated processing steps.

In this step, you will manually annotate such quarantined ballots with
all required information, namely:
  * Cast votes
  * Required ballot attributes, if any (such as precinct number).

If there are no quarantined ballots, then OpenCount will inform you
and this step will not be necessary.

## Discarding Ballots ##

In some cases, a scanned "ballot" may not be a ballot at all. For
instance, scans of non-ballot sheets of paper may be present in the
election dataset. For such cases, you will not want a CVR to be
outputted - to discard such ballots from the election, check the
"Discard Ballot" checkbox.

Proceed to the last step by clicking the `Results` tab on the top
menu bar.

# Generate Cast Vote Records (CVRs) #

Finally, in this final step, OpenCount will generate a CVR for each
ballot. The results are output to the project directory: if the
project name is **"TestElection"**, then the results will be found in:

```
    opencount/opencount/projects_new/TestElection/cvr/
    opencount/opencount/projects_new/TestElection/election_results.txt
    opencount/opencount/projects_new/TestElection/election_results_batches.txt
```

There are two primary output formats: **Ballot-level Cast Vote Records (CVRs)**, and **Cumulative Election Totals**.

## Ballot level CVRs ##

In the following directory will be a CVR for each ballot:
`    opencount/opencount/projects_new/PROJECTNAME/cvr/*`

A CVR is a text document that contains all votes cast by the voter on
that ballot. The format is simple:

```
CONTEST TITLE 0
    CANDIDATE THAT RECEIVED VOTE
CONTEST TITLE 1
    CANDIDATE THAT RECEIVED VOTE
...
```

If an overvote or undervote is detected, then the CANDIDATE field will
be OVERVOTE or UNDERVOTE respectively.

The directory structure of cvr/ matches the directory structure of
the input voted ballots directory.

## Cumulative election totals ##

The following two files contain the total election results:
```
    opencount/opencount/projects_new/TestElection/election_results.txt
    opencount/opencount/projects_new/TestElection/election_results_batches.txt
```

# Conclusion #

In this tutorial, you learned how to tabulate an election dataset
using the OpenCount system.

There are two more topics that you may be interested in:
  * Outputting precinct-level results
  * Determining ballot features such as tally-group.

To learn how to do both, follow the links to these additional
tutorials:
  * Precinct-stamp decoding: TutorialPrecinctResults
  * Ballot attributes

Thanks!