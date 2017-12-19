# In Progress #

This is still being written.

# Introduction #

In this tutorial, you will learn how to output precinct-level
results. OpenCount will perform this by decoding a human-readable
precinct stamp that is present on each ballot.

Note: This tutorial assumes that you have already completed the
main [Tutorial](Tutorial.md), which introduces you the OpenCount pipeline. If
you haven't completed that tutorial yet, please do so before
beginning with this one.

# Getting Started #

Create a new project called "TutorialPrecinct", and use the
same dataset from Tutorial at:
> opencount/test-ballots/tutorial\_dataset/votedballots/

The project configuration should be:
  * **Number of Pages:** 2
  * **Ballot Pairing**: Ballots alternate
  * **Vendor**: Hart

Run the barcode decoding computation. Once it is complete,
proceed to the **Attributes** tab on the top menu bar:

<img src='http://i.imgur.com/U7ph5rG.png' alt='tutorialprecinct_attrtab' title='Hosted by imgur.com' width='90%' />

# Ballot Attributes #

A Ballot Attribute is some part of the image that contains some
relevant information in a human-readable format. Common
ballot attributes are:
  * Precinct Number
  * Tally Group
  * Party Affiliation

For instance, here is a ballot with a few ballot attributes
marked:

<img src='http://i.imgur.com/uSTQJwG.jpg' alt='ballot_attrs_example' title='Hosted by imgur.com' width='75%' />

In the "Ballot Attributes" user interface, you are asked to
mark one ballot from each ballot style with the ballot attributes
that you are interested in.

In this tutorial, we will focus on defining a **Precinct Number**
attribute. This differs from other attributes in that it consists
of a series of **decimal digits**.

To define the **Precinct Number** attribute, click the **Add Attribute**
button on the top button bar, and create a box around the precinct
stamp like so:

<img src='http://i.imgur.com/paiylXJ.png' alt='1a_drawattrbox' title='Hosted by imgur.com' width='90%' />

Tip: It is highly recommended to draw a box that is larger than the
actual precinct stamp. This is to allow the region to still capture
the precinct stamp even in the presence of translation/rotation
on other ballots. In particular, for some elections the printed
precinct stamp location can vary from language to language.

Once you create the box, a pop-up dialog will appear asking for
the name of this ballot attribute. Type in "precinct", and check
the "Is this a digit attribute?" checkbox. Click Ok:

<img src='http://i.imgur.com/Cm7Fewv.png' alt='1b_attrname' title='Hosted by imgur.com' width='60%' />

In the next dialog, perform the following steps:
  * Check the "Is this a digit-based attribute"?
  * In this election, there are 7 digits in a precinct stamp. So, set the "Number of Digits" to: 7.
  * Check the "Is this attribute consistent within each partition (as defined by the barcodes)?" and "Is this attribute for tabulation only?" checkboxes.

<img src='http://i.imgur.com/SZR7wxL.png' alt='1c_attrdetails' title='Hosted by imgur.com' width='75%' />

Note: For most election vendors, you will want to check both the
"Is this attribute consistent within each partition" and "Is this
attribute for tabulation only?" checkboxes. This will be the case
for Hart and Diebold vendors, for instance. The first checkbox is
checked because attributes such as precinct, language, party, and
tally group are implicitly encoded in the barcodes of each ballot -
thus, we are certain that we only have to annotate one ballot from
each style. The second checkbox is true because we only wish to
extract precinct numbers to generate precinct-level results. In
a future tutorial, you will see an example election where we will
not check these two checkboxes.

We are done defining this attribute: click the "Add this Attribute"
button. The newly-defined attribute should be displayed on the
ballot:

<img src='http://i.imgur.com/QsDHrCo.png' alt='1d_attrdefined' title='Hosted by imgur.com' width='90%' />

This is the only ballot attribute we want to define: click the
"Mark All Ballots Done" on the bottom button menu. Move onto the
next step "Label Digit Attrs" at the top menu bar.

# Label Digit Attrs #

In this step, you will annotate the precinct stamp on one ballot
from every ballot style. To reduce workload, you will not
individually annotate each precinct stamp separately. Instead,
you will annotate each individual digit, and OpenCount will
automatically search for instances of that digit across all
precinct stamps.

To start off, draw a box around the '0' digit as shown. You
may use the "Zoom In"/"Zoom Out" buttons on the bottom button
bar if you wish:

<img src='http://i.imgur.com/SGjelnM.png' alt='2a_draw_1st_digit' title='Hosted by imgur.com' width='90%' />

A dialog will appear, asking what digit value this is. Type
0, and click "Ok".

OpenCount will now perform an automated search for all
repeated instances of the '0' digit. Depending on how many
ballot styles there are, this may take some time.

# Overlay Verification #

Once the automated search is complete, an overlay verification
window will pop up:

<img src='http://i.imgur.com/UHFCaSw.png' alt='2b_overlay' title='Hosted by imgur.com' width='90%' />

The **min** and **max** overlay images are useful tools that help
us quickly verify whether all purported matches are indeed '0'
digits.

In this image, OpenCount detected 6 possible matches, and
displayed the **min** and **max** overlays. Your task is to determine
if the **min** and **max** overlay both closely resemble the digit '0'.
If either one of the **min** or **max** overlays don't look like a
'0', then it is not likely that every digit in the overlay is a '0'.

In this particular example, we can safely assume that every digit
in this group is a '0' - click the **Accept (all match)** button. Your
screen should now reflect the matches:

<img src='http://i.imgur.com/RCTFJzZ.png' alt='2c_firstdigitmatchdone' title='Hosted by imgur.com' width='90%' />

For illustration, let's step through an example where the overlays
are ambiguous. Here are the **min** and **max** overlays of several '6'
digits mixed in with '0' digits:

<img src='http://i.imgur.com/gQiXZO1.png' alt='badoverlay_1_noisyoverlay' title='Hosted by imgur.com' width='90%' />

If the **min** and/or **max** overlay looks suspect, then you can
**Split** the current group of images into two different groups to
try to separate the 'wrong' matches from the 'right' matches.
To do this, click the **Split and Continue** button on the bottom
button bar (highlighted in green above).

As you can see, doing one split separated the '6' from the
'0' digits well in this case. Thus, for the good matches we
can click the **Accept (all match)** button.

<img src='http://i.imgur.com/B3VpMHF.png' alt='badoverlay_2_split_is_0' title='Hosted by imgur.com' width='90%' />

For the bad match (the '6'), we will click the **Reject (some don't match)**
button:

<img src='http://i.imgur.com/ehpklqA.png' alt='badoverlay_split_is_6' title='Hosted by imgur.com' width='90%' />

# Completing the Digit Labeling #

Continue labeling the rest of the digits. If you wish to
move or delete a digit box, click the **Modify** button on
the bottom button bar, and:
  * To **move** a box, simply click and drag the box (or use the arrow keys while a box is selected)
  * To **delete** a box, simply click a box and press the **Backspace** or **Delete** key.

To **sort** the precinct stamps by number of detected digits,
click the **Sort Cells** button on the bottom button bar. This
will rearrange the precinct stamps so that the precinct stamps
with the fewest detected digits are displayed first. This is
a helpful feature when there are many precinct stamps to annotate.

<img src='http://i.imgur.com/6mOYcVn.png' alt='sortcells_1' title='Hosted by imgur.com' width='90%' />

<img src='http://i.imgur.com/1CEum2s.png' alt='sortcells_2' title='Hosted by imgur.com' width='90%' />

Sometimes, you may want to manually label a precinct stamp.
For instance, this is useful when a significant portion of the
precinct stamp is covered by voter marks or scanner streaks.

To manually label a precinct stamp, right click on the precinct
stamp, and choose the "Manually Label..." entry from the context
menu:

<img src='http://i.imgur.com/bY8SuA6.png' alt='manuallabel_0' title='Hosted by imgur.com' width='90%' />

Enter the precinct number one digit at a time, separated
by commas, into the dialog:

<img src='http://i.imgur.com/2MWB5iL.png' alt='manuallabel_1' title='Hosted by imgur.com' />

The precinct patch will be updated to reflect that it has been manually labeled:

<img src='http://i.imgur.com/eeQhGnQ.png' alt='manuallabel_2' title='Hosted by imgur.com' width='90%' />

To delete a manual labeling, right click on the precinct patch,
and choose the **"Undo Manual Label"** entry.

Once you have finished annotating both precinct stamps, proceed
to the next step by clicking the "Group" tab on the top menu bar.

# Bookkeeping #

In the "Group" page, click the "Run Grouping" button. Click
through the dialogs. Then go to the next step "Correct Grouping" -
it will tell you that no grouping verification is required. This is
because we selected the "Ballot Attributes are consistent within
each partition" option. Finally, proceed to the "Select Voting Targets"
step, and complete the election as normal.

# Precinct-Level Results #

In the **election\_results.txt** and **election\_results\_batches.txt** files, there will now be additional
precinct-level reporting.

Thanks!