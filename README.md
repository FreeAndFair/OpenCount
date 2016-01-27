OpenCount
===

*OpenCount* is part of the Verfiable Elections software suite of
products, developed by Verifiable Elections, Inc. It has two
components: *OpenCount Tabulator* (hereafter,
*Tabulator*), which tabulates results from ballots cast in an election, and *OpenCount Auditor* (hereafter, *Auditor*), which is designed
to assist with post-election audits of elections conducted using
optical-scan paper ballots.

*Tabulator* helps count a set of paper ballots that were cast in an
election. If you provide scanned images of all of the paper ballots,
*Tabulator* will identify all votes on the ballots and count and
tally the votes. *Tabulator* currently supports optical-scan ballots
associated with Diebold (Premier), ES&S, Hart, and Sequoia ballot
styles.

*Auditor* provides tools to carry out risk-limiting ballot-level audits, which are an innovative, efficient, and cost-effective way to provide transparency and check the accuracy of election results.

Currently, only the *Tabulator* component is implemented. Parts of the current implementation were used to support the
[California Secretary of State's Post Election Risk-Limiting Audit Pilot Program](http://www.sos.ca.gov/voting-systems/oversight/risk-limiting-pilot.htm).
During the development of the current implementation, it was tested on ballots cast in real elections held in ten California counties.


Development Process and Methodology
===

The current OpenCount application was written in Python and uses the OpenCV computer vision library. Future versions of OpenCount will be developed using the Trust-by-Design (TBD) software engineering
methodology, documented in several papers published by Joe
Kiniry and his coauthors.

Requirements
===

What follows are the mandatory and secondary requirements imposed
upon *OpenCount*.

Mandatory Requirements
===

* Must be able to correctly interpret generated ballot templates and scanned ballot images, including those that are skewed by small angles during the scanning process.
* Must allow election officials to annotate generated ballot templates and scanned ballots with the locations of voting targets and contest regions and to record contest text (i.e., contest names, candidate names and referendum choices).
* Must be able to handle write-in votes.
* Must be able to detect ambiguously- or incompletely-marked voting targets.
* Must flag problematic ballots for manual inspection and annotation.
* Must be able to generate accurate cast vote records (CVRs) from scanned ballot images and annotations.
* Must be able to display specific ballot images and the corresponding CVRs as required by the risk-limiting audit process.

Secondary Requirements
===

#### Usability:
* The user interface must be easy to use for non-technical users (i.e., election officials).
* Progress indicators must be provided for operations that are expected to run for long periods of time.

#### Persistence:
* The application will allow election projects in progress to be saved/checkpointed, so that they need not be completed in a single sitting.
* The application will exhibit minimal data loss from an arbitrary failure (e.g., a typical system failure like a Windows crash) of the machine running the application.

#### Automation:
* The application should be able to automatically perform a risk-limiting audit with specified parameters, to the extent that such automation is possible within the risk-limiting audit process.

#### Scalability:
* The application should be able to use multiple processors/cores to process scanned ballots in parallel.

#### Analysis:
* The application should be able to provide an analysis of the processed ballot information (e.g., numbers of undervotes and overvotes, number of problematic ballots, etc.).

Current Status
===

The current version of OpenCount includes only the Tabulator component and provides no direct assistance with risk-limiting audits, other than by naming ballot images and corresponding CVRs in the filesystem appropriately to enable easy lookup during risk-limiting audits.


History
===

OpenCount was originally developed as part of research conducted by
researchers at UC Berkeley and UC San Diego, including Kai Wang, Eric
Kim, Nicholas Carlini, Theron Ji, Arel Cordero, Andrew Chang, George
Yiu, Ivan Motyashov, Daniel Nguyen, Raji Srikantan, Alan Tsai, Keaton
Mowery, David Wagner, and others. It was partially funded by the US
National Science Foundation and by the TRUST center; we gratefully
acknowledge their support. We also thank the California Secretary of
State, election officials at Alameda, Leon, Madera, Marin, Merced,
Napa, Orange, San Luis Obispo, Santa Cruz, Stanislaus, Ventura, and
Yolo counties, and Clear Ballots for data and assistance.
