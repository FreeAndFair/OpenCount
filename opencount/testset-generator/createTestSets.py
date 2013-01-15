# Tool to create a dataset for testing for OpenCount development.
#
# This tool will create datasets in two modes:
#
# (1) Random Ballots - Given an election name, will just grab a random number of ballots
#                      as specified by the user. TODO: Determine how to sufficiently
#                      take randomly distributed ballots from the election directory.
# (2) Grouped Ballots - User will specify the amount of partitions as well as the number of
#                       ballots per partition, and the tool will generate these using
#                       grouping information provided by running previous elections.
# -------------------------------------------------------------------------------------------
# 
# Info for extending to support new elections:
# 
# For mode 2, the tool needs the grouping information either from:
# (1) A grouping_results.csv file, which was created after processing using 
#     OpenCount v1.
# or
# (2) Relevant pickle'd files which are created by the new pipeline, i.e.,
#     ballot_to_images.p, group_to_ballots.p, image_to_page.p
# 
#
# ---------------------------------------------------------------------------------------------
#
# USAGE:
# 
# python createTestSets.py napa 5 50 destdir/
# will create 5 random partitions each with 50 images 
# to the destdir directory using the Napa ballots.
# 
# TODO: Support multiple modes and passing flags(?)


import sys, os
import itertools
import random
import shutil
import TestSetElection

# SINGLE-SIDED ELECTIONS: MADERA, MERCED, STANISLAUS, VENTURA

ELECTION_DEFINITIONS = { 
			 #2012
			 'yolo': ['yolo', '/home/agchang/elections-grouping-info/elections-2012/yolo/grouping_results.csv', None, 'groups' , 1, ['templatepath']],
			 'napa': ['napa', '/home/agchang/elections-grouping-info/elections-2012/napa/data', '/media/data1/audits2012_straight/napa/', 'groups', 2, None],
			 'santa-cruz': ['santa-cruz', '/home/agchang/elections-grouping-info/elections-2012/santa-cruz/grouping_results.csv', None, 'groups', 1, ['templatepath']],
			 'orange': ['orange', '/home/agchang/elections-grouping-info/elections-2012/orange/oc_full_grouped_forAndy.p', None, 'groups', None, None],
			 'marin' : ['marin', '/home/agchang/elections-grouping-info/elections-2012/marin/Pol267_Plus2More/grouping_results_marin-pol267.csv', None, 'groups', 
				   1, ['templatepath']],
			 'madera' : ['madera', None, '/media/data1/audits2012_straight/madera/votedballots', 'nogroups', None, None],
                         #2011
			 'alameda' : ['alameda', '/home/agchang/elections-grouping-info/elections-2011/alameda/grouping_results_alameda.csv', None, 'groups', 1, ['templatepath']],
			 'merced' : ['merced', None, '/media/data1/audits2011_straight/merced/voted', 'nogroups', None, None],
			 'slo' : ['slo', '/home/agchang/elections-grouping-info/elections-2011/slo/grouping_results_slo.csv', None, 'groups', 1, ['templatepath']],
			 'stanislaus' : ['stanislaus', None, '/media/data1/audits2011/stanislaus/stanislaus/11-08-2011/ballots', 'nogroups', None, None],
			 'ventura' : ['ventura', None, '/media/data1/audits2011_straight/ventura/ballots-straight', 'nogroups', None, None]	
			}

# {groupID : [{ballotID : [side1, side2, ...]} , ...] , ...}

def generateTestSets(electionName, numGroups, numImages, destdir):
	# TODO: add error checking for args, generalize this
	#mElection = TestSetElection.TestSetElection(electionName, '/home/agchang/elections-grouping-info/elections-2012/napa/data', '/media/data1/audits2012_straight/napa/', 'groups', 2, None)
	#mElection = TestSetElection.TestSetElection('yolo', '/home/agchang/elections-grouping-info/elections-2012/yolo/grouping_results.csv', None, 'groups', 1, ['precinct'])	
	mElection = TestSetElection.TestSetElection(*ELECTION_DEFINITIONS[electionName])
	groupData = mElection.getSampleSets()
	# print groupData #REMOVE ME

	groupData = {key: val for key, val in groupData.iteritems() if len(val) >= numImages}
	#print groupData
	print groupData.keys()
	#print len(groupData)

	# Handle single template elections when the user specifies more than one group.
	if len(groupData) == 1 and numGroups != 1:
		raise Exception("The election only has a single template(thus group). You cannot specify more than 1 group to select ballots from")
		exit()
	

	if len(groupData) < numGroups:
		raise Exception("There aren't enough groups in this election!")
		exit()

	if len(groupData) == 0:
		raise Exception("No groups are atleast the size of the specified input")
		exit()

	#intSeq = range(len(groupData))
	selectedGroups = []
	for i in xrange(numGroups):
		#randNum = random.choice(intSeq)
		selGroup = random.choice(list(groupData.keys()))
		selectedGroups.append(groupData[selGroup])
		del groupData[selGroup]
		#intSeq.remove(randNum)

	finalGroups = []
	for i in xrange(len(selectedGroups)):
		randImages = random.sample(selectedGroups[i], numImages)
		finalGroups.append(randImages)

	# For the purposes of generating testsets of images
	# with identical ballot layouts, pick either the front or
	# back side to use throughout the entire group.
	
	os.mkdir(destdir)
	side = 0 # Using front side for now.
	for i in xrange(len(finalGroups)):
		groupDir = os.path.join(destdir,str(i))
		print "Creating ", groupDir 
		os.mkdir(groupDir)
		for j in xrange(len(finalGroups[i])):	
			for k, v in finalGroups[i][j].iteritems():
			 	# Selecting the front side
				# Need to append the enclosing directory, in case
				# of identical filenames.
				fileName = os.path.basename(os.path.dirname(v[side])) + '_' + os.path.basename(v[side])
				fullFileName = os.path.join(groupDir, fileName)
				print "Copying ", fullFileName
				shutil.copy(v[side], fullFileName)
	
		
		

if __name__ == "__main__":
	generateTestSets(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])




