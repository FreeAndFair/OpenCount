# Generalized "TestSetElection" class to handle encapsulation
# of election-specific information as well as interface
# to generate random images for test sets.

import sys, os
import pickle
import csv
from itertools import groupby
from operator import itemgetter

#elections2011 = ['alameda', 'merced', 'slo', 'stanislaus', 'ventura']
#elections2012 = ['madera', 'marin', 'napa', 'orange', 'santa-cruz', 'yolo']

class TestSetElection:
	
	def __init__(self, electionName, groupingData, imageSrc, mode, version, attributes):
		self.electionName = electionName
		self.groupingData = groupingData
		self.imageSrc = imageSrc
		self.mode = mode # Mode is either "groups" or "nogroups"
		self.version = version #version 1 => grouping_results.csv 
				       #version 2 => pickle'd files	
		self.attributes = attributes # If grouping_results.csv, need to specify attributes.
				
	# TODO: This is under the assumption that the file names for
	# the new pipeline is standardized, double-check this.

	# Returns a dictionary containing a mapping
	# from group number to a list of imagePaths,
	# in the case of mode == "nogroups", there will
	# only be a single group.
	
	def getSampleSets(self):
		if self.mode == "groups":
			if self.version == 1: # Handle the grouping_results.csv file
				res = self.processVer1GroupingData()
				return res
			elif self.version == 2: # Handle the pickle'd grouping file
				res = self.processVer2GroupingData()
				return res
			elif self.electionName == 'orange': #spec case orange
				res = self.processOrange()
				return res			
			else:
				raise Exception("Election version number not recognized.")
				exit()

		elif self.mode == "nogroups":
			res = self.processNoGroups()
			return res

		else:
			raise Exception("Mode not recognized.") #TODO: More specific Exception
			exit()

	def processVer2GroupingData(self):
		""" Handles pickle'd grouping data files """
		groupingData = sorted([os.path.join(self.groupingData, entry) 
					for entry in os.listdir(self.groupingData)]) # Warning: Assumes standardized filenames
		
		#print groupingData 

		f_ballot_to_images = open(groupingData[0], 'rb')
		f_group_to_ballots = open(groupingData[1], 'rb')
		f_image_to_page = open(groupingData[2], 'rb')

		ballot_to_images = pickle.load(f_ballot_to_images)
		image_to_page = pickle.load(f_image_to_page)
		group_to_ballots = pickle.load(f_group_to_ballots)
	
		#print ballot_to_images
		#print image_to_page
		#print group_to_ballots

		# Combine the three data structures in a single one.
		# {groupID : [{ballotID : [side1, side2, ...]} , ...] , ...}
		res = {}

		for group in group_to_ballots.keys():
			groupLst = []
			for ballotID in group_to_ballots[group]:
				orderedImageLst = ['front','back']
				for imgpath in ballot_to_images[ballotID]:
					if image_to_page[imgpath] == 0:
						orderedImageLst[0] = imgpath
					else:
						orderedImageLst[1] = imgpath

				ballotDict = {}
				ballotDict[ballotID] = orderedImageLst
				groupLst.append(ballotDict)

			res[group] = groupLst

		return res

	def processVer1GroupingData(self):
		""" Handles the grouping_results.csv file 
		    NOTE: This is legacy behavior and is meant
		    to support elections created with OpenCount v1."""		
		grouping_res = csv.reader(open(self.groupingData))
		header = grouping_res.next()
		attr_indices = []
		for attr in self.attributes:
			if attr in header:
				attr_indices.append(header.index(attr))
			else:
				raise Exception("Attribute not found in grouping_results.csv!")

		#print attr_indices

		uniquekeys = []
		groups = {}
		data = sorted(grouping_res, key=itemgetter(*attr_indices))
	
		counter = 0
		counter2 = 0
	
		def checkPath(path):
			""" Another special-case, alameda and SLO need to have their paths changed to match byrd instead of ballotscan """
			ALAMEDA_ROOT = '/media/data1/audits2011_straight/alameda-png/voted'
			SLO_ROOT = '/media/data1/audits2011_straight/slo/voted'
			
			if self.electionName == 'alameda':
				p1 = os.path.split(path)
				fold = os.path.split(p1[0])[1]
				f = p1[1]
				f = os.path.join(fold,f)
				new_path = os.path.join(ALAMEDA_ROOT, f)
				return new_path
			elif self.electionName == 'slo':
				p1 = os.path.split(path)
				fold = os.path.split(p1[0])[1]
				f = p1[1]
				f = os.path.join(fold,f)
				f = os.path.splitext(f)[0] + '.png'
				new_path = os.path.join(SLO_ROOT,f)
				return new_path
			else:
				return path
				
				

		for k, group in groupby(data, itemgetter(*attr_indices)):
			bal_list = []
			for item in group:
				bal_dict = {}
				side_list = []
				side_list.append(checkPath(item[0])) # Should be the voted ballot path
				bal_dict[counter] = side_list
				bal_list.append(bal_dict)
				counter += 1
			groups[counter2] = bal_list
			#uniquekeys.append(k)
			counter2 += 1

		
		return groups

	def processNoGroups(self):
		res = {}
		bals = []
		counter = 0
		for dirname, dirpaths, filenames in os.walk(self.imageSrc):
			# At the top level.
			if dirname == os.path.split(self.imageSrc)[1]:
				continue
			for f in filenames:
				if os.path.splitext(f)[1] not in ['.png','.jpg','.jpeg.']: # incase weird files
					continue
				bal_dict = {}
				bal_dict[counter] = [os.path.join(dirname,f)]
				bals.append(bal_dict)
				counter += 1

		res[0] = bals
		return res

	def processOrange(self):
		#{groupID : list elems}
		# groupID keys of the form (int party_idx, int lang_idx,
		# int page, intbtype)
		# elems = [(str imgpath_i, bool isflipped_i,list bbs_i),...]

		res = {}
		data = pickle.load(open(self.groupingData,'rb'))
		counter = 0
		group_counter = 0
		for k in data.keys():
			bals = []
			for bal in data[k]:
				bal_dict = {}
				if bal[1] == True: #bal is flipped, ignore
					continue
				bal_dict[counter] = [bal[0]]
				bals.append(bal_dict)
				counter += 1
			res[group_counter] = bals
			group_counter += 1

		return res
			
				
				
			
						 


		
		
		
		


