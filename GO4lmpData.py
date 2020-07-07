import numpy as np
import networkx as nx
import math
import sys
from oplsAAFF   import oplsAAFF
from scipy.ndimage.interpolation import rotate

# generator of Atoms, Bonds and Angles sections for Lammps data file in full style
# units real
# AtomType C (7) 1:graphene; 2:C-H; 3:C=O; 4:C-O-C; 5:C-OH; 6:[C]-COOH; 7:[C]OOH; 
# AtomType H (4) 11:C-H; 12:-OH; 13:-COOH; 
# AtomType O (6) 21:C=O; 22:C-O-C; 23:C-OH; 24:[O]=C-OH; 25:O=C-[O]-H;

class GO4lmpData:
	oplsaa = oplsAAFF()
	oplsaa.readNonBonded('ffnonbonded.itp')
	oplsaa.readBondAngle('ffbonded.itp')
	type2oplsaa = {1:147,2:145,3:280,4:184,5:220,6:149,7:235,8:146,9:155,10:270,11:281,12:180,13:154,14:236,15:268,\
	               16:421, 17:146, 18:900, 19:909, 20:202, 21:225}
	boxSize = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
	# atoms: n x 7 mat, [id mol type q x y z]
	atoms = np.array([])
	atomTypeNum = 21
	# bonds: n x 4 mat, [bondid type atomid1 atomid2]
	bonds = np.array([])
	bondGraph = nx.Graph()
	bondTypePara = []
	# angles: n x 5 mat, [angleid type atomid1 atomid2 atomid3]
	angles = np.array([])
	bondBondGraph = nx.Graph()
	angleTypePara = []
	dihedrals = np.array([])
	dihedralTypePara = []

	def boxLengthXYZ(self):
		return np.asarray( [ (self.boxSize[i+3] - self.boxSize[i]) for i in range(3) ] )

	def totalMass(self):
		oplsaaTypes = [self.type2oplsaa[t] for t in self.atoms[:,2]]
		massSum = 0
		for t in oplsaaTypes:		
			mass, charge, sigma, epsilon = self.oplsaa.findNonBonded(t)
			massSum += mass
		return massSum

	def maxAtomID(self):
		allID = self.atoms[:,0]
		return np.max(allID)

	def maxAtomMol(self):
		allMol = self.atoms[:,1]
		return np.max(allMol)

	def atomType2Index(self, atomType):
		allType = self.atoms[:,2]
		return np.where(allType==atomType)[0]

	def atomIDCoordNum(self, atomID):
		neighbors = self.bondGraph.neighbors(atomID)
		return len([k for k in neighbors])

	def atomIndexCoordNum(self, index):
		atomID = self.atoms[index, 0]
		return self.atomIDCoordNum(atomID)

	def atomID2Index(self, atomID):
		atomIDs = self.atoms[:, 0]
		atomIndex = np.where(atomIDs==atomID)[0][0]
		return atomIndex

	def atomID2Type(self, atomID):
		atomType = self.atoms[self.atomID2Index(atomID), 2]
		return atomType

	def bondTuple2Type(self, bondPair):
		# bondPair: tuple (atomID1, atomID2)
		pairType1 = self.atomID2Type(bondPair[0])
		pairType2 = self.atomID2Type(bondPair[1])
		sortedPair = sorted([pairType1, pairType2])
		return ( (sortedPair[0] - 1)*self.atomTypeNum + sortedPair[1] )

	def angleTuple2Type(self, angleTuple):
		# angleTuple: tuple (id1,id2,id3)
		centerAtomType = self.atomID2Type(angleTuple[1])
		otherAtomsType = self.bondTuple2Type([angleTuple[0], angleTuple[2]])
		return ( (centerAtomType-1)*self.atomTypeNum*self.atomTypeNum + otherAtomsType )	

	def dihedralTuple2Type(self, dihedralTuple):
		# dihedralTuple: tuple (id1, id2, id3, id4)
		dihedralType = -1
		dihedralSeq = [dihedralTuple[i] for i in range(len(dihedralTuple))]
		type1 = self.atomID2Type(dihedralTuple[0])
		type2 = self.atomID2Type(dihedralTuple[1])
		type3 = self.atomID2Type(dihedralTuple[2])
		type4 = self.atomID2Type(dihedralTuple[3])
		if( type2 <= type3 ):
			dihedralType = type1*self.atomTypeNum**3 + type2*self.atomTypeNum**2 + type3*self.atomTypeNum + type4
		else:
			dihedralSeq = [dihedralTuple[len(dihedralTuple)-1-i] for i in range(len(dihedralTuple))]
			dihedralType = type4*self.atomTypeNum**3 + type3*self.atomTypeNum**2 + type1*self.atomTypeNum + type1
		return dihedralType, dihedralSeq

	def updateBondsFromGraph(self):
		newBonds = []
		bondCount = 0
		for edge in self.bondGraph.edges:
			bondCount += 1
			newBonds.append([ bondCount, self.bondTuple2Type(edge), edge[0], edge[1] ])
		self.bonds = np.array(newBonds)
		# renumber the type
		allTypes = self.bonds[:,1]
		uniqueTypes = np.unique(allTypes)
		old2newType = {}
		for i in range(len(uniqueTypes)):
			old2newType.update( {uniqueTypes[i]:(i+1)} )
		for i in range(len(self.bonds)):
			self.bonds[i,1] = old2newType[self.bonds[i,1]]
		return

	def updateBondBondGraphFromGraph(self):
		CCCAngleGraph = nx.Graph()
		#topo: node2s[i-1] --- node1 --- node2s[i]
		for adjc in self.bondGraph.adjacency():
			node1 = adjc[0]
			node2s = [key for key in adjc[1].keys()]
			node2Num = len(node2s)
			if(node2Num <= 1):
				continue
			else:
				for i in range(node2Num):
					if(i):
						bond1 = tuple(sorted( (node1, node2s[i-1]) ))
						bond2 = tuple(sorted( (node1, node2s[i]) ))
					else:
						bond1 = tuple(sorted( (node1, node2s[-1]) ) )
						bond2 = tuple(sorted( (node1, node2s[i]) ) )
					CCCAngleGraph.add_edge(bond1, bond2)
		self.bondBondGraph = CCCAngleGraph
		return

	def updateAnglesFromGraph(self):
		newAngles = []
		angleCount = 0
		for edge in self.bondBondGraph.edges:
			angleCount += 1
			angleAtoms = [edge[0][0], edge[0][1], edge[1][0], edge[1][1]]
			if(edge[0][0] == edge[1][0]):
				angleAtoms = [edge[0][1], edge[0][0], edge[1][1]]
			elif(edge[0][0] == edge[1][1]):
				angleAtoms = [edge[0][1], edge[0][0], edge[1][0]]
			elif(edge[0][1] == edge[1][0]):
				angleAtoms = [edge[0][0], edge[0][1], edge[1][1]]
			else:
				angleAtoms = [edge[0][0], edge[0][1], edge[1][0]]
			newAngles.append([ angleCount, self.angleTuple2Type(angleAtoms), angleAtoms[0], angleAtoms[1], angleAtoms[2] ])
		self.angles = np.array(newAngles)
		# renumber the type
		allTypes = self.angles[:,1]
		uniqueTypes = np.unique(allTypes)
		old2newType = {}
		for i in range(len(uniqueTypes)):
			old2newType.update( {uniqueTypes[i]:(i+1)} )
		for i in range(len(self.angles)):
			self.angles[i,1] = old2newType[self.angles[i,1]]
		return

	def updateDihedralsFromGraph(self):
		# topo: bond2 --- bond1 --- bond3 
		newDihedrals = []
		dihedralCount = 0
		for adjc in self.bondBondGraph.adjacency():
			bond1 = adjc[0]
			bond2s = [key for key in adjc[1].keys()]
			bond2s = [x for x in set(bond2s).difference(set([bond1]))]
			bond2sNum = len(bond2s)
			if(bond2sNum <= 1):
				continue
			elif(bond2sNum == 2):
				bond2 = bond2s[0]
				bond3 = bond2s[1]
				node12 = [x for x in set(bond1).intersection(set(bond2))][0]
				node13 = [x for x in set(bond1).intersection(set(bond3))][0]
				if( not (node12 == node13) ):
					dihedralCount += 1
					node2 = [x for x in set(bond2).difference(set([node12]))][0]
					node3 = [x for x in set(bond3).difference(set([node13]))][0]
					if( not (node2 == node3) ):
						dihedralType, dihedralSeq = self.dihedralTuple2Type( (node2, node12, node13, node3) )
						newDihedrals.append([dihedralCount, dihedralType, dihedralSeq[0], dihedralSeq[1], dihedralSeq[2], dihedralSeq[3] ])
			else:
				for i in range(bond2sNum):
					if(i):
						bond2 = bond2s[i-1]
						bond3 = bond2s[i]
					else:
						bond2 = bond2s[-1]
						bond3 = bond2s[i]
					node12 = [x for x in set(bond1).intersection(set(bond2))][0]
					node13 = [x for x in set(bond1).intersection(set(bond3))][0]
					if( not (node12 == node13) ):
						dihedralCount += 1
						node2 = [x for x in set(bond2).difference(set([node12]))][0]
						node3 = [x for x in set(bond3).difference(set([node13]))][0]
						if( not (node2 == node3) ):
							dihedralType, dihedralSeq = self.dihedralTuple2Type( (node2, node12, node13, node3) )
							newDihedrals.append([dihedralCount, dihedralType, dihedralSeq[0], dihedralSeq[1], dihedralSeq[2], dihedralSeq[3] ])
		self.dihedrals = np.array(newDihedrals)
		# renumber the type
		allTypes = self.dihedrals[:,1]
		uniqueTypes = np.unique(allTypes)
		old2newType = {}
		for i in range(len(uniqueTypes)):
			old2newType.update( {uniqueTypes[i]:(i+1)} )
		for i in range(len(self.dihedrals)):
			self.dihedrals[i,1] = old2newType[self.dihedrals[i,1]]
		return

	def updateTopoFromGraph(self):
		self.updateBondsFromGraph()
		self.updateBondBondGraphFromGraph()
		self.updateAnglesFromGraph()
		self.updateDihedralsFromGraph()
		return

	def atomTypePara(self):
		allTypes = self.atoms[:,2]
		uniqueTypes = [x+1 for x in range(self.atomTypeNum)]
		#uniqueTypes = np.unique(allTypes)
		uniqueTypePara = []

		for atomType in uniqueTypes:
			oplsaaType = self.type2oplsaa[atomType]
			mass, charge, sigma, epsilon = self.oplsaa.findNonBonded(oplsaaType)
			uniqueTypePara.append([atomType, oplsaaType, mass, charge, sigma, epsilon])

		for i in range(len(allTypes)):
			atomType = allTypes[i]
			oplsaaType = self.type2oplsaa[atomType]
			mass, charge, sigma, epsilon = self.oplsaa.findNonBonded(oplsaaType)
			self.atoms[i,3] = charge

		return uniqueTypePara

	def bondTypePara(self):
		allBondTypes = self.bonds[:,1]
		uniqueTypes, uniqueIndices = np.unique(allBondTypes, return_index=True)
		uniqueBondAtomID1 = self.bonds[uniqueIndices,2]
		uniqueBondAtomID2 = self.bonds[uniqueIndices,3]
		uniqueTypePara = []

		for i in range(len(uniqueTypes)):
			bondType = uniqueTypes[i]
			atomID1 = uniqueBondAtomID1[i]
			atomID2 = uniqueBondAtomID2[i]
			oplsaaType1 = self.type2oplsaa[self.atomID2Type(atomID1)]
			oplsaaType2 = self.type2oplsaa[self.atomID2Type(atomID2)]
			oplsaaName1 = self.oplsaa.findAtomName(oplsaaType1)
			oplsaaName2 = self.oplsaa.findAtomName(oplsaaType2)
			b0, kb = self.oplsaa.findBond(oplsaaName1, oplsaaName2)
			uniqueTypePara.append( [bondType, oplsaaName1, oplsaaName2, b0, kb] )

		return uniqueTypePara

	def angleTypePara(self):
		allAngleTypes = self.angles[:,1]
		uniqueTypes, uniqueIndices = np.unique(allAngleTypes, return_index=True)
		uniqueAngleAtomID1 = self.angles[uniqueIndices,2]
		uniqueAngleAtomID2 = self.angles[uniqueIndices,3]
		uniqueAngleAtomID3 = self.angles[uniqueIndices,4]
		uniqueTypePara = []

		for i in range(len(uniqueTypes)):
			angleType = uniqueTypes[i]
			atomID1 = uniqueAngleAtomID1[i]
			atomID2 = uniqueAngleAtomID2[i]
			atomID3 = uniqueAngleAtomID3[i]
			oplsaaType1 = self.type2oplsaa[self.atomID2Type(atomID1)]
			oplsaaType2 = self.type2oplsaa[self.atomID2Type(atomID2)]
			oplsaaType3 = self.type2oplsaa[self.atomID2Type(atomID3)]
			oplsaaName1 = self.oplsaa.findAtomName(oplsaaType1)
			oplsaaName2 = self.oplsaa.findAtomName(oplsaaType2)
			oplsaaName3 = self.oplsaa.findAtomName(oplsaaType3)
			theta0, ktheta = self.oplsaa.findAngle(oplsaaName1, oplsaaName2, oplsaaName3)
			uniqueTypePara.append( [angleType, oplsaaName1, oplsaaName2, oplsaaName3, theta0, ktheta] )

		return uniqueTypePara

	def dihedralTypePara(self):
		allDihedralTypes = self.dihedrals[:,1]
		uniqueTypes, uniqueIndices = np.unique(allDihedralTypes, return_index=True)
		uniqueDihedralAtomID1 = self.dihedrals[uniqueIndices,2]
		uniqueDihedralAtomID2 = self.dihedrals[uniqueIndices,3]
		uniqueDihedralAtomID3 = self.dihedrals[uniqueIndices,4]
		uniqueDihedralAtomID4 = self.dihedrals[uniqueIndices,5]
		uniqueTypePara = []

		for i in range(len(uniqueTypes)):
			dihedralType = uniqueTypes[i]
			atomID1 = uniqueDihedralAtomID1[i]
			atomID2 = uniqueDihedralAtomID2[i]
			atomID3 = uniqueDihedralAtomID3[i]
			atomID4 = uniqueDihedralAtomID4[i]
			oplsaaType1 = self.type2oplsaa[self.atomID2Type(atomID1)]
			oplsaaType2 = self.type2oplsaa[self.atomID2Type(atomID2)]
			oplsaaType3 = self.type2oplsaa[self.atomID2Type(atomID3)]
			oplsaaType4 = self.type2oplsaa[self.atomID2Type(atomID4)]
			oplsaaName1 = self.oplsaa.findAtomName(oplsaaType1)
			oplsaaName2 = self.oplsaa.findAtomName(oplsaaType2)
			oplsaaName3 = self.oplsaa.findAtomName(oplsaaType3)
			oplsaaName4 = self.oplsaa.findAtomName(oplsaaType4)
			k1, k2, k3, k4 = self.oplsaa.findDihedral(oplsaaName1, oplsaaName2, oplsaaName3, oplsaaName4)
			uniqueTypePara.append( [dihedralType, oplsaaName1, oplsaaName2, oplsaaName3, oplsaaName4, k1, k2, k3, k4] )

		return uniqueTypePara

	def writeLammpsData(self, outName):
		with open(outName, 'w') as output:
			output.write('LAMMPS Data file\n\n')
			output.write('%d atoms\n' %( int(self.maxAtomID() ) ) )
			output.write('%d bonds\n' %( len(self.bonds) ) )
			output.write('%d angles\n'%( len(self.angles) ) )
			output.write('%d dihedrals\n\n'%( len(self.dihedrals) ) )
			output.write('%d atom types\n' %( self.atomTypeNum ))
			output.write('%d bond types\n' %( np.max(self.bonds[:,1]) ) )
			output.write('%d angle types\n'%( np.max(self.angles[:,1]) ))
			output.write('%d dihedral types\n\n'%( np.max(self.dihedrals[:,1]) ))
			output.write('%.3f %.3f xlo xhi\n'%( self.boxSize[0], self.boxSize[1] ))
			output.write('%.3f %.3f ylo yhi\n'%( self.boxSize[2], self.boxSize[3] ))
			output.write('%.3f %.3f zlo zhi\n\n'%( self.boxSize[4], self.boxSize[5] ))

			atomTypePara = self.atomTypePara()
			output.write('Masses\n\n')
			currentTypeNum = 0
			for para in atomTypePara:
				currentTypeNum += 1
				line2Write = '%d %.5f # %s \n'%(currentTypeNum, para[2], self.oplsaa.findAtomName(para[1]))
				output.write(line2Write)

			output.write('\nPair Coeffs # lj/cut/coul/long\n\n')			
			for para in atomTypePara:
				# para: [type, oplsaaType, mass, charge, sigma, epsilon]
				line2Write = '%d %f %f # oplsaa_%s %s\n'%(int(para[0]), para[5], para[4], para[1], self.oplsaa.findAtomName(para[1]))
				output.write(line2Write)

			output.write('\nBond Coeffs # harmonic\n\n')
			bondTypePara = self.bondTypePara()
			for para in bondTypePara:
				# para: [type, oplsaaName1, oplsaaName2, b0, kb]
				line2Write = '%d %f %f # %s %s\n'%(int(para[0]), para[4], para[3], para[1], para[2])
				output.write(line2Write)

			output.write('\nAngle Coeffs # harmonic\n\n')
			angleTypePara = self.angleTypePara()
			for para in angleTypePara:
				# para: [type, oplsaaName1, oplsaaName2, oplsaaName3, theta0, ktheta]
				line2Write = '%d %f %f # %s %s %s\n'%(int(para[0]), para[5], para[4], para[1], para[2], para[3])
				output.write(line2Write)

			output.write('\nDihedral Coeffs # harmonic\n\n')
			dihedralTypePara = self.dihedralTypePara()
			for para in dihedralTypePara:
				# para: [type, oplsaaName1, oplsaaName2, oplsaaName3, oplsaaName4 k1, k2, k3, k4]
				line2Write = '%d %f %f %f %f # %s %s %s %s\n'%(int(para[0]), para[5], para[6], para[7], para[8], para[1], para[2], para[3], para[4])
				output.write(line2Write)

			output.write('\nAtoms # full\n\n')
			for item in self.atoms:
				line2Write = "%.0f %.0f %.0f %f %f %f %f\n"%(item[0],item[1],item[2],item[3],item[4],item[5],item[6])
				output.write(line2Write)
			output.write('\n Bonds\n\n')
			for item in self.bonds:
				line2Write = "%d %d %d %d\n"%(item[0],item[1],item[2],item[3])
				output.write(line2Write)
			output.write('\n Angles \n\n')
			for item in self.angles:
				line2Write = "%d %d %d %d %d\n"%(item[0],item[1],item[2],item[3],item[4])
				output.write(line2Write)
			output.write('\n Dihedrals \n\n')
			for item in self.dihedrals:
				line2Write = "%d %d %d %d %d %d\n"%(item[0],item[1],item[2],item[3],item[4],item[5])
				output.write(line2Write)
			return

	def writeAtoms(self, outName):
		with open(outName, 'w') as output:
			for item in self.atoms:
				line2Write = "%.0f %.0f %.0f %f %f %f %f\n"%(item[0],item[1],item[2],item[3],item[4],item[5],item[6])
				output.write(line2Write)

	def writeBonds(self, outName):
		with open(outName, 'w') as output:
			for item in self.bonds:
				line2Write = "%d %d %d %d\n"%(item[0],item[1],item[2],item[3])
				output.write(line2Write)

	def writeAngles(self, outName):
		with open(outName, 'w') as output:
			for item in self.angles:
				line2Write = "%d %d %d %d %d\n"%(item[0],item[1],item[2],item[3],item[4])
				output.write(line2Write)

	def readGraphene(self, fileName):
		with open(fileName, 'r') as f:
			contents = f.readlines()

		label0 = 'Masses\n'
		label1 = 'Atoms # full\n'
		label2 = 'Velocities\n'

		index0 = contents.index(label0) - 1
		boxInfo = contents[index0-3:index0]
		for i in range(len(boxInfo)):
			self.boxSize[i*2] = boxInfo[i].split()[0]
			self.boxSize[i*2+1] = boxInfo[i].split()[1]

		index1 = contents.index(label1) + 2
		index2 = contents.index(label2) - 1
		contents = contents[index1:index2]
		atomData = [ [0.0]*7 for i in range(len(contents)) ]
		for i in range(len(contents)):
			currentLine = contents[i].split()
			for j in range(7):
				atomData[i][j] = float(currentLine[j])
		atomData = sorted(atomData, key=lambda x: x[0])
		self.atoms = np.array(atomData)
		self.atoms[:,1] = 1
		return atomData

	def linkGraphene(self, cutoff=1.5):
		# link all type-1 atoms within cutoff
		XYZid = np.array(self.atoms[:,[4,5,6,0]])
		boxSize = [ self.boxSize[1]-self.boxSize[0], self.boxSize[3]-self.boxSize[2], self.boxSize[5]-self.boxSize[4] ]
		# edges of the graph are bonds of the graphene
		CCBondGraph = nx.Graph()
		for i in range(len(self.atoms)):
			iXYZ = XYZid[i,0:3]
			iID = XYZid[i,3]
			for j in range(3):
				delta = np.abs(XYZid[:,j] - iXYZ[j])
				delta[delta>(0.5*boxSize[j])] -= boxSize[j]
				if(j==0):
					diffSqrTotalSqrt = delta**2
				else:
					diffSqrTotalSqrt+= delta**2
				allDist = np.sqrt(diffSqrTotalSqrt)
			jIDs = XYZid[allDist<cutoff,3]
			for iConnect in jIDs:
				if(iID == iConnect):
					continue
				CCBondGraph.add_edge(iID, iConnect)
		self.bondGraph = CCBondGraph
		#self.updateBondsFromGraph()
		#self.updateBondBondGraphFromGraph()
		#self.updateAnglesFromGraph()
		self.updateTopoFromGraph()
		return

	def addH(self, cID):
		index0 = self.atomID2Index(cID)
		self.atoms[index0, 2] = 2
		xyz0 = self.atoms[index0, 4:7]

		neighbors = [neigh for neigh in self.bondGraph.neighbors(cID)]
		xyz1 = self.atoms[self.atomID2Index(neighbors[0]),4:7]
		xyz2 = self.atoms[self.atomID2Index(neighbors[1]),4:7]

		CHVec = ((xyz0 - xyz1) + (xyz0 - xyz2)) / 2
		CHVecLength = math.sqrt(np.sum(CHVec**2))
		CHBondLength = 1.08
		CHVec = CHVec * CHBondLength / CHVecLength

		HID = self.maxAtomID() + 1
		HType = 8
		HMol = self.maxAtomMol() + 1
		Hq = 0.0
		Hx = xyz0[0] + CHVec[0]
		Hy = xyz0[1] + CHVec[1]
		Hz = xyz0[2] + CHVec[2]
		self.atoms = np.append(self.atoms,[[HID, HMol, HType, Hq, Hx, Hy, Hz]],axis=0)
		self.bondGraph.add_edge(cID,HID)
		#self.updateTopoFromGraph()
		return

	def addCarbonyl(self, cID):
		index0 = self.atomID2Index(cID)
		self.atoms[index0, 2] = 3
		xyz0 = self.atoms[index0, 4:7]
		neighbors = [neigh for neigh in self.bondGraph.neighbors(cID)]
		xyz1 = self.atoms[self.atomID2Index(neighbors[0]),4:7]
		xyz2 = self.atoms[self.atomID2Index(neighbors[1]),4:7]

		COVec = ((xyz0 - xyz1) + (xyz0 - xyz2)) / 2
		COVecLength = math.sqrt(np.sum(COVec**2))
		COBondLength = 1.20
		COVec = COVec * COBondLength / COVecLength

		OID = self.maxAtomID() + 1
		OType = 11
		OMol = self.maxAtomMol() + 1
		Oq = 0.0
		Ox = xyz0[0] + COVec[0]
		Oy = xyz0[1] + COVec[1]
		Oz = xyz0[2] + COVec[2]
		self.atoms = np.append(self.atoms,[[OID, OMol, OType, Oq, Ox, Oy, Oz]],axis=0)
		self.bondGraph.add_edge(cID,OID)
		#self.updateTopoFromGraph()
		return

	def addEpoxy(self, cID1, cID2, upOrDown=1):
		index1 = self.atomID2Index(cID1)
		index2 = self.atomID2Index(cID2)
		self.atoms[index1, 2] = 4
		self.atoms[index2, 2] = 4
		xyz1 = self.atoms[index1, 4:7]
		xyz2 = self.atoms[index2 ,4:7]
		OID = self.maxAtomID() + 1
		OType = 12
		OMol = self.maxAtomMol() + 1
		Oq = 0.0
		Ox = (xyz1[0] + xyz2[0])/2
		Oy = (xyz1[1] + xyz2[1])/2
		Oz = (xyz1[2] + xyz2[2])/2 + upOrDown*1.2
		self.atoms = np.append(self.atoms,[[OID, OMol, OType, Oq, Ox, Oy, Oz]],axis=0)
		self.bondGraph.add_edge(cID1,OID)
		self.bondGraph.add_edge(cID2,OID)
		#self.updateTopoFromGraph()
		return

	def addOH(self, cID, upOrDown=1):
		index0 = self.atomID2Index(cID)
		self.atoms[index0, 2] = 5
		xyz = self.atoms[index0, 4:7]
		OID = self.maxAtomID() + 1
		OType = 13
		OMol = self.maxAtomMol() + 1
		Oq = 0.0
		Ox = xyz[0]
		Oy = xyz[1]
		Oz = xyz[2] + upOrDown*1.46
		self.atoms = np.append(self.atoms,[[OID, OMol, OType, Oq, Ox, Oy, Oz]],axis=0)
		HID = self.maxAtomID() + 1
		HType = 9
		HMol = OMol
		Hq = 0.0
		Hx = xyz[0]
		Hy = xyz[1]
		Hz = xyz[2] + upOrDown*(1.46 + 0.967)
		self.atoms = np.append(self.atoms,[[HID, HMol, HType, Hq, Hx, Hy, Hz]],axis=0)
		self.bondGraph.add_edge(cID,OID)
		self.bondGraph.add_edge(OID,HID)
		#self.updateTopoFromGraph()
		return

	def addCOOH(self, cID, upOrDown=1):
		COOHCoords = np.array([[0, 0, 1.56], [-0.99, 0.09, 2.23], [1.23, -0.05, 2.07], [1.20, 0.01, 3.04]])
		index0 = self.atomID2Index(cID)
		self.atoms[index0, 2] = 6
		xyz = self.atoms[index0, 4:7]
		COOHCoords = COOHCoords*upOrDown + xyz
		newCID = self.maxAtomID() + 1
		OID1 = newCID + 1
		OID2 = newCID + 2
		HID  = newCID + 3
		CType = 7
		OType1 = 14
		OType2 = 15
		HType = 10
		anyQ = 0.0
		anyMol = self.maxAtomMol() + 1
		self.atoms = np.append(self.atoms,[[newCID, anyMol, CType, anyQ, COOHCoords[0][0], COOHCoords[0][1], COOHCoords[0][2]]],axis=0)
		self.atoms = np.append(self.atoms,[[OID1, anyMol, OType1, anyQ, COOHCoords[1][0], COOHCoords[1][1], COOHCoords[1][2]]],axis=0)
		self.atoms = np.append(self.atoms,[[OID2, anyMol, OType2, anyQ, COOHCoords[2][0], COOHCoords[2][1], COOHCoords[2][2]]],axis=0)
		self.atoms = np.append(self.atoms,[[HID, anyMol, HType, anyQ, COOHCoords[3][0], COOHCoords[3][1], COOHCoords[3][2]]],axis=0)
		self.bondGraph.add_edge(cID,newCID)
		self.bondGraph.add_edge(newCID,OID1)
		self.bondGraph.add_edge(newCID,OID2)
		self.bondGraph.add_edge(OID2,HID)
		#self.updateTopoFromGraph()
		return

	def addTU(self, location):
		TUAtomID = []
		TUCoords = np.array([ -2.291,   1.898,   0.644, \
		                      -2.168,   0.339,   0.070, \
		                      -3.257,  -0.424,  -0.207, \
		                      -0.978,  -0.244,  -0.230, \
		                      -3.179,  -1.437,  -0.272, \
		                      -4.137,  -0.070,   0.158, \
		                      -0.907,  -1.259,  -0.286, \
		                      -0.158,   0.237,   0.124]).reshape(8,3)
		TUAtomType = [20, 16, 18, 18, 19, 19, 19, 19]
		TUAtomMol = []
		TUCoords += location

		return

	def addTUChain(self, cID1, cID2, xyzImg=[0, 0, 0]):
		self.atoms[self.atomID2Index(cID1),2] = 21
		self.atoms[self.atomID2Index(cID2),2] = 21
		TUAtomNum = 6
		TUAtomID = self.maxAtomID() + 1 + np.array([x for x in range(TUAtomNum)])
		TUAtomMol = self.maxAtomMol() + 1
		TUAtomType = [20, 16, 18, 18, 19, 19]
		TUAtomCharge = []
		for atomType in TUAtomType:
			mass, charge, sigma, epsilon = self.oplsaa.findNonBonded(self.type2oplsaa[atomType])
			TUAtomCharge.append(charge)
		TUCoords = np.array([ -2.291,   1.898,   0.644, \
		                      -2.168,   0.339,   0.070, \
		                      -3.257,  -0.424,  -0.207, \
		                      -0.978,  -0.244,  -0.230, \
		                      -3.179,  -1.437,  -0.272, \
		                      -0.907,  -1.259,  -0.286]).reshape(TUAtomNum,3)
		N1Coord = TUCoords[2,:]
		N2Coord = TUCoords[3,:]
		# recenter the coords
		N21Center = (N1Coord + N2Coord)/2
		TUCoords -= N21Center
		C1Coord = np.array(self.atoms[self.atomID2Index(cID1),4:7])
		C2Coord = np.array(self.atoms[self.atomID2Index(cID2),4:7])
		PBCFix = np.asarray(xyzImg)*self.boxLengthXYZ()
		C21Diff = C2Coord - C1Coord
		for i in range(3):
			if(C21Diff[i] > 0):
				C1Coord[i] += PBCFix[i]
			if(C21Diff[i] < 0):
				C2Coord[i] += PBCFix[i]

		# rotate the molecule
		N21Vec = N2Coord - N1Coord
		C21Vec = C2Coord - C1Coord
		CNCross = np.cross(C21Vec, N21Vec)
		CNAngle = math.acos(np.dot(N21Vec, C21Vec) / (np.linalg.norm(N21Vec) * np.linalg.norm(C21Vec) ))
		TUCoords = np.dot(TUCoords, self.rotation_matrix(CNCross, CNAngle))
		# recenter the molecule
		C21Center = ( C1Coord + C2Coord ) / 2
		TUCoords += C21Center
		# bonds in TU molecule
		TUBonds = [(0,1),(1,2),(1,3),(2,4),(3,5)]
		# add atoms to system	
		for i in range(TUAtomNum):
			self.atoms = np.append(self.atoms,[[TUAtomID[i], TUAtomMol, TUAtomType[i], TUAtomCharge[i],\
			                                    TUCoords[i][0], TUCoords[i][1], TUCoords[i][2]]],axis=0)
		# add bonds to system
		for bond in TUBonds:
			self.bondGraph.add_edge(TUAtomID[bond[0]], TUAtomID[bond[1]])
		self.bondGraph.add_edge(cID1, TUAtomID[2])
		self.bondGraph.add_edge(cID2, TUAtomID[3])
		#self.updateTopoFromGraph()
		return

	def rotation_matrix(self, axis, theta):
	    """
	    Return the rotation matrix associated with counterclockwise rotation about
	    the given axis by theta radians.
	    """
	    axis = np.asarray(axis)
	    axis = axis / math.sqrt(np.dot(axis, axis))
	    a = math.cos(theta / 2.0)
	    b, c, d = -axis * math.sin(theta / 2.0)
	    aa, bb, cc, dd = a * a, b * b, c * c, d * d
	    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
	    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
	                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
	                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


	def addMPDChain(self, cID1, cID2, xyzImg=[0, 0, 0]):
		self.atoms[self.atomID2Index(cID1),2] = 21
		self.atoms[self.atomID2Index(cID2),2] = 21
		MPDAtomNum = 14
		MPDAtomID = self.maxAtomID() + 1 + np.array([x for x in range(MPDAtomNum)])
		MPDAtomMol = self.maxAtomMol() + 1
		MPDAtomType = [16, 16, 16, 16, 16, 16, 18, 18, 17, 17, 17, 17, 19, 19]
		MPDAtomCharge = []
		for atomType in MPDAtomType:
			mass, charge, sigma, epsilon = self.oplsaa.findNonBonded(self.type2oplsaa[atomType])
			MPDAtomCharge.append(charge)
		MPDCoords = np.array([-2.000, -0.030, -0.016, \
							  -1.299, -1.235, -0.083, \
							   0.091, -1.215, -0.041, \
							   0.789, -0.016,  0.061, \
							   0.080,  1.183,  0.123, \
							  -1.302,  1.159,  0.086, \
							  -1.975, -2.431, -0.244, \
							   2.169, -0.019,  0.154, \
							  -3.096, -0.037, -0.055, \
							   0.647, -2.161, -0.091, \
							   0.624,  2.131,  0.206, \
							  -1.857,  2.102,  0.134, \
							  -1.473, -3.258,  0.089, \
							   2.626, -0.832, -0.268]).reshape(MPDAtomNum,3)
		N1Coord = MPDCoords[6,:]
		N2Coord = MPDCoords[7,:]
		# recenter the coords
		N21Center = (N1Coord + N2Coord)/2
		MPDCoords -= N21Center
		C1Coord = np.array(self.atoms[self.atomID2Index(cID1),4:7])
		C2Coord = np.array(self.atoms[self.atomID2Index(cID2),4:7])
		PBCFix = np.asarray(xyzImg)*self.boxLengthXYZ()
		C21Diff = C2Coord - C1Coord
		for i in range(3):
			if(C21Diff[i] > 0):
				C1Coord[i] += PBCFix[i]
			if(C21Diff[i] < 0):
				C2Coord[i] += PBCFix[i]
		# rotate the molecule
		N21Vec = N2Coord - N1Coord
		C21Vec = C2Coord - C1Coord
		CNCross = np.cross(C21Vec, N21Vec)
		CNAngle = math.acos(np.dot(N21Vec, C21Vec) / (np.linalg.norm(N21Vec) * np.linalg.norm(C21Vec) ))
		MPDCoords = np.dot(MPDCoords, self.rotation_matrix(CNCross, CNAngle))
		# recenter the molecule
		C21Center = ( C1Coord + C2Coord ) / 2
		MPDCoords += C21Center
		# bonds in MPD molecule
		MPDBonds = [(1,9),(2,7),(3,10),(4,8),(5,11),(6,12),\
		            (1,2),(2,3),(3,4),(4,5),(5,6),(6,1),\
		            (7,13),(8,14)]
		# add atoms to system	
		for i in range(MPDAtomNum):
			self.atoms = np.append(self.atoms,[[MPDAtomID[i], MPDAtomMol, MPDAtomType[i], MPDAtomCharge[i],\
			                                    MPDCoords[i][0], MPDCoords[i][1], MPDCoords[i][2]]],axis=0)
		# add bonds to system
		for bond in MPDBonds:
			self.bondGraph.add_edge(MPDAtomID[bond[0]-1], MPDAtomID[bond[1]-1])
		self.bondGraph.add_edge(cID1, MPDAtomID[6])
		self.bondGraph.add_edge(cID2, MPDAtomID[7])
		#self.updateTopoFromGraph()
		return