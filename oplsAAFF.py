import numpy as np

class oplsAAFF:
	# unit: kJ/mol, nm
	# nonbondedINFO [oplsNum name mass charge sigma epsilon]
	nonbondedINFO = []
	# bondINFO [name1 name2 b_0 k_b]
	bondINFO = []
	# angleINFO [name1 name2 name3 theta0 cth]
	angleINFO = []
	# dihedralINFO = [name1 name2 name3 name4 k1 k2 k3 k4]
	dihedralINFO = []

	def readNonBonded(self, fileName):
		with open(fileName, 'r') as f:
			lines = f.readlines()
			for i in range(len(lines)):
				if(lines[i].startswith(' opls')):
					info = lines[i].split()
					self.nonbondedINFO.append([ info[0], info[1], float(info[3]), float(info[4]), float(info[6]), float(info[7]) ])
		return

	def readBondAngle(self, fileName):
		label1 = '[ bondtypes ]'
		label2 = '[ angletypes ]'
		label3 = '[ dihedraltypes ]'
		index1 = 0
		index2 = 0
		index3 = 0
		with open(fileName, 'r') as f:
			lines = f.readlines()
			for i in range(len(lines)):
				if(lines[i].startswith(label1)):
					index1 = i
				if(lines[i].startswith(label2)):
					index2 = i
				if(lines[i].startswith(label3)):
					index3 = i
				if(index1 and index2 and index3):
					break
			for i in range(index1, index2):
				info = lines[i].split()
				if(len(info)>=5):
					try:
						self.bondINFO.append([ info[0], info[1], float(info[3]), float(info[4]) ])
					except:
						continue
			for i in range(index2, index3):
				info = lines[i].split()
				if(len(info)>=6):
					try:
						self.angleINFO.append([ info[0], info[1], info[2], float(info[4]), float(info[5]) ])
					except:
						continue
			for i in range(index3, len(lines)):
				info = lines[i].split()
				if(len(info)>=9):
					try:
						self.dihedralINFO.append([ info[0], info[1], info[2], info[3], float(info[5]), float(info[6]), float(info[7]), float(info[8]) ])
					except:
						continue
		return

	def findAtomName(self, atomMark):
		atomLabel = atomMark
		if(type(atomMark) is int):
			atomLabel = 'opls_%03d'%atomMark
		atomName = ''
		for i in range(len(self.nonbondedINFO)):
			info = self.nonbondedINFO[i]
			name0 = info[0]
			name1 = info[1]
			if( atomLabel == name0 ):
				atomName = name1
				break
		return atomName

	def findNonBonded(self, atomMark):
		# give parameters in kcal/mol and A
		kjmol2kcalmol = 1.0/4.184
		nm2A = 10.0

		atomLabel = atomMark
		if(type(atomMark) is int):
			atomLabel = 'opls_%03d'%atomMark

		mass = -1.0
		charge = -1.0
		sigma = -1.0
		epsilon = -1.0

		for i in range(len(self.nonbondedINFO)):
			info = self.nonbondedINFO[i]
			name0 = info[0]
			name1 = info[1]
			if( (atomLabel == name0) or (atomLabel == name1) ):
				mass = info[2]
				charge = info[3]
				sigma = info[4] * nm2A
				epsilon = info[5] * kjmol2kcalmol

		return mass, charge, sigma, epsilon

	def findBond(self, atomName1, atomName2):
		# give parameters in kcal/mol and A
		kjmolnm2kcalmolA = 1.0/4.184 / 100
		nm2A = 10.0

		b0 = -1.0
		kb = -1.0

		for i in range(len(self.bondINFO)):
			info = self.bondINFO[i]
			name1 = info[0]
			name2 = info[1]
			if( (atomName1==name1) and (atomName2==name2) or (atomName1==name2) and (atomName2==name1) ):
				b0 = info[2] * nm2A
				# gromacs to lammps, * 0.5
				kb = info[3] * kjmolnm2kcalmolA * 0.5
		if(b0 < 0):
			print('%s %s bond parameters missing'%(atomName1, atomName2))
		return b0, kb

	def findAngle(self, atomName1, atomName2, atomName3):
		# give parameters in kcal/mol and A
		kjmol2kcalmol = 1.0/4.184

		theta0 = -1.0
		energy = -1.0

		for i in range(len(self.angleINFO)):
			info = self.angleINFO[i]
			name1 = info[0]
			name2 = info[1]
			name3 = info[2]
			if( atomName2 == name2 ):
				if( (atomName1==name1) and (atomName3==name3) or (atomName1==name3) and (atomName3==name1) ):
					theta0 = info[3]
					# gromacs to lammps, * 0.5
					energy = info[4] * kjmol2kcalmol * 0.5
		if(theta0 < 0):
			print('%s %s %s angle parameters missing'%(atomName1, atomName2, atomName3))
		return theta0, energy

	def findDihedral(self, atomName1, atomName2, atomName3, atomName4):
		# give parameters in kcal/mol and A
		kjmol2kcalmol = 1.0/4.184

		k1 = -1.0
		k2 = -1.0
		k3 = -1.0
		k4 = -1.0

		for i in range(len(self.dihedralINFO)):
			info = self.dihedralINFO[i]
			name1 = info[0]
			name2 = info[1]
			name3 = info[2]
			name4 = info[3]
			match1 = (atomName1 == name1) or (name1 == 'X')
			match2 = (atomName2 == name2)
			match3 = (atomName3 == name3)
			match4 = (atomName4 == name4) or (name4 == 'X')
			if(match1 and match2 and match3 and match4):
				k1 = info[4] * kjmol2kcalmol
				k2 = info[5] * kjmol2kcalmol
				k3 = info[6] * kjmol2kcalmol
				k4 = info[7] * kjmol2kcalmol
			else:
				match1 = (atomName4 == name1) or (name1 == 'X')
				match2 = (atomName3 == name2)
				match3 = (atomName2 == name3)
				match4 = (atomName1 == name4) or (name4 == 'X')
				if(match1 and match2 and match3 and match4):
					k1 = info[4] * kjmol2kcalmol
					k2 = info[5] * kjmol2kcalmol
					k3 = info[6] * kjmol2kcalmol
					k4 = info[7] * kjmol2kcalmol
		if(k1 == -1 and k2 == -1 and k3 == -1 and k4 == -1):
			print('%s %s %s %s dihedral parameters missing'%(atomName1, atomName2, atomName3, atomName4))
		return k1, k2, k3, k4
