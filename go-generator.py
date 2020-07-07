from GO4lmpData import GO4lmpData
from oplsAAFF   import oplsAAFF

go = GO4lmpData()
ff = oplsAAFF()

go.readGraphene('go.data')
go.linkGraphene()

#go.addH(121)
go.addTUChain(8,808, [0,0,1])
go.addTUChain(87,887, [0,0,0])
go.addMPDChain(72,872,[0,0,1])
go.addMPDChain(150,950,[0,0,0])

for i in range(1,int(go.maxAtomID())):
	if( go.atomID2Type(i)==1 and go.atomIDCoordNum(i)==2):
		go.addCarbonyl(i)

#for item in go.atomTypePara():
#	print(item)
#
#for item in go.bondTypePara():
#	print(item)
#
#for item in go.angleTypePara():
#	print(item)

#for item in go.dihedralTypePara():
#	print(item)

print(go.totalMass())
go.updateTopoFromGraph()
go.writeLammpsData('out.data')
