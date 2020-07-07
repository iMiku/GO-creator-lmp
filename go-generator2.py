from GO4lmpData import GO4lmpData
from oplsAAFF   import oplsAAFF
import random
import numpy as np
import networkx as nx

go = GO4lmpData()
ff = oplsAAFF()

go.readGraphene('go.data')
go.linkGraphene()

numCarbonPerLayer = 800
numTU2Add = 160

#go.addH(121)
#go.addTUChain(8,808, [0,0,1])
#go.addTUChain(87,887, [0,0,0])
#go.addMPDChain(72,872,[0,0,1])
#go.addMPDChain(150,950,[0,0,0])

grapheneBondGraph = go.bondBondGraph.copy()
bonds2Sub = []
for bond in grapheneBondGraph.nodes():
	numNeigh = len([n for n in grapheneBondGraph.neighbors(bond)])
	if( numNeigh == 4 and bond[0]<=numCarbonPerLayer):
		bonds2Sub.append(bond)

subBondGraph = grapheneBondGraph.subgraph(bonds2Sub).copy()
selectedBonds = []
for i in range(numTU2Add):
	chosenBondIndex = int( random.random()*len(subBondGraph.nodes()) )
	chosenBond = list(subBondGraph.nodes())[chosenBondIndex]
	selectedBonds.append( chosenBond )
	nodes2Del = list(subBondGraph.neighbors(chosenBond)).copy()
	subBondGraph.remove_nodes_from( nodes2Del )
	subBondGraph.remove_node(chosenBond)

linkCount = 0
for pair in selectedBonds:
	linkCount += 1
	if(linkCount%2):
		go.addTUChain( pair[0], pair[0] + 800, [0, 0, 1] )
	else:
		go.addTUChain( pair[0], pair[0] + 800, [0, 0, 0] )

	go.addOH(pair[1])
	go.addOH(pair[1] + 800)

for i in range(1,int(go.maxAtomID())):
	if( go.atomID2Type(i)==1 and go.atomIDCoordNum(i)==2):
		go.addCarbonyl(i)

print(go.totalMass())
go.updateTopoFromGraph()
go.writeLammpsData('out2.data')
