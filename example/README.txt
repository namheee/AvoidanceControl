
_test.pickle file encapsulates the network configurations and identified attractors.

File Name: [network_name]_test.pickle

Objects (in order):
normalNetwork (list): Name of the normal network.
cancerNetwork (list): Names of the abnormal networks.
mutInfo (dict): Mapping of network names to their specific mutation nodes (e.g., {'Network1': ['x01', 'x05']}).
nodeInfo (dict): Dictionary of node names and their corresponding indices for each network.
allAtt (dict): Identified attractors for each network stored as a tuple of strings. (e.g., '101010','1*1*11')
undesiredInfo (dict): Set of attractors present in abnormal networks but absent in the normal network.