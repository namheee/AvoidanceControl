import pickle
from stateAvoid.stateAvoid import saMain

#### INPUT ####
model_dir = '/app/example/' # for .bnet model
sim_dir = '/app/example/' # for attractor simulation results
sim_file = 'n10toyF48_Network_test.pickle' # for attractor simulation restuls

with open(sim_dir + sim_file,'rb') as f:
    normalNetwork = pickle.load(f)
    cancerNetwork = pickle.load(f)
    mutInfo = pickle.load(f)
    nodeInfo = pickle.load(f)
    allAtt = pickle.load(f)
    undesiredInfo = pickle.load(f)

simulation_result = (normalNetwork, cancerNetwork, mutInfo, nodeInfo, allAtt, undesiredInfo)
result = saMain(model_dir, simulation_result)

with open(sim_dir + 'stateAvoidance_result.txt','w') as savefile:
    savefile.write(str(result[-1]))
    savefile.write('\n')    