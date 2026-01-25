import copy
import pickle
from collections import defaultdict, Counter
import time
from stateAvoid.auxiliary_function import *

def saMain(model_file_dir, simulation_result):
    result = []
    normalNetwork, cancerNetwork, mutInfo, nodeInfo, allAtt, undesiredInfo = simulation_result
    nodeList = list(nodeInfo[normalNetwork[0]].keys())    
    # ====================================================================
    # check is_ans?
    start = time.time()
    sim_data = (cancerNetwork, allAtt, undesiredInfo, mutInfo, nodeList)
    is_d, is_cdu, perturbNodeO = is_ans(sim_data, includeMUT = False)
    
    if not np.all([is_d, is_cdu]):
        algtime = time.time()-start
        result.append((normalNetwork[0], algtime, ())) # 해가 없음.
        return result
    # ====================================================================
    
    
    else:
        perturbNode = perturbNodeO
        is_ans_noNew = True
        # ====================================================================
        # main algorithm
        start1 = time.time()
        controlled = []
        possCAND, cand, sol = main2(controlled, model_file_dir, sim_data, perturbNode)
        # ====================================================================
        # compute FVS 
        if len(sol)>0:
            final_sol = sol
        
        else:
            final_sol = []
            cand = possCAND
            while 1:
                if (len(cand) == 0) or (len(perturbNode) == 0): break
                rep_list = []
                for cand1 in cand:
                    fsol1_ = set(itertools.chain(*[x[1] for x in rep_list]))
                    if len(fsol1_)>0:
                        break                        
                    canalized_candNl = list()
                    ctrlList = cand1.split(':') # to list
                    for networkName in cancerNetwork:
                        task_data_ = networkName, model_file_dir, ctrlList, allAtt, undesiredInfo, mutInfo
                        _, _, canalIdx, _, a1, u0 = define_AU(task_data_, ctrlList)
                        canalized_candNl += [nodeList[x] for x in canalIdx]
        
                    candNi = [p for p,q in Counter(canalized_candNl).items() if q == len(cancerNetwork)]
                    perturbNode1 = [px for px in perturbNode if (px not in itertools.chain(*[(x, '~'+x) for x in set(candNi)]))]
        
                    if len(perturbNode1)>0:
                        _, cand1_, sol1_ = main2(ctrlList, model_file_dir, sim_data, perturbNode1)
                        rep_list.append((cand1_, sol1_))
                    else:
                        continue
                    
                final_sol = set(itertools.chain(*[x[1] for x in rep_list]))
                
                if len(final_sol) == 0:
                    cand = set(itertools.chain(*[x[0] for x in rep_list]))
                    perturbNode = perturbNode1
                else:
                    break
                    
        algtime = time.time()-start
        
        if len(final_sol)>0:
            for target in final_sol:
                 result.append((normalNetwork[0], algtime, tuple([target])))
        else:
            result.append((normalNetwork[0], algtime, ()))
        
        return result
