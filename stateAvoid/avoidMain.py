from choonFunction import mFVSs, modeltext_transform, canalizingC
from pyboolnet.state_transition_graphs import primes2stg
from pyboolnet.attractors import compute_attractors_tarjan
from pyboolnet.file_exchange import bnet2primes, primes2bnet
from pyboolnet.prime_implicants import create_constants, create_variables
from pyboolnet.trap_spaces import compute_steady_states
from scipy.spatial.distance import hamming 

from choonFunction import modeltext_transform
from sympy.logic.boolalg import to_dnf, simplify_logic
import random
import pickle
from collections import *
from collections import defaultdict, Counter

import os
import pyboolnet
import itertools
import numpy as np
import pandas as pd
import time
import copy
import cana
import re
import fvs as afvs
import networkx as nx



def computeAtt(primes):
    stg = primes2stg(primes, "synchronous")
    steady, cyclic = compute_attractors_tarjan(stg)
    return steady, cyclic

def cyclic2str(cyclicSet):
    cyclicList = []
    for cyc in cyclicSet:
        catt0 = pd.DataFrame([list(x) for x in cyc])
        catt = catt0.apply(lambda x: len(set(x.values.tolist())) == 1, axis=0).tolist()
        cyclicList.append(''.join([str(x) if i in np.where([x == True for x in catt])[0] else '*' for i,x in enumerate(catt0.iloc[0,:])]))
    return cyclicList

def readBNETfromPrimes(model_file):
    primes = bnet2primes(model_file)
    net = pyboolnet.file_exchange.primes2bnet(primes)
    net = re.sub("&", "and", net)
    net = re.sub("[|]", "or", net)
    net = re.sub("!", "not ",net)
    net = re.sub(",   ", " = ",net)
    net = re.sub("\n\n", "\n",net)

    net_splited = [(x.split(' ')[0],x) for x in net.split('\n')]
    net_ordered = []
    for nodex in primes.keys():
        for eq in net_splited:
            if eq[0] == nodex:
                net_ordered.append(eq[1])

    # print('\n'.join(net_ordered), end="\n\n")
    modeltext = '\n'.join(net_ordered)
    
    return modeltext

def makeDict(x):
    perturb = defaultdict()
    for p in x:
        if '~' not in p: perturb[p] = True
        else: perturb[p[1:]] = False
    return perturb

def pert_dict2List(pert_i, annotType = 'int'):
    if annotType == 'str':      
        return sorted(['~'+x if y == 'False' else x for x,y in pert_i.items()])
    else:
        return sorted(['~'+x if y == False else x for x,y in pert_i.items()])



def _graph_minus(graph, nodeset):
    newgraph = nx.DiGraph()

    remaining_nodes = [n for n in graph.nodes() if n not in nodeset]
    newgraph.add_nodes_from(remaining_nodes)

    for u, v in graph.edges():
        if u not in nodeset and v not in nodeset:
            newgraph.add_edge(u, v)
            
    return newgraph

def _is_acyclic(graph):
    return nx.is_directed_acyclic_graph(graph)


def verify_fvs(G_net, candidate_fvs):
    
    modeltext = G_net.replace("=", "*=")
    net = cana.boolean_network.BooleanNetwork.from_string_boolean(modeltext)

    directed_graph = net.structural_graph()
    graph_copy = directed_graph.copy()

    reduced_graph = _graph_minus(graph_copy, set(candidate_fvs))

    is_valid_fvs = _is_acyclic(reduced_graph)
    
    if is_valid_fvs:
        
        return True
    else:
        try:
            cycle = nx.find_cycle(reduced_graph)
            print(f" cycle?: {cycle}")
        except nx.NetworkXNoCycle:
            pass 
        return False


def mFVSs(modeltext):
    modeltext = modeltext.replace("=", "*=")
    net = cana.boolean_network.BooleanNetwork.from_string_boolean(modeltext)

    # Mapping nodes
    mappind_dic = {}
    for node in net.nodes:
        mappind_dic[node.id] = node.name

    # if len(net.nodes) <= 20: 
    #     # FVS bruteforce
    #     FVS_result = net.feedback_vertex_set_driver_nodes(graph='structural', method='bruteforce', max_search=10, keep_self_loops=True)  # brutuforce
    
    # else:
    #     # FVS GRASP
    #     directed_graph = net.structural_graph()
    #     FVS_result = afvs.fvs_grasp(directed_graph, max_iter=len(net.nodes)*500, keep_self_loops=True)

    
    FVS_result = net.feedback_vertex_set_driver_nodes(graph='structural', method='bruteforce', max_search=10, keep_self_loops=True)  # brutuforce
 
    FVS_list_list = []
    for FVS in FVS_result:
        FVS_list = []
        for node in FVS:
            FVS_list.append(mappind_dic[node])
        FVS_list_list.append(FVS_list)

    return FVS_list_list

# updated: 25.07.21
def define_AU(networkName, networkInfo, ctrlList):
    allAtt, model_file_dir, cancerNetwork, nodeList, mutInfo, undesiredInfo = networkInfo
    # control에 의해 canalized되는 노드를 구한다.
    # canalized되지 않은 남은 노드를 구한다.
    model_file = model_file_dir+networkName+'.bnet'  
    modeltext = readBNETfromPrimes(model_file)
    F_net = modeltext_transform(modeltext)
    _, canalized = cached_canalizingC(F_net, makeDict([x+'_' for x in ctrlList + mutInfo[networkName]]))
    # _, canalized = canalizingC(F_net, makeDict([x+'_' for x in ctrlList + mutInfo[networkName]]))

    canalIdxV = sorted([(nodeList.index(x[:-1]),y==True) for x,y in canalized.items()]) # node index 정렬.
    canalV = ''.join([str(int(x[1])) for x in canalIdxV]) #순서대로 canlV를 만듬. 
    canalIdx = [x[0] for x in canalIdxV] # 순서대로.
    remIdx = list(set(range(len(nodeList)))-set(canalIdx))
    # print(canalIdxV, canalV, canalIdx, remIdx)

    
    # desired attractor
    desiredAttdf = pd.DataFrame([list(x) for x in set(allAtt[networkName])-set(undesiredInfo[networkName]) if '*' not in x])
    desiredAttcanalPart = [''.join(p) for p in desiredAttdf.iloc[:,canalIdx].values]

    if canalV not in desiredAttcanalPart:
        # print('No Desired Attractor') # canalized되는 것이랑 desired canalized영역이 다름.
        return canalIdxV, canalV, canalIdx, remIdx, [], []
    
    else:
        desiredAttdf.index = [''.join(p) for p in desiredAttdf.iloc[:,canalIdx].values]
        if len(remIdx) != 0 :
            a1 = desiredAttdf.loc[canalV,:] # desiredAtt 중 수렴 가능한 집합.
        else:
            a1 = desiredAttdf.loc[canalV,:]
        if sum([x == canalV for x in desiredAttdf.index]) == 1: a1 = pd.DataFrame(a1).T
    
        # undesired attractor
        undf = pd.DataFrame([list(x) for x in undesiredInfo[networkName]])
        undf.index = [''.join(p) for p in undf.iloc[:,canalIdx].values]
        u0 = undf.loc[list(set(undf.index) & set([canalV])),:] # 제어해도 벗어나지 못하는 집합.
        
        return canalIdxV, canalV, canalIdx, remIdx, a1, u0


from sympy.logic.boolalg import to_dnf, simplify_logic
import re
def check_consistency(canalizedValue, G_dict, nodes):
    pattern = r'\bx\d+_\b'
    checked = []
    for node_n in nodes:
        node_function = G_dict[node_n]
        variable = re.findall(pattern, node_function)
        for nodex in variable:
            node_function = re.sub(nodex, str(canalizedValue[nodex]), node_function)
        # node_function2 = re.sub('~', 'not ', node_function)
        # node_function2 = re.sub(' & ', ' and ', node_function2)
        # node_function2 = re.sub(' [:|:] ', ' or ', node_function2)
        node_function2 = re.sub('~False', 'True', node_function)
        node_function2 = re.sub('~True', 'False', node_function2)
        exp_df = to_dnf(node_function2, simplify=True, force=True)
        checked.append(canalizedValue[node_n] == exp_df)
    return np.all(checked)
    
    
def returnstr2list(nodeList, stratt, includeCyclic = True):
    stratt2list = []
    for x,y in zip(nodeList, list(stratt)):
        if y == '1':
            stratt2list.append(x +'_')
        elif y == '0':
            stratt2list.append('~' + x +'_')
        else:
            if includeCyclic == True:
                stratt2list.append(x +'_')
                stratt2list.append('~' + x +'_')
            else:
                continue
    return stratt2list
     
def reverse(canalizedSet):
    reverseList = []
    for x in canalizedSet:
        if '~' in x:
            reverseList.append(x[1:])
        else:
            reverseList.append('~'+x)
    return reverseList




# updated: 25.08.26

def checkcontrolU(undesiredInfo, dsV, sameVidx):
    undesiredI = [list(x) for x in itertools.chain(*undesiredInfo.values())]
    uI = pd.DataFrame(undesiredI)
    unsV = uI.iloc[:,sameVidx].apply(lambda x: ''.join(x), axis=1).to_list()
    uI.index = unsV
    is_NosameU = (len(set([dsV]) & set(unsV)) == 0) # True -> 같을 값을 가지는 undeired가 없다. -> 해가 있다. 
    return is_NosameU
    
    
def checkDU(undesiredInfo, dsV, sameVidx):
    undesiredI = [list(x) for x in itertools.chain(*undesiredInfo.values())]
    uI = pd.DataFrame(undesiredI)
    unsV = uI.iloc[:,sameVidx].apply(lambda x: ''.join(x), axis=1).to_list()
    uI.index = unsV
    is_NosameU = (len(set([dsV]) & set(unsV)) == 0) # True -> 같을 값을 가지는 undeired가 없다. -> 해가 있다. 

    # desired값이 같은 영역 중, sameVidx가 전체가 아니라면 D 중 하나로 수렴시킬 수 있는 다른 해의 가능성이 있을 수 있음.
    if is_NosameU == False: # 0이 아니라는 소리는 겹치는게 있음. 겹치는게 있는데, 
        if np.all(['*' in attx for attx in uI.loc[dsV,:].values]): # 겹치는 attractor가 모두 cyclic이라면 추가 제어가 가능함. 만약 하나라도 point라면 그건 더 이상 벗어날 수가 없음.
            is_NosameU = True
        else: # 아니라면 추가 제어를 통해 undesired를 회피해야하는데 그런 해는 존재할 수 없음.
            is_NosameU = False
    return is_NosameU


def is_ans(cancerNetwork, nodeList, allAtt, undesiredInfo, mutInfo, includeMUT = False):
    # 해가 보장되지 않는 예외 조건 탐색!!
    desireds = []
    for networkName in cancerNetwork:
        desired = [list(x) for x in set(allAtt[networkName])-set(undesiredInfo[networkName]) if '*' not in x]
        desireds.append(desired)
    

    # desired는 각 network별로 비교해서 같은 값을 가지는 노드가 있어야함.
    perturbNodeset = []
    sameVidx_list = []
    any_d_same = []
    for dx in itertools.product(*desireds):
        dsV = ''.join([p for p,q in zip(*dx) if len(set([p,q])) == 1]) # 같은 노드들의 실제 값
        dsVidx = np.where([len(set([p,q])) == 1 for p,q in zip(*dx)])[0] # 같은 노드들의 index
        perturbNodeset += [(nodeList[i], n=='1') for i,n in enumerate(dx[0]) if i in dsVidx]

        any_d_same.append(len(dsVidx)>0) # 노드값이 같은지 확인. 값이 같은게 있어야 수렴할 수 있는 desired가 있는 것임.
        sameVidx_list.append((dsV,dsVidx)) # [('1000110100', array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))]

    if includeMUT: # 돌연변이 노드도 포함해서 해를 찾아봄. 최종 해에서 거름.
        perturbNode = [pert_dict2List({p[0]:p[1]})[0] for p in set(perturbNodeset)]
    else: # 처음부터 돌연변이 노드는 제외하고 찾음.
        mutNode = [x[1:] if '~' in x else x for x in list(itertools.chain(*mutInfo.values()))]
        perturbNode = [pert_dict2List({p[0]:p[1]})[0] for p in set(perturbNodeset) if p[0] not in mutNode]
    
    is_d = np.any(any_d_same) # 하나라도 같으면 된다.
    # is_mut_f = is_mut_fix(mutInfo, desireds, cancerNetwork, perturbNodeset)
    
    # 각 desired별 같은 값을 가지는 노드값이랑 undesired가 같은지를 비교해보았을 때, 
    # 다른 값이 있는 undesired가 없다면 그 undesired는 회피할 수가 없음.
    is_AvoidU = []
    for dsVlist in sameVidx_list:
        dsV, sameVidx = dsVlist # [('1000110100', array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))]
        is_NosameU = checkDU(undesiredInfo, dsV, sameVidx)
        is_AvoidU.append(is_NosameU)
    
    return is_d, np.any(is_AvoidU), perturbNode

# ====
# Main
# =====
# 1. avoid되는 undesired개수가 많으며, desired에 같은 값을 가지는 것으로 바뀔 수 있는 canalizaiton effect를 보이는 타겟 랭크를 매긴다.
def find_target(controlled, networkInfo, perturbNode):
    allAtt, model_file_dir, networkList, nodeList, mutInfo, undesiredInfo = networkInfo
    canalizedList = []
    # cancerCanalNet = defaultdict(dict)
    for networkName in networkList:
        model_file = model_file_dir+networkName+'.bnet'  
        modeltext = readBNETfromPrimes(model_file)
        F_net = modeltext_transform(modeltext)

        # mutNode = [x[1:] if '~' in x else x for x in list(itertools.chain(*mutInfo.values()))]
        # perturbNode = set(['~'+x for x in nodeList] + nodeList) - set(mutNode + ['~'+x for x in mutNode])
        for alln in list(perturbNode):
            # _, canalized = canalizingC(F_net, makeDict([x+'_' for x in controlled + [alln] + mutInfo[networkName]]))
            _, canalized = cached_canalizingC(F_net, makeDict([x+'_' for x in controlled + [alln] + mutInfo[networkName]]))

            # print(alln, canalized)
            # cancerCanalNet[networkName][tuple(set(controlled + [alln]))] = canalized
            canalizedSet = set(pert_dict2List(canalized))
        
            avoid_u = 0
            for stratt in undesiredInfo[networkName]:
                intersect_canalized = set(returnstr2list(nodeList, stratt)) & set(reverse(canalizedSet))
                avoid_u += int(len(intersect_canalized) != 0)
                
    
            include_d = 0
            desiredAtt = [list(x) for x in set(allAtt[networkName])-set(undesiredInfo[networkName]) if '*' not in x]            
            for stratt in desiredAtt:
                same_canalized = (set(returnstr2list(nodeList, stratt, False)) & set(canalizedSet)) == set(canalizedSet)
                include_d += int(same_canalized)
    
            if include_d == 0 : avoid_u = 0 # include_d가 0인 것은 avoid_u가 커도 의미가 없다. 순위가 내려가게 설정.
         
            canalizedList.append([networkName, alln, avoid_u, include_d, len(canalizedSet)])
    rankdf = pd.DataFrame(canalizedList, columns = ['network','ctrl','avoid_u','include_d', 'len_c']).groupby('ctrl').aggregate("sum").sort_values(by='avoid_u', ascending=False)
    rankdf2 = pd.DataFrame(canalizedList, columns = ['network','ctrl','avoid_u','include_d', 'len_c']).groupby('ctrl').aggregate(tuple).sort_values(by='avoid_u', ascending=False)
    rankdf2 = rankdf2.loc[rankdf.index,:]
    return rankdf, rankdf2

def make_unique_combi(candidate):
    candList = sorted(candidate.split(':'))
    nodes = set([x if '~' not in x else x[1:] for x in candList])
    if len(nodes) != len(candList):
        return ''
    else:
        return ':'.join(list(set(candList)))
                

def checkRG(final_cand, model_file_dir, cancerNetwork, mutInfo, reducedThreshold):
    candCheck = defaultdict(int)
    for cand2 in final_cand:    
        ctrlList = cand2.split(':')
        for networkName in cancerNetwork:   
            model_file = model_file_dir+networkName+'.bnet'  
            modeltext = readBNETfromPrimes(model_file)
            F_net = modeltext_transform(modeltext)
            # G_net, _ = canalizingC(F_net, makeDict([x+'_' for x in list(ctrlList) + mutInfo[networkName] ]))
            G_net, _ = cached_canalizingC(F_net, makeDict([x+'_' for x in list(ctrlList) + mutInfo[networkName] ]))
            
            candCheck[cand2] += len(G_net.split('\n'))
    final_cand2 = [p for p,q in candCheck.items() if q < int(reducedThreshold)*len(cancerNetwork)]
    return final_cand2


# updated: 25.10.14

from functools import lru_cache
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial


# Global cache dictionaries
canalization_cache = {}

def cached_canalizingC(network_key, control_dict):
    try:
        if hasattr(control_dict, 'items'):
            control_items = sorted(control_dict.items())
            control_key = '_'.join([f"{k}:{v}" for k, v in control_items])
        else:
            control_key = "empty"
        
        cache_key = f"{network_key}|{control_key}"
        
        if cache_key in canalization_cache:
            return canalization_cache[cache_key]
        
        result = canalizingC(network_key, control_dict)
        canalization_cache[cache_key] = result
        
        return result
        
    except Exception as e:
        print(f"Cache failed, computing directly: {e}")
        return canalizingC(network_key, control_dict)



def compute_single(task_data):
    networkName, model_file_dir, ctrlList, allAtt, undesiredInfo, mutInfo = task_data    
    
    try:
        model_file = model_file_dir + networkName + '.bnet' 
        primes = bnet2primes(model_file)
        nodeList = list(primes.keys())  
        networkInfo = (allAtt, model_file_dir, list(undesiredInfo.keys()), nodeList, mutInfo, undesiredInfo)

        # 가질 수 있는 desired attractor 개수
        _, _, _, _, a1, _ = define_AU(networkName, networkInfo, ctrlList)
       
        modeltext = readBNETfromPrimes(model_file)
        F_net = modeltext_transform(modeltext)
        G_net, G_canalized = cached_canalizingC(F_net, makeDict([x+'_' for x in list(ctrlList) + mutInfo[networkName]]))
        
        if G_net != '':
            G_dict = {x.split(' = ')[0]:x.split(' = ')[1] for x in G_net.split('\n')}    
            G_fvs = mFVSs(G_net)
            
            # FVS 크기 제한 추가 (성능 최적화)
            MAX_FVS_SIZE = 8
            G_fvs = [fvs for fvs in G_fvs[:1] if len(fvs) <= MAX_FVS_SIZE]
            
            if not G_fvs:
                return (networkName, tuple(ctrlList), len(a1), 0)
            
            for fvs_set in G_fvs:
                fvs_combi = [[x, '~'+x] for x in fvs_set]
                fvs_combi = list(itertools.product(*fvs_combi))
                
                fvs_pointAttNUM = []
                for fvs_node in fvs_combi:
                    _, fvs_canalValue = canalizingC(G_net, makeDict(list(fvs_node)))
                    fvs_point = ''.join([str(int(True ==(fvs_canalValue[k]))) for k in [x.split(' = ')[0] for x in G_net.split('\n')]])
                    is_point_stable = check_consistency(fvs_canalValue, G_dict, fvs_set)
                    fvs_pointAttNUM.append(is_point_stable)
        
                return (networkName, tuple(ctrlList), len(a1), np.sum(fvs_pointAttNUM))
        
        else:
            # canalized되는 값이랑 desiredAtt랑 같은지 여부.
            stable_a = []
            canalizedSet2 = pert_dict2List(G_canalized)
            desiredAtt = [list(x) for x in set(allAtt[networkName])-set(undesiredInfo[networkName]) if '*' not in x]            
            for stratt in desiredAtt:
                same_canalized = (set(returnstr2list(nodeList, stratt, False)) & set(canalizedSet2)) == set(canalizedSet2)
                stable_a.append(same_canalized)
            
            return (networkName, tuple(ctrlList), len(a1), np.sum(stable_a))
            
    except Exception as e:
        print(f"Error processing {networkName}, {ctrlList}: {e}")
        return (networkName, tuple(ctrlList), 0, 0)


def process_candidates_parallel(final_cand, model_file_dir, cancerNetwork, allAtt, undesiredInfo, mutInfo, max_workers=None):
    
    if max_workers is None:
        max_workers = min(mp.cpu_count()/2, len(final_cand) * len(cancerNetwork))
    
    # 전체 결과 저장용
    stableN = []
    
    for cand2 in final_cand:
        ctrlList = cand2.split(':')
        stableTmp = []  # 각 candidate별 임시 결과
        
        # 현재 candidate의 모든 네트워크 작업 준비
        tasks = []
        for networkName in cancerNetwork:
            task_data = (networkName, model_file_dir, ctrlList, allAtt, undesiredInfo, mutInfo)
            tasks.append(task_data)
        
        # 병렬 처리 실행
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 작업 제출
            future_to_network = {
                executor.submit(compute_single, task): task[0] 
                for task in tasks
            }
            
            # 결과 수집
            for future in as_completed(future_to_network):
                try:
                    result = future.result()
                    stableN.append(result)
                    stableTmp.append(result)
                except Exception as e:
                    networkName = future_to_network[future]
                    print(f"Network {networkName} generated an exception: {e}")
        
        # 현재 candidate에 대한 조기 종료 로직
        stableTmp_df = pd.DataFrame(stableTmp, columns=['networkName','ctrl','desired_Num','fvs_pointAtt_Num'])
        stableTmp_grouped = stableTmp_df.groupby(['ctrl','networkName']).aggregate(list).map(lambda x: np.max(x))
        resultTableT = stableTmp_grouped.loc[stableTmp_grouped.loc[:,'desired_Num'] == stableTmp_grouped.loc[:,'fvs_pointAtt_Num'],:]
        finalResultT = [p for p,q in Counter([x[0] for x in resultTableT.index]).items() if q == len(cancerNetwork)]
        
        # Target이 1개라도 있으면 멈추기
        if len(finalResultT) > 0:
            break
    
    return stableN




def main2(controlled, networkInfo, perturbNode): 
    allAtt, model_file_dir, cancerNetwork, nodeList, mutInfo, undesiredInfo = networkInfo

    # =================================
    # avoid되는 undesired개수가 많으며, desired에 같은 값을 가지는 것으로 바뀔 수 있는 canalizaiton effect를 보이는 타겟 랭크를 매겨 후보를 결정한다.
    # =================================    
    rankdf, network_rank = find_target(controlled, networkInfo, perturbNode)
    rankdf = rankdf.loc[[np.all(np.array(x)>0) for x in network_rank.loc[:,'include_d']],:] # desired가 모두 있어야 함. 그래야 제어 의미가 있음.
    rankdf = rankdf.loc[[x for x in rankdf.index if x not in controlled],:]

    if rankdf.shape[0]==0: #해가 없음.
        return [], [], []
    else:
        
        # start = time.time() # =============================================================
        poss_cand = [] #가능한 candidate집합을 저장해두기.
        max_i = np.max(rankdf.loc[:,'avoid_u']) # 1등만 고르는 것.
        cand_i = rankdf.loc[[x == max_i for x in rankdf.loc[:,'avoid_u']],:].index.tolist()
        cand = [':'.join(list(set([x] + controlled))) for x in cand_i.copy()]
        cand = list(set(cand))
        # print('combi =', cand)
        poss_cand += [':'.join(list(set([x] + controlled))) for x in rankdf.index]
    
        
        final_cand = []
        canalized_candN = defaultdict(list) # canalized된 노드의 전체 집합.
        while 1:
            if (len(cand) == 0) or (len(poss_cand) == 0): 
                break
            for cand1 in cand:   
                cand_pair = []
                for networkName in cancerNetwork:
                    ctrlList = cand1.split(':') # to list
                    _, _, canalIdx, _, a1, u0 = define_AU(networkName, networkInfo, ctrlList)
                    canalized_candN[cand1] += [nodeList[x] for x in canalIdx]
                    # a1 : 만족하는 desired가 있어야함.
                    # u0 : 남아있는 undesired가 없어야하고,
                    cand_pair.append(np.all([len(a1) != 0, len(u0) == 0]))
                if np.all(cand_pair): # 모든 cancer network에서 만족하면 final_cand로 넘어감.            
                    final_cand.append(cand1)
        
            if len(final_cand)>0: break # candidate을 정함.
            else:
                updated_cand = []
                for cand1 in cand:
                    ctrl1_ = cand1.split(':') # 처음에 정해진 것을 고정.                
    
                    candNi = [p for p,q in Counter(canalized_candN[cand1]).items() if q == len(cancerNetwork)] # 공통인 것만 골라서 제외함.
                    perturbNode1 = [px for px in perturbNode if (px not in itertools.chain(*[(x, '~'+x) for x in set(candNi)]))]
                    
                    if len(perturbNode1) == 0: 
                        break
                        
                    # perturb node 집합만 다시 정의함.
                    # U를 다시 정의한다고 되어있지만, 저장을 너무 많이 해야함.  
                    # 전체 u에서 추가 제어를 하게 되면 남아있는 나머지만 계산됨. 코드에서는 다시 정의하지 않음.
    
                    # 1번, 2번 조건을 만족하지 않는 경우 해를 업데이트 함.
                    rankdf1, network_rank1 = find_target(ctrl1_, networkInfo, perturbNode1)
                    rankdf1 = rankdf1.loc[[np.all(np.array(x)>0) for x in network_rank1.loc[:,'include_d']],:]
                    rankdf1 = rankdf1.loc[[x for x in rankdf1.index if x not in ctrl1_],:]     
        
                    max_j = np.max(rankdf1.loc[:,'avoid_u'])
                    cand_j = rankdf1.loc[[x == max_j for x in rankdf1.loc[:,'avoid_u']],:].index.tolist()
                  
                    if max_j == 0: # undesired를 피할 수 있는 해가 없음.
                        break          
                    else:
                        poss_cand += list(set([make_unique_combi(cand1 + ':' + x) for x in rankdf1.index])) # combinations                    
                        updated_cand += list(set([make_unique_combi(cand1 + ':' + x) for x in cand_j])) # combinations  
        
                # cand update함.
                cand = [x for x in updated_cand if x != '']
                cand = list(set(cand))
                # print('cand', cand)

        
        # final_cand가 결정되는데, 그걸로 network가 너무 안줄어든다면...
        reducedThreshold = 10
        final_cand = checkRG(final_cand, model_file_dir, cancerNetwork, mutInfo, reducedThreshold)
           
        # print(time.time()-start) # =============================================================
        ### 여기가 오래 걸림. ###
        # =================================   
        # 가질 수 있는 attractor개수를 확인함.
        # =================================  

        stableN = process_candidates_parallel(final_cand, model_file_dir, cancerNetwork, allAtt, undesiredInfo, mutInfo)
        
        stableN_df = pd.DataFrame(stableN, columns = ['networkName','ctrl','desired_Num','fvs_pointAtt_Num'])
        stableN_grouped = stableN_df.groupby(['ctrl','networkName']).aggregate(list).map(lambda x: np.max(x))
        resultTable = stableN_grouped.loc[stableN_grouped.loc[:,'desired_Num'] == stableN_grouped.loc[:,'fvs_pointAtt_Num'],:]
        finalResult = [p for p,q in Counter([x[0] for x in resultTable.index]).items() if q == len(cancerNetwork)]

        # possmax = max([len(x.split(':')) for x in poss_cand])
        possmin = min([len(x.split(':')) for x in poss_cand])
        poss_cand2  = [x for x in poss_cand if len(x.split(':')) == possmin]
        
        return poss_cand2, final_cand, finalResult






