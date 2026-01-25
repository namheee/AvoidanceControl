import os
import pyboolnet
from pyboolnet.file_exchange import bnet2primes
from pyboolnet.trap_spaces import compute_steady_states
from scipy.spatial.distance import hamming 
import itertools
import numpy as np
import pandas as pd
import time
import copy
import cana
import re


def mFVSs(modeltext):
    modeltext = modeltext.replace("=", "*=")
    net = cana.boolean_network.BooleanNetwork.from_string_boolean(modeltext)

    # Mapping nodes
    mappind_dic = {}
    for node in net.nodes:
        mappind_dic[node.id] = node.name

    # FVSs
    FVS_bruteforce = net.feedback_vertex_set_driver_nodes(graph='structural', method='bruteforce', max_search=10, keep_self_loops=True)  # brutuforce

    FVS_list_list = []
    for FVS in FVS_bruteforce:
        FVS_list = []
        for node in FVS:
            FVS_list.append(mappind_dic[node])
        FVS_list_list.append(FVS_list)

    return FVS_list_list

from sympy.logic.boolalg import to_dnf, simplify_logic
import cana
from cana.datasets.bio import THALIANA
import re
import itertools
import random
import collections


def modeltext_transform(modeltext):
    # Replace the logics with symbols
    modeltext = re.sub(r"\band\b", "&", modeltext)
    modeltext = re.sub(r"\bor\b", "|", modeltext)
    modeltext = re.sub(r"\bnot \b", "~", modeltext)

    # strip modeltext
    modeltext = modeltext.strip()

    # split the modeltext by line
    modeltext_lines = modeltext.splitlines()

    modeltext_extra = ""
    for modeltext_line in modeltext_lines:
        node_list = re.findall(r'\w+', modeltext_line)
        modeltext_line_extra = modeltext_line

        for node in node_list:
            modeltext_line_extra = re.sub(r"\b" + node + r"\b", node + "_", modeltext_line_extra)

        modeltext_extra += modeltext_line_extra + "\n"
    return modeltext_extra

def canalizingC(modeltext, canalizing_node_dic):
    """
    from github.com/chooonlog
    """
    # Strip whitespace
    modeltext = modeltext.strip()

    # Replace the logics with symbols
    modeltext = re.sub(r"\band\b", "&", modeltext)
    modeltext = re.sub(r"\bor\b", "|", modeltext)
    modeltext = re.sub(r"\bnot\b", "~", modeltext)

    # Split text lines
    modeltext_line = modeltext.splitlines()

    # Get all nodes
    all_node_list = []
    for line in modeltext_line:
        all_node_list += re.findall(r'\w+', line)

    # Deduplication
    all_node_list = [x for i, x in enumerate(all_node_list) if i == all_node_list.index(x)]

    # Create a all state vector dictionary with no values
    all_state_vector_dic = {}
    for node in all_node_list:
        all_state_vector_dic[node] = ""

    # Recursive process
    canalized_state_vector_dic = {}
    step_canalized_state_vector_list = []
    process = True
    while process:

        # Update canalizing node list
        if canalizing_node_dic:
            for node in canalizing_node_dic:
                all_state_vector_dic[node] = canalizing_node_dic[node]

            # Append canalized state vector list according to the step
            step_canalized_state_vector_list.append(canalizing_node_dic)

            # Merge two dictionaries
            canalized_state_vector_dic = dict(**canalized_state_vector_dic, **canalizing_node_dic)

            # Get canalizing node list
            canalizing_node_list = list(canalizing_node_dic.keys())

            # Split text lines
            modeltext_line = modeltext.splitlines()

            # Apply the canalization effect
            new_canalizing_node_dic = {}
            new_modeltext = ""
            for line in modeltext_line:
                str1 = line.split("=")
                state_variable = str1[0].strip()
                Boolean_expression = str1[1].strip()
                if not state_variable in canalizing_node_list:
                    for fixedNode in canalizing_node_dic:
                        Boolean_expression = re.sub(r"\b" + fixedNode + r"\b", str(canalizing_node_dic[fixedNode]).lower(), Boolean_expression)
                        
                    Boolean_expression = re.sub('~False', 'True', Boolean_expression)
                    Boolean_expression = re.sub('~True', 'False', Boolean_expression) 
                    simplifiedExpression = to_dnf(Boolean_expression, simplify=True, force=True)
                    if simplifiedExpression in [True, False]:
                        new_canalizing_node_dic[state_variable] = simplifiedExpression
                    else:
                        new_modeltext += state_variable + " = " + str(simplifiedExpression) + "\n"
            modeltext = new_modeltext
            canalizing_node_dic = new_canalizing_node_dic
        else:
            break

    # Remove whitespace
    modeltext = modeltext.strip()

    return modeltext, canalized_state_vector_dic
