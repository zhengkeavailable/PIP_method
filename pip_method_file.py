# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 13:18:50 2024

@author: zhengke
"""
import triage_file
import calculate_propensity_score as cps
import numpy as np
import gurobipy as gp
import pandas as pd
import time
import csv

# from gurobipy import GRB

df_CTS = pd.read_csv("inspire_CTS_0_120_icuhalfday.csv")

# N = 429  # Sample size
N = 858  # Sample size
p = 37  # Dimension of features
D = 2  # Depth
B_lb = -1e3  # Lower bound of \Phi_st
B_ub = 1e5  # Upper bound of \Phi_st
M_ub = 1e5  # Upper bound: big M
epsilon = 0.01
num_j = 4  # Number of treatments
'''
C_ub = max(df_CTS.iloc[:, 0].values)+1
x = df_CTS.iloc[0:429, 3:40].values  
Trt = df_CTS.iloc[0:429,40].values
y = df_CTS.iloc[0:429, 0].values
y = C_ub - y 
'''
x = df_CTS.iloc[:, 3:40].values
Trt = df_CTS.iloc[:, 40].values
y = df_CTS.iloc[:, 0].values
C_ub = max(df_CTS.iloc[:, 0].values) + 1
y = C_ub - y

# 0. Initialization
a_start = {}
b_start = {}
z_start = {}

# Trt_is[s][j] = 1 if j_s=j
Trt_is = {}
for s in range(N):
    Trt_is[s] = {}
    for j in range(num_j):
        if Trt[s] == j + 1:
            Trt_is[s][j] = 1
        else:
            Trt_is[s][j] = 0

# Calculate propensity score
# estimator='IPW'
estimator = 'DR'
if estimator == 'IPW':
    propensity_model = cps.train_propensity_model(x, Trt)
    treatment_mean = None
elif estimator == 'DR':
    propensity_model = pd.read_csv('propensity_score.csv', header=None).values
    treatment_mean = pd.read_csv('calculate_mean.csv', header=None).values
    # propensity_model=pd.read_csv('propensity_score.csv',header=None).iloc[0:429,:].values
    # treatment_mean=pd.read_csv('calculate_mean.csv',header=None).iloc[0:429,:].values

# 0.Initialization a,b
for k in range(2 ** D - 1):
    a_start[k] = {}
    for i in range(p):
        a_start[k][i] = 0
    b_start[k] = 0

# 0. Initialization from policy tree
'''
if estimator == 'IPW':
    a_start[0][11] = 1
    b_start[0] = 22
elif estimator == 'DR':
    a_start[0][10] = 1
    b_start[0] = 58
'''
# depth = 2
if estimator=='IPW':
    a_start[0][7]=1
    a_start[1][11]=1
    a_start[2][2]=1 
    b_start[0]=98
    b_start[1]=20
    b_start[2]=60
elif estimator=='DR':
    a_start[0][6]=1
    a_start[1][1]=1
    a_start[2][9]=1 
    b_start[0]=82
    b_start[1]=158
    b_start[2]=4.1
'''
if estimator=='IPW':
    a_start[0][18]=1
    a_start[1][2]=1
    a_start[2][16]=1 
    a_start[3][18]=1
    a_start[4][11]=1
    a_start[5][9]=1 
    a_start[6][1]=1
    
    b_start[0]=13
    b_start[1]=62
    b_start[2]=4.5
    b_start[3]=8.0
    b_start[4]=20.0
    b_start[5]=4.0
    b_start[6]=168
elif estimator=='DR':
    a_start[0][2]=1
    a_start[1][13]=1
    a_start[2][0]=1 
    a_start[3][15]=1
    a_start[4][9]=1
    a_start[5][11]=1 
    a_start[6][4]=1
    
    b_start[0]=68
    b_start[1]=96
    b_start[2]=65
    b_start[3]=31.7
    b_start[4]=4.0
    b_start[5]=20.0
    b_start[6]=2.0
'''
A_L = {}
A_R = {}
initial_value_positive = []
initial_value_negative = []
constraint_value = {}
odd_index = 0  # append constraint_value = 0
# Select epsilon_1 and epsilon_2
for t in range(2 ** D):
    constraint_value[t] = {}
    A_L[t] = []
    A_R[t] = []
    real_t = t + 2 ** D - 1
    current_node = real_t
    while current_node != 0:
        parent_node = (current_node - 1) // 2
        if current_node == 2 * parent_node + 1:
            A_L[t].append(parent_node)
        else:
            A_R[t].append(parent_node)
        current_node = parent_node
    for s in range(N):
        constraint_value[t][s] = B_ub
        for k in A_R[t]:
            if sum(a_start[k][i] * x[s][i] for i in range(p)) - b_start[k] < constraint_value[t][s]:
                constraint_value[t][s] = sum(a_start[k][i] * x[s][i] for i in range(p)) - b_start[k]
        for k in A_L[t]:
            if sum(-a_start[k][i] * x[s][i] for i in range(p)) + b_start[k] - epsilon < constraint_value[t][s]:
                constraint_value[t][s] = sum(-a_start[k][i] * x[s][i] for i in range(p)) + b_start[k] - epsilon
        if constraint_value[t][s] < 0:
            initial_value_negative.append(constraint_value[t][s])
        elif constraint_value[t][s] > 0:
            initial_value_positive.append(constraint_value[t][s])
        else:
            if odd_index == 0:
                initial_value_negative.append(constraint_value[t][s])
                odd_index = 1
            else:
                initial_value_positive.append(constraint_value[t][s])
                odd_index = 0

mode = "MIP"
# mode="PIP"
enlargement_rate = 1.4
shrinkage_rate = 0.7
base_rate = 10
pip_max_rate = 40
if mode == "MIP":
    base_rate = 100
    pip_max_rate = 100
# 这里record应该再加一项每个sub problem跑多少秒
with open('output/record.txt', 'a') as f3:
    print('base_rate,enlargement_rate,shrinkage_rate,pip_max_rate:', base_rate, enlargement_rate, shrinkage_rate,
          pip_max_rate, file=f3)
    print('max time ' + str(60) + 's each iteration', file=f3)

epsilon_1 = np.percentile(initial_value_positive, base_rate)
epsilon_2 = -np.percentile(initial_value_negative, 100 - base_rate)

for t in range(2 ** D):
    z_start[t] = {}
    for s in range(N):
        if constraint_value[t][s] >= 0:
            z_start[t][s] = 1
        else:
            z_start[t][s] = 0

# Construct gp model
model = gp.Model("DecisionTree")
model.setParam('IntegralityFocus', 1)
iterations_unchange = 0  # Continuous enlargement iterations
max_rate_reach = 0
iterations = 0  # Total iterations
# Calculate pbjective function, update a,b
start_time = time.time()
f_old, a_start, b_start, z_start, constraint_value, value_negative, value_positive, en_e1, en_e2, en_e1_lb, en_e2_lb, sh_e1, sh_e2, sh_e1_ub, sh_e2_ub = triage_file.build_decision_tree_model(
    model, x, Trt, Trt_is, y, propensity_model, treatment_mean, a_start, b_start, z_start, constraint_value, D, N, B_lb,
    M_ub, epsilon, epsilon_1, epsilon_2, p, num_j, iterations, base_rate, enlargement_rate, shrinkage_rate,
    pip_max_rate, estimator, mode)
end_time = time.time()
execution_time = end_time - start_time
with open('output/time.txt', 'a') as f2:
    print("Total time of pip method after iteration " + str(iterations), execution_time, "s", file=f2)

if mode == "PIP":
    f_new = 0
    base_rate = enlargement_rate * base_rate
    epsilon_1 = np.percentile(value_positive, base_rate)
    epsilon_2 = -np.percentile(value_negative, 100 - base_rate)
    value = [f_old]
    shrinkage = [0]
    e1_list = [en_e1]
    e2_list = [en_e2]
    e1_b_list = [en_e1_lb]
    e2_b_list = [en_e2_lb]

    while iterations_unchange < 10 and iterations < 50 and min(epsilon_1, epsilon_2) > 1e-6 and max_rate_reach <= 1:
        iterations += 1
        # 1. Determine index sets # In function
        # 2. Solve the MIP
        f_new, a_start, b_start, z_start, constraint_value, value_negative, value_positive, en_e1, en_e2, en_e1_lb, en_e2_lb, sh_e1, sh_e2, sh_e1_ub, sh_e2_ub = triage_file.build_decision_tree_model(
            model, x, Trt, Trt_is, y, propensity_model, treatment_mean, a_start, b_start, z_start, constraint_value, D,
            N,
            B_lb, M_ub, epsilon, epsilon_1, epsilon_2, p, num_j, iterations, base_rate, enlargement_rate,
            shrinkage_rate,
            pip_max_rate, estimator, mode)
        # 3. Enlargement
        if f_new - f_old <= 1:
            iterations_unchange = iterations_unchange + 1
            with open('output/output_iter=' + str(iterations) + '.txt', 'a') as f:
                print("Enlargement!", file=f)
            if enlargement_rate * base_rate < pip_max_rate:
                base_rate = enlargement_rate * base_rate
            else:
                base_rate = pip_max_rate
                max_rate_reach += 1
            epsilon_1 = np.percentile(value_positive, base_rate)
            epsilon_2 = -np.percentile(value_negative, 100 - base_rate)
            shrinkage.append(0)
            e1_list.append(en_e1)
            e2_list.append(en_e2)
            e1_b_list.append(en_e1_lb)
            e2_b_list.append(en_e2_lb)

        # 4. Shrinkage
        else:
            iterations_unchange = 0
            with open('output/output_iter=' + str(iterations) + '.txt', 'a') as f:
                print("Shrinkage!", file=f)
            base_rate = shrinkage_rate * base_rate
            epsilon_1 = np.percentile(value_positive, base_rate)
            epsilon_2 = -np.percentile(value_negative, 100 - base_rate)
            shrinkage.append(1)
            e1_list.append(sh_e1)
            e2_list.append(sh_e2)
            e1_b_list.append(sh_e1_ub)
            e2_b_list.append(sh_e2_ub)
        f_old = f_new
        value.append(f_old)
        end_time = time.time()
        execution_time = end_time - start_time
        with open('output/time.txt', 'a') as f2:
            print("Total time of pip method after iteration " + str(iterations), execution_time, "s", file=f2)

    # 5. Terminate
    with open('output/obj_value.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            ['Iterations', 'Value', 'Shrinkage', 'Epsilon_1_next', 'Epsilon_2_next', 'Epsilon_1_bound',
             'Epsilon_2_bound'])
        for i in range(iterations + 1):
            writer.writerow([i, value[i], shrinkage[i], e1_list[i], e2_list[i], e1_b_list[i], e2_b_list[i]])
