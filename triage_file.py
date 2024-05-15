# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import calculate_propensity_score as cps
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import os
from rpy2.robjects import r


def build_decision_tree_model(model, x, Trt, Trt_is, y, propensity_model, treatment_mean, a_start, b_start, z_start,
                              constraint_value, D, N, B_lb, M_ub, epsilon, epsilon_1, epsilon_2, p, num_j, iterations,
                              base_rate, enlargement_rate, shrinkage_rate, pip_max_rate, estimator, mode):
    model = model.copy()
    a = {}
    a_abs = {}
    b = {}
    z = {}
    L = {}
    c = {}
    u = {}
    M = {}
    A_L = {}
    A_R = {}
    J0_set = {}
    J1_set = {}
    J2_set = {}
    with open('output/output_iter=' + str(iterations) + '.txt', 'a') as f:
        print("epsilon_1,epsilon_2:", epsilon_1, epsilon_2, file=f)
    for k in range(2 ** D - 1):
        a[k] = model.addVars(p, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="a_" + str(k))
        # a[k] = model.addVars(p, lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.INTEGER, name="a_" + str(k))
        a_abs[k] = model.addVars(p, lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="a_abs_" + str(k))
        b[k] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="b_" + str(k))
        for i in range(p):
            # a[k][i].VarHintVal = a_start[k][i]
            a[k][i].setAttr(gp.GRB.Attr.Start, a_start[k][i])
        # b[k].VarHintVal = b_start[k]
        b[k].setAttr(gp.GRB.Attr.Start, b_start[k])
        u[k] = model.addVar(vtype=GRB.BINARY, name="u_" + str(k))

    for s in range(N):
        z[s] = {}
        for t in range(2 ** D):
            z[s][t] = model.addVar(vtype=GRB.BINARY, name="z_" + str(s) + "_" + str(t))

    for t in range(2 ** D):
        L[t] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="L_" + str(t))
        # if L_start is not None:
        # L[t].setAttr(gp.GRB.Attr.Start, L_start[t])
        c[t] = {}
        M[t] = {}
        for j in range(num_j):
            c[t][j] = model.addVar(vtype=GRB.BINARY, name="c_" + str(j) + "_" + str(t))
            # if c_start is not None:
            # c[t][j].setAttr(gp.GRB.Attr.Start, c_start[t][j])
            M[t][j] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="M_" + str(t) + "_" + str(j))
            # if M_start is not None:
            # M[t][j].setAttr(gp.GRB.Attr.Start, M_start[t][j])

    # Set Objective function
    obj = gp.quicksum(L[t] for t in range(2 ** D)) + gp.quicksum((1 - u[k]) for k in range(2 ** D - 1))
    # obj = gp.quicksum(L[t] for t in range(2 ** D))
    # model.setObjective(obj, GRB.MAXIMIZE)

    # Decision Tree branching constraints
    J0_count = 0
    J1_count = 0
    J2_count = 0
    for t in range(2 ** D):
        J0_set[t] = []  # J_{epsilon_1,2}^{in}
        J1_set[t] = []  # J_{epsilon_1}^>
        J2_set[t] = []  # J_{epsilon_2}^<
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
            # print("(s,t): (",s,",",t,")","CURRENT:",current_node,"PARENT:",parent_node)
            current_node = parent_node
        for s in range(N):
            # a_k' * x_s - b_k >= B_lb(1 - z_st), k in A_R(t)
            if constraint_value[t][s] > epsilon_1:
                J1_set[t].append(s)
                model.remove(z[s][t])
                J1_count += 1
                for k in A_R[t]:
                    model.addConstr(
                        gp.quicksum(a[k][i] * x[s][i] for i in range(p)) - b[k] - epsilon >= B_lb * (1 - z_start[t][s]))
                for k in A_L[t]:
                    model.addConstr(-(gp.quicksum(a[k][i] * x[s][i] for i in range(p))) + b[k] >= B_lb * (
                            1 - z_start[t][s]))
            elif constraint_value[t][s] >= -epsilon_2:
                z[s][t].setAttr(gp.GRB.Attr.Start, z_start[t][s])
                J0_set[t].append(s)
                J0_count += 1
                for k in A_R[t]:
                    model.addConstr(
                        gp.quicksum(a[k][i] * x[s][i] for i in range(p)) - b[k] - epsilon >= B_lb * (1 - z[s][t]))
                for k in A_L[t]:
                    model.addConstr(
                        -(gp.quicksum(a[k][i] * x[s][i] for i in range(p))) + b[k] >= B_lb * (1 - z[s][t]))
            else:
                J2_set[t].append(s)
                model.remove(z[s][t])
                J2_count += 1

    # \sum_j c_{jt} = 1
    for t in range(2 ** D):
        model.addConstr(gp.quicksum(c[t][j] for j in range(num_j)) == 1)

    # L_t <= \sum_s z_{st} + (1 - c_{jt}), for each t and each j
    for t in range(2 ** D):
        for j in range(num_j):
            if estimator == 'IPW':
                model.addConstr(L[t] <= gp.quicksum(
                    (y[s] * Trt_is[s][j] * z[s][t] / cps.propensity_score(propensity_model, x[s], Trt[s] - 1)) for s in
                    J0_set[t]) +
                                gp.quicksum((y[s] * Trt_is[s][j] * z_start[t][s] / cps.propensity_score(
                                    propensity_model, x[s], Trt[s] - 1)) for s in J1_set[t]) + M_ub * (1 - c[t][j]))
            elif estimator == 'DR':
                model.addConstr(L[t] <= gp.quicksum(((y[s] - treatment_mean[s][Trt[s] - 1]) * Trt_is[s][j] * z[s][t] /
                                                     propensity_model[s][Trt[s] - 1] + treatment_mean[s][j] * z[s][t])
                                                    for s in J0_set[t]) +
                                gp.quicksum(((y[s] - treatment_mean[s][Trt[s] - 1]) * Trt_is[s][j] * z_start[t][s] /
                                             propensity_model[s][Trt[s] - 1] + treatment_mean[s][j] * z_start[t][s]) for
                                            s in J1_set[t]) + M_ub * (1 - c[t][j]))
    for s in range(N):
        fixed_sum = 0
        J0_s = []
        constraint_true = 0
        for t in range(2 ** D):
            if s in J1_set[t]:
                fixed_sum += 1
            elif s in J0_set[t]:
                J0_s.append(t)
                constraint_true = 1
        if constraint_true == 1:
            model.addConstr(gp.quicksum(z[s][t] for t in J0_s) + fixed_sum == 1)

    # ||a_k||_1 <= u_k
    for k in range(2 ** D - 1):
        for i in range(p):
            model.addConstr(a[k][i] <= a_abs[k][i])
            model.addConstr(-a[k][i] <= a_abs[k][i])
        model.addConstr(gp.quicksum(a_abs[k][i] for i in range(p)) <= u[k])

    # Resource constraints
    for t in range(2 ** D):
        for j in range(num_j):
            model.addConstr(
                M[t][j] >= gp.quicksum(z[s][t] for s in J0_set[t]) + sum(z_start[t][s] for s in J1_set[t]) - M_ub * (
                        1 - c[t][j]))

    # Resource proportion
    gamma = [0.4, 0.2, 0.3, 0.5]
    penalty = {}
    for j in range(3):
        penalty[j] = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name="penalty" + str(j))
        model.addConstr(penalty[j] >= gp.quicksum(M[t][j] for t in range(2 ** D)) - gamma[j] * N)
        obj = obj - 100 * penalty[j]
        # obj = obj - 1000*penalty[j]

    model.setObjective(obj, GRB.MAXIMIZE)

    optimal_a = {}
    optimal_b = {}
    optimal_z = {}
    optimal_c = {}
    optimal_L = {}
    optimal_M = {}
    # model.setParam('Heuristics', 0)
    all_vars = model.getVars()
    integer_var_count = sum(1 for var in all_vars if var.vType == gp.GRB.BINARY)
    # if estimator == 'DR' and iterations == 0:
    if mode == "MIP" and iterations == 0:
        model.setParam("Timelimit", 7200)
    else:
        model.setParam("Timelimit", 60)
    model.setParam('LogFile', 'output/solver_log' + str(iterations) + '.txt')
    model.optimize()
    model.write("output/model" + str(iterations) + ".lp")
    optimal_value = model.objVal
    for k in range(2 ** D - 1):
        optimal_a[k] = [a[k][i].X for i in range(p)]
        optimal_b[k] = b[k].X
    for t in range(2 ** D):
        optimal_z[t] = {}
        optimal_M[t] = {}
        for j in range(num_j):
            if c[t][j].X >= 0.99:
                if estimator == 'IPW':
                    optimal_L[t] = sum(
                        (y[s] * Trt_is[s][j] * z[s][t].X / cps.propensity_score(propensity_model, x[s], Trt[s] - 1)) for
                        s in J0_set[t]) + sum(
                        (y[s] * Trt_is[s][j] * z_start[t][s] / cps.propensity_score(propensity_model, x[s], Trt[s] - 1))
                        for s in J1_set[t]) + M_ub * (1 - c[t][j].X)
                elif estimator == 'DR':
                    optimal_L[t] = sum(((y[s] - treatment_mean[s][Trt[s] - 1]) * Trt_is[s][j] * z[s][t].X /
                                        propensity_model[s][Trt[s] - 1] + treatment_mean[s][j] * z[s][t].X) for s in
                                       J0_set[t]) + sum(((y[s] - treatment_mean[s][Trt[s] - 1]) * Trt_is[s][j] *
                                                         z_start[t][s] / propensity_model[s][Trt[s] - 1] +
                                                         treatment_mean[s][j] * z_start[t][s]) for s in
                                                        J1_set[t]) + M_ub * (1 - c[t][j].X)
                optimal_M[t][j] = M[t][j].X
            else:
                optimal_L[t] = 0
                optimal_M[t][j] = 0
        for s in J0_set[t]:
            optimal_z[t][s] = z[s][t].X
        for s in J1_set[t]:
            optimal_z[t][s] = z_start[t][s]
        for s in J2_set[t]:
            optimal_z[t][s] = 0
        optimal_c[t] = [c[t][j].X for j in range(num_j)]

    value_positive = []
    value_negative = []
    constraint_value_0 = []
    constraint_value_1 = []
    constraint_value_2 = []
    odd_index = 0
    for t in range(2 ** D):
        for s in range(N):
            constraint_value[t][s] = 1e4
            for k in A_R[t]:
                if sum(a[k][i].X * x[s][i] for i in range(p)) - b[k].X - epsilon < constraint_value[t][s]:
                    constraint_value[t][s] = sum(a[k][i].X * x[s][i] for i in range(p)) - b[k].X - epsilon
            for k in A_L[t]:
                if -(sum(a[k][i].X * x[s][i] for i in range(p))) + b[k].X < constraint_value[t][s]:
                    constraint_value[t][s] = -(sum(a[k][i].X * x[s][i] for i in range(p))) + b[k].X
            if constraint_value[t][s] < 0:
                value_negative.append(constraint_value[t][s])
            elif constraint_value[t][s] > 0:
                value_positive.append(constraint_value[t][s])
            else:
                if odd_index == 0:
                    value_negative.append(constraint_value[t][s])
                    odd_index = 1
                else:
                    value_positive.append(constraint_value[t][s])
                    odd_index = 0
    # Return new epsilon_1,2
    with open('output/output_iter=' + str(iterations) + '.txt', 'a') as f:
        print("Obj:", optimal_value, file=f)
        print("optimal_a:", optimal_a, file=f)
        print("optimal_b:", optimal_b, file=f)
        print("optimal_c:", optimal_c, file=f)
        # print("optimal_M:",optimal_M,file=f)
        for t in range(2 ** D):
            print("optimal_z_" + str(t) + ":", sum(optimal_z[t][s] for s in range(N)), file=f)
        no_branch = 0
        z_zero = 0
        for s in range(N):
            temp_zs = 0
            for t in range(2 ** D):
                if s in J0_set[t]:
                    constraint_value_0.append(constraint_value[t][s])
                    print('J0_z_' + str(s) + ',' + str(t) + '=', z[s][t].X, file=f)
                    print('z_start' + str(s) + ',' + str(t) + '=', z_start[t][s], file=f)
                    temp_zs += z[s][t].X
                elif s in J1_set[t]:
                    constraint_value_1.append(constraint_value[t][s])
                    print('J1_z_' + str(s) + ',' + str(t) + '=', z_start[t][s], file=f)
                    temp_zs += z_start[t][s]
                else:
                    constraint_value_2.append(constraint_value[t][s])
                    print('J2_z_' + str(s) + ',' + str(t) + '=', 0, file=f)
                print('Constraint_value_' + str(s) + ',' + str(t), constraint_value[t][s], file=f)
                if constraint_value[t][s] > np.percentile(value_positive,
                                                          min(enlargement_rate * base_rate, pip_max_rate)):
                    print("If enlargement," + str(s) + ',' + str(t) + ' in J1_set', file=f)
                elif constraint_value[t][s] < np.percentile(value_negative,
                                                            100 - min(enlargement_rate * base_rate, pip_max_rate)):
                    print("If enlargement," + str(s) + ',' + str(t) + ' in J2_set', file=f)
                else:
                    print("If enlargement," + str(s) + ',' + str(t) + ' in J0_set', file=f)
                if constraint_value[t][s] > np.percentile(value_positive, shrinkage_rate * base_rate):
                    print("If shrinkage," + str(s) + ',' + str(t) + ' in J1_set', file=f)
                elif constraint_value[t][s] < np.percentile(value_negative, 100 - shrinkage_rate * base_rate):
                    print("If shrinkage," + str(s) + ',' + str(t) + ' in J2_set', file=f)
                else:
                    print("If shrinkage," + str(s) + ',' + str(t) + ' in J0_set', file=f)
            print('Sample ' + str(s) + ' in ' + str(temp_zs) + ' leaf nodes', file=f)
            if temp_zs == 0:
                z_zero += 1
            k = 0
            while k < 2 ** D - 1:
                if sum(a[k][i].X * x[s][i] for i in range(p)) - b[k].X - epsilon >= 0:
                    print('Sample ' + str(s) + ' pass node ' + str(k) + ' RIGHT', file=f)
                    k = 2 * k + 2
                elif -(sum(a[k][i].X * x[s][i] for i in range(p))) + b[k].X >= 0:
                    print('Sample ' + str(s) + ' pass node ' + str(k) + ' LEFT', file=f)
                    k = 2 * k + 1
                else:
                    no_branch += 1
                    print('Large epsilon!', sum(a[k][i].X * x[s][i] for i in range(p)) - b[k].X - epsilon,
                          -(sum(a[k][i].X * x[s][i] for i in range(p))) + b[k].X, file=f)
                    k = 2 ** D - 1
        if mode == "MIP" and iterations == 0:
            en_e1, en_e2, en_e1_lb, en_e2_lb, sh_e1, sh_e2, sh_e1_ub, sh_e2_ub = 0, 0, 0, 0, 0, 0, 0, 0
        else:
            en_e1 = np.percentile(value_positive, min(enlargement_rate * base_rate, pip_max_rate))
            print("If enlargement, epsilon_1 = ", en_e1, file=f)
            en_e2 = -np.percentile(value_negative, 100 - min(enlargement_rate * base_rate, pip_max_rate))
            print("If enlargement, epsilon_2 = ", en_e2, file=f)
            en_e1_lb = max(constraint_value_0 + constraint_value_2)
            print("If enlargement, epsilon_1 should >= ", en_e1_lb, file=f)
            en_e2_lb = -min(constraint_value_0)
            print("If enlargement, epsilon_2 should >= ", en_e2_lb, file=f)
            sh_e1 = np.percentile(value_positive, shrinkage_rate * base_rate)
            print("If shrinkage, epsilon_1 = ", sh_e1, file=f)
            sh_e2 = -np.percentile(value_negative, 100 - shrinkage_rate * base_rate)
            print("If shrinkage, epsilon_2 = ", sh_e2, file=f)
            sh_e1_ub = min(constraint_value_1)
            print("If shrinkage, epsilon_1 should < ", sh_e1_ub, file=f)
            sh_e2_ub = -max(constraint_value_2)
            print("If shrinkage, epsilon_2 should < ", sh_e2_ub, file=f)

        print("Number of samples fail to branch due to large epsilon: ", no_branch, file=f)
        print("Number of samples have no treatment:", z_zero, file=f)
        print("J0,J1,J2:", J0_count, J1_count, J2_count, file=f)

    with open('output/output_iter=' + str(iterations) + '.txt', 'a') as f:
        print("length_value_positive,length_value_negative:", len(value_positive), len(value_negative), file=f)
    return optimal_value, optimal_a, optimal_b, optimal_z, constraint_value, value_negative, value_positive, en_e1, en_e2, en_e1_lb, en_e2_lb, sh_e1, sh_e2, sh_e1_ub, sh_e2_ub

# 然后调用函数创建并求解模型
# model = build_decision_tree_model(D, N, L, epsilon)
# 我感觉inbalanced tree不是坏事，只要给同一个treatment就行了，后续可以再把相同的treatment合并
