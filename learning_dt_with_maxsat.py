import math
import os
import time
from data import Dataset
import pysat
from utils.constants import *

from pysat.examples.rc2 import RC2
from pysat.formula import WCNF
from pysat.solvers import Solver
from pysat.card import CardEnc
from pysat.formula import IDPool

import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Learning Decision Trees with MaxSAT')
    parser.add_argument('--data_file', help='Path to the dataset', required=True)
    parser.add_argument('--train_ratio', help='Training data ratio', type=float, default=0.75)
    parser.add_argument('--maxsat', help='MaxSAT solver', default='g3')
    parser.add_argument('--max_depth', help='Maximum depth of the tree', type=int, default=-1)
    parser.add_argument('--size', help='Number of nodes in the decision tree', type=int)
    parser.add_argument('--atleast', help='At least constraint', action='store_true')
    parser.add_argument('--n_learners', help='Number of learners', type=int, required=True)
    parser.add_argument('--depth', help='Depth of the tree', type=int)
    parser.add_argument('--seed', help='Random seed', type=int, default=2024)
    parser.add_argument('--separator', help='Separator for the dataset', default=' ')
    return parser.parse_args()


class DecisionTreeLearner:

    def __init__(self, data, options):
        self.data = data
        self.options = options
        self.wcnf = WCNF()
        self.solver = Solver(name='g3')

        # variables
        self.vpool = IDPool()
        self.left_child_vars = {} # l_ij
        self.right_child_vars = {} # r_ij
        self.parent_vars = {} # p_ji
        self.discrimination_vars = {} # a_rj
        self.activation_vars = {} # u_rj
        self.label_vars = {} # y_s
        self.d0_vars = {} # d0_rj
        self.d1_vars = {} # d1_rj
        self.class_vars = {} # c_j
        self.leaf_vars = {} # v_i
        self.depth_vars = {} # h_it
        self.size_vars = {} # m_i
        self.correct_sampling_vars = {} # b_q
        self.lambda_vars = {} # lambda_it
        self.tau_vars = {} # tau_it
        self.true_var = None
        self.false_var = None

        # clauses
        self.clauses = []
        

    # ===================== Utility functions for variable generation =====================

    def get_l_bounds(self, i, N=None):
        return range(i + 1, min(2 * i, N - 1) + 1)

    def get_r_bounds(self, i, N=None):
        return range(i + 2, min(2 * i + 1, N) + 1)

    def get_p_bounds(self, j, N=None):
        return range(int(max(math.floor(j / 2), 1)), (min((j - 1), N) + 1))

    def get_lambda_bounds(self, i):
        return range(int(math.floor(i / 2)) + 1)

    def get_tau_bounds(self, i):
        return range(i + 1)

    # get the bound of the height for the node i
    def get_h_bounds(self, i, N=None):
        return range(int(math.ceil(np.log2(i+1))) - 1, int(math.ceil((i-1)*1.0/2)) + 1)
    

    # ===================== Variable generation functions

    def generate_left_child_variables(self, N):
        for i in range(1, N + 1):
            self.left_child_vars[i] = {}
            for j in self.get_l_bounds(i=i, N=N):
                if (j % 2 == 0):
                    self.left_child_vars[i][j] = self.vpool.id(f"l_{i}_{j}")


    def generate_right_child_variables(self, N):
        for i in range(1, N + 1):
            self.right_child_vars[i] = {}
            for j in self.get_r_bounds(i=i, N=N):
                if (j % 2 == 1):
                    self.right_child_vars[i][j] = self.vpool.id(f"r_{i}_{j}")


    def generate_parent_variables(self, N):
        for j in range(2, N + 1):
            self.parent_vars[j] = {}
            for i in self.get_p_bounds(j=j, N=N):
                self.parent_vars[j][i] = self.vpool.id(f"p_{j}_{i}")


    def generate_leaf_node_variables(self, N):
        for i in range(1, N + 1):
            self.leaf_vars[i] = self.vpool.id(f"v_{i}")


    def generate_lambda_variables(self, N):
        for i in range(1, N + 1):
            self.lambda_vars[i] = {}
            for t in self.get_lambda_bounds(i=i):
                self.lambda_vars[i][t] = self.vpool.id(f"lambda_{i}_{t}")


    def generate_tau_variables(self, N):
        for i in range(1, N + 1):
            self.tau_vars[i] = {}
            for t in self.get_tau_bounds(i=i):
                self.tau_vars[i][t] = self.vpool.id(f"tau_{i}_{t}")


    def generate_constant_variables(self):
        self.true_var = self.vpool.id(LABEL_TRUE_VARS)
        self.false_var = self.vpool.id(LABEL_FALSE_VARS)


    def generate_discrimination_variables(self, N, K):
        for r in range(1, K + 1):
            self.discrimination_vars[r] = {}
            for j in range(1, N + 1):
                self.discrimination_vars[r][j] = self.vpool.id(f"a_{r}_{j}")


    def generate_activation_variables(self, N, K):
        for r in range(1, K + 1):
            self.activation_vars[r] = {}
            for j in range(1, N + 1):
                self.activation_vars[r][j] = self.vpool.id(f"u_{r}_{j}")


    def generate_d0_variables(self, N, K):
        for r in range(1, K + 1):
            self.d0_vars[r] = {}
            for j in range(1, N + 1):
                self.d0_vars[r][j] = self.vpool.id(f"d0_{r}_{j}")


    def generate_d1_variables(self, N, K):
        for r in range(1, K + 1):
            self.d1_vars[r] = {}
            for j in range(1, N + 1):
                self.d1_vars[r][j] = self.vpool.id(f"d1_{r}_{j}")


    def generate_class_variables(self, N):
        for i in range(1, N + 1):
            self.class_vars[i] = self.vpool.id(f"c_{i}")
    

    def generate_depth_variables(self, N):
        for i in range(1, N+1):
            self.depth_vars[i] = {}
            for t in self.get_h_bounds(i=i, N=N):
                self.depth_vars[i][t] = self.vpool.id(f"h_{i}_{t}")


    # the m variables must be created only for odd nodes
    def generate_size_variables(self, N):
        assert N % 2 == 1
        for i in range(1, N+1):
            if i % 2 == 0: continue
            self.size_vars[i] = self.vpool.id(f"m_{i}")


    def generate_correct_sampling_variables(self, M):
        for q in range(1, M + 1):
            self.correct_sampling_vars[q] = self.vpool.id(f"b_{q}")
            self.add_soft_clause([self.correct_sampling_vars[q]], weight=1)


    def generate_variables(self, N, K, max_depth=-1, depth=-1):
        """
        Generate the variables of the MaxSAT instance using a variable pool.
        """

        self.generate_constant_variables()
        self.generate_left_child_variables(N)
        self.generate_right_child_variables(N)
        self.generate_leaf_node_variables(N)
        self.generate_parent_variables(N)
        
        if self.options.atleast is not None:
            self.generate_size_variables(N)
        if max_depth > 0 or depth > 0:
            self.generate_depth_variables(N)
        self.generate_lambda_variables(N)
        self.generate_tau_variables(N)

        self.generate_discrimination_variables(N, K)
        self.generate_activation_variables(N, K)
        self.generate_d0_variables(N, K)
        self.generate_d1_variables(N, K)
        self.generate_class_variables(N)
        self.generate_correct_sampling_variables(len(self.data.train_samples))


    # ===================== Constraint util functions

    def add_clause(self, clause):

        if self.options.atleast is not None:
            modified_clause = self.adapt_constraint_with_m(clause)

        self.wcnf.append(modified_clause)
        self.clauses.append(modified_clause)


    def add_soft_clause(self, clause, weight):
        self.wcnf.append(clause, weight=weight)
        self.clauses.append(clause)

    
    def adapt_constraint_with_m(self, clause):
        """
        Adapts the clause passed to it by applying a conditional modifier based on the largest node index in each clause.
        If the largest index `j` in a clause `C` is odd, replace `C` with `m_j → C`.
        If `j` is even, replace `C` with `m_{j+1} → C`.

        Args:
            clause (list): List of literals in the clause.

        Returns:
            clause (list): Adapted clause with the conditional modifier.
        """
        largest_index = max(abs(lit) for lit in clause)

        # Find the corresponding `m` variable
        if largest_index % 2 == 1:  # If `j` is odd
            m_j = self.size_vars.get(largest_index)
            if m_j is not None:
                adapted_clause = [-m_j] + clause
            else:
                adapted_clause = clause  # If m_j not defined, keep clause as is
        else:  # If `j` is even
            m_j_plus_1 = self.size_vars.get(largest_index + 1)
            if m_j_plus_1 is not None:
                adapted_clause = [-m_j_plus_1] + clause
            else:
                adapted_clause = clause  # If m_j+1 not defined, keep clause as is

        return adapted_clause

    # ===================== Constraint generation functions

    def add_root_not_leaf_constraint(self, leaf_vars):
        """
        Adds the constraint that the root node is not a leaf: ¬v1
        """
        new_clause = [-leaf_vars[1]]
        self.add_clause(new_clause) # ¬v1


    def add_leaf_no_children_constraint(self):
        """
        Adds constraints that if a node is a leaf, it has no children.
        """
        for i, v_i in self.leaf_vars.items():
            for j in self.left_child_vars.get(i, {}):
                new_clause = [-v_i, -self.left_child_vars[i][j]]
                self.add_clause(new_clause)  # v_i -> ¬l_ij

    
    def add_consecutive_child_constraint(self):
        """Ensure left child of node i has a consecutive right child."""
        for i, left_children in self.left_child_vars.items():
            for j, l_ij in left_children.items():
                r_ij_plus_1 = self.right_child_vars.get(i, {}).get(j + 1)
                if r_ij_plus_1:
                    new_clause = [l_ij, -r_ij_plus_1]
                    self.add_clause(new_clause)  # l_ij -> -r_ij+1
                    new_clause = [-l_ij, r_ij_plus_1]
                    self.add_clause(new_clause)  # ¬l_ij -> r_ij+1


    def add_non_leaf_must_have_child_constraint(self):
        """Ensure non-leaf nodes have exactly one left child."""
        for i, v_i in self.leaf_vars.items():
            children = list(self.left_child_vars.get(i, {}).values())
            if children:
                card = CardEnc.equals(lits=children, bound=1, vpool=self.vpool)
                for new_clause in card.clauses:
                    self.add_clause([-v_i] + new_clause)


    def add_parent_child_relationship(self):
        """Ensure parent-child relationship through left or right indicators."""
        for j, parents in self.parent_vars.items():
            for i, p_ji in parents.items():
                if j in self.left_child_vars.get(i, {}):
                    l_ij = self.left_child_vars[i][j]
                    new_clause = [p_ji, -l_ij]
                    self.add_clause(new_clause)
                    new_clause = [-p_ji, l_ij]
                    self.add_clause(new_clause)
                if j in self.right_child_vars.get(i, {}):
                    r_ij = self.right_child_vars[i][j]
                    new_clause = [p_ji, -r_ij]
                    self.add_clause(new_clause)
                    new_clause = [-p_ji, r_ij]
                    self.add_clause(new_clause)


    def add_tree_structure_constraint(self, N):
        """
        Ensure each non-root node has exactly one parent.

        Args:
            N (int): Number of nodes
        """
        for j in range(2, N + 1):
            parents = list(self.parent_vars[j].values())
            card = CardEnc.equals(lits=parents, bound=1, vpool=self.vpool)
            for new_clause in card.clauses:
                self.add_clause(new_clause)


    def add_discrimination_for_value_0(self, N):
        """
        Define discrimination constraints for feature value 0.

        Args:
            N (int): Number of nodes.
        """
        for r, d0_r in self.d0_vars.items():
            new_clause = [-d0_r[1]]
            self.add_clause(new_clause)
            for j in range(2, N + 1):
                terms = []
                for i in range(j // 2, j):
                    if i in self.parent_vars.get(j, {}) and i in self.d0_vars.get(r, {}) and i in self.right_child_vars.get(i, {}):
                        p_ji = self.parent_vars[j][i]
                        d0_ri = self.d0_vars[r][i]
                        a_ri = self.discrimination_vars[r][i]
                        r_ij = self.right_child_vars[i][j]
                        terms.append([p_ji, d0_ri])
                        terms.append([a_ri, r_ij])
                if terms:
                    new_clause = [d0_r[j], -terms]
                    self.add_clause(new_clause)
                    new_clause = [-d0_r[j], terms]
                    self.add_clause(new_clause)


    def add_discrimination_for_value_1(self, N):
        """
        Define discrimination constraints for feature value 1.

        Args:
            N (int): Number of nodes.
        """
        for r, d1_r in self.d1_vars.items():
            new_clause = [-d1_r[1]]
            self.add_clause(new_clause)
            for j in range(2, N + 1):
                terms = []
                for i in range(j // 2, j):
                    if i in self.parent_vars.get(j, {}) and i in self.d1_vars.get(r, {}) and i in self.left_child_vars.get(i, {}):
                        p_ji = self.parent_vars[j][i]
                        d1_ri = self.d1_vars[r][i]
                        a_ri = self.discrimination_vars[r][i]
                        l_ij = self.left_child_vars[i][j]
                        terms.append([p_ji, d1_ri])
                        terms.append([a_ri, l_ij])
                if terms:
                    new_clause = [d1_r[j], -terms]
                    self.add_clause(new_clause)  
                    new_clause = [-d1_r[j], terms]
                    self.add_clause(new_clause)


    def add_path_activation_constraint(self, N, K):
        """
        Implements Equation (9) of original paper to enforce path activation constraints for features.
        Ensures that if a feature `r` is used at node `j`, then its activation along the path is consistent.

        Args:
            N (int): Number of nodes.
            K (int): Number of features.
        """
        for r in range(1, K + 1):  
            for j in range(2, N + 1):  
                
                for i in range(j // 2, j):
                    if i in self.parent_vars.get(j, {}):
                        u_ri = self.activation_vars[r].get(i)
                        p_ji = self.parent_vars[j].get(i)
                        a_rj = self.discrimination_vars[r].get(j)
                        if u_ri is not None and p_ji is not None and a_rj is not None:
                            self.add_clause([-u_ri, -p_ji, -a_rj]) # ¬u_ri ∨ ¬p_ji ∨ ¬a_rj
                
                u_rj = self.activation_vars[r].get(j)
                a_rj = self.discrimination_vars[r].get(j)
                parent_activation = []
                for i in range(j // 2, j):
                    u_ri = self.activation_vars[r].get(i)
                    p_ji = self.parent_vars[j].get(i)
                    if u_ri is not None and p_ji is not None:
                        parent_activation.append((u_ri, p_ji))

                # Clause: u_rj → (a_rj ∨ ∨ (u_ri ∧ p_ji))
                if u_rj is not None and a_rj is not None:
                    self.add_clause([-u_rj, a_rj] + [p for u, p in parent_activation])
                    # Clause: (a_rj ∨ ∨ (u_ri ∧ p_ji)) → u_rj
                    for u, p in parent_activation:
                        self.add_clause([u_rj, -u, -p])
                    if a_rj is not None:
                        self.add_clause([u_rj, -a_rj])


    def add_feature_usage_constraints(self, N, K):
        """
        Enforces that:
        1. Non-leaf nodes use exactly one feature.
        2. Leaf nodes use no features.
        
        Args:
            N (int): Number of nodes.
            K (int): Number of features.
        """
        for j in range(1, N + 1):
            v_j = self.leaf_vars[j]
            
            feature_usage = [self.discrimination_vars[r][j] for r in range(1, K + 1)]
            
            # Equation (10) of original paper: Exactly one feature is used for non-leaf nodes
            if feature_usage:
                card = CardEnc.equals(lits=feature_usage, bound=1, vpool=self.vpool)
                for clause in card.clauses:
                    self.add_clause([-v_j] + clause)

            # Equation (11) of original paper: No feature is used for leaf nodes
            for a_rj in feature_usage:
                # If node `j` is a leaf (v_j), then no feature should be active (¬a_rj for each feature `r`)
                self.add_clause([v_j, -a_rj])  # v_j -> ¬a_rj


    def add_leaf_discriminative_feature_constraint(self, N, K):
        """
        Enforces that if a node `j` is a leaf and does not have a specific class,
        then at least one discrimination variable must be active for that node
        based on the sign of the feature in the example.
        
        Args:
            N (int): Number of nodes.
            K (int): Number of features.
        """
        positive_samples = self.data.get_positive_train_samples()
        for q, eq in positive_samples:
            b_q = self.correct_sampling_vars[q]
            
            for i in range(1, N + 1):
                v_i = self.leaf_vars[i]
                c_i = self.class_vars.get(i)
                
                discriminative_features = []
                for r in range(1, K + 1):
                    feature_value = eq[r - 1]
                    d_ri = self.d0_vars[r][i] if feature_value == 0 else self.d1_vars[r][i]
                    discriminative_features.append(d_ri)
                
                # Add clause for positive example: ¬b_q ∨ ¬v_i ∨ c_i ∨ ∨ d_{sigma(r, q)}^{r, i}
                clause = [-b_q, -v_i, c_i] + discriminative_features
                self.add_clause(clause)


    def add_leaf_class_constraint(self, N, K):
        """
        Enforces that if a leaf node `j` is assigned a specific class,
        then at least one discrimination variable must be active for that node
        based on the sign of the feature in the example.

        Args:
            N (int): Number of nodes.
            K (int): Number of features.
        """
        negative_samples = self.data.get_negative_train_samples()
        for q, eq in negative_samples:
            b_q = self.correct_sampling_vars[q]
            
            for i in range(1, N + 1):
                v_i = self.leaf_vars[i]
                c_i = self.class_vars.get(i)
                
                discriminative_features = []
                for r in range(1, K + 1):
                    feature_value = eq[r - 1]
                    d_ri = self.d0_vars[r][i] if feature_value == 0 else self.d1_vars[r][i]
                    discriminative_features.append(d_ri)
                
                # Add clause for negative example: ¬b_q ∨ ¬v_i ∨ ¬c_i ∨ ∨ d_{sigma(r, q)}^{r, i}
                clause = [-b_q, -v_i, -c_i] + discriminative_features
                self.add_clause(clause)


    def add_depth_constraints(self, N):
        """
        Adds constraints for the depth of each node, ensuring each node has an exact depth
        within the allowable range, and the root node is at depth 0.

        Args:
            N (int): Number of nodes in the tree.
        """
        for i in range(1, N + 1):
            min_depth = math.ceil(math.log(i + 1, 2)) - 1
            max_depth_i = math.ceil((i - 1) / 2)
            
            depth_vars = []
            for t in range(min_depth, max_depth_i + 1):
                depth_var = self.vpool.id(f"depth_{i}_{t}")
                self.depth_vars.setdefault(i, {})[t] = depth_var
                depth_vars.append(depth_var)
            
            if i == 1:
                self.add_clause([self.depth_vars[1][0]])  # depth1,0 is true for the root

            # Each node must have exactly one depth
            if depth_vars:
                card = CardEnc.equals(lits=depth_vars, bound=1, vpool=self.vpool)
                for clause in card.clauses:
                    self.add_clause(clause)


    def add_depth_consistency_constraints(self, N):
        """
        Adds depth consistency constraints to ensure that if a node `i` is at depth `t` and has a child `j`,
        then the child `j` must be at depth `t + 1`.

        Args:
            N (int): Number of nodes in the tree.
        """
        for i in range(1, N + 1):
            min_depth = math.ceil(math.log(i + 1, 2)) - 1
            max_depth_i = math.ceil((i - 1) / 2)
            
            for t in range(min_depth, max_depth_i + 1):
                depth_it = self.depth_vars[i][t]

                # For left children of `i`
                for j in self.left_child_vars.get(i, {}):
                    l_ij = self.left_child_vars[i][j]
                    depth_j_t1 = self.depth_vars[j].get(t + 1)
                    
                    if depth_j_t1 is not None:
                        # depth_i,t ∧ l_ij → depth_j,t+1
                        clause = [-depth_it, -l_ij, depth_j_t1]
                        self.add_clause(clause)

                # For right children of `i`
                for j in self.right_child_vars.get(i, {}):
                    r_ij = self.right_child_vars[i][j]
                    depth_j_t1 = self.depth_vars[j].get(t + 1)
                    
                    if depth_j_t1 is not None:
                        # depth_i,t ∧ r_ij → depth_j,t+1
                        clause = [-depth_it, -r_ij, depth_j_t1]
                        self.add_clause(clause)

        
    def add_depth_constraints_leafs(self, N, max_depth=None, depth=None):
        """
        Adds constraints to enforce:
        - If `max_depth` is specified, all nodes at `max_depth` must be leaf nodes.
        - If `depth` is specified as an exact depth, at least one node must exist at `depth`,
            and all nodes at `depth` must be leaf nodes.
        
        Args:
            N (int): Total number of nodes in the tree.
            max_depth (int, optional): Maximum depth allowed in the tree.
            depth (int, optional): Exact depth required for the tree.
        """
        if max_depth is not None:
            target_depth = max_depth
        elif depth is not None:
            target_depth = depth
        else:
            raise ValueError("Either max_depth or depth must be provided")

        # Range of nodes at `target_depth`
        start_node = int(2 ** target_depth)
        end_node = int(min((2 ** (target_depth + 1)) - 1, N))
        
        # Enforce that all nodes at `target_depth` are leaf nodes
        for i in range(start_node, end_node + 1):
            depth_i_U = self.depth_vars.get(i, {}).get(target_depth)
            v_i = self.leaf_vars.get(i)

            # depth_{i, target_depth} → v_i, i.e., ¬depth_{i, target_depth} ∨ v_i
            if depth_i_U is not None and v_i is not None:
                clause = [-depth_i_U, v_i]
                self.add_clause(clause)

        # If `depth` is specified, enforce that at least one node is at `depth`
        if depth is not None:
            depth_at_target_clauses = [
                self.depth_vars.get(i, {}).get(target_depth)
                for i in range(start_node, end_node + 1)
                if self.depth_vars.get(i, {}).get(target_depth) is not None
            ]
            
            if depth_at_target_clauses:
                self.add_clause(depth_at_target_clauses)

    
    def transitivity_size_variables(self, N):
        """
        Adds transitivity constraints for the size variables: m_i+2 -> m_i.

        Args:
            N (int): Number of nodes in the tree.
        """

        for i in range(1, N - 1):
            if i % 2 == 0: continue
            m_i = self.size_vars.get(i)
            m_i_plus_2 = self.size_vars.get(i + 2)
            if m_i and m_i_plus_2:
                self.add_clause([-m_i_plus_2, m_i])



    def generate_constraints(self, N):
        self.add_root_not_leaf_constraint(self.leaf_vars)
        self.add_leaf_no_children_constraint()
        self.add_consecutive_child_constraint()
        self.add_non_leaf_must_have_child_constraint()
        self.add_parent_child_relationship()
        self.add_tree_structure_constraint(N)
        self.add_discrimination_for_value_0(N)
        self.add_discrimination_for_value_1(N)
        self.add_path_activation_constraint(N, len(self.data.feature_names) - 1)
        self.add_feature_usage_constraints(N, len(self.data.feature_names) - 1)
        self.add_leaf_discriminative_feature_constraint(N, len(self.data.feature_names) - 1)
        self.add_leaf_class_constraint(N, len(self.data.feature_names) - 1)

        if self.options.max_depth is not None or self.options.depth is not None:
            self.add_depth_constraints(N)
            self.add_depth_consistency_constraints(N)
            self.add_depth_constraints_leafs(N, max_depth=self.options.max_depth, depth=self.options.depth)

        if self.options.atleast is not None:
            self.transitivity_size_variables(N)


    # ===================== Generate Decision Tree

    def generate_decision_tree(self, N, K=None, max_depth=-1, depth=-1, sol_path=None, u_wght_soft=[]):
        
        assert not (max_depth > 0 and depth > 0)
        if K is None:
            K = len(self.data.train_samples[0]) - 1

        # create or update the wcnf file
        if len(u_wght_soft) == 0:

            # create the wcnf file at first -> generating the values in self.wcnf
            self.generate_variables(N=N, K=K, max_depth=max_depth, depth=depth)
            self.generate_constraints(N=N)

            # add the wcnf formula to the .wcnf file
            self.wcnf.topw = sum(self.wcnf.wght) + 1
            file_name = self.options.data_file.split('/')[-1].split('.')[0]
            wcnf_base_path = self.create_dir_solution('binarytree/wcnf', file_name)

            # define the wcnf file path depending on the use of depth or max_depth
            if max_depth > 0:
                wcnf_file_path = wcnf_base_path + '/formula_' + str(self.options.seed) + '_'\
                    + str(N) + '_max-' + str(max_depth)
            elif depth > 0:
                wcnf_file_path = wcnf_base_path + '/formula_' + str(self.options.seed) + '_'\
                    + str(N) + '_exact-' + str(depth)
            else:
                wcnf_file_path = wcnf_base_path + '/formula_' + str(self.options.seed) + '_' + str(N)
                
            wcnf_file_path = wcnf_file_path
            self.wcnf_file_path = wcnf_file_path + '.wcnf'
            self.wcnf.to_file(self.wcnf_file_path)

        else:
            # update the weights of soft clauses in exisiting wcnf formula
            assert len(u_wght_soft) == len(self.wcnf.wght)
            assert not os.path.isfile(self.wcnf_file_path)

            for b_q, weight in zip(self.correct_sampling_vars, u_wght_soft):
                self.wcnf.append([b_q], weight=weight)
                
            self.wcnf_file_path = '_'.join(self.wcnf_file_path.split('_')[:-2]) + '_' + '.wcnf'
            self.wcnf.wght = u_wght_soft
            self.wcnf.topw = sum(u_wght_soft) + 1
            self.wcnf.to_file(self.wcnf_file_path)

        # print the size information of wcnf formula
        nb_var = self.wcnf.nv
        nb_hard = len(self.wcnf.hard)
        nb_soft = len(self.wcnf.soft)
        count_literal = lambda l: sum([len(i) for i in l])
        nb_literals = count_literal(self.wcnf.soft) + count_literal(self.wcnf.hard)

        # logs
        print("nb variables: " + str(nb_var))
        print("nb soft clauses: " + str(nb_soft))
        print("nb hard clauses: " + str(nb_hard))
        print("nb literals: " + str(nb_literals))
        
        var_model = []
        cost = -1

        # generate the model with the maxsat solver
        with RC2(self.wcnf) as rc2:
            print("Solving the problem with RC2")
            var_model = rc2.compute()
            cost = rc2.cost

        # get the results of classification in training samples
        if len(var_model) > 0:
            c_results = self.get_training_example_classification_result(var_model)
        else:
            print("No solution found")

        # delete the wcnf file due to disk space
        print(self.wcnf_file_path)
        assert os.path.isfile(self.wcnf_file_path)
        os.remove(self.wcnf_file_path)

        # prepare the file for the solution
        if sol_path is None:
            sol_file_name = wcnf_file_path + "{}".format("_best") + ".sol"
        else:
            sol_file_name = sol_path
        
        if len(var_model) != 0:
            self.build_graph(N=N, model=var_model, filename=sol_file_name, K=K, labeled=True)

        return cost, sol_file_name, c_results, self.wcnf.wght, nb_var, nb_soft, nb_hard, nb_literals
    

    def print_constraints(self):
        """Print all constraints (clauses) added to the solver."""
        print("Constraints (Clauses):")
        for clause in self.clauses:
            print(clause)


    def create_dir_solution(self, basepath, dataset_name):
        path = os.path.join(basepath, dataset_name)
        print("Saving the wcnf file at: ", path)
        if not os.path.exists(path):
            os.makedirs(path)
        return path
    

    def get_training_example_classification_result(self, var_model):
        """
        Get the classification results based on the current model.
        0 -> wrongly classified, 1 -> correctly classified.

        Args:
            var_model (list of int): Model from the solver, with positive values indicating true literals.
        
        Returns:
            list of int: Array of classification results based on var_model for soft clauses in self.wcnf.
                        Each element is 1 if correctly classified, 0 if wrongly classified.
        """
        c_results = []
        
        # Iterate over each soft clause in the WCNF
        for s_clause in self.wcnf.soft:
            # Assume the clause is satisfied (correct classification) unless proven otherwise
            correct_classification = True
            for literal in s_clause:
                # If any literal in the clause is false (negative in model), classification is wrong
                if var_model[abs(literal) - 1] < 0:
                    c_results.append(0)
                    correct_classification = False
                    break
            
            # If all literals in the clause were true, mark as correctly classified
            if correct_classification:
                c_results.append(1)
        
        return c_results

    

    def build_graph(self, N, model, filename="graph.png", K=None, labeled=False):
        """
        Build the graph based on the model solution.
        
        Args:
            N (int): Total number of nodes in the tree.
            model (list of int): Solution model from the solver.
            filename (str): Output file name to store the graph information.
            K (int): Number of features, if available.
            labeled (bool): Whether to label nodes based on features.
        """
        
        with open(filename, "w") as file_dt:
            nodes = []
            labeldict = None
            
            # Get the number of nodes really used based on the model
            nb_node_used = self.get_number_nodes_really_used(N=N, model=model)
            
            print("Number of nodes used:", nb_node_used)
            file_dt.write("NODES\n")

            if labeled:
                labeldict = {}

                # Iterate over features and nodes to assign labels
                for r in range(1, K + 1):
                    for j in range(1, nb_node_used + 1):
                        # Get variable ID for discrimination variable a_r_j
                        a_r_j_var_id = self.discrimination_vars[r].get(j)
                        if a_r_j_var_id is not None and model[a_r_j_var_id - 1] > 0:
                            labeldict[j] = "" + self.data.feature_names[r - 1] + "_" + str(r)
                            nodes.append((j, self.data.feature_names[r - 1]))
                            file_dt.write("{} {}\n".format(j, self.data.feature_names[r - 1]))

                # Label class nodes based on v_j and c_j variables
                for j in range(1, nb_node_used + 1):
                    v_var_id = self.leaf_vars.get(j)
                    c_var_id = self.class_vars.get(j)
                    if v_var_id is not None and model[v_var_id - 1] > 0:
                        v = 1 if (c_var_id is not None and model[c_var_id - 1] > 0) else 0
                        labeldict[j] = "             c_" + str(v)
                        file_dt.write("{} {}\n".format(j, v))

            file_dt.write("EDGES\n")
            edges = []

            # Add edges for left and right children based on the model
            for i in range(1, nb_node_used + 1):
                # Left child edges
                for j in self.get_l_bounds(i=i, N=nb_node_used):
                    if j % 2 == 0:
                        l_i_j_var_id = self.left_child_vars.get(i, {}).get(j)
                        if l_i_j_var_id is not None and model[l_i_j_var_id - 1] > 0:
                            edges.append((i, j, 1))
                            file_dt.write("{} {} {}\n".format(i, j, 1))

                # Right child edges
                for j in self.get_r_bounds(i=i, N=nb_node_used):
                    if j % 2 == 1:
                        r_i_j_var_id = self.right_child_vars.get(i, {}).get(j)
                        if r_i_j_var_id is not None and model[r_i_j_var_id - 1] > 0:
                            edges.append((i, j, 0))
                            file_dt.write("{} {} {}\n".format(i, j, 0))



    def get_number_nodes_really_used(self, N, model):
        """
        Get the number of nodes really used based on the model solution.
        
        Args:
            N (int): Total number of nodes.
            model (list of int): Solution model returned by the solver, where positive
                                literals indicate satisfied variables.
        
        Returns:
            int: The highest index of nodes used.
        """
        assert N % 2 == 1
        nb_node_used = 1
        
        for i in range(1, N + 1):
            if i % 2 == 0: continue
            
            # Retrieve the variable ID for `m_i` directly from `vpool`
            m_var_id = self.size_vars.get(i)
            
            # Check if this node is used in the model (positive in the solution)
            if m_var_id is not None and model[m_var_id - 1] > 0:
                nb_node_used = i
            else:
                break
            
        return nb_node_used



def main():
    
    args = parse_args()
    data = Dataset(file_path=args.data_file, train_ratio=args.train_ratio, seed=args.seed, separator=args.separator)

    # we get the number of features and samples from the data
    n_features = len(data.feature_names) - 1
    n_samples = len(data.train_samples)
    print(f"Loaded {n_samples} samples with {n_features} features.")

    if args.size <= 0 and (args.max_depth > 0 or args.depth > 0):
        args.atleast = 1
        args.size = int(np.exp2(args.max_depth+1)) - 1 if args.max_depth > 0 else int(np.exp2(args.depth+1)) - 1
    elif args.size <= 0:
        assert("If the size is 0, the max_depth or depth must be greater than 0")

    print(f"Generating a decision tree with {args.size} nodes.")

    # initialize a decision tree learner with the data
    learner = DecisionTreeLearner(data=data, options=args)
    print(learner.generate_decision_tree(N=args.size, K=n_features, max_depth=args.max_depth, depth=args.depth))
    # learner.generate_variables(N=args.size, K=n_features, max_depth=args.max_depth, depth=args.depth)

    # print("Variables generated")
    # print(learner.vpool)

    # learner.generate_constraints(N=args.size)

    # print("Constraints generated")
    # learner.print_constraints()

    




if __name__ == '__main__':
    # get execution time
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))