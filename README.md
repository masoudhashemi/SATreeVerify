# Verifying Tree Ensembles using SAT

This code implements a method to represent ensemble trees (XGBoost and RanomForest) with propsitional logic. We then use this representation to test the robustness of the models using SAT (with Z3 solver).


Check `examples` folder for examples of using the code.


## How the code works:
- It first parses the trees and finds the decision box boundaries and their values.
- Using the boundary thresholds, we discretize the input data. Using this discretization, we can represent the trees as propositional logic formulas easily.
- Using the threshold, we define a set of variables for each decision box. and define a set of clauses for each tree. In addition, we add other clauses to ensuure that the tree decisions do not have overlaps (for each input there is a unique decision).
- For each input, clauses are added to the model to restrict the correct clauses to the ones overlapping with the epsilon ball around the input.
- Some auxiliary clauses are also added to help the solver to prune the cluases. For instance if the input is less than x all the clauses that are greater than x are pruned.
- Using these cluases two methods are used to find the solutions: 1) soft solver, which uses MAX-SAT that assigns the values that maximize the value of the satisfied cluases. 2) hard solver, which uses SAT. To be able to solve SAT, we need to implement the ensembling logic, i.e., we implement the summation of the trees and in XGBoost we compute the positive and negative weights of the trees.