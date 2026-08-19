#!/usr/bin/env python3
"""Module that builds a decision tree."""
import numpy as np


class Node:
    """Represents a decision node."""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Initialize a Node."""
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """Return the maximum depth below this node."""
        left_depth = self.left_child.max_depth_below()
        right_depth = self.right_child.max_depth_below()
        return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        """Count nodes below this node."""
        if only_leaves:
            return (self.left_child.count_nodes_below(only_leaves=True) +
                    self.right_child.count_nodes_below(only_leaves=True))
        else:
            return (1 +
                    self.left_child.count_nodes_below(only_leaves=False) +
                    self.right_child.count_nodes_below(only_leaves=False))

    def left_child_add_prefix(self, text):
        """Add prefix for left child."""
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for x in lines[1:]:
            if x.strip():
                new_text += "    |" + x + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """Add prefix for right child."""
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for x in lines[1:]:
            if x.strip():
                new_text += "    " + x + "\n"
        return new_text

    def __str__(self):
        """Return a string representation of the node."""
        if self.is_root:
            result = (f"root [feature={self.feature}, "
                      f"threshold={self.threshold}]\n")
        else:
            result = (f"node [feature={self.feature}, "
                      f"threshold={self.threshold}]\n")
        if self.left_child:
            result += self.left_child_add_prefix(
                self.left_child.__str__())
        if self.right_child:
            result += self.right_child_add_prefix(
                self.right_child.__str__())
        return result

    def get_leaves_below(self):
        """Return list of all leaves below this node."""
        return (self.left_child.get_leaves_below() +
                self.right_child.get_leaves_below())

    def update_bounds_below(self):
        """Recursively compute lower and upper bounds for each node."""
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -1 * np.inf}

        for child in [self.left_child, self.right_child]:
            child.lower = self.lower.copy()
            child.upper = self.upper.copy()
            if child == self.left_child:
                child.lower[self.feature] = self.threshold
            else:
                child.upper[self.feature] = self.threshold

        for child in [self.left_child, self.right_child]:
            child.update_bounds_below()

    def update_indicator(self):
        """Compute indicator function from lower and upper bounds."""
        def is_large_enough(x):
            return np.array(
                [x[:, key] > self.lower[key]
                 for key in self.lower.keys()]).all(axis=0)

        def is_small_enough(x):
            return np.array(
                [x[:, key] <= self.upper[key]
                 for key in self.upper.keys()]).all(axis=0)

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x),
                      is_small_enough(x)]), axis=0)

    def pred(self, x):
        """Predict class for a single individual."""
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


class Leaf(Node):
    """Represents a leaf node."""

    def __init__(self, value, depth=None):
        """Initialize a Leaf."""
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Return the depth of this leaf."""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """Return 1 as a leaf counts as one node."""
        return 1

    def __str__(self):
        """Return a string representation of the leaf."""
        return f"-> leaf [value={self.value}]"

    def get_leaves_below(self):
        """Return list containing this leaf."""
        return [self]

    def update_bounds_below(self):
        """Pass for leaf nodes."""
        pass

    def pred(self, x):
        """Return the value of this leaf."""
        return self.value


class Decision_Tree():
    """Represents a decision tree."""

    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        """Initialize a Decision_Tree."""
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """Return the maximum depth of the tree."""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Return the number of nodes in the tree."""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """Return a string representation of the tree."""
        return self.root.__str__()

    def get_leaves(self):
        """Return list of all leaves in the tree."""
        return self.root.get_leaves_below()

    def update_bounds(self):
        """Update bounds for all nodes."""
        self.root.update_bounds_below()

    def update_predict(self):
        """Compute the prediction function."""
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()
        self.predict = lambda A: np.sum(
            np.array([leaf.indicator(A) * leaf.value
                      for leaf in leaves]), axis=0)

    def pred(self, x):
        """Predict class for a single individual."""
        return self.root.pred(x)

    def np_extrema(self, arr):
        """Return min and max of array."""
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        """Return a random feature and threshold for splitting."""
        diff = 0
        while diff == 0:
            feature = self.rng.integers(0, self.explanatory.shape[1])
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population])
            diff = feature_max - feature_min
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def fit_node(self, node):
        """Fit a node of the decision tree."""
        node.feature, node.threshold = self.split_criterion(node)

        left_population = (node.sub_population &
                           (self.explanatory[:, node.feature] >
                            node.threshold))
        right_population = (node.sub_population &
                            ~left_population)

        is_left_leaf = (node.depth == self.max_depth - 1 or
                        np.sum(left_population) <= self.min_pop or
                        len(np.unique(self.target[left_population])) == 1)

        if is_left_leaf:
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        is_right_leaf = (node.depth == self.max_depth - 1 or
                         np.sum(right_population) <= self.min_pop or
                         len(np.unique(self.target[right_population])) == 1)

        if is_right_leaf:
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        """Return a leaf child node."""
        value = np.bincount(self.target[sub_population]).argmax()
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.subpopulation = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        """Return a node child."""
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def fit(self, explanatory, target, verbose=0):
        """Fit the decision tree."""
        if self.split_criterion == "random":
            self.split_criterion = self.random_split_criterion
        else:
            self.split_criterion = self.Gini_split_criterion

        self.explanatory = explanatory
        self.target = target
        self.root.sub_population = np.ones_like(self.target, dtype='bool')

        self.fit_node(self.root)
        self.update_predict()

    if verbose == 1:
        print(f"""  Training finished.
    - Depth                     : {self.depth()}
    - Number of nodes           : {self.count_nodes()}
    - Number of leaves          : {self.count_nodes(only_leaves=True)}
    - Accuracy on training data : {self.accuracy(self.explanatory, self.target)}""")

    def accuracy(self, test_explanatory, test_target):
        """Return accuracy of the model."""
        return np.sum(np.equal(self.predict(test_explanatory),
                               test_target)) / test_target.size
