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
            new_text += ("    | " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """Add prefix for right child."""
        lines = text.split("\n")
        new_text = "    +---> " + lines[0] + "\n"
        for x in lines[1:]:
            new_text += ("      " + x) + "\n"
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
        return f"leaf [value={self.value}]"


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
