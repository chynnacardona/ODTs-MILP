import numpy as np


class WarmStartMapper:
    """
    Translates a fitted greedy decision tree (e.g., scikit-learn DecisionTreeClassifier)
    into initial MIP variable values (warm start) for the Optimal Classification Trees (OCT) model.
    """

    def __init__(self, max_depth: int, n_features: int, n_classes: int):
        self.max_depth = max_depth
        self.n_features = n_features
        self.n_classes = n_classes

        # Calculate node sets based on complete binary tree structure
        # Decision nodes: 1 to 2^D - 1
        # Leaf nodes: 2^D to 2^(D+1) - 1
        self.num_nodes = 2 ** (max_depth + 1) - 1
        self.n_branch_nodes = 2**max_depth - 1
        self.branch_nodes = list(range(1, self.n_branch_nodes + 1))
        self.leaf_nodes = list(range(self.n_branch_nodes + 1, self.num_nodes + 1))

    def extract_cart_parameters(self, cart_model, X_train: np.ndarray):
        """
        Extracts structural decision properties from a trained sklearn DecisionTreeClassifier.
        """
        tree_ = cart_model.tree_

        # Variables to populate
        a_init = {
            (t, j): 0.0
            for t in self.branch_nodes
            for j in range(self.n_features)
        }
        b_init = {t: 0.0 for t in self.branch_nodes}
        d_init = {t: 0 for t in self.branch_nodes}
        c_init = {
            (k, t): 0 for t in self.leaf_nodes for k in range(self.n_classes)
        }

        def traverse(cart_node_id: int, mio_node_id: int, current_depth: int):
            if current_depth > self.max_depth:
                return

            # Check if current MIO node is a decision/branch node
            if mio_node_id in self.branch_nodes:
                # If CART node is an internal split node
                if tree_.children_left[cart_node_id] != tree_.children_right[cart_node_id]:
                    feature = tree_.feature[cart_node_id]
                    threshold = tree_.threshold[cart_node_id]

                    d_init[mio_node_id] = 1
                    a_init[(mio_node_id, feature)] = 1.0
                    b_init[mio_node_id] = float(threshold)

                    # Recurse left and right children
                    left_cart = tree_.children_left[cart_node_id]
                    right_cart = tree_.children_right[cart_node_id]
                    traverse(left_cart, 2 * mio_node_id, current_depth + 1)
                    traverse(right_cart, 2 * mio_node_id + 1, current_depth + 1)
                else:
                    # CART pruned early or hit a leaf before max depth; node disabled in MIO
                    d_init[mio_node_id] = 0
                    # Route samples down the left path trivially if pruned early
                    traverse(cart_node_id, 2 * mio_node_id, current_depth + 1)

            # Check if current MIO node is a leaf node
            elif mio_node_id in self.leaf_nodes:
                # Get predicted class count at leaf
                value_counts = tree_.value[cart_node_id][0]
                predicted_class = int(np.argmax(value_counts))
                c_init[(predicted_class, mio_node_id)] = 1

        # Begin traversal from root node (index 1)
        traverse(cart_node_id=0, mio_node_id=1, current_depth=0)

        # Route training samples to establish starting sample assignment z_it
        z_init = self._compute_sample_assignments(X_train, cart_model)

        return {
            "a": a_init,
            "b": b_init,
            "d": d_init,
            "c": c_init,
            "z": z_init,
        }

    def _compute_sample_assignments(self, X_train: np.ndarray, cart_model):
        """
        Maps each training sample to its assigned leaf node in the complete MIO tree structure.
        """
        n_samples = X_train.shape[0]
        z_init = {
            (i, t): 0 for i in range(n_samples) for t in self.leaf_nodes
        }

        # Get the leaf indices where CART places each sample
        leaf_indices = cart_model.apply(X_train.astype(np.float32))

        # Map CART leaf indices to our explicit 1-indexed complete binary tree leaf node IDs
        node_map = {}
        tree_ = cart_model.tree_

        def map_nodes(cart_id: int, mio_id: int):
            if mio_id in self.leaf_nodes or tree_.children_left[cart_id] == tree_.children_right[cart_id]:
                # If MIO reached maximum depth leaf or CART terminates
                target_leaf = mio_id if mio_id in self.leaf_nodes else self._get_leftmost_leaf(mio_id)
                node_map[cart_id] = target_leaf
                return

            map_nodes(tree_.children_left[cart_id], 2 * mio_id)
            map_nodes(tree_.children_right[cart_id], 2 * mio_id + 1)

        map_nodes(cart_id=0, mio_id=1)

        for i in range(n_samples):
            cart_leaf = leaf_indices[i]
            mio_leaf = node_map.get(cart_leaf, self.leaf_nodes[0])
            z_init[(i, mio_leaf)] = 1

        return z_init

    def _get_leftmost_leaf(self, mio_id: int) -> int:
        """Helper to project a non-leaf node down to its leftmost descendant leaf."""
        curr = mio_id
        while curr not in self.leaf_nodes:
            curr = 2 * curr
        return curr