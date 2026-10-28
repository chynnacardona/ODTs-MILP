# Section 5, table 1: Ground truth depth (D = 1, 2, 3, 4, 5), no noise, training size of 100

import sys
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.GreedyCART import greedy_cart
from core.OCT_standard import StandardOCT

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# For Greedy CART metrics
def cart_metrics(node, X_test): 
    leaf_depths = []

    def traverse(cur_node, cur_depth=0):
        if cur_node.value is not None or (cur_node.left is None and cur_node.right is None):
            leaf_depths.append(cur_depth)
            return
        if cur_node.left:
            traverse(cur_node.left, cur_depth + 1)
        if cur_node.right:
            traverse(cur_node.right, cur_depth + 1)

    traverse(node, 0)

    tree_size = len(leaf_depths)
    max_depth = int(np.max(leaf_depths)) if leaf_depths else 0

    def predict_sample(cur_node, x, depth=0):
        if cur_node.value is not None or (cur_node.left is None and cur_node.right is None):
            return cur_node.value, depth
        if x[cur_node.feature] <= cur_node.threshold:
            return predict_sample(cur_node.left, x, depth + 1)
        else:
            return predict_sample(cur_node.right, x, depth + 1)

    predictions = []
    for x in X_test:
        pred, depth = predict_sample(node, x)
        predictions.append(pred)

    return {
        "tree_size": tree_size,
        "max_depth": max_depth,
        "predictions": np.array(predictions)
    }

# For Standard OCT dynamic structure metrics
def oct_dynamic_metrics(oct_inst, X_test):
    preds = oct_inst.predict(X_test)
    tree_s = oct_inst.tree_structure_

    def get_active_leaf_depths(node_id=0, current_depth=0):
        if node_id not in tree_s:
            return [current_depth]
        if "l" in tree_s[node_id]:
            return [current_depth]
        
        node_info = tree_s[node_id]
        if not node_info.get("d", False):
            return [current_depth]
        
        left_child = 2 * node_id + 1
        right_child = 2 * node_id + 2
        depths = []
        if left_child in tree_s:
            depths.extend(get_active_leaf_depths(left_child, current_depth + 1))
        if right_child in tree_s:
            depths.extend(get_active_leaf_depths(right_child, current_depth + 1))
        return depths if depths else [current_depth]

    leaf_depths = get_active_leaf_depths(0, 0)
    tree_size = len(leaf_depths)
    max_depth = int(np.max(leaf_depths)) if leaf_depths else 0

    return {
        "tree_size": tree_size,
        "max_depth": max_depth,
        "predictions": preds
    }

class GroundTruth: 
    def __init__(self, depth, num_features=2, seed=42):
        np.random.seed(seed)
        self.depth = depth
        self.num_features = num_features
        self.tree = self._build_node(0)

    def _build_node(self, cur_depth):
        if cur_depth == self.depth:
            return {'is_leaf': True, 'label': np.random.choice([0, 1])}

        feat = np.random.randint(0, self.num_features)
        thresh = np.round(np.random.uniform(-0.5, 0.5), 2)
        left = self._build_node(cur_depth + 1)
        right = self._build_node(cur_depth + 1)

        if left.get('is_leaf') and right.get('is_leaf') and left['label'] == right['label']:
            right['label'] = 1 - left['label']

        return {'is_leaf': False, 'feature': feat, 'threshold': thresh, 'left': left, 'right': right}

    def predict_sample(self, x, node=None):
            if node is None: 
                node = self.tree
            if node['is_leaf']: 
                return node['label']
            if x[node['feature']] <= node['threshold']:
                return self.predict_sample(x, node['left'])
            return self.predict_sample(x, node['right'])

    def generate_dataset(self, n_samples):
        X = np.random.uniform(-1, 1, size=(n_samples, self.num_features))
        y = np.array([self.predict_sample(x) for x in X])
        return X, y

def run_benchmark():
    target_depths = [1, 2, 3, 4, 5]
    n_seeds = 30  

    print("=" * 115)
    print(f"{'True D':<8} | {'Method':<20} | {'In-Sample Acc (%)':<18} | {'Out-Sample Acc (%)':<18} | {'Tree Size':<12} | {'Max D':<8}")
    print("=" * 115)

    for D in target_depths:
        metrics = {
            'CART-depth': {'in': [], 'out': [], 'size': [], 'max_d': []},
            'CART': {'in': [], 'out': [], 'size': [], 'max_d': []},
            'OCT (Standard)': {'in': [], 'out': [], 'size': [], 'max_d': []},
            'Ground Truth': {'in': [], 'out': [], 'size': [], 'max_d': []}
        }

        for seed_offset in range(n_seeds):
            current_seed = 42 + (D * 100) + seed_offset
            gt = GroundTruth(depth=D, num_features=2, seed=current_seed)
            X_train, y_train = gt.generate_dataset(100)
            
            # Dynamic test set size formula capped at 4000 to prevent depth 5 memory/lag bottlenecks[cite: 3]
            test_samples = min(int((2 ** D) * (gt.num_features - 1) * 500), 4000)
            X_test, y_test = gt.generate_dataset(test_samples)

            # 1. Custom Greedy CART-depth
            cart_root = greedy_cart(X_train, y_train, current_depth=0, max_depth=D)
            cart_m = cart_metrics(cart_root, X_test)
            tr_preds = [cart_metrics(cart_root, np.array([x]))['predictions'][0] for x in X_train]
            metrics['CART-depth']['in'].append(np.mean(np.array(tr_preds) == y_train) * 100)
            metrics['CART-depth']['out'].append(np.mean(cart_m['predictions'] == y_test) * 100)
            metrics['CART-depth']['size'].append(cart_m['tree_size'])
            metrics['CART-depth']['max_d'].append(cart_m['max_depth'])

            # 2. Standard Scikit-Learn CART (Unconstrained)
            sk_cart = DecisionTreeClassifier(random_state=current_seed).fit(X_train, y_train)
            metrics['CART']['in'].append(sk_cart.score(X_train, y_train) * 100)
            metrics['CART']['out'].append(sk_cart.score(X_test, y_test) * 100)
            is_leaf = (sk_cart.tree_.children_left == -1) & (sk_cart.tree_.children_right == -1)
            metrics['CART']['size'].append(int(np.sum(is_leaf)))
            metrics['CART']['max_d'].append(sk_cart.tree_.max_depth)

            # 3. Standard OCT (Gurobi MILP with 15s cap per seed)
            oct_std = StandardOCT(depth=D, alpha=0.0001, time_limit=15, verbose=False)
            oct_std.fit(X_train, y_train)
            oct_m = oct_dynamic_metrics(oct_std, X_test)
            
            metrics['OCT (Standard)']['in'].append(np.mean(oct_std.predict(X_train) == y_train) * 100)
            metrics['OCT (Standard)']['out'].append(np.mean(oct_m['predictions'] == y_test) * 100)
            metrics['OCT (Standard)']['size'].append(oct_m['tree_size'])
            metrics['OCT (Standard)']['max_d'].append(oct_m['max_depth'])

            # 4. Ground Truth
            metrics['Ground Truth']['in'].append(100.0)
            metrics['Ground Truth']['out'].append(100.0)
            metrics['Ground Truth']['size'].append(2**D)
            metrics['Ground Truth']['max_d'].append(D)

        # Print aggregated results as Mean ± Std Dev
        for method in ['CART-depth', 'CART', 'OCT (Standard)', 'Ground Truth']:
            in_mean, in_std = np.mean(metrics[method]['in']), np.std(metrics[method]['in'])
            out_mean, out_std = np.mean(metrics[method]['out']), np.std(metrics[method]['out'])
            size_mean, size_std = np.mean(metrics[method]['size']), np.std(metrics[method]['size'])
            max_mean = np.mean(metrics[method]['max_d'])

            in_str = f"{in_mean:.2f} ± {in_std:.2f}"
            out_str = f"{out_mean:.2f} ± {out_std:.2f}"
            size_str = f"{size_mean:.1f} ± {size_std:.1f}"

            print(f"{D:<8} | {method:<20} | {in_str:<18} | {out_str:<18} | {size_str:<12} | {max_mean:<8.1f}")
            D = "" 
        print("-" * 115)

if __name__ == "__main__":
    run_benchmark()