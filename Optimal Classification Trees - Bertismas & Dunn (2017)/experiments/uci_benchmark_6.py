import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.OCT_standard import StandardOCT
from core.OCT_Hyperlane import OCTH
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

def get_uci_datasets():
    return [
        ("Iris", load_iris().data, load_iris().target),
        ("Breast-cancer-diagnostic", load_breast_cancer().data, load_breast_cancer().target),
        ("Wine", load_wine().data, load_wine().target)
    ]

def run_table6():
    datasets = get_uci_datasets()
    depths = [1, 2, 3, 4]
    
    print("\n" + "="*75)
    print("Table 6 Mean out-of-sample accuracies across all datasets for")
    print("      each method, according to the maximum tree depth")
    print("="*75)
    print(f"{'Max. depth':<12} | {'CART (%)':<10} | {'OCT (%)':<10} | {'OCT-H (%)':<10}")
    print("-" * 50)
    
    for depth in depths:
        all_cart, all_oct, all_octh = [], [], []
        
        for name, X, y in datasets:
            n_samples = X.shape[0]
            n_min_val = max(1, int(0.05 * n_samples))  # N_min = 5% of data[cite: 12]
            
            cart_runs, oct_runs, octh_runs = [], [], []
            for seed in range(5):  # 5 Monte Carlo trials[cite: 12]
                X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
                X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=1/3, random_state=seed)
                
                X_train_val = np.vstack((X_train, X_val))
                y_train_val = np.concatenate((y_train, y_val))
                
                # 1. CART
                cart = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=n_min_val, random_state=seed)
                cart.fit(X_train_val, y_train_val)
                cart_runs.append(accuracy_score(y_test, cart.predict(X_test)) * 100)
                
                # 2. Standard OCT (Univariate)
                oct_model = StandardOCT(depth=depth, alpha=0.0001, time_limit=60, verbose=False)
                oct_model.fit(X_train_val, y_train_val)
                oct_runs.append(accuracy_score(y_test, oct_model.predict(X_test)) * 100)

                # 3. OCT-H (Multivariate Hyperplane)
                octh_model = OCTH(depth=depth, alpha=0.01, time_limit=60, verbose=False)
                octh_model.fit(X_train_val, y_train_val)
                octh_runs.append(accuracy_score(y_test, octh_model.predict(X_test)) * 100)
                
            all_cart.append(np.mean(cart_runs))
            all_oct.append(np.mean(oct_runs))
            all_octh.append(np.mean(octh_runs))
            
        print(f"{depth:<12} | {np.mean(all_cart):6.1f}     | {np.mean(all_oct):6.1f}     | {np.mean(all_octh):6.1f}")
        
    print("="*75)

if __name__ == "__main__":
    run_table6()