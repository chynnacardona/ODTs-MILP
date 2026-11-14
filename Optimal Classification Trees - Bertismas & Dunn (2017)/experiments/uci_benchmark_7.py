import sys
from pathlib import Path
import warnings
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.OCT_standard import StandardOCT
from datasets.uci_loader import  load_all_uci_datasets # <--- Imported here!

def run_table7(depth=2):
    datasets =  load_all_uci_datasets()

    if not datasets:
        print("No datasets found in data/ folder!")
        return

    print("\n" + "="*95)
    print("Table 7 Full results for CART and OCT at depth 2")
    print("="*95)
    print(f"{'Dataset':<28} | {'n':<5} | {'p':<4} | {'K':<3} | {'Mean out-of-sample accuracy':<26} | {'Mean improvement':<18}")
    print(f"{'Name':<28} | {'':<5} | {'':<4} | {'':<3} | {'CART':<8} | {'OCT':<8} |")
    print("-" * 95)

    for name, X, y in datasets:
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        n_min_val = max(1, int(0.05 * n_samples))  # N_min = 5% rule
        
        cart_accuracies, oct_accuracies, improvements = [], [], []

        for seed in range(5):  # 5 Monte Carlo trials
            X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
            X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=1/3, random_state=seed)
            
            X_train_val = np.vstack((X_train, X_val))
            y_train_val = np.concatenate((y_train, y_val))

            # CART
            cart = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=n_min_val, random_state=seed)
            cart.fit(X_train_val, y_train_val)
            c_acc = accuracy_score(y_test, cart.predict(X_test)) * 100
            cart_accuracies.append(c_acc)

            # Standard OCT
            oct_model = StandardOCT(depth=depth, alpha=0.0001, time_limit=15, verbose=False)
            oct_model.fit(X_train_val, y_train_val)
            o_acc = accuracy_score(y_test, oct_model.predict(X_test)) * 100
            oct_accuracies.append(o_acc)
            
            improvements.append(o_acc - c_acc)

        mean_cart = np.mean(cart_accuracies)
        mean_oct = np.mean(oct_accuracies)
        imp_mean = np.mean(improvements)
        imp_std = np.std(improvements)
        
        sign_str = "+" if imp_mean >= 0 else ""
        imp_formatted = f"{sign_str}{imp_mean:.2f} ± {imp_std:.2f}"

        print(f"{name:<28} | {n_samples:<5} | {n_features:<4} | {n_classes:<3} | {mean_cart:6.1f}   | {mean_oct:6.1f}   | {imp_formatted:<18}")

    print("="*95)

if __name__ == "__main__":
    run_table7(depth=2)