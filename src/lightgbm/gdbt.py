import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler


TARGET = "FloodProbability"


train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")


FEATURES = [c for c in train_df.columns if c not in ["id", TARGET]]

def feature_engineering(df):
    df = df.copy()
    df["env_interact"] = df["MonsoonIntensity"] * df["Deforestation"] * df["ClimateChange"]
    human = ["Deforestation", "Urbanization", "AgriculturalPractices", "Encroachments",
             "IneffectiveDisasterPreparedness", "InadequatePlanning", "PoliticalFactors"]
    df["human_sum"] = df[human].sum(axis=1)
    df["infra_ratio"] = df["DrainageSystems"] / (df["DamsQuality"] + 1e-6)
    return df

train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)
FEATURES = [c for c in train_df.columns if c not in ["id", TARGET]]


scaler = StandardScaler()
X_all = scaler.fit_transform(train_df[FEATURES])
X_test = scaler.transform(test_df[FEATURES])
y_all = train_df[TARGET].values

class SimpleGBDTRegression:
    def __init__(self, n_estimators=200, learning_rate=0.1, max_depth=3, min_samples_leaf=20):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.trees = []
        self.leaf_values = []

    def fit(self, X, y, verbose=True):
        
        y_pred = np.full(len(y), np.mean(y), dtype=float)

        for t in range(self.n_estimators):
            
            residual = y - y_pred
            tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf)
            tree.fit(X, residual)
            leaf_idx = tree.apply(X)
            gamma = {}
            for leaf in np.unique(leaf_idx):
                mask = leaf_idx == leaf
                gamma[leaf] = residual[mask].mean()
            self.trees.append(tree)
            self.leaf_values.append(gamma)
            updates = np.array([gamma[l] for l in leaf_idx])
            y_pred = y_pred + self.learning_rate * updates
            if verbose and ((t+1) % 50 == 0 or t == 0):
                mse = mean_squared_error(y, y_pred)
                print(f"[{t+1}/{self.n_estimators}] MSE={mse:.6f}")

    def predict(self, X):
        pred = np.zeros(X.shape[0], dtype=float)
        for tree, gamma in zip(self.trees, self.leaf_values):
            leaf_idx = tree.apply(X)
            pred += self.learning_rate * np.array([gamma[leaf] for leaf in leaf_idx])
        return pred  


kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(y_all))
test_preds = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_all)):
    print(f"\n=== Fold {fold+1} ===")
    X_tr, X_val = X_all[tr_idx], X_all[val_idx]
    y_tr, y_val = y_all[tr_idx], y_all[val_idx]
    model = SimpleGBDTRegression(n_estimators=200, learning_rate=0.05, max_depth=4, min_samples_leaf=50)
    baseline = np.mean(y_tr)
    model.fit(X_tr, y_tr - baseline, verbose=True)  
    val_pred = baseline + model.predict(X_val)
    test_pred = baseline + model.predict(X_test)
    val_pred = np.clip(val_pred, 0.0, 1.0)
    test_pred = np.clip(test_pred, 0.0, 1.0)

    oof[val_idx] = val_pred
    test_preds.append(test_pred)
    print("Fold MSE:", mean_squared_error(y_val, val_pred), "R2:", r2_score(y_val, val_pred))


print("\nOverall OOF MSE:", mean_squared_error(y_all, oof))
print("Overall OOF R2:", r2_score(y_all, oof))
final_test = np.mean(test_preds, axis=0)
submission = pd.DataFrame({"id": test_df["id"], "FloodProbability": final_test})
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")
