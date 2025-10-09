import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

TARGET = "FloodProbability"
N_BINS = 10
LAMBDA_L2 = 1.0

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

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

# 标准化
scaler = StandardScaler()
X_all = scaler.fit_transform(train_df[FEATURES])
X_test = scaler.transform(test_df[FEATURES])
y_all = train_df[TARGET].values.astype(float)

class LightGBM:
    def __init__(self, n_estimators=50, learning_rate=0.05, max_depth=3, min_samples_leaf=10, n_bins=10, lambda_l2=1.0):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_bins = n_bins
        self.lambda_l2 = lambda_l2
        self.trees = []
        self.baseline = 0.0

    def _compute_gradients(self, y, y_pred):
        g = y_pred - y
        h = np.ones_like(y)
        return g, h

    def _find_best_split(self, X, g, h):
        best_gain = -np.inf
        best_feat, best_thresh = None, None
        G_total, H_total = np.sum(g), np.sum(h)

        for f_idx in range(X.shape[1]):
            f_values = X[:, f_idx]
            bins = np.linspace(f_values.min(), f_values.max(), self.n_bins + 1)
            for i in range(1, len(bins)):
                thresh = bins[i]
                left_mask = f_values <= thresh
                right_mask = ~left_mask
                G_L, H_L = g[left_mask].sum(), h[left_mask].sum()
                G_R, H_R = g[right_mask].sum(), h[right_mask].sum()
                gain = (G_L**2)/(H_L + self.lambda_l2) + (G_R**2)/(H_R + self.lambda_l2) - (G_total**2)/(H_total + self.lambda_l2)
                if gain > best_gain:
                    best_gain = gain
                    best_feat = f_idx
                    best_thresh = thresh
        return best_feat, best_thresh

    def _build_tree(self, X, g, h, depth=0):
        node = {}
        if depth >= self.max_depth or len(X) <= self.min_samples_leaf:
            G, H = g.sum(), h.sum()
            node['leaf'] = -G / (H + self.lambda_l2)
            return node

        feat, thresh = self._find_best_split(X, g, h)
        if feat is None:
            G, H = g.sum(), h.sum()
            node['leaf'] = -G / (H + self.lambda_l2)
            return node

        node['feat'] = feat
        node['thresh'] = thresh
        left_mask = X[:, feat] <= thresh
        right_mask = ~left_mask
        node['left'] = self._build_tree(X[left_mask], g[left_mask], h[left_mask], depth+1)
        node['right'] = self._build_tree(X[right_mask], g[right_mask], h[right_mask], depth+1)
        return node

    def _predict_tree(self, x, node):
        if 'leaf' in node:
            return node['leaf']
        if x[node['feat']] <= node['thresh']:
            return self._predict_tree(x, node['left'])
        else:
            return self._predict_tree(x, node['right'])

    def fit(self, X, y):
        y_pred = np.zeros(len(y), dtype=float)
        self.trees = []

        for t in range(self.n_estimators):
            g, h = self._compute_gradients(y, y_pred)
            tree = self._build_tree(X, g, h)
            self.trees.append(tree)
            pred_inc = np.array([self._predict_tree(x, tree) for x in X])
            y_pred += self.lr * pred_inc

    def predict(self, X):
        y_pred = np.zeros(X.shape[0], dtype=float)
        for tree in self.trees:
            pred_inc = np.array([self._predict_tree(x, tree) for x in X])
            y_pred += self.lr * pred_inc
        return y_pred


kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(y_all))
test_preds = []

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_all)):
    print(f"\n=== Fold {fold+1} ===")
    X_tr, X_val = X_all[tr_idx], X_all[val_idx]
    y_tr, y_val = y_all[tr_idx], y_all[val_idx]

    baseline = y_tr.mean()
    model = LightGBM(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        min_samples_leaf=10,
        n_bins=20,
        lambda_l2=0.1
    )
    model.fit(X_tr, y_tr - baseline)

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
