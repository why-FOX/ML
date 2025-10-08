"""
改进点:
数据采样使用有放回采样 Bootstrap
限制没棵树只考虑部分特征中的最佳分割点
树的参数是完整训练集，无min_leaf限制

"""

from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import time
import sys

#读取数据
data = pd.read_csv("../playground-series-s4e5/train.csv")

#删掉"id"列表
data.drop("id",axis=1,inplace=True)

#获取特征列名和目标列名
target = "FloodProbability"
features = [col for col in data.columns if col!= target]

# #增加聚合特征
# data['fsum'] = data[features].sum(axis=1)
# data['fmean'] = data[features].mean(axis=1)
# data['fstd'] = data[features].std(axis=1)
# data['fmax'] = data[features].max(axis=1)
# data['fmin'] = data[features].min(axis=1)
# data['frange'] = data['fmax'] - data['fmin']
# features.extend(['fsum', 'fmean', 'fstd', 'fmax', 'fmin', 'frange'])
#增加聚合特征
data['fsum'] = data[features].sum(axis = 1)
features.append('fsum')
# 进度条函数


def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')

def r2_score(groundtruth, prediction):
    y_true = np.array(groundtruth)
    y_pred = np.array(prediction)

    y_mean = np.mean(y_true)
    ss_total = np.sum((y_true - y_mean) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)

    if ss_total == 0:
        if ss_residual == 0:
            return 1.0
        else:
            return 0.0
    r2 = 1 - (ss_residual / ss_total)
    return float(r2)

#k折交叉验证
"""
model:模型
data:数据集
features:特征
target:目标
n:交叉验证次数
shuffle:是否打乱顺序
random_state:随机种子,确保可复现

注:每次交叉验证只训练一次,测试一次
"""
def cross_validation(model,data,features,target,n,shuffle=True,random_state = 1):
    scores = []

    kf = KFold(n_splits=n,shuffle = shuffle,random_state=random_state)
    print(f"\n开始{n}折交叉验证...")
    for k,(train_idx,test_idx) in enumerate(kf.split(data)):
        print_progress_bar(k+1, n, prefix=f'交叉验证进度:', suffix=f'第{k+1}/{n}折', length=30)

        x_train = data.iloc[train_idx][features]
        y_train = data.iloc[train_idx][target]
        x_test = data.iloc[test_idx][features]
        y_test = data.iloc[test_idx][target]

        print(f"  训练第{k+1}折模型...")
        model.fit(x_train,y_train)
        print(f"  预测第{k+1}折验证集...")
        y_pred = model.predict(x_test)

        score = r2_score(y_test,y_pred)
        scores.append(score)
        print(f"  第{k+1}折R²分数: {score:.4f}")


    return scores,sum(scores)/len(scores)

class RandomForest(object):
    def __init__(self, nr_trees=100, sample_sz=None, max_features='sqrt', min_leaf=5, max_depth=None, random_state=None):
        self.nr_trees = nr_trees
        self.sample_sz = sample_sz
        self.max_features = max_features
        self.min_leaf = min_leaf
        self.max_depth = max_depth
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
        self.trees = [self.create_tree() for i in range(self.nr_trees)]

    def get_sample_data(self):
        # Bootstrap采样：有放回采样，采样大小=原始数据集大小
        n_samples = self.sample_sz if self.sample_sz is not None else len(self.x)
        return np.random.choice(len(self.x), n_samples, replace=True)
    

    def create_tree(self):
        return DecesionTree(min_leaf=self.min_leaf, max_depth=self.max_depth,
                          max_features=self.max_features)
    
    def single_tree_fit(self,tree):
        idxs = self.get_sample_data()
        return tree.fit(self.x.iloc[idxs],self.y.iloc[idxs])
    
    def fit(self,x,y):
        self.x = x
        self.y = y
        for i, tree in enumerate(self.trees):
            self.single_tree_fit(tree)

    def predict(self,x):
        all_tree_preds = np.stack([tree.predict(x) for tree in self.trees])
        return np.mean(all_tree_preds,axis = 0)

class DecesionTree(object):
    def __init__(self, min_leaf=1, max_depth=None, max_features='sqrt', current_depth=0):
        self.min_leaf = min_leaf   #决策树中每次分裂两节点中至少特征个数
        self.max_depth = max_depth #
        self.max_features = max_features
        self.current_depth = current_depth
        self.score = float('inf')
    def fit(self,x,y):
        self.x = x
        self.y = y
        self.features = self.x.columns
        self.sample_size = len(x)
        self.val = np.mean(y)
        self.find_best_split()
        return self


    def find_best_split(self):
        # 特征随机选择
        n_features = len(self.features)
        if self.max_features == 'sqrt':
            n_selected = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            n_selected = int(np.log2(n_features))
        elif isinstance(self.max_features, float):
            n_selected = int(self.max_features * n_features)
        else:
            n_selected = min(n_features, self.max_features)

        # 随机选择特征子集
        selected_features = np.random.choice(n_features, n_selected, replace=False)

        #在随机选择的若干特征中找到最优的一个分裂特征，及其分裂值
        for col_idx in selected_features:
            self.find_col_best_split_point(col_idx)
        #当分裂效果不好 或分裂深度大于其指定深度时，该决策树停止分裂
        if self.is_leaf or (self.max_depth is not None and self.current_depth >= self.max_depth):
            return

        # 确保找到了有效的分割点
        if not hasattr(self, 'split_point'):
            return

        left_child_sample_idxs = np.nonzero(self.split_col <= self.split_point)[0]
        right_child_sample_idxs = np.nonzero(self.split_col > self.split_point)[0]

        # 检查分割后的样本数量
        if len(left_child_sample_idxs) < self.min_leaf or len(right_child_sample_idxs) < self.min_leaf:
            return

        try:
            self.left_child_tree = (DecesionTree(self.min_leaf, self.max_depth, self.max_features, self.current_depth + 1)
                                     .fit(self.x.iloc[left_child_sample_idxs], self.y.iloc[left_child_sample_idxs]))
            self.right_child_tree = (DecesionTree(self.min_leaf, self.max_depth, self.max_features, self.current_depth + 1)
                                      .fit(self.x.iloc[right_child_sample_idxs], self.y.iloc[right_child_sample_idxs]))
        except Exception as e:
            # 如果创建子节点失败，保持为叶子节点
            print(f"Warning: Failed to create child nodes: {e}")
            return




    def find_col_best_split_point(self,col_idx):
        x_col = self.x.values[:,col_idx]
        sorted_idxs = np.argsort(x_col)
        sorted_x_col = x_col[sorted_idxs]
        sorted_y = self.y.iloc[sorted_idxs].values
        
        lchild_n_samples = 0
        lchild_y_sum  = 0.0
        lchild_y_square_sum = 0.0

        rchild_n_samples = self.sample_size
        rchild_y_sum = sorted_y.sum()
        rchild_y_square_sum = (sorted_y ** 2).sum()

        node_y_sum = rchild_y_sum
        node_y_square_sum = rchild_y_square_sum 

        for i in range(0, self.sample_size - self.min_leaf):
            xi, yi = sorted_x_col[i], sorted_y[i]
            
            rchild_n_samples -= 1
            rchild_y_sum -= yi
            rchild_y_square_sum -= (yi ** 2)
            
            lchild_n_samples  +=  1
            lchild_y_sum += yi
            lchild_y_square_sum += (yi ** 2)

            if i< self.min_leaf or xi == sorted_x_col[i+1]:
                continue
            #计算 不纯度 平方平均误差 越小越好
            lchild_impurity = self.calc_mse_inpurity(lchild_y_square_sum,lchild_y_sum, lchild_n_samples)
            rchild_impurity = self.calc_mse_inpurity(rchild_y_square_sum,rchild_y_sum, rchild_n_samples)
            split_score = (lchild_n_samples * lchild_impurity + rchild_n_samples * rchild_impurity) / self.sample_size

            if split_score < self.score:
                self.score = split_score
                self.split_point = xi
                self.split_col_idx = col_idx


    def calc_mse_inpurity(self,y_squared_sum,y_sum,n_y):
        return (y_squared_sum / n_y) - (y_sum / n_y) ** 2

    def predict(self, x):
        if type(x) == pd.DataFrame:
            x = x.values
        return np.array([self.predict_row(row) for row in x])
    

    #以最终分裂后节点的均值作为预测结果
    def predict_row(self, row):
        if self.is_leaf:
            return self.val

        # 安全检查：确保子节点存在
        if not hasattr(self, 'split_col_idx') or not hasattr(self, 'left_child_tree'):
            return self.val  # 如果子节点不存在，返回当前节点值

        # 安全检查：确保right_child_tree也存在
        if not hasattr(self, 'right_child_tree'):
            return self.val

        t = (self.left_child_tree if row[self.split_col_idx] <= self.split_point else self.right_child_tree)
        return t.predict_row(row)

    @property
    def split_name(self):
        return self.features[self.split_col_idx]


    @property
    def is_leaf(self):
        return self.score == float('inf') or not hasattr(self, 'split_point')
    
    @property
    def split_col(self):
        return self.x.iloc[:,self.split_col_idx].values
  
    def __repr__(self):
        pr =  f'sample: {self.sample_size}, value: {self.val}\r\n'
        if not self.is_leaf:
            pr += f'split column: {self.split_name}, \
                split point: {self.split_point}, score: {self.score} '
        return pr    
    
def main():
    print("开始训练洪水预测模型...")
    print(f"数据集大小: {data.shape}")
    print(f"特征数量: {len(features)}")
    print(f"目标变量: {target}")
    model = RandomForest(nr_trees=100, sample_sz=20000, max_features='sqrt',
                        min_leaf=30, max_depth=15, random_state=42)
    scores, avg_score = cross_validation(model, data, features, target, n=5, random_state=42)
    print(f"\n交叉验证结果:")
    print(f"   各折R²分数: {[f'{score:.4f}' for score in scores]}")
    print(f"   平均R²分数: {avg_score:.4f}")
    return

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()

    print(f"\n模型训练完成！")
    print(f"总用时: {end_time - start_time:.2f}秒")
