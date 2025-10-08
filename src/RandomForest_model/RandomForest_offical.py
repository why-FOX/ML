import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import time
import sys

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

#读取数据
data = pd.read_csv("../playground-series-s4e5/train.csv")

#删掉"id"列表
data.drop("id",axis=1,inplace=True)


#获取特征列名和目标列名
target = "FloodProbability"
features = [col for col in data.columns if col!= target]

#增加聚合特征
data['fsum'] = data[features].sum(axis = 1)
features.append('fsum')





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

    print_progress_bar(n, n, prefix='交叉验证进度:', suffix='完成!', length=30)
    return scores,sum(scores)/len(scores)



def main():
    print("开始训练洪水预测模型...")
    print(f"数据集大小: {data.shape}")
    print(f"特征数量: {len(features)}")
    print(f"目标变量: {target}")

    # 创建随机森林回归器实例
    model = RandomForestRegressor(n_estimators=100, random_state=42, min_samples_leaf = 100,n_jobs=-1)

    # 进行5折交叉验证
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
