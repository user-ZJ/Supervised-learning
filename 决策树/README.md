# 决策树
## 熵（Entropy）
表示粒子移动的自由度，熵越大，自由度越高
![信息](image/entropy.png)

## 信息熵
信息熵（information_entropy）是度量样本集合纯度最常用的指标
![信息熵](image/information_entropy.png)
**信息熵值越小，样本纯度越高**

## 信息增益（information gain）
分类前的信息熵减去分类后的信息熵
假定离散属性a有V个可能的取值，若使用a来对样本集D进行划分，则会产生V个分支节点，其中第v个分支节点包含了D中所有在属性a上取值为av的样本，记为![DV](image/Dv.png)，![|DV|](image/Dv_abs.png)为![DV](image/Dv.png)中样本数，信息增益公式如下：
![信息增益](image/information_gain.png)
**信息增益越大，使用属性a来进行划分所获得的纯度提升越大**

## 交叉熵
交叉熵作为损失函数可以衡量两个分布的相似性

## 条件熵

## ID3决策树
ID3算法(Iterative Dichotomiser 3，迭代二叉树3代)是一种贪心算法，用来构造决策树。ID3算法起源于概念学习系统（CLS），以信息熵的下降速度为选取测试属性的标准，即在每个节点选取还尚未被用来划分的具有最高信息增益的属性作为划分标准，然后继续这个过程，直到生成的决策树能完美分类训练样例

## 增益率

## C4.5决策树

## 连续与缺失值处理

## 随机森林(Random Forest)
随机森林就是通过集成学习的思想将多棵树集成的一种算法，它的基本单元是决策树，而它的本质属于机器学习的一大分支——集成学习（Ensemble Learning）方法
从直观角度来解释，每棵决策树都是一个分类器（假设现在针对的是分类问题），那么对于一个输入样本，N棵树会有N个分类结果。而随机森林集成了所有的分类投票结果，将投票次数最多的类别指定为最终的输出，这就是一种最简单的 Bagging 思想

## 决策树中的超参数
1. 最大深度
2. 每片叶子最小样本数
3. 每次分裂的最小样本数
4. 最大特征数
有时，我们会遇到特征数量过于庞大，而无法建立决策树的情况。在这种状况下，对于每一个分裂，我们都需要检查整个数据集中的每一个特征。这种过程极为繁琐。而解决方案之一是限制每个分裂中查找的特征数。如果这个数字足够庞大，我们很有可能在查找的特征中找到良好特征（尽管也许并不是完美特征）。然而，如果这个数字小于特征数，这将极大加快我们的计算速度

## sklearn中使用决策树
```
# Import statements 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# Read the data.
data = np.asarray(pd.read_csv('data.csv', header=None))
# Assign the features to the variable X, and the labels to the variable y. 
X = data[:,0:2]
y = data[:,2]

# TODO: Create the decision tree model and assign it to the variable model.
# You won't need to, but if you'd like, play with hyperparameters such
# as max_depth and min_samples_leaf and see what they do to the decision
# boundary.
model = DecisionTreeClassifier()

# TODO: Fit the model.
model.fit(X, y)

# TODO: Make predictions. Store them in the variable y_pred.
y_pred = model.predict(X)


# TODO: Calculate the accuracy and assign it to the variable acc.
acc = accuracy_score(y, y_pred)
```

## 决策树实际应用
查看Titanic Solutions-zh.ipynb