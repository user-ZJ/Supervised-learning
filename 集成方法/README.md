# 集成方法（Ensemble Method）
将弱学习器整合在一起组成强学习器  

## Bagging（自助聚类 bootstrap aggregating）
弱学习器通过投票的方式，决定样本分类结果  

## Boosting
Boosting方法是一种用来提高弱分类算法准确度的方法,这种方法通过构造一个预测函数系列,然后以一定的方式将他们组合成一个预测函数。  
Boosting是一种框架算法，拥有系列算法，如AdaBoost，GradientBoosting，LogitBoost等算法。  
Boosting中所有的弱分类器可以是不同类的分类器.  

### AdaBoost
首先以最高准确率拟合一个学习器，再拟合第二个学习器修正第一个学习器错误，以此类推  
计算出各个弱学习器的模型权重，根据权重进行投票，决定样本的分类结果   

**数据权重**  
将分类错误的数据权重调整为：  （分类正确样本权重之和）/（分类错误样本权重之和）  
**模型权重**  
![](image/model_weight.png)  

## sklearn中使用AdaBoost
```
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
# base_estimator: 弱学习器使用的模型
# n_estimators: 使用的弱学习器的最大数量
 model = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=2), n_estimators = 4)
model.fit(x_train, y_train)
model.predict(x_test)
```
