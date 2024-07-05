# 机器学习理论课课程设计
## 一 数据集及其来源
1. 帕尔默群岛（南极洲）企鹅数据
   - 数据文件:data/penguins/penguins_size.csv
   - 来源: [Kaggle - Archipelago Antarctica Penguin Data](https://www.kaggle.com/datasets/parulpandeypalmer-archipelago-antarctica-penguin-data)

2. 酒数据集
   - 数据文件:data/wine/wine.data
   - 来源:[UCI Machine Learning Repository - Wine](https://archive.ics.uci.edu/dataset/109/wine)

3. 电离层数据集
   - 数据文件:data/ionosphere/ionosphere.data
   - 来源:[UCI Machine Learning Repository - Ionosphere](https://archive.ics.uci.edu/dataset/52/ionosphere)

4. 鸢尾花数据集
   - 数据文件:data/iris/iris.data
   - 来源:[UCI Machine Learning Repository - iris](https://archive.ics.uci.edu/dataset/53/iris)

5. 皮肤病数据集
   - 数据文件:data/dermatology/dermatology.data
   - 来源:[UCI Machine Learning Repository - dermatology](https://archive.ics.uci.edu/dataset/33/dermatology)

## 二、文件说明
- **data文件夹**：本文件夹用于存放所有原始数据集文件。每个数据集有其独立的子文件夹
  - `data/penguins/` - 存放帕尔默群岛企鹅数据集。
  - `data/wine/` - 存放酒数据集。
  - `data/ionosphere/` - 存放电离层数据集。
  - `data/iris/` - 存放鸢尾花数据集。
  - `data/dermatology/` - 存放皮肤病数据集。
- **model文件夹**：此文件夹用于存放模型文件
  - `model/naive_bayes_classifier.py` - 朴素贝叶斯模型
  - `model/network.py` - 神经网络模型
  - `random_forest.py` - 随机森林模型
- **train-code文件夹**：该部分包含用于训练和测试机器学习模型的代码
  - `train-code/penguins/` - 存放帕尔默群岛企鹅训练文件
  - `train-code/wine/` - 存放酒训练文件
  - `train-code/ionosphere/` - 存放电离层训练文件
  - `train-code/iris/` - 存放鸢尾花训练文件
  - `train-code/dermatology/` - 存放皮肤病训练文件

## 三、训练模型
1. **朴素贝叶斯 (Naive Bayes)**
    - **原理**：朴素贝叶斯是一种基于贝叶斯定理的分类方法。它的核心假设是特征条件独立性，即给定类别标签时，一个特征出现的概率与其它特征的出现概率是相互独立的。
    - **应用**：由于其实现简单且在处理大量特征时计算效率高，朴素贝叶斯在文本分类、垃圾邮件识别等领域有着广泛的应用。
    - **优点**：计算简便，对于小样本数据表现良好。
    - **缺点**：特征独立性的假设在现实中往往不成立，可能会影响分类的准确性。
2. **神经网络 (Neural Networks, NN)**
    - **结构**：神经网络由多个层组成，包括输入层、一个或多个隐藏层以及输出层。每一层由多个神经元组成，神经元之间通过权重连接。
    - **功能**：通过学习输入数据与输出数据之间的映射关系，神经网络能够捕捉复杂的非线性模式。
    - **应用**：在图像识别、语音识别和自然语言处理等领域有着广泛的应用。
    - **优点**：强大的非线性拟合能力，能够处理高维数据。
    - **缺点**：需要大量的数据进行训练，容易过拟合。

3. **随机森林 (Random Forest)**
    - **原理**：随机森林是一种集成学习方法，它构建多个决策树并将它们的预测结果进行汇总，以提高模型的准确性和鲁棒性。
    - **特点**：在训练每个决策树时，随机森林通过随机选择样本和特征来增加模型的多样性，减少过拟合。
    - **应用**：由于其易于实现、训练速度快且通常不需要太多的调参，随机森林在许多分类和回归问题中都非常流行。
    - **优点**：模型稳定性高，对数据的适应性强。
    - **缺点**：模型可能会变得相对复杂，需要较多的解释工作。

4. **Scikit-learn GaussianNB (朴素贝叶斯实现)**
    - **实现**：`GaussianNB`是Scikit-learn库中实现的高斯朴素贝叶斯分类器，可以用于分类任务。

5. **Scikit-learn MLPClassifier (神经网络实现)**
    - **实现**：`MLPClassifier`是Scikit-learn库中实现的多层感知器分类器，它是一个基本的前馈神经网络，可以用于分类任务。

6. **Scikit-learn RandomForestClassifier (随机森林实现)**
    - **实现**：`RandomForestClassifier`是Scikit-learn库中实现的随机森林分类器，它构建多个决策树并进行投票或平均以得到最终的预测结果。

## 四、需求库
```
numpy==1.24.3
scikit-learn==1.3.0
scipy==1.11.1
pandas==2.0.3
```

## 五、训练结果
|                | dermatology | ionosphere | iris  | penguins | wine  |
|----------------|-------------|------------|-------|----------|-------|
| 朴素贝叶斯          | 0.875       | 0.843      | 0.867 | 0.881    | 0.917 |
| 朴素贝叶斯(sklearn) | 0.875       | 0.843      | 0.867 | 0.895    | 0.917 |
| 神经网络           | 0.986       | 0.871      | 0.933 | 0.970    | 0.861 |
| 神经网络(sklearn)  | 1.000       | 0.914      | 0.933 | 1.000    | 0.917 |
| 随机森林           | 0.9722      | 0.871      | 0.900 | 1.000    | 0.889 |
| 随机森林(sklearn)  | 0.9861      | 0.914      | 0.900 | 1.000    | 0.944 |