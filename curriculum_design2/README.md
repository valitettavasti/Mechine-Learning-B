# 项目设计文档

## 一、项目结构

项目文件和目录结构如下：

- `/data/`
  - 存放所有原始数据集文件，包括训练集和测试集。

- `/CNN-number-recognition.py`
  - 卷积神经网络实现的MNIST数字识别模型。

- `/data_conversion.py`
  - 数据预处理脚本，用于将`torchvision`的`datasets`下载的数据转换为适用于`sklearn`的数据格式。

- `/decision_tree.py`
  - 决策树模型实现，使用`sklearn.tree.DecisionTreeClassifier`，针对MNIST数字识别任务。

- `/logistic.py`
  - 逻辑回归模型实现，使用`sklearn.linear_model.LogisticRegression`，用于MNIST数字识别。

- `/SVM.py`
  - 支持向量机模型实现，使用`sklearn.svm.SVC`，用于MNIST数字识别。

# 项目设计文档

## 二、模型参数

### 卷积神经网络模型

#### 网络代码实现：

```python
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 定义卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # 第一卷积层，输入通道1，输出通道64
            nn.ReLU(),                                   # 激活函数
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 第二卷积层，输入通道64，输出通道128
            nn.ReLU(),                                   # 激活函数
            nn.MaxPool2d(2, 2)                            # 池化层，窗口大小和步长都为2
        )
        
        # 定义全连接层
        self.dense = nn.Sequential(
            nn.Linear(14*14*128, 1024),                # 全连接层，输入特征数量为14*14*128，输出1024
            nn.ReLU(),                                   # 激活函数
            nn.Dropout(0.5),                              # Dropout正则化，概率为0.5
            nn.Linear(1024, 10)                          # 最终全连接层，输出10个类别
        )
    
    def forward(self, x):
        # 前向传播函数
        x = self.conv1(x)                # 通过卷积层
        x = x.view(-1, 14*14*128)       # 展平特征图以匹配全连接层的输入
        x = self.dense(x)                # 通过全连接层
        return x
```
卷积层:

  - `self.conv1 = Sequential(...)`: 这是一个Sequential容器，它按顺序包含了一系列层。在这个容器中：
  - `Conv2d(1, 64, kernel_size=3, padding=1)`: 第一个卷积层，它将输入通道数为1（灰度图像）的图像映射到64个特征图，使用3x3的卷积核，并且padding设置为1，以保持特征图的空间尺寸不变。
  - `ReLU()`: 激活函数，引入非线性，帮助模型学习更复杂的特征。
  - `Conv2d(64, 128, kernel_size=3, padding=1)`: 第二个卷积层，将64个输入特征图映射到128个特征图。
  - `MaxPool2d(stride=2, kernel_size=2)`: 池化层，用于降低特征图的空间尺寸，减少参数数量和计算量，同时使特征检测更加鲁棒。

全连接层:

  - `self.dense = Sequential(...)`: 这是另一个Sequential容器，包含全连接层： 
  - `Linear(14 * 14 * 128, 1024)`: 第一个全连接层，将卷积层输出的14x14x128的特征图（在经过池化后）展平后映射到1024维的空间。
  - `ReLU()`: 同上，引入非线性。
  - `Dropout(p=0.5)`: 正则化技术，随机丢弃50%的神经元输出，以防止过拟合。
  - `Linear(1024, 10)`: 最后一个全连接层，将1024维的特征映射到10维，对应于MNIST数据集中的10个类别（0到9的数字）。

#### 超参数配置

在模型训练过程中，超参数的选择对模型性能有着显著影响。以下是一些关键超参数的配置选项：

##### 优化器 (Optimizers)
选择了以下优化器来进行模型的参数更新:

- `torch.optim.Adam`: 一种自适应学习率优化算法，广泛用于各种任务。
- `torch.optim.SGD`: 标准随机梯度下降优化器，带动量项。
- `torch.optim.RMSprop`: 另一种自适应学习率方法，主要在处理非平稳目标时表现良好。
- `torch.optim.Adamax`: Adam的一个变种，使用无限小的梯度进行自适应学习率调整。

##### 学习率步长 (Learning Rate Scheduling)
尝试了不同的步长:

- 0.0001
- 0.001
- 0.005
- 0.01



## 决策树模型

### 实现：
使用 `sklearn.tree.DecisionTreeClassifier` 实现的决策树模型

## 逻辑回归模型

### 实现：
使用 `sklearn.linear_model.LogisticRegression` 实现的逻辑回归模型

## 支持向量机模型

### 实现：
使用 `sklearn.svm.SVC` 实现的支持向量机模型

### 采用参数：
- `kernel='linear'`: 使用线性核。
- `kernel='rbf'`: 使用径向基函数核，适用于非线性问题。
- `kernel='poly'`: 使用多项式核，为数据提供高维空间的映射。

## 三、需要的库
```angular2html
matplotlib==3.9.0
numpy==1.24.3
scikit_learn==1.4.2
torch==2.3.0
torchvision==0.18.0
```

## 四、评价指标

在模型的性能评估中，我选用了以下模型衡量模型效果：

### Loss（损失函数）
损失函数用于衡量模型预测值与实际值之间的差异。常用的损失函数是交叉熵损失（Cross-Entropy Loss）：


$$ \text{cost} = -\sum_{c=1}^{C} y_{o,c} \log(p_{o,c}) $$

其中，$C$是类别的数量，$y_{o,c}$是一个二进制指示器（如果类别$c$是观测样本的分类，则为1，否则为0），$p_{o,c}$是模型预测观测样本属于类别$c$的概率。

### Test Accuracy（测试准确率）
测试准确率是衡量模型在测试集上正确分类的比例：

$$ \text{Test Accuracy} = \frac{\text{正确预测的数量}}{\text{总预测数量}} $$

### Precision（精确度）
精确度是针对每个类别的，表示为模型预测为正类别中实际为正类别的比例：

$$ \text{Precision} = \frac{\text{真正例（TP）}}{\text{真正例（TP）} + \text{假正例（FP）}} $$

### Recall（召回率）
召回率也称为真正例率或灵敏度，表示为实际正类别中被正确预测的比例：

$$ \text{Recall} = \frac{\text{真正例（TP）}}{\text{真正例（TP）} + \text{假负例（FN）}} $$

### F1 Score（F1 分数）
F1 分数是精确度和召回率的调和平均数，用于衡量模型的准确性和完整性的平衡：

$$ F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} $$

F1 分数在精确度和召回率之间取得平衡，是一个综合考虑两者的指标，特别适用于类别不平衡的情况。

这些指标为模型的选择和优化提供了重要的参考，帮助我们全面理解模型的性能。
## 五、训练结果

### 步长
- 模型：`torch.optim.Adam`
- 超参数：步长

|                 | 0.00001  | 0.0001   | 0.0005   | 0.001    | 0.0015   | 0.003    | 0.005    | 0.01     | 
|-----------------|----------|----------|----------|----------|----------|----------|----------|----------|
| Loss:           | 0.0110   | 0.0042   | 0.0023   | 0.0024   | 0.0024   | 0.0051   | 0.0053   | 0.0076   |
| Test Accuracy:  | 91.1100% | 97.3100% | 98.5000% | 98.3700% | 97.7900% | 95.0000% | 95.1200% | 93.8000% |
| Precision:      | 0.9111   | 0.9733   | 0.9849   | 0.9838   | 0.9779   | 0.9504   | 0.9517   | 0.9408   |
| Recall:         | 0.9098   | 0.9729   | 0.9850   | 0.9836   | 0.9777   | 0.9497   | 0.9504   | 0.9376   |
| F1:             | 0.9105   | 0.9731   | 0.9850   | 0.9837   | 0.9778   | 0.9501   | 0.9510   | 0.9392   |

- 结论：在0.0005，0.001附近，各项指标数据较好，当步长变大/变小时，指标数据出现了不同程度的变差，可以理解为过拟合/欠拟合

### 优化器

-超级参数：优化器

参数分别为

| 步长        |                 |    | 
|-----------|-----------------|----|
| Loss      |Test Accuracy    |    |
| Precision |Recall           |F1  |



<table>
 <col width=158 style='mso-width-source:userset;mso-width-alt:5603;width:118pt'>
 <col width=64 span=12 style='width:48pt'>
 <tr height=18 style='height:13.8pt'>
  <td height=18 class=xl65 width=158 style='height:13.8pt;width:118pt'></td>
  <td colspan=3 class=xl65 width=192 style='width:144pt'>0.0001</td>
  <td colspan=3 class=xl65 width=192 style='width:144pt'>0.001</td>
  <td colspan=3 class=xl65 width=192 style='width:144pt'>0.005</td>
  <td colspan=3 class=xl65 width=192 style='width:144pt'>0.01</td>
 </tr>
 <tr height=18 style='height:13.8pt'>
  <td rowspan=2 height=36 class=xl65 style='height:27.6pt'>`torch.optim.Adam`</td>
  <td class=xl65>0.0042</td>
  <td class=xl67>97.31%</td>
  <td></td>
  <td class=xl65>0.0024</td>
  <td class=xl67>98.37%</td>
  <td></td>
  <td class=xl65>0.0053</td>
  <td class=xl67>95.12%</td>
  <td></td>
  <td class=xl65>0.0076</td>
  <td class=xl67>93.80%</td>
  <td></td>
 </tr>
 <tr height=18 style='height:13.8pt'>
  <td height=18 class=xl65 style='height:13.8pt'>0.9733</td>
  <td class=xl65>0.9729</td>
  <td class=xl65>0.9731</td>
  <td class=xl65>0.9838</td>
  <td class=xl65>0.9836</td>
  <td class=xl65>0.9837</td>
  <td class=xl65>0.9517</td>
  <td class=xl65>0.9504</td>
  <td class=xl65>0.951</td>
  <td class=xl65>0.9408</td>
  <td class=xl65>0.9376</td>
  <td class=xl65>0.9392</td>
 </tr>
 <tr height=18 style='height:13.8pt'>
  <td rowspan=2 height=36 class=xl65 style='height:27.6pt'>`torch.optim.SGD`</td>
  <td class=xl65>0.0359</td>
  <td class=xl67>26.23%</td>
  <td class=xl65></td>
  <td class=xl65>0.0322</td>
  <td class=xl67>74.09%</td>
  <td></td>
  <td class=xl65>0.0131</td>
  <td class=xl67>90.43%</td>
  <td></td>
  <td class=xl65>0.0098</td>
  <td class=xl67>90.87%</td>
  <td></td>
 </tr>
 <tr height=18 style='height:13.8pt'>
  <td height=18 class=xl65 style='height:13.8pt'>0.3023</td>
  <td class=xl65>0.2633</td>
  <td class=xl65>0.2814</td>
  <td class=xl65>0.7632</td>
  <td class=xl65>0.7332</td>
  <td class=xl65>0.7479</td>
  <td class=xl65>0.9033</td>
  <td class=xl65>0.9031</td>
  <td class=xl65>0.9032</td>
  <td class=xl65>0.9145</td>
  <td class=xl65>0.9072</td>
  <td class=xl65>0.9108</td>
 </tr>
 <tr height=18 style='height:13.8pt'>
  <td rowspan=2 height=36 class=xl68 style='height:27.6pt'><span
  style='mso-spacerun:yes'>&nbsp;</span>`torch.optim.RMSprop`<span
  style='mso-spacerun:yes'>&nbsp;</span></td>
  <td class=xl65>0.0032</td>
  <td class=xl67>97.86%</td>
  <td class=xl65></td>
  <td class=xl65>0.0053</td>
  <td class=xl67>97.86%</td>
  <td class=xl65></td>
  <td class=xl65>0.1091</td>
  <td class=xl67>96.01%</td>
  <td class=xl65></td>
  <td class=xl65>1.0133</td>
  <td class=xl67>94.26%</td>
  <td class=xl65></td>
 </tr>
 <tr height=18 style='height:13.8pt'>
  <td height=18 class=xl65 style='height:13.8pt'>0.9786</td>
  <td class=xl65>0.9785</td>
  <td class=xl65>0.9785</td>
  <td class=xl65>0.9785</td>
  <td class=xl65>0.9786</td>
  <td class=xl65>0.9786</td>
  <td class=xl65>0.9602</td>
  <td class=xl65>0.9599</td>
  <td class=xl65>0.96</td>
  <td class=xl65>0.9458</td>
  <td class=xl65>0.9418</td>
  <td class=xl65>0.9438</td>
 </tr>
 <tr height=18 style='height:13.8pt'>
  <td rowspan=2 height=36 class=xl65 style='height:27.6pt'>`torch.optim.Adamax`</td>
  <td class=xl65>0.0065</td>
  <td class=xl67>94.23%</td>
  <td class=xl65></td>
  <td class=xl65>0.0027</td>
  <td class=xl66 align=right>98.09%</td>
  <td></td>
  <td class=xl65>0.0051</td>
  <td class=xl67>96.37%</td>
  <td class=xl65></td>
  <td class=xl65>0.008</td>
  <td class=xl67>94.68%</td>
  <td class=xl65></td>
 </tr>
 <tr height=18 style='height:13.8pt'>
  <td height=18 class=xl65 style='height:13.8pt'>0.9422</td>
  <td class=xl65>0.9416</td>
  <td class=xl65>0.9419</td>
  <td class=xl65>0.9809</td>
  <td class=xl65>0.9808</td>
  <td class=xl65>0.9809</td>
  <td class=xl65>0.9639</td>
  <td class=xl65>0.9633</td>
  <td class=xl65>0.9636</td>
  <td class=xl65>0.9469</td>
  <td class=xl65>0.9464</td>
  <td class=xl65>0.9467</td>
 </tr>
 <![if supportMisalignedColumns]>
 <tr height=0 style='display:none'>
  <td width=158 style='width:118pt'></td>
  <td width=64 style='width:48pt'></td>
  <td width=64 style='width:48pt'></td>
  <td width=64 style='width:48pt'></td>
  <td width=64 style='width:48pt'></td>
  <td width=64 style='width:48pt'></td>
  <td width=64 style='width:48pt'></td>
  <td width=64 style='width:48pt'></td>
  <td width=64 style='width:48pt'></td>
  <td width=64 style='width:48pt'></td>
  <td width=64 style='width:48pt'></td>
  <td width=64 style='width:48pt'></td>
  <td width=64 style='width:48pt'></td>
 </tr>
 <![endif]>
</table>



### 不同模型
|                 |神经网络(Adam)lr=0.001   |决策树  |逻辑回归  |SVM('linear')|SVM('rbf')|SVM('poly')|
|-----------------|-------------------------|-------|----------|-------------|----------|----------|
| Test Accuracy:  |98.3700%                 |87.59% |92.56%    |94.04%       |97.92%    |97.71%    |
| Precision:      | 0.9733                  |0.9247 |0.9247    |0.9399       |0.9792    |0.9771    |
| Recall:         |0.9729                   |0.9245 |0.9245    |0.9394       |0.9791    |0.9769    |
| F1:             |0.9731                   |0.9246 |0.9246    |0.9397       |0.9791    |0.9770    |
