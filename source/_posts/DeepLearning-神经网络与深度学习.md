---
title: DeepLearning|神经网络与深度学习
date: 2022-06-29 20:00:13
tags:
  - Deep_learning
  - Andrew_Ng
categories:
  - deep-learning
mathjax: true
---

{% timeline 2022 %}

<!-- timeline 07-01 -->

完成笔记整理

<!-- endtimeline -->

<!-- timeline 06-29 -->

整理到了逻辑回归的损失函数

<!-- endtimeline -->

{% endtimeline %}

# 前言

> 参考链接：
>
> - [吴恩达推荐笔记：22张图总结深度学习全部知识_学习_程序员生活志_InfoQ写作社区](https://xie.infoq.cn/article/2d4ffe7bb8f7fadaa7aec35f8)
> - [fengdu78/deeplearning_ai_books: deeplearning.ai（吴恩达老师的深度学习课程笔记及资源） (github.com)](https://github.com/fengdu78/deeplearning_ai_books)
> - [DeepLearning.AI中国官网 - 深度学习专业 (deeplearningai.net)](https://www.deeplearningai.net/specializationDetail/Deep_Learning_Specialization)
>
> 目的：学习吴恩达老师深度学习课程第一课《神经网络与深度学习》过程中，将所得总结成该笔记。有助于后期温习。

# 第一门课：神经网络与深度学习

## 第一周：Introduction to Deep Learning

### Welcome

1. 神经网络与深度学习：学习如何建立神经网络，以及如何训练
2. 改善深层神经网络：学习超参数调整、正则化、诊断偏差和方差以及一些高级优化算法
3. 搭建机器学习项目：学习如何搭建机器学习项目
4. CNN：学习如何构建 CNN
5. 序列模型：学习序列模型，以及如何将其应用于自然语言处理

### What is a Neural Network

NN 相当于一个函数 $\hat y = f(x)$，只要将特征向量 $x$ 输入到模型 $f(\centerdot)$ 中，就能输出预测值 $\hat y$。

以房价预测为例，area，bedrooms，location 和 wealth 决定了该房子的价值。所以构建一个有4个输入特征的 NN。只要将相关的特征输入到模型中，就能预测对应的房价。**但是要经过一定的训练才能得到理想的精度。**

<img src="Untitled.png" alt="Untitled" style="zoom: 80%;" />

模型的中间结点和所有输入特征相连，通过训练，这些特征结点可以调节和每个输入特征之间的相关联性大小。比如第一个结点可能代表家庭人口，而家庭人口取决于 size 和 bedrooms，而和 location 和 wealth 不相干。 

### Supervised Learning with Neural Networks

**监督学习，即所有输入数据都有确定的对应输出数据**，在各种网络架构中，输入数据和输出数据的节点层都位于网络的两端，训练过程就是不断地调整它们之间的网络连接权重。

<img src="Untitled 1.png" alt="Untitled" style="zoom:80%;" />

有多种不同架构的监督学习，比如标准的 NN 可用于训练房子特征和房价之间的函数，CNN 可用于训练图像和类别之间的函数，RNN 可用于训练语音和文本之间的函数。他们的模型架构如上图所示。

<img src="Untitled 2.png" alt="Untitled" style="zoom: 67%;" />

### Why is Deep Learning taking off?

<img src="Untitled 3.png" alt="Untitled" style="zoom: 67%;" />

深度学习能发展起来主要是由于大数据的出现。

在小规模的数据量下，模型的性能取决于个人的特征工程能力。而随着数据量的不断增大，神经网络的优势就愈发明显了。从图中可以看出，随着规模【数据规模，网络规模】的不断扩大，神经网络的性能越来越好。

因此为了更好的性能，人们要么训练一个更大的神经网络，要么投入更多的数据。但是 labeled data 总是有限的，因此目前人们倾向于训练出更大的 NN。但是一个更大的神经网络意味着更长的训练时间，因此目前的许多算法都是在提升神经网络的运行速度。

例如新型激活函数的出现，因为sigmoid 函数在正无穷处和负无穷处会出现趋于零的导数，这正是梯度消失导致训练缓慢甚至失败的主要原因。**而用 ReLU 函数替换 sigmoid 函数可以在反向传播中保持快速的梯度下降过程。**要研究深度学习，需要学会「idea — 代码 — 实验 — idea」的良性循环。

## 第二周：Basics of Neural Network programming

### Binary Classification

<img src="Untitled 4.png" alt="Untitled" style="zoom:67%;" />

关于二分类问题，可以以判断一张图像是不是猫为例子。将一张图像作为模型的输入，如果该图像是猫，则输出1，否则输出0。**Logistic Regression 是解决二分类问题的一种常见算法。**

> 符号约定
>
> - $x$：表示一个 $n_{x}$ 维的输入数据，x.shape 为 $\left(n_{x},1\right)$
> - $y$：表示输出结果，取值为 $\left(0,1\right)$
> - $\left(x^{(i)}, y^{(i)}\right)$：表示第 i 组数据，可能是训练数据，也可能是测试数据
> - $X=\left[x^{(1)}, x^{(2)}, \ldots, x^{(m)}\right]$：表示所有的训练集的输入值，放在一个 $n_{x} \times m$ 的矩阵中，其中 m 表示样本数目
> - $Y=\left[y^{(1)}, y^{(2)}, \ldots, y^{(m)}\right]$：表示所有训练集的输出值，维度为 $1 \times m$

### Logistic Regression Hypothesis Function

<img src="Untitled 5.png" alt="Untitled" style="zoom:80%;" />

我们可以将 logistic 回归看成将两组数据点分离的问题，如果仅有线性回归（激活函数为线性），则对于非线性边界的数据点（例如，一组数据点被另一组包围）是无法有效分离的，因此在这里需要用非线性激活函数替换线性激活函数。

关于识别一张图片是否为猫的问题，我们使用逻辑回归，将特征向量 $X$ 【将图片 `reshape` 成向量】输入模型后，输出预测值 $\hat y$，表示对实际值 $y$ 的估计。**令 $\hat y$ 表示 $y$ 等于1的概率，即 $\hat y = p(y=1|x)$。** 

如果将假设函数设为 $\hat{y}=w^{T} x+b$ ，其中 $w$ 表示模型参数，$b$ 表示偏差。但是该假设函数得到的输出范围可能大于1，或小于0。不符合概率的定义，所以需要对其附加 sigmoid 函数。于是 Logistics Regression 的假设函数应该为 $\hat{y}=\sigma \left(w^{T} x+b\right)$

### Logistic Regression Cost Function

<img src="Untitled 6.png" alt="Untitled" style="zoom:67%;" />

神经网络的训练目标是确定最合适的权重 w 和偏置项 b，那这个过程是怎么样的呢？

这其实就是一个优化问题，优化过程的目的是使预测值 $\hat y$ 更加接近真实值 $y$ 。形式上可以通过寻找目标函数的最小值来实现。所以我们首先确定目标函数（损失函数、代价函数）的形式，然后用梯度下降逐步更新 $w$，$b$，当目标函数达到最小值或者足够小时，我们就能获得很好的预测结果。

Logistic Regression 中用到的损失函数是：$L(\hat{y}, y)=-y \log (\hat{y})-(1-y) \log (1-\hat{y})$

### Gradient Descent

<img src="Untitled 7.png" alt="Untitled" style="zoom:80%;" />

1. **首先初始化 $w$，$b$。**由于 Logistics Regression 的代价函数是凸函数，所以无论在哪里初始化，都可以到达最低点。
2. **找到下坡的方向【求梯度】后，以固定的学习率 $\alpha$ 不断地迭代。**
3. **直到到达全局最优解或者接近全局最优解的地方**。

用公式来说明：

$$
\begin{array}{l}
w:=w-\alpha \frac{\partial J(w, b)}{\partial w} \\
b:=b-\alpha \frac{\partial J(w, b)}{\partial b}
\end{array}
$$

其中学习率的大小可以决定收敛的速度和最终结果。

学习率较大时，初期收敛很快，不易停留在局部极小值，但后期难以收敛到稳定的值；学习率较小时，情况刚好相反。一般而言，我们希望训练初期学习率较大，后期学习率较小。

### Derivatives

导数其实相当于函数的斜率。

以函数 $f(a)=3a$ 为例。导数的数学定义就是将 $a$ 右移无限小的距离，$f(a)$ 增加了3倍于 $a$ 右移的距离的值。所以 $\frac{d f(a)}{d a}=3$。

### More Derivative Examples

导数就是斜率，而函数的斜率，在不同的点是不同的。对于函数 $f(a) = a^2$，或者$f(a)=\log{a}$，它们的斜率是变化的，所以它们的导数或者斜率，在曲线上不同的点处是不同的。

### Computation Graph

![Untitled](Untitled 8.png)

上图是函数 $J=3\left(a+bc\right)$ 的计算图。假设 $a=5$，$b=3$，$c=2$。

通过上图可以看出，**通过一个前向传播的过程，可以计算 $J$ 的值。再通过一个反向传播的过程，计算梯度**。

### Derivatives with a Computation Graph

![Untitled](Untitled 9.png)

函数 $J=3\left(a+bc\right)$ 。假设要计算 $\frac{d J}{d b}$，根据从左到右的方向传播箭头开始计算。

1. 先计算 $\frac{d J}{d v}$，按照导数的计算方法，得到 $\frac{d J}{d v}=3$ 
2. 再计算$\frac{d J}{d u} = \frac{d J}{d v} \cdot \frac{d v}{d u} = 3 \cdot \frac{d v}{d u} = 3 \times 1 = 3$ 
3. 最后计算$\frac{d J}{d b}= \frac{d J}{d u} \cdot \frac{d u}{d b} = 3 \cdot \frac{d u}{d b} = 3 \cdot c = 6$  

### Logistic Regression Gradient Descent

对逻辑回归的损失函数，**仅针对一个训练样本**进行梯度下降，过程如下

![Untitled](Untitled 10.png)

目标是要计算出 $\frac{d L}{d w_{1}}$，$\frac{d L}{d w_{2}}$，$\frac{d L}{d b}$，然后结合学习率进行更新。反向传播的过程如上图所示，先计算 $\frac{d L}{d a}$，再计算 $\frac{d L}{d z}$，最后根据链式法则能得到$\frac{d L}{d w_{1}}=x_{1}dz$，$\frac{d L}{d w_{2}}=x_2dz$，$\frac{d L}{d b} = d z$。

### Gradient Descent on m Examples

对逻辑回归的代价函数，**针对 m 个样本**进行梯度下降，过程如下

Logistics Regression 的代价函数为 $J(w, b)=\frac{1}{m} \sum \limits_{i=1}^{{m}} L\left(a^{(i)}, y^{(i)}\right)$ ，其中 $a_{i}$，$y_{i}$ 分别表示第 i 个样本的预测值和真实值。目标是是计算出 $\frac{d J(w,b)}{d w_{1}}$，$\frac{d J(w,b)}{d w_{2}}$，$\frac{d J(w,b)}{d b}$，然后结合学习率进行更新。

反向传播的过程如下

1. 先针对第 i 个样本进行梯度下降计算，得到 $\frac{d J(w,b)}{d w_{1} ^{i}}$，$\frac{d J(w,b)}{d w_{2} ^{i}}$，$\frac{d J(w,b)}{d b ^{i}}$ 【**第一个for循环**】
2. 分别对总的 m 个样本进行梯度下降计算，然后将得到的梯度求和 $\sum \limits_{i=1}^{{m}} \frac{d J(w,b)}{d w_{1} ^{i}}$，$\sum \limits_{i=1}^{{m}} \frac{d J(w,b)}{d w_{2} ^{i}}$，$\sum \limits_{i=1}^{{m}} \frac{d J(w,b)}{d b ^{i}}$ 【**第二个for循环**】
3. 对上一步得到的梯度和除于 m ，得到 $dw_{1} = \frac{d J(w,b)}{d w_{1}}$，$dw_{2} =\frac{d J(w,b)}{d w_{2}}$，$db = \frac{d J(w,b)}{d b}$ 
4. 然后结合学习率进行更新
5. 重复上述过程，直到得到合适的参数【**迭代次数，这个循环无法除去**】

### Vectorization

向量化可以提升速度，比如计算两个向量之间的点积。

如果使用 for 循环遍历每个元素速度很慢，使用 numpy 提供的 `[np.dot](http://np.dot)` 进行向量化运算，可以提高速度。

### More Examples of Vectorization

当你想写循环时候，检查 **numpy** 是否存在类似的内置函数，从而避免使用循环方式。

### Vectorizing Logistic Regression

之前需要使用 for 循环针对每个样本分别进行前向传播，得到预测值。

但是可以借助向量化的方式，摆脱 for 循环。

假设 $a^{(i)}$ 表示第 i 个样本的预测值，它的公式为 $a^{(i)} = \sigma \left(z^{(i)}\right)$，其中 $z^{(i)}=w^{T} x^{(i)}+b$ ，$x^{(i)} \in R^{1 \times n_{x}}$，表示第 i 个样本的输入，$w^T \in R ^{1 \times n_{x}}$，$b \in R$ 是要优化的参数。

可以将 m 个样本输入 $x$ 进行堆叠，得到一个输入矩阵 $X \in R ^{n_{x} \times m}$：

$$
X = \begin{bmatrix}    \vdots & \vdots & \vdots & \vdots \\    x^{(1)} & x^{(2)} & \cdots & x^{(m)} \\    \vdots & \vdots & \vdots & \vdots \end{bmatrix} 
$$

初始化一个向量 $Z = \left[z^{(1)} ,z^{(2)} ,\ldots ,z^{(m)}\right]$，令 $Z = w^{T} X+b=\left[w^{T} x^{(1)}+b, w^{T} x^{(2)}+b, \ldots, w^{T} x^{(m)}+b\right]$，$Z \in R ^{1 \times m}$。【**这里对 $b$ 的运算用到了广播机制**】

再初始化一个向量 $A = \left[a^{(1)} ,a^{(2)} ,\ldots ,a^{(m)}\right]$，令 $A = \sigma \left( Z\right) = \sigma \left( w^{T}  X+b \right)=\left[\sigma \left( w^{T} x^{(1)}+b \right), \sigma \left( w^{T} x^{(2)}+b \right), \ldots, \sigma \left( w^{T} x^{(m)}+b \right)\right]$，$A \in R ^{1 \times m}$。

可以看出 $A$ 中的每一个元素就是每一个样本输入的预测值。**从而摆脱了第一个 for 循环**。

### Vectorizing Logistic Regression's Gradient

对每一个样本输入的梯度进行求和后取平均，产生了第二个 for 循环。

$$
\begin{array}{l}d w=0 \\d w+=x^{(1)} * d z^{(1)} \\d w+=x^{(2)} * d z^{(2)} \\ \cdots \\d w+=x^{(m)} * d z^{(m)} \\d w=\frac{d w}{m} \\\\db=0 \\d b+=d z^{(1)} \\d b+=d z^{(2)} \\ \cdots \\ d b+=dz^{(m)} \\d b=\frac{d b}{m}\end{array}
$$

利用向量化的方式可以去掉该循环。

因为 $d b=\frac{1}{m} \sum_{i=1}^{m} d z^{(i)}$ ，所以 $d b=\frac{1}{m} * n p . \operatorname{sum}(d Z)$。 

因为 $d w= \frac{1}{m} *\left(x^{(1)} d z^{(1)}+x^{(2)} d z^{(2)}+\ldots+x^{m} d z^{m}\right)$，所以 $X = \frac{1}{m} * X * d Z^{T}$。

所以逻辑回归中一次梯度下降更新的向量化实现版本如下：

$$
\begin{array}{l}Z=w^{T} X+b=n p \cdot d o t(w \cdot T, X)+b \\A=\sigma(Z) \\d Z=A-Y \\d w=\frac{1}{m} * X * d Z^{T} \\d b=\frac{1}{m} * n p \cdot \operatorname{sum}(d Z) \\w:=w-a * d w \\b:=b-a * d b\end{array}
$$

Logistic Regression 的总体过程如下图所示

<img src="Untitled 11.png" alt="Untitled" style="zoom:67%;" />

### Broadcasting in Python

![Untitled](Untitled 12.png)

**如果两个数组的后缘维度（即矩阵维度元组中最后一个位置的值，shape[-1]）的轴长度相符或其中一方的轴长度为1，则认为它们是广播兼容的。广播会在缺失维度和轴长度为1的维度上进行。**

### A note on python or numpy vectors

1. 不要使用一维数组，改用行向量/列向量代替。

   ```python
   a = np.array([1, 2, 3])  # a.shape=(3,), 这是一维数组
   # a.T 与 a 一摸一样
   
   # 改用
   a = np.array([[1, 2, 3]])  # a.shape=(1, 3), 这是行向量
   ```

2. 不知道一个向量的维度的时候，使用断言语句进行判断。

3. 为了确保你的矩阵或向量所需要的维数时，不要羞于 **`reshape`** 操作
