---
title: DeepLearning|神经网络与深度学习
date: 2022-06-29 20:00:13
tags:
  - Deep Learning
  - Andrew Ng
categories:
  - deep learning
katex: true 
---

{% timeline 2022 %}

<!-- timeline 07-07 -->

完成第四周笔记整理

<!-- endtimeline -->

<!-- timeline 07-05 -->

完成第三周笔记整理

<!-- endtimeline -->

<!-- timeline 07-01 -->

完成第二周笔记整理

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

# 第一周：Introduction to Deep Learning

## Welcome

1. 神经网络与深度学习：学习如何建立神经网络，以及如何训练
2. 改善深层神经网络：学习超参数调整、正则化、诊断偏差和方差以及一些高级优化算法
3. 搭建机器学习项目：学习如何搭建机器学习项目
4. CNN：学习如何构建 CNN
5. 序列模型：学习序列模型，以及如何将其应用于自然语言处理

## What is a Neural Network

NN 相当于一个函数 $\hat y = f(x)$，只要将特征向量 $x$ 输入到模型 $f(\centerdot)$ 中，就能输出预测值 $\hat y$。

以房价预测为例，area，bedrooms，location 和 wealth 决定了该房子的价值。所以构建一个有4个输入特征的 NN。只要将相关的特征输入到模型中，就能预测对应的房价。**但是要经过一定的训练才能得到理想的精度。**

<img src="Untitled.png" alt="Untitled" style="zoom: 80%;" />

模型的中间结点和所有输入特征相连，通过训练，这些特征结点可以调节和每个输入特征之间的相关联性大小。比如第一个结点可能代表家庭人口，而家庭人口取决于 size 和 bedrooms，而和 location 和 wealth 不相干。 

## Supervised Learning with Neural Networks

**监督学习，即所有输入数据都有确定的对应输出数据**，在各种网络架构中，输入数据和输出数据的节点层都位于网络的两端，训练过程就是不断地调整它们之间的网络连接权重。

<img src="Untitled 1.png" alt="Untitled" style="zoom:80%;" />

有多种不同架构的监督学习，比如标准的 NN 可用于训练房子特征和房价之间的函数，CNN 可用于训练图像和类别之间的函数，RNN 可用于训练语音和文本之间的函数。他们的模型架构如上图所示。

<img src="Untitled 2.png" alt="Untitled" style="zoom: 67%;" />

## Why is Deep Learning taking off?

<img src="Untitled 3.png" alt="Untitled" style="zoom: 67%;" />

深度学习能发展起来主要是由于大数据的出现。

在小规模的数据量下，模型的性能取决于个人的特征工程能力。而随着数据量的不断增大，神经网络的优势就愈发明显了。从图中可以看出，随着规模【数据规模，网络规模】的不断扩大，神经网络的性能越来越好。

因此为了更好的性能，人们要么训练一个更大的神经网络，要么投入更多的数据。但是 labeled data 总是有限的，因此目前人们倾向于训练出更大的 NN。但是一个更大的神经网络意味着更长的训练时间，因此目前的许多算法都是在提升神经网络的运行速度。

例如新型激活函数的出现，因为sigmoid 函数在正无穷处和负无穷处会出现趋于零的导数，这正是梯度消失导致训练缓慢甚至失败的主要原因。**而用 ReLU 函数替换 sigmoid 函数可以在反向传播中保持快速的梯度下降过程。**要研究深度学习，需要学会「idea — 代码 — 实验 — idea」的良性循环。

# 第二周：Basics of Neural Network programming

## Binary Classification

<img src="Untitled 4.png" alt="Untitled" style="zoom:67%;" />

关于二分类问题，可以以判断一张图像是不是猫为例子。将一张图像作为模型的输入，如果该图像是猫，则输出1，否则输出0。**Logistic Regression 是解决二分类问题的一种常见算法。** 

> 符号约定
>
> - $x$：表示一个$n_{x}$维的输入数据，x.shape 为 $\left(n_{x},1\right)$
> - $y$：表示输出结果，取值为 $\left(0,1\right)$
> - $\left(x^{(i)}, y^{(i)}\right)$：表示第 i 组数据，可能是训练数据，也可能是测试数据
> - $X=\left[x^{(1)}, x^{(2)}, \ldots, x^{(m)}\right]$：表示所有的训练集的输入值，放在一个 $n_{x} \times m$ 的矩阵中，其中 m 表示样本数目
> - $Y=\left[y^{(1)}, y^{(2)}, \ldots, y^{(m)}\right]$：表示所有训练集的输出值，维度为 $1 \times m$

## Logistic Regression Hypothesis Function

<img src="Untitled 5.png" alt="Untitled" style="zoom:80%;" />

我们可以将 logistic 回归看成将两组数据点分离的问题，如果仅有线性回归（激活函数为线性），则对于非线性边界的数据点（例如，一组数据点被另一组包围）是无法有效分离的，因此在这里需要用非线性激活函数替换线性激活函数。

关于识别一张图片是否为猫的问题，我们使用逻辑回归，将特征向量 $X$ 【将图片 `reshape` 成向量】输入模型后，输出预测值 $\hat y$，表示对实际值 $y$ 的估计。**令 $\hat y$ 表示 $y$ 等于1的概率，即 $\hat y = p(y=1|x)$。** 

如果将假设函数设为 $\hat{y}=w^{T} x+b$ ，其中 $w$ 表示模型参数，$b$ 表示偏差。但是该假设函数得到的输出范围可能大于1，或小于0。不符合概率的定义，所以需要对其附加 sigmoid 函数。于是 Logistics Regression 的假设函数应该为 $\hat{y}=\sigma \left(w^{T} x+b\right)$

## Logistic Regression Cost Function

<img src="Untitled 6.png" alt="Untitled" style="zoom:67%;" />

神经网络的训练目标是确定最合适的权重 w 和偏置项 b，那这个过程是怎么样的呢？

这其实就是一个优化问题，优化过程的目的是使预测值 $\hat y$ 更加接近真实值 $y$ 。形式上可以通过寻找目标函数的最小值来实现。所以我们首先确定目标函数（损失函数、代价函数）的形式，然后用梯度下降逐步更新 $w$，$b$，当目标函数达到最小值或者足够小时，我们就能获得很好的预测结果。

Logistic Regression 中用到的损失函数是：$L(\hat{y}, y)=-y \log (\hat{y})-(1-y) \log (1-\hat{y})$

## Gradient Descent

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

## Derivatives

导数其实相当于函数的斜率。

以函数 $f(a)=3a$ 为例。导数的数学定义就是将 $a$ 右移无限小的距离，$f(a)$ 增加了3倍于 $a$ 右移的距离的值。所以 $\frac{d f(a)}{d a}=3$。

## More Derivative Examples

导数就是斜率，而函数的斜率，在不同的点是不同的。对于函数 $f(a) = a^2$，或者$f(a)=\log{a}$，它们的斜率是变化的，所以它们的导数或者斜率，在曲线上不同的点处是不同的。

## Computation Graph

![Untitled](Untitled 8.png)

上图是函数 $J=3\left(a+bc\right)$ 的计算图。假设 $a=5$，$b=3$，$c=2$。

通过上图可以看出，**通过一个前向传播的过程，可以计算 $J$ 的值。再通过一个反向传播的过程，计算梯度**。

## Derivatives with a Computation Graph

![Untitled](Untitled 9.png)

函数 $J=3\left(a+bc\right)$ 。假设要计算 $\frac{d J}{d b}$，根据从左到右的方向传播箭头开始计算。

1. 先计算 $\frac{d J}{d v}$，按照导数的计算方法，得到 $\frac{d J}{d v}=3$ 
2. 再计算$\frac{d J}{d u} = \frac{d J}{d v} \cdot \frac{d v}{d u} = 3 \cdot \frac{d v}{d u} = 3 \times 1 = 3$ 
3. 最后计算$\frac{d J}{d b}= \frac{d J}{d u} \cdot \frac{d u}{d b} = 3 \cdot \frac{d u}{d b} = 3 \cdot c = 6$  

## Logistic Regression Gradient Descent

对逻辑回归的损失函数，**仅针对一个训练样本**进行梯度下降，过程如下

![Untitled](Untitled 10.png)

目标是要计算出 $\frac{d L}{d w_{1}}$，$\frac{d L}{d w_{2}}$，$\frac{d L}{d b}$，然后结合学习率进行更新。反向传播的过程如上图所示，先计算 $\frac{d L}{d a}$，再计算 $\frac{d L}{d z}$，最后根据链式法则能得到$\frac{d L}{d w_{1}}=x_{1}dz$，$\frac{d L}{d w_{2}}=x_2dz$，$\frac{d L}{d b} = d z$。

## Gradient Descent on m Examples

对逻辑回归的代价函数，**针对 m 个样本**进行梯度下降，过程如下

Logistics Regression 的代价函数为 $J(w, b)=\frac{1}{m} \sum \limits_{i=1}^{{m}} L\left(a^{(i)}, y^{(i)}\right)$ ，其中 $a_{i}$，$y_{i}$ 分别表示第 i 个样本的预测值和真实值。目标是是计算出 $\frac{d J(w,b)}{d w_{1}}$，$\frac{d J(w,b)}{d w_{2}}$，$\frac{d J(w,b)}{d b}$，然后结合学习率进行更新。

反向传播的过程如下

1. 先针对第 i 个样本进行梯度下降计算，得到 $\frac{d J(w,b)}{d w_{1} ^{i}}$，$\frac{d J(w,b)}{d w_{2} ^{i}}$，$\frac{d J(w,b)}{d b ^{i}}$ 【**第一个for循环**】
2. 分别对总的 m 个样本进行梯度下降计算，然后将得到的梯度求和 $\sum \limits_{i=1}^{{m}} \frac{d J(w,b)}{d w_{1} ^{i}}$，$\sum \limits_{i=1}^{{m}} \frac{d J(w,b)}{d w_{2} ^{i}}$，$\sum \limits_{i=1}^{{m}} \frac{d J(w,b)}{d b ^{i}}$ 【**第二个for循环**】
3. 对上一步得到的梯度和除于 m ，得到 $dw_{1} = \frac{d J(w,b)}{d w_{1}}$，$dw_{2} =\frac{d J(w,b)}{d w_{2}}$，$db = \frac{d J(w,b)}{d b}$ 
4. 然后结合学习率进行更新
5. 重复上述过程，直到得到合适的参数【**迭代次数，这个循环无法除去**】

## Vectorization

向量化可以提升速度，比如计算两个向量之间的点积。

如果使用 for 循环遍历每个元素速度很慢，使用 numpy 提供的 `np.dot` 进行向量化运算，可以提高速度。

## More Examples of Vectorization

当你想写循环时候，检查 **numpy** 是否存在类似的内置函数，从而避免使用循环方式。

## Vectorizing Logistic Regression

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

## Vectorizing Logistic Regression's Gradient

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

## Broadcasting in Python

![Untitled](Untitled 12.png)

**如果两个数组的后缘维度（即矩阵维度元组中最后一个位置的值，shape[-1]）的轴长度相符或其中一方的轴长度为1，则认为它们是广播兼容的。广播会在缺失维度和轴长度为1的维度上进行。**

## A note on python or numpy vectors

1. 不要使用一维数组，改用行向量/列向量代替。

   ```python
   a = np.array([1, 2, 3])  # a.shape=(3,), 这是一维数组
   # a.T 与 a 一摸一样
   
   # 改用
   a = np.array([[1, 2, 3]])  # a.shape=(1, 3), 这是行向量
   ```

2. 不知道一个向量的维度的时候，使用断言语句进行判断。

3. 为了确保你的矩阵或向量所需要的维数时，不要羞于 **`reshape`** 操作

# 第三周：浅层神经网络

## Neural Network Overview

可以将多个 sigmoid 单元堆叠起来形成一个神经网络。

![Untitled](Untitled 13.png)

将包含三个特征 $x_{1}$，$x_{2}$，$x_{3}$ 的输入样本输入模型中。$W^{[1]} \in R^{3 \times 3}$，表示第一层中，与每一个神经元相关的参数。

于是上图中第一层的前向传播计算过程如下，其中**方括号上标表示与网络的第 i 层相关**：

$$
\left.\begin{array}{r}x \\W^{[1]} \\b^{[1]}\end{array}\right\} \Longrightarrow z^{[1]}=W^{[1]} x+b^{[1]} \Longrightarrow a^{[1]}=\sigma\left(z^{[1]}\right)
$$

第二层的前向传播计算过程如下，$W^{[2]} \in R^{1 \times 3}$：

$$
\begin{array}{l}\left.\begin{array}{r}a^{[1]}=\sigma\left(z^{[1]}\right) \\W^{[2]} \\b^{[2]}\end{array}\right\} \Longrightarrow z^{[2]}=W^{[2]} a^{[1]}+b^{[2]} \Longrightarrow a^{[2]}=\sigma\left(z^{[2]}\right)\Longrightarrow L\left(a^{[2]}, y\right)\end{array}
$$

反向传播的过程

$$
\begin{array}{l}\left.\begin{array}{r}d a^{[1]}=d \sigma\left(z^{[1]}\right) \\d W^{[2]} \\d b^{[2]}\end{array}\right\} \Longleftarrow d z^{[2]}=d\left(W^{[2]} \alpha^{[1]}+b^{[2]}\right) \Longleftarrow d a^{[2]}=d \sigma\left(z^{[2]}\right) \Longleftarrow d L\left(a^{[2]}, y\right)\end{array}
$$

其实可以看出是上图中的网络前向传播和反向传播的过程就是多次逻辑回归的重复。

## Neural Network Representation

<img src="Untitled 14.png" alt="Untitled" style="zoom:50%;" />

上图的 NN 由3部分构成

1. Input Layer：输入特征 $x_{1}$，$x_{2}$，$x_{3}$，构成 NN 的输入
2. Hidden Layer：该层中的每一个神经元包含两个动作，计算 $z$，然后对 $z$ 使用激活函数。
3. Output Layer：产生预测值

从技术实现角度出发，这是个三层的 NN。**但是在计算 NN 的层数时，通常输入层不会被计入，所以会把该 NN 称为两层的 NN。**

## Computing a Neural Network's output

<img src="Untitled 15.png" alt="Untitled" style="zoom: 67%;" />

**针对单个训练样本**，隐藏层中的每个神经元的计算过程如下【**下标表示第 i 个神经元**】：

$$
\begin{array}{l}z_{1}^{[1]}=w_{1}^{[1] T} x+b_{1}^{[1]}, a_{1}^{[1]}=\sigma\left(z_{1}^{[1]}\right) \\z_{2}^{[1]}=w_{2}^{[1] T} x+b_{2}^{[1]}, a_{2}^{[1]}=\sigma\left(z_{2}^{[1]}\right) \\z_{3}^{[1]}=w_{3}^{[1] T} x+b_{3}^{[1]}, a_{3}^{[1]}=\sigma\left(z_{3}^{[1]}\right)\end{array}
$$

对上面的计算进行向量化，将隐藏层的参数 $w_{1}^{[1]}$，$w_{2}^{[1]}$，$w_{3}^{[1]}$ 堆叠起来，得到一个矩阵 $W^{[1]} \in R^{3 \times 3}$，同时将 $b_{1}^{[1]}$，$b_{2}^{[1]}$，$b_{3}^{[1]}$ 堆叠起来，得到矩阵 $b^{[1]} \in R^{3 \times 1}$。进行向量化计算，得到 $z^{[1]}$：

$$
z^{[1]}=\left[\begin{array}{c}
z_{1}^{[1]} \\
z_{2}^{[1]} \\
z_{3}^{[1]} 
\end{array}\right]=\overbrace{\left[\begin{array}{c}
\ldots W_{1}^{[1] T} \ldots \\
\ldots W_{2}^{[1] T} \ldots \\
\ldots W_{3}^{[1] T} \ldots 
\end{array}\right]}^{W ^ {[1]}}*
\overbrace{\left[\begin{array}{l}
x_{1} \\
x_{2} \\
x_{3}
\end{array}\right]}^{i n p u t}+
\overbrace{\left[\begin{array}{l}
b_{1}^{[1]} \\
b_{2}^{[1]} \\
b_{3}^{[1]} 
\end{array}\right]}^{b^{[1]}}
$$

然后再对向量 $z^{[1]}$ 进行激活函数运算，得到隐藏层的输出

$$
\\a^{[1]}=\left[\begin{array}{c}
a_{1}^{[1]} \\
a_{2}^{[1]} \\
a_{3}^{[1]} 
\end{array}\right]=\sigma\left(z^{[1]}\right)
$$

所以针对单个训练样本，上图中的 NN 向量化实现前向传播过程如下：

$$
z^{[1]}=W^{[1]}x+b^{[1]}\\ a^{[1]}=\sigma \left(z^{[1]} \right)\\z^{[2]}=W^{[2]}a^{[1]}+b^{[2]}\\a^{[2]}=\sigma \left(z^{[2]} \right)
$$

## Vectorizing across multiple examples

对于多个样本的非向量化前向传播的过程如下【**小括号上标表示与第 i 个训练样本相关，下面的公式是运用在多个训练样本上的**】：

$$
\begin{array}{l}z^{[1](i)}=W^{[1]} x^{(i)}+b^{[1]} \\a^{[1](i)}=\sigma\left(z^{[1](i)}\right) \\z^{[2](i)}=W^{[2]} a^{[1](i)}+b^{[2]} \\a^{[2](i)}=\sigma\left(z^{[2](i)}\right)\end{array}
$$

向量化实现过程：

1. 将 m 个训练样本 $x$ 横向堆叠成为矩阵 $X \in R^{n_x \times m}$，$n_x$ 为输入特征数。

   $$
   X=\left[\begin{array}{cccc}\vdots & \vdots & \vdots & \vdots \\x^{(1)} & x^{(2)} & \cdots & x^{(m)} \\\vdots & \vdots & \vdots & \vdots\end{array}\right]
   $$

2. 将每个神经元线性运算的结果 $z$ 进行堆叠，得到矩阵 $Z$。**该矩阵的列数，取决于训练样本的数量，行数取决于该层神经元的数量。**

   $$
   Z^{[1]}=\left[\begin{array}{cccc}\vdots & \vdots & \vdots & \vdots \\z^{[1](1)} & z^{[1](2)} & \cdots & z^{[1](m)} \\\vdots & \vdots & \vdots & \vdots\end{array}\right]
   $$

3. 再对矩阵 $Z$ 使用激活函数，得到矩阵 $A$ 。它由每一个训练样本【**列数**】，在每一个神经元【**行数**】的激活值组成。

   $$
   A^{[1]}=\left[\begin{array}{cccc}\vdots & \vdots & \vdots & \vdots \\a^{[1](1)} & a^{[1](2)} & \cdots & a^{[1](m)} \\\vdots & \vdots & \vdots & \vdots\end{array}\right]
   $$

于是多个样本的非向量化前向传播转变为向量化前向传播过程如下：

$$
\left.\begin{array}{r}
z^{[1](i)}=W^{[1]} x^{(i)}+b^{[1]} \\
a^{[1](i)}=\sigma\left(z^{[1](i)}\right) \\
z^{[2](i)}=W^{[2]} a^{[1](i)}+b^{[2]} \\
a^{[2](i)}=\sigma\left(z^{[2](i)}\right)
\end{array}\right\} \Longrightarrow\left\{\begin{array}{l} Z^{[1]}=W^{[1]}X + b^{[1]}\\
A^{[1]}=\sigma\left(Z^{[1]}\right) \\
Z^{[2]}=W^{[2]} A^{[1]}+b^{[2]} \\
A^{[2]}=\sigma\left(Z^{[2]}\right)
\end{array}\right.
$$

## Justification for vectorized implementation

<img src="Untitled 16.png" alt="Untitled" style="zoom: 67%;" />

## Activation functions

<img src="Untitled 17.png" alt="Untitled" style="zoom:67%;" />

1. **sigmoid**：sigmoid 函数**常用于二分类问题，或者多分类问题的最后一层，主要是由于其归一化特性**。sigmoid 函数在两侧会出现梯度趋于零的情况，会导致训练缓慢。
2. **tanh**：相对于 sigmoid，tanh 函数的**优点是梯度值更大，可以使训练速度变快**。缺点是函数在两侧会出现梯度趋于零的情况，会导致训练缓慢。
3. **ReLU**：该函数很常用，基本是默认选择的激活函数，**优点是不会导致训练缓慢的问题，并且由于激活值为零的节点不会参与反向传播，该函数还有稀疏化网络的效果**。
4. **Leaky ReLU**：避免了零激活值的结果，使得反向传播过程始终执行，但在实践中很少用。

## why need a nonlinear activation function?

<img src="Untitled 18.png" alt="Untitled" style="zoom:50%;" />

上图中的实例可以看出，没有激活函数的神经网络经过两层的传播，最终得到的结果和单层的线性运算是一样的，**也就是说，没有使用非线性激活函数的话，无论多少层的神经网络都等价于单层神经网络（不包含输入层）**。唯一可以考虑使用线性激活函数的地方就是输出层。

## Derivatives of activation functions

1. **sigmoid**

   - 公式为：$g(z) = \sigma(z) =  \frac{1}{1+e^{-z}}$
   - 求导为：$\frac{d}{d z} g(z)=\frac{1}{1+e^{-z}}\left(1-\frac{1}{1+e^{-z}}\right)=g(z)(1-g(z))$

2. **Tanh**

   - 公式为：$g(z)=\tanh (z)=\frac{e^{z}-e^{-z}}{e^{z}+e^{-z}}$
   - 求导为：$\frac{d}{d z} g(z)=1-(g (z))^{2}$

3. **ReLU**

   - 公式为：$g(z) = \max(0,z)$

   - 求导如下，通常在 z = 0 的时候给定其导数1，0；但是 z = 0 的情况很少

     $$
     g(z)^{\prime}=\left\{\begin{array}{ll}0 & \text { if } \mathrm{z}<0 \\1 & \text { if } \mathrm{z}>0 \\\text { undefined } & \text { if } \mathrm{z}=0\end{array}\right.
     $$

4. **Leaky ReLU**

   - 公式为：$g(z) = \max(0.01z,z)$

   - 求导为

     $$
     g(z)^{\prime}=\left\{\begin{array}{ll}0.01 & \text { if } \mathrm{z}<0 \\1 & \text { if } \mathrm{z}>0 \\\text { undefined } & \text { if } \mathrm{z}=0\end{array}\right.
     $$

## Gradient descent for neural networks

针对上文提到的一个两层 NN，有 $W^{[1]} \in R^{n^{[1]} \times n_x}$，$b^{[1]} \in R^{n^{[1]} \times 1}$，$W^{[2]} \in R^{n^{[2]} \times n^{[1]}}$，$b^{[2]} \in R^{n^{[2]} \times 1}$ 这些参数。其中 $n_x$，$n^{[1]}$，$n^{[2]}$ 分别表示输入特征数，第一层的神经元数和第二层的神经元数。

假设进行二分类任务，则代价函数为：$J\left(W^{[1]}, b^{[1]}, W^{[2]}, b^{[2]}\right)=\frac{1}{m} \sum_{i=1}^{m} L(a, y)$。

1. 前向传播过程如下

   $$
   (1)  Z^{[1]}=W^{[1]} x+b^{[1]} \\ (2)  a^{[1]}=\sigma\left(Z^{[1]}\right) \\ (3)  Z^{[2]}=W^{[2]} a^{[1]}+b^{[2]} \\(4)  a^{[2]}=g^{[2]}\left(Z^{[z]}\right)=\sigma\left(Z^{[2]}\right) \\
   $$

2. 反向传播过程如下

   $$
   (1)d Z^{[2]}=A^{[2]}-Y, Y=\left[y^{[1]} \quad y^{[2]} \quad \cdots \quad y^{[m]}\right] \\(2)d W^{[2]}=\frac{1}{m} d Z^{[2]} A^{[1] T} \\(2)d b^{[2]}=\frac{1}{m} n p . \operatorname{sum}\left(d Z^{[2]}, \text { axis }=1, \text { keepdims }=\text { True }\right) \\(4)d Z^{[1]}=\underbrace{W^{[2] T} \mathrm{~d} Z^{[2]}}_{\left(n^{[1]}, m\right)} * \underbrace{g^{[1]^{\prime}}}_{\text {activation function of hidden layer }} * \underbrace{\left(Z^{[1]}\right)}_{\left(n^{[1]}, m\right)} \\(5)d W^{[1]}=\frac{1}{m} d Z^{[1]} x^{T} \\ (6)\underbrace{d b^{[1]}}_{\left(n^{[1]}, 1\right)}=\frac{1}{m} n p . \operatorname{sum}\left(d Z^{[1]}, \text { axis }=1, \text { keepdims }=\text { True }\right)
   $$

## Random+Initialization

<img src="Untitled 19.png" alt="Untitled" style="zoom:67%;" />

当将所有参数初始化为零的时候，会使所有的神经元变得相同，在训练过程中只能学到相同的特征，而无法学到多层级、多样化的特征。

解决办法是随机初始化所有参数，但仅需少量的方差就行，因此使用 Rand（0.01）进行初始化，其中 0.01 也是超参数之一，**一般选择较小的数，因为如果太大的话，而激活函数是 tanh 或 sigmoid 的话，可能会落到梯度较小的位置，导致训练速度特别慢**。

# 第四周：Deep Neural Networks

## Deep L-layer neural network

为什么要使用深层 NN，是因为有一些函数，只有采用更深的 NN 才可以拟合，而较浅的模型做不到。

关于 NN 的符号定义：[http://www.ai-start.com/dl2017/html/notation.html](http://www.ai-start.com/dl2017/html/notation.html)

## Forward Propagation in a Deep Network

前向传播过程和第三周中讲授的浅层神经网络类似，不过只是将一层的前向传播过程多重复了几遍。

针对单个训练样本，前向传播的过程就是对以下计算过程的关于层数进行多次迭代，$l$ 表示第几层：

$$
z^{[l]}=w^{[l]} a^{[l-1]}+b^{[l]}, a^{[l]}=g^{[l]}\left(z^{[l]}\right)
$$

针对多个训练样本，向量化前向传播的过程如下：

$$
Z^{[l]}=W^{[l]} A^{[l-1]}+b^{[l]}, A^{[l]}=g^{[l]}\left(Z^{[l]}\right)
$$

**关于层数的迭代，必须用到一个显式的 for 循环，这是无法避免的。**

## Getting your Matrix Dimensions Right

非向量化实现情况下，$w^{[l]}$ 的维度一般是 $R^{n^{[l]} \times n^{[l-1]}}$，$b^{[l]}$ 的维度一般是 $R^{n^{[l]} \times 1}$，$z^{[l]}$，$a^{[l]}$ 的维度一般是 $R^{n^{[l]} \times 1}$。

并且 $dw^{[l]}$ 和 $w^{[l]}$ 的维度相同，$db^{[l]}$ 和 $b^{[l]}$ 的维度相同。

**如果要进行向量化，则 $w$ 和 $b$ 的向量化维度不变 $W^{[l]} \in R^{n^{[l]} \times n^{[l-1]}}$， $b^{[l]} \in R^{n^{[l]} \times 1}$，但 $z$，$a$ 以及 $x$ 的维度向量化后会发生变化**。

$Z^{[l]}$ 可以看出有每一个训练样本得到的 $z^{[l]}$ 堆叠得到的，$Z^{[l]}=\left(z^{[l][1]}, \quad z^{[l][2]}, \quad z^{[l][3]}, \ldots, z^{[l][m]}\right)$，其中 m 表示训练样本的个数。则 $Z^{[l]} \in R^{n^{[l]} \times m}$。

$A^{[l]}$ 是对 $Z^{[l]}$ 进行激活函数运算后得到，$A^{[l]} \in R^{n^{[l]} \times m}$。

## Why Deep Representations?

<img src="Untitled 20.png" alt="Untitled" style="zoom: 33%;" />

**神经网络的参数化容量随层数增加而指数式地增长，即某些深度神经网络能解决的问题，浅层神经网络需要相对的指数量级的计算才能解决。**

CNN 的深度网络可以将底层的简单特征逐层组合成越来越复杂的特征，深度越大，其能分类的图像的复杂度和多样性就越大。RNN 的深度网络也是同样的道理，可以将语音分解为音素，再逐渐组合成字母、单词、句子，执行复杂的语音到文本任务。

## Building Blocks of Deep Neural Networks

![Untitled](Untitled 21.png)

正向传播的本质其实是实现一个正向函数。假设在第 $l$ 层，该函数输入 $a^{[l-1]}$，通过计算 $a^{[l]} = g^{[l]} \left(z^{[l]} \right)=g^{[l]} \left(W^{[l]} a^{[l-1]}+b^{[l]} \right)$，输出 $a^{[l]}$。同时将 $z^{[l]}$ 缓存下来，因为 $z^{[l]}$ 在正向和反向传播的过程中都起重要作用。

反向传播也是要实现一个反向函数。假设在第 $l$ 层，该函数输入 $da^{[l]}$ 以及缓存的 $z^{[l]}$ ，输出 $da^{[l-1]}$，同时也要输出 $dW^{[l]}$，$db^{[l]}$ 用来更新参数。

![Untitled](Untitled 22.png)

整体步骤如下：

1. 正向传播：把输入特征 $a^{[0]}$ 作为输入，并结合 $W^{[1]}$ 和 $b^{[1]}$ 计算第一层的激活函数，得到结果用 $a^{[1]}$ 表示，同时缓存 $z^{[l]}$。之后将 $a^{[1]}$ 喂到第二层，结合 $W^{[2]}$ 和 $b^{[2]}$ 计算第二层的激活函数，得到结果用 $a^{[2]}$ 表示。后面几层以此类推，直到最后你算出了 $a^{[l]} = \hat y$。**在这些过程里我们缓存了所有的 $z^{[l]}$ 值，**这就是正向传播的步骤。
2. 反向传播：需要算一系列的反向迭代，你需要把 $da^{[l]}$ 的值作为输入，然后输出 $da^{[l-1]}$，以此类推，直到我们得到 $da^{[1]}$。因为 $da^{[0]}$ 是输入特征的导数，而输入特征不需要进行更新，所以反向传播到 $da^{[1]}$ 就可以止步了。**反向传播的过程中也会输出每一步的 $dW^{[l]}$ 和 $db^{[l]}$**。

## Forward and Backward Propagation

![Untitled](Untitled 23.png)

1. 前向传播：输入 $a^{[l-1]}$，输出 $a^{[l]}$，缓存 $z^{[l]}$。

   非向量化实现为 $a^{[l]} = g^{[l]} \left(z^{[l]} \right)=g^{[l]} \left(W^{[l]} a^{[l-1]}+b^{[l]} \right)$。向量化实现为 $A^{[l]} = g^{[l]} \left(Z^{[l]} \right)=g^{[l]} \left(W^{[l]} A^{[l-1]}+b^{[l]} \right)$ 

2. 反向传播：输入 $da^{[l]}$，输出 $da^{[l-1]}$，$dw^{[l]}$，$db^{[l]}$。

   非向量化实现为

   $$
   (1)  d z^{[l]}=d a^{[l]} * g^{[l]^{\prime}}\left(z^{[l]}\right) \\
   (2)  d w^{[l]}=d z^{[l]} \cdot a^{[l-1]} \\
   (3)  d b^{[l]}=d z^{[l]} \\
   (4)  d a^{[l-1]}=w^{[l] T} \cdot d z^{[l]} \\
   (5)  d z^{[l]}=w^{[l+1] T} d z^{[l+1]} \cdot g^{[l]^{\prime}}\left(z^{[l]}\right) 
   $$

   向量化实现为

   $$
   (1)  d Z^{[l]}=d A^{[l]} * g^{[l]^{\prime}}\left(Z^{[l]}\right) \\(2)  d W^{[l]}=\frac{1}{m} d Z^{[l]} \cdot A^{[l-1] T} \\(3)  d b^{[l]}=\frac{1}{m}  np.  \operatorname{sum}\left(d z^{[l]}\right. , axis  =1 , keepdims  =  True  )\\ (4)  d A^{[l-1]}=W^{[l] T} \cdot d Z^{[l]} 
   $$

## Parameters vs Hyperparameters

<img src="Untitled 24.png" alt="Untitled" style="zoom:33%;" />

深度网络的特点是需要大量的训练数据和计算资源，其中涉及大量的矩阵运算，可以在 GPU 上并行执行，还包含了大量的超参数，例如学习率、迭代次数、隐藏层数、激活函数选择、学习率调整方案、批尺寸大小、正则化方法等。

关于超参数的选择，需要不断的在实验的基础上进行调试，直到找到理想的超参数。同时**超参数的设定不是一劳永逸的**，因为 GPU，CPU 的算例不断提升，相关问题不断变化，超参数也需要不断变化。

## What does this have to do with the brain?

深度学习和大脑有什么关联性吗？关联不大。
