title: 统计学的基本概念和方法
tags:
  - Learning
  - math
id: 256
categories:
  - Life
date: 2013-10-03 22:42:08
---

如果这本书读完《统计学的基本概念和方法》，一本价值500页的书，最后给忘记了，我会为我这几天的功夫感到遗憾，姑且高兴高兴把报告亮出来。谈谈感想。

### 前言

“统计学是用以收集数据、分析数据和由数据得出结论的一组概念、原则和方法。” 这是埃维森在本书中给统计学下的定义。而在读完本书后，我觉得统计学不仅仅是那些曾经学习的数学定义和公式，本书用大量的例子有趣地说明了统计就在生活中，如政府部门的政策和评估、人口调查、医学、心理学、和法学等等领域，统计学都有重要的角色。而如何正确的收集数据，如何选择合适统计分析方法，以及如何用怀疑的眼光来看待一些社会生活中的统计结论，不盲从以及冷静的分析，是我觉得本书给我的一大益处。

总的来说，统计学可以说是在各种随机的数据中寻找寻找规律性的方法。统计学家所做的许多工作都是为了回答四个问题：

问题1：在数据中，变量之间是否有关系？

问题2：变量之间的关系有多强？

问题3：总体中是否存在该关系？

问题4：观测到的关系是一种因果关系吗？[more...]

纵观本书，为了回答这四个问题，首先要定义需要观测的变量并收集数据；对数据进行整理，用圆饼图、条形图、点图或直方图等，形象地对数据做出表示；计算汇总统计量，如均值，方差，标准差等；从原始数据做出初步猜测，所观测的变量之间是否有关系；为了验证猜想，运用公式，计算变量之间的相关系数，如对于数量型变量的分析，还需计算出线性回归公式；接着，对总体中是否存在该关系做出假设，如零假设是总体中不存在这些关系，再运用假设检验的方法拒绝零假设，即统计上显著，从而可以将样本的结论推广到总体，假设检验中会用到z，t，χ2, F分布；最后，对于因果关系，这是很难回答的，统计上相关未必有因果关系，这需要综合考虑多方面的因素才能下结论，然而 否定因果关系是容易的，可以通过引入第三方变量造成变量中的关系消失来否定因果关系。

### 数据的收集

统计结果是基于：数据的收集和统计方法的运用。

而正确的进行数据收集是对后期得出正确的统计结论的重要保障。而实践中我们由于成本、社会道德等原因，很难进行全部总体的普查，一般是进行抽样，这样可以节约成本，加快数据收集过程，提高效率并减少偏差。一个合适的能够被用来推广到总体的统计样本叫随机样本。首先我们要定义清楚总体是什么，即采样的种群，是所有成年人或者某公司所有员工；其次我们要选择正确的采样方法的保证采样的随机性，如概率采样，分层采样和聚类采样；最后，我们要决定采样规模，样本数越大，抽样误差越小。例如调查总体中样本的百分比，往往要求误差在3%以内，那么需要至少1200个响应者。

实际中，采样会遇到很多问题，对人的研究比研究土豆难得多，例如组织问题，心里问题，道德问题和成本问题等，都应在进行数据抽样前多加考虑。

### 数据描述

在得到采样数据后，我们一般会采用表的形式来记录数据，而为了展现统计结果，图是分析数据的一种极富有信息的方法，直观易于理解。

* 分类变量

一般采用圆饼图和条形图来展现。当类别不是很多时，圆饼图表达更易于表达出每一个类别的相对大小。但是圆饼图用于显示每一组有多少个观测数时不是很好。

而条形图，优势在于可以显示每一个组的观测数，可以有相同高度不同宽度的条形图，也可以选择不同高度和相同宽度的条形图。

* 度量变量

度量变量一般用点线图，茎叶图，盒形图，直方图，散点图和时间序列图来展现。

点线图：在一条连续的线上表明一个小的数据阵，原始值得以保留，适合展现小规模数据的分布和变化。盒形图：对数据进行了简化，取5个值，最大值，最小值，中位数，25%和75%分位数来构成。表明了两个极值和中间值的范围，在分析若干组数据时很有用。但无法恢复原始数据。直方图：用矩形的面积表示变量值的观测和相对数量，可用于显示大量的观测数据，但是无法保持原始值。茎叶图：适合展现小数据集，但不适合展现数据变化范围小的数据。散点图：可以用于表现变量之间的关系，在回归分析中常用到，表现了数据的分布和变化。时间序列图：在横轴上均匀的显示时间，纵轴上是变量的值，可以展现多个变量的数据。

* 地图作图

根据数据，按区域在地图上染色，标识数据，如人口分布等。但地图可能会误导，地图上面积小，但是人口密度大，但面积和人口密度不成正比。

如何选择正确的图呢？Edward Tufte定义了图优性：图要能够在最短的时间内用最少的笔墨，在最小的空间里给观众最多的思想。在统计图选择中，我们面临两个矛盾：数据的简化和数据的完整，我们需要在简单化受益和信息的丢失之间权衡。

### 分析统计量

*   平均数

    数据的统计量计算需要正确的方法。平均数包括众数，中位数和均值。众数适用于分类变量，特别是有多个值的分类变量，如宗教、种族的划分。中位数：一组数据中的中间值，不受极值的影响，当数据出现非对称分布时应该用中位数，例如收入、房子价格。但中位数只是中间点的数据，但没有充分利用数据。均值：在对总体进行点估计时常用均值，均值可以充分地利用观测数据，但均值容易受极值的影响。 如果均值和中位数差不多，采用均值，但如果非常不同，采用中为数据，对于非对称数据，采用中位数更好。

*   变差

    极差：很容易受极端值的影响，为了避免这种影响，我们可以采用四分位数极差。

    标准差：度量数据在平均意义上和均值的偏离。

*   标准误差：多个样本均值的标准差，一般比某一组观察值的标准差小。

*   标准得分=(观察值 - 均值) / 标准差，可以对不同的观察值进行比较。

### 重要的概率分布

统计的样本一般假定是随机样本，对于随机样本的分析，需要计算概率，概率可以从等可能事件，相对频数和主观概率中得到。而了解概率的分布是在进行参数假设检验等分析中重要的一步。

* 离散型变量的概率分布

主要有几何分布和二项分布，主要在分类变量且样本数量很少时才很方便。对有很少的样本而可能性很多的分析，可以采用Poisson分布。

* 连续型变量的概率分布

主要有z（标准正太分布），t，χ2, F分布。标准正态分布是一个钟形曲线，均值=0，标准差=1，曲线下面有95%的面积在-1.96~1.96之间。一般假设检验和参数估计中会计算出标准的z值再查表得到概率。t分布，又叫学生分布，和正态分布很像，是最常用的统计分布，t分布的前提是总体要符合正态分布，根据自由度选择t分布。χ2, F分布都是有偏的，取值从0开始，按自由度选择。

* p值：即极端概率发生的概率，p值在说明统计显著方面很重要。

### 统计推断：估计和假设检验

对样本的分析最终需要做出一个关于总体的估计。
* 点估计

在估计总体均值时一般采用样本均值而不是中位数或众数来估计。

* 区间估计

只给出一个点是不够准确的，总要说明误差范围。在来自不同样本的多个置信区间当中包含未知总体参数的区间所占的百分比称为置信水平。我们一般采用95%的置信区间，意思不是该区间中包含真值的概率，而是多次抽样中有95%的置信区间包含未知总体参数的值。可以通过增加样本容量或降低置信水平来获得较短的置信区间。一般对于1200个响应者，抽样误差可控在-3%~3%，且置信水平是95%.

* 统计显著

显著是在本书开端就提出的问题，统计显著是对参数假设检验而言的，一个检验的显著水平a是抽样所得的数据拒绝了本来是正确的零假设的概率。p值越小越显著。一般p取0.05，但对于双边检验p=0.025。

假设检验可以进行总体均值的检验（t检验），总体比例检验(z检验)。当然也可以采用计算置信区间的方法，如果零假设中的相关参数值在置信区间，就不拒绝。

### 变量间的关系

对于从试验和观测中得到的数据，我们最终关心的是变量之间的关系问题。因此统计量的计算，数据的描述，统计推断都在为回答变量之间的关系做铺垫。四个重要的关系问题：问题1：在数据中，变量之间是否有关系？问题2：变量之间的关系有多强？问题3：总体中是否存在该关系？问题4：观测到的关系是一种因果关系吗？

前三个问题，可以通过统计方法得到准确的结论，但是因果关系确很难回答，我们在判断因果关系时：(1) 用常识进行推断. (2) 主意自变量和因变量的发生顺序。(3) 即使自变量发生在因变量之前，但当引入第三个变量，自变量和因变量的关系消失，他们也没有因果关系。然而，统计关系强，确实可以反映可以从多大程度上用自变量去估计因变量。

变量分为三类：分类型变量，顺序型变量和数量型变量，不同的变量，分析方法不同。

1\. 两个分类型变量的关系： χ2分析

常用列联表来表示分类型变量。

变量间是否有关系？可以通过观察数据的百分比分布得出。

变量之间的关系强度？通过公式计算相关系数。

总体中是否有这个关系？通过 χ2分析得到p值，进而拒绝零假设。

因果关系？分类变量的统计关系，不能说明因果关系。

2\. 两个数值型变量的分析：回归分析和相关分析

回归分析描述的是一个或多个自变量的变化是如何影响因变量的方法。相关分析描述的是两个数值型变量的关系强度。在对数值型变量的回归分析前，需要观察散点图，如果数据很分散，则变量的关系就很弱，同时回归分析时需要抛弃极端值。

回归分析重要参数如下

回归方程y = a + bx， b即回归系数，表示自变量变化1，因变量的变化值。

相关系数r：r^2表示自变量在决定因变量的取值变化效应中所占的比例，其他的变化影响来自残差变量。

总平方和=⨊(观测值 - 平均值)^2 = 残差平方和 + 回归平方和

残差平方和=⨊(估计值 - 观测值)^2

回归平方和 = ⨊(估计值 - 平均值)^2 

r^2 = 回归平方和 /  总平方和  

￼￼回归方程是否能对总体做出估计？需要进行相关分析，置信区间的计算，相关分析中采用t假设检验。

3\. 一个数值型变量和一个分类型变量的关系

变量：分类型变量，因变量：数值型变量。例如地区和犯罪率的关系。对于关系强度的计算，需要计算总平方和，残差平方和以及自变量效应的平方和。

总平方和(TSS)=⨊⨊(观测值 - 平均值)^2 = 残差平方和 + 回归平方和

残差平方和(RSS)=⨊⨊(观测值 - 组均值)^2

分类变量平方和(CSS) = ⨊⨊ni(组均值 - 平均值)^2  (ni---每组中的观测值数量) 

相关系数R ^ 2 = 回归平方和 /  总平方和  

采用F检验进行分析。

在书中，还提到配对数据的差异检验，如分析各个地区犯罪率与年份的关系，配对数据检验采用t检验，有时如果无法确定总体是正态分布，可以采用符号检验，但符号检验采用二项分布，如果数据量大不太合适。

4\. 两个顺序型变量的分析：秩方法

顺序型变量很少出现，尽管有些变量如富裕程度，看上去是分类型变量，但是富裕程度有低、中、高的顺序，如果采用分类型变量的χ2分析，数据就不能充分利用。

一般地，当一个数量型变量和一个顺序型变量出现时，都作为顺序型变量来分析，反之不行。

当一个分类型变量和一个顺序型变量进行分析时，都看做分类型变量来分析。

(1) 用词作为两个顺序型变量

例如：学生对数学的掌握程度(一般、中等、较好)与老师上课的有趣程度(一般、中等，较好)的关系，书中给的例子是政党身份的强弱与对总统选举感兴趣的程度的关系。

系数 γ 度量两个取值为词的顺序变量的相关程度。

γ = （相同的次序 - 不同的次序）/ （相同的次序  +  不同的次序）

可以将 γ转化为z值，进行检验，根据p值拒绝零假设。

(2) 用数字作为两个顺序性变量

例如：马刺队和灰熊队2011年和2012年的排名关系，或者一个人的经济地位与人的能力的关系。

采用Spearman   rs相关系数来度量两个取值为数量的顺序变量的相关程度。一般将rs转化为t变量进行检验。

5\. 多元分析

多元统计分析考虑两个或两个以上的变量对一个因变量的相关的影响。多元分析可以通过引入控制变量，来观察自变量和因变量的关系，如果关系消失，则否定因果关系。

(1) 三个分类型变量

偏Φ，是当控制变量取特定值时，度量自变量和因变量之间的相关程度。可以认为是对每一个控制变量的值，得出的自变量和因变量的相关系数Φ的平均值。例如：控制性别，研究收入和投票的关系。

(2) 多个数值型变量间的关系

采用多元回归分析，计算每一个自变量的偏归系数，构造回归方程。

相关系数R是因变量的实际值和回归方程预测值之间关系的强度。R^2是所有自变量共同解释因变量差异的总和。

一般将R转化为F比，进行假设检验其统计上的显著性。

(3) 因变量是数量型变量，自变量是两个分类型变量.

采用双因子方差分析，由于双因子方差分析可以考虑两个分类变量各自效应之外的交互效应，所以优于单因子方差分析。交互效应，即两个自变量对因变量的联合作用。
交互效应大于每个因变量各自效应的总和。通过研究交互效应和残差效应的F比，可以得出两个分类变量对因变量的统计显著性。

### 总结

在读完本书后，我觉得这么多统计方法，并没有带来很乐观的统计现状。统计从收集数据到分析得出结论，每一步都可能误用，而大多数误用不是故意的，且有些别有用心的人会对用一些方法对统计结果进行有意的歪曲，统计中需要警惕：

1\. 数据收集中的危险，研究者往往会在数据收集的随机性上打折扣，例如研究成年人的购物兴趣，研究者为了方便会选取一些积极性较高的大学生，样本是随机的，但样本的结果只能说明大学生的购买兴趣，统计推断只能把他推广到它产生的总体。

2\. 调查的环境，调查的身份，提问的方式，被访者是否在一个舒适和私下的环境下接受采访，也会对数据产生影响。

3\. 问题的内容和提问的时机，例如薪水问题如果提问不当，被访者会很反感。

4\. 结果的记录方式，如果一个人吸烟的兴趣是4分，两个相同分数的人对分数的理解是不同的。而且大多数研究场合下人们都会对回答有某种程度的夸张。

5\. 分析方法的误用，统计推断的误用，例如收入的平均数应该用中位数而不是均值；用最小二乘法计算回归方程，而不是绝对值，双边假设检验的p=0.025，而不是0.05。

6\. 数字的错误解释，如Benz公司广告说过去15年来被注册的他们公司的车种97%仍然在使用，其实97%中也包含近几年来注册的新车，消费者总是会被误导。

总之，任何有收集经验数据的场合都需要统计方法，而统计的结果依赖于数据和分析方法。统计很神奇，可以从混乱与随机中找到规律，而我们要做地是不被规律所误导，不盲从详细各种社会统计，官方统计，要拥有一双怀疑的眼睛。