title: System Analysis and Design (Kenneth Kendall) Review Part4
tags:
  - Design
  - Learning
  - SystemAnalysis
id: 493
categories:
  - System Analysis and Design
date: 2013-11-27 17:45:25
---

## 9\. 数据库设计

### 9.1.基本概念

1.数据库与文件系统的区别
传统的文件系统：是扁平文件，无结构的文件，存储简单地数据，读取速度快，存储形式单一；存在数据冗余，更新时间长，不一致性，安全性的问题，冗余会带来插入、更新、删除异常。
数据库系统：是堆文件，采用数据表存储，数据存放随机且没有特定的顺序，每一个表需要提供主键；数据库提供索引功能；可以通过范式来保证数据的一致性；有事务处理功能,ACID；可以避免插入、更新、和删除异常；提供安全功能。[more...]

2.数据库设计的目标：
a)能向用户提供数据
b)准确性，一致性，完整性
c)有效的存储数据以及有效的更新和检索
d)有目的的检索数据，获取的数据要有利于管理和决策。

3.有效性目标
a)保证数据能被各种应用程序的用户共享。
b)维护数据的准确性，一致性，完整性
c)确保当前的和未来的应用程序所需的所有数据能立即可用
d)允许数据库随用户需求的增加而不断演进。
e)允许用户建立自己的数据视图，而不用关心数据的实际存储方式。

4.ERD图,要会画，可以看书中的例子
实体：任何由某人为手机数据而选择的对象或事件叫做实体。包括：一般实体，关联实体，属性实体。
关系：1对1，1对多，多对多
属性：实体的特征
![](http://i40.tinypic.com/bf2o8n.jpg "ERD图")

5.键的类型：基于模式而不基于实例
a)主键：唯一标识一条记录
b)候选键：一个或一组可当作主键使用的键
c)辅助键：不能唯一标识一条记录，可以唯一，也可以标识多重记录
d)链接键(组合键)：可以选择两个或多个数据组成一个键
e)外键：一个属性，它是另外一个表的键

### 9.2.规范化

合理规范化的模型可应对需求变更，规范化数据重复降至最少.
1.第一范式1NF：数据原子性，每一列只有单值属性

2.第二范式2NF：没有部分依赖，每一个属性不能依赖主键的部分
如  (学号, 课程名称) → (姓名, 年龄, 成绩, 学分) ，不符合2NF
因为 (课程名称) –> (学分)，(学号)->(姓名，年龄)，存在部分依赖
此时存在数据冗余，插入异常，删除异常，更新异常

分解为：学生：Student(学号, 姓名, 年龄)； 
Course(课程名称, 学分)
SelectCourse(学号, 课程名称, 成绩)

3.第三范式3NF：没有传递依赖，不存在非主属性依赖于非主属性
例如： (学号) → (姓名, 年龄, 所在学院, 学院地点, 学院电话),符合2NF， 但不符合3NF，因为 (所在学院) -> (学院地点, 学院电话)是传递依赖，分解为：
(学号) → (姓名, 年龄, 所在学院)
(所在学院) -> (学院地点, 学院电话)

4.BCNF(Boyce-Codd Normaml Form): nothing but the key, 首先要满足3NF，在此基础上在候选键之间不能有传递依赖。
如：(仓库ID, 存储物品ID) →(管理员ID, 数量) 
　　 (管理员ID, 存储物品ID) → (仓库ID, 数量) 
(仓库ID, 存储物品ID)和(管理员ID, 存储物品ID)都是候选关键字，表中的唯一非关键字段为数量，符合3NF。但是，由于存在如下决定关系：
　　 (仓库ID) → (管理员ID) 　 (管理员ID) → (仓库ID) 
候选键之间有依赖，不符合BCNF。

分解为：
(仓库ID) –> (存储物品ID，数量)
(仓库ID) -> (管理员ID)
BCNF分解过细，导致查询连接性能低，所以一般3NF就可以了。

[数据库范式详解](http://www.cnblogs.com/zxsoft/archive/2007/08/03/840826.html  "数据库范式解析")

### 9.3.3NF无损连接分解方法

分解要做到无损连接，依赖保持。
考虑 关系  R= {ABCDEFGHI}
H -> GD  E->D  HD->CE  BD -> A
1\. Right reduced
H -> G
H->D
E->D
HD->C
HD->E
BD->A

2\. Left reduced
H->G
H->D
E->D
H->C
H->E
BD->A

3\. Find minimal cover
尝试删除一个依赖，然后验证是否无损和依赖保持，直到不能再删除
可以删除H->D,就不能再删了
H->G
E->D
H->C
H->E
BD->A
Minimal cover R’={H->GCE, E-D, BD->A}

4\. 补充没有的依赖
我们发现I还没有在R’中，因此，提取出R’中的键HB构造关系HB->I

5\. 最终结果
R = { H->GCE, E-D, BD->A , HB->I}

### 9.4.主文件/数据库关系设计指导原则

1\. 指导原则
每个单独的数据实体应创建一个主数据表。
一个特定的数据字段只应出现在一个主文件中。
每个数据表支持CRUD操作

2\. 完整性约束
实体完整性：主键不能有空值，唯一键可以有空值
引用完整性：一对多关系中，“一端”是父表，“多端”是子表，子表的外键必须在父表中有匹配记录；父表记录只在没有子表记录关联时才能删除；父表主键更新，要进行级联更新，即对应的子表外键要更新。
域完整性：对数据进行有效性检验，如数据必须大于0

3\. 异常
数据冗余：可以通过3NF解决
插入异常：如果主键重复，或未知，导致插入异常，因为违反实体完整性。
删除异常：删除导致相关数据丢失导致
更新异常：更新导致不一致性。

4\. 检索和显示数据的步骤
1)从数据库中选择一个关系

2)连接两个关系: join(inner join, outer join, left join, right join)
a)inner join： A join B 只有交集会出现在结果中
b)left (outer) join:  A 中与B没有匹配的记录会保留
c)right(outer) join: B中与A没有匹配的会保留
d)outer join, 又叫full outer join = left join 与right join的并集，所有匹配和未匹配的记录都有 

3)从关系中投影出列
4)从关系中选择所需的行
5)导出新的属性
6)行索引或排序
7)计算总计值和进行性能测量
8)显示数据

### 9.5.反规范化

1\. 原因
规范化的方式可以减少数据冗余，但查询速度会下降，一定的冗余，可以提升查询响应速度，避免重复引用检查表。
2\. 反规范化的6种方法
1)合并1：1关系
2)部分合并1：*关系，不复制外关键字而复制非key的常用的字段
3)在1：*关系中复制FK（外关键字），减少join的表数量，将另一个表的主键复制变成外键。
4)在*：*关系中复制属性，避免3表join
5)引入重复组，将多值属性写在主表里：例如user表中多个电话号码的列，常用地址和常用电话
6)为了避免查询和更新这两个不可调和的矛盾，可以将更新和查询放在两张表中，从工作表提取出查询表，专门用于查询。这个方法演化成了数据仓库。这个方法只适用于查询的实时性要求不高的情况

## 10.	设计准确的数据输入规程

数据输入准确性的目标：
•为数据创建有意义的代码
•设计有效的数据获取方法
•保证数据的完整性和有效性
•通过有效性检查确保数据的质量

### 10.1 有效的编码

#### 10.1.1.基本概念

1.优点：
a)提高数据处理效率
b)有助于数据的排序
c)节约内存和存储空间
d)提高数据输入的准确性和效率
e)特定类型的编码允许我们按特定的方式处理数据

2.目的
a)记录某些事物
b)分类信息
c)隐蔽信息
d)展示信息
e)请求相应地处理

3.编码的一般指导原则
a)保持代码简洁，短代码比长代码更容易输入和记忆
b)保持代码稳定，不应虽每次接收新数据而变化
c)代码要独一无二
d)允许排序代码
e)避免使人迷惑的代码
f)保持代码统一
g)允许修改代码
h)代码要有意义

### 10.1.2.编码的类型

1.记录某些事物
a)简单地顺序码
•优点：减少指派重复数字的可能性；让使用者估计出订单合适收到
•适用于：作业按顺序处理，需要知道数据项输入系统的顺序，需要知道事件的发生顺序。
•缺点：容易泄露商业信息，泄露当前已经指派了多少编码

b)字母衍生码
•例如：名字的编码： 取前2个辅音字母+名字长度+一个随机数
•例如编码女装：考虑品牌，类别，产地，生产日期，款号，尺码，颜色，价格
•常用于标识一个账号，邮寄地址标签
•注意：避免重复，分布要均匀
•缺点：如果采用名字的前三个辅音字母，当辅音字母少于3个，就会生成“RXX”这样的类型； 如果一些数据发生变化，如名字或地址变化，就会改变字母衍生码，而改变文件中的主键

2.分类信息
a)分类码
•目的：将一组具有特殊信息的数据从其他数据中区分出来，可以由单个字母或数字组成。是描述人物、位置、事物或事件的一种简写形式
•例如：计算所得税，分类有支付利息，医药费，税，捐款，应付款，生活用品支出，给每一个分类指派一个字母； 
•使用单个字母标识分类可能会存在扩展瓶颈，可使用两个以上的字母，如计算机中的快捷键。

b)块顺序码
•是顺序码的扩展，对数据分类，对每一个分类分配一个编码范围，在该类别的项目按顺序编码
•例如浏览器分配100~999， 数据库 200~299
•优点：根据普通的特征对数据分类，还能简单地为下一个需要标识的项目（在同一块）指派一个可用的数字代码

3.隐藏信息
密码，隐藏信息，如医药处方，凯撒密码对称加密，Hash密码单向加密

4.展示信息：为用户展示信息，使数据输入更有意义
a)有效数字集编码
•例如衣服采用有效数字集描述产品信息，“414-219-19-12:表示浅褐色冬季外套，款式219，尺码12”
•优点：让员工方便的定位产品类别；查询效率高；有助于销售

b)助记码
•帮助记忆，结合字母和符号，醒目而又清晰的编码产品，例如国家简称
c)Unicode：显示我们不能输入和看到的字符， 国际标准组织ISO定义Unicode字符集，包括所有标准语言字符，还有65535个空位

5.请求相应的处理
a)功能码：用简短的数字或字母来标识一个计算机对数据执行的功能，通常采用顺序码或助记码的形式
优点：执行功能只需输入功能码，提高输入效率

### 10.2.快速而高效的数据获取

1.决定要获取什么样的数据
如果输入无用，输出也会无用
输入的数据分类：随每个事务而改变的数据；能简明的将正在处理的项目与所有其他项目区分出来的数据

2.让计算机完成数据处理：处理重复的任务，如记录事务时间，根据输入计算新值，随时保存和检索数据

3.减少瓶颈和减少额外输入步骤：步骤越少，引入错误的机会就越少

4.选择有效的数据输入方法
a)键盘
b)扫描仪
c)视频，音频
d)磁性墨水
e)标记识别表单：如答题纸，但缺点是用户可能会一时大意填图出错
f)条形码：准确度高
g)RFID：射频识别技术

<div style="color: #EF4808">
两个好用的Chrome插件，小伙伴们看完复习资料支持一下吧：
[剪影截图：好用的网页截图，一键人人分享工具，快捷键ctrl+shift+z, 双击确定!](https://chrome.google.com/webstore/detail/剪影截图/gkloklemhahnoipikedmafefilidffko "剪影截图")
[TSS下载助手：让你方便一键批量下载](https://chrome.google.com/webstore/detail/tss下载助手/odhkpoplnhfnhhhkgphckabboemiifle "TSS下载助手")
</div>

资源：
[系统分析与设计part1](http://cyanny/myblog/2013/11/27/system-analysis-design-kenneth-kendall-review-part1/ "System Analysis and Design (Kenneth Kendall) Review Part1")
[系统分析与设计part2](http://cyanny/myblog/2013/11/27/system-analysis-design-kenneth-kendall-review-part2/ "System Analysis and Design (Kenneth Kendall) Review Part2")
[系统分析与设计part3](http://cyanny/myblog/2013/11/27/system-analysis-design-kenneth-kendall-review-part3/ "System Analysis and Design (Kenneth Kendall) Review Part3")
[系统分析与设计part4](http://cyanny/myblog/2013/11/27/system-analysis-design-kenneth-kendall-review-part4/ "System Analysis and Design (Kenneth Kendall) Review Part4")