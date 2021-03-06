title: 分布式系统总结part3
tags:
  - Learning
id: 414
categories:
  - Distributed System
  - Life
date: 2013-11-24 14:26:20
---

## 复制和一致性

数据的复制是为了提高性能和可靠性，但是我们需要保持各个副本的一致性：

问题一： 数据更新的实际分发问题，它需要关心副本的位置，以及如何在副本之间传播更新。

问题二：如何保持多个副本的一致性[more...]

### 以数据为中心的一致性

#### 1\. 严格的一致性

(1)条件：所有访问按绝对时间排序，数据项x的任何读操作将返回最近一次对x的写操作的结果所对应的值。

(2)要求：所有写操作是瞬间可见的，系统维持一个绝对的全局时间顺序，如果发生写操作，后续的读操作都会得到最新的写入的值。

(3)缺点：依赖于绝对的全局时间，分布式系统中为每个操作分配准确的全局时间是不可能的，一般是将时间分割成一系列连续的、不重叠的时间间隔，保证没个时间间隔内最多只发生一个单一操作。
精确的定义最后一次写操作是困难的，如图可以看到严格一致性
[![Distributed System02](http://cyanny/myblog/wp-content/uploads/2013/11/Distributed-System02.png)](http://cyanny/myblog/wp-content/uploads/2013/11/Distributed-System02.png)

#### 2.线性化和顺序的一致性

(1)顺序一致性

条件：所有进程以相同顺序看到所有的共享访问，访问不按时间排序。任何执行结果都是相同的，就好像所有进程对数据存储的读、写操作按照某种序列顺序执行的，并且每个进程的操作按照程序所定制的顺序出现在这个序列中。

缺点：严重的性能问题，对于任何的顺序一致性存储，改变协议以提高读操作的性能必将降低写操作的性能，反之亦然。
[![Distributed System03](http://cyanny/myblog/wp-content/uploads/2013/11/Distributed-System03-580x350.png)](http://cyanny/myblog/wp-content/uploads/2013/11/Distributed-System03.png)
(2)线性化一致性

条件：所有进程以相同顺序看到所有的共享访问，访问根据(并非唯一的)全局时间戳排序。

在实际应用中，线性化主要用于开发算法的形式验证。关于根据时间戳维护顺序的附加限制使得线性化的实现比顺序一致性的实现开销更大。 

#### 3.因果一致性

条件：是一种弱化的顺序一致性模型，所有进程必须以相同的顺序看到具有潜在因果关系的写操作。不同机器上的进程可以以不同的顺序被看到并发的写操作.

当一个读操作后面跟着一个写操作时，这两个事件就具有潜在的因果关系。

[![Distributed System04](http://cyanny/myblog/wp-content/uploads/2013/11/Distributed-System04-580x394.png)](http://cyanny/myblog/wp-content/uploads/2013/11/Distributed-System04.png)

#### 4.FIFO一致性

条件：所有进程以某个单一进程(有两个以上的写操作)提出写操作的顺序看到这些写操作，但是不同进程可以以不同的顺序看到不同的进程提出的写操作。

优点：容易实现，不需要保证不同进程看到相同的写操作顺序，除非两个以上的写操作是同一个进程提出的。这种情况下，写操作必须按顺序达到。

缺点：仍然对许多应用存在不必要的限制，因为这些应用需要从任何位置都可以看到按顺序看到某个单一进程所产生的写操作。同时并不是所有的应用程序都要求看到所有的写操作。这就引入了弱一致性。

[![Distributed System05](http://cyanny/myblog/wp-content/uploads/2013/11/Distributed-System05.png)](http://cyanny/myblog/wp-content/uploads/2013/11/Distributed-System05.png)

#### 5.弱一致性

条件：引入同步变量，只有执行一次同步，共享数据才能保持一致。  

a)对数据存储所关联的同步变量的访问是顺序一致的；b)每个拷贝完成所有先前执行的写操作之前，不允许对同步变量进行任何操作；c)所有先前对同步变量执行的操作都执行完毕之前，不允许对数据项进行任何读或者写操作。

特点：弱一致性是在一组操作，而非单个操作上强迫执行顺序一致性。同步变量用于划分操作的组。

缺点：即当同步变量被访问时，数据存储不知道此次访问是因为进程已经结束对数据存储的写操作还是因为进程将开始读数据而进行的。我们需要区分进程进入临界区和离开临界区的区别，引入了释放一致性。 

[![Distributed System06](http://cyanny/myblog/wp-content/uploads/2013/11/Distributed-System06-580x347.png)](http://cyanny/myblog/wp-content/uploads/2013/11/Distributed-System06.png)

#### 6.释放一致性

条件：退出临界区时，让共享数据保持一致，提供两种类型的同步变量。获取(acquire) 操作是用于通知数据存储进程进入临界区的操作，而释放(release)操作是表明进程刚刚离开临界区的操作。

*   对共享数据执行读操作或写操作之前，所有进程先前执行的获取操作都必须已经成功完成；
*   在释放操作被允许执行前，所有进程先前执行的读操作和写操作都必须已经完成；
*   对同步变量的访问是FIFO一致的(不需要顺序一致)

[![Distributed System07](http://cyanny/myblog/wp-content/uploads/2013/11/Distributed-System07.png)](http://cyanny/myblog/wp-content/uploads/2013/11/Distributed-System07.png)

#### 7.入口一致性

条件：它需要程序员(或编译器)在每个临界区的开始和结束处分别使用获取和释放操作，入口一致性要求每个普通的共享数据项都要与某种同步变量(如锁或障碍)关联。即让进程在进入临界区时，让属于同一临界区的共享数据保持一致。

优点：第一，将一系列共享数据项与各自的同步变量关联起来可以减少获取和释放一个同步变量所带来的额外开销。因为只有少数的数据项需要同步。第二，增加了并行度：这也使得多个包含不同共享数据的临界区可以同时执行

缺点：第一，同步变量和共享数据的关联带来了额外的复杂性和负载，程序设计可能更加复杂，容易出错。第二，关于入口一致性的一个程序设计问题是如何正确地将数据与同步变量关联起来。解决这个问题的一种方法是使用分布式的共享对象，向用户屏蔽低层的同步细节。

#### 以数据为中心的一致性的比较

[![Distributed System18](http://cyanny/myblog/wp-content/uploads/2013/11/Distributed-System18-580x276.png)](http://cyanny/myblog/wp-content/uploads/2013/11/Distributed-System18.png)
[![Distributed System19](http://cyanny/myblog/wp-content/uploads/2013/11/Distributed-System19-580x222.png)](http://cyanny/myblog/wp-content/uploads/2013/11/Distributed-System19.png)

### 以客户为中心的一致性模型

#### 1\. 最终一致性

条件： 没有更新操作时，所有副本逐渐成为相互完全相同的拷贝。最终一致性实际上只要求更新操作被保证传播到所有副本

优点：开销小

缺点：移动用户访问分布式数据库的不同副本时，如果副本没有更新，会出现不一致。

#### 2\. 单调读

条件：总是读最新值，如果一个进程读取数据x的值，那么该进程对执行任何后续读操作将总是得到第一次读取的那个值或更新的值。
如果一个进程已经在t时刻看到x的值，那么以后他不再会看到较老的版本的x的值。

[![Distributed System08](http://cyanny/myblog/wp-content/uploads/2013/11/Distributed-System08-580x359.png)](http://cyanny/myblog/wp-content/uploads/2013/11/Distributed-System08.png)

#### 3\. 单调写

条件：总是在最新的拷贝上写，一个进程对数据项x执行的写操作必须在该进程对x执行任何后续写操作之前完成，保证写操作以正确的顺序传播到数据存储的所有拷贝。 
在单调写一致性的定义中，同一进程的写操作的执行顺序与这些操作的启动顺序相同。

比较：注意，单调写一致性与数据为中心的FIFO一致性相似。FIFO一致性的本质是，同一进程执行的写操作必须在任何地方以正确的顺序执行。

这一顺序限制也适用于单调写一致性，只是我们这里考虑的是仅为单一进程维持的一致性，而不是为许多并发进程维持的一致性。

[![Distributed System09](http://cyanny/myblog/wp-content/uploads/2013/11/Distributed-System09-580x382.png)](http://cyanny/myblog/wp-content/uploads/2013/11/Distributed-System09.png)

### 4.写后读

条件：一个进程对数据项x执行一次写操作的结果总是会被该进程对x执行的后续读操作看见。 一个写操作总是在同一进程执行的后续读操作之前完成，而不管这个后续的读操作发生在什么位置。 

[![Distributed System10](http://cyanny/myblog/wp-content/uploads/2013/11/Distributed-System10-580x344.png)](http://cyanny/myblog/wp-content/uploads/2013/11/Distributed-System10.png)

### 5.读后写

条件：同一个进程对数据项x执行的读操作之后的写操作，保证发生在与x读取值相同或比之更新的值上。进程对数据项上x所执行的任何后续的写操作都会在x的拷贝上执行，而该拷贝是用该进程最近读取的值更新的。

在读取的最新值上进行写操作。 
[![Distributed System11](http://cyanny/myblog/wp-content/uploads/2013/11/Distributed-System11-580x340.png)](http://cyanny/myblog/wp-content/uploads/2013/11/Distributed-System11.png)

## 容错

### 1\. 两军问题

两军的通信信道不稳定，需要达成一致才能发动攻击，无论双方发了多少次确认，都不能确定通信兵是否把自己的消息带给对方，永远不会达成协议。
在不可靠传输的条件下，即使是无错误的进程，在两个进程之间达成协议也是不可能的。

### 2\. 拜占庭将军问题

通信良好，进程确不好，正如拜占庭的将军有叛徒的情况。

在这个问题中，红军还是在山谷中扎营，但是在附近的山上有n个带领部队的蓝军将领。通信是通过电话双向进行的，及时而且质量很好。但是有n个将军是叛徒(故障)，他们通过给忠诚的将军发送错误的和矛盾的信息(模拟故障进程)来阻止他们达成协议。现在问题在于忠诚的将军是否还能达成协议。 

Lamport采用递归算法来让无故障的进程达成协议。

N个进程，其中M个故障进程：只有当**N>=3m+1**时，才能达成一致，即有三分之二的进程是正常的。

### 3\. 两阶段提交

[![Distributed System12](http://cyanny/myblog/wp-content/uploads/2013/11/Distributed-System12-580x293.png)](http://cyanny/myblog/wp-content/uploads/2013/11/Distributed-System12.png)
状态机很能说明问题。

两个阶段：vote_commit and global_commit, 3(n-1)条消息

缺点：两阶段提交的一个问题在于当协调者崩溃时，参与者不能做出最后的决定。因此参与者可能在协调者恢复之前保持阻塞。

因此引入3PC，三阶段提交，但3PC实际中用的少，因为2PC阻塞情况很少出现。

### 4\. 三阶段提交

[![Distributed System13](http://cyanny/myblog/wp-content/uploads/2013/11/Distributed-System13-580x302.png)](http://cyanny/myblog/wp-content/uploads/2013/11/Distributed-System13.png)
没有一个可以直接转换到Commit或者Abort状态的单独状态。

没有一个这样的状态：它不能做出最后决定，而且可以从它直接转换到Commit状态。 Commit之前需要经过PRCOMMIT状态。

三阶段：vote-commit, prepare-commit, global-commit

引入了PRECOMMIT状态。

*   2PC：崩溃的参与者可能恢复到了Commit状态而所有参与者还处于Ready状态。在这种情况下，其余的可能操作进程不能做出最后的决定，不得不在崩溃的进程恢复之前阻塞。
*   在3PC中，只要有可操作的进程处于Ready状态，就没有崩溃的进程可以恢复到Init、Abort或Precommit之外的状态。因此存活进程总是可以做出的最后决定。当协调者崩溃，达到了预备提交阶段，在2PC中其他进程就会阻塞，但在3PC中，在预备提交阶段，即使没有协调者，也可以做出决定。

### 资源

[分布式系统总结part1 中间件，进程迁移，移动通信失效，名称解析，移动实体定位](http://cyanny/myblog/2013/11/24/distributed-system-review-part1/ "分布式系统总结part1")

[分布式系统总结part2 Lamport同步与向量时间戳，两大选举算法，三大互斥算法](http://cyanny/myblog/2013/11/24/distributed-system-review-part2/ "分布式系统总结part2")

[分布式系统总结part3 复制和一致性(以数据和以客户为中心的一致性)，容错（拜占庭将军问题，两阶段与三阶段提交）](http://cyanny/myblog/2013/11/24/distributed-system-review-part3/ "分布式系统总结part3")

[分布式系统总结part4 Petri网解决哲学家问题和生产者、消费者问题](http://cyanny/myblog/2013/11/24/distributed-system-review-part4/ "分布式系统总结part4")