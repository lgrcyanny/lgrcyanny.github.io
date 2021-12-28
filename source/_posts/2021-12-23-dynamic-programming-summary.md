---
layout: posts
title: Dynamic Programming Summary
date: 2021-12-23 11:11:31
tags:
    - Algorithm
    - Dynamic Programming 
---

# 1. 动态规划之道
- **DP问题的特征**
    - 最优子结构: 原问题是一个最优化问题, 可递归地拆分为多个子问题, 通过数学方法组合各个子问题的最优解, 可以求得问题的最优解
    - 重叠的子问题: 子问题相互重叠, 例如斐波拉契数列问题, 而子问题如果不重叠, 可以用一般的递归
    - 无后效性: 某一个状态一旦确定, 就不受这个状态以后决策的影响, 例如地下城游戏
<!--more-->
- **DP解题思路**
    - 状态定义, 即定义子问题, 弄清楚原问题和子问题的关系, 例如dp[n], 表示0..n上问题的解, 可以考虑缩减问题规模
    - 状态转移, 即定义子问题之间的转换关系, 写出状态转移方程, 例如dp[i] = f(dp[i - 1]), i < n
    - 子问题相互重叠, 自底向上, 利用存储表, 一般是一维或二维数组, 如果数组稀疏, 可以用HashMap
    - 注意初始化和边界条件
- **DP与其他算法的区别**
    - DP与分治算法: 分治算法不要求子问题相互重叠
    - DP与贪心算法
        - 贪心要求最优子结构, 每一步的最优解包括上一步的最优解, 每次只求解一个最优子问题
        - 贪心不保证全局最优, DP保证
- **DP问题类型: 按结果类型分类**
    - 最值问题: 求最大值, 最小值
    - 计数问题: 求组合总数, 路径总数等
- **DP问题类型: 按状态转移方程形式分类**
    - 线性问题: 问题规模i从小到大, 大规模问题的解依赖小规模问题的最优解, 例如LIS, 最大子数组和
    - 前缀和问题: 求区间和 sum(i, j) = sums(0, j + 1) - sums(0, j)
    - 区间问题: 例如: 最长回文字符串, 最长回文子序列
    - 背包问题: 0-1背包问题, 完全背包问题
    - 状态压缩: 例如旅行商问题, 求经过所有点的最短路径
    - 计数问题: 例如卡特兰数
    - 数位问题
    - 矩阵快速幂: 对于线性递归式求解, 时间复杂度可以优化到O(logN)


# 2. 动态规划经典实战
## 2.1 线性动态规划
线性动态规划是指状态的推导按问题规模的大小从小到大, 较大问题的求解可以划分为小规模问题.
状态定义一般是一维数组dp[i], 状态转移会依赖O(n)个子问题或O(1)个子问题. 按问题类型分, 主要有:
- 单串问题: 包括最长上升子序列, 最大子数组和, 打家劫舍(不相邻子序列最大和), 需要两个位置的问题, 带维度的单串问题, 股票问题(带状态的单串问题)
- 双串问题: 包括最长公共子序列, 字符串匹配(例如最短编辑距离, 通配符匹配), 带维度的双串问题
- 矩阵问题: 最小路径和, 最大正方形, 最大矩形, 矩形区域不超过K的最大数值和
- 无串线性问题: 没有显示的字符串和数组, 但可以用线性动规, 例如丑数, 完全平方数

### 2.1.1 单串问题-最长上升子序列(LIS)
状态转移依赖O(n)个子问题
+ 状态定义
dp[i], 表示以nums[i]结尾的最长上升子序列的长度, 最终结果为max(dp[i]), 0 <= i < n
+ 状态转移方程
p[i] = max(dp[j]) + 1,  0<= j < i, nums[j] < nums[i]

```scala
def getLISLengthDP(arr: Array[Int]): Int = {
    val n = arr.size
    val memo = Array.ofDim[Int](n) // memo[i] is LIS from 0 to i
    memo(0) = 1
    var maxLen = 1
    for (i <- 1 until n) {
      memo(i) = 1
      for (j <- 0 until i) {
        if (arr(j) < arr(i) && memo(j) + 1 > memo(i)) {
          memo(i) = memo(j) + 1
        }
      }
      if (memo(i) > maxLen) {
        maxLen = memo(i)
      }
    }
    maxLen
}
```

### 2.1.2 单串问题-求具有最大和的连续子数组

给一个整数数组nums, 找一个具有最大和的连续子数组, 输出最大和
- 状态定义 
    - dp[i] 表示以nums[i]结尾的最大连续子数组和
    - 整个数组的最大连续子序和即所有dp[i]的最大值, res = max(dp[i]), 0 <= i < n
- 状态转移: (Kanade算法)
    - dp[i] = max(dp[i - 1], 0) + nums[i] 
    - 或者 dp[i] = max(nums[i], nums[i] + dp[i - 1])

```scala
def getMaxSubArr(nums: Array[Int]): Int = {
    var maxSum = nums(0)
    var maxEnding = 0
    for (i <- 0 until nums.size) {
      maxEnding = Math.max(maxEnding, 0) + nums(i)
      if (maxEnding > maxSum) {
        maxSum = maxEnding
      }
    }
    maxSum
}
```

**问题变种: 例如求循环子数组的最大和**
解法巧秒的是, 求子数组的最大和, 最小和, 最终结果为max(max_sub, all_sum - min_sub).
- 状态定义: 
    - max_dp[i] 表示以nums[i]结尾的最大连续子数组和
    - min_dp[i] 表示以nums[i]结尾的最小连续子数组和
- 状态转移
    - max_dp[i] = max(max_dp[i - 1] + nums(i), nums(i))
    - in_dp[i] = min(min_dp[i - 1] + nums(i), nums(i))
- 最终结果
    - max_dp = max(max_dp[i]),  0 <= i < n
    - min_dp = min(min_dp[i]),  0 <= i < n
    - max(max_dp, all_sum - min_dp)
这里为了优化空间复杂度为O(1), 只需记录max_dp[i]的最大值和min_dp[i]的最小值

```scala
def maxSubArraySumCircular(nums: Array[Int]): Int = {
    val n = nums.size
    if (n == 1) {
      nums(0)
    } else {
      var maxSum = nums.max
      var minSum = nums.min
      var maxEnding = 0
      var minEnding = 0
      for (i <- 0 until n) {
        maxEnding = Math.max(maxEnding + nums(i), nums(i))
        minEnding = Math.min(minEnding + nums(i), nums(i))
        maxSum = Math.max(maxSum, maxEnding)
        minSum = Math.min(minSum, minEnding)
      }
      val allSum = nums.sum
      if (maxSum < 0) {
        maxSum
      } else {
        Math.max(maxSum, allSum - minSum)
      }
    }
}
```


### 2.1.3 单串问题-打家劫舍(不相邻子序列最大和)

问题是一个小偷沿途偷盗房屋, 每个房屋内有现金, 相邻房屋有警报, 不能触发报警, 求能偷到的现金的最大值. 该问题是求不连续子序列的最大值
- 状态定义: dp[i]表示以i结尾的最大非连续子序列的和
- 状态转移: dp[i] = max(dp[i - 1], dp[i - 2] + nums(i))

```scala
def rob(nums: Array[Int]): Int = {
    val n = nums.size
    if (n == 1) {
      nums(0)
    } else {
      var maxEnding0 = nums(0)
      var maxEnding1 = Math.max(nums(1), maxEnding0)
      var maxSum = maxEnding1
      for (i <- 2 until n) {
        val t = Math.max(maxEnding1, maxEnding0 + nums(i))
        maxEnding0 = maxEnding1
        maxEnding1 = t
        maxSum = Math.max(maxSum, t)
      }
      maxSum
    }
}
```

### 2.1.4 单串问题-需要两个位置

求一个数组中最长斐波那契子序列的长度, 这里问题定义需要考虑两个位置
- 状态定义 
    - dp[i][j]表示以i, j结尾的最长斐波拉契子序列的长度
- 状态转移
    - dp[j][k] = dp[i][j] + 1, if A[i] + A[j] == A[k]
- 实现细节
    - 需要用一个HashSet维护一个Arr[k] -> k的索引表
    - 由于dp是稀疏二维矩阵, 用hashmap替代

```scala
def longestFibSubSeq(nums: Array[Int]): Int = {
    val n = nums.size
    val index = new mutable.HashMap[Int, Int]()
    for (i <- 0 until n) {
      index.put(nums(i), i)
    }
    var res = 0
    // dp as HashMap because 2-dimension array is sparse
    val dp = new mutable.HashMap[Int, Int]()
    for (k <- 0 until n) {
      for (j <- 0 until k) {
        // test if A[k] = A[i] + A[j]
        val i = index.getOrElse(nums(k) - nums(j), -1)
        if (i >= 0 && i < j) {
          dp(j * n + k) = dp.getOrElse(i * n + j, 2) + 1
          res = Math.max(res, dp(j * n + k))
        }
      }
    }
    if (res >= 3) {
      res
    } else {
      0
    }
}
```


### 2.1.5 单串问题-带维度的问题
经典的鸡蛋掉落问题, n层楼, k个鸡蛋, 求鸡蛋不碎的最少掉落次数
- 状态定义: dp[i][k]表示层数为i,k个鸡蛋的最小操作次数
- 状态转移: 假设在第f层抛鸡蛋, 两种情况
    - 鸡蛋碎了, 剩余k-1个鸡蛋, 在f层以下下搜索, 问题转换为dp[f - 1][k - 1]
    - 鸡蛋没有碎, 还是k个鸡蛋, 在f层以上搜索, 问题转换为dp[i - f][k]
    - 因此, dp[i][k] = 1 + min(max(dp[f - 1][k - 1], dp[i - f][k])), 1<= f <= i
    - 在搜索时需要用二分查找, 优化时间复杂度

```scala
def superEggDrop(K: Int, n: Int): Int = {
    val dp = Array.ofDim[Int](n + 1, K + 1)
    for (k <- 0 to K) {
      dp(0)(k) = 0
      dp(1)(k) = 1
    }
    for (i <- 1 to n) {
      dp(i)(1) = i
    }
    for (i <- 2 to n) {
      for (k <- 2 to K) {
        var left = 1
        var right = i
        while (left + 1 < right) {
          val x = (left + right) / 2
          val t1 = dp(x - 1)(k - 1)
          val t2 = dp(i - x)(k)
          if (t1 < t2) {
            left = x
          } else if (t1 > t2) {
            right = x
          } else {
            left = x
            right = x
          }
        }
        dp(i)(k) = 1 + Math.min(
          Math.max(dp(left - 1)(k - 1), dp(i - left)(k)),
          Math.max(dp(right - 1)(k - 1), dp(i - right)(k))
        )
      }
    }
    dp(n)(K)
}
```


### 2.1.6 单串问题-股票买卖, 考虑状态
给定一个数组, price[i]表示第i天的股票价格, 假设可以完成多笔交易, 最终获得股票的最大利润
- 状态定义: dp[i][0] 表示第i天结束, 手里没有股票的最大收益, dp[i][1]表示第i天结束, 手里有股票的最大收益
- 状态转移
    - dp[i][0] = max{dp[i - 1][0], dp[i - 1][1] + prices[i]}
    - dp[i][1] = max{dp[i - 1][1], dp[i - 1][0] - prices[i]}
- 边界条件
    - dp[0][0] = 0, dp[0][1] = -prices[0]
- 计算优化
    - 由于只依赖前一个值dp[i-1], 因此只需要保存两个变量dp0, dp1

```scala
def maxProfit(prices: Array[Int]): Int = {
    val n = prices.size
    var dp0 = 0
    var dp1 = -prices(0)
    for (i <- 1 until n) {
      dp0 = Math.max(dp0, dp1 + prices(i))
      dp1 = Math.max(dp1, dp0 - prices(i))
    }
    dp0
}
```


### 2.1.7 双串问题-最长公共子序列(LCS)

- 状态定义: dp[i][j]表示字符串text1 = 0..i和字符串text2=0..j的最大公共子序列长度
- 状态转移
    - dp[i][j] = max(dp[i-1][j], dp[i][j-1]), if text1[i] != text2[j]
    - dp[i][j] = dp[i-1][j-1] + 1, if text1[i] == text2[j]

```scala
def longestCommonSubsequence(text1: String, text2: String): Int = {
    val m = text1.size
    val n = text2.size
    val dp = Array.ofDim[Int](m + 1, n + 1)
    for (i <- 0 to m) {
      dp(i)(0) = 0
    }
    for (j <- 0 to n) {
      dp(0)(j) = 0
    }
    for (i <- 1 to m) {
      for (j <- 1 to n) {
        if (text1(i - 1) == text2(j - 1)) {
          dp(i)(j) = dp(i - 1)(j - 1) + 1
        } else {
          dp(i)(j) = Math.max(dp(i - 1)(j), dp(i)(j - 1))
        }
      }
    }
    println(printLCS(text1, text2, dp))
    dp(m)(n)
}
```

### 2.1.8 双串问题-字符串匹配
编辑距离: 求把字符串word1通过插入, 删除, 替换为word2的最短操作次数
- 状态定义: dp[i][j]表示从word1[0..i]变为word2[0..j]的最少操作步骤
- 状态转移:
    - if word1[i] == word2[j]
        - dp[i][j] = dp[i - 1][j - 1]
    - if word1[i] != word2[j]
        - deleteCost = dp[i-1][j] + 1
        - insertCost = dp[i][j - 1] + 1
        - updateCost = dp[i -1][j - 1] + 1
        - dp[i][j] = min(deleteCost, insertCost, updateCost)

```scala
def minDistance(word1: String, word2: String): Int = {
    val m = word1.size
    val n = word2.size
    val dp = Array.ofDim[Int](m + 1, n + 1)
    for (i <- 0 to m) {
      dp(i)(0) = i
    }
    for (j <- 0 to n) {
      dp(0)(j) = j
    }
    for (i <- 1 to m) {
      for (j <- 1 to n) {
        if (word1(i - 1) == word2(j - 1)) {
          dp(i)(j) = dp(i - 1)(j - 1)
        } else {
          dp(i)(j) = Array(dp(i - 1)(j), dp(i)(j - 1), dp(i - 1)(j - 1)).min + 1 // delete, insert, update cost
        }
      }
    }
    dp(m)(n)
}
```


**问题变种: 通配符匹配问题**
- 给定一个字符串 (s) 和一个字符模式 (p) ，实现一个支持 '?' 和 '*' 的通配符匹配, 输入s和p, 输出s是否和p匹配
- 状态定义: dp[i][j]表示字符串s[0..i], p[0..j]是否匹配
- 状态转移:
    - dp[i][j] = (dp[i - 1][j - 1])  if (s[i] == p[j]) or p[j] == '?'
    - dp[i][j] = or(dp[i-1][j], dp[i][j - 1]) , if p[j] == '*', 匹配时使用或不使用星号
    - dp[i][j] = false, 其他情况
- 边界条件
    - dp[0][0] = true
    - dp[i][0] = false
    - dp[0][j] = true, if p[0..j]都是星号

```scala
def minDistance(word1: String, word2: String): Int = {
    val m = word1.size
    val n = word2.size
    val dp = Array.ofDim[Int](m + 1, n + 1)
    for (i <- 0 to m) {
      dp(i)(0) = i
    }
    for (j <- 0 to n) {
      dp(0)(j) = j
    }
    for (i <- 1 to m) {
      for (j <- 1 to n) {
        if (word1(i - 1) == word2(j - 1)) {
          dp(i)(j) = dp(i - 1)(j - 1)
        } else {
          dp(i)(j) = Array(dp(i - 1)(j), dp(i)(j - 1), dp(i - 1)(j - 1)).min + 1 // delete, insert, update cost
        }
      }
    }
    dp(m)(n)
}
```


### 2.1.9 矩阵问题-最大正方形
在由'0'和'1'组成的二维矩阵内, 找到只包含'1'的最大正方形, 并返回面积
例如如下矩阵, 最大正方形面积为4

```scala
val matrix = Array(
      Array("1", "0", "1", "0", "0"),
      Array("1", "0", "1", "1", "1"),
      Array("1", "1", "1", "1", "1"),
      Array("1", "0", "0", "1", "0")
    )
```

- 状态定义: dp[i][j]表示以(i,j)为右下角的只包含1的最大正方形的边长
- 状态转移
    - dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1, if matrix[i][j] == 1
    - dp[i][j] = 0, if matrix[i][j] == 0
- 边界条件
    - dp[0][j] = 1, if matrix[0][j] == 1 
    - dp[i][0] = 1, if matrix[i][0] == 1

```scala
def maximalSquare(matrix: Array[Array[Char]]): Int = {
    val m = matrix.size
    val n = matrix(0).size
    val dp = Array.ofDim[Int](m, n)
    var res = 0
    for (i <- 0 until m) {
      for (j <- 0 until n) {
        if (matrix(i)(j) == '1') {
          if (i == 0 || j == 0) {
            dp(i)(j) = 1
          } else {
            dp(i)(j) = Array(dp(i - 1)(j), dp(i)(j - 1), dp(i - 1)(j - 1)).min + 1
          }
        } else {
          dp(i)(j) = 0
        }
        res = Math.max(res, dp(i)(j))
      }
    }
    res * res
}
```


### 2.1.10 矩阵问题-最大子矩阵
给定一个正整数、负整数和 0 组成的 N × M矩阵，编写代码找出元素总和最大的子矩阵。
返回一个数组 [r1, c1, r2, c2]，其中 r1, c1 分别代表子矩阵左上角的行号和列号，r2, c2 分别代表右下角的行号和列号。若有多个满足条件的子矩阵，返回任意一个均可

本题目的技巧是将二维矩阵问题转化为一维的动态规划问题
设数组colSum[k]表示从第i行到第j行的列和, 问题转化为求一维数组colSum的最大子数组和

```scala
def getMaxMatrix(matrix: Array[Array[Int]]): Array[Int] = {
    val res = Array.ofDim[Int](4)
    val N = matrix.length
    val M = matrix(0).length
    var maxSum = matrix(0)(0)
    var maxEnding = 0
    var tempX = 0 // temp coordinate
    var tempY = 0 // temp coordinate
    val colSum = Array.ofDim[Int](M) // column sum from i to j

    for (i <- 0 until N) {
      for (t <- 0 until M) {
        colSum(t) = 0
      }
      for (j <- i until N) {
        maxEnding = 0 // dp[i - 1]
        for (k <- 0 until M) {
          colSum(k) = colSum(k) + matrix(j)(k)
          if (maxEnding > 0) {
            maxEnding = colSum(k) + maxEnding
          } else {
            maxEnding = colSum(k)
            tempX = i
            tempY = k
          }
          if (maxEnding > maxSum) {
            maxSum = maxEnding
            res(0) = tempX
            res(1) = tempY
            res(2) = j
            res(3) = k
          }
        }
      }
    }
    res
}
```


## 2.2 前缀和问题

前缀和是线性动态规划的一种, 前缀和隐含了动态规划的思想
**1.什么是前缀和**
+ 状态定义：sums[i] := [0..i-1] 的和
+ 状态转移：sums[i] = a[i - 1] + sums[i - 1]
+ 初始化：sums[0] = 0

**2.常见问题**
+ 求区间和: sum(i, j) = sums(0, j + 1) - sums(0, j)
+ 快速求矩形和: sum(abcd)=sum(od)−sum(ob)−sum(oc)+sum(oa)
+ 结合数据结构哈希表, 记录查询前缀和结果
    + 和为k的最长子数组, key为前缀和, value为索引
    + 和为k的子数组个数, key为前缀和, value为count数
+ 逆运算差分: 差分数组的前缀和是原数组


### 2.2.1 求数组的区间和
给定一个整数数组  nums，求出数组从索引 i 到 j（i ≤ j）范围内元素的总和，包含 i、j 两点

```scala
class NumArray(_nums: Array[Int]) {
    val n = _nums.size
    val prefix = Array.ofDim[Int](n + 1)
    buildPrefixSum()

    private def buildPrefixSum(): Unit = {
      prefix(0) = 0
      for (i <- 1 to n) {
        prefix(i) = prefix(i - 1) + _nums(i - 1)
      }
    }

    def sumRange(left: Int, right: Int): Int = {
      prefix(right + 1) - prefix(left)
    }
}
```


### 2.2.2 用HashMap维护前缀和
给你一个整数数组 nums 和一个整数 k ，请你统计并返回该数组中和为 k 的连续子数组的个数
hashmap<前缀和的值, 出现次数>, 检测当pre[j−1] == pre[i]−k, count值累加
```scala
def subarraySum(nums: Array[Int], k: Int): Int = {
    val n = nums.size
    val map = new mutable.HashMap[Int, Int]()
    map.put(0, 1)
    var prefixSum = 0
    var maxCount = 0
    for (i <- 0 until n) {
      prefixSum = prefixSum + nums(i)
      val t = prefixSum - k
      if (map.contains(t)) {
        maxCount = maxCount + map(t)
      }
      map.put(prefixSum, map.getOrElse(prefixSum, 0) + 1)
    }
    maxCount
}
```



## 2.3 区间问题
- 状态定义dp[i][j], 表示原问题在区间[i..j]上的解
- 状态转移
    - 与常数个规模较小的子问题相关, 时间复杂度为O(n^2)
        - 例如: 最长回文子串, dp[i][j] = f(dp[i + 1][j], dp[i + 1][j - 1], dp[i][j - 1])
    - 与O(n)个更小规模的子问题有关, O(n^3)
        - dp[i][j] = g(f(dp[i][k], dp[k + 1][j])) 其中 k = i .. j-1


### 2.3.1 最长回文子串
求字符串s的最长回文子串, 
- 例如s="abbacd", 答案是"abba"
- s="a", 答案是"a"

解题思路:
注意最长回文子串, 不是最长回文子序列, 前者是连续的, 后者不要求连续, 因此dp数组要用boolean类型:
- 状态定义: dp[i][j]表示从i到j的子串是否是回文
- 状态转移:
    - dp[i][j] = dp[i+1][j - 1],  if s[i] == s[j], 长度大于2
    - dp[i][j] = false, if s[i] != s[j]
- 边界条件:
dp[i][i] = true
dp[i][j] = true, if j - i < 3, 类似"aa", "aca"这样的长度为2或3的字符串

```scala
def longestPalindrome(s: String): String = {
    val n = s.size
    val dp = Array.ofDim[Boolean](n, n)
    var maxLen = 1
    var begin = 0
    for (len <- 1 to n) {
      for (i <- 0 until n) {
        val j = i + len - 1
        if (j < n) {
          if (i == j) {
            dp(i)(j) = true
          } else if (s(i) != s(j)) {
            dp(i)(j) = false
          } else {
            if (j - i < 3) {
              dp(i)(j) = true // such as "aca"
            } else {
              dp(i)(j) = dp(i + 1)(j - 1)
            }
          }
          if (dp(i)(j) && j - i + 1 > maxLen) {
            maxLen = j - i + 1
            begin = i
          }
        }
      }
    }
    s.substring(begin, begin + maxLen)
}
```


### 2.3.2 最长回文子序列
给你一个字符串s ，找出其中最长的回文子序列，并返回该序列的长度, 例如s="bbbab", 输出是"bbbb"
- 状态定义: dp[i][j]表示从i到j的最长回文子串长度, 最终结果为dp[0][n - 1]
- 状态转移
    - dp[i][j] = dp[i+1][j - 1] + 2, if s[i] == s[j]
    - dp[i][j] = max{dp[i + 1][j], dp[i][j - 1]}, if s[i] != s[j]
```scala
def longestPalindromeSubseq(s: String): Int = {
    val n = s.size
    val dp = Array.ofDim[Int](n, n) // it's sparse, can replace with hashmap
    for (len <- 1 to n) {
      for (i <- 0 until n) {
        val j = i + len - 1
        if (j < n) {
          if (i == j) {
            dp(i)(j) = 1
          } else if (s(i) != s(j)) {
            dp(i)(j) = Math.max(dp(i + 1)(j), dp(i)(j - 1))
          } else {
            dp(i)(j) = dp(i + 1)(j - 1) + 2
          }
        }
      }
    }
    dp(0)(n - 1)
}
```


## 2.4 背包问题
**1.0-1背包问题**
- 问题定义: 有n种物品，物品j的体积为v(j), 价值为w(j), 有一个体积限制V。每种物品只有 1 个，只有选或者不选
- 状态定义: dp[i][j] := 考虑了[0..i]里的物品，占用了j空间，所能取得的最大价值
- 状态转移:
    - dp[i][j] = max(dp[i - 1][j] 当前物品不选, dp[i - 1][j - v[i]] + w[i] 当前物品选), if j - v[i] >= 0
- 空间优化
    - 用一维数组, 防止遍历时产生覆盖, j需要从大到小遍历
    - dp[j] = max{dp[j], dp[j - v[i]] + w[i]}
- 如果背包要求装满
    - 初始化dp[i][j]为-1, 表示方案不可取, dp[0][0] = 0
    - 状态转移时，需要判断dp[i - 1][j - v[i]] != -1

**2.完全背包问题**
- 问题定义: 有n种物品，物品j的体积为v[j]，价值为w[i]，有一个体积限制V, 每种物品有无限个
- 状态定义: dp[i][j] := 考虑了[0..i]里的物品，占用了j空间，所能取得的最大价值
- 状态转移:
    - dp[i][j] = max(dp[i - 1][j] 当前物品不选, **dp[i][j - v[i]]** + w[i] 当前物品选)，if j - v[i] >= 0
- 空间优化
    - 用一维数组, j从小到大遍历
    - dp[j] = max{dp[j], dp[j - v[i] + w[i]]}

**3.多重背包问题**
- 问题定义
有 n 种物品，物品 j 的体积为 v[j]，价值为 w[i]，有一个体积限制 V 。每种物品还有一个c[i] ，表示每种物品的个数
- 问题思路
对于物品 i, 数量限制是c[i] , 可以将其分成若干物品，它们的价值和体积为：(w[i], v[i]), (2 * w[i], 2 * v[i]) .., 超过体积限制的就不要, 然后对这些物品做0-1背包问题 

**4.问题类型**
    - 最值问题: 例如零钱兑换
    - 恰好取到背包容量: 例如分割等和子集
    - 组合问题（求方案数）: 需要考虑组合的顺序问题, 例如零钱兑换就是无顺序的


### 2.4.1 最值问题
零钱兑换问题
给你一个整数数组coins，表示不同面额的硬币, 硬币数量无限多；以及一个整数amount，表示总金额
计算并返回可以凑成总金额所需的最少的硬币个数。如果没有任何一种硬币组合能组成总金额，返回-1

这是一个完全背包问题
- 状态定义: dp[j]表示0..i种硬币, 找零总金额为j的最小硬币数量
- 状态转移: dp[j] = min{[j], dp[j - coins[i]] + 1} ,  j >= coins[i]
- 初始化
    - dp[j] = Int.Max - 1
    - dp[0] = 0

```scala
def coinChangeOpt(coins: Array[Int], amount: Int): Int = {
    val n = coins.size
    val dp = Array.ofDim[Int](amount + 1)
    dp(0) = 0
    for (j <- 1 to amount) {
      dp(j) = Int.MaxValue - 1
    }
    for (j <- 0 to amount) {
      for (i <- 0 until n) {
        if (coins(i) <= j) {
          dp(j) = Math.min(dp(j), dp(j - coins(i)) + 1)
        }
      }
    }
    val res = dp(amount)
    if (res > amount) -1 else res
}
```


### 2.4.2 恰好取到背包容量
给你一个只包含正整数的非空数组nums。请你判断是否可以将这个数组分割成两个子集，使得两个子集的元素和相等
先求数组总和total, 如果total是奇数, 则不可分, 如果是偶数可以继续下面的步骤
令halfSum = total / 2, 问题转化为0-1背包问题, 从数组中取一个子集, 其总和为halfSum, 并要求背包填满
```scala
def canPartition(nums: Array[Int]): Boolean = {
    val n = nums.size
    val totalSum = nums.sum
    if (n == 1 || totalSum % 2 != 0) {
      false
    } else {
      val halfSum = totalSum / 2
      val dp = Array.ofDim[Int](halfSum + 1)
      dp(0) = 0
      for (j <- 1 to halfSum) {
        dp(j) = -1
      }
      for (i <- 0 until n) {
        for (j <- halfSum to 0 by -1) {
          if (j >= nums(i) && dp(j - nums(i)) != -1) {
            dp(j) = Math.max(dp(j), dp(j - nums(i)) + nums(i))
          }
        }
      }
      dp(halfSum) != -1
    }
}
```


### 2.4.3 求组合的方案数
求零钱兑换问题的方案总数
完全背包问题, 组合没有顺序
- 状态定义: dp[j]表达前i个硬币, target为j时的组合数
- 状态转移
    - dp[j] = dp[j] + dp[j - coins[i]], if j >= nums[i]
    - j的遍历方向是从小到大, 因为coins[i]可以被选择多次
    - 由于组合没有顺序性, i的遍历在外层, j的遍历在内层
- 边界条件
    - dp[0] = 1, 表达target为0, 方案数为空集

```scala
def change(amount: Int, coins: Array[Int]): Int = {
    val n = coins.size
    val dp = Array.ofDim[Int](amount + 1)
    dp(0) = 1
    for (i <- 0 until n) {
      for (j <- 1 to amount) {
        if (j >= coins(i)) {
          dp(j) = dp(j) + dp(j - coins(i))
        }
      }
    }
    dp(amount)
}
```


## 2.5 状态压缩
状态压缩动态规划，用于NP问题的小规模求解, 是利用计算机二进制的性质来描述状态


### 2.5.1 旅行商问题
一个商人想要旅行各地并进行贸易。各地之间有若干条单向的通道相连，商人从一个地方出发，想要用最短的路程把所有地区环游一遍，请问环游需要的最短路程是多少？这里graph用邻接链表表示, 例如graph=[[1, 2, 3], [0], [0], [0]], 最短路径长度为4
- 预处理: 用floyd算法计算任意两个点对之间的最短路径distance[i][j]
- 状态定义:
    - dp[s][i]表达最后一个节点是i, 状态是s的最短路径
    - 状态s是一个mask, s的二进制s[v]表示已经搜索过节点v
- 状态转移
    - dp[s][i] =  min {dp[s\i][v] + distance[v][i]}, 遍历上一个节点v, v的状态是没有搜索过节点i, 求最短路径
    - 最终结果: min{dp[2^n - 1][i]}
    - 边界条件
        - 当s中只有一个1, 表示开始节点, dp[s][i] = 0
        - 默认dp[s][i] = Int.Max

```scala
  def shortestPathLength(graph: Array[Array[Int]]): Int = {
    val n = graph.size
    val states = 1 << n
    val dp = Array.ofDim[Int](states, n)
    val distance = Array.ofDim[Int](n, n + 1)
    // floyd algorithm for min distance between each pair
    for (i <- 0 until n) {
      for (j <- 0 until n) {
        distance(i)(j) = n + 1
        if (graph(i).contains(j)) {
          distance(i)(j) = 1
        }
      }
    }
    for (k <- 0 until n) {
      for (i <- 0 until n) {
        for (j <- 0 until n) {
          distance(i)(j) = Math.min(distance(i)(j), distance(i)(k) + distance(k)(j))
        }
      }
    }
    // dp algorithm
    for (s <- 0 until states) {
      for (i <- 0 until n) {
        dp(s)(i) = Int.MaxValue
      }
    }
    for (s <- 1 until states) {
      for (i <- 0 until n) {
        // s is 2 ^ k
        if ((s & (s - 1)) == 0) {
          dp(s)(i) = 0
        } else {
          if ((s & 1 << i) != 0) {
            for (v <- 0 until n) {
              if ((s & (1 << v)) != 0 && (v != i)) {
                dp(s)(i) = Math.min(dp(s)(i), dp(s ^ (1 << i))(v) + distance(v)(i))
              }
            }
          }
        }
      }
    }
    dp(states - 1).min
}
```

**思路二: 用广度优先遍历+状态压缩**

```scala
def shortestPathLengthBFS(graph: Array[Array[Int]]): Int = {
    val visited = new mutable.HashSet[(Int, Int)]()
    val queue = new mutable.Queue[(Int, Int, Int)]() // identity, mask, distance
    val n = graph.size
    for (i <- 0 until n) {
      visited.add((i, 1 <<i))
      queue.enqueue((i, 1 << i, 0))
    }
    var isDone = false
    var res = 0
    while (queue.nonEmpty && !isDone) {
      val (i, mask, dist) = queue.dequeue()
      if (mask == ((1 << n) - 1)) {
        isDone = true
        res = dist
      }
      for (j <- graph(i)) {
        val maskForJ = mask | 1 << j
        if (!visited.contains((j, maskForJ))) {
          queue.enqueue((j, maskForJ, dist + 1))
          visited.add((j, maskForJ))
        }
      }
    }
    res
}
```


## 2.6 计数问题
有两种模式:
1.找到组合数公式，然后用DP的方式或者用含阶乘的公式求组合数, 例如: 路径问题
2.找到递归关系，然后以DP的方式求这个递推关系，如果是线性递推关系，可以用矩阵快速幂加速 例如:隐晦的递推关系: 栅栏涂色


### 2.6.1 组合数问题
给你一个整数n ，求恰由n个节点组成且节点值从1到n互不相同的二叉搜索树有多少种
- 状态定义
    - 令f(i, n): 表达序列长度为n, 以i为根的二叉搜索树长度
    - g(n) 表达长度为n的二叉搜索树长度
- 状态转移
    - g(n) = sum{f(i, n)}
    - f(i, n) = g(i - 1)*g(n - i)
    - g(n) = sum{g(i - 1)*g(n - i)}, g(n)这样的组合数, 是数学上的卡特兰数

```scala
def numTrees(n: Int): Int = {
    val dp = Array.ofDim[Int](n + 1)
    dp(0) = 1
    dp(1) = 1
    for (i <- 2 to n) {
      for (j <- 1 to i) {
        dp(i) = dp(i) + dp(j - 1) * dp(i - j)
      }
    }
    dp(n)
}
```


### 2.5.1 隐晦的递推关系
有n个一样的骰子，每个骰子上都有f个面，分别标号为 1, 2, ..., f
约定：掷骰子的得到总点数为各骰子面朝上的数字的总和, 求总点数为target的组合总数, 结果比较大模10^9+7
- 状态定义:
    - dp[i][j] 表示i个骰子, target为j的组合总数
- 状态转移:
    - dp[i][j] = d[i - 1][j - 1] + .. + d[i - 1][j - f], 遍历f种骰子的值
- 初始化:
    - dp[0][0] = 0
- 空间优化
    - 用滚动数组, 类似0-1背包问题, 用dp[j]就可以,
    - 为了防止覆盖, 遍历方向是从大到小
    - 每一轮遍历j时, 需要dp[j] = 0

```scala
def numRollsToTarget(n: Int, f: Int, target: Int): Int = {
    val mod = (1e9 + 7).toInt
    val dp = Array.ofDim[Int](target + 1)
    dp(0) = 1
    for (i <- 1 to n) {
      for (j <- target to 0 by -1) {
        dp(j) = 0
        for (k <- 1 to f if j >= k) {
          dp(j) = (dp(j) + dp(j - k)) % mod
        }
      }
    }
    dp(target)
}
```


## 2.7 数位问题
求解在一段区间上[L, R]上满足条件的数字的个数
例如, 求最大为N的数字组合
我们有一组排序的数字 D，它是{'1','2','3','4','5','6','7','8','9'} 的非空子集.（请注意，'0' 不包括在内）
用这些数字写数字, 例如'112', '335', 给定一个整数N, 返回可以用D中的数字能写出的小于或等于N的正整数的数目

- 状态定义
    - 令N的数位总数为K
    - dp[i]表示除掉N前面的i位, 剩余的K - i位的合法组合总数, 例如N=2345, dp[0]表示2345, dp[1]表示345,
- 状态转移
    - 当s[i] == d, dp[i] += dp[i+ 1], d是digits的十进制值, 顶到s[i]的上边界
    - 当s[i] > d, dp[i] += d_length ** (K - i - 1)
- 最终答案
    - dp[0] += 所有数位小于K的组合总数

```scala
def atMostNGivenDigitSet(digits: Array[String], N: Int): Int = {
    val nStr = N.toString
    val dLen = digits.size
    val digitsValue = digits.map(_.toInt)
    val K = nStr.size
    val dp = Array.ofDim[Int](K + 1)
    dp(K) = 1
    for (i <- K - 1 to 0 by -1) {
      val iValue = nStr(i) - '0'
      for (d <- digitsValue) {
        if (d < iValue) {
          dp(i) = dp(i) + Math.pow(dLen, K - i - 1).toInt
        } else if (d == iValue) {
          dp(i) = dp(i) + dp(i + 1)
        }
      }
    }

    for (i <- 1 until K) {
      dp(0) = dp(0) + Math.pow(dLen, i).toInt
    }
    dp(0)
}
```


## 写在后面
+ 动态规划的主要特征是最优子结构(最值问题, 组合方案数问题等), 重叠子问题和无后效性
+ 实际问题的难点是识别动态规划的模式, 本篇中主要的模式包括线性问题, 前缀和问题, 区间问题, 背包问题, 状态压缩问题, 计数问题, 数位问题
+ 其实工作中能遇到用动规的场景不多, 学了这个技艺有什么用呢? 我想可以用王国维的一句话, "无用之用, 实为大用"

参考:
- [动态规划精讲 By Leetcode](https://leetcode-cn.com/leetbook/detail/dynamic-programming-1-plus/)
- Dynamic Programming for Coding Interviews

