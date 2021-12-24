---
layout: posts
title: Dynamic Programming Summary
date: 2021-12-23 11:11:31
tags:
    - Algorithm
    - Dynamic Programming 
---

# 1. 动态规划之道
    • DP问题的特征
        ○ 最优子结构: 原问题是一个最优化问题, 可递归地拆分为多个子问题, 通过数学方法组合各个子问题的最优解, 可以求得问题的最优解
        ○ 重叠的子问题: 子问题相互重叠, 例如斐波拉契数列问题, 而子问题如果不重叠, 可以用一般的递归
        ○ 无后效性: 某一个状态一旦确定, 就不受这个状态以后决策的影响, 例如地下城游戏
    • DP解题方法
        ○ 状态定义, 即定义子问题, 弄清楚原问题和子问题的关系, 例如dp[n], 表示0..n上问题的解, 可以考虑缩减问题规模
        ○ 状态转移, 即定义子问题之间的转换关系, 写出状态转移方程, 例如dp[i] = f(dp[i - 1]), i < n
        ○ 子问题相互重叠, 自底向上, 利用存储表, 一般是一维或二维数组, 如果数组稀疏, 可以用HashMap
        ○ 注意初始化和边界条件
    • DP与其他算法的区别
        ○ DP与分治算法
            § 分治算法不要求子问题相互重叠
        ○ DP与贪心算法
            § 贪心:
                □ 要求最优子结构, 每一步的最优解包括上一步的最优解, 每次只求解一个最优子问题
                □ 不保证全局最优, DP保证
    • DP问题类型: 按结果类型
        ○ 最值问题: 求最大值, 最小值
        ○ 计数问题: 求组合总数, 路径总数等
    • DP问题类型: 按状态转移方程形式分类
        ○ 线性问题
            § 问题规模i从小到大, 大规模问题的解依赖小规模问题的最优解, 例如LIS, 最大子数组和
        ○ 前缀和问题
            § 求区间和 sum(i, j) = sums(0, j + 1) - sums(0, j)
        ○ 区间问题, 例如: 最长回文字符串, 最长回文子序列
        ○ 背包问题, 0-1背包问题, 完全背包问题
        ○ 状态压缩: 例如旅行商问题, 求经过所有点的最短路径
        ○ 计数问题: 例如卡特兰数
        ○ 数位问题
        ○ 矩阵快速幂: 对于线性递归式求解, 时间复杂度可以优化到O(logN)


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

问题变种: 例如求循环子数组的最大和
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
状态定义: dp[i]表示以i结尾的最大非连续子序列的和
状态转移:
dp[i] = max(dp[i - 1], dp[i - 2] + nums(i))
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
        ○ dp[i][j] = max(dp[i-1][j], dp[i][j-1]), if text1[i] != text2[j]
        ○ dp[i][j] = dp[i-1][j-1] + 1, if text1[i] == text2[j]

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

















