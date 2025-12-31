## 两方隐私保护 ANN 检索协议

**模型介绍**：两方半诚实场景下的 Client-Server 架构隐私保护 ANN（Approximate Nearest Neighbor）检索协议。基于 Garnet 系统的 Kona KNN 实现进行改造。

**参与方**：
- **P0（检察院）**：持有查询 embedding，最终获得 top-k 最近的 fileId 列表
- **P1（法院）**：持有目标 embedding 库（每条对应一个 fileId）

**方案特点（A 方案）**：
- 聚类结果（clusterId）对 P1 可见
- 不涉及分类标签与投票逻辑
- 复用 Kona 的欧几里得距离计算优化（零在线通信）
- 复用 Kona 的 DQBubble Top-k 选择算法

---

### 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    离线阶段（Offline Phase）                      │
├─────────────────────────────────────────────────────────────────┤
│  阶段1: P1 明文 KMeans + 建索引                                   │
│    └── 输入: P1 的 embedding 库                                   │
│    └── 输出: centroids, clusterIndex, recordIndex->fileId 映射    │
├─────────────────────────────────────────────────────────────────┤
│  阶段2: DatasetShare (数据秘密共享)                               │
│    └── 输入: 每条 record 的 embedding                             │
│    └── 输出: P0/P1 各自的 (maskedVectorU, share) 数据             │
├─────────────────────────────────────────────────────────────────┤
│  阶段3: 生成离线三元组                                            │
│    └── 欧几里得距离计算所需的预计算数据                            │
│    └── DCF 密钥（用于安全比较）                                    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    在线阶段（Online Phase）                       │
├─────────────────────────────────────────────────────────────────┤
│  阶段1: AssignCluster (安全选类)                                  │
│    └── 输入: P0 的查询 embedding, P1 的 centroids                 │
│    └── 输出: selectedClusterId (A方案: P1可见)                    │
├─────────────────────────────────────────────────────────────────┤
│  阶段2: 类内 KNN                                                  │
│    └── 输入: 候选集的秘密共享数据                                  │
│    └── 处理: 安全距离计算 + DQBubble Top-k                        │
│    └── 输出: 仅 P0 获得 topKFileIds                               │
└─────────────────────────────────────────────────────────────────┘
```

---

### 数据结构定义

#### 输入文件格式

##### P1 的 Embedding 库文件: `{dataDir}/{datasetName}-P1-embeddings`
```
fileId emb[0] emb[1] ... emb[dim-1]
1001 0.123 -0.456 0.789 ...
1002 0.234 -0.567 0.890 ...
...
```

##### P0 的查询文件: `{dataDir}/{datasetName}-P0-queries`
```
queryId emb[0] emb[1] ... emb[dim-1]
0 0.111 -0.222 0.333 ...
1 0.444 -0.555 0.666 ...
...
```

#### 离线阶段输出文件

| 文件名 | 描述 | 格式 |
|--------|------|------|
| `{name}-meta` | 元数据 | 文本: `embDim numRecords numClusters` |
| `{name}-centroids` | 聚类中心 | 二进制: C×D 个 long long |
| `{name}-cluster-index` | 聚类索引 | 文本: 每行 `clusterId idx1 idx2 ...` |
| `{name}-P0-shares` | P0 的秘密共享数据 | 二进制 |
| `{name}-P1-shares` | P1 的秘密共享数据 | 二进制 |
| `{name}-P0-triples` | P0 的距离计算三元组 | 二进制 |
| `{name}-P1-triples` | P1 的距离计算三元组 | 二进制 |
| `2-fss/k0, k1, r0, r1` | DCF 密钥 | 文本 |

#### 核心数据结构

```cpp
// 秘密共享记录
struct SharedRecord {
    int recordIndex;              // 记录索引
    int fileId;                   // 文件ID
    int clusterId;                // 所属聚类ID
    vector<Z2<K>> maskedVectorU;  // U = v + delta_0 + delta_1 (公开值)
    vector<Z2<K>> share;          // P0: delta_0, P1: delta_1
};

// 距离-FileId 对（用于 Top-k 选择）
// array<Z2<K>, 2>: [0]=距离份额, [1]=fileId份额
vector<array<Z2<K>, 2>> distFileId_vec;
```

---

### 代码运行流程

#### 1. 编译

在 Garnet 根目录下执行：

```bash
# 编译离线阶段程序
make -j8 ann-party-offline.x

# 编译在线阶段程序
make -j8 ann-party.x
```

#### 2. 生成测试数据

```bash
# 生成测试数据：1000条记录，128维 embedding，10条查询
./ann-party-offline.x --gen-test 1000 128 10
```

#### 3. 运行离线阶段

```bash
# 执行 KMeans 聚类和数据秘密共享
./ann-party-offline.x --data-dir ./Player-Data/ANN-Data/ --dataset test --clusters 10
```

输出示例：
```
========================================
  ANN 离线阶段 - P1 KMeans + DatasetShare
========================================
数据目录: ./Player-Data/ANN-Data/
数据集名: test
聚类数量: 10
----------------------------------------
[IO] 加载 1000 条 embedding 记录

[阶段1] 执行 KMeans 聚类...
[KMeans] 开始聚类，数据量=1000, 维度=128, 聚类数=10
[KMeans] 迭代 0, 最大中心偏移=156.234
[KMeans] 迭代 10, 最大中心偏移=12.456
[KMeans] 在第 23 次迭代后收敛
[KMeans] 聚类完成，各聚类大小：
  Cluster 0: 98 条记录
  Cluster 1: 102 条记录
  ...

[阶段2] 生成秘密共享数据...
[DatasetShare] 开始生成秘密共享，记录数=1000, 维度=128
[DatasetShare] 秘密共享生成完成

[阶段3] 生成欧几里得距离三元组...
[Triples] 生成欧几里得距离三元组，记录数=1000, 维度=128
[Triples] 三元组生成完成

[阶段4] 生成 DCF 密钥...
[DCF] DCF 密钥已生成

========================================
  ANN 离线阶段完成！
========================================
```

#### 4. 生成 SSL 证书

```bash
./Scripts/setup-ssl.sh 2
```

#### 5. 运行在线阶段

打开两个终端窗口：

**终端1（P0 - 检察院）：**
```bash
./ann-party.x 0 -pn 11126 -h localhost -d ./Player-Data/ANN-Data/ -n test -k 5
```

**终端2（P1 - 法院）：**
```bash
./ann-party.x 1 -pn 11126 -h localhost -d ./Player-Data/ANN-Data/ -n test -k 5
```

输出示例（P0）：
```
========================================
  两方隐私保护 ANN 检索 - P0
========================================
数据目录: ./Player-Data/ANN-Data/
数据集名: test
Top-K: 5
----------------------------------------
[Network] 网络连接成功
[IO] 加载 P0 共享数据，记录数=1000
[IO] 加载 10 条查询

========================================
  ANN 在线阶段 - 处理 10 条查询
========================================

--- 查询 0 ---
[AssignCluster] P0 选中的聚类: 3
[Prepare] P0 收到 98 条候选记录
[KNN] 计算 98 条候选的距离...
[KNN] 执行 top-5 选择...
[KNN] Reveal top-5 fileId 给 P0...
[Result] 查询 0 的 Top-5 结果:
  #1: fileId=1045
  #2: fileId=1023
  #3: fileId=1067
  #4: fileId=1012
  #5: fileId=1089

...

========================================
  性能统计
========================================
总通信轮次: 1234
总运行时间: 2.45 秒
总通信量: 0.567 MB
```

---

### 核心算法说明

#### 1. KMeans 聚类

使用 KMeans++ 初始化，确保聚类中心分布均匀：

```cpp
// KMeans++ 初始化
vector<vector<long long>> initializeCentroids(records) {
    // 1. 随机选择第一个中心
    // 2. 后续中心按距离概率选择（距离越远概率越高）
}

// 迭代优化
while (!converged) {
    // 分配阶段：每个点分配到最近中心
    assignToNearestCenter(records, centers);
    
    // 更新阶段：重新计算聚类中心
    centers = updateCenters(records, centers);
}
```

#### 2. 数据秘密共享

采用 Kona 风格的 "Masked-Plain + Additive Share" 表示：

```cpp
// 对每个 embedding 维度
for (d in 0..embDim) {
    v = record.embedding[d];      // 原始值
    
    // 生成随机份额
    delta0 = random();
    delta1 = random();
    
    // 掩码明文（公开）
    U = v + delta0 + delta1;
    
    // P0 持有: (U, delta0)
    // P1 持有: (U, delta1)
}
```

#### 3. AssignCluster（A 方案）

A 方案允许 P1 知道选中的 clusterId：

```cpp
// P0 发送查询向量给 P1
if (playerno == 0) {
    send(queryVec);
    clusterId = receive();
}

// P1 明文计算最近聚类
if (playerno == 1) {
    queryVec = receive();
    for (cen in centroids) {
        dist = computeDistance(queryVec, cen.center);
        if (dist < minDist) {
            clusterId = cen.id;
        }
    }
    send(clusterId);
}
```

#### 4. 类内 KNN

基于 Kona 的实现进行改造：

**关键改动**：
- payload 从 label 改为 fileId
- 去掉多数投票逻辑（label_compute）
- Reveal 阶段只将 top-k 的 fileId 打开给 P0

```cpp
// 距离计算（复用 Kona 的零在线通信优化）
for (candidate in candidates) {
    distShare = computeEuclideanDistance(candidate, queryShare);
    fileIdShare = secretShare(candidate.fileId);
    distFileId_vec.push_back({distShare, fileIdShare});
}

// Top-k 选择（DQBubble 算法）
for (i in 0..k) {
    top_1(distFileId_vec, size - i, true);
}

// Reveal top-k fileId 给 P0
for (i in 0..k) {
    fileId = reveal_to_P0(distFileId_vec[size-1-i][1]);
    topKFileIds.push_back(fileId);
}
```

#### 5. DQBubble Top-k 选择

使用分治思想，将通信轮次从 O(kn) 降低到 O(k log n)：

```cpp
void top_1(shares, size, min_in_last) {
    compare_idx_vec = [0, 1, 2, ..., size-1];
    
    while (compare_idx_vec.size() > 1) {
        // 批量比较配对元素
        compare_in_vec(shares, compare_idx_vec, compare_res);
        
        // 安全交换
        SS_vec(shares, compare_idx_vec, compare_res);
        
        // 保留每对的较小/较大者
        compare_idx_vec = [1, 3, 5, ...];  // 奇数索引
    }
}
```

---

### 通信复杂度分析

| 阶段 | 通信轮次 | 通信量 |
|------|----------|--------|
| AssignCluster | O(1) | O(d) |
| 距离计算 | O(1)（零在线通信） | 0 |
| Top-k 选择 | O(k log n) | O(k·n) |
| Reveal | O(k) | O(k) |

其中：
- d = embedding 维度
- n = 候选集大小（cluster 内记录数）
- k = top-k 参数

---

### 安全性说明

**半诚实安全模型**：
- 假设双方遵循协议但可能尝试从收到的消息中推断额外信息
- P0 只能获得最终的 top-k fileId，无法得知候选集的其他信息
- P1 可以知道 clusterId（A 方案允许），但不知道查询结果

**数据保护**：
- P1 的 embedding 库：通过秘密共享保护，P0 只能看到掩码明文
- P0 的查询 embedding：在 AssignCluster 阶段发送给 P1（A 方案）
- 距离计算：使用预计算三元组，在线阶段无明文泄露
- Top-k 选择：使用安全比较和安全交换，中间结果不泄露

---

### Makefile 配置

在 Garnet 的 Makefile 中添加：

```makefile
# ANN 相关目标
ann-party-offline.x: Machines/ann-party-offline.o $(COMMON_OBJ)
	$(CXX) $(CFLAGS) -o $@ $^ $(LDLIBS)

ann-party.x: Machines/ann-party.o $(COMMON_OBJ)
	$(CXX) $(CFLAGS) -o $@ $^ $(LDLIBS)
```

---

### 参考文献

1. Kona KNN 实现：基于本项目的 `Machines/kona.cpp`
2. SecKNN [TIFs'24]: 安全 KNN 协议的通信优化
3. ABY2 秘密共享：两方加法秘密共享方案
4. DCF（分布式比较函数）：用于安全比较的函数秘密共享
