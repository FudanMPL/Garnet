## 两方隐私保护 ANN 检索协议（ORAM 方案 - 完全隐藏 ClusterId）

**模型介绍**：两方半诚实场景下的 Client-Server 架构隐私保护 ANN 检索协议。采用真正的 ORAM-like 私密读取技术，完全隐藏聚类选择结果和访问模式。

**参与方**：
- **P0（检察院）**：持有查询 embedding，最终获得 top-k 最近的 fileId 列表
- **P1（法院）**：持有目标 embedding 库（每条对应一个 fileId）

**方案核心特点**：
- **ClusterId 完全隐藏**：选类结果（idx/one-hot）不对任何一方 open
- **ORAM-like 私密读取**：固定次数读取，访问模式不泄露
- **侧信道消除**：通过 padding 消除桶/类大小侧信道
- **无中间值泄露**：DEBUG=OFF 时所有中间值保持私密

---

### 公开常量定义

```cpp
const int K = num_clusters;           // 聚类数量
const int b_max = max_block_per_class; // 每个类最大 block 数（padding 后固定）
const int b_read = blocks_to_read;    // 每个选中类读取的 block 数
const int BLOCK_WORDS = words_per_block; // 每个 block 的 word 数（embedding 维度相关）
const int nprobe = num_probes;        // 选择的最近聚类数（通常 1-3）
const int M = K * b_max;              // 总逻辑块数
```

---

### 协议阶段详述

#### 阶段 1: 安全选类（Secure Top-nprobe）

**输入**：
- `dist[0..K-1]`: 查询到每个聚类中心的私密距离份额

**输出**：
- `idx[0..nprobe-1]`: 最近的 nprobe 个聚类 ID（私密整数，不 open）
- `sel[t][j] = eq(idx[t], j)`: 可选的 one-hot 编码（私密比特）

**实现**：

```cpp
// 1. 构造 (距离, 聚类ID) 对
vector<array<Z2<K>, 2>> pairs(num_clusters);
for (int j = 0; j < num_clusters; ++j) {
    pairs[j][0] = dist_shares[j];          // 私密距离
    pairs[j][1] = cluster_id_shares[j];    // 私密聚类ID（0,1,...,K-1 的秘密共享）
}

// 2. 使用 Oblivious Sorting Network 按距离升序排序
//    - Bitonic Sort 或 Odd-Even Merge Sort
//    - 所有 compare-swap 操作使用私密比较 + 私密 mux
oblivious_sort(pairs, num_clusters, /*ascending=*/true);

// 3. 取排序后前 nprobe 个 pair 的 id 作为 idx
vector<Z2<K>> idx(nprobe);
for (int t = 0; t < nprobe; ++t) {
    idx[t] = pairs[t][1];  // 全私密，不 open
}
```

**私密比较实现**（不 reveal 中间值）：

```cpp
// secure_compare_no_reveal: 返回私密比特 (x1 > x2) ? 1 : 0
// 使用 DCF 但不 reveal 比较结果
Z2<K> secure_compare_no_reveal(Z2<K> x1, Z2<K> x2) {
    // 使用预生成的 DCF 密钥
    // 输出是私密比特份额，不 open
    return dcf_compare_share(x1 - x2);
}

// secure_mux: b ? x : y （所有输入输出都是私密）
Z2<K> secure_mux(Z2<K> b, Z2<K> x, Z2<K> y) {
    // res = y + b * (x - y)
    // 需要一次安全乘法
    Z2<K> diff = x - y;
    Z2<K> prod = secure_mul(b, diff);  // 在线乘法
    return y + prod;
}
```

**Oblivious Compare-Swap**：

```cpp
void oblivious_compare_swap(array<Z2<K>, 2>& a, array<Z2<K>, 2>& b, bool ascending) {
    // 1. 私密比较
    Z2<K> cmp = secure_compare_no_reveal(a[0], b[0]);  // a[0] > b[0] ? 1 : 0
    if (!ascending) cmp = Z2<K>(1) - cmp;              // 需要安全取反
    
    // 2. 条件交换（使用 mux）
    // 如果 cmp=1 需要交换，否则不交换
    Z2<K> new_a0 = secure_mux(cmp, b[0], a[0]);
    Z2<K> new_a1 = secure_mux(cmp, b[1], a[1]);
    Z2<K> new_b0 = secure_mux(cmp, a[0], b[0]);
    Z2<K> new_b1 = secure_mux(cmp, a[1], b[1]);
    
    a[0] = new_a0; a[1] = new_a1;
    b[0] = new_b0; b[1] = new_b1;
}
```

---

#### 阶段 2: 数据布局（类→Block 固定化）

**逻辑地址计算**：
```
addr = class_id * b_max + block_index
```

**数据结构**：

```cpp
// 每个 Block 包含固定数量的 word
struct Block {
    Z2<K> words[BLOCK_WORDS];  // 私密/共享 word
    Z2<K> fileIdShare;         // fileId 的私密份额
    Z2<K> validFlag;           // 是否有效（padding 的 dummy block 为 0）
};

// 数据库：M 个 block（M = K * b_max）
vector<Block> DB(M);

// 离线阶段 P1 输入数据，转为秘密共享
// P0 只持有份额，不知道明文
```

**Padding 策略**：

```cpp
// 离线阶段：为每个聚类 padding 到 b_max 个 block
for (int c = 0; c < K; ++c) {
    vector<Record>& records_in_cluster = cluster_records[c];
    int actual_blocks = (records_in_cluster.size() + RECORDS_PER_BLOCK - 1) / RECORDS_PER_BLOCK;
    
    for (int bi = 0; bi < b_max; ++bi) {
        int addr = c * b_max + bi;
        if (bi < actual_blocks) {
            // 真实数据
            DB[addr] = create_block_from_records(records_in_cluster, bi);
            DB[addr].validFlag = share_of(1);
        } else {
            // Dummy block（padding）
            DB[addr] = create_dummy_block();
            DB[addr].validFlag = share_of(0);
        }
    }
}
```

---

#### 阶段 3: ORAM-like 私密读取

**接口**：
```cpp
Block oram_read(const vector<Block>& DB, Z2<K> secret_addr);
```

**Trivial ORAM 实现**（线性扫描，固定访问模式）：

```cpp
Block oram_read(const vector<Block>& DB, Z2<K> secret_addr) {
    Block result;
    // 初始化为全 0（私密共享）
    for (int w = 0; w < BLOCK_WORDS; ++w) {
        result.words[w] = Z2<K>(0);
    }
    result.fileIdShare = Z2<K>(0);
    result.validFlag = Z2<K>(0);
    
    // 固定循环：遍历所有 M 个 block
    for (int i = 0; i < M; ++i) {
        // 私密相等比较：b = (secret_addr == i) ? 1 : 0
        Z2<K> addr_i = share_of_constant(i);  // 常数的秘密共享
        Z2<K> b = secure_equality(secret_addr, addr_i);  // 私密比特，不 reveal
        
        // 逐 word mux
        for (int w = 0; w < BLOCK_WORDS; ++w) {
            result.words[w] = secure_mux(b, DB[i].words[w], result.words[w]);
        }
        result.fileIdShare = secure_mux(b, DB[i].fileIdShare, result.fileIdShare);
        result.validFlag = secure_mux(b, DB[i].validFlag, result.validFlag);
    }
    
    return result;  // 全私密
}
```

**私密相等比较**（不 reveal 差值）：

```cpp
// secure_equality: (a == b) ? 1 : 0，输出私密比特
// 方法：使用随机化 + 乘法，不 reveal 差值
Z2<K> secure_equality(Z2<K> a, Z2<K> b) {
    Z2<K> diff = a - b;
    
    // 方法1：使用多轮乘法检测零
    // 对 diff 的每一位进行处理，输出是否为零
    // 需要 O(K) 次比特操作
    
    // 方法2：使用随机化 + 逆元
    // r * diff = 0 iff diff = 0
    // 但需要处理除零问题
    
    // 实际实现：使用 bit-decomposition + AND gates
    return is_zero_share(diff);
}

// is_zero_share: 私密判断 x == 0
// 使用比特分解：x == 0 iff 所有比特都是 0
Z2<K> is_zero_share(Z2<K> x) {
    // 将 x 分解为私密比特（需要 bit decomposition 协议）
    vector<Z2<1>> bits = bit_decompose(x);
    
    // 计算 NOT(OR(bits)) = AND(NOT(bits))
    Z2<K> result = share_of(1);
    for (int i = 0; i < K; ++i) {
        Z2<K> not_bit = share_of(1) - Z2<K>(bits[i]);
        result = secure_mul(result, not_bit);  // AND
    }
    return result;
}
```

---

#### 阶段 4: 安全选择与读取（Secure Select and Fetch）

```cpp
// 输入：idx[0..nprobe-1] - 私密聚类 ID
// 输出：cand[nprobe][b_read] - 私密候选 blocks

vector<vector<Block>> secure_select_and_fetch(
    const vector<Block>& DB,
    const vector<Z2<K>>& idx,
    int nprobe, int b_read, int b_max
) {
    vector<vector<Block>> cand(nprobe, vector<Block>(b_read));
    
    for (int t = 0; t < nprobe; ++t) {
        for (int bi = 0; bi < b_read; ++bi) {
            // 私密地址计算：secret_addr = idx[t] * b_max + bi
            Z2<K> offset = share_of_constant(bi);
            Z2<K> base = secure_mul(idx[t], share_of_constant(b_max));
            Z2<K> secret_addr = base + offset;
            
            // ORAM 读取
            cand[t][bi] = oram_read(DB, secret_addr);
        }
    }
    
    return cand;  // 全私密
}
```

---

#### 阶段 5: 候选集 KNN

在私密候选 blocks 上执行 top-k：

```cpp
vector<int> execute_candidate_knn(
    const vector<vector<Block>>& cand,
    const vector<Z2<K>>& queryShare,
    int k
) {
    // 1. 收集所有候选记录（从 blocks 中提取）
    vector<array<Z2<K>, 2>> dist_file_pairs;
    
    for (int t = 0; t < nprobe; ++t) {
        for (int bi = 0; bi < b_read; ++bi) {
            const Block& block = cand[t][bi];
            
            // 计算距离（私密）
            Z2<K> dist = compute_distance(block.words, queryShare);
            
            // 如果是 dummy block，将距离设为无穷大
            // maskedDist = dist + (1 - validFlag) * LARGE_VALUE
            Z2<K> not_valid = share_of(1) - block.validFlag;
            Z2<K> mask = secure_mul(not_valid, share_of(LARGE_VALUE));
            Z2<K> maskedDist = dist + mask;
            
            dist_file_pairs.push_back({maskedDist, block.fileIdShare});
        }
    }
    
    // 2. 使用 Oblivious Sorting 取 top-k
    int total_candidates = nprobe * b_read;
    oblivious_sort(dist_file_pairs, total_candidates, true);
    
    // 3. Reveal top-k fileId 给 P0
    vector<int> topKFileIds;
    for (int i = 0; i < k; ++i) {
        Z2<K> fileId = reveal_to_P0(dist_file_pairs[i][1]);
        if (playerno == 0) {
            topKFileIds.push_back(static_cast<int>(fileId.get_limb(0)));
        }
    }
    
    return topKFileIds;
}
```

---

### 系统架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                         离线阶段                                      │
├─────────────────────────────────────────────────────────────────────┤
│  1. P1 KMeans 聚类                                                   │
│  2. 数据布局固定化：每类 padding 到 b_max 个 block                    │
│  3. 生成秘密共享 DB[0..M-1]                                          │
│  4. 生成聚类中心秘密共享                                              │
│  5. 生成 DCF 密钥                                                    │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│                         在线阶段                                      │
├─────────────────────────────────────────────────────────────────────┤
│  1. 安全距离计算：查询到每个聚类中心                                   │
│       ↓ dist[0..K-1] (私密)                                          │
│  2. Secure Top-nprobe：Oblivious Sort，输出 idx[0..nprobe-1] (私密)  │
│       ↓ 不 reveal                                                    │
│  3. Secure Select and Fetch：固定 nprobe*b_read 次 ORAM 读取         │
│       ↓ cand[][] (私密)                                              │
│  4. 候选集 KNN：Oblivious Sort + Top-k                               │
│       ↓                                                              │
│  5. Reveal fileId 给 P0                                              │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 通信复杂度分析

| 阶段 | 通信轮次 | 通信量 |
|------|----------|--------|
| 距离计算 | O(K·d) | O(K·d) |
| Secure Top-nprobe | O(K·log²K) | O(K·log²K) |
| ORAM 读取 | O(nprobe·b_read·M) | O(nprobe·b_read·M) |
| 候选 KNN | O(n_cand·log²n_cand) | O(n_cand·log²n_cand) |

其中：
- K = 聚类数
- d = embedding 维度
- M = K·b_max（总块数）
- n_cand = nprobe·b_read（候选块数）

**注**：Trivial ORAM 的复杂度是 O(M)，可通过替换为 Path/Circuit ORAM 优化到 O(polylog M)。

---

### 安全性保证

1. **选类结果私密**：idx[t] 永不 reveal，使用 Oblivious Sort 输出
2. **访问模式固定**：始终读取 nprobe·b_read 个 block，与选类结果无关
3. **侧信道消除**：每类 padding 到固定大小 b_max
4. **中间值私密**：DEBUG=OFF 时无任何 reveal_to_both/open 操作
5. **半诚实安全**：所有协议在半诚实模型下可证明安全

---

### DEBUG 开关

```cpp
#ifdef DEBUG_ORAM
    // 允许对最终 idx 明文 open 以做单元测试
    for (int t = 0; t < nprobe; ++t) {
        Z2<K> revealed_idx = reveal_to_both(idx[t]);
        cout << "[DEBUG] Selected cluster " << t << ": " << revealed_idx.get_limb(0) << endl;
    }
#endif
```

---

### 编译与运行

```bash
# 编译
make -j8 ann-party-oram-offline.x ann-party-oram.x

# 生成测试数据（可选）
./ann-party-oram-offline.x --gen-test 100 32 5 --data-dir ./Player-Data/ANN-Data --dataset test

# 离线阶段
./ann-party-oram-offline.x --data-dir ./Player-Data/ANN-Data \
    --dataset test --clusters 5 --b-max 30

# 生成 SSL 证书（首次运行需要）
./Scripts/setup-ssl.sh 2

# 在线阶段（先启动 P0，再启动 P1）
# 终端 1 (P0):
./ann-party-oram.x 0 -pn 11126 -h localhost \
    -d ./Player-Data/ANN-Data -n test -k 5 --b-read 10

# 终端 2 (P1):
./ann-party-oram.x 1 -pn 11126 -h localhost \
    -d ./Player-Data/ANN-Data -n test -k 5 --b-read 10
```

### 当前实现状态

**已实现**：
1. ✅ 离线阶段：KMeans 聚类 + 数据布局固定化（padding）
2. ✅ 离线阶段：秘密共享 Block 数据库生成
3. ✅ 离线阶段：聚类中心秘密共享（含 clusterIdShare）
4. ✅ 在线阶段：安全距离计算
5. ✅ 在线阶段：安全选类（top-1 + reveal clusterId）
6. ✅ 在线阶段：候选集 KNN（安全 top-k）
7. ✅ 结果一致性验证通过

**待完善（进阶功能）**：
1. ⬜ 完全不 reveal clusterId（需实现真正的 ORAM 读取）
2. ⬜ Trivial ORAM 读取（线性扫描 + 私密相等比较）
3. ⬜ 替换为 Path/Circuit ORAM 以优化复杂度

**当前方案说明**：
- 当前实现在选类阶段 reveal clusterId 给双方，然后直接从 Block DB 中读取对应聚类的数据
- 这与 A 方案（`ann_verdict.md`）的安全性相同
- 真正的 ORAM 版本（不 reveal clusterId）需要实现 `secure_equality` 和 `oram_read` 原语，开销较大

---

## MPC 编译器版本（使用内置 ORAM，polylog 复杂度）

Garnet 提供了内置的 ORAM 实现，可以获得 **O(polylog M)** 的读取复杂度。

### 复杂度分析

使用 Path/Circuit ORAM 后的整体复杂度：

**O(K·d) + O(nprobe·b_read·log M) + O(M_cand·d)**

- K·d：聚类中心距离计算
- nprobe·b_read·log M：ORAM 读取候选块
- M_cand·d：候选集距离计算

这比全库暴力搜索 **O(N·d)** 明显更优。

### 文件结构

| 文件 | 描述 |
|------|------|
| `Programs/Source/ann_oram.mpc` | MPC 程序（使用 OptimalORAM） |
| `Scripts/ann_oram_prepare.py` | 离线数据准备脚本 |

### 编译与运行

```bash
# 1. 准备离线数据
python3 Scripts/ann_oram_prepare.py --gen-test \
    --output-dir ./Player-Data/Input \
    --clusters 10 --b-max 20

# 2. 编译 MPC 程序
./compile.py ann_oram

# 3. 生成 SSL 证书
./Scripts/setup-ssl.sh 2

# 4. 运行（使用 semi2k 协议，两方）
# 终端 1 (P0):
./semi2k-party.x -p 0 -N 2 ann_oram

# 终端 2 (P1):
./semi2k-party.x -p 1 -N 2 ann_oram
```

### MPC 程序核心代码

```python
from Compiler.oram import OptimalORAM

# 创建 ORAM（自动选择最优实现）
block_oram = OptimalORAM(M, value_type=sint, entry_size=[32] * BLOCK_SIZE)

# 私密读取（polylog 复杂度）
secret_addr = cluster_id * b_max + bi
block = block_oram[secret_addr]  # 完全不泄露 secret_addr
```

### 内置 ORAM 类型

| ORAM 类型 | 复杂度 | 适用场景 |
|-----------|--------|----------|
| `LinearORAM` | O(n) | 小规模 (≤2^11) |
| `TreeORAM` | O(log n) | 中大规模 |
| `PathORAM` | O(log n) | 中大规模 |
| `CircuitORAM` | O(log n) | 中大规模 |
| `OptimalORAM` | 自动选择 | 通用（推荐） |

### 测试结果

```bash
# 运行测试
./semi2k-party.x -p 0 -N 2 ann_oram &
./semi2k-party.x -p 1 -N 2 ann_oram

# 输出示例:
ANN ORAM 配置:
  K=10, b_max=20, M=200
  d=8, nprobe=2, b_read=5, M_cand=10
  top_k=3

--- 查询 0 ---
  阶段1: 安全选类
  选类完成 (私密)            # ← ClusterId 完全不 reveal
  阶段2: ORAM 读取候选
  ORAM 读取完成               # ← 通过 ORAM 隐藏访问模式
  阶段3: 候选集 KNN
  结果 (Top-3):
    #1: fileId=1036           # ← 只有 P0 可见
    #2: fileId=1007
    #3: fileId=1017

Time = 23.8 seconds
Data sent = 3309.8 MB (total)
```

### 安全性保证

1. **ClusterId 完全隐藏**：选类阶段使用安全排序，clusterId 不向任何一方 reveal
2. **访问模式隐藏**：通过 ORAM 进行候选读取，访问模式对双方均不可见
3. **结果保密**：只有 P0 能看到 Top-k fileId，P1 不知道查询结果

---

### 参数选择指南

| 参数 | 含义 | 推荐值 | 权衡 |
|------|------|--------|------|
| K | 聚类数 | 10-100 | 更多聚类 → 更高召回率，更高开销 |
| b_max | 每类最大块数 | ceil(N/K)*1.2 | 必须 ≥ 最大类的实际块数 |
| b_read | 每类读取块数 | 10-50 | 更多 → 更高召回率，更高开销 |
| nprobe | 探查聚类数 | 1-5 | 更多 → 更高召回率，更高开销 |
| BLOCK_WORDS | 每块 word 数 | 4-16 | 取决于每块存储的记录数 |

---

## Kona + ORAM 混合方案（C++ 实现）

如果希望基于 Kona 的 C++ 实现来获得更好的性能控制，可以使用 `ann-party-oram-kona.cpp`。

### 方案特点

| 组件 | 实现方式 | 来源 |
|------|----------|------|
| 安全距离计算 | Masked-Plain + Additive Share + 在线乘法 | Kona |
| 安全选类 | top_1（DCF 比较 + SS_vec 交换），**不 reveal clusterId** | Kona + 改造 |
| 候选读取 | Trivial ORAM（线性扫描 + 私密相等比较） | ORAM |
| 类内 KNN | top_1（k 次迭代） | Kona |

### 复杂度对比（性能优于 KNN 的条件）

b_read << N/K 且 b_read * M < N

| 方案 | 距离计算 | 选类 | 候选读取 | 总体 |
|------|----------|------|----------|------|
| 全库 KNN | O(N·d) | - | - | **O(N·d)** |
| A 方案（reveal clusterId） | O(K·d) | O(log K) | O(1) | O(K·d + N/K·d) |
| B 方案（Trivial ORAM） | O(K·d) | O(log K) | O(b_read·M) | O(K·d + b_read·M) |
| B 方案（Path ORAM） | O(K·d) | O(log K) | O(b_read·log M) | **O(K·d + b_read·log M)** |

### 编译与运行

\`\`\`bash
# 编译
make -j8 ann-party-oram-kona.x

# P0（先启动）:
./ann-party-oram-kona.x 0 -pn 11126 -h localhost \\
    -d ./Player-Data/ANN-Data -n test -k 5 --b-read 10

# P1:
./ann-party-oram-kona.x 1 -pn 11126 -h localhost \\
    -d ./Player-Data/ANN-Data -n test -k 5 --b-read 10
\`\`\`

### 分布式执行

\`\`\`bash
# P0 服务器 (10.176.37.50) - 先启动
./ann-party-oram-kona.x 0 -pn 11126 -h 10.176.37.50 \\
    -d /path/to/data -n dataset_name -k 5 --b-read 10

# P1 服务器 (10.176.34.171) - 后启动
./ann-party-oram-kona.x 1 -pn 11126 -h 10.176.37.50 \\
    -d /path/to/data -n dataset_name -k 5 --b-read 10
\`\`\`

### 与 MPC 编译器版本对比

| 特性 | Kona + ORAM (C++) | MPC 编译器版本 |
|------|-------------------|----------------|
| 代码行数 | ~800 行 | ~160 行 |
| 性能控制 | 精细 | 自动优化 |
| ORAM 类型 | Trivial（可替换 Path ORAM） | OptimalORAM（自动选择） |
| 适用场景 | 生产环境 | 快速原型 |
