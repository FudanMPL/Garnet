#!/usr/bin/env python3
"""
ANN vs KNN 消融实验

对比两种方案：
1. ANN（Approximate Nearest Neighbor）：先聚类，再在候选类中计算
2. KNN（K-Nearest Neighbor）：直接在所有数据中计算

实验目标：
- 对比 2000、5000、8000 条数据下的执行效率
- 对比计算的记录数（ANN 只计算候选类，KNN 计算全部）

数据格式参考: fileId emb[0] emb[1] ... emb[dim-1]
"""

import numpy as np
import time
import os
import sys
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 配置参数
DATA_SIZES = [2000, 5000, 8000]
CLUSTER_NUMS = [10]  # 默认聚类数
TOP_K = 5  # 返回Top-K个最近邻
N_RUNS = 3  # 每个配置运行次数
RANDOM_SEED = 42


def load_embeddings(filepath, max_records=None):
    """
    加载 embedding 文件
    格式: fileId emb[0] emb[1] ... emb[dim-1]
    """
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            file_id = int(parts[0])
            emb = [float(x) for x in parts[1:]]
            records.append((file_id, emb))
            if max_records and len(records) >= max_records:
                break
    return records


def generate_test_data(n_samples, dim, n_clusters=10, seed=42):
    """
    生成测试数据 - 使用混合高斯分布生成具有聚类结构的数据
    """
    np.random.seed(seed)
    
    # 生成多个高斯分布中心
    centers = np.random.uniform(-1, 1, (n_clusters, dim))
    
    samples_per_center = n_samples // n_clusters
    remainder = n_samples % n_clusters
    
    data = []
    file_ids = []
    
    for i, center in enumerate(centers):
        n = samples_per_center + (1 if i < remainder else 0)
        # 每个中心周围添加噪声
        cluster_data = center + np.random.randn(n, dim) * 0.1
        data.append(cluster_data)
        file_ids.extend(range(1000 + len(file_ids), 1000 + len(file_ids) + n))
    
    data = np.vstack(data)
    
    # 打乱顺序
    indices = np.random.permutation(len(data))
    data = data[indices]
    file_ids = [file_ids[i] for i in indices]
    
    return file_ids, data


def create_test_data_from_existing(existing_data, target_size, seed=42):
    """
    基于现有数据创建测试数据
    """
    np.random.seed(seed)
    
    if len(existing_data) >= target_size:
        indices = np.random.choice(len(existing_data), target_size, replace=False)
        return existing_data[indices]
    
    # 数据不够，通过添加噪声扩充
    n_existing = len(existing_data)
    n_needed = target_size - n_existing
    
    repeated_indices = np.random.choice(n_existing, n_needed, replace=True)
    noise = np.random.randn(n_needed, existing_data.shape[1]) * 0.01
    augmented = existing_data[repeated_indices] + noise
    
    return np.vstack([existing_data, augmented])


class ANNRetriever:
    """
    ANN 检索器：先聚类，再在候选类中计算
    """
    
    def __init__(self, n_clusters=10, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.cluster_data = {}  # 每个聚类的数据
        self.cluster_indices = {}  # 每个聚类的原始索引
        
    def fit(self, X, file_ids):
        """
        离线阶段：执行 KMeans 聚类
        """
        self.X = X
        self.file_ids = file_ids
        
        # KMeans 聚类
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            init='k-means++',
            random_state=self.random_state,
            n_init=10
        )
        self.labels = self.kmeans.fit_predict(X)
        
        # 按聚类分组数据
        for i in range(self.n_clusters):
            mask = self.labels == i
            self.cluster_data[i] = X[mask]
            self.cluster_indices[i] = np.where(mask)[0]
        
        return self
    
    def query(self, query_vec, top_k=5):
        """
        在线阶段：先选类，再在类内计算
        返回: (top_k_indices, top_k_file_ids, n_computed, cluster_id, cluster_size)
        """
        query_vec = query_vec.reshape(1, -1)
        
        # 阶段1: 选择最近的聚类中心
        center_distances = euclidean_distances(query_vec, self.kmeans.cluster_centers_)[0]
        selected_cluster = np.argmin(center_distances)
        
        # 阶段2: 在选中的聚类内计算距离
        cluster_X = self.cluster_data[selected_cluster]
        cluster_indices = self.cluster_indices[selected_cluster]
        
        if len(cluster_X) == 0:
            return [], [], 0, selected_cluster, 0
        
        distances = euclidean_distances(query_vec, cluster_X)[0]
        
        # 获取Top-K
        k = min(top_k, len(distances))
        top_k_local = np.argsort(distances)[:k]
        
        # 映射回原始索引
        top_k_global = cluster_indices[top_k_local]
        top_k_file_ids = [self.file_ids[i] for i in top_k_global]
        
        n_computed = len(cluster_X)
        cluster_size = len(cluster_X)
        
        return top_k_global.tolist(), top_k_file_ids, n_computed, selected_cluster, cluster_size


class KNNRetriever:
    """
    KNN 检索器：直接在所有数据中计算
    """
    
    def __init__(self):
        pass
    
    def fit(self, X, file_ids):
        """
        离线阶段：只保存数据（无聚类）
        """
        self.X = X
        self.file_ids = file_ids
        return self
    
    def query(self, query_vec, top_k=5):
        """
        在线阶段：直接计算所有距离
        返回: (top_k_indices, top_k_file_ids, n_computed)
        """
        query_vec = query_vec.reshape(1, -1)
        
        # 直接计算所有距离
        distances = euclidean_distances(query_vec, self.X)[0]
        
        # 获取Top-K
        k = min(top_k, len(distances))
        top_k_indices = np.argsort(distances)[:k]
        
        top_k_file_ids = [self.file_ids[i] for i in top_k_indices]
        
        n_computed = len(self.X)
        
        return top_k_indices.tolist(), top_k_file_ids, n_computed


def run_ablation_experiment(data, file_ids, data_sizes, n_clusters=10, top_k=5, n_runs=3):
    """
    运行消融实验：对比 ANN 和 KNN
    """
    results = {}
    
    print("=" * 80)
    print("ANN vs KNN 消融实验")
    print("=" * 80)
    print(f"数据量测试范围: {data_sizes}")
    print(f"聚类数 (ANN): {n_clusters}")
    print(f"Top-K: {top_k}")
    print(f"每个配置运行次数: {n_runs}")
    print(f"数据维度: {data.shape[1]}")
    print("=" * 80)
    print()
    
    for n_samples in data_sizes:
        print(f"\n{'='*70}")
        print(f"数据量: {n_samples} 条")
        print(f"{'='*70}")
        
        # 准备数据
        test_data = create_test_data_from_existing(data, n_samples)
        test_file_ids = list(range(1000, 1000 + n_samples))
        
        # 标准化
        scaler = StandardScaler()
        test_data_scaled = scaler.fit_transform(test_data)
        
        # 生成查询向量
        np.random.seed(RANDOM_SEED + n_samples)
        query_vec = test_data_scaled[np.random.randint(0, len(test_data_scaled))]
        
        # ============= ANN 方案 =============
        print(f"\n  [ANN] 先聚类 + 候选类计算:")
        print("-" * 60)
        
        ann_offline_times = []
        ann_online_times = []
        ann_computed_records = []
        ann_cluster_sizes = []
        
        for run in range(n_runs):
            # 离线阶段：KMeans 聚类
            ann_retriever = ANNRetriever(n_clusters=n_clusters, random_state=RANDOM_SEED + run)
            
            offline_start = time.time()
            ann_retriever.fit(test_data_scaled, test_file_ids)
            offline_time = time.time() - offline_start
            ann_offline_times.append(offline_time)
            
            # 在线阶段：选类 + 类内计算
            online_start = time.time()
            top_k_idx, top_k_fids, n_computed, cluster_id, cluster_size = ann_retriever.query(query_vec, top_k)
            online_time = time.time() - online_start
            ann_online_times.append(online_time)
            ann_computed_records.append(n_computed)
            ann_cluster_sizes.append(cluster_size)
        
        ann_results = {
            'offline_time': np.mean(ann_offline_times),
            'online_time': np.mean(ann_online_times),
            'total_time': np.mean(ann_offline_times) + np.mean(ann_online_times),
            'computed_records': np.mean(ann_computed_records),
            'cluster_size': np.mean(ann_cluster_sizes),
            'top_k_result': top_k_fids
        }
        
        print(f"    离线阶段 (KMeans): {ann_results['offline_time']*1000:.2f} ms")
        print(f"    在线阶段 (选类+计算): {ann_results['online_time']*1000:.4f} ms")
        print(f"    候选类大小: {ann_results['cluster_size']:.0f} 条")
        print(f"    实际计算记录数: {ann_results['computed_records']:.0f} 条")
        print(f"    Top-{top_k} 结果: {top_k_fids}")
        
        # ============= KNN 方案 =============
        print(f"\n  [KNN] 全量计算:")
        print("-" * 60)
        
        knn_offline_times = []
        knn_online_times = []
        knn_computed_records = []
        
        for run in range(n_runs):
            knn_retriever = KNNRetriever()
            
            # 离线阶段：只保存数据
            offline_start = time.time()
            knn_retriever.fit(test_data_scaled, test_file_ids)
            offline_time = time.time() - offline_start
            knn_offline_times.append(offline_time)
            
            # 在线阶段：全量计算
            online_start = time.time()
            top_k_idx, top_k_fids, n_computed = knn_retriever.query(query_vec, top_k)
            online_time = time.time() - online_start
            knn_online_times.append(online_time)
            knn_computed_records.append(n_computed)
        
        knn_results = {
            'offline_time': np.mean(knn_offline_times),
            'online_time': np.mean(knn_online_times),
            'total_time': np.mean(knn_offline_times) + np.mean(knn_online_times),
            'computed_records': np.mean(knn_computed_records),
            'top_k_result': top_k_fids
        }
        
        print(f"    离线阶段 (无聚类): {knn_results['offline_time']*1000:.4f} ms")
        print(f"    在线阶段 (全量计算): {knn_results['online_time']*1000:.4f} ms")
        print(f"    实际计算记录数: {knn_results['computed_records']:.0f} 条")
        print(f"    Top-{top_k} 结果: {top_k_fids}")
        
        # ============= 对比结果 =============
        print(f"\n  [对比分析]")
        print("-" * 60)
        
        records_saved = knn_results['computed_records'] - ann_results['computed_records']
        records_saved_pct = records_saved / knn_results['computed_records'] * 100
        
        online_speedup = knn_results['online_time'] / ann_results['online_time'] if ann_results['online_time'] > 0 else 0
        
        print(f"    ANN 少计算记录数: {records_saved:.0f} 条 ({records_saved_pct:.1f}%)")
        print(f"    在线阶段加速比: {online_speedup:.2f}x")
        
        # 检查结果准确性
        ann_set = set(ann_results['top_k_result'])
        knn_set = set(knn_results['top_k_result'])
        overlap = len(ann_set & knn_set)
        recall = overlap / len(knn_set) * 100 if len(knn_set) > 0 else 0
        print(f"    ANN Top-{top_k} 召回率: {recall:.1f}% (与精确KNN对比)")
        
        results[n_samples] = {
            'ann': ann_results,
            'knn': knn_results,
            'records_saved': records_saved,
            'records_saved_pct': records_saved_pct,
            'online_speedup': online_speedup,
            'recall': recall
        }
    
    return results


def print_summary_table(results, data_sizes):
    """
    打印汇总表格
    """
    print("\n" + "=" * 90)
    print("消融实验汇总表")
    print("=" * 90)
    
    # 表头
    print("\n【执行效率对比】")
    print("-" * 90)
    header = f"{'数据量':^10}|{'ANN离线(ms)':^14}|{'ANN在线(ms)':^14}|{'KNN在线(ms)':^14}|{'加速比':^10}|{'少计算记录':^14}"
    print(header)
    print("-" * 90)
    
    for n_samples in data_sizes:
        r = results[n_samples]
        row = f"{n_samples:^10}|{r['ann']['offline_time']*1000:^14.2f}|{r['ann']['online_time']*1000:^14.4f}|{r['knn']['online_time']*1000:^14.4f}|{r['online_speedup']:^10.2f}x|{r['records_saved']:^14.0f}"
        print(row)
    
    print("\n【计算记录数对比】")
    print("-" * 90)
    header = f"{'数据量':^10}|{'总记录数':^12}|{'ANN计算数':^12}|{'KNN计算数':^12}|{'节省比例':^12}|{'召回率':^10}"
    print(header)
    print("-" * 90)
    
    for n_samples in data_sizes:
        r = results[n_samples]
        row = f"{n_samples:^10}|{n_samples:^12}|{r['ann']['computed_records']:^12.0f}|{r['knn']['computed_records']:^12.0f}|{r['records_saved_pct']:^12.1f}%|{r['recall']:^10.1f}%"
        print(row)
    
    print("\n" + "=" * 90)


def save_results_to_csv(results, data_sizes, output_path):
    """
    保存结果到 CSV 文件
    """
    with open(output_path, 'w') as f:
        # 写入表头
        f.write("数据量,ANN离线时间(ms),ANN在线时间(ms),KNN离线时间(ms),KNN在线时间(ms),")
        f.write("ANN计算记录数,KNN计算记录数,节省记录数,节省比例(%),在线加速比,召回率(%)\n")
        
        for n_samples in data_sizes:
            r = results[n_samples]
            f.write(f"{n_samples},{r['ann']['offline_time']*1000:.4f},{r['ann']['online_time']*1000:.6f},")
            f.write(f"{r['knn']['offline_time']*1000:.4f},{r['knn']['online_time']*1000:.6f},")
            f.write(f"{r['ann']['computed_records']:.0f},{r['knn']['computed_records']:.0f},")
            f.write(f"{r['records_saved']:.0f},{r['records_saved_pct']:.2f},{r['online_speedup']:.4f},{r['recall']:.2f}\n")
    
    print(f"\n结果已保存到: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='ANN vs KNN 消融实验')
    parser.add_argument('--embeddings', '-e', type=str, default=None,
                        help='P1 embedding 文件路径')
    parser.add_argument('--dim', '-d', type=int, default=1000,
                        help='embedding 维度（生成测试数据时使用）')
    parser.add_argument('--clusters', '-c', type=int, default=10,
                        help='聚类数')
    parser.add_argument('--top-k', '-k', type=int, default=5,
                        help='返回的最近邻数量')
    parser.add_argument('--runs', '-r', type=int, default=3,
                        help='每个配置运行次数')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='结果输出 CSV 文件路径')
    
    args = parser.parse_args()
    
    # 加载或生成数据
    if args.embeddings and os.path.exists(args.embeddings):
        print(f"从文件加载数据: {args.embeddings}")
        records = load_embeddings(args.embeddings)
        file_ids = [r[0] for r in records]
        data = np.array([r[1] for r in records])
        print(f"加载了 {len(records)} 条记录，维度 = {data.shape[1]}")
    else:
        print(f"生成测试数据: 维度 = {args.dim}")
        file_ids, data = generate_test_data(max(DATA_SIZES) + 1000, args.dim)
        print(f"生成了 {len(data)} 条测试记录")
    
    # 运行消融实验
    results = run_ablation_experiment(
        data, file_ids,
        DATA_SIZES,
        n_clusters=args.clusters,
        top_k=args.top_k,
        n_runs=args.runs
    )
    
    # 打印汇总表
    print_summary_table(results, DATA_SIZES)
    
    # 保存结果
    if args.output:
        save_results_to_csv(results, DATA_SIZES, args.output)
    else:
        default_output = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'ann_vs_knn_ablation_results.csv'
        )
        save_results_to_csv(results, DATA_SIZES, default_output)
    
    print("\n实验完成！")


if __name__ == '__main__':
    main()
