#!/usr/bin/env python3
"""
KMeans 聚类算法性能测试脚本

测试不同聚类数（6-12）在不同数据量（2000、5000、8000）下的执行时间与准确度

数据格式参考: fileId emb[0] emb[1] ... emb[dim-1]
"""

import numpy as np
import time
import os
import sys
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 配置参数
CLUSTER_NUMS = [6, 7, 8, 9, 10, 11, 12]
DATA_SIZES = [2000, 5000, 8000]
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


def generate_test_data(n_samples, dim, seed=42):
    """
    生成测试数据
    使用混合高斯分布生成更真实的聚类数据
    """
    np.random.seed(seed)
    
    # 生成多个高斯分布中心
    n_centers = 10
    centers = np.random.uniform(-1, 1, (n_centers, dim))
    
    samples_per_center = n_samples // n_centers
    remainder = n_samples % n_centers
    
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
    如果现有数据不够，通过添加噪声来扩充
    """
    np.random.seed(seed)
    
    if len(existing_data) >= target_size:
        # 随机采样
        indices = np.random.choice(len(existing_data), target_size, replace=False)
        return existing_data[indices]
    
    # 数据不够，需要扩充
    n_existing = len(existing_data)
    n_needed = target_size - n_existing
    
    # 通过添加噪声复制现有数据
    repeated_indices = np.random.choice(n_existing, n_needed, replace=True)
    noise = np.random.randn(n_needed, existing_data.shape[1]) * 0.01
    augmented = existing_data[repeated_indices] + noise
    
    return np.vstack([existing_data, augmented])


class KMeansClusterer:
    """KMeans 聚类器，支持 KMeans++ 初始化"""
    
    def __init__(self, n_clusters, max_iter=100, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.model = None
    
    def fit(self, X):
        """执行聚类"""
        self.model = KMeans(
            n_clusters=self.n_clusters,
            init='k-means++',
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            n_init=10
        )
        self.model.fit(X)
        return self
    
    def predict(self, X):
        """预测聚类标签"""
        return self.model.predict(X)
    
    @property
    def labels_(self):
        return self.model.labels_
    
    @property
    def cluster_centers_(self):
        return self.model.cluster_centers_
    
    @property
    def inertia_(self):
        return self.model.inertia_
    
    @property
    def n_iter_(self):
        return self.model.n_iter_


def evaluate_clustering(X, labels, centers):
    """
    评估聚类质量
    
    返回指标:
    - SSE (Sum of Squared Errors): 越小越好
    - Silhouette Score: [-1, 1]，越大越好
    - Calinski-Harabasz Index: 越大越好
    - Davies-Bouldin Index: 越小越好
    """
    metrics = {}
    
    # SSE (也称 inertia)
    sse = 0
    for i, label in enumerate(labels):
        sse += np.sum((X[i] - centers[label]) ** 2)
    metrics['SSE'] = sse
    
    # 轮廓系数 (需要至少 2 个样本)
    try:
        if len(set(labels)) > 1:
            metrics['Silhouette'] = silhouette_score(X, labels)
        else:
            metrics['Silhouette'] = -1
    except Exception:
        metrics['Silhouette'] = -1
    
    # Calinski-Harabasz Index
    try:
        if len(set(labels)) > 1:
            metrics['CH_Index'] = calinski_harabasz_score(X, labels)
        else:
            metrics['CH_Index'] = 0
    except Exception:
        metrics['CH_Index'] = 0
    
    # Davies-Bouldin Index
    try:
        if len(set(labels)) > 1:
            metrics['DB_Index'] = davies_bouldin_score(X, labels)
        else:
            metrics['DB_Index'] = float('inf')
    except Exception:
        metrics['DB_Index'] = float('inf')
    
    # 各聚类大小统计
    unique_labels, counts = np.unique(labels, return_counts=True)
    metrics['cluster_sizes'] = dict(zip(unique_labels.tolist(), counts.tolist()))
    metrics['min_cluster_size'] = int(min(counts))
    metrics['max_cluster_size'] = int(max(counts))
    metrics['avg_cluster_size'] = float(np.mean(counts))
    
    return metrics


def run_benchmark(data, cluster_nums, data_sizes, n_runs=3):
    """
    运行基准测试
    
    Args:
        data: 原始数据 (numpy array)
        cluster_nums: 聚类数列表
        data_sizes: 数据量列表
        n_runs: 每个配置运行次数（用于计算平均时间）
    
    Returns:
        结果字典
    """
    results = {}
    
    print("=" * 80)
    print("KMeans 聚类算法性能测试")
    print("=" * 80)
    print(f"聚类数测试范围: {cluster_nums}")
    print(f"数据量测试范围: {data_sizes}")
    print(f"每个配置运行次数: {n_runs}")
    print(f"数据维度: {data.shape[1]}")
    print("=" * 80)
    print()
    
    for n_samples in data_sizes:
        print(f"\n{'='*60}")
        print(f"数据量: {n_samples} 条")
        print(f"{'='*60}")
        
        # 准备数据
        test_data = create_test_data_from_existing(data, n_samples)
        
        # 标准化数据
        scaler = StandardScaler()
        test_data_scaled = scaler.fit_transform(test_data)
        
        for n_clusters in cluster_nums:
            print(f"\n  聚类数 K={n_clusters}:")
            print("-" * 50)
            
            # 多次运行取平均
            times = []
            all_metrics = []
            
            for run in range(n_runs):
                start_time = time.time()
                
                clusterer = KMeansClusterer(
                    n_clusters=n_clusters,
                    random_state=RANDOM_SEED + run
                )
                clusterer.fit(test_data_scaled)
                
                end_time = time.time()
                elapsed = end_time - start_time
                times.append(elapsed)
                
                # 评估聚类质量
                metrics = evaluate_clustering(
                    test_data_scaled,
                    clusterer.labels_,
                    clusterer.cluster_centers_
                )
                metrics['n_iter'] = clusterer.n_iter_
                all_metrics.append(metrics)
            
            # 计算平均值
            avg_time = np.mean(times)
            std_time = np.std(times)
            
            avg_metrics = {
                'SSE': np.mean([m['SSE'] for m in all_metrics]),
                'Silhouette': np.mean([m['Silhouette'] for m in all_metrics]),
                'CH_Index': np.mean([m['CH_Index'] for m in all_metrics]),
                'DB_Index': np.mean([m['DB_Index'] for m in all_metrics]),
                'n_iter': np.mean([m['n_iter'] for m in all_metrics]),
                'min_cluster_size': np.mean([m['min_cluster_size'] for m in all_metrics]),
                'max_cluster_size': np.mean([m['max_cluster_size'] for m in all_metrics]),
            }
            
            # 保存结果
            key = (n_samples, n_clusters)
            results[key] = {
                'time_avg': avg_time,
                'time_std': std_time,
                'metrics': avg_metrics
            }
            
            # 打印结果
            print(f"    执行时间: {avg_time:.4f}s (±{std_time:.4f}s)")
            print(f"    迭代次数: {avg_metrics['n_iter']:.1f}")
            print(f"    SSE: {avg_metrics['SSE']:.2f}")
            print(f"    轮廓系数 (Silhouette): {avg_metrics['Silhouette']:.4f}")
            print(f"    CH 指数: {avg_metrics['CH_Index']:.2f}")
            print(f"    DB 指数: {avg_metrics['DB_Index']:.4f}")
            print(f"    聚类大小范围: [{avg_metrics['min_cluster_size']:.0f}, {avg_metrics['max_cluster_size']:.0f}]")
    
    return results


def print_summary_table(results, cluster_nums, data_sizes):
    """打印汇总表格"""
    
    print("\n" + "=" * 80)
    print("性能测试汇总表")
    print("=" * 80)
    
    # 执行时间表
    print("\n【执行时间 (秒)】")
    print("-" * 60)
    header = "数据量\\聚类数".ljust(15) + "".join([f"K={k}".center(10) for k in cluster_nums])
    print(header)
    print("-" * 60)
    
    for n_samples in data_sizes:
        row = f"{n_samples}".ljust(15)
        for n_clusters in cluster_nums:
            key = (n_samples, n_clusters)
            if key in results:
                row += f"{results[key]['time_avg']:.4f}".center(10)
            else:
                row += "N/A".center(10)
        print(row)
    
    # SSE 表
    print("\n【SSE (误差平方和)】")
    print("-" * 60)
    print(header)
    print("-" * 60)
    
    for n_samples in data_sizes:
        row = f"{n_samples}".ljust(15)
        for n_clusters in cluster_nums:
            key = (n_samples, n_clusters)
            if key in results:
                sse = results[key]['metrics']['SSE']
                if sse > 10000:
                    row += f"{sse:.0f}".center(10)
                else:
                    row += f"{sse:.2f}".center(10)
            else:
                row += "N/A".center(10)
        print(row)
    
    # 轮廓系数表
    print("\n【轮廓系数 (Silhouette Score)】")
    print("-" * 60)
    print(header)
    print("-" * 60)
    
    for n_samples in data_sizes:
        row = f"{n_samples}".ljust(15)
        for n_clusters in cluster_nums:
            key = (n_samples, n_clusters)
            if key in results:
                row += f"{results[key]['metrics']['Silhouette']:.4f}".center(10)
            else:
                row += "N/A".center(10)
        print(row)
    
    # CH 指数表
    print("\n【Calinski-Harabasz 指数】")
    print("-" * 60)
    print(header)
    print("-" * 60)
    
    for n_samples in data_sizes:
        row = f"{n_samples}".ljust(15)
        for n_clusters in cluster_nums:
            key = (n_samples, n_clusters)
            if key in results:
                row += f"{results[key]['metrics']['CH_Index']:.1f}".center(10)
            else:
                row += "N/A".center(10)
        print(row)
    
    # DB 指数表
    print("\n【Davies-Bouldin 指数 (越小越好)】")
    print("-" * 60)
    print(header)
    print("-" * 60)
    
    for n_samples in data_sizes:
        row = f"{n_samples}".ljust(15)
        for n_clusters in cluster_nums:
            key = (n_samples, n_clusters)
            if key in results:
                row += f"{results[key]['metrics']['DB_Index']:.4f}".center(10)
            else:
                row += "N/A".center(10)
        print(row)
    
    print("\n" + "=" * 80)


def save_results_to_csv(results, cluster_nums, data_sizes, output_path):
    """保存结果到 CSV 文件"""
    with open(output_path, 'w') as f:
        # 写入表头
        f.write("数据量,聚类数,执行时间(秒),执行时间标准差,SSE,轮廓系数,CH指数,DB指数,迭代次数,最小聚类大小,最大聚类大小\n")
        
        for n_samples in data_sizes:
            for n_clusters in cluster_nums:
                key = (n_samples, n_clusters)
                if key in results:
                    r = results[key]
                    m = r['metrics']
                    f.write(f"{n_samples},{n_clusters},{r['time_avg']:.6f},{r['time_std']:.6f},"
                           f"{m['SSE']:.4f},{m['Silhouette']:.6f},{m['CH_Index']:.4f},"
                           f"{m['DB_Index']:.6f},{m['n_iter']:.1f},"
                           f"{m['min_cluster_size']:.0f},{m['max_cluster_size']:.0f}\n")
    
    print(f"\n结果已保存到: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='KMeans 聚类算法性能测试')
    parser.add_argument('--embeddings', '-e', type=str, default=None,
                        help='P1 embedding 文件路径')
    parser.add_argument('--dim', '-d', type=int, default=768,
                        help='embedding 维度（生成测试数据时使用）')
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
        # 生成足够多的数据
        file_ids, data = generate_test_data(max(DATA_SIZES) + 1000, args.dim)
        print(f"生成了 {len(data)} 条测试记录")
    
    # 运行基准测试
    results = run_benchmark(
        data,
        CLUSTER_NUMS,
        DATA_SIZES,
        n_runs=args.runs
    )
    
    # 打印汇总表
    print_summary_table(results, CLUSTER_NUMS, DATA_SIZES)
    
    # 保存结果
    if args.output:
        save_results_to_csv(results, CLUSTER_NUMS, DATA_SIZES, args.output)
    else:
        default_output = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'kmeans_benchmark_results.csv'
        )
        save_results_to_csv(results, CLUSTER_NUMS, DATA_SIZES, default_output)
    
    print("\n测试完成！")


if __name__ == '__main__':
    main()
