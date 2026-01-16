#!/usr/bin/env python3
"""
ANN ORAM 离线数据准备脚本

功能：
1. 执行 KMeans 聚类
2. 生成 Block 数据库（padding 到固定大小）
3. 生成 MPC 输入文件

用法：
    python3 ann_oram_prepare.py --embeddings <file> --output-dir <dir> \
        --clusters 10 --b-max 20 --queries <file>
"""

import argparse
import numpy as np
from sklearn.cluster import KMeans
import os
import struct

def parse_args():
    parser = argparse.ArgumentParser(description='ANN ORAM 离线数据准备')
    parser.add_argument('--embeddings', type=str, required=False, default=None,
                        help='P1 embedding 文件 (格式: fileId emb[0] emb[1] ...)')
    parser.add_argument('--queries', type=str, default=None,
                        help='P0 查询文件 (格式: queryId emb[0] emb[1] ...)')
    parser.add_argument('--output-dir', type=str, default='./Player-Data/Input',
                        help='输出目录')
    parser.add_argument('--clusters', '-k', type=int, default=10,
                        help='聚类数量 K')
    parser.add_argument('--b-max', type=int, default=20,
                        help='每类最大块数')
    parser.add_argument('--scale', type=int, default=1000,
                        help='定点化缩放因子')
    parser.add_argument('--gen-test', action='store_true',
                        help='生成测试数据')
    parser.add_argument('--test-records', type=int, default=100,
                        help='测试记录数')
    parser.add_argument('--test-dim', type=int, default=32,
                        help='测试维度')
    parser.add_argument('--test-queries', type=int, default=3,
                        help='测试查询数')
    return parser.parse_args()

def load_embeddings(filepath, scale=1000):
    """加载 embedding 文件"""
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            file_id = int(parts[0])
            emb = [int(float(x) * scale + 0.5) for x in parts[1:]]
            records.append((file_id, emb))
    return records

def load_queries(filepath, scale=1000):
    """加载查询文件"""
    queries = []
    with open(filepath, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            query_id = int(parts[0])
            emb = [int(float(x) * scale + 0.5) for x in parts[1:]]
            queries.append((query_id, emb))
    return queries

def kmeans_cluster(records, k):
    """执行 KMeans 聚类"""
    print(f"[KMeans] 聚类 {len(records)} 条记录到 {k} 个聚类...")
    
    embeddings = np.array([r[1] for r in records])
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    centers = kmeans.cluster_centers_
    
    # 分配记录到聚类
    cluster_records = {i: [] for i in range(k)}
    for idx, label in enumerate(labels):
        cluster_records[label].append(records[idx])
    
    print("[KMeans] 聚类完成:")
    for c in range(k):
        print(f"  Cluster {c}: {len(cluster_records[c])} 条记录")
    
    return centers, cluster_records

def create_block_db(cluster_records, k, b_max, dim, scale=1000):
    """创建 Block 数据库，padding 到固定大小"""
    print(f"[BlockDB] 创建 Block 数据库 (K={k}, b_max={b_max})...")
    
    M = k * b_max
    # Block: embedding[dim] + fileId + valid
    block_db = []
    
    for c in range(k):
        records = cluster_records.get(c, [])
        for bi in range(b_max):
            if bi < len(records):
                # 真实数据
                file_id, emb = records[bi]
                block = emb + [file_id, 1]  # valid = 1
            else:
                # Dummy block
                block = [0] * dim + [0, 0]  # valid = 0
            block_db.append(block)
    
    print(f"[BlockDB] 创建 {len(block_db)} 个 block")
    return block_db

def write_mpc_input(output_dir, centers, block_db, queries, k, dim, scale):
    """写入 MPC 输入文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    # P1 输入: 聚类中心 + Block 数据库
    p1_file = os.path.join(output_dir, 'Input-P1-0')
    print(f"[IO] 写入 P1 输入: {p1_file}")
    
    with open(p1_file, 'w') as f:
        # 聚类中心
        for c in range(k):
            center = centers[c]
            for val in center:
                f.write(f"{int(val * scale)}\n")
            f.write(f"{c}\n")  # clusterId
        
        # Block 数据库 - 格式: embedding[dim] + fileId + valid
        for block in block_db:
            # block = embedding[dim] + [fileId, valid]
            for val in block:
                f.write(f"{val}\n")
    
    # P0 输入: 查询向量
    if queries:
        p0_file = os.path.join(output_dir, 'Input-P0-0')
        print(f"[IO] 写入 P0 输入: {p0_file}")
        
        with open(p0_file, 'w') as f:
            for query_id, emb in queries:
                for val in emb:
                    f.write(f"{val}\n")
    
    # 写入配置文件
    config_file = os.path.join(output_dir, 'ann_oram_config.txt')
    with open(config_file, 'w') as f:
        f.write(f"K={k}\n")
        f.write(f"b_max={len(block_db)//k}\n")
        f.write(f"d={dim}\n")
        f.write(f"M={len(block_db)}\n")
        f.write(f"N_queries={len(queries) if queries else 0}\n")
    
    print(f"[IO] 配置已写入: {config_file}")

def generate_test_data(output_dir, n_records, dim, n_queries, scale=1000):
    """生成测试数据"""
    print(f"[TestData] 生成测试数据: {n_records} 条记录, {dim} 维, {n_queries} 查询")
    
    os.makedirs(output_dir, exist_ok=True)
    
    np.random.seed(12345)
    
    # 生成 embedding
    emb_file = os.path.join(output_dir, 'test-P1-embeddings')
    with open(emb_file, 'w') as f:
        for i in range(n_records):
            emb = np.random.uniform(-1, 1, dim)
            f.write(f"{1000 + i} " + " ".join(f"{v:.6f}" for v in emb) + "\n")
    print(f"[TestData] P1 embedding: {emb_file}")
    
    # 生成查询
    query_file = os.path.join(output_dir, 'test-P0-queries')
    with open(query_file, 'w') as f:
        for i in range(n_queries):
            emb = np.random.uniform(-1, 1, dim)
            f.write(f"{i} " + " ".join(f"{v:.6f}" for v in emb) + "\n")
    print(f"[TestData] P0 查询: {query_file}")
    
    return emb_file, query_file

def main():
    args = parse_args()
    
    print("========================================")
    print("  ANN ORAM 离线数据准备")
    print("========================================")
    
    if not args.gen_test and not args.embeddings:
        print("错误: 必须提供 --embeddings 或 --gen-test")
        return
    
    if args.gen_test:
        # 生成测试数据
        emb_file, query_file = generate_test_data(
            args.output_dir,
            args.test_records,
            args.test_dim,
            args.test_queries,
            args.scale
        )
        args.embeddings = emb_file
        args.queries = query_file
    
    # 加载数据
    records = load_embeddings(args.embeddings, args.scale)
    dim = len(records[0][1])
    print(f"[IO] 加载 {len(records)} 条记录, 维度={dim}")
    
    queries = None
    if args.queries:
        queries = load_queries(args.queries, args.scale)
        print(f"[IO] 加载 {len(queries)} 条查询")
    
    # 自动调整 b_max
    avg_per_cluster = len(records) // args.clusters
    if args.b_max < avg_per_cluster * 2:
        args.b_max = avg_per_cluster * 2
        print(f"[自动调整] b_max = {args.b_max}")
    
    # KMeans 聚类
    centers, cluster_records = kmeans_cluster(records, args.clusters)
    
    # 创建 Block 数据库
    block_db = create_block_db(
        cluster_records, 
        args.clusters, 
        args.b_max, 
        dim,
        args.scale
    )
    
    # 写入 MPC 输入
    write_mpc_input(
        args.output_dir,
        centers,
        block_db,
        queries,
        args.clusters,
        dim,
        args.scale
    )
    
    print("\n========================================")
    print("  离线数据准备完成！")
    print("========================================")
    print(f"  K (聚类数) = {args.clusters}")
    print(f"  b_max (每类最大块) = {args.b_max}")
    print(f"  M (总块数) = {len(block_db)}")
    print(f"  d (维度) = {dim}")
    print(f"  输出目录: {args.output_dir}")

if __name__ == '__main__':
    main()
