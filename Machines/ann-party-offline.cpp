/**
 * @file ann-party-offline.cpp
 * @brief 两方隐私保护 ANN 检索 - 离线阶段
 * 
 * 功能：
 * 1. P1 明文 KMeans 聚类 + 建索引
 * 2. DatasetShare：将 P1 的 embedding 数据转为秘密共享形式
 * 
 * 参与方：
 * - P0（检察院）：持有查询 embedding
 * - P1（法院）：持有目标 embedding 库（每条对应一个 fileId）
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <cmath>
#include <random>
#include <algorithm>
#include <limits>
#include <chrono>
#include <cstring>
#include <sys/stat.h>

#include "../Math/Z2k.hpp"
#include "../Math/bigint.h"
#include "Tools/random.h"

using namespace std;

const int K = 64;  // 环大小

// ============== 数据结构定义 ==============

/**
 * @brief Embedding 记录结构
 */
struct EmbeddingRecord {
    int recordIndex;                  // 记录索引
    int fileId;                       // 文件ID
    vector<long long> embedding;      // 定点化后的 embedding 向量
    int clusterId;                    // 所属聚类ID（KMeans后填充）
    
    EmbeddingRecord() : recordIndex(-1), fileId(-1), clusterId(-1) {}
    EmbeddingRecord(int idx, int fid, const vector<long long>& emb)
        : recordIndex(idx), fileId(fid), embedding(emb), clusterId(-1) {}
};

/**
 * @brief 聚类中心结构
 */
struct Centroid {
    int clusterId;
    vector<long long> center;  // 定点化后的中心向量
    
    Centroid() : clusterId(-1) {}
    Centroid(int cid, const vector<long long>& c) : clusterId(cid), center(c) {}
};

/**
 * @brief KMeans 聚类结果
 */
struct KMeansResult {
    int numClusters;                              // 聚类数量
    int embDim;                                   // embedding 维度
    vector<Centroid> centroids;                   // 聚类中心列表
    map<int, vector<int>> clusterIndex;           // clusterId -> list(recordIndex)
    vector<EmbeddingRecord> records;              // 所有记录
    
    KMeansResult() : numClusters(0), embDim(0) {}
};

/**
 * @brief 秘密共享后的记录（P0/P1 各自持有的份额）
 */
struct SharedRecord {
    int recordIndex;
    int fileId;
    int clusterId;
    vector<Z2<K>> maskedVectorU;     // U = v + delta_0 + delta_1 (公开值)
    vector<Z2<K>> share;             // P0: delta_0, P1: delta_1
    
    SharedRecord() : recordIndex(-1), fileId(-1), clusterId(-1) {}
};

/**
 * @brief 秘密共享后的聚类中心（用于安全选类）
 */
struct SharedCentroid {
    int clusterId;
    vector<Z2<K>> maskedVectorU;     // U = v + delta_0 + delta_1 (公开值)
    vector<Z2<K>> share;             // P0: delta_0, P1: delta_1
    
    SharedCentroid() : clusterId(-1) {}
};

/**
 * @brief ANN 离线阶段配置
 */
struct ANNOfflineConfig {
    string dataDir;                   // 数据目录
    string datasetName;               // 数据集名称
    int numClusters;                  // 聚类数量
    int maxIterations;                // KMeans 最大迭代次数
    double convergenceThreshold;      // 收敛阈值
    
    ANNOfflineConfig()
        : dataDir("./Player-Data/ANN-Data/")
        , datasetName("test")
        , numClusters(10)
        , maxIterations(100)
        , convergenceThreshold(1e-6) {}
};

// ============== 工具函数 ==============

/**
 * @brief 创建目录（如果不存在）
 */
void ensureDirectoryExists(const string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        mkdir(path.c_str(), 0755);
    }
}

/**
 * @brief 计算两个向量的欧几里得距离平方（定点数）
 */
long long computeSquaredDistance(const vector<long long>& a, const vector<long long>& b) {
    assert(a.size() == b.size());
    long long dist = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        long long diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

/**
 * @brief 浮点数定点化（乘以缩放因子后取整）
 */
long long toFixedPoint(double val, int scaleFactor = 1000) {
    return static_cast<long long>(val * scaleFactor + 0.5);
}

/**
 * @brief 定点数还原为浮点数
 */
double fromFixedPoint(long long val, int scaleFactor = 1000) {
    return static_cast<double>(val) / scaleFactor;
}

// ============== KMeans 聚类实现 ==============

class KMeansClusterer {
public:
    KMeansClusterer(int numClusters, int maxIter = 100, double threshold = 1e-6)
        : m_numClusters(numClusters)
        , m_maxIterations(maxIter)
        , m_convergenceThreshold(threshold) {}
    
    /**
     * @brief 执行 KMeans 聚类
     * @param records 输入的 embedding 记录
     * @return 聚类结果
     */
    KMeansResult cluster(vector<EmbeddingRecord>& records) {
        KMeansResult result;
        if (records.empty()) return result;
        
        int embDim = records[0].embedding.size();
        result.embDim = embDim;
        result.numClusters = m_numClusters;
        
        // 1. 初始化聚类中心（KMeans++ 方式）
        vector<vector<long long>> centers = initializeCentroids(records);
        
        cout << "[KMeans] 开始聚类，数据量=" << records.size() 
             << ", 维度=" << embDim 
             << ", 聚类数=" << m_numClusters << endl;
        
        // 2. 迭代优化
        bool converged = false;
        for (int iter = 0; iter < m_maxIterations && !converged; ++iter) {
            // 2.1 分配阶段：将每个点分配到最近的中心
            assignToNearestCenter(records, centers);
            
            // 2.2 更新阶段：重新计算聚类中心
            vector<vector<long long>> newCenters = updateCenters(records, centers, embDim);
            
            // 2.3 检查收敛
            double maxShift = computeMaxCenterShift(centers, newCenters);
            centers = newCenters;
            
            if (iter % 10 == 0) {
                cout << "[KMeans] 迭代 " << iter << ", 最大中心偏移=" << maxShift << endl;
            }
            
            if (maxShift < m_convergenceThreshold) {
                converged = true;
                cout << "[KMeans] 在第 " << iter << " 次迭代后收敛" << endl;
            }
        }
        
        // 3. 构建结果
        for (int c = 0; c < m_numClusters; ++c) {
            result.centroids.emplace_back(c, centers[c]);
        }
        
        for (auto& rec : records) {
            result.clusterIndex[rec.clusterId].push_back(rec.recordIndex);
        }
        result.records = records;
        
        // 输出聚类统计
        cout << "[KMeans] 聚类完成，各聚类大小：" << endl;
        for (const auto& kv : result.clusterIndex) {
            cout << "  Cluster " << kv.first << ": " << kv.second.size() << " 条记录" << endl;
        }
        
        return result;
    }
    
private:
    int m_numClusters;
    int m_maxIterations;
    double m_convergenceThreshold;
    
    /**
     * @brief KMeans++ 初始化聚类中心
     */
    vector<vector<long long>> initializeCentroids(const vector<EmbeddingRecord>& records) {
        vector<vector<long long>> centers;
        mt19937 rng(42);  // 固定随机种子便于复现
        
        // 随机选择第一个中心
        uniform_int_distribution<int> firstDist(0, records.size() - 1);
        centers.push_back(records[firstDist(rng)].embedding);
        
        // KMeans++ 选择剩余中心
        vector<double> distances(records.size(), numeric_limits<double>::max());
        
        for (int c = 1; c < m_numClusters; ++c) {
            // 更新距离（到最近中心的距离）
            double totalDist = 0;
            for (size_t i = 0; i < records.size(); ++i) {
                long long d = computeSquaredDistance(records[i].embedding, centers.back());
                distances[i] = min(distances[i], static_cast<double>(d));
                totalDist += distances[i];
            }
            
            // 按距离概率选择下一个中心
            uniform_real_distribution<double> probDist(0, totalDist);
            double threshold = probDist(rng);
            double cumSum = 0;
            int selected = 0;
            for (size_t i = 0; i < records.size(); ++i) {
                cumSum += distances[i];
                if (cumSum >= threshold) {
                    selected = i;
                    break;
                }
            }
            centers.push_back(records[selected].embedding);
        }
        
        return centers;
    }
    
    /**
     * @brief 将每个点分配到最近的聚类中心
     */
    void assignToNearestCenter(vector<EmbeddingRecord>& records, 
                               const vector<vector<long long>>& centers) {
        for (auto& rec : records) {
            long long minDist = numeric_limits<long long>::max();
            int bestCluster = 0;
            
            for (int c = 0; c < m_numClusters; ++c) {
                long long dist = computeSquaredDistance(rec.embedding, centers[c]);
                if (dist < minDist) {
                    minDist = dist;
                    bestCluster = c;
                }
            }
            rec.clusterId = bestCluster;
        }
    }
    
    /**
     * @brief 更新聚类中心
     */
    vector<vector<long long>> updateCenters(const vector<EmbeddingRecord>& records,
                                            const vector<vector<long long>>& oldCenters,
                                            int embDim) {
        vector<vector<long long>> newCenters(m_numClusters, vector<long long>(embDim, 0));
        vector<int> clusterCounts(m_numClusters, 0);
        
        // 累加每个聚类的所有点
        for (const auto& rec : records) {
            int c = rec.clusterId;
            clusterCounts[c]++;
            for (int d = 0; d < embDim; ++d) {
                newCenters[c][d] += rec.embedding[d];
            }
        }
        
        // 计算平均值
        for (int c = 0; c < m_numClusters; ++c) {
            if (clusterCounts[c] > 0) {
                for (int d = 0; d < embDim; ++d) {
                    newCenters[c][d] /= clusterCounts[c];
                }
            } else {
                // 空聚类保持原中心
                newCenters[c] = oldCenters[c];
            }
        }
        
        return newCenters;
    }
    
    /**
     * @brief 计算聚类中心的最大偏移量
     */
    double computeMaxCenterShift(const vector<vector<long long>>& oldCenters,
                                  const vector<vector<long long>>& newCenters) {
        double maxShift = 0;
        for (size_t c = 0; c < oldCenters.size(); ++c) {
            double shift = sqrt(static_cast<double>(
                computeSquaredDistance(oldCenters[c], newCenters[c])));
            maxShift = max(maxShift, shift);
        }
        return maxShift;
    }
};

// ============== 数据秘密共享 ==============

class DatasetSharer {
public:
    DatasetSharer() {
        prng.ReSeed();
    }
    
    /**
     * @brief 生成秘密共享数据
     * @param kmeansResult KMeans 聚类结果
     * @param p0Records P0 侧的共享记录（输出）
     * @param p1Records P1 侧的共享记录（输出）
     */
    void generateShares(const KMeansResult& kmeansResult,
                        vector<SharedRecord>& p0Records,
                        vector<SharedRecord>& p1Records) {
        
        int embDim = kmeansResult.embDim;
        const auto& records = kmeansResult.records;
        
        cout << "[DatasetShare] 开始生成秘密共享，记录数=" << records.size() 
             << ", 维度=" << embDim << endl;
        
        p0Records.clear();
        p1Records.clear();
        p0Records.reserve(records.size());
        p1Records.reserve(records.size());
        
        for (const auto& rec : records) {
            SharedRecord p0Rec, p1Rec;
            
            // 基本信息（两方都持有）
            p0Rec.recordIndex = p1Rec.recordIndex = rec.recordIndex;
            p0Rec.fileId = p1Rec.fileId = rec.fileId;
            p0Rec.clusterId = p1Rec.clusterId = rec.clusterId;
            
            // 生成随机掩码份额
            p0Rec.maskedVectorU.resize(embDim);
            p1Rec.maskedVectorU.resize(embDim);
            p0Rec.share.resize(embDim);
            p1Rec.share.resize(embDim);
            
            for (int d = 0; d < embDim; ++d) {
                Z2<K> v(rec.embedding[d]);  // 原始值
                Z2<K> delta0, delta1;
                
                // 生成随机份额
                delta0.randomize(prng);
                delta1.randomize(prng);
                
                // U = v + delta0 + delta1 (公开的掩码明文)
                Z2<K> U = v + delta0 + delta1;
                
                p0Rec.maskedVectorU[d] = U;
                p1Rec.maskedVectorU[d] = U;
                p0Rec.share[d] = delta0;
                p1Rec.share[d] = delta1;
            }
            
            p0Records.push_back(p0Rec);
            p1Records.push_back(p1Rec);
        }
        
        cout << "[DatasetShare] 秘密共享生成完成" << endl;
    }
    
    /**
     * @brief 生成用于欧几里得距离计算的三元组
     * 三元组结构: (delta_x - delta_y)^2 的份额
     */
    void generateEuclideanTriples(int numRecords, int embDim,
                                  vector<vector<Z2<K>>>& triples0,
                                  vector<vector<Z2<K>>>& triples1,
                                  vector<Z2<K>>& queryDelta0,
                                  vector<Z2<K>>& queryDelta1) {
        
        cout << "[Triples] 生成欧几里得距离三元组，记录数=" << numRecords 
             << ", 维度=" << embDim << endl;
        
        // 查询向量的随机掩码
        queryDelta0.resize(embDim);
        queryDelta1.resize(embDim);
        for (int d = 0; d < embDim; ++d) {
            queryDelta0[d].randomize(prng);
            queryDelta1[d].randomize(prng);
        }
        
        // 为每条记录生成三元组
        triples0.resize(numRecords);
        triples1.resize(numRecords);
        
        for (int i = 0; i < numRecords; ++i) {
            triples0[i].resize(embDim);
            triples1[i].resize(embDim);
            
            for (int d = 0; d < embDim; ++d) {
                // 生成随机值 r
                Z2<K> r;
                r.randomize(prng);
                
                // 这里简化处理，实际应该基于数据的 delta 来计算
                // 完整实现需要在 DatasetShare 时同时生成
                triples0[i][d] = r;
                
                Z2<K> tmp;
                tmp.randomize(prng);
                triples1[i][d] = tmp;
            }
        }
        
        cout << "[Triples] 三元组生成完成" << endl;
    }
    
    /**
     * @brief 生成聚类中心的秘密共享
     * @param centroids 聚类中心列表
     * @param p0Centroids 输出：P0 的份额
     * @param p1Centroids 输出：P1 的份额
     */
    void generateCentroidShares(const vector<Centroid>& centroids,
                                 vector<SharedCentroid>& p0Centroids,
                                 vector<SharedCentroid>& p1Centroids) {
        if (centroids.empty()) return;
        
        int embDim = centroids[0].center.size();
        int numCentroids = centroids.size();
        
        cout << "[CentroidShare] 生成聚类中心秘密共享，聚类数=" << numCentroids 
             << ", 维度=" << embDim << endl;
        
        p0Centroids.clear();
        p1Centroids.clear();
        p0Centroids.resize(numCentroids);
        p1Centroids.resize(numCentroids);
        
        for (int c = 0; c < numCentroids; ++c) {
            SharedCentroid& p0Cen = p0Centroids[c];
            SharedCentroid& p1Cen = p1Centroids[c];
            
            p0Cen.clusterId = p1Cen.clusterId = centroids[c].clusterId;
            p0Cen.maskedVectorU.resize(embDim);
            p1Cen.maskedVectorU.resize(embDim);
            p0Cen.share.resize(embDim);
            p1Cen.share.resize(embDim);
            
            for (int d = 0; d < embDim; ++d) {
                Z2<K> v(centroids[c].center[d]);  // 原始值
                Z2<K> delta0, delta1;
                
                delta0.randomize(prng);
                delta1.randomize(prng);
                
                Z2<K> U = v + delta0 + delta1;
                
                p0Cen.maskedVectorU[d] = U;
                p1Cen.maskedVectorU[d] = U;
                p0Cen.share[d] = delta0;
                p1Cen.share[d] = delta1;
            }
        }
        
        cout << "[CentroidShare] 聚类中心秘密共享生成完成" << endl;
    }
    
    /**
     * @brief 生成用于选类阶段的安全距离计算三元组
     * @param numCentroids 聚类中心数量
     * @param embDim embedding 维度
     * @param centroidTriples0/1 输出：各方的三元组
     * @param clusterQueryDelta0/1 输出：查询向量的 delta 份额
     */
    void generateClusterTriples(int numCentroids, int embDim,
                                 vector<vector<Z2<K>>>& centroidTriples0,
                                 vector<vector<Z2<K>>>& centroidTriples1,
                                 vector<Z2<K>>& clusterQueryDelta0,
                                 vector<Z2<K>>& clusterQueryDelta1) {
        
        cout << "[ClusterTriples] 生成选类阶段三元组，聚类数=" << numCentroids 
             << ", 维度=" << embDim << endl;
        
        // 查询向量的随机掩码（用于选类）
        clusterQueryDelta0.resize(embDim);
        clusterQueryDelta1.resize(embDim);
        for (int d = 0; d < embDim; ++d) {
            clusterQueryDelta0[d].randomize(prng);
            clusterQueryDelta1[d].randomize(prng);
        }
        
        // 为每个聚类中心生成三元组
        centroidTriples0.resize(numCentroids);
        centroidTriples1.resize(numCentroids);
        
        for (int c = 0; c < numCentroids; ++c) {
            centroidTriples0[c].resize(embDim);
            centroidTriples1[c].resize(embDim);
            
            for (int d = 0; d < embDim; ++d) {
                Z2<K> r;
                r.randomize(prng);
                centroidTriples0[c][d] = r;
                
                Z2<K> tmp;
                tmp.randomize(prng);
                centroidTriples1[c][d] = tmp;
            }
        }
        
        cout << "[ClusterTriples] 选类阶段三元组生成完成" << endl;
    }
    
private:
    PRNG prng;
};

// ============== 文件 I/O ==============

class ANNDataIO {
public:
    /**
     * @brief 加载 P1 的 embedding 数据
     * 文件格式：每行 "fileId emb[0] emb[1] ... emb[dim-1]"
     */
    static vector<EmbeddingRecord> loadEmbeddings(const string& filename, int scaleFactor = 1000) {
        vector<EmbeddingRecord> records;
        ifstream fin(filename);
        
        if (!fin.is_open()) {
            cerr << "[Error] 无法打开文件: " << filename << endl;
            return records;
        }
        
        string line;
        int recordIndex = 0;
        
        while (getline(fin, line)) {
            if (line.empty()) continue;
            
            istringstream iss(line);
            int fileId;
            iss >> fileId;
            
            vector<long long> emb;
            double val;
            while (iss >> val) {
                emb.push_back(toFixedPoint(val, scaleFactor));
            }
            
            if (!emb.empty()) {
                records.emplace_back(recordIndex, fileId, emb);
                recordIndex++;
            }
        }
        
        fin.close();
        cout << "[IO] 加载 " << records.size() << " 条 embedding 记录" << endl;
        return records;
    }
    
    /**
     * @brief 保存 KMeans 结果
     */
    static void saveKMeansResult(const string& dir, const string& name, 
                                  const KMeansResult& result) {
        ensureDirectoryExists(dir);
        string basePath = dir + "/" + name;
        
        // 保存元数据
        ofstream metaFile(basePath + "-meta");
        if (!metaFile.is_open()) {
            cerr << "[Error] 无法写入文件 " << basePath << "-meta" << endl;
            cerr << "[Error] 请检查目录权限: " << dir << endl;
            throw runtime_error("无法写入文件，请检查目录权限");
        }
        metaFile << result.embDim << " " << result.records.size() 
                 << " " << result.numClusters << endl;
        for (int c = 0; c < result.numClusters; ++c) {
            metaFile << (result.clusterIndex.count(c) > 0 ? result.clusterIndex.at(c).size() : 0);
            if (c < result.numClusters - 1) metaFile << " ";
        }
        metaFile << endl;
        metaFile.close();
        
        // 保存聚类中心
        ofstream centroidFile(basePath + "-centroids", ios::binary);
        for (const auto& cen : result.centroids) {
            for (auto val : cen.center) {
                centroidFile.write(reinterpret_cast<const char*>(&val), sizeof(val));
            }
        }
        centroidFile.close();
        
        // 保存聚类索引
        ofstream indexFile(basePath + "-cluster-index");
        for (const auto& kv : result.clusterIndex) {
            indexFile << kv.first;
            for (int idx : kv.second) {
                indexFile << " " << idx;
            }
            indexFile << endl;
        }
        indexFile.close();
        
        cout << "[IO] KMeans 结果已保存到 " << basePath << "-*" << endl;
    }
    
    /**
     * @brief 保存共享记录到文件
     */
    static void saveSharedRecords(const string& dir, const string& name,
                                   int partyId,
                                   const vector<SharedRecord>& records) {
        ensureDirectoryExists(dir);
        string filename = dir + "/" + name + "-P" + to_string(partyId) + "-shares";
        
        ofstream fout(filename, ios::binary);
        if (!fout.is_open()) {
            cerr << "[Error] 无法写入文件 " << filename << endl;
            throw runtime_error("无法写入文件，请检查目录权限");
        }
        
        // 写入记录数和维度
        int numRecords = records.size();
        int embDim = records.empty() ? 0 : records[0].share.size();
        fout.write(reinterpret_cast<const char*>(&numRecords), sizeof(numRecords));
        fout.write(reinterpret_cast<const char*>(&embDim), sizeof(embDim));
        
        // 写入每条记录
        for (const auto& rec : records) {
            fout.write(reinterpret_cast<const char*>(&rec.recordIndex), sizeof(rec.recordIndex));
            fout.write(reinterpret_cast<const char*>(&rec.fileId), sizeof(rec.fileId));
            fout.write(reinterpret_cast<const char*>(&rec.clusterId), sizeof(rec.clusterId));
            
            // 写入 maskedVectorU 和 share
            for (int d = 0; d < embDim; ++d) {
                fout.write(reinterpret_cast<const char*>(&rec.maskedVectorU[d]), sizeof(Z2<K>));
            }
            for (int d = 0; d < embDim; ++d) {
                fout.write(reinterpret_cast<const char*>(&rec.share[d]), sizeof(Z2<K>));
            }
        }
        
        fout.close();
        cout << "[IO] P" << partyId << " 共享数据已保存到 " << filename << endl;
    }
    
    /**
     * @brief 加载共享记录
     */
    static vector<SharedRecord> loadSharedRecords(const string& dir, const string& name,
                                                    int partyId) {
        vector<SharedRecord> records;
        string filename = dir + "/" + name + "-P" + to_string(partyId) + "-shares";
        
        ifstream fin(filename, ios::binary);
        if (!fin.is_open()) {
            cerr << "[Error] 无法打开文件: " << filename << endl;
            return records;
        }
        
        int numRecords, embDim;
        fin.read(reinterpret_cast<char*>(&numRecords), sizeof(numRecords));
        fin.read(reinterpret_cast<char*>(&embDim), sizeof(embDim));
        
        records.resize(numRecords);
        for (int i = 0; i < numRecords; ++i) {
            fin.read(reinterpret_cast<char*>(&records[i].recordIndex), sizeof(int));
            fin.read(reinterpret_cast<char*>(&records[i].fileId), sizeof(int));
            fin.read(reinterpret_cast<char*>(&records[i].clusterId), sizeof(int));
            
            records[i].maskedVectorU.resize(embDim);
            records[i].share.resize(embDim);
            
            for (int d = 0; d < embDim; ++d) {
                fin.read(reinterpret_cast<char*>(&records[i].maskedVectorU[d]), sizeof(Z2<K>));
            }
            for (int d = 0; d < embDim; ++d) {
                fin.read(reinterpret_cast<char*>(&records[i].share[d]), sizeof(Z2<K>));
            }
        }
        
        fin.close();
        cout << "[IO] 加载 P" << partyId << " 共享数据，记录数=" << numRecords << endl;
        return records;
    }
    
    /**
     * @brief 保存三元组数据
     */
    static void saveTriples(const string& dir, const string& name,
                            int partyId,
                            const vector<vector<Z2<K>>>& triples,
                            const vector<Z2<K>>& queryDelta) {
        ensureDirectoryExists(dir);
        string filename = dir + "/" + name + "-P" + to_string(partyId) + "-triples";
        
        ofstream fout(filename, ios::binary);
        
        int numRecords = triples.size();
        int embDim = numRecords > 0 ? triples[0].size() : 0;
        fout.write(reinterpret_cast<const char*>(&numRecords), sizeof(numRecords));
        fout.write(reinterpret_cast<const char*>(&embDim), sizeof(embDim));
        
        // 写入查询向量的 delta
        for (int d = 0; d < embDim; ++d) {
            fout.write(reinterpret_cast<const char*>(&queryDelta[d]), sizeof(Z2<K>));
        }
        
        // 写入每条记录的三元组
        for (int i = 0; i < numRecords; ++i) {
            for (int d = 0; d < embDim; ++d) {
                fout.write(reinterpret_cast<const char*>(&triples[i][d]), sizeof(Z2<K>));
            }
        }
        
        fout.close();
        cout << "[IO] P" << partyId << " 三元组已保存到 " << filename << endl;
    }
    
    /**
     * @brief 保存聚类中心秘密共享
     */
    static void saveCentroidShares(const string& dir, const string& name,
                                    int partyId,
                                    const vector<SharedCentroid>& centroids) {
        ensureDirectoryExists(dir);
        string filename = dir + "/" + name + "-P" + to_string(partyId) + "-centroid-shares";
        
        ofstream fout(filename, ios::binary);
        
        int numCentroids = centroids.size();
        int embDim = numCentroids > 0 ? centroids[0].share.size() : 0;
        fout.write(reinterpret_cast<const char*>(&numCentroids), sizeof(numCentroids));
        fout.write(reinterpret_cast<const char*>(&embDim), sizeof(embDim));
        
        for (const auto& cen : centroids) {
            fout.write(reinterpret_cast<const char*>(&cen.clusterId), sizeof(cen.clusterId));
            for (int d = 0; d < embDim; ++d) {
                fout.write(reinterpret_cast<const char*>(&cen.maskedVectorU[d]), sizeof(Z2<K>));
            }
            for (int d = 0; d < embDim; ++d) {
                fout.write(reinterpret_cast<const char*>(&cen.share[d]), sizeof(Z2<K>));
            }
        }
        
        fout.close();
        cout << "[IO] P" << partyId << " 聚类中心共享数据已保存到 " << filename << endl;
    }
    
    /**
     * @brief 加载聚类中心秘密共享
     */
    static vector<SharedCentroid> loadCentroidShares(const string& dir, const string& name,
                                                       int partyId) {
        vector<SharedCentroid> centroids;
        string filename = dir + "/" + name + "-P" + to_string(partyId) + "-centroid-shares";
        
        ifstream fin(filename, ios::binary);
        if (!fin.is_open()) {
            cerr << "[Error] 无法打开文件: " << filename << endl;
            return centroids;
        }
        
        int numCentroids, embDim;
        fin.read(reinterpret_cast<char*>(&numCentroids), sizeof(numCentroids));
        fin.read(reinterpret_cast<char*>(&embDim), sizeof(embDim));
        
        centroids.resize(numCentroids);
        for (int c = 0; c < numCentroids; ++c) {
            fin.read(reinterpret_cast<char*>(&centroids[c].clusterId), sizeof(int));
            centroids[c].maskedVectorU.resize(embDim);
            centroids[c].share.resize(embDim);
            
            for (int d = 0; d < embDim; ++d) {
                fin.read(reinterpret_cast<char*>(&centroids[c].maskedVectorU[d]), sizeof(Z2<K>));
            }
            for (int d = 0; d < embDim; ++d) {
                fin.read(reinterpret_cast<char*>(&centroids[c].share[d]), sizeof(Z2<K>));
            }
        }
        
        fin.close();
        cout << "[IO] 加载 P" << partyId << " 聚类中心共享数据，聚类数=" << numCentroids << endl;
        return centroids;
    }
    
    /**
     * @brief 保存选类阶段三元组
     */
    static void saveClusterTriples(const string& dir, const string& name,
                                    int partyId,
                                    const vector<vector<Z2<K>>>& triples,
                                    const vector<Z2<K>>& queryDelta) {
        ensureDirectoryExists(dir);
        string filename = dir + "/" + name + "-P" + to_string(partyId) + "-cluster-triples";
        
        ofstream fout(filename, ios::binary);
        
        int numCentroids = triples.size();
        int embDim = numCentroids > 0 ? triples[0].size() : 0;
        fout.write(reinterpret_cast<const char*>(&numCentroids), sizeof(numCentroids));
        fout.write(reinterpret_cast<const char*>(&embDim), sizeof(embDim));
        
        // 写入查询向量的 delta
        for (int d = 0; d < embDim; ++d) {
            fout.write(reinterpret_cast<const char*>(&queryDelta[d]), sizeof(Z2<K>));
        }
        
        // 写入每个聚类中心的三元组
        for (int c = 0; c < numCentroids; ++c) {
            for (int d = 0; d < embDim; ++d) {
                fout.write(reinterpret_cast<const char*>(&triples[c][d]), sizeof(Z2<K>));
            }
        }
        
        fout.close();
        cout << "[IO] P" << partyId << " 选类三元组已保存到 " << filename << endl;
    }
};

// ============== 生成 DCF 密钥（用于安全比较） ==============

void gen_fake_dcf(int beta, int n, const string& outputDir) {
    ensureDirectoryExists(outputDir);
    
    int lambda_bytes = 16;
    PRNG prng;
    prng.InitSeed();
    
    fstream k0, k1, r0, r1;
    k0.open(outputDir + "/k0", ios::out);
    k1.open(outputDir + "/k1", ios::out);
    r0.open(outputDir + "/r0", ios::out);
    r1.open(outputDir + "/r1", ios::out);
    
    octet seed[2][lambda_bytes];
    bigint s[2][2], v[2][2], t[2][2], tmp_t[2], convert[2], tcw[2], a, scw, vcw, va, tmp, tmp1, tmp_out;
    
    prng.get(tmp, n);
    bytesFromBigint(&seed[0][0], tmp, lambda_bytes);
    k0 << tmp << " ";
    prng.get(tmp1, n);
    bytesFromBigint(&seed[1][0], tmp1, lambda_bytes);
    k1 << tmp1 << " ";
    prng.get(a, n);
    prng.get(tmp, n);
    r1 << a - tmp << " ";
    r0 << tmp << " ";
    r0.close();
    r1.close();
    
    tmp_t[0] = 0;
    tmp_t[1] = 1;
    int keep, lose;
    va = 0;
    
    for (int i = 0; i < n - 1; i++) {
        keep = bigint(a >> (n - i - 1)).get_ui() & 1;
        lose = 1 ^ keep;
        for (int j = 0; j < 2; j++) {
            prng.SetSeed(seed[j]);
            for (int k = 0; k < 2; k++) {
                prng.get(t[k][j], 1);
                prng.get(v[k][j], n);
                prng.get(s[k][j], n);
            }
        }
        scw = s[lose][0] ^ s[lose][1];
        bytesFromBigint(&seed[0][0], v[lose][0], lambda_bytes);
        prng.SetSeed(seed[0]);
        prng.get(convert[0], n);
        bytesFromBigint(&seed[0][0], v[lose][1], lambda_bytes);
        prng.SetSeed(seed[0]);
        prng.get(convert[1], n);
        if (tmp_t[1])
            vcw = convert[0] + va - convert[1];
        else
            vcw = convert[1] - convert[0] - va;
        if (keep)
            vcw = vcw + tmp_t[1] * (-beta) + (1 - tmp_t[1]) * beta;
        bytesFromBigint(&seed[0][0], v[keep][0], lambda_bytes);
        prng.SetSeed(seed[0]);
        prng.get(convert[0], n);
        bytesFromBigint(&seed[0][0], v[keep][1], lambda_bytes);
        prng.SetSeed(seed[0]);
        prng.get(convert[1], n);
        va = va - convert[1] + convert[0] + tmp_t[1] * (-vcw) + (1 - tmp_t[1]) * vcw;
        tcw[0] = t[0][0] ^ t[0][1] ^ keep ^ 1;
        tcw[1] = t[1][0] ^ t[1][1] ^ keep;
        k0 << scw << " " << vcw << " " << tcw[0] << " " << tcw[1] << " ";
        k1 << scw << " " << vcw << " " << tcw[0] << " " << tcw[1] << " ";
        bytesFromBigint(&seed[0][0], s[keep][0] ^ (tmp_t[0] * scw), lambda_bytes);
        bytesFromBigint(&seed[1][0], s[keep][1] ^ (tmp_t[1] * scw), lambda_bytes);
        bigintFromBytes(tmp_out, &seed[0][0], lambda_bytes);
        bigintFromBytes(tmp_out, &seed[1][0], lambda_bytes);
        tmp_t[0] = t[keep][0] ^ (tmp_t[0] * tcw[keep]);
        tmp_t[1] = t[keep][1] ^ (tmp_t[1] * tcw[keep]);
    }
    
    prng.SetSeed(seed[0]);
    prng.get(convert[0], n);
    prng.SetSeed(seed[1]);
    prng.get(convert[1], n);
    k0 << tmp_t[1] * (-1 * (convert[1] - convert[0] - va)) + (1 - tmp_t[1]) * (convert[1] - convert[0] - va) << " ";
    k1 << tmp_t[1] * (-1 * (convert[1] - convert[0] - va)) + (1 - tmp_t[1]) * (convert[1] - convert[0] - va) << " ";
    k0.close();
    k1.close();
    
    cout << "[DCF] DCF 密钥已生成到 " << outputDir << endl;
}

// ============== 生成测试数据 ==============

void generateTestData(const string& dir, const string& name,
                      int numRecords, int embDim,
                      int numQueries) {
    ensureDirectoryExists(dir);
    
    mt19937 rng(12345);
    uniform_real_distribution<double> dist(-1.0, 1.0);
    
    // 生成 P1 的 embedding 库
    string p1File = dir + "/" + name + "-P1-embeddings";
    ofstream fout(p1File);
    for (int i = 0; i < numRecords; ++i) {
        fout << (1000 + i);  // fileId
        for (int d = 0; d < embDim; ++d) {
            fout << " " << dist(rng);
        }
        fout << endl;
    }
    fout.close();
    cout << "[TestData] P1 embedding 库已生成: " << p1File << endl;
    
    // 生成 P0 的查询 embedding
    string p0File = dir + "/" + name + "-P0-queries";
    fout.open(p0File);
    for (int i = 0; i < numQueries; ++i) {
        fout << i;  // queryId
        for (int d = 0; d < embDim; ++d) {
            fout << " " << dist(rng);
        }
        fout << endl;
    }
    fout.close();
    cout << "[TestData] P0 查询已生成: " << p0File << endl;
}

// ============== 主函数 ==============

int main(int argc, char* argv[]) {
    ANNOfflineConfig config;
    config.dataDir = "./Player-Data/ANN-Data/";
    config.datasetName = "test";
    config.numClusters = 10;
    
    // 解析命令行参数
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--data-dir" && i + 1 < argc) {
            config.dataDir = argv[++i];
        } else if (arg == "--dataset" && i + 1 < argc) {
            config.datasetName = argv[++i];
        } else if (arg == "--clusters" && i + 1 < argc) {
            config.numClusters = atoi(argv[++i]);
        } else if (arg == "--gen-test") {
            // 生成测试数据
            int numRecords = 1000, embDim = 128, numQueries = 10;
            if (i + 1 < argc) numRecords = atoi(argv[++i]);
            if (i + 1 < argc) embDim = atoi(argv[++i]);
            if (i + 1 < argc) numQueries = atoi(argv[++i]);
            
            generateTestData(config.dataDir, config.datasetName, 
                           numRecords, embDim, numQueries);
            return 0;
        }
    }
    
    cout << "========================================" << endl;
    cout << "  ANN 离线阶段 - P1 KMeans + DatasetShare" << endl;
    cout << "========================================" << endl;
    cout << "数据目录: " << config.dataDir << endl;
    cout << "数据集名: " << config.datasetName << endl;
    cout << "聚类数量: " << config.numClusters << endl;
    cout << "----------------------------------------" << endl;
    
    // 1. 加载 P1 的 embedding 数据
    string embFile = config.dataDir + "/" + config.datasetName + "-P1-embeddings";
    vector<EmbeddingRecord> records = ANNDataIO::loadEmbeddings(embFile);
    
    if (records.empty()) {
        cerr << "[Error] 未加载到数据，请先生成测试数据:" << endl;
        cerr << "  ./ann-party-offline.x --gen-test 1000 128 10" << endl;
        return 1;
    }
    
    // 2. KMeans 聚类
    cout << "\n[阶段1] 执行 KMeans 聚类..." << endl;
    KMeansClusterer clusterer(config.numClusters);
    KMeansResult kmeansResult = clusterer.cluster(records);
    
    // 保存 KMeans 结果
    ANNDataIO::saveKMeansResult(config.dataDir, config.datasetName, kmeansResult);
    
    // 3. 数据秘密共享
    cout << "\n[阶段2] 生成秘密共享数据..." << endl;
    DatasetSharer sharer;
    vector<SharedRecord> p0Records, p1Records;
    sharer.generateShares(kmeansResult, p0Records, p1Records);
    
    // 保存共享数据
    ANNDataIO::saveSharedRecords(config.dataDir, config.datasetName, 0, p0Records);
    ANNDataIO::saveSharedRecords(config.dataDir, config.datasetName, 1, p1Records);
    
    // 4. 生成欧几里得距离三元组
    cout << "\n[阶段3] 生成欧几里得距离三元组..." << endl;
    vector<vector<Z2<K>>> triples0, triples1;
    vector<Z2<K>> queryDelta0, queryDelta1;
    sharer.generateEuclideanTriples(records.size(), kmeansResult.embDim,
                                    triples0, triples1,
                                    queryDelta0, queryDelta1);
    
    ANNDataIO::saveTriples(config.dataDir, config.datasetName, 0, triples0, queryDelta0);
    ANNDataIO::saveTriples(config.dataDir, config.datasetName, 1, triples1, queryDelta1);
    
    // 5. 生成聚类中心秘密共享（用于安全选类）
    cout << "\n[阶段4] 生成聚类中心秘密共享..." << endl;
    vector<SharedCentroid> p0Centroids, p1Centroids;
    sharer.generateCentroidShares(kmeansResult.centroids, p0Centroids, p1Centroids);
    
    ANNDataIO::saveCentroidShares(config.dataDir, config.datasetName, 0, p0Centroids);
    ANNDataIO::saveCentroidShares(config.dataDir, config.datasetName, 1, p1Centroids);
    
    // 6. 生成选类阶段三元组
    cout << "\n[阶段5] 生成选类阶段三元组..." << endl;
    vector<vector<Z2<K>>> centroidTriples0, centroidTriples1;
    vector<Z2<K>> clusterQueryDelta0, clusterQueryDelta1;
    sharer.generateClusterTriples(config.numClusters, kmeansResult.embDim,
                                   centroidTriples0, centroidTriples1,
                                   clusterQueryDelta0, clusterQueryDelta1);
    
    ANNDataIO::saveClusterTriples(config.dataDir, config.datasetName, 0, 
                                   centroidTriples0, clusterQueryDelta0);
    ANNDataIO::saveClusterTriples(config.dataDir, config.datasetName, 1, 
                                   centroidTriples1, clusterQueryDelta1);
    
    // 7. 生成 DCF 密钥（用于安全比较）
    cout << "\n[阶段6] 生成 DCF 密钥..." << endl;
    string fssDir = config.dataDir + "/2-fss";
    gen_fake_dcf(1, K, fssDir);
    
    cout << "\n========================================" << endl;
    cout << "  ANN 离线阶段完成！" << endl;
    cout << "========================================" << endl;
    cout << "生成的文件列表:" << endl;
    cout << "  - " << config.dataDir << "/" << config.datasetName << "-meta" << endl;
    cout << "  - " << config.dataDir << "/" << config.datasetName << "-centroids" << endl;
    cout << "  - " << config.dataDir << "/" << config.datasetName << "-cluster-index" << endl;
    cout << "  - " << config.dataDir << "/" << config.datasetName << "-P0-shares" << endl;
    cout << "  - " << config.dataDir << "/" << config.datasetName << "-P1-shares" << endl;
    cout << "  - " << config.dataDir << "/" << config.datasetName << "-P0-triples" << endl;
    cout << "  - " << config.dataDir << "/" << config.datasetName << "-P1-triples" << endl;
    cout << "  - " << config.dataDir << "/" << config.datasetName << "-P0-centroid-shares" << endl;
    cout << "  - " << config.dataDir << "/" << config.datasetName << "-P1-centroid-shares" << endl;
    cout << "  - " << config.dataDir << "/" << config.datasetName << "-P0-cluster-triples" << endl;
    cout << "  - " << config.dataDir << "/" << config.datasetName << "-P1-cluster-triples" << endl;
    cout << "  - " << fssDir << "/k0, k1, r0, r1" << endl;
    
    return 0;
}

