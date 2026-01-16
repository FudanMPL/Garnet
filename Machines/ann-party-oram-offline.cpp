/**
 * @file ann-party-oram-offline.cpp
 * @brief 两方隐私保护 ANN 检索 - ORAM 版本离线阶段
 * 
 * 实现：
 * 1. P1 明文 KMeans 聚类
 * 2. 数据布局固定化：每类 padding 到 b_max 个 block
 * 3. 生成秘密共享 DB[0..M-1]
 * 4. 生成聚类中心秘密共享
 * 5. 生成 DCF 密钥
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

const int K_BITS = 64;  // 环大小

// ============== ORAM 配置参数 ==============
struct ORAMConfig {
    int numClusters;      // K: 聚类数量
    int bMax;             // b_max: 每类最大 block 数
    int blockWords;       // BLOCK_WORDS: 每 block 的 word 数
    int recordsPerBlock;  // 每 block 存储的记录数
    
    ORAMConfig() 
        : numClusters(10)
        , bMax(20)
        , blockWords(4)  // embedding_dim / some_factor
        , recordsPerBlock(1)  // 简化：每 block 存 1 条记录
    {}
    
    int totalBlocks() const { return numClusters * bMax; }
};

// ============== 数据结构定义 ==============

struct EmbeddingRecord {
    int recordIndex;
    int fileId;
    vector<long long> embedding;
    int clusterId;
    
    EmbeddingRecord() : recordIndex(-1), fileId(-1), clusterId(-1) {}
    EmbeddingRecord(int idx, int fid, const vector<long long>& emb)
        : recordIndex(idx), fileId(fid), embedding(emb), clusterId(-1) {}
};

struct Centroid {
    int clusterId;
    vector<long long> center;
    
    Centroid() : clusterId(-1) {}
    Centroid(int cid, const vector<long long>& c) : clusterId(cid), center(c) {}
};

struct KMeansResult {
    int numClusters;
    int embDim;
    vector<Centroid> centroids;
    map<int, vector<int>> clusterIndex;
    vector<EmbeddingRecord> records;
    
    KMeansResult() : numClusters(0), embDim(0) {}
};

/**
 * @brief ORAM Block 结构（秘密共享）
 * 每个 block 包含固定数量的 word，以及元数据
 */
struct SharedBlock {
    vector<Z2<K_BITS>> maskedU;     // embedding 的掩码明文
    vector<Z2<K_BITS>> share;       // embedding 的份额
    Z2<K_BITS> fileIdShare;         // fileId 的秘密共享
    Z2<K_BITS> validShare;          // 有效标志（1=真实数据，0=dummy）
    
    SharedBlock() {}
    SharedBlock(int blockWords) 
        : maskedU(blockWords), share(blockWords) {}
};

/**
 * @brief 秘密共享的聚类中心
 */
struct SharedCentroid {
    int clusterId;
    vector<Z2<K_BITS>> maskedU;
    vector<Z2<K_BITS>> share;
    Z2<K_BITS> clusterIdShare;  // clusterId 的秘密共享（用于选类）
    
    SharedCentroid() : clusterId(-1) {}
};

struct ANNORAMOfflineConfig {
    string dataDir;
    string datasetName;
    ORAMConfig oram;
    int maxIterations;
    double convergenceThreshold;
    
    ANNORAMOfflineConfig()
        : dataDir("./Player-Data/ANN-Data/")
        , datasetName("test")
        , maxIterations(100)
        , convergenceThreshold(1e-6) {}
};

// ============== 工具函数 ==============

void ensureDirectoryExists(const string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        mkdir(path.c_str(), 0755);
    }
}

long long computeSquaredDistance(const vector<long long>& a, const vector<long long>& b) {
    assert(a.size() == b.size());
    long long dist = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        long long diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

long long toFixedPoint(double val, int scaleFactor = 1000) {
    return static_cast<long long>(val * scaleFactor + 0.5);
}

// ============== KMeans 聚类 ==============

class KMeansClusterer {
public:
    KMeansClusterer(int numClusters, int maxIter = 100, double threshold = 1e-6)
        : m_numClusters(numClusters), m_maxIterations(maxIter), m_convergenceThreshold(threshold) {}
    
    KMeansResult cluster(vector<EmbeddingRecord>& records) {
        KMeansResult result;
        if (records.empty()) return result;
        
        int embDim = records[0].embedding.size();
        result.embDim = embDim;
        result.numClusters = m_numClusters;
        
        vector<vector<long long>> centers = initializeCentroids(records);
        
        cout << "[KMeans] 开始聚类，数据量=" << records.size() 
             << ", 维度=" << embDim << ", 聚类数=" << m_numClusters << endl;
        
        bool converged = false;
        for (int iter = 0; iter < m_maxIterations && !converged; ++iter) {
            assignToNearestCenter(records, centers);
            vector<vector<long long>> newCenters = updateCenters(records, centers, embDim);
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
        
        for (int c = 0; c < m_numClusters; ++c) {
            result.centroids.emplace_back(c, centers[c]);
        }
        
        for (auto& rec : records) {
            result.clusterIndex[rec.clusterId].push_back(rec.recordIndex);
        }
        result.records = records;
        
        cout << "[KMeans] 聚类完成:" << endl;
        for (const auto& kv : result.clusterIndex) {
            cout << "  Cluster " << kv.first << ": " << kv.second.size() << " 条记录" << endl;
        }
        
        return result;
    }
    
private:
    int m_numClusters, m_maxIterations;
    double m_convergenceThreshold;
    
    vector<vector<long long>> initializeCentroids(const vector<EmbeddingRecord>& records) {
        vector<vector<long long>> centers;
        mt19937 rng(42);
        
        uniform_int_distribution<int> firstDist(0, records.size() - 1);
        centers.push_back(records[firstDist(rng)].embedding);
        
        vector<double> distances(records.size(), numeric_limits<double>::max());
        
        for (int c = 1; c < m_numClusters; ++c) {
            double totalDist = 0;
            for (size_t i = 0; i < records.size(); ++i) {
                long long d = computeSquaredDistance(records[i].embedding, centers.back());
                distances[i] = min(distances[i], static_cast<double>(d));
                totalDist += distances[i];
            }
            
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
    
    vector<vector<long long>> updateCenters(const vector<EmbeddingRecord>& records,
                                            const vector<vector<long long>>& oldCenters,
                                            int embDim) {
        vector<vector<long long>> newCenters(m_numClusters, vector<long long>(embDim, 0));
        vector<int> clusterCounts(m_numClusters, 0);
        
        for (const auto& rec : records) {
            int c = rec.clusterId;
            clusterCounts[c]++;
            for (int d = 0; d < embDim; ++d) {
                newCenters[c][d] += rec.embedding[d];
            }
        }
        
        for (int c = 0; c < m_numClusters; ++c) {
            if (clusterCounts[c] > 0) {
                for (int d = 0; d < embDim; ++d) {
                    newCenters[c][d] /= clusterCounts[c];
                }
            } else {
                newCenters[c] = oldCenters[c];
            }
        }
        
        return newCenters;
    }
    
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

// ============== ORAM 数据生成器 ==============

class ORAMDataGenerator {
public:
    ORAMDataGenerator() { prng.ReSeed(); }
    
    /**
     * @brief 生成 ORAM Block 数据库
     * 每个类 padding 到 b_max 个 block
     */
    void generateBlockDB(
        const KMeansResult& kmeansResult,
        const ORAMConfig& config,
        vector<SharedBlock>& p0Blocks,
        vector<SharedBlock>& p1Blocks
    ) {
        int M = config.totalBlocks();
        int embDim = kmeansResult.embDim;
        int blockWords = embDim;  // 简化：每 block 存整个 embedding
        
        cout << "[ORAM-DB] 生成 Block 数据库，M=" << M 
             << ", blockWords=" << blockWords << endl;
        
        p0Blocks.resize(M);
        p1Blocks.resize(M);
        
        // 初始化所有 block
        for (int i = 0; i < M; ++i) {
            p0Blocks[i].maskedU.resize(blockWords);
            p0Blocks[i].share.resize(blockWords);
            p1Blocks[i].maskedU.resize(blockWords);
            p1Blocks[i].share.resize(blockWords);
        }
        
        // 为每个聚类填充 blocks
        for (int c = 0; c < config.numClusters; ++c) {
            const auto& recordIndices = kmeansResult.clusterIndex.count(c) > 0 
                ? kmeansResult.clusterIndex.at(c) : vector<int>();
            
            int actualRecords = recordIndices.size();
            int blocksNeeded = min(actualRecords, config.bMax);
            
            for (int bi = 0; bi < config.bMax; ++bi) {
                int addr = c * config.bMax + bi;
                
                if (bi < blocksNeeded) {
                    // 真实数据
                    int recIdx = recordIndices[bi];
                    const auto& rec = kmeansResult.records[recIdx];
                    
                    createSharedBlock(rec, blockWords, p0Blocks[addr], p1Blocks[addr], true);
                } else {
                    // Dummy block (padding)
                    createDummyBlock(blockWords, p0Blocks[addr], p1Blocks[addr]);
                }
            }
        }
        
        cout << "[ORAM-DB] Block 数据库生成完成" << endl;
    }
    
    /**
     * @brief 生成聚类中心的秘密共享
     * 包括 clusterIdShare 用于私密选类
     */
    void generateCentroidShares(
        const vector<Centroid>& centroids,
        vector<SharedCentroid>& p0Centroids,
        vector<SharedCentroid>& p1Centroids
    ) {
        if (centroids.empty()) return;
        
        int embDim = centroids[0].center.size();
        int numCentroids = centroids.size();
        
        cout << "[CentroidShare] 生成聚类中心秘密共享，K=" << numCentroids << endl;
        
        p0Centroids.resize(numCentroids);
        p1Centroids.resize(numCentroids);
        
        for (int c = 0; c < numCentroids; ++c) {
            p0Centroids[c].clusterId = p1Centroids[c].clusterId = c;
            p0Centroids[c].maskedU.resize(embDim);
            p0Centroids[c].share.resize(embDim);
            p1Centroids[c].maskedU.resize(embDim);
            p1Centroids[c].share.resize(embDim);
            
            // embedding 秘密共享
            for (int d = 0; d < embDim; ++d) {
                Z2<K_BITS> v(centroids[c].center[d]);
                Z2<K_BITS> delta0, delta1;
                delta0.randomize(prng);
                delta1.randomize(prng);
                
                Z2<K_BITS> U = v + delta0 + delta1;
                
                p0Centroids[c].maskedU[d] = p1Centroids[c].maskedU[d] = U;
                p0Centroids[c].share[d] = delta0;
                p1Centroids[c].share[d] = delta1;
            }
            
            // clusterId 秘密共享（用于私密选类比较）
            Z2<K_BITS> idRandom;
            idRandom.randomize(prng);
            p0Centroids[c].clusterIdShare = Z2<K_BITS>(c) - idRandom;
            p1Centroids[c].clusterIdShare = idRandom;
        }
        
        cout << "[CentroidShare] 完成" << endl;
    }
    
private:
    PRNG prng;
    
    void createSharedBlock(
        const EmbeddingRecord& rec,
        int blockWords,
        SharedBlock& p0Block,
        SharedBlock& p1Block,
        bool isValid
    ) {
        // embedding 秘密共享
        for (int d = 0; d < blockWords && d < (int)rec.embedding.size(); ++d) {
            Z2<K_BITS> v(rec.embedding[d]);
            Z2<K_BITS> delta0, delta1;
            delta0.randomize(prng);
            delta1.randomize(prng);
            
            Z2<K_BITS> U = v + delta0 + delta1;
            
            p0Block.maskedU[d] = p1Block.maskedU[d] = U;
            p0Block.share[d] = delta0;
            p1Block.share[d] = delta1;
        }
        
        // fileId 秘密共享
        Z2<K_BITS> fileIdRandom;
        fileIdRandom.randomize(prng);
        p0Block.fileIdShare = Z2<K_BITS>(rec.fileId) - fileIdRandom;
        p1Block.fileIdShare = fileIdRandom;
        
        // valid 标志秘密共享
        Z2<K_BITS> validRandom;
        validRandom.randomize(prng);
        if (isValid) {
            p0Block.validShare = Z2<K_BITS>(1) - validRandom;
            p1Block.validShare = validRandom;
        } else {
            p0Block.validShare = Z2<K_BITS>(0) - validRandom;
            p1Block.validShare = validRandom;
        }
    }
    
    void createDummyBlock(int blockWords, SharedBlock& p0Block, SharedBlock& p1Block) {
        // 用随机值填充（看起来和真实数据一样）
        for (int d = 0; d < blockWords; ++d) {
            Z2<K_BITS> delta0, delta1;
            delta0.randomize(prng);
            delta1.randomize(prng);
            
            // Dummy 数据使用 0，但仍然加密
            Z2<K_BITS> U = delta0 + delta1;
            
            p0Block.maskedU[d] = p1Block.maskedU[d] = U;
            p0Block.share[d] = delta0;
            p1Block.share[d] = delta1;
        }
        
        // fileId = 0（dummy）
        Z2<K_BITS> fileIdRandom;
        fileIdRandom.randomize(prng);
        p0Block.fileIdShare = Z2<K_BITS>(0) - fileIdRandom;
        p1Block.fileIdShare = fileIdRandom;
        
        // valid = 0（dummy）
        Z2<K_BITS> validRandom;
        validRandom.randomize(prng);
        p0Block.validShare = Z2<K_BITS>(0) - validRandom;
        p1Block.validShare = validRandom;
    }
};

// ============== 文件 I/O ==============

class ORAMDataIO {
public:
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
    
    static void saveORAMConfig(const string& dir, const string& name,
                                const ORAMConfig& config, int embDim, int numRecords) {
        ensureDirectoryExists(dir);
        string filename = dir + "/" + name + "-oram-config";
        
        ofstream fout(filename);
        fout << config.numClusters << " " << config.bMax << " " 
             << config.blockWords << " " << config.recordsPerBlock << " "
             << embDim << " " << numRecords << endl;
        fout.close();
        
        cout << "[IO] ORAM 配置已保存到 " << filename << endl;
    }
    
    static void saveKMeansResult(const string& dir, const string& name, 
                                  const KMeansResult& result) {
        ensureDirectoryExists(dir);
        string basePath = dir + "/" + name;
        
        // 元数据
        ofstream metaFile(basePath + "-meta");
        metaFile << result.embDim << " " << result.records.size() 
                 << " " << result.numClusters << endl;
        for (int c = 0; c < result.numClusters; ++c) {
            metaFile << (result.clusterIndex.count(c) > 0 ? result.clusterIndex.at(c).size() : 0);
            if (c < result.numClusters - 1) metaFile << " ";
        }
        metaFile << endl;
        metaFile.close();
        
        // 聚类中心
        ofstream centroidFile(basePath + "-centroids", ios::binary);
        for (const auto& cen : result.centroids) {
            for (auto val : cen.center) {
                centroidFile.write(reinterpret_cast<const char*>(&val), sizeof(val));
            }
        }
        centroidFile.close();
        
        // 聚类索引
        ofstream indexFile(basePath + "-cluster-index");
        for (const auto& kv : result.clusterIndex) {
            indexFile << kv.first;
            for (int idx : kv.second) {
                indexFile << " " << idx;
            }
            indexFile << endl;
        }
        indexFile.close();
        
        cout << "[IO] KMeans 结果已保存" << endl;
    }
    
    static void saveBlockDB(const string& dir, const string& name, int partyId,
                            const vector<SharedBlock>& blocks, int blockWords) {
        ensureDirectoryExists(dir);
        string filename = dir + "/" + name + "-P" + to_string(partyId) + "-oram-blocks";
        
        ofstream fout(filename, ios::binary);
        
        int numBlocks = blocks.size();
        fout.write(reinterpret_cast<const char*>(&numBlocks), sizeof(numBlocks));
        fout.write(reinterpret_cast<const char*>(&blockWords), sizeof(blockWords));
        
        for (const auto& block : blocks) {
            // maskedU
            for (int d = 0; d < blockWords; ++d) {
                fout.write(reinterpret_cast<const char*>(&block.maskedU[d]), sizeof(Z2<K_BITS>));
            }
            // share
            for (int d = 0; d < blockWords; ++d) {
                fout.write(reinterpret_cast<const char*>(&block.share[d]), sizeof(Z2<K_BITS>));
            }
            // fileIdShare
            fout.write(reinterpret_cast<const char*>(&block.fileIdShare), sizeof(Z2<K_BITS>));
            // validShare
            fout.write(reinterpret_cast<const char*>(&block.validShare), sizeof(Z2<K_BITS>));
        }
        
        fout.close();
        cout << "[IO] P" << partyId << " Block DB 已保存，共 " << numBlocks << " 个 block" << endl;
    }
    
    static void saveCentroidShares(const string& dir, const string& name, int partyId,
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
                fout.write(reinterpret_cast<const char*>(&cen.maskedU[d]), sizeof(Z2<K_BITS>));
            }
            for (int d = 0; d < embDim; ++d) {
                fout.write(reinterpret_cast<const char*>(&cen.share[d]), sizeof(Z2<K_BITS>));
            }
            fout.write(reinterpret_cast<const char*>(&cen.clusterIdShare), sizeof(Z2<K_BITS>));
        }
        
        fout.close();
        cout << "[IO] P" << partyId << " 聚类中心共享已保存" << endl;
    }
};

// ============== DCF 密钥生成 ==============

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

// ============== 测试数据生成 ==============

void generateTestData(const string& dir, const string& name,
                      int numRecords, int embDim, int numQueries) {
    ensureDirectoryExists(dir);
    
    mt19937 rng(12345);
    uniform_real_distribution<double> dist(-1.0, 1.0);
    
    string p1File = dir + "/" + name + "-P1-embeddings";
    ofstream fout(p1File);
    for (int i = 0; i < numRecords; ++i) {
        fout << (1000 + i);
        for (int d = 0; d < embDim; ++d) {
            fout << " " << dist(rng);
        }
        fout << endl;
    }
    fout.close();
    cout << "[TestData] P1 embedding 库已生成: " << p1File << endl;
    
    string p0File = dir + "/" + name + "-P0-queries";
    fout.open(p0File);
    for (int i = 0; i < numQueries; ++i) {
        fout << i;
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
    ANNORAMOfflineConfig config;
    
    // 解析命令行参数
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--data-dir" && i + 1 < argc) {
            config.dataDir = argv[++i];
        } else if (arg == "--dataset" && i + 1 < argc) {
            config.datasetName = argv[++i];
        } else if (arg == "--clusters" && i + 1 < argc) {
            config.oram.numClusters = atoi(argv[++i]);
        } else if (arg == "--b-max" && i + 1 < argc) {
            config.oram.bMax = atoi(argv[++i]);
        } else if (arg == "--block-words" && i + 1 < argc) {
            config.oram.blockWords = atoi(argv[++i]);
        } else if (arg == "--gen-test") {
            int numRecords = 100, embDim = 32, numQueries = 3;
            if (i + 1 < argc) numRecords = atoi(argv[++i]);
            if (i + 1 < argc) embDim = atoi(argv[++i]);
            if (i + 1 < argc) numQueries = atoi(argv[++i]);
            
            generateTestData(config.dataDir, config.datasetName, 
                           numRecords, embDim, numQueries);
            return 0;
        }
    }
    
    cout << "========================================" << endl;
    cout << "  ANN ORAM 离线阶段（数据布局固定化）" << endl;
    cout << "========================================" << endl;
    cout << "数据目录: " << config.dataDir << endl;
    cout << "数据集名: " << config.datasetName << endl;
    cout << "聚类数量 K: " << config.oram.numClusters << endl;
    cout << "每类最大块 b_max: " << config.oram.bMax << endl;
    cout << "----------------------------------------" << endl;
    
    // 1. 加载数据
    string embFile = config.dataDir + "/" + config.datasetName + "-P1-embeddings";
    vector<EmbeddingRecord> records = ORAMDataIO::loadEmbeddings(embFile);
    
    if (records.empty()) {
        cerr << "[Error] 未加载到数据" << endl;
        return 1;
    }
    
    int embDim = records[0].embedding.size();
    config.oram.blockWords = embDim;  // 每 block 存完整 embedding
    
    // 自动计算 b_max
    int avgRecordsPerCluster = (records.size() + config.oram.numClusters - 1) / config.oram.numClusters;
    if (config.oram.bMax < avgRecordsPerCluster * 2) {
        config.oram.bMax = avgRecordsPerCluster * 2;
        cout << "[自动调整] b_max = " << config.oram.bMax << endl;
    }
    
    // 2. KMeans 聚类
    cout << "\n[阶段1] KMeans 聚类..." << endl;
    KMeansClusterer clusterer(config.oram.numClusters);
    KMeansResult kmeansResult = clusterer.cluster(records);
    
    ORAMDataIO::saveKMeansResult(config.dataDir, config.datasetName, kmeansResult);
    ORAMDataIO::saveORAMConfig(config.dataDir, config.datasetName, config.oram, embDim, records.size());
    
    // 3. 生成 ORAM Block 数据库
    cout << "\n[阶段2] 生成 ORAM Block 数据库（Padding）..." << endl;
    ORAMDataGenerator generator;
    vector<SharedBlock> p0Blocks, p1Blocks;
    generator.generateBlockDB(kmeansResult, config.oram, p0Blocks, p1Blocks);
    
    ORAMDataIO::saveBlockDB(config.dataDir, config.datasetName, 0, p0Blocks, embDim);
    ORAMDataIO::saveBlockDB(config.dataDir, config.datasetName, 1, p1Blocks, embDim);
    
    // 4. 生成聚类中心秘密共享
    cout << "\n[阶段3] 生成聚类中心秘密共享..." << endl;
    vector<SharedCentroid> p0Centroids, p1Centroids;
    generator.generateCentroidShares(kmeansResult.centroids, p0Centroids, p1Centroids);
    
    ORAMDataIO::saveCentroidShares(config.dataDir, config.datasetName, 0, p0Centroids);
    ORAMDataIO::saveCentroidShares(config.dataDir, config.datasetName, 1, p1Centroids);
    
    // 5. 生成 DCF 密钥
    cout << "\n[阶段4] 生成 DCF 密钥..." << endl;
    string fssDir = config.dataDir + "/2-fss";
    gen_fake_dcf(1, K_BITS, fssDir);
    
    cout << "\n========================================" << endl;
    cout << "  ANN ORAM 离线阶段完成！" << endl;
    cout << "========================================" << endl;
    cout << "ORAM 参数:" << endl;
    cout << "  K (聚类数) = " << config.oram.numClusters << endl;
    cout << "  b_max (每类最大块) = " << config.oram.bMax << endl;
    cout << "  M (总块数) = " << config.oram.totalBlocks() << endl;
    cout << "  Block Words = " << embDim << endl;
    
    return 0;
}
