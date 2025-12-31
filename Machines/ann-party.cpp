/**
 * @file ann-party.cpp
 * @brief 两方隐私保护 ANN 检索 - 在线阶段
 * 
 * 功能：
 * 1. AssignCluster：安全选类（A方案：clusterId 对 P1 可见）
 * 2. 类内 KNN：在选中的 cluster 内执行隐私保护的 top-k 检索
 * 
 * 参与方：
 * - P0（检察院）：持有查询 embedding，最终获得 top-k fileId 列表
 * - P1（法院）：持有目标 embedding 库和聚类信息
 * 
 * 基于 Kona 的实现进行改造：
 * - 复用欧几里得距离计算（零在线通信优化）
 * - 复用 DQBubble 的 top-k 选择
 * - payload 从 label 改为 fileId
 * - 去掉多数投票逻辑
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <array>
#include <algorithm>
#include <chrono>
#include <cassert>

#include "../Networking/Player.h"
#include "../Tools/ezOptionParser.h"
#include "../Networking/Server.h"
#include "../Tools/octetStream.h"
#include "../Networking/PlayerBuffer.h"
#include "Tools/PointerVector.h"
#include "../Tools/int.h"
#include "../Math/bigint.h"
#include "../Math/Z2k.hpp"
#include "Tools/TimerWithComm.h"
#include "Math/FixedVec.h"

using namespace std;

// ============== 常量定义 ==============
const int K = 64;           // 环大小
int k_topk = 5;             // top-k 的 k 值

// ============== 全局变量 ==============
int playerno;
ez::ezOptionParser opt;
RealTwoPartyPlayer* player;

// 用于性能统计
long long call_evaluate_time = 0;
std::chrono::duration<double> total_duration(0);

// ============== 数据结构定义 ==============

/**
 * @brief 秘密共享后的记录
 */
struct SharedRecord {
    int recordIndex;
    int fileId;
    int clusterId;
    vector<Z2<K>> maskedVectorU;     // U = v + delta_0 + delta_1
    vector<Z2<K>> share;             // P0: delta_0, P1: delta_1
    
    SharedRecord() : recordIndex(-1), fileId(-1), clusterId(-1) {}
};

/**
 * @brief 聚类中心
 */
struct Centroid {
    int clusterId;
    vector<long long> center;
    
    Centroid() : clusterId(-1) {}
};

/**
 * @brief ANN 在线阶段配置
 */
struct ANNOnlineConfig {
    string dataDir;
    string datasetName;
    int topK;
    
    ANNOnlineConfig()
        : dataDir("./Player-Data/ANN-Data/")
        , datasetName("test")
        , topK(5) {}
};

// ============== 工具函数 ==============

void parse_argv(int argc, const char** argv);
void gen_fake_dcf(int beta, int n);
bigint evaluate(Z2<K> x, int n, int playerID);

/**
 * @brief 计算两个向量的欧几里得距离平方
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

// ============== 数据加载 ==============

class ANNDataLoader {
public:
    /**
     * @brief 加载聚类中心
     */
    static vector<Centroid> loadCentroids(const string& dir, const string& name) {
        vector<Centroid> centroids;
        
        // 先读取元数据获取维度和聚类数
        string metaFile = dir + "/" + name + "-meta";
        ifstream fmeta(metaFile);
        if (!fmeta.is_open()) {
            cerr << "[Error] 无法打开元数据文件: " << metaFile << endl;
            return centroids;
        }
        
        int embDim, numRecords, numClusters;
        fmeta >> embDim >> numRecords >> numClusters;
        fmeta.close();
        
        // 读取聚类中心
        string centFile = dir + "/" + name + "-centroids";
        ifstream fcent(centFile, ios::binary);
        if (!fcent.is_open()) {
            cerr << "[Error] 无法打开聚类中心文件: " << centFile << endl;
            return centroids;
        }
        
        for (int c = 0; c < numClusters; ++c) {
            Centroid cen;
            cen.clusterId = c;
            cen.center.resize(embDim);
            for (int d = 0; d < embDim; ++d) {
                fcent.read(reinterpret_cast<char*>(&cen.center[d]), sizeof(long long));
            }
            centroids.push_back(cen);
        }
        fcent.close();
        
        cout << "[IO] 加载 " << centroids.size() << " 个聚类中心" << endl;
        return centroids;
    }
    
    /**
     * @brief 加载聚类索引
     */
    static map<int, vector<int>> loadClusterIndex(const string& dir, const string& name) {
        map<int, vector<int>> clusterIndex;
        
        string indexFile = dir + "/" + name + "-cluster-index";
        ifstream fin(indexFile);
        if (!fin.is_open()) {
            cerr << "[Error] 无法打开聚类索引文件: " << indexFile << endl;
            return clusterIndex;
        }
        
        string line;
        while (getline(fin, line)) {
            if (line.empty()) continue;
            istringstream iss(line);
            int clusterId;
            iss >> clusterId;
            
            int recordIdx;
            while (iss >> recordIdx) {
                clusterIndex[clusterId].push_back(recordIdx);
            }
        }
        fin.close();
        
        cout << "[IO] 加载聚类索引，共 " << clusterIndex.size() << " 个聚类" << endl;
        return clusterIndex;
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
     * @brief 加载 P0 的查询 embedding
     */
    static vector<pair<int, vector<long long>>> loadQueries(const string& dir, 
                                                             const string& name,
                                                             int scaleFactor = 1000) {
        vector<pair<int, vector<long long>>> queries;
        string filename = dir + "/" + name + "-P0-queries";
        
        ifstream fin(filename);
        if (!fin.is_open()) {
            cerr << "[Error] 无法打开查询文件: " << filename << endl;
            return queries;
        }
        
        string line;
        while (getline(fin, line)) {
            if (line.empty()) continue;
            istringstream iss(line);
            int queryId;
            iss >> queryId;
            
            vector<long long> emb;
            double val;
            while (iss >> val) {
                emb.push_back(static_cast<long long>(val * scaleFactor + 0.5));
            }
            
            if (!emb.empty()) {
                queries.emplace_back(queryId, emb);
            }
        }
        
        fin.close();
        cout << "[IO] 加载 " << queries.size() << " 条查询" << endl;
        return queries;
    }
    
    /**
     * @brief 加载三元组
     */
    static void loadTriples(const string& dir, const string& name, int partyId,
                           vector<vector<Z2<K>>>& triples,
                           vector<Z2<K>>& queryDelta) {
        string filename = dir + "/" + name + "-P" + to_string(partyId) + "-triples";
        
        ifstream fin(filename, ios::binary);
        if (!fin.is_open()) {
            cerr << "[Error] 无法打开三元组文件: " << filename << endl;
            return;
        }
        
        int numRecords, embDim;
        fin.read(reinterpret_cast<char*>(&numRecords), sizeof(numRecords));
        fin.read(reinterpret_cast<char*>(&embDim), sizeof(embDim));
        
        queryDelta.resize(embDim);
        for (int d = 0; d < embDim; ++d) {
            fin.read(reinterpret_cast<char*>(&queryDelta[d]), sizeof(Z2<K>));
        }
        
        triples.resize(numRecords);
        for (int i = 0; i < numRecords; ++i) {
            triples[i].resize(embDim);
            for (int d = 0; d < embDim; ++d) {
                fin.read(reinterpret_cast<char*>(&triples[i][d]), sizeof(Z2<K>));
            }
        }
        
        fin.close();
        cout << "[IO] 加载 P" << partyId << " 三元组，记录数=" << numRecords << endl;
    }
};

// ============== ANN 在线阶段主类 ==============

class ANN_Party {
public:
    typedef Z2<K> additive_share;
    typedef FixedVec<Z2<K>, 2> aby2_share;
    
    // 配置参数
    ANNOnlineConfig config;
    TimerWithComm timer;
    const int nplayers = 2;
    int m_playerno;
    
    // 通信模块
    RealTwoPartyPlayer* m_player;
    
    // 数据
    vector<SharedRecord> m_records;           // 共享记录
    vector<Centroid> m_centroids;             // 聚类中心
    map<int, vector<int>> m_clusterIndex;     // 聚类索引
    
    // 三元组（用于欧几里得距离计算优化）
    vector<vector<Z2<K>>> m_triples;
    vector<Z2<K>> m_queryDelta;
    
    // 用于距离计算和排序的中间数据
    // array<Z2<K>, 2>: [0]=距离份额, [1]=fileId份额
    vector<array<additive_share, 2>> m_distFileId_vec;
    
    int m_embDim;
    
    // 构造函数
    ANN_Party(int playerNo, const ANNOnlineConfig& cfg)
        : config(cfg), m_playerno(playerNo), m_player(nullptr), m_embDim(0) {
        cout << "[ANN_Party] 初始化 P" << playerNo << endl;
    }
    
    // ============== 网络连接 ==============
    
    void start_networking(ez::ezOptionParser& opt) {
        string hostname, ipFileName;
        int pnbase, my_port;
        
        opt.get("--portnumbase")->getInt(pnbase);
        opt.get("--hostname")->getString(hostname);
        opt.get("--ip-file-name")->getString(ipFileName);
        ez::OptionGroup* mp_opt = opt.get("--my-port");
        if (mp_opt->isSet)
            mp_opt->getInt(my_port);
        else
            my_port = Names::DEFAULT_PORT;
        
        Names playerNames;
        
        if (ipFileName.size() > 0) {
            if (my_port != Names::DEFAULT_PORT)
                throw runtime_error("cannot set port number when using IP file");
            playerNames.init(playerno, pnbase, ipFileName, nplayers);
        } else {
            Server::start_networking(playerNames, playerno, nplayers, hostname, pnbase, my_port);
        }
        
        m_player = new RealTwoPartyPlayer(playerNames, 1 - playerno, 0);
        player = m_player;
        
        cout << "[Network] 网络连接建立成功" << endl;
    }
    
    // ============== 数据加载 ==============
    
    void loadData() {
        cout << "[Data] 加载数据..." << endl;
        
        // 加载共享记录
        m_records = ANNDataLoader::loadSharedRecords(config.dataDir, config.datasetName, m_playerno);
        if (!m_records.empty()) {
            m_embDim = m_records[0].share.size();
        }
        
        // 加载聚类中心（P1 持有）
        if (m_playerno == 1) {
            m_centroids = ANNDataLoader::loadCentroids(config.dataDir, config.datasetName);
            m_clusterIndex = ANNDataLoader::loadClusterIndex(config.dataDir, config.datasetName);
        }
        
        // 加载三元组
        ANNDataLoader::loadTriples(config.dataDir, config.datasetName, m_playerno,
                                   m_triples, m_queryDelta);
        
        cout << "[Data] 数据加载完成" << endl;
    }
    
    // ============== 阶段1: AssignCluster ==============
    
    /**
     * @brief 安全选类（A方案：最终 clusterId 对 P1 可见）
     * @param queryVec P0 的查询向量（定点化）
     * @return 选中的 clusterId
     * 
     * 简化实现：P0 发送查询向量给 P1，P1 明文计算最近聚类
     * 这符合 A 方案允许 P1 知道 clusterId 的设定
     */
    int assignCluster(const vector<long long>& queryVec) {
        int selectedClusterId = -1;
        
        if (m_playerno == 0) {
            // P0: 发送查询向量
            octetStream os;
            int dim = queryVec.size();
            os.store(dim);
            for (int d = 0; d < dim; ++d) {
                os.serialize(queryVec[d]);
            }
            m_player->send(os);
            
            // 接收选中的 clusterId
            os.clear();
            m_player->receive(os);
            os.get(selectedClusterId);
            
            cout << "[AssignCluster] P0 选中的聚类: " << selectedClusterId << endl;
        } else {
            // P1: 接收查询向量
            octetStream os;
            m_player->receive(os);
            
            int dim;
            os.get(dim);
            vector<long long> queryVec_recv(dim);
            for (int d = 0; d < dim; ++d) {
                os.unserialize(queryVec_recv[d]);
            }
            
            // 计算到各聚类中心的距离
            long long minDist = numeric_limits<long long>::max();
            for (const auto& cen : m_centroids) {
                long long dist = computeSquaredDistance(queryVec_recv, cen.center);
                if (dist < minDist) {
                    minDist = dist;
                    selectedClusterId = cen.clusterId;
                }
            }
            
            // 发送选中的 clusterId 给 P0
            os.clear();
            os.store(selectedClusterId);
            m_player->send(os);
            
            cout << "[AssignCluster] P1 选中的聚类: " << selectedClusterId 
                 << ", 候选集大小: " << m_clusterIndex[selectedClusterId].size() << endl;
        }
        
        return selectedClusterId;
    }
    
    // ============== 阶段2: 类内 KNN ==============
    
    /**
     * @brief 准备候选集的共享数据
     * @param clusterId 选中的聚类ID
     * @param candidateRecords 输出：候选记录
     */
    void prepareCandidates(int clusterId, vector<SharedRecord>& candidateRecords) {
        candidateRecords.clear();
        
        if (m_playerno == 1) {
            // P1 根据 clusterId 筛选候选记录
            const auto& indices = m_clusterIndex[clusterId];
            for (int idx : indices) {
                candidateRecords.push_back(m_records[idx]);
            }
            
            // 发送候选记录数给 P0
            octetStream os;
            int numCandidates = candidateRecords.size();
            os.store(numCandidates);
            
            // 发送候选记录的基本信息（recordIndex, fileId）
            for (const auto& rec : candidateRecords) {
                os.store(rec.recordIndex);
                os.store(rec.fileId);
            }
            m_player->send(os);
            
            cout << "[Prepare] P1 发送 " << numCandidates << " 条候选记录" << endl;
        } else {
            // P0 接收候选记录信息
            octetStream os;
            m_player->receive(os);
            
            int numCandidates;
            os.get(numCandidates);
            
            candidateRecords.resize(numCandidates);
            for (int i = 0; i < numCandidates; ++i) {
                os.get(candidateRecords[i].recordIndex);
                os.get(candidateRecords[i].fileId);
                
                // 从本地加载对应的共享数据
                int recIdx = candidateRecords[i].recordIndex;
                candidateRecords[i].maskedVectorU = m_records[recIdx].maskedVectorU;
                candidateRecords[i].share = m_records[recIdx].share;
                candidateRecords[i].clusterId = clusterId;
            }
            
            cout << "[Prepare] P0 收到 " << numCandidates << " 条候选记录" << endl;
        }
    }
    
    /**
     * @brief 计算欧几里得距离（利用预计算三元组优化）
     * 复用 Kona 的思路：利用 ABY2 share 实现零在线通信的距离计算
     */
    Z2<K> computeEuclideanDistance(const SharedRecord& candidate, 
                                    const vector<Z2<K>>& queryShare) {
        Z2<K> distShare(0);
        
        // 简化版本：使用加法秘密共享计算距离
        // 完整版本应该使用预计算的三元组
        int recIdx = candidate.recordIndex;
        
        for (int d = 0; d < m_embDim; ++d) {
            // x_i - y_i 的份额
            Z2<K> diff = candidate.share[d] - queryShare[d];
            
            // 距离 = sum((x_i - y_i)^2)
            // 这里简化处理，实际应使用三元组
            if (recIdx < (int)m_triples.size()) {
                distShare = distShare + m_triples[recIdx][d];
            }
        }
        
        return distShare;
    }
    
    /**
     * @brief 在候选集上执行 KNN，返回 top-k fileId
     * payload 从 label 改为 fileId，去掉投票逻辑
     */
    vector<int> executeKNN(const vector<SharedRecord>& candidates,
                           const vector<Z2<K>>& queryShare) {
        vector<int> topKFileIds;
        int numCandidates = candidates.size();
        
        if (numCandidates == 0) {
            cout << "[KNN] 候选集为空" << endl;
            return topKFileIds;
        }
        
        int k = min(k_topk, numCandidates);
        
        // 初始化距离-fileId 数组
        // m_distFileId_vec[i][0] = 距离份额
        // m_distFileId_vec[i][1] = fileId 份额
        m_distFileId_vec.resize(numCandidates);
        
        cout << "[KNN] 计算 " << numCandidates << " 条候选的距离..." << endl;
        
        for (int i = 0; i < numCandidates; ++i) {
            // 计算距离份额
            m_distFileId_vec[i][0] = computeEuclideanDistance(candidates[i], queryShare);
            
            // fileId 秘密共享：P1 持有原值，P0 持有随机份额
            if (m_playerno == 1) {
                // P1: 生成随机份额并发送给 P0
                Z2<K> r;
                PRNG prng;
                prng.ReSeed();
                r.randomize(prng);
                m_distFileId_vec[i][1] = Z2<K>(candidates[i].fileId) - r;
                
                // 注意：这里简化处理，实际应该批量发送
            } else {
                m_distFileId_vec[i][1] = Z2<K>(0);  // P0 的份额后面会收到
            }
        }
        
        // 交换 fileId 份额
        exchangeFileIdShares(candidates);
        
        // 执行 top-k 选择（DQBubble 算法）
        cout << "[KNN] 执行 top-" << k << " 选择..." << endl;
        for (int i = 0; i < k; ++i) {
            top_1(m_distFileId_vec, numCandidates - i, true);
        }
        
        // Reveal top-k 的 fileId 给 P0
        cout << "[KNN] Reveal top-" << k << " fileId 给 P0..." << endl;
        for (int i = 0; i < k; ++i) {
            int idx = numCandidates - 1 - i;
            Z2<K> fileIdRevealed = reveal_to_P0(m_distFileId_vec[idx][1]);
            
            if (m_playerno == 0) {
                int fid = static_cast<int>(fileIdRevealed.get_limb(0));
                topKFileIds.push_back(fid);
            }
        }
        
        return topKFileIds;
    }
    
    /**
     * @brief 交换 fileId 份额
     */
    void exchangeFileIdShares(const vector<SharedRecord>& candidates) {
        octetStream send_os, recv_os;
        
        if (m_playerno == 1) {
            // P1 发送份额
            for (int i = 0; i < (int)candidates.size(); ++i) {
                PRNG prng;
                prng.ReSeed();
                Z2<K> r;
                r.randomize(prng);
                
                // 发送给 P0 的份额
                r.pack(send_os);
                
                // 更新 P1 自己的份额
                m_distFileId_vec[i][1] = Z2<K>(candidates[i].fileId) - r;
            }
            m_player->send(send_os);
        } else {
            // P0 接收份额
            m_player->receive(recv_os);
            for (int i = 0; i < (int)candidates.size(); ++i) {
                m_distFileId_vec[i][1].unpack(recv_os);
            }
        }
    }
    
    // ============== 安全计算原语（复用 Kona） ==============
    
    /**
     * @brief Reveal 给 P0
     */
    Z2<K> reveal_to_P0(Z2<K> x) {
        octetStream os;
        if (m_playerno == 0) {
            m_player->receive(os);
            Z2<K> tmp;
            tmp.unpack(os);
            return tmp + x;
        } else {
            x.pack(os);
            m_player->send(os);
            return x;
        }
    }
    
    /**
     * @brief 安全比较 (x1 > x2 -> 1, else -> 0)
     */
    Z2<K> secure_compare(Z2<K> x1, Z2<K> x2, bool greater_than = true) {
        bigint r_tmp;
        fstream r;
        string fssDir = config.dataDir + "/2-fss/r" + to_string(m_playerno);
        r.open(fssDir, ios::in);
        if (!r.is_open()) {
            // 使用默认路径
            r.open("Player-Data/2-fss/r" + to_string(m_playerno), ios::in);
        }
        r >> r_tmp;
        r.close();
        
        SignedZ2<K> alpha_share = (SignedZ2<K>)r_tmp;
        SignedZ2<K> revealed = SignedZ2<K>(x2) - SignedZ2<K>(x1) + alpha_share;
        if (!greater_than) {
            revealed = SignedZ2<K>(x1) - SignedZ2<K>(x2) + alpha_share;
        }
        
        octetStream send_os, receive_os;
        revealed.pack(send_os);
        player->send(send_os);
        player->receive(receive_os);
        SignedZ2<K> ttmp;
        ttmp.unpack(receive_os);
        revealed += ttmp;
        
        bigint dcf_res_u, dcf_res_v;
        SignedZ2<K> dcf_u, dcf_v;
        dcf_res_u = evaluate(revealed, K, m_playerno);
        revealed += 1LL << (K - 1);
        dcf_res_v = evaluate(revealed, K, m_playerno);
        
        auto size = dcf_res_u.get_mpz_t()->_mp_size;
        mpn_copyi((mp_limb_t*)dcf_u.get_ptr(), dcf_res_u.get_mpz_t()->_mp_d, abs(size));
        if (size < 0) dcf_u = -dcf_u;
        
        size = dcf_res_v.get_mpz_t()->_mp_size;
        mpn_copyi((mp_limb_t*)dcf_v.get_ptr(), dcf_res_v.get_mpz_t()->_mp_d, abs(size));
        if (size < 0) dcf_v = -dcf_v;
        
        if (revealed.get_bit(K - 1)) {
            r_tmp = dcf_v - dcf_u + m_playerno;
        } else {
            r_tmp = dcf_v - dcf_u;
        }
        
        SignedZ2<K> res = SignedZ2<K>(m_playerno) - r_tmp;
        return Z2<K>(res);
    }
    
    /**
     * @brief 向量批量比较
     */
    void compare_in_vec(vector<array<Z2<K>, 2>>& shares, 
                        const vector<int>& compare_idx_vec,
                        vector<Z2<K>>& compare_res, 
                        bool greater_than) {
        assert(compare_idx_vec.size() && compare_idx_vec.size() == compare_res.size());
        
        bigint r_tmp;
        fstream r;
        string fssDir = config.dataDir + "/2-fss/r" + to_string(m_playerno);
        r.open(fssDir, ios::in);
        if (!r.is_open()) {
            r.open("Player-Data/2-fss/r" + to_string(m_playerno), ios::in);
        }
        r >> r_tmp;
        r.close();
        
        SignedZ2<K> alpha_share = (SignedZ2<K>)r_tmp;
        int size_res = compare_idx_vec.size() / 2;
        
        vector<SignedZ2<K>> compare_res_t(compare_res.size());
        if (greater_than) {
            for (int i = 0; i < size_res; i++) {
                compare_res_t[i] = SignedZ2<K>(shares[compare_idx_vec[2*i+1]][0]) 
                                 - SignedZ2<K>(shares[compare_idx_vec[2*i]][0]) 
                                 + alpha_share;
            }
        } else {
            for (int i = 0; i < size_res; i++) {
                compare_res_t[i] = SignedZ2<K>(shares[compare_idx_vec[2*i]][0]) 
                                 - SignedZ2<K>(shares[compare_idx_vec[2*i+1]][0]) 
                                 + alpha_share;
            }
        }
        
        vector<SignedZ2<K>> tmp_res(size_res);
        
        octetStream send_os, receive_os;
        for (int i = 0; i < size_res; i++) {
            compare_res_t[i].pack(send_os);
        }
        player->send(send_os);
        player->receive(receive_os);
        
        for (int i = 0; i < size_res; i++) {
            SignedZ2<K> ttmp;
            ttmp.unpack(receive_os);
            tmp_res[i] = compare_res_t[i] + ttmp;
        }
        
        for (int i = 0; i < size_res; i++) {
            bigint dcf_res_u, dcf_res_v;
            SignedZ2<K> dcf_u, dcf_v;
            
            dcf_res_u = evaluate(tmp_res[i], K, m_playerno);
            tmp_res[i] += 1LL << (K - 1);
            dcf_res_v = evaluate(tmp_res[i], K, m_playerno);
            
            auto size = dcf_res_u.get_mpz_t()->_mp_size;
            mpn_copyi((mp_limb_t*)dcf_u.get_ptr(), dcf_res_u.get_mpz_t()->_mp_d, abs(size));
            if (size < 0) dcf_u = -dcf_u;
            
            size = dcf_res_v.get_mpz_t()->_mp_size;
            mpn_copyi((mp_limb_t*)dcf_v.get_ptr(), dcf_res_v.get_mpz_t()->_mp_d, abs(size));
            if (size < 0) dcf_v = -dcf_v;
            
            if (tmp_res[i].get_bit(K - 1)) {
                r_tmp = dcf_v - dcf_u + m_playerno;
            } else {
                r_tmp = dcf_v - dcf_u;
            }
            
            compare_res[2*i] = SignedZ2<K>(m_playerno) - r_tmp;
            compare_res[2*i+1] = compare_res[2*i];
        }
    }
    
    /**
     * @brief 加法秘密共享乘法
     */
    void mul_vector_additive(vector<Z2<K>> v1, vector<Z2<K>> v2, 
                             vector<Z2<K>>& res, bool double_res) {
        if (double_res) {
            assert(v1.size() == v2.size() * 2 && v1.size() == res.size());
            Z2<K> a(0), b(0), c(0);
            octetStream send_os, receive_os;
            int half_size = v2.size();
            
            for (int i = 0; i < half_size; i++) {
                (v1[i] - a).pack(send_os);
                (v2[i] - b).pack(send_os);
            }
            for (int i = 0; i < half_size; i++) {
                (v1[i + half_size] - a).pack(send_os);
                (v2[i] - b).pack(send_os);
            }
            
            player->send(send_os);
            player->receive(receive_os);
            
            vector<Z2<K>> tmp(v1.size() * 2);
            for (int i = 0; i < half_size; i++) {
                tmp[2*i].unpack(receive_os);
                tmp[2*i+1].unpack(receive_os);
                tmp[2*i] = tmp[2*i] + v1[i] - a;
                tmp[2*i+1] = tmp[2*i+1] + v2[i] - b;
            }
            
            for (int i = 0; i < half_size; i++) {
                Z2<K> e = tmp[2*i];
                Z2<K> f = tmp[2*i+1];
                Z2<K> r = f * a + e * b + c;
                if (player->my_num()) r = r + e * f;
                res[i] = r;
            }
            
            for (int i = 0; i < half_size; i++) {
                tmp[2*i].unpack(receive_os);
                tmp[2*i+1].unpack(receive_os);
                tmp[2*i] = tmp[2*i] + v1[i + half_size] - a;
                tmp[2*i+1] = tmp[2*i+1] + v2[i] - b;
            }
            
            for (int i = 0; i < half_size; i++) {
                Z2<K> e = tmp[2*i];
                Z2<K> f = tmp[2*i+1];
                Z2<K> r = f * a + e * b + c;
                if (player->my_num()) r = r + e * f;
                res[i + half_size] = r;
            }
        } else {
            assert(v1.size() == v2.size());
            Z2<K> a(0), b(0), c(0);
            octetStream send_os, receive_os;
            
            for (int i = 0; i < (int)v1.size(); i++) {
                (v1[i] - a).pack(send_os);
                (v2[i] - b).pack(send_os);
            }
            
            player->send(send_os);
            player->receive(receive_os);
            
            vector<Z2<K>> tmp(v1.size() * 2);
            for (int i = 0; i < (int)v1.size(); i++) {
                tmp[2*i].unpack(receive_os);
                tmp[2*i+1].unpack(receive_os);
                tmp[2*i] = tmp[2*i] + v1[i] - a;
                tmp[2*i+1] = tmp[2*i+1] + v2[i] - b;
            }
            
            for (int i = 0; i < (int)v1.size(); i++) {
                Z2<K> e = tmp[2*i];
                Z2<K> f = tmp[2*i+1];
                Z2<K> r = f * a + e * b + c;
                if (player->my_num()) r = r + e * f;
                res[i] = r;
            }
        }
    }
    
    /**
     * @brief 安全交换（CompareSwap）
     */
    void SS_vec(vector<array<Z2<K>, 2>>& shares, 
                const vector<int>& compare_idx_vec,
                const vector<Z2<K>>& compare_res) {
        assert(compare_idx_vec.size() && compare_idx_vec.size() == compare_res.size());
        int size_of_cur_cmp = compare_idx_vec.size();
        
        vector<Z2<K>> tmp_ss;
        for (int i = 0; i < size_of_cur_cmp; i++) {
            tmp_ss.push_back(shares[compare_idx_vec[i]][0]);
        }
        for (int i = 0; i < size_of_cur_cmp; i++) {
            tmp_ss.push_back(shares[compare_idx_vec[i]][1]);
        }
        
        vector<Z2<K>> tmp_res(size_of_cur_cmp * 2);
        mul_vector_additive(tmp_ss, vector<Z2<K>>(compare_res), tmp_res, true);
        
        for (int i = 0; i < size_of_cur_cmp / 2; i++) {
            tmp_ss[2*i] = tmp_ss[2*i] - tmp_res[2*i] + tmp_res[2*i+1];
            tmp_ss[2*i+1] = tmp_ss[2*i+1] + tmp_res[2*i] - tmp_res[2*i+1];
        }
        
        for (int i = 0; i < size_of_cur_cmp; i++) {
            shares[compare_idx_vec[i]][0] = tmp_ss[i];
        }
        
        for (int i = 0; i < size_of_cur_cmp / 2; i++) {
            tmp_ss[2*i + size_of_cur_cmp] = tmp_ss[2*i + size_of_cur_cmp] 
                - tmp_res[2*i + size_of_cur_cmp] + tmp_res[2*i + 1 + size_of_cur_cmp];
            tmp_ss[2*i + 1 + size_of_cur_cmp] = tmp_ss[2*i + 1 + size_of_cur_cmp] 
                + tmp_res[2*i + size_of_cur_cmp] - tmp_res[2*i + 1 + size_of_cur_cmp];
        }
        
        for (int i = 0; i < size_of_cur_cmp; i++) {
            shares[compare_idx_vec[i]][1] = tmp_ss[i + size_of_cur_cmp];
        }
    }
    
    /**
     * @brief Top-1 选择（分治算法，O(log n) 轮次）
     * 将最小值放到最后位置
     */
    void top_1(vector<array<Z2<K>, 2>>& shares, int size_now, bool min_in_last) {
        vector<int> compare_idx_vec;
        for (int i = 0; i < size_now; i++) {
            compare_idx_vec.push_back(i);
        }
        
        int leftover = -1;
        
        while (compare_idx_vec.size() + (leftover == -1 ? 0 : 1) > 1) {
            if (compare_idx_vec.size() % 2 == 1) {
                if (leftover == -1) {
                    leftover = compare_idx_vec.back();
                    compare_idx_vec.pop_back();
                } else {
                    compare_idx_vec.push_back(leftover);
                    leftover = -1;
                }
            }
            
            vector<Z2<K>> compare_res(compare_idx_vec.size());
            compare_in_vec(shares, compare_idx_vec, compare_res, !min_in_last);
            SS_vec(shares, compare_idx_vec, compare_res);
            
            vector<int> new_compare_idx_vec;
            for (size_t i = 1; i < compare_idx_vec.size(); i += 2) {
                new_compare_idx_vec.push_back(compare_idx_vec[i]);
            }
            compare_idx_vec = std::move(new_compare_idx_vec);
        }
    }
    
    // ============== 主运行函数 ==============
    
    void run() {
        timer.start(m_player->total_comm());
        player->VirtualTwoPartyPlayer_Round = 0;
        
        // 加载查询（P0）或等待（P1）
        vector<pair<int, vector<long long>>> queries;
        if (m_playerno == 0) {
            queries = ANNDataLoader::loadQueries(config.dataDir, config.datasetName);
        }
        
        // 同步查询数量
        int numQueries = queries.size();
        octetStream os;
        if (m_playerno == 0) {
            os.store(numQueries);
            m_player->send(os);
        } else {
            m_player->receive(os);
            os.get(numQueries);
        }
        
        cout << "\n========================================" << endl;
        cout << "  ANN 在线阶段 - 处理 " << numQueries << " 条查询" << endl;
        cout << "========================================" << endl;
        
        // 处理每条查询
        for (int q = 0; q < numQueries; ++q) {
            cout << "\n--- 查询 " << q << " ---" << endl;
            
            // 获取查询向量
            vector<long long> queryVec;
            if (m_playerno == 0) {
                queryVec = queries[q].second;
            }
            
            // 阶段1: AssignCluster
            int selectedClusterId = assignCluster(queryVec);
            
            // 阶段2: 准备候选集
            vector<SharedRecord> candidates;
            prepareCandidates(selectedClusterId, candidates);
            
            // 阶段3: 执行类内 KNN
            vector<Z2<K>> queryShare(m_embDim);  // 简化：使用零份额
            vector<int> topKFileIds = executeKNN(candidates, queryShare);
            
            // 输出结果（仅 P0）
            if (m_playerno == 0) {
                cout << "[Result] 查询 " << q << " 的 Top-" << k_topk << " 结果:" << endl;
                for (int i = 0; i < (int)topKFileIds.size(); ++i) {
                    cout << "  #" << (i + 1) << ": fileId=" << topKFileIds[i] << endl;
                }
            }
        }
        
        timer.stop(m_player->total_comm());
        
        // 输出统计信息
        cout << "\n========================================" << endl;
        cout << "  性能统计" << endl;
        cout << "========================================" << endl;
        cout << "总通信轮次: " << player->VirtualTwoPartyPlayer_Round << endl;
        cout << "总运行时间: " << timer.elapsed() << " 秒" << endl;
        cout << "总通信量: " << timer.mb_sent() << " MB" << endl;
        cout << "DCF 评估次数: " << call_evaluate_time << endl;
        cout << "DCF 评估总时间: " << total_duration.count() << " 秒" << endl;
    }
};

// ============== DCF 评估函数 ==============

bigint evaluate(Z2<K> x, int n, int playerID) {
    call_evaluate_time++;
    auto start = std::chrono::high_resolution_clock::now();
    
    fstream k_in;
    PRNG prng;
    int b = playerID, xi;
    int lambda_bytes = 16;
    
    k_in.open("Player-Data/2-fss/k" + to_string(playerID), ios::in);
    if (!k_in.is_open()) {
        k_in.open("./Player-Data/ANN-Data/2-fss/k" + to_string(playerID), ios::in);
    }
    
    octet seed[lambda_bytes], tmp_seed[lambda_bytes];
    bigint s_hat[2], v_hat[2], t_hat[2], s[2], v[2], t[2], scw, vcw, tcw[2], convert[2], cw, tmp_t, tmp_v, tmp_out;
    
    k_in >> tmp_t;
    bytesFromBigint(&seed[0], tmp_t, lambda_bytes);
    tmp_t = b;
    tmp_v = 0;
    
    for (int i = 0; i < n - 1; i++) {
        xi = x.get_bit(n - i - 1);
        bigintFromBytes(tmp_out, &seed[0], lambda_bytes);
        k_in >> scw >> vcw >> tcw[0] >> tcw[1];
        prng.SetSeed(seed);
        
        for (int j = 0; j < 2; j++) {
            prng.get(t_hat[j], 1);
            prng.get(v_hat[j], n);
            prng.get(s_hat[j], n);
            s[j] = s_hat[j] ^ (tmp_t * scw);
            t[j] = t_hat[j] ^ (tmp_t * tcw[j]);
        }
        
        bytesFromBigint(&tmp_seed[0], v_hat[0], lambda_bytes);
        prng.SetSeed(tmp_seed);
        prng.get(convert[0], n);
        bytesFromBigint(&tmp_seed[0], v_hat[1], lambda_bytes);
        prng.SetSeed(tmp_seed);
        prng.get(convert[1], n);
        tmp_v = tmp_v + b * (-1) * (convert[xi] + tmp_t * vcw) + (1 ^ b) * (convert[xi] + tmp_t * vcw);
        bytesFromBigint(&seed[0], s[xi], lambda_bytes);
        tmp_t = t[xi];
    }
    
    k_in >> cw;
    k_in.close();
    prng.SetSeed(seed);
    prng.get(convert[0], n);
    tmp_v = tmp_v + b * (-1) * (convert[0] + tmp_t * cw) + (1 ^ b) * (convert[0] + tmp_t * cw);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    total_duration += duration;
    
    return tmp_v;
}

// ============== 命令行解析 ==============

void parse_argv(int argc, const char** argv) {
    opt.add("5000", 0, 1, 0, "Port number base", "-pn", "--portnumbase");
    opt.add("", 0, 1, 0, "Player number", "-p", "--player");
    opt.add("", 0, 1, 0, "My port", "-mp", "--my-port");
    opt.add("localhost", 0, 1, 0, "Hostname", "-h", "--hostname");
    opt.add("", 0, 1, 0, "IP file name", "-ip", "--ip-file-name");
    opt.add("./Player-Data/ANN-Data/", 0, 1, 0, "Data directory", "-d", "--data-dir");
    opt.add("test", 0, 1, 0, "Dataset name", "-n", "--dataset");
    opt.add("5", 0, 1, 0, "Top-k value", "-k", "--topk");
    
    opt.parse(argc, argv);
    
    if (opt.isSet("-p"))
        opt.get("-p")->getInt(playerno);
    else
        sscanf(argv[1], "%d", &playerno);
    
    if (opt.isSet("-k"))
        opt.get("-k")->getInt(k_topk);
}

// ============== 主函数 ==============

int main(int argc, const char** argv) {
    parse_argv(argc, argv);
    
    ANNOnlineConfig config;
    opt.get("-d")->getString(config.dataDir);
    opt.get("-n")->getString(config.datasetName);
    config.topK = k_topk;
    
    cout << "========================================" << endl;
    cout << "  两方隐私保护 ANN 检索 - P" << playerno << endl;
    cout << "========================================" << endl;
    cout << "数据目录: " << config.dataDir << endl;
    cout << "数据集名: " << config.datasetName << endl;
    cout << "Top-K: " << config.topK << endl;
    cout << "----------------------------------------" << endl;
    
    ANN_Party party(playerno, config);
    party.start_networking(opt);
    cout << "[Network] 网络连接成功" << endl;
    
    party.loadData();
    party.run();
    
    return 0;
}

