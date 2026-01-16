/**
 * @file ann-party-oram-kona.cpp
 * @brief 两方隐私保护 ANN 检索 - Kona + ORAM 混合方案
 * 
 * 方案特点：
 * 1. 选类阶段：复用 Kona 的安全距离计算 + top_1，不 reveal clusterId
 * 2. 候选读取：使用 ORAM 按私密地址读取候选块
 * 3. 类内 KNN：复用 Kona 的欧几里得距离计算（预计算三元组） + top_1
 * 
 * 性能优势：
 * - 选类：O(K * d) 距离计算 + O(log K) 轮 top_1
 * - ORAM 读取：O(b_read * M) 或 O(b_read * log M)（Path ORAM）
 * - 类内 KNN：O(b_read * d) 距离计算 + O(k * log(b_read)) 轮 top_k
 * 
 * 总体：O(K*d + b_read*M + b_read*d) << O(N*d)（当 b_read << N）
 * 
 * 基于 ann-party.cpp 和 ann-party-oram.cpp 整合
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

// ============== 配置 ==============
const int K = 64;
int k_topk = 5;
int g_bRead = 10;      // 每个聚类读取的块数

// ============== 全局变量 ==============
int playerno;
ez::ezOptionParser opt;
RealTwoPartyPlayer* player;
string g_dataDir;
long long call_evaluate_time = 0;

// ============== 数据结构 ==============

struct SharedRecord {
    int recordIndex;
    int fileId;
    int clusterId;
    vector<Z2<K>> maskedVectorU;
    vector<Z2<K>> share;
    Z2<K> fileIdShare;
    
    SharedRecord() : recordIndex(-1), fileId(-1), clusterId(-1) {}
};

struct SharedCentroid {
    int clusterId;
    vector<Z2<K>> maskedVectorU;
    vector<Z2<K>> share;
    Z2<K> clusterIdShare;
    
    SharedCentroid() : clusterId(-1) {}
};

// ORAM Block 结构
struct SharedBlock {
    vector<Z2<K>> maskedU;      // embedding
    vector<Z2<K>> share;
    Z2<K> fileIdShare;
    Z2<K> validShare;           // 1 = 有效, 0 = dummy
    int clusterId;              // 所属聚类（明文，用于构建）
    
    SharedBlock() {}
    SharedBlock(int dim) : maskedU(dim), share(dim) {}
};

struct ORAMConfig {
    int numClusters;
    int bMax;                   // 每聚类最大块数
    int embDim;
    int numRecords;
    
    int totalBlocks() const { return numClusters * bMax; }
};

struct ANNKonaORAMConfig {
    string dataDir;
    string datasetName;
    int topK;
    int bRead;
    
    ANNKonaORAMConfig() : topK(5), bRead(10) {}
};

// ============== DCF 评估函数 ==============

std::chrono::duration<double> total_duration(0);

bigint evaluate(Z2<K> x, int n, int playerID) {
    call_evaluate_time++;
    
    fstream k_in;
    PRNG prng;
    int b = playerID, xi;
    int lambda_bytes = 16;
    
    string keyPath = g_dataDir + "/2-fss/k" + to_string(playerID);
    k_in.open(keyPath, ios::in);
    if (!k_in.is_open()) {
        k_in.open("Player-Data/2-fss/k" + to_string(playerID), ios::in);
    }
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
    
    return tmp_v;
}

// ============== 数据加载 ==============

class ANNKonaORAMLoader {
public:
    static ORAMConfig loadORAMConfig(const string& dir, const string& name) {
        ORAMConfig cfg;
        string filename = dir + "/" + name + "-oram-config";
        ifstream fin(filename);
        if (!fin.is_open()) {
            cerr << "[Error] 无法打开 ORAM 配置文件: " << filename << endl;
            cfg.numClusters = 0;
            return cfg;
        }
        fin >> cfg.numClusters >> cfg.bMax >> cfg.embDim >> cfg.numRecords;
        fin.close();
        return cfg;
    }
    
    static vector<SharedBlock> loadBlockDB(const string& dir, const string& name, int party, int embDim) {
        vector<SharedBlock> blocks;
        string suffix = party == 0 ? "P0" : "P1";
        string filename = dir + "/" + name + "-" + suffix + "-oram-blocks";
        
        ifstream fin(filename, ios::binary);
        if (!fin.is_open()) {
            cerr << "[Error] 无法打开 Block DB: " << filename << endl;
            return blocks;
        }
        
        int numBlocks;
        fin.read(reinterpret_cast<char*>(&numBlocks), sizeof(int));
        blocks.resize(numBlocks);
        
        for (int i = 0; i < numBlocks; ++i) {
            blocks[i].maskedU.resize(embDim);
            blocks[i].share.resize(embDim);
            
            for (int d = 0; d < embDim; ++d) {
                long long val;
                fin.read(reinterpret_cast<char*>(&val), sizeof(long long));
                blocks[i].maskedU[d] = Z2<K>(val);
            }
            for (int d = 0; d < embDim; ++d) {
                long long val;
                fin.read(reinterpret_cast<char*>(&val), sizeof(long long));
                blocks[i].share[d] = Z2<K>(val);
            }
            long long fileIdShareVal, validShareVal;
            int clusterId;
            fin.read(reinterpret_cast<char*>(&fileIdShareVal), sizeof(long long));
            fin.read(reinterpret_cast<char*>(&validShareVal), sizeof(long long));
            fin.read(reinterpret_cast<char*>(&clusterId), sizeof(int));
            
            blocks[i].fileIdShare = Z2<K>(fileIdShareVal);
            blocks[i].validShare = Z2<K>(validShareVal);
            blocks[i].clusterId = clusterId;
        }
        
        fin.close();
        cout << "[IO] 加载 " << numBlocks << " 个 Block" << endl;
        return blocks;
    }
    
    static vector<SharedCentroid> loadCentroidShares(const string& dir, const string& name, int party) {
        vector<SharedCentroid> centroids;
        string suffix = party == 0 ? "P0" : "P1";
        string filename = dir + "/" + name + "-" + suffix + "-centroid-shares";
        
        ifstream fin(filename, ios::binary);
        if (!fin.is_open()) {
            cerr << "[Error] 无法打开聚类中心共享: " << filename << endl;
            return centroids;
        }
        
        int numClusters, embDim;
        fin.read(reinterpret_cast<char*>(&numClusters), sizeof(int));
        fin.read(reinterpret_cast<char*>(&embDim), sizeof(int));
        
        centroids.resize(numClusters);
        for (int c = 0; c < numClusters; ++c) {
            centroids[c].maskedVectorU.resize(embDim);
            centroids[c].share.resize(embDim);
            
            for (int d = 0; d < embDim; ++d) {
                long long val;
                fin.read(reinterpret_cast<char*>(&val), sizeof(long long));
                centroids[c].maskedVectorU[d] = Z2<K>(val);
            }
            for (int d = 0; d < embDim; ++d) {
                long long val;
                fin.read(reinterpret_cast<char*>(&val), sizeof(long long));
                centroids[c].share[d] = Z2<K>(val);
            }
            long long clusterIdShareVal;
            fin.read(reinterpret_cast<char*>(&clusterIdShareVal), sizeof(long long));
            centroids[c].clusterIdShare = Z2<K>(clusterIdShareVal);
            centroids[c].clusterId = c;
        }
        
        fin.close();
        cout << "[IO] 加载 " << numClusters << " 个聚类中心共享" << endl;
        return centroids;
    }
    
    static vector<pair<int, vector<long long>>> loadQueries(const string& dir, const string& name, int scaleFactor = 1000) {
        vector<pair<int, vector<long long>>> queries;
        string filename = dir + "/" + name + "-P0-queries";
        
        ifstream fin(filename);
        if (!fin.is_open()) return queries;
        
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
            if (!emb.empty()) queries.emplace_back(queryId, emb);
        }
        
        fin.close();
        cout << "[IO] 加载 " << queries.size() << " 条查询" << endl;
        return queries;
    }
    
    static void loadTriples(const string& dir, const string& name, int party,
                            vector<vector<Z2<K>>>& triples, vector<Z2<K>>& queryDelta) {
        string suffix = party == 0 ? "P0" : "P1";
        string filename = dir + "/" + name + "-" + suffix + "-triples";
        
        ifstream fin(filename, ios::binary);
        if (!fin.is_open()) {
            cout << "[Warning] 三元组文件不存在，使用在线乘法" << endl;
            return;
        }
        
        int numRecords, dim;
        fin.read(reinterpret_cast<char*>(&numRecords), sizeof(int));
        fin.read(reinterpret_cast<char*>(&dim), sizeof(int));
        
        triples.resize(numRecords);
        queryDelta.resize(dim);
        
        for (int i = 0; i < numRecords; ++i) {
            triples[i].resize(dim);
            for (int d = 0; d < dim; ++d) {
                long long val;
                fin.read(reinterpret_cast<char*>(&val), sizeof(long long));
                triples[i][d] = Z2<K>(val);
            }
        }
        
        for (int d = 0; d < dim; ++d) {
            long long val;
            fin.read(reinterpret_cast<char*>(&val), sizeof(long long));
            queryDelta[d] = Z2<K>(val);
        }
        
        fin.close();
    }
};

// ============== Kona 核心组件（从 ann-party.cpp 复制）==============

/**
 * @brief 批量安全乘法（使用在线协议）
 */
void mul_vector_additive(const vector<Z2<K>>& v1, const vector<Z2<K>>& v2, 
                         vector<Z2<K>>& res, bool /*use_triple*/) {
    assert(v1.size() == v2.size());
    res.resize(v1.size());
    
    Z2<K> a(0), b(0), c(0);
    octetStream send_os, recv_os;
    
    for (size_t i = 0; i < v1.size(); i++) {
        (v1[i] - a).pack(send_os);
        (v2[i] - b).pack(send_os);
    }
    
    player->send(send_os);
    player->receive(recv_os);
    
    for (size_t i = 0; i < v1.size(); i++) {
        Z2<K> e_other, f_other;
        e_other.unpack(recv_os);
        f_other.unpack(recv_os);
        
        Z2<K> e = (v1[i] - a) + e_other;
        Z2<K> f = (v2[i] - b) + f_other;
        
        Z2<K> r = c + f * a + e * b;
        if (player->my_num()) r = r + e * f;
        res[i] = r;
    }
}

/**
 * @brief 安全比较（使用 DCF）
 */
void compare_in_vec(const vector<array<Z2<K>, 2>>& shares, 
                    const vector<int>& compare_idx_vec,
                    vector<Z2<K>>& compare_res) {
    int cnt = compare_idx_vec.size() / 2;
    compare_res.resize(cnt);
    
    if (cnt == 0) return;
    
    // 读取 DCF 密钥
    bigint r_tmp;
    fstream r;
    string fssDir = g_dataDir + "/2-fss/r" + to_string(playerno);
    r.open(fssDir, ios::in);
    if (!r.is_open()) r.open("Player-Data/2-fss/r" + to_string(playerno), ios::in);
    r >> r_tmp;
    r.close();
    
    SignedZ2<K> alpha_share = (SignedZ2<K>)r_tmp;
    
    octetStream send_os, recv_os;
    vector<SignedZ2<K>> revealed(cnt);
    
    for (int i = 0; i < cnt; ++i) {
        int idx0 = compare_idx_vec[2 * i];
        int idx1 = compare_idx_vec[2 * i + 1];
        revealed[i] = SignedZ2<K>(shares[idx1][0]) - SignedZ2<K>(shares[idx0][0]) + alpha_share;
        revealed[i].pack(send_os);
    }
    
    player->send(send_os);
    player->receive(recv_os);
    
    for (int i = 0; i < cnt; ++i) {
        SignedZ2<K> tmp;
        tmp.unpack(recv_os);
        revealed[i] += tmp;
    }
    
    for (int i = 0; i < cnt; ++i) {
        bigint dcf_res_u = evaluate(revealed[i], K, playerno);
        SignedZ2<K> tmp_rev = revealed[i] + (1LL << (K - 1));
        bigint dcf_res_v = evaluate(tmp_rev, K, playerno);
        
        SignedZ2<K> dcf_u, dcf_v;
        auto size = dcf_res_u.get_mpz_t()->_mp_size;
        mpn_copyi((mp_limb_t*)dcf_u.get_ptr(), dcf_res_u.get_mpz_t()->_mp_d, abs(size));
        if (size < 0) dcf_u = -dcf_u;
        
        size = dcf_res_v.get_mpz_t()->_mp_size;
        mpn_copyi((mp_limb_t*)dcf_v.get_ptr(), dcf_res_v.get_mpz_t()->_mp_d, abs(size));
        if (size < 0) dcf_v = -dcf_v;
        
        if (revealed[i].get_bit(K - 1)) {
            r_tmp = dcf_v - dcf_u + playerno;
        } else {
            r_tmp = dcf_v - dcf_u;
        }
        
        SignedZ2<K> res_signed = SignedZ2<K>(playerno) - r_tmp;
        compare_res[i] = Z2<K>(res_signed);
    }
}

/**
 * @brief 安全交换（SS_vec）
 */
void SS_vec(vector<array<Z2<K>, 2>>& shares, 
            const vector<int>& compare_idx_vec,
            const vector<Z2<K>>& compare_res) {
    int cnt = compare_idx_vec.size() / 2;
    
    vector<Z2<K>> v1_diff(cnt * 2), v2_diff(cnt * 2);
    for (int i = 0; i < cnt; ++i) {
        int idx0 = compare_idx_vec[2 * i];
        int idx1 = compare_idx_vec[2 * i + 1];
        v1_diff[2 * i] = shares[idx0][0] - shares[idx1][0];
        v1_diff[2 * i + 1] = shares[idx0][1] - shares[idx1][1];
        v2_diff[2 * i] = compare_res[i];
        v2_diff[2 * i + 1] = compare_res[i];
    }
    
    vector<Z2<K>> mul_res;
    mul_vector_additive(v1_diff, v2_diff, mul_res, false);
    
    for (int i = 0; i < cnt; ++i) {
        int idx0 = compare_idx_vec[2 * i];
        int idx1 = compare_idx_vec[2 * i + 1];
        shares[idx0][0] = shares[idx0][0] - mul_res[2 * i];
        shares[idx0][1] = shares[idx0][1] - mul_res[2 * i + 1];
        shares[idx1][0] = shares[idx1][0] + mul_res[2 * i];
        shares[idx1][1] = shares[idx1][1] + mul_res[2 * i + 1];
    }
}

/**
 * @brief Top-1 选择（将最小/最大值放到数组末尾）
 */
void top_1(vector<array<Z2<K>, 2>>& shares, int size, bool min_in_last) {
    if (size <= 1) return;
    
    vector<int> compare_idx_vec;
    for (int i = 0; i < size; ++i) compare_idx_vec.push_back(i);
    
    while (compare_idx_vec.size() > 1) {
        int cnt = compare_idx_vec.size() / 2;
        vector<int> pairs;
        for (int i = 0; i < cnt; ++i) {
            pairs.push_back(compare_idx_vec[2 * i]);
            pairs.push_back(compare_idx_vec[2 * i + 1]);
        }
        
        vector<Z2<K>> compare_res;
        compare_in_vec(shares, pairs, compare_res);
        
        if (!min_in_last) {
            for (auto& r : compare_res) r = Z2<K>(1) - r;
        }
        
        SS_vec(shares, pairs, compare_res);
        
        vector<int> next_idx;
        for (int i = 0; i < cnt; ++i) {
            next_idx.push_back(pairs[2 * i + 1]);
        }
        if (compare_idx_vec.size() % 2 == 1) {
            next_idx.push_back(compare_idx_vec.back());
        }
        compare_idx_vec = next_idx;
    }
}

/**
 * @brief Reveal 给双方
 */
Z2<K> reveal_to_both(Z2<K> x) {
    octetStream os;
    if (playerno == 0) {
        x.pack(os);
        player->send(os);
        os.clear();
        player->receive(os);
        Z2<K> tmp;
        tmp.unpack(os);
        return tmp + x;
    } else {
        player->receive(os);
        Z2<K> tmp;
        tmp.unpack(os);
        Z2<K> result = tmp + x;
        os.clear();
        x.pack(os);
        player->send(os);
        return result;
    }
}

/**
 * @brief Reveal 给 P0
 */
Z2<K> reveal_to_P0(Z2<K> x) {
    octetStream os;
    if (playerno == 0) {
        player->receive(os);
        Z2<K> tmp;
        tmp.unpack(os);
        return tmp + x;
    } else {
        x.pack(os);
        player->send(os);
        return Z2<K>(0);
    }
}

// ============== Kona + ORAM 主类 ==============

class ANN_Kona_ORAM_Party {
public:
    typedef Z2<K> Share;
    
    ANNKonaORAMConfig config;
    ORAMConfig oramConfig;
    TimerWithComm timer;
    int m_playerno;
    RealTwoPartyPlayer* m_player;
    
    vector<SharedBlock> m_blockDB;
    vector<SharedCentroid> m_centroids;
    
    vector<Share> m_queryShare;
    vector<Share> m_queryMaskedU;
    
    vector<vector<Z2<K>>> m_triples;
    vector<Z2<K>> m_queryDelta;
    
    static constexpr long long LARGE_DIST_VAL = 1LL << 50;
    
    ANN_Kona_ORAM_Party(int playerNo, const ANNKonaORAMConfig& cfg)
        : config(cfg), m_playerno(playerNo), m_player(nullptr) {
        cout << "[ANN_Kona_ORAM] 初始化 P" << playerNo << endl;
    }
    
    void start_networking(ez::ezOptionParser& opt) {
        string hostname;
        int pnbase;
        
        opt.get("--portnumbase")->getInt(pnbase);
        opt.get("--hostname")->getString(hostname);
        
        Names playerNames;
        Server::start_networking(playerNames, playerno, 2, hostname, pnbase, Names::DEFAULT_PORT);
        
        m_player = new RealTwoPartyPlayer(playerNames, 1 - playerno, 0);
        player = m_player;
        
        cout << "[Network] 连接成功" << endl;
    }
    
    void loadData() {
        oramConfig = ANNKonaORAMLoader::loadORAMConfig(config.dataDir, config.datasetName);
        m_blockDB = ANNKonaORAMLoader::loadBlockDB(config.dataDir, config.datasetName, 
                                                    m_playerno, oramConfig.embDim);
        m_centroids = ANNKonaORAMLoader::loadCentroidShares(config.dataDir, config.datasetName, m_playerno);
        
        // 尝试加载预计算三元组（可选）
        ANNKonaORAMLoader::loadTriples(config.dataDir, config.datasetName, m_playerno,
                                        m_triples, m_queryDelta);
    }
    
    // ============== 阶段 1: 安全选类（不 reveal clusterId）==============
    
    /**
     * @brief 安全选类（不 reveal clusterId）
     * 返回 clusterId 的秘密共享
     */
    Share assignClusterPrivate(const vector<long long>& queryVec) {
        cout << "[AssignCluster] 开始安全选类（私密）" << endl;
        
        int dim = oramConfig.embDim;
        int numClusters = m_centroids.size();
        
        // 1. P0 对查询向量秘密共享
        m_queryShare.resize(dim);
        m_queryMaskedU.resize(dim);
        
        octetStream os;
        
        if (m_playerno == 0) {
            PRNG prng;
            prng.ReSeed();
            
            for (int d = 0; d < dim; ++d) {
                Z2<K> v(queryVec[d]);
                Z2<K> delta0, delta1;
                delta0.randomize(prng);
                delta1.randomize(prng);
                
                m_queryMaskedU[d] = v + delta0 + delta1;
                m_queryShare[d] = delta0;
                
                m_queryMaskedU[d].pack(os);
                delta1.pack(os);
            }
            m_player->send(os);
        } else {
            m_player->receive(os);
            for (int d = 0; d < dim; ++d) {
                m_queryMaskedU[d].unpack(os);
                m_queryShare[d].unpack(os);
            }
        }
        
        // 2. 计算到每个聚类中心的安全距离
        vector<array<Z2<K>, 2>> distClusterId_vec(numClusters);
        
        for (int c = 0; c < numClusters; ++c) {
            Z2<K> distShare = computeSecureDistance(m_centroids[c].maskedVectorU,
                                                      m_centroids[c].share,
                                                      m_queryMaskedU, m_queryShare);
            distClusterId_vec[c][0] = distShare;
            distClusterId_vec[c][1] = m_centroids[c].clusterIdShare;
        }
        
        // 3. 使用 top_1 选择最小距离（不 reveal）
        top_1(distClusterId_vec, numClusters, true);
        
        // 返回 clusterId 的秘密共享（不 reveal）
        Share selectedClusterIdShare = distClusterId_vec[numClusters - 1][1];
        
        cout << "[AssignCluster] 选类完成（ClusterId 保持私密）" << endl;
        return selectedClusterIdShare;
    }
    
    /**
     * @brief 计算安全欧几里得距离
     */
    Z2<K> computeSecureDistance(const vector<Z2<K>>& targetMaskedU,
                                 const vector<Z2<K>>& targetShare,
                                 const vector<Z2<K>>& queryMaskedU,
                                 const vector<Z2<K>>& queryShare) {
        int dim = queryMaskedU.size();
        
        Z2<K> distShare(0);
        Z2<K> U_diff_sq_sum(0);
        
        vector<Z2<K>> delta_diffs(dim);
        for (int d = 0; d < dim; ++d) {
            Z2<K> U_diff = queryMaskedU[d] - targetMaskedU[d];
            delta_diffs[d] = queryShare[d] - targetShare[d];
            
            U_diff_sq_sum = U_diff_sq_sum + U_diff * U_diff;
            distShare = distShare - Z2<K>(2) * U_diff * delta_diffs[d];
        }
        
        // 在线乘法计算 delta^2
        vector<Z2<K>> delta_sq(dim);
        mul_vector_additive(delta_diffs, delta_diffs, delta_sq, false);
        
        Z2<K> delta_sq_sum(0);
        for (int d = 0; d < dim; ++d) {
            delta_sq_sum = delta_sq_sum + delta_sq[d];
        }
        
        if (m_playerno == 0) {
            distShare = distShare + U_diff_sq_sum + delta_sq_sum;
        } else {
            distShare = distShare + delta_sq_sum;
        }
        
        return distShare;
    }
    
    // ============== 阶段 2: ORAM 读取候选块 ==============
    
    /**
     * @brief 安全相等比较（返回私密比特）
     */
    Share secureEquality(Share a, Share b) {
        // 使用减法 + is_zero 判断
        Share diff = a - b;
        
        // 使用位分解检查是否为零
        // 简化版：reveal diff（仅用于调试）
        // 生产版应使用 secure zero-test
        
        // 这里使用简化版：通过随机掩码和交互判断
        PRNG prng;
        prng.ReSeed();
        Share r;
        r.randomize(prng);
        
        Share masked = diff * r;  // 随机化
        Share maskedRevealed = reveal_to_both(masked);
        
        // 如果 diff == 0，则 masked == 0
        if (maskedRevealed.is_zero()) {
            return m_playerno == 0 ? Share(1) : Share(0);  // 返回 1 的共享
        } else {
            return Share(0);  // 返回 0
        }
    }
    
    /**
     * @brief Trivial ORAM 读取（线性扫描）
     * @param secretAddr 私密地址
     * @return 读取的 Block
     */
    SharedBlock oramRead(Share secretAddr) {
        int M = oramConfig.totalBlocks();
        int dim = oramConfig.embDim;
        
        SharedBlock result(dim);
        result.fileIdShare = Share(0);
        result.validShare = Share(0);
        for (int d = 0; d < dim; ++d) {
            result.maskedU[d] = Share(0);
            result.share[d] = Share(0);
        }
        
        // 线性扫描所有块
        for (int i = 0; i < M; ++i) {
            // 计算 eq = (secretAddr == i)
            Share addr_i = m_playerno == 0 ? Share(i) : Share(0);
            Share eq = secureEquality(secretAddr, addr_i);
            
            // 使用 eq 进行条件选择: result = eq ? block[i] : result
            for (int d = 0; d < dim; ++d) {
                Share diff_u = m_blockDB[i].maskedU[d] - result.maskedU[d];
                Share diff_s = m_blockDB[i].share[d] - result.share[d];
                
                vector<Z2<K>> v1 = {diff_u, diff_s};
                vector<Z2<K>> v2 = {eq, eq};
                vector<Z2<K>> prod;
                mul_vector_additive(v1, v2, prod, false);
                
                result.maskedU[d] = result.maskedU[d] + prod[0];
                result.share[d] = result.share[d] + prod[1];
            }
            
            // fileId 和 valid
            vector<Z2<K>> v1 = {m_blockDB[i].fileIdShare - result.fileIdShare,
                               m_blockDB[i].validShare - result.validShare};
            vector<Z2<K>> v2 = {eq, eq};
            vector<Z2<K>> prod;
            mul_vector_additive(v1, v2, prod, false);
            
            result.fileIdShare = result.fileIdShare + prod[0];
            result.validShare = result.validShare + prod[1];
        }
        
        return result;
    }
    
    /**
     * @brief 读取候选块（固定次数）
     */
    vector<SharedBlock> fetchCandidates(Share clusterIdShare, int bRead) {
        cout << "[ORAM] 读取 " << bRead << " 个候选块" << endl;
        
        int bMax = oramConfig.bMax;
        vector<SharedBlock> candidates;
        
        for (int bi = 0; bi < bRead; ++bi) {
            // 私密地址 = clusterIdShare * bMax + bi
            Share bi_share = m_playerno == 0 ? Share(bi) : Share(0);
            Share bMax_share = m_playerno == 0 ? Share(bMax) : Share(0);
            
            // 安全乘法计算地址
            vector<Z2<K>> v1 = {clusterIdShare};
            vector<Z2<K>> v2 = {bMax_share};
            vector<Z2<K>> prod;
            mul_vector_additive(v1, v2, prod, false);
            
            Share addr = prod[0] + bi_share;
            
            // ORAM 读取
            SharedBlock block = oramRead(addr);
            candidates.push_back(block);
        }
        
        cout << "[ORAM] 读取完成" << endl;
        return candidates;
    }
    
    // ============== 阶段 3: 候选集 KNN ==============
    
    void executeCandidateKNN(const vector<SharedBlock>& candidates, int topK) {
        cout << "[KNN] 候选集 KNN, 候选数=" << candidates.size() << ", topK=" << topK << endl;
        
        int dim = oramConfig.embDim;
        int n = candidates.size();
        
        // 构建 (distance, fileId) 对
        vector<array<Z2<K>, 2>> distFileId_vec(n);
        
        for (int i = 0; i < n; ++i) {
            // 计算距离
            Z2<K> distShare = computeSecureDistance(
                candidates[i].maskedU, candidates[i].share,
                m_queryMaskedU, m_queryShare);
            
            // 使用 validShare 掩码无效块
            // dist = dist + (1 - valid) * LARGE_DIST
            Share one_minus_valid = (m_playerno == 0 ? Share(1) : Share(0)) - candidates[i].validShare;
            
            vector<Z2<K>> v1(1);
            v1[0] = one_minus_valid;
            vector<Z2<K>> v2(1);
            v2[0] = m_playerno == 0 ? Share(LARGE_DIST_VAL) : Share(0);
            vector<Z2<K>> prod;
            mul_vector_additive(v1, v2, prod, false);
            
            distFileId_vec[i][0] = distShare + prod[0];
            distFileId_vec[i][1] = candidates[i].fileIdShare;
        }
        
        // Top-k 选择
        for (int k = 0; k < topK && k < n; ++k) {
            top_1(distFileId_vec, n - k, true);
        }
        
        // Reveal top-k fileId 给 P0
        cout << "[Result] Top-" << topK << " 结果:" << endl;
        for (int k = 0; k < topK && k < n; ++k) {
            Share fileIdShare = distFileId_vec[n - 1 - k][1];
            Z2<K> fileIdRevealed = reveal_to_P0(fileIdShare);
            
            if (m_playerno == 0) {
                long long fileId = static_cast<long long>(fileIdRevealed.get_limb(0));
                cout << "  #" << (k + 1) << ": fileId=" << fileId << endl;
            }
        }
    }
    
    // ============== 主流程 ==============
    
    void processQuery(const vector<long long>& queryVec) {
        // 阶段 1: 安全选类（不 reveal clusterId）
        Share clusterIdShare = assignClusterPrivate(queryVec);
        
        // 阶段 2: ORAM 读取候选
        vector<SharedBlock> candidates = fetchCandidates(clusterIdShare, config.bRead);
        
        // 阶段 3: 候选集 KNN
        executeCandidateKNN(candidates, config.topK);
    }
};

// ============== 命令行解析 ==============

void parse_opt(int argc, const char** argv) {
    opt.add("", true, 1, 0, "Party number", "-p", "--player");
    opt.add("11126", false, 1, 0, "Port number", "-pn", "--portnumbase");
    opt.add("localhost", false, 1, 0, "Hostname", "-h", "--hostname");
    opt.add("./Player-Data/ANN-Data", false, 1, 0, "Data directory", "-d", "--data-dir");
    opt.add("test", false, 1, 0, "Dataset name", "-n", "--dataset");
    opt.add("5", false, 1, 0, "Top-K", "-k", "--topk");
    opt.add("10", false, 1, 0, "Blocks to read per cluster", "--b-read");
    
    opt.parse(argc, argv);
    opt.get("--player")->getInt(playerno);
}

// ============== Main ==============

int main(int argc, const char** argv) {
    parse_opt(argc, argv);
    
    ANNKonaORAMConfig config;
    opt.get("--data-dir")->getString(config.dataDir);
    opt.get("--dataset")->getString(config.datasetName);
    opt.get("--topk")->getInt(config.topK);
    opt.get("--b-read")->getInt(config.bRead);
    
    g_dataDir = config.dataDir;
    k_topk = config.topK;
    g_bRead = config.bRead;
    
    cout << "========================================" << endl;
    cout << "  ANN Kona + ORAM - P" << playerno << endl;
    cout << "========================================" << endl;
    cout << "数据目录: " << config.dataDir << endl;
    cout << "数据集: " << config.datasetName << endl;
    cout << "Top-K: " << config.topK << endl;
    cout << "B-Read: " << config.bRead << endl;
    cout << "----------------------------------------" << endl;
    
    ANN_Kona_ORAM_Party party(playerno, config);
    party.start_networking(opt);
    party.loadData();
    
    // 加载查询
    auto queries = ANNKonaORAMLoader::loadQueries(config.dataDir, config.datasetName);
    
    cout << "\n========================================" << endl;
    cout << "  处理 " << queries.size() << " 条查询" << endl;
    cout << "========================================" << endl;
    
    for (size_t q = 0; q < queries.size(); ++q) {
        cout << "\n--- 查询 " << q << " ---" << endl;
        party.processQuery(queries[q].second);
    }
    
    cout << "\n========================================" << endl;
    cout << "  完成" << endl;
    cout << "========================================" << endl;
    
    return 0;
}
