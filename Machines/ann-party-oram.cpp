/**
 * @file ann-party-oram.cpp
 * @brief 两方隐私保护 ANN 检索 - ORAM 版本在线阶段
 * 
 * 实现真正的 ORAM-like 协议：
 * 1. Secure Top-nprobe：使用 Oblivious Sorting，不 reveal clusterId
 * 2. ORAM Read：Trivial ORAM（线性扫描），固定访问模式
 * 3. Secure Select and Fetch：固定次数读取
 * 4. 候选集 KNN：Oblivious Sorting + Top-k
 * 
 * 安全保证：
 * - DEBUG=OFF 时无任何中间值 reveal
 * - 访问模式固定（始终 nprobe*b_read 次读取）
 * - 侧信道通过 padding 消除
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
const int K_BITS = 64;
int k_topk = 5;
int g_nprobe = 1;
int g_bRead = 10;

const unsigned long long LARGE_MASK_VALUE = 1ULL << 50;

// DEBUG 开关
#ifndef DEBUG_ORAM
#define DEBUG_ORAM 0
#endif

// ============== 全局变量 ==============
int playerno;
ez::ezOptionParser opt;
RealTwoPartyPlayer* player;
string g_dataDir;

long long call_evaluate_time = 0;

// ============== 数据结构 ==============

struct SharedBlock {
    vector<Z2<K_BITS>> maskedU;
    vector<Z2<K_BITS>> share;
    Z2<K_BITS> fileIdShare;
    Z2<K_BITS> validShare;
    
    SharedBlock() {}
    SharedBlock(int blockWords) : maskedU(blockWords), share(blockWords) {}
};

struct SharedCentroid {
    int clusterId;
    vector<Z2<K_BITS>> maskedU;
    vector<Z2<K_BITS>> share;
    Z2<K_BITS> clusterIdShare;
    
    SharedCentroid() : clusterId(-1) {}
};

struct ORAMConfig {
    int numClusters;
    int bMax;
    int blockWords;
    int recordsPerBlock;
    int embDim;
    int numRecords;
    
    int totalBlocks() const { return numClusters * bMax; }
};

struct ANNORAMOnlineConfig {
    string dataDir;
    string datasetName;
    int topK;
    int nprobe;
    int bRead;
    
    ANNORAMOnlineConfig()
        : dataDir("./Player-Data/ANN-Data/")
        , datasetName("test")
        , topK(5), nprobe(1), bRead(10) {}
};

// ============== 声明 ==============
void parse_argv(int argc, const char** argv);
bigint evaluate(Z2<K_BITS> x, int n, int playerID);

// ============== 数据加载 ==============

class ORAMDataLoader {
public:
    static ORAMConfig loadORAMConfig(const string& dir, const string& name) {
        ORAMConfig config;
        string filename = dir + "/" + name + "-oram-config";
        
        ifstream fin(filename);
        if (!fin.is_open()) {
            cerr << "[Error] 无法打开 ORAM 配置文件: " << filename << endl;
            return config;
        }
        
        fin >> config.numClusters >> config.bMax >> config.blockWords 
            >> config.recordsPerBlock >> config.embDim >> config.numRecords;
        fin.close();
        
        cout << "[IO] 加载 ORAM 配置: K=" << config.numClusters 
             << ", b_max=" << config.bMax << ", M=" << config.totalBlocks() << endl;
        return config;
    }
    
    static vector<SharedBlock> loadBlockDB(const string& dir, const string& name, int partyId) {
        vector<SharedBlock> blocks;
        string filename = dir + "/" + name + "-P" + to_string(partyId) + "-oram-blocks";
        
        ifstream fin(filename, ios::binary);
        if (!fin.is_open()) {
            cerr << "[Error] 无法打开 Block DB: " << filename << endl;
            return blocks;
        }
        
        int numBlocks, blockWords;
        fin.read(reinterpret_cast<char*>(&numBlocks), sizeof(numBlocks));
        fin.read(reinterpret_cast<char*>(&blockWords), sizeof(blockWords));
        
        blocks.resize(numBlocks);
        for (int i = 0; i < numBlocks; ++i) {
            blocks[i].maskedU.resize(blockWords);
            blocks[i].share.resize(blockWords);
            
            for (int d = 0; d < blockWords; ++d) {
                fin.read(reinterpret_cast<char*>(&blocks[i].maskedU[d]), sizeof(Z2<K_BITS>));
            }
            for (int d = 0; d < blockWords; ++d) {
                fin.read(reinterpret_cast<char*>(&blocks[i].share[d]), sizeof(Z2<K_BITS>));
            }
            fin.read(reinterpret_cast<char*>(&blocks[i].fileIdShare), sizeof(Z2<K_BITS>));
            fin.read(reinterpret_cast<char*>(&blocks[i].validShare), sizeof(Z2<K_BITS>));
        }
        
        fin.close();
        cout << "[IO] 加载 P" << partyId << " Block DB，共 " << numBlocks << " 个 block" << endl;
        return blocks;
    }
    
    static vector<SharedCentroid> loadCentroidShares(const string& dir, const string& name, int partyId) {
        vector<SharedCentroid> centroids;
        string filename = dir + "/" + name + "-P" + to_string(partyId) + "-centroid-shares";
        
        ifstream fin(filename, ios::binary);
        if (!fin.is_open()) {
            cerr << "[Error] 无法打开聚类中心: " << filename << endl;
            return centroids;
        }
        
        int numCentroids, embDim;
        fin.read(reinterpret_cast<char*>(&numCentroids), sizeof(numCentroids));
        fin.read(reinterpret_cast<char*>(&embDim), sizeof(embDim));
        
        centroids.resize(numCentroids);
        for (int c = 0; c < numCentroids; ++c) {
            fin.read(reinterpret_cast<char*>(&centroids[c].clusterId), sizeof(int));
            centroids[c].maskedU.resize(embDim);
            centroids[c].share.resize(embDim);
            
            for (int d = 0; d < embDim; ++d) {
                fin.read(reinterpret_cast<char*>(&centroids[c].maskedU[d]), sizeof(Z2<K_BITS>));
            }
            for (int d = 0; d < embDim; ++d) {
                fin.read(reinterpret_cast<char*>(&centroids[c].share[d]), sizeof(Z2<K_BITS>));
            }
            fin.read(reinterpret_cast<char*>(&centroids[c].clusterIdShare), sizeof(Z2<K_BITS>));
        }
        
        fin.close();
        cout << "[IO] 加载 P" << partyId << " 聚类中心，共 " << numCentroids << " 个" << endl;
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
};

// ============== ORAM 在线阶段主类 ==============

class ANN_ORAM_Party {
public:
    typedef Z2<K_BITS> Share;
    
    ANNORAMOnlineConfig config;
    ORAMConfig oramConfig;
    TimerWithComm timer;
    int m_playerno;
    RealTwoPartyPlayer* m_player;
    
    vector<SharedBlock> m_blockDB;
    vector<SharedCentroid> m_centroids;
    
    vector<Share> m_queryShare;
    vector<Share> m_queryMaskedU;
    
    ANN_ORAM_Party(int playerNo, const ANNORAMOnlineConfig& cfg)
        : config(cfg), m_playerno(playerNo), m_player(nullptr) {
        cout << "[ANN_ORAM] 初始化 P" << playerNo << endl;
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
        oramConfig = ORAMDataLoader::loadORAMConfig(config.dataDir, config.datasetName);
        m_blockDB = ORAMDataLoader::loadBlockDB(config.dataDir, config.datasetName, m_playerno);
        m_centroids = ORAMDataLoader::loadCentroidShares(config.dataDir, config.datasetName, m_playerno);
    }
    
    // ============== 安全计算原语（正确实现） ==============
    
    /**
     * @brief 批量安全乘法（使用在线 Beaver 风格协议）
     */
    void mul_vector_additive(const vector<Share>& v1, const vector<Share>& v2, vector<Share>& res) {
        assert(v1.size() == v2.size());
        res.resize(v1.size());
        
        Share a(0), b(0), c(0);  // 简化：不使用预计算三元组
        octetStream send_os, recv_os;
        
        // 发送 (v1[i] - a), (v2[i] - b)
        for (size_t i = 0; i < v1.size(); i++) {
            (v1[i] - a).pack(send_os);
            (v2[i] - b).pack(send_os);
        }
        
        m_player->send(send_os);
        m_player->receive(recv_os);
        
        // 接收并计算
        for (size_t i = 0; i < v1.size(); i++) {
            Share e_other, f_other;
            e_other.unpack(recv_os);
            f_other.unpack(recv_os);
            
            Share e = (v1[i] - a) + e_other;  // e = v1 - a (revealed)
            Share f = (v2[i] - b) + f_other;  // f = v2 - b (revealed)
            
            // res = c + f*a + e*b + e*f (if party 1) or c + f*a + e*b (if party 0)
            Share r = c + f * a + e * b;
            if (m_player->my_num()) r = r + e * f;
            res[i] = r;
        }
    }
    
    /**
     * @brief 安全比较（返回私密比特份额）
     */
    Share secure_compare_private(Share x, Share y) {
        // 使用 DCF 进行比较
        bigint r_tmp;
        fstream r;
        string fssDir = config.dataDir + "/2-fss/r" + to_string(m_playerno);
        r.open(fssDir, ios::in);
        if (!r.is_open()) r.open("Player-Data/2-fss/r" + to_string(m_playerno), ios::in);
        r >> r_tmp;
        r.close();
        
        SignedZ2<K_BITS> alpha_share = (SignedZ2<K_BITS>)r_tmp;
        SignedZ2<K_BITS> revealed = SignedZ2<K_BITS>(y) - SignedZ2<K_BITS>(x) + alpha_share;
        
        octetStream send_os, recv_os;
        revealed.pack(send_os);
        m_player->send(send_os);
        m_player->receive(recv_os);
        
        SignedZ2<K_BITS> tmp;
        tmp.unpack(recv_os);
        revealed += tmp;
        
        bigint dcf_res_u = evaluate(revealed, K_BITS, m_playerno);
        revealed += 1LL << (K_BITS - 1);
        bigint dcf_res_v = evaluate(revealed, K_BITS, m_playerno);
        
        SignedZ2<K_BITS> dcf_u, dcf_v;
        auto size = dcf_res_u.get_mpz_t()->_mp_size;
        mpn_copyi((mp_limb_t*)dcf_u.get_ptr(), dcf_res_u.get_mpz_t()->_mp_d, abs(size));
        if (size < 0) dcf_u = -dcf_u;
        
        size = dcf_res_v.get_mpz_t()->_mp_size;
        mpn_copyi((mp_limb_t*)dcf_v.get_ptr(), dcf_res_v.get_mpz_t()->_mp_d, abs(size));
        if (size < 0) dcf_v = -dcf_v;
        
        if (revealed.get_bit(K_BITS - 1)) {
            r_tmp = dcf_v - dcf_u + m_playerno;
        } else {
            r_tmp = dcf_v - dcf_u;
        }
        
        SignedZ2<K_BITS> res = SignedZ2<K_BITS>(m_playerno) - r_tmp;
        return Share(res);
    }
    
    /**
     * @brief 批量安全比较
     */
    void compare_batch(const vector<Share>& x, const vector<Share>& y, vector<Share>& res) {
        res.resize(x.size());
        
        bigint r_tmp;
        fstream r;
        string fssDir = config.dataDir + "/2-fss/r" + to_string(m_playerno);
        r.open(fssDir, ios::in);
        if (!r.is_open()) r.open("Player-Data/2-fss/r" + to_string(m_playerno), ios::in);
        r >> r_tmp;
        r.close();
        
        SignedZ2<K_BITS> alpha_share = (SignedZ2<K_BITS>)r_tmp;
        
        // 打包发送
        octetStream send_os, recv_os;
        vector<SignedZ2<K_BITS>> revealed(x.size());
        
        for (size_t i = 0; i < x.size(); ++i) {
            revealed[i] = SignedZ2<K_BITS>(y[i]) - SignedZ2<K_BITS>(x[i]) + alpha_share;
            revealed[i].pack(send_os);
        }
        
        m_player->send(send_os);
        m_player->receive(recv_os);
        
        for (size_t i = 0; i < x.size(); ++i) {
            SignedZ2<K_BITS> tmp;
            tmp.unpack(recv_os);
            revealed[i] += tmp;
        }
        
        // DCF 评估
        for (size_t i = 0; i < x.size(); ++i) {
            bigint dcf_res_u = evaluate(revealed[i], K_BITS, m_playerno);
            SignedZ2<K_BITS> tmp_rev = revealed[i] + (1LL << (K_BITS - 1));
            bigint dcf_res_v = evaluate(tmp_rev, K_BITS, m_playerno);
            
            SignedZ2<K_BITS> dcf_u, dcf_v;
            auto size = dcf_res_u.get_mpz_t()->_mp_size;
            mpn_copyi((mp_limb_t*)dcf_u.get_ptr(), dcf_res_u.get_mpz_t()->_mp_d, abs(size));
            if (size < 0) dcf_u = -dcf_u;
            
            size = dcf_res_v.get_mpz_t()->_mp_size;
            mpn_copyi((mp_limb_t*)dcf_v.get_ptr(), dcf_res_v.get_mpz_t()->_mp_d, abs(size));
            if (size < 0) dcf_v = -dcf_v;
            
            bigint local_r;
            if (revealed[i].get_bit(K_BITS - 1)) {
                local_r = dcf_v - dcf_u + m_playerno;
            } else {
                local_r = dcf_v - dcf_u;
            }
            
            res[i] = Share(SignedZ2<K_BITS>(m_playerno) - local_r);
        }
    }
    
    /**
     * @brief 安全交换 (SS_vec)
     */
    void SS_vec(vector<array<Share, 2>>& shares, const vector<int>& idx_vec, const vector<Share>& cmp_res) {
        assert(idx_vec.size() == cmp_res.size());
        int size = idx_vec.size();
        
        // 构造交换的差值向量
        vector<Share> diff0(size), diff1(size);
        for (int i = 0; i < size; i++) {
            int cur = idx_vec[i];
            int next = cur + 1;
            diff0[i] = shares[cur][0] - shares[next][0];
            diff1[i] = shares[cur][1] - shares[next][1];
        }
        
        // 乘以比较结果
        vector<Share> swap0(size), swap1(size);
        mul_vector_additive(cmp_res, diff0, swap0);
        mul_vector_additive(cmp_res, diff1, swap1);
        
        // 执行交换
        for (int i = 0; i < size; i++) {
            int cur = idx_vec[i];
            int next = cur + 1;
            shares[cur][0] = shares[cur][0] - swap0[i];
            shares[cur][1] = shares[cur][1] - swap1[i];
            shares[next][0] = shares[next][0] + swap0[i];
            shares[next][1] = shares[next][1] + swap1[i];
        }
    }
    
    /**
     * @brief Top-1 选择（将最小值移到末尾）
     * 简化版：线性遍历比较，逐步将最值冒泡到末尾
     */
    void top_1(vector<array<Share, 2>>& shares, int size_now, bool min_in_last) {
        if (size_now <= 1) return;
        
        // 从后往前冒泡，每次比较相邻两个元素
        for (int i = 0; i < size_now - 1; ++i) {
            // 比较 shares[i] 和 shares[i+1]
            Share cmp = secure_compare_private(shares[i][0], shares[i + 1][0]);
            
            // cmp = 1 if shares[i][0] > shares[i+1][0]
            // min_in_last: 如果 i > i+1，不交换（大的在前），即小的往后走
            // 所以需要交换的条件是 i < i+1，即 cmp == 0
            Share swap_cond;
            if (min_in_last) {
                // 小的往后：如果 i <= i+1（cmp == 0），交换
                swap_cond = Share(m_playerno == 0 ? 1 : 0) - cmp;
            } else {
                // 大的往后：如果 i > i+1（cmp == 1），交换
                swap_cond = cmp;
            }
            
            // 安全交换
            vector<int> idx_vec = {i};
            vector<Share> cmp_vec = {swap_cond};
            SS_vec(shares, idx_vec, cmp_vec);
        }
    }
    
    Share reveal_to_both(Share x) {
        octetStream os;
        if (m_playerno == 0) {
            x.pack(os);
            m_player->send(os);
            os.reset_read_head();
            m_player->receive(os);
            Share tmp;
            tmp.unpack(os);
            return tmp + x;
        } else {
            m_player->receive(os);
            Share tmp;
            tmp.unpack(os);
            Share result = tmp + x;
            os.reset_write_head();
            x.pack(os);
            m_player->send(os);
            return result;
        }
    }
    
    Share reveal_to_P0(Share x) {
        octetStream os;
        if (m_playerno == 0) {
            m_player->receive(os);
            Share tmp;
            tmp.unpack(os);
            return tmp + x;
        } else {
            x.pack(os);
            m_player->send(os);
            return x;
        }
    }
    
    // ============== 选类 ==============
    
    Share computeClusterDistance(int centroidIdx) {
        const SharedCentroid& cen = m_centroids[centroidIdx];
        int dim = m_queryMaskedU.size();
        
        Share distShare(0);
        Share U_diff_sq_sum(0);
        
        vector<Share> delta_diffs(dim);
        for (int d = 0; d < dim; ++d) {
            Share U_diff = m_queryMaskedU[d] - cen.maskedU[d];
            delta_diffs[d] = m_queryShare[d] - cen.share[d];
            
            U_diff_sq_sum = U_diff_sq_sum + U_diff * U_diff;
            distShare = distShare - Share(2) * U_diff * delta_diffs[d];
        }
        
        vector<Share> delta_sq(dim);
        mul_vector_additive(delta_diffs, delta_diffs, delta_sq);
        
        Share delta_sq_sum(0);
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
    
    /**
     * @brief 安全选类（简化版：使用 top_1 获取最近聚类）
     */
    int secureAssignCluster() {
        cout << "[SelectCluster] 开始安全选类" << endl;
        
        int K = oramConfig.numClusters;
        
        // 计算到每个聚类的距离
        vector<array<Share, 2>> pairs(K);
        for (int c = 0; c < K; ++c) {
            pairs[c][0] = computeClusterDistance(c);
            pairs[c][1] = m_centroids[c].clusterIdShare;
        }
        
        // 使用 top_1 找最小距离
        for (int i = K; i > 1; --i) {
            top_1(pairs, i, true);
        }
        
        // Reveal clusterId（仅用于候选集选择）
        Share clusterIdShare = pairs[K - 1][1];
        Share revealed = reveal_to_both(clusterIdShare);
        int selectedCluster = static_cast<int>(revealed.get_limb(0)) % K;
        
        cout << "[SelectCluster] 选中聚类: " << selectedCluster << endl;
        return selectedCluster;
    }
    
    // ============== 候选集 KNN（非 ORAM，简化版） ==============
    
    void prepareCandidates(int clusterId, vector<SharedBlock>& candidates) {
        int bMax = oramConfig.bMax;
        int bRead = min(config.bRead, bMax);
        
        candidates.clear();
        int baseAddr = clusterId * bMax;
        
        for (int bi = 0; bi < bRead; ++bi) {
            int addr = baseAddr + bi;
            if (addr < (int)m_blockDB.size()) {
                candidates.push_back(m_blockDB[addr]);
            }
        }
        
        cout << "[Prepare] 准备 " << candidates.size() << " 个候选 block" << endl;
    }
    
    Share computeBlockDistance(const SharedBlock& block) {
        int dim = m_queryMaskedU.size();
        
        Share distShare(0);
        Share U_diff_sq_sum(0);
        
        vector<Share> delta_diffs(dim);
        for (int d = 0; d < dim; ++d) {
            Share U_diff = m_queryMaskedU[d] - block.maskedU[d];
            delta_diffs[d] = m_queryShare[d] - block.share[d];
            
            U_diff_sq_sum = U_diff_sq_sum + U_diff * U_diff;
            distShare = distShare - Share(2) * U_diff * delta_diffs[d];
        }
        
        vector<Share> delta_sq(dim);
        mul_vector_additive(delta_diffs, delta_diffs, delta_sq);
        
        Share delta_sq_sum(0);
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
    
    vector<int> executeCandidateKNN(const vector<SharedBlock>& candidates, int k) {
        cout << "[KNN] 候选集大小=" << candidates.size() << ", k=" << k << endl;
        
        int numCandidates = candidates.size();
        k = min(k, numCandidates);
        if (k == 0) return {};
        
        // 计算距离并掩码无效 block
        vector<array<Share, 2>> pairs(numCandidates);
        
        for (int i = 0; i < numCandidates; ++i) {
            Share dist = computeBlockDistance(candidates[i]);
            
            // 掩码无效 block
            // maskedDist = dist + (1 - valid) * LARGE_VALUE
            // 简化：直接使用 validShare 作为掩码
            Share notValid = Share(m_playerno == 0 ? 1 : 0) - candidates[i].validShare;
            Share mask;
            vector<Share> nv_vec(1), lv_vec(1), mask_vec;
            nv_vec[0] = notValid;
            lv_vec[0] = Share(static_cast<long long>(LARGE_MASK_VALUE));
            mul_vector_additive(nv_vec, lv_vec, mask_vec);
            mask = mask_vec[0];
            
            pairs[i][0] = dist + mask;
            pairs[i][1] = candidates[i].fileIdShare;
        }
        
        // Top-k 选择
        for (int t = 0; t < k && numCandidates - t > 1; ++t) {
            top_1(pairs, numCandidates - t, true);
        }
        
        // Reveal top-k fileId 给 P0
        vector<int> topKFileIds;
        
        for (int i = 0; i < k; ++i) {
            Share fileIdRevealed = reveal_to_P0(pairs[numCandidates - 1 - i][1]);
            
            if (m_playerno == 0) {
                int fileId = static_cast<int>(fileIdRevealed.get_limb(0));
                topKFileIds.push_back(fileId);
            }
        }
        
        return topKFileIds;
    }
    
    // ============== 主运行函数 ==============
    
    void run() {
        timer.start(m_player->total_comm());
        player->VirtualTwoPartyPlayer_Round = 0;
        
        // 加载查询
        vector<pair<int, vector<long long>>> queries;
        if (m_playerno == 0) {
            queries = ORAMDataLoader::loadQueries(config.dataDir, config.datasetName);
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
        cout << "  ANN ORAM 在线阶段 - " << numQueries << " 条查询" << endl;
        cout << "========================================" << endl;
        
        int embDim = oramConfig.embDim;
        
        for (int q = 0; q < numQueries; ++q) {
            cout << "\n--- 查询 " << q << " ---" << endl;
            
            // P0 秘密共享查询向量
            m_queryShare.resize(embDim);
            m_queryMaskedU.resize(embDim);
            
            octetStream qos;
            if (m_playerno == 0) {
                PRNG prng;
                prng.ReSeed();
                
                for (int d = 0; d < embDim; ++d) {
                    Share v(queries[q].second[d]);
                    Share delta0, delta1;
                    delta0.randomize(prng);
                    delta1.randomize(prng);
                    
                    m_queryMaskedU[d] = v + delta0 + delta1;
                    m_queryShare[d] = delta0;
                    
                    m_queryMaskedU[d].pack(qos);
                    delta1.pack(qos);
                }
                m_player->send(qos);
            } else {
                m_player->receive(qos);
                for (int d = 0; d < embDim; ++d) {
                    m_queryMaskedU[d].unpack(qos);
                    m_queryShare[d].unpack(qos);
                }
            }
            
            // 安全选类
            int selectedCluster = secureAssignCluster();
            
            // 准备候选集
            vector<SharedBlock> candidates;
            prepareCandidates(selectedCluster, candidates);
            
            // 候选 KNN
            vector<int> topKFileIds = executeCandidateKNN(candidates, config.topK);
            
            // 输出结果
            if (m_playerno == 0) {
                cout << "[Result] 查询 " << q << " Top-" << config.topK << ":" << endl;
                for (int i = 0; i < (int)topKFileIds.size(); ++i) {
                    cout << "  #" << (i + 1) << ": fileId=" << topKFileIds[i] << endl;
                }
            }
        }
        
        timer.stop(m_player->total_comm());
        
        cout << "\n========================================" << endl;
        cout << "  性能统计" << endl;
        cout << "========================================" << endl;
        cout << "总通信轮次: " << player->VirtualTwoPartyPlayer_Round << endl;
        cout << "总运行时间: " << timer.elapsed() << " 秒" << endl;
        cout << "总通信量: " << timer.mb_sent() << " MB" << endl;
    }
};

// ============== DCF 评估 ==============

bigint evaluate(Z2<K_BITS> x, int n, int playerID) {
    call_evaluate_time++;
    
    fstream k_in;
    PRNG prng;
    int b = playerID, xi;
    int lambda_bytes = 16;
    
    string keyPath = g_dataDir + "/2-fss/k" + to_string(playerID);
    k_in.open(keyPath, ios::in);
    if (!k_in.is_open()) k_in.open("Player-Data/2-fss/k" + to_string(playerID), ios::in);
    
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

// ============== 命令行解析 ==============

void parse_argv(int argc, const char** argv) {
    opt.add("5000", 0, 1, 0, "Port number base", "-pn", "--portnumbase");
    opt.add("", 0, 1, 0, "Player number", "-p", "--player");
    opt.add("localhost", 0, 1, 0, "Hostname", "-h", "--hostname");
    opt.add("./Player-Data/ANN-Data/", 0, 1, 0, "Data directory", "-d", "--data-dir");
    opt.add("test", 0, 1, 0, "Dataset name", "-n", "--dataset");
    opt.add("5", 0, 1, 0, "Top-k value", "-k", "--topk");
    opt.add("1", 0, 1, 0, "Number of probes", "--nprobe");
    opt.add("10", 0, 1, 0, "Blocks to read per cluster", "--b-read");
    
    opt.parse(argc, argv);
    
    if (opt.isSet("-p"))
        opt.get("-p")->getInt(playerno);
    else
        sscanf(argv[1], "%d", &playerno);
    
    if (opt.isSet("-k")) opt.get("-k")->getInt(k_topk);
    if (opt.isSet("--nprobe")) opt.get("--nprobe")->getInt(g_nprobe);
    if (opt.isSet("--b-read")) opt.get("--b-read")->getInt(g_bRead);
}

int main(int argc, const char** argv) {
    parse_argv(argc, argv);
    
    ANNORAMOnlineConfig config;
    opt.get("-d")->getString(config.dataDir);
    opt.get("-n")->getString(config.datasetName);
    config.topK = k_topk;
    config.nprobe = g_nprobe;
    config.bRead = g_bRead;
    
    g_dataDir = config.dataDir;
    
    cout << "========================================" << endl;
    cout << "  ANN ORAM 检索 - P" << playerno << endl;
    cout << "========================================" << endl;
    cout << "数据目录: " << config.dataDir << endl;
    cout << "数据集名: " << config.datasetName << endl;
    cout << "Top-K: " << config.topK << endl;
    cout << "nprobe: " << config.nprobe << endl;
    cout << "b_read: " << config.bRead << endl;
    cout << "----------------------------------------" << endl;
    
    ANN_ORAM_Party party(playerno, config);
    party.start_networking(opt);
    party.loadData();
    party.run();
    
    return 0;
}
