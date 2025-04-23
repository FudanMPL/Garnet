#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <fstream>
#include <numeric>
#include <omp.h>

using namespace std;

typedef vector<int> Vec;
typedef long long ll;
string dataset_name;
string dir;

int k_left=5,k_right=30;
class Sample {
    public:
        Vec features;
        int label;
        Sample() : label(0) {}
        Sample(int n) : features(n),label(0) {};
        Sample(Vec features_,int label_):features(features_),label(label_){};
};
    
struct Kmeans_Result {
    int num_features, num_train_data, num_test_data;
    vector<Sample> m_sample; // 训练集
    vector<Sample> m_test;   // 测试集

    Kmeans_Result(int features, int train_num, int test_num,
                    const vector<Sample>& samples,
                    const vector<Sample>& tests)
        : num_features(features),
            num_train_data(train_num),
            num_test_data(test_num),
            m_sample(samples),
            m_test(tests) {}

    Kmeans_Result() : num_features(0), num_train_data(0), num_test_data(0) {}
};

class K_means {
    public:
        int num_features;
        int num_train_data;
        int num_test_data;
        int k_means_k ;
    
        vector<Sample> m_sample;
        vector<Sample> m_test;
        vector<int> labels;
        vector<vector<Vec>> newClusters;
    
        vector<vector<int>>labels_clusters;
    
    
        void read_data() {
            string base_path = "Player-Data/Knn-Data/" + dir + dataset_name + "-data/";
    
            ifstream meta_file(base_path + "Knn-meta");
            meta_file >> num_features >> num_train_data >> num_test_data;
            meta_file.close();
    
            m_sample.resize(num_train_data, Sample(num_features));
            m_test.resize(num_test_data, Sample(num_features));
    
            ifstream sample_file(base_path + "P0-0-X-Train");
            for (int i = 0; i < num_train_data; i++)
                for (int j = 0; j < num_features; j++)
                    sample_file >> m_sample[i].features[j];
            sample_file.close();
    
            ifstream label_file(base_path + "P0-0-Y-Train");
            for (int i = 0; i < num_train_data; i++)
                label_file >> m_sample[i].label;
            label_file.close();
    
            ifstream test_file(base_path + "P1-0-X-Test");
            for (int i = 0; i < num_test_data; i++)
                for (int j = 0; j < num_features; j++)
                    test_file >> m_test[i].features[j];
            test_file.close();
    
            ifstream test_label_file(base_path + "P1-0-Y-Test");
            for (int i = 0; i < num_test_data; i++)
                test_label_file >> m_test[i].label;
            test_label_file.close();
        }
    
        ll euclideanSquareDistance(const Vec& a, const Vec& b) {
            ll sum = 0;
            for (size_t i = 0; i < a.size(); ++i)
                sum += (a[i] - b[i]) * (ll)(a[i] - b[i]);
            return sum;
        }
    
        Vec computeCentroid(const vector<Vec>& cluster) {
            size_t dim = cluster[0].size();
            Vec centroid(dim, 0);
            vector<ll> temp_sum(dim, 0);
            for (const auto& point : cluster)
                for (size_t i = 0; i < dim; ++i)
                    temp_sum[i] += point[i];
            for (size_t i = 0; i < dim; ++i)
                centroid[i] = static_cast<int>(temp_sum[i] / cluster.size());
            return centroid;
        }
    
        void kMeansPlusPlusInit(int k, vector<Vec>& centers) {
            centers.push_back(m_sample[rand() % num_train_data].features);
            vector<ll> dist(num_train_data, 0);
    
            for (int c = 1; c < k; ++c) {
                ll total_dist = 0;
    
                #pragma omp parallel for reduction(+:total_dist)
                for (int i = 0; i < num_train_data; ++i) {
                    ll minDist = numeric_limits<ll>::max();
                    for (const auto& center : centers)
                        minDist = min(minDist, euclideanSquareDistance(m_sample[i].features, center));
                    dist[i] = minDist;
                    total_dist += minDist;
                }
    
                ll r = rand() % total_dist, cumulative = 0;
                for (int i = 0; i < num_train_data; ++i) {
                    cumulative += dist[i];
                    if (cumulative >= r) {
                        centers.push_back(m_sample[i].features);
                        break;
                    }
                }
            }
        }
    
        void kMeans(int k, vector<int>& labels, vector<Vec>& centers) {
            labels.resize(num_train_data, 0);
            centers.clear();
            srand(time(0));
    
            kMeansPlusPlusInit(k, centers);
    
            bool changed;
            do {
                changed = false;
    
                #pragma omp parallel for
                for (int i = 0; i < num_train_data; ++i) {
                    ll minDist = numeric_limits<ll>::max();
                    int bestCluster = 0;
                    for (int j = 0; j < k; ++j) {
                        ll dist = euclideanSquareDistance(m_sample[i].features, centers[j]);
                        if (dist < minDist) {
                            minDist = dist;
                            bestCluster = j;
                        }
                    }
                    if (labels[i] != bestCluster) {
                        labels[i] = bestCluster;
                        changed = true;
                    }
                }
    
                newClusters.assign(k, {});
                labels_clusters.assign(k,{});
                for (int i = 0; i < num_train_data; ++i){
                    newClusters[labels[i]].push_back(m_sample[i].features);
                    labels_clusters[labels[i]].push_back(m_sample[i].label);
                }
    
                for (int j = 0; j < k; ++j)
                    if (!newClusters[j].empty())
                        centers[j] = computeCentroid(newClusters[j]);
            } while (changed);
        }
    
        void printVector(const Vec& v) {
            cout << "(";
            for (size_t i = 0; i < v.size(); ++i) {
                cout << v[i];
                if (i < v.size() - 1) cout << ", ";
            }
            cout << ")" << endl;
        }
    
        void print_cluster(int k) {
            cout << "Final clustering result:" << endl;
            for (size_t i = 0; i < k; ++i) {
                cout << "Cluster " << i << ":\n";
                for (const auto& item : newClusters[i])
                    printVector(item);
            }
        }
    
        double calculateSSE(const vector<Vec>& centers, const vector<int>& labels) {
            double sse = 0.0;
            #pragma omp parallel for reduction(+:sse)
            for (int i = 0; i < num_train_data; ++i) {
                sse += sqrt(euclideanSquareDistance(m_sample[i].features, centers[labels[i]]));
            }
            return sse;
        }
    
        int findBestK(int min_k = 2, int max_k = 10) {
            vector<double> sses;
            for (int k = min_k; k <= max_k; ++k) {
                vector<int> tmp_labels;
                vector<Vec> tmp_centers;
                kMeans(k, tmp_labels, tmp_centers);
                double sse = calculateSSE(tmp_centers, tmp_labels);
                cout << "k=" << k << ", SSE=" << sse << endl;
                sses.push_back(sse);
            }
        
            // 用“最大拐点法”计算最佳 k
            int best_k = min_k;
            double max_delta = -1.0;
            for (int i = 1; i < (int)sses.size() - 1; ++i) {
                double delta1 = sses[i - 1] - sses[i];
                double delta2 = sses[i] - sses[i + 1];
                double second_derivative = delta1 - delta2;
        
                if (second_derivative > max_delta) {
                    max_delta = second_derivative;
                    best_k = min_k + i;
                }
            }
        
            cout << "Best k selected by elbow method: " << best_k << endl;
            return best_k;
        }
        
    
        int predictCluster(const Vec& input,vector<Vec>&final_centers) {
            ll minDist = numeric_limits<ll>::max();
            int bestCluster = -1;
            for (int i = 0; i < final_centers.size(); ++i) {
                ll dist = euclideanSquareDistance(input, final_centers[i]);
                if (dist < minDist) {
                    minDist = dist;
                    bestCluster = i;
                }
            }
            return bestCluster;
        }
    
    
        Kmeans_Result run(const string& ds_name) {
            dataset_name = ds_name;
            read_data();
            k_means_k = findBestK(k_left, k_right); // auto-select best k
            vector<Vec> centers(k_means_k);
            kMeans(k_means_k, labels, centers);
            // print_cluster(k_means_k);
            cout<<endl;
            for(int i=0;i<k_means_k;i++){
                cout<<"Cluster_"<<i<<":"<<newClusters[i].size()<<endl;
            }
            int idx=predictCluster(m_test[0].features,centers);
            cout<<"Selected Cluster:"<<idx<<endl;
            vector<Sample> m_sample_selected;
            vector<Sample> m_test_selected;
            int sze=newClusters[idx].size();
            for(int i=0;i<sze;i++){
                m_sample_selected.push_back(Sample(newClusters[idx][i],labels_clusters[idx][i]));
            }
            // cout<<"Kmeans_Result run(const string& ds_name) :"<<m_sample_selected[0].features.size()<<endl;
            Kmeans_Result Res(num_features,sze,num_test_data,m_sample_selected,m_test);
            return Res;
        }
    
    
    };
    


    
    void save_result_to_file(const Kmeans_Result& result, const string& filename) {
        ofstream out(filename);
        if (!out.is_open()) {
            cerr << "Failed to open file for writing: " << filename << endl;
            return;
        }
    
        out << result.num_features << " " << result.num_train_data << " " << result.num_test_data << endl;
    
        // 保存训练集
        for (const auto& sample : result.m_sample) {
            for (int val : sample.features)
                out << val << " ";
            out << sample.label << endl;
        }
    
        // 保存测试集
        for (const auto& sample : result.m_test) {
            for (int val : sample.features)
                out << val << " ";
            out << sample.label << endl;
        }
    
        out.close();
        cout << "Saved Kmeans_Result to " << filename << endl;
    }
    
    Kmeans_Result load_result_from_file(const string& filename) {
        ifstream in(filename);
        if (!in.is_open()) {
            cerr << "Failed to open file for reading: " << filename << endl;
            exit(1);
        }
    
        Kmeans_Result result;
        in >> result.num_features >> result.num_train_data >> result.num_test_data;
    
        // 读取训练集
        for (int i = 0; i < result.num_train_data; ++i) {
            Vec features(result.num_features);
            for (int j = 0; j < result.num_features; ++j) {
                in >> features[j];
            }
            int label;
            in >> label;
            result.m_sample.emplace_back(features, label);
        }
    
        // 读取测试集
        for (int i = 0; i < result.num_test_data; ++i) {
            Vec features(result.num_features);
            for (int j = 0; j < result.num_features; ++j) {
                in >> features[j];
            }
            int label;
            in >> label;
            result.m_test.emplace_back(features, label);
        }
    
        in.close();
        return result;
    }

int main() {
    dir="knn-1/";
    // vector<string>dataset_name_list={"Iris","Wine","Cancer","Spambase","Adult","Mnist","Dota2Games", "Toxicity", "arcene", "RNA-seq", "PEMS-SF"};
    vector<string>dataset_name_list={"Mnist"};
    for(int i=0;i<dataset_name_list.size();i++){
        dataset_name=dataset_name_list[i];
        cout<<"--------DataSet:"<<dataset_name<<"--------------"<<endl;
        K_means kmeans;
        cout << "**Start kMeans**" << endl;
        Kmeans_Result res=kmeans.run(dataset_name);
        save_result_to_file(res,"KmeansRes");
    }
    return 0;
}

