#include <random>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;
 
int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <number_of_files> <numbers_per_file>" << endl;
        return 1;
    }

    int number_of_files = stoi(argv[1]);
    int numbers_per_file = stoi(argv[2]);

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::normal_distribution<double> distribution(0.0, 1.0);

    for (int file_index = 1; file_index <= number_of_files; ++file_index) {
        ostringstream filename;
        filename << "data" << file_index << ".txt";
        
        ofstream f(filename.str());
        if (!f.is_open()) {
            cerr << "Failed to open file " << filename.str() << endl;
            return 1;
        }

        for (int i = 0; i < numbers_per_file; ++i) {
            double noise = distribution(generator);
            f << noise << endl;
        }

        f.close();
    }
    
    ofstream count_file("file_count.txt");
    if (!count_file.is_open()) {
        cerr << "Failed to open file_count.txt" << endl;
        return 1;
    }
    count_file << number_of_files << endl;
    count_file.close();
    
    // fstream f;
	// f.open("data1.txt",ios::out);

    // for(int i = 0; i < 100; i ++) {
    //     double noise = distribution(generator);
    //     f << noise << endl;
    // }
	
	// f.close();

    // f.open("data2.txt",ios::out);

    // for(int i = 0; i < 100; i ++) {
    //     double noise = distribution(generator);
    //     f << noise << endl;
    // }
	
	// f.close();

    // f.open("data3.txt",ios::out);

    // for(int i = 0; i < 100; i ++) {
    //     double noise = distribution(generator);
    //     f << noise << endl;
    // }
	
	// f.close();

	return 0;
}