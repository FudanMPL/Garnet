#include "Math/gf2n.h"
#include "Math/gfp.hpp"
#include "Math/Z2k.h"
#include "Math/Setup.h"
#include "Protocols/Share.h"

#include <fstream>
#include <vector>
#include <numeric>

#include "Math/Z2k.hpp"

template <class T>
void check_triples(int n_players, string type_char = "")
{
    ifstream* inputFiles = new ifstream[n_players];
    for (int i = 0; i < n_players; i++)
    {
        stringstream ss;
        ss << get_prep_sub_dir<Share<T>>(n_players) << "Triples-";
        if (type_char.size())
            ss << type_char;
        else
            ss << T::type_char();
        ss << "-P" << i;
        inputFiles[i].open(ss.str().c_str());
        cout << "Opening file " << ss.str() << endl;
        octetStream tmp, tmp2 = file_signature<T>();
        tmp.input(inputFiles[i]);
        assert(tmp == tmp2);
    }

    int j = 0;
    while (inputFiles[0].peek() != EOF)
    {
        T a,b,c,cc,tmp,prod;
        vector<T> as(n_players), bs(n_players), cs(n_players);
        for (int i = 0; i < n_players; i++)
        {
            as[i].input(inputFiles[i], false);
            bs[i].input(inputFiles[i], false);
            cs[i].input(inputFiles[i], false);
        }

        a = accumulate(as.begin(), as.end(), T());
        b = accumulate(bs.begin(), bs.end(), T());
        c = accumulate(cs.begin(), cs.end(), T());

        prod = a * b;
        if (prod != c)
        {
            cout << T::type_string() << ": Error in " << j << endl;
            cout << "a " << a << " " << as[0] << " " << as[1] << endl;
            cout << "b " << b << " " << bs[0] << " " << bs[1] << endl;
            cout << "c " << c << " " << cs[0] << " " << cs[1] << endl;
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 2; j++)
                {
                    tmp = as[i] * bs[j];
                    cc += tmp;
                    cout << "a" << i << " * b" << j << " " << tmp << endl;
                }
            cout << "cc " << cc << endl;
            cout << "a*b " << prod << endl;
            cout << "DID YOU INDICATE THE CORRECT NUMBER OF PLAYERS?" << endl;

            return;
        }

        j++;
    }

    cout << j << " correct triples of type " << T::type_string() << endl;
    delete[] inputFiles;
}

int main(int argc, char** argv)
{
    int n_players = 2;
    if (argc > 1)
        n_players = atoi(argv[1]);
    read_setup(get_prep_sub_dir<Share<gfp>>(PREP_DIR, n_players, 128));
    gfp::init_field(gfp::pr(), false);
    gf2n::init_field();
    check_triples<gf2n>(n_players);
    check_triples<gfp>(n_players);
}
