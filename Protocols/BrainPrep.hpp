/*
 * BrainPrep.cpp
 *
 */

#include "BrainPrep.h"
#include "Processor/Processor.h"
#include "Protocols/MaliciousRepMC.h"
#include "Tools/Subroutines.h"
#include "Math/gfp.h"

template<class T> class ZProtocol;

template<int L>
class Zint : public SignedZ2<L + 2>
{
    typedef SignedZ2<L + 2> super;

public:
    static string type_string()
    {
        return "Zint" + to_string(L);
    }

    Zint()
    {
    }

    template<class T>
    Zint(const T& other) : super(other)
    {
    }

    void randomize(PRNG& G, int n = -1)
    {
        (void) n;
        *this = G.get<Z2<L>>();
    }
};

template<class U>
class ZShare : public Rep3Share<Zint<U::Z_BITS>>
{
public:
    typedef ZProtocol<U> Protocol;
    typedef ReplicatedMC<ZShare> MAC_Check;

    ZShare()
    {
    }

    template<class T>
    ZShare(const FixedVec<T, 2>& other)
    {
        FixedVec<T, 2>::operator=(other);
    }
};

template<class U>
class ZProtocol : public Replicated<Rep3Share<Zint<U::Z_BITS>>>
{
    typedef Rep3Share<Zint<U::Z_BITS>> T;
    vector<T> random;
    SeededPRNG G;
    ReplicatedInput<Rep3Share<Zint<U::N_MASK_BITS>>> input;

public:
    ZProtocol(Player& P) : Replicated<T>(P), input(0, this->P)
    {
    }

    T get_random()
    {
        if (random.empty())
        {
            int buffer_size = OnlineOptions::singleton.batch_size;
            input.reset_all(this->P);
            for (int i = 0; i < buffer_size; i++)
            {
                typename U::clear tmp;
                tmp.randomize(G);
                input.add_mine(tmp);
            }
            for (int i = 0; i < this->P.num_players(); i++)
                input.add_other(i);
            input.exchange();
            for (int i = 0; i < buffer_size; i++)
            {
                random.push_back({});
                for (int j = 0; j < 3; j++)
                    random.back() += input.finalize(j);
            }
        }

        auto res = random.back();
        random.pop_back();
        return res;
    }
};

template<class T>
void BrainPrep<T>::basic_setup(Player&)
{
    gfp2::init_default(DIV_CEIL(T::Z_BITS + 3, 8) * 8);
}

template<class T>
void BrainPrep<T>::buffer_triples()
{
    if(gfp2::get_ZpD().pr_bit_length
            <= ZProtocol<T>::share_type::clear::N_BITS)
        throw runtime_error(
                to_string(gfp2::get_ZpD().pr_bit_length)
                        + "-bit prime too short for "
                        + to_string(ZProtocol<T>::share_type::clear::N_BITS)
                        + "-bit integer computation");
    typedef Rep3Share<gfp2> pShare;
    auto buffer_size = OnlineOptions::singleton.batch_size;
    Player& P = this->protocol->P;
    vector<array<ZShare<T>, 3>> triples;
    vector<array<Rep3Share<gfp2>, 3>> check_triples;
    DataPositions usage;
    HashMaliciousRepMC<pShare> MC;
    vector<Rep3Share<gfp2>> masked, checks;
    vector<gfp2> opened;
    ZProtocol<T> Z_protocol(P);
    Replicated<pShare> p_protocol(P);
    generate_triples(triples, buffer_size, &Z_protocol);
    check_triples.resize(buffer_size);
    p_protocol.init_mul();
    for (int i = 0; i < buffer_size; i++)
    {
        pShare a = p_protocol.get_random();
        pShare b = triples[i][1];
        p_protocol.prepare_mul(a, b);
        check_triples[i][0] = a;
        check_triples[i][1] = b;
    }
    p_protocol.exchange();
    auto t = Create_Random<gfp2>(P);
    for (int i = 0; i < buffer_size; i++)
    {
        check_triples[i][2] = p_protocol.finalize_mul();
        pShare a = triples[i][0];
        auto& f = check_triples[i][0];
        masked.push_back(a * t - f);
    }
    MC.POpen(opened, masked, P);
    for (int i = 0; i < buffer_size; i++)
    {
        auto& b = check_triples[i][1];
        pShare c = triples[i][2];
        auto& h = check_triples[i][2];
        auto& rho = opened[i];
        checks.push_back(t * c - h - rho * b);
    }
    MC.CheckFor(0, checks, P);
    MC.Check(P);
    for (auto& x : triples)
        this->triples.push_back({{x[0], x[1], x[2]}});
}

template<class T>
void BrainPrep<T>::buffer_inputs(int player)
{
    this->buffer_inputs_as_usual(player, this->proc);
}
