/*
 * SpecificPrivateOutput.h
 *
 */

#ifndef PROCESSOR_SPECIFICPRIVATEOUTPUT_H_
#define PROCESSOR_SPECIFICPRIVATEOUTPUT_H_

template<class T>
class SpecificPrivateOutput
{
    deque<T> secrets;
    vector<typename T::PO*> pos;
    Player& P;
    vector<bool> active;

public:
    SpecificPrivateOutput(SubProcessor<T>& proc) :
            P(proc.P)
    {
        for (int i = 0; i < P.num_players(); i++)
            pos.push_back(new typename T::PO(proc.P));
        active.resize(P.num_players());
    }

    ~SpecificPrivateOutput()
    {
        for (auto& x : pos)
            delete x;
    }

    void prepare_sending(const T& secret, int player)
    {
        pos[player]->prepare_sending(secret, player);
        if (P.my_num() == player)
            secrets.push_back(secret);
        active[player] = true;
    }

    void exchange()
    {
        for (int i = 0; i < this->P.num_players(); i++)
            if (active[i])
            {
                if (i == this->P.my_num())
                    pos[i]->receive();
                else
                    pos[i]->send(i);
            }
    }

    typename T::clear finalize(int player)
    {
        if (player == this->P.my_num())
        {
            T secret = secrets.front();
            secrets.pop_front();
            return pos[player]->finalize(secret);
        }
        else
            return {};
    }
};

#endif /* PROCESSOR_SPECIFICPRIVATEOUTPUT_H_ */
