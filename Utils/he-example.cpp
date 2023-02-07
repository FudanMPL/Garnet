/*
 * he-example.cpp
 *
 */

#include "FHE/FHE_Params.h"
#include "FHE/NTL-Subs.h"
#include "FHE/FHE_Keys.h"
#include "FHE/Plaintext.h"

void first_phase(string filename, int n_mults, int circuit_sec);
void second_phase(string filename);

int main()
{
    for (int n_mults = 0; n_mults < 2; n_mults++)
        for (int sec = 0; sec <= 120; sec += 40)
        {
            string filename = "mp-spdz-he";
            first_phase(filename, n_mults, sec);
            second_phase(filename);
        }
}

void first_phase(string filename, int n_mults, int circuit_sec)
{
    // specify number of multiplications (at most one) and function privacy parameter
    // increase the latter to accommodate more operations
    FHE_Params params(n_mults, circuit_sec);

    // generate parameters for computation modulo a 32-bit prime
    params.basic_generation_mod_prime(32);

    // find computation modulus (depends on parameter generation)
    cout << "computation modulo " << params.get_plaintext_modulus() << endl;

    // generate key pair
    FHE_KeyPair pair(params);
    pair.generate();

    Plaintext_mod_prime plaintext(params);

    // set first two plaintext slots
    plaintext.set_element(0, 4);
    plaintext.set_element(1, -1);

    // encrypt
    Ciphertext ciphertext = pair.pk.encrypt(plaintext);

    // store for second phase
    octetStream os;
    params.pack(os);
    pair.pk.pack(os);
    ciphertext.pack(os);
    plaintext.pack(os);
    pair.sk.pack(os);
    ofstream out(filename);
    os.output(out);
}

void second_phase(string filename)
{
    // read from file
    ifstream in(filename);
    octetStream os;
    os.input(in);
    FHE_Params params;
    FHE_PK pk(params);
    FHE_SK sk(params);
    Plaintext_mod_prime plaintext(params);
    Ciphertext ciphertext(params);

    // parameter must be set correctly first
    params.unpack(os);
    pk.unpack(os);
    ciphertext.unpack(os);
    plaintext.unpack(os);

    if (params.n_mults() == 0)
        // public-private multiplication is always available
        ciphertext *= plaintext;
    else
        // private-private multiplication only with matching parameters
        ciphertext = ciphertext.mul(pk, ciphertext);

    // re-randomize for circuit privacy
    ciphertext.rerandomize(pk);

    // read secret key and decrypt
    sk.unpack(os);
    plaintext = sk.decrypt(ciphertext);

    cout << "should be 16: " << plaintext.element(0) << endl;
    cout << "should be 1: " << plaintext.element(1) << endl;
    assert(plaintext.element(0) == 16);
    assert(plaintext.element(1) == 1);
}
