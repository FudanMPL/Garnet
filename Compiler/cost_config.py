import re
from collections import defaultdict, deque
import math

#Basic operations, these operations are necessary for one protocol
# share:
# open:
# muls:
# matmuls:

#advance operations, these operations are optional
# TruncPr:
# LTZ:
# Trunc
# Mod2m:
# EQZ:
# SDiv:
# Pow2:
# FPDiv:
# sin:
# cos:
# tan:
# exp2_fx:
# log2_fx:
# sqrt:
# atan:


class Cost(object):
    f = 0
    bit_length = 1
    _security = 1
    n_parties = 2
    computation_security = 1
    cost_dict = defaultdict(lambda: -1)
    cost_dict_func = defaultdict(lambda: -1)
    subcls = None
    @classmethod
    def update_cost(cls):
        for key, value in cls.cost_dict_constasnt_func.items():
            if value.__code__.co_argcount == 5:
                cls.cost_dict[key] = value(cls.bit_length, cls._security, cls.computation_security, cls.f, cls.n_parties)
            else:
                cls.cost_dict[key] = value  
        for key, value in cls.cost_dict_func.items():
            if value.__code__.co_argcount == 5:
                cls.cost_dict[key] = value(cls.bit_length, cls._security,cls.computation_security, cls.f, cls.n_parties)                    
            else:   
                cls.cost_dict[key] = value
        cls.cost_dict["square"] = (0, 0, (cls.cost_dict['muls'][0]+cls.cost_dict['muls'][2]), (cls.cost_dict['muls'][1]+cls.cost_dict['muls'][3]))
        
        if cls.program.protocol=="SPDZ":
            cls.cost_dict["randbit"] = (0, 0, (cls.cost_dict['open'][0]+cls.cost_dict['open'][2])*cls.n_parties+\
                (cls.cost_dict['muls'][0]+cls.cost_dict['muls'][2])*(cls.n_parties-1), \
                    (cls.cost_dict['open'][1]+cls.cost_dict['open'][3])+\
                        (cls.cost_dict['muls'][1]+cls.cost_dict['muls'][3])*(cls.n_parties-1))            
        elif cls.program.options.ring:
            cls.cost_dict["randbit"] = (0, 0, (cls.cost_dict['share'][0]+cls.cost_dict['share'][2])*cls.n_parties+\
                (cls.cost_dict['muls'][0]+cls.cost_dict['muls'][2])*(cls.n_parties-1), \
                    (cls.cost_dict['share'][1]+cls.cost_dict['share'][3])+\
                        (cls.cost_dict['muls'][1]+cls.cost_dict['muls'][3])*(cls.n_parties-1))
        else:
            cls.cost_dict["randbit"] = (0, 0, cls.cost_dict['open'][0], cls.cost_dict['open'][1])     
            cls.cost_dict["dabit"] = (0, 0, cls.cost_dict["randbit"][2] +\
             (cls.cost_dict['share'][0]+cls.cost_dict['share'][2])*cls.n_parties, \
                 cls.cost_dict["randbit"][3]+(cls.cost_dict['share'][0]+cls.cost_dict['share'][2]))
                 
    @classmethod
    def init(cls, program):
        for arg in program.args:
                m = re.match('f([0-9]+)$', arg)
                if m:
                    cls.f = int(m.group(1))
        if program.options.ring:
            cls.bit_length = program.bit_length+1
        else:
            cls.bit_length = program.bit_length
        cls._security = cls.bit_length
        cls.computation_security = program.c_security
        cls.n_parties = program.n_parties
        cls.program = program
        cls.update_cost()
        Cost.subcls = cls
    
    @classmethod
    def set_precision(self, precision):
        Cost.f = precision
        Cost.subcls.update_cost()

    cost_dict_constasnt_func = {
        "triple":lambda bit_length, kappa_s ,kapaa, precision, n_parties: (0, 0, bit_length * (bit_length+kapaa), 1), # currently, only for two party
        "sedabit": lambda bit_length,  kappa_s ,kapaa, precision, n_parties, len: (0, 0, 0, 0),
        "edabit": lambda bit_length,  kappa_s ,kapaa, precision, n_parties, len: (0, 0, 0, 0),
        "dabit": lambda bit_length,  kappa_s ,kapaa, precision, n_parties: (0, 0, 0, 0), #done
        "shufflegen": lambda bit_length,  kappa_s ,kapaa, precision, n_parties, size: (0, 0, 0, 0),
        "shuffleapply": lambda bit_length,  kappa_s ,kapaa, precision, n_parties, size, record_size: (0, 1, 0, 0),
        "randbit": lambda bit_length,  kappa_s , kapaa, precision, n_parties: (0, 0, 1, 1), #done
    }
    @classmethod
    def get_cost(self, name):
        return self.cost_dict[name]


class ABY3(Cost): #done
    cost_dict_func = {
        "share": lambda bit_length,  kappa_s , kapaa, precision, n_parties: (bit_length*3, 1, 0, 0),
        "open" : lambda bit_length, kappa_s , kapaa, precision, n_parties: (bit_length*3, 1, 0, 0),
        "muls" : lambda bit_length, kappa_s , kapaa, precision, n_parties: (bit_length*3, 1, 0, 0),
        "matmuls": lambda bit_length, kappa_s , kapaa, precision, n_parties, p ,q, r: (p*r*bit_length*3, 1, 0, 0),
        "TruncPr": lambda bit_length, kappa_s , kapaa, precision, n_parties: (bit_length, 1, 0, 0),
        "LTZ": lambda bit_length, kappa_s , kapaa, precision, n_parties: (bit_length*9, math.log2(bit_length)+2, 0, 0),
   }
        # "trunc": lambda bit_length, kappa_s , kapaa, precision, n_parties: (bit_length, 1, 0, 0),
        # "bit_share":lambda bit_length, kappa_s , kapaa, precision, n_parties: (3, 1, 0, 0),
        # "ands":lambda bit_length, kappa_s , kapaa, precision, n_parties: (3, 1, 0, 0)
        
class SecureML(Cost): #done
    cost_dict_func = {
        "share": lambda bit_length, kappa_s , kapaa, precision, n_parties: (0, 0, 0, 0),
        "open" : lambda bit_length, kappa_s , kapaa, precision, n_parties: (bit_length*2, 1, 0, 0),
        "muls" : lambda bit_length,  kappa_s ,kapaa, precision, n_parties: (bit_length*4, 1, bit_length * (bit_length+kapaa), 1),
        "matmuls": lambda bit_length, kappa_s , kapaa, precision, n_parties, p ,q, r: (p*q*bit_length*2+q*r*bit_length*2, 1, p*q*r*bit_length * (bit_length+kapaa), 1),
        "TruncPr": lambda bit_length, kappa_s , kapaa, precision, n_parties: (0, 0, 0, 0)
   }

class ABY(Cost):
    cost_dict_func = {
        "share": lambda bit_length, kappa_s , kapaa, precision, n_parties: (0, 0, 0, 0),
        "open" : lambda bit_length, kappa_s , kapaa, precision, n_parties: (bit_length*2, 1, 0, 0),
        "muls" : lambda bit_length, kappa_s , kapaa, precision, n_parties: (bit_length*4, 1, bit_length*(2*kapaa+bit_length+1), 2),
        "matmuls": lambda bit_length, kappa_s , kapaa, precision, n_parties, p ,q, r: (p*q*r*bit_length*2, 1, p*q*r*bit_length*(2*kapaa+bit_length+1), 2),
        "TruncPr" : lambda bit_length,  kappa_s ,kapaa, precision, n_parties: (0, 0, 0, 0),
        "LTZ": lambda bit_length,  kappa_s ,kapaa, precision, n_parties: (bit_length*kapaa*2+kapaa+bit_length, 4, bit_length*kapaa*4+kapaa, 2),
   }
    
class MPCFormer(Cost):
    cost_dict_func = {
    "share": lambda bit_length, kappa_s , kapaa, precision, n_parties: (0, 0, 0, 0),
    "open" : lambda bit_length, kappa_s , kapaa, precision, n_parties: (bit_length*2, 1, 0, 0),
    "muls" : lambda bit_length, kappa_s , kapaa, precision, n_parties: (bit_length*4, 1, 0, 0),
    "matmuls": lambda bit_length, kappa_s , kapaa, precision, n_parties, p ,q, r: ((p*q + q*r)*bit_length*2, 1, 0, 0),
    "TruncPr": lambda bit_length,  kappa_s ,kapaa, precision, n_parties: (0, 0, 0, 0),
    "exp_fx":lambda bit_length, kappa_s , kapaa, precision, n_parties: (bit_length*(16), 8, 0, 0),
    "LTZ":lambda bit_length,  kappa_s ,kapaa, precision, n_parties: (bit_length*54, 8, 0, 0),
    "EQZ":lambda bit_length,  kappa_s ,kapaa, precision, n_parties: (bit_length*26, 8, 0, 0),
    "Reciprocal":lambda bit_length,  kappa_s ,kapaa, precision, n_parties: (bit_length*138, 38, 0, 0)
}
    
def cryptflow2_cost(l, m,kappa):
    q=math.ceil(l/m)
    r = l%m
    if r == 0 :
        r=m
    R = 2**(r)
    M = 2**m
    if q>0:
        return kappa*(4*q-math.ceil(math.log(q))-2)+M*(2*q-3)+R*2+22*(q-1)-2*math.ceil(math.log(q))
    else:
        return 10^11

def cryptflow2_search(l,kappa):
    res = 10^10
    for i in range(1, 10):
        res = min(res, cryptflow2_cost(l, i, kappa))  
    return res 
class CryptoFlow2(Cost):
    cost_dict_func = {
        "share": lambda bit_length,  kappa_s ,kapaa, precision, n_parties: (0, 0, 0, 0),
        "open" : lambda bit_length,  kappa_s ,kapaa, precision, n_parties: (bit_length*2, 1, 0, 0),
        "muls" : lambda bit_length, kappa_s , kapaa, precision, n_parties: (bit_length*(math.ceil((bit_length+1)/2)+kapaa), 2, 0, 0),
        "matmuls": lambda bit_length, kappa_s , kapaa, precision, n_parties, p ,q, r: (q*r*bit_length*(p*math.ceil((bit_length+1))/2+kapaa), 2, 0, 0),
        "LTZ": lambda bit_length, kappa_s , kapaa, precision,  n_parties: ((kapaa+18)*bit_length, math.log2(bit_length)+2, 0, 0),
        "TruncPr": lambda bit_length, kappa_s , kapaa, precision, n_parties: ((kapaa+2)*bit_length+19*bit_length+kapaa*precision+14*precision, 2*math.ceil(math.log(bit_length))+2, 0, 0)
   }
    
class SPDZ(Cost):
    cost_dict_func = {
        "share": lambda bit_length, kappa_s , kapaa, precision, n_parties: ((kappa_s+bit_length)*(n_parties-1), 1, (kappa_s*bit_length+bit_length+kappa_s)*n_parties*(n_parties-1), 3), # offline: Fmac 
        "open" : lambda bit_length, kappa_s , kapaa, precision, n_parties: ((bit_length+kappa_s)*n_parties*(n_parties-1), 1, (kappa_s*bit_length+bit_length+kappa_s)*n_parties*(n_parties-1), 3), #((bit_length+kappas_s)*n_parties)
        "muls" : lambda bit_length, kappa_s , kapaa, precision, n_parties: ((bit_length+kappa_s)*n_parties*(n_parties-1)*2, 1, (kappa_s*bit_length+bit_length+kappa_s)*n_parties*(n_parties-1)*2+ n_parties*(n_parties-1)*(18*kappa_s*kappa_s+4*bit_length*bit_length+17*bit_length*kappa_s), 8), #(Fmac+(bit_length+kappas_s)*n_parties)
        "matmuls": lambda bit_length, kappa_s , kapaa, precision, n_parties, p ,q, r: (p*q*r*(bit_length+kappa_s)*n_parties*(n_parties-1)*2, 1, p*q*r*((kappa_s*bit_length+bit_length+kappa_s)*n_parties*(n_parties-1)*2+ n_parties*(n_parties-1)*(18*kappa_s*kappa_s+4*bit_length*bit_length+17*bit_length*kappa_s)), 8), #pqr(Fmac+(bit_length+kappas_s)*n_parties)
        "TruncPr": lambda bit_length, kappa_s , kapaa, precision, n_parties: ((bit_length+kappa_s)*n_parties*(n_parties-1), 1, bit_length*((kappa_s+bit_length)*(n_parties-1)+(kappa_s*bit_length+bit_length+kappa_s)*n_parties*(n_parties-1)+(bit_length+kappa_s)*n_parties*(n_parties-1)*2+(kappa_s*bit_length+bit_length+kappa_s)*n_parties*(n_parties-1)*2+ n_parties*(n_parties-1)*(18*kappa_s*kappa_s+4*bit_length*bit_length+17*bit_length*kappa_s)), 13)
   }

class BGW(Cost): #done
    cost_dict_func = {
        "share": lambda bit_length, kappa_s , kapaa, precision, n_parties: (n_parties-math.floor(n_parties/2)-1, 1, 0, 0),
        "open" : lambda bit_length, kappa_s , kapaa, precision, n_parties: (n_parties*math.floor(n_parties/2), 1, 0, 0),
        "muls" : lambda bit_length, kappa_s , kapaa, precision, n_parties: (n_parties*(n_parties-math.floor(n_parties/2)-1), 1, 0, 0),
        "matmuls": lambda bit_length, kappa_s , kapaa, precision, n_parties, p ,q, r: (p*r*n_parties*(n_parties-math.floor(n_parties/2)-1), 1, 0, 0)
   }

class Falcon(Cost):
    cost_dict_func = {
        "share": lambda bit_length, kappa_s , kapaa, precision, n_parties: (bit_length*3, 1, 0, 0),
        "open" : lambda bit_length, kappa_s , kapaa, precision, n_parties: (bit_length*6, 1, 0, 0),
        "muls" : lambda bit_length, kappa_s , kapaa, precision, n_parties: (bit_length*6, 1, 0, 0),
        "matmuls": lambda bit_length, kappa_s , kapaa, precision, n_parties, p ,q, r: (p*r*bit_length*6, 1, 0, 0),
        "LTZ": lambda bit_length, kappa_s , kapaa, precision,  n_parties: (24*bit_length, math.log2(bit_length)+5, bit_length*2+8*6*2+bit_length+bit_length*math.log2(bit_length)+bit_length*2+bit_length, 8+math.log2(bit_length)+1+2+math.log2(bit_length)+1),
        "TruncPr": lambda bit_length, kappa_s , kapaa, precision, n_parties: (bit_length, 1, bit_length*8+bit_length+bit_length*math.log2(bit_length)+(bit_length-precision)+(bit_length-precision)*math.ceil(math.log2((bit_length-precision))), 1+math.log2(bit_length)+1),
        "Pow2":lambda bit_length, kappa_s , kapaa, precision, n_parties: (bit_length*bit_length*(24),  bit_length*(math.log2(bit_length)+5), (bit_length*2+8*6*2+bit_length+bit_length*math.log2(bit_length)+bit_length*2+bit_length)*bit_length, 8+math.log2(bit_length)+1+2+math.log2(bit_length)+1),
        "Reciprocal":lambda bit_length,  kappa_s ,kapaa, precision, n_parties: (bit_length*bit_length*(24) + 36*bit_length, bit_length*(math.log2(bit_length)+5)+5, (bit_length*2+8*6*2+bit_length+bit_length*math.log2(bit_length)+bit_length*2+bit_length)*bit_length, 8+math.log2(bit_length)+1+2+math.log2(bit_length)+1),
   }

protocol_store = {
    "ABY3" : ABY3(),
    "SecureML" : SecureML(),
    "ABY": ABY(),
    "Falcon": Falcon(),
    "BGW": BGW(),
    "CryptFlow2": CryptoFlow2(),
    "MPCFormer": MPCFormer(),
    "SPDZ": SPDZ()
}

def get_cost_config(name):
    return protocol_store[name]



if __name__ == "__main__":
    print(128*59+4*59)
    print(cryptflow2_cost(32, 7, 128))