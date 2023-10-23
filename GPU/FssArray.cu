#define BYTE_LEN 8

struct FssArray{
  int byte_len = BYTE_LEN;
  uint8_t value[BYTE_LEN];
  
  __device__ FssArray operator+(const FssArray & other) const{
    FssArray result;

    bool carry = 0;
    uint16_t tmp, need_carry = 1<<8;
    for(int i = 0; i < byte_len; i++){
      tmp = value[i] + other.value[i] + carry;
      if(tmp < need_carry)
        carry = 0;
      else{
        carry = 1;
        tmp = tmp % need_carry;
      }
    }
    result.value[i] = tmp;
    return result;
  }
}