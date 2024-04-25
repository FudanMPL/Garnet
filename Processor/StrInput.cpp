/*
 * StrInput.cpp
 *
 */

#include "StrInput.h"

#include <math.h>

template<>
void StrInput_<Integer>::read(std::istream& in, const int* params)
{
   unsigned int code = 0;
    char ch = 0;
    int count = 0;
    while (in.get(ch)) {
        if (std::isspace(ch)) {
            break;
        }
        // cout<<(int) (( unsigned char) ch)<<'\t';
        code = code + ( unsigned char) ch * (1 << count);
        count = count + 8;
    }

    items[0]=(int ) code;
}

template<>
void StrInput_<bigint>::read(std::istream& in, const int* params)
{
#ifdef HIGH_PREC_INPUT
    mpf_class x;
    in >> x;
    items[0] = x << *params;
#else
   unsigned int code = 0;
    char ch = 0;
    int count = 0;

    in.get(ch);
    while(ch==' ' || ch=='\n')in.get(ch);
    unsigned char byte = static_cast<unsigned char>(ch);
    int numBytes = 0;
    bool flag=false;//是否为转义字符
    if(byte==0x5C)
    {
        in.get(ch);
        byte = static_cast<unsigned char>(ch);
        flag=true;
    }
    if (byte <= 0x7F) {
        numBytes = 1;  // 单字节字符
    } else if ((byte >> 5) == 0x6) {
        numBytes = 2;  // 双字节字符
    } else if ((byte >> 4) == 0xE) {
        numBytes = 3;  // 三字节字符
    } else if ((byte >> 3) == 0x1E) {
        numBytes = 4;  // 四字节字符
    }


    code = code + ( unsigned char) ch * (1 << count);
    count = count + 8;
    if(flag && numBytes==1)
    {
        switch(code)
        {
            case 0x74: // \tableofcontents
                items[0]=(int)( unsigned char) '\t';
                break;
            case 0x6E: // \n
                items[0]=(int)( unsigned char) '\n';
                break;
            case 0x5C: // \n
                items[0]=(int)( unsigned char) '\\';
                break;
            case 0x78:// 表示十六进制数x
            // ToDo:转义十六进制数据
                char hexStr[2];
                in.get(hexStr[0]);
                in.get(hexStr[1]);
                items[0] = strtoul(hexStr, NULL, 16);
                break;
            default:
                printf("Wrong escape character!\n");
                break;
        }
    }
    else
    {
        for(int i=1;i< numBytes;++i)
        {
            in.get(ch);
            code = code + ( unsigned char) ch * (1 << count);
            count = count + 8;
        }
        items[0]=(int ) code;
    }
   
#endif
}