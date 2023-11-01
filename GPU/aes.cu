// ........ jsaes: AES in JavaScript (... B. Poettering) ... C ....
// ... http://point-at-infinity.org/jsaes/.... GNU GPL ...
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <cuda.h>
#include "fss_struct.h"
#define BYTE uint8_t


#ifndef AES_CU_
#define AES_CU_

using namespace std;


void printBytes(BYTE b[], int len) {
int i;
for (i=0; i<len; i++)
    printf("%x ", b[i]);
//    cout << hex << b[i] << " " ;
printf("\n");
}


void f1printBytes(BYTE b[], int len, FILE* fp) {
int i;
for (i=0; i<len; i++)
   fprintf(fp, "%02x ", b[i]);
//    cout << hex << b[i] << " " ;
fprintf(fp, "\n");
}
int flag=0;
void f2printBytes(BYTE b[], int len, FILE* fp) {
int i;
for (i=0; i<len; i++){
   fprintf(fp, "%c", b[i]);
   if(b[i]=='\n')
        flag++;
   }
//    cout << hex << b[i] << " " ;
//fprintf(fp, "\n");
}
void f3printBytes(BYTE b[], int len, FILE* fp) {
int i;
for (i=0; i<len; i++){
 if(b[i]=='\0'){
   return ;
   }
   fprintf(fp, "%c", b[i]);
   //printf("%x ", b[i]);
   if(b[i]=='\n')
        flag++;
   }
//    cout << hex << b[i] << " " ;
//fprintf(fp, "\n");
}
/******************************************************************************/
// The following lookup tables and functions are for internal use only!


BYTE AES_Sbox[] =
{   /*0    1    2    3    4    5    6    7    8    9    a    b    c    d    e    f */
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76, /*0*/ 
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0, /*1*/
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15, /*2*/
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75, /*3*/
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84, /*4*/
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf, /*5*/
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8, /*6*/ 
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2, /*7*/
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73, /*8*/
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb, /*9*/
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79, /*a*/
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08, /*b*/
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a, /*c*/
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e, /*d*/
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf, /*e*/
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16  /*f*/
};


//BYTE AES_ShiftRowTab[] = {0,5,10,15,4,9,14,3,8,13,2,7,12,1,6,11};

//BYTE AES_Sbox_Inv[256];
//BYTE AES_ShiftRowTab_Inv[16];
//BYTE AES_xtime[256];

__device__ void AES_SubBytes(BYTE state[], BYTE sbox[]) {
int i;
for(i = 0; i < 16; i++)
    state[i] = sbox[state[i]];
}

__device__ void AES_AddRoundKey(BYTE state[], BYTE rkey[]) {
    int i;
    for(i = 0; i < 16; i++)
        state[i] ^= rkey[i];
}

__device__ void AES_ShiftRows(BYTE state[], BYTE shifttab[]) {
    BYTE h[16];
    memcpy(h, state, 16);
    int i;
    for(i = 0; i < 16; i++)
        state[i] = h[shifttab[i]];
}

__device__ void AES_MixColumns(BYTE state[], BYTE AES_xtime[]) {
    int i;
    //ared__ BYTE AES_xtime[];
for(i = 0; i < 16; i += 4) {
    BYTE s0 = state[i + 0], s1 = state[i + 1];
    BYTE s2 = state[i + 2], s3 = state[i + 3];
    BYTE h = s0 ^ s1 ^ s2 ^ s3;
    state[i + 0] ^= h ^ AES_xtime[s0 ^ s1];
    state[i + 1] ^= h ^ AES_xtime[s1 ^ s2];
    state[i + 2] ^= h ^ AES_xtime[s2 ^ s3];
    state[i + 3] ^= h ^ AES_xtime[s3 ^ s0];
}
}

__device__ void AES_MixColumns_Inv(BYTE state[], BYTE AES_xtime[]) {
    int i;
    for(i = 0; i < 16; i += 4) {
        BYTE s0 = state[i + 0], s1 = state[i + 1];
        BYTE s2 = state[i + 2], s3 = state[i + 3];
        BYTE h = s0 ^ s1 ^ s2 ^ s3;
        BYTE xh = AES_xtime[h];
        BYTE h1 = AES_xtime[AES_xtime[xh ^ s0 ^ s2]] ^ h;
        BYTE h2 = AES_xtime[AES_xtime[xh ^ s1 ^ s3]] ^ h;
        state[i + 0] ^= h1 ^ AES_xtime[s0 ^ s1];
        state[i + 1] ^= h2 ^ AES_xtime[s1 ^ s2];
        state[i + 2] ^= h1 ^ AES_xtime[s2 ^ s3];
        state[i + 3] ^= h2 ^ AES_xtime[s3 ^ s0];
}
}

// AES_Init: initialize the tables needed at runtime. 
// Call this function before the (first) key expansion.
__device__ void AES_Init(BYTE AES_Sbox[], BYTE AES_ShiftRowTab[], BYTE AES_Sbox_Inv[], BYTE AES_xtime[], BYTE AES_ShiftRowTab_Inv[]) {
    //__shared__ BYTE AES_ShiftRowTab[16] ;
    AES_ShiftRowTab[0]=0;
    AES_ShiftRowTab[1]=5;
    AES_ShiftRowTab[2]=10;
    AES_ShiftRowTab[3]=15;
    AES_ShiftRowTab[4]=4;
    AES_ShiftRowTab[5]=9;
    AES_ShiftRowTab[6]=14;
    AES_ShiftRowTab[7]=3;
    AES_ShiftRowTab[8]=8;
    AES_ShiftRowTab[9]=13;
    AES_ShiftRowTab[10]=2;
    AES_ShiftRowTab[11]=7;
    AES_ShiftRowTab[12]=12;
    AES_ShiftRowTab[13]=1;
    AES_ShiftRowTab[14]=6;
    AES_ShiftRowTab[15]=11;
    
AES_Sbox[0] = 0x63;AES_Sbox[1] = 0x7c;AES_Sbox[2] = 0x77;AES_Sbox[3] = 0x7b;AES_Sbox[4] = 0xf2;AES_Sbox[5] = 0x6b;AES_Sbox[6] = 0x6f;AES_Sbox[7] = 0xc5;AES_Sbox[8] = 0x30;AES_Sbox[9] = 0x1;AES_Sbox[10] = 0x67;AES_Sbox[11] = 0x2b;AES_Sbox[12] = 0xfe;AES_Sbox[13] = 0xd7;AES_Sbox[14] = 0xab;AES_Sbox[15] = 0x76;
AES_Sbox[16] = 0xca;AES_Sbox[17] = 0x82;AES_Sbox[18] = 0xc9;AES_Sbox[19] = 0x7d;AES_Sbox[20] = 0xfa;AES_Sbox[21] = 0x59;AES_Sbox[22] = 0x47;AES_Sbox[23] = 0xf0;AES_Sbox[24] = 0xad;AES_Sbox[25] = 0xd4;AES_Sbox[26] = 0xa2;AES_Sbox[27] = 0xaf;AES_Sbox[28] = 0x9c;AES_Sbox[29] = 0xa4;AES_Sbox[30] = 0x72;AES_Sbox[31] = 0xc0;
AES_Sbox[32] = 0xb7;AES_Sbox[33] = 0xfd;AES_Sbox[34] = 0x93;AES_Sbox[35] = 0x26;AES_Sbox[36] = 0x36;AES_Sbox[37] = 0x3f;AES_Sbox[38] = 0xf7;AES_Sbox[39] = 0xcc;AES_Sbox[40] = 0x34;AES_Sbox[41] = 0xa5;AES_Sbox[42] = 0xe5;AES_Sbox[43] = 0xf1;AES_Sbox[44] = 0x71;AES_Sbox[45] = 0xd8;AES_Sbox[46] = 0x31;AES_Sbox[47] = 0x15;
AES_Sbox[48] = 0x4;AES_Sbox[49] = 0xc7;AES_Sbox[50] = 0x23;AES_Sbox[51] = 0xc3;AES_Sbox[52] = 0x18;AES_Sbox[53] = 0x96;AES_Sbox[54] = 0x5;AES_Sbox[55] = 0x9a;AES_Sbox[56] = 0x7;AES_Sbox[57] = 0x12;AES_Sbox[58] = 0x80;AES_Sbox[59] = 0xe2;AES_Sbox[60] = 0xeb;AES_Sbox[61] = 0x27;AES_Sbox[62] = 0xb2;AES_Sbox[63] = 0x75;
AES_Sbox[64] = 0x9;AES_Sbox[65] = 0x83;AES_Sbox[66] = 0x2c;AES_Sbox[67] = 0x1a;AES_Sbox[68] = 0x1b;AES_Sbox[69] = 0x6e;AES_Sbox[70] = 0x5a;AES_Sbox[71] = 0xa0;AES_Sbox[72] = 0x52;AES_Sbox[73] = 0x3b;AES_Sbox[74] = 0xd6;AES_Sbox[75] = 0xb3;AES_Sbox[76] = 0x29;AES_Sbox[77] = 0xe3;AES_Sbox[78] = 0x2f;AES_Sbox[79] = 0x84;
AES_Sbox[80] = 0x53;AES_Sbox[81] = 0xd1;AES_Sbox[82] = 0x0;AES_Sbox[83] = 0xed;AES_Sbox[84] = 0x20;AES_Sbox[85] = 0xfc;AES_Sbox[86] = 0xb1;AES_Sbox[87] = 0x5b;AES_Sbox[88] = 0x6a;AES_Sbox[89] = 0xcb;AES_Sbox[90] = 0xbe;AES_Sbox[91] = 0x39;AES_Sbox[92] = 0x4a;AES_Sbox[93] = 0x4c;AES_Sbox[94] = 0x58;AES_Sbox[95] = 0xcf;
AES_Sbox[96] = 0xd0;AES_Sbox[97] = 0xef;AES_Sbox[98] = 0xaa;AES_Sbox[99] = 0xfb;AES_Sbox[100] = 0x43;AES_Sbox[101] = 0x4d;AES_Sbox[102] = 0x33;AES_Sbox[103] = 0x85;AES_Sbox[104] = 0x45;AES_Sbox[105] = 0xf9;AES_Sbox[106] = 0x2;AES_Sbox[107] = 0x7f;AES_Sbox[108] = 0x50;AES_Sbox[109] = 0x3c;AES_Sbox[110] = 0x9f;AES_Sbox[111] = 0xa8;
AES_Sbox[112] = 0x51;AES_Sbox[113] = 0xa3;AES_Sbox[114] = 0x40;AES_Sbox[115] = 0x8f;AES_Sbox[116] = 0x92;AES_Sbox[117] = 0x9d;AES_Sbox[118] = 0x38;AES_Sbox[119] = 0xf5;AES_Sbox[120] = 0xbc;AES_Sbox[121] = 0xb6;AES_Sbox[122] = 0xda;AES_Sbox[123] = 0x21;AES_Sbox[124] = 0x10;AES_Sbox[125] = 0xff;AES_Sbox[126] = 0xf3;AES_Sbox[127] = 0xd2;
AES_Sbox[128] = 0xcd;AES_Sbox[129] = 0xc;AES_Sbox[130] = 0x13;AES_Sbox[131] = 0xec;AES_Sbox[132] = 0x5f;AES_Sbox[133] = 0x97;AES_Sbox[134] = 0x44;AES_Sbox[135] = 0x17;AES_Sbox[136] = 0xc4;AES_Sbox[137] = 0xa7;AES_Sbox[138] = 0x7e;AES_Sbox[139] = 0x3d;AES_Sbox[140] = 0x64;AES_Sbox[141] = 0x5d;AES_Sbox[142] = 0x19;AES_Sbox[143] = 0x73;
AES_Sbox[144] = 0x60;AES_Sbox[145] = 0x81;AES_Sbox[146] = 0x4f;AES_Sbox[147] = 0xdc;AES_Sbox[148] = 0x22;AES_Sbox[149] = 0x2a;AES_Sbox[150] = 0x90;AES_Sbox[151] = 0x88;AES_Sbox[152] = 0x46;AES_Sbox[153] = 0xee;AES_Sbox[154] = 0xb8;AES_Sbox[155] = 0x14;AES_Sbox[156] = 0xde;AES_Sbox[157] = 0x5e;AES_Sbox[158] = 0xb;AES_Sbox[159] = 0xdb;
AES_Sbox[160] = 0xe0;AES_Sbox[161] = 0x32;AES_Sbox[162] = 0x3a;AES_Sbox[163] = 0xa;AES_Sbox[164] = 0x49;AES_Sbox[165] = 0x6;AES_Sbox[166] = 0x24;AES_Sbox[167] = 0x5c;AES_Sbox[168] = 0xc2;AES_Sbox[169] = 0xd3;AES_Sbox[170] = 0xac;AES_Sbox[171] = 0x62;AES_Sbox[172] = 0x91;AES_Sbox[173] = 0x95;AES_Sbox[174] = 0xe4;AES_Sbox[175] = 0x79;
AES_Sbox[176] = 0xe7;AES_Sbox[177] = 0xc8;AES_Sbox[178] = 0x37;AES_Sbox[179] = 0x6d;AES_Sbox[180] = 0x8d;AES_Sbox[181] = 0xd5;AES_Sbox[182] = 0x4e;AES_Sbox[183] = 0xa9;AES_Sbox[184] = 0x6c;AES_Sbox[185] = 0x56;AES_Sbox[186] = 0xf4;AES_Sbox[187] = 0xea;AES_Sbox[188] = 0x65;AES_Sbox[189] = 0x7a;AES_Sbox[190] = 0xae;AES_Sbox[191] = 0x8;
AES_Sbox[192] = 0xba;AES_Sbox[193] = 0x78;AES_Sbox[194] = 0x25;AES_Sbox[195] = 0x2e;AES_Sbox[196] = 0x1c;AES_Sbox[197] = 0xa6;AES_Sbox[198] = 0xb4;AES_Sbox[199] = 0xc6;AES_Sbox[200] = 0xe8;AES_Sbox[201] = 0xdd;AES_Sbox[202] = 0x74;AES_Sbox[203] = 0x1f;AES_Sbox[204] = 0x4b;AES_Sbox[205] = 0xbd;AES_Sbox[206] = 0x8b;AES_Sbox[207] = 0x8a;
AES_Sbox[208] = 0x70;AES_Sbox[209] = 0x3e;AES_Sbox[210] = 0xb5;AES_Sbox[211] = 0x66;AES_Sbox[212] = 0x48;AES_Sbox[213] = 0x3;AES_Sbox[214] = 0xf6;AES_Sbox[215] = 0xe;AES_Sbox[216] = 0x61;AES_Sbox[217] = 0x35;AES_Sbox[218] = 0x57;AES_Sbox[219] = 0xb9;AES_Sbox[220] = 0x86;AES_Sbox[221] = 0xc1;AES_Sbox[222] = 0x1d;AES_Sbox[223] = 0x9e;
AES_Sbox[224] = 0xe1;AES_Sbox[225] = 0xf8;AES_Sbox[226] = 0x98;AES_Sbox[227] = 0x11;AES_Sbox[228] = 0x69;AES_Sbox[229] = 0xd9;AES_Sbox[230] = 0x8e;AES_Sbox[231] = 0x94;AES_Sbox[232] = 0x9b;AES_Sbox[233] = 0x1e;AES_Sbox[234] = 0x87;AES_Sbox[235] = 0xe9;AES_Sbox[236] = 0xce;AES_Sbox[237] = 0x55;AES_Sbox[238] = 0x28;AES_Sbox[239] = 0xdf;
AES_Sbox[240] = 0x8c;AES_Sbox[241] = 0xa1;AES_Sbox[242] = 0x89;AES_Sbox[243] = 0xd;AES_Sbox[244] = 0xbf;AES_Sbox[245] = 0xe6;AES_Sbox[246] = 0x42;AES_Sbox[247] = 0x68;AES_Sbox[248] = 0x41;AES_Sbox[249] = 0x99;AES_Sbox[250] = 0x2d;AES_Sbox[251] = 0xf;AES_Sbox[252] = 0xb0;AES_Sbox[253] = 0x54;AES_Sbox[254] = 0xbb; AES_Sbox[255] = 0x16;
    //__shared__ BYTE AES_Sbox_Inv[256];
    //__shared__ BYTE AES_ShiftRowTab_Inv[16];
    //__shared__ BYTE AES_xtime[256];
    int i;
    for(i = 0; i < 256; i++){
        AES_Sbox_Inv[AES_Sbox[i]] = i;
    }
    for(i = 0; i < 16; i++)
        AES_ShiftRowTab_Inv[AES_ShiftRowTab[i]] = i;
    for(i = 0; i < 128; i++) {
        AES_xtime[i] = i << 1;
        AES_xtime[128 + i] = (i << 1) ^ 0x1b;
    }
}

__device__ void AES_Init2(BYTE AES_Sbox[], BYTE AES_ShiftRowTab[], BYTE AES_Sbox_Inv[], BYTE AES_xtime[], BYTE AES_ShiftRowTab_Inv[]) {
    //__shared__ BYTE AES_ShiftRowTab[16] ;
    AES_ShiftRowTab[0]=0;
    AES_ShiftRowTab[1]=5;
    AES_ShiftRowTab[2]=10;
    AES_ShiftRowTab[3]=15;
    AES_ShiftRowTab[4]=4;
    AES_ShiftRowTab[5]=9;
    AES_ShiftRowTab[6]=14;
    AES_ShiftRowTab[7]=3;
    AES_ShiftRowTab[8]=8;
    AES_ShiftRowTab[9]=13;
    AES_ShiftRowTab[10]=2;
    AES_ShiftRowTab[11]=7;
    AES_ShiftRowTab[12]=12;
    AES_ShiftRowTab[13]=1;
    AES_ShiftRowTab[14]=6;
    AES_ShiftRowTab[15]=11;
    
    //__shared__ BYTE AES_Sbox_Inv[256];
    //__shared__ BYTE AES_ShiftRowTab_Inv[16];
    //__shared__ BYTE AES_xtime[256];

    
AES_Sbox_Inv[0] = 0x52;AES_Sbox_Inv[1] = 0x9;AES_Sbox_Inv[2] = 0x6a;AES_Sbox_Inv[3] = 0xd5;AES_Sbox_Inv[4] = 0x30;AES_Sbox_Inv[5] = 0x36;AES_Sbox_Inv[6] = 0xa5;AES_Sbox_Inv[7] = 0x38;AES_Sbox_Inv[8] = 0xbf;AES_Sbox_Inv[9] = 0x40;AES_Sbox_Inv[10] = 0xa3;AES_Sbox_Inv[11] = 0x9e;AES_Sbox_Inv[12] = 0x81;AES_Sbox_Inv[13] = 0xf3;AES_Sbox_Inv[14] = 0xd7;AES_Sbox_Inv[15] = 0xfb;
AES_Sbox_Inv[16] = 0x7c;AES_Sbox_Inv[17] = 0xe3;AES_Sbox_Inv[18] = 0x39;AES_Sbox_Inv[19] = 0x82;AES_Sbox_Inv[20] = 0x9b;AES_Sbox_Inv[21] = 0x2f;AES_Sbox_Inv[22] = 0xff;AES_Sbox_Inv[23] = 0x87;AES_Sbox_Inv[24] = 0x34;AES_Sbox_Inv[25] = 0x8e;AES_Sbox_Inv[26] = 0x43;AES_Sbox_Inv[27] = 0x44;AES_Sbox_Inv[28] = 0xc4;AES_Sbox_Inv[29] = 0xde;AES_Sbox_Inv[30] = 0xe9;AES_Sbox_Inv[31] = 0xcb;
AES_Sbox_Inv[32] = 0x54;AES_Sbox_Inv[33] = 0x7b;AES_Sbox_Inv[34] = 0x94;AES_Sbox_Inv[35] = 0x32;AES_Sbox_Inv[36] = 0xa6;AES_Sbox_Inv[37] = 0xc2;AES_Sbox_Inv[38] = 0x23;AES_Sbox_Inv[39] = 0x3d;AES_Sbox_Inv[40] = 0xee;AES_Sbox_Inv[41] = 0x4c;AES_Sbox_Inv[42] = 0x95;AES_Sbox_Inv[43] = 0xb;AES_Sbox_Inv[44] = 0x42;AES_Sbox_Inv[45] = 0xfa;AES_Sbox_Inv[46] = 0xc3;AES_Sbox_Inv[47] = 0x4e;
AES_Sbox_Inv[48] = 0x8;AES_Sbox_Inv[49] = 0x2e;AES_Sbox_Inv[50] = 0xa1;AES_Sbox_Inv[51] = 0x66;AES_Sbox_Inv[52] = 0x28;AES_Sbox_Inv[53] = 0xd9;AES_Sbox_Inv[54] = 0x24;AES_Sbox_Inv[55] = 0xb2;AES_Sbox_Inv[56] = 0x76;AES_Sbox_Inv[57] = 0x5b;AES_Sbox_Inv[58] = 0xa2;AES_Sbox_Inv[59] = 0x49;AES_Sbox_Inv[60] = 0x6d;AES_Sbox_Inv[61] = 0x8b;AES_Sbox_Inv[62] = 0xd1;AES_Sbox_Inv[63] = 0x25;
AES_Sbox_Inv[64] = 0x72;AES_Sbox_Inv[65] = 0xf8;AES_Sbox_Inv[66] = 0xf6;AES_Sbox_Inv[67] = 0x64;AES_Sbox_Inv[68] = 0x86;AES_Sbox_Inv[69] = 0x68;AES_Sbox_Inv[70] = 0x98;AES_Sbox_Inv[71] = 0x16;AES_Sbox_Inv[72] = 0xd4;AES_Sbox_Inv[73] = 0xa4;AES_Sbox_Inv[74] = 0x5c;AES_Sbox_Inv[75] = 0xcc;AES_Sbox_Inv[76] = 0x5d;AES_Sbox_Inv[77] = 0x65;AES_Sbox_Inv[78] = 0xb6;AES_Sbox_Inv[79] = 0x92;
AES_Sbox_Inv[80] = 0x6c;AES_Sbox_Inv[81] = 0x70;AES_Sbox_Inv[82] = 0x48;AES_Sbox_Inv[83] = 0x50;AES_Sbox_Inv[84] = 0xfd;AES_Sbox_Inv[85] = 0xed;AES_Sbox_Inv[86] = 0xb9;AES_Sbox_Inv[87] = 0xda;AES_Sbox_Inv[88] = 0x5e;AES_Sbox_Inv[89] = 0x15;AES_Sbox_Inv[90] = 0x46;AES_Sbox_Inv[91] = 0x57;AES_Sbox_Inv[92] = 0xa7;AES_Sbox_Inv[93] = 0x8d;AES_Sbox_Inv[94] = 0x9d;AES_Sbox_Inv[95] = 0x84;
AES_Sbox_Inv[96] = 0x90;AES_Sbox_Inv[97] = 0xd8;AES_Sbox_Inv[98] = 0xab;AES_Sbox_Inv[99] = 0x0;AES_Sbox_Inv[100] = 0x8c;AES_Sbox_Inv[101] = 0xbc;AES_Sbox_Inv[102] = 0xd3;AES_Sbox_Inv[103] = 0xa;AES_Sbox_Inv[104] = 0xf7;AES_Sbox_Inv[105] = 0xe4;AES_Sbox_Inv[106] = 0x58;AES_Sbox_Inv[107] = 0x5;AES_Sbox_Inv[108] = 0xb8;AES_Sbox_Inv[109] = 0xb3;AES_Sbox_Inv[110] = 0x45;AES_Sbox_Inv[111] = 0x6;
AES_Sbox_Inv[112] = 0xd0;AES_Sbox_Inv[113] = 0x2c;AES_Sbox_Inv[114] = 0x1e;AES_Sbox_Inv[115] = 0x8f;AES_Sbox_Inv[116] = 0xca;AES_Sbox_Inv[117] = 0x3f;AES_Sbox_Inv[118] = 0xf;AES_Sbox_Inv[119] = 0x2;AES_Sbox_Inv[120] = 0xc1;AES_Sbox_Inv[121] = 0xaf;AES_Sbox_Inv[122] = 0xbd;AES_Sbox_Inv[123] = 0x3;AES_Sbox_Inv[124] = 0x1;AES_Sbox_Inv[125] = 0x13;AES_Sbox_Inv[126] = 0x8a;AES_Sbox_Inv[127] = 0x6b;
AES_Sbox_Inv[128] = 0x3a;AES_Sbox_Inv[129] = 0x91;AES_Sbox_Inv[130] = 0x11;AES_Sbox_Inv[131] = 0x41;AES_Sbox_Inv[132] = 0x4f;AES_Sbox_Inv[133] = 0x67;AES_Sbox_Inv[134] = 0xdc;AES_Sbox_Inv[135] = 0xea;AES_Sbox_Inv[136] = 0x97;AES_Sbox_Inv[137] = 0xf2;AES_Sbox_Inv[138] = 0xcf;AES_Sbox_Inv[139] = 0xce;AES_Sbox_Inv[140] = 0xf0;AES_Sbox_Inv[141] = 0xb4;AES_Sbox_Inv[142] = 0xe6;AES_Sbox_Inv[143] = 0x73;
AES_Sbox_Inv[144] = 0x96;AES_Sbox_Inv[145] = 0xac;AES_Sbox_Inv[146] = 0x74;AES_Sbox_Inv[147] = 0x22;AES_Sbox_Inv[148] = 0xe7;AES_Sbox_Inv[149] = 0xad;AES_Sbox_Inv[150] = 0x35;AES_Sbox_Inv[151] = 0x85;AES_Sbox_Inv[152] = 0xe2;AES_Sbox_Inv[153] = 0xf9;AES_Sbox_Inv[154] = 0x37;AES_Sbox_Inv[155] = 0xe8;AES_Sbox_Inv[156] = 0x1c;AES_Sbox_Inv[157] = 0x75;AES_Sbox_Inv[158] = 0xdf;AES_Sbox_Inv[159] = 0x6e;
AES_Sbox_Inv[160] = 0x47;AES_Sbox_Inv[161] = 0xf1;AES_Sbox_Inv[162] = 0x1a;AES_Sbox_Inv[163] = 0x71;AES_Sbox_Inv[164] = 0x1d;AES_Sbox_Inv[165] = 0x29;AES_Sbox_Inv[166] = 0xc5;AES_Sbox_Inv[167] = 0x89;AES_Sbox_Inv[168] = 0x6f;AES_Sbox_Inv[169] = 0xb7;AES_Sbox_Inv[170] = 0x62;AES_Sbox_Inv[171] = 0xe;AES_Sbox_Inv[172] = 0xaa;AES_Sbox_Inv[173] = 0x18;AES_Sbox_Inv[174] = 0xbe;AES_Sbox_Inv[175] = 0x1b;
AES_Sbox_Inv[176] = 0xfc;AES_Sbox_Inv[177] = 0x56;AES_Sbox_Inv[178] = 0x3e;AES_Sbox_Inv[179] = 0x4b;AES_Sbox_Inv[180] = 0xc6;AES_Sbox_Inv[181] = 0xd2;AES_Sbox_Inv[182] = 0x79;AES_Sbox_Inv[183] = 0x20;AES_Sbox_Inv[184] = 0x9a;AES_Sbox_Inv[185] = 0xdb;AES_Sbox_Inv[186] = 0xc0;AES_Sbox_Inv[187] = 0xfe;AES_Sbox_Inv[188] = 0x78;AES_Sbox_Inv[189] = 0xcd;AES_Sbox_Inv[190] = 0x5a;AES_Sbox_Inv[191] = 0xf4;
AES_Sbox_Inv[192] = 0x1f;AES_Sbox_Inv[193] = 0xdd;AES_Sbox_Inv[194] = 0xa8;AES_Sbox_Inv[195] = 0x33;AES_Sbox_Inv[196] = 0x88;AES_Sbox_Inv[197] = 0x7;AES_Sbox_Inv[198] = 0xc7;AES_Sbox_Inv[199] = 0x31;AES_Sbox_Inv[200] = 0xb1;AES_Sbox_Inv[201] = 0x12;AES_Sbox_Inv[202] = 0x10;AES_Sbox_Inv[203] = 0x59;AES_Sbox_Inv[204] = 0x27;AES_Sbox_Inv[205] = 0x80;AES_Sbox_Inv[206] = 0xec;AES_Sbox_Inv[207] = 0x5f;
AES_Sbox_Inv[208] = 0x60;AES_Sbox_Inv[209] = 0x51;AES_Sbox_Inv[210] = 0x7f;AES_Sbox_Inv[211] = 0xa9;AES_Sbox_Inv[212] = 0x19;AES_Sbox_Inv[213] = 0xb5;AES_Sbox_Inv[214] = 0x4a;AES_Sbox_Inv[215] = 0xd;AES_Sbox_Inv[216] = 0x2d;AES_Sbox_Inv[217] = 0xe5;AES_Sbox_Inv[218] = 0x7a;AES_Sbox_Inv[219] = 0x9f;AES_Sbox_Inv[220] = 0x93;AES_Sbox_Inv[221] = 0xc9;AES_Sbox_Inv[222] = 0x9c;AES_Sbox_Inv[223] = 0xef;
AES_Sbox_Inv[224] = 0xa0;AES_Sbox_Inv[225] = 0xe0;AES_Sbox_Inv[226] = 0x3b;AES_Sbox_Inv[227] = 0x4d;AES_Sbox_Inv[228] = 0xae;AES_Sbox_Inv[229] = 0x2a;AES_Sbox_Inv[230] = 0xf5;AES_Sbox_Inv[231] = 0xb0;AES_Sbox_Inv[232] = 0xc8;AES_Sbox_Inv[233] = 0xeb;AES_Sbox_Inv[234] = 0xbb;AES_Sbox_Inv[235] = 0x3c;AES_Sbox_Inv[236] = 0x83;AES_Sbox_Inv[237] = 0x53;AES_Sbox_Inv[238] = 0x99;AES_Sbox_Inv[239] = 0x61;
AES_Sbox_Inv[240] = 0x17;AES_Sbox_Inv[241] = 0x2b;AES_Sbox_Inv[242] = 0x4;AES_Sbox_Inv[243] = 0x7e;AES_Sbox_Inv[244] = 0xba;AES_Sbox_Inv[245] = 0x77;AES_Sbox_Inv[246] = 0xd6;AES_Sbox_Inv[247] = 0x26;AES_Sbox_Inv[248] = 0xe1;AES_Sbox_Inv[249] = 0x69;AES_Sbox_Inv[250] = 0x14;AES_Sbox_Inv[251] = 0x63;AES_Sbox_Inv[252] = 0x55;AES_Sbox_Inv[253] = 0x21;AES_Sbox_Inv[254] = 0xc;AES_Sbox_Inv[255] = 0x7d;

    int i;
    for(i = 0; i < 16; i++)
        AES_ShiftRowTab_Inv[AES_ShiftRowTab[i]] = i;
    for(i = 0; i < 128; i++) {
        AES_xtime[i] = i << 1;
        AES_xtime[128 + i] = (i << 1) ^ 0x1b;
    }
}

// AES_Done: release memory reserved by AES_Init. 
// Call this function after the last encryption/decryption operation.
void AES_Done() {}

/* AES_ExpandKey: expand a cipher key. Depending on the desired encryption 
strength of 128, 192 or 256 bits 'key' has to be a byte array of length 
16, 24 or 32, respectively. The key expansion is done "in place", meaning 
that the array 'key' is modified.
*/
int AES_ExpandKey(BYTE key[], int keyLen) {
    int kl = keyLen, ks, Rcon = 1, i, j;
    BYTE temp[4], temp2[4];
    switch (kl) {
        case 16: ks = 16 * (10 + 1); break;
        case 24: ks = 16 * (12 + 1); break;
        case 32: ks = 16 * (14 + 1); break;
        default: 
        printf("AES_ExpandKey: Only key lengths of 16, 24 or 32 bytes allowed!");
}
    for(i = kl; i < ks; i += 4) {
        memcpy(temp, &key[i-4], 4);
    if (i % kl == 0) {
        temp2[0] = AES_Sbox[temp[1]] ^ Rcon;
        temp2[1] = AES_Sbox[temp[2]];
        temp2[2] = AES_Sbox[temp[3]];
        temp2[3] = AES_Sbox[temp[0]];
        memcpy(temp, temp2, 4);
        if ((Rcon <<= 1) >= 256)
            Rcon ^= 0x11b;
}
    else if ((kl > 24) && (i % kl == 16)) {
        temp2[0] = AES_Sbox[temp[0]];
        temp2[1] = AES_Sbox[temp[1]];
        temp2[2] = AES_Sbox[temp[2]];
        temp2[3] = AES_Sbox[temp[3]];
        memcpy(temp, temp2, 4);
    }
    for(j = 0; j < 4; j++)
        key[i + j] = key[i + j - kl] ^ temp[j];
    }
    return ks;
}

__global__ void AES_Encrypt_Gen(aes_gen_block * cuda_aes_block_array, KeyBlock cuda_key_block[2], int keyLen, int j, int block_number) {
    int global_thread_index = blockDim.x*blockIdx.x + threadIdx.x;   
    __shared__ BYTE AES_ShiftRowTab[16];
    __shared__ BYTE AES_Sbox[256];
    __shared__ BYTE AES_ShiftRowTab_Inv[16];
    __shared__ BYTE AES_Sbox_Inv[256];
    __shared__ BYTE AES_xtime[256];
    if(global_thread_index < block_number){

        if(threadIdx.x == 0 ){
            AES_Init(AES_Sbox, AES_ShiftRowTab, AES_Sbox_Inv, AES_xtime, AES_ShiftRowTab_Inv);
        }
        __syncthreads();
        BYTE block[16]; 

        for(int k = 0; k < 2; k++){
            for(int i=0; i<16; i++){
                block[i] = cuda_aes_block_array[global_thread_index].block[j][k*16+i];
            }
            int l = keyLen, i;
            //printBytes(block, 16);
            AES_AddRoundKey(block, &cuda_key_block[k].cuda_key[0]);
            for(i = 16; i < l - 16; i += 16) {
                AES_SubBytes(block, AES_Sbox);
                AES_ShiftRows(block, AES_ShiftRowTab);
                AES_MixColumns(block, AES_xtime);
                AES_AddRoundKey(block, &cuda_key_block[k].cuda_key[i]);
            }
            AES_SubBytes(block, AES_Sbox);
            AES_ShiftRows(block, AES_ShiftRowTab);
            AES_AddRoundKey(block, &cuda_key_block[k].cuda_key[i]);
            for(int i=0; i<16; i++){
                cuda_aes_block_array[global_thread_index].block[j][k*16+i] = block[i];
            }
        }
    }
};

__global__ void AES_Encrypt_Eval(aes_eval_block * cuda_aes_block_array, KeyBlock cuda_key_block[2], int keyLen, int block_number) {
    int global_thread_index = blockDim.x*blockIdx.x + threadIdx.x;
   
    __shared__ BYTE AES_ShiftRowTab[16];
    __shared__ BYTE AES_Sbox[256];
    __shared__ BYTE AES_ShiftRowTab_Inv[16];
    __shared__ BYTE AES_Sbox_Inv[256];
    __shared__ BYTE AES_xtime[256];
    if(global_thread_index < block_number){
        if(threadIdx.x == 0 ){
            AES_Init(AES_Sbox, AES_ShiftRowTab, AES_Sbox_Inv, AES_xtime, AES_ShiftRowTab_Inv);
        }
        __syncthreads();
        BYTE block[16]; 
        for(int k = 0; k < 2; k++){
            for(int i=0; i<16; i++){
                block[i] = cuda_aes_block_array[global_thread_index].block[k*16+i];
            }
            int l = keyLen, i;
            AES_AddRoundKey(block, &cuda_key_block[k].cuda_key[0]);
            for(i = 16; i < l - 16; i += 16) {
                AES_SubBytes(block, AES_Sbox);
                AES_ShiftRows(block, AES_ShiftRowTab);
                AES_MixColumns(block, AES_xtime);
                AES_AddRoundKey(block, &cuda_key_block[k].cuda_key[i]);
            }
            AES_SubBytes(block, AES_Sbox);
            AES_ShiftRows(block, AES_ShiftRowTab);
            AES_AddRoundKey(block, &cuda_key_block[k].cuda_key[i]);
            for(int i=0; i<16; i++){
                cuda_aes_block_array[global_thread_index].block[k*16+i] = block[i];
            }
        }
    }
};





// ===================== test ============================================
// int main(int argc, char* argv[]) {
//     int block_number = 1 ;
//     int number_of_zero_pending = 0;
//     aes_block* aes_block_array;

//     BYTE key[16 * (14 + 1)];
//     int keyLen = 16;
//     int blockLen = 16;

//     for(int i = 0; i < blockLen; i++){
//         key[i] = i;
//     }

//     int expandKeyLen = AES_ExpandKey(key, keyLen);
//     printf("expand key length is %d",expandKeyLen);
//     if(number_of_zero_pending != 0)
//         aes_block_array = new aes_block [ block_number + 1];
//     else
//         aes_block_array = new aes_block[ block_number ];
//     for(int i = 0; i < 16; i++)
//         aes_block_array[0].block[i] = 'A';

//     cudaSetDevice(0);	//device 0: Tesla K20c, device 1: GTX 770, device 1 is faster for this application
//     cudaDeviceProp prop;
//     cudaGetDeviceProperties(&prop, 0);
//     int num_sm = prop.multiProcessorCount; 

//     aes_block *cuda_aes_block_array;
//     BYTE *cuda_key;//, *cuda_Sbox;

//     int thrdperblock = block_number/num_sm;
//     if(block_number%num_sm>0)
//         thrdperblock++;

//     if(thrdperblock>1024){
//         thrdperblock = 1024;
//         num_sm = block_number/1024;
//         if(block_number%1024>0){
//             num_sm++;
//         }
//     }
//     dim3 ThreadperBlock(thrdperblock);
//     dim3 BlockperGrid(num_sm);
//     cudaMalloc(&cuda_aes_block_array, block_number*sizeof(class aes_block));
//     cudaMalloc(&cuda_key,16*15*sizeof(BYTE) );
//     cudaMemcpy(cuda_aes_block_array, aes_block_array, block_number*sizeof(class aes_block), cudaMemcpyHostToDevice);
//     cudaMemcpy(cuda_key, key, 16*15*sizeof(BYTE), cudaMemcpyHostToDevice);
//     AES_Encrypt <<< BlockperGrid, ThreadperBlock>>>(cuda_aes_block_array[0].block, cuda_key, expandKeyLen, block_number);
//     cudaMemcpy(aes_block_array, cuda_aes_block_array, block_number*sizeof(class aes_block), cudaMemcpyDeviceToHost);

    
//     return 0;
// }

#endif