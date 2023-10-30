/*
 * @Author: SkyTu 1336923451@qq.com
 * @Date: 2023-10-24 10:47:57
 * @LastEditors: SkyTu 1336923451@qq.com
 * @LastEditTime: 2023-10-30 21:17:42
 * @FilePath: /txy/Garnet/GPU/gpu.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include "Math/bigint.h"
#include <iostream>
#include "fss_struct.h"

void fss_dpf_generate(FssDpfGenerateBlock * gen_block_cpu, CorrectionWord * cw_cpu, int parallel, int create_size);