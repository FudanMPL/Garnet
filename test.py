k0 = open("/home/txy/Garnet/Player-Data/2-fss/k_conv_relu_0")
k1 = open("/home/txy/Garnet/Player-Data/2-fss/k_conv_relu_1")
r0 = open("/home/txy/Garnet/Player-Data/2-fss/r_conv_relu_0")
r1 = open("/home/txy/Garnet/Player-Data/2-fss/r_conv_relu_1")



def read_file_k(file):
    all_file = file.read().split(" ")
    
    scw = "{"
    vcw = []
    tcw0 = []
    tcw1 = []
    final_cw = 0
    tree_height = int(all_file[0])
    seed = int(all_file[1])
    cnt = 1
    while cnt < (tree_height-1) * 4 + 1:
        scw += "bigint(\"" + all_file[cnt + 1] + "\"),"
        # scw.append(int(all_file[cnt + 1]))
        vcw.append(int(all_file[cnt + 2]))
        tcw0.append(int(all_file[cnt + 3]))
        tcw1.append(int(all_file[cnt + 4]))
        cnt += 4
    scw = scw[:-1] + "}"
    final_cw = int(all_file[-2])
    print(seed)
    print("-------------")
    print(scw)  
    print("-------------")
    print(vcw)
    print("-------------")
    print(tcw0)
    print(tcw1)
    print(final_cw)

def read_file_r(file):
    all_file = file.read().split(" ")
    print("this->reshare_value = bigint(\""+all_file[0]+"\");")
    print("this->r_mask_share = bigint(\""+all_file[1]+"\");")
    print("this->r_drelu_share = "+all_file[2]+";")
    print("this->r_select_share = "+all_file[3]+";")
    print("this->u_select_share = bigint(\""+all_file[4]+"\");")
    print("this->reverse_u_select_share = bigint(\""+all_file[5]+"\");")
    print("this->reverse_1_u_select_share = bigint(\""+all_file[6]+"\");")
    print("this->o_select_share = bigint(\""+all_file[7]+"\");")
    print("this->p_select_share = bigint(\""+all_file[8]+"\");")
    print("this->v_select_share = bigint(\""+all_file[9]+"\");")
    print("this->w_select_share = bigint(\""+all_file[10]+"\");")

read_file_k(k0)
print("-------------")
read_file_k(k1)
print("-------------")
read_file_r(r0)
print("-------------")
read_file_r(r1)