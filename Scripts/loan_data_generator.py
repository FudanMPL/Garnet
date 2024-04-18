import random

party_number = 3
sample_number_of_parties = [
0,
10,
20]
feature_number = 4 # 四个属性假设分别为：公司id，公司的坏账数，公司坏账贷款总金额，公司坏账贷款未还总金额

max_sample_id = 100

max_feature_value = [
    max_sample_id,
    30,
    1000000,
    1000000
]


def generate_train_data():
    train_samples = 100
    f = open("Player-Data/Input-P0-0", 'w')
    for i in range(train_samples):
        tmp = random.randint(0, 1)
        f.write(str(tmp))
        if i != train_samples - 1:
            f.write(' ')
    f.write("\n")
    for k in range(feature_number):
        for i in range(train_samples):
            tmp = random.randint(1, max_feature_value[k])
            f.write(str(tmp))
            if i != train_samples - 1:
                f.write(' ')
        f.write("\n")

    f.close()


def generate_test_data(party_id):
    if sample_number_of_parties[party_id] == 0:
        return
    f = open("Player-Data/Input-P"+str(party_id)+"-0", 'w')
    for i in range(sample_number_of_parties[party_id]):
        for k in range(feature_number):
            tmp = random.randint(1, max_feature_value[k])
            f.write(str(tmp))
            if k != feature_number - 1:
                f.write(' ')
        f.write("\n")
    f.close()



if __name__ == '__main__':
    generate_train_data()
    # for i in range(party_number):
    #     generate_test_data(i)
