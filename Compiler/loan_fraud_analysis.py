from Compiler.types import *
from Compiler.sorting import *
from Compiler.group_ops import *


class LoanFraudAnalysis:
    def __init__(self, party_number, partys_sample_number, feature_number, model):
        self.party_number = party_number
        self.partys_sample_number = partys_sample_number
        self.sample_number = sum(partys_sample_number)
        self.feature_number = feature_number
        self.data_before_merge = sint.Matrix(self.sample_number, self.feature_number)
        self.model = model

    def model_setting(self, model):
        self.model = model


    def share_data(self):
        count = 0
        party_id = 0
        for x in self.data_before_merge:
            if count == self.sample_number[self.partys_sample_number[party_id] * self.sample_number]:
                count = 0
                party_id = party_id + 1
            x.input_from(party_id)
            count = count + 1

    def merge_data(self):
        key = self.data_before_merge.get_column(0)
        perm = gen_perm_by_radix_sort(key)
        self.data = sint.Matrix(self.sample_number, self.feature_number)
        for i in range(self.feature_number):
            self.data.set_column(i, perm.apply(self.data_before_merge.get_column(i)))
        flag = sint(0, self.sample_number)
        flag[0] = 1
        key = self.data.get_column(0)
        for i in range(1, self.sample_number):
            flag[i] = key[i] != key[i-1]
        for i in range(self.feature_number):
            self.data.set_column(i, GroupSum(flag, self.data.get_column(i)))
        for i in range(self.sample_number):
            self.data[i] = self.data[i].get_vector() * flag[i]
        perm = SortPerm(1 - flag)
        for i in range(self.feature_number):
            self.data.set_column(i, perm.apply(self.data.get_column(i)))

    def predict_and_reveal(self):
        predict = self.model(self.data)
        res = predict * self.data.get_column(0)
        print(res.reveal())
