from Compiler.types import *
from Compiler.sorting import *
from Compiler.group_ops import *


class LoanFraudFramework:
    def __init__(self, party_number, sample_number_of_parties, feature_number, model):
        self.party_number = party_number
        self.sample_number_of_parties = sample_number_of_parties
        self.sample_number = sum(sample_number_of_parties)
        self.feature_number = feature_number
        self.data_before_merge = sint.Matrix(self.sample_number, self.feature_number)
        self.model = model

    def model_setting(self, model):
        self.model = model


    def input_data_from_parties(self):
        count = 0
        party_id = 0
        for x in self.data_before_merge:
            if count == self.sample_number_of_parties[party_id]:
                count = 0
                party_id = party_id + 1
            x.input_from(party_id)
            count = count + 1

    def merge_data(self):
        key = self.data_before_merge.get_column(0)
        perm = gen_perm_by_radix_sort(key)
        self.data = sint.Matrix(self.sample_number, self.feature_number)
        for i in range(self.feature_number):
            col = perm.apply(self.data_before_merge.get_column(i))
            self.data.set_column(i, col.get_vector())
        flag = sint.Array(size=self.sample_number)
        flag[0] = 1
        key = self.data.get_column(0)
        for i in range(1, self.sample_number):
            flag[i] = key[i] != key[i-1]
        for i in range(1, self.feature_number):
            self.data.set_column(i, GroupSum(flag, self.data.get_column(i)))
        for i in range(self.sample_number):
            self.data[i] = self.data[i].get_vector() * flag[i]
        perm = SortPerm(flag.get_vector().bit_not())
        for i in range(self.feature_number):
            self.data.set_column(i, perm.apply(self.data.get_column(i)).get_vector())

    def predict_and_reveal(self):
        predict = sint.Array(self.sample_number)
        for i in range(self.sample_number):
            predict[i] = self.model(self.data[i])
        pred_res = predict * self.data.get_column(0)
        perm = SortPerm(predict.get_vector().bit_not())
        pred_res = perm.apply(pred_res)
        res = pred_res.reveal()
        print_str("fraud company id: ")
        for id in res:
            @if_(id != 0)
            def _():
                print_str("%s ", id)
        print_ln("")
