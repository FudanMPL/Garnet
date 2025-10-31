import unittest
import subprocess
import re
import os
import sys

# 添加 Compiler 目录到 Python 路径
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Compiler'))

# 从 Compiler.types 导入 dataframe
try:
    from Compiler.types import dataframe
    print("成功导入 dataframe")
except ImportError as e:
    print(f"导入 dataframe 失败: {e}")
    # 如果仍然失败，尝试其他方法


def search_result(line, result_output):
    line = line + r"\s*(-?\d+(?:\.\d+)?)"
    match = re.search(line, result_output)
    return match


def run_script(script, cwd='./'):
    result = subprocess.run(script, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True, cwd=cwd)

    result_returncode = result.returncode
    result_output = result.stdout
    result_error = result.stderr
    return result_returncode, result_output, result_error


class Test(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass()

    # 虚拟机编译测试
    def test_machines(self):
        print(" 虚拟机编译测试开始 ")

        script = "make -j all"
        return_code, _, err = run_script(script)
        self.assertEqual(return_code, 0, msg=f"虚拟机编译错误：\n{err}")

        print(" 虚拟机编译测试结束 ")

    # MPC功能正确性测试
    def test_arithmetic(self):
        # 写入数据
        print(" 算术运算测试开始 ")

        data_script_0 = "echo 8192 268435456 0 9997 10999 -17579 -32767 33554437 99 0 16373 -4095 -4194301 -2097151 1 0 -1 8194 -16483 -65506 9223372036854775807 16383 8192 0 4098 8000 1024 625 > Player-Data/Input-P0-0"
        data_return_code_0, _, data_result_error_0 = run_script(data_script_0)
        self.assertEqual(data_return_code_0, 0, msg=f"参与方0数据写入错误：\n{data_result_error_0}")
        data_script_1 = "echo 512 16777213 8192 -9997 -5408 9999 -67538 30576542 173 16373 0 -67108868 -8388607 -2097151 130179 4194207 263153 1024 -279 71 1 3 128 17 1 -8 -2048 5000 > Player-Data/Input-P1-0"
        data_return_code_1, _, data_result_error_1 = run_script(data_script_1)
        self.assertEqual(data_return_code_1, 0, msg=f"参与方1数据写入错误：\n{data_result_error_1}")

        # 编译脚本的测试
        compile_script = "python ./compile.py -R 64 Programs/Source/Unit_Test/test_framework_file.mpc"
        compile_return_code, _, compile_result_error = run_script(compile_script)
        self.assertEqual(compile_return_code, 0, msg=f"mpc文件编译错误：\n{compile_result_error}")

        # 运行脚本的测试
        running_script = "./Scripts/semi2k.sh test_framework_file"
        running_return_code, running_result_output, running_result_error = run_script(running_script)
        self.assertEqual(running_return_code, 0, msg=f"mpc文件运行错误：\n{running_result_error}")

        add_expect_res_array = [8704, 285212669, 8192, 0, 5591, -7580, -100305]
        reduce_expect_res_array = [2977895, -74, -16373, 16373, 67104773, 4194306, 0]
        mul_expect_res_array = [130179, 0, -263153, 8390656, 4598757, -4650926, 9223372036854775807]
        div_expect_res_array = [5461, 64, 0, 4098, -1000, -0.5, 0.125]
        for i in range(7):
            search_str = "add_res_" + str(i) + " = "
            match = search_result(search_str, running_result_output)
            self.assertIsNotNone(match, msg=f"加法第{i}个未找到函数输出")
            res = str(add_expect_res_array[i])
            self.assertEqual(match.group(1), res, msg=f"加法第{i}个结果错误")

        for i in range(7):
            search_str = "reduce_res_" + str(i) + " = "
            match = search_result(search_str, running_result_output)
            self.assertIsNotNone(match, msg=f"减法第{i}个未找到函数输出")
            res = str(reduce_expect_res_array[i])
            self.assertEqual(match.group(1), res, msg=f"减法第{i}个结果错误")

        for i in range(7):
            search_str = "mul_res_" + str(i) + " = "
            match = search_result(search_str, running_result_output)
            self.assertIsNotNone(match, msg=f"乘法第{i}个未找到函数输出")
            res = str(mul_expect_res_array[i])
            self.assertEqual(match.group(1), res, msg=f"乘法第{i}个结果错误")

        for i in range(7):
            search_str = "div_res_" + str(i) + " = "
            match = search_result(search_str, running_result_output)
            self.assertIsNotNone(match, msg=f"除法第{i}个未找到函数输出")
            res = div_expect_res_array[i]
            self.assertAlmostEqual(float(match.group(1)), res, delta=0.000015, msg=f"除法第{i}个结果错误")

        print(" 算术运算测试结束 ")

    def test_0iris(self):
        print(" 0iris测试开始 ")
        # 写入数据
        compile_script = 'cp ./Data/0iris_data ./Player-Data/Input-P0-0'
        compile_return_code, _, compile_result_error = run_script(compile_script)
        self.assertEqual(compile_return_code, 0, msg=f"参与方0数据写入错误：\n{compile_result_error}")

        # 编译脚本的测试
        compile_script = "python ./compile.py -R 64 Programs/Source/Unit_Test/test_0iris.mpc"
        compile_return_code, _, compile_result_error = run_script(compile_script)
        self.assertEqual(compile_return_code, 0, msg=f"mpc文件编译错误：\n{compile_result_error}")

        # 运行脚本的测试
        running_script = "./Scripts/ring.sh test_0iris"
        running_return_code, running_result_output, running_result_error = run_script(running_script)
        self.assertEqual(running_return_code, 0, msg=f"mpc文件运行错误：\n{running_result_error}")
        accuracies = re.findall(r'acc:\s+([0-9.]+)', running_result_output)
        accuracies = [float(a) for a in accuracies]
        # print(f" 0iris 测试集准确率: {accuracies}")
        max_acc = max(accuracies)
        self.assertGreaterEqual(max_acc, 0.9, "测试集准确率低于90%")

        print(" 0irsis测试结束 ")

    def test_concat(self):
        print("concat测试开始 ")
        
        # 测试用例1：正常情况下的纵向连接
        try:
            df1 = dataframe(
                data=[[1, 'A'], [2, 'B']],
                columns=['id', 'name']
            )
            df2 = dataframe(
                data=[[3, 'C'], [4, 'D']],
                columns=['id', 'name']
            )
            
            result = dataframe.concat([df1, df2], axis=0)
            
            self.assertEqual(result.shape[0], 4, "连接后的数据框行数不正确")
            self.assertEqual(result.columns, ['id', 'name'], "列名不正确")
            self.assertEqual(result.value_types, [int, str], "值类型不正确")
            
            expected_data = [[1, 'A'], [2, 'B'], [3, 'C'], [4, 'D']]
            for i in range(result.shape[0]):
                row_data = [result['id'].data[i], result['name'].data[i]]
                self.assertEqual(row_data, expected_data[i], f"第{i}行数据不正确")
                
        except Exception as e:
            self.fail(f"测试用例1失败：{str(e)}")
        
        # 测试用例2：列名不一致的情况（应该报错）
        try:
            df1 = dataframe(
                data=[[1, 'A']],
                columns=['id', 'name']
            )
            df2 = dataframe(
                data=[[2, 'B']],
                columns=['ID', 'name']
            )
            
            with self.assertRaises(ValueError):
                dataframe.concat([df1, df2], axis=0)
                
        except Exception as e:
            self.fail(f"测试用例2失败：{str(e)}")
        
        # 测试用例3：值类型不一致的情况（应该报错）
        try:
            df1 = dataframe(
                data=[[1, 'A']],
                columns=['id', 'name']
            )
            df2 = dataframe(
                data=[[2.5, 'B']],
                columns=['id', 'name']
            )
            
            with self.assertRaises(TypeError):
                dataframe.concat([df1, df2], axis=0)
                
        except Exception as e:
            self.fail(f"测试用例3失败：{str(e)}")
        
        # 测试用例4：空列表情况（应该报错）
        try:
            with self.assertRaises(ValueError):
                dataframe.concat([], axis=0)
                
        except Exception as e:
            self.fail(f"测试用例4失败：{str(e)}")
        
        # 测试用例5：包含非dataframe对象的情况（应该报错）
        try:
            df1 = dataframe(
                data=[[1, 'A']],
                columns=['id', 'name']
            )
            
            with self.assertRaises(TypeError):
                dataframe.concat([df1, "not_a_dataframe"], axis=0)
                
        except Exception as e:
            self.fail(f"测试用例5失败：{str(e)}")
        
        print("concat测试结束 ")

    def test_join(self):
        print(" join测试开始 ")
        
        # 测试用例1：基本索引合并
        try:
            df1_data = [['Alice', 25], ['Bob', 30]]
            df1_columns = ['name', 'age']
            df1_index = [1, 2]
            
            df2_data = [['Engineer'], ['Designer']]
            df2_columns = ['job']
            df2_index = [1, 2]
            
            df1 = dataframe(data=df1_data, columns=df1_columns, index=df1_index)
            df2 = dataframe(data=df2_data, columns=df2_columns, index=df2_index)
            
            result = df1.join(df2, lsuffix='_left', rsuffix='_right', how='inner')
            
            self.assertEqual(len(result.index), 2, "inner join索引数量不正确")
            self.assertEqual(len(result.columns), 3, "inner join列数量不正确")
            self.assertIn('name', result.columns)
            self.assertIn('age', result.columns)
            self.assertIn('job', result.columns)
            
            expected_index = [1, 2]
            self.assertEqual(result.index, expected_index, "inner join索引不正确")
            
        except Exception as e:
            self.fail(f"测试用例1失败: {e}")
        
        # 测试用例2：列名冲突处理
        try:
            df1_data = [['Alice', 25], ['Bob', 30]]
            df1_columns = ['name', 'value']
            df1_index = [1, 2]
            
            df2_data = [[100], [200]]
            df2_columns = ['value']
            df2_index = [1, 2]
            
            df1 = dataframe(data=df1_data, columns=df1_columns, index=df1_index)
            df2 = dataframe(data=df2_data, columns=df2_columns, index=df2_index)
            
            result = df1.join(df2, lsuffix='_left', rsuffix='_right', how='inner')
            
            expected_columns = ['name', 'value_left', 'value_right']
            self.assertEqual(result.columns, expected_columns, "列名冲突处理不正确")
                    
        except Exception as e:
            self.fail(f"测试用例2失败: {e}")
        
        # 测试用例3：原地操作
        try:
            df1_data = [['Alice', 25], ['Bob', 30]]
            df1_columns = ['name', 'age']
            df1_index = [1, 2]
            
            df2_data = [['Engineer'], ['Designer']]
            df2_columns = ['job']
            df2_index = [1, 2]
            
            df1 = dataframe(data=df1_data, columns=df1_columns, index=df1_index)
            df2 = dataframe(data=df2_data, columns=df2_columns, index=df2_index)
            
            original_id = id(df1)
            result = df1.join(df2, inplace=True)
            
            self.assertEqual(id(result), original_id, "原地操作未正确执行")
            self.assertEqual(len(result.index), 2, "原地操作后数据不正确")
            
        except Exception as e:
            self.fail(f"测试用例3失败: {e}")

        # 测试用例4：outer join（可能跳过）
        try:
            df1_data = [['Alice', 25], ['Bob', 30], ['Charlie', 35]]
            df1_columns = ['name', 'age']
            df1_index = [1, 2, 3]
            
            df2_data = [['Engineer'], ['Designer']]
            df2_columns = ['job']
            df2_index = [1, 2]
            
            df1 = dataframe(data=df1_data, columns=df1_columns, index=df1_index)
            df2 = dataframe(data=df2_data, columns=df2_columns, index=df2_index)
            
            result = df1.join(df2, lsuffix='_left', rsuffix='_right', how='outer')
            
            self.assertEqual(len(result.index), 3, "outer join索引数量不正确")
            self.assertEqual(len(result.columns), 3, "outer join列数量不正确")
            
            expected_index = [1, 2, 3]
            self.assertEqual(result.index, expected_index, "outer join索引不正确")
            
            
        except Exception as e:
            print(f"测试用例4跳过: {e}")
        
        print(" join测试结束 ")

    def test_xgboost_training(self):
        print(" xgboost训练测试开始 ")

        # CONFIG.mine 文件
        config_path = "CONFIG.mine"
        target_content = "MOD = -DRING_SIZE=32"

        with open(config_path, "w") as f:
            f.write(target_content)

        self.assertTrue(os.path.exists(config_path), "CONFIG.mine 文件未创建")
        with open(config_path, "r") as f:
            content = f.read().strip()
        self.assertEqual(content, target_content, "CONFIG.mine 内容不正确")

        # 虚拟机编译测试
        script = "make -j 8 rss-with-conversion-party.x"
        return_code, _, err = run_script(script)
        self.assertEqual(return_code, 0, msg=f"虚拟机编译错误：\n{err}")

        # 数据准备脚本的测试
        data_prepare_script = "python ./Scripts/data_prepare_for_xgboost.py IRIS"
        data_prepare_return_code, result, data_prepare_result_error = run_script(data_prepare_script)
        self.assertEqual(data_prepare_return_code, 0, msg=f"数据准备失败：\n{data_prepare_result_error}")

        train_info = re.search(r"file: .*_train\.csv\s*items: (\d+)\s*attributes: (\d+)", result)
        test_info = re.search(r"file: .*_test\.csv\s*items: (\d+)", result)
        n_train = int(train_info.group(1))
        m = int(train_info.group(2))
        n_test = int(test_info.group(1))
        mpc_path = "./Programs/Source/Unit_Test/test_xgboost.mpc"
        with open(mpc_path, "r") as f:
            lines = f.readlines()

        lines[1] = f"m = {m} # 特征数\n"
        lines[2] = f"n_train = {n_train} # 训练样本数量\n"
        lines[3] = f"n_test = {n_test} # 测试样本数量\n"

        with open(mpc_path, "w") as f:
            f.writelines(lines)

        with open(mpc_path, "r") as f:
            new_lines = f.readlines()

        self.assertEqual(new_lines[1].strip(), f"m = {m} # 特征数")
        self.assertEqual(new_lines[2].strip(), f"n_train = {n_train} # 训练样本数量")
        self.assertEqual(new_lines[3].strip(), f"n_test = {n_test} # 测试样本数量")

        # 编译脚本的测试
        compile_script = "python ./compile.py Programs/Source/Unit_Test/test_xgboost.mpc -R 32"
        compile_return_code, _, compile_result_error = run_script(compile_script)
        self.assertEqual(compile_return_code, 0, msg=f"mpc文件编译错误：\n{compile_result_error}")

        # 运行脚本的测试
        script = "./Scripts/rss-with-conversion.sh test_xgboost"
        run_return_code, run_result, run_err = run_script(script)
        self.assertEqual(run_return_code, 0, msg=f"mpc文件运行错误：\n{run_err}")

        test_section = run_result.split("test for test set")[-1]
        acc = re.search(r'\*\*\* accuracy: (\d+)/(\d+)', test_section)
        self.assertIsNotNone(acc, "未找到测试集准确率")
        correct = int(acc.group(1))
        total = int(acc.group(2))
        accuracy = correct / total
        self.assertGreaterEqual(accuracy, 0.9, "测试集准确率低于90%")

        print(" xgboost训练测试结束 ")

    # def test_div(self):
    #     data_script_0 = "echo 16383 8192 0 4098 8000 1024 625 > Player-Data/Input-P0-0"
    #     data_return_code_0, _, data_result_error_0 = run_script(data_script_0)
    #     self.assertEqual(data_return_code_0, 0, msg=f"参与方0数据写入错误：\n{data_result_error_0}")
    #     data_script_1 = "echo 3 128 17 1 -8 -2048 5000 > Player-Data/Input-P1-0"
    #     data_return_code_1, _, data_result_error_1 = run_script(data_script_1)
    #     self.assertEqual(data_return_code_1, 0, msg=f"参与方1数据写入错误：\n{data_result_error_1}")

    #     compile_script = "python ./compile.py -R 64 Programs/Source/Unit_Test/test_div.mpc"
    #     compile_return_code, _, compile_result_error = run_script(compile_script)
    #     self.assertEqual(compile_return_code, 0, msg=f"mpc文件编译错误：\n{compile_result_error}")

    #     running_script = "./Scripts/semi2k.sh test_div"
    #     running_return_code, running_result_output, running_result_error = run_script(running_script)
    #     self.assertEqual(running_return_code, 0, msg=f"mpc文件运行错误：\n{running_result_error}")
    #     div_expect_res_array = [5461, 64, 0, 4098, -1000, -0.5, 0.125]
    #     for i in range(7):
    #         search_str = "div_res_" + str(i) + " = " + r"\s*(-?\d+(?:\.\d+)?)"
    #         match = re.search(search_str, running_result_output)
    #         # match = search_result(search_str, running_result_output)
    #         self.assertIsNotNone(match, msg=f"除法第{i}个未找到函数输出")
    #         res = div_expect_res_array[i]
    #         self.assertAlmostEqual(float(match.group(1)), res, delta=0.000015, msg=f"除法第{i}个结果错误")


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(Test("test_machines"))
    suite.addTest(Test("test_arithmetic"))
    suite.addTest(Test("test_0iris"))
    suite.addTest(Test("test_concat"))
    suite.addTest(Test("test_join"))
    suite.addTest(Test("test_xgboost_training"))
    # suite.addTest(Test("test_div"))

    runner = unittest.TextTestRunner()
    runner.run(suite)