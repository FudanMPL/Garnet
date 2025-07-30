import unittest
import subprocess
import re
import os


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
    suite.addTest(Test("test_xgboost_training"))
    # suite.addTest(Test("test_div"))

    runner = unittest.TextTestRunner()
    runner.run(suite)