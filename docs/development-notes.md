## 开发须知

在主目录下，包含一个文件名为 UnitTest_Framework.py 的单元测试框架，可以在框架中添加名称以 test 开头的单元测试函数，自动化地测试开发内容。

### 一、单元测试
开发者在开发完代码后，需要执行以下两个步骤对自己开发的代码进行测试。

#### 步骤一

首先，单元测试框架中现已包含了若干单元测试，开发者需要先确保现有的单元测试能够顺利通过。现有的单元测试包括所有虚拟机的编译测试 test_machines、semi2k 虚拟机中的加减乘除测试 test_arithmetic、训练Iris数据集的神经网络测试 test_0iris 和 XGBoost 模型安全训练的测试 test_xgboost_training。可以直接通过以下命令运行现有的单元测试：
```
python UnitTest_Framework.py
```
运行现有测试大约需要十分钟，若测试成功，将会出现类似输出：
```
 虚拟机编译测试开始 
 虚拟机编译测试结束 
. 算术运算测试开始 
 算术运算测试结束 
. 0iris测试开始 
 0irsis测试结束 
. xgboost训练测试开始 
 xgboost训练测试结束 
.
----------------------------------------------------------------------
Ran 4 tests in 557.550s

OK
```
若测试失败，需要按照系统报错对 bug 进行修复。

#### 步骤二

第二个步骤是在单元测试框架中给自己的开发内容添加单元测试函数，在测试中需要给出至少3组测试数据，能够覆盖通常情况以及边界条件。
接下来以除法的单元测试函数 test_div 和 XGBoost 模型安全训练的单元测试函数 test_xgboost_training 为例，说明如何针对新的开发内容撰写一个单元测试。

##### test_div

假设现在需要测试 semi2k 虚拟机能否正确执行除法操作，现已准备了七组测试数据，执行七次除法操作的 mpc 文件如下：
```
div_array = sfix.Array(14)
for i in range(14):
    if i % 2 == 0 :div_array[i] = sfix.get_input_from(0)
    else: div_array[i] = sfix.get_input_from(1)

div_res_array = sfix.Array(7)
@for_range(7)
def _(i):
    div_res_array[i] = div_array[2 * i] / div_array[2 * i + 1]
    print_ln("div_res_%s = %s", i, div_res_array[i].reveal())
```
除法的执行可以分为三步：数据准备、编译和运行，在 test_div 中可为每个步骤设置一个断言来检查预期行为是否成立。
首先是数据准备，需要将测试数据写入 Player-Data 目录下的 Input-P0-0 和 Input-P1-0。框架提供 run_script 函数运行脚本，该函数接收 string 类型的输入（即脚本），返回状态码、标准输出和标准错误输出。可以在单元测试函数 test_div 中通过 run_script 运行写入数据的命令，并通过断言检查状态码从而判断数据写入是否成功：
```
data_script_0 = "echo 16383 8192 0 4098 8000 1024 625 > Player-Data/Input-P0-0"
data_return_code_0, _, data_result_error_0 = run_script(data_script_0)
self.assertEqual(data_return_code_0, 0, msg=f"参与方0数据写入错误：\n{data_result_error_0}")
data_script_1 = "echo 3 128 17 1 -8 -2048 5000 > Player-Data/Input-P1-0"
data_return_code_1, _, data_result_error_1 = run_script(data_script_1)
self.assertEqual(data_return_code_1, 0, msg=f"参与方1数据写入错误：\n{data_result_error_1}")
```
然后使用 run_script 运行编译上述 mpc 文件的命令，通过断言检查状态码从而判断编译是否成功：
```
compile_script = "python ./compile.py -R 64 Programs/Source/Unit_Test/test_div.mpc"
compile_return_code, _, compile_result_error = run_script(compile_script)
self.assertEqual(compile_return_code, 0, msg=f"mpc文件编译错误：\n{compile_result_error}")
```
最后是运行，同样调用 run_script 函数运行命令。此时，需要结合具体测试内容，决定断言检查的内容。该例子是测试除法的执行结果是否正确，所以需要通过 python 的正则表达式从标准输出中获取七个除法的执行结果，并依次与正确结果进行比对。在 semi2k 虚拟机中，得到的除法的执行结果可能与正确结果相差 0.000015，因此断言中需要考虑该误差。
```
running_script = "./Scripts/semi2k.sh test_div"
running_return_code, running_result_output, running_result_error = run_script(running_script)
self.assertEqual(running_return_code, 0, msg=f"mpc文件运行错误：\n{running_result_error}")

div_expect_res_array = [5461, 64, 0, 4098, -1000, -0.5, 0.125]
for i in range(7):
     search_str = "div_res_" + str(i) + " = " + r"\s*(-?\d+(?:\.\d+)?)"
     match = re.search(search_str, running_result_output)
     self.assertIsNotNone(match, msg=f"除法第{i}个未找到函数输出")
     res = div_expect_res_array[i]
     self.assertAlmostEqual(float(match.group(1)), res, delta=0.000015, msg=f"除法第{i}个结果错误")
```
添加了 test_div 单元测试函数后，需要在 main 函数中添加如下代码，再次运行 python UnitTest_Framework.py 后，test_div 会自动化地执行。
```
suite.addTest(Test("test_div"))
```

##### test_xgboost_training

运行XGBoost模型安全训练的具体步骤可以查看xgboost-training.md。运行XGBoost模型安全训练分为环境配置、数据准备、脚本配置、编译和运行五步，每个步骤均可设置一个断言来检查预期行为是否成立。

首先是环境配置，需要修改（或新建）CONFIG.mine文件，在开头加入一行代码：
```
MOD = -DRING_SIZE=32
```
可以通过 python 语句进行上述配置，并通过断言检查是否配置正确：
```
config_path = "CONFIG.mine"
target_content = "MOD = -DRING_SIZE=32"

with open(config_path, "w") as f:
        f.write(target_content)

self.assertTrue(os.path.exists(config_path), "CONFIG.mine 文件未创建")
with open(config_path, "r") as f:
        content = f.read().strip()
self.assertEqual(content, target_content, "CONFIG.mine 内容不正确")
```
之后还需要编译所需的虚拟机，并测试虚拟机是否编译成功。此处以 rss-with-conversion-party.x 为例：
```
script = "make -j 8 rss-with-conversion-party.x"
return_code, _, err = run_script(script)
self.assertEqual(return_code, 0, msg=f"虚拟机编译错误：\n{err}")
```
接下来是数据准备，需要下载IRIS训练集至Data目录，然后运行如下命令处理数据集。
```
python ./Scripts/data_prepare_for_xgboost.py IRIS
```
可以通过 run_script 函数运行上述命令，并通过断言检查状态码从而判断数据准备是否成功：
```
data_prepare_script = "python ./Scripts/data_prepare_for_xgboost.py IRIS"
data_prepare_return_code, result, data_prepare_result_error = run_script(data_prepare_script)
self.assertEqual(data_prepare_return_code, 0, msg=f"数据准备失败：\n{data_prepare_result_error}")
```
在脚本配置中，需要根据上述数据准备的命令的输出信息修改 mpc 文件，可以通过 python 的正则表达式从 run_script 获取的标准输出中匹配输出信息对 mpc 文件进行修改，最后通过断言检查脚本配置是否正确：
```
train_info = re.search(r"file: .*_train\.csv\s*items: (\d+)\s*attributes: (\d+)", result)
test_info = re.search(r"file: .*_test\.csv\s*items: (\d+)", result)
n_train = int(train_info.group(1))
m = int(train_info.group(2))
n_test = int(test_info.group(1))
mpc_path = "./Programs/Source/Unit_Test/xgboost.mpc"
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
```
在编译阶段，直接调用 run_script 函数运行编译命令，通过断言检查编译命令状态码从而判断编译是否成功即可：
```
compile_script = "python ./compile.py Programs/Source/Unit_Test/xgboost.mpc -R 32"
compile_return_code, _, compile_result_error = run_script(compile_script)
self.assertEqual(compile_return_code, 0, msg=f"mpc文件编译错误：\n{compile_result_error}")
```
最后是运行，该例子是对XGBoost模型安全训练的测试，因此最后以测试集的准确率为标准来检查训练过程。同样可以通过 python 的正则表达式从标准输出中得到测试集的准确率，然后通过断言检查测试集准确率是否大于90%：
```
script = "./Scripts/rss-with-conversion.sh xgboost"
run_return_code, run_result, run_err = run_script(script)
self.assertEqual(run_return_code, 0, msg=f"mpc文件运行错误：\n{run_err}")

test_section = run_result.split("test for test set")[-1]
acc = re.search(r'\*\*\* accuracy: (\d+)/(\d+)', test_section)
self.assertIsNotNone(acc, "未找到测试集准确率")
correct = int(acc.group(1))
total = int(acc.group(2))
accuracy = correct / total
self.assertGreaterEqual(accuracy, 0.9, "测试集准确率低于90%")
```
添加了 test_xgboost_training 函数后，需要在 main 函数中添加如下代码：
```
suite.addTest(Test("test_xgboost_training"))
```

### 二、待补充