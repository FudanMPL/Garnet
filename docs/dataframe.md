# 明文下的表结构dataframe

我们在Garnet中增加了明文下的表结构`dataframe`，支持的数据类型为`cint/cifx/sint/sfix`。相关数据结构、功能及使用方法如下：

## series

`series`是一个带标签的一维数组，用于表示表结构`dataframe`的一行或者一列。

### 构造

其构造方法为调用：
```
s = series(data, index=index, name=name)
```
其中，`data`为一个一维列表；`index`为一个一维列表，默认为`None`；`name`为一个字符串，默认为`None`，为`None`时表示一个行，否则为一个列。例如，创建一个表示行的实例时：
```
row_s = series(data=[cint(1), sfix(1000.0)], index=['ID', 'Salary'])
## row_s
# ID       1
# Salary   1000.0 
```
创建一个表示列的实例时：
```
col_s = series(data=[cint(170), cint(160)], name='Height')
## col_s
# 0   170
# 1   60 
```

### 索引
`index`为`series`的索引，可以根据索引获取`series`相应位置的值。若不传入`index`，则索引从0开始。例如：
```
## row_s['Salary'].reveal()
# 1000.0
## col_s[0]
# 170
```

## dataframe

`dataframe`是二维带标签的数据结构，类似于SQL数据库中的表，仿照Python中Pandas库的Dataframe编写。可能包含具有不同数据类型的列，每一个列的数据的类型相同。

### 构造

其构造方法为调用：
```
df = dataframe(data, columns=columns, index=index)
```
其中，`data`为一个二维列表，`data`中的每一个一维列表表示`df`的一行；`columns`为一个一维列表；`index`为一个一维列表，默认为`None`。例如：
```
ID1 = cint(1)
ID2 = cint(2)
ID3 = cint(3)
salary1 = sfix(1000.0)
salary2 = sfix(1500.0)
salary3 = sfix(800.0)

data = [[ID1, salary1], [ID2, salary2], [ID3, salary3]]
df = dataframe(data, ['ID', 'salary'])
## df
#    ID   Salary
# 0   1   1000.0
# 1   2   1500.0
# 2   3    800.0
```

### 查

可以使用`dataframe`包含的列名作为索引获取`dataframe`的一列，返回一个表示列的`series`。例如：
```
## df['Salary']
# 0   1000.0
# 1   1500.0
# 2    800.0
```
若传入多个列名组成的列表来获取多列，则返回一个`dataframe`。例如：
```
## df[['ID', 'Salary']]
#    ID   Salary
# 0   1   1000.0
# 1   2   1500.0
# 2   3    800.0
```
另一方面，可以使用`loc[]`来获取`dataframe`的一行，返回一个表示行的`series`，索引为`dataframe`的列名。例如：
```
## df.loc[0]
# ID       1
# Salary   1000.0 
```
若传入多个整数组成的列表来获取多行，则返回一个`dataframe`。例如：
```
## df.loc[[0, 2]]
#    ID   Salary
# 0   1   1000.0
# 2   3    800.0
```
若传入一个切片来获取一行或多行，则返回一个`dataframe`。例如：
```
## df.loc[0:1]
#    ID   Salary
# 0   1   1000.0
# 1   2   1500.0
```
对于一个单元格的数据，可以用如下方法获取：
```
## df['Salary'][2].reveal()
# 800.0
```

### 增

可以使用`df['new_column'] = column_data`的方式向`dataframe`添加一列数据，例如：
```
df['Age'] = [sint(25), sint(40), sint(37)]
## df
#    ID   Salary   Age
# 0   1   1000.0    25
# 1   2   1500.0    40
# 2   3    800.0    37
```
类似地，可以使用`loc[]`按照上述方式向`dataframe`添加一行数据，例如：
```
df.loc[3] = [cint(4), sfix(2000.0)]
## df
#    ID   Salary 
# 0   1   1000.0 
# 1   2   1500.0 
# 2   3    800.0
# 3   4   2000.0
```
此外，可以使用`df[['new_column1', 'new_column2', ...]] = [row1_data, row2_data, ...]`的方式向`dataframe`添加多列数据，例如：
```
age1 = sint(25)
age2 = sint(40)
age3 = sint(37)
height1 = cint(175)
height2 = cint(170)
height3 = cint(160)

df[['Age', 'Height']] = [[age1, height1], [age2, height2], [age3, height3]]
## df
#    ID   Salary   Age   Height
# 0   1   1000.0    25      175
# 1   2   1500.0    40      170
# 2   3    800.0    37      160
```

### 改

可以使用`df['column'][row_index] = new_data`的方式修改`dataframe`相应单元格的数据，例如：
```
df['Salary'][0] = sfix(5000.0)
## df['Salary'][0].reveal()
# 5000.0
```
可以使用`df['column'] = new_column_data`的方式修改`dataframe`的一列数据，例如：
```
df['ID'] = [cint(4), cint(3), cint(2)]
## df
#    ID   Salary 
# 0   4   1000.0 
# 1   3   1500.0 
# 2   2    800.0
```
可以使用`df['row'] = new_row_data`的方式修改`dataframe`的一行数据，例如：
```
df.loc[2] = [cint(5), sfix(2000.0)]
## df
#    ID   Salary 
# 0   1   1000.0 
# 1   2   1500.0 
# 2   5   2000.0
```

### 删

可以调用`drop()`方法实现删除`dataframe`的若干行或列：
```
def drop(self, index=None, column=None, inplace=False)
```
其中，`index`可以为一个整数或一个整数的列表，默认为`None`；`column`可以为一个字符串或字符串的列表，默认为`None`；`inplace`为布尔值，当`inplace`为`True`时，表示直接对调用`drop()`方法的对象进行删除操作并返回，否则返回一个新的`dataframe`对象，与调用`drop()`函数的对象进行删除操作后相同。例如：
```
new_df = df.drop(index=[0, 1], inplace=True)
## df
#    ID   Salary
# 2   3    800.0
## new_df
#    ID   Salary
# 2   3    800.0

new_df = df.drop(column='Salary', inplace=False)
## df
#    ID   Salary
# 0   1   1000.0
# 1   2   1500.0
# 2   3    800.0
## new_df
#    ID
# 0   1
# 1   2
# 2   3
```

### 合并

可以调用`merge()`方法实现多个`dataframe`的求并或求交：
```
def merge(self, obj, on, join='outer', inplace=False)
```
其中，`obj`可以为一个`dataframe`或一个`dataframe`的列表；`on`为一个字符串，需为一个列名，表示基于该列标识表中的一行；`join`为一个字符串，取值为`outer`或`inner`，表示求并集还是求交集，默认为`outer`；`inplace`为布尔值，当`inplace`为`True`时，表示直接在调用`merge()`方法的对象上进行求并或求交并返回，否则返回一个新的`dataframe`对象作为求并或求交的结果。此外，还会返回一个`num`，表示并集/交集的行数，当`on`的类型为`cint/cfix`时，`num`的类型为`cint`；当`on`的类型为`sint/sfix`时，`num`的类型为`sint`。例如：
```
## df1
#    ID   Salary
# 0   1   1000.0
# 1   2   1500.0
# 2   3    800.0
## df2
#    ID   Age
# 0   2    37
# 1   4    40
## df3
#    ID   Height
# 0   2    170
# 1   5    160

df_outer_join, num = df1.merge(obj=[df2, df3], on='ID', join='outer', inplace=False)
## df_outer_join
#    ID   Salary   Age   Height
# 0   1   1000.0     0        0
# 1   2   1500.0    37      170
# 2   3    800.0     0        0
# 3   4        0    40        0
# 4   5        0     0      160
## num
# 5

df_inner_join, num = df1.merge(obj=[df2, df3], on='ID', join='inner', inplace=False)
## df_inner_join
#    ID   Salary   Age   Height
# 0   2   1500.0    37      170
## num
# 1
```
若`on`一列的数据类型为`sint/sfix`，且合并的几个`dataframe`中包括类型为`cint/cfix`的列，则不能进行合并，系统将进行报错提示。

## 场景示例

在两方半诚实的场景下，假设P0和P1两方各自拥有一批相同用户的不同信息，我们需要对这些信息进行汇总。

### mpc代码

假设P0的数据存于Player-Data/Input-P0-0中：
```
1 2 3
39 25 20
60 72 50
```
P1的数据存于Player-Data/Input-P1-0中：
```
1 2 3
4500.5 5000.3 3000.4
11 4 1
```
在Programs/Source/test_dataframe.mpc中编写代码：
```
P0_ID0 = sint.get_input_from(0)
P0_ID1 = sint.get_input_from(0)
P0_ID2 = sint.get_input_from(0)
P0_Age0 = sint.get_input_from(0)
P0_Age1 = sint.get_input_from(0)
P0_Age2 = sint.get_input_from(0)
P0_Weight0 = sint.get_input_from(0)
P0_Weight1 = sint.get_input_from(0)
P0_Weight2 = sint.get_input_from(0)

P1_ID0 = sint.get_input_from(1)
P1_ID1 = sint.get_input_from(1)
P1_ID2 = sint.get_input_from(1)
P1_Salary0 = sfix.get_input_from(1)
P1_Salary1 = sfix.get_input_from(1)
P1_Salary2 = sfix.get_input_from(1)
P1_Year0 = sint.get_input_from(1)
P1_Year1 = sint.get_input_from(1)
P1_Year2 = sint.get_input_from(1)

df0 = dataframe(data=[[P0_ID0, P0_Age0, P0_Weight0], [P0_ID1, P0_Age1, P0_Weight1], [P0_ID2, P0_Age2, P0_Weight2]], columns=['ID', 'Age', 'Weight'])
df1 = dataframe(data=[[P1_ID0, P1_Salary0, P1_Year0], [P1_ID1, P1_Salary1, P1_Year1], [P1_ID2, P1_Salary2, P1_Year2]], columns=['ID', 'Salary', 'Year'])

union_df, n = df0.merge(obj=df1, on='ID', join='outer', inplace=False)
```

### 编译

编写好上述mpc代码后，运行下述命令来获得编译后的二进制文件：
```
python ./compile.py test_dataframe
```

### 运行
在两方半诚实场景下运行，使用下述命令：
```
./Scripts/semi2k.sh test_dataframe
```