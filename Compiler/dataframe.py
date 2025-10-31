from .types import series, sint, sfix, cint, cfix, cchr, _dataframeLoc
from Compiler import library
from Compiler.sorting import gen_perm_by_radix_sort, SortPerm

class DataFrame:
    def __init__(self, data, columns, index = None):
        if len(set(columns)) != len(columns):
            raise ValueError("Replicated column name. ")
    
        if index and not all(isinstance(i, int) for i in index):
            raise ValueError("Each index must be type integer.")

        self.columns = columns
        self.index = index if index else list(range(len(data)))
        # self.index = list(range(len(data)))

        max_len = max(len(row) for row in data)
        padded_data = [row + [None] * (max_len - len(row)) for row in data]
        transposed_data = list(zip(*padded_data))

        self.value_types = []
        self.data = {}

        for col, col_data in zip(self.columns, transposed_data):
            non_none_values = [val for val in col_data if val is not None]

            if not non_none_values:
                raise ValueError(f"Column '{col}' contains only None values; unable to determine type.")

            first_type = type(non_none_values[0])
            if not all(isinstance(val, first_type) for val in non_none_values):
                raise TypeError(f"Column '{col}' contains mixed types: {[type(val) for val in non_none_values]}")

            self.value_types.append(first_type)
            self.data[col] = series(data=list(col_data), index=self.index, name=col)

    def convert_value(self, value, value_type):
        if not isinstance(value, value_type):
            try:
                return value_type.conv(value)
            except AttributeError:
                raise TypeError(f"Cannot convert {type(value)} to {value_type}")
        return value

    def drop(self, index=None, column=None, inplace=False):

        if index is None and column is None:
            raise ValueError("At least one of 'index' or 'column' must be specified.")

        if not inplace:
            df_copy = DataFrame(
                data=[[self.data[col].data[i] for col in self.columns] for i in range(len(self.index))],
                columns=self.columns[:],
                index=self.index[:]
            )
            return df_copy.drop(index=index, column=column, inplace=True)

        if column is not None:
            if isinstance(column, str):
                column = [column]  
            if not all(col in self.columns for col in column):
                missing = [col for col in column if col not in self.columns]
                raise KeyError(f"Columns {missing} not found in DataFrame.")

            self.columns = [col for col in self.columns if col not in column]
            self.value_types = [self.value_types[i] for i, col in enumerate(self.columns) if col not in column]
            self.data = {col: self.data[col] for col in self.columns}

        if index is not None:
            if isinstance(index, int):
                index = [index]
            elif isinstance(index, slice):
                index = list(range(*index.indices(len(self.index))))
            elif not isinstance(index, list):
                raise TypeError("Index must be an int, list, or slice.")

            invalid_indices = [i for i in index if i not in self.index]
            if invalid_indices:
                raise IndexError(f"Indices {invalid_indices} are out of range.")

            remaining_indices = [i for i in self.index if i not in index]
            # self.index = list(range(len(remaining_indices)))
            self.index = remaining_indices
            for col in self.columns:
                self.data[col].data = [self.data[col].data[i] for i in remaining_indices]
                self.data[col].index = self.index

        return self 

    def merge(self, obj, on, join='outer', inplace=False):
        if join not in ['outer', 'inner']:
            raise ValueError("The join type can only be 'inner' or 'outer'.")

        if not isinstance(obj, list):
            obj = [obj]

        if not all(isinstance(df, DataFrame) for df in obj):
            raise TypeError("All elements in obj must be instances of DataFrame.")
    
        new_columns = self.columns[:]
        for df in obj:
            for col in df.columns:
                if col not in new_columns:
                    new_columns.append(col)

        new_value_types = []
        for col in new_columns:
            col_types = set()
            if col in self.columns:
                col_types.add(self.value_types[self.columns.index(col)])
            for df in obj:
                if col in df.columns:
                    col_types.add(df.value_types[df.columns.index(col)])
            
            if len(col_types) > 1:
                raise TypeError(f"Column '{col}' has inconsistent types across DataFrames: {col_types}")
            
            new_value_types.append(next(iter(col_types)))

        new_index = self.index[:]
        max_index = max(self.index) if self.index else -1
        for df in obj:
            new_index += [max_index + 1 + i for i in range(len(df.index))]
            max_index = new_index[-1]

        new_data = {col: [None] * len(new_index) for col in new_columns}

        for col in self.columns:
            for i, v in enumerate(self.data[col].data):
                new_data[col][i] = v

        row_offset = len(self.index)
        for df in obj:
            for col in df.columns:
                for i, v in enumerate(df.data[col].data):
                    new_data[col][row_offset + i] = v
            row_offset += len(df.index)

        new_df = DataFrame(
            data=[list(row) for row in zip(*new_data.values())],
            columns=new_columns,
            index=new_index
        )

        if on not in new_df.columns:
            raise ValueError(f"Column '{on}' does not exist in either DataFrame.")

        if inplace:
            self.index = new_df.index
            self.columns = new_df.columns
            self.value_types = new_value_types
            self.data = {col: series(new_df[col].data, index=new_df.index, name=col) for col in new_columns}

            on_type = self.value_types[self.columns.index(on)]
            if on_type is sint or on_type is sfix:
                return self._ss_groupBy(on=on, party_number=len(obj) + 1, join=join)
            elif on_type is cint or on_type is cfix:
                return self._groupBy(on=on, party_number=len(obj) + 1, join=join)
        else:
            on_type = new_df.value_types[new_df.columns.index(on)]
            if on_type is sint or on_type is sfix: 
                return new_df._ss_groupBy(on=on, party_number=len(obj) + 1, join=join)
            elif on_type is cint or on_type is cfix:
                return new_df._groupBy(on=on, party_number=len(obj) + 1, join=join)
            
    def concat(objs, axis=0):
        if not isinstance(objs, list) or len(objs) == 0:
            raise ValueError("objs must be a non-empty list of DataFrames")
    
        if not all(isinstance(df, DataFrame) for df in objs):
            raise TypeError("All elements in objs must be instances of DataFrame.")
        
        if axis == 0:
            first_columns = objs[0].columns
            first_value_types = objs[0].value_types
            
            for i, df in enumerate(objs[1:], 1):
                if df.columns != first_columns:
                    raise ValueError(f"All DataFrames must have the same columns when axis=0. "
                                f"DataFrame 0 has columns {first_columns}, "
                                f"but DataFrame {i} has columns {df.columns}")
                
                for j, col in enumerate(first_columns):
                    if df.value_types[j] != first_value_types[j]:
                        raise TypeError(f"Column '{col}' has inconsistent types across DataFrames: "
                                    f"{first_value_types[j]} vs {df.value_types[j]}")
            
            new_data = {col: [] for col in first_columns}
            new_index = []
   
            max_index = max(max(df.index) for df in objs) if objs[0].index else -1
            for df in objs:
                for col in first_columns:
                    new_data[col].extend(df.data[col].data)
                offset = max_index + 1 if new_index else 0
                new_index.extend([idx + offset for idx in df.index])
                max_index = new_index[-1]
            
            return DataFrame(
                data=[list(row) for row in zip(*new_data.values())],
                columns=first_columns,
                index=new_index
            )
        
        elif axis == 1:
            first_length = len(objs[0].index)
            for i, df in enumerate(objs[1:], 1):
                if len(df.index) != first_length:
                    raise ValueError(f"All DataFrames must have the same number of rows when axis=1. "
                                f"DataFrame 0 has {first_length} rows, "
                                f"but DataFrame {i} has {len(df.index)} rows")
            
            new_columns = []
            new_value_types = []
            new_data = {}
            new_index = objs[0].index
            
            for df in objs:
                for col in df.columns:
                    if col in new_columns:
                        raise ValueError(f"Column name '{col}' appears in multiple DataFrames. "
                                    f"Column names must be unique when axis=1.")
                    
                    new_columns.append(col)
                    col_idx = df.columns.index(col)
                    new_value_types.append(df.value_types[col_idx])
                    new_data[col] = df.data[col].data
            
            return DataFrame(
                data=[list(row) for row in zip(*new_data.values())],
                columns=new_columns,
                index=new_index
            )
        
        else:
            raise ValueError("axis must be either 0 (vertical) or 1 (horizontal)")

    def join(self, other, lsuffix='_left', rsuffix='_right', how='outer', inplace=False):
        if how not in ['outer', 'inner']:
            raise ValueError("The join type can only be 'inner' or 'outer'.")

        if not isinstance(other, DataFrame):
            raise TypeError("The other object must be an instance of DataFrame.")

        new_columns = []
        left_columns_map = {}
        right_columns_map = {}
        
        for col in self.columns:
            if col in other.columns:
                new_col_name = col + lsuffix
                left_columns_map[col] = new_col_name
                new_columns.append(new_col_name)
            else:
                left_columns_map[col] = col
                new_columns.append(col)
        
        for col in other.columns:
            if col in self.columns:
                new_col_name = col + rsuffix
                right_columns_map[col] = new_col_name
                if new_col_name not in new_columns:
                    new_columns.append(new_col_name)
            else:
                right_columns_map[col] = col
                new_columns.append(col)

        if how == 'outer':
            new_index = sorted(set(self.index) | set(other.index))
        else:
            new_index = sorted(set(self.index) & set(other.index))

        new_value_types = []
        new_data = {col: [None] * len(new_index) for col in new_columns}

        for col in new_columns:
            col_types = set()
            
            orig_col = None
            for k, v in left_columns_map.items():
                if v == col:
                    orig_col = k
                    break
            if orig_col and orig_col in self.columns:
                col_idx = self.columns.index(orig_col)
                col_types.add(self.value_types[col_idx])
            
            orig_col = None
            for k, v in right_columns_map.items():
                if v == col:
                    orig_col = k
                    break
            if orig_col and orig_col in other.columns:
                col_idx = other.columns.index(orig_col)
                col_types.add(other.value_types[col_idx])
            
            if len(col_types) > 1:
                if str in col_types:
                    new_value_types.append(str)
                else:
                    non_none_types = [t for t in col_types if t is not type(None)]
                    if non_none_types:
                        new_value_types.append(non_none_types[0])
                    else:
                        new_value_types.append(object)
            elif col_types:
                new_value_types.append(next(iter(col_types)))
            else:
                new_value_types.append(object)

        for orig_col, new_col in left_columns_map.items():
            col_type = new_value_types[new_columns.index(new_col)]
            for i, idx_val in enumerate(new_index):
                if idx_val in self.index:
                    pos = self.index.index(idx_val)
                    value = self.data[orig_col].data[pos]
                    if value is not None and not isinstance(value, col_type) and col_type != object:
                        try:
                            if col_type == str:
                                value = str(value)
                            elif col_type == int:
                                value = int(value)
                            elif col_type == float:
                                value = float(value)
                        except (ValueError, TypeError):
                            pass
                    new_data[new_col][i] = value

        for orig_col, new_col in right_columns_map.items():
            col_type = new_value_types[new_columns.index(new_col)]
            for i, idx_val in enumerate(new_index):
                if idx_val in other.index:
                    pos = other.index.index(idx_val)
                    value = other.data[orig_col].data[pos]
                    if value is not None and not isinstance(value, col_type) and col_type != object:
                        try:
                            if col_type == str:
                                value = str(value)
                            elif col_type == int:
                                value = int(value)
                            elif col_type == float:
                                value = float(value)
                        except (ValueError, TypeError):
                            pass
                    new_data[new_col][i] = value

        result_data = [list(row) for row in zip(*new_data.values())]
        
        try:
            new_df = DataFrame(data=result_data, columns=new_columns, index=new_index)
        except Exception as e:
            print(f"警告: 创建DataFrame时遇到问题: {e}")
            raise e

        if inplace:
            self.index = new_df.index
            self.columns = new_df.columns
            self.value_types = new_value_types
            self.data = {col: series(new_df[col].data, index=new_df.index, name=col) for col in new_columns}
            return self
        else:
            return new_df

    def _groupBy(self, on, party_number, join = 'outer'):
        on_type = self.value_types[self.columns.index(on)]
        on_array = on_type.Array(size=len(self.index))
        for i in range(len(self.index)): on_array[i] = self[on][i]

        inner_array = on_type.Array(size=len(self.index))
        outer_array = on_type.Array(size=len(self.index))

        for col in self.columns:
            if col is on: continue
            col_type = self.value_types[self.columns.index(col)]
            col_array = col_type.Array(size=len(self.index))
            for i in range(len(self.index)): col_array[i] = self[col][i]

            for i in range(len(self.index)):
                for j in range(i):
                    @library.if_(on_array[i] == on_array[j])
                    def body():
                        inner_array[j] = inner_array[j] + 1
                        outer_array[i] = outer_array[i] + 1

                        tmp_i = col_array[i]
                        tmp_j = col_array[j]
                        # ne = col_array[j] != 0
                        e = col_array[j] == 0
                        col_array[i] = e * tmp_i + (1 - e) * tmp_j
                        col_array[j] = e * tmp_i + (1 - e) * tmp_j

            for i in range(len(self.index)): self[col][i] = col_array[i]
        
        # for i in range(len(self.index)): library.print_ln("inner_array[%s]: %s", i, inner_array[i])
        # for i in range(len(self.index)): library.print_ln("outer_array[%s]: %s", i, outer_array[i])
        self.index = list(range(len(self.index)))
        for col in self.columns:
            self[col].index = self.index
        
        # outer join
        if join == 'outer': 
            ct_array = cint.Array(size=1)
            ct_array[0] = 0
            for col in self.columns:
                col_type = self.value_types[self.columns.index(col)]
                col_array = col_type.Array(size=len(self.index))

                idx = cint.Array(size=1)
                idx[0] = 0
                for i in range(len(self.index)):
                    @library.if_(outer_array[i] == 0)
                    def _body():
                        col_array[idx[0]] = self[col][i]
                        idx[0] = idx[0] + 1
                ct_array[0] = idx[0]
                
                # for i in range(len(self.index)): library.print_ln("%s_array[%s]: %s", col, i, col_array[i].reveal())
                for i in range(len(self.index)): self[col][i] = col_array[i]

            return self, ct_array[0]

        # inner join
        elif join == 'inner':
            party_number_cint = cint(party_number - 1)
            col_num_without_on = cint(len(self.columns) - 1)
            times = party_number_cint * col_num_without_on

            ct_array = cint.Array(size=1)
            ct_array[0] = 0

            for col in self.columns:
                col_type = self.value_types[self.columns.index(col)]
                col_array = col_type.Array(size=len(self.index))

                idx = cint.Array(size=1)
                idx[0] = 0
                for i in range(len(self.index)):
                    @library.if_(inner_array[i] == times)
                    def _body():
                        col_array[idx[0]] = self[col][i]
                        idx[0] = idx[0] + 1
                ct_array[0] = idx[0]
                
                # for i in range(len(self.index)): library.print_ln("%s_array[%s]: %s", col, i, col_array[i].reveal())
                for i in range(len(self.index)): self[col][i] = col_array[i]

            return self, ct_array[0]

    def _ss_groupBy(self, on, party_number, join = 'outer'):
        from Compiler.sorting import gen_perm_by_radix_sort, SortPerm

        if cint in self.value_types or cfix in self.value_types or cchr in self.value_types: 
            raise ValueError(f"Cannot merge the tables with secret column '{on}' and clear text column.")

        # sort
        ids = self[on]
        on_type = self.value_types[self.columns.index(on)]
        ids_Array = sint.Array(size=len(self.index))
        for i in range(len(self.index)): ids_Array[i] = ids.data[i]
        
        perm = gen_perm_by_radix_sort(ids_Array)
        for col in self.columns:
            col_value_type = self.value_types[self.columns.index(col)]
            col_Array = col_value_type.Array(size=len(self.index))
            for i in range(len(self.index)): col_Array[i] = self[col].data[i]
            library.print_ln("%s:", col)
            col_Array = perm.apply(col_Array)
            for i in range(len(self.index)): self[col].data[i] = col_Array[i]

        for i in range(len(self.index)): ids_Array[i] = self[on].data[i]
        flag = sint.Array(size=len(self.index))
        flag[0] = 1
        flag.assign_vector(ids_Array.get_vector(size=len(ids_Array) - 1) !=
                           ids_Array.get_vector(size=len(ids_Array) - 1, base=1), base=1)
        
        # 保留除0外第一个行的数据
        for col in self.columns:
            if col is on: continue
            col_value_type = self.value_types[self.columns.index(col)]
            col_Array = col_value_type.Array(size=len(self.index))
            for i in range(len(self.index)): col_Array[i] = self[col].data[i]
            
            for i in range(len(self.index) - 1, -1, -1):
                if i == 0: continue
                j = i - 1
                tmp_i = col_Array[i]
                tmp_j = col_Array[j]
                e = tmp_i == 0
                e2 = tmp_j == 0
                col_Array[i] = flag[i] * tmp_i + (1 - flag[i]) * ((1 - e) * e2 * tmp_i + (1 - (1 - e) * e2) * tmp_j)
                col_Array[j] = flag[i] * tmp_j + (1 - flag[i]) * ((1 - e) * e2 * tmp_i + (1 - (1 - e) * e2) * tmp_j)
            
            for i in range(len(self.index)): self[col].data[i] = col_Array[i]
        
        self.index = list(range(len(self.index)))
        for col in self.columns:
            self[col].index = self.index

        # outer join
        if join == 'outer':
            for col in self.columns:
                col_value_type = self.value_types[self.columns.index(col)]
                col_Array = col_value_type.Array(size=len(self.index))
                for i in range(len(self.index)): col_Array[i] = self[col].data[i]
                col_Array = col_Array * flag
                for i in range(len(self.index)): self[col].data[i] = col_Array[i]
            
            perm = SortPerm(flag.get_vector().bit_not())
            for col in self.columns:
                col_value_type = self.value_types[self.columns.index(col)]
                col_Array = col_value_type.Array(size=len(self.index))
                for i in range(len(self.index)): col_Array[i] = self[col].data[i]
                col_Array = perm.apply(col_Array)
                for i in range(len(self.index)): self[col].data[i] = col_Array[i]

            return self, sum(flag)
        
        # inner join
        elif join == 'inner': 
            in_intersection = sint.Array(size=len(self.index))
            in_intersection.assign_vector(ids_Array.get_vector(size=len(ids_Array) - 1) ==
                                ids_Array.get_vector(size=len(ids_Array) - 1, base=party_number - 1), base=0)
            
            for col in self.columns:
                col_value_type = self.value_types[self.columns.index(col)]
                col_Array = col_value_type.Array(size=len(self.index))
                for i in range(len(self.index)): col_Array[i] = self[col].data[i]
                col_Array = col_Array * in_intersection
                for i in range(len(self.index)): self[col].data[i] = col_Array[i]
            
            perm = SortPerm(in_intersection.get_vector().bit_not())
            for col in self.columns:
                col_value_type = self.value_types[self.columns.index(col)]
                col_Array = col_value_type.Array(size=len(self.index))
                for i in range(len(self.index)): col_Array[i] = self[col].data[i]
                col_Array = perm.apply(col_Array)
                for i in range(len(self.index)): self[col].data[i] = col_Array[i]

            return self, sum(in_intersection)

    def __getitem__(self, key):
        if isinstance(key, str): 
            if key not in self.columns:
                raise KeyError(f"Column '{key}' not found in DataFrame.")
            return self.data[key]
        elif isinstance(key, list):
            missing_cols = [col for col in key if col not in self.columns]
            if missing_cols:
                raise KeyError(f"Columns {missing_cols} not found in DataFrame.")

            new_data = {col: self.data[col] for col in key}
            return DataFrame(
                data=[list(row) for row in zip(*[new_data[col].data for col in key])],
                columns=key,
                index=self.index[:]
            )

        else:
            raise TypeError(f"Invalid key type: {type(key)}. Must be str or list of str.")

    def __setitem__(self, key, value):
        if isinstance(key, str): 
            self._assign_single_column(key, value)
            
        elif isinstance(key, list):
            if not isinstance(value, list) or not all(isinstance(row, list) for row in value):
                raise TypeError("Value must be a list of lists for multiple column assignment.")

            num_rows = len(value)
            max_existing_rows = len(self.index)
            max_new_rows = max(num_rows, max_existing_rows)
            self._expand_index(max_new_rows)

            col_data_dict = {col: [] for col in key}
            for row in value:
                for col, val in zip(key, row):
                    col_data_dict[col].append(val)
                for col in key[len(row):]:
                    col_data_dict[col].append(None)

            for col, col_values in col_data_dict.items():
                self._assign_single_column(col, col_values, target_rows=max_new_rows)

        else:
            raise KeyError(f"Invalid key type: {type(key)}")

    def _assign_single_column(self, col, value, target_rows=None):
        num_values = len(value)
        max_existing_rows = len(self.index)
        
        required_rows = max(num_values, max_existing_rows)
        self._expand_index(required_rows)

        if col in self.columns:
            expected_type = self.data[col].value_type

            self.data[col].data = value + [None] * (required_rows - num_values)
            self.data[col].data = [self.convert_value(val, expected_type) 
                                   for val in self.data[col].data]

        else:
            inferred_type = self._infer_column_type(value)
            self.columns.append(col)
            self.value_types.append(inferred_type)
            self.data[col] = series(
                value + [None] * (required_rows - num_values), 
                index=self.index, 
                name=col
            )

    def _expand_index(self, new_size):
        if new_size <= len(self.index):
            return
        
        max_index = 0
        for i in self.index: max_index = max(max_index, i)
        new_indices_num = new_size - len(self.index)
        for i in range(0, new_indices_num):
            self.index = self.index + [max_index + i + 1]
        
        # self.index = list(range(new_size))
        for col in self.columns:
            self.data[col].index = self.index
            self.data[col].data.extend([None] * (new_size - len(self.data[col].data)))
            self.data[col].data = [self.convert_value(val, self.value_types[self.columns.index(col)]) for val in self.data[col].data]

    def _infer_column_type(self, col_data):
        non_none_values = [val for val in col_data if val is not None]
        if not non_none_values:
            raise ValueError("At least one element is not None")
        
        first_type = type(non_none_values[0])
        if not all(isinstance(v, first_type) for v in non_none_values):
            raise TypeError(f"Column contains mixed types: {[type(v) for v in non_none_values]}")
        
        return first_type

    @property
    def shape(self):
        num_rows = len(self.index)
        num_cols = len(self.columns)
        return (num_rows, num_cols)

    @property
    def loc(self):
        return _DataFrameLoc(self)
    
    # to check the table format for test
    def __repr__(self):
        return self._format_DataFrame()

    def _format_DataFrame(self):
        col_widths = [max(len(str(col)), max(len(str(val)) for val in self.data[col].data)) for col in self.columns]
        index_width = max(len(str(idx)) for idx in self.index)  # 计算索引宽度

        header = " " * (index_width + 2) + "  ".join(f"{col:<{col_widths[i]}}" for i, col in enumerate(self.columns))

        rows = []
        for i, idx in enumerate(self.index):
            row_values = [str(self.data[col].data[i]) for col in self.columns]
            formatted_row = f"{idx:<{index_width}}  " + "  ".join(f"{val:<{col_widths[j]}}" for j, val in enumerate(row_values))
            rows.append(formatted_row)

        return header + "\n" + "\n".join(rows)