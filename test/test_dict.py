# 初始字典
data_dict = {
    "key1": "value1",
    "key2": "value2",
}

# 新的键值对，可能会有新键，也可能会有更新的键
new_data = {
    "key2": "new_value2",  # 更新 "key2" 的值
    "key3": "value3",      # 新的键 "key3"
}

# 更新字典
data_dict.update(new_data)

print(data_dict)
