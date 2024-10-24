import json

# 输入文件路径
input_file_path = '/sharedata/dataset/arcee-ai-The-Tome/the_tome.jsonl'
# 输出文件的基础路径
output_base_path = '/sharedata/zimoliu/data/chat_sft/arcee-ai-The-Tome/src/the_tome'

# 初始化变量
line_count = 0
chunk_size = 10000  # 每个文件包含的JSON对象数量
file_index = 1  # 输出文件的起始索引
js_lst = []
# 打开输入文件进行读取
with open(input_file_path, 'r', encoding='utf-8') as input_file:
    # 循环遍历输入文件的每一行
    for line in input_file:
        line_count += 1
        js_lst.append(json.loads(line.strip()))
        # 每读取1000行，写入一个新的jsonl文件
        if line_count % chunk_size == 0:
            # 将当前行（JSON对象）写入当前的jsonl文件
            with open(f'{output_base_path}_{file_index-1}.jsonl', 'a', encoding='utf-8') as output_file:
                output_file.write("\n".join([json.dumps(j, ensure_ascii=False) for j in js_lst]))
            print("write to {} successfuly".format(f'{output_base_path}_{file_index-1}.jsonl'))
            js_lst = []
            # 创建当前jsonl文件的路径
            file_index += 1
    if len(js_lst) != 0:
        with open(f'{output_base_path}_{file_index-1}.jsonl', 'a', encoding='utf-8') as output_file:
                output_file.write("\n".join([json.dumps(j, ensure_ascii=False) for j in js_lst]))
        print("write to {} successfuly".format(f'{output_base_path}_{file_index-1}.jsonl'))

    
        
            
# 检查是否还有剩余的行未写入文件
# if line_count % chunk_size != 0:
#     # 创建最后一个jsonl文件的路径
#     output_file_path = f'{output_base_path}_{file_index}.jsonl'
#     # 打开最后一个jsonl文件进行写入
#     with open(output_file_path, 'w', encoding='utf-8') as output_file:
#         for i in range(line_count % chunk_size):
#             output_file.write(input_file.next())

# print('所有JSON行已成功分割并保存为多个jsonl文件。')