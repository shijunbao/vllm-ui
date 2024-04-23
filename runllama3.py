from openai import OpenAI
import gradio as gr
import re
import jieba

#########################################
#参数
#本次输入联合历史记录作为输入，最大的token上限
max_window_tokens = 8192
#保留几轮历史记录,比如保留一轮问答，设置为1，  
# 不要历史记录设置为0，这时只处理input文本
history_rounds = 6
#########################################


#x版：带有连续对话和最大token截断功能。

# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1/"

# 创建一个 OpenAI 客户端，用于与 API 服务器进行交互
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

#判断是否是汉字字符。
def is_chinese_character(char):
    # 正则表达式匹配所有基本汉字和汉字扩展A
    pattern = re.compile(r'[\u4e00-\u9fff]')
    # 使用match方法检查字符是否匹配
    return bool(pattern.match(char))


#统计一个字符串的token数
def count_string_tokens(s):
    # 初始化结果列表
    result = []

    # 遍历字符串中的每个字符
    i = 0
    while i < len(s):
        # 检查当前字符是否为英文字母
        if s[i].isalpha():
            # 从当前位置开始，找到连续的英文字母
            j = i + 1
            while j < len(s) and s[j].isalpha() and is_chinese_character(s[j])==False:
                #print(s[j])
                #print("is alpha")
                j += 1
            # 将连续的英文字母作为一个单词添加到结果列表中
            result.append(s[i:j])
            i = j
        else:
            # 将其他字符（包括中文、数字、标点符号等）作为一个单独的字符添加到结果列表中
            result.append(s[i])
            i += 1

    # 获取result列表元素数量，也就获取了分割后token数量       
    count_toks = len(result)

    return(count_toks)




#截取字符串 s ，同时返回：最后n个字符串，
def split_string(s, n):
    # 初始化结果列表
    result = []

    # 遍历字符串中的每个字符
    i = 0
    while i < len(s):
        # 检查当前字符是否为英文字母
        if s[i].isalpha():
            # 从当前位置开始，找到连续的英文字母
            j = i + 1
            while j < len(s) and s[j].isalpha() and is_chinese_character(s[j])==False:
                #print(s[j])
                #print("is alpha")
                j += 1
            # 将连续的英文字母作为一个单词添加到结果列表中
            result.append(s[i:j])
            i = j
        else:
            # 将其他字符（包括中文、数字、标点符号等）作为一个单独的字符添加到结果列表中
            result.append(s[i])
            i += 1

    # 截取最后n个分割，重新组合成字符串
    last_n_segments = ''.join(result[-n:])

    #return last_n_segments,result
    #return len(result),last_n_segments,result
    return last_n_segments


def predict(message, history):
    #清空工作窗口列表
    work_content = []

    # 计算message输入的token数量
    total_tokens = 0
    count_message_tokens = count_string_tokens(message)
    # 历史记录允许的最大token数计算      总的token-message的token数
    history_total_tokens = max_window_tokens - count_message_tokens
    tokens_left = history_total_tokens   #中间变量，统计每个添加后的剩余token数  初始化

    # 将所有聊天历史转换为 OpenAI 格式
    history_openai_format = [{"role": "system", "content": "你是个靠谱的 AI 助手，尽量详细的解答用户的提问。"}]
  


    #计数器，统计history_openai_format添加次数
    rounds_count = 0
    if history_rounds > 0: #需要调取历史记录
        # 从最后一组数据开始，每次添加到列表的第一个元素之前，先添加assistant，再添加user
        for human, assistant in reversed(history):
            # 先添加assistant的对话
            history_openai_format.insert(0, {"role": "assistant", "content": assistant})
            # 再添加user的对话
            history_openai_format.insert(0, {"role": "user", "content": human})
            rounds_count = rounds_count + 1  #计数器增加一轮
            if rounds_count >= history_rounds:
                break
                
        
    
    
    #遍历，一步一步将history加入work_content
    
    #倒着统计每一个列表中的content的token数,  从最新到最旧历史记录
    reversed_token_counts = []
    reversed_history = []    
    for item in reversed(history_openai_format):   #倒着统计这个列表 也就是从最后一个最新的记录开始统计
        
        item_count = count_string_tokens(item['content']) 
        if item_count <= tokens_left:  #如果token数量够用   
            reversed_history.append(item)
            tokens_left = tokens_left - item_count  #剩余token计数器累减
            # print("1111111111111111111111")
            # print("add history content to history list")
            # print(item_count)
            # print(reversed_history)
        else:
            if tokens_left >=0:
                #截取最早的内容
                item['content'] = split_string(item['content'],tokens_left)
                reversed_history.append(item)
                # print("2222222222222")
                # print("add last history content to history list")
                # print(tokens_left)
                # print(reversed_history)
            break

    # print("reversed_history")
    # print(reversed_history)

    #从最早的可用history，添加到work_content
    for item in reversed(reversed_history):
        work_content.append(item)
    
    #加入message到work_content列表
    work_content.append({"role": "user", "content": message})
    

    # 添加第一个节点的检查，不是user增加user
    # 检查第一个元素的 'role' 字段是否是 'assistant'
    if work_content[0]['role'] != 'system':
        if work_content[0]['role'] == 'assistant':
            # 如果第一个是 'assistant'，则在第一个位置插入新的字典元素
            work_content.insert(0, {"role": "user", "content": "你是个靠谱的 AI 助手，尽量详细的解答用户的提问。"})
            work_content.insert(0, {"role": "system", "content": "你是个靠谱的 AI 助手，尽量详细的解答用户的提问。"})


    # print("work_content")
    # print(work_content)
 
    # 创建一个聊天完成请求，并将其发送到 API 服务器
    stream = client.chat.completions.create(
        model='gpt-3.5-turbo',   # 使用的模型名称  这个只是适配one api所以  使用gpt3.5  实际模型是FusionNet_34Bx2_MoE-AWQ
        messages= work_content,  # 聊天历史
        temperature=0.8,                  # 控制生成文本的随机性
        stream=True,                      # 是否以流的形式接收响应
        extra_body={
            'repetition_penalty': 1, 
            'stop_token_ids': []
        }
    )

    ## 从响应流中读取并返回生成的文本
    #partial_message = ""
    #for chunk in stream:
        #partial_message += (chunk.choices[0].delta.content or "")
        #yield partial_message
        
    # 从响应流中读取并返回生成的文本
    partial_message = ""
    for chunk in stream:
        chunk_content = (chunk.choices[0].delta.content or "")
        # 检查是否遇到了停止标记
        if '<|eot_id|>' in chunk_content:
            yield partial_message
            break
        partial_message += chunk_content
        yield partial_message



    # #测试：  打印   work_content

    print("#########")
    print("max_window_tokens最大窗口总token数")
    print(max_window_tokens)
    print("#########")
    print("history_total_tokens总共的历史可用token数（减去最新用户输入）")
    print(history_total_tokens)
    print("#########")
    # #输出work_content到一个字符串变量，用于统计测试当前总token
    # # 初始化一个空字符串变量来存储输出
    # work_content_string = ""
    # # 遍历history_openai_format列表中的每个元素
    # for item in work_content:
    #     # 将每个元素的"role"和"content"键对应的值拼接到输出字符串中
    #     work_content_string += f"Role: {item['role']}, Content: {item['content']}\n"
    # print("work_content的token总数（本轮提问总计喂给AI的token数量）")
    # print(count_string_tokens(work_content_string))
    print("#########")
    print("本轮提问喂给AI的内容：")
    print(work_content)
    print("#####################################################\n\n\n\n")
    
    

# 创建一个聊天界面，并启动它，share=True 让 gradio 为我们提供一个 debug 用的域名
gr.ChatInterface(predict).queue().launch(server_port=9010, share=True, server_name='0.0.0.0')
