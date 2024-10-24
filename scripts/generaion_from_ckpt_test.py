import os
os.environ['NPU_VISIBLE_DEVICES']='0'
os.environ['ASCEND_RT_VISIBLE_DEVICES']='0'
import torch
import torch_npu
torch.npu.set_device(['npu:0'])

import argparse
import time

from aigcode.config import (
    TrainConfig,
)
from aigcode.model import AIGCcode
from aigcode.tokenizer import Tokenizer
from aigcode.beam_search import Sampler, TopKSampler, TopPSampler


def load_model(model_path, tokenizer_path, npu=0):
        
    yaml_path = os.path.join(model_path, "config.yaml")
    print("yaml:{}\n".format(yaml_path))
    # tokenizer_path = os.path.join(model_path, "tokenizer.json")

    model_pt_path = os.path.join(model_path, "model.pt")
    print("model_pt_path:{}\n".format(model_pt_path))
    # 找到对应的pt文件
    # print("yaml path is:", yaml_path)
    # 加载模型
    # device = torch.device("cuda")
    # yaml_path = ""
    config = TrainConfig.load(yaml_path, [])
    # config.model.init_device = "cuda"
    config.model.init_device = "npu"
    print("Config: {}\n".format(config))
    model = AIGCcode(config=config.model)
    print("model construct")
    state_dict = torch.load(model_pt_path, map_location="npu")
    print("state dict source: {}\n".format(state_dict))
    # model = torch.load(model_pt_path)
    model.load_state_dict(model._make_state_dict_compatible(state_dict)[0])
    print("state dict loaded: {}\n".format(model.state_dict()))
    model = model.to('npu:{}'.format(npu))
    # .eval()
    
    tokenizer = Tokenizer.from_file(tokenizer_path, eos_token_id=2, bos_token_id=1, pad_token_id=0, truncate_to=None)
    return model, tokenizer

    
def model_gen(model, input_ids):
    i = 0
    input_ids = [1] + input_ids
    with torch.no_grad():
        while True:
            if i > 100:
            # if i > 100:
            # if i > 100:
                break
            i += 1
            # 二维
            # print("input_ids: {}\n".format(input_ids))
            # print("input len: {}\n".format(len(input_ids)))
    ## model eval     
            outputs = model(torch.tensor([input_ids], device=model.device))
            
            batch_logits = outputs.logits
            print("batch_logits shape:{}\n".format(batch_logits.shape))
            # 转纬度
            batch_probs = torch.softmax(batch_logits, dim=-1)
            # if i > 1:
            #exclude_ids = input_ids[len_token-1:]
            #batch_probs = penalize_repeats(batch_probs, exclude_ids)
            batch_prediction_indices = torch.argmax(batch_probs, dim=-1)
            
            # print("batch_prediction_indices -1 is:", batch_prediction_indices[])
            # 转文字
            token_ids = batch_prediction_indices.tolist()
            # print("outputs: {}\n".format(token_ids))
            # print("token_ids is:", token_ids)
            output_ids = token_ids[0]
            # print("tmp res: {}\n".format(tokenizer.decode(token_ids[0][:])))
            ### input_ids 追加 last_id
            last_id = output_ids[-1]
            input_ids.append(last_id)

    ### model gen
            # outputs = model.generate(input_ids=torch.tensor([input_ids], device=model.device),max_steps=10,beam_size=1)
            # print("gen_output: {}\n".format(outputs))
            # token_ids = outputs.token_ids.tolist()[0][0]
            # input_ids.extend(token_ids)
            # last_id = input_ids[-1]
            
            # print("tmp res add: {}\n".format(tokenizer.decode(input_ids)), file=sys.stdout)
            # if input_ids[-1] == 151643:
            #     break
            if last_id == 2:
                break

            if len(input_ids) > model.config.max_sequence_length:
                input_ids = input_ids[len(input_ids)-model.config.max_sequence_length:]
            # input_ids = torch.cat([input_ids, torch.tensor([last_id]).to(model.device)])
        # response = tokenizer.decode(token_ids[0], skip_special_tokens=True)

        # generation_output = model.generate(
        #     input_ids=input_ids, max_new_tokens=answer_ids.shape[-1], num_beams=1, use_cache=use_cache
        # )
        # print("prediction all: {}\n".format(tokenizer.decode(token_ids[0])))
        # print("prediction all: {}\n".format(tokenizer.decode(token_ids[0])), file=sys.stdout)
        # # print("answer: {}\n".format(answer))
        # # print("res: {}\n".format(tokenizer.decode(token_ids[0][len_token-1:])))
        # print("res: {}\n".format(tokenizer.decode(input_ids[len_token:])), file=sys.stdout)
        # # print("[ground truth]: ", answer, ", answer_id is:", answer_ids)
        # print("--------")
    return input_ids


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()

    argparser.add_argument("--model_dir", type=str, default="/sharedata/aigcode_7b_checkpoints/814/step129600-unsharded")
    argparser.add_argument("--tokenizer_path", type=str, default="/sharedata/zimoliu/code/zimo_aigcode_moe/tokenizers/neo_tokenizer.json")
    argparser.add_argument("--npu_indce", type=int, default=0)
    argparser.add_argument("--gen_type", type=str, default="gen")

    args = argparser.parse_args()
    print("args:{}\n".format(args))

    model_dir = args.model_dir
    tokenizer_path = args.tokenizer_path
    npu_indce = args.npu_indce
    


    start_time = time.time()
    # gpu_num = sc % 8
    # print("gpu_num:{}".format(gpu_num))
    model, tokenizer = load_model(model_dir, tokenizer_path, npu_indce)
    print("loading tokenizer and model")
    end_time = time.time()
    # model_lst.append((tokenizer, model))

    print("加载模型耗时: {:.2f}秒".format(end_time - start_time))
    # tokenizer = AutoTokenizer.from_pretrained("/cpfs01/shared/public/zimoliu/code/OrionStar-Yi-34B-Chat",
    #                                           use_fast=False, trust_remote_code=True)
    # print("loading model")
    # model = AutoModelForCausalLM.from_pretrained("/cpfs01/shared/public/zimoliu/code/OrionStar-Yi-34B-Chat",
    #                                              torch_dtype=torch.bfloat16,
    #                                              trust_remote_code=True).half().to('cuda:{}'.format(sc)).eval()

    # print("loading config")

    # model.generation_config = GenerationConfig.from_pretrained(
    #     "/cpfs01/shared/public/zimoliu/code/OrionStar-Yi-34B-Chat",
    #     trust_remote_code=True)  # 可指定不同的生成长度、top_p等相关超参

    # print("eval model")

    # model = model.half().to('cuda:{}'.format(sc)).eval()
    
    hist = []
    if args.gen_type == "gen":
        k_sampler = TopKSampler(k=50, temperature=2.0)
    while True:
        # if his > 0:
        #     if len(hist) >= his:
        #         if not clear:
        #             print("out of length")
        #             hist = hist[-his:]
        #         else:
        #             print("out of length, clear history")
        #             hist = []
        # else:
        #     hist = []
        text = input("Input: ")
        print("-"*20)
        # res, hist = generate(model, tokenizer, text, hist)
        input_ids = tokenizer.encode(text, add_special_tokens=False)
        
        print("orig input_ids {}\n".format(input_ids))

        len_token = len(input_ids)

        print("orig input len: {}\n".format(len_token))
        print("orig tokens: {}\n".format(tokenizer.decode(input_ids)))
        # print("max_len: {}\n".format(model.config.max_sequence_length), file=sys.stdout)
        # input_ids = input_ids[:model.config.max_sequence_length]
        if args.gen_type == "gen":
            res = model.generate(torch.tensor([input_ids], device=model.device), sampler=k_sampler, max_steps=20, beam_size=10, per_node_beam_size=5)
        else:
            res = model_gen(model, input_ids)
        # print(len(hist))
        print("-" * 20)
        print("output orig:{}".format(res))
        print("-" * 20)
        if args.gen_type == "gen":
            print("output txt:{}".format(tokenizer.decode(res.token_ids.tolist()[0][0])))
            print("output txt:{}".format(tokenizer.decode(res.token_ids.tolist()[0][1])))
            print("output txt:{}".format(tokenizer.decode(res.token_ids.tolist()[0][2])))
            print("output txt:{}".format(tokenizer.decode(res.token_ids.tolist()[0][3])))
            print("output txt:{}".format(tokenizer.decode(res.token_ids.tolist()[0][4])))
        else:
            print("output txt:{}".format(tokenizer.decode(res)))
        print("=" * 20)
        # print("=" * 40)

