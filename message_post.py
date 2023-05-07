import multiprocessing
import random
import time

import requests
import pandas as pd
import numpy as np
import secrets
import ecvrf_edwards25519_sha512_elligator2
import json

# post_object = {
#     "node_address" : 'http://127.0.0.1:8001'
# }
#
# r = requests.post('http://127.0.0.1:8000/register_node', json=post_object)
# print(r.text)

# def normfun(x, mu, sigma): #正态分布
#     pdf = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
#     return pdf



# 加权抽样算法
def randon_weight(weight_data):
    templist = []
    for i in weight_data:   # 信任值小于0的视为0
        if float(i) < 0:
            templist.append(0)
        else:
            templist.append(float(i))
    weight_data = templist
    total = sum(weight_data)
    ra = random.uniform(0, total)
    curr_sum = 0
    ret = 0
    for i in range(0, len(weight_data)):   #
        curr_sum += weight_data[i]
        if ra <= curr_sum:
            ret = i
            break
    return ret

def post_message(message_sender, message_context, message_receiver, NOL):
    # message_receiver = random.randint(8001, 8000+node_number)   # 随机产生接受者
    post_object = {
        "NOL_trust" : NOL.loc[message_sender-8001].values[0],  # 加上常数项0.5是因为贝叶斯公式原理，暂时用均匀分布代替
        "message_context" : message_context,
        "sender_ID" : message_sender
    }
    r = requests.post('http://127.0.0.1:' + str(message_receiver) + '/message_receiver', json=post_object)
    # print(r.text)

def pow_test(target_port, event):
    r = requests.get('http://127.0.0.1:' + str(target_port+8001) + '/mine', params='i='+str(target_port))
    if r.text == "finish":
        event.set()

def pot_test():
    print('正在进行', valid)
    starttime = time.time()
    total_weight = sum(miners_stake)
    winner_num = 1
    winner_PoT = 0
    m = valid  # 验证者个数
    validator_set = set()

    while winner_num != 0:  # winner选举
        candidate = {}
        for i in range(miner_number):
            if miner_list[i] in validator_set:  # 已经被选过的不参与
                continue
            post_object = {
                "p": (miners_stake[i] / total_weight) if total_weight != 0 else 1,
                "i": i
            }
            print(post_object)
            print(miners_stake)
            r = requests.post('http://127.0.0.1:' + str(miner_list[i]) + '/vrf_prove', json=post_object)
            if r.text == "selected":
                # wait_to_select.remove()
                candidate[miner_list[i]] = miners_stake[i]
        if len(candidate) >= 1:
            candidate = sorted(candidate.items(), key=lambda kv: kv[1], reverse=True)
            for i in range(len(candidate)):
                winner_PoT = candidate[i][0]
                total_weight -= candidate[i][1]
                winner_num -= 1
                if winner_num == 0:
                    break

    while m != 0:  # validator选举
        candidate = {}
        for i in range(miner_number):
            if miner_list[i] in validator_set or miner_list[i] == winner_PoT:  # 已经被选过的不参与
                continue
            post_object = {
                "p": (miners_stake[i] / total_weight) if total_weight != 0 else 1,
                "m": m
            }
            print(post_object)
            print(miners_stake)
            r = requests.post('http://127.0.0.1:' + str(miner_list[i]) + '/vrf_prove', json=post_object)
            if r.text == "selected":
                # wait_to_select.remove()
                candidate[miner_list[i]] = miners_stake[i]
        if len(candidate) >= 1:
            candidate = sorted(candidate.items(), key=lambda kv: kv[1], reverse=True)
            for i in range(len(candidate)):
                validator_set.add(candidate[i][0])
                total_weight -= candidate[i][1]
                m -= 1
                if m == 0:
                    break

    endtime = time.time()
    print('validator选举时间:', round(endtime - starttime, 2), 'secs')
    POT_time.append(round(endtime - starttime, 2))
    print('winner is ', winner_PoT)
    print(validator_set)

if __name__ == "__main__":
    for ii in range(21):
        NOL = pd.read_excel("NOL_test100.xlsx", header=None)
        message_context = 'The context is 1'
        node_number = 100
        miner_number = 20
        # NOL = pd.read_excel("NOL_facebook.xlsx", header=None)
        # node_number = 517
        # miner_number = 100


        mat1 = []
        with open(
                r'C:\Users\user\Desktop\博士手稿\data\KONECT-Social-Network-Datasets-master\facebook-wosn-wall\100test_matrix.csv',
                'r') as infile:
            for line in infile:
                temp = line.split(",")
                temp = [int(x.strip()) for x in temp]
                mat1.append(temp)
        mat1 = np.array(mat1)
        sender_list = [j for j in range(8001, 8001 + node_number)]  # 创建发送者ID名单
        sender_list = random.sample(sender_list, 50)  # 随机抽取?人作为发送者
        pool = multiprocessing.Pool(processes=10)   # 进程池一共有10进程
        for i in sender_list:
            receiver_set = set()
            for j in range(node_number):    # generate receiver set based on adjacency matrix
                if mat1[i - 8001][j] == 1:
                    receiver_set.add(j)
                if mat1[j][i - 8001] == 1:
                    receiver_set.add(j)
            for j in receiver_set:
                pool.apply_async(post_message, (i, message_context, j+8001, NOL, ))    # 异步开启进程
        pool.close()
        pool.join()

        # sender_list = [i for i in range(8001, 8011)]
        # sender_list = random.sample(sender_list, 5)
        # message_context = 'The context is 3'
        # node_number = 10
        # pool = multiprocessing.Pool(processes=10)   # 进程池一共有10进程
        # for i in range(1, 4):  # 每一个发送者的发送次数
        #     for sender in sender_list:
        #         pool.apply_async(post_message, (sender, message_context, node_number, ))    # 异步开启进程
        # pool.close()
        # pool.join() # 进程阻塞于主进程结束前

        prior_probability = '0.5'
        for i in range(8001, 8001+node_number): # 全部node执行rating
            r = requests.get('http://127.0.0.1:' + str(i) + '/rating', params='p='+prior_probability)
            # print(r.text)

        # 让旷工结算offset
        miner_list = [8099,8040,8052,8051,8069,8065,8083,8071,8096,8093,8095,8081,8090,8053,8082,8098,8085,8091,8094,8097]
        for i in miner_list:
            requests.get('http://127.0.0.1:' + str(i) + '/compute_offset')


        # 获取矿工的stake
        miners_stake = []
        for i in miner_list:
            # 防出错
            temp2 = requests.get('http://127.0.0.1:' + str(i) + '/forge_select').text
            if temp2 != '':
                temp = float(temp2)
            else:
                temp = 0

            if temp < 0:    # stake 为负数的设置为0
                temp = 0
            miners_stake.append(temp)
        # print(miners_stake)

        # POT_time = []
        # # PoT实验
        # for valid in range(1,11):
        #     print('正在进行',valid)
        #     starttime = time.time()
        #     total_weight = sum(miners_stake)
        #     winner_num = 1
        #     winner_PoT = 0
        #     m = valid # 验证者个数
        #     validator_set = set()
        #
        #     while winner_num != 0: # winner选举
        #         candidate = {}
        #         for i in range(miner_number):
        #             if miner_list[i] in validator_set:  # 已经被选过的不参与
        #                 continue
        #             post_object = {
        #                 "p": (miners_stake[i] / total_weight) if total_weight != 0 else 1,
        #                 "i": i
        #             }
        #             print(post_object)
        #             print(miners_stake)
        #             r = requests.post('http://127.0.0.1:' + str(miner_list[i]) + '/vrf_prove', json=post_object)
        #             if r.text == "selected":
        #                 # wait_to_select.remove()
        #                 candidate[miner_list[i]] = miners_stake[i]
        #         if len(candidate) >= 1:
        #             candidate = sorted(candidate.items(), key=lambda kv: kv[1], reverse=True)
        #             for i in range(len(candidate)):
        #                 winner_PoT = candidate[i][0]
        #                 total_weight -= candidate[i][1]
        #                 winner_num -= 1
        #                 if winner_num == 0:
        #                     break
        #
        #
        #     while m != 0:   # validator选举
        #         candidate = {}
        #         for i in range(miner_number):
        #             if miner_list[i] in validator_set or miner_list[i] == winner_PoT:  # 已经被选过的不参与
        #                 continue
        #             post_object = {
        #                 "p": (miners_stake[i] / total_weight) if total_weight != 0 else 1,
        #                 "m": m
        #             }
        #             print(post_object)
        #             print(miners_stake)
        #             r = requests.post('http://127.0.0.1:' + str(miner_list[i]) + '/vrf_prove', json=post_object)
        #             if r.text == "selected":
        #                 # wait_to_select.remove()
        #                 candidate[miner_list[i]] = miners_stake[i]
        #         if len(candidate) >= 1:
        #             candidate = sorted(candidate.items(), key=lambda kv: kv[1], reverse=True)
        #             for i in range(len(candidate)):
        #                 validator_set.add(candidate[i][0])
        #                 total_weight -= candidate[i][1]
        #                 m -= 1
        #                 if m == 0:
        #                     break
        #
        #     endtime = time.time()
        #     print('validator选举时间:', round(endtime - starttime, 2),'secs')
        #     POT_time.append(round(endtime - starttime, 2))
        #     print('winner is ', winner_PoT)
        #     print(validator_set)
        # print('时间列表:', POT_time)

        #PoW实验
        # starttime = time.time()
        # event = multiprocessing.Event()
        # jobs = []
        # for i in range(10):
        #     p = multiprocessing.Process(target=pow_test, args=(i, event,))  # 要把参数传过去！！！！！！！
        #     p.start()
        #     jobs.append(p)
        # while True:
        #     if event.is_set():
        #         for i in jobs:
        #             # Terminate each process
        #             i.terminate()   # 终止子进程
        #         break
        #     time.sleep(2)
        # endtime = time.time()
        # print('pow时间:', round(endtime - starttime, 2), 'secs')

        winner = miner_list[randon_weight(miners_stake)]
        print('The winner is %s' % winner)
        r = requests.get('http://127.0.0.1:' + str(winner) + '/forge')

        # print(r.text)

        # test
        for i in miner_list:
            fileName = 'chain' + str(i) + '.txt'
            r = requests.get('http://127.0.0.1:' + str(i) + '/chain')
            with open(fileName, 'w', encoding='utf-8') as file:
                file.write(str(r.text) + '\n')
            file.close()

        print('All has been done!')