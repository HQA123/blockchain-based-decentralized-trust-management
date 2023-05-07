import copy
import random
import sys
from hashlib import sha256
import json
import time
from ast import literal_eval
from flask import Flask, request
import requests
import secrets
import ecvrf_edwards25519_sha512_elligator2
from scipy.stats import binom
import multiprocessing


class Block:
    def __init__(self, index, transactions, timestamp, previous_hash, block_log_message, nonce=0, trust_value=[]):
        self.index = index
        self.transactions = transactions
        self.timestamp = timestamp
        self.previous_hash = previous_hash
        self.nonce = nonce
        self.block_log_message = block_log_message  # 这是一个列表，记载了所有节点的NOL值
        self.trust_value = trust_value

    def compute_hash(self):
        """
        A function that return the hash of the block contents.
        """
        block_string = json.dumps(self.__dict__, sort_keys=True)
        return sha256(block_string.encode()).hexdigest()


class Blockchain:
    # difficulty of our PoW algorithm
    difficulty = 5
    n = 100 # 要和节点数同步
    def __init__(self):
        self.unconfirmed_transactions = []
        self.chain = []

    def create_genesis_block(self):
        """
        A function to generate genesis block and appends it to
        the chain. The block has index 0, previous_hash as 0, and
        a valid hash.
        """
        genesis_block = Block(0, [], 0, "0", block_log_message=[], trust_value=[0 for i in range(n)])
        genesis_block.hash = genesis_block.compute_hash()
        self.chain.append(genesis_block)

    @property
    def last_block(self):
        return self.chain[-1]

    def add_block(self, block, proof):
        """
        A function that adds the block to the chain after verification.
        Verification includes:
        * Checking if the proof is valid.
        * The previous_hash referred in the block and the hash of latest block
          in the chain match.
        """
        previous_hash = self.last_block.hash

        # if previous_hash != block.previous_hash:
        #     return False

        # if not Blockchain.is_valid_proof(block, proof):
        #     return False

        #block.hash = proof
        block.hash = block.compute_hash()   # pos方法不需要计算特定的哈希
        self.chain.append(block)
        return True

    @staticmethod
    def proof_of_trust(block):  # 或者叫proof of stake
        """
        consensus based on trust_value * message_number
        """
        message_number = len(block_log_message)

        # message_number = 0
        # for i in offset_list:
        #     message_number += i[0] + i[1]

        # print(message_number)
        # print(block.__dict__)
        # print(block.trust_value)
        stake = message_number * block.trust_value[int(port)-8001]
        return stake

    @staticmethod
    def proof_of_work(block, initial_num):
        """
        Function that tries different values of nonce to get a hash
        that satisfies our difficulty criteria.
        """
        global pow_flag
        block.nonce = 0
        j = 0
        computed_hash = block.compute_hash()
        while not computed_hash.startswith('0' * Blockchain.difficulty):
            j += 1
            block.nonce += initial_num * j
            print(block.nonce)
            # time.sleep(2)
            computed_hash = block.compute_hash()
            if pow_flag:
                break
        return computed_hash

    def add_new_transaction(self, transaction):
        self.unconfirmed_transactions.append(transaction)

    @classmethod
    def is_valid_proof(cls, block, block_hash):
        """
        Check if block_hash is valid hash of block and satisfies
        the difficulty criteria.
        """
        return (block_hash.startswith('0' * Blockchain.difficulty) and
                block_hash == block.compute_hash())

    @classmethod
    def check_chain_validity(cls, chain):
        result = True
        previous_hash = "0"

        for block in chain:
            block_hash = block.hash
            # remove the hash field to recompute the hash again
            # using `compute_hash` method.
            delattr(block, "hash")

            if not cls.is_valid_proof(block, block_hash) or \
                    previous_hash != block.previous_hash:
                result = False
                break

            block.hash, previous_hash = block_hash, block_hash

        return result

    def forge(self):
        """
        锻造一个新区块
        """
        global offset_list, block_log_message, n
        # print(block_log_message)
        last_block = self.last_block
        new_block = Block(index=last_block.index + 1,
                          transactions=self.unconfirmed_transactions,
                          timestamp=time.time(),
                          previous_hash=last_block.hash,
                          block_log_message=copy.deepcopy(block_log_message),
                          trust_value=copy.deepcopy(trust_value_list))
        print(new_block.__dict__)
        proof = self.proof_of_trust(new_block)
        self.add_block(new_block, proof)    # proof是stake

        # announce the recently mined block to the network
        post_object = {
            "node_address": "http://127.0.0.1:" + str(port)
        }
        # print(miner_list)
        for i in miner_list:
            if i != int(port):
                r = requests.post('http://127.0.0.1:' + str(i) + '/register_with', json=post_object)

        # 重置
        block_log_message.clear()
        offset_list = [[0, 0] for i in range(n)]

        return True

    def mine(self, initial_num):
        """
        This function serves as an interface to add the pending
        transactions to the blockchain by adding them to the block
        and figuring out Proof Of Work.
        """
        # if not self.unconfirmed_transactions:
        #     return False

        global offset_list, block_log_message, n
        # print(block_log_message)
        last_block = self.last_block
        new_block = Block(index=last_block.index + 1,
                          transactions=self.unconfirmed_transactions,
                          timestamp=time.time(),
                          previous_hash=last_block.hash,
                          block_log_message=copy.deepcopy(block_log_message),
                          trust_value=copy.deepcopy(trust_value_list))
        print(new_block.__dict__)
        proof = self.proof_of_work(new_block, initial_num)

        return True

def bayes_inference(p, credibility_list):
    """
    bayes method to compute decision probability.
    p is the prior probability of event e.
    """
    credits_positive = credits_negative = 1
    for i in credibility_list:
        credits_positive = credits_positive * i
        credits_negative = credits_negative * (1-i)
    bayes_probability = p * credits_positive / (p * credits_positive + (1-p) * credits_negative)
    return bayes_probability

def vrfverify(target_port):
    requests.get('http://127.0.0.1:' + str(target_port) + '/vrf_verify')


app = Flask(__name__)

n = 100  # 节点数
peers = set()   # the address to other participating members of the network
message_storage = []    # test for sending and receiving message
block_log_message = [] # test for storing message_storage+rated_score
offset_list = [[0, 0] for i in range(n)]    # store the offset every round. need to be clear when the block generate sucessfully
trust_value_list = [0 for i in range(n)]    # 信任值列表，初始值设为0
miner_list = [8099,8040,8052,8051,8069,8065,8083,8071,8096,8093,8095,8081,8090,8053,8082,8098,8085,8091,8094,8097]
pow_flag = False

# the node's copy of blockchain
blockchain = Blockchain()
blockchain.create_genesis_block()

# VRF verify
@app.route('/vrf_verify', methods=['GET'])
def vrf_verify():
    global public_key, pi_string
    alpha_string = b'I bid $100 for the horse named IntegrityChain'
    # public_key = bytes(request.get_json()["public_key"], 'utf-8')
    # pi_string = bytes(request.get_json()["pi_string"], 'utf-8')
    b_status, beta_string = ecvrf_edwards25519_sha512_elligator2.ecvrf_proof_to_hash(pi_string)
    #
    # Alice initially shares ONLY the beta_string with Bob
    #
    # Later, Bob validates Alice's subsequently shared public_key, pi_string, and alpha_string
    result, beta_string2 = ecvrf_edwards25519_sha512_elligator2.ecvrf_verify(public_key, pi_string, alpha_string)
    return "OK"
# VRF prove
@app.route('/vrf_prove', methods=['POST'])
def vrf_prove():
    global public_key, pi_string
    secret_key = secrets.token_bytes(nbytes=32)
    public_key = ecvrf_edwards25519_sha512_elligator2.get_public_key(secret_key)
    alpha_string = b'I bid $100 for the horse named IntegrityChain'
    p_status, pi_string = ecvrf_edwards25519_sha512_elligator2.ecvrf_prove(secret_key, alpha_string)
    p = request.get_json()["p"]
    # W = request.get_json()["W"]
    # winner_num = request.get_json()["winner_num"]
    # n = request.get_json()["n"]
    r = random.random()

    # print(winner_num, type(winner_num))

    # if r <= binom.pmf(k=1, n=n, p=p) or p == 1:
    if r <= p:
        pool = multiprocessing.Pool(processes=10)  # 进程池一共有10进程
        for i in miner_list:
            if i != int(port):
                pool.apply_async(vrfverify, (i,))  # 要把参数传过去！！！！！！！
        pool.close()
        pool.join()
        return "selected"
    return "not selected"

# 计算trust_value的增量
@app.route('/compute_offset', methods=['GET'])
def compute_offset():
    for i in range(len(trust_value_list)):
        # 更新trust_value_list
        if offset_list[i][0] + offset_list[i][1] == 0:
            trust_value_list[i] += 0
        else:
            trust_value_list[i] += (offset_list[i][0] - offset_list[i][1]) / (offset_list[i][0] + offset_list[i][1])

    # test
    fileName = 'collect_offset' + str(port) + '.txt'
    with open(fileName, 'w', encoding='utf-8') as file:
        file.write(str(offset_list) + '\n')
    file.close()

    # test
    fileName = 'compute_offset' + str(port) + '.txt'
    with open(fileName, 'w', encoding='utf-8') as file:
        file.write(str(trust_value_list) + '\n')
    file.close()

    return 'success!'

# 给miner使用的
@app.route('/collect_offset', methods=['POST']) # 接受10次
def collect_offset():
    '''
    input: rated_list from one user node
    output: offset_list = [[positive rate, negative rate], ... , ]
    '''
    # r = request.data.decode(encoding='utf-8')
    # print(r)
    # return r
    global block_log_message
    received_rating = request.data.decode(encoding='utf-8') # 得到字符串类型
    received_rating = literal_eval(received_rating)
    block_log_message.extend(received_rating)
    # print(block_log_message)
    for item in received_rating:
        if int(item['rating']) == 1:
            offset_list[int(item['UID_sender']) - 8001][0] += 1 # 要注意与序号对其，offset列表从0开始
        elif int(item['rating']) == -1:
            offset_list[int(item['UID_sender']) - 8001][1] += 1

    return json.dumps(offset_list)

@app.route('/rating', methods=['GET'])
def rating(Thr = 0.5):  # 传N次
    """
    Input: coming from message_storage list = [{NOL_trust, message_context, sender_ID}]
    Output: [UID_receiver, UID_sender, message_context, rating]
    """
    # def normfun(x, mu, sigma): #正态分布
    #     pdf = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    #     return pdf
    rated_list = []
    credibility_list = []
    prior_probability = float(request.args.get('p'))

    # print(prior_probability)
    # mu = 0
    # for i in range(report_numbers):
    #     x = np.random.normal(loc=mu)
    #     P = stats.norm.cdf(x, loc=0)
    #     credibility_set.append(P)

    # test
    fileName = 'message' + str(port) + '.txt'
    with open(fileName, 'w', encoding='utf-8') as file:
        for message in message_storage:
            file.write(str(message) + '\n')
    file.close()

    for message in message_storage:
        # print(message)
        credibility_list.append(message['NOL_trust'])

    # get bayes value
    bayes_value = bayes_inference(prior_probability, credibility_list)
    # 制作评分列表
    for message in message_storage:
        rated_message = {'UID_receiver' : int(port), 'UID_sender' : message['sender_ID'],
                         'message_context' : message['message_context']}
        if bayes_value >= Thr:
            # 评分全为1
            rated_message['rating'] = 1
        else:
            # 评分全为-1
            rated_message['rating'] = -1
        rated_list.append(rated_message)

    # print(message_storage)
    message_storage.clear() # 清除列表元素

    # 将rated_list转发给miner
    data = json.dumps(rated_list)
    miner_port = random.sample(miner_list, 1)    # rated_list 要发送给少量miner，一般认为miner占10%
    requests.post('http://127.0.0.1:' + str(miner_port[0]) + '/collect_offset', data=data)  # random.sample返回的是个列表

    # test
    fileName = 'rate' + str(port) + '.txt'
    with open(fileName, 'w', encoding='utf-8') as file:
        for rated_message in rated_list:
            file.write(str(rated_message) + '\n')
    file.close()

    # # test简化了vrfverify的传参过程，如果下面要测量PoT出块时间，则需要激活此段代码
    # global public_key, pi_string
    # secret_key = secrets.token_bytes(nbytes=32)
    # public_key = ecvrf_edwards25519_sha512_elligator2.get_public_key(secret_key)
    # alpha_string = b'I bid $100 for the horse named IntegrityChain'
    # p_status, pi_string = ecvrf_edwards25519_sha512_elligator2.ecvrf_prove(secret_key, alpha_string)

    return json.dumps(rated_list)

@app.route('/message_receiver', methods=['GET', 'POST'])
def message_receiver():
    if request.method == 'GET': # get方法为查询
        return json.dumps(message_storage)
    else:
        message = request.get_json()    # post方法为添加
        message_storage.append(message)
        # print(message_storage)
        return message


# endpoint to submit a new transaction. This will be used by
# our application to add new data (posts) to the blockchain
@app.route('/new_transaction', methods=['POST'])
def new_transaction():
    tx_data = request.get_json()
    required_fields = ["author", "content"]

    for field in required_fields:
        if not tx_data.get(field):
            return "Invalid transaction data", 404

    tx_data["timestamp"] = time.time()

    blockchain.add_new_transaction(tx_data)

    return "Success", 201


# endpoint to return the node's copy of the chain.
# Our application will be using this endpoint to query
# all the posts to display.
@app.route('/chain', methods=['GET'])
def get_chain():
    chain_data = []
    for block in blockchain.chain:
        chain_data.append(block.__dict__)
    return json.dumps({"length": len(chain_data),
                       "chain": chain_data,
                       "peers": list(peers)})


# endpoint to request the node to mine the unconfirmed
# transactions (if any). We'll be using it to initiate
# a command to mine from our application itself.
@app.route('/mine', methods=['GET'])
def mine_unconfirmed_transactions():
    global pow_flag
    pow_flag = False
    initial_num = float(request.args.get('i'))
    result = blockchain.mine(initial_num)
    # if not result:
    #     return "No transactions to mine"
    # else:
    #     # Making sure we have the longest chain before announcing to the network
    #     chain_length = len(blockchain.chain)
    #     # consensus()
    #     if chain_length == len(blockchain.chain):
    #         # announce the recently mined block to the network
    #         announce_new_block(blockchain.last_block)
    for i in range(10):
        requests.get('http://127.0.0.1:' + str(i+8001) + '/change_pow_flag')
    return "finish"

@app.route('/change_pow_flag', methods=['GET'])
def change_pow_flag():
    global pow_flag
    pow_flag = True
    return "OK"

# 选出区块
@app.route('/forge_select', methods=['GET'])
def forge_select():
    """
    return the stake of the forger
    """
    return json.dumps(blockchain.proof_of_trust(blockchain.last_block))

# 锻造区块
@app.route('/forge', methods=['GET'])
def forge_block():
    result = blockchain.forge()
    # Making sure we have the longest chain before announcing to the network
    chain_length = len(blockchain.chain)
    # consensus()




    # announce_new_block(blockchain.last_block)
    return "Block #{} is mined.".format(blockchain.last_block.block_log_message)

# endpoint to add new peers to the network.
# This is a middle function to realize function register_with
@app.route('/register_node', methods=['POST'])
def register_new_peers():
    node_address = request.get_json()["node_address"]
    # node_address = request.get_data().decode()
    # node_address = request.form["node_address"]
    # print(node_address.encode())

    if not node_address:
        return "Invalid data", 400

    # Add the node to the peer list
    peers.add(node_address)


    # Return the consensus blockchain to the newly registered node
    # so that he can sync
    return get_chain()


@app.route('/register_with', methods=['POST'])
def register_with_existing_node():
    """
    Internally calls the `register_node` endpoint to
    register current node with the node specified in the
    request, and sync the blockchain as well as peer data.
    """
    node_address = request.get_json()["node_address"]   # 需要程序自动发送给的地址
    if not node_address:
        return "Invalid data", 400

    data = {"node_address": request.host_url}   # 本程序的端口地址
    headers = {'Content-Type': "application/json"}

    # Make a request to register with remote node and obtain information
    response = requests.post(node_address + "/register_node",
                             data=json.dumps(data), headers=headers)

    if response.status_code == 200:
        global blockchain, peers
        # update chain and the peers
        chain_dump = response.json()['chain']
        blockchain = create_chain_from_dump(chain_dump)
        peers.update(response.json()['peers'])  # peers会和源节点同步

        global trust_value_list
        trust_value_list = blockchain.last_block.__dict__['trust_value']
        return "Registration successful", 200
    else:
        # if something goes wrong, pass it on to the API response
        return response.content, response.status_code


def create_chain_from_dump(chain_dump):
    """
    create a existive blockchain and check if it is tampered
    """
    generated_blockchain = Blockchain()
    generated_blockchain.create_genesis_block()
    for idx, block_data in enumerate(chain_dump):
        if idx == 0:
            continue  # skip genesis block
        block = Block(index=block_data["index"],
                      transactions=block_data["transactions"],
                      timestamp=block_data["timestamp"],
                      previous_hash=block_data["previous_hash"],
                      nonce=block_data["nonce"],
                      block_log_message=block_data["block_log_message"],
                      trust_value=block_data["trust_value"]
                      )
        proof = block_data['hash']
        added = generated_blockchain.add_block(block, proof)
        # print(block.__dict__["trust_value"])
        if not added:
            raise Exception("The chain dump is tampered!!")
    return generated_blockchain


# endpoint to add a block mined by someone else to
# the node's chain. The block is first verified by the node
# and then added to the chain.
@app.route('/add_block', methods=['POST'])
def verify_and_add_block():
    block_data = request.get_json()
    block = Block(block_data["index"],
                  block_data["transactions"],
                  block_data["timestamp"],
                  block_data["previous_hash"],
                  block_data["nonce"])

    proof = block_data['hash']
    added = blockchain.add_block(block, proof)

    if not added:
        return "The block was discarded by the node", 400

    return "Block added to the chain", 201


# endpoint to query unconfirmed transactions
@app.route('/pending_tx')
def get_pending_tx():
    return json.dumps(blockchain.unconfirmed_transactions)


def consensus():
    """
    Our naive consnsus algorithm. If a longer valid chain is
    found, our chain is replaced with it.
    """
    global blockchain

    longest_chain = None
    current_len = len(blockchain.chain)

    for node in peers:
        response = requests.get('{}chain'.format(node))
        length = response.json()['length']
        chain = response.json()['chain']
        if length > current_len and blockchain.check_chain_validity(chain):
            current_len = length
            longest_chain = chain

    if longest_chain:
        blockchain = longest_chain
        return True

    return False


def announce_new_block(block):
    """
    A function to announce to the network once a block has been mined.
    Other blocks can simply verify the proof of work and add it to their
    respective chains.
    """
    for peer in peers:
        url = "{}add_block".format(peer)
        headers = {'Content-Type': "application/json"}
        requests.post(url,
                      data=json.dumps(block.__dict__, sort_keys=True),
                      headers=headers)

# Uncomment this line if you want to specify the port number in the code
# type python node_server [port]
if __name__ == '__main__':
    port = sys.argv[1]
    app.run(debug=False, port=port, threaded = True)
