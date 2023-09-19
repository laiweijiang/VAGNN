import world
import torch
import numpy as np
from dataloader import BasicDataset
from time import time

import os
from datetime import datetime
sample_ext = False

def UniformSample_original(dataset, neg_k):
    dataset: BasicDataset
    #allPos = dataset.allPos
    #start = time()
    S, S2 = UniformSample_original_python(dataset, neg_k)
    return S, S2


def UniformSample_original_python(dataset, neg_k):
    """
    the original impliment of BPR Sampling in VAGNN
    :return:
        np.array
    """
    total_start = time()
    dataset: BasicDataset
    user_num = dataset.trainDataSize
    user_num2 = dataset.trainDataSize2
    users = np.random.randint(0, dataset.n_users, user_num)
    users2 = np.random.randint(0, dataset.n_users, user_num2)
    allPos = dataset.allPos
    allPos2 = dataset.allPos2
    S = []
    S2 = []

    # u-i sample
    for i, user in enumerate(users):

        posVideoForUser = allPos[0][user]
        posVloggerForUser = allPos[1][user]

        if len(posVideoForUser) == 0:
            continue

        posindex = np.random.randint(0, len(posVideoForUser))
        posvideo = posVideoForUser[posindex]
        posvlogger = dataset.vlogger_list[posvideo]

        for i in range(neg_k):
            while True:
                # negvlogger = np.random.randint(0, dataset.n_vloggers)
                negvideo = np.random.randint(0, dataset.n_videos)
                negvlogger = dataset.vlogger_list[negvideo]
                if (negvideo in posVideoForUser) or (negvlogger in posVloggerForUser):
                    continue
                else:
                    break
            S.append([user, posvideo, negvideo, posvlogger, negvlogger])

    # u-a sample
    for i, user in enumerate(users2):

        posForUser = allPos2[user]
        if len(posForUser) == 0:
            continue

        posindex = np.random.randint(0, len(posForUser))
        posvlogger = posForUser[posindex]
        for i in range(neg_k):
            while True:
                negvlogger = np.random.randint(0, dataset.n_vloggers)
                if negvlogger in posForUser:
                    continue
                else:
                    break
            S2.append([user, posvlogger, negvlogger])

    return np.array(S), np.array(S2)


# ===================end samplers==========================
# =====================utils====================================

def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


def getFileName():
    file = f"lgn-{world.dataset}-{world.config['VAGNN_n_layers']}-{world.config['vlogger_reg']}-{world.config['cl_reg']}-{world.config['cl_temp']}-{world.config['latent_dim_rec']}"+datetime.fromtimestamp(time()).strftime('%m%d%H%M')
    return os.path.join(world.FILE_PATH, file)


def minibatch(*tensors, **kwargs):
    batch_size = kwargs.get('batch_size', world.config['bpr_batch_size'])

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):
    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


class timer:
    """
    Time context manager for code block
        with timer():
            do something
        timer.get()
    """
    from time import time
    TAPE = [-1]  # global time record
    NAMED_TAPE = {}

    @staticmethod
    def get():
        if len(timer.TAPE) > 1:
            return timer.TAPE.pop()
        else:
            return -1

    @staticmethod
    def dict(select_keys=None):
        hint = "|"
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                hint = hint + f"{key}:{value:.2f}|"
        else:
            for key in select_keys:
                value = timer.NAMED_TAPE[key]
                hint = hint + f"{key}:{value:.2f}|"
        return hint

    @staticmethod
    def zero(select_keys=None):
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                timer.NAMED_TAPE[key] = 0
        else:
            for key in select_keys:
                timer.NAMED_TAPE[key] = 0

    def __init__(self, tape=None, **kwargs):
        if kwargs.get('name'):
            timer.NAMED_TAPE[kwargs['name']] = timer.NAMED_TAPE[
                kwargs['name']] if timer.NAMED_TAPE.get(kwargs['name']) else 0.
            self.named = kwargs['name']
            if kwargs.get("group"):
                # TODO: add group function
                pass
        else:
            self.named = False
            self.tape = tape or timer.TAPE

    def __enter__(self):
        self.start = timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.named:
            timer.NAMED_TAPE[self.named] += timer.time() - self.start
        else:
            self.tape.append(timer.time() - self.start)


# ====================Metrics==============================
# =========================================================
def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos videos. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred / recall_n)
    precis = np.sum(right_pred) / precis_n
    return {'recall': recall, 'precision': precis}


def MRRatK_r(r, k):
    """
    Mean Reciprocal Rank
    """
    pred_data = r[:, :k]
    scores = np.log2(1. / np.arange(1, k + 1))
    pred_data = pred_data / scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)


def NDCGatK_r(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, videos in enumerate(test_data):
        length = k if k <= len(videos) else len(videos)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)




def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

# ====================end Metrics=============================
# =========================================================
