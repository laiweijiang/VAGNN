import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import world
from world import cprint


class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")

    @property
    def n_users(self):
        raise NotImplementedError

    @property
    def n_videos(self):
        raise NotImplementedError

    @property
    def n_vloggers(self):
        raise NotImplementedError

    @property
    def trainDataSize(self):
        raise NotImplementedError

    @property
    def trainDataSize2(self):
        raise NotImplementedError

    @property
    def validDict(self):
        raise NotImplementedError

    @property
    def testDict(self):
        raise NotImplementedError

    @property
    def allPos(self):
        raise NotImplementedError

    @property
    def vlogger_list(self):
        raise NotImplementedError

    @property
    def allPos2(self):
        raise NotImplementedError

    def getUserVideoFeedback(self, users, videos):
        raise NotImplementedError

    def getUserPosVideos(self, users):
        raise NotImplementedError

    def getUserNegVideos(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg videos in super large dataset
        """
        raise NotImplementedError

    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A =
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError

class Loader(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    gowalla dataset
    """

    def __init__(self, config=world.config, path="../data/gowalla"):
        # train or test
        cprint(f'loading [{path}]')
        self.mode_dict = {'train': 0, "test": 1, "valid": 2}
        self.mode = self.mode_dict['train']
        self.n_user = 0
        self.n_video = 0
        self.n_vlogger = 0
        user_video_file = path + '/user_video_data.txt'
        user_vlogger_file = path + '/user_vlogger_data.txt'
        user_vlogger_file2 = path + '/user_vlogger_data2.txt'
        vlogger_video_file = path + '/vlogger_video_data.txt'
        size_file = path + '/data_size.txt'
        self.path = path
        trainUniqueUsers, trainVideo, trainUser,trainVlogger = [], [], [], []
        trainUniqueUsers2, trainVlogger2, trainUser2 = [], [], []
        testUniqueUsers, testVideo, testUser = [], [], []
        validUniqueUsers, validVideo, validUser = [], [], []

        self.traindataSize = 0
        self.traindataSize2 = 0
        self.validDataSize = 0
        self.testDataSize = 0

        with open(size_file) as f:
            self.n_user, self.n_video, self.n_vlogger = [int(s) for s in f.readline().split('\t')][:3]

        # u-i train
        with open(user_video_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    videos = [int(i) for i in l[1:-2]]
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(videos))
                    trainVideo.extend(videos)
                    self.traindataSize += len(videos)
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainVideo = np.array(trainVideo)

        # u-a
        with open(user_vlogger_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    vloggers = [int(i) for i in l[1:-2]]
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(vloggers))
                    trainVlogger.extend(vloggers)
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainVlogger = np.array(trainVlogger)

        # u-a train
        with open(user_vlogger_file2) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    vloggers = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    trainUniqueUsers2.append(uid)
                    trainUser2.extend([uid] * len(vloggers))
                    trainVlogger2.extend(vloggers)
                    self.traindataSize2 += len(vloggers)
        self.trainUniqueUsers2 = np.array(trainUniqueUsers2)
        self.trainUser2 = np.array(trainUser2)
        self.trainVlogger2 = np.array(trainVlogger2)

        # valid
        with open(user_video_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    videos = [int(i) for i in l[-2:-1]]
                    uid = int(l[0])
                    validUniqueUsers.append(uid)
                    validUser.extend([uid] * len(videos))
                    validVideo.extend(videos)

        self.validUniqueUsers = np.array(validUniqueUsers)
        self.validUser = np.array(validUser)
        self.validVideo = np.array(validVideo)

        # test
        with open(user_video_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    videos = [int(i) for i in l[-1:]]
                    uid = int(l[0])
                    testUniqueUsers.append(uid)
                    testUser.extend([uid] * len(videos))
                    testVideo.extend(videos)

        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testVideo = np.array(testVideo)


        # (vloggers,videos)
        with open(vlogger_video_file, 'r') as f:
            self.A_I_pairs = list(map(lambda s: tuple(int(i) for i in s[:-1].split('\t')), f.readlines()))
        indice = np.array(self.A_I_pairs, dtype=np.int32)
        values = np.ones(len(self.A_I_pairs), dtype=np.float32)
        self.ground_truth_a_i = sp.coo_matrix(
            (values, (indice[:, 0], indice[:, 1])), shape=(self.n_vlogger, self.n_video)).tocsr()

        self.A_I_pairs = sorted(self.A_I_pairs, key=lambda x: x[-1])

        # print(self.A_I_pairs)
        self.i_a_dict = dict(list(map(lambda x: x[::-1], self.A_I_pairs)))
        self._vloggerList = list(self.i_a_dict.values())

        # (users,videos), bipartite graph
        self.UserVideoNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainVideo)),
                                      shape=(self.n_user, self.n_video))
        self.users_D = np.array(self.UserVideoNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.] = 1
        self.videos_D = np.array(self.UserVideoNet.sum(axis=0)).squeeze()
        self.videos_D[self.videos_D == 0.] = 1.

        # (users,vloggers), bipartite graph
        self.UserVloggerNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainVlogger)),
                                         shape=(self.n_user, self.n_vlogger))
        self.users2_D = np.array(self.UserVloggerNet.sum(axis=1)).squeeze()
        self.users2_D[self.users2_D == 0.] = 1
        self.vloggers_D = np.array(self.UserVloggerNet.sum(axis=0)).squeeze()
        self.vloggers_D[self.vloggers_D == 0.] = 1.

        # (users,vloggers), bipartite graph
        self.UserVloggerNet2 = csr_matrix((np.ones(len(self.trainUser2)), (self.trainUser2, self.trainVlogger2)),
                                        shape=(self.n_user, self.n_vlogger))
        self.users2_D = np.array(self.UserVloggerNet2.sum(axis=1)).squeeze()
        self.users2_D[self.users2_D == 0.] = 1
        self.vloggers_D = np.array(self.UserVloggerNet2.sum(axis=0)).squeeze()
        self.vloggers_D[self.vloggers_D == 0.] = 1.

        # pre-calculate
        self._allPos = self.getUserPosVideos(list(range(self.n_user)))
        self._allPos2 = self.getUserPosVloggers(list(range(self.n_user)))
        self.__validDict = self.__build_valid()
        self.__testDict = self.__build_test()
        print(f"{world.dataset} is ready to go")

    @property
    def n_users(self):
        return self.n_user

    @property
    def n_videos(self):
        return self.n_video

    @property
    def n_vloggers(self):
        return self.n_vlogger

    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def trainDataSize2(self):
        return self.traindataSize2

    @property
    def testDict(self):
        return self.__testDict

    @property
    def validDict(self):
        return self.__validDict

    @property
    def allPos(self):
        return self._allPos

    @property
    def vlogger_list(self):
        return self._vloggerList

    @property
    def allPos2(self):
        return self._allPos2

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph(self):
        try:
            pre_adj_mat = sp.load_npz(self.path + '/user_video_s_pre_adj_mat.npz')
            print("successfully loaded...")
            user_video_graph = self._convert_sp_mat_to_sp_tensor(pre_adj_mat)
            user_video_graph = user_video_graph.coalesce().to(world.device)

            pre_adj_mat = sp.load_npz(self.path + '/user_vlogger_s_pre_adj_mat.npz')
            print("successfully loaded...")
            user_vlogger_graph = self._convert_sp_mat_to_sp_tensor(pre_adj_mat)
            user_vlogger_graph = user_vlogger_graph.coalesce().to(world.device)

            pre_adj_mat = sp.load_npz(self.path + '/pooling1_s_pre_adj_mat.npz')
            print("successfully loaded...")
            pool1_graph = self._convert_sp_mat_to_sp_tensor(pre_adj_mat)
            pool1_graph = pool1_graph.coalesce().to(world.device)

            pre_adj_mat = sp.load_npz(self.path + '/pooling2_s_pre_adj_mat.npz')
            print("successfully loaded...")
            pool2_graph = self._convert_sp_mat_to_sp_tensor(pre_adj_mat)
            pool2_graph = pool2_graph.coalesce().to(world.device)

            pre_adj_mat = sp.load_npz(self.path + '/pooling3_s_pre_adj_mat.npz')
            print("successfully loaded...")
            pool3_graph = self._convert_sp_mat_to_sp_tensor(pre_adj_mat)
            pool3_graph = pool3_graph.coalesce().to(world.device)

        except:
            print("generating adjacency matrix")

            user_video_adj, user_vlogger_adj = self.generate_user_video_vlogger_user_adj(self.UserVideoNet,
                                                                                         self.UserVloggerNet,
                                                                                         self.ground_truth_a_i, p=0.4,
                                                                                         q=0.1
                                                                                         )
            # u-i graph
            adj_mat = sp.dok_matrix((self.n_user + self.n_video, self.n_user + self.n_video), dtype=np.float32)
            adj_mat = adj_mat.tolil()
            R = user_video_adj.tolil()
            adj_mat[:self.n_user, self.n_user:] = R
            adj_mat[self.n_user:, :self.n_user] = R.T
            adj_mat = adj_mat.todok()
            # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

            rowsum = np.array(adj_mat.sum(axis=1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)
            norm_adj = d_mat.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat)
            norm_adj = norm_adj.tocsr()
            sp.save_npz(self.path + '/user_video_s_pre_adj_mat.npz', norm_adj)
            user_video_graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            user_video_graph = user_video_graph.coalesce().to(world.device)

            # u-a graph
            adj_mat = sp.dok_matrix((self.n_user + self.n_vlogger, self.n_user + self.n_vlogger), dtype=np.float32)
            adj_mat = adj_mat.tolil()
            R = user_vlogger_adj.tolil()
            adj_mat[:self.n_user, self.n_user:] = R
            adj_mat[self.n_user:, :self.n_user] = R.T
            adj_mat = adj_mat.todok()
            # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

            rowsum = np.array(adj_mat.sum(axis=1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)

            norm_adj = d_mat.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat)
            norm_adj = norm_adj.tocsr()
            sp.save_npz(self.path + '/user_vlogger_s_pre_adj_mat.npz', norm_adj)
            user_vlogger_graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            user_vlogger_graph = user_vlogger_graph.coalesce().to(world.device)

            # pooling garph1 (a-i)
            bundle_size = self.ground_truth_a_i.sum(axis=1) + 1e-8
            ai_graph = sp.diags(1 / bundle_size.A.ravel()) @ self.ground_truth_a_i
            sp.save_npz(self.path + '/pooling1_s_pre_adj_mat.npz', ai_graph)
            pool1_graph = self._convert_sp_mat_to_sp_tensor(ai_graph)
            pool1_graph = pool1_graph.coalesce().to(world.device)

            #  pooling graph2 (i-a)
            sp.save_npz(self.path + '/pooling2_s_pre_adj_mat.npz', ai_graph.T)
            pool2_graph = self._convert_sp_mat_to_sp_tensor(ai_graph.T)
            pool2_graph = pool2_graph.coalesce().to(world.device)

            # pooling graph3 (i-u)
            video_size = self.UserVideoNet.sum(axis=1) + 1e-8
            u_i_graph = sp.diags(1 / video_size.A.ravel()) @ self.UserVideoNet
            sp.save_npz(self.path + '/pooling3_s_pre_adj_mat.npz', u_i_graph.T)
            pool3_graph = self._convert_sp_mat_to_sp_tensor(u_i_graph.T)
            pool3_graph = pool3_graph.coalesce().to(world.device)

        self.Graph=[user_video_graph,user_vlogger_graph,pool1_graph,pool2_graph,pool3_graph]

        return self.Graph

    def generate_user_video_vlogger_user_adj(self,user_video_adj, user_vlogger_adj, vlogger_video_adj,p, q):
        user_video_adj = user_video_adj.tolil()
        user_vlogger_adj= user_vlogger_adj.tolil()
        vlogger_video_adj = vlogger_video_adj.tolil()

        new_user_video_adj = user_video_adj.copy()
        new_user_vlogger_adj = user_vlogger_adj.copy()
        for user in range(self.n_user):
            if np.random.random() <= p:
                current_node = np.random.choice(user_vlogger_adj[user].nonzero()[1])
                #U-A-U
                if np.random.random() <= q:
                    # get users interacted with vlogger
                    neighbors = np.array(user_vlogger_adj.T[current_node].nonzero()[1])
                    neighbors = neighbors[~np.in1d(neighbors, user)]
                    current_node = np.random.choice(neighbors)
                    related_videos = user_video_adj[user].nonzero()[1]
                    for video in related_videos:
                        new_user_video_adj[current_node, video] += 1
                #U-A-I
                else:
                    # get videos interacted with vlogger
                    neighbors = np.array(vlogger_video_adj[current_node].nonzero()[1])
                    current_node = np.random.choice(neighbors)
                    new_user_video_adj[user, current_node] += 1

            if np.random.random() <= p:
                current_node = np.random.choice(user_video_adj[user].nonzero()[1])  # 随机选择一个已交互的video作为起点
                #U-I-U
                if np.random.random() <= q:
                    # get users interacted with videos
                    neighbors = np.array(user_video_adj.T[current_node].nonzero()[1])
                    neighbors = neighbors[~np.in1d(neighbors, user)]
                    current_node = np.random.choice(neighbors)
                    related_vloggers = user_vlogger_adj[user].nonzero()[1]
                    for vlogger in related_vloggers:
                        new_user_vlogger_adj[current_node, vlogger] += 1

        return new_user_video_adj.tocsr(), new_user_vlogger_adj.tocsr()


    def __build_valid(self):
        """
        return:
            dict: {user: [videos]}
        """
        valid_data = {}
        for i, video in enumerate(self.validVideo):
            user = self.validUser[i]
            if valid_data.get(user):
                valid_data[user].append(video)
            else:
                valid_data[user] = [video]
        return valid_data

    def __build_test(self):
        """
        return:
            dict: {user: [videos]}
        """
        test_data = {}
        for i, video in enumerate(self.testVideo):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(video)
            else:
                test_data[user] = [video]
        return test_data

    def getUserVideoFeedback(self, users, videos):
        """
        users:
            shape [-1]
        videos:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserVideoNet[users, videos])
        return np.array(self.UserVideoNet[users, videos]).astype('uint8').reshape((-1,))

    def getUserPosVideos(self, users):
        posVideos = []
        posVloggers = []
        for user in users:
            posVideos.append(self.UserVideoNet[user].nonzero()[1])
            posVloggers.append(self.UserVloggerNet[user].nonzero()[1])
        return posVideos,posVloggers

    def getUserPosVloggers(self, users):
        posVloggers = []
        for user in users:
            posVloggers.append(self.UserVloggerNet2[user].nonzero()[1])
        return posVloggers
