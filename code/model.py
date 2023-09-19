import torch
from dataloader import BasicDataset
from torch import nn
import torch.nn.functional as F

class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def getUsersRating(self, users):
        raise NotImplementedError

class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()

    def bpr_loss(self, users, posVideo, negVideo, posVlogger, negVlogger, users2, posVlogger2, negVlogger2):
        """
        Parameters:
            users: users list
            pos: positive videos for corresponding users
            neg: negative videos for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError


class ProjectionHead(nn.Module):
    def __init__(
            self,
            config: dict,
    ):
        super().__init__()
        self.config = config
        projection_dim = self.config['projection_dim']
        dropout = self.config['projection_dropout']
        embedding_dim= self.config['latent_dim_rec']
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class VAGNN(BasicModel):
    def __init__(self,
                 config: dict,
                 dataset: BasicDataset):
        super(VAGNN, self).__init__()
        self.config = config
        self.dataset: dataloader.BasicDataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_videos = self.dataset.n_videos
        self.num_vloggers = self.dataset.n_vloggers
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['VAGNN_n_layers']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_video = torch.nn.Embedding(
            num_embeddings=self.num_videos, embedding_dim=self.latent_dim)
        self.embedding_vlogger = torch.nn.Embedding(
            num_embeddings=self.num_vloggers, embedding_dim=self.latent_dim)
        if self.config['pretrain'] == 0:
            #nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
            #nn.init.xavier_uniform_(self.embedding_video.weight, gain=1)
            #nn.init.xavier_uniform_(self.embedding_vlogger.weight, gain=1)
            #print('use xavier initilizer')
            # random normal init seems to be a better choice when VAGNN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_video.weight, std=0.1)
            nn.init.normal_(self.embedding_vlogger.weight, std=0.1)
            #world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_video.weight.data.copy_(torch.from_numpy(self.config['video_emb']))
            self.embedding_vlogger.weight.data.copy_(torch.from_numpy(self.config['vlogger_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()

        self.q = torch.nn.Parameter(torch.FloatTensor(self.latent_dim, self.latent_dim), requires_grad=True)
        nn.init.normal_(self.q, std=0.1)

        self.vlogger_reg=self.config['vlogger_reg']


        self.cl_temp=self.config['cl_temp']

        self.projection = ProjectionHead(self.config)
        self.userProjection = ProjectionHead(self.config)
        self.videoProjection = ProjectionHead(self.config)
        self.vloggerProjection = ProjectionHead(self.config)
        self.userProjection2 = ProjectionHead(self.config)
        self.videoProjection2 = ProjectionHead(self.config)
        self.vloggerProjection2 = ProjectionHead(self.config)


    def one_propagate(self, graph, A_feature, B_feature):
        # propagate
        features = torch.cat([A_feature, B_feature])
        all_features = [features]
        for i in range(self.n_layers):
            if isinstance(graph, list):
                all_emb = torch.sparse.mm(graph[i], features)
            else:
                all_emb = torch.sparse.mm(graph, features)
            all_features.append(all_emb)
        all_features = torch.stack(all_features, dim=1)
        light_out = torch.mean(all_features, dim=1)
        A_feature, B_feature = torch.split(light_out, [A_feature.size()[0], B_feature.size()[0]])
        return A_feature, B_feature

    def computer(self, ui_graph, ua_graph):
        #  =============================  video level propagation  =============================

        atom_users_feature, atom_videos_feature = self.one_propagate(
            ui_graph, self.embedding_user.weight, self.embedding_video.weight)
        atom_vloggers_feature = F.normalize(torch.matmul(self.Graph[2], atom_videos_feature))

        #  ============================= vlogger level propagation =============================

        non_atom_users_feature, non_atom_vloggers_feature = self.one_propagate(
            ua_graph, self.embedding_user.weight, self.embedding_vlogger.weight)

        non_atom_videos_feature = F.normalize(torch.matmul(self.Graph[4], non_atom_users_feature))

        users_feature = [atom_users_feature, non_atom_users_feature]
        vloggers_feature = [atom_vloggers_feature, non_atom_vloggers_feature]
        videos_feature = [atom_videos_feature, non_atom_videos_feature]

        return users_feature, videos_feature, vloggers_feature


    def getUsersRating(self, users):
        all_users, all_videos, all_vloggers = self.computer(self.Graph[0], self.Graph[1])

        users_feature_atom, users_feature_non_atom = [i[users] for i in all_users]  # batch_f
        vloggers_feature_atom, vloggers_feature_non_atom = all_vloggers  # b_f
        videos_feature_atom, videos_feature_non_atom = all_videos  # b_f

        vloggers_feature_atom = vloggers_feature_atom[self.dataset.vlogger_list]
        vloggers_feature_non_atom = vloggers_feature_non_atom[self.dataset.vlogger_list]

        ui = self.f(torch.mm(users_feature_atom, videos_feature_atom.t()) \
                    + torch.mm(users_feature_non_atom, videos_feature_non_atom.t()))  # batch_b
        ua = self.f(torch.mm(users_feature_atom, vloggers_feature_atom.t()) \
                    + torch.mm(users_feature_non_atom, vloggers_feature_non_atom.t()))

        videos_feature = (videos_feature_atom + videos_feature_non_atom) / 2
        vloggers_feature = (vloggers_feature_atom + vloggers_feature_non_atom) / 2

        c = torch.mm(videos_feature, self.q)  # i*q
        d = (torch.sum(c * vloggers_feature, 1))  # i*q*a

        weight = torch.sigmoid(d)


        return weight * ui + (1 - weight) * ua

    def getEmbedding(self, users, pos_videos, neg_videos, pos_vloggers, neg_vloggers, users2, pos_vloggers2, neg_vloggers2):
        all_users, all_videos, all_vloggers = self.computer(self.Graph[0], self.Graph[1])

        # train loader1
        users_emb0 = all_users[0][users]
        users_emb1 = all_users[1][users]
        posVideo_emb0 = all_videos[0][pos_videos]
        posVideo_emb1 = all_videos[1][pos_videos]
        negVideo_emb0 = all_videos[0][neg_videos]
        negVideo_emb1 = all_videos[1][neg_videos]
        posVlogger_emb0 = all_vloggers[0][pos_vloggers]
        posVlogger_emb1 = all_vloggers[1][pos_vloggers]
        negVlogger_emb0 = all_vloggers[0][neg_vloggers]
        negVlogger_emb1 = all_vloggers[1][neg_vloggers]

        users_emb_ego = self.embedding_user(users)
        posVideo_emb_ego = self.embedding_video(pos_videos)
        negVideo_emb_ego = self.embedding_video(neg_videos)
        posVlogger_emb_ego = self.embedding_vlogger(pos_vloggers)
        negVlogger_emb_ego = self.embedding_vlogger(neg_vloggers)

        # train loader2
        users_emb20 = all_users[0][users2]
        users_emb21 = all_users[1][users2]
        posVlogger_emb20 = all_vloggers[0][pos_vloggers2]
        posVlogger_emb21 = all_vloggers[1][pos_vloggers2]
        negVlogger_emb20 = all_vloggers[0][neg_vloggers2]
        negVlogger_emb21 = all_vloggers[1][neg_vloggers2]

        users_emb_ego2 = self.embedding_user(users2)
        posVlogger_emb_ego2 = self.embedding_vlogger(pos_vloggers2)
        negVlogger_emb_ego2 = self.embedding_vlogger(neg_vloggers2)

        return [users_emb0, users_emb1], [posVideo_emb0, posVideo_emb1], [negVideo_emb0, negVideo_emb1], \
               [posVlogger_emb0, posVlogger_emb1], [negVlogger_emb0, negVlogger_emb1], \
               users_emb_ego, posVideo_emb_ego, negVideo_emb_ego, posVlogger_emb_ego, negVlogger_emb_ego, \
               [users_emb20, users_emb21], [posVlogger_emb20, posVlogger_emb21], [negVlogger_emb20, negVlogger_emb21], \
               users_emb_ego2, posVlogger_emb_ego2, negVlogger_emb_ego2

    
    def calc_crosscl_loss(self,users_emb,posVideo_emb,posVlogger_emb):
        p_user_emb1 = users_emb[0]
        p_user_emb2 = users_emb[1]
        p_video_emb1 = posVideo_emb[0]
        p_video_emb2 = posVideo_emb[1]
        p_vlogger_emb1 = posVlogger_emb[0]
        p_vlogger_emb2 = posVlogger_emb[1]

        p_user_emb1 = self.userProjection(p_user_emb1)
        p_user_emb2 = self.userProjection(p_user_emb2)
        p_video_emb1 = self.videoProjection(p_video_emb1)
        p_video_emb2 = self.videoProjection(p_video_emb2)
        p_vlogger_emb1 = self.vloggerProjection(p_vlogger_emb1)
        p_vlogger_emb2 = self.vloggerProjection(p_vlogger_emb2)

        # group contrast
        normalize_emb_user1 = F.normalize(p_user_emb1, dim=1)
        normalize_emb_user2 = F.normalize(p_user_emb2, dim=1)
        normalize_emb_video1 = F.normalize(p_video_emb1, dim=1)
        normalize_emb_video2 = F.normalize(p_video_emb2, dim=1)
        normalize_emb_vlogger1 = F.normalize(p_vlogger_emb1, dim=1)
        normalize_emb_vlogger2 = F.normalize(p_vlogger_emb2, dim=1)

        pos_score_u = torch.sum(torch.mul(normalize_emb_user1, normalize_emb_user2), dim=1)
        pos_score_i = torch.sum(torch.mul(normalize_emb_video1, normalize_emb_video2), dim=1)
        pos_score_a = torch.sum(torch.mul(normalize_emb_vlogger1, normalize_emb_vlogger2), dim=1)

        ttl_score_u = torch.matmul(normalize_emb_user1, normalize_emb_user2.T)
        ttl_score_i = torch.matmul(normalize_emb_video1, normalize_emb_video2.T)
        ttl_score_a = torch.matmul(normalize_emb_vlogger1, normalize_emb_vlogger2.T)


        if self.cl_temp<0.05:
            ssl_logits_user = ttl_score_u - pos_score_u[:, None]  # [batch_size, num_users]
            ssl_logits_video = ttl_score_i - pos_score_i[:, None]  # [batch_size, num_users]
            ssl_logits_vlogger = ttl_score_a - pos_score_a[:, None]  # [batch_size, num_users]

            # InfoNCE Loss
            clogits_user = torch.mean(torch.logsumexp(ssl_logits_user / self.cl_temp, dim=1))
            clogits_video = torch.mean(torch.logsumexp(ssl_logits_video / self.cl_temp, dim=1))
            clogits_vlogger = torch.mean(torch.logsumexp(ssl_logits_vlogger / self.cl_temp, dim=1))
            cl_loss = (torch.mean(clogits_user) + torch.mean(clogits_video) + torch.mean(clogits_vlogger)) / 3
        else:
            pos_score_u = torch.exp(pos_score_u / self.cl_temp)
            ttl_score_u = torch.sum(torch.exp(ttl_score_u / self.cl_temp), dim=1)
            pos_score_i = torch.exp(pos_score_i / self.cl_temp)
            ttl_score_i = torch.sum(torch.exp(ttl_score_i / self.cl_temp), dim=1)
            pos_score_a = torch.exp(pos_score_a / self.cl_temp)
            ttl_score_a = torch.sum(torch.exp(ttl_score_a / self.cl_temp), dim=1)

            cl_loss = (torch.mean(torch.log(pos_score_u / ttl_score_u)) + torch.mean(
                torch.log(pos_score_i / ttl_score_i)) + torch.mean(torch.log(pos_score_a / ttl_score_a))) / 3
        return cl_loss

    def bpr_loss(self, users, posVideo, negVideo, posVlogger, negVlogger, users2, posVlogger2, negVlogger2):
        (users_emb, posVideo_emb, negVideo_emb, posVlogger_emb, negVlogger_emb,
         userEmb0, posVideoEmb0, negVideoEmb0, posVloggerEmb0, negVloggerEmb0,
         users_emb2, posVlogger_emb2, negVlogger_emb2,
         userEmb02, posVloggerEmb02, negVloggerEmb02
         ) = self.getEmbedding(users.long(), posVideo.long(), negVideo.long(), posVlogger.long(), negVlogger.long(),
                               users2.long(), posVlogger2.long(), negVlogger2.long())

        reg_loss = (1 / 2) * (torch.sum(torch.pow(userEmb0, 2))+
                              torch.sum(torch.pow(posVideoEmb0, 2))+
                              torch.sum(torch.pow(negVideoEmb0, 2))+
                              torch.sum(torch.pow(posVloggerEmb0, 2))+
                              torch.sum(torch.pow(negVloggerEmb0, 2))+
                              torch.sum(torch.pow(userEmb02, 2))+
                              torch.sum(torch.pow(posVloggerEmb02, 2))+
                              torch.sum(torch.pow(negVloggerEmb02, 2))
                              )/ float(len(users))

        ui_pos_scores = (torch.sum(users_emb[0] * posVideo_emb[0], 1)
                         + torch.sum(users_emb[1] * posVideo_emb[1], 1))
        ui_neg_scores = (torch.sum(users_emb[0] * negVideo_emb[0], 1)
                         + torch.sum(users_emb[1] * negVideo_emb[1], 1))
        ua_pos_scores = (torch.sum(users_emb[0] * posVlogger_emb[0], 1)
                         + torch.sum(users_emb[1] * posVlogger_emb[1], 1))
        ua_neg_scores = (torch.sum(users_emb[0] * negVlogger_emb[0], 1)
                         + torch.sum(users_emb[1] * negVlogger_emb[1], 1))
        ua_pos_scores2 = (torch.sum(users_emb2[0] * posVlogger_emb2[0], 1)
                          + torch.sum(users_emb2[1] * posVlogger_emb2[1], 1))
        ua_neg_scores2 = (torch.sum(users_emb2[0] * negVlogger_emb2[0], 1)
                          + torch.sum(users_emb2[1] * negVlogger_emb2[1], 1))

        # pos_weight
        videos_feature = (posVideo_emb[0] + posVideo_emb[1]) / 2
        vloggers_feature = (posVlogger_emb[0] + posVlogger_emb[1]) / 2

        c = torch.mm(videos_feature, self.q)  # i*q
        d = (torch.sum(c * vloggers_feature, 1))  # i*q*a
        pos_weight = torch.sigmoid(d)

        # neg_weight
        videos_feature = (negVideo_emb[0] + negVideo_emb[1]) / 2
        vloggers_feature = (negVlogger_emb[0] + negVlogger_emb[1]) / 2

        c = torch.mm(videos_feature, self.q)  # i*q
        d = (torch.sum(c * vloggers_feature, 1))  # i*q*a
        neg_weight = torch.sigmoid(d)

        pos_ui_weight = pos_weight * ui_pos_scores + (1 - pos_weight) * ua_pos_scores
        neg_ui_weight = neg_weight * ui_neg_scores + (1 - neg_weight) * ua_neg_scores

        bpr_loss =-torch.mean(F.logsigmoid(pos_ui_weight - neg_ui_weight)+ 10e-8) -self.vlogger_reg * torch.mean(
            F.logsigmoid(ua_pos_scores2 - ua_neg_scores2) + 10e-8)

        cl_loss=self.calc_crosscl_loss(users_emb,posVideo_emb,posVlogger_emb)
        return bpr_loss, reg_loss, cl_loss


    def forward(self, users, videos):
        # compute embedding
        all_users, all_videos, all_vloggers = self.computer(self.Graph[0], self.Graph[1])
        print('forward')
        # all_users, all_videos = self.computer()
        users_emb = all_users[users]
        videos_emb = all_videos[videos]
        inner_pro = torch.mul(users_emb, videos_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma
