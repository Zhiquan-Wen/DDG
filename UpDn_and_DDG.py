import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier, PaperClassifier
from fc import FCNet, GTH
from attention import Att_0, Att_1, Att_2, Att_3, Att_P, Att_PD, Att_3S
import torch
import random


class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        num_hid = opt.num_hid
        activation = opt.activation
        dropG = opt.dropG
        dropW = opt.dropW
        dropout = opt.dropout
        dropL = opt.dropL
        norm = opt.norm
        dropC = opt.dropC
        self.opt = opt

        self.w_emb = WordEmbedding(opt.ntokens, emb_dim=300, dropout=dropW)
        self.w_emb.init_embedding(opt.dataroot + 'glove6b_init_300d.npy')
        self.q_emb = QuestionEmbedding(in_dim=300, num_hid=num_hid, nlayers=1,
                                       bidirect=False, dropout=dropG, rnn_type='GRU')

        self.q_net = FCNet([self.q_emb.num_hid, num_hid], dropout=dropL, norm=norm, act=activation)
        self.gv_net = FCNet([opt.v_dim, num_hid], dropout=dropL, norm=norm, act=activation)

        self.gv_att_1 = Att_3(v_dim=opt.v_dim, q_dim=self.q_emb.num_hid, num_hid=num_hid, dropout=dropout, norm=norm,
                              act=activation)
        self.gv_att_2 = Att_3(v_dim=opt.v_dim, q_dim=self.q_emb.num_hid, num_hid=num_hid, dropout=dropout, norm=norm,
                              act=activation)
        self.classifier = SimpleClassifier(in_dim=num_hid, hid_dim=2 * num_hid, out_dim=opt.ans_dim,
                                           dropout=dropC, norm=norm, act=activation)

        self.normal = nn.BatchNorm1d(num_hid,affine=False)

    def forward(self, q, q_pos, img_idx, gv_pos, self_sup=True, k=10, train=True):

        """Forward
        q: [batch_size, seq_length]
        gv_pos: [batch, K, v_dim]
        self_sup: use negative images or not
        return: logits, not probs
        """ 

        out = {}
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb)  # run GRU on word embeddings [batch, q_dim]
        q_repr = self.q_net(q_emb)
        batch_size = q.size(0)
        logits, joint_main, _ = self.compute_predict(q_repr, q_emb, gv_pos)
        out['logit'] = logits

        if train:
            # positive q
            w_pos_emb = self.w_emb(q_pos)
            q_pos_emb = self.q_emb(w_pos_emb)
            q_pos_repr = self.q_net(q_pos_emb)

            # positive img
            pos_gv_pos = torch.stack([v_fea[img_idx[index][:k]] for index, v_fea in enumerate(gv_pos)], 0)

            logits_q, joint_main_q, _ = self.compute_predict(q_pos_repr, q_pos_emb, gv_pos)
            logits_v, joint_main_v, _ = self.compute_predict(q_repr, q_emb, pos_gv_pos)
            ensemble_logits = (logits + logits_q + logits_v) / 3
            
            out['ensemble_logit'] = ensemble_logits
            out['joint_pos_q'] = joint_main_q
            out['joint_pos_v'] = joint_main_v
            out['joint_fea'] = joint_main

        if self_sup:
            # construct an irrelevant Q-I pair for each instance
            index = random.sample(range(0, batch_size), batch_size)
            gv_neg = gv_pos[index]
            logits_neg_v, joint_neg_img, att_gv_neg = \
                self.compute_predict(q_repr, q_emb, gv_neg)

            index_q = random.sample(range(0, batch_size), batch_size)
            q_repr_neg = q_repr[index_q]
            q_emb_neg = q_emb[index_q]
            logits_neg_q, joint_neg_q, att_neg_q = self.compute_predict(q_repr_neg, q_emb_neg, gv_pos)


            out['logit_neg_v'] = logits_neg_v
            out['joint_neg_v'] = joint_neg_img
            out['logit_neg_q'] = logits_neg_q
            out['joint_neg_q'] = joint_neg_q
            
        # else:
        #     return logits_pos, att_gv_pos
        return out

    def compute_predict(self, q_repr, q_emb, v):

        att_1 = self.gv_att_1(v, q_emb)  # [batch, 1, v_dim]
        att_2 = self.gv_att_2(v, q_emb)  # [batch, 1, v_dim]
        att_gv = att_1 + att_2

        gv_embs = (att_gv * v)  # [batch, v_dim]
        gv_emb = gv_embs.sum(1)
        gv_repr = self.gv_net(gv_emb)

        joint_repr = q_repr * gv_repr

        joint_repr_normal = self.normal(joint_repr)
        logits = self.classifier(joint_repr_normal)

        return logits, joint_repr, att_gv

