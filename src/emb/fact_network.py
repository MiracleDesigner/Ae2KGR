import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripleE(nn.Module):
    def __init__(self, args, num_entities):
        super(TripleE, self).__init__()
        conve_args = copy.deepcopy(args)    
        conve_args.model = 'conve'
        self.conve_nn = ConvE(conve_args, num_entities)
        conve_state_dict = torch.load(args.conve_state_dict_path)
        conve_nn_state_dict = get_conve_nn_state_dict(conve_state_dict)
        self.conve_nn.load_state_dict(conve_nn_state_dict)

        complex_args = copy.deepcopy(args)
        complex_args.model = 'complex'
        self.complex_nn = ComplEx(complex_args)

        distmult_args = copy.deepcopy(args)
        distmult_args.model = 'distmult'
        self.distmult_nn = DistMult(distmult_args)

    def forward(self, e1, r, conve_kg, secondary_kgs):
        complex_kg = secondary_kgs[0]
        distmult_kg = secondary_kgs[1]
        return (self.conve_nn.forward(e1, r, conve_kg)
                + self.complex_nn.forward(e1, r, complex_kg)
                + self.distmult_nn.forward(e1, r, distmult_kg)) / 3

    def forward_fact(self, e1, r, conve_kg, secondary_kgs):
        complex_kg = secondary_kgs[0]
        distmult_kg = secondary_kgs[1]
        return (self.conve_nn.forward_fact(e1, r, conve_kg)
                + self.complex_nn.forward_fact(e1, r, complex_kg)
                + self.distmult_nn.forward_fact(e1, r, distmult_kg)) / 3

class HyperE(nn.Module):
    def __init__(self, args, num_entities):
        super(HyperE, self).__init__()
        self.conve_nn = ConvE(args, num_entities)
        conve_state_dict = torch.load(args.conve_state_dict_path)
        conve_nn_state_dict = get_conve_nn_state_dict(conve_state_dict)
        self.conve_nn.load_state_dict(conve_nn_state_dict)

        complex_args = copy.deepcopy(args)
        complex_args.model = 'complex'
        self.complex_nn = ComplEx(complex_args)

    def forward(self, e1, r, conve_kg, secondary_kgs):
        complex_kg = secondary_kgs[0]
        return (self.conve_nn.forward(e1, r, conve_kg)
                + self.complex_nn.forward(e1, r, complex_kg)) / 2

    def forward_fact(self, e1, r, e2, conve_kg, secondary_kgs):
        complex_kg = secondary_kgs[0]
        return (self.conve_nn.forward_fact(e1, r, e2, conve_kg)
                + self.complex_nn.forward_fact(e1, r, e2, complex_kg)) / 2

class ComplEx(nn.Module):
    def __init__(self, args):
        super(ComplEx, self).__init__()

    def forward(self, e1, r, kg):
        def dist_mult(E1, R, E2):
            return torch.mm(E1 * R, E2.transpose(1, 0))

        E1_real = kg.get_entity_embeddings(e1)
        R_real = kg.get_relation_embeddings(r)
        E2_real = kg.get_all_entity_embeddings()
        E1_img = kg.get_entity_img_embeddings(e1)
        R_img = kg.get_relation_img_embeddings(r)
        E2_img = kg.get_all_entity_img_embeddings()

        rrr = dist_mult(R_real, E1_real, E2_real)
        rii = dist_mult(R_real, E1_img, E2_img)
        iri = dist_mult(R_img, E1_real, E2_img)
        iir = dist_mult(R_img, E1_img, E2_real)
        S = rrr + rii + iri - iir
        S = F.sigmoid(S)
        return S

    def forward_fact(self, e1, r, e2, kg):
        def dist_mult_fact(E1, R, E2):
            return torch.sum(E1 * R * E2, dim=1, keepdim=True)

        E1_real = kg.get_entity_embeddings(e1)
        R_real = kg.get_relation_embeddings(r)
        E2_real = kg.get_entity_embeddings(e2)
        E1_img = kg.get_entity_img_embeddings(e1)
        R_img = kg.get_relation_img_embeddings(r)
        E2_img = kg.get_entity_img_embeddings(e2)

        rrr = dist_mult_fact(R_real, E1_real, E2_real)
        rii = dist_mult_fact(R_real, E1_img, E2_img)
        iri = dist_mult_fact(R_img, E1_real, E2_img)
        iir = dist_mult_fact(R_img, E1_img, E2_real)
        S = rrr + rii + iri - iir
        S = F.sigmoid(S)
        return S

class ConvE(nn.Module):
    def __init__(self, args, num_entities):
        super(ConvE, self).__init__()
        self.args = args
        self.entity_dim = args.entity_dim
        self.relation_dim = args.relation_dim
        assert(args.emb_2D_d1 * args.emb_2D_d2 == args.entity_dim)
        assert(args.emb_2D_d1 * args.emb_2D_d2 == args.relation_dim)
        self.emb_2D_d1 = args.emb_2D_d1
        self.emb_2D_d2 = args.emb_2D_d2
        self.num_out_channels = args.num_out_channels
        self.w_d = args.kernel_size
        self.HiddenDropout = nn.Dropout(args.hidden_dropout_rate)
        self.FeatureDropout = nn.Dropout(args.feat_dropout_rate)

        # stride = 1, padding = 0, dilation = 1, groups = 1
        self.conv1 = nn.Conv2d(1, self.num_out_channels, (self.w_d, self.w_d), 1, 0)
        self.bn0 = nn.BatchNorm2d(1)
        self.bn1 = nn.BatchNorm2d(self.num_out_channels)
        self.bn2 = nn.BatchNorm1d(self.entity_dim)
        self.register_parameter('b', nn.Parameter(torch.zeros(num_entities)))
        h_out = 2 * self.emb_2D_d1 - self.w_d + 1
        w_out = self.emb_2D_d2 - self.w_d + 1
        self.feat_dim = self.num_out_channels * h_out * w_out
        self.fc = nn.Linear(self.feat_dim, self.entity_dim)

        if 'use_attention' in args.__dict__.keys():
            if args.use_attention:
                self.parameters_dim = 300
                self.W1 = nn.Parameter(torch.zeros(
                    size=(2 * self.entity_dim, self.parameters_dim)))
                nn.init.xavier_normal_(self.W1.data, gain=1.414)
                self.W2 = nn.Parameter(torch.zeros(size=(self.parameters_dim, self.entity_dim)))
                nn.init.xavier_normal_(self.W2.data, gain=1.414)
                self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, e1, r, kg):
        E1 = kg.get_entity_embeddings(e1).view(-1, 1, self.emb_2D_d1, self.emb_2D_d2)
        R = kg.get_relation_embeddings(r).view(-1, 1, self.emb_2D_d1, self.emb_2D_d2)
        E2 = kg.get_all_entity_embeddings() #137*200

        stacked_inputs = torch.cat([E1, R], 2)
        stacked_inputs = self.bn0(stacked_inputs)

        X = self.conv1(stacked_inputs)
        # X = self.bn1(X)
        X = F.relu(X)
        X = self.FeatureDropout(X)
        X = X.view(-1, self.feat_dim)
        X = self.fc(X)
        X = self.HiddenDropout(X)
        X = self.bn2(X)
        X = F.relu(X)
        X = torch.mm(X, E2.transpose(1, 0))
        X += self.b.expand_as(X)

        if 'use_attention' in self.args.__dict__.keys():
            if self.args.use_attention:
                E1_A = kg.get_entity_embeddings(e1)  # 512*200
                R_A = kg.get_relation_embeddings(r)  # 512*200
                query = torch.cat((E1_A, R_A), dim=1)
                Y = torch.mm(query, self.W1)
                Y = torch.mm(Y, self.W2)
                # Y = F.relu(Y)
                Y = self.leakyrelu(Y)
                Y = torch.mm(Y, E2.transpose(1, 0))

                X = self.args.alpha * X + (1 - self.args.alpha) * Y

        S = F.sigmoid(X) # 512*137
        return S

    def forward_fact(self, e1, r, e2, kg):
        """
        Compute network scores of the given facts.
        :param e1: [batch_size]
        :param r:  [batch_size]
        :param e2: [batch_size]
        :param kg:
        """
        # print(e1.size(), r.size(), e2.size())
        # print(e1.is_contiguous(), r.is_contiguous(), e2.is_contiguous())
        # print(e1.min(), r.min(), e2.min())
        # print(e1.max(), r.max(), e2.max())
        E1 = kg.get_entity_embeddings(e1).view(-1, 1, self.emb_2D_d1, self.emb_2D_d2)
        R = kg.get_relation_embeddings(r).view(-1, 1, self.emb_2D_d1, self.emb_2D_d2)
        E2 = kg.get_entity_embeddings(e2)

        stacked_inputs = torch.cat([E1, R], 2)
        stacked_inputs = self.bn0(stacked_inputs)

        X = self.conv1(stacked_inputs)
        # X = self.bn1(X)
        X = F.relu(X)
        X = self.FeatureDropout(X)
        X = X.view(-1, self.feat_dim)
        X = self.fc(X)
        X = self.HiddenDropout(X)
        X = self.bn2(X)
        X = F.relu(X)
        X = torch.matmul(X.unsqueeze(1), E2.unsqueeze(2)).squeeze(2)
        X += self.b[e2].unsqueeze(1)

        S = F.sigmoid(X)
        return S

class DistMult(nn.Module):
    def __init__(self, args):
        super(DistMult, self).__init__()

    def forward(self, e1, r, kg):
        E1 = kg.get_entity_embeddings(e1)
        R = kg.get_relation_embeddings(r)
        E2 = kg.get_all_entity_embeddings()
        S = torch.mm(E1 * R, E2.transpose(1, 0))
        S = F.sigmoid(S)
        return S

    def forward_fact(self, e1, r, e2, kg):
        E1 = kg.get_entity_embeddings(e1)
        R = kg.get_relation_embeddings(r)
        E2 = kg.get_entity_embeddings(e2)
        S = torch.sum(E1 * R * E2, dim=1, keepdim=True)
        S = F.sigmoid(S)
        return S

class GraphAttentionLayer:
    def __init__(self, args, state_dict={}):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = args.entity_dim
        self.parameters_dim = args.parameters_dim
        self.alpha = args.alpha
        self.dropout_rate = 0.3

        if not bool(state_dict):
            self.a = nn.Parameter(torch.zeros(
                size=(self.parameters_dim, 3 * self.in_features)))
            nn.init.xavier_normal_(self.a.data, gain=1.414)
            self.a_2 = nn.Parameter(torch.zeros(size=(1, self.parameters_dim)))
            nn.init.xavier_normal_(self.a_2.data, gain=1.414)
        else:
            self.a = state_dict['a']
            self.a_2 = state_dict['a_2']

        self.dropout = nn.Dropout(self.dropout_rate)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward_fact(self, e1, r, e2, kg):
        E1 = kg.get_entity_embeddings(e1)
        R = kg.get_relation_embeddings(r)
        E2 = kg.get_entity_embeddings(e2)

        S = torch.sum(E1 * R * E2, dim=1, keepdim=True)
        S = F.sigmoid(S)

        # [E1, E2, R]
        # facts = torch.cat((E1, E2, R), dim=1).t()
        # edge_m = self.a.mm(facts)
        # powers = -self.leakyrelu(self.a_2.mm(edge_m).squeeze())
        # S = torch.exp(powers).unsqueeze(1)

        # powers = F.relu(self.a_2.mm(edge_m).squeeze())
        # S = F.sigmoid(powers.unsqueeze(1))
        # S = F.sigmoid(S)
        return S

    # def init_model(self, state_dict):
    #     self.a = state_dict['a']
    #     self.a_2 = state_dict['a_2']

class MultiHeadGraphAttention(nn.Module):
    def __init__(self, args):
        super(MultiHeadGraphAttention, self).__init__()
        self.args = args
        self.attentions = [GraphAttentionLayer(args)
                           for _ in range(args.num_head)]
        # for i, attention in enumerate(self.attentions):
        #     self.add_module('attention_{}'.format(i), attention)

    def forward_fact(self, e1, r, e2, kg):
        S = torch.cat([att.forward_fact(e1, r, e2, kg)
                       for att in self.attentions], dim=1)
        # S = S.mean(dim=1)
        # S = S.unsqueeze(1)

        # S = torch.sum(S, dim=1)
        # S = S.unsqueeze(1)
        # S = F.sigmoid(S)

        # S = torch.sum(S, dim=1)
        S = S[:, 0]
        S = self.sigmoid(S)
        S = S.unsqueeze(1)

        return S

    def sigmoid(self, x):
        return 1/(1+torch.exp(-x))

    def load_attention_state_dict(self, state_dict):
        self.attentions = [GraphAttentionLayer(self.args, state_dict['attention_{}'.format(i)])
                           for i in range(self.args.num_head)]

def get_conve_nn_state_dict(state_dict):
    conve_nn_state_dict = {}
    for param_name in ['mdl.b', 'mdl.conv1.weight', 'mdl.conv1.bias', 'mdl.bn0.weight', 'mdl.bn0.bias',
                       'mdl.bn0.running_mean', 'mdl.bn0.running_var', 'mdl.bn1.weight', 'mdl.bn1.bias',
                       'mdl.bn1.running_mean', 'mdl.bn1.running_var', 'mdl.bn2.weight', 'mdl.bn2.bias',
                       'mdl.bn2.running_mean', 'mdl.bn2.running_var', 'mdl.fc.weight', 'mdl.fc.bias']:
        conve_nn_state_dict[param_name.split('.', 1)[1]] = state_dict['state_dict'][param_name]
    # for key in state_dict['state_dict']:
    #     conve_nn_state_dict[key.split('.', 1)[1]] = state_dict['state_dict'][key]
    return conve_nn_state_dict

def get_conve_kg_state_dict(state_dict):
    kg_state_dict = dict()
    for param_name in ['kg.entity_embeddings.weight', 'kg.relation_embeddings.weight']:
        kg_state_dict[param_name.split('.', 1)[1]] = state_dict['state_dict'][param_name]
    return kg_state_dict

def get_complex_kg_state_dict(state_dict):
    kg_state_dict = dict()
    for param_name in ['kg.entity_embeddings.weight', 'kg.relation_embeddings.weight',
                       'kg.entity_img_embeddings.weight', 'kg.relation_img_embeddings.weight']:
        kg_state_dict[param_name.split('.', 1)[1]] = state_dict['state_dict'][param_name]
    return kg_state_dict

def get_distmult_kg_state_dict(state_dict):
    kg_state_dict = dict()
    for param_name in ['kg.entity_embeddings.weight', 'kg.relation_embeddings.weight']:
        kg_state_dict[param_name.split('.', 1)[1]] = state_dict['state_dict'][param_name]
    return kg_state_dict

def get_attention_nn_state_dict(state_dict):
    # attention_nn_state_dict = {}
    # for i in range(len(state_dict)):
    #     param_name = 'attention_{}'.format(i)
    #     for sub_param_name in ['a', 'a_2']:
    #         attention_nn_state_dict[param_name][sub_param_name] = state_dict['params']['attention_{}'.format(i)][sub_param_name]
    # return attention_nn_state_dict
    return state_dict['params']

def get_attention_kg_state_dict(state_dict):
    kg_state_dict = dict()
    kg_state_dict['entity_embeddings.weight'] = state_dict['final_entity_embeddings']
    kg_state_dict['relation_embeddings.weight'] = state_dict['final_relation_embeddings']
    return kg_state_dict