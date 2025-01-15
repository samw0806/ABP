# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import ipdb
# from models.omics_encoder import SNN_Block

# class AdaptFormer(nn.Module):
#     def __init__(self, num_latents, dim):
#         super(AdaptFormer, self).__init__()

#         # Spectrogram (Spec)
#         self.spec_norm1 = nn.LayerNorm(256)  # 代替原来的 spec_enc.norm1
#         self.spec_attn = nn.MultiheadAttention(embed_dim=256, num_heads=1)  # 代替原来的 spec_enc.attn
#         self.spec_norm2 = nn.LayerNorm(256)  # 代替原来的 spec_enc.norm2
#         self.spec_mlp = nn.Sequential(  # 代替原来的 spec_enc.mlp
#             nn.Linear(256, dim),
#             nn.GELU(),
#             nn.Dropout(0.1),
#             nn.Linear(dim, 256)
#         )

#         # RGB
#         self.rgb_norm1 = nn.LayerNorm(256)  # 代替原来的 rgb_enc.norm1
#         self.rgb_attn = nn.MultiheadAttention(embed_dim=256, num_heads=1)  # 代替原来的 rgb_enc.attn
#         self.rgb_norm2 = nn.LayerNorm(256)  # 代替原来的 rgb_enc.norm2
#         self.rgb_mlp = nn.Sequential(  # 代替原来的 rgb_enc.mlp
#             nn.Linear(256, dim),
#             nn.GELU(),
#             nn.Dropout(0.1),
#             nn.Linear(dim, 256)
#         )

#         # Adapter params
#         self.act = QuickGELU()  # 你已有的激活函数
#         self.dropout = nn.Dropout(0.1)
#         self.dim = dim

#         # Spectrogram (Spec) 网络层
#         self.spec_down = nn.Linear(256, dim)
#         self.spec_up = nn.Linear(dim, 256)
#         nn.init.xavier_uniform_(self.spec_down.weight)
#         nn.init.zeros_(self.spec_down.bias)
#         nn.init.zeros_(self.spec_up.weight)
#         nn.init.zeros_(self.spec_up.bias)
#         self.spec_scale = nn.Parameter(torch.ones(1))

#         # RGB 图像处理网络层
#         self.rgb_down = nn.Linear(256, dim)
#         self.rgb_up = nn.Linear(dim, 256)
#         nn.init.xavier_uniform_(self.rgb_down.weight)
#         nn.init.zeros_(self.rgb_down.bias)
#         nn.init.zeros_(self.rgb_up.weight)
#         nn.init.zeros_(self.rgb_up.bias)
#         self.rgb_scale = nn.Parameter(torch.ones(1))

#         # Latents 参数
#         self.num_latents = num_latents
#         self.latents = nn.Parameter(torch.empty(1, num_latents, 256).normal_(std=0.02))
#         self.scale_a = nn.Parameter(torch.zeros(1))
#         self.scale_v = nn.Parameter(torch.zeros(1))

#     def attention(self,q,k,v): # requires q,k,v to have same dim
#         B, N, C = q.shape
#         attn = (q @ k.transpose(-2, -1)) * (C ** -0.5) # scaling
#         attn = attn.softmax(dim=-1)
#         x = (attn @ v).reshape(B, N, C)
#         return x
    
#     # Latent Fusion
#     def fusion(self, audio_tokens, visual_tokens):
#         # shapes
#         BS = audio_tokens.shape[0]
#         # concat all the tokens
#         concat_ = torch.cat((audio_tokens,visual_tokens),dim=1)
#         # cross attention (AV -->> latents)
#         fused_latents = self.attention(q=self.latents.expand(BS,-1,-1), k=concat_, v=concat_)
#         # cross attention (latents -->> AV)
#         audio_tokens = audio_tokens + self.scale_a * self.attention(q=audio_tokens, k=fused_latents, v=fused_latents)
#         visual_tokens = visual_tokens + self.scale_v * self.attention(q=visual_tokens, k=fused_latents, v=fused_latents)
#         return audio_tokens, visual_tokens

#     def forward_audio_AF(self, x):
#         x_down = self.spec_down(x)
#         x_down = self.act(x_down)
#         x_down = self.dropout(x_down)
#         x_up = self.spec_up(x_down)
#         return x_up

#     def forward_visual_AF(self, x):
#         x_down = self.rgb_down(x)
#         x_down = self.act(x_down)
#         x_down = self.dropout(x_down)
#         x_up = self.rgb_up(x_down)
#         return x_up


#     def forward(self, x, y):

#         # Bottleneck Fusion
#         x,y = self.fusion(x,y)
#         # Attn skip connections
#         x = x + self.spec_attn(self.spec_norm1(x),self.spec_norm1(x),self.spec_norm1(x))[0]
#         y = y + self.rgb_attn(self.rgb_norm1(y),self.rgb_norm1(y),self.rgb_norm1(y))[0]

#         # FFN + skip conections
#         x = x + self.spec_mlp(self.spec_norm2(x)) + self.forward_audio_AF(self.spec_norm2(x)) * self.spec_scale
#         y = y + self.rgb_mlp(self.rgb_norm2(y)) + self.forward_visual_AF(self.rgb_norm2(y)) * self.rgb_scale
#         return x,y

# # Example activation function for QuickGELU
# class QuickGELU(nn.Module):
#     def forward(self, x):
#         return x * torch.sigmoid(1.702 * x)
    
# class ABP(nn.Module):
#     def __init__(self,args,dim=126,omic_names=[],omics_input_dim=0, ):
#         super(ABP, self).__init__()
#         num_latents = args.num_latents
#         m_dim = args.m_dim
#         self.omic_sizes = args.omic_sizes
#         self.num_pathways = len(args.omic_sizes)
#         self.omics_input_dim = omics_input_dim
#         self.wsi_embedding_dim = args.encoding_dim
#         self.wsi_projection_dim = args.wsi_projection_dim
#         self.init_per_path_model(self.omic_sizes,args.omics_format)  # --->dim 256
#         num_classes = args.n_classes
#         encoder_layers = []
#         self.encoder_wsi = nn.Sequential(
#             nn.Linear(self.wsi_embedding_dim, self.wsi_projection_dim*2),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.2),
#             nn.Linear(self.wsi_projection_dim*2, self.wsi_projection_dim*2),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.2),
#             nn.Linear(self.wsi_projection_dim*2, self.wsi_projection_dim),
#         )
#         for i in range(m_dim):

#             # Vanilla Transformer Encoder (use for full fine tuning)
#             # encoder_layers.append(VanillaEncoder(num_latents=num_latents, spec_enc=self.v1.blocks[i], rgb_enc=self.v2.blocks[i]))

#             # Frozen Transformer Encoder with AdaptFormer 
#             encoder_layers.append(AdaptFormer(num_latents=num_latents, dim=dim))
             
#         self.audio_visual_blocks = nn.Sequential(*encoder_layers)

#         # final norm
#         self.spec_post_norm = nn.LayerNorm(256)
#         self.rgb_post_norm = nn.LayerNorm(256)

#         # classifier
#         self.classifier = nn.Sequential(
#             nn.Linear(256*2,num_classes)
#         )
    
#     def forward_encoder(self,x,y):     
#         # encoder forward pass
#         for blk in self.audio_visual_blocks:
#             x,y = blk(x,y)

#         x = self.spec_post_norm(x)
#         y = self.rgb_post_norm(y)

#         # return class token alone
#         x = x[:, 0]
#         y = y[:, 0]
#         return x,y
    
#     def init_per_path_model(self, omic_sizes, omics_format):

#         if omics_format in ['pathways','groups']:
#             # strategy 1, same with SurvPath
#             hidden = [self.wsi_projection_dim, self.wsi_projection_dim]
#             sig_networks = []
#             for input_dim in omic_sizes:
#                 fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
#                 for i, _ in enumerate(hidden[1:]):
#                     fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i + 1], dropout=0.25))
#                 sig_networks.append(nn.Sequential(*fc_omic))
#             self.sig_networks = nn.ModuleList(sig_networks)
#         elif omics_format == 'gene':
#             self.sig_networks = SNN_Block(dim1=self.omics_input_dim, dim2=self.wsi_projection_dim) # TODO: Maybe can use geneformer?

#         else:
#             raise ValueError('omics_format should be pathways, gene or groups')
    
#     def forward(self, **kwargs):
#         wsi = kwargs['x_wsi']  # wsi features (batch_size, feature_num, wsi_embedding_dim:768)
#         wsi = self.encoder_wsi(wsi)
#         x_omic = [kwargs['x_omic%d' % i] for i in range(1, self.num_pathways + 1)]  # omic features list (omic_size)

#         # ---> get
#         h_omic = [self.sig_networks[idx].forward(sig_feat) for idx, sig_feat in
#                   enumerate(x_omic)]  # each omic signature goes through it's own FC layer
#         h_omic_bag = torch.stack(h_omic)  # omic embeddings are stacked (to be used in co-attention)
#         h_omic_bag = h_omic_bag.permute(1, 0, 2)  # (batch_size, num_pathways, 256)


#         x,y = self.forward_encoder(wsi,h_omic_bag)
#         logits = torch.cat((x,y),dim=1)
#         logits = self.classifier(logits)
#         return logits    


from utils.loss_func import *
from utils.loss_func import NLLSurvLoss
from models.club import MIEstimator
from models.disentangle_transformer import MITransformerLayer
from models.omics_encoder import SNN_Block
import numpy as np
import ipdb
import torch.nn.functional as F

'''hyper-parameters of PIBD'''
BAG_SIZE = 512


'''KL divergence between two normal distributions'''
def KL_between_normals(q_distr, p_distr):
    mu_q, sigma_q = q_distr
    mu_p, sigma_p = p_distr
    k = mu_q.size(1)

    mu_diff = mu_p - mu_q
    mu_diff_sq = torch.mul(mu_diff, mu_diff)
    logdet_sigma_q = torch.sum(2 * torch.log(torch.clamp(sigma_q, min=1e-8)), dim=1)
    logdet_sigma_p = torch.sum(2 * torch.log(torch.clamp(sigma_p, min=1e-8)), dim=1)

    fs = torch.sum(torch.div(sigma_q ** 2, sigma_p ** 2), dim=1) + torch.sum(torch.div(mu_diff_sq, sigma_p ** 2), dim=1)
    two_kl = fs - k + logdet_sigma_p - logdet_sigma_q
    return two_kl * 0.5


'''Product of Experts: getting the joint distribution of multiple modalities.'''
class PoE(nn.Module):
    def __init__(self,modality_num=2, sample_num=50, seed = 1):
        super(PoE, self).__init__()

        self.sample_num = sample_num
        self.seed = seed

        phi = torch.ones(modality_num, requires_grad=True)
        self.phi = torch.nn.Parameter(phi)

    def forward(self, mu_list, var_list, eps=1e-8):
        t_sum = 0
        mu_t_sum = 0

        alpha = F.softmax(self.phi, dim=0)

        for idx, (mu, var) in enumerate(zip(mu_list, var_list)):
            T = 1 / (var + eps)

            t_sum += alpha[idx] * T
            mu_t_sum += mu * alpha[idx] * T

        mu = mu_t_sum / t_sum
        var = 1 / t_sum

        dim = mu.shape[1]
        batch_size = mu.shape[0]
        eps = self.gaussian_noise(samples=(batch_size, self.sample_num), k=dim, seed = self.seed)
        poe_features = torch.unsqueeze(mu, dim=1) + torch.unsqueeze(var, dim=1) * eps

        return poe_features

    def gaussian_noise(self, samples, k, seed):
        # works with integers as well as tuples
        if self.training:
            return torch.normal(torch.zeros(*samples, k), torch.ones(*samples, k)).cuda()
        else:
            return torch.normal(torch.zeros(*samples, k), torch.ones(*samples, k),
                                generator=torch.manual_seed(seed)).cuda() # must be the same seed as the training seed



'''Prototypical Information Bottleneck'''
class PIB(nn.Module):
    def __init__(self,
                 x_dim,
                 z_dim = 256,
                 beta = 1e-2,
                 sample_num = 50,
                 topk = 256,
                 num_classes = 4,
                 seed = 1):
        super(PIB, self).__init__()

        self.beta = beta
        self.sample_num = sample_num
        self.topk = topk
        self.num_classes = num_classes
        self.seed = seed
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Linear(x_dim, z_dim*2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(z_dim*2, z_dim*2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(z_dim*2, z_dim),
        )

        # decoder a simple logistic regression as in the paper
        self.decoder_logits = nn.Linear(z_dim, num_classes)

        # design proxies for histology images
        self.proxies = nn.Parameter(torch.empty([num_classes*2, z_dim*2]))
        torch.nn.init.xavier_uniform_(self.proxies, gain=1.0)
        self.proxies_dict = {"0,0":0,"0,1":1,"0,2":2,"0,3":3,"1,0":4,"1,1":5,"1,2":6,"1,3":7} #"censor,laberl":index

    def gaussian_noise(self, samples, K, seed):
        # works with integers as well as tuples
        if self.training:
            return torch.normal(torch.zeros(*samples, K), torch.ones(*samples, K)).cuda()
        else:
            return torch.normal(torch.zeros(*samples, K), torch.ones(*samples, K),
                                generator=torch.manual_seed(seed)).cuda() # must be the same seed as the training seed

    def encoder_result(self, x):

        encoder_output = self.encoder(x)

        return encoder_output

    def encoder_proxies(self):
        mu_proxy = self.proxies[:, :self.z_dim]
        sigma_proxy = torch.nn.functional.softplus(self.proxies[:, self.z_dim:]) # Make sigma always positive

        return mu_proxy, sigma_proxy

    def forward(self,x, y=None, c=None):
        #feature number
        feature_num = x.shape[1]

        # get z from encoder
        z = self.encoder_result(x)

        # get mu and sigma from proxies
        mu_proxy, sigma_proxy = self.encoder_proxies()

        # sample
        eps_proxy = self.gaussian_noise(samples=([self.num_classes*2, self.sample_num]), K=self.z_dim,seed = self.seed)
        z_proxy_sample = mu_proxy.unsqueeze(dim=1) + sigma_proxy.unsqueeze(dim=1) * eps_proxy
        z_proxy = torch.mean(z_proxy_sample, dim=1)

        # get attention maps
        z_norm = F.normalize(z,dim=2) # normalize z in the feature dimension
        z_proxy_norm = torch.unsqueeze(F.normalize(z_proxy),dim=0)
        att = torch.matmul(z_norm, torch.transpose(z_proxy_norm, 1, 2))

        if y is None and c is None:
            '''validating and testing'''

            # get the proxy with the highest attention map
            att_unbind_proxy = torch.cat(torch.unbind(att, dim=1), dim=1)
            _, att_topk_proxy_idx = torch.topk(att_unbind_proxy, self.topk, dim=1)
            att_topk_proxy_idx = att_topk_proxy_idx % (self.num_classes*2)

            # get the positive proxy index
            positive_proxy_idx,_ = torch.mode(att_topk_proxy_idx, dim=1)
            positive_proxy_idx = positive_proxy_idx.unsqueeze(1).repeat(1, self.z_dim).unsqueeze(dim=1)#batch_size, index,z_dim

            proxy_loss = None

            # # Original code
            # # "strategy 1: use the proxy with the highest frequency"
            # # get the proxy with the highest attention map for each sample
            # mu_topk = torch.empty([x.shape[0], self.z_dim]).cuda()
            # sigma_topk = torch.empty([x.shape[0], self.z_dim]).cuda()
            # for i in range(x.shape[0]):
            #     positive_proxy_idx = torch.argmax(torch.bincount(att_topk_proxy_idx[i]))
            #     positive_proxy_idx = positive_proxy_idx.repeat(1, self.z_dim)
            #     mu_topk[i] = torch.gather(mu_proxy, 0, positive_proxy_idx)
            #     sigma_topk[i] = torch.gather(sigma_proxy, 0, positive_proxy_idx)

            # # "strategy 2: use the proxy with the highest attention map" (In practice, two strategies are similar in performance.)
            # # get the topk attention of each prototype
            # if self.topk > att.shape[1]:
            #     attn_topk_new = att
            # else:
            #     attn_topk_new,_ = torch.topk(att,self.topk,dim=1)
            # #caculate the mean of the topk attention
            # attn_topk_new_mean = torch.mean(attn_topk_new,dim=1)
            # #get the index of the topk attention
            # _, attn_topk_new_idx = torch.topk(attn_topk_new_mean,1, dim=1)
            # mu_topk = torch.empty([x.shape[0], self.z_dim]).cuda()
            # sigma_topk = torch.empty([x.shape[0], self.z_dim]).cuda()
            # for i in range(x.shape[0]):
            #     positive_proxy_idx = attn_topk_new_idx[i].repeat(1, self.z_dim)
            #     mu_topk[i] = torch.gather(mu_proxy, 0, positive_proxy_idx)
            #     sigma_topk[i] = torch.gather(sigma_proxy, 0, positive_proxy_idx)

        else:
            '''training'''
            # get proxy_index for each sample
            proxy_indices = [self.proxies_dict[str(int(c_item)) + "," + str(int(y_item))] for c_item, y_item in
                             zip(c, y)]
            proxy_indices = torch.tensor(proxy_indices).long().cuda()
            mask = torch.zeros_like(att, dtype=torch.bool).cuda()
            mask[torch.arange(att.size(0)), :, proxy_indices] = True
            # get att_positive for each sample
            att_positive = torch.masked_select(att, mask).view(att.size(0), att.size(1), 1)
            # get att_negative for each sample
            att_negative = torch.masked_select(att, ~mask).view(att.size(0), att.size(1), -1)

            # calculate proxy loss
            att_topk_positive, att_topk_idx_positive = torch.topk(att_positive.squeeze(dim=2), self.topk, dim=1)
            att_topk_negative, _ = torch.topk(att_negative, self.topk, dim=1)
            att_positive_mean = torch.mean(att_topk_positive, dim=1)
            att_negative_mean = torch.mean(torch.mean(att_topk_negative, dim=1),dim=1)
            proxy_loss = -(att_positive_mean-att_negative_mean).mean()

            positive_proxy_idx = proxy_indices.unsqueeze(1).repeat(1, self.z_dim).unsqueeze(dim=1)


            ## Original code
            # att = torch.matmul(z_norm, torch.transpose(z_proxy_norm, 1, 2))
            # att_positive = torch.empty([x.shape[0], x.shape[1], 1]).cuda()
            # att_negative = torch.empty([x.shape[0], x.shape[1], self.num_classes * 2 - 1]).cuda()
            # mu_topk = torch.empty([x.shape[0], self.z_dim]).cuda()
            # sigma_topk = torch.empty([x.shape[0], self.z_dim]).cuda()
            # for i in range(x.shape[0]):
            #     proxy_index = self.proxies_dict[str(int(c[i])) + "," + str(int(y[i]))]
            #     z_proxy_spe = z_proxy_norm[:, int(proxy_index), :].unsqueeze(dim=0)
            #     att_positive[i] = torch.matmul(z_norm[i, :, :].unsqueeze(dim=0), torch.transpose(z_proxy_spe, 1, 2))
            #     att_negative[i] = torch.cat((att[i, :, :proxy_index], att[i, :, proxy_index + 1:]), dim=1)
            #     mu_topk[i] = mu_proxy[proxy_index]
            #     sigma_topk[i] = sigma_proxy[proxy_index]

        # Gather mu_proxy and sigma_proxy for each sample
        mu_proxy_repeat = mu_proxy.repeat(x.shape[0], 1, 1)  # batch_size, num_classes*2, z_dim
        sigma_proxy_repeat = sigma_proxy.repeat(x.shape[0], 1, 1)
        mu_topk = torch.gather(mu_proxy_repeat, 1, positive_proxy_idx).squeeze(dim=1)
        sigma_topk = torch.gather(sigma_proxy_repeat, 1, positive_proxy_idx).squeeze(dim=1)

        att_unbind = torch.cat(torch.unbind(att, dim=2), dim=1)
        # get topk z features from attention maps
        att_topk, att_topk_idx = torch.topk(att_unbind, self.topk, dim=1)
        att_topk_idx = att_topk_idx % feature_num
        # get topk z features from z
        z_topk = torch.gather(z, 1, att_topk_idx.unsqueeze(dim=2).repeat(1, 1, self.z_dim))

        decoder_logits_proxy = torch.mean(self.decoder_logits(z_proxy_sample), dim=1)

        return decoder_logits_proxy, mu_proxy, sigma_proxy, z_topk, mu_topk, sigma_topk, proxy_loss


'''Prototypical Information Bottleneck and Disentanglement'''
class ABP(nn.Module):
    def __init__(
            self,
            args,
            omic_names=[],
            omics_input_dim=0,
    ):
        super(ABP, self).__init__()

        # ---> general props
        self.num_pathways = len(args.omic_sizes)
        self.num_classes = args.n_classes
        self.omic_sizes = args.omic_sizes
        self.args = args

        # ---> wsi props
        self.wsi_embedding_dim = args.encoding_dim
        self.wsi_projection_dim = args.wsi_projection_dim

        # ---> omics preprocessing for captum
        if omic_names != []:
            self.omic_names = omic_names
            all_gene_names = []
            for group in omic_names:
                all_gene_names.append(group)
            all_gene_names = np.asarray(all_gene_names)
            all_gene_names = np.concatenate(all_gene_names)
            all_gene_names = np.unique(all_gene_names)
            all_gene_names = list(all_gene_names)
            self.all_gene_names = all_gene_names

        # ---> omic props
        self.omics_input_dim = omics_input_dim
        self.init_per_path_model(self.omic_sizes,args.omics_format)  # --->dim 256

        self.bag_size = BAG_SIZE
        self.topk_wsi = int(self.bag_size * args.ratio_wsi)
        self.topk_omics = int(self.num_pathways * args.ratio_omics)
        self.PIB_wsi = PIB(self.wsi_embedding_dim, self.wsi_projection_dim, num_classes=self.num_classes,
                                     topk=self.topk_wsi, sample_num=args.sample_num, seed=args.seed)
        self.PIB_omics = PIB(self.wsi_projection_dim, self.wsi_projection_dim, num_classes=self.num_classes,
                                      topk=self.topk_omics,sample_num=args.sample_num, seed=args.seed)

        # ---> Disentanglement
        self.PID = MITransformerLayer(
            dim=self.wsi_projection_dim,
            num_heads=4,
            mlp_ratio=1.,
            qkv_bias=True,
            attn_drop=0.1,
            proj_drop=0.1,
            drop_path=0.1,
        )

        # ---> classifier
        self.to_logits = nn.Sequential(
            nn.Linear(self.wsi_projection_dim * 3, self.wsi_projection_dim),
            nn.ReLU(),
            nn.Linear(self.wsi_projection_dim, self.num_classes),
        )

        # ---> PoE
        self.PoE = PoE(modality_num=2, sample_num=args.sample_num, seed=args.seed)

        # --->CLUB
        self.CLUB = MIEstimator(self.wsi_projection_dim)

        # SurvLoss
        self.loss_surv = NLLSurvLoss(alpha=0.5)
        self.is_net = args.net
        if args.net == True:
            self.net = FCNetWithMI(self.wsi_projection_dim)
            

    def init_per_path_model(self, omic_sizes, omics_format):

        if omics_format in ['pathways','groups']:
            # strategy 1, same with SurvPath
            hidden = [self.wsi_projection_dim, self.wsi_projection_dim]
            sig_networks = []
            for input_dim in omic_sizes:
                fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
                for i, _ in enumerate(hidden[1:]):
                    fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i + 1], dropout=0.25))
                sig_networks.append(nn.Sequential(*fc_omic))
            self.sig_networks = nn.ModuleList(sig_networks)

            # strategy 2, less parameters
            # max_size = max(omic_sizes)
            # fc_omic = [SNN_Block(dim1=max_size, dim2=hidden[0])]
            # for i, _ in enumerate(hidden[1:]):
            #     fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i + 1], dropout=0.25))
            # self.sig_networks = nn.Sequential(*fc_omic)

        elif omics_format == 'gene':
            self.sig_networks = SNN_Block(dim1=self.omics_input_dim, dim2=self.wsi_projection_dim) # TODO: Maybe can use geneformer?

        else:
            raise ValueError('omics_format should be pathways, gene or groups')

    def info_nec_loss(self,z1, z2, temperature=0.1, device='cuda'):
        # z1, z2: batch_size x embedding_dim
        # 计算相似性（点积）
        sim = torch.matmul(z1, z2.T) / temperature  # (batch_size, batch_size)
        
        # 使用softmax来计算对比损失
        labels = torch.arange(z1.size(0)).to(device)  # 正样本的标签，应该匹配对角线上的元素
        
        # 正样本分数
        pos_sim = torch.diagonal(sim)  # 取对角线作为正样本的相似度
        
        # 对比损失（负对比损失）
        loss = F.cross_entropy(sim, labels)
        
        return loss

    def forward(self, **kwargs):
        wsi = kwargs['x_wsi']  # wsi features (batch_size, feature_num, wsi_embedding_dim:768)
        x_omic = [kwargs['x_omic%d' % i] for i in range(1, self.num_pathways + 1)]  # omic features list (omic_size)
        return_attn = kwargs["return_attn"]
        y = kwargs["y"]
        c = kwargs["c"]

        # ---> get
        h_omic = [self.sig_networks[idx].forward(sig_feat) for idx, sig_feat in
                  enumerate(x_omic)]  # each omic signature goes through it's own FC layer
        h_omic_bag = torch.stack(h_omic)  # omic embeddings are stacked (to be used in co-attention)
        h_omic_bag = h_omic_bag.permute(1, 0, 2)  # (batch_size, num_pathways, 256)

        decoder_logits_proxy_omic, mu_proxy_omic, sigma_proxy_omic, h_omic_bag, \
        mu_topk_omic, sigma_topk_omic, proxy_loss_omic = self.PIB_omics(h_omic_bag, y, c)

        iter_num = self.args.num_patches // self.bag_size # use subset of patches, and average the results
        logits = torch.empty((iter_num, wsi.size(0), self.num_classes)).cuda()
        mimin_total,mimin_loss_total,IB_loss_proxy,proxy_loss = 0.0,0.0,0.0,0.0
        for i in range(iter_num):
            # strategy 1 random sampling from patches for each bag, which can reduce the computation
            idx = torch.randperm(wsi.size(1))[:self.bag_size]
            h_wsi_bag = wsi[:, idx, :]  #(batch_size, bag_num, 768)

            # ---> get wsi embeddings
            decoder_logits_proxy_wsi, mu_proxy_wsi, sigma_proxy_wsi, h_wsi_bag, mu_topk_wsi, sigma_topk_wsi,\
            proxy_loss_wsi = self.PIB_wsi(h_wsi_bag,y=y,c=c)

            # ---> PoE
            mu_list = [mu_topk_wsi, mu_topk_omic]
            var_list = [sigma_topk_wsi, sigma_topk_omic]
            poe_features = self.PoE(mu_list, var_list)
            poe_features = torch.mean(poe_features, dim=1).unsqueeze(dim=1)

            poe_embed = poe_features.expand(h_wsi_bag.shape[0], 1,-1)

            if return_attn:
                histology, pathways, global_embed, attns = self.PID(h_wsi_bag, h_omic_bag, poe_embed,return_attn)
            else:
                histology, pathways, global_embed, = self.PID(h_wsi_bag, h_omic_bag, poe_embed, return_attn)

            # ----> aggregate
            histology = torch.mean(histology, dim=1)
            pathways = torch.mean(pathways, dim=1)
            global_embed = torch.mean(global_embed, dim=1)

            mimin = self.CLUB.forward_glob(histology, pathways, global_embed)
            mimin_loss = self.CLUB.learning_loss_glob(histology, pathways, global_embed)
            mimin += self.info_nec_loss(histology, pathways)
            # mimin += self.info_nec_loss(torch.cat((histology, pathways), dim=1), global_embed)

            if self.is_net == True:
                pathways,l1,l2 = self.net(pathways)


            logits[i] = self.to_logits(torch.cat([histology, pathways, global_embed], dim=-1))

            if self.training:
                mimin_total += mimin
                mimin_loss_total += mimin_loss
                if self.is_net == True:
                    mimin_total += l1
                    mimin_loss_total += l2

                IB_loss_proxy += self.args.alpha * self.get_loss_proxy(decoder_logits_proxy_wsi, self.loss_surv) \
                                 + self.args.alpha * self.get_loss_proxy(decoder_logits_proxy_omic, self.loss_surv) \
                                 + self.args.beta * self.get_KL_loss(mu_proxy_wsi, sigma_proxy_wsi) \
                                 + self.args.beta * self.get_KL_loss(mu_proxy_omic, sigma_proxy_omic)
                proxy_loss += (proxy_loss_wsi + proxy_loss_omic)

        IB_loss_proxy /= iter_num
        proxy_loss /= iter_num
        mimin_total /= iter_num
        mimin_loss_total /= iter_num

        # if self.training:
        #     print('mimin:{:.4f},mimin_loss:{:.4f}'.format(mimin_total, mimin_loss_total))

        logits = torch.mean(logits, dim=0)

        if return_attn:
            return logits, attns
        else:
            return logits, IB_loss_proxy, proxy_loss, mimin_total, mimin_loss_total
            # return logits, IB_loss_proxy, proxy_loss, mimin_total


    def get_loss_proxy(self, x, loss):
        censor = torch.empty([self.num_classes * 2]).cuda()
        for i in range(self.num_classes):
            censor[i] = 0
            censor[i + self.num_classes] = 1
        y = torch.arange(0, self.num_classes).repeat(2).cuda()
        loss_proxy = loss(h=x, y=y, t=None, c=censor)

        return loss_proxy

    def get_KL_loss(self, mu, std):
        '''
        :param mu: [batch_size, dimZ]
        :param std: [batch_size, dimZ]
        :return:
        '''
        # KL divergence between prior and posterior
        prior_z_distr = torch.zeros_like(mu), torch.ones_like(std)
        encoder_z_distr = mu, std

        I_zx_bound = torch.mean(KL_between_normals(encoder_z_distr, prior_z_distr))

        return torch.mean(I_zx_bound)
    

class FCNetWithMI(nn.Module):
    def __init__(self, dim=256):
        super(FCNetWithMI, self).__init__()

        # 网络结构
        self.fc1 = nn.Linear(dim, dim//2)  # 第一层降维
        self.fc2 = nn.Linear(dim//2, dim//4)   # 第二层降维
        self.fc3 = nn.Linear(dim//4, dim//2)   # 第三层升维
        self.fc4 = nn.Linear(dim//2, dim)  # 第四层升维回256
        # 可学习的压缩率
        self.compress_rate = nn.Parameter(torch.ones(1))  # 可学习的压缩率
        self.CLUB1 = MIEstimator(dim,dim//4)
        self.CLUB2 = MIEstimator(dim//2,dim//4)
        # 输出层
        self.fc_out = nn.Linear(dim, dim)

    def mutual_info_loss(self, x, y):
        """
        计算输入x和y的互信息，基于内积相似度。
        """
        # 计算内积相似度，简化的方式
        sim = torch.matmul(x, y.T)  # (batch_size, batch_size)
        return -torch.mean(sim)

    def forward(self, x):
        # 前向传播
        z1 = F.relu(self.fc1(x))   # 第一个全连接层输出 z1
        z2 = self.fc2(z1)          # 第二个全连接层输出 z2
        z3 = self.fc3(z2)          # 第三个全连接层输出 z3
        z4 = self.fc4(z3)          # 第四个全连接层输出 z4

        # 计算互信息
        # 最小化互信息
        mi_x_z1 = self.CLUB1.forward_(x, z2)  # x 和 z1 的互信息
        mi_z1_z2 = self.CLUB2.forward_(z1, z2)  # z1 和 z2 的互信息

        # 最大化互信息
        mi_z3_y = self.CLUB2.forward_(z3, z2)   # z3 和 y 的互信息
        mi_z4_y = self.CLUB1.forward_(z4, z2)   # z4 和 y 的互信息

        l1 = self.CLUB1.learning_loss_(x, z2)
        l2 = self.CLUB2.learning_loss_(z1, z2)
        l3 = self.CLUB2.learning_loss_(z3, z2)
        l4 = self.CLUB1.learning_loss_(z4, z2)

        # 综合损失
        loss1 = mi_x_z1 + mi_z1_z2 - mi_z3_y - mi_z4_y
        loss2 = l1 + l2 + l3 + l4

        return z4, loss1, loss2









