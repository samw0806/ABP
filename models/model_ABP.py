import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb
from models.omics_encoder import SNN_Block

class AdaptFormer(nn.Module):
    def __init__(self, num_latents, dim):
        super(AdaptFormer, self).__init__()

        # Spectrogram (Spec)
        self.spec_norm1 = nn.LayerNorm(256)  # 代替原来的 spec_enc.norm1
        self.spec_attn = nn.MultiheadAttention(embed_dim=256, num_heads=8)  # 代替原来的 spec_enc.attn
        self.spec_norm2 = nn.LayerNorm(256)  # 代替原来的 spec_enc.norm2
        self.spec_mlp = nn.Sequential(  # 代替原来的 spec_enc.mlp
            nn.Linear(256, dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, 256)
        )

        # RGB
        self.rgb_norm1 = nn.LayerNorm(256)  # 代替原来的 rgb_enc.norm1
        self.rgb_attn = nn.MultiheadAttention(embed_dim=256, num_heads=8)  # 代替原来的 rgb_enc.attn
        self.rgb_norm2 = nn.LayerNorm(256)  # 代替原来的 rgb_enc.norm2
        self.rgb_mlp = nn.Sequential(  # 代替原来的 rgb_enc.mlp
            nn.Linear(256, dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, 256)
        )

        # Adapter params
        self.act = QuickGELU()  # 你已有的激活函数
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

        # Spectrogram (Spec) 网络层
        self.spec_down = nn.Linear(256, dim)
        self.spec_up = nn.Linear(dim, 256)
        nn.init.xavier_uniform_(self.spec_down.weight)
        nn.init.zeros_(self.spec_down.bias)
        nn.init.zeros_(self.spec_up.weight)
        nn.init.zeros_(self.spec_up.bias)
        self.spec_scale = nn.Parameter(torch.ones(1))

        # RGB 图像处理网络层
        self.rgb_down = nn.Linear(256, dim)
        self.rgb_up = nn.Linear(dim, 256)
        nn.init.xavier_uniform_(self.rgb_down.weight)
        nn.init.zeros_(self.rgb_down.bias)
        nn.init.zeros_(self.rgb_up.weight)
        nn.init.zeros_(self.rgb_up.bias)
        self.rgb_scale = nn.Parameter(torch.ones(1))

        # Latents 参数
        self.num_latents = num_latents
        self.latents = nn.Parameter(torch.empty(1, num_latents, 256).normal_(std=0.02))
        self.scale_a = nn.Parameter(torch.zeros(1))
        self.scale_v = nn.Parameter(torch.zeros(1))

    def attention(self,q,k,v): # requires q,k,v to have same dim
        B, N, C = q.shape
        attn = (q @ k.transpose(-2, -1)) * (C ** -0.5) # scaling
        attn = attn.softmax(dim=-1)
        x = (attn @ v).reshape(B, N, C)
        return x
    
    # Latent Fusion
    def fusion(self, audio_tokens, visual_tokens):
        # shapes
        BS = audio_tokens.shape[0]
        # concat all the tokens
        concat_ = torch.cat((audio_tokens,visual_tokens),dim=1)
        # cross attention (AV -->> latents)
        fused_latents = self.attention(q=self.latents.expand(BS,-1,-1), k=concat_, v=concat_)
        # cross attention (latents -->> AV)
        audio_tokens = audio_tokens + self.scale_a * self.attention(q=audio_tokens, k=fused_latents, v=fused_latents)
        visual_tokens = visual_tokens + self.scale_v * self.attention(q=visual_tokens, k=fused_latents, v=fused_latents)
        return audio_tokens, visual_tokens

    def forward_audio_AF(self, x):
        x_down = self.spec_down(x)
        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.spec_up(x_down)
        return x_up

    def forward_visual_AF(self, x):
        x_down = self.rgb_down(x)
        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.rgb_up(x_down)
        return x_up


    def forward(self, x, y):

        # Bottleneck Fusion
        x,y = self.fusion(x,y)
        # Attn skip connections
        x = x + self.spec_attn(self.spec_norm1(x),self.spec_norm1(x),self.spec_norm1(x))[0]
        y = y + self.rgb_attn(self.rgb_norm1(y),self.rgb_norm1(y),self.rgb_norm1(y))[0]

        # FFN + skip conections
        x = x + self.spec_mlp(self.spec_norm2(x)) + self.forward_audio_AF(self.spec_norm2(x)) * self.spec_scale
        y = y + self.rgb_mlp(self.rgb_norm2(y)) + self.forward_visual_AF(self.rgb_norm2(y)) * self.rgb_scale
        return x,y

# Example activation function for QuickGELU
class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)
    
class ABP(nn.Module):
    def __init__(self,args,dim=126,omic_names=[],omics_input_dim=0, ):
        super(ABP, self).__init__()
        num_latents = args.num_latents
        m_dim = args.m_dim
        self.omic_sizes = args.omic_sizes
        self.num_pathways = len(args.omic_sizes)
        self.omics_input_dim = omics_input_dim
        self.wsi_embedding_dim = args.encoding_dim
        self.wsi_projection_dim = args.wsi_projection_dim
        self.init_per_path_model(self.omic_sizes,args.omics_format)  # --->dim 256
        num_classes = args.n_classes
        encoder_layers = []
        self.encoder_wsi = nn.Sequential(
            nn.Linear(self.wsi_embedding_dim, self.wsi_projection_dim*2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(self.wsi_projection_dim*2, self.wsi_projection_dim*2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(self.wsi_projection_dim*2, self.wsi_projection_dim),
        )
        for i in range(m_dim):

            # Vanilla Transformer Encoder (use for full fine tuning)
            # encoder_layers.append(VanillaEncoder(num_latents=num_latents, spec_enc=self.v1.blocks[i], rgb_enc=self.v2.blocks[i]))

            # Frozen Transformer Encoder with AdaptFormer 
            encoder_layers.append(AdaptFormer(num_latents=num_latents, dim=dim))
             
        self.audio_visual_blocks = nn.Sequential(*encoder_layers)

        # final norm
        self.spec_post_norm = nn.LayerNorm(256)
        self.rgb_post_norm = nn.LayerNorm(256)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(256*2,num_classes)
        )
    
    def forward_encoder(self,x,y):     
        # encoder forward pass
        for blk in self.audio_visual_blocks:
            x,y = blk(x,y)

        x = self.spec_post_norm(x)
        y = self.rgb_post_norm(y)

        # return class token alone
        x = x[:, 0]
        y = y[:, 0]
        return x,y
    
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
        elif omics_format == 'gene':
            self.sig_networks = SNN_Block(dim1=self.omics_input_dim, dim2=self.wsi_projection_dim) # TODO: Maybe can use geneformer?

        else:
            raise ValueError('omics_format should be pathways, gene or groups')
    
    def forward(self, **kwargs):
        wsi = kwargs['x_wsi']  # wsi features (batch_size, feature_num, wsi_embedding_dim:768)
        wsi = self.encoder_wsi(wsi)
        x_omic = [kwargs['x_omic%d' % i] for i in range(1, self.num_pathways + 1)]  # omic features list (omic_size)

        # ---> get
        h_omic = [self.sig_networks[idx].forward(sig_feat) for idx, sig_feat in
                  enumerate(x_omic)]  # each omic signature goes through it's own FC layer
        h_omic_bag = torch.stack(h_omic)  # omic embeddings are stacked (to be used in co-attention)
        h_omic_bag = h_omic_bag.permute(1, 0, 2)  # (batch_size, num_pathways, 256)


        x,y = self.forward_encoder(wsi,h_omic_bag)
        logits = torch.cat((x,y),dim=1)
        logits = self.classifier(logits)
        return logits    
    





