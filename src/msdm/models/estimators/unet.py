
import torch 
import torch.nn as nn 
from monai.networks.blocks import UnetOutBlock

from msdm.models.utils.conv_blocks import BasicBlock, UpBlock, DownBlock, UnetBasicBlock, UnetResBlock, save_add, BasicDown, BasicUp, SequentialEmb
from msdm.models.embedders.time_embedder import TimeEmbbeding
from msdm.models.utils.attention_blocks import MedfusionAttention, PerceiverResampler, zero_module, AttentionPooling


class TextTimeconditionPooling(nn.Module):
    def __init__(self, encoder_dim, time_embed_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(encoder_dim)
        self.pool = AttentionPooling(encoder_dim, encoder_dim)
        self.proj = nn.Linear(encoder_dim, time_embed_dim)
        self.norm2 = nn.LayerNorm(time_embed_dim)

    def forward(self, cond_emb, mask):
        c = self.norm1(cond_emb)
        c = self.pool(c, mask)
        c = self.proj(c)
        c = self.norm2(c)
        return c


class UNet(nn.Module):
    def __init__(self, 
            in_ch=1, 
            out_ch=1, 
            spatial_dims=3,
            hid_chs=[256, 256, 512, 1024],
            kernel_sizes=[3, 3, 3, 3],
            strides=[1, 2, 2, 2], # WARNING, last stride is ignored (follows OpenAI)
            act_name=("SWISH", {}),
            norm_name=("GROUP", {'num_groups': 32, "affine": True}),
            time_embedder=None,
            time_embedder_kwargs={},
            deep_supervision=True, # True = all but last layer, 0/False=disable, 1=only first layer, ... 
            use_res_block=True,
            estimate_variance=False ,
            use_self_conditioning =False, 
            dropout=0, 
            learnable_interpolation=True,
            use_attention='none',
            num_res_blocks=2,
            cond_emb_hidden_dim=768,
            cond_emb_bias=False
        ):
        super().__init__()
        use_attention = use_attention if isinstance(use_attention, list) else [use_attention]*len(strides) 
        self.use_self_conditioning = use_self_conditioning
        self.use_res_block = use_res_block
        self.depth = len(strides)
        self.num_res_blocks = num_res_blocks
        self.num_channels = in_ch

        # ------------- Time-Embedder-----------
        if time_embedder is not None:
            self.time_embedder=time_embedder(**time_embedder_kwargs)
            time_emb_dim = self.time_embedder.emb_dim
        else:
            self.time_embedder = None 
            time_emb_dim = None 

        # ------------- Condition-Embedder-----------

        ConvBlock = UnetResBlock if use_res_block else UnetBasicBlock

        # ----------- In-Convolution ------------
        in_ch = in_ch*2 if self.use_self_conditioning else in_ch 
        self.in_conv = BasicBlock(spatial_dims, in_ch, hid_chs[0], kernel_size=kernel_sizes[0], stride=strides[0])
        
        # ----------- Encoder ------------
        in_blocks = [] 
        for i in range(1, self.depth):
            for k in range(num_res_blocks):
                seq_list = [] 
                seq_list.append(
                    ConvBlock(
                        spatial_dims=spatial_dims,
                        in_channels=hid_chs[i-1 if k==0 else i],
                        out_channels=hid_chs[i],
                        kernel_size=kernel_sizes[i],
                        stride=1,
                        norm_name=norm_name,
                        act_name=act_name,
                        dropout=dropout,
                        emb_channels=time_emb_dim
                    )
                )

                seq_list.append(
                    MedfusionAttention(
                        spatial_dims=spatial_dims,
                        in_channels=hid_chs[i],
                        out_channels=hid_chs[i],
                        num_heads=8,
                        ch_per_head=hid_chs[i]//8,
                        depth=1,
                        norm_name=norm_name,
                        dropout=dropout,
                        emb_dim=time_emb_dim,
                        attention_type=use_attention[i]
                    )
                )
                in_blocks.append(SequentialEmb(*seq_list))

            if i < self.depth-1:
                in_blocks.append(
                    BasicDown(
                        spatial_dims=spatial_dims,
                        in_channels=hid_chs[i],
                        out_channels=hid_chs[i],
                        kernel_size=kernel_sizes[i],
                        stride=strides[i],
                        learnable_interpolation=learnable_interpolation 
                    )
                )

        self.in_blocks = nn.ModuleList(in_blocks)
        
        # ----------- Middle ------------
        self.middle_block = SequentialEmb(
            ConvBlock(
                spatial_dims=spatial_dims,
                in_channels=hid_chs[-1],
                out_channels=hid_chs[-1],
                kernel_size=kernel_sizes[-1],
                stride=1,
                norm_name=norm_name,
                act_name=act_name,
                dropout=dropout,
                emb_channels=time_emb_dim
            ),
            MedfusionAttention(
                spatial_dims=spatial_dims,
                in_channels=hid_chs[-1],
                out_channels=hid_chs[-1],
                num_heads=8,
                ch_per_head=hid_chs[-1]//8,
                depth=1,
                norm_name=norm_name,
                dropout=dropout,
                emb_dim=time_emb_dim,
                attention_type=use_attention[-1]
            ),
            ConvBlock(
                spatial_dims=spatial_dims,
                in_channels=hid_chs[-1],
                out_channels=hid_chs[-1],
                kernel_size=kernel_sizes[-1],
                stride=1,
                norm_name=norm_name,
                act_name=act_name,
                dropout=dropout,
                emb_channels=time_emb_dim
            )
        )
     
        # ------------ Decoder ----------
        out_blocks = [] 
        for i in range(1, self.depth):
            for k in range(num_res_blocks+1):
                seq_list = [] 
                out_channels=hid_chs[i-1 if k==0 else i]
                seq_list.append(
                    ConvBlock(
                        spatial_dims=spatial_dims,
                        in_channels=hid_chs[i]+hid_chs[i-1 if k==0 else i],
                        out_channels=out_channels,
                        kernel_size=kernel_sizes[i],
                        stride=1,
                        norm_name=norm_name,
                        act_name=act_name,
                        dropout=dropout,
                        emb_channels=time_emb_dim
                    )
                )
            
                seq_list.append(
                    MedfusionAttention(
                        spatial_dims=spatial_dims,
                        in_channels=out_channels,
                        out_channels=out_channels,
                        num_heads=8,
                        ch_per_head=out_channels//8,
                        depth=1,
                        norm_name=norm_name,
                        dropout=dropout,
                        emb_dim=time_emb_dim,
                        attention_type=use_attention[i]
                    )
                )

                if (i >1) and k==0:
                    seq_list.append(
                        BasicUp(
                            spatial_dims=spatial_dims,
                            in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=strides[i],
                            stride=strides[i],
                            learnable_interpolation=learnable_interpolation 
                        )
                    )
        
                out_blocks.append(SequentialEmb(*seq_list))
        self.out_blocks = nn.ModuleList(out_blocks)
        
        # --------------- Out-Convolution ----------------
        out_ch_hor = out_ch*2 if estimate_variance else out_ch
        self.outc = zero_module(UnetOutBlock(spatial_dims, hid_chs[0], out_ch_hor, dropout=None))
        if isinstance(deep_supervision, bool):
            deep_supervision = self.depth-2 if deep_supervision else 0 
        self.outc_ver = nn.ModuleList([
            zero_module(UnetOutBlock(spatial_dims, hid_chs[i]+hid_chs[i-1], out_ch, dropout=None) )
            for i in range(2, deep_supervision+2)
        ])
        #-----------------------------------------------------------------------------------------------
        #self.attn_pool = PerceiverResampler(dim=1024, depth=2, dim_head=64, heads=8, num_latents=1024)
        #self.cond_pooling = TextTimeconditionPooling(self.cond_embedder.hidden_dim, self.time_embedder.emb_dim)
        #self.cond_linear = nn.Linear(self.cond_embedder.hidden_dim, self.cond_embedder.emb_dim)
        self.cond_pooling = AttentionPooling(self.time_embedder.emb_dim, self.time_embedder.emb_dim)
        self.cond_projection_linear = nn.Linear(cond_emb_hidden_dim, self.time_embedder.emb_dim, bias=cond_emb_bias)
        self.cond_act = nn.SiLU()
        self.cond_projection_linear2 = nn.Linear(self.time_embedder.emb_dim, self.time_embedder.emb_dim, bias=cond_emb_bias)
        #self.cond_projection_linear3 = nn.Linear(cond_emb_hidden_dim, self.time_embedder.emb_dim)

        #self.cond_norm = nn.LayerNorm(cond_emb_hidden_dim)
        #self.cond_norm2 = nn.LayerNorm(self.time_embedder.emb_dim)

    def forward(self, x_t, t=None, condition_embed=None, condition_mask=None, self_cond=None):
        # x_t [B, C, *]
        # t [B,]
        # condition [B,]
        # self_cond [B, C, *]
        # -------- Time Embedding (Gloabl) -----------
        if t is None:
            time_emb = None 
        else:
            time_emb = self.time_embedder(t) # [B, C]
        # -------- Condition Embedding (Gloabl) -----------
        if condition_mask is not None:
            condition_mask = (1 - condition_mask.to(x_t.dtype)) * -10000.0
            condition_mask = condition_mask.unsqueeze(1)
        if condition_embed is None:
            cond_emb = None
            emb = save_add(time_emb, condition_embed)
        else:
            cond_emb = self.cond_projection_linear(condition_embed)
            cond_emb = self.cond_act(cond_emb)
            cond_emb = self.cond_projection_linear2(cond_emb)
            aug_emb = self.cond_pooling(cond_emb, condition_mask)
            emb = save_add(time_emb, aug_emb)
        # ---------- Self-conditioning-----------
        if self.use_self_conditioning:
            self_cond = torch.zeros_like(x_t) if self_cond is None else x_t 
            x_t = torch.cat([x_t, self_cond], dim=1)  
    
        # --------- Encoder --------------
        x = [self.in_conv(x_t)]
        for i in range(len(self.in_blocks)):
            x.append(self.in_blocks[i](x[i], emb, cond_emb, condition_mask))

        # ---------- Middle --------------
        h = self.middle_block(x[-1], emb, cond_emb, condition_mask)
        
        # -------- Decoder -----------
        y_ver = []
        for i in range(len(self.out_blocks), 0, -1):
            h = torch.cat([h, x.pop()], dim=1)

            depth, j = i//(self.num_res_blocks+1), i%(self.num_res_blocks+1)-1
            y_ver.append(self.outc_ver[depth-1](h)) if (len(self.outc_ver)>=depth>0) and (j==0) else None 

            h = self.out_blocks[i-1](h, emb, cond_emb, condition_mask)

        # ---------Out-Convolution ------------
        y = self.outc(h)

        return y, y_ver[::-1]


if __name__=='__main__':
    model = UNet(in_ch=3, use_res_block=False, learnable_interpolation=False)
    input = torch.randn((1, 3, 16, 32, 32))
    time = torch.randn((1,))
    out_hor, out_ver = model(input, time)
    print(out_hor[0].shape)