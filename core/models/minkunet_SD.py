import MinkowskiEngine as ME
from MinkowskiEngine.modules.resnet_block import BasicBlock
from core.models.resnet import ResNetBase
import torch
import torch.nn as nn
import math

    
class ProtoAttn(nn.Module):
    def __init__(
            self, 
            prototype: dict, 
            proj_channel: int = 256, 
            num_heads: int = 8, 
            kv_bias: bool = False, 
            dropout_ratio: float = 0.2, 
         ):
        super(ProtoAttn, self).__init__()
        assert ( proj_channel % num_heads == 0 ), f"proj_channels {feat_channel} should be divided by num_heads {num_heads}."
     
        self.num_heads = num_heads
        # self.attn_channel = proj_channel // num_heads
        self.proj_channel = proj_channel
        self.proj_layers = nn.ModuleDict({})
        self.unproj_layers = nn.ModuleDict({})
        for k, v in prototype.items():
            feat_channel = v.shape[1]
            self.proj_layers.update({k: nn.Linear(feat_channel, proj_channel)})
            self.unproj_layers.update({k: nn.Linear(proj_channel, feat_channel)})
            
        # self.proj = ME.MinkowskiLinear(feat_channel, proj_channels, bias=True, dimension=D)
        # self.q_embed = nn.Sequential(
        #     ME.MinkowskiLinear(feat_channel, feat_channel, bias=qkv_bias, dimension=D),
        #     ME.MinkowskiToFeature
        # )
        self.attn = nn.MultiheadAttention(
            embed_dim=proj_channel,
            num_heads=num_heads, 
            dropout=dropout_ratio, 
            batch_first=True, 
            add_bias_kv=kv_bias
        )

        self.init_weights()

        # self.bn = ME.MinkowskiBatchNorm(feat_channel)
        # self.ln = ME.LayerNorm(feat_channel)

        # self.dropout = ME.MinkowskiDropout(dropout_ratio)

        # self.norm_fact = 1 / (feat_channel // num_heads) ** 0.5

        # self.multi_head_proj = nn.Sequential(
        #     ME.MinkowskiLinear(feat_channel, feat_channel, bias=True, dimension=D),
        #     ME.MinkowskiBatchNorm(feat_channel),
        #     ME.MinkowskiReLU(inplace=True)
        #     )
        # self.softmax = ME.MinkowskiSoftmax(dim=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiLinear):
                ME.utils.kaiming_normal_(m.linear.weight, mode="fan_out", nonlinearity="relu")
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")
            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def forward(self, x): # x是一个prototype的dict
        proj_proto = [self.proj_layers[k](v.Proto.detach()) for k, v in x.items()] # (l. d_model)
        proj_proto = torch.stack(proj_proto, dim=1) # (num_class, l. d_model)
        q, k, v = proj_proto.clone(), proj_proto.clone(), proj_proto.clone()
        # q = k = v = proj_proto # 参考SPFormer的写法

        attn, _ = self.attn(q, k, v)
        # out = out.view(16, 6, 256).permute(1, 0, 2).contiguous()
        out_dict = {k: v for k, v in zip(x.keys(), [t.squeeze(1) for t in torch.split(attn, split_size_or_sections=1, dim=1)])}

        out = {k: self.unproj_layers[k](v) for k, v in out_dict.items()} # 反投影回初始维度，用于对应block蒸馏

        return out

        # n, dim_in = x.shape # 在进行类原型间注意力时，n表示层数，dim_in表示经过proj后相同维度的特征
        # q = self.q_embed(x).view(-1, self.num_heads, self.attn_channel).contiguous()
        # q = self.k_embed(x).view(-1, self.num_heads, self.attn_channel).contiguous()
        # q = self.v_embed(x).view(-1, self.num_heads, self.attn_channel).contiguous()

        # # q = self.q_embed(x).reshape(n, self.num_heads, self.feat_channel).transpose(1, 2)
        # # k = self.k_embed(x).reshape(n, self.num_heads, self.feat_channel).transpose(1, 2)
        # # v = self.v_embed(x).reshape(n, self.num_heads, self.feat_channel).transpose(1, 2)

        # dist = torch.matmul(q, k.transpose(2, 3)) * self.norm_fact
        # dist = torch.softmax(dist, dim=-1)

        # attn = torch.matmul(dist, v)
        # attn = attn.transpose(1, 2).reshape(b, n, dim_in)

        # return ME.SparseTensor(
        #     features = attn,
        #     coordinates = x.coordinates
        # )
    
class self_attn(nn.Module):
    def __init__(
            self, 
            emded_dim: int = 256, 
            num_heads: int = 8, 
            kv_bias: bool = False, 
            dropout_ratio: float = 0.2, 
         ):
        super(self_attn, self).__init__()
        assert ( emded_dim % num_heads == 0 ), f"proj_channels {emded_dim} should be divided by num_heads {num_heads}."
            
        self.self_attn = nn.MultiheadAttention(
            embed_dim=emded_dim,
            num_heads=num_heads, 
            dropout=dropout_ratio, 
            batch_first=True, 
            add_bias_kv=kv_bias
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiLinear):
                ME.utils.kaiming_normal_(m.linear.weight, mode="fan_out", nonlinearity="relu")
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")
            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def forward(self, x): # x是一个prototype的dict
        q = k = v = x
        out, _ = self.self_attn(q, k, v)
        return out

        # n, dim_in = x.shape # 在进行类原型间注意力时，n表示层数，dim_in表示经过proj后相同维度的特征
        # q = self.q_embed(x).view(-1, self.num_heads, self.attn_channel).contiguous()
        # q = self.k_embed(x).view(-1, self.num_heads, self.attn_channel).contiguous()
        # q = self.v_embed(x).view(-1, self.num_heads, self.attn_channel).contiguous()

        # # q = self.q_embed(x).reshape(n, self.num_heads, self.feat_channel).transpose(1, 2)
        # # k = self.k_embed(x).reshape(n, self.num_heads, self.feat_channel).transpose(1, 2)
        # # v = self.v_embed(x).reshape(n, self.num_heads, self.feat_channel).transpose(1, 2)

        # dist = torch.matmul(q, k.transpose(2, 3)) * self.norm_fact
        # dist = torch.softmax(dist, dim=-1)

        # attn = torch.matmul(dist, v)
        # attn = attn.transpose(1, 2).reshape(b, n, dim_in)

        # return ME.SparseTensor(
        #     features = attn,
        #     coordinates = x.coordinates
        # )
    
class cross_attn(nn.Module):
    def __init__(
            self, 
            emded_dim: int = 256, 
            num_heads: int = 8, 
            kv_bias: bool = False, 
            dropout_ratio: float = 0.2, 
         ):
        super(self_attn, self).__init__()
        assert ( emded_dim % num_heads == 0 ), f"proj_channels {emded_dim} should be divided by num_heads {num_heads}."
            
        self.self_attn = nn.MultiheadAttention(
            embed_dim=emded_dim,
            num_heads=num_heads, 
            dropout=dropout_ratio, 
            batch_first=True, 
            add_bias_kv=kv_bias
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiLinear):
                ME.utils.kaiming_normal_(m.linear.weight, mode="fan_out", nonlinearity="relu")
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")
            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def forward(self, q, k): # x是一个prototype的dict
        out, _ = self.self_attn(q, k, k)
        return out
    

class fuse_layer(nn.Module):
    def __init__(self, in_channels=None, D=3):
        nn.Module.__init__(self)
        # self.weight = nn.Parameter(torch.ones(in_channels * 2, requires_grad=True))
        self.gated_fuse = nn.Sequential(
            ME.MinkowskiConvolution(in_channels * 2, in_channels, kernel_size=1, bias=True, dimension=D),
            # ME.MinkowskiBatchNorm(in_channels),
            # ME.MinkowskiLeakyReLU(0.1, inplace=True),
            # ME.MinkowskiConvolution(in_channels, in_channels, kernel_size=1, bias=True, dimension=D),
            ME.MinkowskiSigmoid()
        )

        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiLinear):
                ME.utils.kaiming_normal_(m.linear.weight, mode="fan_out", nonlinearity="relu")
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")
            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def forward(self, original_feat, enhanced_feat):
        feat_g = ME.cat(original_feat, enhanced_feat)
        feat_g = self.gated_fuse(feat_g)
        one_st = ME.SparseTensor(features = torch.ones(feat_g.shape).cuda(), coordinates=feat_g.C, coordinate_manager=feat_g.coordinate_manager)
        out = feat_g * original_feat + (one_st-feat_g) * enhanced_feat
        return out


    # def forward(self, original_feat, enhanced_feat):
    #     out = original_feat + enhanced_feat
    #     # feat_g = ME.cat(original_feat, enhanced_feat)
    #     # feat_g = self.gated_fuse(feat_g)
    #     # one_st = ME.SparseTensor(features = torch.ones([feat_g.shape[0],1]).cuda(), coordinates=feat_g.C, coordinate_manager=feat_g.coordinate_manager)
    #     # out_f = feat_g.F * original_feat.F + (one_st.F-feat_g.F) * enhanced_feat.F
    #     # out = ME.SparseTensor(features = out_f, coordinates=original_feat.C, coordinate_manager=original_feat.coordinate_manager)
    #     # # out = feat_g * original_feat + (one_st-feat_g) * enhanced_feat

    #     # # out_f = feat_g_f.F * original_feat.F + (torch.tensor(1).cuda()-feat_g_f.F) * enhanced_feat.F
    #     # print(original_feat.coordinate_manager)
    #     # # out = ME.SparseTensor(features = out_f, coordinates=original_feat.C, coordinate_manager=original_feat.coordinate_manager)
    #     # print(out.coordinate_manager)
    #     return out



class HB_learner(nn.Module):
    def __init__(self, in_channels=None, hidden_channels=None, out_channels=None, D=3):
        nn.Module.__init__(self)
        self.network_initialization(in_channels, hidden_channels, out_channels, D)
        self.weight_initialization()

    def CBR_k1(self, in_channels, out_channels, D=3):
        CBR_k1 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=1, bias=True, dimension=D),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True)
        ) 
        return CBR_k1 
        
    def network_initialization(self, in_channels, hidden_channels, out_channels, D):
        self.CBR_k1s1 = self.CBR_k1(in_channels, hidden_channels)
        self.CBR_k1s2 = self.CBR_k1(hidden_channels, 2 * hidden_channels)
        self.CBR_k1s3 = self.CBR_k1(2 * hidden_channels, 4 * hidden_channels)
        self.CBR_k1s4 = self.CBR_k1(4 * hidden_channels, out_channels)
        
        # self.fuse = nn.Sequential(
        #     ME.MinkowskiConvolution(in_channels + out_channels, out_channels, kernel_size=1, bias=True, dimension=D),
        #     ME.MinkowskiBatchNorm(self.out_channels),
        #     ME.MinkowskiLeakyReLU(0.1, inplace=True)
        # )

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiLinear):
                ME.utils.kaiming_normal_(m.linear.weight, mode="fan_out", nonlinearity="relu")
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")
            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def forward(self, x: ME.SparseTensor):
        # residual = x
        x = self.CBR_k1s1(x)
        x = self.CBR_k1s2(x)
        x = self.CBR_k1s3(x)
        x = self.CBR_k1s4(x)
        # x = self.fuse(ME.cat(residual, x))
        # fuse_feat = fuse_feat + residual
        return x
    

class classifier(nn.Module):
    def __init__(self, in_channels, num_classes, D=3):
        nn.Module.__init__(self)
        self.final = ME.MinkowskiConvolution(in_channels, num_classes, kernel_size=1, bias=True, dimension=D)
    def forward(self, x):
        return self.final(x)
        
        
class MinkUNetSDBase(ResNetBase):
    BLOCK = None
    PLANES = None
    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)
    INIT_DIM = 32
    OUT_TENSOR_STRIDE = 1

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling
    # initialize_coords
    def __init__(self, in_channels, out_channels, D=3, return_feat=True, CL_LAYERS=[], RETURN_BLOCKS=[]):
        self.return_feat = return_feat
        self.CL_LAYERS = CL_LAYERS # LAYERS for conducting Contrastive Learning
        self.RETURN_BLOCKS = RETURN_BLOCKS # Return LAYERS
        ResNetBase.__init__(self, in_channels, out_channels, D)

    def network_initialization(self, in_channels, out_channels, D):
        # Output of the first conv concated to conv6
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = ME.MinkowskiConvolution(
            in_channels, self.inplanes, kernel_size=5, dimension=D)

        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        self.conv1p1s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0],
                                       self.LAYERS[0])

        self.conv2p2s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1],
                                       self.LAYERS[1])

        self.conv3p4s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)

        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2],
                                       self.LAYERS[2])

        self.conv4p8s2 = ME.MinkowskiConvolution(
            self.inplanes, self.inplanes, kernel_size=2, stride=2, dimension=D)
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3],
                                       self.LAYERS[3])

        self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[4], kernel_size=2, stride=2, dimension=D)
        self.bntr4 = ME.MinkowskiBatchNorm(self.PLANES[4])

        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[4],
                                       self.LAYERS[4])
        self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[5], kernel_size=2, stride=2, dimension=D)
        self.bntr5 = ME.MinkowskiBatchNorm(self.PLANES[5])

        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[5],
                                       self.LAYERS[5])
        self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[6], kernel_size=2, stride=2, dimension=D)
        self.bntr6 = ME.MinkowskiBatchNorm(self.PLANES[6])

        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[6],
                                       self.LAYERS[6])
        self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(
            self.inplanes, self.PLANES[7], kernel_size=2, stride=2, dimension=D)
        self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[7])

        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[7],
                                       self.LAYERS[7])
        
        if self.return_feat == False:
            self.final = ME.MinkowskiConvolution(
                self.PLANES[7] * self.BLOCK.expansion,
                out_channels,
                kernel_size=1,
                bias=True,
                dimension=D)
        self.relu = ME.MinkowskiReLU(inplace=True)

        self.dropout = ME.MinkowskiDropout(p=0.5)

        # 根据所选层级生成对应的hb_learner学习器
        self.hb_learner = nn.ModuleDict()
        for l in self.CL_LAYERS:
            if l == "block4":
                self.hb_learner.update({l: HB_learner(self.PLANES[3], self.PLANES[3], self.PLANES[3])})
            if l == "block5":
                self.hb_learner.update({l: HB_learner(self.PLANES[4], self.PLANES[4], self.PLANES[4])})
            if l == "block6":
                self.hb_learner.update({l: HB_learner(self.PLANES[5], self.PLANES[5], self.PLANES[5])})
            if l == "block7":
                self.hb_learner.update({l: HB_learner(self.PLANES[6], self.PLANES[6], self.PLANES[6])})
            if l == "block8":
                self.hb_learner.update({l: HB_learner(self.PLANES[7], self.PLANES[7], self.PLANES[7])})


        for l in self.CL_LAYERS:
            if l == "block4":
                self.fuse4 = fuse_layer(self.PLANES[3])
            if l == "block5":
                self.fuse5 = fuse_layer(self.PLANES[4])
            if l == "block6":
                self.fuse6 = fuse_layer(self.PLANES[5])
            if l == "block7":
                self.fuse7 = fuse_layer(self.PLANES[6])
            if l == "block8":
                self.fuse8 = fuse_layer(self.PLANES[7])
        

    def forward(self, x):
        if self.return_feat:
            hb_learner_feature = {}

        # ENCODER
        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)

        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)

        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)

        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)

        # tensor_stride=16
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out_bottle = self.block4(out)

        if 'block4' in self.CL_LAYERS:
            hb_learner_feature.update({'block4': out_bottle})
            enhanced_feat = self.hb_learner['block4'](out_bottle)
            # out_bottle = enhanced_feat + out_bottle
            out_bottle = self.fuse4(out_bottle, enhanced_feat)
        elif 'block4' in self.RETURN_BLOCKS:
            hb_learner_feature.update({'block4': out_bottle})

        # DECODER
        # tensor_stride=8
        out = self.convtr4p16s2(out_bottle)
        out = self.bntr4(out)
        out = self.relu(out)

        out = ME.cat(out, out_b3p8)
        out = self.block5(out)

        if 'block5' in self.CL_LAYERS:
            hb_learner_feature.update({'block5': out})
            enhanced_feat = self.hb_learner['block5'](out)
            # out = enhanced_feat + out
            out = self.fuse5(out, enhanced_feat)
        elif 'block5' in self.RETURN_BLOCKS:
            hb_learner_feature.update({'block5': out})

        # tensor_stride=4
        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)

        out = ME.cat(out, out_b2p4)
        out = self.block6(out)

        if 'block6' in self.CL_LAYERS:
            hb_learner_feature.update({'block6': out})
            enhanced_feat = self.hb_learner['block6'](out)
            # out = enhanced_feat + out
            out = self.fuse6(out, enhanced_feat)
        elif 'block6' in self.RETURN_BLOCKS:
            hb_learner_feature.update({'block6': out})

        # tensor_stride=2
        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu(out)

        out = ME.cat(out, out_b1p2)
        out = self.block7(out)

        if 'block7' in self.CL_LAYERS:
            hb_learner_feature.update({'block7': out})
            enhanced_feat = self.hb_learner['block7'](out)
            # out = enhanced_feat + out
            out = self.fuse7(out, enhanced_feat)
        elif 'block7' in self.RETURN_BLOCKS:
            hb_learner_feature.update({'block7': out})

        # tensor_stride=1
        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.relu(out)

        out = ME.cat(out, out_p1)
        out = self.block8(out)

        if 'block8' in self.CL_LAYERS:
            hb_learner_feature.update({'block8': out})
            enhanced_feat = self.hb_learner['block8'](out)
            # out = enhanced_feat + out
            out = self.fuse8(out, enhanced_feat)
        elif 'block8' in self.RETURN_BLOCKS:
            hb_learner_feature.update({'block8': out})

        if self.return_feat:
            return out, hb_learner_feature
        else:
            return out, self.final(out)


class MinkUNet34_SD(MinkUNetSDBase):
    BLOCK = BasicBlock
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)