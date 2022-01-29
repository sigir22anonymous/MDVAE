import torch
from torch import nn
from torch.nn import functional as F
import scipy
from utils import *
from torchvision import models
class specific_encoder(nn.Module):
    def __init__(self,config,domain,use_cuda = False):
        super(specific_encoder,self).__init__()
        self.input_dim = config['input_dim'][domain]
        self.neef = config['neef']
        self.exdim = config['exdim']
        self.bn_momentum = config['bn_momentum']
        self.use_cuda = use_cuda
        self.net = nn.Sequential(
            nn.Linear(self.input_dim,self.neef*4),
            nn.BatchNorm1d(self.neef*4,   momentum=self.bn_momentum),
            nn.LeakyReLU(0.2),
            nn.Linear(self.neef*4,self.neef*2),
            nn.BatchNorm1d(self.neef*2,   momentum=self.bn_momentum),
            nn.LeakyReLU(0.2),
            nn.Linear(self.neef*2,self.exdim*2)
        )
        if self.use_cuda:
            self.net =self.net.cuda()
    
    def forward(self,input):
        output = self.net(input)
        q_z_exclusive_mean, q_z_exclusive_logvar = torch.split(output, [self.exdim,self.exdim], dim = -1)
        return q_z_exclusive_mean ,q_z_exclusive_logvar

class invariant_encoder(nn.Module):
    def __init__(self,config,domain:int,use_cuda = False):
        super(invariant_encoder,self).__init__()
        self.input_dim = config['input_dim'][domain]
        self.nsef = config['nsef']
        self.shdim = config['shdim']
        self.bn_momentum = config['bn_momentum']
        self.use_cuda =use_cuda
        self.vae_individual_shared_encoder1 = nn.Sequential(
            nn.Linear(self.input_dim,self.nsef*4),
            nn.BatchNorm1d(self.nsef*4,  momentum=self.bn_momentum),
            nn.LeakyReLU(0.2)
            nn.Linear(self.nsef*4,self.nsef*2),
            nn.BatchNorm1d(self.nsef*2,   momentum=self.bn_momentum),
            nn.LeakyReLU(0.2),
            nn.Linear(self.nsef*2,self.shdim*2)
        )

        if self.use_cuda:
            self.vae_individual_shared_encoder =self.vae_individual_shared_encoder.cuda()

    
    def forward(self,input):
        output = self.vae_individual_shared_encoder(input)
        r_z_shared_mean, r_z_shared_logvar = torch.split(output, [self.shdim,self.shdim], dim = -1)
        return r_z_shared_mean, r_z_shared_logvar

class feature_decoder(nn.Module):
    def __init__(self,config,domain,use_cuda = False):
        super(feature_decoder,self).__init__()
        self.bn_momentum = config['bn_momentum']
        self.use_cuda = use_cuda
        self.ndf = config['ndf']
        self.input_dim = config['shdim'] + config['exdim']
        self.out_dim = config['input_dim'][domain]
        self.vae_decoder = nn.Sequential(
            nn.Linear(self.input_dim,self.ndf),
            nn.BatchNorm1d(self.ndf,momentum= self.bn_momentum),
            nn.ReLU(),
            nn.Linear(self.ndf,self.out_dim)
        )
        if self.use_cuda:
            self.vae_decoder = self.vae_decoder.cuda()
    def forward(self,input):
        return self.vae_decoder(input)

class c_encoder(nn.Module):
    def __init__(self,config:dict,use_cuda = False):
        super(c_encoder,self).__init__()
        self.input_dim = sum(config['input_dim'])
        self.ncef = config['ncef']
        self.cdim = config['cdim']
        self.bn_momentum = config['bn_momentum']
        self.class_num = config['class_num']
        self.use_cuda =use_cuda

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim,self.ncef*8),
            nn.BatchNorm1d(self.ncef*8,  momentum=self.bn_momentum),
            nn.LeakyReLU(0.2),
            nn.Linear(self.ncef*8,self.ncef*4),
            nn.BatchNorm1d(self.ncef*4,  momentum=self.bn_momentum),
            nn.LeakyReLU(0.2),
            nn.Linear(self.ncef*4,self.cdim*2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.cdim*2,self.class_num),
        )
        if self.use_cuda:
            self.encoder =self.encoder.cuda()
            self.classifier =self.classifier.cuda()
    
    def forward(self,input):
        output =self.encoder(input)
        label = self.classifier(output)
        return output,label

class distribution_gengrator(nn.Module):
    def __init__(self,config:dict,use_cuda = False):
        super(distribution_gengrator,self).__init__()
        self.input_dim = config['cdim']*2
        self.nzcef = config['nzcef']
        self.shdim = config['shdim']
        self.bn_momentum = config['bn_momentum']
        self.use_cuda =use_cuda

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim,self.nzcef*4),
            nn.BatchNorm1d(self.nzcef*4,  momentum=self.bn_momentum),
            nn.LeakyReLU(0.2),
            nn.Linear(self.nzcef*4,self.shdim*2)
        )
        if self.use_cuda:
            self.encoder =self.encoder.cuda()
    
    def forward(self,input):
        output =self.encoder(input)
        prior_mean, prior_logvar = torch.split(output, [self.shdim,self.shdim], dim = -1)
        return prior_mean, prior_logvar

class c_decoder(nn.Module):
    def __init__(self,config:dict,use_cuda = False):
        super(c_decoder,self).__init__()
        self.input_dim = config['shdim']
        self.nzcef = config['nzcef']
        self.outdim= sum(config['input_dim'])
        self.bn_momentum = config['bn_momentum']
        self.use_cuda =use_cuda

        self.decoder = nn.Sequential(
            nn.Linear(self.input_dim,self.nzcef*4),
            nn.BatchNorm1d(self.nzcef*4,  momentum=self.bn_momentum),
            nn.LeakyReLU(0.2),
            nn.Linear(self.nzcef*4,self.outdim*2)
        )
        if self.use_cuda:
            self.decoder =self.decoder.cuda()
    
    def forward(self,input):
        output =self.dedcoder(input)
        return output
class CMR_NET(nn.Module):
    def __init__(self,config:dict,use_cuda = False):
        super(CMR_NET,self).__init__()
        self.use_cuda = use_cuda
        self.config = config
        self.reset(config)
    def reset(self):

    def forward(self,x,y,label,step):
        qzx_mean,qzx_logvar  = self.qzx_extractor(x)
        qzy_mean,qzy_logvar  = self.qzy_extractor(y)
        
        rzx_mean,rzx_logvar  = self.rzx_extractor(x)
        rzy_mean,rzy_logvar  = self.rzy_extractor(y)

        c_features,x_y_label = self.c_extractor(torch.cat([x,y],dim = -1))
        prior_mean, prior_logvar = self.gengrator(c_features)

        qzx_logvar = torch.min(qzx_logvar, torch.Tensor([9.21035]).cuda() )
        qzy_logvar = torch.min(qzy_logvar, torch.Tensor([9.21035]).cuda() )
        rzx_logvar = torch.min(rzx_logvar, torch.Tensor([9.21035]).cuda() )
        rzy_logvar = torch.min(rzy_logvar, torch.Tensor([9.21035]).cuda() )

        
        qzx = sample_normal(qzx_logvar,qzx_mean,self.use_cuda,self.training)
        qzy = sample_normal(qzy_logvar,qzy_mean,self.use_cuda,self.training)
        rzx = sample_normal(rzx_logvar,rzx_mean,self.use_cuda,self.training)
        rzy = sample_normal(rzy_logvar,rzy_mean,self.use_cuda,self.training)
        xy = self.c_recon(prior_mean)

        xfromx  = self.decode_x(torch.cat([rzx,qzx],dim = 1))
        yfromy  = self.decode_y(torch.cat([rzy,qzy],dim = 1))
        yfromx  = self.decode_x(torch.cat([rzy,qzx],dim = 1))
        xfromy  = self.decode_y(torch.cat([rzx,qzy],dim = 1))
        
        x_recon_loss = torch.mean((xfromx - x)**2) *self.config['l1_weight']
        y_recon_loss = torch.mean((yfromy - y)**2) *self.config['l1_weight']
        
        x_f_recon_loss = torch.mean((yfromx - y)**2) *self.config['l1_weight']
        y_f_recon_loss = torch.mean((xfromy - x)**2) *self.config['l1_weight']

        aligin_x_loss = kl_loss2(rzx_logvar,rzx_mean,prior_logvar,prior_mean)
        aligin_y_loss = kl_loss2(rzy_logvar,rzy_mean,prior_logvar,prior_mean)
        kl_x_loss = kl_loss(qzx_logvar,qzx_mean)
        kl_y_loss = kl_loss(qzy_logvar,qzy_mean)
        label_recon_loss = self.crossentropy(x_y_label,label)
        xy_recon_loss = torch.mean((xy - torch.cat([x,y],dim = -1))**2) 

        reg_coeff= torch.Tensor( 1.0 - torch.exp( -torch.Tensor([step]).float() / self.ar) ).detach().cuda()

        joint_loss = x_recon_loss +  y_recon_loss \
                     + self.ALPHA *  reg_coeff*kl_x_loss \
                     + self.ALPHA* reg_coeff*kl_y_loss \
                     + self.BETA  *reg_coeff*kl_intery_loss  \
                     + self.GAMMA  * reg_coeff*x_f_recon_loss \
                     + self.GAMMA *reg_coeff*y_f_recon_loss  \
                     + self.LAMBDA *label_recon_loss

        return {
            'x_recon_loss' :x_recon_loss,
            'y_recon_loss' :y_recon_loss,
            'x_crecon_loss' :x_f_recon_loss,
            'y_crecon_loss' :y_f_recon_loss,
            'aligin_x_loss' :aligin_x_loss,
            'aligin_y_loss' :aligin_y_loss,
            'kl_x_loss' :kl_x_loss, 
            'kl_y_loss' :kl_y_loss,
            'joint_loss' : joint_loss,
            'label_recon_loss':label_recon_loss,
        }
