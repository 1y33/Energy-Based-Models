import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class EBM(nn.Module):
    def __init__(self,energy_net,alpha,sigma,ld_steps,D):
        super().__init__()

        self.energy_net = energy_net
        self.nll = nn.NLLLoss(reduction="none")
        self.D=D
        self.sigma=sigma
        self.alpha = torch.FloatTensor([alpha])
        self.ld_steps = ld_steps

    def classify(self,x):
        f_xy = self.energy_net(x)
        y_pred = torch.softmax(f_xy,1)
        return torch.argmax(y_pred,dim=1)

    def class_loss(self,f_xy,y):
        y_pred = torch.softmax(f_xy,1)
        return self.nll(torch.log(y_pred),y)

    def gen_loss(self,x,f_xy):
        x_sample = self.sample(x=None,batch_size=x.shape[0])
        f_x_sample_y = self.energy_net(x_sample)

        return -(torch.logsumexp(f_xy,1)-torch.logsumexp(f_x_sample_y,1))

    def forward(self,x,y,reduction="avg"):
        f_xy = self.energy_net(x)
        L_clf = self.class_loss(f_xy,y)
        L_gen = self.gen_loss(x.f_xy)

        if reduction == "sum":
            loss = (L_clf + L_gen).sum()
        else:
            loss = (L_clf + L_gen).mean()

        return loss

    def energy_gradient(self,x):
        self.energy_net.eval()

        x_i = torch.FloatTensor(x.data)
        x_i.requires_grad = True

        x_i_grad = torch.autograd.grad(torch.logsumexp(self.energy_net(x_i),1).sum(),[x_i],retain_graph=True)[0]

        self.energy_net.train()

        return x_i_grad

    def langevine_dynamics_step(self,x_old,alpha):
        grad_energy = self.energy_gradient(x_old)
        epsilon = torch.randn_like(grad_energy) * self.sigma

        x_new = x_old + alpha * grad_energy + epsilon

        return x_new

    def sample(self,batch_size=64,x=None):
        x_sample = 2. * torch.rand([batch_size,self.D])-1
        for i in range(self.ld_steps):
            x_sample = self.langevine_dynamics_step(x_sample,alpha= self.alpha)

        return x_sample
