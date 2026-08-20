import torch
import torch.nn as nn
import torch.nn.functional as F


class NtXentLoss(nn.Module):
    def __init__(self, batch_size, temperature):
        super(NtXentLoss, self).__init__()
        #self.batch_size = batch_size
        self.temperature = temperature
        #self.device = device

        #self.mask = self.mask_correlated_samples(batch_size)
        self.similarityF = nn.CosineSimilarity(dim = 2)
        self.criterion = nn.CrossEntropyLoss(reduction = 'sum')
    

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size 
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask
    

    def forward(self, z_i, z_j, device):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N - 1) augmented examples within a minibatch as negative examples.
        """
        batch_size = z_i.shape[0]
        N = 2 * batch_size

        z = torch.cat((z_i, z_j), dim=0)
        

        sim = self.similarityF(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        #sim = 0.5 * (z_i.shape[1] - torch.tensordot(z.unsqueeze(1), z.T.unsqueeze(0), dims = 2)) / z_i.shape[1] / self.temperature

        sim_i_j = torch.diag(sim, batch_size )
        sim_j_i = torch.diag(sim, -batch_size )
        
        mask = self.mask_correlated_samples(batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).view(N, 1)
        negative_samples = sim[mask].view(N, -1)

        labels = torch.zeros(N).to(device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss
    

class InfoNCELoss(nn.Module):
    def __init__(self, batch_size, temperature):
        super(InfoNCELoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, z_i, z_j, device):
        batch_size = self.batch_size
        N = 2 * batch_size

        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        sim = torch.matmul(z_i, z_j.T) / self.temperature

        positive_samples = torch.diag(sim).view(batch_size, 1)
        negative_samples = sim.view(batch_size, -1)

        labels = torch.arange(batch_size).to(device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= batch_size
        return loss