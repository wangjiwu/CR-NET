import torch
import torch.nn as nn
import torch.nn.functional as func

class BellingLoss(nn.Module):
    def __init__(self):
        super(BellingLoss, self).__init__()
        return

    def forward(self, pred, label):


        loss = \
            torch.mul(torch.add(torch.neg(torch.exp
                                 (torch.div(torch.pow(torch.add(pred, torch.neg(label)), 2), -81*2))), 1), 300)


        #logger.info('mean of bell loss ', torch.sum(loss).cpu().data/torch.numel(loss))

        return torch.sum(loss)/torch.numel(loss) # mean loss
