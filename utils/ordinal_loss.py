import torch
import torch.nn as nn

class OrdinalLossBasic(nn.Module):
    def __init__(self, num_classes, device):
        super(OrdinalLossBasic, self).__init__()
        self.num_classes = num_classes
        self.class_encoding = torch.zeros(num_classes, num_classes).to(device)  # [num_classes, num_classes]
        for i in range(num_classes):
            self.class_encoding.data[i, :i + 1] = 1
        #print(self.class_encoding.device)
        #print("class encoding matrix")
        #print(self.class_encoding)

    def forward(self, *input):
        raise NotImplementedError

class OrdinalXELoss(OrdinalLossBasic):
    def __init__(self, num_classes, device):
        super(OrdinalXELoss, self).__init__(num_classes, device)
        self.bce_loss = nn.BCELoss(reduction='mean')

    def forward(self, input_ordinal_dist, rating):
        """
        :param input_ordinal_dist: [batch, num_classes]
        :param rating: [batch]
        :return:
        """
        #print(rating.detach().cpu().numpy())
        batch_size = rating.size(0)
        #input_ordinal_dist_flattened = input_ordinal_dist.view(batch_size * self.num_classes)  # [batch * num_classes]
        ground_truth_ordinal_dist = self.class_encoding[rating]  # [batch, num_classes]
        #print(ground_truth_ordinal_dist.detach().cpu().numpy())
        #print(input_ordinal_dist.detach().cpu().numpy())
        #ground_truth_ordinal_dist_flattened = ground_truth_ordinal_dist.view(batch_size * self.num_classes)
        loss = self.bce_loss(input_ordinal_dist, ground_truth_ordinal_dist)
        #print(loss.detach().item())
        return loss

class OrdinalMSELoss(OrdinalLossBasic):
    def __init__(self, num_classes, device):
        super(OrdinalMSELoss, self).__init__(num_classes, device)
        self.mse_loss = nn.MSELoss(reduction='mean')

    def forward(self, input_ordinal_dist, rating):
        """
        :param input_ordinal_dist: [batch, num_classes]
        :param rating: [batch]
        :return:
        """
        #print(rating.detach().cpu().numpy())
        batch_size = rating.size(0)
        ground_truth_ordinal_dist = self.class_encoding[rating]  # [batch, num_classes]
        #print(ground_truth_ordinal_dist.detach().cpu().numpy())
        #print(input_ordinal_dist.detach().cpu().numpy())
        loss = self.mse_loss(input_ordinal_dist, ground_truth_ordinal_dist)
        #print(loss.detach().item())
        return loss
