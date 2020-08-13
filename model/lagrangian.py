import torch
import torch.nn as nn


class Lagrangian(nn.Module):
    def __init__(self, num_cost_types, cost_threshold, lagrangian_init_val=0.2, use_hinge_loss=False):
        """
        :param num_cost_types:
        :param cost_threshold: a list of float
        """
        super(Lagrangian, self).__init__()
        assert len(cost_threshold) == num_cost_types
        self.lagrangian_multiplier = nn.Parameter(torch.FloatTensor(num_cost_types))  # [num_cost_types]
        self.cost_threshold = nn.Parameter(torch.FloatTensor(cost_threshold))  # [num_cost_types]
        self.cost_threshold.requires_grad = False
        self.num_cost_types = num_cost_types
        self.init_lagrangian_multiplier(lagrangian_init_val)
        self.use_hinge_loss = use_hinge_loss

    def init_lagrangian_multiplier(self, lagrangian_init_val):
        self.lagrangian_multiplier.data.fill_(lagrangian_init_val)

    def forward(self, cumulative_cost):
        """
        Function for computing \lambda \dot (C_t - \alpha)
        :param cumulative_cost: [batch_size, num_cost_types]
        :return:
        """
        batch_size = cumulative_cost.size(0)
        # (C_t - \alpha)
        constraint_term = cumulative_cost - self.cost_threshold.unsqueeze(0).expand(batch_size, self.num_cost_types)  # [batch, num_cost_types]
        constraint_term_detached = constraint_term.detach()
        # constraint_penalty = \lambda \dot (C_t - \alpha)
        lagrangian_multiplier_expanded = self.lagrangian_multiplier.unsqueeze(0).expand(batch_size, self.num_cost_types)  # [batch, num_cost_types]
        constraint_penalty = -torch.bmm(lagrangian_multiplier_expanded.view(batch_size, 1, self.num_cost_types),
                                        constraint_term.view(batch_size, self.num_cost_types, 1)).squeeze(2).squeeze(1)
        # hinge loss, clamp the loss above zero
        if self.use_hinge_loss:
            constraint_penalty[constraint_penalty > 0] = 0

        # count the number of times the policy violates the constrain, disable it
        #violate_amount = torch.sum(constraint_term_detached[constraint_term_detached > 0]).item()
        violate_amount = 0

        return constraint_penalty.sum(), violate_amount

    def get_lagrangian_multiplier_array(self):
        return self.lagrangian_multiplier.detach().cpu().numpy()

    def compute_regularization(self, cumulative_cost):
        """
        Function for computing \lambda \dot C_t
        :param cumulative_cost: [batch_size, num_cost_types]
        :return:
        """
        batch_size = cumulative_cost.size(0)
        # compute \lambda \dot C_t
        lagrangian_multiplier_expanded = self.lagrangian_multiplier.unsqueeze(0).expand(batch_size,
                                                                                        self.num_cost_types)  # [batch, num_cost_types]
        constraint_regularization = torch.bmm(lagrangian_multiplier_expanded.view(batch_size, 1, self.num_cost_types),
                                        cumulative_cost.view(batch_size, self.num_cost_types, 1)).squeeze(2).squeeze(1)
        return constraint_regularization

    def clamp_lagrangian_multiplier(self):
        # ensure lagrangian_multiplier >= 0
        with torch.no_grad():
            self.lagrangian_multiplier.data = torch.clamp(self.lagrangian_multiplier, min=0.0)
