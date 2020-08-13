import math
import time


class LossStatistics:
    """
    Accumulator for loss staistics. Modified from OpenNMT
    """

    def __init__(self, loss=0.0, n_tokens=0, n_batch=0, forward_time=0.0, loss_compute_time=0.0, backward_time=0.0):
        assert type(loss) is float or type(loss) is int
        assert type(n_tokens) is int
        self.loss = loss
        if math.isnan(loss):
            raise ValueError("Loss is NaN")
        self.n_tokens = n_tokens
        self.n_batch = n_batch
        self.forward_time = forward_time
        self.loss_compute_time = loss_compute_time
        self.backward_time = backward_time

    def update(self, stat, update_n_src_words=False):
        """
        Update statistics by suming values with another `LossStatistics` object

        Args:
            stat: another statistic object
        """
        self.loss += stat.loss
        if math.isnan(stat.loss):
            raise ValueError("Loss is NaN")
        self.n_tokens += stat.n_tokens
        self.n_batch += stat.n_batch
        self.forward_time += stat.forward_time
        self.loss_compute_time += stat.loss_compute_time
        self.backward_time += stat.backward_time

    def xent(self):
        """ compute normalized cross entropy """
        assert self.n_tokens > 0, "n_tokens must be larger than 0"
        return self.loss / self.n_tokens

    def ppl(self):
        """ compute normalized perplexity """
        assert self.n_tokens > 0, "n_tokens must be larger than 0"
        return math.exp(min(self.loss / self.n_tokens, 100))

    def total_time(self):
        return self.forward_time, self.loss_compute_time, self.backward_time

    def clear(self):
        self.loss = 0.0
        self.n_tokens = 0
        self.n_batch = 0
        self.forward_time = 0.0
        self.loss_compute_time = 0.0
        self.backward_time = 0.0


class RewardStatistics:
    """
    Accumulator for reward staistics.
    """
    def __init__(self, final_reward=0.0, pg_loss=0.0, n_batch=0, sample_time=0, q_estimate_compute_time=0, backward_time=0):
        self.final_reward = final_reward
        self.pg_loss = pg_loss
        if math.isnan(pg_loss):
            raise ValueError("Policy gradient loss is NaN")
        self.n_batch = n_batch
        self.sample_time = sample_time
        self.q_estimate_compute_time = q_estimate_compute_time
        self.backward_time = backward_time

    def update(self, stat):
        self.final_reward += stat.final_reward
        if math.isnan(stat.pg_loss):
            raise ValueError("Policy gradient loss is NaN")
        self.pg_loss += stat.pg_loss
        self.n_batch += stat.n_batch
        self.sample_time += stat.sample_time
        self.q_estimate_compute_time += stat.q_estimate_compute_time
        self.backward_time += stat.backward_time

    def reward(self):
        assert self.n_batch > 0, "n_batch must be positive"
        return self.final_reward / self.n_batch

    def loss(self):
        assert self.n_batch > 0, "n_batch must be positive"
        return self.pg_loss / self.n_batch

    def total_time(self):
        return self.sample_time, self.q_estimate_compute_time, self.backward_time

    def clear(self):
        self.final_reward = 0.0
        self.pg_loss = 0.0
        self.n_batch = 0.0
        self.sample_time = 0.0
        self.q_estimate_compute_time = 0.0
        self.backward_time = 0.0


class LagrangianStatistics:
    def __init__(self, lagrangian_loss=0.0, lagrangian_grad_norm=0.0, violate_amount=0, n_batch=0):
        self.lagrangian_loss = lagrangian_loss
        if math.isnan(lagrangian_loss):
            raise ValueError("Loss is NaN")
        self.lagrangian_grad_norm = lagrangian_grad_norm
        self.violate_amount = violate_amount
        self.n_batch = n_batch

    def update(self, stat):
        self.lagrangian_loss += stat.lagrangian_loss
        self.n_batch += stat.n_batch
        self.lagrangian_grad_norm += stat.lagrangian_grad_norm
        self.violate_amount += stat.violate_amount

    def clear(self):
        self.lagrangian_loss = 0.0
        self.lagrangian_grad_norm = 0.0
        self.n_batch = 0
        self.violate_amount = 0

    def loss(self):
        assert self.n_batch > 0, "n_batch must be positive"
        return self.lagrangian_loss / self.n_batch

    def grad_norm(self):
        assert self.n_batch > 0, "n_batch must be positive"
        return self.lagrangian_grad_norm / self.n_batch

    def violate_amt(self):
        assert self.n_batch > 0, "n_batch must be positive"
        return self.violate_amount / self.n_batch


class JointLossStatistics:
    """
    Accumulator for loss staistics. Modified from OpenNMT
    """

    def __init__(self, joint_loss=0.0, generation_loss=0.0, enc_classification_loss=0.0, dec_classification_loss=0.0, inconsist_loss=0.0, n_iterations=0, n_tokens=0, forward_time=0.0, loss_compute_time=0.0, backward_time=0.0):
        assert type(joint_loss) is float or type(joint_loss) is int
        assert type(n_iterations) is int
        self._joint_loss = joint_loss
        if math.isnan(joint_loss):
            raise ValueError("Loss is NaN")
        self._generation_loss = generation_loss
        self._enc_classification_loss = enc_classification_loss
        self._dec_classification_loss = dec_classification_loss
        self._inconsist_loss = inconsist_loss
        self._n_iterations = n_iterations
        self._n_tokens = n_tokens
        self._forward_time = forward_time
        self._loss_compute_time = loss_compute_time
        self._backward_time = backward_time

    def update(self, stat):
        """
        Update statistics by suming values with another `LossStatistics` object

        Args:
            stat: another statistic object
        """
        self._joint_loss += stat._joint_loss
        self._generation_loss += stat._generation_loss
        self._enc_classification_loss += stat._enc_classification_loss
        self._dec_classification_loss += stat._dec_classification_loss
        self._inconsist_loss += stat._inconsist_loss
        if math.isnan(stat._joint_loss):
            raise ValueError("Loss is NaN")
        self._n_iterations += stat._n_iterations
        self._n_tokens += stat._n_tokens
        self._forward_time += stat._forward_time
        self._loss_compute_time += stat._loss_compute_time
        self._backward_time += stat._backward_time

    def generation_loss(self):
        """ compute normalized cross entropy """
        assert self._n_tokens > 0, "n_tokens must be larger than 0"
        return self._generation_loss / self._n_tokens

    def ppl(self):
        """ compute normalized perplexity """
        assert self._n_tokens > 0, "n_tokens must be larger than 0"
        return math.exp(min(self._generation_loss / self._n_tokens, 100))

    def joint_loss(self):
        assert self._n_iterations > 0, "n_tokens must be larger than 0"
        return self._joint_loss / self._n_iterations

    def classification_loss(self):
        assert self._n_iterations > 0, "n_tokens must be larger than 0"
        return (self._enc_classification_loss + self._dec_classification_loss) / self._n_iterations

    def enc_classification_loss(self):
        assert self._n_iterations > 0, "n_tokens must be larger than 0"
        return self._enc_classification_loss / self._n_iterations

    def dec_classification_loss(self):
        assert self._n_iterations > 0, "n_tokens must be larger than 0"
        return self._dec_classification_loss / self._n_iterations

    def inconsist_loss(self):
        assert self._n_iterations > 0, "n_tokens must be larger than 0"
        return self._inconsist_loss / self._n_iterations

    def total_time(self):
        return self._forward_time, self._loss_compute_time, self._backward_time

    def clear(self):
        self._joint_loss = 0.0
        self._generation_loss = 0.0
        self._enc_classification_loss = 0.0
        self._dec_classification_loss = 0.0
        self._inconsist_loss = 0.0
        self._n_iterations = 0
        self._n_tokens = 0
        self._forward_time = 0.0
        self._loss_compute_time = 0.0
        self._backward_time = 0.0
