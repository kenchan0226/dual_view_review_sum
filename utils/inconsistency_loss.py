import torch


def inconsistency_loss_func(enc_logit, dec_logit, type='KL_div', detach_y=False):
    """
    The function to compute the inconsistency loss between the probability predictions of enc classifier and dec classifier.
    :param enc_logit, FloatTensor: output after the logsoftmax function, [batch, num_classes]
    :param dec_logit, FloatTensor: output after the logsoftmax function, [batch, num_classes]
    :param type: string, valid choices, ['cross_entropy', 'KL_div']
    :return:
    """
    x = enc_logit
    if detach_y:
        y = torch.exp(dec_logit).detach()
    else:
        y = torch.exp(dec_logit)
    if type == 'KL_div':
        # KL divergence loss normalized by batch size
        loss = torch.nn.KLDivLoss(reduction='none', size_average=False, reduce=False)(x, y)
        loss = torch.mean(torch.sum(loss, 1))
    elif type == "cross_entropy":
        # cross entropy loss normalized by batch size
        loss = torch.mean(torch.sum(-y * x, 1))
    else:
        raise ValueError("This kind ({}) of inconsistency loss function is not valid.".format(type))
    return loss