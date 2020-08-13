import numpy as np
import torch

def binary_results_to_rating_preds(binary_results):
    """
    :param binary_results: np array: [batch, num_classes]
    :return:
    """
    predicted_classes = list(map(binary_to_rating, binary_results))
    return np.array(predicted_classes)


def binary_to_rating(binary_result):
    """
    :param binary_result: np array: [num_classes]
    :return:
    """
    predicted_rating = -1
    for i in range(binary_result.shape[0]):
        if binary_result[i] >= .5:
            predicted_rating += 1
        else:
            break
    if predicted_rating == -1:
        predicted_rating = 0
    return predicted_rating


def binary_results_tensor_to_rating_preds_tensor(binary_results):
    """
    :param binary_results: tensor: [batch, num_classes]
    :return:
    """
    batch_size, num_classes = binary_results.size()
    predicted_rating_list = []
    for i in range(batch_size):
        predicted_rating = -1
        for j in range(num_classes):
            if binary_results.data[i, j] >= .5:
                predicted_rating += 1
            else:
                break
        if predicted_rating == -1:
            predicted_rating = 0
        predicted_rating_list.append(predicted_rating)
    return torch.LongTensor(predicted_rating_list).to(binary_results.device)

