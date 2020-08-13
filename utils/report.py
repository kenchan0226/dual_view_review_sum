import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from os.path import join


def export_lagrangian_stats(lagrangian_loss, lagrangian_multipliers, lagrangian_grad_norms, violate_amount, plot_every, path):
    num_cost_types = lagrangian_multipliers[0].shape[0]
    # Export the loss, grad norm, and violate amount
    export_csv_and_plot("$L_\lambda$", 'lagrangian_loss',lagrangian_loss, path)
    export_csv_and_plot("gradient of $\lambda$", 'lagrangian_grad', lagrangian_grad_norms, path)
    export_csv_and_plot("amount of violation", 'violate_amount', violate_amount, path)

    # Export the values of lagrangian multiplier
    value_exp_path = join(path, 'lagrangian_value')
    for i in range(num_cost_types):
        lagrangian_i_list = [multiplier[i] for multiplier in lagrangian_multipliers]
        export_csv_and_plot("$\lambda_{}$".format(i), "lagrangian_value_{}".format(i), lagrangian_i_list, path)


def export_csv_and_plot(label, filename, value_list, path):
    with open(join(path, "{}.csv".format(filename)), 'w') as result_csv:
        result_csv.write(concat_float_list(value_list, ',') + '\n')
    plot_curve(value_list, join(path, filename), label)


def export_train_and_valid_reward(train_reward, valid_reward, plot_every, path):
    # Export the results to a csv file
    labels = ['Training reward:,', 'Validation reward:,']
    float_lists = [train_reward, valid_reward]
    with open(path + '.csv', 'w') as result_csv:
        for i in range(len(labels)):
            result_csv.write(labels[i] + concat_float_list(float_lists[i], ',') + '\n')
    print("Training and valid loss saved to: {}".format(path + '.csv'))
    # Export the plots to pdf file
    plot_train_valid_curve(train_reward, valid_reward, plot_every, path, 'Reward')
    print("Training and valid loss plot saved to: {}".format(path + '_reward.pdf'))


def export_train_and_valid_loss(train_loss, valid_loss, train_ppl, valid_ppl, plot_every, path):
    """
    :param train_loss: a list of float
    :param valid_loss: a list of float
    :param train_ppl: a list of float
    :param valid_ppl: a list of float
    :param plot_every: int
    :param path: str
    :return:
    """
    # Export the results to a csv file
    labels = ['Training loss:,', 'Validation loss:,', 'Training perplexity:,', 'Validation Perplexity:,']
    float_lists = [train_loss, valid_loss, train_ppl, valid_ppl]
    with open(path + '.csv', 'w') as result_csv:
        for i in range(len(labels)):
            result_csv.write(labels[i] + concat_float_list(float_lists[i], ',') + '\n')
    print("Training and valid loss saved to: {}".format(path + '.csv'))
    # Export the plots to pdf file
    plot_train_valid_curve(train_loss, valid_loss, plot_every, path, 'Loss')
    plot_train_valid_curve(train_ppl, valid_ppl, plot_every, path, 'Perplexity')
    print("Training and valid loss plot saved to: {}".format(path + '_loss.pdf'))
    print("Training and valid ppl plot saved to: {}".format(path + '_perplexity.pdf'))


def concat_float_list(list, delimiter=','):
    return delimiter.join([str(l) for l in list])


def plot_curve(value_list, path, value_label):
    plt.figure()
    plt.xlabel("Checkpoints")
    plt.ylabel(value_label)
    num_checkpoints = len(value_list)
    X = list(range(num_checkpoints))
    plt.plot(X, value_list, label="training")
    plt.legend()
    plt.savefig("%s.pdf" % (path))


def plot_train_curve(train_loss, plot_every, path, loss_label):
    #plt.ioff()
    title = "Training %s for every %d iterations" % (loss_label.lower(), plot_every)
    plt.figure()
    plt.title(title)
    plt.xlabel("Checkpoints")
    plt.ylabel(loss_label)
    num_checkpoints = len(train_loss)
    X = list(range(num_checkpoints))
    plt.plot(X, train_loss, label="training")
    plt.legend()
    plt.savefig("%s_%s.pdf" % (path, loss_label.lower()))


def plot_train_valid_curve(train_loss, valid_loss, plot_every, path, loss_label):
    #plt.ioff()
    title = "Training and validation %s for every %d iterations" % (loss_label.lower(), plot_every)
    plt.figure()
    plt.title(title)
    plt.xlabel("Checkpoints")
    plt.ylabel(loss_label)
    num_checkpoints = len(train_loss)
    X = list(range(num_checkpoints))
    plt.plot(X, train_loss, label="training")
    plt.plot(X, valid_loss, label="validation")
    plt.legend()
    plt.savefig("%s_%s.pdf" % (path, loss_label.lower()))

if __name__ == '__main__':
    train_loss = [20.1,15.3,12.3,11.0,10.0]
    valid_loss = [30.2,29.2,25.2,21.3,20.2]
    train_ppl = [10.1,5.3,2.3,1.0,1.0]
    valid_ppl = [20.2,19.2,15.2,11.3,10.2]

    plot_every = 4000
    path = '../exp/debug/valid_train_curve'
    export_train_and_valid_loss(train_loss, valid_loss, train_ppl, valid_ppl, plot_every, path)
