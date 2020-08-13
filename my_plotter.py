import matplotlib.pyplot as plt
import numpy as np

#
key_list = ['ROUGE-1 F1', 'ROUGE-2 F1', 'ROUGE-L F1', 'Macro F1']
incon_weight_results = {'ROUGE-1 F1': [15.32, 15.58, 16.82, 14.95, 14.97],
                        'ROUGE-2 F1': [5.68, 5.91, 7.05, 5.51, 5.42],
                        'ROUGE-L F1': [15.07, 15.34, 16.58, 14.72, 14.75],
                        'Macro F1': [56.64, 57.13, 57.37, 54.63, 54.50]}

dec_weight_results = {'ROUGE-1 F1': [17.08, 16.96, 16.82, 16.77, 16.60],
                      'ROUGE-2 F1': [7.17, 7.03, 7.05, 6.94, 6.83],
                      'ROUGE-L F1': [16.83, 16.73, 16.58, 16.51, 16.39],
                      'Macro F1': [55.52, 55.12, 57.37, 57.23, 55.88]}

bottom_list = [5, 0, 5, 50]
top_list = [20, 15, 20, 65]
objects = ('0.01', '0.03', '0.1', '0.3', '1.0')
y_pos = np.arange(len(objects))
# plot_results = incon_weight_results
# for i in range(2):
#     for j in range(2):
#         plot_num = i * 2 + j + 1
#         plt.subplot(2, 2, plot_num)
#         bottom = bottom_list[plot_num - 1]
#         top = top_list[plot_num - 1]
#         key = key_list[plot_num - 1]
#         performance = plot_results[key]
#         # performance = [value - bottom for value in plot_results[key]]
#         # plt.bar(y_pos, performance, align='center', alpha=0.5, bottom=bottom)
#         plt.plot(y_pos, performance, alpha=0.5)
#         plt.ylim(ymin=bottom)
#         plt.ylim(ymax=top)
#         plt.xticks(y_pos, objects)
#         plt.ylabel(key)
#         plt.xlabel(r'$\gamma_4$')

plt.subplot(2, 2, 1)
bottom = 5
top = 20
key = "ROUGE-L F1"
performance = dec_weight_results[key]
plt.plot(y_pos, performance, alpha=0.5, marker='^')
plt.ylim(ymin=bottom)
plt.ylim(ymax=top)
plt.xticks(y_pos, objects)
plt.ylabel(key)
plt.xlabel(r'$\gamma_3$')

plt.subplot(2, 2, 2)
bottom = 50
top = 65
key = "Macro F1"
performance = dec_weight_results[key]
plt.plot(y_pos, performance, alpha=0.5, marker='^')
plt.ylim(ymin=bottom)
plt.ylim(ymax=top)
plt.xticks(y_pos, objects)
plt.ylabel(key)
plt.xlabel(r'$\gamma_3$')

plt.subplot(2, 2, 3)
bottom = 5
top = 20
key = "ROUGE-L F1"
performance = incon_weight_results[key]
plt.plot(y_pos, performance, alpha=0.5, marker='*')
plt.ylim(ymin=bottom)
plt.ylim(ymax=top)
plt.xticks(y_pos, objects)
plt.ylabel(key)
plt.xlabel(r'$\gamma_4$')

plt.subplot(2, 2, 4)
bottom = 50
top = 65
key = "Macro F1"
performance = incon_weight_results[key]
plt.plot(y_pos, performance, alpha=0.5, marker='*')
plt.ylim(ymin=bottom)
plt.ylim(ymax=top)
plt.xticks(y_pos, objects)
plt.ylabel(key)
plt.xlabel(r'$\gamma_4$')


plt.tight_layout()
plt.savefig('figs\\merged.pdf')
plt.show()