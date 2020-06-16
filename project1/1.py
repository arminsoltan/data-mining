import matplotlib.pyplot as plt
import numpy as np


def show_scatter_plot():
    scatter_plot = dict()
    for x in ['L', 'B', 'R']:
        scatter_plot[x] = list()
    for line in file:
        data = line.split(',')
        scatter_plot[data[0]].append((int(data[1]) * int(data[2]), int(data[3]) * int(data[4])))
        for x in ['L', 'B', 'R']:
            plt.scatter([lis[0] for lis in scatter_plot[x]], [lis[1] for lis in scatter_plot[x]],
                        label='class {}'.format(x))
    plt.ylabel('left_distance * left_weight')
    plt.xlabel('right_distance * right_weight')
    plt.title('scatter plot')
    plt.show()


def show_box_plot():
    box_plot = dict()
    data_type = ['left_distance', 'left_weight', 'right_distance', 'right_weight']
    class_type = ['L', 'B', 'R']
    for x in ['L', 'B', 'R']:
        box_plot[x] = dict()
        for t in data_type:
            box_plot[x][t] = list()
    for line in file:
        data = line.split(',')
        for index, t in enumerate(data_type, start=1):
            box_plot[data[0]][t].append(int(data[index]))
    fig, axs = plt.subplots(3, sharex=True)
    for index, x in enumerate(class_type, start=0):
        data_to_plot = [box_plot[x]['left_distance'], box_plot[x]['left_weight'],
                        box_plot[x]['right_distance'], box_plot[x]['right_weight']]
        axs[index].boxplot(data_to_plot)
        axs[index].set_title('class {}'.format(x))
        axs[index].set_xticklabels(['left_distance', 'left_weight', 'right_distance', 'right_weight'])
    plt.show()


file = open('balance-scale-data.csv', 'r+')
choice = int(input('Do you want show scatter plot (1) or box plot (2)'))
if choice == 1:
    show_scatter_plot()
else:
    show_box_plot()
