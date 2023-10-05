import pandas as pd
import numpy as np
import math
import csv
import time
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
def read_csv_as_numpy_matrix(file_path):
    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        data_list = list(csvreader)

    # 将数据列表转换为NumPy矩阵
    numpy_matrix = np.array(data_list, dtype=np.float32)  # 可能需要根据数据类型进行调整

    return numpy_matrix

def plot_grid_new(layer,y_values):
    # y_values 为列表
    for x_tuple in y_values:
        a_point = (x_tuple[0], x_tuple[1])
        b_point = (x_tuple[2], x_tuple[3])
        plt.plot([a_point[0], b_point[0]], [a_point[1], b_point[1]])
    if(layer==4 or layer==8):
        plt.gca().set_yticklabels([])
    else:
        plt.gca().set_xticklabels([])
    # 设置图表标题和标签
    plt.title(f'layer:{layer}')
    plt.show()


def save_to_csv(data, filename):
   with open(filename, 'w', newline='') as csvfile:
      writer = csv.writer(csvfile)
      writer.writerows(data)
def read_csv_and_get_shape(csv_file_path):
    try:
        df = pd.read_csv(csv_file_path,header=None)
        shape = df.shape
        return shape
    except FileNotFoundError:
        # print("文件不存在或路径错误。")
        return None
    except pd.errors.EmptyDataError:
        # print("CSV 文件为空。")
        return None
def set_values_in_column(matrix, column_index, start_row, end_row, value):
    matrix[start_row:end_row + 1, column_index] = value
def set_values_in_row(matrix, row_index, start_column, end_column, value):
    matrix[row_index,start_column:end_column+1] = value
def find_connected_blocks(sorted_list, threshold):
    connected_blocks = []
    current_block = []
    for value in sorted_list:
        if not current_block or (value - current_block[-1]) <= threshold:
            current_block.append(value)
        else:
            connected_blocks.append(current_block)
            current_block = [value]
    # If there is any remaining block at the end of the loop, add it to the list.
    if current_block:
        connected_blocks.append(current_block)

    return connected_blocks

def get_via_pitch_csv(x_pth,x):
    # csv_pth = os.path.join(x_pth, 'BeGAN_{:04d}_current.csv'.format(x))
    # netlist_pth = os.path.join(x_pth, 'BeGAN_{:04d}.sp'.format(x))
    csv_pth = os.path.join(x_pth,'BeGAN_{:03d}_current_map.csv'.format(x))
    netlist_pth = os.path.join(x_pth,'BeGAN_{:03d}_reg_grid.sp'.format(x))
    # csv_pth = os.path.join(x_pth, 'current_map{:02d}_current.csv'.format(x))
    # netlist_pth = os.path.join(x_pth, 'current_map{:02d}.sp'.format(x))
    if os.path.exists(netlist_pth):
        csv_shape = read_csv_and_get_shape(csv_pth)
        grid1 = {}
        grid4 = {}
        grid5 = {}
        maxlength = csv_shape[0]  ##默认大小
        data_array_r = np.zeros((0, 8))  # 这几个是i,r,v的array
        data_array_viar = [[0] * maxlength for _ in range(maxlength)]
        data_array_layer_r = [[0] * maxlength for _ in range(maxlength)]  # m1 m4 m7 m8 m9 合并的阻值
        with open(netlist_pth, 'r') as data:
            for temp_line in data:
                components = temp_line.split(" ")
                if components[0][0] == 'R':
                    position1 = components[1].split("_")
                    position2 = components[2].split("_")
                    # 提取各个部分数据
                    node1x = position1[2]  # 节点1
                    node1y = position1[3]
                    layer1 = position1[1]
                    layer2 = position2[1]
                    node2x= position2[2]
                    node2y= position2[3]
                    value = float(components[3])
                    if (layer1 != layer2):
                        x_round = round(int(node1x) / 2000) - 1
                        y_round = round(int(node1y) / 2000) - 1
                        data_array_viar[x_round][y_round] = value
                    else:
                        if layer1=="met1":
                            node1xx = float(node1x) / 2000  # x坐标
                            node1yy = float(node1y) / 2000  # y坐标
                            node2xx = float(node2x) / 2000  # y坐标
                            # node2yy = float(node2y) / 2000  # y坐标
                            # print(node1xx,node2xx,node1yy,node2yy)
                            if grid1.get(float(node1yy)) is None:
                                grid1[float(node1yy)] = []
                            grid1[float(node1yy)].append(float(node1xx))
                            grid1[float(node1yy)].append(float(node2xx))
                        elif layer1=="met4":
                            node1xx = float(node1x) / 2000  # x坐标
                            node1yy = float(node1y) / 2000  # y坐标
                            node2yy = float(node2y) / 2000  # y坐标
                            if grid4.get(float(node1xx)) is None:
                                grid4[float(node1xx)] = []
                            grid4[float(node1xx)].append(float(node1yy))
                            grid4[float(node1xx)].append(float(node2yy))
                        elif layer1=="met5":
                            node1xx = float(node1x) / 2000  # x坐标
                            node1yy = float(node1y) / 2000  # y坐标
                            node2xx = float(node2x) / 2000  # y坐标
                            if grid5.get(float(node1yy)) is None:
                                grid5[float(node1yy)] = []
                            grid5[float(node1yy)].append(float(node1xx))
                            grid5[float(node1yy)].append(float(node2xx))
        #
        matrix = np.zeros((maxlength, maxlength))
        # for key, value in grid4.items():
        #     listx = list(set(value))
        #     listx.sort()
        #     threshold = 2.41
        #     connected_blocks_list = find_connected_blocks(listx, threshold)
        #     for idx, block in enumerate(connected_blocks_list, 1):
        #          set_values_in_column(matrix, int(key), int(block[0]), int(block[-1]), 1)
        # for key, value in grid5.items():
        #     listx = list(set(value))
        #     listx.sort()
        #     threshold = 2.41
        #     connected_blocks_list = find_connected_blocks(listx, threshold)
        #     for idx, block in enumerate(connected_blocks_list, 1):
        #          set_values_in_row(matrix, int(key), int(block[0]), int(block[-1]), 1)
        for key, value in grid1.items():
            listx = list(set(value))
            listx.sort()
            threshold = 2.41
            connected_blocks_list = find_connected_blocks(listx, threshold)
            for idx, block in enumerate(connected_blocks_list, 1):
                 set_values_in_row(matrix, int(key), int(block[0]), int(block[-1]), 1)
        print(matrix)
        df = pd.DataFrame(matrix)
        csv_file=f"pdn_density{x}.csv"
        df.to_csv(csv_file, index=False, header=None)
        # 调用函数读取CSV文件并获得NumPy矩阵
        csv_data_matrix = read_csv_as_numpy_matrix(csv_file)
        matrix = csv_data_matrix
        vmin = np.min(matrix)
        vmax = np.max(matrix)

        colors = [(0, 'white'), (1 / 5, 'cyan'), (2 / 5, 'green'), (3 / 5, 'yellow'), (4 / 5, 'orange'), (1, 'red')]
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
        print(vmin, vmax)
        # vmin=0
        # vmax=1
        # # 创建热图
        plt.imshow(matrix, cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax)
        plt.colorbar(label='Value')  # 添加颜色条以显示值与颜色的对应关系
        plt.title(f'pitch {x}')
        # 显示热图
        plt.show()


for x in range(1,100):
    x_pth='/Users/chococolate/Documents/cad_paper/sky130hd'
    get_via_pitch_csv(x_pth,x)