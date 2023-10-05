import pandas as pd
import numpy as np
import math
import csv
import time
import os
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
    csv_pth = os.path.join(x_pth,'BeGAN_{:03d}_current_map.csv'.format(x))
    netlist_pth = os.path.join(x_pth,'BeGAN_{:03d}_reg_grid.sp'.format(x))
    csv_shape = read_csv_and_get_shape(csv_pth)
    if csv_shape == None:
        print("jump")
    else:
        grid4 = {}
        maxlength = csv_shape[0]  ##默认大小
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
                    flag = 1
                    if (layer1 != layer2):
                        x_round = round(int(node1x) / 2000) - 1
                        y_round = round(int(node1y) / 2000) - 1
                        data_array_viar[x_round][y_round] = value
                    else:
                        if layer1 == 'met4':
                            node1xx = float(node1x) / 2000  # x坐标
                            node1yy = float(node1y) / 2000  # y坐标
                            node2yy = float(node2y) / 2000  # y坐标
                            if grid4.get(float(node1xx)) is None:
                                grid4[float(node1xx)] = []
                            grid4[float(node1xx)].append(float(node1yy))
                            grid4[float(node1xx)].append(float(node2yy))
                        if layer1 == 'met1':
                            y_round = math.floor(int(node1y) / 2000)
                            x_start = min(int(str(node1x)), int(str(node2x))) / 2000
                            x_start_round = math.floor(x_start)
                            x_end = max(int(str(node1x)), int(str(node2x))) / 2000  # 调整下数据的顺序
                            x_end_round = math.floor(x_end)
                            while (flag == 1):
                                seg_r1 = value / (x_end - x_start)
                                flag = 0  # 每层只执行第一次
                            i = x_end_round - x_start_round

                            data_array_layer_r[y_round][x_start_round] += seg_r1

                            while (i > 0):
                                data_array_layer_r[y_round][x_start_round + i] += seg_r1

                                i = i - 1
                        elif layer1 == 'met4':

                            x_round = math.floor(int(node1x) / 2000)
                            y_start = min(int(str(node1y)), int(str(node2y))) / 2000
                            y_start_round = math.floor(y_start)
                            y_end = max(int(str(node1y)), int(str(node2y))) / 2000  # 调整下数据的顺序
                            y_end_round = math.floor(y_end)
                            while (flag == 1):
                                seg_r4 = value / (y_end - y_start)
                                flag = 0  # 每层只执行第一次
                            i = y_end_round - y_start_round

                            data_array_layer_r[y_start_round][x_round] += seg_r4

                            while (i > 0):
                                data_array_layer_r[y_start_round + i][x_round] += seg_r4

                                i = i - 1

                        else:

                            y_round = math.floor(int(node1y) / 2000)
                            x_start = min(int(str(node1x)), int(str(node2x))) / 2000
                            x_start_round = math.floor(x_start)
                            x_end = max(int(str(node1x)), int(str(node2x))) / 2000  # 调整下数据的顺序
                            x_end_round = math.floor(x_end)
                            while (flag == 1):
                                seg_r9 = value / (x_end - x_start)
                                flag = 0  # 每层只执行第一次
                            i = x_end_round - x_start_round

                            data_array_layer_r[y_round][x_start_round] += seg_r9

                            while (i > 0):
                                data_array_layer_r[y_round][x_start_round + i] += seg_r9

                                i = i - 1
        matrix = np.zeros((maxlength, maxlength))
        for key, value in grid4.items():
            listx = list(set(value))
            listx.sort()
            threshold = 2.41
            connected_blocks_list = find_connected_blocks(listx, threshold)
            for idx, block in enumerate(connected_blocks_list, 1):
                set_values_in_column(matrix, int(key), int(block[0]), int(block[-1]), 1)
        df = pd.DataFrame(matrix)
        df.to_csv(f"sky130hd_ncsv/pitch{x}.csv", index=False, header=None)
        save_to_csv(data_array_viar,
                    f'sky130hd_ncsv/via{x}.csv')
        np_data_array_layer_r = np.array(data_array_layer_r, dtype=np.float32)
        save_to_csv(np_data_array_layer_r, f'sky130hd_ncsv/r{x}.csv')

for x in range(1000):
    x_pth='/Users/xxx/sky130hd'
    get_via_pitch_csv(x_pth,x)