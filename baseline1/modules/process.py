import csv
import glob
import re
import pandas as pd
import os
import torch
import numpy as np

def read_info(file_path): # 74x74
    with open(file_path, "r") as file:
         reader = csv.reader(file)
         data = list(reader)
    num_rows = len(data)
    num_columns = len(data[0]) if data else 0
    #print("Number of columns:", num_columns)
    return num_rows,num_columns


def read_data(path,str):
    file_list=glob.glob(path+str)
    #row,column=read_info(file_list[0])
    row=128
    column=128
    num=len(glob.glob(path+str))
    #print("row=",row,",column=",column)
    #print("file_len:", num)
    current_map=np.zeros((num,row,column))
    for i,fname in enumerate(file_list):
        #print("fname",fname)
        with open(fname) as csvfile:
            read_csv=csv.reader(csvfile,delimiter=',')
            #rows=[row for row in read_csv]
            #print(f"row={len(rows)},col={len(rows[0])}")
            for row_num,row in enumerate(read_csv):
                for col_num,value in enumerate(row):
                    current_map[i,row_num,col_num] =float(value)

    return current_map

def read_fake_data_nangate45(folder_path):
    file_pattern=['BeGAN_{:04d}_current.csv','BeGAN_{:04d}_eff_dist.csv','BeGAN_{:04d}_pdn_density.csv']
    tfile_pattern='BeGAN_{:04d}_ir_drop.csv'
    # add_pth='/home/wangmingyue/nangate45_netcsv'
    # matrix_patterns=['pitch{:d}.csv']
    # vias_patterns=['via{:d}.csv']
    # r_patterns = ['r{:d}.csv']
    data_list=[]
    tdata_list=[]
    for i in range(800):
        tfile_path = os.path.join(folder_path, tfile_pattern.format(i))
        if(os.path.exists(tfile_path)):
            concat_data=None
            for pattern in file_pattern:
                file_path=os.path.join(folder_path,pattern.format(i))
                csv_content=pd.read_csv(file_path,header=None)
                torch_content=torch.tensor(csv_content.values)
                expand_torch=torch.unsqueeze(torch_content,dim=0)
                if concat_data is None:
                    concat_data=expand_torch
                else:
                    concat_data=torch.cat((concat_data,expand_torch),dim=0)
            # for matrix in matrix_patterns:
            #     file_pth=os.path.join(add_pth,matrix.format(i))
            #     csv_content=pd.read_csv(file_pth,header=None)
            #     torch_content=torch.tensor(csv_content.values)
            #     # torch_content=torch.flip(torch_content,[0])
            #     expand_torch=torch.unsqueeze(torch_content,dim=0)
            #     concat_data=torch.cat((concat_data,expand_torch),dim=0)
            # for matrix in vias_patterns:
            #     file_pth=os.path.join(add_pth,matrix.format(i))
            #     csv_content=pd.read_csv(file_pth,header=None)
            #     torch_content=torch.tensor(csv_content.values)
            #     # torch_content=torch.flip(torch_content,[0])
            #     expand_torch=torch.unsqueeze(torch_content,dim=0)
            #     concat_data=torch.cat((concat_data,expand_torch),dim=0)
            # for matrix in r_patterns:
            #     file_pth=os.path.join(add_pth,matrix.format(i))
            #     csv_content=pd.read_csv(file_pth,header=None)
            #     torch_content=torch.tensor(csv_content.values)
            #     # torch_content=torch.flip(torch_content,[0])
            #     expand_torch=torch.unsqueeze(torch_content,dim=0)
            #     concat_data=torch.cat((concat_data,expand_torch),dim=0)
            concat_data=torch.unsqueeze(concat_data,0)
            data_list.append(concat_data)
        ####3
            tcsv_content = pd.read_csv(tfile_path, header=None)
            ttorch_content = torch.tensor(tcsv_content.values)
            texpand_torch = torch.unsqueeze(ttorch_content, dim=0)
            texpand_torch=torch.unsqueeze(texpand_torch,0)
            tdata_list.append(texpand_torch)
    # tensor_data=torch.Tensor(tensor_list)
    # wx, wtx, hx, htx, whx, whtx = flip(data_list, tdata_list)
    # data_list = data_list + wx
    # data_list = data_list + hx
    # data_list = data_list + whx
    # tdata_list = tdata_list + wtx
    # tdata_list = tdata_list + htx
    # tdata_list = tdata_list + whtx
    # print(len(data_list), len(tdata_list))
    return data_list,tdata_list



def read_fake_data(folder_path):
    file_pattern=['current_map{:02d}_current.csv','current_map{:02d}_eff_dist.csv','current_map{:02d}_pdn_density.csv']
    tfile_pattern='current_map{:02d}_ir_drop.csv'
    data_list=[]
    tdata_list=[]
    for i in range(90):
        tfile_path = os.path.join(folder_path, tfile_pattern.format(i))
        if(os.path.exists(tfile_path)):
            concat_data=None
            for pattern in file_pattern:
                file_path=os.path.join(folder_path,pattern.format(i))
                csv_content=pd.read_csv(file_path,header=None)
                torch_content=torch.tensor(csv_content.values)
                expand_torch=torch.unsqueeze(torch_content,dim=0)
                if concat_data is None:
                    concat_data=expand_torch
                else:
                    concat_data=torch.cat((concat_data,expand_torch),dim=0)

            concat_data=torch.unsqueeze(concat_data,0)
            data_list.append(concat_data)
        ####3
            tcsv_content = pd.read_csv(tfile_path, header=None)
            ttorch_content = torch.tensor(tcsv_content.values)
            texpand_torch = torch.unsqueeze(ttorch_content, dim=0)
            texpand_torch=torch.unsqueeze(texpand_torch,0)
            tdata_list.append(texpand_torch)
    # tensor_data=torch.Tensor(tensor_list)
    return data_list,tdata_list


#with open(file_path,'r') as file:
#read_info(file_path)


def read_real_data(folder_path):
    data_list=[]
    tdata_list=[]
    train_set = [1,2,5,6,3,4,11,12,17,18]
    for i in train_set:
        file_pattern = ['current_map.csv', 'eff_dist_map.csv', 'pdn_density.csv']
        folder_name = 'testcase{}'.format(i)
        file_path = os.path.join(folder_path, folder_name)
        if os.path.exists(file_path):
            concat_data = None
            for pattern in file_pattern:
                csv1 = os.path.join(folder_path, folder_name, pattern)
                csv1 = pd.read_csv(csv1, header=None)
                tsv1 = torch.tensor(csv1.values)
                expand_torch = torch.unsqueeze(tsv1, dim=0)
                if concat_data is None:
                    concat_data = expand_torch
                else:
                    concat_data = torch.cat((concat_data, expand_torch), dim=0)
            concat_data = torch.unsqueeze(concat_data, 0)
            data_list.append(concat_data)
            # tdata
            tfile_path = os.path.join(folder_path, folder_name, 'ir_drop_map.csv')
            tcsv_content = pd.read_csv(tfile_path, header=None)
            ttorch_content = torch.tensor(tcsv_content.values)
            texpand_torch = torch.unsqueeze(ttorch_content, dim=0)
            texpand_torch = torch.unsqueeze(texpand_torch, 0)
            tdata_list.append(texpand_torch)
    wx, wtx, hx, htx, whx, whtx = flip(data_list, tdata_list)
    data_list = data_list + wx
    data_list = data_list + hx
    data_list = data_list + whx
    tdata_list = tdata_list + wtx
    tdata_list = tdata_list + htx
    tdata_list = tdata_list + whtx
    print(len(data_list), len(tdata_list))
    return data_list, tdata_list
    #     folder_name='testcase{}'.format(i)
    #     file_path=os.path.join(folder_path,folder_name)
    #     matrix_pth = "/home/wangmingyue/Desktop/iccad/real_netlist_csv"
    #     matrix = "matrix{:d}.csv"
    #     if os.path.exists(file_path):
    #         csv1=os.path.join(folder_path,folder_name,"current_map.csv")
    #         csv2=os.path.join(folder_path,folder_name,"eff_dist_map.csv")
    #         csv3=os.path.join(folder_path,folder_name,"pdn_density.csv")
    #         tcsv=os.path.join(folder_path,folder_name,"ir_drop_map.csv")
    #         mcsv = os.path.join(matrix_pth, matrix.format(i))
    #         csv1=pd.read_csv(csv1, header=None)
    #         csv2 = pd.read_csv(csv2, header=None)
    #         csv3 = pd.read_csv(csv3, header=None)
    #         csv4 = pd.read_csv(tcsv, header=None)
    #         mcsv = pd.read_csv(mcsv, header=None)
    #         tsv1=torch.tensor(csv1.values)
    #         esv1=torch.unsqueeze(tsv1,dim=2)
    #         tsv2 = torch.tensor(csv2.values)
    #         esv2 = torch.unsqueeze(tsv2, dim=2)
    #         tsv3 = torch.tensor(csv3.values)
    #         esv3 = torch.unsqueeze(tsv3, dim=2)
    #         tsv4 = torch.tensor(csv4.values)
    #         esv4 = torch.unsqueeze(tsv4, dim=2)
    #         mtsv = torch.tensor(mcsv.values)
    #         mesv = torch.unsqueeze(mtsv, dim=0)
    #         concat=torch.cat((esv1,esv2,esv3,mesv),dim=2)
    #         data_list.append(concat)
    #         tdata_list.append(esv4)
    # return data_list,tdata_list

def flip(data_list,tdata_list):
    #水平翻转
    flipped_tensors_x_datalist=[]
    flipped_tensors_x_tdatalist = []
    #垂直翻转
    flipped_tensors_y_datalist = []
    flipped_tensors_y_tdatalist = []
    #对角翻转
    flipped_x_y_datalist=[]
    flipped_x_y_tdatalist = []
    for (i,j) in zip(data_list,tdata_list):
        bz,ch,h,w=i.shape
        flipped=i.clone()
        for x in range(ch):
            matrix=i[:,x,:,:]
            # print(matrix)
            flip_matrix=torch.flip(matrix,[2])
            # print(flip_matrix)
            flipped[:,x,:,:]=flip_matrix
        flipped2 = torch.flip(j, [3])

        flipped_tensors_x_datalist.append(flipped)
        flipped_tensors_x_tdatalist.append(flipped2)


    for (i, j) in zip(data_list, tdata_list):
        bz, ch, h, w = i.shape
        flipped = i.clone()
        for x in range(ch):
            matrix = i[:, x, :, :]
            flip_matrix = torch.flip(matrix, [1])
            flipped[:, x, :, :] = flip_matrix
        flipped2 = torch.flip(j, [2])
        flipped_tensors_y_tdatalist.append(flipped2)
        flipped_tensors_y_datalist.append(flipped)

    for (i, j) in zip(data_list, tdata_list):
        flipped=torch.flip(i,[2,3])
        flipped3=torch.flip(j,[2,3])
        flipped_x_y_datalist.append(flipped)
        flipped_x_y_tdatalist.append(flipped3)

    return flipped_tensors_x_datalist,flipped_tensors_x_tdatalist,flipped_tensors_y_datalist,flipped_tensors_y_tdatalist,flipped_x_y_datalist,flipped_x_y_tdatalist

def read_fake_data_nangate45_test(folder_path):
    file_pattern=['BeGAN_{:04d}_current.csv','BeGAN_{:04d}_eff_dist.csv','BeGAN_{:04d}_pdn_density.csv']
    tfile_pattern='BeGAN_{:04d}_ir_drop.csv'
    add_pth = '/home/wangmingyue/date/nangate45_ncsv'
    matrix_patterns = ['pitch{:d}.csv']
    vias_patterns = ['via{:d}.csv']
    r_patterns = ['r{:d}.csv']
    data_list=[]
    tdata_list=[]
    for i in range(800,1001):
        tfile_path = os.path.join(folder_path, tfile_pattern.format(i))
        if(os.path.exists(tfile_path)):
            concat_data=None
            for pattern in file_pattern:
                file_path=os.path.join(folder_path,pattern.format(i))
                csv_content=pd.read_csv(file_path,header=None)
                torch_content=torch.tensor(csv_content.values)
                expand_torch=torch.unsqueeze(torch_content,dim=0)
                if concat_data is None:
                    concat_data=expand_torch
                else:
                    concat_data=torch.cat((concat_data,expand_torch),dim=0)
            for matrix in matrix_patterns:
                file_pth=os.path.join(add_pth,matrix.format(i))
                csv_content=pd.read_csv(file_pth,header=None)
                torch_content=torch.tensor(csv_content.values)
                # torch_content=torch.flip(torch_content,[0])
                expand_torch=torch.unsqueeze(torch_content,dim=0)
                concat_data=torch.cat((concat_data,expand_torch),dim=0)
            for matrix in vias_patterns:
                file_pth=os.path.join(add_pth,matrix.format(i))
                csv_content=pd.read_csv(file_pth,header=None)
                torch_content=torch.tensor(csv_content.values)
                # torch_content=torch.flip(torch_content,[0])
                expand_torch=torch.unsqueeze(torch_content,dim=0)
                concat_data=torch.cat((concat_data,expand_torch),dim=0)
            for matrix in r_patterns:
                file_pth=os.path.join(add_pth,matrix.format(i))
                csv_content=pd.read_csv(file_pth,header=None)
                torch_content=torch.tensor(csv_content.values)
                # torch_content=torch.flip(torch_content,[0])
                expand_torch=torch.unsqueeze(torch_content,dim=0)
                concat_data=torch.cat((concat_data,expand_torch),dim=0)
            concat_data=torch.unsqueeze(concat_data,0)
            data_list.append(concat_data)
        ####3
            tcsv_content = pd.read_csv(tfile_path, header=None)
            ttorch_content = torch.tensor(tcsv_content.values)
            texpand_torch = torch.unsqueeze(ttorch_content, dim=0)
            texpand_torch=torch.unsqueeze(texpand_torch,0)
            tdata_list.append(texpand_torch)
    # tensor_data=torch.Tensor(tensor_list)
    return data_list,tdata_list


def read_fake_data_test_asap7(folder_path):
    file_pattern = ['BeGAN_{:03d}_current_map.csv', 'eff_dist_{:03d}.csv', 'pdn_density_{:d}.csv']
    tfile_pattern = 'BeGAN_{:03d}_voltage_map_regular.csv'
    # add_pth='/home/wangmingyue/nangate45_netcsv'
    # matrix_patterns=['pitch{:d}.csv']
    # vias_patterns=['via{:d}.csv']
    # r_patterns = ['r{:d}.csv']
    data_list = []
    tdata_list = []
    for i in range(800,1001):
        tfile_path = os.path.join(folder_path, 'pdn_density_{:d}.csv'.format(i))
        if (os.path.exists(tfile_path)):
            concat_data = None
            shape = 0
            for pattern in file_pattern:
                file_path = os.path.join(folder_path, pattern.format(i))
                csv_content = pd.read_csv(file_path, header=None)
                torch_content = torch.tensor(csv_content.values)
                if concat_data is None:
                    shape = torch_content.shape
                    expand_torch = torch.unsqueeze(torch_content, dim=0)
                    concat_data = expand_torch
                else:
                    ex_result = np.zeros(shape)
                    ex_result[:torch_content.shape[0], :torch_content.shape[1]] = torch_content
                    expand_torch = torch.unsqueeze(torch.tensor(ex_result), dim=0)
                    concat_data = torch.cat((concat_data, expand_torch), dim=0)

            concat_data = torch.unsqueeze(concat_data, 0)
            data_list.append(concat_data)
            ####3
            tfile_path = os.path.join(folder_path, tfile_pattern.format(i))
            tcsv_content = pd.read_csv(tfile_path, header=None)
            ttorch_content = torch.tensor(tcsv_content.values)
            texpand_torch = torch.unsqueeze(ttorch_content, dim=0)
            texpand_torch = torch.unsqueeze(texpand_torch, 0)
            tdata_list.append(texpand_torch)

    return data_list, tdata_list

def read_fake_data_test_sky130hd(folder_path):
    file_pattern = ['BeGAN_{:03d}_current_map.csv', 'eff_dist_{:03d}.csv', 'pdn_density{:d}.csv']
    tfile_pattern = 'BeGAN_{:03d}_voltage_map_regular.csv'
    # file_pattern = ['BeGAN_{:03d}_current_map.csv', 'eff_dist_{:03d}.csv', 'pdn_density_{:d}.csv']
    # tfile_pattern = 'BeGAN_{:03d}_voltage_map_regular.csv'
    # add_pth='/home/wangmingyue/nangate45_netcsv'
    # matrix_patterns=['pitch{:d}.csv']
    # vias_patterns=['via{:d}.csv']
    # r_patterns = ['r{:d}.csv']
    data_list = []
    tdata_list = []
    for i in range(800,1001):
        tfile_path = os.path.join(folder_path, 'pdn_density{:d}.csv'.format(i))
        # print(tfile_path)
        if (os.path.exists(tfile_path)):
            concat_data = None
            shape = 0
            for pattern in file_pattern:
                file_path = os.path.join(folder_path, pattern.format(i))
                csv_content = pd.read_csv(file_path, header=None)
                torch_content = torch.tensor(csv_content.values)
                if concat_data is None:
                    shape = torch_content.shape
                    expand_torch = torch.unsqueeze(torch_content, dim=0)
                    concat_data = expand_torch
                else:
                    ex_result = np.zeros(shape)
                    ex_result[:torch_content.shape[0], :torch_content.shape[1]] = torch_content
                    expand_torch = torch.unsqueeze(torch.tensor(ex_result), dim=0)
                    concat_data = torch.cat((concat_data, expand_torch), dim=0)

            concat_data = torch.unsqueeze(concat_data, 0)
            data_list.append(concat_data)
            ####3
            tfile_path = os.path.join(folder_path, tfile_pattern.format(i))
            tcsv_content = pd.read_csv(tfile_path, header=None)
            ttorch_content = torch.tensor(tcsv_content.values)
            texpand_torch = torch.unsqueeze(ttorch_content, dim=0)
            texpand_torch = torch.unsqueeze(texpand_torch, 0)
            tdata_list.append(texpand_torch)

    return data_list, tdata_list