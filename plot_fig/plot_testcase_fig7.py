import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
i=0

index=[1,5,6,8,10]
avg_mae45=[]
avg_mae130=[]
avg_real=[]
max_mae45=[]
max_mae130=[]
max_real=[]
for i in index:
    truth=pd.read_csv("fig7_data/valid_case{}/ir_drop_map.csv".format(i),header=None)
    nan45=pd.read_csv("fig7_data/nangate45/valid_case{}output.csv".format(i),header=None)
    sky130=pd.read_csv("fig7_data/sky130hd/valid_case{}output.csv".format(i),header=None)
    real = pd.read_csv("fig7_data/non/valid_case{}output.csv".format(i), header=None)
    ttruth=torch.tensor(truth.values)
    tnan45 = torch.tensor(nan45.values)
    tsky130= torch.tensor(sky130.values)
    treal = torch.tensor(real.values)
    ####
    mae45=torch.mean(torch.abs(ttruth-tnan45)).item()

    avg_mae45.append(mae45)

    mae45max=torch.max(torch.abs(ttruth-tnan45)).item()

    max_mae45.append(mae45max)
    ####
    mae130 = torch.mean(torch.abs(ttruth - tsky130)).item()

    avg_mae130.append(mae130)
    ####
    mae130max = torch.max(torch.abs(ttruth - tsky130)).item()
    max_mae130.append(mae130max)
    #####
    mae_real = torch.mean(torch.abs(ttruth - treal)).item()

    avg_real.append(mae_real)

    mae45max = torch.max(torch.abs(ttruth - treal)).item()

    max_real.append(mae45max)

    ###
x=[0,4,8,12,16]

mavg_mae45=[x*1000 for x in avg_mae45]
mavg_mae130=[x*1000 for x in avg_mae130]
mavg_real=[x*1000 for x in avg_real]
max_mae45=[x*1000 for x in max_mae45]
max_mae130=[x*1000 for x in max_mae130]
max_real=[x*1000 for x in max_real]

# plt.bar(x,max_mae45,width=0.4,label='Nangate 45nm',align="center",color="#2878B5")
# x_shift=[i+0.5 for i in x]
# plt.bar(x_shift,max_mae130,width=0.4,label="Skywater 130nm PDK",align="center",color="orange")
# x_shift=[i+1 for i in x]
# plt.bar(x_shift,max_real,width=0.4,label="Real trained",align="center",color="pink")
plt.figure(figsize=(12,6))
x_shift=[i+2 for i in x]
plt.bar(x_shift,mavg_mae45,width=0.6,align="center",color="#2878B5",label="Nangate45-Based")
x_shift=[i+2.6 for i in x]
plt.bar(x_shift,mavg_mae130,width=0.6,align="center",color="orange",label="Skywater130-Based")

x_shift=[i+3.2 for i in x]
plt.bar(x_shift,mavg_real,width=0.6,align="center",color="#32B897",label="Non-Transfer")
plt.ylabel("MAE (mV)",fontsize=18)

labels=['Case 1','Case 2',"Case 3","Case 4","Case 5"]
plt.legend(fontsize=18)
plt.xticks([])
plt.yticks(fontsize=14)
x=[2,6,10,14,18]
# labels=['Avg','Max','Avg','Max','Avg',]
plt.xticks(x,labels,fontsize="18")
plt.savefig("testcase.pdf")
plt.show()