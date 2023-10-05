import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
plt.figure(figsize=(10,6))
# x=[2,3,4,5]
# nan45=[0.607,0.564,0.446,0.422]
# real=[0.572,0.861,0.878,0.775]
# sky130hd=[0.586,0.561,0.576,0.528]
x=[2,3,4,5]
nan45=[0.617,0.564,0.446,0.422]
real=[1.66,1.46,1.115,1.10]
sky130hd=[0.586,0.561,0.576,0.528]
plt.plot(x,nan45,linestyle="-",color="blue",marker='o',label="Nangate45-Based",linewidth=2,markersize=10)
plt.plot(x,real,linestyle="--",color="red",marker='s',label="Non-Transfer",linewidth=2,markersize=10)
plt.plot(x,sky130hd,linestyle="-.",color="green",marker="*",label="Skywater130-Based",linewidth=2,markersize=10)
plt.legend(fontsize=19)
xtickes=[2,3,4,5]
plt.xticks(xtickes,fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("Fine-tuning case number",fontsize=19)
plt.ylabel("MAE (mV)",fontsize=19)
plt.savefig("mae_transfer.png")
plt.show()