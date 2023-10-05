from PIL import Image,ImageDraw
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
csv_pth=['fig5_data/BeGAN_0803_ir_drop.csv',
    'fig5_data/baseline1_803output.csv',
         'fig5_data/MAUnet_803output.csv',]
# df=pd.read_csv(csv_pth)
# matrix=df.to_numpy()
# threshold90=np.percentile(matrix,90)
# threshold80=np.percentile(matrix,80)
# threshold70=np.percentile(matrix,70)
# matrix_size=len(matrix)
fig,axes=plt.subplots(3,3,figsize=(12,12))
# above=matrix>threshold
# plt.plot(np.where(above)[1],np.where(above)[0],'ro')
# plt.plot(np.where(~above)[1],np.where(~above)[0],'bo')
for d in range(3):
    df = pd.read_csv(csv_pth[d])
    matrix = df.to_numpy()
    threshold90 = np.percentile(matrix, 90)
    threshold80 = np.percentile(matrix, 80)
    threshold70 = np.percentile(matrix, 70)
    matrix_size = len(matrix)
    index=1
    for i in range(matrix_size):
        for j in range(matrix_size):
            if matrix[i,j]>threshold90:
                axes[index,d].plot(i,j,marker='o',color='#D76364',markersize=2)
            elif matrix[i,j]>threshold80:
                axes[index,d].plot(i,j,marker='o',color='#63E398',markersize=2)
            elif matrix[i, j] > threshold70:
                axes[index,d].plot(i, j, marker='o',color='#9DC3E7', markersize=2)
    # axes[index,0].set_title("Grund Truth",fontsize=20)
    # axes[index,1].set_title("IREDGe Prediction",fontsize=20)
    # axes[index,2].set_title("MAUnet Prediction",fontsize=20)
    axes[index, d].set_xticks([])
    axes[index, d].set_yticks([])

axes[0,0].set_title("Grund Truth",fontsize=20)
axes[0,1].set_title("IREDGe Prediction",fontsize=20)
axes[0,2].set_title("MAUnet Prediction",fontsize=20)

csv_pth=['fig5_data/BeGAN_814_voltage_map_regular.csv',
    'fig5_data/baseline1_814output.csv',
         'fig5_data/MAUnet_814output.csv',]
index=0
for d in range(3):
    df = pd.read_csv(csv_pth[d])
    matrix = df.to_numpy()
    threshold90 = np.percentile(matrix, 90)
    threshold80 = np.percentile(matrix, 80)
    threshold70 = np.percentile(matrix, 70)
    matrix_size = len(matrix)
    for i in range(matrix_size):
        for j in range(matrix_size):
            if matrix[i,j]>threshold90:
                axes[index,d].plot(i,j,marker='o',color='#D76364',markersize=2)
            elif matrix[i,j]>threshold80:
                axes[index,d].plot(i,j,marker='o',color='#63E398',markersize=2)
            elif matrix[i, j] > threshold70:
                axes[index,d].plot(i, j, marker='o',color='#9DC3E7', markersize=2)
    axes[index, d].set_xticks([])
    axes[index, d].set_yticks([])
    # axes[0].set_title("Grund Truth",fontsize=20)
    # axes[1].set_title("IREDGe Prediction",fontsize=20)
    # axes[2].set_title("MAUnet Prediction",fontsize=20)

csv_pth = ['fig5_data/BeGAN_902_voltage_map_regular.csv',
           'fig5_data/baseline1_902output.csv',
           'fig5_data/MAUnet_902output.csv', ]
index=2
for d in range(3):
    df = pd.read_csv(csv_pth[d])
    matrix = df.to_numpy()
    threshold90 = np.percentile(matrix, 90)
    threshold80 = np.percentile(matrix, 80)
    threshold70 = np.percentile(matrix, 70)
    matrix_size = len(matrix)
    for i in range(matrix_size):
        for j in range(matrix_size):
            if matrix[i, j] > threshold90:
                axes[index,d].plot(i, j, marker='o', color='#D76364', markersize=2)
            elif matrix[i, j] > threshold80:
                axes[index,d].plot(i, j, marker='o', color='#63E398', markersize=2)
            elif matrix[i, j] > threshold70:
                axes[index,d].plot(i, j, marker='o', color='#9DC3E7', markersize=2)
    axes[index,d].set_xticks([])
    axes[index, d].set_yticks([])
    # axes[0].set_title("Grund Truth", fontsize=20)
    # axes[1].set_title("IREDGe Prediction", fontsize=20)
    # axes[2].set_title("MAUnet Prediction", fontsize=20)
    #
# plt.xlim(0,matrix_size)
# plt.ylim(0,matrix_size)
plt.tight_layout()
# plt.axis('off')
plt.savefig("pred.png")
plt.show()