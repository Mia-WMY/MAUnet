import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
# 用于存储从文件中读取的值
values = []
dataset="asap7"
our_pth='fig4_data/MAUnet_seed1_{}_result.txt'.format(dataset)
baseline_path='fig4_data/baseline1_seed1_{}_result.txt'.format(dataset)
with open(baseline_path, 'r') as file:
    lines = file.readlines()[1:]  # 读取第二行到最后一行
    for line in lines:
        # 分割每行数据并获取第一个字符串
        parts = line.strip().split()
        if parts:
            values.append(parts[1])

# 将值转换为数值类型（如果是字符串）
values1 = [float(val) for val in values]
values = []
with open(our_pth, 'r') as file:
    lines = file.readlines()[1:]  # 读取第二行到最后一行
    for line in lines:
        # 分割每行数据并获取第一个字符串
        parts = line.strip().split()
        if parts:
            values.append(parts[1])
# 将值转换为数值类型（如果是字符串）
values2 = [float(val) for val in values]
dataset="nangate45"
our_pth='fig4_data/MAUnet_seed1_{}_result.txt'.format(dataset)
baseline_path='fig4_data/baseline1_seed1_{}_result.txt'.format(dataset)
values = []
with open(baseline_path, 'r') as file:
    lines = file.readlines()[1:]  # 读取第二行到最后一行
    for line in lines:
        # 分割每行数据并获取第一个字符串
        parts = line.strip().split()
        if parts:
            values.append(parts[1])

# 将值转换为数值类型（如果是字符串）
values3 = [float(val) for val in values]
values = []
with open(our_pth, 'r') as file:
    lines = file.readlines()[1:]  # 读取第二行到最后一行
    for line in lines:
        # 分割每行数据并获取第一个字符串
        parts = line.strip().split()
        if parts:
            values.append(parts[1])

# 将值转换为数值类型（如果是字符串）
values4 = [float(val) for val in values]
dataset="sky130hd"
dataset2="sky130"
values = []
our_pth='fig4_data/MAUnet_seed1_{}_result.txt'.format(dataset2)
baseline_path='fig4_data/baseline1_seed1_{}_result.txt'.format(dataset2)
with open(baseline_path, 'r') as file:
    lines = file.readlines()[1:]  # 读取第二行到最后一行
    for line in lines:
        # 分割每行数据并获取第一个字符串
        parts = line.strip().split()
        if parts:
            values.append(parts[1])

# 将值转换为数值类型（如果是字符串）
values5 = [float(val) for val in values]

values = []
with open(our_pth, 'r') as file:
    lines = file.readlines()[1:]  # 读取第二行到最后一行
    for line in lines:
        # 分割每行数据并获取第一个字符串
        parts = line.strip().split()
        if parts:
            values.append(parts[1])

# 将值转换为数值类型（如果是字符串）
values6= [float(val) for val in values]




binxs=np.arange(0,1.1,0.1)
fig,axes=plt.subplots(2,3,figsize=(12,8))

# 绘制直方图

hist,bins,_=axes[0,0].hist(values1, bins=binxs, edgecolor='#5F97D2',color='white',hatch='//',linewidth=2 ,label="IREDGe")  # 设置柱子数量为20，颜色为天蓝色，边框为黑色
# plt.title('(a) IREDGe',fontsize=18)  # 设置标题
# plt.xlabel('F1-score',fontsize=18)  # 设置x轴标签
plt.ylabel("Frequency",fontsize=18)  # 设置y轴标签
hist/=len(values1)
print(hist)
print(bins)
print(bins[1]-bins[0])
axes[0,0].clear()
axes[0,0].bar(bins[:-1],hist,width=(bins[1]-bins[0]),edgecolor='#5F97D2',color='white',hatch='//',linewidth=2,label="IREDGe")
axes[0,0].set_ylim(0,0.8)
axes[0,0].set_ylabel("Frequency",fontsize=19)  # 设置y轴标签
axes[0,0].set_title("Asap 7nm",fontsize=19)
plt.subplot(2,3,2)# fig2
# 绘制直方图

hist,bins,_=axes[0,1].hist(values3, bins=binxs, edgecolor='#5F97D2',color='white',hatch='//',linewidth=2 )  # 设置柱子数量为20，颜色为天蓝色，边框为黑色
# plt.title('(a) IREDGe',fontsize=18)  # 设置标题
# plt.xlabel('F1-score',fontsize=18)  # 设置x轴标签
# plt.ylabel("Number of Samples",fontsize=18)  # 设置y轴标签
hist/=len(values3)
print(hist)
print(bins)
print(bins[1]-bins[0])
axes[0,1].clear()
axes[0,1].bar(bins[:-1],hist,width=(bins[1]-bins[0]),edgecolor='#5F97D2',color='white',hatch='//',linewidth=2)
axes[0,1].set_ylim(0,0.8)
axes[0,1].set_title("Nangate 45nm",fontsize=19)
plt.subplot(2,3,3)# fig3
# 绘制直方图
hist,bins,_=axes[0,2].hist(values5, bins=binxs, edgecolor='#5F97D2', color='white',hatch='//',linewidth=2)  # 设置柱子数量为20，颜色为天蓝色，边框为黑色
# plt.title('(a) IREDGe',fontsize=18)  # 设置标题
# plt.xlabel('F1-score',fontsize=18)  # 设置x轴标签
# plt.ylabel("Number of Samples",fontsize=18)  # 设置y轴标签
hist/=len(values5)
print(hist)
print(bins)
print(bins[1]-bins[0])
axes[0,2].clear()
axes[0,2].bar(bins[:-1],hist,width=(bins[1]-bins[0]),edgecolor='#5F97D2',color='white',hatch='//',linewidth=2)
axes[0,2].set_ylim(0,1)
axes[0,2].set_title("Skywater 130nm PDK",fontsize=19)
plt.subplot(2,3,4)# fig4
hist,bins,_=axes[1,0].hist(values2, bins=binxs, color='#5F97D2', edgecolor='black')  # 设置柱子数量为20，颜色为天蓝色，边框为黑色
# plt.title('(b) MAUnet',fontsize=18)  # 设置标题
plt.xlabel('F1-score',fontsize=19)  # 设置x轴标签
# plt.ylabel("Number of Samples",fontsize=18)  # 设置y轴标签
hist/=len(values2)
print(hist)
print(bins)
print(bins[1]-bins[0])
axes[1,0].clear()
axes[1,0].bar(bins[:-1],hist,width=(bins[1]-bins[0]),color='#5F97D2', edgecolor='black')
axes[1,0].set_ylim(0,0.8)
axes[1,0].set_xlabel('F1-score',fontsize=19)
axes[1,0].set_ylabel('Frequency',fontsize=19)
plt.subplot(2,3,5)# fig5
# 绘制直方图
hist,bins,_=axes[1,1].hist(values4, bins=binxs, color='#5F97D2', edgecolor='black')  # 设置柱子数量为20，颜色为天蓝色，边框为黑色
# plt.title('(a) IREDGe',fontsize=18)  # 设置标题
plt.xlabel('F1-score',fontsize=19)  # 设置x轴标签
# plt.ylabel("Number of Samples",fontsize=18)  # 设置y轴标签
hist/=len(values4)
print(hist)
print(bins)
print(bins[1]-bins[0])
axes[1,1].clear()
axes[1,1].bar(bins[:-1],hist,width=(bins[1]-bins[0]),color='#5F97D2', edgecolor='black')
axes[1,1].set_ylim(0,0.8)
axes[1,1].set_xlabel('F1-score',fontsize=19)
plt.subplot(2,3,6)# fig6
# 绘制直方图
hist,bins,_=axes[1,2].hist(values6, bins=binxs, color='#D76364', edgecolor='black')  # 设置柱子数量为20，颜色为天蓝色，边框为黑色
# plt.title('(a) IREDGe',fontsize=18)  # 设置标题
plt.xlabel('F1-score',fontsize=19)  # 设置x轴标签
# plt.ylabel("Number of Samples",fontsize=18)  # 设置y轴标签
hist/=len(values6)
print(hist)
print(bins)
print(bins[1]-bins[0])
axes[1,2].clear()
axes[1,2].bar(bins[:-1],hist,width=(bins[1]-bins[0]),color='#5F97D2', edgecolor='black')
axes[1,2].set_ylim(0,1)
axes[1,2].set_xlabel('F1-score',fontsize=19)
for i in range(2):
    for j in range(3):
        ax=axes[i,j]
        ax.tick_params(axis='both',labelsize=14)
legend_patches=[mpatches.Patch(color="#5F97D2",label="MAUnet"),
                mpatches.Patch(edgecolor='#5F97D2',hatch='//',linewidth=2 ,label="IREDGe")
                ]
# fig.legend(handles=legend_patches,loc="upper center")
plt.subplots_adjust(wspace=0.3)
plt.savefig("hist.pdf")

# 显示图形
plt.show()
