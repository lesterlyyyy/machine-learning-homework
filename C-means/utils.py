import matplotlib.pyplot as plt
import matplotlib.lines as mlines
def draw_disturbution(train_data,labels):
    fig, axs = plt.subplots(nrows=3, ncols=2, sharex=False, sharey=False, figsize=(19, 9))
    LabelsColors = []
    for i in labels:
        if i == 0:
            LabelsColors.append('black')
        if i == 1:
            LabelsColors.append('orange')
        if i == 2:
            LabelsColors.append('red')
        if i == 3:
            LabelsColors.append('yellow')
        if i == 4:
            LabelsColors.append('blue')
        if i == 5:
            LabelsColors.append('green')
        if i == 6:
            LabelsColors.append('cyan')
        if i == 7:
            LabelsColors.append('purple')
        if i == 8:
            LabelsColors.append('peru')
        if i == 9:
            LabelsColors.append('gold')
    # 画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第二列(玩游戏)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][0].scatter(x=train_data[:, 0], y=train_data[:, 1], color=LabelsColors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs0_title_text = axs[0][0].set_title(u'身高和体重')
    axs0_xlabel_text = axs[0][0].set_xlabel(u'身高')
    axs0_ylabel_text = axs[0][0].set_ylabel(u'体重')
    plt.setp(axs0_title_text, size=9, weight='bold', color='red')
    plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')

        # 画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][1].scatter(x=train_data[:, 0], y=train_data[:, 2], color=LabelsColors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs1_title_text = axs[0][1].set_title(u'身高和50m成绩', )
    axs1_xlabel_text = axs[0][1].set_xlabel(u'身高')
    axs1_ylabel_text = axs[0][1].set_ylabel(u'50m成绩')
    plt.setp(axs1_title_text, size=9, weight='bold', color='red')
    plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')

    # 画出散点图,以datingDataMat矩阵的第二(玩游戏)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[1][0].scatter(x=train_data[:, 1], y=train_data[:, 2], color=LabelsColors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs2_title_text = axs[1][0].set_title(u'体重和50m成绩')
    axs2_xlabel_text = axs[1][0].set_xlabel(u'体重')
    axs2_ylabel_text = axs[1][0].set_ylabel(u'50m成绩')
    plt.setp(axs2_title_text, size=9, weight='bold', color='red')
    plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')

    axs[1][1].scatter(x=train_data[:, 0], y=train_data[:, 3], color=LabelsColors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs3_title_text = axs[1][1].set_title(u'身高和肺活量')
    axs3_xlabel_text = axs[1][1].set_xlabel(u'身高')
    axs3_ylabel_text = axs[1][1].set_ylabel(u'肺活量')
    plt.setp(axs3_title_text, size=9, weight='bold', color='red')
    plt.setp(axs3_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs3_ylabel_text, size=7, weight='bold', color='black')

    axs[2][0].scatter(x=train_data[:, 1], y=train_data[:, 3], color=LabelsColors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs4_title_text = axs[2][0].set_title(u'体重和肺活量')
    axs4_xlabel_text = axs[2][0].set_xlabel(u'体重')
    axs4_ylabel_text = axs[2][0].set_ylabel(u'肺活量')
    plt.setp(axs4_title_text, size=9, weight='bold', color='red')
    plt.setp(axs4_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs4_ylabel_text, size=7, weight='bold', color='black')

    axs[2][1].scatter(x=train_data[:, 2], y=train_data[:, 3], color=LabelsColors, s=15, alpha=.5)
    # 设置标题,x轴label,y轴label
    axs5_title_text = axs[2][1].set_title(u'50m成绩和肺活量')
    axs5_xlabel_text = axs[2][1].set_xlabel(u'50m成绩')
    axs5_ylabel_text = axs[2][1].set_ylabel(u'肺活量')
    plt.setp(axs5_title_text, size=9, weight='bold', color='red')
    plt.setp(axs5_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs5_ylabel_text, size=7, weight='bold', color='black')

    # 设置图例
    class_1 = mlines.Line2D([], [], color='black', marker='.',
                              markersize=6, label='类别1')
    class_2 = mlines.Line2D([], [], color='orange', marker='.',
                               markersize=6, label='类别2')
    class_3 = mlines.Line2D([], [], color='red', marker='.',
                               markersize=6, label='类别3')
    class_4 = mlines.Line2D([], [], color='yellow', marker='.',
                            markersize=6, label='类别4')
    class_5 = mlines.Line2D([], [], color='blue', marker='.',
                            markersize=6, label='类别5')
    class_6 = mlines.Line2D([], [], color='green', marker='.',
                            markersize=6, label='类别6')
    class_7 = mlines.Line2D([], [], color='cyan', marker='.',
                            markersize=6, label='类别7')
    class_8 = mlines.Line2D([], [], color='purple', marker='.',
                            markersize=6, label='类别8')
    class_9 = mlines.Line2D([], [], color='peru', marker='.',
                            markersize=6, label='类别9')
    class_10 = mlines.Line2D([], [], color='gold', marker='.',
                            markersize=6, label='类别10')

    # 添加图例
    axs[0][0].legend(handles=[class_1, class_2, class_3,class_4,class_5,class_6,class_7,class_8,class_9,class_10])
    axs[0][1].legend(
        handles=[class_1, class_2, class_3, class_4, class_5, class_6, class_7, class_8, class_9, class_10])
    axs[1][0].legend(
        handles=[class_1, class_2, class_3, class_4, class_5, class_6, class_7, class_8, class_9, class_10])
    axs[1][1].legend(
        handles=[class_1, class_2, class_3, class_4, class_5, class_6, class_7, class_8, class_9, class_10])
    axs[2][0].legend(
        handles=[class_1, class_2, class_3, class_4, class_5, class_6, class_7, class_8, class_9, class_10])
    axs[2][1].legend(
        handles=[class_1, class_2, class_3, class_4, class_5, class_6, class_7, class_8, class_9, class_10])
    plt.tight_layout()
    plt.show()