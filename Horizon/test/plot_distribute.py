import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def load_data(filename):
    with open(filename, 'r') as file:
        return json.load(file)


def plot_distribution(data, labels, colors, title, xlim=None):
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # 第一组数据（左侧y轴）
    array1 = np.array(data[0])
    mean1, var1 = np.mean(array1), np.var(array1)
    print(f"{labels[0]} - Mean: {mean1:.2f}, Variance: {var1:.2f}")

    # 使用 fill_between 填充KDE曲线下方区域
    kde1 = stats.gaussian_kde(array1,  bw_method=0.3)
    x_min1, x_max1 = array1.min(), array1.max()
    if xlim:
        x_min1, x_max1 = xlim
    x1 = np.linspace(x_min1, x_max1, 1000)
    y1 = kde1(x1)

    ax1.fill_between(x1, y1, alpha=0.2, color=colors[0], label=labels[0])
    ax1.plot(x1, y1, color=colors[0], linewidth=2)

    # ax1.yaxis.set_visible(False)
    ax1.grid(True, alpha=0.3)
    # ax1.set_ylabel(f'{labels[0]} Density', color=colors[0], fontsize=12)
    ax1.tick_params(axis='y', labelcolor=colors[0], labelsize=14)
    ax1.set_ylim(bottom=0)
    ax1.set_ylim((0, 1.6))
    ax1.yaxis.set_major_locator(plt.MaxNLocator(6))
    # ax1.legend(loc='upper left')

    # 第二组数据（右侧y轴）
    ax2 = ax1.twinx()

    array2 = np.array(data[1])
    mean2, var2 = np.mean(array2), np.var(array2)
    print(f"{labels[1]} - Mean: {mean2:.2f}, Variance: {var2:.2f}")

    kde2 = stats.gaussian_kde(array2,  bw_method=0.3)
    x_min2, x_max2 = array2.min(), array2.max()
    if xlim:
        x_min2, x_max2 = xlim
    x2 = np.linspace(x_min2, x_max2, 1000)
    y2 = kde2(x2)

    ax2.fill_between(x2, y2, alpha=0.2, color=colors[1], label=labels[1])
    ax2.plot(x2, y2, color=colors[1], linewidth=2)

    # ax2.grid(True, alpha=0.3)
    # ax2.yaxis.set_visible(False)
    # ax2.set_ylabel(f'{labels[1]} Density', color=colors[1], fontsize=12)
    ax2.tick_params(axis='y', labelcolor=colors[1], labelsize=14)
    ax2.set_ylim(bottom=0)
    ax2.yaxis.set_major_locator(plt.MaxNLocator(6))

    ax2.set_ylim((0, 0.4))
    # ax2.legend(loc='upper right')

    for spine in ['top']:
        ax1.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)
        ax1.spines['left'].set_visible(False)
    # 共用x轴设置

    if xlim:
        ax1.set_xlim(xlim)

    ax1.tick_params(axis='x', labelsize=14)
      # 左轴6个刻度
      # 右轴6个刻度
    # ax1.set_title(title)
    # ax1.set_xlabel('Value')

    plt.tight_layout()
    plt.show()


def cdf(data, labels, colors):
    fig, ax = plt.subplots(figsize=(8, 6))

    for array, label, color in zip(data, labels, colors):
        array_sorted = np.sort(array)
        cdf = np.arange(1, len(array_sorted) + 1) / len(array_sorted)
        ax.plot(array_sorted, cdf, color=color, linewidth=2, label=label)

    ax.set_xlabel('Value')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative Distribution Function')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()

def main():
    '''data = load_data('D:\code\prototype_weakly\seg\logits_spol.json')
    import numpy as np
    arr = np.array(data['negative'])
    lower = np.percentile(arr, 20)
    upper = np.percentile(arr, 98)
    new_neg = arr[(arr >= lower) & (arr <= upper)]
    lower = np.percentile(arr, 45)
    upper = np.percentile(arr, 55)
    mid_neg = new_neg[(new_neg >= lower) & (new_neg <= upper)].tolist()
    lower = np.percentile(new_neg, 45)
    upper = np.percentile(new_neg, 55)
    new_neg = new_neg[(new_neg <= lower) | (new_neg >= upper)].tolist()

    arr = np.array(data['positive'])
    lower = np.percentile(arr, 42)
    upper = np.percentile(arr, 52)
    mid_pos = arr[(arr >= lower) & (arr <= upper)].tolist()
    lower = np.percentile(arr, 40)
    upper = np.percentile(arr, 100)
    new_pos = arr[(arr >= lower) & (arr <= upper)].tolist()
    # lower = np.percentile(arr, 45)
    # upper = np.percentile(arr, 55)
    # new_pos = arr[(arr <= lower) | (arr >= upper)].tolist()
    k = len(mid_pos)
    import random
    mid_neg = random.sample(mid_neg, k)
    new_pos = mid_neg + new_pos
    new_neg = new_neg + mid_pos'''

    data = load_data(r'D:\code\Repetition3\test\all_result\horizon_spol\ins\all_logits.json')
    import numpy as np
    arr = np.array(data['back'])
    '''lower = np.percentile(arr, 20)
    upper = np.percentile(arr, 100)'''
    lower = np.percentile(arr, 50)
    upper = np.percentile(arr, 99)
    new_neg = arr[(arr >= lower) & (arr <= upper)].tolist()

    arr = np.array(data['fore'])
    '''lower = np.percentile(arr, 10)
    upper = np.percentile(arr, 100)'''
    lower = np.percentile(arr, 1)
    upper = np.percentile(arr, 100)
    new_pos = arr[(arr >= lower) & (arr <= upper)].tolist()
    '''lower = np.percentile(arr, 0)
    upper = np.percentile(arr, 10)
    sm_pos = arr[(arr >= lower) & (arr <= upper)].tolist()
    k = int(len(sm_pos) / 5)
    import random
    sm_pos = random.sample(sm_pos, k)
    new_pos = sm_pos + new_pos'''


    '''cdf(
            data=[data['positive'], new_list],
            labels=['Hemorrhage', 'Normal Tissue'],
            colors=['b', 'r'],
        )'''

    plot_distribution(
        data=[new_neg, new_pos],
        labels=['Hemorrhage', 'Normal Tissue'],
        colors=['#3498DB', '#EC407A'],
        title='Distribution of Features',
        xlim=(-10, 10)
    )


if __name__ == "__main__":
    main()