import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go

plotly.offline.init_notebook_mode()
from linear_regression import LinearRegression

data = pd.read_csv('../data/optical density-sugar degree.csv')

#信息导入
train_data = data.sample(frac=0.8)
test_data = data.drop(train_data.index)

input_param_name_1 = 'wavelength.180'
input_param_name_2 = 'wavelength.280'
input_param_name_3 = 'wavelength.380'
input_param_name_4 = 'wavelength.480'
input_param_name_5 = 'wavelength.580'
input_param_name_6 = 'wavelength.680'
input_param_name_7 = 'wavelength.780'
output_param_name = 'sugar'

x_train = train_data[[input_param_name_1,
                      input_param_name_2,
                      input_param_name_3,
                      input_param_name_4,
                      input_param_name_5,
                      input_param_name_6,
                      input_param_name_7
                      ]].values
y_train = train_data[[output_param_name]].values

x_test = test_data[[input_param_name_1,
                    input_param_name_2,
                    input_param_name_3,
                    input_param_name_4,
                    input_param_name_5,
                    input_param_name_6,
                    input_param_name_7
                    ]].values
y_test = test_data[[output_param_name]].values

# 训练散点图-测试集和训练集-暂时先不打印多维数据
"""
#训练值
plot_training_trace = go.Scatter3d(
    x=x_train[:, 0].flatten(),
    y=x_train[:, 1].flatten(),
    z=y_train.flatten(),
    name='Training Set',
    mode='markers',
    marker={
        'size': 10,
        'opacity': 1,
        'line': {
            'color': 'rgb(255, 255, 255)',
            'width': 1
        },
    }
)

#测试值
plot_test_trace = go.Scatter3d(
    x=x_test[:, 0].flatten(),
    y=x_test[:, 1].flatten(),
    z=y_test.flatten(),
    name='Test Set',
    mode='markers',
    marker={
        'size': 10,
        'opacity': 1,
        'line': {
            'color': 'rgb(255, 255, 255)',
            'width': 1
        },
    }
)

#图形配置
plot_layout = go.Layout(
    title='Date Sets',
    scene={
        'xaxis': {'title': input_param_name_4},
        'yaxis': {'title': input_param_name_5},
        'zaxis': {'title': output_param_name}
    },
    margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
)

plot_data = [plot_training_trace, plot_test_trace]

plot_figure = go.Figure(data=plot_data, layout=plot_layout)

plotly.offline.plot(plot_figure, auto_open=False)
"""
#数据处理
num_iterations = 500 #迭代次数
learning_rate = 0.01
polynomial_degree = 0
sinusoid_degree = 0

linear_regression = LinearRegression(x_train, y_train, polynomial_degree, sinusoid_degree)

(theta, cost_history) = linear_regression.train(
    learning_rate,
    num_iterations
)

print('开始损失', cost_history[0])
print('结束损失', cost_history[-1])

plt.plot(range(num_iterations), cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Gradient Descent Progress')
plt.show()

"""
predictions_num = 10

x1_min = x_train[:,:,0].min()#返回所有中的最小值
x1_max = x_train[:,:, 0].max()

x2_min = x_train[:, 1].min()
x2_max = x_train[:, 1].max()

x3_min = 
x3_max = 
"""

#设置三维图坐标范围
"""
x_axis = np.linspace(x_min, x_max, predictions_num)
y_axis = np.linspace(y_min, y_max, predictions_num)
"""

"""
x_predictions = np.zeros((predictions_num * predictions_num, 1))
y_predictions = np.zeros((predictions_num * predictions_num, 1))
"""

"""
x_y_index = 0
for x_index, x_value in enumerate(x_axis):
    for y_index, y_value in enumerate(y_axis):
        x_predictions[x_y_index] = x_value
        y_predictions[x_y_index] = y_value
        x_y_index += 1
"""


y_predictions = linear_regression.predict(x_train)
#y_predictions = linear_regression.predict(np.hstack((x_predictions, y_predictions)))#组合
"""
平均绝对误差(MAE)  MAE用来衡量预测值与真实值之间的平均绝对误差，MAE越小表示模型越好
"""
def MAE(y, y_pre):
    return np.mean(np.abs(y - y_pre))
"""
均方误差(MSE)   MSE也许是回归中最普通的评价指标，MSE越小表示模型越好
"""
def MSE(y, y_pre):
    return np.mean((y - y_pre) ** 2)
"""
均方根误差(RMSE)   RMSE是在MSE的基础之上开根号而来，RMSE越小表示模型越好
"""
def RMSE(y, y_pre):
    return np.sqrt(MSE(y, y_pre))
"""
平均绝对百分比误差(MAPE)  MAPE和MAE类似，只是在MAE的基础上做了标准化处理，MAPE越小表示模型越好
"""
def MAPE(y, y_pre):
    return np.mean(np.abs((y - y_pre) / y))
"""
R^2评价指标  sklearn在实现线性回归时默认采用了R^2指标，R^2越大表示模型越好
"""
def R2(y, y_pre):
    u = np.sum((y - y_pre) ** 2)
    v = np.sum((y - np.mean(y)) ** 2)
    return 1 - (u / v)
#print("model score: ", model.score(x, y))
print("MAE: ", MAE(y_train, y_predictions))
print("MSE: ", MSE(y_train, y_predictions))
print("MAPE: ", MAPE(y_train, y_predictions))
print("R^2: ", R2(y_train, y_predictions))
#绘制预测面
"""
plot_predictions_trace = go.Scatter3d(
    x=x_predictions.flatten(),
    y=y_predictions.flatten(),
    z=z_predictions.flatten(),
    name='Prediction Plane',
    mode='markers',
    marker={
        'size': 1,
    },
    opacity=0.8,
    surfaceaxis=2,
)

plot_data = [plot_training_trace, plot_test_trace, plot_predictions_trace]
plot_figure = go.Figure(data=plot_data, layout=plot_layout)
#plotly.offline.plot(plot_figure,auto_open=False)
"""
print(theta)
#single_data = pd.read_csv('../data/single.csv')

# 读取Excel文件
df = pd.read_excel('../data/single.xlsx')
# 将DataFrame转换为数组
row_vector = df.values.flatten()
#转化为方便计算的行向量
single_data = np.array(row_vector)
b = single_data.reshape(-1,1)
#得到结果值
y_result_test = 0
for i in range(len(theta)):
  y_result_test=y_result_test+b[i]*theta[i]
#调整结果格式
a = np.round(y_result_test * 100, 4)+9.5
y_final = [str(val) + "%" for val in a]
output = " ".join(y_final)
#输出结果
print("正在计算该桃子甜度···")
print("该桃子甜度为:")
print(output)

#窗口部分代码
"""
import tkinter as tk

# 创建主窗口
root = tk.Tk()
root.title("慧眼识桃")
root.geometry('428x926')#iphone12 pro max

# 创建标签
label = tk.Label(root, text="甜度查询", font=("Helvetica", 16))
label.pack(pady=20)


def read_txt_file(filename):
    with open(filename, 'r') as file:
        content = file.read()  # 读取整个文件内容
        words = content.split()  # 按空格分割成单词列表
    return words

filename = '../data/single.txt'
numbers = read_txt_file(filename)
"""

"""
#计算
def calculate():
    # 获取用户输入的数据
    #input_str =(0.00301026 0.002505293 0.073708363 0.240671474 0.462785954 0.03087108 0.001451069 1)
    # 将用户输入的字符串按空格分割成数字字符串列表
    #input_nums_str = input_str.split()

    # 将数字字符串列表转换为浮点数列表，形成行向量
    #input_vector = [float(num) for num in input_nums_str]

    # 计算逐个相乘后的结果并相加
    #y_result_test= sum(input_num * known_num for input_num, known_num in zip(input_vector, theta))+0.09
    single_data = np.array([0.00301026,0.002505293,0.073708363,0.240671474,0.462785954,0.03087108,0.001451069,1])
    b = single_data.reshape(-1, 1)
    y_result_test = 0
    for i in range(len(theta)):
        y_result_test = y_result_test + b[i] * theta[i]
    # 在标签中显示结果
    result_label.config(text=f"结果：{y_result_test+0.085}")

# 创建输入框和标签

result_label = tk.Label(root, text="")
result_label.pack()


# 创建按钮
button = tk.Button(root, text="点击我得甜度结果", command=calculate)
button.pack(pady=5)


# 运行主循环
root.mainloop()
"""