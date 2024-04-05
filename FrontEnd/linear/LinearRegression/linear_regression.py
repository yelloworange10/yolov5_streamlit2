import numpy as np

from linear.utils.features import prepare_for_training

class LinearRegression:

    def __init__(self, data, labels, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
        """
        1.对数据进行预处理操作
        2.先得到所有的特征个数
        3.初始化参数矩阵
        """
        (data_processed, #预处理之后的结果
         features_mean,  #标准化操作之后的 mean值和标准差
         features_deviation) = prepare_for_training(data, polynomial_degree, sinusoid_degree, normalize_data=True)

        self.data = data_processed #数据为预处理之后的数据-更新
        self.labels = labels #标签不变
        self.features_mean = features_mean
        self.features_deviation = features_deviation
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data

        num_features = self.data.shape[1] #数据特征个数获取，一行中的列数
        self.theta = np.zeros((num_features, 1)) #构建参数矩阵-行数与特征点相同，列1列

    def train(self, alpha, num_iterations=500): #学习率alpha，迭代次数
        """
        训练模块，执行梯度下降
        """
        #梯度下降函数里得到损失
        cost_history = self.gradient_descent(alpha, num_iterations)
        return self.theta, cost_history

    def gradient_descent(self, alpha, num_iterations):
        """
        实际迭代模块，会迭代num_iterations次
        """
        cost_history = [] #每次损失的变化，建立矩阵
        for _ in range(num_iterations):
            self.gradient_step(alpha)
            cost_history.append(self.cost_function(self.data, self.labels)) #向列表末尾添加一个元素
        return cost_history

    def gradient_step(self, alpha):
        """
        梯度下降参数更新计算方法，注意是矩阵运算
        """
        num_examples = self.data.shape[0] #样本个数，行数
        prediction = LinearRegression.hypothesis(self.data,self.theta) #调用类中方法
        delta = prediction - self.labels #预测值减去真实值-残差
        theta = self.theta #获取当前theta值
        theta = theta - alpha * (1 / num_examples) * (np.dot(delta.T, self.data)).T #转置，矩阵计算方便
        self.theta = theta

    def cost_function(self, data, labels):
        """
        损失计算方法
        """
        num_examples = data.shape[0] #样本总个数
        prediction = LinearRegression.hypothesis(self.data,self.theta)  # 调用类中方法
        delta = prediction - self.labels  #预测值减去实际值
        cost = (1 / 2) * np.dot(delta.T, delta) / num_examples #最小二乘法
        return cost[0][0]

    @staticmethod  #装饰的静态方法
    def hypothesis(data,theta): #获取预测值
        predictions = np.dot(data,theta) #矩阵相乘
        return predictions

    def get_cost(self, data, labels):
        data_processed = prepare_for_training(data,
                                              self.polynomial_degree,
                                              self.sinusoid_degree,
                                              self.normalize_data
                                              )[0]

        return self.cost_function(data_processed, labels)

    def predict(self, data):
        """
        用训练的参数模型，与预测得到回归值结果
        """
        data_processed = prepare_for_training(data,
                                              self.polynomial_degree,
                                              self.sinusoid_degree,
                                              self.normalize_data
                                              )[0]

        predictions = LinearRegression.hypothesis(data_processed, self.theta)

        return predictions
