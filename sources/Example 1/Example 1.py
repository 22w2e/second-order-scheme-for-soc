import time
import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as normal

tf.keras.backend.set_floatx('float64')


class Solver(object):
    def __init__(self, scheme='euler'):
        self.valid_size = 512
        self.batch_size = 64
        self.num_iterations = 10000
        self.logging_frequency = 100
        self.lr_values = [5e-2, 5e-3, 1e-3]
        self.lr_boundaries = [5000, 8000]
        self.config = Config()
        self.scheme = scheme  # 'euler' or 'cn' for Crank-Nicolson
        self.model = WholeNet(scheme=scheme)
        self.y_init = self.model.y_init
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            self.lr_boundaries, self.lr_values)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-8)

        # 初始化用于存储损失函数统计量的列表
        self.loss_history = []
        self.steps_history = []
        self.loss_mean_history = []
        self.loss_5_percentile_history = []
        self.loss_95_percentile_history = []
        self.cost_history = []

    def train(self):
        """Training the model"""
        start_time = time.time()
        training_history = []

        # 使用相同的验证集评估模型
        dW = self.config.sample(self.valid_size)
        valid_data = dW

        for step in range(self.num_iterations + 1):
            if step % self.logging_frequency == 0:
                loss, cost = self.model(valid_data, training=True)
                y_init = self.y_init.numpy()[0][0]
                elapsed_time = time.time() - start_time
                training_history.append([step, cost, y_init, loss])

                # 记录损失值和步数
                self.loss_history.append(loss.numpy())
                self.cost_history.append(cost.numpy())
                self.steps_history.append(step)

                # 计算移动窗口统计量
                self.calculate_moving_statistics()

                # 在Solver类的train方法中，修改打印语句：
                print(
                    f"Scheme: {self.scheme.upper()}, step: {step:5d}, loss: {loss:.4e}, Y0: {y_init:.4e}, cost: {cost:.4e}, elapsed time: {elapsed_time:3.0f}")

            # 训练步骤
            self.train_step(self.config.sample(self.batch_size))

        print(f'Scheme: {self.scheme.upper()}, Y0_true: {y_init:.4e}')
        self.training_history = training_history

    def calculate_moving_statistics(self):
        """计算移动窗口统计量"""
        window_size = 20  # 滑动窗口大小
        if len(self.loss_history) >= window_size:
            window_losses = self.loss_history[-window_size:]
            self.loss_mean_history.append(np.mean(window_losses))
            self.loss_5_percentile_history.append(np.percentile(window_losses, 5))
            self.loss_95_percentile_history.append(np.percentile(window_losses, 95))
        else:
            # 窗口未填满时使用当前所有数据
            self.loss_mean_history.append(np.mean(self.loss_history))
            self.loss_5_percentile_history.append(np.percentile(self.loss_history, 5))
            self.loss_95_percentile_history.append(np.percentile(self.loss_history, 95))

    @tf.function
    def train_step(self, train_data):
        """Updating the gradients"""
        with tf.GradientTape(persistent=True) as tape:
            loss, cost = self.model(train_data, training=True)
        grad = tape.gradient(loss, self.model.trainable_variables)
        del tape
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))


class WholeNet(tf.keras.Model):
    """Building the neural network architecture"""

    def __init__(self, scheme='euler'):
        super(WholeNet, self).__init__()
        self.config = Config()
        self.scheme = scheme
        self.y_init = tf.Variable(tf.random.normal([1, self.config.dim_y], mean=0, stddev=1, dtype=tf.dtypes.float64))
        self.z_net = [FNNet() for _ in range(self.config.num_time_interval)]

    def call(self, dw, training=True):
        config = self.config
        batch_size = tf.shape(dw)[0]
        delta_t = config.delta_t
        t_stamps = config.t_stamp

        # 初始状态
        x = tf.ones([batch_size, config.dim_x], dtype=tf.float64)
        y = tf.matmul(tf.ones([batch_size, 1], dtype=tf.float64), self.y_init)

        l = 0.0  # 代价函数

        for i in range(config.num_time_interval):
            t_i = t_stamps[i]
            dW = dw[:, :, i]

            z_i = self.z_net[i]((t_i, x, y), training=training)
            u_i = config.u_fn(t_i, x, y, z_i)
            b_i = config.b_fn(t_i, x, y, z_i)
            sigma_i = config.sigma_fn(t_i, x, y, z_i)
            f_i = config.Hx_fn(t_i, x, y, z_i)

            # 累加代价
            l += config.f_fn(t_i, x, u_i) * delta_t

            if self.scheme == 'euler':
                # Euler 离散
                x_next = x + b_i * delta_t + sigma_i * dW
                y_next = y - f_i * delta_t + z_i * dW
            else:
                # Crank-Nicolson 离散
                # 初始预测 (Euler)
                x_next = x + b_i * delta_t + sigma_i * dW
                y_next = y - f_i * delta_t + z_i * dW

                # Picard 迭代 (3次)
                for _ in range(3):
                    z_next = self.z_net[i]((t_i + delta_t, x_next, y_next), training=training)
                    u_next = config.u_fn(t_i + delta_t, x_next, y_next, z_next)
                    b_next = config.b_fn(t_i + delta_t, x_next, y_next, z_next)
                    sigma_next = config.sigma_fn(t_i + delta_t, x_next, y_next, z_next)
                    f_next = config.Hx_fn(t_i + delta_t, x_next, y_next, z_next)

                    # CN 更新
                    x_next = x + 0.5 * (b_i + b_next) * delta_t + 0.5 * (sigma_i + sigma_next) * dW
                    y_next = y - 0.5 * (f_i + f_next) * delta_t + 0.5 * (z_i + z_next) * dW

            x = x_next
            y = y_next

        delta = y + config.hx_tf(config.total_T, x)
        loss = tf.reduce_mean(tf.reduce_sum(delta ** 2, axis=1, keepdims=True))

        l += config.h_fn(config.total_T, x)
        cost = tf.reduce_mean(l)

        return loss, cost


class FNNet(tf.keras.Model):
    """ Define the feedforward neural network """

    def __init__(self):
        super(FNNet, self).__init__()
        self.config = Config()
        num_hiddens = [self.config.dim_x + 10, self.config.dim_x + 10, self.config.dim_x + 10]
        self.bn_layers = [
            tf.keras.layers.BatchNormalization(
                momentum=0.99,
                epsilon=1e-6,
                beta_initializer=tf.random_normal_initializer(0.0, stddev=0.1),
                gamma_initializer=tf.random_uniform_initializer(0.1, 0.5)
            )
            for _ in range(len(num_hiddens) + 2)]
        self.dense_layers = [tf.keras.layers.Dense(num_hiddens[i],
                                                   use_bias=False,
                                                   activation=None)
                             for i in range(len(num_hiddens))]
        # final output should be gradient of size dim_z
        self.dense_layers.append(tf.keras.layers.Dense(self.config.dim_z, activation=None))

    def call(self, inputs, training=True):
        """structure: bn -> (dense -> bn -> relu) * len(num_hiddens) -> dense -> bn"""
        t, x, y = inputs
        ts = tf.ones([tf.shape(x)[0], 1], dtype=tf.dtypes.float64) * t
        x = tf.concat([ts, x, y], axis=1)
        x = self.bn_layers[0](x, training=training)
        for i in range(len(self.dense_layers) - 1):
            x = self.dense_layers[i](x)
            x = self.bn_layers[i + 1](x, training=training)
            x = tf.nn.relu(x)
        x = self.dense_layers[-1](x)
        x = self.bn_layers[-1](x, training=training)
        return x


class Config(object):
    """Define the configs in the systems"""

    def __init__(self):
        super(Config, self).__init__()
        self.dim_x = 100
        self.dim_y = 100
        self.dim_z = 100
        self.num_time_interval = 25
        self.total_T = 0.1
        self.delta_t = (self.total_T + 0.0) / self.num_time_interval
        self.sqrth = np.sqrt(self.delta_t)
        self.t_stamp = np.arange(0, self.num_time_interval) * self.delta_t

    def sample(self, num_sample):
        dw_sample = normal.rvs(size=[num_sample, self.num_time_interval]) * self.sqrth
        return dw_sample[:, np.newaxis, :]

    def f_fn(self, t, x, u):
        return 0.25 * tf.reduce_sum(x ** 2, 1, keepdims=True) + tf.reduce_sum(u ** 2, 1, keepdims=True)

    def h_fn(self, t, x):
        ones = tf.ones(shape=tf.stack([self.dim_x, self.dim_x]), dtype=tf.dtypes.float64)
        inputs = tf.matmul(x, ones)
        return 0.5 * tf.reduce_sum(inputs * x, 1, keepdims=True)

    def b_fn(self, t, x, y, z):
        return -0.25 * x + 0.5 * y + 0.5 * z

    def sigma_fn(self, t, x, y, z):
        return 0.2 * x + 0.5 * y + 0.5 * z

    def Hx_fn(self, t, x, y, z):
        return -0.5 * x - 0.25 * y + 0.2 * z

    def hx_tf(self, t, x):
        ones = tf.ones(shape=tf.stack([self.dim_x, self.dim_x]), dtype=tf.dtypes.float64) * 1.0
        return tf.matmul(x, ones)

    def u_fn(self, t, x, y, z):
        return 0.5 * (y + z)


def plot_comparison(euler_solver, cn_solver):
    """绘制两种离散方法的对比图（分开绘制）"""
    # 检查可用的样式
    print("Available matplotlib styles:", plt.style.available)

    # 选择可用的样式（例如'ggplot'或'seaborn-v0_8'）
    try:
        plt.style.use('default')  # 较新版本的seaborn样式
    except:
        plt.style.use('ggplot')  # 备选样式

    # 损失函数对比图
    plt.figure(figsize=(12, 6))
    plt.plot(euler_solver.steps_history, euler_solver.loss_history,
             label='deep SMP-BSDE', color='blue', linewidth=2, alpha=0.8)
    plt.plot(cn_solver.steps_history, cn_solver.loss_history,
             label='improved deep SMP-BSDE', color='red', linewidth=2, alpha=0.8)

    plt.xlabel('Training Step', fontsize=18)
    plt.ylabel('Loss Value', fontsize=18)
    plt.title('Loss Function Comparison ', fontsize=18)
    plt.legend(fontsize=18)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('loss_comparison.png', dpi=600, bbox_inches='tight')

    # 代价泛函对比图
    plt.figure(figsize=(12, 6))
    plt.plot(euler_solver.steps_history, euler_solver.cost_history,
             label='deep SMP-BSDE', color='blue', linewidth=2, alpha=0.8)
    plt.plot(cn_solver.steps_history, cn_solver.cost_history,
             label='improved deep SMP-BSDE', color='red', linewidth=2, alpha=0.8)

    plt.xlabel('Training Step', fontsize=18)
    plt.ylabel('Cost Value', fontsize=18)
    plt.title('Cost Functional Comparison ', fontsize=18)
    plt.legend(fontsize=18)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('cost_comparison.png', dpi=600, bbox_inches='tight')
    plt.show()

    print("对比图已保存为 loss_comparison.png 和 cost_comparison.png")


def main():
    # 训练Euler离散模型
    print('Training Euler scheme...')
    euler_solver = Solver(scheme='euler')
    euler_solver.train()

    # 训练Crank-Nicolson离散模型
    print('\nTraining Crank-Nicolson scheme...')
    cn_solver = Solver(scheme='cn')
    cn_solver.train()

    # 绘制对比图
    plot_comparison(euler_solver, cn_solver)

    # 保存结果
    def save_results(solver, filename):
        data = np.array(solver.training_history)
        output = np.zeros((len(data[:, 0]), 4))
        output[:, 0] = data[:, 0]  # step
        output[:, 1] = data[:, 2]  # y_init
        output[:, 2] = data[:, 3]  # loss
        output[:, 3] = data[:, 1]  # cost
        np.savetxt(filename, output, fmt=['%d', '%.5e', '%.5e', '%.5e'], delimiter=',')

    save_results(euler_solver, './euler_results.csv')
    save_results(cn_solver, './cn_results.csv')

    print('\nSolving is done!')


if __name__ == '__main__':
    main()