import time
import numpy as np
import tensorflow as tf
import math
from scipy.stats import multivariate_normal as normal
import matplotlib.pyplot as plt

tf.keras.backend.set_floatx('float64')


class Solver(object):
    def __init__(self, scheme='euler'):
        self.valid_size = 512
        self.batch_size = 64
        self.num_iterations = 10000
        self.logging_frequency = 100
        self.lr_values = [5e-3, 5e-3, 5e-3]
        self.lr_boundaries = [2000, 4000]
        self.config = Config()
        self.scheme = scheme  # 'euler' or 'cn'

        # 根据离散方案选择模型
        if scheme == 'euler':
            self.model = EulerWholeNet()
        else:
            self.model = CNWholeNet()

        self.y_init = self.model.y_init
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            self.lr_boundaries, self.lr_values)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-8)

        # 存储训练历史
        self.loss_history = []
        self.cost_history = []
        self.steps_history = []

    def train(self):
        """Training the model"""
        start_time = time.time()
        training_history = []
        dW = self.config.sample(self.valid_size)
        valid_data = dW

        for step in range(self.num_iterations + 1):
            if step % self.logging_frequency == 0:
                loss, cost = self.model(valid_data, training=True)
                y_init = self.y_init.numpy()[0][0]
                elapsed_time = time.time() - start_time
                training_history.append([step, cost, y_init, loss])

                # 记录训练过程
                self.loss_history.append(loss.numpy())
                self.cost_history.append(cost.numpy())
                self.steps_history.append(step)

                print(
                    f"Scheme: {self.scheme.upper()}, step: {step:5d}, loss: {loss:.4e}, Y0: {y_init:.4e}, cost: {cost:.4e}, elapsed time: {elapsed_time:.1f}")

            self.train_step(self.config.sample(self.batch_size))

        print(f'Scheme: {self.scheme.upper()}, Y0_true: {y_init:.4e}')
        self.training_history = training_history
        return training_history

    @tf.function
    def train_step(self, train_data):
        """Updating the gradients"""
        with tf.GradientTape(persistent=True) as tape:
            loss, cost = self.model(train_data, training=True)
        grad = tape.gradient(loss, self.model.trainable_variables)
        del tape
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))


class EulerWholeNet(tf.keras.Model):
    """Euler离散方法的网络"""

    def __init__(self):
        super(EulerWholeNet, self).__init__()
        self.config = Config()
        self.y_init = tf.Variable(tf.random.normal([1, self.config.dim_y], mean=0, stddev=1, dtype=tf.dtypes.float64))
        self.z_net = FNNet()

    def call(self, dw, training):
        x_init = tf.ones([1, self.config.dim_x], dtype=tf.dtypes.float64) * 0.0
        time_stamp = np.arange(0, self.config.num_time_interval) * self.config.delta_t
        all_one_vec = tf.ones([tf.shape(dw)[0], 1], dtype=tf.dtypes.float64)
        x = tf.matmul(all_one_vec, x_init)
        y = tf.matmul(all_one_vec, self.y_init)
        l = 0.0

        for t in range(0, self.config.num_time_interval):
            data = time_stamp[t], x, y
            z = self.z_net(data, training=training)
            u = self.config.u_fn(time_stamp[t], x, y, z)
            l = l + self.config.f_fn(time_stamp[t], x, u) * self.config.delta_t

            b_ = self.config.b_fn(time_stamp[t], x, y, z)
            sigma_ = self.config.sigma_fn(time_stamp[t], x, y, z)
            f_ = self.config.Hx_fn(time_stamp[t], x, y, z)

            x = x + b_ * self.config.delta_t + sigma_ * dw[:, :, t]
            y = y - f_ * self.config.delta_t + z * dw[:, :, t]

        delta = y + self.config.hx_tf(self.config.total_T, x)
        loss = tf.reduce_mean(tf.reduce_sum(delta ** 2, 1, keepdims=True))
        l = l + self.config.h_fn(self.config.total_T, x)
        cost = tf.reduce_mean(l)
        return loss, cost


class CNWholeNet(tf.keras.Model):
    """Crank-Nicolson离散方法的网络"""

    def __init__(self):
        super(CNWholeNet, self).__init__()
        self.config = Config()
        self.y_init = tf.Variable(tf.random.normal([1, self.config.dim_y], mean=0, stddev=1, dtype=tf.dtypes.float64))
        self.z_net = FNNet()

    def call(self, dw, training):
        config = self.config
        delta_t = config.delta_t
        t_stamps = config.t_stamp
        picard_steps = 3  # Picard迭代次数

        # 初始状态
        x = tf.ones([tf.shape(dw)[0], config.dim_x], dtype=tf.float64) * 0.0
        y = tf.matmul(tf.ones([tf.shape(dw)[0], 1], dtype=tf.float64), self.y_init)
        l = 0.0

        for i in range(config.num_time_interval):
            t_i = t_stamps[i]
            dW = dw[:, :, i]

            z_i = self.z_net((t_i, x, y), training=training)
            u_i = config.u_fn(t_i, x, y, z_i)
            b_i = config.b_fn(t_i, x, y, z_i)
            sigma_i = config.sigma_fn(t_i, x, y, z_i)
            f_i = config.Hx_fn(t_i, x, y, z_i)

            l += config.f_fn(t_i, x, u_i) * delta_t

            # 初始预测 (Euler)
            x_next = x + b_i * delta_t + sigma_i * dW
            y_next = y - f_i * delta_t + z_i * dW

            # Picard迭代
            for _ in range(picard_steps):
                z_next = self.z_net((t_i + delta_t, x_next, y_next), training=training)
                u_next = config.u_fn(t_i + delta_t, x_next, y_next, z_next)
                b_next = config.b_fn(t_i + delta_t, x_next, y_next, z_next)
                sigma_next = config.sigma_fn(t_i + delta_t, x_next, y_next, z_next)
                f_next = config.Hx_fn(t_i + delta_t, x_next, y_next, z_next)

                # CN更新
                x_next = x + 0.5 * (b_i + b_next) * delta_t + 0.5 * (sigma_i + sigma_next) * dW
                y_next = y - 0.5 * (f_i + f_next) * delta_t + 0.5 * (z_i + z_next) * dW

            x, y = x_next, y_next

        delta = y + config.hx_tf(config.total_T, x)
        loss = tf.reduce_mean(tf.reduce_sum(delta ** 2, axis=1, keepdims=True))
        l += config.h_fn(config.total_T, x)
        cost = tf.reduce_mean(l)
        return loss, cost


class FNNet(tf.keras.Model):
    """共享的神经网络结构"""

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
            ) for _ in range(len(num_hiddens) + 2)]

        self.dense_layers = [tf.keras.layers.Dense(num_hiddens[i],
                                                   use_bias=False,
                                                   activation=None)
                             for i in range(len(num_hiddens))]
        self.dense_layers.append(tf.keras.layers.Dense(self.config.dim_z, activation=None))

    def call(self, inputs, training):
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
    """系统配置"""

    def __init__(self):
        self.dim_x = 200
        self.dim_y = 200
        self.dim_z = 200
        self.num_time_interval = 25
        self.total_T = 1.0
        self.delta_t = (self.total_T + 0.0) / self.num_time_interval
        self.sqrth = np.sqrt(self.delta_t)
        self.t_stamp = np.arange(0, self.num_time_interval) * self.delta_t

    def sample(self, num_sample):
        dw_sample = normal.rvs(size=[num_sample, self.dim_x, self.num_time_interval]) * self.sqrth
        return dw_sample

    def f_fn(self, t, x, u):
        return tf.reduce_sum(u ** 2, 1, keepdims=True)

    def h_fn(self, t, x):
        return tf.math.log(0.5 * (1 + tf.reduce_sum(x ** 2, 1, keepdims=True)))

    def b_fn(self, t, x, y, z):
        return 2 * y

    def sigma_fn(self, t, x, y, z):
        return np.sqrt(2)

    def Hx_fn(self, t, x, y, z):
        return 0

    def hx_tf(self, t, x):
        a = 1 + tf.reduce_sum(x ** 2, 1, keepdims=True)
        return 2 * x / a

    def u_fn(self, t, x, y, z):
        return y


def plot_comparison(euler_history, cn_history):
    """绘制两种方法的比较图"""
    # 准备数据
    euler_steps = [x[0] for x in euler_history]
    euler_loss = [x[3] for x in euler_history]
    euler_cost = [x[1] for x in euler_history]

    cn_steps = [x[0] for x in cn_history]
    cn_loss = [x[3] for x in cn_history]
    cn_cost = [x[1] for x in cn_history]

    # 损失函数比较
    plt.figure(figsize=(12, 6))
    plt.plot(euler_steps, euler_loss, label='deep SMP-BSDE', color='blue', linewidth=2)
    plt.plot(cn_steps, cn_loss, label='improved deep SMP-BSDE', color='red', linewidth=2)
    plt.xlabel('Training Steps', fontsize=18)
    plt.ylabel('Loss', fontsize=18)
    plt.title('Loss Comparison ', fontsize=18)
    plt.legend(fontsize=18)
    plt.grid(True)
    plt.savefig('ex2loss_comparison.png', dpi=600, bbox_inches='tight')

    # 代价函数比较
    plt.figure(figsize=(12, 6))
    plt.plot(euler_steps, euler_cost, label='deep SMP-BSDEr', color='blue', linewidth=2)
    plt.plot(cn_steps, cn_cost, label='improved deep SMP-BSDE', color='red', linewidth=2)
    plt.xlabel('Training Steps', fontsize=18)
    plt.ylabel('Cost', fontsize=18)
    plt.title('Cost Comparison ', fontsize=18)
    plt.legend(fontsize=18)
    plt.grid(True)
    plt.savefig('ex2cost_comparison.png', dpi=600, bbox_inches='tight')

    plt.show()

def main():
    # 训练Euler方法
    print("=" * 50)
    print("Training Euler scheme...")
    euler_solver = Solver(scheme='euler')
    euler_history = euler_solver.train()

    # 训练Crank-Nicolson方法
    print("=" * 50)
    print("\nTraining Crank-Nicolson scheme...")
    cn_solver = Solver(scheme='cn')
    cn_history = cn_solver.train()

    # 绘制比较图
    plot_comparison(euler_history, cn_history)

    # 保存结果
    def save_results(history, filename):
        data = np.array(history)
        output = np.zeros((len(data[:, 0]), 4))
        output[:, 0] = data[:, 0]  # step
        output[:, 1] = data[:, 2]  # y_init
        output[:, 2] = data[:, 3]  # loss
        output[:, 3] = data[:, 1]  # cost
        np.savetxt(filename, output, fmt=['%d', '%.5e', '%.5e', '%.5e'], delimiter=',')

    save_results(euler_history, './euler_results.csv')
    save_results(cn_history, './cn_results.csv')

    print("\nAll training completed! Results saved to:")
    print("- euler_results.csv")
    print("- cn_results.csv")
    print("- ex2loss_comparison.png")
    print("- ex2cost_comparison.png")


if __name__ == '__main__':
    main()