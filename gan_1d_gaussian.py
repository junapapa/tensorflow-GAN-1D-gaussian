"""
    @ file : gan_1d_gaussian.py
    @ brief

    @ author : Younghyun Lee <yhlee109@gmail.com>
    @ date : 2017.12.20
    @ version : 1.0
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# reproducibility
seed = 200
np.random.seed(seed)
tf.set_random_seed(seed)


# Define Classes
class DataDistribution(object):
    """ DataDistribution(mu1, sigma1, mu2, sigma2)

        Distribution of original data : p_data(x) ~ N(mu1, sigma1) or N(mu2, sigma2)
    """
    def __init__(self, mu1, sigma1, mu2, sigma2):
        """
        생성자
        :param mu1: 첫번째 분포의 평균
        :param sigma1: 첫번째 분포의 분산
        :param mu2: 두번째 분포의 평균
        :param sigma2: 두번째 분포의 분산
        """
        self.mu1 = mu1
        self.sigma1 = sigma1
        self.mu2 = mu2
        self.sigma2 = sigma2

    def sampling_data(self, num_samples):
        """
        샘플 데이터 생성 (create sample data as much as 'num_samples')
        :param num_samples: 생성할 데이터 수
        :return: 생성된 데이터 (크기순으로 정렬)
        """
        samples1 = np.random.normal(self.mu1, self.sigma1, num_samples//2)
        samples2 = np.random.normal(self.mu2, self.sigma2, num_samples//2)
        samples = np.concatenate((samples1, samples2))
        samples.sort()      # for stratified sampling
        return samples


class NoiseDistribution(object):
    """ NoiseDistribution(data_range)

        Distribution of noise : p_z(z) ~ U(-range, range) + N(0, 1)*0.01
        for stratified sampling,
            create equally spaced sample data in [-range, range] as much as 'num_samples'
            then, random noise N(0,1)*0.01 is added.
    """
    def __init__(self, data_range):
        """
        생성자
        :param data_range: 랜덤 노이즈 분포 범위
        """
        self.data_range = data_range

    def sampling_data(self, num_samples):
        """
        샘플 데이터 생성(create noise data as much as 'num_samples')
        :param num_samples: 생성할 데이터 수
        :return: 생성된 데이터
        """
        samples_equal = np.linspace(-self.data_range, self.data_range, num_samples)
        samples = samples_equal + np.random.random(num_samples) * 0.01
        return samples


# Define functions
def generator(x, num_hidden_layers):
    """
    Generator G(z)
    :param x: 입력 벡터 변수
    :param num_hidden_layers: 히든 레이어 수
    :return: 생성자 결과값 (스칼라)
    """
    # 1st hidden layer
    w0 = tf.get_variable('w0',
                         [x.get_shape()[1], num_hidden_layers],
                         initializer=tf.truncated_normal_initializer(stddev=2.0)
                         )
    b0 = tf.get_variable('b0',
                         [num_hidden_layers],
                         initializer=tf.constant_initializer(0.0)
                         )
    h0 = tf.nn.relu(tf.matmul(x, w0) + b0)

    # output layer
    w1 = tf.get_variable('w1',
                         [h0.get_shape()[1], 1],
                         initializer=tf.truncated_normal_initializer(stddev=2.0)
                         )
    b1 = tf.get_variable('b1',
                         [1],
                         initializer=tf.constant_initializer(0.0)
                         )
    hypothesis = tf.matmul(h0, w1) + b1

    return hypothesis


def discriminator(x, num_hidden_layers):
    """
    Discriminator D(x)
    :param x: 입력 벡터 변수
    :param num_hidden_layers: 히든 레이어 수
    :return: 확률값(0~1) - 1에 가까울수록 real, 0에 가까울수록 fake
    """
    # 1st hidden layer
    w0 = tf.get_variable('w0',
                         [x.get_shape()[1], num_hidden_layers],
                         initializer=tf.contrib.layers.variance_scaling_initializer()
                         )
    b0 = tf.get_variable('b0',
                         [num_hidden_layers],
                         initializer=tf.constant_initializer(0.0)
                         )
    h0 = tf.nn.relu(tf.matmul(x, w0) + b0)

    # output layer
    w1 = tf.get_variable('w1',
                         [h0.get_shape()[1], 1],
                         initializer=tf.contrib.layers.variance_scaling_initializer()
                         )
    b1 = tf.get_variable('b1',
                         [1],
                         initializer=tf.constant_initializer(0.0)
                         )
    hypothesis = tf.sigmoid(tf.matmul(h0, w1) + b1)

    return hypothesis


def optimizer(loss, var_list, num_decay_steps=400, initial_learning_rate=0.03):
    """
    최적화 도구(exponential_decay) - Gradient Descent
    :param loss: 비용 함수
    :param var_list: 최적화 대상 변수 리스트
    :param num_decay_steps: decay step 횟수
    :param initial_learning_rate: 초기 학습 비율
    :return: 최적화 도구
    """
    decay = 0.95
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        initial_learning_rate,
        batch,
        num_decay_steps,
        decay,
        staircase=True
    )
    opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        loss,
        global_step=batch,
        var_list=var_list
    )
    return opt


class GAN_Model(object):
    """ GAN_Model(sess, name, num_hidden_layers, learning_rate)

        Build GAN graph
    """
    def __init__(self, sess, name, num_hidden_layers=32, learning_rate=0.03):
        """
        생성자
        :param sess: 세션
        :param name: 이름
        :param num_hidden_layers: 생성자 및 구별자의 히든 레이어 수
        :param learning_rate: 학습 비율
        """
        self.sess = sess
        self.name = name

        # pre-trained discriminator
        with tf.variable_scope('D_pre_train'):
            self.x_pre = tf.placeholder(tf.float32, shape=(None, 1))
            self.y_pre = tf.placeholder(tf.float32, shape=(None, 1))
            self.D_pre = discriminator(self.x_pre, num_hidden_layers)

        # generator
        with tf.variable_scope('Generator'):
            self.z = tf.placeholder(tf.float32, shape=(None, 1))
            self.G = generator(self.z, num_hidden_layers)

        # discriminator
        with tf.variable_scope('Discriminator') as scope:
            self.x = tf.placeholder(tf.float32, shape=(None, 1))
            self.D1 = discriminator(self.x, num_hidden_layers)
            scope.reuse_variables()
            self.D2 = discriminator(self.G, num_hidden_layers)

        # Define the loss for discriminator and generator
        eps = 0.0001  # to prevent log(0)
        self.loss_pre = tf.reduce_mean(tf.square(self.D_pre - self.y_pre))
        self.loss_d = tf.reduce_mean(-tf.log(self.D1 + eps) - tf.log(1 - self.D2 + eps))
        self.loss_g = tf.reduce_mean(-tf.log(self.D2 + eps))

        # trainable parameters
        self.params_pre = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D_pre_train')
        self.params_d = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')
        self.params_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')

        # optimizer
        self.opt_pre = optimizer(self.loss_pre, self.params_pre, 400, learning_rate)
        self.opt_d = optimizer(self.loss_d, self.params_d, 400, learning_rate)
        self.opt_g = optimizer(self.loss_g, self.params_g, 400, learning_rate/2)

    def pre_train(self, x_data, y_data):
        """
        GAN 학습 전 구별자를 선행 학습 (gradient descent 1 step)
        :param x_data: data
        :param y_data: label
        :return: loss value
        """
        loss_pre, _ = self.sess.run([self.loss_pre, self.opt_pre], feed_dict={self.x_pre: x_data, self.y_pre: y_data})
        return loss_pre

    def get_pre_trained_weights(self):
        """
        선행 학습된 구별자의 파라메터 가져오기
        :return: 선행 학습된 구별자의 파라메터
        """
        return self.sess.run(self.params_pre)

    def update_discriminator(self, x_data, z_data):
        """
        GAN 구별자 학습 (gradient descent 1 step)
        :param x_data: input data for D(x)
        :param z_data: noise data for G(z)
        :return: loss value of discriminator
        """
        loss_d, _ = self.sess.run([self.loss_d, self.opt_d], feed_dict={self.x: x_data, self.z: z_data})
        return loss_d

    def update_generator(self, z_data):
        """
        GAN 생성자 학습 (gradient descent 1 step)
        :param z_data: noise data for G(z)
        :return: loss value of generator
        """
        loss_g, _ = self.sess.run([self.loss_g, self.opt_g], feed_dict={self.z: z_data})
        return loss_g


def run_pre_train(model, real_data, num_steps):
    """
    구별자 선행 학습 실행
    :param model: GAN model
    :param real_data: real data class
    :param num_steps: number of iteration
    :return: pre-trained weights of D(x)
    """

    num_samples = 1000
    num_hist_bins = 100

    print('\n===== Start : pre-training =====\n')

    for step in range(num_steps):
        samples = real_data.sampling_data(num_samples)
        hist_samples, edges = np.histogram(samples, bins=num_hist_bins, density=True)

        max_val = np.max(hist_samples)
        min_val = np.min(hist_samples)
        labels = (hist_samples - min_val) / (max_val - min_val)
        data = edges[1:]

        x_data = np.reshape(data, (num_hist_bins, 1))
        y_data = np.reshape(labels, (num_hist_bins, 1))

        loss_pre = model.pre_train(x_data, y_data)

        if step % 100 == 0:
            print('pre-training step : %d/%d (loss: %f)' % (step, num_steps, loss_pre))

    print('\n===== Finish : pre-training =====\n')

    return model.get_pre_trained_weights()


def run_train(model, real_data, noise_data, num_steps, batch_size):
    """
    GAN 학습 실행
    :param model: GAN model
    :param real_data: real data class (p_data)
    :param noise_data: nose data class (p_z)
    :param num_steps: number of epoch
    :param batch_size: size of batch data
    """

    print('\n===== Start : GAN training =====\n')

    # training-loop
    for step in range(num_steps):

        # update discriminator
        x = real_data.sampling_data(batch_size)
        z = noise_data.sampling_data(batch_size)

        x_data = np.reshape(x, (batch_size, 1))
        z_data = np.reshape(z, (batch_size, 1))

        loss_d = model.update_discriminator(x_data, z_data)

        # update generator
        z = noise_data.sampling_data(batch_size)

        z_data = np.reshape(z, (batch_size, 1))

        loss_g = model.update_generator(z_data)

        if step % 100 == 0:
            print('[%d/%d]: loss_d : %.3f, loss_g : %.3f' % (step, num_steps, loss_d, loss_g))

    print('\n===== Finish : GAN training =====\n')


class Display(object):
    """ Display(num_points, num_bins, mu1, sigma1, mu2, sigma2, data_range)

    """
    def __init__(self, num_points, num_bins, mu1, sigma1, mu2, sigma2, data_range):
        """
        생성자
        :param num_points: x 축 표현 데이터 수
        :param num_bins:
        :param mu1: 첫번째 분포 평균
        :param sigma1: 첫번째 분포 분산
        :param mu2: 두번째 분포 평균
        :param sigma2: 두번째 분포 분산
        :param data_range: 데이터 분포 범위
        """
        self.num_points = num_points
        self.num_bins = num_bins
        self.mu = (mu1 + mu2) / 2.0
        self.sigma = max(sigma1, sigma2)
        self.data_range = data_range
        self.xs = np.linspace(-data_range, data_range, num_points)
        self.bins = np.linspace(-data_range, data_range, num_bins)

    def draw_results(self, db_init, db_pre_trained, db_trained, pd, pg):
        """
        결과 출력
        :param db_init: initial decision boundary
        :param db_pre_trained: pre-trained decision boundary
        :param db_trained: trained decision boundary
        :param pd: distribution of real data
        :param pg: distribution of fake data
        """
        db_x = np.linspace(-self.data_range, self.data_range, len(db_trained))
        p_x = np.linspace(-self.data_range, self.data_range, len(pd))
        f, ax = plt.subplots(1)
        ax.plot(db_x, db_init, 'g--', linewidth=2, label='db_init')
        ax.plot(db_x, db_pre_trained, 'c--', linewidth=2, label='db_pre_trained')
        ax.plot(db_x, db_trained, 'g-', linewidth=2, label='db_trained')
        ax.set_ylim(0, max(1, np.max(pd) * 1.1))
        ax.set_xlim(max(self.mu - self.sigma * 3, -self.data_range * 0.9),
                    min(self.mu + self.sigma * 3, self.data_range * 0.9))
        plt.plot(p_x, pd, 'b-', linewidth=2, label='real data')
        plt.plot(p_x, pg, 'r-', linewidth=2, label='generated data')
        plt.title('1D Generative Adversarial Network: ' + '(mu : %3g,' % self.mu + ' sigma : %3g)' % self.sigma)
        plt.xlabel('Data values')
        plt.ylabel('Probability density')
        plt.legend()
        plt.grid(True)

        plt.show()


def main():
    # parameters
    mu1 = -2.0
    sigma1 = 1.0
    mu2 = 2.0
    sigma2 = 1.0

    data_range = 5

    num_hidden_layers = 32
    learning_rate = 0.03
    batch_size = 150

    # data source
    real_data = DataDistribution(mu1, sigma1, mu2, sigma2)
    noise_data = NoiseDistribution(data_range)

    # display option
    num_points = 10000
    num_bins = 20
    display = Display(num_points=num_points,
                      num_bins=num_bins,
                      mu1=mu1,
                      sigma1=sigma1,
                      mu2=mu2,
                      sigma2=sigma2,
                      data_range=data_range)

    # initialization
    sess = tf.Session()
    gan = GAN_Model(sess=sess,
                    name='gan',
                    num_hidden_layers=num_hidden_layers,
                    learning_rate=learning_rate,
                    )

    sess.run(tf.global_variables_initializer())

    # plot : initial decision boundary
    db_init = np.zeros((num_points, 1))
    for i in range(num_points // batch_size):
        db_init[batch_size * i: batch_size * (i + 1)] = \
            sess.run(gan.D1,
                     feed_dict={gan.x: np.reshape(display.xs[batch_size * i:batch_size * (i + 1)], (batch_size, 1))})

    # pre-training discriminator
    num_pre_steps = 1000
    weights_pre = run_pre_train(gan, real_data, num_pre_steps)

    # copy weights from pre-training over to new D network
    sess.run(tf.global_variables_initializer())
    for i, v in enumerate(gan.params_d):
        sess.run(v.assign(weights_pre[i]))

    # plot : pre-trained decision boundary
    db_pre_trained = np.zeros((num_points, 1))
    for i in range(num_points // batch_size):
        db_pre_trained[batch_size * i: batch_size * (i + 1)] = \
            sess.run(gan.D1,
                     feed_dict={gan.x: np.reshape(display.xs[batch_size * i:batch_size * (i + 1)], (batch_size, 1))})

    # training
    num_train_steps = 3000
    run_train(gan, real_data, noise_data, num_train_steps, batch_size)

    # plot : trained decision boundary
    db_trained = np.zeros((num_points, 1))
    for i in range(num_points // batch_size):
        db_trained[batch_size * i: batch_size * (i + 1)] = \
            sess.run(gan.D1,
                     feed_dict={gan.x: np.reshape(display.xs[batch_size * i:batch_size * (i + 1)], (batch_size, 1))})

    # plot : pdf of data distribution
    x_samples = real_data.sampling_data(num_points)
    x_pdf, _ = np.histogram(x_samples, bins=num_bins, density=True)

    # plot : pdf of generated samples
    zs = np.linspace(-data_range, data_range, num_points)
    g_samples = np.zeros((num_points, 1))
    for i in range(num_points // batch_size):
        g_samples[batch_size * i: batch_size * (i + 1)] = \
            sess.run(gan.G,
                     feed_dict={gan.z: np.reshape(zs[batch_size * i:batch_size * (i + 1)], (batch_size, 1))})
    g_pdf, _ = np.histogram(g_samples, bins=num_bins, density=True)

    display.draw_results(db_init, db_pre_trained, db_trained, x_pdf, g_pdf)


if __name__ == '__main__':
    main()

