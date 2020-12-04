import tensorflow.compat.v1 as tf

def cnn_predictor(input_num, inputs, previous_action, scope):
    with tf.variable_scope(scope):

        # input shape [batch, height, width, channels]
        asset_dim = inputs.get_shape()[1]
        L = inputs.get_shape()[2]    # window length
        N = inputs.get_shape()[3]    # feature

        # filter shape [height, width, channels, number of filters]
        conv1_W = tf.Variable(tf.truncated_normal([1,3,N,3], stddev=0.05))
        layer = tf.nn.conv2d(inputs, filter=conv1_W, padding='VALID', strides=[1, 1, 1, 1])
        norm1 = tf.layers.batch_normalization(layer)
        x = tf.nn.relu(norm1)

        conv2_W = tf.Variable(tf.random_normal([1, L-2, 3, 20], stddev=0.05))
        conv2 = tf.nn.conv2d(x, filter=conv2_W, strides=[1, 1, 1, 1], padding='VALID')
        norm2 = tf.layers.batch_normalization(conv2)
        x = tf.nn.relu(norm2)

        w = tf.reshape(previous_action, [-1, int(asset_dim), 1, 1])
        x = tf.concat([x, w], axis=3)

        conv3_W = tf.Variable(tf.random_normal([1, 1, 21, 1], stddev=0.05))
        conv3 = tf.nn.conv2d(x, filter=conv3_W, strides=[1, 1, 1, 1], padding='VALID')
        norm3 = tf.layers.batch_normalization(conv3)
        net = tf.nn.relu(norm3)

        net = tf.layers.flatten(net)
        w_init = tf.random_uniform_initializer(-0.003, 0.003)
        out = tf.layers.dense(net, asset_dim, activation=tf.nn.softmax, kernel_initializer=w_init)    #[1, 6]

    return out

def rnn_predictor(input_num, inputs, previous_action, scope):
    with tf.variable_scope(scope):

        asset_dim = inputs.get_shape()[1]
        L = inputs.get_shape()[2]  # window length
        N = inputs.get_shape()[3]  # feature

        x=tf.reshape(inputs, shape=[-1, asset_dim, L*N])
        hidden_size = 10

        rnn_cells = []
        for i in range(asset_dim):
            rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)
            rnn_cells.append(rnn_cell)
        cell = tf.nn.rnn_cell.MultiRNNCell(rnn_cells)
        initial_state = cell.zero_state(input_num, tf.float32)
        net, state = tf.nn.dynamic_rnn(cell, x, initial_state=initial_state, dtype=tf.float32)
        # state: [batch_size, hidden_size];  outputs: [batch_size, L, hidden_size]

        net = tf.reshape(net, [-1, int(asset_dim), 1, hidden_size])
        w = tf.reshape(previous_action, [-1, int(asset_dim), 1, 1])
        x = tf.concat([net, w], axis=3)

        x = tf.layers.flatten(x)
        net = tf.layers.dense(x, 256, activation=tf.nn.relu)
        w_init = tf.random_uniform_initializer(-0.003, 0.003)
        out = tf.layers.dense(net, asset_dim, activation=tf.nn.softmax, kernel_initializer=w_init)  # [1, 6]

    return out


class StockActor:

    def __init__(self,sess, asset_dim, window_size, feature_dim, learning_rate, tau, bench_size, nn = "cnn"):
        """
        :param sess:
        :param asset_dim:  non-cash assets + one cash
        :param window_size:  the moving window
        :param feature_dim:  feasure: close price, open price, ...
        :param learning_rate:  for the optimization
        :param tau:  for the target network, tau is for network_param and (1-tau) is for target_network_params
        :param bench_size:  the bench_size of the training
        :param neural: string: "cnn" or "rnn"
        """
        self.sess = sess
        self.asset_dim = asset_dim
        self.window_size = window_size
        self.feasure_dim = feature_dim
        self.learning_rate = learning_rate
        self.tau = tau  # tau: target network update parameter
        self.batch_size = bench_size
        self.nn = nn

        self.scopes=['online/actor','target/actor']
        self.input_num, self.state, self.previous_action, self.out = self.build_actor_network(self.scopes[0], self.nn)
        self.network_params = tf.trainable_variables()

        self.target_input_num, self.target_state, self.target_previous_action, self.target_out = self.build_actor_network(self.scopes[1], self.nn)
        self.target_network_params = tf.trainable_variables()[len(self.network_params):]

        self.update_target_network_params = [self.target_network_params[i].assign(
            tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        # the action gradient will be provided by the critic network, then combine the gradients here
        self.action_gradient= tf.placeholder(tf.float32,[None]+[self.asset_dim])
        self.unnormalized_actor_gradients=tf.gradients(self.out,
                                                       self.network_params,
                                                       -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.math.divide(x, self.batch_size), self.unnormalized_actor_gradients))

        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(
            zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def build_actor_network(self, scope, nn):

        with tf.variable_scope(scope):
            input_num = tf.placeholder(tf.int32, shape=[])
            input_state = tf.placeholder(tf.float32, shape=[None, self.asset_dim,
                                                             self.window_size,
                                                             self.feasure_dim], name= "input")
            previous_action = tf.placeholder(tf.float32, shape=[None, self.asset_dim])

            if nn == "cnn":
                out = cnn_predictor(input_num, input_state, previous_action, scope)
            elif nn == "rnn":
                out = rnn_predictor(input_num, input_state, previous_action, scope)

        return input_num, input_state, previous_action, out


    def train(self,input_num, state, previous_action, action_gradient):
        self.sess.run(self.optimize,feed_dict={self.input_num: input_num,
                                               self.state: state,
                                               self.previous_action: previous_action,
                                               self.action_gradient: action_gradient
                                               })

    def predict(self,input_num, state, previous_action):
        return self.sess.run(self.out,feed_dict={self.input_num: input_num,
                                                 self.state:state,
                                                 self.previous_action:previous_action
                                                 })

    def predict_target(self,input_num, state, previous_action):
        return self.sess.run(self.target_out,feed_dict={self.target_input_num: input_num,
                                                        self.target_state: state,
                                                        self.target_previous_action:previous_action
                                                        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

if __name__=="__main__":
    num_stock = 6
    window_size = 10
    num_feature = 2

    learning_rate = 0.0001
    tau = 0.001
    bench_size = 32
    sesson = tf.Session()
    actor = StockActor(sesson, num_stock, window_size, num_feature, learning_rate, tau, bench_size, "cnn")

    net = actor.build_actor_network("a", "cnn")

