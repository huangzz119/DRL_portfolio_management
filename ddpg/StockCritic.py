import tensorflow.compat.v1 as tf

def cnn_predictor(input_num, inputs, actions, previous_action, scope):
    """
    the cnn predictor to give the corresponding q value of the value function  
    """
    with tf.variable_scope(scope):

        asset_dim = inputs.get_shape()[1]
        L = inputs.get_shape()[2]  # window length
        N = inputs.get_shape()[3]  # feature

        # filter shape [height, width, channels, number of filters]
        conv1_W = tf.Variable(tf.truncated_normal([1, 3, N, 3], stddev=0.05))  # eg: [?, 10, 2, 32]
        layer = tf.nn.conv2d(inputs, filter=conv1_W, padding='VALID', strides=[1, 1, 1, 1])  # result: [?, 6, 1, 32]
        norm1 = tf.layers.batch_normalization(layer)
        x = tf.nn.relu(norm1)

        conv2_W = tf.Variable(tf.random_normal([1, L - 2, 3, 20], stddev=0.05))
        conv2 = tf.nn.conv2d(x, filter=conv2_W, strides=[1, 1, 1, 1], padding='VALID')  # [1, 6, 1, 20]
        norm2 = tf.layers.batch_normalization(conv2)
        x = tf.nn.relu(norm2)

        previous_w = tf.reshape(previous_action, [-1, int(asset_dim), 1, 1])
        x = tf.concat([x, previous_w], axis=3)
        w = tf.reshape(actions, [-1, int(asset_dim), 1, 1])
        x = tf.concat([x, w], axis=3)

        conv3_W = tf.Variable(tf.random_normal([1, 1, 22, 1], stddev=0.05))
        conv3 = tf.nn.conv2d(x, filter=conv3_W, strides=[1, 1, 1, 1], padding='VALID')
        norm3 = tf.layers.batch_normalization(conv3)
        net = tf.nn.relu(norm3)

        net = tf.layers.flatten(net)
        out = tf.layers.dense(net, 1, kernel_initializer=tf.random_uniform_initializer(-0.003, 0.003))

    return out

def rnn_predictor(input_num, inputs, actions, previous_action, scope):
    """
    the rnn predictor to give the corresponding q value of the value function  
    """
    with tf.variable_scope(scope):

        asset_dim = inputs.get_shape()[1]
        L = inputs.get_shape()[2]  # window length
        N = inputs.get_shape()[3]  # feature

        x=tf.reshape(inputs, shape=[-1, asset_dim, L*N])
        hidden_size = 10

        rnn_cells = []
        for i in range(asset_dim):
            rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)   ## create a BasicRNNCell
            rnn_cells.append(rnn_cell)
        cell = tf.nn.rnn_cell.MultiRNNCell(rnn_cells)
        initial_state = cell.zero_state(input_num, tf.float32)
        net, state = tf.nn.dynamic_rnn(cell, x, initial_state=initial_state, dtype=tf.float32)
        # state: [batch_size, hidden_size];  outputs: [batch_size, L, hidden_size]

        x = tf.reshape(net, [-1, int(asset_dim), 1, hidden_size])
        previous_w = tf.reshape(previous_action, [-1, int(asset_dim), 1, 1])
        x = tf.concat([x, previous_w], axis=3)
        w = tf.reshape(actions, [-1, int(asset_dim), 1, 1])
        net = tf.concat([x, w], axis=3)

        net = tf.layers.flatten(net)
        net = tf.layers.dense(net, 64,  activation=tf.nn.relu)
        out = tf.layers.dense(net, 1, kernel_initializer=tf.random_uniform_initializer(-0.01, 0.01))

    return out


class StockCritic:
    def __init__(self,sess, asset_dim, window_size, feature_dim, learning_rate, tau, gamma, num_actor_vars, nn = "cnn"):
        self.sess = sess
        self.asset_dim = asset_dim
        self.window_size = window_size
        self.feature_dim = feature_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma
        self.nn = nn

        self.scopes=['rnn/online/critic','rnn/target/critic']
        self.inputs_num, self.state, self.previous_action, self.action, self.out = self.build_critic_network(self.scopes[0], self.nn)
        self.network_params = tf.trainable_variables()[num_actor_vars:]

        self.target_inputs_num, self.target_state, self.target_previous_action, self.target_action, self.target_out = self.build_critic_network(self.scopes[1], self.nn)
        self.target_network_params = tf.trainable_variables()[(len(self.network_params)+num_actor_vars): ]

        self.update_target_network_params = [self.target_network_params[i].assign(
            tf.multiply(self.network_params[i], self.tau) + tf.multiply(self.target_network_params[i], 1. - self.tau))
             for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])
        self.loss = tf.losses.mean_squared_error(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action
        self.action_grads = tf.gradients(self.out, self.action)

    def build_critic_network(self, scope, nn):
        with tf.variable_scope(scope):
            input_num = tf.placeholder(tf.int32, shape=[])
            inputs = tf.placeholder(tf.float32, shape=[None, self.asset_dim,
                                                       self.window_size,
                                                       self.feature_dim])
            previous_action = tf.placeholder(tf.float32, shape=[None, self.asset_dim])
            action = tf.placeholder(tf.float32, shape=[None, self.asset_dim])

            if nn == "cnn":
                out = cnn_predictor(input_num, inputs, previous_action, action, scope)
            elif nn == "rnn":
                out = rnn_predictor(input_num, inputs, previous_action, action, scope)

        return input_num, inputs, previous_action, action, out

    def train(self, input_num, inputs, previous_action, actions, predicted_q_value):
        critic_loss,q_value,_= self.sess.run([self.loss,self.out,self.optimize],
                                            feed_dict={self.inputs_num: input_num,
                                                       self.state: inputs,
                                                       self.previous_action:previous_action,
                                                       self.action: actions,
                                                       self.predicted_q_value: predicted_q_value})
        return critic_loss,q_value

    def predict(self,input_num, inputs, previous_action, actions):
        return self.sess.run(self.out,feed_dict={self.input_num: input_num,
                                                 self.state:inputs,
                                                 self.previous_action: previous_action,
                                                 self.actions:actions})

    def predict_target(self,input_num, inputs,previous_action, actions):
        return self.sess.run(self.target_out,feed_dict={self.target_inputs_num: input_num,
                                                        self.target_state: inputs,
                                                        self.target_previous_action: previous_action,
                                                        self.target_action: actions})

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def action_gradients(self, input_num, inputs,previous_action, actions):
        return self.sess.run(self.action_grads,feed_dict={self.inputs_num:input_num,
                                                          self.state:inputs,
                                                          self.previous_action:previous_action,
                                                          self.action:actions})


