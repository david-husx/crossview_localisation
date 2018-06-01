import tensorflow as tf


class TransNet:

    def conv_layer(self, x, kernel_dim, in_dimen, out_dimen, trainable, name='conv', activation_fn=tf.nn.relu):
        with tf.variable_scope(name):
            weight = tf.get_variable(name='weights', shape=[kernel_dim, kernel_dim, in_dimen, out_dimen],
                                     trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable(name='biases', shape=[out_dimen],
                                   trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())

            y = tf.nn.bias_add(tf.nn.conv2d(x, weight, strides=[1, 1, 1, 1], padding='SAME'), bias)
            if activation_fn is not None:
                y = activation_fn(y)

        return y


    def conv_layer_batch(self, x, kernel_dim, in_dimen, out_dimen, trainable, name='conv', activation_fn=tf.nn.relu):
        with tf.variable_scope(name):
            weight = tf.get_variable(name='weights', shape=[kernel_dim, kernel_dim, in_dimen, out_dimen],
                                     trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable(name='biases', shape=[out_dimen],
                                   trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())

            y = tf.nn.bias_add(tf.nn.conv2d(x, weight, strides=[1, 1, 1, 1], padding='SAME'), bias)
            if activation_fn is not None:
                y = activation_fn(y)

        return y


    def cnn_network(self, x, keep_prob, trainable):
        trans_x = self.conv_layer(x, 3, 512, 512, trainable, 'trans_conv1')
        trans_x = tf.nn.dropout(trans_x, keep_prob, name='trans_conv1_dropout')
        trans_x = self.conv_layer(trans_x, 1, 512, 512, trainable, 'trans_conv2')
        trans_x = tf.nn.dropout(trans_x, keep_prob, name='trans_conv2_dropout')
        return trans_x


    def transform(self, x_sat, x_grd, keep_prob, trainable, name):
        print('TransNet: trainable =', trainable)

        with tf.variable_scope(name + '_trans'):
            trans_sat = self.conv_layer_batch(x_sat, 1, 512, 512, trainable, 'sat_trans_conv1')
            trans_sat = tf.nn.dropout(trans_sat, keep_prob, name='sat_trans_conv1_dropout')
            trans_grd = self.conv_layer_batch(x_grd, 1, 512, 512, trainable, 'grd_trans_conv1')
            trans_grd = tf.nn.dropout(trans_grd, keep_prob, name='grd_trans_conv1_dropout')

        with tf.variable_scope(name + '_trans_shared') as scope:
            trans_sat = self.cnn_network(trans_sat, keep_prob, trainable)
            scope.reuse_variables()
            trans_grd = self.cnn_network(trans_grd, keep_prob, trainable)

        return trans_sat, trans_grd


