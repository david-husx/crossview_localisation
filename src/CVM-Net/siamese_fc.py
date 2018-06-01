import tensorflow as tf


class Siamese_FC:

    def fc_layer(self, x, input_dim, output_dim, init_dev, init_bias,
                 trainable, name='fc_layer', activation_fn=tf.nn.relu):
        with tf.variable_scope(name):
            weight = tf.get_variable(name='weights', shape=[input_dim, output_dim],
                                     trainable=trainable,
                                     initializer=tf.truncated_normal_initializer(mean=0.0, stddev=init_dev))
            bias = tf.get_variable(name='biases', shape=[output_dim],
                                   trainable=trainable, initializer=tf.constant_initializer(init_bias))

            if activation_fn is not None:
                out = tf.nn.xw_plus_b(x, weight, bias)
                out = activation_fn(out)
            else:
                out = tf.nn.xw_plus_b(x, weight, bias)

        return out


    def siamese_fc(self, x_sat, x_grd, trainable, scope_name):
        print('Siamese_FC:', scope_name, ' trainable =', trainable)

        with tf.variable_scope(scope_name) as scope:
            fc_sat = self.fc_layer(x_sat, 64*512, 4096, 0.005, 0.1, trainable, 'fc1', activation_fn=None)
            sat_global = tf.nn.l2_normalize(fc_sat, dim=1)

            scope.reuse_variables()

            fc_grd = self.fc_layer(x_grd, 64*512, 4096, 0.005, 0.1, trainable, 'fc1', activation_fn=None)
            grd_global = tf.nn.l2_normalize(fc_grd, dim=1)

        return sat_global, grd_global
