import tensorflow as tf

x = tf.constant(1, name='x')
fileID = tf.placeholder(tf.string, [None, 1], name='FileID')
y = tf.constant(2, name='y')
op_add = tf.add(x, y, name='add_x_y')

with tf.Session() as sess:
    writer = tf.summary.FileWriter('logs/', sess.graph)
    sess.run(tf.global_variables_initializer())
    print(sess.run(op_add))
