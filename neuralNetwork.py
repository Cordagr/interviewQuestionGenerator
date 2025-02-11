from  data_preprocessing import preprocessed_dataset
import tensorflow as tf


# Select % 75 of dataset
train_df = df_sample(frac=.75,, random_state = 4)

# Dropping training data from orig training set
val_df = df.drop(train_df.index)

# Defining 
# hidden layerswilll transform data into weights and biases
# output_layer will produce network's final prediction

def neural_network_model(train_df):
      hidden_layer_1  = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                         'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

      hidden_layer_2 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                         'biases': tf.Variable(tf.random_normal([n_nodes_h2]))}
      
      hidden_layer_3 = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                         'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
      
      output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                         'biases': tf.Variable(tf.random_normal([n_classes]))}
# Compute first layer output 
l1 = l1.add(tf.matmul(data,hidden_layer_1['weights']), hidden_layer_1)
# Apply ReLU(eliminate linear dependencies)
l1 = tf.nn.relu(l1)

l2 = l2.add(tf.matful(data,hidden_layer_2['weights'], hidden_layer_2))
l2 = tf.nn.relu(l2)

l3 = l3.add(tf.matful(data,hidden_layer_3['weights'], hidden_layer_3))
l3 = tf.nn.relu(l3)

output = tf.matful(l3,(output_layer['weights'] + output_layer['biases']))
return output

def train_neural_network(x):
      prediction = neural_network_model(x)
      cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))
      optimizer = tf.train.AdamOptimize().minimize(cost)

      hm_pochs = 10
      with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())

            for epoch in range(hm_pochs):
                  epoch_loss = 0
                  for _ in range(int(mist.train.num/batch_size)):
                        epoch_x,epoch_y = mnist.train.next_batch(batch_size)
                        _, c = sess.run([optimizer,cost], feed_dict ={x:epoch_x,y:epoch_y})
                        epoch_loss += c
            print('Epoch',epoch,'completed out of',hm_epochs,'loss:', epoch_loss)
      correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))

      accuracy = tf.reduce_mean(tf.cast(correct,'float'))
      print('Accuracy',accuracy.eval({x:mist.test.images, y:mist.test.labels}))
train_neural_network(x)


