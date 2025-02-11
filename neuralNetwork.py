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
# Apply ReLU
l1 = tf.nn.relu(l1)



