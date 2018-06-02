
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd



# In[2]:


from sklearn.preprocessing import StandardScaler


# In[3]:


loan_data = pd.read_csv('input.csv')


# In[11]:


scaler = StandardScaler()
scaler.fit(loan_data)


# In[12]:


scaled_data = scaler.transform(loan_data)


# In[13]:


scaled_data


# ## Loading the inputs and labels

# In[14]:


input_matrix = scaled_data.astype(np.float32)


# In[15]:


input_matrix.shape


# In[16]:


type(input_matrix[0][0])


# In[35]:


labels = np.loadtxt("output.csv", delimiter=",", ndmin=2).astype(np.float32)[1:9578,:]


# In[36]:


labels.shape


# In[37]:


type(labels[0][0])


# In[39]:


import tensorflow as tf


# In[40]:


from math import floor,ceil


# ## Splitting data into training and testing set

# In[41]:


train_size = 0.75

train_cnt = floor(input_matrix.shape[0] * train_size)
x_train = input_matrix[0:train_cnt]
y_train = labels[0:train_cnt]
x_test = input_matrix[train_cnt:]
y_test = labels[train_cnt:]


# In[42]:


x_train.shape


# In[43]:


y_train.shape


# In[44]:


x_test.shape


# In[45]:


y_test


# In[46]:


y_test.shape


# ## setting up the network parameters

# In[47]:


learning_rate = 0.01
training_epochs = 5000
batch_size = 30


# In[49]:


n_classes = labels.shape[1]
n_samples = 9578
n_inputs = input_matrix.shape[1]


# In[50]:


n_classes


# In[51]:


n_samples


# In[115]:


n_inputs


# In[52]:


n_hidden_1 = 20
n_hidden_2 = 20


# ## Defining the multilayer perceptron function

# In[53]:


def multilayer_network(X,weights,biases,keep_prob):
    '''
    X: Placeholder for data inputs
    weights: dictionary of weights
    biases: dictionary of bias values
    
    '''
    #first hidden layer with sigmoid activation
    # sigmoid(X*W+b)
    
    with tf.name_scope('layer1'):
        layer_1 = tf.add(tf.matmul(X,weights['h1']),biases['h1'])
        layer_1 = tf.nn.sigmoid(layer_1)
        layer_1 = tf.nn.dropout(layer_1,keep_prob)
    
    #second hidden layer
    with tf.name_scope('layer2'):
        layer_2 = tf.add(tf.matmul(layer_1,weights['h2']),biases['h2'])
        layer_2 = tf.nn.sigmoid(layer_2)
        layer_2 = tf.nn.dropout(layer_2,keep_prob)
    
    #output layer
    with tf.name_scope('output_layer'):
        out_layer = tf.matmul(layer_2,weights['out']) + biases['out']
        tf.summary.histogram("output_for_the_layer",out_layer)
    
    return out_layer


# ### Defining the values for weights and biases

# In[54]:


#defining the weights and biases dictionary

with tf.name_scope("weights"):
    weights = {
        'h1': tf.Variable(tf.random_normal([n_inputs,n_hidden_1]),name='W_input'),
        'h2': tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2]),name='W_layer1'),
        'out': tf.Variable(tf.random_normal([n_hidden_2,n_classes]),name='W_layer2')
    }
    tf.summary.histogram("weights1",weights['h1'])
    tf.summary.histogram("weights2",weights['h2'])
    tf.summary.histogram("weights_out",weights['out'])
with tf.name_scope("biases"):
    biases = {
        'h1': tf.Variable(tf.random_normal([n_hidden_1]),name='b_input'),
        'h2': tf.Variable(tf.random_normal([n_hidden_2]),name='b_layer1'),
        'out': tf.Variable(tf.random_normal([n_classes]),name='b_layer2')
    }
    tf.summary.histogram("bias_input",biases['h1'])
    tf.summary.histogram("bias_layer1",biases['h2'])
    tf.summary.histogram("bias_layer2",biases['out'])
keep_prob = tf.placeholder("float")


# ### Defining the tensor objects for inputs and labels

# In[55]:


with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32,[None,n_inputs],name='x_inputs')
    Y = tf.placeholder(tf.float32,[None,n_classes],name='y_inputs')


# In[56]:


#obtaining predictions of the model
predictions = multilayer_network(X,weights,biases,keep_prob)


# In[57]:


#cost function(loss) and optimizer function
with tf.name_scope('loss'):
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=predictions,labels=Y))
    tf.summary.scalar('loss',cost)

with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar("Accuracy",accuracy)


# ## Training the model (backpropagation)

# In[58]:



#initializing all variables
init = tf.global_variables_initializer()


# In[59]:


with tf.Session() as sess:
    sess.run(init)
    
    
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("logs/",sess.graph)

    #for loop

    for epoch in range(training_epochs):
        avg_cost = 0.0
        total_batch = int(len(x_train) / batch_size)
        x_batches = np.array_split(x_train, total_batch)
        y_batches = np.array_split(y_train, total_batch)
        for i in range(total_batch):
            batch_x, batch_y = x_batches[i], y_batches[i]
            _,c,p = sess.run([optimizer,cost,predictions], 
                            feed_dict={
                                X: batch_x, 
                                Y: batch_y, 
                                keep_prob: 0.8
                            })
            avg_cost += c / total_batch
            
        print("Epoch:", '%04d' % (epoch+1), "cost=", 
                "{:.9f}".format(avg_cost))
        
        acc,res = sess.run([accuracy,merged], feed_dict={X: x_test, Y: y_test,keep_prob:1.0})
        writer.add_summary(res,epoch)
        print('Accuracy:', acc)
        print ('---------------')
        
    print("Model has completed {} epochs of training".format(training_epochs))
    


# In[37]:


import tensorboard as tb

