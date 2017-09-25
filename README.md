# A voir et à savoir : 

#### Le LSTM est un réseau de neurones récurrents (RNN) qui permet de surmonter le problème du "vanishing" gradient dans la phase d'entraînement du réseau de neurone. Le LSTM est un RNN qui a de la mémoire! 

* Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM): 
* A friendly introduction to Recurrent Neural Networks: https://youtu.be/UNmqTiOnRfg
* Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM): https://youtu.be/WCUNPb-5EYI

### Il est composé de trois gates : input gate, forget gate, output gate. Le LSTM est utilisé dans la cas des traduction, des analyses de sentiments et la génaration de textes. 
* TensorFlow and Deep Learning without a PhD, Part 2 (Google Cloud Next '17):  https://youtu.be/fTUwdXUFfI8

### LSTM Modèle 

```
def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']    
```




