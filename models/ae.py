import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, ReLU
from tensorflow.math import exp, sqrt, square
from models.tf_basemodel import tf_BasedModel


def reparametrize(mu, logvar):
    """
    Differentiably sample random Gaussian data with specified mean and variance using the
    reparameterization trick.

    Suppose we want to sample a random number z from a Gaussian distribution with mean mu and
    standard deviation sigma, such that we can backpropagate from the z back to mu and sigma.
    We can achieve this by first sampling a random value epsilon from a standard Gaussian
    distribution with zero mean and unit variance, then setting z = sigma * epsilon + mu.

    For more stable training when integrating this function into a neural network, it helps to
    pass this function the log of the variance of the distribution from which to sample, rather
    than specifying the standard deviation directly.

    Inputs:
    - mu: Tensor of shape (N, Z) giving means
    - logvar: Tensor of shape (N, Z) giving log-variances

    Returns:
    - z: Estimated latent vectors, where z[i, j] is a random value sampled from a Gaussian with
         mean mu[i, j] and log-variance logvar[i, j].
    """
    ################################################################################################
    # TODO: Reparametrize by initializing epsilon as a normal distribution and scaling by          #
    # posterior mu and sigma to estimate z                                                         #
    ################################################################################################
    # Replace "pass" statement with your code
    sigma = sqrt(exp(logvar))
    epsilon = tf.random.normal(sigma.shape)
    z = sigma * epsilon + mu
    ################################################################################################
    #                              END OF YOUR CODE                                                #
    ################################################################################################
    return z

class AutoEncoder(tf_BasedModel):
    def __init__(
        self,
        meta_data,
        batch_sz,
        hidden_size=100,
        num_directions=2,
        num_layers=1,
        window_size=None,
        use_attention=False,
        embedding_dim=16,
        model_save_path="./ae_models",
        feature_type="sequentials",
        label_type="next_log",
        eval_type="session",
        topk=5,
        use_tfidf=False,
        freeze=False,
        gpu=-1,
        **kwargs
    ):
        super().__init__(
            meta_data=meta_data,
            batch_sz=batch_sz,
            model_save_path=model_save_path,
            feature_type=feature_type,
            label_type=label_type,
            eval_type=eval_type,
            topk=topk,
            use_tfidf=use_tfidf,
            embedding_dim=embedding_dim,
            freeze=freeze,
            gpu=gpu,
            **kwargs
        )
        self.num_labels = meta_data["num_labels"]
        self.num_layers = num_layers
        self.feature_type = feature_type
        self.label_type = label_type
        self.hidden_size = hidden_size
        self.num_directions = num_directions
        self.use_tfidf = use_tfidf
        self.embedding_dim = embedding_dim
        rnn_cells = [tf.keras.layers.LSTMCell(self.hidden_size) for _ in range(self.num_layers)]
        stacked_lstm = tf.keras.layers.StackedRNNCells(rnn_cells)
        self.lstm = tf.keras.layers.RNN(stacked_lstm)

        self.linear1 = Dense(
            self.hidden_size // 2, activation = 'relu')
        self.linear2 = Dense(self.hidden_size * self.num_directions, activation = None)
        self.linear3 = Dense(
            self.num_labels, activation=None,
        )


    def call(self, input_dict):
        if self.label_type == "anomaly":
            y = tf.convert_to_tensor(input_dict["window_anomalies"])
        elif self.label_type == "next_log":
            y = tf.convert_to_tensor(input_dict["window_labels"])

        # y = tf.one_hot(y, self.num_labels)
        # y = tf.squeeze(y)

        x = tf.convert_to_tensor(input_dict["features"])
        x = tf.nn.embedding_lookup(self.embedding_matrix, x)
        outputs = self.lstm(x)
        # outputs = self.lstm1(x)
        # outputs = self.lstm2(outputs)
        # outputs = self.lstm3(outputs)

        logits = self.linear3(self.linear2(self.linear1(tf.cast(outputs, dtype=tf.float32))))
        y_pred = tf.nn.softmax(logits)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y, y_pred)
        loss = tf.reduce_sum(loss)

        return_dict = {"loss": loss, "y_pred": y_pred}
        return return_dict
