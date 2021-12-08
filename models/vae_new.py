from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, ReLU
from tensorflow.math import exp, sqrt, square
from models.tf_basemodel import tf_BasedModel
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
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
        self.latent_size = 64
        self.hidden_dim = 512  # H_d
        self.encoder = Sequential(layers=[
            Dense(self.hidden_size, activation="relu"),
            Dense(self.hidden_size, activation="relu"),
            Dense(self.hidden_size, activation="relu")
        ])
        self.mu_layer = Dense(self.latent_size)
        self.logvar_layer = Dense(self.latent_size)
        self.decoder = Sequential(layers=[
            Dense(self.hidden_size, activation="relu"),
            Dense(self.hidden_size // 2, activation="relu"),
            Dense(self.hidden_size * self.num_directions, activation="relu"),
            Dense(self.num_labels, activation="sigmoid"),
        ])

    def call(self, input_dict):
        if self.label_type == "anomaly":
            y = tf.convert_to_tensor(input_dict["window_anomalies"])
        elif self.label_type == "next_log":
            y = tf.convert_to_tensor(input_dict["window_labels"])

        y = tf.one_hot(y, self.num_labels)
        y = tf.squeeze(y)

        x = tf.convert_to_tensor(input_dict["features"])

        encoded = self.encoder(x)
        mu = self.mu_layer(encoded)
        logvar = self.logvar_layer(encoded)
        y_pred = self.decoder(reparametrize(mu, logvar))

        bce_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        loss = bce_fn(y, y_pred) * y_pred.shape[-1]
        

        return_dict = {"loss": loss, "y_pred": y_pred}
        # print(return_dict)
        return return_dict

