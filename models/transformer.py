from models.tf_basemodel import tf_BasedModel
import tensorflow as tf
from tensorflow.keras import layers
import models.transformer_funcs as transformer


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class Transformer(tf_BasedModel):
    def __init__(
            self,
            meta_data,
            batch_sz,
            embedding_dim=16,
            nhead=4,
            hidden_size=100,
            num_layers=1,
            model_save_path="./transformer_models",
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
        )
        self.num_labels = meta_data["num_labels"]
        self.feature_type = feature_type
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.use_tfidf = use_tfidf

        self.cls = tf.zeros((1, 1, embedding_dim))


        self.encoder_layer = TokenAndPositionEmbedding(64, meta_data["vocab_size"], self.embedding_dim)
        self.transformer_encoder = TransformerBlock(self.embedding_dim, nhead, hidden_size, False)

        #self.criterion = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.prediction_layer = tf.keras.layers.Dense(self.num_labels)

    def call(self, input_dict):
        if self.label_type == "anomaly":
            y = tf.convert_to_tensor(input_dict["window_anomalies"])
        elif self.label_type == "next_log":
            y = tf.convert_to_tensor(input_dict["window_labels"])

        x = input_dict["features"]
        x = tf.nn.embedding_lookup(self.embedding_matrix, x)
        if self.feature_type == "semantics":
            if not self.use_tfidf:
                x = x.sum(dim=-2)  # add tf-idf

        x_t = tf.transpose(x, perm=[1, 0, 2])

        x_transformed = self.transformer_encoder(x_t)
        representation = tf.reduce_mean(tf.transpose(x_transformed, perm=[1, 0, 2]), axis=1)

        y_pred = self.prediction_layer(representation)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y, y_pred)
        return_dict = {"loss": loss, "y_pred": y_pred}
        return return_dict
