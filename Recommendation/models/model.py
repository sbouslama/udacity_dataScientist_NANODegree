from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Lambda, Embedding
from keras.optimizers import Adam

def categorize(inputs, emb_input_dims, emb_output_dims):
    """
    Replaces categorical features with trainable embeddings
    :param inputs: tensor with encoded categorical features in first columns
    :param emb_input_dims: number of unique classes in categorical features
    :param emb_output_dims: embedding dimensions of categorical features
    :return: transformed tensor
    """
    n_embs = len(emb_input_dims)
    if n_embs > 0:
        embs = []

        # iterate over categorical features
        for i, nunique, dim in zip(range(n_embs), emb_input_dims, emb_output_dims):
            # separate their values with Lambda layer
            tmp = Lambda(lambda x: x[:, i])(inputs)
            # pass them through Embedding layer
            embs.append(Embedding(nunique, dim)(tmp))

        # pass all the numerical features directly
        embs.append(Lambda(lambda x: x[:, n_embs:])(inputs))
        # and concatenate them
        outputs = Concatenate()(embs)
    else:
        outputs = inputs

    return outputs

class Model(Model):
    def __init__(self, que_dim, prof_dim, 
                 que_input_embs, que_outp_embs,
                 prof_input_embs, prof_outp_embs,
                 dense1_dim, dense2_dim
                ):
        
        super().__init__()
        
        self.que_feats = Input((que_dim,))
        self.prof_feats = Input((prof_dim,))
        
        self.que_catecorized= categorize(self.que_feats, que_input_embs, que_outp_embs)
        self.prof_catecorized= categorize(self.prof_feats, prof_input_embs, prof_outp_embs)
        
        self.merged = Concatenate()([self.que_catecorized, self.prof_catecorized])
#         self.merged = Concatenate()([self.que_feats, self.prof_feats])
        self.dense1 = Dense(dense1_dim, activation='tanh')(self.merged)
        self.dense2 = Dense(dense2_dim, activation='tanh')(self.dense1)
        self.outputs = Dense(1, activation='sigmoid')(self.dense2)

        super().__init__([self.que_feats, self.prof_feats], self.outputs)