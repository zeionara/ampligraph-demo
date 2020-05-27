from ampligraph.evaluation import train_test_split_no_unseen
from ampligraph.latent_features import ComplEx, TransE


def split(graph: iter, train_portion: float = 0.8):
    n_test_samples = int(len(graph) * (1 - train_portion))
    return train_test_split_no_unseen(graph, test_size=n_test_samples, seed=0, allow_duplication=False)


def train_complex(train_samples: iter):
    model = ComplEx(
        batches_count=100,
        seed=0,
        epochs=200,
        k=150,
        eta=5,
        optimizer='adam',
        optimizer_params={
            'lr': 1e-3
        },
        loss='multiclass_nll',
        regularizer='LP',
        regularizer_params={
            'p': 3,
            'lambda': 1e-5
        },
        verbose=True
    )
    model.fit(train_samples, early_stopping=False)
    return model


def train_transe(train_samples: iter):
    model = TransE(
        batches_count=100,
        seed=0,
        epochs=200,
        k=150,
        eta=5,
        optimizer='adam',
        optimizer_params={
            'lr': 1e-3
        },
        loss='multiclass_nll',
        regularizer='LP',
        regularizer_params={
            'p': 3,
            'lambda': 1e-5
        },
        verbose=True
    )
    model.fit(train_samples, early_stopping=False)
    return model
