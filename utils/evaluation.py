import numpy as np
import pandas as pd
from ampligraph.evaluation import mrr_score, hits_at_n_score, evaluate_performance
from scipy.special import expit


def compute_metrics(model, train_samples, test_samples):
    ranks = evaluate_performance(
        test_samples,
        model=model,
        filter_triples=train_samples,  # Corruption strategy filter defined above
        use_default_protocol=True,  # corrupt subj and obj separately while evaluating
        verbose=True
    )
    return {
        'MRR': mrr_score(ranks),
        'Hits@10': hits_at_n_score(ranks, n=10),
        'Hits@3': hits_at_n_score(ranks, n=3),
        'Hits@1': hits_at_n_score(ranks, n=1)
    }


def score_samples(model, evaluation_samples, train_samples):
    skipped_samples = np.array(
        list({
            tuple(i) for i in np.vstack((
                train_samples,
                evaluation_samples
            ))
        })
    )
    ranks = evaluate_performance(
        evaluation_samples,
        model=model,
        filter_triples=skipped_samples,  # Corruption strategy filter defined above
        corrupt_side='s+o',
        use_default_protocol=False,  # corrupt subj and obj separately while evaluating
        verbose=True
    )
    return model.predict(evaluation_samples), ranks


def summarize(scores: iter, evaluation_samples: iter, ranks: iter):
    return pd.DataFrame(
        list(
            zip(
                map(' '.join, evaluation_samples),
                ranks,
                np.squeeze(scores),
                np.squeeze(expit(scores))
            )
        ),
        columns=['statement', 'rank', 'score', 'prob']
    ).sort_values("prob", ascending=False)
