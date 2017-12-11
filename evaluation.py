import numpy as np
import json
import os
from permute.core import one_sample

eval_save_path = 'save_dir/evals'

# Comment in the one measure you want to compare the methods on.
measures = (
    'ROUGE-1-F',
    # 'ROUGE-1-R',
    # 'ROUGE-2-F',
    # 'ROUGE-2-R',
)

# Comment in the one dataset you want to compare the methods on.
dataset_name = 'opinosis'
# dataset_name = 'duc'
# dataset_name = 'tac'

# Include names of systems for comparison here.
# The current tuple contains all original names used in the thesis.
systems = (
    'docemb_blendle',
    'docemb_google',
    'embdist_sentence_blendle',
    'embdist_sentence_google',
    'embdist_word_blendle',
    'embdist_word_google',
    'kageback_blendle',
    'kageback_google',
    'lb_2010',
    'lb_2011',
    'lead',
    'lead_3',
    'mmr_w2v_blendle',
    'mmr_w2v_google',
    'msv',

    'rnn_ranker_blendle_1epoch',
    'rnn_ranker_blendle_pretrained_1epoch',
    'rnn_ranker_blendle_cotrained_25_1epoch',
    'rnn_ranker_blendle_cotrained_1epoch',
    'rnn_ranker_blendle_cotrained_only',
    'rnn_ranker_blendle_pretrain',

    'rnn_ranker_cotrained_1epoch',
    'rnn_ranker_cotrained_25_1epoch',
    'rnn_ranker_google_cotrained_only',
    'rnn_ranker_google_1epoch',
    'rnn_ranker_pretrain',
    'rnn_ranker_pretrained_1epoch',

    'textrank_raw',
    'textrank_stopwords_stemming',
    'textrank_postag',
    'textrank_complete',
    'textrank_w2v_google',
    'textrank_w2v_blendle',
    'textrank_tfidf_w2v_blendle',
    'textrank_tfidf_w2v_google',
    'textrank_sif_blendle',
    'textrank_sif_google',
)
evals = {}


for name in systems:
    evals[name] = json.load(open(os.path.join(eval_save_path, '{d}/{n}.json'.format(d=dataset_name, n=name))))

methods = {key: [[x[measure] for x in eval] for measure in measures] for key, eval in evals.items()}

rouge_results = {key: {measure: float("{0:.2f}".format(np.mean([x[measure] for x in evals[key]]) * 100))
                       for measure in measures}
                 for key in evals.keys()}
best_system = list(rouge_results.keys())[np.argmax([x[measures[0]] for x in rouge_results.values()])]

for name in systems:
    print('{:>30}: {:5}\n'.format(name, ' & '.join([str(rouge_results[name][m]) for m in measures])))
print("\nBest system: {} (score: {})\n".format(best_system, rouge_results[best_system][measures[0]]))

alpha = .05
number_of_tests = len(systems) - 1
print('\nStatistically comparable results:')


p_values = []
for method_name, method in methods.items():
    if method_name != best_system:
        # make list of (measure, (Tuple(result))-tuples for statistical test
        ttest_input = list(zip(measures, zip(*(method, methods[best_system]))))
        ttest_outputs = [(one_sample(*result, stat='mean', reps=10**5, seed=42, alternative='two-sided')[0], measure)
                         for measure, result in ttest_input]
        p_values.append((ttest_outputs[0][0], ttest_outputs[0][1], method_name))

for i, (p_value, measure, method_name) in enumerate(sorted(p_values)):
    # Compute corrected alpha according to Holm-Bonferroni method
    corrected_alpha = alpha / (number_of_tests - i)
    if p_value > corrected_alpha:
        print("{:>40}    {:10}    p-value: {:<6.4f}    score: {:<10}".format(method_name,
                                                                             measure,
                                                                             p_value,
                                                                             rouge_results[method_name][measure]))
