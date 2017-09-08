from pythonrouge.pythonrouge import Pythonrouge

def evaluate(system_summary, reference_summaries, stemming=False, stopwords=False, use_cf=False, ngram=2):
    ROUGE_path = "rouge_files/ROUGE-1.5.5/ROUGE-1.5.5.pl"
    data_path = "rouge_files/ROUGE-1.5.5/data/"

    # initialize setting of ROUGE, eval ROUGE-1, 2
    rouge = Pythonrouge(n_gram=ngram,
                        ROUGE_SU4=False,
                        ROUGE_L=False,
                        stemming=stemming,
                        stopwords=stopwords,
                        word_level=True,
                        length_limit=True,
                        length=100,
                        use_cf=use_cf,
                        cf=95,
                        scoring_formula="average",
                        resampling=True,
                        samples=1000,
                        favor=True,
                        p=0.5)

    # system summary: list of summaries, where each summary is a list of sentences
    summary = [system_summary]

    # reference summaries: list of (list of summaries per article), where each summary is a list of sentences
    reference = [[[summary] for summary in reference_summaries]]

    setting_file = rouge.setting(files=False, summary=summary, reference=reference, temp_root='')

    result = rouge.eval_rouge(setting_file, ROUGE_path=ROUGE_path, data_path=data_path)

    return result