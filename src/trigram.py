import re
import numpy as np
from collections import Counter
from tqdm import tqdm

class ShannonVisualizationMethodTrigram():
    def __init__(self, corpus_txt_path_list):
        self.tokens = self._aggregate_corpus(corpus_txt_path_list)
    
    def _generate_tokens(self, corpus_txt_path):
        
        regex = "[a-zA-ZçÇãÃõÕáÁéÉíÍóÓúÚâÂêÊîÎôÔûÛàÀ]+"
        corpus = open(corpus_txt_path, encoding='UTF-8').read()
        tokens = re.findall(regex, corpus)
        
        return tokens
    
    def _aggregate_corpus(self, corpus_txt_path_list):
        tokens = []
        
        for corpus_txt_path in corpus_txt_path_list:
            tokens += self._generate_tokens(corpus_txt_path)
        return tokens

    def _choose_next_word(self, second_last_word, last_word):
        # Denominator cal (Count (w1,w2))
        count_w1_w2 = Counter(zip(self.tokens, self.tokens[1:]))[(second_last_word, last_word)]

        if count_w1_w2 == 0:
            raise Exception("The bigram does not exist in the corpus")

        trigrams = Counter(zip(self.tokens, self.tokens[1:], self.tokens[2:]))

        # Count (w1, w2, w3) for each Token
        count_w1_w2_w3s = [trigrams[(second_last_word, last_word, token)] for token in self.tokens]

        # Calculate Probability for each token
        trigram_probabilities = np.array(count_w1_w2_w3s) / count_w1_w2

        # Remove zero probabilities
        set_words = [(trigram_probabilities[i], self.tokens[i]) for i in range(len(trigram_probabilities)) if
                     trigram_probabilities[i] > 0]

        # Make a set of words
        set_words = set(set_words)

        # Choose next word
        prob_words = [st[0] for st in set_words]
        candidate_words = [st[1] for st in set_words]

        next_word = np.random.choice(candidate_words, 1, p=prob_words).item()

        return next_word

    def generate_text(self, first_word, second_word, words_count = 10):

        generated_text_str = f"{first_word} {second_word}"

        for _ in range(0,words_count):
            generated_word = self._choose_next_word(first_word, second_word)
            first_word = second_word
            second_word = generated_word

            generated_text_str = generated_text_str + " " + generated_word

        return generated_text_str  