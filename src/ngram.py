import re
import math


# Method for making vocabs from train dataset
def make_vocab(train_path) :
  corpus = []
  corpus_length = 0
  freq_table = dict()
  
  # Only use training data
  with open(train_path, 'r') as source :
    corpus = [re.split('\s+', line) for line in source]
      
  # Make token frequency table
  for sentence in corpus :
    for word in sentence :
      freq_table[word] = freq_table.get(word, 0) + 1
      corpus_length += 1
      
  # 'UNK' token processing
  for word in list(freq_table.keys()) :
    if freq_table[word] < 3 :
      freq_table['UNK'] = freq_table.get('UNK', 0) + freq_table[word]
      del(freq_table[word])
      
  return freq_table, corpus_length

# Unigram Model
class UnigramModel :
  def __init__(self, freq_table, corpus_length, smoothing=False) :
    self.freq_table = freq_table
    self.corpus_length = corpus_length
    self.smoothing = smoothing
  
  # Calculate unigram's probability
  def calculate_unigram_prob(self, word) :
    word_freq = self.freq_table[word]
    denominator = self.corpus_length
    # Laplace smoothing
    if self.smoothing :
      word_freq += 1
      denominator += len(self.freq_table)
    unigram_prob = word_freq / float(denominator)
    return unigram_prob
  
  # Calculate sentence's probability
  def calculate_sentence_prob(self, sentence) :
    sentence_log_prob = 0
    
    # Divide into words
    for word in sentence :
      # 'UNK' token processing
      if self.freq_table.get(word, 0) == 0 :
        word_prob = self.calculate_unigram_prob('UNK')
      else :
        word_prob = self.calculate_unigram_prob(word)
      sentence_log_prob += math.log(word_prob)
      
    return math.exp(sentence_log_prob)

# Caculate unigram's perplexity
def calculate_unigram_perplexity(model, file_path) :
  corpus_length = 0
  total_perplexity = 0
  
  # Load test file
  with open(file_path, 'r') as source :
    corpus = [re.split('\s+', line) for line in source]
  
  # Divide into setences
  for sentence in corpus :
    # Caculate each sentence's perplexity
    sentence_length = len(sentence)
    sentence_exp = -1 / sentence_length
    unigram_sentence_prob = model.calculate_sentence_prob(sentence)
    
    # Exception Handling : 0 ^ (-small)
    try :
      unigram_sentence_perplexity = math.pow(unigram_sentence_prob, sentence_exp)
    except :
      # unigram_sentence_perplexity = float('inf')
      unigram_sentence_perplexity = 0
      
    total_perplexity += sentence_length * unigram_sentence_perplexity
    corpus_length += sentence_length
    
  return total_perplexity / corpus_length
    
def unigram() :
  train_path = '../datasets/1b_benchmark.train.tokens'
  test_path = '../datasets/1b_benchmark.test.tokens'
  
  freq_table, corpus_length = make_vocab(train_path)
  unigram_model = UnigramModel(freq_table, corpus_length, False)
  unigram_perplexity = calculate_unigram_perplexity(unigram_model, train_path)
  print(unigram_perplexity)
  
  
if __name__ == "__main__" :
  unigram()
  print("Success!\n")
