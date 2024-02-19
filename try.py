import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from sacrebleu import corpus_chrf, sentence_chrf
from rouge import Rouge
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('wordnet')
from collections import Counter
from nltk.util import ngrams
import sys
from nltk.translate import meteor
from crystalbleu import corpus_bleu

sys.setrecursionlimit(5000)

# Reading text and reference from files
#file_path = "/Users/miyamotoeiji/Desktop/res/nvBench_/gpt-zero.txt"
#file_path = "/Users/miyamotoeiji/Desktop/res/nvBench_/gpt-few.txt"
#file_path = "/Users/miyamotoeiji/Desktop/res/nvBench_/gpt-cot.txt"
#file_path = "/Users/miyamotoeiji/Desktop/res/nvBench_/gpt-emo.txt"
#file_path = "/Users/miyamotoeiji/Desktop/res/nvBench_/gpt-role.txt"
#file_path = "/Users/miyamotoeiji/Desktop/res/nvBench_/gpt-meta.txt"
#file_path = "/Users/miyamotoeiji/Desktop/res/nvBench_/gemini-zero.txt"
#file_path = "/Users/miyamotoeiji/Desktop/res/nvBench_/gemini-few.txt"
#file_path = "/Users/miyamotoeiji/Desktop/res/nvBench_/gemini-cot.txt"
#file_path = "/Users/miyamotoeiji/Desktop/res/nvBench_/gemini-emo.txt"
#file_path = "/Users/miyamotoeiji/Desktop/res/nvBench_/gemini-role.txt"
#file_path = "/Users/miyamotoeiji/Desktop/res/nvBench_/gemini-meta.txt"
#file_path ="/Users/miyamotoeiji/Desktop/res/nvBench_/mist-zero.txt"
#file_path ="/Users/miyamotoeiji/Desktop/res/nvBench_/mist-few.txt"
#file_path ="/Users/miyamotoeiji/Desktop/res/nvBench_/mist-cot.txt"
#file_path ="/Users/miyamotoeiji/Desktop/res/nvBench_/mist-emo.txt"
#file_path ="/Users/miyamotoeiji/Desktop/res/nvBench_/mist-role.txt"
#file_path ="/Users/miyamotoeiji/Desktop/res/nvBench_/mist-meta.txt"
#file_path ="/Users/miyamotoeiji/Desktop/res/nvBench_/mixtral-zero.txt"
#file_path ="/Users/miyamotoeiji/Desktop/res/nvBench_/mixtral-few.txt"
#file_path ="/Users/miyamotoeiji/Desktop/res/nvBench_/mix-cot.txt"
#file_path ="/Users/miyamotoeiji/Desktop/res/nvBench_/mix-emo.txt"
#file_path ="/Users/miyamotoeiji/Desktop/res/nvBench_/mix-role.txt"
#file_path ="/Users/miyamotoeiji/Desktop/res/nvBench_/mix-meta.txt"
#file_path ="/Users/miyamotoeiji/Desktop/res/nvBench_/claude-zero.txt"
#file_path ="/Users/miyamotoeiji/Desktop/res/nvBench_/claude-few.txt"
#file_path ="/Users/miyamotoeiji/Desktop/res/nvBench_/claude-cot.txt"
#file_path ="/Users/miyamotoeiji/Desktop/res/nvBench_/claude-emo.txt"
#file_path ="/Users/miyamotoeiji/Desktop/res/nvBench_/claude-role.txt"
file_path ="/Users/miyamotoeiji/Desktop/res/nvBench_/claude-meta.txt"

ref_file_path = "/Users/miyamotoeiji/Desktop/res/nvBench_/ref.txt"

with open(file_path, 'r', encoding='utf-8') as file:
    generated_codes = file.read()

with open(ref_file_path, "r", encoding="utf-8") as f:
    reference_codes = f.read()

# Tokenization
tGen = word_tokenize(generated_codes)
tRef = [word_tokenize(reference_codes)]  # Reference for BLEU must be a list of list of tokens
tRef2 = word_tokenize(reference_codes)

# BLEU Score (using NLTK, assuming single reference)
bleu_score = corpus_bleu([tRef], [tGen], smoothing_function=SmoothingFunction().method1)
print(f"BLEU Score: {bleu_score}")

# ROUGE Scores (using python-rouge, input must be strings, not tokens)
rouge = Rouge()
scores = rouge.get_scores([generated_codes], [reference_codes], avg=True)
print(f"ROUGE-1: {scores['rouge-1']['f']}")
print(f"ROUGE-2: {scores['rouge-2']['f']}")
print(f"ROUGE-L: {scores['rouge-l']['f']}")

# METEOR Score (using NLTK, expects strings)
score_m = round(meteor(
        tRef,tGen), 4)

#tokenized text required
print(f"METEOR Score: {score_m}")

# chrF Score (using sacrebleu, expects strings)
chrf_score = corpus_chrf(tRef2,tGen,remove_whitespace=True)#6, 3, remove_whitespace=True)
print(f"chrF Score: {chrf_score}")

#CrystalBLEU score
k = 500
tokenized_corpus = tGen + tRef2
# <tokenized_corpus> is a list of strings
# Extract all n-grams of length 1-4
all_ngrams = []
for n in range(1, 5):
    all_ngrams.extend(list(ngrams(tokenized_corpus, n)))
# Calculate frequencies of all n-grams
frequencies = Counter(all_ngrams)
trivially_shared_ngrams = dict(frequencies.most_common(k))

# 3. Calculate CrystalBLEU
crystalBLEU_score = corpus_bleu(
    [tRef], [tGen], smoothing_function=SmoothingFunction().method1, ignoring=trivially_shared_ngrams)
print(f"CrystalBLEU Score: {crystalBLEU_score}")