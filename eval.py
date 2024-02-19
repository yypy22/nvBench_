import nltk
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from sacrebleu import corpus_chrf
from rouge import Rouge
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('wordnet')
from collections import Counter
from nltk.util import ngrams
# 1. Import CrystalBLEU
from crystalbleu import corpus_bleu
import sys

# Increase the recursion limit
sys.setrecursionlimit(3000)  # The default is usually 1000

# generated_codes = ["""



# """
#             ]
file_path = "/Users/miyamotoeiji/Desktop/res/nvBench_/gpt-zero.txt"
text = ""
ref = ""

# Read the content of the file
with open(file_path, 'r', encoding='utf-8') as file:
    codes = file.read()
    text = codes
with open("/Users/miyamotoeiji/Desktop/res/nvBench_/ref.txt","r",encoding="utf-8") as f:
    code2 = f.read()
    ref = code2

generated_codes = text
reference_codes = ref
ref2 = [reference_codes]

# tGen = word_tokenize(generated_codes)
# tRef = word_tokenize(reference_codes)
tGen = generated_codes.split()
tRef = [reference_codes.split()]

# BLEU Score
# bleu_score = corpus_bleu(reference_codes, generated_codes)
bleu_score = corpus_bleu([tRef], [tGen])
print(f"BLEU Score: {bleu_score}")

# ROUGE Scores
rouge = Rouge()
scores = rouge.get_scores(tGen, tRef, avg=True)
print(f"ROUGE-1: {scores['rouge-1']['f']}")
print(f"ROUGE-2: {scores['rouge-2']['f']}")
print(f"ROUGE-L: {scores['rouge-l']['f']}")

#METEOR score
score_m = nltk.translate.meteor_score.meteor_score(
    tGen, tRef)
#tokenized text required
print(f"METEOR Score: {score_m}")

# chrF Score
chrf_score = corpus_chrf(generated_codes, reference_codes).score
print(f"chrF Score: {chrf_score}")

#CrystalBLEU score
k = 500
tokenized_corpus = tGen + tRef
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
    tRef, tGen, ignoring=trivially_shared_ngrams)
print(f"CrystalBLEU Score: {crystalBLEU_score}")
