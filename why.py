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
file_path = "/Users/miyamotoeiji/Desktop/res/nvBench_/gpt-zero.txt"
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
#file_path ="/Users/miyamotoeiji/Desktop/res/nvBench_/claude-meta.txt"

ref_file_path = "/Users/miyamotoeiji/Desktop/res/nvBench_/ref.txt"

with open(file_path, 'r', encoding='utf-8') as file:
    generated_codes = file.read()

with open(ref_file_path, "r", encoding="utf-8") as f:
    reference_codes = f.read()

# Tokenization
tGen = word_tokenize(generated_codes)
tRef = [word_tokenize(reference_codes)]  # Reference for BLEU must be a list of list of tokens
tRef2 = word_tokenize(reference_codes)

# chrF Score (using sacrebleu, expects strings)
chrf_score = sentence_chrf(generated_codes,[reference_codes],remove_whitespace=True)#6, 3, remove_whitespace=True)
print(f"chrF Score: {chrf_score}")
