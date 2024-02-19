import nltk
from nltk.tokenize import word_tokenize
from sacrebleu.metrics import CHRF
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

nltk.download('punkt')

# List of all file paths for generated texts
file_paths = [
    "/Users/miyamotoeiji/Desktop/res/nvBench_/gpt-zero.txt",
"/Users/miyamotoeiji/Desktop/res/nvBench_/gpt-few.txt",
"/Users/miyamotoeiji/Desktop/res/nvBench_/gpt-cot.txt",
"/Users/miyamotoeiji/Desktop/res/nvBench_/gpt-emo.txt",
"/Users/miyamotoeiji/Desktop/res/nvBench_/gpt-role.txt",
"/Users/miyamotoeiji/Desktop/res/nvBench_/gpt-meta.txt",
"/Users/miyamotoeiji/Desktop/res/nvBench_/gemini-zero.txt",
"/Users/miyamotoeiji/Desktop/res/nvBench_/gemini-few.txt",
"/Users/miyamotoeiji/Desktop/res/nvBench_/gemini-cot.txt",
"/Users/miyamotoeiji/Desktop/res/nvBench_/gemini-emo.txt",
"/Users/miyamotoeiji/Desktop/res/nvBench_/gemini-role.txt",
"/Users/miyamotoeiji/Desktop/res/nvBench_/gemini-meta.txt",
"/Users/miyamotoeiji/Desktop/res/nvBench_/mist-zero.txt",
"/Users/miyamotoeiji/Desktop/res/nvBench_/mist-few.txt",
"/Users/miyamotoeiji/Desktop/res/nvBench_/mist-cot.txt",
"/Users/miyamotoeiji/Desktop/res/nvBench_/mist-emo.txt",
"/Users/miyamotoeiji/Desktop/res/nvBench_/mist-role.txt",
"/Users/miyamotoeiji/Desktop/res/nvBench_/mist-meta.txt",
"/Users/miyamotoeiji/Desktop/res/nvBench_/mixtral-zero.txt",
"/Users/miyamotoeiji/Desktop/res/nvBench_/mixtral-few.txt",
"/Users/miyamotoeiji/Desktop/res/nvBench_/mix-cot.txt",
"/Users/miyamotoeiji/Desktop/res/nvBench_/mix-emo.txt",
"/Users/miyamotoeiji/Desktop/res/nvBench_/mix-role.txt",
"/Users/miyamotoeiji/Desktop/res/nvBench_/mix-meta.txt",
"/Users/miyamotoeiji/Desktop/res/nvBench_/claude-zero.txt",
"/Users/miyamotoeiji/Desktop/res/nvBench_/claude-few.txt",
"/Users/miyamotoeiji/Desktop/res/nvBench_/claude-cot.txt",
"/Users/miyamotoeiji/Desktop/res/nvBench_/claude-emo.txt",
"/Users/miyamotoeiji/Desktop/res/nvBench_/claude-role.txt", 
"/Users/miyamotoeiji/Desktop/res/nvBench_/claude-meta.txt"
]

# Path to the reference file (assuming the same reference for all generated texts)
ref_file_path = "/Users/miyamotoeiji/Desktop/res/nvBench_/ref.txt"

# Read the reference text
with open(ref_file_path, "r", encoding="utf-8") as f:
    reference_codes = f.read()

# Tokenize the reference text for BLEU and store plain text for chrF
tRef_bleu = [word_tokenize(reference_codes)]  # For BLEU, reference must be a list of lists of tokens
tRef_chrf = reference_codes  # For chrF, reference is a string

# Initialize CHRF object for chrF score calculation
chrf = CHRF()

# Iterate over each file path, read the content, tokenize, and calculate scores
for file_path in file_paths:
    with open(file_path, 'r', encoding='utf-8') as file:
        generated_codes = file.read()
        
        # Tokenize the generated text for BLEU
        tGen_bleu = word_tokenize(generated_codes)
        
        # Calculate chrF score using strings directly
        chrf_score = chrf.sentence_score(generated_codes, [tRef_chrf])
        print(f"File: {file_path}")
        print(f"chrF Score: {chrf_score}\n")

# Note: Example shows chrF score calculation. Integrate BLEU, METEOR, or other metrics as needed.
