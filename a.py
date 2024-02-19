import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize

# s = """
# {
#   "$schema": "https://vega.github.io/schema/vega-lite/v4.json",
#   "description": "A line chart showing the count of sections per year grouped by semester",
#   "data": {
#     "values": [
#       {"year": "2001", "semester": "Fall", "count": 2},
#       {"year": "2001", "semester": "Spring", "count": 3},
#       {"year": "2002", "semester": "Fall", "count": 9},
#       {"year": "2002", "semester": "Spring", "count": 4},
#       {"year": "2003", "semester": "Fall", "count": 6},
#       {"year": "2003", "semester": "Spring", "count": 6},
#       {"year": "2004", "semester": "Fall", "count": 3},
#       {"year": "2004", "semester": "Spring", "count": 4},
#       {"year": "2005", "semester": "Fall", "count": 4},
#       {"year": "2005", "semester": "Spring", "count": 4},
#       {"year": "2006", "semester": "Fall", "count": 8},
#       {"year": "2006", "semester": "Spring", "count": 5},
#       {"year": "2007", "semester": "Fall", "count": 6},
#       {"year": "2007", "semester": "Spring", "count": 6},
#       {"year": "2008", "semester": "Fall", "count": 1},
#       {"year": "2008", "semester": "Spring", "count": 9},
#       {"year": "2009", "semester": "Fall", "count": 7},
#       {"year": "2009", "semester": "Spring", "count": 2},
#       {"year": "2010", "semester": "Fall", "count": 5},
#       {"year": "2010", "semester": "Spring", "count": 6}
#     ]
#   },
#   "mark": "line",
#   "encoding": {
#     "x": {"field": "year", "type": "ordinal"},
#     "y": {"aggregate": "count", "field": "count", "type": "quantitative"},
#     "color": {"field": "semester", "type": "nominal"}
#   }
# }
# """
# b = "encoding is mark and line"
# a = word_tokenize(s)
# c = word_tokenize(b)
# print (nltk.translate.meteor_score.meteor_score(
#     [a], c)
#     )


# print("I love dogs")
# print("I love dogs".split())
# a = "I love dogs"
# print(word_tokenize(a))