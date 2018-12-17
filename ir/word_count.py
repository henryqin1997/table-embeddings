import nltk
from nltk.tokenize import RegexpTokenizer
import json
import os

wiki_dir = 'ir/wiki/0'
nltk.download('stopwords')
nltk.download('punkt')
stopwords = set(nltk.corpus.stopwords.words('english'))
tokenizer = RegexpTokenizer(r'\w+')

if __name__ == '__main__':
    for file in os.listdir(wiki_dir):
        if file.endswith('.json'):
            data = json.load(open(os.path.join(wiki_dir, file)))
            summaries = ' '.join([item['summary'] for item in data if item['summary']])

            # Tokenize Wikipedia summaries
            # words = nltk.word_tokenize(summaries)
            words = tokenizer.tokenize(summaries)

            # Remove single-character tokens (mostly punctuation)
            words = [word for word in words if len(word) > 1]

            # Remove numbers
            words = [word for word in words if not word.isnumeric()]

            # Lowercase all words (default_stopwords are lowercase too)
            words = [word.lower() for word in words]

            # Stemming words seems to make matters worse, disabled
            # stemmer = nltk.stem.snowball.SnowballStemmer('german')
            # words = [stemmer.stem(word) for word in words]

            # Remove stopwords
            words = [word for word in words if word not in stopwords]

            # Calculate frequency distribution
            fdist = nltk.FreqDist(words)

            # Output top 50 words
            # for word, frequency in fdist.most_common(50):
            #     print(u'{};{}'.format(word, frequency))

            json.dump(fdist.most_common(50), open(os.path.join('ir', 'word_count', '0', file), 'w+'),
                      ensure_ascii=False, indent=4)
