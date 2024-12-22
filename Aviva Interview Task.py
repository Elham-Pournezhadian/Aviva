# Title: **Aviva Interview Task**  
# By: **Ella Pournezhadian**  
# Interview Date: **23 Dec 2023**  


import pandas as pd
import json
import uuid
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
import unittest



# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4') 
nltk.download('averaged_perceptron_tagger') 


# Setup stop words and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()




# Loading the JSON data
try:
    with open("input_data.json", "r") as file:
        petitions = json.load(file)
except FileNotFoundError:
    print("The file 'input_data.json' was not found.")
except json.JSONDecodeError:
    print("Error decoding the JSON file.")


# Extracting and preprocess data
data = []
for petition in petitions:
    title = petition['label']['_value']
    abstract = petition['abstract']['_value']
    num_signatures = petition['numberOfSignatures']
    data.append({
        "title": title,
        "abstract": abstract,
        "num_signatures": num_signatures
    })

# Converting to DataFrame
df = pd.DataFrame(data)






# Defining Functions

def tokenize_and_filter(text):
    '''This function Removes the stop words, and Lemmatize them, then tokenizes them.'''
    
    words = word_tokenize(text.lower())
    tagged_words = pos_tag(words)
    filtered_words = []
    for word, tag in tagged_words:
        if word not in stop_words:
            if tag.startswith('N'):  
                lemmatized_word = lemmatizer.lemmatize(word, pos='n') 
                if len(lemmatized_word) >= 5:
                    filtered_words.append(lemmatized_word)
            else:
                if len(word) >= 5:
                    filtered_words.append(word)
    
    return filtered_words


# Generating word counts
def generate_word_counts(tokens, common_words):
    word_counts = {word: tokens.count(word) for word in common_words}
    return word_counts






# Generating uniqueid
df["petition_id"] = [str(uuid.uuid4()) for _ in range(len(df))]
df["all_text"] = df["title"] + " " + df["abstract"]


df_lemmatize = df.copy()


# Applying tokenizing (with&without Lemmatizing)
df_lemmatize["tokens"] = df_lemmatize["all_text"].apply(tokenize_and_filter)


# Finding the 20 most common words across all petitions
all_words_lemmatize = [word for tokens in df_lemmatize["tokens"] for word in tokens]
most_common_words_lemmatize = [word for word, _ in Counter(all_words_lemmatize).most_common(20)]


# Generating common word counts
df_lemmatize = df_lemmatize.join(df_lemmatize["tokens"].apply(lambda tokens: pd.Series(generate_word_counts(tokens, most_common_words_lemmatize))))


# Final output with petition_id and word counts
lemmatize_output = df_lemmatize[["petition_id"] + most_common_words_lemmatize]





# Unit tests for key functions
class TestPetitionProcessing(unittest.TestCase):

    def test_tokenize_and_filter(self):
        text = "Testing, voting, generating, these are random words for children."
        result = tokenize_and_filter(text)
        expected = ["testing", "voting", "generating", "random", "word", "child"] 
        self.assertEqual(result, expected)
        
if __name__ == "__main__":
    unittest.main(argv=[''], exit=False)


    


# Save to CSV
output_csv_path = "output_petitions.csv"
lemmatize_output.to_csv(output_csv_path, index=False)

print(f"Output saved to {output_csv_path}")



