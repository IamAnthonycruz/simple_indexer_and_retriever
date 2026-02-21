import os
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

def parse_file(file_path):
	hash_map = {}
	with open(rf"{file_path}", "r") as file:
		key, val = "", ""

		for line in file:
			line = line.strip()

			if ":" in line:
				
				if key:
					hash_map[key] = val

				key, val = line.split(":", 1)
			else:
				val += " " + line  

		if key:
			hash_map[key] = val
	return hash_map

main_dict = {}
count = 0
folder_path = r"C:\Users\cruzi\OneDrive\Documents\CS4422 Assignment 2\data"
for filename in os.listdir(folder_path):
	count += 1
	full_path = os.path.join(folder_path, filename)
	if os.path.isfile(full_path):
		main_dict.update(parse_file(full_path))
		print(main_dict)
	if count > 3: break


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def text_preprocessing(file_dict):
    processed_dict = {}

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    for key, val in file_dict.items():
        val = val.lower()
        val = re.sub(r'[^\w\s]', "", val)
        tokens = nltk.word_tokenize(val)
        tokens = [word for word in tokens if word not in stop_words]
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        processed_dict[key] = tokens

    return processed_dict

def build_vocabulary(preprocessed_dict):
	all_tokens = set()

	for tokens in preprocessed_dict.values():
		all_tokens.update(tokens)
	vocab = sorted(all_tokens)
	tok2idx = {token:idx for idx, token in enumerate(vocab)}
	idx2tok= {idx:token for idx, token in enumerate(vocab)}
	return tok2idx, idx2tok