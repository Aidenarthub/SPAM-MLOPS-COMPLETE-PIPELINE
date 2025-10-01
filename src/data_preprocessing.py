import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
# importing tokenizer library from the nltk
from nltk.tokenize import TreebankWordTokenizer

# Ensure the "logs" directory exists
log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

# Setting up logger
logger=logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

cansole_handler=logging.StreamHandler()
cansole_handler.setLevel('DEBUG')

log_file_path=os.path.join(log_dir,'data_preprocessing.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
cansole_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(cansole_handler)
logger.addHandler(file_handler)


def transform_text(text):
    """
    Transforms the input text by converting it to lowercase, tokenizing, removing stopwords and punctualtion, and stemming.
    """
    tokenizer = TreebankWordTokenizer()
    ps=PorterStemmer()
    # Convert to lowercase
    text=text.lower()
    # Tokenize the text
    text=tokenizer.tokenize(text)
    # Remove non-alphanumeric tokens
    text=[word for word in text if word.isalnum()]
    # Remove stopwords and punctuations
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word not in stop_words]
    # Stem the word
    text=[ps.stem(word) for word in text]
    # join the tokens back into a single string
    return " ".join(text)

def preprocess_df(df,text_column='text',target_column='target'):
    """
    Preprocess the DataFrame by encoding the target column, removing duplicates, and transforming the text column.
    """
    try:
        logger.debug('starting preprocessing for DataFrame')
        # Encode the target column
        encoder=LabelEncoder()
        df[target_column]=encoder.fit_transform(df[target_column])
        logger.debug('Target column encoded')

        # Remove duplicate rows
        df=df.drop_duplicates(keep='first')
        logger.debug('Duplicate removed')

        # Apply text transformation to the specified text column
        df.loc[:,text_column]=df[text_column].apply(transform_text)
        logger.debug('Text column transformed')
        return df
    
    except KeyError as e:
        logger.error('column not found: %s',e)
        raise
    except Exception as e:
        logger.error('Error during text normalization: %s', e)
        raise

from pathlib import Path

def main(text_column='text', target_column='target'):
    try:
        # Load raw data
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Data loaded properly')

        # Preprocess
        train_processed_data = preprocess_df(train_data, text_column, target_column)
        test_processed_data = preprocess_df(test_data, text_column, target_column)

        # Save to data/interim
        data_path = Path('./data') / 'interim'
        data_path.mkdir(parents=True, exist_ok=True)

        train_processed_data.to_csv(data_path / 'train_processed.csv', index=False)
        test_processed_data.to_csv(data_path / 'test_processed.csv', index=False)

        logger.debug('Processed data saved to %s', data_path.as_posix())
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s', e)
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)


if __name__ == "__main__":
    main()
