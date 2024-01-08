import pandas
import re


def clean_text(text):
    """
    Function to perform basic text cleaning:
    - Lowercase
    - Removing special characters
    - Removing  punctuations
    """
    # Convert text to lowercase
    text = text.lower()

    # Remove special characters and punctuations
    text = re.sub(r'[^a-z0-9\s]', '', text)

    return text


# TODO: add method to use a smaller sample of the dataset
def create_subset():
    pass