%%writefile app.py
import streamlit as st
import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
import os
import gdown # Import gdown

# Define the path where the model will be saved locally in the Streamlit environment
LOCAL_SAVE_DIRECTORY = "./saved_sentiment_model"

# Define the Google Drive File ID of your saved model (or a zipped archive of it)
# You need to replace 'YOUR_GOOGLE_DRIVE_FILE_ID' with the actual ID from your shared link
# Example ID from your previous link: '1l0JXajXq4bjf0bOgfQkrxJNwyuzzww-G' (if that's the model file)
GOOGLE_DRIVE_FILE_ID = 'YOUR_GOOGLE_DRIVE_FILE_ID'

# Define the local path to save the downloaded file
# If you zipped the model directory, this should be the zip file name
# If it's the .h5 file, use the .h5 file name
DOWNLOADED_FILE_NAME = "tf_model.h5" # Or "saved_sentiment_model.zip" if you zipped

# --- Download the model from Google Drive if it doesn't exist locally ---
if not os.path.exists(LOCAL_SAVE_DIRECTORY):
    st.info(f"Downloading model from Google Drive...")
    try:
        # Ensure the directory for the downloaded file exists
        # If you're downloading a zip, you might create LOCAL_SAVE_DIRECTORY later after unzipping
        # If downloading an .h5 file directly into a folder structure, create parent dirs as needed
        # For simplicity, let's assume you're downloading the .h5 file directly to the main app directory for now
        # Or you might download a zip and unzip it. Let's refine this based on what was uploaded.

        # Assuming 'YOUR_GOOGLE_DRIVE_FILE_ID' is the ID of the 'tf_model.h5' file
        # and you want to save it inside LOCAL_SAVE_DIRECTORY structure
        # A better approach for the whole directory is to zip it and download the zip.

        # Let's assume the user uploaded the whole 'saved_sentiment_model' directory as a zip
        # If you uploaded the directory as a zip, replace GOOGLE_DRIVE_FILE_ID and DOWNLOADED_FILE_NAME accordingly
        # And add unzip logic here.

        # **Alternative (Simpler if you just uploaded the .h5 file):**
        # If your GOOGLE_DRIVE_FILE_ID is for a single large file like tf_model.h5
        # and your model loading logic expects this file structure, you might need to download
        # the tokenizer files separately or save the whole model directory as a zip.

        # Let's revert to downloading the directory as a zip as it's more likely for a full BERT model save.
        # --- ASSUMPTION: The user zipped the 'saved_sentiment_model' directory and uploaded the zip to Drive ---
        # If this assumption is wrong, we need to adjust the download and loading logic.
        # Let's assume GOOGLE_DRIVE_FILE_ID is the ID of 'saved_sentiment_model.zip'
        # And DOWNLOADED_FILE_NAME is 'saved_sentiment_model.zip'

        # Adjusting variables based on zipping the directory:
        # LOCAL_SAVE_DIRECTORY remains './saved_sentiment_model'
        # DOWNLOADED_ZIP_NAME = "saved_sentiment_model.zip"
        # GOOGLE_DRIVE_FILE_ID_OF_ZIP = 'YOUR_GOOGLE_DRIVE_ZIP_FILE_ID' # <-- You need this ID

        # Let's assume the user wants to download the directory itself as a zip from the link provided earlier
        # If the link '1l0JXajXq4bjf0bOgfQkrxJNwyuzzww-G' is actually the directory zipped:
        GOOGLE_DRIVE_FILE_ID_OF_ZIP = '1l0JXajXq4bjf0bOgfQkrxJNwyuzzww-G' # Assuming this ID is for the zipped model
        DOWNLOADED_ZIP_NAME = "saved_sentiment_model.zip"

        gdown.download(f'https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID_OF_ZIP}', DOWNLOADED_ZIP_NAME, fuzzy=True)

        # Unzip the downloaded file
        import zipfile
        with zipfile.ZipFile(DOWNLOADED_ZIP_NAME, 'r') as zip_ref:
            zip_ref.extractall(".") # Extract to the current directory

        st.success("Model downloaded and extracted successfully!")
    except Exception as e:
        st.error(f"Error downloading or extracting model: {e}")
        st.stop() # Stop the app if model cannot be loaded

# --- End Download Logic ---

# Load the trained model and tokenizer from the local directory
@st.cache_resource # Cache the model to avoid reloading on each interaction
def load_model_from_local(local_save_directory):
    try:
        # Assuming the model and tokenizer were saved using save_pretrained
        model = TFBertForSequenceClassification.from_pretrained(local_save_directory)
        tokenizer = BertTokenizer.from_pretrained(local_save_directory)
        st.success("Model and tokenizer loaded from local directory!")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model or tokenizer from local directory: {e}")
        return None, None

model, tokenizer = load_model_from_local(LOCAL_SAVE_DIRECTORY)


st.title("IMDB TV Show Review Sentiment Analysis")
st.markdown("Enter a TV show review below to predict its sentiment (Positive/Negative).")

# Text input for the review
user_review = st.text_area("Enter Review Here:", "")

if st.button("Predict Sentiment"):
    if model is not None and tokenizer is not None and user_review:
        # Preprocess and encode the input review
        # Using the same preprocessing steps as in the notebook
        import string
        import re
        import nltk
        # Note: NLTK downloads within Streamlit can be problematic.
        # It's better to ensure these are downloaded in the environment before running the app.
        # Adding try-except for robustness, but pre-downloading is recommended.
        try:
             nltk.data.find('corpora/stopwords')
             nltk.data.find('tokenizers/punkt')
             nltk.data.find('corpora/wordnet')
             nltk.data.find('corpora/omw-1.4')
             from nltk.corpus import stopwords
             from nltk.tokenize import word_tokenize
             from nltk.stem import WordNetLemmatizer
             stop_words = set(stopwords.words('english'))
             lemmatizer = WordNetLemmatizer()
        except nltk.downloader.DownloadError:
             st.error("NLTK data not found. Please ensure 'stopwords', 'punkt', 'wordnet', and 'omw-1.4' are downloaded.")
             st.stop()


        def preprocess_text(text):
            # Remove HTML tags
            text = re.sub(r'<.*?>', '', text)
            # Remove punctuation and convert to lowercase
            text = text.translate(str.maketrans('', '', string.punctuation)).lower()
            # Tokenize the text
            tokens = word_tokenize(text)
            # Remove stop words and lemmatize
            processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
            return " ".join(processed_tokens)

        processed_review = preprocess_text(user_review)

        # Encode the processed review
        max_length = 128 # Use the same max_length as during training
        encoded_review = tokenizer(
            processed_review,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='tf'
        )

        # Make prediction
        with st.spinner("Predicting..."):
            # The model expects input as a dictionary
            tf_inputs = {key: tf.constant(val) for key, val in encoded_review.items()}
            outputs = model(tf_inputs)
            logits = outputs.logits
            # Get the predicted class (0 for negative, 1 for positive)
            predicted_class_id = tf.argmax(logits, axis=1).numpy()[0]

            # Map the class ID back to sentiment label
            sentiment_map = {0: 'Negative', 1: 'Positive'}
            predicted_sentiment = sentiment_map[predicted_class_id]

        st.subheader("Prediction:")
        if predicted_sentiment == 'Positive':
            st.success(f"Sentiment: {predicted_sentiment} 😊")
        else:
            st.error(f"Sentiment: {predicted_sentiment} 😟")

    elif not user_review:
        st.warning("Please enter a review to predict sentiment.")
    else:
         st.error("Model could not be loaded. Please check the model path and ensure NLTK data is available.")
