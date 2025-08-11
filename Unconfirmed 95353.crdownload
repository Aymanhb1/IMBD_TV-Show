import streamlit as st
import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
import os

# Define the path to the saved model
# Make sure this path is correct relative to where you run the script,
# or provide the absolute path.
# If running locally after downloading from Colab, it might be './saved_sentiment_model'
# If deploying elsewhere, adjust the path accordingly.
SAVE_DIRECTORY = "./saved_sentiment_model"

# Load the trained model and tokenizer
@st.cache_resource # Cache the model to avoid reloading on each interaction
def load_model(save_directory):
    try:
        model = TFBertForSequenceClassification.from_pretrained(save_directory)
        tokenizer = BertTokenizer.from_pretrained(save_directory) # Load tokenizer if saved
        st.success("Model and tokenizer loaded successfully!")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        st.info("Please ensure the model and tokenizer files are in the correct directory.")
        return None, None

model, tokenizer = load_model(SAVE_DIRECTORY)

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
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        from nltk.stem import WordNetLemmatizer

        # Download necessary NLTK data (if not already downloaded) - might need to handle this outside Streamlit for deployment
        try:
             nltk.data.find('corpora/stopwords')
        except nltk.downloader.DownloadError:
             nltk.download('stopwords', quiet=True)
        try:
             nltk.data.find('tokenizers/punkt')
        except nltk.downloader.DownloadError:
             nltk.download('punkt', quiet=True)
        try:
             nltk.data.find('corpora/wordnet')
        except nltk.downloader.DownloadError:
             nltk.download('wordnet', quiet=True)
        try:
             nltk.data.find('corpora/omw-1.4')
        except nltk.downloader.DownloadError:
             nltk.download('omw-1.4', quiet=True)


        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()

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
         st.error("Model could not be loaded. Please check the model path.")
