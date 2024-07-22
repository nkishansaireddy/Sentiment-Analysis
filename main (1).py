import streamlit as st
import pandas as pd
from transformers import pipeline
import base64

# Load sentiment analysis pipeline
nlp = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")


# Define function to get sentiment score for a given text
def get_sentiment(text):
    # Add special tokens to indicate the beginning and end of the text
    text = f"<s>{text}</s>"

    # Truncate the text to the maximum length allowed by the RoBERTa model
    max_length = nlp.tokenizer.max_model_input_sizes['roberta-base']
    text = text[:max_length - 2] + '</s>'

    result = nlp(text)[0]
    label_id = result['label']

    # Map the output label to the corresponding sentiment label
    if label_id == 'LABEL_0':
        label = 'negative'
    elif label_id == 'LABEL_2':
        label = 'positive'
    else:
        label = 'neutral'

    score = result['score']
    return label, score


# search for a specific word in dataset
def find_word(word, df):
    neg_score = 0;
    pos_score = 0;
    neu_score = 0
    word = word.lower()

    for index, row in df.iterrows():

        text = row['texts'].lower()
        if word in text.split():

            if row['sentiment_label'] == 'negative':
                neg_score += 1

            elif row['sentiment_label'] == 'positive':
                pos_score += 1

            else:
                neu_score += 1

    return pos_score, neg_score, neu_score


# Create a button to download the dataset
def download_button(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="dataset_after_sentiment.csv">Download New dataset</a>'
    return href


# Define Streamlit app
def main():
    # --------------------------------- First Section ---------------------------------

    # Set page style and title
    st.set_page_config(page_title='SenTrake | Sentiment Analysis', page_icon=':memo:', layout="wide",
                       initial_sidebar_state="expanded")

    # Add custom CSS to the app
    st.markdown(
        """
        <style>
            /* Change the border color and hover text color of the upload button */
            div.stFileUploader:hover button {
                border-color: #f7ac54;
            }
    
            div.stFileUploader button:hover span {
                color: #f7ac54;
            }
        body {
            background-image: url("https://cdn.dorik.com/6231c5ff317f6e0012e1bed0/64456811304170004cf83658/images/milad-fakurian-VbC-EiOTDqA-unsplash_kaus0fr0.jpg");
            background-attachment: fixed;
            background-size: cover;
            color: white;
            font-family: sans-serif;
        }
        </style>
        """,
        unsafe_allow_html=True)

    # Add header
    st.header(':dart: SenTrack | Sentiment Analysis')

    st.write(
        '#### <span style="color:#f7ac54"> Important Note:</span> Please make sure your target texts are inside the column called "texts".',
        unsafe_allow_html=True)

    # Get file from user
    uploaded_file = st.file_uploader('Upload CSV file', type=['csv'])

    if uploaded_file is not None:

        # Read CSV file into DataFrame
        df = pd.read_csv(uploaded_file)

        # Add sentiment column to DataFrame
        df[['sentiment_label', 'sentiment_score']] = df['texts'].apply(get_sentiment).apply(pd.Series)

        # Show DataFrame in Streamlit
        st.write(df)

        # download Button
        st.markdown(download_button(df), unsafe_allow_html=True)

        # --------------------------- Second Section -----------------------------------

        # Add some space
        st.write('')

        # Add a horizontal line
        st.write('---')

        # Add header
        st.header(':traffic_light: Sentiment Label Percentages')
        # Define a dictionary of colors for each sentiment
        color_dict = {
            'positive': '#00FF00',  # Green
            'negative': '#FF0000',  # Red
            'neutral': '#FFFF00',  # Yellow
        }

        # Count the number of positive and negative labels in the column
        counts = df['sentiment_label'].value_counts()

        # Calculate the percentage for each label
        total = counts.sum()
        percentages = [count / total * 100 for count in counts]

        # Print the label with the percentage and size based on percentage
        for i, label in enumerate(counts.index):
            pct = percentages[i]
            color = color_dict[label]  # Look up the color from the dict
            size = int(pct * 3)  # Scale the percentage to a font size (adjust 3 as needed)
            styled_text = f'<span style="color:{color}; font-size:{size}px">{label}</span>'  # Apply the color and size
            st.write(f'{styled_text}     {pct:.1f}%', unsafe_allow_html=True)

        # -------------------------------- Third Section -------------------------------------

        # Add some space
        st.write('')

        # Add a horizontal line
        st.write('---')

        # get more information about specific word
        st.header(":mag: Search for any word/s and see how your customers talk about.")
        user_input = st.text_input("Enter word/s separated by spaces for more knowledge")
        words_list = user_input.split()
        for word in words_list:
            positive, negative, neutral = find_word(word, df)
            st.write(f"Results For {word}: ")
            st.write("Positive occurrence: ", positive, ' Negative occurrence: ', negative, " Neutral occurrence: ",
                     neutral)


# Run Streamlit app
if __name__ == '__main__':
    main()
