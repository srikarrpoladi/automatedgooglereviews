import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.util import ngrams
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import itertools
import streamlit as st


st.set_option('deprecation.showPyplotGlobalUse', False)



nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('vader_lexicon')
nltk.download('wordnet')


def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

# Sentiment Analysis
def classify_sentiments(df, column_name):
    sid = SentimentIntensityAnalyzer()

    df['vader_scores'] = df[column_name].apply(sid.polarity_scores)
    df['compound_score'] = df['vader_scores'].apply(lambda x: x['compound'])
    
    df['sentiment'] = df['compound_score'].apply(
        lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral')
    )

    return df

# Text Cleaning Function
def clean_reviews(df, column_name, min_length=5):
    df_copy = df.copy()

    pattern = r'[^\w\s]'
    df_copy[column_name] = df_copy[column_name].replace(pattern, '', regex=True)
    df_copy[column_name] = df_copy[column_name].str.lower()

    df_copy = df_copy.drop_duplicates().dropna().reset_index(drop=True)
    df_copy = df_copy[df_copy[column_name].str.len() >= min_length].reset_index(drop=True)

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def clean_text(text):
        words = word_tokenize(text)
        filtered_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        return ' '.join(filtered_words)

    df_copy['cleaned_text'] = df_copy[column_name].apply(clean_text)
    df_copy['tokenized'] = df_copy['cleaned_text'].apply(word_tokenize)

    return df_copy

# Sentiment Distribution Plot
def plot_sentiment_distribution(df):
    sentiment_counts = df['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['sentiment', 'count']

    plt.figure(figsize=(8, 6))
    sns.barplot(data=sentiment_counts, x='sentiment', y='count', palette='viridis')
    plt.title('Distribution of Sentiment Categories')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    st.pyplot()

# Word Frequency Analysis
def word_frequency_analysis(df, column_name, top_n=10):
    all_reviews = ' '.join(df[column_name])
    all_words = all_reviews.split()
    word_counts = Counter(all_words)
    
    word_freq_df = pd.DataFrame(word_counts.most_common(), columns=['Word', 'Frequency'])
    top_words = word_freq_df.head(top_n)
    
    plt.figure(figsize=(10, 6))
    plt.bar(top_words['Word'], top_words['Frequency'], color='skyblue')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title(f'Top {top_n} Frequent Words')
    plt.xticks(rotation=45)
    st.pyplot()
    
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_reviews)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Word Cloud")
    st.pyplot()

# Topic Modeling with LDA
def perform_topic_modeling(df, column_name, n_components=5, max_features=1000):
    count_vectorizer = CountVectorizer(max_features=max_features, stop_words='english')
    dt_matrix = count_vectorizer.fit_transform(df[column_name])

    lda = LatentDirichletAllocation(n_components=n_components, random_state=42)
    lda.fit(dt_matrix)

    words = count_vectorizer.get_feature_names_out()
    topics = []

    for idx, topic in enumerate(lda.components_):
        topics.append([words[i] for i in topic.argsort()[-10:]])

    return topics

# Aspect-Based Analysis
def label_themes(text, themes):
    aspects = []
    for theme, keywords in themes.items():
        if any(keyword in text for keyword in keywords):
            aspects.append(theme)
    return aspects

# Perform Aspect Analysis
def perform_aspect_analysis(df, column_name, themes):
    df['themes'] = df[column_name].apply(lambda x: label_themes(x, themes))

    theme_counts = Counter(itertools.chain(*df['themes']))
    
    return theme_counts

def extract_phrases(text, n):
            tokens = text.split()
            return list(ngrams(tokens, n))

def analyze_sentiment_clusters(df, text_column, sentiment_column, n=2):
    sentiment_groups = df.groupby(sentiment_column).size().reset_index(name='count')
    st.write("Sentiment Counts:")
    st.dataframe(sentiment_groups)

    def extract_phrases(text, n):
        tokens = text.split()
        return list(ngrams(tokens, n))
    
    st.write("Frequent Phrases by Sentiment:")
    for sentiment in ['positive', 'neutral', 'negative']:
        reviews = ' '.join(df[df[sentiment_column] == sentiment][text_column])
        phrases = extract_phrases(reviews, n)
        st.write(f"Frequent phrases for {sentiment}:")
        st.write(pd.DataFrame(Counter(phrases).most_common(10), columns=['Phrase', 'Frequency']))

def plot_theme_distribution(theme_counts):
    plt.figure(figsize=(8, 6))
    sns.barplot(x=list(theme_counts.keys()), y=list(theme_counts.values()), palette='coolwarm')
    plt.title('Theme Distribution')
    plt.xlabel('Themes')
    plt.ylabel('Count')
    st.pyplot()

# Streamlit UI
def main():
    st.title("Customer Review Analysis")
    st.sidebar.header("File Upload")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file:
        df = load_data(uploaded_file)

        st.write("### Raw Data", df.head())

        df_cleaned = clean_reviews(df, column_name='reviews')
        
        df_cleaned = classify_sentiments(df_cleaned, column_name='reviews')
        plot_sentiment_distribution(df_cleaned)

        word_frequency_analysis(df_cleaned, column_name='cleaned_text')

        n_topics = st.sidebar.slider("Number of Topics for LDA", 2, 10, 5)
        topics = perform_topic_modeling(df_cleaned, column_name='cleaned_text', n_components=n_topics)
        st.write("### Topics Identified:", topics)



        themes = {
            'food': ['delicious', 'tasty', 'flavor', 'dish', 'shawarma', 'rice', 'chicken', 'bowl', 'eat', 'spicy'],
            'service': ['friendly', 'rude', 'wait', 'helpful', 'positive', 'thank', 'rating', 'appreciate'],
            'ambiance': ['clean', 'cozy', 'atmosphere', 'decor', 'wonderful', 'experience']
        }
        theme_counts = perform_aspect_analysis(df_cleaned, column_name='cleaned_text', themes=themes)
        plot_theme_distribution(theme_counts)

        # Sentiment Clustering and Frequent Phrases
        st.write("### Sentiment Clustering and Frequent Phrases:")
        sentiment_groups = df_cleaned.groupby('sentiment').size().reset_index(name='count')
        st.write("Sentiment Counts:")
        st.dataframe(sentiment_groups)
        
        st.write("Frequent Phrases by Sentiment:")
        for sentiment in ['positive', 'neutral', 'negative']:
            reviews = ' '.join(df_cleaned[df_cleaned['sentiment'] == sentiment]['cleaned_text'])
            phrases = extract_phrases(reviews, n=2)
            frequent_phrases = Counter(phrases).most_common(10)
            
            st.write(f"#### {sentiment.capitalize()} Sentiment:")
            st.table(pd.DataFrame(frequent_phrases, columns=['Phrase', 'Frequency']))


        if st.sidebar.button("Export Cleaned Data"):
            export_filename = "cleaned_reviews.csv"
            df_cleaned.to_csv(export_filename, index=False)
            st.success(f"Data exported to {export_filename}")

if __name__ == "__main__":
    main()