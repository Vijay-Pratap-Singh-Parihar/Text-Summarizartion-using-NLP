import streamlit as st 
from textblob import TextBlob 
import spacy

# Sumy Summary Pkg
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

# Function for Sumy Summarization
def sumy_summarizer(docx):
	parser = PlaintextParser.from_string(docx, Tokenizer("english"))
	text_rank_summarizer = TextRankSummarizer()
	summary = text_rank_summarizer(parser.document, sentences_count=3)
	summary_list = [str(sentence) for sentence in summary]
	result = ' '.join(summary_list)
	return result

# Function to Analyse Tokens and Lemma
@st.cache
def text_analyzer(my_text):
	nlp = spacy.load('en')
	docx = nlp(my_text)
	allData = [('"Token":{},\n"Lemma":{}'.format(token.text, token.lemma_)) for token in docx]
	return allData

# Function For Extracting Entities
@st.cache
def entity_analyzer(my_text):
	nlp = spacy.load('en')
	docx = nlp(my_text)
	tokens = [ token.text for token in docx]
	entities = [(entity.text, entity.label_) for entity in docx.ents]
	allData = ['"Token":{},\n"Entities":{}'.format(tokens,entities)]
	return allData


from summa.summarizer import summarize as summa_summarizer # import summa_summarizer

def main():
    """ NLP Based App with Streamlit """
    # Title
    st.title("TEXT SUMMARIZATION USING NLP")
    st.subheader("Natural Language Processing On the Go..")
    st.markdown("""
        #### Description
        + This is a Natural Language Processing(NLP) Based App useful for basic NLP task Sentiment,Summarization
        """)

    # # Tokenization
    # if st.checkbox("Show Tokens and Lemma"):
    #     st.subheader("Tokenize Your Text")
    #     message = st.text_area("Enter Text", "Type Here ..")
    #     if st.button("Analyze"):
    #         nlp_result = text_analyzer(message)
    #         st.json(nlp_result)

    # # Entity Extraction
    # if st.checkbox("Show Named Entities"):
    #     st.subheader("Analyze Your Text")
    #     message = st.text_area("Enter Text", "Type Here ..")
    #     if st.button("Extract"):
    #         entity_result = entity_analyzer(message)
    #         st.json(entity_result)

    # Sentiment Analysis
    if st.checkbox("Show Sentiment Analysis"):
        st.subheader("Analyze Your Text")
        message = st.text_area("Enter Text", "Type Here ..")
        if st.button("Analyze"):
            blob = TextBlob(message)
            result_sentiment = blob.sentiment
            st.success(result_sentiment)

    # Summarization
    if st.checkbox("Show Text Summarization"):
        st.subheader("Summarize Your Text")
        message = st.text_area("Enter Text", "Type Here ..")
        summary_options = st.selectbox("Choose Summarizer", ["sumy", "summa"]) # replace gensim with summa
        if st.button("Summarize"):
            if summary_options == "sumy":
                st.text("Using Sumy Summarizer ..")
                summary_result = sumy_summarizer(message)
            elif summary_options == "summa":
                st.text("Using Summa Summarizer ..")
                summary_result = summa_summarizer(message) 
            else:
                st.warning("Using Default Summarizer")
                st.text("Using Summa Summarizer ..")
                summary_result = summa_summarizer(message) 
            st.success(summary_result)

    st.sidebar.subheader("About App")
    # st.sidebar.text("")
    st.sidebar.info("Text Summarization using NLP")

    st.sidebar.subheader("By")
    st.sidebar.text("Vijay Pratap Singh Parihar")
    st.sidebar.text("Vasudha Bhat")
    st.sidebar.text("Vaibhav Shilpi")
    st.sidebar.text("Vijay Chauhan")



if __name__ == "__main__":
    main()
