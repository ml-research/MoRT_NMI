from sentence_transformers import SentenceTransformer

st = SentenceTransformer("average_word_embeddings_glove.840B.300d")

enc = st.encode("This is a test")
print(enc)