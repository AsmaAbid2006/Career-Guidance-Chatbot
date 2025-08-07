import streamlit as st
import joblib
import string

# Loading model and vectorizer
model = joblib.load("intent_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Preprocessing text
def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Setting full-width layout
st.set_page_config(page_title="Career Guidance Chatbot", layout="wide")

# Creating three columns: left (about), center (chatbot), right (examples)
left_col, center_col, right_col = st.columns([1.5, 2.5, 1.5])

# ---------- LEFT COLUMN (About Section) ----------
with left_col:
    with st.expander("📘 About this Chatbot"):
     st.markdown("""
    This AI-powered career guidance chatbot helps you explore potential career paths
    based on your interests, skills, or personality.  
    
    Type your question or a description of your interests — and get matched with a suitable career role.

    ✅ Built with Machine Learning  
    ✅ Covers 50+ tech and non-tech roles  
    ✅ Ideal for students, job seekers, and explorers
    """)

# ---------- CENTER COLUMN (Main Chatbot UI) ----------
with center_col:
    st.title("🎓 Welcome to our Career Guidance Chatbot")
    st.markdown("Type your interests, goals, or curiosities to get a career suggestion:")

    user_input = st.text_input("📝 Your question or interest:")

    if user_input:
        cleaned = preprocess(user_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]
        st.success(f"🔍 **Best Suggested Career Role according to your input is:** {prediction}")

# ---------- RIGHT COLUMN (Sample Questions) ----------
with right_col:
    with st.expander("💡 Example questions you can ask"):
     st.markdown(""" 
    #### 🚀 Technical Interests
    - I love analyzing data and solving math problems.
    - What career is best for someone interested in artificial intelligence?
    - I enjoy building software and writing code.
    - I'm passionate about ethical hacking and cybersecurity.

    #### 🎨 Creative Skills
    - I like drawing, designing interfaces, and working with colors.
    - What job suits someone who enjoys video editing and animation?
    - I love making music and mixing audio.

    #### 🧠 Thinking & Strategy
    - I’m good at planning and managing teams.
    - I like organizing workflows and increasing productivity.
    - I enjoy researching and writing reports.

    #### 🗣 Communication & People Skills
    - I enjoy public speaking and presenting ideas.
    - What can I become if I love teaching others?
    - I like interviewing people and telling stories.

    #### ❤️ Personality-Based Prompts
    - I'm a creative thinker and love solving real-world problems.
    - I like working independently and learning new tech.
    - I'm very social and love working in teams.

    #### 🤔 Not Sure Yet?
    - I don’t know what I want to do — can you help?
    - Suggest a career path based on my interests.
    - Which tech field fits me best?
    """)
