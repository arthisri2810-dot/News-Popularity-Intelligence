import streamlit as st
import torch
import pickle
from transformers import AutoTokenizer, AutoModel

# ----------------------------------
# Page Configuration
# ----------------------------------
st.set_page_config(
    page_title="News Popularity Intelligence System",
    page_icon="📰",
    layout="centered"
)

# ----------------------------------
# Load Models
# ----------------------------------
@st.cache_resource
def load_models():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert_model = AutoModel.from_pretrained("bert-base-uncased")

    with open("popularity_model.pkl", "rb") as f:
        reg_model = pickle.load(f)

    return tokenizer, bert_model, reg_model

tokenizer, bert_model, reg_model = load_models()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)

# ----------------------------------
# Predict Popularity Score
# ----------------------------------
def predict_score(title, description):
    text = title + " " + description

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = bert_model(**inputs)

    cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    score = reg_model.predict(cls_embedding)[0]

    return float(score)

# ----------------------------------
# Predict Priority
# ----------------------------------
def predict_priority(title, description):
    text = (title + " " + description).lower()
    score = predict_score(title, description)

    # HIGH
    high_keywords = [
        "world cup", "historic", "record",
        "scientist", "scientists", "ai system",
        "cancer", "breakthrough",
        "earthquake", "flood", "cyclone",
        "prime minister", "national",
        "global", "international",
        "space mission", "satellite", "disaster"
    ]
    if any(word in text for word in high_keywords):
        return score, "HIGH", "🟢", "90%"

    # LOW
    low_keywords = [
        "college", "hostel", "department",
        "library", "school", "apartment",
        "association", "monthly meeting",
        "parent-teacher", "local",
        "routine", "internal", "residential",
        "community center", "yoga session"
    ]
    if any(word in text for word in low_keywords):
        return score, "LOW", "🔴", "75%"

    # MEDIUM
    medium_keywords = [
        "government", "state government",
        "scholarship", "education policy",
        "training program", "skill development",
        "employment scheme", "startup",
        "business", "funding", "expands",
        "digital portal", "policy", "scheme"
    ]
    if any(word in text for word in medium_keywords):
        return score, "MEDIUM", "🟡", "80%"

    # Fallback
    if score < 0.9:
        return score, "LOW", "🔴", "70%"
    elif score < 1.2:
        return score, "MEDIUM", "🟡", "80%"
    else:
        return score, "HIGH", "🟢", "85%"

# ----------------------------------
# Sidebar Navigation
# ----------------------------------
page = st.sidebar.radio(
    "📌 Navigation",
    ["🏠 Home", "📰 News Intelligence", "🧠 Model Reasoning"]
)

# =================================================
# PAGE 1: HOME
# =================================================
if page == "🏠 Home":

    st.title("📰 News Popularity Intelligence System")

    st.subheader("🔍 Problem Overview")
    st.write(
        "Predicting the popularity of news articles is challenging because "
        "real engagement metrics such as likes, shares, and views are often unavailable."
    )

    st.subheader("❓ Why Popularity Labels Are Unavailable")
    st.write(
        "- Social media platforms restrict engagement data\n"
        "- News datasets rarely contain popularity labels\n"
        "- Popularity varies across platforms and time"
    )

    st.subheader("🏗 System Architecture")
    st.code(
        """
        Title + Description
                ↓
           BERT Encoder
                ↓
         CLS Embedding
                ↓
       Regression Model
                ↓
     Popularity Score
                ↓
      Priority Level
        """
    )

# =================================================
# PAGE 2: NEWS INTELLIGENCE
# =================================================
elif page == "📰 News Intelligence":

    st.title("📰 News Intelligence")

    title = st.text_input("Enter News Title")
    description = st.text_area("Enter News Description")

    if st.button("🚀 Analyze News"):

        if title and description:

            with st.spinner("Analyzing news content..."):

                score, level, color, confidence = predict_priority(title, description)
                score = round(score, 2)

                st.subheader("📊 Output")
                st.write(f"**Popularity Score:** {score}")
                st.markdown(f"### {color} Priority Level: **{level}**")

                st.subheader("🧠 Key Explanatory Highlights")

                if level == "HIGH":
                    st.write(
                        "- Contains national or global impact keywords\n"
                        "- High semantic importance\n"
                        "- Strong engagement potential"
                    )
                elif level == "MEDIUM":
                    st.write(
                        "- Government, policy, or business related\n"
                        "- Moderate public relevance\n"
                        "- Informative content"
                    )
                else:
                    st.write(
                        "- Institutional or local updates\n"
                        "- Limited audience reach\n"
                        "- Routine information"
                    )

        else:
            st.warning("Please enter both title and description.")

# =================================================
# PAGE 3: MODEL REASONING
# =================================================
elif page == "🧠 Model Reasoning":

    st.title("🧠 Model Reasoning")

    st.subheader("⚙ Scoring Logic")
    st.write(
        "- Text is converted into embeddings using BERT\n"
        "- CLS token captures overall meaning\n"
        "- Regression model predicts popularity score"
    )

    st.subheader("📊 Example Comparisons")
    st.write(
        "**College Library Extends Working Hours** → LOW\n\n"
        "**Government Launches Skill Program** → MEDIUM\n\n"
        "**India Wins Cricket World Cup** → HIGH"
    )

    st.subheader("🔎 Model Interpretation")
    st.write(
        "- Model measures textual richness\n"
        "- Does not use real-time social media data\n"
        "- Rule-based logic improves clarity"
    )

    st.subheader("⚠ Limitations")
    st.write(
        "- No real engagement labels\n"
        "- Topic popularity may vary over time\n"
        "- Can be improved with labeled datasets"
    )
