import streamlit as st
import pandas as pd
from docx import Document
from collections import Counter
import nltk

# โหลดข้อมูลภาษาศาสตร์ของ NLTK (ใช้ @st.cache_resource เพื่อไม่ให้โหลดซ้ำซ้อน)
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('averaged_perceptron_tagger_eng')

download_nltk_data()

st.title("🔤 Word Frequency & POS Analyzer")
st.write("อัปเกรดใหม่: วิเคราะห์ความถี่และแยกประเภทคำ (Part of Speech) ช่วยนักแปลทำ Glossary ง่ายขึ้น!")

# ให้ผู้ใช้เลือกว่าจะกรองคำประเภทไหน
pos_option = st.radio(
    "เลือกประเภทคำที่ต้องการวิเคราะห์:",
    ("ทั้งหมด (All)", "เฉพาะคำนาม (Nouns)", "เฉพาะคำกริยา (Verbs)", "เฉพาะคำคุณศัพท์ (Adjectives)")
)

uploaded_file = st.file_uploader("อัปโหลดเอกสารภาษาอังกฤษ (.txt หรือ .docx)", type=["txt", "docx"])

if uploaded_file is not None:
    text = ""
    if uploaded_file.name.endswith(".txt"):
        text = uploaded_file.read().decode("utf-8")
    elif uploaded_file.name.endswith(".docx"):
        doc = Document(uploaded_file)
        text = "\n".join([para.text for para in doc.paragraphs])

    st.info("กำลังวิเคราะห์ Part of Speech...")
    
    # แยกคำและล้างเครื่องหมายวรรคตอน
    words = nltk.word_tokenize(text)
    words = [w.lower() for w in words if w.isalpha()]
    
    # วิเคราะห์ Part of Speech (POS Tagging)
    pos_tags = nltk.pos_tag(words)
    
    # กรองประเภทคำตามที่ผู้ใช้เลือก
    filtered_words = []
    for word, tag in pos_tags:
        if pos_option == "เฉพาะคำนาม (Nouns)" and tag.startswith('NN'):
            filtered_words.append(word)
        elif pos_option == "เฉพาะคำกริยา (Verbs)" and tag.startswith('VB'):
            filtered_words.append(word)
        elif pos_option == "เฉพาะคำคุณศัพท์ (Adjectives)" and tag.startswith('JJ'):
            filtered_words.append(word)
        elif pos_option == "ทั้งหมด (All)":
            filtered_words.append(word)
            
    # ตัด Stopwords พื้นฐาน
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or', 'it', 'that', 'this', 'by', 'with', 'as', 'be', 'from'}
    final_words = [w for w in filtered_words if w not in stopwords and len(w) > 1]

    # นับจำนวนและแสดงผล
    word_counts = Counter(final_words)
    df = pd.DataFrame(word_counts.most_common(30), columns=["คำศัพท์", "ความถี่"])

    if not df.empty:
        st.subheader(f"📊 Top 30 {pos_option} ที่ใช้บ่อยที่สุด")
        st.dataframe(df)
        st.bar_chart(df.set_index("คำศัพท์"))

        # ปุ่มดาวน์โหลด
        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 ดาวน์โหลดผลลัพธ์ (CSV)", data=csv, file_name='pos_frequency.csv', mime='text/csv')
    else:
        st.warning("ไม่พบคำที่ตรงกับเงื่อนไข ลองเปลี่ยนประเภทคำดูนะครับ")
