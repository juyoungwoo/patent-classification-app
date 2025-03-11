import streamlit as st
import pandas as pd
import openai

# 🎯 Streamlit 웹 앱 설정
st.set_page_config(page_title="특허 분류", layout="wide")

# 🔑 OpenAI API 키 입력
st.title("📂 OpenAI 기반 특허 분류 웹 앱")
api_key = st.text_input("🔑 OpenAI API 키를 입력하세요", type="password")

# 📂 표준산업기술분류표 (GitHub에서 읽음)
@st.cache_data
def load_category_data():
    return pd.read_csv("category.csv", encoding="utf-8-sig")

category_df = load_category_data()

# 📂 CSV 파일 업로드
uploaded_file = st.file_uploader("📂 CSV 파일을 업로드하세요", type="csv")

if api_key and uploaded_file:
    openai.api_key = api_key

    # 📊 CSV 데이터 로드
    df = pd.read_csv(uploaded_file)
    st.write("업로드된 CSV 데이터", df.head())

    # 🔍 LLM 기반 분류 함수
    def classify_major_category(text, categories):
        prompt = f"""
        다음 특허명을 보고 가장 적절한 대분류를 선택하세요.  
        반드시 아래 목록 중 하나만 선택해야 합니다.  

        가능 목록:
        {', '.join(categories)}

        특허명: {text}  
        **출력: (오직 대분류 단어 하나만)**
        """
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.2
        )
        return response["choices"][0]["message"]["content"].strip()

    # 📊 대/중/소 분류 실행
    df['대분류'] = df['특허명'].apply(lambda x: classify_major_category(x, category_df['대분류'].unique().tolist()))

    # 결과 저장 및 다운로드
    output_file = "processed_patents.csv"
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    st.download_button(label="📥 결과 CSV 다운로드", data=open(output_file, "rb"), file_name="processed_patents.csv")
