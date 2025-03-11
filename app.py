import streamlit as st
import pandas as pd
import openai

# 🎯 Streamlit 웹 앱 설정
st.set_page_config(page_title="특허 분류", layout="wide")

# 🔑 OpenAI API 키 입력
st.title("📂 LLM 기반 특허명 표준산업기술분류 앱")
api_key = st.text_input("🔑 API 키를 입력하세요", type="password")

# 📂 표준산업기술분류표 (GitHub에서 읽음)
@st.cache_data
def load_category_data():
    return pd.read_csv("category.csv", encoding="utf-8")

category_df = load_category_data()

# 📂 CSV 파일 업로드 
uploaded_file = st.file_uploader("📂 특허명 CSV 파일을 업로드하세요", type="csv")

if api_key and uploaded_file:
    # ✅ 최신 OpenAI API 방식 적용
    client = openai.OpenAI(api_key=api_key)

    # 📊 CSV 데이터 로드
    df = pd.read_csv(uploaded_file, encoding="utf-8")

    # ✅ 데이터 정리: 개행 문자 및 탭 문자 제거
    df = df.applymap(lambda x: x.replace("\n", " ").replace("\t", " ") if isinstance(x, str) else x)

    # ✅ CSS 적용 (긴 문자열 자동 줄바꿈)
    st.markdown("""
        <style>
        div[data-testid="stDataFrame"] div[role="gridcell"] {
            white-space: normal !important;
            word-wrap: break-word !important;
            overflow-wrap: break-word !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # ✅ 긴 문자열이 잘리지 않도록 설정
    st.write("📊 **업로드된 CSV 데이터**")
    st.dataframe(df, height=600, use_container_width=True)  # ✅ 너비 자동 확장

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
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.2
        )
        return response.choices[0].message.content.strip()

    def classify_mid_category(text, major_category, df):
        mid_categories = df[df['대분류'] == major_category]['중분류'].unique().tolist()
        prompt = f"""
        특허명: {text}  
        이 특허는 **'{major_category}' 대분류**에 속합니다.  
        아래 목록에서 **가장 적절한 중분류 하나만** 출력하세요.  

        가능 목록:
        {', '.join(mid_categories)}

        **출력: (오직 중분류 단어 하나만)**
        """
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.2
        )
        return response.choices[0].message.content.strip()

    def classify_sub_category(text, major_category, mid_category, df):
        sub_categories = df[(df['대분류'] == major_category) & (df['중분류'] == mid_category)]['소분류'].unique().tolist()
        prompt = f"""
        특허명: {text}  
        이 특허는 **'{major_category}' 대분류, '{mid_category}' 중분류**에 속합니다.  
        아래 목록에서 **가장 적절한 소분류 하나만** 출력하세요.  

        가능 목록:
        {', '.join(sub_categories)}

        **출력: (오직 소분류 단어 하나만)**
        """
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.2
        )
        return response.choices[0].message.content.strip()

    # ✅ 대/중/소 분류 적용
    def classify_patent(row):
        text = row["특허명"]
        major_category = classify_major_category(text, category_df['대분류'].unique().tolist())
        mid_category = classify_mid_category(text, major_category, category_df)
        sub_category = classify_sub_category(text, major_category, mid_category, category_df)
        return pd.Series([major_category, mid_category, sub_category])

    # ✅ 데이터프레임에 적용 (대/중/소분류 모두 저장)
    df[['대분류', '중분류', '소분류']] = df.apply(classify_patent, axis=1)

    # ✅ 결과 확인: 데이터프레임 출력 (끊김 없이 정상 출력)
    st.write("📊 **분류된 데이터**")
    st.dataframe(df, height=600, use_container_width=True)  # ✅ 너비 자동 확장

    # ✅ 결과 저장 및 다운로드
    output_file = "processed_patents.csv"
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    st.download_button(label="📥 결과 CSV 다운로드", data=open(output_file, "rb"), file_name="processed_patents.csv")
