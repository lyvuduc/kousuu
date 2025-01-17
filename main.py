import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import pickle  # モデルをロードするため
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.lines as mlines



# Define the category mapping (fallback for unknown categories)
CATEGORY_MAP = {
    '会議': 'Meeting',
    '研修': 'Training',
    '開発': 'Development',
    '休暇': 'Vacation',
    '有休': 'Vacation',
}

# 学習済みモデルとベクトライザーのロード
with open('trained_model.pkl', 'rb') as model_file, open('vectorizer.pkl', 'rb') as vec_file:
    trained_model = pickle.load(model_file)
    vectorizer = pickle.load(vec_file)

# 学習済みモデルを使ったカテゴリー推測関数
def predict_category(件名):
    件名_vectorized = vectorizer.transform([件名])  # ベクトル化
    predicted_category = trained_model.predict(件名_vectorized)[0]
    return predicted_category

# 件名の分類ルール（学習済みモデルと連携）
def map_category(件名):
    if not isinstance(件名, str):  # 件名が文字列でない場合（NaNやfloatを含む）
        return 'Unknown'
    try:
        # 学習済みモデルで予測
        return predict_category(件名)
    except Exception:
        # 学習済みモデルでエラーの場合、ルールベースで判定
        if '開発' in 件名:
            return 'Development'
        elif '会' in 件名 or '打ち合わせ' in 件名 or 'ミーティング' in 件名:
            return 'Meeting'
        elif '研修' in 件名:
            return 'Training'
        elif '有休' in 件名:
            return 'Vacation'
        else:
            return 'Other'

# CSVデータを読み込む
def load_data(file):
    return pd.read_csv(file)

# カテゴリーを分類し、工数を計算する
def categorize_and_add_hours(df):
    df['件名'] = df['件名'].astype(str)  # 件名列を文字列型に変換
    df['Category'] = df['件名'].apply(map_category)
    df['Start'] = pd.to_datetime(df['開始日'] + ' ' + df['開始時刻'])
    df['End'] = pd.to_datetime(df['終了日'] + ' ' + df['終了時刻'])
    df['Duration'] = abs((df['End'] - df['Start']).dt.total_seconds()) / 3600
    df['Month'] = pd.to_datetime(df['開始日']).dt.month
    df['Date'] = pd.to_datetime(df['開始日']).dt.date
    return df

def aggregate_hours_by_month(df, start_month, end_month):
    df_filtered = df[(df['Month'] >= start_month) & (df['Month'] <= end_month)]
    return df_filtered.groupby(['Month', 'Category'])['Duration'].sum().unstack(fill_value=0)

def aggregate_hours_by_day(df, start_date, end_date):
    df_filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    return df_filtered.groupby(['Date', 'Category'])['Duration'].sum().unstack(fill_value=0)

# 横軸に "hours" 単位を追加した棒グラフ
def plot_bar_chart_with_unit(data, title, x_label, y_label):
    # 日本語カテゴリを英語に変換するマッピング
    category_translation = {
        '会議': 'Meeting',
        '研修': 'Training',
        '開発': 'Development',
        '休暇': 'Vacation',
        '有休': 'Vacation',
        'その他': 'Other'
    }

    # カテゴリラベルを英語に変換
    translated_data = data.rename(columns=category_translation)

    plt.figure(figsize=(10, 6))
    
    # 棒グラフを生成
    ax = translated_data.plot(kind='bar', stacked=True, alpha=0.7, colormap="tab10", figsize=(10, 6))
    
    # グラフの設定
    plt.title(title, fontsize=15)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    
    # 凡例の再設定（英語ラベルを使用）
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title="Category", bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    
    # レイアウト調整
    plt.tight_layout()
    
    # Streamlit 上で表示
    st.pyplot(plt)


# 積み上げエリアチャート
def plot_stacked_area_chart(data, title, x_label):
    data_percentage = data.div(data.sum(axis=1), axis=0) * 100  # Convert to percentage
    plt.figure(figsize=(24, 10))
    ax = data_percentage.plot(kind='area', stacked=True, alpha=0.7, colormap="tab10")
    plt.title(title, fontsize=15)
    plt.ylabel("Percentage (%)")
    plt.xlabel(x_label)
    handles, labels = ax.get_legend_handles_labels()
    english_category_names = {
        '会議': 'Meeting',
        '研修': 'Training',
        '開発': 'Development',
        '休暇': 'Vacation',
        '有休': 'Vacation',
        'その他': 'Other'
    }
    translated_labels = [english_category_names.get(label, label) for label in labels]
    custom_handles = [
        mlines.Line2D([0], [0], marker='o', color='w', markerfacecolor=h.get_facecolor()[0], markersize=10) 
        for h in handles
    ]
    ax.legend(custom_handles, translated_labels, title="Category", bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    plt.tight_layout()
    st.pyplot(plt)

# アプリのデザイン
st.markdown("""
    <style>
    .stApp {
        background-color: #f5f5f5;
        font-family: 'Roboto', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# アプリの表示ロジック
def display_results():
    st.title("⏳ 工数自動入力アプリ ⏳")
    st.markdown("<h2 style='color:#FF6347;'>工数入力はめんどくさい → AIに任せて！</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📂 CSVファイルをアップロードしてください", type=["csv"])
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        df = categorize_and_add_hours(df)
        
        st.subheader("📅 月別工数")
        min_month, max_month = df['Month'].min(), df['Month'].max()
        start_month, end_month = st.slider("月範囲を選択", min_value=min_month, max_value=max_month, value=(min_month, max_month))
        result_month = aggregate_hours_by_month(df, start_month, end_month)
        plot_bar_chart_with_unit(result_month, "Monthly Work Hours Breakdown by Category", "Month", "Work Hours")
        
        st.subheader("📊 月別工数（積み上げエリアチャート）")
        result_month_area = aggregate_hours_by_month(df, start_month, end_month)
        plot_stacked_area_chart(result_month_area, "Work Hours Breakdown by Category (Percentage)", "Month")
        
        st.subheader("📅 日別工数")
        min_date, max_date = df['Date'].min(), df['Date'].max()
        start_date, end_date = st.slider("日範囲を選択", min_value=min_date, max_value=max_date, value=(min_date, max_date))
        result_day = aggregate_hours_by_day(df, start_date, end_date)
        st.bar_chart(result_day)
        
        st.subheader("📊 日別工数（％-積み上げエリアチャート）")
        result_day_area = aggregate_hours_by_day(df, start_date, end_date)
        plot_stacked_area_chart(result_day_area, "Daily Work Hours Breakdown by Category (Percentage)", "Date")

if __name__ == "__main__":
    display_results()
