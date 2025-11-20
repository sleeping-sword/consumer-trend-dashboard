import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

import matplotlib
import matplotlib.font_manager as fm

font_path = "NotoSansTC-Regular.otf"  # 你剛放的字體檔
fm.fontManager.addfont(font_path)
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans TC']
matplotlib.rcParams['axes.unicode_minus'] = False

'''
LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=1, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 取最後一個時間步的輸出
        out = self.fc(out)
        return out

# -------------------------
# 訓練函式
# -------------------------
def train_lstm(df, feature_cols, target_col='SalesQty', seq_length=10, epochs=50, lr=0.01, model_path="lstm_model.pth"):
    """
    訓練 LSTM 模型並保存
    """
    # 選擇數值型欄位
    features_df = df[feature_cols].select_dtypes(include=[np.number])
    target_df = df[[target_col]].select_dtypes(include=[np.number])
    
    # 標準化
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(features_df)
    y_scaled = scaler_y.fit_transform(target_df)
    
    # 建立序列資料
    X, y = [], []
    for i in range(len(X_scaled) - seq_length):
        X.append(X_scaled[i:i+seq_length])
        y.append(y_scaled[i+seq_length, 0])
    
    X = np.array(X)
    y = np.array(y)
    
    # 轉成 tensor
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y).unsqueeze(-1)
    
    # 建立模型
    input_size = X_tensor.shape[2]
    model = LSTMModel(input_size=input_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 訓練
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
    
    # 保存模型（只存 state_dict + feature list + scaler 參數）
    torch.save({
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_size": model.lstm.hidden_size,
        "num_layers": model.lstm.num_layers,
        "feature_cols": feature_cols,
        "scaler_X_min": scaler_X.min_,
        "scaler_X_scale": scaler_X.scale_,
        "scaler_y_min": scaler_y.min_,
        "scaler_y_scale": scaler_y.scale_
    }, model_path)
    
    print(f"模型已保存到 {model_path}")
    
    return model, scaler_X, scaler_y

# -------------------------
# 預測函式
# -------------------------
def predict_lstm(df, model_path, feature_cols, seq_len=10):
    """
    使用已保存模型預測
    """
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    # 重建模型架構
    model = LSTMModel(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"]
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    
    # 重建 scaler
    scaler_X = MinMaxScaler()
    scaler_X.min_ = checkpoint["scaler_X_min"]
    scaler_X.scale_ = checkpoint["scaler_X_scale"]
    
    scaler_y = MinMaxScaler()
    scaler_y.min_ = checkpoint["scaler_y_min"]
    scaler_y.scale_ = checkpoint["scaler_y_scale"]
    
    # 取特徵欄位資料並標準化
    data = df[feature_cols].select_dtypes(include=[np.number]).values
    data_scaled = scaler_X.transform(data)
    
    # 取最後 seq_len 筆資料
    if len(data_scaled) < seq_len:
        raise ValueError(f"資料不足，需要至少 {seq_len} 筆資料")
    recent_seq = data_scaled[-seq_len:]
    X = torch.FloatTensor(recent_seq).unsqueeze(0)  # (1, seq_len, features)
    
    # 預測
    with torch.no_grad():
        pred_scaled = model(X).numpy()
    
    pred = scaler_y.inverse_transform(pred_scaled)[0][0]
    print("預測值:", pred)
    return pred
'''

def linearPre(data, select):
    # 假設這是一個簡單的線性回歸模型來預測消費趨勢
    for word in select['欄位']:
        inputData = data[word].values.reshape(-1, 1)
    outputData = data['Sales'].values
    model = LinearRegression()
    model.fit(inputData, outputData)
    return model


st.set_page_config(page_title="消費趨勢智慧分析平台", layout="wide")

st.title("📊 消費趨勢智慧分析平台")

page = st.sidebar.selectbox(
    "功能選擇",
    ["可預測消費趨勢模型", "分析市場趨勢", "試算獲利潛力組合"]
)

# === 功能一：可預測消費趨勢模型 ===
if page == "可預測消費趨勢模型":
    st.subheader("📈 可預測消費趨勢模型")
    st.write("上傳包含 `date`（或月份）與 `sales` 欄位的 CSV，系統會自動畫出趨勢並預測下一期銷售量。")

    uploaded_file = st.file_uploader("📤 上傳銷售資料 CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("✅ 已成功讀取資料：")
        st.dataframe(df.head())
        row_values = df.values.flatten()
        new_df = pd.DataFrame([row_values])

                    # 3️⃣ 行列互換
        transposed_df = df.head(5)
        transposed_df = transposed_df.T  # 行列互換
        transposed_df.reset_index(inplace=True)  # 把 index 變成欄位
        transposed_df.rename(columns={"index": "欄位"}, inplace=True)  # 改名
            
            # 4️⃣ 加上行選取欄位
        transposed_df["_selected"] = False

            # 5️⃣ 顯示 DataEditor
        edited = st.data_editor(
                transposed_df,
                hide_index=True,
                width="stretch",
                column_config={
                    "_selected": st.column_config.CheckboxColumn("選取這行")
                },
                key="editor",
            )

            # 6️⃣ 取得選取的行
        selected_rows = edited[edited["_selected"] == True]

        st.subheader("你選到的『行』：")
        st.dataframe(selected_rows)
        # 日期欄位處理
        # 日期欄位處理（自動辨識大小寫）

        date_cols = [col for col in df.columns if col.lower() == 'date' or col == '月份']
        if date_cols:
            date_col = date_cols[0]  # 抓第一個符合的欄位名稱
            if date_col.lower() == 'date':
                df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col)
            df['time_index'] = np.arange(len(df))
        else:
            st.error("❌ 必須包含欄位 'date'、'Date' 或 '月份'")
            st.stop()

        # 趨勢線回歸預測（支援大小寫與同義字）
        sales_cols = [col for col in df.columns if any(k in col.lower() for k in ['sale', 'sales', 'revenue', 'amount', 'profit', '銷售', '營收'])]
        if st.button('開始預測'):
            sales_col = sales_cols[0]  # 抓第一個符合的欄位名稱
            model = linearPre(df, selected_rows)
            next_idx = [[len(df)]]
            prediction = model.predict(next_idx)[0]
            X = df[["time_index"]]
            pre_low = prediction * 0.98
            pre_high = prediction * 1.02
            # 畫圖
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(df['time_index'], df[sales_col], marker='o', label='實際銷售量')
            ax.plot(df['time_index'], model.predict(X), linestyle='--', color='orange', label='回歸趨勢線')
            plt.vlines(x=len(df), ymin=pre_low, ymax=pre_high, color='red', label='下一期預測區間')
            ax.scatter(len(df), pre_low, color='red')
            ax.scatter(len(df), pre_high, color='red')
            ax.set_xlabel("時間")
            ax.set_ylabel("銷售量")
            ax.set_title("銷售趨勢預測")
            ax.legend()
            st.pyplot(fig)
        
            st.success(f"📅 下一期預測銷售量：約為 **{pre_low:.0f}~{pre_high:.0f}** 單位")
        else:
            st.error("❌ 必須包含與銷售相關的欄位（如 'Sales', 'sale', '銷售額', '營收' 等）")


# === 功能二：分析市場趨勢 ===
elif page == "分析市場趨勢":
    st.subheader("📊 分析市場趨勢")
    st.write("分析不同地區或季節性需求變化。")

    regions = ['北部', '中部', '南部', '東部']
    spending = [50, 40, 70, 30]
    fig, ax = plt.subplots()
    ax.bar(regions, spending, color=['#007bff','#17a2b8','#28a745','#ffc107'])
    ax.set_ylabel("平均月支出（千元）")
    ax.set_title("地域性消費差異")
    st.pyplot(fig)

# === 功能三：試算獲利潛力組合 ===
else:
    st.subheader("💡 試算獲利最具潛力的品項或組合")
    st.write("根據產品特性與價格彈性模擬不同策略。")

    price = st.slider("產品價格 (元)", 50, 500, 200, step=10)
    discount = st.slider("折扣比例 (%)", 0, 50, 10, step=5)
    demand = max(0, 1000 - (price - 200) * 2 + discount * 5)
    profit = demand * (price * (1 - discount / 100) * 0.3)

    st.metric(label="📈 預估銷售量", value=f"{int(demand)} 件")
    st.metric(label="💰 預估獲利", value=f"{profit:,.0f} 元")
