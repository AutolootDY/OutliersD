import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tvDatafeed import TvDatafeed, Interval
import pandas as pd
import numpy as np

# 🚀 ดึงข้อมูลจาก TradingView
tv = TvDatafeed()

# 📌 ตั้งค่าหุ้นและตลาด
stock_symbol = "DELTA"
exchange_stock = "SET"
set_index_symbol = "SET"
exchange_index = "SET"
interval = Interval.in_daily

# 📌 ดึงข้อมูลดัชนี SET และหุ้น DELTA
set_index_data = tv.get_hist(symbol=set_index_symbol, exchange=exchange_index, interval=interval, n_bars=5000)
delta_data = tv.get_hist(symbol=stock_symbol, exchange=exchange_stock, interval=interval, n_bars=5000)

# ✅ **ดึงข้อมูลหุ้นใน SET50**
set50_stocks = ["PTT", "AOT", "CPALL", "SCB", "KBANK", "DELTA", "ADVANC", "AWC", "BANPU", "BBL", "BDMS","BH","BJC","BTS","CBG","CCET","COM7","CPF","CPN","CRC","EGCO","GLOBAL","GPSC","GULF","HMPRO","INTUCH","ITC","IVL","KTC","KTB","LH","MINT","MTC","OR","OSP","PTT","PTTEP","PTTGC","RATCH","SAWAD","SCC","SCGP","TISCO","TLI","TOP","TRUE","TTB","TU","WHA"]
stock_data = {}

for stock in set50_stocks:
    data = tv.get_hist(symbol=stock, exchange="SET", interval=interval, n_bars=5000)
    if data is not None and not data.empty:
        stock_data[stock] = data['close']

# 🔹 รวมข้อมูลหุ้นทั้งหมดเป็น DataFrame
stock_df = pd.DataFrame(stock_data)
stock_df.reset_index(inplace=True)
stock_df.rename(columns={'datetime': 'Date'}, inplace=True)

# ✅ **คำนวณ Market Breadth**
stock_df_numeric = stock_df.select_dtypes(include=[np.number])  # ดึงเฉพาะคอลัมน์ตัวเลข
advance = (stock_df_numeric.diff() > 0).sum(axis=1)
decline = (stock_df_numeric.diff() < 0).sum(axis=1)
stock_df['AD Line'] = (advance - decline).cumsum()
stock_df['McClellan Osc'] = (advance - decline).rolling(19).mean() - (advance - decline).rolling(39).mean()

# ✅ **ทำให้ Date เป็น datetime format**
stock_df['Date'] = pd.to_datetime(stock_df['Date'])
delta_data['Date'] = pd.to_datetime(delta_data.index)

# ✅ **รวมข้อมูล AD Line และ McClellan Osc กับ DELTA**
merged_df = stock_df[['Date', 'AD Line', 'McClellan Osc']].merge(
    delta_data[['Date', 'close']], on='Date', how='left'
)
merged_df.rename(columns={'close': 'DELTA Price'}, inplace=True)

# # ✅ **จับแพทเทิร์น Scenario**
# merged_df['AD Line Diff'] = merged_df['AD Line'].diff(3)  # ดู 3 วันก่อนหน้า
# merged_df['McClellan Osc Diff'] = merged_df['McClellan Osc'].diff(3)
# merged_df['DELTA Price Diff'] = merged_df['DELTA Price'].diff(3)

# ✅ คำนวณการเปลี่ยนแปลงภายในช่วง 5 วัน
merged_df['AD Line Diff 5d'] = merged_df['AD Line'].diff(5)
merged_df['McClellan Osc Diff 5d'] = merged_df['McClellan Osc'].diff(5)
merged_df['DELTA Price Diff 5d'] = merged_df['DELTA Price'].diff(5)


# ✅ ฟังก์ชันสำหรับแยก Scenario และอธิบายแต่ละ Scenario ในช่วงเวลา 7 วัน
def detect_scenario(row, threshold_ad=5, threshold_mc=1, threshold_delta=0.5):
    ad_diff = row['AD Line Diff 5d']
    mc_diff = row['McClellan Osc Diff 5d']
    delta_diff = row['DELTA Price Diff 5d']

    # ใช้ Threshold เพื่อลด Noise จากค่าใกล้ 0
    ad_signal = "up" if ad_diff > threshold_ad else "down" if ad_diff < -threshold_ad else "neutral"
    mc_signal = "up" if mc_diff > threshold_mc else "down" if mc_diff < -threshold_mc else "neutral"
    delta_signal = "up" if delta_diff > threshold_delta else "down" if delta_diff < -threshold_delta else "neutral"

    # 🔹 Scenario 1: ทุกตัวขึ้น = แนวโน้มขาขึ้น
    if ad_signal == "up" and mc_signal == "up" and delta_signal == "up":
        return "Scenario 1", "✅ แนวโน้มขาขึ้น"

    # 🔹 Scenario 2: ทุกตัวลง = แนวโน้มขาลง
    elif ad_signal == "down" and mc_signal == "down" and delta_signal == "down":
        return "Scenario 2", "❌ แนวโน้มขาลง"

    # 🔹 Scenario 3: AD Line & DELTA ลง แต่ McClellan Osc ยังขึ้น = ตลาดอ่อนแอ
    elif ad_signal == "down" and mc_signal == "up" and delta_signal == "down":
        return "Scenario 3", "⚠️ ตลาดอ่อนแอ"

    elif ad_signal == "down" and mc_signal == "down" and delta_signal == "up":
        return "Scenario 4", "⚠️ DELTA ยังขึ้น อาจเด้งเพื่อลงต่อ!"

    # ถ้าไม่เข้าเงื่อนไขใดเลย
    else:
        return "No Scenario", ""

# ✅ ใช้ฟังก์ชันกับ DataFrame
merged_df[['Scenario', 'Description']] = merged_df.apply(detect_scenario, axis=1, result_type='expand')

# ✅ ฟิลเตอร์กราฟ Scenario ที่ตรงเงื่อนไข
scenario_results = merged_df[merged_df['Scenario'] != "No Scenario"]

# 📈 สร้างกราฟ Interactive สำหรับ AD Line
fig_adline = go.Figure()

fig_adline.add_trace(go.Scatter(x=merged_df['Date'], y=merged_df['AD Line'],
                               mode='lines', name='AD Line', line=dict(color='blue')))

# for index, row in scenario_results.iloc[::10].iterrows():
#     fig_adline.add_trace(go.Scatter(x=[row['Date']], y=[row['AD Line']],
#                                    mode='markers+text', name=f"{row['Scenario']}",
#                                    marker=dict(color='red', size=8),
#                                    text=[f"{row['Description']}"],
#                                    textposition="top right"))

fig_adline.update_layout(
    title="AD Line (7-Day Check)",
    xaxis_title="Date",
    yaxis_title="AD Line",
    hovermode="x unified",
    template="plotly_dark",
    xaxis=dict(tickformat="%Y-%m-%d"),
    height=600,
    width=1000,
    margin=dict(l=80, r=80, t=80, b=80),
    autosize=True,
    showlegend=True,
    xaxis_rangeslider_visible=True,
    xaxis_rangeslider_thickness=0.05
)

# 📈 สร้างกราฟ Interactive สำหรับ McClellan Oscillator
fig_mcclellan = go.Figure()

fig_mcclellan.add_trace(go.Scatter(x=merged_df['Date'], y=merged_df['McClellan Osc'],
                                   mode='lines', name='McClellan Osc', line=dict(color='green')))

# for index, row in scenario_results.iloc[::10].iterrows():
#     fig_mcclellan.add_trace(go.Scatter(x=[row['Date']], y=[row['McClellan Osc']],
#                                       mode='markers+text', name=f"{row['Scenario']}",
#                                       marker=dict(color='red', size=8),
#                                       text=[f"{row['Description']}"],
#                                       textposition="top right"))

fig_mcclellan.update_layout(
    title="McClellan Oscillator (7-Day Check)",
    xaxis_title="Date",
    yaxis_title="McClellan Oscillator",
    hovermode="x unified",
    template="plotly_dark",
    xaxis=dict(tickformat="%Y-%m-%d"),
    height=600,
    width=1000,
    margin=dict(l=80, r=80, t=80, b=80),
    autosize=True,
    showlegend=True,
    xaxis_rangeslider_visible=True,
    xaxis_rangeslider_thickness=0.05
)

# 📈 สร้างกราฟ Interactive สำหรับ DELTA Stock Price
fig_delta = go.Figure()

fig_delta.add_trace(go.Scatter(x=merged_df['Date'], y=merged_df['DELTA Price'],
                              mode='lines', name='DELTA Price', line=dict(color='red')))

# for index, row in scenario_results.iloc[::10].iterrows():
#     fig_delta.add_trace(go.Scatter(x=[row['Date']], y=[row['DELTA Price']],
#                                   mode='markers+text', name=f"{row['Scenario']}",
#                                   marker=dict(color='red', size=8),
#                                   text=[f"{row['Description']}"],
#                                   textposition="top right"))
# 🔹 **Highlight จุดที่ตรงกับ Scenario**
# จำกัดแค่แสดง Scenario ทุกๆ 10 วัน (ลดจำนวนข้อมูลที่แสดง)
for index, row in scenario_results.iloc[::10].iterrows():
    fig_delta.add_trace(go.Scatter(x=[row['Date']], y=[row['DELTA Price']],
                             mode='markers+text', name=f"{row['Scenario']}",
                             marker=dict(color='red', size=8),
                             text=[f"{row['Description']}"],
                             textposition="top right"))
    
fig_delta.update_layout(
    title="DELTA Stock Price (7-Day Check)",
    xaxis_title="Date",
    yaxis_title="Stock Price",
    hovermode="x unified",
    template="plotly_dark",
    xaxis=dict(tickformat="%Y-%m-%d"),
    height=600,
    width=1000,
    margin=dict(l=80, r=80, t=80, b=80),
    autosize=True,
    showlegend=True,
    xaxis_rangeslider_visible=True,
    xaxis_rangeslider_thickness=0.05
)

# 📌 แสดงกราฟ Interactive สำหรับทั้ง 3 กราฟใน Streamlit
st.title("Scenario Analysis with AD Line, McClellan Oscillator, and DELTA Stock Price")

# แสดงกราฟ AD Line
st.subheader("AD Line")
st.plotly_chart(fig_adline)

# แสดงกราฟ McClellan Oscillator
st.subheader("McClellan Oscillator")
st.plotly_chart(fig_mcclellan)

# แสดงกราฟ DELTA Stock Price
st.subheader("DELTA Stock Price")
st.plotly_chart(fig_delta)

# 📌 บันทึกข้อมูลลง CSV (สามารถให้ผู้ใช้ดาวน์โหลดได้)
st.write("### Download Scenario Data")
st.download_button(
    label="Download Scenario Data as CSV",
    data=scenario_results.to_csv(index=False),
    file_name="Scenario_Analysis_7d_Separate_Graphs.csv",
    mime="text/csv"
)
