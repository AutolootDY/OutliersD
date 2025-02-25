# Market Breadth Indicators & Detect Scenario Conditions

## 1. Advance-Decline Line (AD Line) คืออะไร?
AD Line เป็นตัวบ่งชี้ความแข็งแกร่งของตลาดโดยดูจากจำนวนหุ้นที่ปรับตัวขึ้นและลง

### สูตรการคำนวณ AD Line
\[ AD_t = AD_{t-1} + (\text{จำนวนหุ้นที่ขึ้น} - \text{จำนวนหุ้นที่ลง}) \]

### วิธีการอ่านค่า AD Line
- 📌 หาก AD Line สูงขึ้น → หุ้นส่วนใหญ่ปรับตัวขึ้น → ตลาดอยู่ในแนวโน้มขาขึ้น
- 📌 หาก AD Line ลดลง → หุ้นส่วนใหญ่ปรับตัวลง → ตลาดอยู่ในแนวโน้มขาลง

### ✅ ประโยชน์ของ AD Line
- ใช้ดูว่า SET50 วิ่งขึ้นจากหุ้นจำนวนมาก หรือถูกขับเคลื่อนโดยหุ้นตัวใหญ่เพียงไม่กี่ตัว
- หาก SET50 ขึ้น แต่ AD Line ลดลง แสดงว่าเป็นสัญญาณ Divergence → ตลาดอาจจะอ่อนแอ

## 2. McClellan Oscillator คืออะไร?
McClellan Oscillator เป็นตัวบ่งชี้ที่ช่วยดูแนวโน้มของ Market Breadth โดยอิงจากค่าเฉลี่ยของ Advance-Decline Difference (หุ้นขึ้น - หุ้นลง)

### สูตรการคำนวณ McClellan Oscillator
\[ \text{McClellan Osc} = EMA_{19} (\text{หุ้นขึ้น} - \text{หุ้นลง}) - EMA_{39} (\text{หุ้นขึ้น} - \text{หุ้นลง}) \]

### วิธีการอ่านค่า McClellan Oscillator
- 📌 หาก McClellan Oscillator สูงกว่า 0 → ตลาดมีแรงซื้อ
- 📌 หาก McClellan Oscillator ต่ำกว่า 0 → ตลาดมีแรงขาย
- 📌 หาก McClellan Oscillator กลับตัวขึ้นจากจุดต่ำสุด → อาจเป็นสัญญาณซื้อ
- 📌 หาก McClellan Oscillator ร่วงลงต่ำกว่าศูนย์ → อาจเป็นสัญญาณขาย

### ✅ ประโยชน์ของ McClellan Oscillator
- ช่วยดูโมเมนตัมของ Market Breadth (หุ้นส่วนใหญ่กำลังแข็งแกร่งหรืออ่อนแอ)
- ใช้ตรวจจับ Divergence ระหว่าง Market Breadth และราคาของ SET50

## 3. เงื่อนไขใหม่ของ detect_scenario

| Scenario | เงื่อนไข | สัญลักษณ์ |
|-----------|----------|------------|
| Scenario 1 | ทุกตัวขึ้น + EMA10 > EMA50 | ✅ ตลาดขาขึ้นชัดเจน |
| Scenario 2 | ทุกตัวลง + EMA10 < EMA50 | ❌ ตลาดขาลงชัดเจน |
| Scenario 3 | AD Line & DELTA ลง แต่ McClellan Osc ยังขึ้น + EMA10 < EMA50 | ⚠️ ตลาดอ่อนแอ |
| Scenario 4 | AD Line & McClellan Osc ลง แต่ DELTA ยังขึ้น + EMA10 > EMA50 | ⚠️ DELTA อาจเด้งเพื่อลงต่อ |
| Scenario 5 | ทุกตัวเป็นกลาง + EMA10 ≈ EMA50 | ⏳ ตลาดไม่มีแนวโน้ม |

---

ไฟล์นี้เป็นเอกสารที่ช่วยอธิบายแนวคิดของ Market Breadth Indicators และเงื่อนไขใหม่ของ `detect_scenario` เพื่อใช้วิเคราะห์แนวโน้มของตลาด

🔗 [ดูกราฟวิเคราะห์ตลาดได้ที่นี่](https://outliersd-weumrtvxrsaksiumrd2hgg.streamlit.app/?fbclid=IwY2xjawIqDPxleHRuA2FlbQIxMAABHbOdhuREUEkS7ihxxQVh53NjTcNj_JmXBaIWkggStvQryXkhezXcWrLeow_aem_5tTig1IM16WiCzTi7l9fnQ) 🚀📊
