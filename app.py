import os
from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.utils
import json

app = Flask(__name__)

# --- 1. LOAD CÁC FILE ĐÃ LƯU ---
model = joblib.load('xgboost_fraud_model.pkl')
stats_dict = joblib.load('category_stats.pkl')
model_features = joblib.load('model_features.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "Không tìm thấy file"
    
    file = request.files['file']
    if file.filename == '':
        return "Tên file trống"

    if file:
        try:
            # Đọc file (CSV hoặc Excel)
            if file.filename.endswith('.csv'):
                df_input = pd.read_csv(file)
            else:
                df_input = pd.read_excel(file, engine='openpyxl')
            
            # Đồng bộ tên cột trong file upload
            column_mapping = {
                'merchant_category': 'category',
                'distance_from_home': 'distance'
            }
            df_input = df_input.rename(columns=column_mapping)

            results = []
            known_categories = list(stats_dict.keys())

            for index, row in df_input.iterrows():
                # A. Lấy dữ liệu cơ bản
                amount = float(row.get('amount', 0))
                currency_str = str(row.get('currency', 'USD')).upper()
                raw_category = str(row.get('category', 'Others'))
                hour = int(row.get('hour', 12))
                distance = float(row.get('distance', 0))

                # B. Feature Engineering & Xử lý "Others"
                amount_usd = amount * 0.00004 if currency_str == 'VND' else amount
                category_for_model = raw_category if raw_category in known_categories else 'Others'
                
                z_score = 0
                if category_for_model in stats_dict:
                    mean_val = stats_dict[category_for_model]['mean']
                    std_val = stats_dict[category_for_model]['std']
                    z_score = (amount_usd - mean_val) / (std_val + 1e-6)

                # C. TẠO DF 23 CỘT KHỚP TẬP TRAIN
                input_row = pd.DataFrame(0.0, index=[0], columns=model_features)
                input_row['Amount USD'] = float(amount_usd)
                input_row['Distance From Home'] = float(distance)
                input_row['Is_High_Risk_Hour'] = int(1 if (0 <= hour <= 5) else 0)
                input_row['amt_zscore_category'] = float(z_score)
                input_row['Product Category'] = 1 
                input_row['Currency'] = 1 if currency_str == 'USD' else 2
                
                # D. DỰ BÁO
                final_features = input_row[model_features].astype(float)
                prob = model.predict_proba(final_features)[0][1]
                pred = 1 if prob > 0.5 else 0

                # E. TẠO LÝ GIẢI (REASONING)
                reasons = []
                if 0 <= hour <= 5: reasons.append("Giao dịch ban đêm")
                if distance > 200: reasons.append(f"Khoảng cách xa")
                if z_score > 2.5: reasons.append("Số tiền bất thường")

                if pred == 1:
                    reason_text = " + ".join(reasons) if reasons else "Hành vi nghi vấn"
                else:
                    reason_text = "Bình thường"

                # Lưu kết quả (Bao gồm các cột ẩn để Filter)
                results.append({
                    'Số tiền': f"{amount:,.0f} {currency_str}",
                    'Danh mục': raw_category,
                    'Lý do phân tích': reason_text,
                    'Rủi ro (%)': f"{prob:.2%}",
                    'Kết luận': "⚠️ CẢNH BÁO" if pred == 1 else "✅ AN TOÀN",
                    # Cột ẩn hỗ trợ JS Filter
                    'data-category': raw_category,
                    'data-currency': currency_str,
                    'data-hour': hour
                })

            res_df = pd.DataFrame(results)
            fraud_df = res_df[res_df['Kết luận'] == '⚠️ CẢNH BÁO']
            
            # --- 1. BIỂU ĐỒ TRÒN ---
            fig_pie = px.pie(res_df, names='Kết luận', title='Tình trạng rủi ro',
                             color='Kết luận', color_discrete_map={'✅ AN TOÀN':'#28a745', '⚠️ CẢNH BÁO':'#dc3545'})
            graphJSON_pie = json.dumps(fig_pie, cls=plotly.utils.PlotlyJSONEncoder)
            
            # --- 2. BIỂU ĐỒ DANH MỤC ---
            if not fraud_df.empty:
                cat_data = fraud_df['Danh mục'].value_counts().reset_index()
                cat_data.columns = ['Danh mục', 'Số vụ']
                fig_cat = px.bar(cat_data, x='Danh mục', y='Số vụ', color='Số vụ', color_continuous_scale='Reds')
            else:
                cat_data = res_df['Danh mục'].value_counts().reset_index()
                fig_cat = px.bar(cat_data, x='Danh mục', y='count', title='Phân bổ giao dịch')
            graphJSON_category = json.dumps(fig_cat, cls=plotly.utils.PlotlyJSONEncoder)

            # --- 3. BIỂU ĐỒ GIỜ ---
            hour_stats = []
            for h in range(24):
                h_data = res_df[res_df['data-hour'] == h]
                f_rate = (len(h_data[h_data['Kết luận'] == '⚠️ CẢNH BÁO']) / len(h_data) * 100) if len(h_data) > 0 else 0
                hour_stats.append({'Giờ': f"{h:02d}h", 'Tỷ lệ rủi (%)': f_rate})
            fig_hour = px.line(pd.DataFrame(hour_stats), x='Giờ', y='Tỷ lệ rủi (%)', title='Xu hướng rủi ro theo giờ')
            graphJSON_hour = json.dumps(fig_hour, cls=plotly.utils.PlotlyJSONEncoder)

            # --- 4. BIỂU ĐỒ TIỀN TỆ ---
            curr_data = res_df['data-currency'].value_counts().reset_index()
            fig_curr = px.bar(curr_data, x='data-currency', y='count', title='Giao dịch theo tiền tệ')
            graphJSON_currency = json.dumps(fig_curr, cls=plotly.utils.PlotlyJSONEncoder)

            # --- XUẤT BẢNG ---
            # Chỉ hiển thị 5 cột chính nhưng giữ lại thuộc tính data-* trong HTML
            display_cols = ['Số tiền', 'Danh mục', 'Lý do phân tích', 'Rủi ro (%)', 'Kết luận']
            # Thủ thuật: Thêm các cột data vào class hoặc thuộc tính ẩn là cực khó với to_html, 
            # nên ta sẽ render bảng đơn giản và dùng JS xử lý tiếp ở index.html.
            table_html = res_df[display_cols].to_html(classes='table table-hover', index=False)

            return render_template('index.html', 
                                   tables=table_html,
                                   graphJSON_pie=graphJSON_pie,
                                   graphJSON_category=graphJSON_category,
                                   graphJSON_hour=graphJSON_hour,
                                   graphJSON_currency=graphJSON_currency,
                                   total_trans=len(res_df),
                                   fraud_count=len(fraud_df))
        except Exception as e:
            return f"Lỗi: {str(e)}"
    return "File không hợp lệ"

if __name__ == "__main__":
    app.run(debug=True)