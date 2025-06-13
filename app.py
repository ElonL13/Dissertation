import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


if 'rf_result' not in st.session_state:
    st.session_state['rf_result'] = None
if 'knn_result' not in st.session_state:
    st.session_state['knn_result'] = None
# 页面设置
st.set_page_config(page_title="Bike Demand Forecasting", layout="wide")
st.title("📊 Bike Demand Forecasting App")

# 文件上传
st.header("1. Upload CSV files")
train_files = st.file_uploader("Upload 2 or more training CSV files", accept_multiple_files=True, type="csv")
test_file = st.file_uploader("Upload a testing CSV file", type="csv")

# 初始化全局变量
train_df = pd.DataFrame()
test_df = pd.DataFrame()

# 数据加载与清洗
if train_files and test_file:
    train_dfs = []
    for file in train_files:
        df = pd.read_csv(file)
        df['started_at'] = pd.to_datetime(df['started_at'], errors='coerce')
        df['date'] = df['started_at'].dt.date
        df['start_station_id'] = df['start_station_id'].astype(str)
        train_dfs.append(df)
    train_df = pd.concat(train_dfs, ignore_index=True)

    test_df = pd.read_csv(test_file)
    test_df['started_at'] = pd.to_datetime(test_df['started_at'], errors='coerce')
    test_df['date'] = test_df['started_at'].dt.date
    test_df['start_station_id'] = test_df['start_station_id'].astype(str)

    for df in [train_df, test_df]:
        df['start_lat'] = pd.to_numeric(df['start_lat'], errors='coerce')
        df['start_lng'] = pd.to_numeric(df['start_lng'], errors='coerce')
        df['day_of_week'] = df['started_at'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'] >= 5

    train = train_df.groupby(['start_station_id', 'date', 'rideable_type']).agg({
        'started_at': 'count',
        'start_lat': 'first',
        'start_lng': 'first',
        'day_of_week': 'first',
        'is_weekend': 'first'
    }).reset_index().rename(columns={'started_at': 'trip_count'})

    test = test_df.groupby(['start_station_id', 'date', 'rideable_type']).agg({
        'started_at': 'count',
        'start_lat': 'first',
        'start_lng': 'first',
        'day_of_week': 'first',
        'is_weekend': 'first'
    }).reset_index().rename(columns={'started_at': 'trip_count'})

    train_pivot = train.pivot_table(
        index=['start_station_id', 'date', 'start_lat', 'start_lng', 'day_of_week', 'is_weekend'],
        columns='rideable_type',
        values='trip_count',
        fill_value=0
    ).reset_index()

    test_pivot = test.pivot_table(
        index=['start_station_id', 'date', 'start_lat', 'start_lng', 'day_of_week', 'is_weekend'],
        columns='rideable_type',
        values='trip_count',
        fill_value=0
    ).reset_index()

    X_train = train_pivot.drop(columns=['classic_bike', 'electric_bike'])
    y_train = train_pivot[['classic_bike', 'electric_bike']]
    X_test = test_pivot.drop(columns=['classic_bike', 'electric_bike'])
    y_test = test_pivot[['classic_bike', 'electric_bike']]

    categorical = ['start_station_id', 'is_weekend']
    numerical = ['start_lat', 'start_lng', 'day_of_week']
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
        ('num', 'passthrough', numerical)
    ])

    st.success("✅ Data cleaned and processed successfully.")

    if st.checkbox("Show processed training data"):
        st.dataframe(train_pivot.head())

    if st.checkbox("Show processed testing data"):
        st.dataframe(test_pivot.head())

    st.header("2. 📊 Visualization")

    # 可视化 Trip Count 趋势图
    fig, ax = plt.subplots(figsize=(14, 4))
    sample_station = train_pivot['start_station_id'].unique()[0]
    sample_data = train_pivot[train_pivot['start_station_id'] == sample_station]
    ax.plot(sample_data['date'], sample_data['classic_bike'], label='Classic')
    ax.plot(sample_data['date'], sample_data['electric_bike'], label='Electric')
    ax.set_title(f"Trip Count Over Time - Station {sample_station}")
    ax.legend()
    st.pyplot(fig)

    # 可视化 Top 20 Total Trip Count by Station
    top_stations = train.groupby('start_station_id')['trip_count'].sum().sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(14, 5))
    top_stations.plot(kind='bar', ax=ax)
    ax.set_title("Top 20 Stations by Total Trip Count")
    ax.set_ylabel("Trip Count")
    st.pyplot(fig)

    st.header("3. 🚀 Run Prediction Models")

    if st.button("Run Random Forest"):
        pipeline_rf = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', MultiOutputRegressor(RandomForestRegressor(random_state=42)))
        ])
        pipeline_rf.fit(X_train, y_train)
        y_pred_rf = np.round(pipeline_rf.predict(X_test)).astype(int)

        st.session_state['rf_result'] = {
            'model': pipeline_rf,
            'pred': y_pred_rf
        }

        st.success("Random Forest completed. Select a station below to visualize results.")

        st.subheader("Random Forest Results")
        for i, label in enumerate(['classic_bike', 'electric_bike']):
            mae = mean_absolute_error(y_test[label], y_pred_rf[:, i])
            r2 = r2_score(y_test[label], y_pred_rf[:, i])
            st.write(f"{label} — MAE (Average error): {mae:.2f}, R² (Model fit): {r2:.3f}")

    if st.button("Run KNN"):
        pipeline_knn = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', MultiOutputRegressor(KNeighborsRegressor(n_neighbors=5)))
        ])
        pipeline_knn.fit(X_train, y_train)
        y_pred_knn = np.round(pipeline_knn.predict(X_test)).astype(int)

        st.session_state['knn_result'] = {
            'model': pipeline_knn,
            'pred': y_pred_knn
        }

        st.success("KNN completed. Select a station below to visualize results.")
        st.subheader("KNN Results")
        for i, label in enumerate(['classic_bike', 'electric_bike']):
            mae = mean_absolute_error(y_test[label], y_pred_knn[:, i])
            r2 = r2_score(y_test[label], y_pred_knn[:, i])
            st.write(f"{label} — MAE (Average error): {mae:.2f}, R² (Model fit): {r2:.3f}")


# 可视化选择区域
    st.header("4. 📌 Select Station to Compare Actual vs Predicted")

    station_options = test_pivot['start_station_id'].unique().tolist()
    selected_station = st.selectbox("Choose a station", station_options)

# 显示图（如果模型有结果）
    if st.session_state['rf_result']:
        rf_pred = st.session_state['rf_result']['pred']
        station_data = test_pivot[test_pivot['start_station_id'] == selected_station].copy()
        dates = station_data['date'].values

        fig_rf, ax_rf = plt.subplots(figsize=(14, 4))
        ax_rf.plot(dates, station_data['classic_bike'].values, label='Actual Classic')
        ax_rf.plot(dates, rf_pred[test_pivot['start_station_id'] == selected_station, 0], 
                   label='Predicted Classic (RF)', linestyle='--')
        ax_rf.set_title(f"Random Forest - Station {selected_station}")
        ax_rf.legend()
        st.pyplot(fig_rf)

    if st.session_state['knn_result']:
        knn_pred = st.session_state['knn_result']['pred']
        station_data = test_pivot[test_pivot['start_station_id'] == selected_station].copy()
        dates = station_data['date'].values

        fig_knn, ax_knn = plt.subplots(figsize=(14, 4))
        ax_knn.plot(dates, station_data['electric_bike'].values, label='Actual Electric')
        ax_knn.plot(dates, knn_pred[test_pivot['start_station_id'] == selected_station, 1], 
                    label='Predicted Electric (KNN)', linestyle='--')
        ax_knn.set_title(f"KNN - Station {selected_station}")
        ax_knn.legend()
        st.pyplot(fig_knn)
