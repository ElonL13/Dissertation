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

st.set_page_config(page_title="Bike Demand Forecasting", layout="wide")
st.title("📊 Bike Demand Forecasting App")

st.header("1. Upload CSV files")
train_files = st.file_uploader("Upload 2 or more training CSV files", accept_multiple_files=True, type="csv")
test_file = st.file_uploader("Upload a testing CSV file", type="csv")

train_df = pd.DataFrame()
test_df = pd.DataFrame()

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

    fig, ax = plt.subplots(figsize=(14, 4))
    sample_station = train_pivot['start_station_id'].unique()[0]
    sample_data = train_pivot[train_pivot['start_station_id'] == sample_station]
    ax.plot(sample_data['date'], sample_data['classic_bike'], label='Classic')
    ax.plot(sample_data['date'], sample_data['electric_bike'], label='Electric')
    ax.set_title(f"Trip Count Over Time - Station {sample_station}")
    ax.legend()
    st.pyplot(fig)

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

        if st.session_state.get('rf_result') and st.session_state.get('knn_result'):
            st.subheader("📋 Model Comparison Table (MAE and R²)")

            rf_pred = st.session_state['rf_result']['pred']
            knn_pred = st.session_state['knn_result']['pred']

            comparison_data = []
            for i, label in enumerate(['classic_bike', 'electric_bike']):
                mae_rf = mean_absolute_error(y_test[label], rf_pred[:, i])
                r2_rf = r2_score(y_test[label], rf_pred[:, i])

                mae_knn = mean_absolute_error(y_test[label], knn_pred[:, i])
                r2_knn = r2_score(y_test[label], knn_pred[:, i])

                comparison_data.append({
                    "Bike Type": label,
                    "Random Forest MAE": f"{mae_rf:.2f}",
                    "Random Forest R²": f"{r2_rf:.3f}",
                    "KNN MAE": f"{mae_knn:.2f}",
                    "KNN R²": f"{r2_knn:.3f}"
                })

            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df)
    st.header("4. 📌 Select Station to Compare Actual vs Predicted")

    station_options = test_pivot['start_station_id'].unique().tolist()
    selected_station = st.selectbox("Choose a station", station_options)

    def plot_predictions_vs_actual(dates, actual, predicted, label, model_name):
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(dates, actual, label=f'Actual {label.capitalize()}')
        ax.plot(dates, predicted, label=f'Predicted {label.capitalize()} ({model_name})', linestyle='--')
        ax.set_title(f'{model_name} - {label.capitalize()} - Actual vs Predicted')
        ax.legend()
        st.pyplot(fig)

    if selected_station:
        if st.session_state.get('rf_result'):
            rf_pred = st.session_state['rf_result']['pred']
            idx = test_pivot['start_station_id'] == selected_station
            station_data = test_pivot[idx].copy()
            dates = station_data['date'].values

            for i, label in enumerate(['classic_bike', 'electric_bike']):
                plot_predictions_vs_actual(
                    dates,
                    station_data[label].values,
                    rf_pred[idx.values, i],
                    label,
                    model_name="Random Forest"
                )

        if st.session_state.get('knn_result'):
            knn_pred = st.session_state['knn_result']['pred']
            idx = test_pivot['start_station_id'] == selected_station
            station_data = test_pivot[idx].copy()
            dates = station_data['date'].values

            for i, label in enumerate(['classic_bike', 'electric_bike']):
                plot_predictions_vs_actual(
                    dates,
                    station_data[label].values,
                    knn_pred[idx.values, i],
                    label,
                    model_name="KNN"
                )
 
        st.header("5. 🎯 Typical Station Examples")

        if st.session_state.get('rf_result') and st.session_state.get('knn_result'):
            st.subheader("High-frequency vs Low-frequency Station Comparison")
            station_total = test_pivot.groupby('start_station_id')[['classic_bike', 'electric_bike']].sum()
            station_total['total'] = station_total['classic_bike'] + station_total['electric_bike']

            high_freq_station = station_total['total'].idxmax()
            low_freq_station = station_total['total'].idxmin()

            for station_id, station_label in [(high_freq_station, "High-frequency Station"), (low_freq_station, "Low-frequency Station")]:
                station_data = test_pivot[test_pivot['start_station_id'] == station_id]
                dates = station_data['date'].values

                rf_pred = st.session_state['rf_result']['pred']
                knn_pred = st.session_state['knn_result']['pred']
                mask = test_pivot['start_station_id'] == station_id

                st.markdown(f"### {station_label} — Station ID: `{station_id}`")

                # RF - Classic
                fig1, ax1 = plt.subplots(figsize=(13, 3))
                ax1.plot(dates, station_data['classic_bike'], label="Actual Classic")
                ax1.plot(dates, rf_pred[mask, 0], label="Predicted Classic (RF)", linestyle="--")
                ax1.set_title("Random Forest – Classic Bike")
                ax1.legend()
                st.pyplot(fig1)

                # RF - Electric
                fig2, ax2 = plt.subplots(figsize=(13, 3))
                ax2.plot(dates, station_data['electric_bike'], label="Actual Electric")
                ax2.plot(dates, rf_pred[mask, 1], label="Predicted Electric (RF)", linestyle="--")
                ax2.set_title("Random Forest – Electric Bike")
                ax2.legend()
                st.pyplot(fig2)

                # KNN - Classic
                fig3, ax3 = plt.subplots(figsize=(13, 3))
                ax3.plot(dates, station_data['classic_bike'], label="Actual Classic")
                ax3.plot(dates, knn_pred[mask, 0], label="Predicted Classic (KNN)", linestyle="--")
                ax3.set_title("KNN – Classic Bike")
                ax3.legend()
                st.pyplot(fig3)

                # KNN - Electric
                fig4, ax4 = plt.subplots(figsize=(13, 3))
                ax4.plot(dates, station_data['electric_bike'], label="Actual Electric")
                ax4.plot(dates, knn_pred[mask, 1], label="Predicted Electric (KNN)", linestyle="--")
                ax4.set_title("KNN – Electric Bike")
                ax4.legend()
                st.pyplot(fig4)
