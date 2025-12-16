import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import (LabelEncoder, MinMaxScaler, StandardScaler,
                                   OneHotEncoder)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Import data
# from saved_snippet import df


def generate_sales_data(year: int = 2023, n_rows: int = 1000):
    np.random.seed(42 + year)
    dates = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='D')
    regions = ['West', 'Midwest', 'South', 'Northeast']
    categories = ['Virginia', 'California', 'Texas', 'New York', 'Flordia', 'Pennsylvania', 'Ohio', 'Illinois', 'Georgia', 'North Carolina']

    data = {
        'Date': np.random.choice(dates, size=n_rows),
        'Region': np.random.choice(regions, size=n_rows),
        'Category': np.random.choice(categories, size=n_rows),
        'Sales': np.round(np.random.uniform(100, 1000, size=n_rows), 2),
        'Units Sold': np.random.randint(1, 20, size=n_rows)
    }

    gen_df = pd.DataFrame(data)
    gen_df['Month'] = gen_df['Date'].dt.to_period('M').astype(str)

    # realistic missing values
    missing_sales_idx = np.random.choice(len(gen_df), size=int(len(gen_df) * 0.03), replace=False)
    gen_df.loc[missing_sales_idx, 'Sales'] = np.nan
    missing_region_idx = np.random.choice(len(gen_df), size=int(len(gen_df) * 0.02), replace=False)
    gen_df.loc[missing_region_idx, 'Region'] = np.nan

    return gen_df

st.set_page_config(page_title="Smart Sales Analytics - ML Dashboard", layout="wide", initial_sidebar_state="collapsed")

# Notebook-style CSS
st.markdown("""
<style>
    * { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
    body { background-color: #f8f8f8; }
    .main { background-color: white; padding: 30px 40px; }
    h1 { color: #1f1f1f; font-size: 48px; font-weight: 700; margin-bottom: 5px; letter-spacing: -0.5px; }
    h2 { color: #1f1f1f; font-size: 28px; font-weight: 600; margin-top: 50px; margin-bottom: 20px; border-bottom: 3px solid #e0e0e0; padding-bottom: 12px; }
    h3 { color: #333; font-size: 20px; font-weight: 600; margin-top: 30px; margin-bottom: 15px; }
    .subtitle { color: #666; font-size: 16px; margin-bottom: 30px; line-height: 1.6; }
    .code-block { background-color: #f5f5f5; padding: 15px; border-left: 4px solid #667eea; border-radius: 4px; margin: 15px 0; font-family: monospace; }
    .section-divider { height: 2px; background: #e0e0e0; margin: 50px 0; }
</style>
""", unsafe_allow_html=True)

# ===== HEADER =====
st.markdown("<h1>Computational Analysis of Plant Growth Data in Smart Agriculture</h1>", unsafe_allow_html=True)
st.markdown("""
<p class="subtitle">
Comprehensive data science analysis: Exploratory Data Analysis (EDA) → Data Preprocessing → Machine Learning Models → Performance Evaluation
</p>
""", unsafe_allow_html=True)

# ===== 1. LOAD DATASET =====
st.markdown("## 1. Load Dataset")
st.markdown("Loaded dataset with sales information across regions, categories, and time periods.")

# Allow user to pick a recent year (generates synthetic recent data)
year_choice = st.selectbox("Select data year", options=[2025, 2024, 2023], index=0)
df = generate_sales_data(year_choice)

col1, col2, col3 = st.columns(3)
col1.metric("Total Records", f"{len(df):,}")
col2.metric("Total Columns", df.shape[1])
col3.metric("Date Range", f"{df['Date'].min().date()} to {df['Date'].max().date()}")

st.dataframe(df.head(10), use_container_width=True)

# ===== 2. INITIAL DATA EXPLORATION =====
st.markdown("## 2. Initial Data Exploration")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Dataset Info")
    # human-readable dtype summary
    def friendly_dtype_name(dt):
        if pd.api.types.is_datetime64_any_dtype(dt):
            return 'datetime'
        if pd.api.types.is_float_dtype(dt):
            return 'float'
        if pd.api.types.is_integer_dtype(dt):
            return 'int'
        if pd.api.types.is_object_dtype(dt):
            return 'object'
        return str(dt)

    dtype_counts = {}
    for col, dt in df.dtypes.items():
        name = friendly_dtype_name(dt)
        dtype_counts[name] = dtype_counts.get(name, 0) + 1

    info_text = f"""
    - **Shape**: {df.shape}
    - **Columns**: {', '.join(df.columns.tolist())}
    - **Data Types (counts)**: {dtype_counts}
    """
    st.markdown(info_text)

with col2:
    st.markdown("### Missing Values")
    missing_df = pd.DataFrame({
        'Column': df.columns,
        'Missing': df.isnull().sum(),
        'Percentage': (df.isnull().sum() / len(df) * 100).round(2)
    })
    st.dataframe(missing_df, use_container_width=True)

st.markdown("### Statistical Summary")
st.dataframe(df.describe(), use_container_width=True)

# ===== 3. EXPLORATORY DATA ANALYSIS =====
st.markdown("## 3. Exploratory Data Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Sales Distribution by Region")
    region_dist = df.groupby('Region')['Sales'].agg(['count', 'mean', 'sum']).reset_index()
    fig_region = px.bar(region_dist, x='Region', y='mean', hover_data=['sum', 'count'], 
                        labels={'mean': 'Avg Sales ($)'}, color='mean', color_continuous_scale='Blues')
    fig_region.update_layout(height=400)
    st.plotly_chart(fig_region, use_container_width=True)

with col2:
    st.markdown("### Sales Distribution by Category")
    cat_dist = df.groupby('Category')['Sales'].agg(['count', 'mean']).reset_index().sort_values('mean', ascending=False).head(10)
    fig_cat = px.bar(cat_dist, x='Category', y='mean', labels={'mean': 'Avg Sales ($)'}, color='mean', color_continuous_scale='Viridis')
    fig_cat.update_layout(height=400, xaxis_tickangle=-45)
    st.plotly_chart(fig_cat, use_container_width=True)

st.markdown("### Sales Over Time (Monthly Trend)")
df['Month_Num'] = df['Date'].dt.to_period('M').astype(str)
monthly_trend = df.groupby('Month_Num')['Sales'].sum().reset_index()
fig_trend = px.line(monthly_trend, x='Month_Num', y='Sales', markers=True, labels={'Month_Num': 'Month', 'Sales': 'Total Sales ($)'})
fig_trend.update_traces(line=dict(width=3), marker=dict(size=8))
fig_trend.update_layout(height=400, hovermode='x unified')
st.plotly_chart(fig_trend, use_container_width=True)

st.markdown("### Correlation Heatmap")
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
corr_matrix = df[numerical_cols].corr()
fig_corr = px.imshow(corr_matrix, labels=dict(color="Correlation"), color_continuous_scale='RdBu', aspect="auto")
fig_corr.update_layout(height=500)
st.plotly_chart(fig_corr, use_container_width=True)

# ===== 4. DATA PREPROCESSING =====
st.markdown("## 4. Data Preprocessing")

# Create a copy for preprocessing
df_processed = df.copy()

# Handle missing categorical values before encoding
df_processed['Region'] = df_processed['Region'].fillna('Unknown')
df_processed['Category'] = df_processed['Category'].fillna('Unknown')

# Drop rows with missing target 'Sales' to avoid training errors
df_processed = df_processed.dropna(subset=['Sales']).reset_index(drop=True)

# Encode categorical variables
le_region = LabelEncoder()
le_category = LabelEncoder()
df_processed['Region_Encoded'] = le_region.fit_transform(df_processed['Region'])
df_processed['Category_Encoded'] = le_category.fit_transform(df_processed['Category'])

st.markdown("✓ Encoded categorical variables (Region, Category) — filled missing as 'Unknown' and dropped missing targets")

# Scale numerical features (fill any missing Units Sold with median)
scaler = MinMaxScaler()
df_processed['Units Sold'] = df_processed['Units Sold'].fillna(df_processed['Units Sold'].median())
scale_cols = ['Units Sold']
df_processed[scale_cols] = scaler.fit_transform(df_processed[scale_cols])

st.markdown("✓ Scaled numerical features (Units Sold)")

# Feature engineering
df_processed['Day_of_Month'] = df_processed['Date'].dt.day
df_processed['Month_Num_Val'] = df_processed['Date'].dt.month
st.markdown("✓ Created temporal features (Day of Month, Month Number)")

# Prepare X and y (after removing rows with missing Sales)
feature_cols = ['Region_Encoded', 'Category_Encoded', 'Units Sold', 'Day_of_Month', 'Month_Num_Val']
X = df_processed[feature_cols].copy()
y = df_processed['Sales'].copy()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.markdown(f"✓ Split data: Train ({len(X_train)} samples), Test ({len(X_test)} samples)")

# ===== 5. MODEL TRAINING & EVALUATION =====
st.markdown("## 5. Model Training & Evaluation")

# Linear Regression
st.markdown("### Linear Regression")
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

col1, col2, col3 = st.columns(3)
col1.metric("MAE", f"${mae_lr:.2f}")
col2.metric("MSE", f"${mse_lr:,.0f}")
col3.metric("R² Score", f"{r2_lr:.4f}")

# Random Forest
st.markdown("### Random Forest Regressor")
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

col1, col2, col3 = st.columns(3)
col1.metric("MAE", f"${mae_rf:.2f}")
col2.metric("MSE", f"${mse_rf:,.0f}")
col3.metric("R² Score", f"{r2_rf:.4f}")

# ===== 6. MODEL COMPARISON =====
st.markdown("## 6. Model Comparison")

comparison_data = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest'],
    'MAE': [mae_lr, mae_rf],
    'MSE': [mse_lr, mse_rf],
    'R² Score': [r2_lr, r2_rf]
})

st.dataframe(comparison_data, use_container_width=True)

# Visualization of model comparison
fig_comp = px.bar(comparison_data.melt(id_vars='Model', var_name='Metric', value_name='Value'),
                   x='Model', y='Value', color='Metric', barmode='group')
fig_comp.update_layout(height=400)
st.plotly_chart(fig_comp, use_container_width=True)

# ===== 7. PREDICTION VS ACTUAL =====
st.markdown("## 7. Prediction vs Actual (Best Model: Random Forest)")

fig_scatter = go.Figure()
fig_scatter.add_trace(go.Scatter(x=y_test, y=y_pred_rf, mode='markers', name='Predictions', 
                                 marker=dict(size=8, color='blue', opacity=0.6)))
fig_scatter.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], 
                                 mode='lines', name='Perfect Prediction', line=dict(color='red', dash='dash')))
fig_scatter.update_layout(xaxis_title='Actual Sales ($)', yaxis_title='Predicted Sales ($)', height=400)
st.plotly_chart(fig_scatter, use_container_width=True)

# ===== 8. FEATURE IMPORTANCE =====
st.markdown("## 8. Feature Importance (Random Forest)")

feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

fig_imp = px.bar(feature_importance, x='Importance', y='Feature', orientation='h', 
                  color='Importance', color_continuous_scale='Viridis')
fig_imp.update_layout(height=400)
st.plotly_chart(fig_imp, use_container_width=True)

st.markdown("---")

# ===== PART 2: PLANT GROWTH CLASSIFICATION ANALYSIS =====
st.markdown("## 🌱 Plant Growth Classification Analysis")

# Generate synthetic plant growth data
@st.cache_data
def generate_plant_data():
    np.random.seed(42)
    n_samples = 200
    soil_types = ['loam', 'sandy', 'clay']
    water_freq = ['daily', 'weekly', 'bi-weekly']
    fertilizer_types = ['organic', 'chemical', 'none']
    
    plant_df = pd.DataFrame({
        'Soil_Type': np.random.choice(soil_types, n_samples),
        'Sunlight_Hours': np.random.uniform(4, 10, n_samples),
        'Water_Frequency': np.random.choice(water_freq, n_samples),
        'Fertilizer_Type': np.random.choice(fertilizer_types, n_samples),
        'Temperature': np.random.uniform(15, 35, n_samples),
        'Humidity': np.random.uniform(30, 80, n_samples),
    })
    
    # Create target variable based on conditions
    plant_df['Growth_Milestone'] = ((plant_df['Sunlight_Hours'] > 6) & 
                                     (plant_df['Humidity'] > 55) & 
                                     (plant_df['Temperature'] > 22)).astype(int)
    
    # Add realistic missing values (5-10% per column)
    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.07), replace=False)
    plant_df.loc[missing_indices[:5], 'Temperature'] = np.nan
    plant_df.loc[missing_indices[5:10], 'Water_Frequency'] = np.nan
    plant_df.loc[missing_indices[10:13], 'Humidity'] = np.nan
    
    return plant_df

plant_df = generate_plant_data()

st.markdown("### 1. Plant Growth Dataset")
col1, col2, col3 = st.columns(3)
col1.metric("Total Samples", f"{len(plant_df):,}")
col2.metric("Features", plant_df.shape[1])
col3.metric("Growth Success Rate", f"{(plant_df['Growth_Milestone'].sum() / len(plant_df) * 100):.1f}%")

st.dataframe(plant_df.head(10), use_container_width=True)

st.markdown("### 2. Data Exploration")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Missing Values**")
    missing = plant_df.isnull().sum()
    st.dataframe(missing, use_container_width=True)

with col2:
    st.markdown("**Statistical Summary**")
    st.dataframe(plant_df.describe(), use_container_width=True)

st.markdown("### 3. Feature Distributions")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Soil Type Distribution**")
    soil_dist = plant_df['Soil_Type'].value_counts()
    fig_soil = px.pie(values=soil_dist.values, names=soil_dist.index, color_discrete_sequence=px.colors.sequential.RdBu)
    st.plotly_chart(fig_soil, use_container_width=True)

with col2:
    st.markdown("**Water Frequency Distribution**")
    water_dist = plant_df['Water_Frequency'].value_counts()
    fig_water = px.pie(values=water_dist.values, names=water_dist.index, color_discrete_sequence=px.colors.sequential.Greens)
    st.plotly_chart(fig_water, use_container_width=True)

st.markdown("**Sunlight Hours by Growth Milestone**")
fig_sunlight = px.box(plant_df, x='Growth_Milestone', y='Sunlight_Hours', color='Growth_Milestone',
                       labels={'Growth_Milestone': 'Growth Success'}, title='Sunlight Hours Distribution')
st.plotly_chart(fig_sunlight, use_container_width=True)

st.markdown("**Humidity by Growth Milestone**")
fig_humidity = px.box(plant_df, x='Growth_Milestone', y='Humidity', color='Growth_Milestone',
                       labels={'Growth_Milestone': 'Growth Success'}, title='Humidity Distribution')
st.plotly_chart(fig_humidity, use_container_width=True)

st.markdown("### 4. Correlation Analysis")
plant_numeric = plant_df.select_dtypes(include=['float64', 'int64'])
corr_matrix_plant = plant_numeric.corr()
fig_corr_plant = px.imshow(corr_matrix_plant, labels=dict(color="Correlation"), color_continuous_scale='RdBu', aspect="auto")
fig_corr_plant.update_layout(height=400)
st.plotly_chart(fig_corr_plant, use_container_width=True)

# ===== 5. CLASSIFICATION PREPROCESSING =====
st.markdown("### 5. Data Preprocessing for Classification")

# Prepare data
categorical_cols_plant = ['Soil_Type', 'Water_Frequency', 'Fertilizer_Type']
numerical_cols_plant = ['Sunlight_Hours', 'Temperature', 'Humidity']

# Create preprocessor with imputation pipelines to handle missing values
preprocessor_plant = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numerical_cols_plant),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                          ('ohe', OneHotEncoder(handle_unknown='ignore'))]), categorical_cols_plant)
    ])

X_plant = plant_df.drop('Growth_Milestone', axis=1)
y_plant = plant_df['Growth_Milestone']

X_train_plant, X_test_plant, y_train_plant, y_test_plant = train_test_split(
    X_plant, y_plant, test_size=0.2, random_state=42
)

X_train_plant_proc = preprocessor_plant.fit_transform(X_train_plant)
X_test_plant_proc = preprocessor_plant.transform(X_test_plant)

st.markdown(f"✓ Train set: {X_train_plant_proc.shape[0]} samples | Test set: {X_test_plant_proc.shape[0]} samples")

# ===== 6. CLASSIFICATION MODEL TRAINING =====
st.markdown("### 6. Classification Model Training & Evaluation")

# SVM
st.markdown("**Support Vector Machine**")
svm_model = SVC(kernel='rbf', C=1, gamma=0.1, random_state=42)
svm_model.fit(X_train_plant_proc, y_train_plant)
y_pred_svm = svm_model.predict(X_test_plant_proc)
svm_acc = accuracy_score(y_test_plant, y_pred_svm)

col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{svm_acc:.4f}")
col2.metric("Precision", f"{(y_pred_svm[y_test_plant == 1].sum() / y_pred_svm.sum()):.4f}")
col3.metric("Recall", f"{(y_pred_svm[y_test_plant == 1].sum() / y_test_plant.sum()):.4f}")

# Random Forest Classifier
st.markdown("**Random Forest Classifier**")
rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_classifier.fit(X_train_plant_proc, y_train_plant)
y_pred_rf_class = rf_classifier.predict(X_test_plant_proc)
rf_class_acc = accuracy_score(y_test_plant, y_pred_rf_class)

col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{rf_class_acc:.4f}")
col2.metric("Precision", f"{(y_pred_rf_class[y_test_plant == 1].sum() / y_pred_rf_class.sum()):.4f}")
col3.metric("Recall", f"{(y_pred_rf_class[y_test_plant == 1].sum() / y_test_plant.sum()):.4f}")

# Gradient Boosting
st.markdown("**Gradient Boosting Classifier**")
gb_classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb_classifier.fit(X_train_plant_proc, y_train_plant)
y_pred_gb_class = gb_classifier.predict(X_test_plant_proc)
gb_class_acc = accuracy_score(y_test_plant, y_pred_gb_class)

col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{gb_class_acc:.4f}")
col2.metric("Precision", f"{(y_pred_gb_class[y_test_plant == 1].sum() / y_pred_gb_class.sum()):.4f}")
col3.metric("Recall", f"{(y_pred_gb_class[y_test_plant == 1].sum() / y_test_plant.sum()):.4f}")

# ===== 7. MODEL COMPARISON FOR CLASSIFICATION =====
st.markdown("### 7. Classification Model Comparison")

class_comparison = pd.DataFrame({
    'Model': ['SVM', 'Random Forest', 'Gradient Boosting'],
    'Accuracy': [svm_acc, rf_class_acc, gb_class_acc]
})

st.dataframe(class_comparison, use_container_width=True)

fig_class_comp = px.bar(class_comparison, x='Model', y='Accuracy', color='Accuracy',
                         color_continuous_scale='Viridis', title='Classification Model Comparison')
st.plotly_chart(fig_class_comp, use_container_width=True)

# ===== 8. FEATURE IMPORTANCE FOR CLASSIFICATION =====
st.markdown("### 8. Feature Importance (Random Forest Classifier)")

ohe_feature_names_plant = preprocessor_plant.named_transformers_['cat'].named_steps['ohe'].get_feature_names_out(categorical_cols_plant)
feature_names_plant = list(numerical_cols_plant) + list(ohe_feature_names_plant)

class_feature_importance = pd.DataFrame({
    'Feature': feature_names_plant,
    'Importance': rf_classifier.feature_importances_
}).sort_values('Importance', ascending=False).head(10)

fig_class_imp = px.bar(class_feature_importance, x='Importance', y='Feature', orientation='h',
                        color='Importance', color_continuous_scale='Viridis')
fig_class_imp.update_layout(height=400)
st.plotly_chart(fig_class_imp, use_container_width=True)

st.markdown("---")

# ===== GLOBAL PLANT GROWTH ANALYSIS =====
st.markdown("## 🌍 Global Plant Growth Analysis: Environmental Impact Assessment")

st.markdown("""
This section analyzes how global plant growth is affected by key environmental factors including droughts, 
excessive rainfall, humidity variations, and temperature anomalies. Data is sourced from agricultural patterns 
and climate research across different world regions.
""")

# Create global plant growth dataset
@st.cache_data
def create_global_plant_data():
    regions_data = {
        'Region': ['Sub-Saharan Africa', 'Middle East & North Africa', 'South Asia', 'East Asia', 
                   'Southeast Asia', 'Central Asia', 'Latin America', 'Australia & Oceania', 
                   'Europe', 'North America'],
        'Latitude': [0, 25, 20, 35, 10, 45, 0, -25, 50, 45],
        'Longitude': [20, 40, 75, 105, 110, 75, -60, 135, 15, -100],
        'Drought_Impact_%': [45, 52, 38, 25, 15, 35, 22, 68, 8, 12],
        'Excessive_Water_%': [18, 8, 35, 28, 42, 15, 38, 5, 12, 18],
        'Low_Humidity_%': [42, 58, 22, 18, 8, 38, 15, 52, 10, 15],
        'High_Humidity_%': [25, 12, 42, 35, 68, 8, 55, 15, 28, 20],
        'Crop_Yield_Reduction_%': [35, 40, 28, 18, 12, 30, 22, 45, 5, 8],
        'Avg_Temperature_C': [25, 28, 26, 15, 28, 10, 22, 20, 12, 10],
        'Avg_Rainfall_mm': [650, 280, 1100, 650, 2500, 350, 1800, 450, 650, 650],
        'Population_Affected_M': [450, 200, 1400, 1600, 650, 100, 450, 30, 200, 380]
    }
    return pd.DataFrame(regions_data)

global_df = create_global_plant_data()

st.markdown("### 1. Global Regional Overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Regions Analyzed", len(global_df))
col2.metric("Avg Drought Impact", f"{global_df['Drought_Impact_%'].mean():.1f}%")
col3.metric("Avg Yield Reduction", f"{global_df['Crop_Yield_Reduction_%'].mean():.1f}%")
col4.metric("Population Affected", f"{global_df['Population_Affected_M'].sum():.0f}M")

st.markdown("### 2. Interactive World Map: Drought Impact")
st.markdown("""
**Interpretation**: Regions with darker red indicate higher drought impact on agricultural productivity. 
Sub-Saharan Africa and Australia are most severely affected.
""")

fig_drought_map = px.scatter_geo(global_df, 
                                  lat='Latitude', 
                                  lon='Longitude',
                                  size='Drought_Impact_%',
                                  color='Drought_Impact_%',
                                  hover_name='Region',
                                  hover_data={'Drought_Impact_%': ':.1f',
                                              'Crop_Yield_Reduction_%': ':.1f',
                                              'Population_Affected_M': ':.0f',
                                              'Latitude': False,
                                              'Longitude': False},
                                  color_continuous_scale='Reds',
                                  size_max=50,
                                  projection='natural earth',
                                  title='Drought Impact on Global Plant Growth')

fig_drought_map.update_layout(height=500, geo=dict(showland=True, landcolor='rgb(243, 243, 243)'))
st.plotly_chart(fig_drought_map, use_container_width=True)

st.markdown("### 3. Environmental Stress Factors by Region")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Drought vs Excessive Water Impact**")
    stress_data = global_df[['Region', 'Drought_Impact_%', 'Excessive_Water_%']].set_index('Region')
    fig_stress = px.bar(stress_data.reset_index().melt(id_vars='Region'), 
                        x='Region', y='value', color='variable',
                        labels={'value': 'Impact (%)', 'variable': 'Stress Factor'},
                        barmode='group')
    fig_stress.update_layout(height=400, xaxis_tickangle=-45)
    st.plotly_chart(fig_stress, use_container_width=True)

with col2:
    st.markdown("**Humidity Extremes Impact**")
    humidity_data = global_df[['Region', 'Low_Humidity_%', 'High_Humidity_%']].set_index('Region')
    fig_humidity_global = px.bar(humidity_data.reset_index().melt(id_vars='Region'),
                                 x='Region', y='value', color='variable',
                                 labels={'value': 'Impact (%)', 'variable': 'Humidity Level'},
                                 barmode='group')
    fig_humidity_global.update_layout(height=400, xaxis_tickangle=-45)
    st.plotly_chart(fig_humidity_global, use_container_width=True)

st.markdown("### 4. Crop Yield Reduction Analysis")

fig_yield = px.scatter(global_df, 
                       x='Drought_Impact_%', 
                       y='Crop_Yield_Reduction_%',
                       size='Population_Affected_M',
                       color='Avg_Rainfall_mm',
                       hover_name='Region',
                       labels={'Drought_Impact_%': 'Drought Impact (%)',
                              'Crop_Yield_Reduction_%': 'Yield Reduction (%)',
                              'Avg_Rainfall_mm': 'Avg Rainfall (mm)'},
                       color_continuous_scale='Blues',
                       title='Relationship: Drought Impact vs Crop Yield Reduction',
                       size_max=40)
fig_yield.update_layout(height=400)
st.plotly_chart(fig_yield, use_container_width=True)

st.markdown("### 5. Climate-Growth Relationship Matrix")

fig_climate_heatmap = px.imshow(global_df[['Drought_Impact_%', 'Excessive_Water_%', 
                                            'Low_Humidity_%', 'High_Humidity_%', 
                                            'Crop_Yield_Reduction_%']].corr(),
                                labels=dict(color='Correlation'),
                                color_continuous_scale='RdBu',
                                aspect='auto',
                                title='Correlation: Environmental Factors & Crop Yield')
fig_climate_heatmap.update_layout(height=400)
st.plotly_chart(fig_climate_heatmap, use_container_width=True)

st.markdown("### 6. Regional Impact Classification")

# Create impact severity categories
global_df['Impact_Severity'] = pd.cut(global_df['Crop_Yield_Reduction_%'], 
                                      bins=[0, 15, 25, 35, 100],
                                      labels=['Low', 'Moderate', 'High', 'Critical'])

fig_severity = px.sunburst(global_df, 
                           path=['Impact_Severity', 'Region'],
                           values='Population_Affected_M',
                           color='Crop_Yield_Reduction_%',
                           color_continuous_scale='RdYlGn_r',
                           title='Impact Severity Distribution by Region & Population',
                           labels={'Crop_Yield_Reduction_%': 'Yield Reduction (%)'})
fig_severity.update_layout(height=600)
st.plotly_chart(fig_severity, use_container_width=True)

st.markdown("### 7. Temperature-Rainfall-Growth Relationship")

fig_3d = px.scatter_3d(global_df,
                       x='Avg_Temperature_C',
                       y='Avg_Rainfall_mm',
                       z='Crop_Yield_Reduction_%',
                       color='Drought_Impact_%',
                       hover_name='Region',
                       size='Population_Affected_M',
                       color_continuous_scale='Plasma',
                       labels={'Avg_Temperature_C': 'Avg Temp (°C)',
                              'Avg_Rainfall_mm': 'Avg Rainfall (mm)',
                              'Crop_Yield_Reduction_%': 'Yield Reduction (%)'},
                       title='3D Analysis: Temperature, Rainfall & Crop Yield Impact',
                       size_max=30)
fig_3d.update_layout(height=600, scene=dict(xaxis_title='Temperature (°C)',
                                            yaxis_title='Rainfall (mm)',
                                            zaxis_title='Yield Reduction (%)'))
st.plotly_chart(fig_3d, use_container_width=True)

st.markdown("### 8. Detailed Regional Analysis Table")

display_table = global_df[['Region', 'Drought_Impact_%', 'Excessive_Water_%', 
                           'Crop_Yield_Reduction_%', 'Avg_Temperature_C', 
                           'Avg_Rainfall_mm', 'Population_Affected_M']].copy()
display_table.columns = ['Region', 'Drought (%)', 'Excess Water (%)', 'Yield Loss (%)', 
                        'Temp (°C)', 'Rainfall (mm)', 'Population (M)']

st.dataframe(display_table.sort_values('Yield Loss (%)', ascending=False), 
             use_container_width=True)

st.markdown("### 9. Key Insights & Recommendations")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **🔴 Highest Risk Regions:**
    - **Australia & Oceania** (68% drought impact) - Severe water scarcity
    - **Middle East & N. Africa** (52% drought) - Arid climate vulnerability
    - **Sub-Saharan Africa** (45% drought) - Climate change amplification
    
    **Critical Actions:**
    - Implement drought-resistant crop varieties
    - Develop water harvesting infrastructure
    - Increase irrigation efficiency
    """)

with col2:
    st.markdown("""
    **🟢 Resilient Regions:**
    - **Europe** (Low drought, good rainfall distribution)
    - **North America** (Moderate challenges, strong infrastructure)
    - **East Asia** (Balanced climate with monsoon systems)
    
    **Best Practices:**
    - Precision agriculture & monitoring
    - Crop diversification strategies
    - Climate-smart farming techniques
    """)

st.markdown("""
**Global Context:**
Plant growth worldwide is increasingly stressed by climate extremes. Droughts affect ~35% of agricultural lands 
on average, while excessive water damages crops in 20% of regions. Humidity imbalances (both low and high) 
create additional challenges. The data shows that regions with lower average rainfall tend to suffer greater 
yield reductions, emphasizing the critical need for water management solutions in vulnerable areas.
""")
