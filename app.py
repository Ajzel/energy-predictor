import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from io import StringIO

class StreamlitEnergyPredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = ['temperature', 'humidity', 'light', 'motion']
        self.model_metrics = {}
    
    def train_with_user_data(self, data_file=None, data_df=None):
        """Train model with user-provided data"""
        try:
            # Load data
            if data_df is not None:
                data = data_df
            elif data_file and os.path.exists(data_file):
                if data_file.endswith('.csv'):
                    data = pd.read_csv(data_file)
                elif data_file.endswith(('.xlsx', '.xls')):
                    data = pd.read_excel(data_file)
                else:
                    raise ValueError("Unsupported file format")
            else:
                # Generate sample data if no user data provided
                data = self._generate_sample_data()
            
            st.write(f"Training with {len(data)} records")
            st.write(f"Columns: {list(data.columns)}")
            
            # Prepare features and target
            feature_cols = [col for col in self.feature_names if col in data.columns]
            if not feature_cols:
                raise ValueError(f"No matching feature columns found. Expected: {self.feature_names}")
            
            X = data[feature_cols]
            
            # Target column (energy consumption)
            target_cols = ['energy_consumption', 'energy', 'power', 'consumption', 'kwh']
            target_col = None
            for col in target_cols:
                if col in data.columns:
                    target_col = col
                    break
            
            if target_col is None:
                raise ValueError(f"No target column found. Expected one of: {target_cols}")
            
            y = data[target_col]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred_train = self.model.predict(X_train_scaled)
            y_pred_test = self.model.predict(X_test_scaled)
            
            self.model_metrics = {
                'train_mse': mean_squared_error(y_train, y_pred_train),
                'test_mse': mean_squared_error(y_test, y_pred_test),
                'train_r2': r2_score(y_train, y_pred_train),
                'test_r2': r2_score(y_test, y_pred_test),
                'feature_importance': dict(zip(feature_cols, self.model.feature_importances_)),
                'training_samples': len(data),
                'features_used': feature_cols,
                'target_column': target_col
            }
            
            self.feature_names = feature_cols
            self.is_trained = True
            
            return self.model_metrics
            
        except Exception as e:
            st.error(f"Training error: {e}")
            return {'error': str(e)}
    
    def predict_single(self, **kwargs):
        """Make a single prediction with given parameters"""
        if not self.is_trained:
            return {'error': 'Model not trained yet'}
        
        try:
            # Prepare input data
            input_values = []
            missing_features = []
            
            for feature in self.feature_names:
                if feature in kwargs:
                    input_values.append(float(kwargs[feature]))
                else:
                    missing_features.append(feature)
            
            if missing_features:
                return {'error': f'Missing required features: {missing_features}'}
            
            # Scale and predict
            input_array = np.array([input_values])
            input_scaled = self.scaler.transform(input_array)
            prediction = self.model.predict(input_scaled)[0]
            
            # Get prediction confidence
            confidence = self._calculate_confidence(input_values)
            
            return {
                'prediction': max(0, prediction),
                'confidence': confidence,
                'input_features': dict(zip(self.feature_names, input_values))
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_confidence(self, input_values):
        """Calculate prediction confidence based on input ranges"""
        confidence_scores = []
        
        # Define typical ranges for each feature
        typical_ranges = {
            'temperature': (15, 35),
            'humidity': (30, 80),
            'light': (0, 100),
            'motion': (0, 1)
        }
        
        for i, feature in enumerate(self.feature_names):
            value = input_values[i]
            min_val, max_val = typical_ranges.get(feature, (0, 100))
            
            if min_val <= value <= max_val:
                confidence_scores.append(1.0)
            else:
                distance = min(abs(value - min_val), abs(value - max_val))
                confidence_scores.append(max(0.1, 1.0 - (distance / max_val)))
        
        return round(np.mean(confidence_scores) * 100, 1)
    
    def _generate_sample_data(self, num_samples=1000):
        """Generate sample training data"""
        np.random.seed(42)
        
        temperature = np.random.normal(25, 5, num_samples)
        humidity = np.random.normal(60, 15, num_samples)
        light = np.random.uniform(0, 100, num_samples)
        motion = np.random.binomial(1, 0.3, num_samples)
        
        # Calculate energy consumption
        energy = (
            temperature * 2.5 +
            humidity * 0.8 +
            (100 - light) * 1.2 +
            motion * 15 +
            np.random.normal(0, 5, num_samples)
        )
        energy = np.maximum(energy, 10)
        
        return pd.DataFrame({
            'temperature': temperature,
            'humidity': humidity,
            'light': light,
            'motion': motion,
            'energy_consumption': energy
        })
    
    def save_model(self, filepath='energy_model.pkl'):
        """Save trained model"""
        if self.is_trained:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'metrics': self.model_metrics
            }, filepath)
            return True
        return False
    
    def load_model(self, filepath='energy_model.pkl'):
        """Load trained model"""
        try:
            saved_objects = joblib.load(filepath)
            self.model = saved_objects['model']
            self.scaler = saved_objects['scaler']
            self.feature_names = saved_objects.get('feature_names', self.feature_names)
            self.model_metrics = saved_objects.get('metrics', {})
            self.is_trained = True
            return True
        except:
            return False

# Initialize the app
st.set_page_config(page_title="Energy Consumption Predictor", page_icon="⚡", layout="wide")

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = StreamlitEnergyPredictor()
    # Try to load existing model
    if os.path.exists('energy_model.pkl'):
        st.session_state.predictor.load_model()

if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []

# Main app
st.title("⚡ Energy Consumption Predictor")
st.markdown("Predict energy consumption based on environmental conditions")

# Sidebar for model training
st.sidebar.header("🔧 Model Training")

# Option to train with sample data
if st.sidebar.button("Train with Sample Data"):
    with st.spinner("Training model with sample data..."):
        metrics = st.session_state.predictor.train_with_user_data()
        if 'error' not in metrics:
            st.session_state.predictor.save_model()
            st.sidebar.success("Model trained successfully!")
            st.sidebar.write("**Training Metrics:**")
            st.sidebar.write(f"Training R²: {metrics['train_r2']:.3f}")
            st.sidebar.write(f"Test R²: {metrics['test_r2']:.3f}")
            st.sidebar.write(f"Training Samples: {metrics['training_samples']}")
        else:
            st.sidebar.error(f"Training failed: {metrics['error']}")

# Option to upload custom data
st.sidebar.subheader("📁 Upload Custom Training Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=['csv'])

if uploaded_file is not None:
    try:
        # Read the uploaded file
        df = pd.read_csv(uploaded_file)
        st.sidebar.write("**Data Preview:**")
        st.sidebar.dataframe(df.head())
        
        if st.sidebar.button("Train with Uploaded Data"):
            with st.spinner("Training model with uploaded data..."):
                metrics = st.session_state.predictor.train_with_user_data(data_df=df)
                if 'error' not in metrics:
                    st.session_state.predictor.save_model()
                    st.sidebar.success("Model trained successfully!")
                    st.sidebar.write("**Training Metrics:**")
                    st.sidebar.write(f"Training R²: {metrics['train_r2']:.3f}")
                    st.sidebar.write(f"Test R²: {metrics['test_r2']:.3f}")
                    st.sidebar.write(f"Training Samples: {metrics['training_samples']}")
                else:
                    st.sidebar.error(f"Training failed: {metrics['error']}")
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")

# Model status
if st.session_state.predictor.is_trained:
    st.sidebar.success("✅ Model is trained and ready!")
    if st.session_state.predictor.model_metrics:
        st.sidebar.write("**Current Model Info:**")
        metrics = st.session_state.predictor.model_metrics
        st.sidebar.write(f"Test R²: {metrics.get('test_r2', 'N/A'):.3f}")
        st.sidebar.write(f"Features: {', '.join(metrics.get('features_used', []))}")
else:
    st.sidebar.warning("⚠️ Model not trained yet!")

# Main prediction interface
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🎯 Make Prediction")
    
    if st.session_state.predictor.is_trained:
        # Input form
        with st.form("prediction_form"):
            col_temp, col_humid = st.columns(2)
            with col_temp:
                temperature = st.number_input("Temperature (°C)", min_value=-50.0, max_value=60.0, value=25.0)
            with col_humid:
                humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
            
            col_light, col_motion = st.columns(2)
            with col_light:
                light = st.number_input("Light Level (%)", min_value=0.0, max_value=100.0, value=50.0)
            with col_motion:
                motion = st.selectbox("Motion Detection", options=[0, 1], format_func=lambda x: "No Motion" if x == 0 else "Motion Detected")
            
            notes = st.text_area("Notes (optional)", placeholder="Add any additional notes...")
            
            submitted = st.form_submit_button("🔮 Predict Energy Consumption")
            
            if submitted:
                prediction_params = {
                    'temperature': temperature,
                    'humidity': humidity,
                    'light': light,
                    'motion': motion
                }
                
                result = st.session_state.predictor.predict_single(**prediction_params)
                
                if 'error' not in result:
                    # Display prediction
                    st.success(f"**Predicted Energy Consumption: {result['prediction']:.2f} kWh**")
                    st.info(f"**Confidence: {result['confidence']:.1f}%**")
                    
                    # Add to history
                    history_entry = {
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'temperature': temperature,
                        'humidity': humidity,
                        'light': light,
                        'motion': motion,
                        'prediction': result['prediction'],
                        'confidence': result['confidence'],
                        'notes': notes
                    }
                    st.session_state.predictions_history.append(history_entry)
                else:
                    st.error(f"Prediction failed: {result['error']}")
    else:
        st.info("Please train the model first using the sidebar options.")

with col2:
    st.header("📊 Model Performance")
    
    if st.session_state.predictor.is_trained and st.session_state.predictor.model_metrics:
        metrics = st.session_state.predictor.model_metrics
        
        # Display metrics
        st.metric("Test R² Score", f"{metrics.get('test_r2', 0):.3f}")
        st.metric("Training Samples", metrics.get('training_samples', 0))
        
        # Feature importance chart
        if 'feature_importance' in metrics:
            feature_imp = metrics['feature_importance']
            fig = go.Figure(data=[
                go.Bar(
                    x=list(feature_imp.values()),
                    y=list(feature_imp.keys()),
                    orientation='h'
                )
            ])
            fig.update_layout(
                title="Feature Importance",
                xaxis_title="Importance",
                yaxis_title="Features",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

# Predictions history
if st.session_state.predictions_history:
    st.header("📜 Prediction History")
    
    # Convert to DataFrame for better display
    history_df = pd.DataFrame(st.session_state.predictions_history)
    
    # Display recent predictions
    st.dataframe(history_df.tail(10), use_container_width=True)
    
    # Plot predictions over time
    if len(history_df) > 1:
        fig = px.line(
            history_df.tail(20), 
            x='timestamp', 
            y='prediction',
            title="Recent Predictions Timeline",
            markers=True
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Download history
    csv = history_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Prediction History",
        data=csv,
        file_name=f"energy_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

# Batch prediction feature
st.header("📦 Batch Predictions")
st.markdown("Upload a CSV file with multiple data points for batch predictions")

batch_file = st.file_uploader("Choose CSV file for batch prediction", type=['csv'])

if batch_file is not None and st.session_state.predictor.is_trained:
    try:
        batch_df = pd.read_csv(batch_file)
        st.write("**Batch Data Preview:**")
        st.dataframe(batch_df.head())
        
        if st.button("🚀 Run Batch Predictions"):
            with st.spinner("Processing batch predictions..."):
                predictions = []
                
                for _, row in batch_df.iterrows():
                    pred_params = {
                        'temperature': row.get('temperature', 25),
                        'humidity': row.get('humidity', 60),
                        'light': row.get('light', 50),
                        'motion': row.get('motion', 0)
                    }
                    
                    result = st.session_state.predictor.predict_single(**pred_params)
                    
                    if 'error' not in result:
                        predictions.append({
                            **pred_params,
                            'predicted_energy': result['prediction'],
                            'confidence': result['confidence']
                        })
                
                if predictions:
                    pred_df = pd.DataFrame(predictions)
                    st.success(f"Successfully processed {len(predictions)} predictions!")
                    st.dataframe(pred_df)
                    
                    # Download batch results
                    csv_results = pred_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Batch Results",
                        data=csv_results,
                        file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.error("No valid predictions could be made")
    
    except Exception as e:
        st.error(f"Error processing batch file: {e}")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • Energy Consumption Predictor")