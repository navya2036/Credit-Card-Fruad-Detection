import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 30px;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
    }
    .fraud-box {
        background-color: #ffebee;
        border: 2px solid #f44336;
        color: #c62828;
    }
    .normal-box {
        background-color: #e8f5e9;
        border: 2px solid #4caf50;
        color: #2e7d32;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">💳 Credit Card Fraud Detection System</p>', unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_models():
    """Load the trained model and scaler"""
    try:
        model_path = Path('models/best_fraud_detection_model.pkl')
        scaler_path = Path('models/scaler.pkl')
        
        if model_path.exists() and scaler_path.exists():
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            return model, scaler, True
        else:
            return None, None, False
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, False

model, scaler, models_loaded = load_models()

# Sidebar
st.sidebar.title("📊 Navigation")
app_mode = st.sidebar.selectbox("Choose Mode", 
                                 ["Single Prediction", "Batch Prediction", "Model Info"])

if not models_loaded:
    st.error("⚠️ Models not found! Please train the model first by running the Jupyter notebook.")
    st.info("👉 Run `fraud_detection.ipynb` to train and save the model.")
    st.stop()

# Model Information
if app_mode == "Model Info":
    st.header("📈 Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Details")
        st.write(f"**Model Type:** {type(model).__name__}")
        st.write(f"**Number of Features:** {model.n_features_in_ if hasattr(model, 'n_features_in_') else 'N/A'}")
        
        if hasattr(model, 'feature_importances_'):
            st.subheader("🔍 Feature Importance")
            feature_names = [f"Feature_{i}" for i in range(len(model.feature_importances_))]
            importance_df = pd.DataFrame({
                'Feature': feature_names[:10],  # Top 10
                'Importance': model.feature_importances_[:10]
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                        title='Top 10 Most Important Features')
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📋 About This System")
        st.markdown("""
        This fraud detection system uses machine learning to identify potentially fraudulent credit card transactions.
        
        **How it works:**
        1. Input transaction features
        2. Model analyzes patterns
        3. Predicts fraud probability
        4. Provides risk assessment
        
        **Key Metrics:**
        - High precision to minimize false alarms
        - High recall to catch actual fraud
        - Balanced F1-score for overall performance
        """)
        
        st.subheader("⚠️ Important Notes")
        st.warning("""
        - This is a predictive model and not 100% accurate
        - Always verify suspicious transactions
        - Use as a decision support tool
        """)

# Single Prediction Mode
elif app_mode == "Single Prediction":
    st.header("🔍 Single Transaction Analysis")
    
    st.info("Enter the transaction details below to check for potential fraud.")
    
    # Create input form
    with st.form("prediction_form"):
        st.subheader("Transaction Details")
        
        # Assuming standard creditcard dataset format
        col1, col2 = st.columns(2)
        
        with col1:
            time = st.number_input("Time (seconds since first transaction)", 
                                   min_value=0.0, value=0.0, step=1.0)
            amount = st.number_input("Transaction Amount ($)", 
                                     min_value=0.0, value=100.0, step=0.01)
        
        # V1-V28 features (PCA transformed)
        st.subheader("Anonymized Features (V1-V28)")
        st.caption("These are PCA-transformed features from the original dataset")
        
        v_features = []
        cols = st.columns(4)
        for i in range(28):
            with cols[i % 4]:
                v_val = st.number_input(f"V{i+1}", value=0.0, step=0.01, 
                                       format="%.4f", key=f"v{i+1}")
                v_features.append(v_val)
        
        submitted = st.form_submit_button("🔍 Check Transaction", use_container_width=True)
        
        if submitted:
            # Prepare input data
            input_data = pd.DataFrame([[time, *v_features, amount]], 
                                     columns=['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount'])
            
            # Scale the input
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction == 1:
                    st.markdown('<div class="prediction-box fraud-box">⚠️ FRAUD DETECTED</div>', 
                               unsafe_allow_html=True)
                else:
                    st.markdown('<div class="prediction-box normal-box">✅ NORMAL TRANSACTION</div>', 
                               unsafe_allow_html=True)
            
            with col2:
                st.metric("Fraud Probability", f"{probability[1]:.2%}")
                st.metric("Normal Probability", f"{probability[0]:.2%}")
            
            with col3:
                risk_level = "🔴 HIGH" if probability[1] > 0.7 else "🟡 MEDIUM" if probability[1] > 0.3 else "🟢 LOW"
                st.metric("Risk Level", risk_level)
                st.metric("Transaction Amount", f"${amount:,.2f}")
            
            # Probability gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = probability[1] * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Fraud Probability", 'font': {'size': 24}},
                delta = {'reference': 50, 'increasing': {'color': "red"}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "darkblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 30], 'color': '#4caf50'},
                        {'range': [30, 70], 'color': '#ffc107'},
                        {'range': [70, 100], 'color': '#f44336'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.subheader("💡 Recommendations")
            if prediction == 1:
                st.error("""
                **Suggested Actions:**
                - Block or flag this transaction immediately
                - Contact the cardholder for verification
                - Review recent transaction history
                - Monitor account for additional suspicious activity
                """)
            else:
                st.success("""
                **Transaction appears normal:**
                - No immediate action required
                - Continue standard monitoring
                - Process transaction as usual
                """)

# Batch Prediction Mode
elif app_mode == "Batch Prediction":
    st.header("📁 Batch Transaction Analysis")
    
    st.info("Upload a CSV file with multiple transactions to analyze them in bulk.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            
            st.subheader("📄 Uploaded Data Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            st.write(f"**Total Transactions:** {len(df)}")
            
            if st.button("🚀 Analyze All Transactions", use_container_width=True):
                with st.spinner("Analyzing transactions..."):
                    # Prepare data
                    X = df.drop('Class', axis=1) if 'Class' in df.columns else df
                    
                    # Scale and predict
                    X_scaled = scaler.transform(X)
                    predictions = model.predict(X_scaled)
                    probabilities = model.predict_proba(X_scaled)[:, 1]
                    
                    # Add predictions to dataframe
                    df['Prediction'] = predictions
                    df['Fraud_Probability'] = probabilities
                    df['Risk_Level'] = pd.cut(probabilities, 
                                              bins=[0, 0.3, 0.7, 1.0], 
                                              labels=['Low', 'Medium', 'High'])
                
                # Display results
                st.success("✅ Analysis Complete!")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Transactions", len(df))
                with col2:
                    fraud_count = (predictions == 1).sum()
                    st.metric("Fraudulent Transactions", fraud_count)
                with col3:
                    fraud_rate = (fraud_count / len(df)) * 100
                    st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
                with col4:
                    high_risk = (df['Risk_Level'] == 'High').sum()
                    st.metric("High Risk Transactions", high_risk)
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # Fraud distribution
                    fraud_dist = pd.Series(predictions).value_counts()
                    fig = px.pie(values=fraud_dist.values, 
                                names=['Normal', 'Fraud'],
                                title='Transaction Classification',
                                color_discrete_sequence=['#4caf50', '#f44336'])
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Risk level distribution
                    risk_dist = df['Risk_Level'].value_counts()
                    fig = px.bar(x=risk_dist.index, y=risk_dist.values,
                                title='Risk Level Distribution',
                                labels={'x': 'Risk Level', 'y': 'Count'},
                                color=risk_dist.index,
                                color_discrete_map={'Low': '#4caf50', 
                                                   'Medium': '#ffc107', 
                                                   'High': '#f44336'})
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show flagged transactions
                st.subheader("⚠️ Flagged Transactions (Fraud Predictions)")
                fraud_df = df[df['Prediction'] == 1].sort_values('Fraud_Probability', ascending=False)
                
                if len(fraud_df) > 0:
                    st.dataframe(fraud_df, use_container_width=True)
                    
                    # Download button
                    csv = fraud_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Flagged Transactions",
                        data=csv,
                        file_name="flagged_transactions.csv",
                        mime="text/csv",
                    )
                else:
                    st.success("No fraudulent transactions detected!")
                
                # Show all results
                with st.expander("📊 View All Results"):
                    st.dataframe(df, use_container_width=True)
                    
                    csv_all = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download All Results",
                        data=csv_all,
                        file_name="all_predictions.csv",
                        mime="text/csv",
                    )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Make sure your CSV has the correct format with all required features.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Credit Card Fraud Detection System | Built with Streamlit 💙</p>
    <p>⚠️ For demonstration purposes only. Always verify predictions with domain experts.</p>
</div>
""", unsafe_allow_html=True)
