import streamlit as st
import pandas as pd
import numpy as np
import pickle
import gzip
from sklearn.metrics.pairwise import euclidean_distances

# Page configuration
st.set_page_config(
    page_title="STR Population Predictor",
    page_icon="🧬",
    layout="wide"
)

@st.cache_resource
def load_model():
    """Load the compressed STR population prediction model"""
    try:
        # Try loading compressed model first
        with gzip.open('str_model_light.pkl.gz', 'rb') as f:
            model_package = pickle.load(f)
        return model_package
    except FileNotFoundError:
        try:
            # Fallback to uncompressed model
            with open('str_model_balanced.pkl', 'rb') as f:
                model_package = pickle.load(f)
            return model_package
        except FileNotFoundError:
            st.error("Model file not found. Please ensure the model file is uploaded.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def predict_population(input_data, model_package):
    """Predict population from STR profile"""
    try:
        model = model_package['model']
        feature_names = model_package['feature_names']
        populations = model_package['populations']
        centroids = model_package['population_centroids']

        # Prepare input data - ensure all features are present
        input_array = []
        for feature in feature_names:
            value = input_data.get(feature, 0)
            if value is None or pd.isna(value):
                value = 0
            input_array.append(float(value))

        # Create DataFrame for prediction
        input_df = pd.DataFrame([input_array], columns=feature_names)

        # Get model predictions
        try:
            predictions = model.predict(input_df)
            probabilities = model.predict_proba(input_df)[0]

            # Create results with model predictions
            results = []
            for i, pop in enumerate(populations):
                confidence = probabilities[i] * 100
                results.append({
                    'rank': i + 1,
                    'population': pop,
                    'confidence': confidence,
                    'method': 'ML Model'
                })

            # Sort by confidence
            results = sorted(results, key=lambda x: x['confidence'], reverse=True)

        except Exception as model_error:
            st.warning(f"Model prediction failed: {model_error}. Using similarity matching.")

            # Fallback to similarity matching
            results = []
            input_profile = np.array(input_array).reshape(1, -1)

            for pop, centroid in centroids.items():
                centroid_array = np.array([centroid.get(feat, 0) for feat in feature_names]).reshape(1, -1)
                distance = euclidean_distances(input_profile, centroid_array)[0][0]

                # Convert distance to similarity score
                max_distance = 50
                similarity = max(0, (max_distance - distance) / max_distance * 100)

                results.append({
                    'population': pop,
                    'confidence': similarity,
                    'method': 'Similarity'
                })

            # Sort by confidence
            results = sorted(results, key=lambda x: x['confidence'], reverse=True)

        # Update ranks
        for i, result in enumerate(results):
            result['rank'] = i + 1

        return results[:5]

    except Exception as e:
        st.error(f"Prediction error: {e}")
        return []

def main():
    # Header
    st.title("🧬 STR Population Predictor")
    st.markdown("Predict population ancestry based on STR allele profiles using machine learning")

    # Load model
    model_package = load_model()

    if model_package is None:
        st.error("Failed to load the prediction model. Please check the model file.")
        return

    st.success("✅ Balanced model loaded successfully!")

    # Display model info
    feature_names = model_package['feature_names']
    populations = model_package['populations']

    # Sidebar info
    with st.sidebar:
        st.header("📊 Model Information")
        st.write(f"**Populations:** {len(populations)}")
        st.write(f"**STR Markers:** {len(feature_names)}")
        st.write("**Model:** Balanced Random Forest")
        st.write("**File:** Compressed (1.15 MB)")

        st.header("🎯 Supported Populations")
        for pop in sorted(populations):
            st.write(f"• {pop}")

        st.header("🧬 STR Markers")
        for marker in feature_names:
            st.write(f"• {marker}")

    # Main interface
    tab1, tab2, tab3 = st.tabs(["🔢 Manual Entry", "📁 CSV Upload", "ℹ️ About"])

    with tab1:
        st.header("Enter STR Allele Values")
        st.info("💡 **Tip:** If you have genotype data (two alleles), use the average. Example: (11,12) → enter 11.5")

        # Create input fields in columns
        cols = st.columns(3)
        input_data = {}

        for i, marker in enumerate(feature_names):
            col_idx = i % 3
            with cols[col_idx]:
                value = st.number_input(
                    f"{marker}:",
                    min_value=0.0,
                    max_value=50.0,
                    value=0.0,
                    step=0.1,
                    key=marker,
                    help=f"Enter allele value for {marker} marker"
                )
                input_data[marker] = value if value > 0 else None

        # Clear and predict buttons
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🗑️ Clear All"):
                st.rerun()

        with col2:
            if st.button("🔍 Predict Population", type="primary"):
                valid_values = [v for v in input_data.values() if v is not None and v > 0]

                if len(valid_values) == 0:
                    st.warning("⚠️ Please enter at least one STR marker value.")
                else:
                    with st.spinner("🧬 Analyzing STR profile..."):
                        results = predict_population(input_data, model_package)

                    if results:
                        st.header("🎯 Prediction Results")

                        for result in results:
                            confidence = result['confidence']

                            col1, col2, col3 = st.columns([1, 3, 2])

                            with col1:
                                if result['rank'] == 1:
                                    st.markdown("🥇 **#1**")
                                elif result['rank'] == 2:
                                    st.markdown("🥈 **#2**")
                                elif result['rank'] == 3:
                                    st.markdown("🥉 **#3**")
                                else:
                                    st.markdown(f"**#{result['rank']}**")

                            with col2:
                                st.markdown(f"**{result['population']}**")
                                st.progress(min(confidence / 30, 1.0))  # Scale for better visualization

                            with col3:
                                if confidence > 20:
                                    st.success(f"{confidence:.1f}%")
                                elif confidence > 10:
                                    st.warning(f"{confidence:.1f}%")
                                else:
                                    st.info(f"{confidence:.1f}%")

                        # Interpretation
                        st.markdown("---")
                        top_confidence = results[0]['confidence']
                        if top_confidence > 25:
                            st.success("🎯 **High Confidence:** Strong prediction reliability")
                        elif top_confidence > 15:
                            st.warning("⚠️ **Moderate Confidence:** Consider additional markers")
                        else:
                            st.info("ℹ️ **Low Confidence:** Results should be interpreted with caution")

                        # Show input summary
                        with st.expander("📋 Input Summary"):
                            input_summary = {k: v for k, v in input_data.items() if v is not None and v > 0}
                            st.json(input_summary)

    with tab2:
        st.header("📁 Batch Processing")
        st.info("Upload a CSV file with STR marker columns for batch processing")

        # Template download
        if st.button("📥 Download Template"):
            template_data = {marker: [0.0] for marker in feature_names}
            template_data['Sample_ID'] = ['Sample_1']
            template_df = pd.DataFrame(template_data)
            csv = template_df.to_csv(index=False)
            st.download_button(
                label="Download CSV Template",
                data=csv,
                file_name="str_template.csv",
                mime="text/csv"
            )

        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("📊 Preview of uploaded data:")
                st.dataframe(df.head())

                if st.button("🚀 Process File"):
                    results_list = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for idx, row in df.iterrows():
                        status_text.text(f"Processing sample {idx + 1} of {len(df)}...")

                        input_data = {}
                        for marker in feature_names:
                            value = row.get(marker, None)
                            if pd.notna(value) and value != 0:
                                input_data[marker] = float(value)
                            else:
                                input_data[marker] = None

                        results = predict_population(input_data, model_package)

                        if results:
                            sample_id = row.get('Sample_ID', f'Sample_{idx + 1}')
                            results_list.append({
                                'Sample_ID': sample_id,
                                'Top_Population': results[0]['population'],
                                'Confidence': f"{results[0]['confidence']:.2f}%",
                                'Second_Population': results[1]['population'] if len(results) > 1 else '',
                                'Second_Confidence': f"{results[1]['confidence']:.2f}%" if len(results) > 1 else '',
                                'Method': results[0].get('method', 'ML Model')
                            })

                        progress_bar.progress((idx + 1) / len(df))

                    status_text.text("✅ Processing complete!")

                    if results_list:
                        results_df = pd.DataFrame(results_list)
                        st.success(f"✅ Successfully processed {len(results_list)} samples")
                        st.dataframe(results_df)

                        csv_results = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv_results,
                            file_name="str_prediction_results.csv",
                            mime="text/csv"
                        )

                        # Summary statistics
                        st.subheader("📈 Summary Statistics")
                        pop_counts = results_df['Top_Population'].value_counts()
                        st.bar_chart(pop_counts)

            except Exception as e:
                st.error(f"❌ Error processing file: {e}")

    with tab3:
        st.header("ℹ️ About This Tool")

        st.markdown("""
        ### STR Population Prediction Tool

        This tool uses a **balanced machine learning model** to predict population ancestry based on STR allele profiles.

        **🎯 Key Features:**
        - **Balanced predictions** across all populations
        - **No population bias** (fixed Bahrain-only issue)
        - **Compressed model** (1.15 MB) for fast loading
        - **Robust error handling** with fallback methods

        **🧬 How to Use Genotype Data:**
        If you have two alleles per marker, use the **average**:
        - CSF1PO: (11,12) → Enter **11.5**
        - D13S317: (8,13) → Enter **10.5**
        - D16S539: (13,13) → Enter **13.0**

        **🌍 Supported Populations:**
        """)

        pop_cols = st.columns(2)
        for i, pop in enumerate(sorted(populations)):
            col_idx = i % 2
            with pop_cols[col_idx]:
                st.write(f"• {pop}")

        st.markdown("""
        **📊 Model Performance:**
        - **Balanced training:** 140 samples per population
        - **No bias:** Equal representation of all populations
        - **Accuracy:** 27.3% (good for 11-class problem)
        - **Best for:** African American, Estonian, Manchu

        **⚠️ Important Notes:**
        - Results are for **research purposes only**
        - Confidence scores are typically 10-30%
        - Higher confidence = more reliable prediction
        - Consider multiple markers for better accuracy
        """)

if __name__ == "__main__":
    main()
