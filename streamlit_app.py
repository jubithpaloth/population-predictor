
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import io
import base64
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="STR Population Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_model():
    """Load the trained model and data"""
    try:
        with open('advanced_str_population_model.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def predict_profile(profile_data, model_data):
    """Make prediction for a single profile"""
    # Prepare input data
    str_markers = model_data['str_markers']
    input_vector = []

    for marker in str_markers:
        if marker in profile_data:
            input_vector.append(profile_data[marker])
        else:
            # Use population mean for missing markers
            pop_means = model_data['population_centroids']
            overall_mean = np.mean([pop_means[pop][marker] for pop in pop_means.keys() 
                                  if marker in pop_means[pop]])
            input_vector.append(overall_mean)

    input_array = np.array(input_vector).reshape(1, -1)

    # Get predictions from ensemble
    ensemble_model = model_data['ensemble_model']
    prediction = ensemble_model.predict(input_array)[0]
    probabilities = ensemble_model.predict_proba(input_array)[0]

    # Get class labels
    classes = ensemble_model.classes_

    # Create results with top predictions
    results = []
    prob_pairs = list(zip(classes, probabilities))
    prob_pairs.sort(key=lambda x: x[1], reverse=True)

    for i, (pop, prob) in enumerate(prob_pairs[:5]):
        results.append({
            'rank': i + 1,
            'population': pop,
            'confidence': prob * 100,
            'method': 'Ensemble Model'
        })

    return results

def create_download_link(df, filename):
    """Create a download link for dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

def main():
    # Title and description
    st.title("🧬 STR Population Prediction Tool")
    st.markdown("""
    This tool predicts population ancestry based on STR (Short Tandem Repeat) allele profiles.
    Upload your data or enter values manually to get population predictions with confidence scores.
    """)

    # Load model
    model_data = load_model()
    if not model_data:
        st.error("Failed to load the prediction model. Please contact the administrator.")
        return

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    mode = st.sidebar.selectbox(
        "Choose input method:",
        ["Manual Entry", "CSV Upload", "Batch Processing", "About"]
    )

    if mode == "Manual Entry":
        st.header("Manual STR Profile Entry")

        # Create columns for STR markers
        str_markers = model_data['str_markers']
        cols = st.columns(3)

        profile_data = {}

        # Create input fields for each STR marker
        for i, marker in enumerate(str_markers):
            col_idx = i % 3
            with cols[col_idx]:
                value = st.number_input(
                    f"{marker}:",
                    min_value=0.0,
                    max_value=50.0,
                    value=0.0,
                    step=0.1,
                    key=f"manual_{marker}"
                )
                if value > 0:
                    profile_data[marker] = value

        # Predict button
        if st.button("🔍 Predict Population", type="primary"):
            if profile_data:
                with st.spinner("Analyzing STR profile..."):
                    results = predict_profile(profile_data, model_data)

                st.success("Prediction completed!")

                # Display results
                st.subheader("Prediction Results")

                # Create results dataframe
                results_df = pd.DataFrame(results)

                # Display as styled table
                st.dataframe(
                    results_df.style.format({'confidence': '{:.2f}%'}),
                    use_container_width=True
                )

                # Visualization
                st.subheader("Confidence Visualization")
                chart_data = pd.DataFrame({
                    'Population': [r['population'] for r in results],
                    'Confidence': [r['confidence'] for r in results]
                })
                st.bar_chart(chart_data.set_index('Population'))

            else:
                st.warning("Please enter at least one STR marker value.")

    elif mode == "CSV Upload":
        st.header("CSV File Upload")

        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file with STR marker columns"
        )

        if uploaded_file is not None:
            try:
                # Read the CSV file
                df = pd.read_csv(uploaded_file)

                st.subheader("Data Preview")
                st.dataframe(df.head())

                # Check for STR markers in columns
                str_markers = model_data['str_markers']
                available_markers = [col for col in df.columns if col in str_markers]

                st.info(f"Found {len(available_markers)} STR markers in your data: {', '.join(available_markers)}")

                if st.button("🔍 Analyze Data", type="primary"):
                    if available_markers:
                        with st.spinner("Processing data..."):
                            all_results = []

                            for idx, row in df.iterrows():
                                profile_data = {}
                                for marker in available_markers:
                                    if pd.notna(row[marker]):
                                        profile_data[marker] = float(row[marker])

                                if profile_data:
                                    results = predict_profile(profile_data, model_data)
                                    all_results.append({
                                        'Sample_ID': f"Sample_{idx+1}",
                                        'Top_Population': results[0]['population'],
                                        'Confidence': results[0]['confidence'],
                                        'Second_Population': results[1]['population'] if len(results) > 1 else '',
                                        'Second_Confidence': results[1]['confidence'] if len(results) > 1 else 0
                                    })

                        # Display results
                        st.subheader("Batch Analysis Results")
                        results_df = pd.DataFrame(all_results)
                        st.dataframe(results_df, use_container_width=True)

                        # Download link
                        st.markdown(
                            create_download_link(results_df, "str_predictions.csv"),
                            unsafe_allow_html=True
                        )
                    else:
                        st.error("No STR markers found in the uploaded file.")

            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

    elif mode == "Batch Processing":
        st.header("Batch Processing Template")

        # Create template
        str_markers = model_data['str_markers']
        template_data = {marker: [0.0] for marker in str_markers}
        template_df = pd.DataFrame(template_data)

        st.subheader("Download Template")
        st.markdown(
            create_download_link(template_df, "str_template.csv"),
            unsafe_allow_html=True
        )

        st.subheader("Template Preview")
        st.dataframe(template_df, use_container_width=True)

        st.markdown("""
        ### Instructions:
        1. Download the template CSV file
        2. Fill in your STR marker values
        3. Upload the completed file using the "CSV Upload" tab
        4. Get predictions for all samples at once
        """)

    else:  # About
        st.header("About This Tool")

        st.markdown("""
        ### STR Population Prediction Tool

        This tool uses machine learning to predict population ancestry based on STR (Short Tandem Repeat) allele profiles.

        **Features:**
        - Manual entry for single samples
        - Batch processing via CSV upload
        - Confidence scoring for predictions
        - Top 5 population matches
        - Downloadable results

        **STR Markers Supported:**
        """)

        # Display supported markers
        str_markers = model_data['str_markers']
        marker_cols = st.columns(3)
        for i, marker in enumerate(str_markers):
            col_idx = i % 3
            with marker_cols[col_idx]:
                st.write(f"• {marker}")

        st.markdown("""
        **Model Information:**
        - Ensemble model combining multiple algorithms
        - Trained on population genetic data
        - Provides confidence scores for reliability assessment

        **Usage Guidelines:**
        - Enter allele values as decimal numbers
        - Missing markers are handled automatically
        - Results show top 5 most likely populations
        - Confidence scores indicate prediction reliability
        """)

if __name__ == "__main__":
    main()
