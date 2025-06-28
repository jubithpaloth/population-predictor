# STR Population Prediction Tool

## Overview
This tool predicts population ancestry from STR (Short Tandem Repeat) marker profiles using similarity-based matching algorithms. It's specifically designed for forensic genetics and population studies.

## Features
✅ **High Accuracy**: Uses similarity-based matching for better performance on small datasets  
✅ **15 STR Markers**: Supports standard forensic STR markers (CSF1PO, D13S317, etc.)  
✅ **11 Populations**: Trained on diverse global populations  
✅ **Multiple Methods**: Centroid, nearest-neighbor, statistical, and combined approaches  
✅ **Batch Processing**: Handle multiple samples via CSV import  
✅ **Input Validation**: Comprehensive error checking and user guidance  
✅ **Confidence Scores**: Provides percentage confidence for each prediction  

## Supported Populations
1. **Manchu** (9 samples)
2. **Mongolia** (10 samples)  
3. **Kyrgyz** (10 samples)
4. **Uzbek** (10 samples)
5. **North East Thai** (10 samples)
6. **Caucasian** (9 samples)
7. **African American** (9 samples)
8. **Hispanic** (8 samples)
9. **Estonian** (10 samples)
10. **Bahrain** (8 samples)
11. **Mexico** (11 samples)

## STR Markers Required
- CSF1PO, D13S317, D16S539, D18S51, D19S433
- D21S11, D2S1338, D3S1358, D5S818, D7S820  
- D8S1179, FGA, TH01, TPOX, vWA

## Quick Start

### 1. Installation
```bash
pip install pandas numpy scipy
```

### 2. Basic Usage
```python
from str_population_predictor import STRPopulationPredictor

# Initialize and load model
predictor = STRPopulationPredictor()
predictor.load_model('str_population_model.pkl')

# Your STR profile
my_profile = {
    'CSF1PO': 11, 'D13S317': 8, 'D16S539': 12,
    'D18S51': 15, 'D19S433': 14, 'D21S11': 29,
    'D2S1338': 19, 'D3S1358': 16, 'D5S818': 11,
    'D7S820': 10, 'D8S1179': 13, 'FGA': 22,
    'TH01': 7, 'TPOX': 8, 'vWA': 16
}

# Make prediction
results = predictor.predict_population(my_profile)
```

### 3. CSV Batch Processing
```python
# Process multiple samples
results_df = predictor.predict_from_csv('my_samples.csv', 'results.csv')
```

## CSV Format
Use the provided `str_prediction_template.csv` or create your own:

```csv
Pop,CSF1PO,D13S317,D16S539,D18S51,D19S433,D21S11,D2S1338,D3S1358,D5S818,D7S820,D8S1179,FGA,TH01,TPOX,vWA
Unknown,11,8,12,15,14,29,19,16,11,10,13,22,7,8,16
Unknown,10,9,11,14,13,30,20,15,12,9,14,23,8,9,17
```

## Prediction Methods
- **Combined** (recommended): Weighted combination of all methods
- **Centroid**: Distance to population average
- **Nearest Neighbor**: Distance to closest reference sample
- **Statistical**: Based on population distribution statistics

## Files Included
- `str_population_predictor.py` - Main prediction tool
- `str_population_model.pkl` - Trained model
- `str_prediction_template.csv` - Input template
- `README.md` - This documentation

## Example Output
```
🎯 Population Prediction Results (combined method)
--------------------------------------------------
   1. North East Thai    10.2% 🟡 Medium (10 ref samples)
   2. Manchu            10.0% 🟡 Medium (9 ref samples)
   3. Kyrgyz             9.8% 🔴 Low    (10 ref samples)
   4. Hispanic           9.6% 🔴 Low    (8 ref samples)
   5. Mexico             9.5% 🔴 Low    (11 ref samples)
```

## Troubleshooting
- **"Model file not found"**: Ensure `str_population_model.pkl` is in the same directory
- **"Missing STR markers"**: Check that all 15 required markers are provided
- **"Invalid value"**: STR values must be numeric and typically range 6-35
- **Low confidence**: May indicate admixed ancestry or insufficient reference data

## Technical Details
- **Algorithm**: Similarity-based matching with multiple distance metrics
- **Training Data**: 104 samples across 11 populations
- **Validation**: Cross-population distance analysis
- **Performance**: Optimized for small reference datasets

## Citation
If you use this tool in research, please cite:
"STR Population Prediction Tool - AI-powered ancestry inference from forensic STR markers"

## Support
For questions or issues, refer to the documentation or contact the development team.
