# Multilingual App Reviews Analytics

A comprehensive analytics project for analyzing multilingual mobile app reviews across different languages and platforms.

## Project Goals

- Analyze sentiment and patterns in multilingual mobile app reviews
- Extract insights from user feedback across 40+ apps in 24 languages
- Provide data-driven recommendations for app improvement
- Demonstrate multilingual text analysis techniques

## Dataset

This project uses the **"Multilingual Mobile App Review Dataset August 2025"** by Pratyush Puri from Kaggle. The dataset contains 2,514 mobile app reviews across 40+ apps in 24 languages, stored in CSV format. It includes messy data types and missing values, making it perfect for data cleaning practice.

### Dataset Setup

**Important:** Place the dataset 'multilingual_mobile_app_reviews_2025.csv' (downloaded from Kaggle) into `data/raw/` before running any scripts.

The dataset file should be located at:
```
data/raw/multilingual_mobile_app_reviews_2025.csv
```

## Project Structure

```
├── data/
│   ├── raw/                    # Original dataset (not tracked by git)
│   └── processed/             # Cleaned and processed data (not tracked by git)
├── notebooks/
│   └── 01_eda.ipynb          # Exploratory data analysis
├── src/
│   ├── data_prep.py          # Data cleaning and preprocessing
│   └── analytics.py          # Analytics and visualization
├── reports/
│   └── figures/              # Generated plots and charts (not tracked by git)
├── .github/
│   └── workflows/            # CI/CD workflows
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Setup

1. Clone this repository:
```bash
git clone https://github.com/AbdulSohail018/Multilingual-App-Reviews-Analytics.git
cd Multilingual-App-Reviews-Analytics
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the dataset from Kaggle and place it in `data/raw/multilingual_mobile_app_reviews_2025.csv`

## Usage

### Data Preprocessing
Clean and prepare the dataset:
```bash
python src/data_prep.py
```

### Analytics
Generate insights and visualizations:
```bash
python src/analytics.py
```

### Exploratory Data Analysis
Open and run the Jupyter notebook:
```bash
jupyter notebook notebooks/01_eda.ipynb
```

## Features

- **Data Cleaning**: Handle missing values, normalize data types, and deduplicate records
- **Language Detection**: Automatic language identification for reviews
- **Sentiment Analysis**: Extract sentiment patterns across different languages
- **Visualization**: Generate insightful charts and graphs
- **Multi-language Support**: Process reviews in 24+ languages

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.