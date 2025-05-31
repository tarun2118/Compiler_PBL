# CPU Scheduling Algorithm Comparison

This project compares traditional CPU scheduling algorithms with real-time scheduling algorithms using machine learning models. It generates synthetic datasets, trains multiple ML models, and provides a Streamlit interface for visualization and prediction.

## Algorithms Compared

### Traditional Algorithms
- First Come First Serve (FCFS)
- Shortest Job First (SJF)
- Round Robin (RR)
- Priority Scheduling

### Real-time Algorithms
- Clock-Driven
- Earliest Deadline First (EDF)
- Weighted Round Robin
- Priority-Driven
- Least Slack Time First (LSTF)
- Rate Monotonic

## Machine Learning Models Used
- Random Forest
- Support Vector Machine (SVM)
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Naive Bayes

## Project Structure
- `data_generator.py`: Generates synthetic datasets for both traditional and real-time algorithms
- `model_trainer.py`: Trains and evaluates different ML models
- `train_models.py`: Main script to generate data and train models
- `app.py`: Streamlit web application for visualization and prediction
- `models/`: Directory containing saved ML models
- `*_dataset.csv`: Generated datasets

## Setup and Usage

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Generate datasets and train models:
```bash
python train_models.py
```

3. Run the Streamlit application:
```bash
streamlit run app.py
```

## Features
- Generate synthetic datasets for both traditional and real-time algorithms
- Train and evaluate 6 different ML models
- Compare model accuracies between traditional and real-time algorithms
- Interactive web interface for:
  - Input parameter adjustment
  - Algorithm prediction
  - Dataset visualization
  - Feature correlation analysis

## Note
The synthetic data generation is designed to slightly favor real-time algorithms to demonstrate their theoretical advantages in certain scenarios. This bias is implemented through controlled randomization in the waiting time calculations. 