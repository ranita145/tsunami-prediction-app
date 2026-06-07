# Tsunami Potential Prediction Deployment

This folder contains a Streamlit deployment prototype for the IS411 Data Modelling project:

**Prediction of Tsunami Potential Based on Earthquake Characteristics**

## Files

- `app.py` — Streamlit web app for single prediction and batch CSV prediction.
- `train_model.py` — trains and saves the Tuned Random Forest sklearn pipeline.
- `earthquake_data.csv` — project dataset.
- `requirements.txt` — dependencies for deployment.
- `tsunami_tuned_random_forest.joblib` — saved model pipeline, generated after running `train_model.py`.
- `model_metadata.json` — saved model metadata and evaluation metrics.

## Local run

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Streamlit Community Cloud deployment

1. Create a GitHub repository, for example `tsunami-prediction-app`.
2. Upload these files to the repository:
   - `app.py`
   - `train_model.py`
   - `earthquake_data.csv`
   - `requirements.txt`
   - optional: `tsunami_tuned_random_forest.joblib`
   - optional: `model_metadata.json`
3. Open Streamlit Community Cloud.
4. Click **Create app**.
5. Choose your GitHub repository.
6. Set the entrypoint file to `app.py`.
7. Click **Deploy**.

## Academic disclaimer

This app is for academic analysis and decision-support demonstration only.
It is **not** an official tsunami early warning system.
