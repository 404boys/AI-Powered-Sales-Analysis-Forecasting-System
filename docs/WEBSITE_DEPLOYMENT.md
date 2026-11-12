# Run and deploy the Streamlit website

## Run locally

1. Train the model from the repository root:

   ```powershell
   python src/modeling/train_lightgbm.py --n-estimators 300
   ```

2. Start the app:

   ```powershell
   streamlit run app/app.py
   ```

3. Open the local URL printed by Streamlit, usually http://localhost:8501.

The model file is generated locally under `artifacts/model/`. It is intentionally excluded from Git because it is a generated binary. The cleaned CSV is also required by the app.

## Optional public deployment with Streamlit Community Cloud

The app can be deployed from the GitHub repository, but deployment requires a trained model artifact and the cleaned dataset to be available to the hosted app. Because generated model binaries and the dataset may be large or governed by dataset redistribution terms, do not commit them blindly.

Recommended deployment path:

1. Confirm that you are allowed to redistribute the cleaned dataset under its dataset licence/terms.
2. Decide how the model artifact and dataset will be supplied to the hosting environment (for a student demo, a private storage location or an approved small demo dataset may be more appropriate).
3. In Streamlit Community Cloud, create a new app and select this repository, the `main` branch, and `app/app.py` as the entry point.
4. Configure any required secrets or data-download process using the host's secrets facility. Never commit passwords, access tokens, or private URLs to GitHub.
5. Deploy and test all five pages after the app starts.

A GitHub repository commit alone does not publish a running website. The hosting setup and model/data availability must also be completed.
