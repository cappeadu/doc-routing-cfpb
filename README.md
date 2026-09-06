# CFPB Complaint Routing API

A text classification system for automatically routing Consumer Financial Protection Bureau (CFPB) consumer complaints to the appropriate financial product or service category.

The project compares three approaches to text classification and exposes them through a FastAPI application:

* **TF-IDF + classical machine learning**
* **Sentence embeddings + classical machine learning**
* **Fine-tuned BERT**

The application is containerized with Docker and provides a single API through which the user can select the classification model and submit a complaint narrative.

---

## Project Overview

Financial institutions receive large volumes of consumer complaints that need to be directed to the appropriate team or department. Manually reviewing and routing every complaint can be time-consuming and inconsistent.

This project treats complaint routing as a **multi-class text classification problem**.

Given a complaint narrative such as:

> "I do not recognize a charge on my credit card."

the system predicts the most appropriate complaint category.

The project uses CFPB complaint narratives and compares increasingly sophisticated approaches to determine how much improvement is gained from moving from traditional NLP methods to semantic embeddings and transformer-based models.

---

## Models

The API supports three models.

### 1. TF-IDF + Classical Machine Learning

The complaint text is converted into TF-IDF features and passed to a classical machine-learning classifier.

This provides a strong and relatively lightweight baseline and makes the model's behaviour easier to inspect.

**Pipeline:**

```text
Complaint
    ↓
Text preprocessing
    ↓
TF-IDF
    ↓
Classifier
    ↓
Predicted category
```

### 2. Sentence Embeddings + Classical Machine Learning

Instead of representing text using individual word/ngram frequencies, complaint narratives are converted into dense semantic embeddings using a Sentence Transformer.

The resulting embeddings are then passed to a classical classifier.

```text
Complaint
    ↓
Text preprocessing
    ↓
Sentence Transformer
    ↓
Dense embedding
    ↓
Classifier
    ↓
Predicted category
```

This approach tests whether semantic representations provide an advantage over sparse TF-IDF features.

### 3. Fine-tuned BERT

A pretrained transformer model is fine-tuned specifically for the complaint classification task.

```text
Complaint
    ↓
Tokenizer
    ↓
Fine-tuned BERT
    ↓
Class probabilities
    ↓
Predicted category
```

The fine-tuned model is hosted on Hugging Face and loaded by the API at inference time.

---

## Dataset

The project uses the **Consumer Financial Protection Bureau Consumer Complaint Database**.

The dataset contains consumer complaints submitted to the CFPB covering a range of financial products and services.

The original product/service categories include:

* Debt collection
* Checking or savings account
* Credit card
* Money transfer, virtual currency, or money service
* Mortgage
* Vehicle loan or lease
* Payday loan, title loan, personal loan, or advance loan
* Credit reporting or other personal consumer reports
* Student loan
* Prepaid card
* Debt or credit management

For the routing task, the original categories are consolidated into a smaller set of operationally meaningful classes where appropriate. This reduces extremely small classes and produces a more practical classification problem. The following are the consolidated classes and mappings:

* Collections & Recovery -> [Debt collection, Debt or credit management]
* Banking Operations -> [Checking or savings account]
* Cards & Payments -> [Credit card, Prepaid card]
* Consumer Lending -> [Vehicle loan or lease, Payday loan, title loan, personal loan, or advance loan, Student loan]
* Money Transfer and Payments -> [Money transfer, virtual currency, or money service]
* Mortgage & Home Lending -> Mortgage
* Credit Reporting & Disputes -> Credit reporting or other personal consumer reports


### Obtaining the data

The raw dataset is **not included in the Git repository**.

The project notebook contains the data acquisition (download_datset.ipynb) and preparation steps and data splits (explore_data.ipynb) required to obtain the dataset and reproduce the training workflow. Embeddings data (.npy) can be obtained by running get_embeddings.ipynb locally or on Google COLAB and saving the numpy object.

This keeps the repository lightweight while allowing the dataset to be retrieved directly from its source.

---

## Data Preprocessing

The complaint narratives contain CFPB-specific anonymisation markers such as:

```text
XXXX
XX/XX/year
XXXX XXXX XXXX
```

These markers represent scrubbed personal information.

Because the underlying information is intentionally anonymised, it is generally not possible to determine whether a particular `XXXX` represents an account number, name, date, address, or another piece of personal information.

The preprocessing therefore treats these anonymised sequences as a generic `[MASK]` token rather than attempting to infer the original information.

Other preprocessing includes operations such as:

* lowercasing
* lemmatisation where applicable
* removal of stop words
* removal of punctuation
* removal of numeric/currency information where appropriate

The exact preprocessing differs depending on the modelling approach.

---

## Project Structure

```text
complaint-project/
│
├── datasets/
│
├── deploy/
│   └── serve.py
│
├── models/
│   ├── artifacts.joblib
│   └── embeddings_model.joblib
│
├── notebooks/
│   ├── download_dataset.ipynb
│   ├── explore_data.ipynb
│   ├── fine_tune_bert_COLAB.ipynb
│   ├── get_embeddings.ipynb
│   ├── training_embeddings.ipynb
│   └── training_tfid.ipynb
│
├── src/
│   ├── config.py
│   ├── data.py
│   ├── evaluate.py
│   ├── models.py
│   ├── predict.py
│   ├── train.py
│   └── utils.py
│
├── src_bert/
│   └── predict.py
│
├── src_embeddings/
│   ├── data.py
│   ├── models.py
│   ├── predict.py
│   └── train.py
│
├── .dockerignore
├── .gitignore
├── compose.yaml
├── Dockerfile
├── README.md
├── requirements-dev.txt
└── requirements.txt
```

The exact structure may evolve as the project develops.

---

### **Experiment Tracking**

MLFLOW is used to track experiments for TF-IDF and Embeddings model. Spin up MLFLOW before training using the command below:

```pwsh
mlflow server `
--backend-store-uri sqlite:///mlflow.db `
--default-artifact-root ./mlruns `
--host 127.0.0.1 `
--port 5000
```

If you are running this locally then head to http://127.0.0.1:5000/ to view experiments.


---

### **Training**

Example of how to run training scripts: 
* TDIF.

```pwsh
$train_data_loc_tdif="C:\Users\Lenovo\code\complaint-project\datasets\train.csv"
$val_data_loc_tdif="C:\Users\Lenovo\code\complaint-project\datasets\val.csv"
$model_params='{"random_state":42, "max_iter":1000}'
$experiment_name="doc_routing"
$directory_name="models"
$vectorizer_params='{"stop_words":"english", "max_features":5000, "ngram_range":[1,3]}'

python -m src.train `                                   
>> --train-data-loc $train_data_loc_tdif `
>> --val-data-loc $val_data_loc_tdif `
>> --model-params $model_params `
>> --experiment-name $experiment_name `
>> --directory-to-save-model $directory_name `
>> --vectorizer-params $vectorizer_params
```

* EMBEDDINGS.

```pwsh
$train_data_loc="C:\Users\Lenovo\code\complaint-project\datasets\train_embeddings.npy"
$val_data_loc="C:\Users\Lenovo\code\complaint-project\datasets\val_embeddings.npy"
$model_params='{"random_state":42, "max_iter":1000}'
$experiment_name="doc_routing"
$directory_name="models"

python -m src_embeddings.train `                        
>> --train-data-loc $train_data_loc `
>> --val-data-loc $val_data_loc `
>> --model-params $model_params `
>> --experiment-name $experiment_name `
>> --directory-to-save-model $directory_name
```

---

### **Inference from CLI**

Example of how to run Inference from command-line: 

<details>
<summary><strong>1. TF-IDF + Classical ML</strong></summary>


```pwsh
python -m src.predict `                                 
>> --text "I do not recognize a charge on my credit card."
```

```json
{
  "results": [
    {
      "prediction": "Cards & Payments",
      "probabilities": {
        "Cards & Payments": 0.981,
        "Collections & Recovery": 0.0134,
        "Credit Reporting & Disputes": 0.0024,
        "Banking Operations": 0.0013,
        "Consumer Lending": 0.0009,
        "Money Transfer and Payments": 0.0005,
        "Mortgage & Home Lending": 0.0005
      },
      "latency": "15.47 ms",
      "decision": "auto_route"
    }
  ],
  "model": "tfidf"
}
```
</details>

<details>
<summary><strong>2. EMBEDDINGS + Classical ML</strong></summary>


```pwsh
python -m src_embeddings.predict `                                 
>> --text "I do not recognize a charge on my credit card."
```

```json
{
  "results": [
    {
      "prediction": "Cards & Payments",
      "probabilities": {
        "Cards & Payments": 0.9567,
        "Collections & Recovery": 0.0275,
        "Banking Operations": 0.0072,
        "Credit Reporting & Disputes": 0.0041,
        "Consumer Lending": 0.0037,
        "Money Transfer and Payments": 0.0008,
        "Mortgage & Home Lending": 0
      },
      "latency": "31.26 ms",
      "decision": "auto_route"
    }
  ],
  "model": "embeddings"
}
```
</details>


<details>
<summary><strong>3. BERT</strong></summary>


```pwsh
python -m src_bert.predict `                                 
>> --text "I do not recognize a charge on my credit card."
```

```json
{
  "results": {
    "prediction": "Cards & Payments",
    "probabilities": {
      "Cards & Payments": 0.9711,
      "Collections & Recovery": 0.0236,
      "Banking Operations": 0.003,
      "Credit Reporting & Disputes": 0.0012,
      "Consumer Lending": 0.0006,
      "Money Transfer and Payments": 0.0003,
      "Mortgage & Home Lending": 0.0002
    },
    "latency": "226.51 ms",
    "decision": "auto_route"
  },
  "model": "bert"
}
```
</details>

---
### **Serving Locally**

```pwsh
uvicorn deploy.serve:app
```

Once the app is running you can use it via Python. eg:
```python
import requests
text = "I do not recognize a charge on my credit card."
params = {"model":"tfidf"} # select different model eg "bert" or "embeddings"
response = requests.post("http://127.0.0.1:8000/predict", params=params, json={"text":text}).json()
```

---
## API

The application is built with **FastAPI**.

The API accepts a complaint narrative and allows the client to select the model used for prediction.

### Example request

```json
{
    "text": "I do not recognize a charge on my credit card.",
}
```

The available model names depend on the model-selection configuration in the application.

### Example response

```json
{
  "results": [
    {
      "prediction": "Cards & Payments",
      "probabilities": {
        "Cards & Payments": 0.981,
        "Collections & Recovery": 0.0134,
        "Credit Reporting & Disputes": 0.0024,
        "Banking Operations": 0.0013,
        "Consumer Lending": 0.0009,
        "Money Transfer and Payments": 0.0005,
        "Mortgage & Home Lending": 0.0005
      },
      "latency": "13.64 ms",
      "decision": "auto_route"
    }
  ],
  "model": "tfidf"
}
```

### Decision-Making and Automated Routing

The system goes beyond classification by using model confidence to support routing decisions.

For each complaint, the API returns:

* The predicted business category
* The probability assigned to each category
* Prediction latency
* A routing decision based on a configurable confidence threshold

The `decision` field indicates whether the prediction meets the configured confidence threshold for automatic routing. Predictions that do not meet the threshold can instead be flagged for further review rather than being routed automatically.

This allows the system to operate as a **decision-making and triage layer**, rather than simply returning a predicted class.

The API also provides the standard FastAPI interactive documentation.

When running locally, it is available at:

```text
http://localhost:8000/docs
```


---

## Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/cappeadu/doc-routing-cfpb.git
cd complaint-project
```

### 2. Create the Python environment

Create a virtual environment or Conda environment and install the required dependencies.

```bash
pip install -r requirements-dev.txt
```

The runtime dependencies include the libraries required for the three inference approaches.

---

## Running with Docker

The application can be run using Docker Compose.

Build the image:

```bash
docker compose -f compose.yaml build
```

Start the API:

```bash
docker compose -f compose.yaml up
```

The API will then be available at:

```text
http://localhost:8000
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

---

## Hugging Face Model

The fine-tuned BERT model is hosted separately on Hugging Face rather than being stored directly inside the Docker image.

The application retrieves the model when the BERT inference path is used.

> Hugging Face model: **appcle/distilbert-base-uncased-cfpd**

---

## Environment Variables

Secrets and credentials should not be committed to GitHub.

For example:

```text
HF_TOKEN=<your-token>
```

When deploying, the token should be configured through the hosting platform's environment-variable/secrets mechanism.

Also, a .env should be created at root folder with the following MLFLOW variable. Example of the URI (localhost):

```text
ML_FLOW_TRACKING_URI=http://127.0.0.1:5000
```
---

## Model Evaluation

The models are evaluated as a multi-class classification problem.

Accuracy is reported, but it is not treated as the only measure of performance because the complaint categories are not perfectly balanced.

The main evaluation metrics include:

* **Precision**
* **Recall**
* **Macro F1**
* **Weighted F1**
* **Confusion matrix**

Macro F1 is particularly useful for assessing whether the model performs reasonably across smaller as well as larger complaint categories.

The model comparison is therefore based on both overall performance and class-level behaviour.

### **Evaluation using TF-IDF**

```pwsh
$dataset_loc="C:\Users\Lenovo\code\complaint-project\datasets\val.csv"
$checkpoint='C:\Users\Lenovo\code\complaint-project\models\artifacts.joblib'

python -m src.evaluate `                                
>> --dataset-loc $dataset_loc `
>> --checkpoint $checkpoint
```

```json
{
  "time stamp": "September 05, 2026 09:22:41 PM",
  "total time": "2.75 mins",
  "overall": {
    "accuracy": 0.8346478873239437,
    "precision_macro": 0.8336530848047446,
    "recall_macro": 0.7883438349834414,
    "f1_macro": 0.8081514744643693,
    "f1_weighted": 0.8336757538457182,
    "num_samples": 3550.0
  }
...
```

---

## Why Three Models?

The purpose of the project is not simply to train a single classifier.

The three approaches represent different levels of text representation:

```text
TF-IDF
  ↓
Sparse lexical representation

Sentence Embeddings
  ↓
Dense semantic representation

BERT
  ↓
Contextual transformer representation
```

This makes it possible to compare:

1. A traditional NLP baseline
2. Semantic embeddings combined with classical machine learning
3. End-to-end transformer fine-tuning

The comparison provides a practical view of the trade-offs between model complexity, semantic representation, inference requirements, and predictive performance.

---

## Deployment Architecture

```text
                    Client
                      │
                      ▼
                FastAPI API
                      │
              ┌───────┴────────┐
              │ Model Selection│
              └───────┬────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
     TF-IDF       Embeddings       BERT
        │             │             │
   Classifier     Classifier    Hugging Face
        │             │             │
        └─────────────┼─────────────┘
                      ▼
                 Prediction
```

The application is packaged as a Docker container and can be deployed to a cloud hosting platform.

---

## Limitations

### Anonymised text

The CFPB dataset contains extensive anonymisation. Personal information is represented by markers such as `XXXX`. This removes potentially useful information from the original complaint narratives and introduces a limitation to the classification task.

The project therefore treats these markers as anonymised information rather than attempting to reconstruct what they represent.

### Class imbalance

Some complaint categories contain substantially more examples than others. A model can therefore achieve reasonable overall performance while performing poorly on smaller classes.

For this reason, class-level precision, recall and F1-score are considered alongside aggregate metrics.

### Dataset changes

The CFPB Consumer Complaint Database is continuously updated. Reproducing the exact training dataset may therefore require using the same dataset snapshot or download date used during the original experiment.

### Model trade-offs

The more sophisticated models require greater computational resources and may have higher inference latency than the TF-IDF baseline.

---

## Future Improvements

Potential extensions include:

* automated testing of the API and prediction pipelines
* monitoring prediction latency and model performance
* improved confidence/threshold handling
* model versioning
* experiment tracking
* automated retraining pipelines
* additional complaint categories or alternative routing taxonomies
* evaluation using newly collected complaints
* production monitoring and feedback loops

---

## Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* spaCy
* PyTorch
* Transformers
* Sentence Transformers
* FastAPI
* Docker
* Docker Compose
* Hugging Face

---

## Project Status

The application has been containerized and tested locally with all three classification approaches.

The next stage is cloud deployment and further production-oriented improvements such as automated testing, monitoring, and model/version management.
