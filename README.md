# aisafety-chat
A chatbot for AI Safety 

## Clone the Repo
Start by cloning the repo with the command: 

```bash
git clone https://github.com/Danselem/aisafety-chat.git
```

Next, run the command: 

```bash
cd aisafety-chat
```
Then run the folloing command:

```bash
python3.10 -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install --upgrade pip && pip install -r requirements.txt
```

## Provide API Keys
First, create your local environment by running the command below 

```bash
mv .env_copy .env
```
then entering your API key values in the `.env` file.

## Ingesting data
`ingest.py` allows accepts a pdf file, load it, chunk it, process it into embedding before storing it in a Pinecone vectorstore on the cloud. Use the command below to perform the ingest operation.

```bash
python ingest.py
```

## Querying the Vector Database
To query the Vector Database, run the command below.
```bash
streamlit run query_app.py &
```

It creates a `Streamlit UI and enables a RAG framework for querying the database

## Licence
This work is implememnyted based on [MIT License](LICENSE).