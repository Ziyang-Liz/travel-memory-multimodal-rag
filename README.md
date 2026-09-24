## Installation

First, open a terminal in the project root folder:

Install the required Python packages:

pip install -r requirements.txt

The required dependencies are listed in requirements.txt.

The system can run with or without an OpenAI API key. If you want to use the API-based answer generator, create a .env file in the project root folder and add:

OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini

If no API key is provided, the system will still run by using the local fallback answer generator.

## How to Run
1. Build the retrieval index

Before running the chatbot, build the local retrieval index:

python scripts/build_index.py

This command creates or updates the following files:

storage/documents.jsonl
storage/tfidf_vector_store.joblib

Run this command again if you edit the travel notes or metadata files.

2. Run the Streamlit chatbot

Start the web chatbot with:

streamlit run app.py

After running the command, a local URL will appear in the terminal, usually:

http://localhost:8501

Open this URL in a browser to use the chatbot.

Example questions:

What did I eat at Sierra Café in Auckland?
Find the memory where the image shows a small Hobbit house.
Find an image which includes toast.
Compare Auckland and Queenstown. Which city gave me more landscape memories?
3. Run the command-line chatbot

A command-line version is also available:

python scripts/run_cli.py
4. Run the evaluation

To evaluate the final multimodal agent:

python scripts/run_eval.py

To evaluate the text-only RAG ablation:

python scripts/run_eval_text_only.py

To evaluate the plain LLM baseline:

python scripts/run_eval_plain_llm.py

The evaluation outputs will be saved in:

data/evaluation/