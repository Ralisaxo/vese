# vese

# Semantic Website Search Engine

This project is a simple semantic search engine built with Python and Streamlit. It allows users to search through content from a website using natural language queries. The search is performed by comparing the semantic meaning of the query against pre-computed vector embeddings of the website's pages.

The core functionality relies on:
1.  **Pre-computed Embeddings:** A CSV file (`website_embeddings.csv`) containing URLs and their corresponding vector embeddings. These embeddings were generated using Google's `text-embedding-004` model.
2.  **Query Embedding:** User queries are embedded in real-time using the same `text-embedding-004` model via the Gemini API.
3.  **Similarity Search:** Cosine similarity is used to find the most relevant pages by comparing the query embedding with the pre-computed page embeddings.

##Deployed Application

*   **Streamlit Community Cloud URL:** [Add Link Here Once Deployed - e.g., https://your-app-name.streamlit.app]

## Technologies Used

*   **Python 3.x**
*   **Streamlit:** For building the interactive web interface.
*   **Pandas:** For loading and managing the data from the CSV file.
*   **NumPy:** For numerical operations, especially with embedding vectors.
*   **Scikit-learn:** For calculating cosine similarity.
*   **Google Generative AI SDK (`google-generativeai`):** For accessing the Gemini API to embed user queries with the `text-embedding-004` model.

## Project Structure

*   `app.py`: The main Streamlit application script.
*   `website_embeddings.csv`: (Or your specific CSV filename) The CSV file containing `url` and `Vector Embedding text-embedding-004` columns. **This file must be present in the root of the repository for the app to work.**
*   `requirements.txt`: A list of Python dependencies required to run the project.
*   `README.md`: This file, providing an overview of the project.
*   `.gitignore`: Specifies intentionally untracked files that Git should ignore (Python template used).
*   `LICENSE`: Contains the MIT License terms for this project.

## Setup and Running Locally

To run this application on your local machine:

1.  **Prerequisites:**
    *   Python 3.7+ installed.
    *   `pip` (Python package installer).

2.  **Clone the Repository (Optional if you already have the files):**
    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```

3.  **Create and Activate a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    # venv\Scripts\activate
    # On macOS/Linux:
    # source venv/bin/activate
    ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Set Up Gemini API Key:**
    *   You will need a Google Gemini API key.
    *   The application expects this key to be available as an environment variable named `GEMINI_API_KEY`.
    *   Set the environment variable in your terminal before running the app:
        *   **macOS/Linux:**
            ```bash
            export GEMINI_API_KEY="YOUR_API_KEY_HERE"
            ```
        *   **Windows (PowerShell):**
            ```bash
            $env:GEMINI_API_KEY="YOUR_API_KEY_HERE"
            ```
        *   **Windows (CMD):**
            ```bash
            set GEMINI_API_KEY=YOUR_API_KEY_HERE
            ```
    *   *(Note: For local runs, `app.py` will try to read this environment variable. For deployed apps on Streamlit Community Cloud, this key must be set as a "Secret".)*

6.  **Ensure Data File is Present:**
    *   Make sure your CSV file (e.g., `website_embeddings.csv`) is in the root directory of the project (same folder as `app.py`).
    *   Verify that the `CSV_FILE_PATH` and column names (`URL_COLUMN_NAME`, `EMBEDDING_COLUMN_NAME`) in `app.py` match your CSV file.

7.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```
    This will typically open the application in your default web browser.

## Deployment on Streamlit Community Cloud

This application is designed to be deployed on Streamlit Community Cloud.

1.  **Prerequisites:**
    *   A GitHub account.
    *   This project pushed to a public GitHub repository.
    *   A Streamlit Community Cloud account (can sign up with GitHub).

2.  **Deployment Steps:**
    *   Log in to [share.streamlit.io](https://share.streamlit.io/).
    *   Click "New app" and choose "Deploy from existing repo".
    *   Select your GitHub repository, branch (usually `main`), and the main Python file (`app.py`).
    *   Under "Advanced settings," go to "Secrets." Add a new secret:
        *   **Secret name:** `GEMINI_API_KEY`
        *   **Secret value:** `YOUR_ACTUAL_GEMINI_API_KEY`
    *   Click "Deploy!".

## Key Reminders for Future Self

*   The quality of search results heavily depends on the quality and relevance of the pre-computed embeddings in the `website_embeddings.csv` file.
*   The `EMBEDDING_COLUMN_NAME` in `app.py` must *exactly* match the column header in the CSV for the vector embeddings.
*   The `task_type="RETRIEVAL_QUERY"` parameter is important when embedding user queries for search tasks with Gemini models.
*   The `text-embedding-004` model produces 768-dimensional embeddings. Ensure your pre-computed embeddings are also of this dimension if from the same model.
*   If the app fails to load data, check `CSV_FILE_PATH` in `app.py` and ensure the CSV file exists in the repository and is correctly formatted.
*   If there are API errors, double-check the `GEMINI_API_KEY` (environment variable for local, Streamlit secret for deployed).

---

This README was generated with assistance from an AI Sparring Partner.
