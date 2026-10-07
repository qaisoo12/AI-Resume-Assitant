# AI-Resume-Assitant
# 📄 AI Resume ATS Checker

A Streamlit app that scores a resume for ATS (Applicant Tracking System) compatibility and gives specific, prioritised suggestions to improve it. It uses Google's **Gemini Flash** model for the analysis.

## Features

- Upload a resume as **PDF, DOCX or TXT**
- **ATS score (0-100)** with a breakdown: formatting, keywords, experience impact, skills, readability
- Prioritised improvements (High / Medium / Low), missing keywords and example bullet rewrites
- Optional **job description** input for keyword matching and a job-match score
- Instant local checks (contact info, standard sections, bullets, metrics, length)
- Download the full report as Markdown

## Project structure

```
.
├── app.py             # Streamlit app
├── requirements.txt   # Python dependencies
└── README.md
```

## Run locally

1. Install Python 3.10 or newer.
2. Get a free Gemini API key at https://aistudio.google.com/apikey
3. Install and run:

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

4. Paste your API key in the sidebar, or set it so you don't have to retype it:

```bash
export GEMINI_API_KEY="your-key"        # Windows PowerShell: $env:GEMINI_API_KEY="your-key"
```

You can also create `.streamlit/secrets.toml` (never commit this file):

```toml
GEMINI_API_KEY = "your-key"
# GEMINI_MODEL = "gemini-flash-latest"   # optional
```

## Configuration

| Setting | Where | Default |
|---|---|---|
| `GEMINI_API_KEY` | secrets / env var / sidebar | none (required) |
| `GEMINI_MODEL` | secrets / env var / sidebar | `gemini-flash-latest` |

`gemini-flash-latest` is an alias for the newest Flash model. To pin a specific version, set `GEMINI_MODEL` (for example `gemini-2.5-flash`).

## Deploy on Streamlit Community Cloud

1. Push this project to a GitHub repository.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, choose your repository, branch `main` and main file `app.py`.
4. Open **Advanced settings → Secrets** and add:

```toml
GEMINI_API_KEY = "your-key"
```

5. Click **Deploy**.

## Troubleshooting

- **"Almost no text could be extracted"**: the PDF is probably a scanned image. Export a text-based PDF or use DOCX.
- **API key rejected**: check the key at https://aistudio.google.com/apikey.
- **Model not found**: set a valid model name in the sidebar or `GEMINI_MODEL`.
- **Rate limit reached**: the free tier has per-minute limits; wait and retry.

## Privacy

The extracted resume text is sent to the Gemini API for analysis and is not stored by this app.

## Disclaimer

The score is an AI-generated estimate, not the output of a real ATS. Use it as guidance, not a guarantee.
