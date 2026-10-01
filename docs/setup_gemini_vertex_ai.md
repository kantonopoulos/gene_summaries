## Using Gemini via Vertex AI

All model calls go through Vertex AI in GCP project `scilifelab-hpa-proj-1` and are billed
to that project. Access is granted per Google account by the project owner.

### 1. Get access
Ask the project owner to grant your Google account these roles on `scilifelab-hpa-proj-1`:

| Role | Needed for |
|---|---|
| Vertex AI User (`roles/aiplatform.user`) | All model calls, local and batch |
| BigQuery User (`roles/bigquery.user`) | Batch runs (input/output tables in BigQuery) |

### 2. Log in (once per machine)
Install the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install), then:
```bash
gcloud auth application-default login
```
This stores Application Default Credentials (ADC) in
`~/.config/gcloud/application_default_credentials.json`. The scripts pick them up
automatically. **Never copy this file into the repo or share it**; it grants access to your
account.

### 3. Install the client
```bash
pip install google-genai
```

### Client setup
```python
from google import genai
from google.genai import types

client = genai.Client(vertexai=True, project="scilifelab-hpa-proj-1", location="global")
# location MUST be "global"; regional locations (e.g. "us-central1") 404 on these models
```

### Text generation (Gemini Flash 3.8)
```python
resp = client.models.generate_content(model="gemini-3.8-flash", contents="Hello")
print(resp.text)
```

### Image generation (Nano Banana Pro)
```python
resp = client.models.generate_content(
    model="gemini-3-pro-image",
    contents="Generate an image of a banana.",
    config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"]),
)
for part in resp.candidates[0].content.parts:
    if part.inline_data:
        with open("out.png", "wb") as f:
            f.write(part.inline_data.data)
```
