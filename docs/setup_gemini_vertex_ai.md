## Using Gemini via Vertex AI (already set up on this machine)

Auth is already done — ADC credentials exist at
`~/.config/gcloud/application_default_credentials.json`, tied to a
collaborator's Google identity on GCP project
`scilifelab-hpa-proj-1`. No login, no link, no code needed. Just call the API.

### Requirements
- `pip install google-genai` (already installed)

### Client setup
```python
from google import genai
from google.genai import types

client = genai.Client(vertexai=True, project="scilifelab-hpa-proj-1", location="global")
# location MUST be "global" — regional locations (e.g. "us-central1") 404 on these models
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