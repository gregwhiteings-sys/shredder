# TyreCycle — tyre-piece photo review

Mobile-responsive web app for iPhone Safari and Android Chrome. Capture or upload a photo, tap two ruler marks, enter their real distance in cm, run the **published** Roboflow workflow, review masks/count, then export human-reviewed CSV/PDF reports.

## Setup

Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
# Set ROBOFLOW_API_KEY in your server environment; never expose it in browser code.
export ROBOFLOW_API_KEY='your_private_key'
python app.py
```

Open http://127.0.0.1:8000. To use the camera on another device, deploy behind HTTPS with authentication, upload limits, and private storage. `capture=environment` is a mobile browser hint; users can also choose from the gallery.

Workflow: `gregwhiteings-gmail-com/tyre-piece-sam-3-label-sample-1794525875800` on `https://serverless.roboflow.com`, using model `gregwhiteings-gmail-com/recycling-yard-tyre-counting-1-rfdetr-seg-small-t1`. Inputs: `image`, `reference_distance_px`, `reference_distance_cm`. Outputs: `output_image`, `predictions`, `count`, `measurements`. Server uses `inference-sdk` and header-based API key transport. Raw predictions are not sent to the browser or persisted. The annotated image is saved separately.

Calibration is optional; both pixel and cm distances must be positive to enable cm units. Tap two marks on a ruler in the same plane as the tyre pieces. Review every mask. The workflow estimates **visible projected area**, not volume, weight or a perspective-correct physical area. The count is **unverified until a person confirms it**. Model weights are **not** included.

## Important deployment limitations

This is a starter application, **not a public multi-user deployment**. Reviews are stored in process memory, and uploaded photos/overlays are kept on local disk. Restarting the server loses review metadata. There is no login, access control, retention/deletion UI, or durable audit trail. Run behind authentication before exposing to the internet. Use a persistent database, private object storage, HTTPS, request rate limits and per-user authorization before production use. Use a single worker until in-memory storage is replaced.

## Tests

`python -m unittest discover -s tests -v` checks review/export rules using a mocked workflow response. Live smoke test (requires an API key and representative image): `ROBOFLOW_SAMPLE_IMAGE=/path/to/photo.jpg python -m unittest tests.test_live -v`. Live inference has not been verified without a sample image and server key.
