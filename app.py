import base64
import csv
import hmac
import io
import math
import os
import tempfile
import time
from pathlib import Path
from threading import Lock
from uuid import uuid4

from flask import Flask, jsonify, request, send_file, send_from_directory, Response
from werkzeug.exceptions import HTTPException
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

WORKSPACE = 'gregwhiteings-gmail-com'
WORKFLOW = 'tyre-piece-sam-3-label-sample-1794525875800'
MODEL = 'gregwhiteings-gmail-com/recycling-yard-tyre-counting-1-rfdetr-seg-small-t1'
app = Flask(__name__, static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 12 * 1024 * 1024
DATA_DIR = Path(os.environ.get('DATA_DIR', tempfile.gettempdir() + '/tyrecycle-reviews'))
DATA_DIR.mkdir(parents=True, exist_ok=True)
# In-memory jobs are intentionally ephemeral; deploy with a single worker or replace with a DB.
JOBS = {}
LOCK = Lock()

@app.before_request
def require_access():
    password = os.environ.get('APP_ACCESS_PASSWORD', '')
    if not password:
        return jsonify(error='Server access password not configured'), 503
    auth = request.authorization
    if not auth or not hmac.compare_digest(auth.password or '', password):
        return Response('Authentication required', 401, {'WWW-Authenticate': 'Basic realm="TyreCycle Review"'})

class WorkflowError(Exception):
    pass

def infer(image_path, reference_px, reference_cm):
    key = os.environ.get('ROBOFLOW_API_KEY')
    if not key:
        raise WorkflowError('ROBOFLOW_API_KEY is not configured on the server')
    from inference_sdk import InferenceHTTPClient, InferenceConfiguration
    client = InferenceHTTPClient(api_url='https://serverless.roboflow.com', api_key=key)
    client.configure(InferenceConfiguration(api_key_transport='header'))
    last = None
    for attempt in range(3):
        try:
            result = client.run_workflow(
                workspace_name=WORKSPACE, workflow_id=WORKFLOW,
                images={'image': str(image_path)},
                parameters={'reference_distance_px': float(reference_px), 'reference_distance_cm': float(reference_cm)},
            )
            if not isinstance(result, list) or not result or not isinstance(result[0], dict):
                raise WorkflowError('Unexpected workflow response format')
            item = result[0]
            if not {'predictions', 'count', 'measurements'}.issubset(item):
                raise WorkflowError('Workflow response is missing required outputs')
            return item
        except WorkflowError:
            raise
        except Exception as exc:
            last = exc
            if attempt < 2:
                time.sleep(0.7 * (2 ** attempt))
    raise WorkflowError(f'Roboflow request failed after retries: {type(last).__name__}') from last

def safe_number(value, default=0):
    try:
        result = float(value)
        return result if math.isfinite(result) and result >= 0 else default
    except (TypeError, ValueError):
        return default

def image_bytes(value):
    if isinstance(value, dict):
        value = value.get('value') or value.get('image')
    if not isinstance(value, str):
        return None
    if value.startswith('data:'):
        value = value.split(',', 1)[-1]
    try:
        return base64.b64decode(value, validate=True)
    except (ValueError, base64.binascii.Error):
        return None

def public_job(job):
    return {k: job[k] for k in ('id', 'filename', 'count', 'measurements', 'pieces', 'reviewed', 'reviewer', 'notes', 'excluded_ids', 'reviewed_count', 'calibration_px', 'calibration_cm', 'has_overlay')}

@app.get('/')
def index():
    return send_from_directory('static', 'index.html')

@app.post('/api/analyze')
def analyze():
    upload = request.files.get('image')
    if not upload or not upload.filename:
        return jsonify(error='Choose a photograph first'), 400
    raw = upload.read()
    if len(raw) > 10 * 1024 * 1024:
        return jsonify(error='Maximum photo size is 10 MB'), 400
    from PIL import Image, UnidentifiedImageError
    try:
        im = Image.open(io.BytesIO(raw))
        im.verify()
        im = Image.open(io.BytesIO(raw))
        original_width, original_height = im.size
        im.thumbnail((2400, 2400))
        calibration_scale = im.width / original_width
        if im.mode != 'RGB':
            im = im.convert('RGB')
    except (UnidentifiedImageError, OSError, ValueError):
        return jsonify(error='Unsupported or invalid image'), 400
    px = safe_number(request.form.get('reference_distance_px'))
    cm = safe_number(request.form.get('reference_distance_cm'))
    if px and cm:
        px *= calibration_scale
    if bool(px) != bool(cm):
        return jsonify(error='Enter both ruler calibration values, or neither'), 400
    job_id = uuid4().hex
    photo_path = DATA_DIR / f'{job_id}.jpg'
    im.save(photo_path, format='JPEG', quality=88)
    try:
        result = infer(photo_path, px, cm)
    except WorkflowError as exc:
        photo_path.unlink(missing_ok=True)
        return jsonify(error=str(exc)), 502
    overlay = image_bytes(result.get('output_image'))
    if overlay and len(overlay) < 20 * 1024 * 1024:
        (DATA_DIR / f'{job_id}-overlay.png').write_bytes(overlay)
    measurements = result.get('measurements') if isinstance(result.get('measurements'), dict) else {}
    pieces = measurements.get('pieces') if isinstance(measurements.get('pieces'), list) else []
    # Deliberately omit raw predictions, polygons and base64 images from browser response.
    cleaned = []
    for i, piece in enumerate(pieces):
        if not isinstance(piece, dict):
            continue
        cleaned.append({k: piece.get(k) for k in ('id', 'area_px2', 'area_cm2', 'longest_dimension_px', 'longest_dimension_cm')})
    job = dict(id=job_id, filename=upload.filename[:100], count=result['count'],
               measurements={k: v for k, v in measurements.items() if k != 'pieces'}, pieces=cleaned,
               reviewed=False, reviewer='', notes='', excluded_ids=[], reviewed_count=None,
               calibration_px=px, calibration_cm=cm, has_overlay=bool(overlay))
    with LOCK:
        JOBS[job_id] = job
    return jsonify(public_job(job))

@app.get('/api/jobs/<job_id>/image/<kind>')
def job_image(job_id, kind):
    if job_id not in JOBS or kind not in ('original', 'overlay'):
        return jsonify(error='Not found'), 404
    path = DATA_DIR / (f'{job_id}.jpg' if kind == 'original' else f'{job_id}-overlay.png')
    if not path.is_file():
        return jsonify(error='Not found'), 404
    return send_file(path, mimetype='image/jpeg' if kind == 'original' else 'image/png')

@app.post('/api/jobs/<job_id>/review')
def review(job_id):
    with LOCK:
        job = JOBS.get(job_id)
        if not job:
            return jsonify(error='Not found'), 404
        data = request.get_json(silent=True) or {}
        if data.get('confirm') is not True:
            return jsonify(error='Explicit review confirmation is required'), 400
        reviewer = str(data.get('reviewer', '')).strip()[:100]
        if not reviewer:
            return jsonify(error='Enter the reviewer name'), 400
        try:
            count = int(data.get('reviewed_count'))
        except (TypeError, ValueError):
            return jsonify(error='Enter a valid reviewed count'), 400
        if count < 0 or count > 100000:
            return jsonify(error='Reviewed count is out of range'), 400
        valid_ids = {str(p['id']) for p in job['pieces']}
        excluded = [str(x) for x in data.get('excluded_ids', [])]
        if any(x not in valid_ids for x in excluded):
            return jsonify(error='Invalid excluded piece ID'), 400
        job.update(reviewed=True, reviewer=reviewer, notes=str(data.get('notes', ''))[:1000],
                   excluded_ids=excluded, reviewed_count=count)
        return jsonify(public_job(job))

def reviewed_job(job_id):
    job = JOBS.get(job_id)
    if not job or not job['reviewed']:
        return None
    return job

@app.get('/api/jobs/<job_id>/export.csv')
def export_csv(job_id):
    job = reviewed_job(job_id)
    if not job:
        return jsonify(error='Review and confirm before exporting'), 403
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(['Job', job_id, 'File', job['filename']])
    writer.writerow(['Review status', 'Human reviewed', 'Reviewer', job['reviewer']])
    writer.writerow(['AI count (unverified)', job['count'], 'Reviewed count', job['reviewed_count']])
    writer.writerow(['Reference pixels', job['calibration_px'], 'Reference cm', job['calibration_cm']])
    writer.writerow(['Notes', job['notes']])
    writer.writerow([])
    writer.writerow(['Piece ID', 'Included', 'Area px2', 'Area cm2', 'Longest dimension px', 'Longest dimension cm'])
    for p in job['pieces']:
        writer.writerow([p.get('id'), str(p.get('id')) not in job['excluded_ids'], p.get('area_px2'), p.get('area_cm2'), p.get('longest_dimension_px'), p.get('longest_dimension_cm')])
    return send_file(io.BytesIO(out.getvalue().encode('utf-8-sig')), as_attachment=True, download_name=f'tyre-review-{job_id[:8]}.csv', mimetype='text/csv')

@app.get('/api/jobs/<job_id>/export.pdf')
def export_pdf(job_id):
    job = reviewed_job(job_id)
    if not job:
        return jsonify(error='Review and confirm before exporting'), 403
    buf = io.BytesIO()
    pdf = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    y = height - 48
    def line(value, size=10):
        nonlocal y
        if y < 55:
            pdf.showPage()
            y = height - 48
        pdf.setFont('Helvetica', size)
        pdf.drawString(42, y, str(value)[:110])
        y -= 19
    line('TYRE PIECE | HUMAN REVIEW REPORT', 16)
    y -= 12
    for value in [f'Job: {job_id}', f'Photo: {job["filename"]}', 'Status: HUMAN REVIEWED',
                  f'Reviewer: {job["reviewer"]}', f'AI count (unverified): {job["count"]}',
                  f'Reviewed count: {job["reviewed_count"]}',
                  f'Calibration: {job["calibration_px"]} px = {job["calibration_cm"]} cm',
                  f'Notes: {job["notes"]}', 'Visible projected area only. Not mass or volume.',
                  'Perspective and calibration can affect accuracy.']:
        line(value)
    y -= 12
    line('PIECES (excluded items marked)', 12)
    for p in job['pieces']:
        excluded = 'EXCLUDED' if str(p.get('id')) in job['excluded_ids'] else 'included'
        line(f'#{p.get("id")} {excluded} | area: {p.get("area_cm2") or p.get("area_px2")} | longest: {p.get("longest_dimension_cm") or p.get("longest_dimension_px")}')
    pdf.save()
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name=f'tyre-review-{job_id[:8]}.pdf', mimetype='application/pdf')

@app.errorhandler(Exception)
def errors(exc):
    if isinstance(exc, HTTPException):
        return jsonify(error=exc.description), exc.code
    app.logger.exception('Request failed')
    return jsonify(error='Server error'), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=int(os.environ.get('PORT', '8000')))
