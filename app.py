"""Gradio app: AI-assisted panoramic dental X-ray analysis.

    python app.py                 # local only
    SHARE=1 python app.py         # public gradio.live link (see privacy note)

Environment variables: MODEL_PATH, HF_MODEL_REPO, HF_MODEL_FILE, MODEL_GDRIVE_ID,
TOOTH_MODEL_PATH, MIN_RELIABLE_AP50, MIN_TEST_INSTANCES, MIN_TRAIN_INSTANCES,
ANTHROPIC_API_KEY, SHARE, PORT.
"""
import os

import gradio as gr
import numpy as np
from PIL import Image

from dental_ai import chatbot
from dental_ai.cam import DetectionCAM, overlay_heatmap
from dental_ai.config import gating_rule, reliability_note
from dental_ai.fdi import generate_fdi_chart
from dental_ai.inference import annotate, detect, finding_label, load_models
from dental_ai.report import cost_text, export_pdf, report_text, summarize

model, tooth_model = load_models()
cam_engine = DetectionCAM(model)
print(f'Model loaded: {len(model.names)} classes. Tooth detector: {"yes" if tooth_model else "no (estimating)"}')

ALL_FINDINGS = 'All findings (top 8)'


def _patient(name, age, clinic):
    return {'name': name, 'age': age, 'clinic': clinic}


def _cam_image(img_rgb, detections):
    heat, note, used = cam_engine.explain(img_rgb, detections)
    return Image.fromarray(overlay_heatmap(img_rgb, heat, used, note.split(':')[0])), note


def _findings_table(detections):
    rows = []
    for d in sorted(detections, key=lambda d: (not d['reliable'], -d['conf'])):
        tooth = '' if d['tooth'] is None else f"{d['tooth']}{'~' if d['tooth_source'] == 'estimated' else ''}"
        rows.append([tooth, d['class'], d['severity'], f"{d['conf']*100:.0f}%",
                     'yes' if d['reliable'] else f"no ({reliability_note(d['class'])})"])
    return rows


def analyze(image, conf_threshold, name, age, clinic):
    if image is None:
        raise gr.Error('Please upload an X-ray image.')
    img_rgb = np.array(image.convert('RGB'))
    detections = detect(model, img_rgb, conf_threshold, tooth_model=tooth_model)

    annotated = Image.fromarray(annotate(img_rgb, detections))
    cam_img, cam_note = _cam_image(img_rgb, detections)
    fdi = generate_fdi_chart(detections)
    summary = summarize(detections, conf_threshold, _patient(name, age, clinic), cam_note)

    state = {'img': img_rgb, 'detections': detections, 'summary': summary,
             'annotated': annotated, 'cam': cam_img, 'fdi': fdi}
    choices = [ALL_FINDINGS] + [f'{i}: {finding_label(d)}' for i, d in enumerate(detections)]
    return (annotated, cam_img, fdi, report_text(summary), _findings_table(detections),
            cost_text(summary), gr.update(choices=choices, value=ALL_FINDINGS), state, None)


def explain_one(choice, state):
    if not state or not choice:
        return gr.update()
    dets = state['detections']
    if choice != ALL_FINDINGS:
        dets = [dets[int(choice.split(':', 1)[0])]]
    cam_img, _ = _cam_image(state['img'], dets)
    state['cam'] = cam_img
    return cam_img


def make_pdf(state, name, age, clinic):
    if not state:
        raise gr.Error('Run an analysis first.')
    s = dict(state['summary'], patient={k: (v or '').strip() for k, v in _patient(name, age, clinic).items()})
    return export_pdf(s, state['annotated'], state['cam'], state['fdi'])


def ask(question, state):
    return chatbot.ask(question, state['summary'] if state else None)


with gr.Blocks(title='Dental AI Analyzer') as demo:
    gr.Markdown(f"""
# 🦷 Dental X-ray AI Analyzer — v3
Detection · HiResCAM explanations · FDI chart · PDF report

> ⚠️ **Research tool, not a diagnosis.** Classes with {gating_rule()} are marked
> *unvalidated* and excluded from risk score, costs and recommendations.
> Do not upload identifiable patient images without consent{', especially on a public share link' if os.environ.get('SHARE') else ''}.
""")
    state = gr.State(None)

    with gr.Tab('🔍 Analysis'):
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(type='pil', label='Panoramic X-ray')
                conf = gr.Slider(0.10, 0.75, value=0.25, step=0.05,
                                 label='Confidence threshold (lower = more, less certain detections)')
                with gr.Accordion('Patient details (PDF only, never sent to the chatbot)', open=False):
                    name = gr.Textbox(label='Patient name')
                    age = gr.Textbox(label='Age')
                    clinic = gr.Textbox(label='Clinic / doctor')
                btn = gr.Button('Analyze X-ray', variant='primary')
            with gr.Column(scale=2):
                out_ann = gr.Image(label='Findings (~ = estimated tooth number)', height=420)

        with gr.Row():
            with gr.Column():
                explain = gr.Dropdown(label='Explain', choices=[ALL_FINDINGS], value=ALL_FINDINGS)
                out_cam = gr.Image(label='Explanation map', height=380)
            out_fdi = gr.Image(label='FDI tooth chart', height=300)

        out_table = gr.Dataframe(headers=['Tooth', 'Finding', 'Severity', 'Confidence', 'Validated'],
                                 label='Findings', interactive=False, wrap=True)
        out_report = gr.Textbox(label='Report', lines=30)
        with gr.Row():
            pdf_btn = gr.Button('📄 Export PDF')
            pdf_out = gr.File(label='PDF report')

    with gr.Tab('💰 Costs'):
        out_cost = gr.Textbox(label='Indicative treatment costs', lines=16)

    with gr.Tab('🤖 Ask about the results'):
        gr.Markdown('Only the de-identified findings are sent to the AI model. Requires `ANTHROPIC_API_KEY`.')
        q = gr.Textbox(label='Your question', placeholder='What does "Caries on tooth 36" mean?')
        ask_btn = gr.Button('Ask', variant='primary')
        answer = gr.Markdown()

    with gr.Tab('ℹ️ How to read the results'):
        gr.Markdown("""
### Tooth numbers (FDI)
Two digits: quadrant (1 upper-right, 2 upper-left, 3 lower-left, 4 lower-right — *patient's* sides) then
tooth position from the midline (1 central incisor … 8 third molar). `36` = lower-left first molar.
A `~` means the number was estimated from the finding's position in the image.

### Severity vs confidence
*Severity* is how serious that kind of finding usually is. *Confidence* is how sure the model is about this
detection. They are reported separately.

### Validated vs unvalidated
A class is *validated* when its per-class test-set AP@50 (from `scripts/evaluate.py`) is high enough, or —
until that evaluation has been run — when it had enough training examples. Unvalidated classes are shown greyed out with corner marks only, and are not used for the risk score, costs or recommendations.

### Explanation map
HiResCAM (a Grad-CAM variant) shows which regions most increased the score of the selected detection(s). Pick a single finding in
the *Explain* dropdown to see its own map. With no detections, a class-agnostic EigenCAM map is shown.
""")

    btn.click(analyze, [inp, conf, name, age, clinic],
              [out_ann, out_cam, out_fdi, out_report, out_table, out_cost, explain, state, pdf_out])
    explain.input(explain_one, [explain, state], out_cam)
    pdf_btn.click(make_pdf, [state, name, age, clinic], pdf_out)
    ask_btn.click(ask, [q, state], answer)
    q.submit(ask, [q, state], answer)

if __name__ == '__main__':
    demo.launch(share=bool(os.environ.get('SHARE')), theme=gr.themes.Soft(),
                server_port=int(os.environ.get('PORT', 7860)))
