import os

from PIL import Image
from pypdf import PdfReader

from dental_ai.fdi import generate_fdi_chart
from dental_ai.findings import make_finding
from dental_ai.report import export_pdf, report_text, summarize


def sample_summary():
    dets = [make_finding('Caries', 0.82, [0, 0, 10, 10], 36, 'estimated'),
            make_finding('Caries', 0.61, [0, 0, 10, 10], 46, 'estimated'),
            make_finding('Malaligned', 0.7, [0, 0, 10, 10], 12, 'estimated'),
            make_finding('Filling', 0.95, [0, 0, 10, 10], 26, 'estimated'),
            make_finding('Supra Eruption', 0.76, [0, 0, 10, 10], 37, 'estimated'),
            make_finding('impacted tooth', 0.76, [0, 0, 10, 10], 38, 'estimated')]
    return summarize(dets, 0.25, {'name': 'Test Patient', 'age': '28', 'clinic': 'Clinic — Ünïcode'})


def test_text_report_sections():
    text = report_text(sample_summary())
    assert 'UNVALIDATED FINDINGS' in text
    assert 'Supra Eruption' in text and '39 training examples' in text
    assert 'Possible impacted tooth' in text
    assert 'Caries' in text and '×2' in text


def test_pdf_has_unicode_and_all_pages_footer(tmp_path):
    s = sample_summary()
    img = Image.new('RGB', (1200, 600), (90, 90, 90))
    path = export_pdf(s, img, img, generate_fdi_chart(s['detections']), out_dir=tmp_path)
    assert os.path.exists(path)
    reader = PdfReader(path)
    text = '\n'.join(p.extract_text() for p in reader.pages)
    assert '?' * 4 not in text
    assert 'Ünïcode' in text
    assert '₹' in text
    for i, page in enumerate(reader.pages, 1):
        assert f'Page {i}/{len(reader.pages)}' in page.extract_text()


def test_pdf_paths_are_unique(tmp_path):
    s = sample_summary()
    assert export_pdf(s, out_dir=tmp_path) != export_pdf(s, out_dir=tmp_path)
