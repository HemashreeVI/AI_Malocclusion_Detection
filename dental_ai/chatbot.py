"""Plain-language Q&A about the findings, using Claude.

Only de-identified findings are sent — never the patient's name, age or
clinic. Requires ``pip install anthropic`` and an ANTHROPIC_API_KEY.
"""
from .config import reliability_note

MODEL = 'claude-opus-5-5'

SYSTEM_PROMPT = (
    'You explain dental X-ray AI findings to patients in plain, friendly language. '
    'You are not a dentist and must not diagnose: explain what a finding usually means, '
    'what questions to ask the dentist, and recommend a professional exam. '
    'Findings marked "unvalidated" come from classes the model performs poorly on; say so '
    'if asked about them. Keep answers under 200 words.'
)


def findings_context(summary):
    """De-identified description of the analysis for the model."""
    if not summary:
        return 'No analysis has been run yet.'
    lines = [f"Risk score {summary['risk']:.0f}/100 ({summary['status']}).", 'Findings:']
    for d in summary['detections']:
        tooth = f"tooth {d['tooth']}" if d['tooth'] is not None else 'no tooth'
        status = 'validated' if d['reliable'] else f"unvalidated ({reliability_note(d['class'])})"
        lines.append(f"- {d['class']} at {tooth}, confidence {d['conf']*100:.0f}%, {status}")
    if len(lines) == 2:
        lines.append('- none')
    return '\n'.join(lines)


def ask(question, summary):
    if not question or not question.strip():
        return 'Please type a question.'
    if not summary:
        return 'Run an analysis first, then ask about the results.'
    try:
        import anthropic
    except ImportError:
        return 'Install the Anthropic SDK (pip install anthropic) and set ANTHROPIC_API_KEY.'

    client = anthropic.Anthropic()
    try:
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            output_config={'effort': 'low'},
            betas=['server-side-fallback-2026-07-01'],
            fallbacks='default',
            system=SYSTEM_PROMPT,
            messages=[{'role': 'user',
                       'content': f'Analysis results:\n{findings_context(summary)}\n\nQuestion: {question}'}],
        )
    except TypeError as e:  # raised before sending when no credentials are configured
        if 'authentication' not in str(e):
            raise
        return 'Chatbot unavailable: set ANTHROPIC_API_KEY to enable it.'
    except anthropic.AuthenticationError:
        return 'Chatbot unavailable: ANTHROPIC_API_KEY is invalid.'
    except anthropic.RateLimitError:
        return 'Chatbot is busy (rate limited). Please try again shortly.'
    except anthropic.APIStatusError as e:
        return f'Chatbot error ({e.status_code}).'
    except anthropic.APIConnectionError:
        return 'Chatbot unavailable: could not reach the API.'

    if response.stop_reason == 'refusal':
        return "Sorry, I can't help with that question. Please ask your dentist."
    return ''.join(b.text for b in response.content if b.type == 'text').strip()
