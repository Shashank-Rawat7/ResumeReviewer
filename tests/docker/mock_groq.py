"""A tiny stand-in for the Groq chat-completions API, used ONLY for offline / Docker smoke tests.

Standard library only. It answers POST /openai/v1/chat/completions with canned, valid JSON for each of the five
pipeline stages (recognised by a phrase in the prompt), plus usage numbers, so the real parser, LangChain and the
real Groq SDK run end to end without any network access or API key. It also serves:
    GET /health  -> {"status": "ok"}
    GET /stats   -> {"requests": <number of completions served>}
It never logs prompts or resume text.

Run:  python tests/docker/mock_groq.py [port]      (default 8080, binds 0.0.0.0)
"""
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

JD_SENTENCE = "Experience with Kubernetes is required."          # the smoke test's job description contains this

STAGES = (
    ("optimizing resumes for Applicant Tracking Systems", "e3", {
        "optimized_bullets": [{"section": "Projects - Mock Project", "bullets": ["Built a mock service end to end."]}]}),
    ("Compare the job description with the candidate's full resume", "e2", {
        "crucial": [{"skill": "Kubernetes", "category": "technical_skills", "status": "missing",
                     "reason": "Mock reason: the resume does not show it.", "jd_evidence": JD_SENTENCE}],
        "good_to_have": [], "optional": []}),
    ("analyzing job descriptions to extract required competencies", "e1", {
        "technical_skills": ["Python", "Kubernetes"], "soft_skills": ["Communication"],
        "academic_qualifications": ["Bachelor's degree"]}),
    ("extract three categories of information", "credentials", {
        "technical_skills": ["Python", "SQL"], "soft_skills": ["Teamwork"], "academic_qualifications": ["B.Tech"]}),
    ("generate a professional summary in structured JSON format", "summary", {
        "skills": ["Python", "SQL"],
        "personal_info_and_education": {"name": "Mock Candidate", "email": "mock@example.test", "education": []},
        "work_experience": [], "key_projects": [{"project_name": "Mock Project", "objective": "Demo",
                                                  "technologies_used": ["Python"], "outcomes": "n/a", "bullets": []}],
        "extracurricular_activities_and_soft_skills": []}),
)

_lock = threading.Lock()
_count = 0


def pick(text):
    for phrase, name, body in STAGES:
        if phrase in text:
            return name, body
    return "unknown", {"note": "mock"}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):          # keep the server silent: no request bodies in logs
        return

    def _send(self, code, payload):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok"})
        elif self.path == "/stats":
            with _lock:
                self._send(200, {"requests": _count})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        global _count
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length).decode("utf-8", "replace") if length else "{}"
        if not self.path.endswith("/chat/completions"):
            self._send(404, {"error": "not found"})
            return
        try:
            request = json.loads(body)
            text = " ".join(str(m.get("content", "")) for m in request.get("messages", []))
        except ValueError:
            self._send(400, {"error": "bad request"})
            return
        stage, answer = pick(text)
        with _lock:
            _count += 1
        content = json.dumps(answer)
        prompt_tokens = max(1, len(text) // 4)
        completion_tokens = max(1, len(content) // 4)
        self._send(200, {
            "id": "mock-%d" % _count, "object": "chat.completion", "created": int(time.time()),
            "model": request.get("model", "mock"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                      "total_tokens": prompt_tokens + completion_tokens},
        })


def serve(port, host="0.0.0.0"):
    server = ThreadingHTTPServer((host, port), Handler)
    return server


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    print("mock groq listening on %d" % port, flush=True)
    serve(port).serve_forever()
