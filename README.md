# GapReady

An AI-powered interview preparation tool that shows students exactly where they stand against a job or internship they're applying for, and helps them prepare for it.

Upload a resume and paste a job description. GapReady tells you your match score, exactly which skills you're missing, and generates tailored interview questions with hints and sample answers, all backed by a built-in AI coach.

---

## What It Does

1. **Upload your resume** (.docx) and **paste a job description**
2. GapReady runs a **hybrid analysis**:
   - A statistical layer identifies the job description's actual requirements
   - An LLM verifies which skills your resume demonstrates, even under different wording
3. You get:
   - An **honest match score**, based on the job's real requirement count
   - A **gap list**, grouped into categories
   - **Interview questions** tailored to your gaps, your real projects, and core behavioral topics, each with a hint and an on-demand sample answer
   - **GapBot**, an AI chat assistant that knows your latest results and helps you prepare
   - **Export** your full prep sheet as a PDF or text file

---

## Features

### Resume & Job Description Analysis
- Extracts text from `.docx` resumes (paragraphs and tables)
- Validates that the uploaded resume is a genuine resume (not a report, thesis, or single-project document)
- Validates that the pasted text is a genuine job/internship posting (not technical notes, advice, or unrelated text)
- Structural validation does **not** compare the resume and job description against each other. A computer science resume paired with a marketing internship posting is a valid input pair, it will simply produce a low match score and many gaps, which is the correct outcome for that scenario.

### Match Score
- Computed as `(total job requirements − missing requirements) ÷ total job requirements`
- The total requirement count is extracted directly from the job description's bullet points using code, not the AI, so the score stays consistent across repeated runs on the same input
- A skill counts as "covered" even if phrased differently on the resume than in the job description (e.g. "Flask REST API development" satisfies a "backend development" requirement)

### Skill Gap Detection (Hybrid NLP + LLM)
- A TF-IDF based statistical layer proposes candidate gaps from the job description
- An LLM (Llama 3.3 70B via Groq) verifies each candidate: discards noise (non-skill fragments), discards false positives (skills the resume demonstrates under different wording), and merges related fragments into one clean, named gap
- Every gap is grounded in an explicit line of the job description; gaps are never invented from general assumptions about the role

### Skill Gap Clustering (K-Means)
- When there are enough gaps to make grouping useful, K-Means clusters them into related categories (e.g. "Data related," "Deployment related") for a more organized display
- Falls back to a plain list when there are too few gaps for grouping to help

### Interview Question Generation
- Questions are generated in a fixed structure:
  - One question per identified gap, tagged `based_on: gap`
  - 2 to 3 questions based on the student's actual named projects, tagged `based_on: project`
  - At least 2 behavioral questions (teamwork, communication, conflict, motivation), tagged `based_on: general`
- Each question includes a category (`technical` or `behavioral`), a short hint, and a natural follow-up question
- Question difficulty matches the job description's actual requirements, not the student's apparent experience level
- The number of gap-tagged questions is enforced in code to always match the number of identified gaps

### Sample Answers
- Generated on demand per question, so students practice recalling an answer themselves before seeing one
- Written in first person, natural spoken tone

### GapBot (AI Chat Assistant)
- A scoped chat assistant for interview prep, resume advice, explaining job description terms, and understanding GapReady's own results
- Aware of the student's latest match score, covered skills, and gaps, so it can give specific rather than generic advice
- Politely declines unrelated requests (e.g. general coding help, homework) and steers back to career topics
- Structured (bulleted/headed) responses for multi-part answers, plain sentences for simple ones

### Export
- Full prep sheet exportable as a PDF (via print) or a plain text file

---

## Tech Stack

**Backend:** Flask, flask-cors, python-docx, scikit-learn, gunicorn
**AI:** Groq API, Llama 3.3 70B
**Frontend:** HTML, CSS, vanilla JavaScript (no framework)
**NLP:** TF-IDF vectorization and cosine similarity (scikit-learn), K-Means clustering (scikit-learn)

---

## Project Structure

```
interview_question_generator/
├── app.py                 # Flask backend: routes, prompts, scoring logic
├── nlp_analysis.py         # TF-IDF analysis and K-Means clustering
├── requirements.txt         # Python dependencies
└── static/
    └── index.html           # Frontend: UI, chat widget, all client-side logic
```

---

## Setup & Running Locally

### 1. Install dependencies

```
python -m pip install -r requirements.txt
```

(Use `python -m pip` rather than plain `pip` if your system blocks direct pip execution.)

### 2. Set your Groq API key as an environment variable

GapReady never stores the API key in code. Set it as an environment variable named `interview_qs_groq_key`.

**Windows (Command Prompt), temporary for the current session:**
```
set interview_qs_groq_key=your_key_here
```

**Windows, permanent:** Search "Edit the system environment variables" → Environment Variables → New (under User variables) → Name: `interview_qs_groq_key`, Value: your key.

### 3. Run the app

```
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

---

## Deployment (Render)

1. Push the project to a GitHub repository (excluding `venv/`, and anything containing API keys)
2. Create a new Web Service on Render, connected to the repository
3. Build command: `pip install -r requirements.txt`
4. Start command: `gunicorn app:app`
5. Add an environment variable in Render's dashboard: Key `interview_qs_groq_key`, Value your Groq API key
6. Deploy. Render auto-redeploys on every subsequent `git push` to the connected branch.

---

## Known Limitations

- The system reads resumes and job descriptions literally rather than inferring implied tools. For example, a resume listing "Logistic Regression, Random Forest" without naming "scikit-learn" may not receive credit for scikit-learn experience. This is a deliberate tradeoff: earlier, more permissive inference caused scoring instability during testing, so reliability was chosen over generosity. Users get more accurate results by naming specific tools and libraries on their resume.
- Only `.docx` resumes are supported; PDF resumes are not currently accepted.
- The chat assistant only has access to a summary of the latest analysis (score, gaps, covered skills), not the full resume text.
- As with any third-party LLM API, availability is subject to the provider's uptime and rate limits.

---

## Author

Jamiha Tashfeen, BSCS15 A
