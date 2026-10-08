"""
quiz_generation.py
OWNER: Person 4 — Quiz Generation & UI

NOTE: retrieve() (from retrieval.py, Person 2) returns a list of DICTS:
{chunk_id, text, source, chunk_index, distance} — not plain strings.
Use result["text"] to get the chunk content.

Uses Google's Gemini API (google-genai SDK).
Set the GEMINI_API_KEY environment variable before running.
"""
import time 
import os
import json
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from retrieval import retrieve


# Load .env file
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# Get Gemini API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY was not found. "
        "Please check your .env file."
    )

# Create Gemini client
llm_client = genai.Client(
    api_key=GEMINI_API_KEY
)
MODEL_NAME = "gemini-3.8-flash"


def generate_quiz(topic_query: str, num_questions: int = 3, top_k: int = 2):
    """Generate multiple-choice quiz questions grounded in retrieved context."""
    results = retrieve(topic_query, top_k=top_k)
    if not results:
        return []

    context = "\n\n".join(r["text"] for r in results)
    prompt = f"""Based on the following text, generate {num_questions} multiple-choice questions
to test a student's understanding. For each question, provide exactly 4 options (A-D)
and indicate the correct answer letter.

Respond with ONLY a JSON array in this exact format, no other text, no markdown fences:
[{{"question": "...", "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}}, "answer": "A"}}]

Text:
{context}
"""

    # Try the Gemini request up to 3 times.
    # 503 = temporary service overload/high demand.
    for attempt in range(3):

        try:
            response = llm_client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config={
                    "response_mime_type": "application/json"
                }
            )

            raw = (response.text or "").strip()

            if not raw:
                return []

            # Remove markdown fences if Gemini happens to add them.
            if raw.startswith("```"):
                raw = raw.strip("`")

                if raw.lower().startswith("json"):
                    raw = raw[4:].strip()

            quiz = json.loads(raw)

            # Basic validation
            if not isinstance(quiz, list):
                return []

            valid_questions = []

            for question in quiz:
                if not isinstance(question, dict):
                    continue

                if not all(
                    key in question
                    for key in ["question", "options", "answer"]
                ):
                    continue

                options = question["options"]

                if not isinstance(options, dict):
                    continue

                if not all(letter in options for letter in ["A", "B", "C", "D"]):
                    continue

                if question["answer"] not in ["A", "B", "C", "D"]:
                    continue

                valid_questions.append(question)

            return valid_questions

        except Exception as e:

            error_message = str(e).lower()

            # Gemini is temporarily overloaded.
            if "503" in error_message or "unavailable" in error_message:

                if attempt < 2:
                    wait_time = 5 * (attempt + 1)

                    time.sleep(wait_time)
                    continue

                raise RuntimeError(
                    "Gemini is currently experiencing high demand. "
                    "Please wait a little and try generating the quiz again."
                ) from e

            # Other Gemini/API errors should be shown normally.
            raise


QUIZ_REVIEW_TOPICS = [
    # "network security basics",
]


def run_quiz_review(topics=None, num_questions=3):
    topics = topics or QUIZ_REVIEW_TOPICS
    for topic in topics:
        quiz = generate_quiz(topic, num_questions=num_questions)
        print(f"=== Topic: {topic} ===")
        for i, q in enumerate(quiz, 1):
            print(f"Q{i}: {q['question']}")
            for k, v in q["options"].items():
                print(f"   {k}. {v}")
            print(f"   Correct: {q['answer']}\n")


if __name__ == "__main__":
    run_quiz_review()