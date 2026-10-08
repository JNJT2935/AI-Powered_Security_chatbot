import os
import re
import html
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)


from llm_answering import answer_question
from quiz_generation import generate_quiz


st.set_page_config(
    page_title="AI Security Chatbot",
    page_icon="🛡️",
    layout="centered"
)


citation_component = st.components.v2.component(
    name="inline_citations",

    html="""
    <div id="citation-container"></div>
    """,

    css="""
    .citation-answer {
        font-size: 16px;
        line-height: 1.7;
        color: var(--st-text-color);
    }

    .citation {
        position: relative;
        display: inline-block;
        color: var(--st-primary-color);
        font-weight: 600;
        cursor: help;
        margin-left: 2px;
        text-decoration: none;
    }

    .citation:hover {
        text-decoration: underline;
    }

    .citation-tooltip {
        visibility: hidden;
        opacity: 0;

        position: absolute;
        z-index: 9999;

        width: 360px;
        max-width: 80vw;

        left: 50%;
        bottom: calc(100% + 10px);

        transform: translateX(-50%) translateY(5px);

        padding: 14px 16px;

        background: var(--st-background-color);
        color: var(--st-text-color);

        border: 1px solid var(--st-border-color);
        border-radius: 10px;

        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.25);

        font-size: 13px;
        line-height: 1.5;

        text-align: left;

        transition:
            opacity 0.15s ease,
            transform 0.15s ease;

        pointer-events: none;
    }

    .citation:hover .citation-tooltip {
        visibility: visible;
        opacity: 1;
        transform: translateX(-50%) translateY(0);
    }

    .citation-source {
        font-weight: 700;
        margin-bottom: 8px;
        color: var(--st-primary-color);
    }

    .citation-excerpt {
        color: var(--st-text-color);
        opacity: 0.9;
    }

    /* Keep tooltip inside the viewport when possible */
    .citation:last-child .citation-tooltip {
        left: auto;
        right: 0;
        transform: translateY(5px);
    }

    .citation:last-child:hover .citation-tooltip {
        transform: translateY(0);
    }
    """,

    js="""
    export default function(component) {

        const {
            data,
            parentElement
        } = component;

        const container =
            parentElement.querySelector("#citation-container");

        if (!container) {
            return;
        }

        const answer = data?.answer || "";
        const chunks = data?.chunks || [];

        // Escape HTML so course material is treated as text
        // rather than executable HTML.
        function escapeHtml(value) {

            return String(value)
                .replace(/&/g, "&amp;")
                .replace(/</g, "&lt;")
                .replace(/>/g, "&gt;")
                .replace(/"/g, "&quot;")
                .replace(/'/g, "&#039;");
        }


        // Convert [1], [2], [3] into hoverable citations.
        function renderAnswer(text) {

            const escaped = escapeHtml(text);

            return escaped.replace(
                /\\[(\\d+)\\]/g,
                function(match, number) {

                    const index =
                        parseInt(number, 10) - 1;

                    if (
                        index < 0 ||
                        index >= chunks.length
                    ) {
                        return match;
                    }

                    const chunk = chunks[index];

                    const source =
                        escapeHtml(
                            chunk.source ||
                            "Unknown source"
                        );

                    const excerpt =
                        escapeHtml(
                            chunk.text ||
                            "No excerpt available."
                        );

                    return `
                        <span class="citation">
                            [${number}]

                            <span class="citation-tooltip">

                                <div class="citation-source">
                                    [${number}] ${source}
                                </div>

                                <div class="citation-excerpt">
                                    ${excerpt}
                                </div>

                            </span>
                        </span>
                    `;
                }
            );
        }


        container.innerHTML = `
            <div class="citation-answer">
                ${renderAnswer(answer)}
            </div>
        `;
    }
    """
)


def display_answer_with_citations(answer, chunks, key=None):
    """
    Display the AI answer with hoverable [1], [2], [3]
    citations.

    Hovering over a citation displays the corresponding
    retrieved course-material source and excerpt.
    """

    if not answer:
        return

    # Make sure chunks are valid dictionaries.
    safe_chunks = []

    for chunk in chunks or []:

        if isinstance(chunk, dict):

            safe_chunks.append({
                "source": chunk.get(
                    "source",
                    "Unknown source"
                ),
                "text": chunk.get(
                    "text",
                    "No excerpt available."
                )
            })

    citation_component(
        data={
            "answer": answer,
            "chunks": safe_chunks
        },
        key=key
    )


st.sidebar.title("🛡️ AI Security Learning")

page = st.sidebar.radio(
    "Choose a feature:",
    [
        "💬 Chatbot",
        "📝 Quiz Generator"
    ]
)


if page == "💬 Chatbot":

    st.title("🛡️ AI-Powered Security Chatbot")

    st.caption(
        "Ask questions about cybersecurity based on the course material."
    )


    if "messages" not in st.session_state:
        st.session_state.messages = []


    for message_index, message in enumerate(
        st.session_state.messages
    ):

        with st.chat_message(message["role"]):

            if message["role"] == "assistant":

                display_answer_with_citations(
                    message["content"],
                    message.get("chunks", []),
                    key=f"citation_history_{message_index}"
                )

            else:

                st.markdown(
                    message["content"]
                )


    if prompt := st.chat_input(
        "Ask a cybersecurity question..."
    ):

        # Display user message
        st.chat_message("user").markdown(prompt)


        # Save user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })


        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                try:

                    result = answer_question(
                        prompt,
                        history=st.session_state.messages[:-1],
                        top_k=7
                    )


                    # Get AI answer
                    bot_response = result["answer"]


                    # Get retrieved RAG chunks
                    chunks = result.get(
                        "chunks",
                        []
                    )

                    display_answer_with_citations(
                        bot_response,
                        chunks,
                        key=f"citation_current_{len(st.session_state.messages)}"
                    )


                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": bot_response,
                        "chunks": chunks
                    })


                except Exception as e:

                    st.error(
                        f"Error: {e}"
                    )


elif page == "📝 Quiz Generator":

    st.title("📝 Cybersecurity Quiz Generator")

    st.caption(
        "Generate multiple-choice questions based on your "
        "cybersecurity course material."
    )

    topic = st.text_input(
        "Enter a cybersecurity topic",
        placeholder="e.g. Network Security, Phishing, Malware"
    )


    num_questions = st.number_input(
        "Number of questions",
        min_value=1,
        max_value=10,
        value=3,
        step=1
    )


    if st.button(
        "Generate Quiz",
        type="primary"
    ):

        if not topic.strip():

            st.warning(
                "Please enter a cybersecurity topic."
            )

        else:

            with st.spinner(
                "Generating your quiz..."
            ):

                try:

                    quiz = generate_quiz(
                        topic_query=topic,
                        num_questions=num_questions,
                        top_k=2
                    )


                    if not quiz:

                        st.error(
                            "No quiz questions could be generated. "
                            "Try a different topic."
                        )


                    else:

                        st.session_state.quiz = quiz

                        st.session_state.quiz_submitted = False

                        st.session_state.quiz_answers = {}


                except Exception as e:

                    st.error(
                        f"Error generating quiz: {e}"
                    )


    if "quiz" in st.session_state:

        st.divider()

        st.subheader("Your Quiz")


        answers = {}


        for i, question in enumerate(
            st.session_state.quiz
        ):

            st.markdown(
                f"### Question {i + 1}"
            )


            st.write(
                question["question"]
            )


            selected_answer = st.radio(
                "Choose your answer:",

                options=list(
                    question["options"].keys()
                ),

                format_func=lambda x, q=question: (
                    f"{x}. {q['options'][x]}"
                ),

                key=f"quiz_question_{i}"
            )


            answers[i] = selected_answer


        st.divider()


        if st.button(
            "Submit Quiz",
            type="primary"
        ):

            score = 0


            for i, question in enumerate(
                st.session_state.quiz
            ):

                if answers[i] == question["answer"]:

                    score += 1


            total = len(
                st.session_state.quiz
            )


            percentage = (
                score / total
            ) * 100


            st.session_state.quiz_answers = answers

            st.session_state.quiz_score = score

            st.session_state.quiz_submitted = True


        if st.session_state.get(
            "quiz_submitted",
            False
        ):

            score = st.session_state.quiz_score

            total = len(
                st.session_state.quiz
            )

            percentage = (
                score / total
            ) * 100


            st.subheader(
                "Quiz Results"
            )


            if percentage >= 80:

                st.success(
                    f"🎉 Excellent! You scored "
                    f"{score}/{total} "
                    f"({percentage:.0f}%)."
                )


            elif percentage >= 50:

                st.info(
                    f"👍 Good job! You scored "
                    f"{score}/{total} "
                    f"({percentage:.0f}%)."
                )


            else:

                st.warning(
                    f"You scored {score}/{total} "
                    f"({percentage:.0f}%). "
                    f"Keep reviewing the course material!"
                )


            st.subheader(
                "Answer Review"
            )


            for i, question in enumerate(
                st.session_state.quiz
            ):

                user_answer = (
                    st.session_state.quiz_answers[i]
                )

                correct_answer = question["answer"]


                if user_answer == correct_answer:

                    st.success(
                        f"Question {i + 1}: Correct ✓"
                    )


                else:

                    st.error(
                        f"Question {i + 1}: "
                        f"Your answer: {user_answer} | "
                        f"Correct answer: {correct_answer}"
                    )


                    st.write(
                        f"**Correct answer:** "
                        f"{question['options'][correct_answer]}"
                    )