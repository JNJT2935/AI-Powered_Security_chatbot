import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


# Load environment variables BEFORE importing modules that use the API key
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)


from llm_answering import answer_question
from quiz_generation import generate_quiz


# Page configuration
st.set_page_config(
    page_title="AI Security Chatbot",
    page_icon="🛡️",
    layout="centered"
)


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("🛡️ AI Security Learning")

page = st.sidebar.radio(
    "Choose a feature:",
    [
        "💬 Chatbot",
        "📝 Quiz Generator"
    ]
)


# ============================================================
# CHATBOT
# ============================================================

if page == "💬 Chatbot":

    st.title("🛡️ AI-Powered Security Chatbot")
    st.caption(
        "Ask questions about cybersecurity based on the course material."
    )


    # Initialise chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []


    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    # Chat input
    if prompt := st.chat_input("Ask a cybersecurity question..."):

        # Display user message
        st.chat_message("user").markdown(prompt)

        # Save user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })


        # Generate response
        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                try:

                    result = answer_question(
                        prompt,
                        history=st.session_state.messages[:-1],
                        top_k=7
                    )

                    # Only display the actual answer
                    bot_response = result["answer"]

                    st.markdown(bot_response)


                    # Save only the answer to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": bot_response
                    })


                except Exception as e:

                    st.error(f"Error: {e}")


# ============================================================
# QUIZ GENERATOR
# ============================================================

elif page == "📝 Quiz Generator":

    st.title("📝 Cybersecurity Quiz Generator")

    st.caption(
        "Generate multiple-choice questions based on your cybersecurity course material."
    )


    # Topic input
    topic = st.text_input(
        "Enter a cybersecurity topic",
        placeholder="e.g. Network Security, Phishing, Malware"
    )


    # Number of questions
    num_questions = st.number_input(
        "Number of questions",
        min_value=1,
        max_value=10,
        value=3,
        step=1
    )


    # Generate quiz button
    if st.button("Generate Quiz", type="primary"):

        if not topic.strip():

            st.warning("Please enter a cybersecurity topic.")

        else:

            with st.spinner("Generating your quiz..."):

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

                        # Store generated quiz
                        st.session_state.quiz = quiz

                        # Reset previous answers
                        st.session_state.quiz_submitted = False

                        # Reset answer storage
                        st.session_state.quiz_answers = {}


                except Exception as e:

                    st.error(f"Error generating quiz: {e}")


    # DISPLAYING GENERATED QUIZ
    
    if "quiz" in st.session_state:

        st.divider()

        st.subheader("Your Quiz")


        # Store user's answers
        answers = {}


        for i, question in enumerate(st.session_state.quiz):

            st.markdown(
                f"### Question {i + 1}"
            )

            st.write(question["question"])


            # Answer options
            selected_answer = st.radio(
                "Choose your answer:",
                options=list(question["options"].keys()),
                format_func=lambda x, q=question: (
                    f"{x}. {q['options'][x]}"
                ),
                key=f"quiz_question_{i}"
            )


            answers[i] = selected_answer


        st.divider()


        # Submit quiz
        if st.button("Submit Quiz", type="primary"):

            score = 0

            for i, question in enumerate(st.session_state.quiz):

                if answers[i] == question["answer"]:

                    score += 1


            total = len(st.session_state.quiz)

            percentage = (score / total) * 100


            # Save results
            st.session_state.quiz_answers = answers
            st.session_state.quiz_score = score
            st.session_state.quiz_submitted = True



        # DISPLAYING SCORE

        if st.session_state.get("quiz_submitted", False):

            score = st.session_state.quiz_score
            total = len(st.session_state.quiz)
            percentage = (score / total) * 100


            st.subheader("Quiz Results")

            if percentage >= 80:

                st.success(
                    f"🎉 Excellent! You scored {score}/{total} "
                    f"({percentage:.0f}%)."
                )

            elif percentage >= 50:

                st.info(
                    f"👍 Good job! You scored {score}/{total} "
                    f"({percentage:.0f}%)."
                )

            else:

                st.warning(
                    f"You scored {score}/{total} "
                    f"({percentage:.0f}%). Keep reviewing the course material!"
                )


            # SHOW CORRECT ANSWERS

            st.subheader("Answer Review")


            for i, question in enumerate(st.session_state.quiz):

                user_answer = st.session_state.quiz_answers[i]
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