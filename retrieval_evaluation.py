from retrieval import retrieve


# ============================================================
# EVALUATION QUESTIONS
# ============================================================

TEST_QUESTIONS = [
    {
        "question": "What is the formal definition of auditing according to professional standards?",
        "expected_chunks": ["au001"]
    },

    {
        "question": "What is the relationship between inherent risk, control risk, and residual risk?",
        "expected_chunks": ["au451", "au452", "au453"]
    },

    {
        "question": "How do the statutory penalties for unauthorized access to computer data differ from cyberterrorism under the Cybersecurity and Cybercrime Act 2021?",
        "expected_chunks": ["cy156", "cy1171","cy470","cy496"]
    },

    {
        "question": "What are Cialdini's original six core principles of persuasion, and what seventh principle was introduced in 2016?",
        "expected_chunks": ["hu850","hu852"]
    },

    {
        "question": "What three core principles guide Zero Trust Architecture (ZTA) across enterprise environments, and how does micro-segmentation enforce Zero Trust at the network level?",
        "expected_chunks": ["se824","hu194","in080"]
    },

    {
        "question": "How is the CIA Triad defined in security fundamentals, how does an IS audit verify its controls, and what specific technical controls support each element in databases?",
        "expected_chunks": ["se518", "au053", "au054", "au073"]
    },

    {
        "question": "How is a digital signature defined legally under the Electronic Transactions Act (ETA 2000), how does it technically operate using asymmetric cryptography and hashing, and how is it implemented in S/MIME email security?",
        "expected_chunks": ["cy1108","se445","ne369"]
    },

    {
        "question": "How do Discretionary Access Control (DAC) and Mandatory Access Control (MAC) differ in how permissions are assigned?",
        "expected_chunks": ["se525","se528"]
    },

    {
        "question": "How do Spear Phishing and Whaling differ from conventional phishing, what psychological principles drive social engineering attacks, and how do SETA programs mitigate human error?",
        "expected_chunks": ["hu883","hu886","hu914","hu923","hu913","hu1233"]
    },

    {
        "question": "What four phases comprise the NIST SP 800-86 digital forensic model, how does the order of evidence volatility dictate collection priorities, and why is the Chain of Custody critical for legal admissibility?",
        "expected_chunks": ["di054", "cy563", "cy567", "di296"]
    }
]


# ============================================================
# K VALUES TO TEST
# ============================================================

K_VALUES = [1, 3, 5, 7, 10]


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print("\n" + "=" * 80)
    print("RETRIEVAL K-VALUE EVALUATION")
    print("=" * 80)

    print(f"\nTesting K values: {K_VALUES}")
    print(f"Number of questions: {len(TEST_QUESTIONS)}")


    # Store overall results
    evaluation_results = {}


    # ========================================================
    # TEST EACH K
    # ========================================================

    for k in K_VALUES:

        print("\n" + "=" * 80)
        print(f"K = {k}")
        print("=" * 80)

        total_relevant_retrieved = 0
        total_relevant = 0

        total_retrieved = 0

        hit_count = 0


        # ----------------------------------------------------
        # Run every question
        # ----------------------------------------------------

        for question_number, test_case in enumerate(
            TEST_QUESTIONS,
            start=1
        ):

            question = test_case["question"]
            expected_chunks = set(
                test_case["expected_chunks"]
            )

            results = retrieve(
                query=question,
                top_k=k
            )

            retrieved_ids = [
                result["chunk_id"]
                for result in results
            ]

            # Find relevant chunks that were retrieved
            relevant_retrieved = (
                expected_chunks.intersection(
                    retrieved_ids
                )
            )

            # -----------------------------------------------
            # Metrics for this question
            # -----------------------------------------------

            number_relevant_retrieved = len(
                relevant_retrieved
            )

            number_relevant = len(
                expected_chunks
            )

            question_recall = (
                number_relevant_retrieved /
                number_relevant
            )

            question_precision = (
                number_relevant_retrieved /
                len(retrieved_ids)
                if retrieved_ids
                else 0
            )

            question_hit = (
                1
                if number_relevant_retrieved > 0
                else 0
            )


            # -----------------------------------------------
            # Add to totals
            # -----------------------------------------------

            total_relevant_retrieved += (
                number_relevant_retrieved
            )

            total_relevant += number_relevant

            total_retrieved += len(retrieved_ids)

            hit_count += question_hit


            # -----------------------------------------------
            # Display
            # -----------------------------------------------

            print(
                f"\nQ{question_number}: "
                f"{question}"
            )

            print(
                f"Expected:  "
                f"{list(expected_chunks)}"
            )

            print(
                f"Retrieved: "
                f"{retrieved_ids}"
            )

            print(
                f"Relevant retrieved: "
                f"{number_relevant_retrieved}"
            )

            print(
                f"Recall@{k}: "
                f"{question_recall:.2%}"
            )

            print(
                f"Precision@{k}: "
                f"{question_precision:.2%}"
            )

            print(
                f"Hit@{k}: "
                f"{'PASS' if question_hit else 'FAIL'}"
            )


        # ====================================================
        # OVERALL METRICS FOR THIS K
        # ====================================================

        recall_at_k = (
            total_relevant_retrieved /
            total_relevant
            if total_relevant
            else 0
        )

        precision_at_k = (
            total_relevant_retrieved /
            total_retrieved
            if total_retrieved
            else 0
        )

        hit_at_k = (
            hit_count /
            len(TEST_QUESTIONS)
        )


        evaluation_results[k] = {
            "recall": recall_at_k,
            "precision": precision_at_k,
            "hit": hit_at_k
        }


        print("\n" + "-" * 80)
        print(f"SUMMARY FOR K = {k}")
        print("-" * 80)

        print(
            f"Recall@{k}:    "
            f"{recall_at_k:.2%}"
        )

        print(
            f"Precision@{k}: "
            f"{precision_at_k:.2%}"
        )

        print(
            f"Hit@{k}:       "
            f"{hit_at_k:.2%}"
        )


    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    print("\n\n" + "=" * 80)
    print("FINAL K COMPARISON")
    print("=" * 80)

    print(
        f"{'K':<8}"
        f"{'Recall@K':<15}"
        f"{'Precision@K':<15}"
        f"{'Hit@K':<15}"
    )

    print("-" * 53)

    for k in K_VALUES:

        results = evaluation_results[k]

        print(
            f"{k:<8}"
            f"{results['recall']:<15.2%}"
            f"{results['precision']:<15.2%}"
            f"{results['hit']:<15.2%}"
        )

    print("=" * 80)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()