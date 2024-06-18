# Automated Question Answering Using FLAN-T5 with HuggingFace

This project focuses on generating answers to questions using the FLAN-T5 language model from HuggingFace. It reads questions from an input CSV file, generates answers using the model, and saves the results back into an output CSV file.

## Project Overview

- **Model Used:** FLAN-T5 from HuggingFace.
- **Task:** Automated question answering.
- **Input:** CSV file containing questions.
- **Output:** CSV file containing questions and generated answers.

## File Structure

- `main.py`: The main script for reading questions from a CSV file, generating answers using FLAN-T5, and saving the results to an output CSV file.

## Requirements

To run this project, you'll need the following dependencies:

- Python 3.7+
- Transformers (Hugging Face)
- Pandas
- Sentence Transformers
- FAISS
- NumPy

You can install the required packages using:

```bash
pip install transformers pandas sentence-transformers faiss-cpu numpy
