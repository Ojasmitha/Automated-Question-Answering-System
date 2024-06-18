import os
import pandas as pd
import argparse
from langchain.llms import HuggingFaceHub 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import evaluate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
# from evaluate import load as evaluate_load
from transformers import T5ForConditionalGeneration, T5Tokenizer

def generate_answers(input_file, output_file):
    # Set the Hugging Face Hub API token as an environment variable
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

    # Initialize the HuggingFace model with explicit temperature parameter
    model = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        task="text-generation",
        model_kwargs={"temperature": 1, "max_length": 256}
    )

    # Initialize the output parser
    output_parser = StrOutputParser()

    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_file)

    # Define the chat prompt template
    prompt = ChatPromptTemplate.from_template("Answer the following question: {question}")

    # Generate answers for each question
    answers = []
    for question in df["question"]:
        # Set up the chain with the question
        output = prompt | model | output_parser
        # Invoke the chain
        answer = output.invoke({"question": question})
        answers.append(answer)

    # Add answers to the DataFrame
    df["Answer"] = answers

    # Save the DataFrame with questions and answers to the output CSV file
    df.to_csv(output_file, index=False)

    print(f"Answers generated and saved to '{output_file}'.")

    val_questions_path = "./val_questions.csv"  # Adjust this path to the actual location of your val_questions.csv

    # Initialize the HuggingFace model
    model = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        task="text-generation",
        model_kwargs={"temperature": 0.7, "max_length": 256}
    )

    # Initialize the output parser
    output_parser = StrOutputParser()

    # Initialize the SQuAD metric pipeline
    squad_metric = evaluate.load("squad_v2")

    # Define the score function
    def score(preds, refs):
        preds = [{'id': str(idx), 'prediction_text': ans, 'no_answer_probability': 0.0} for idx, ans in enumerate(preds)]
        refs = [{'id': str(idx), 'answers': {'answer_start': [0] * len(ans), 'text': ans}} for idx, ans in enumerate(refs)]
        results = squad_metric.compute(predictions=preds, references=refs)
        return dict(f1=results['f1'], exact_match=results['exact'], total=results['total'])

    # Read the CSV file into a DataFrame for validation
    df_val = pd.read_csv(val_questions_path)
    prompt = ChatPromptTemplate.from_template("Answer the following question: {question}")

    # Generate answers for each question in the validation dataset
    predicted_answers = []
    for question in df_val["question"]:
        output = prompt | model | output_parser
        answer = output.invoke({"question": question})
        predicted_answers.append(answer.strip())

    # Calculate metrics using the scoring function
    metrics = score(preds=predicted_answers, refs=df_val["answer"].tolist())

    # Print the calculated metrics
    print("Validation Metrics:")
    print(metrics)

    val_df = pd.read_csv(val_questions_path)
    val_answers = []

    for question in val_df["question"]:
        output = prompt | model | output_parser
        answer = output.invoke({"question": question})
        val_answers.append(answer.strip())

    # Preprocess ground truth answers to handle possible lists
   

    val_df["Processed Answer"] = val_df["answer"].apply(lambda x: eval(x)[0].lower() if isinstance(x, str) and x.startswith("[") else x.lower())
    val_df["Processed Generated Answer"] = val_df["Generated Answer"].apply(lambda x: x.lower())

# Calculate accuracy
    correct_predictions = (val_df["Processed Answer"] == val_df["Processed Generated Answer"]).sum()
    total_predictions = len(val_df)
    accuracy = correct_predictions / total_predictions * 100

    print(f"Accuracy (ignoring case sensitivity): {accuracy:.2f}%")

    
def generate_langchain_rag_answers(input_file, passages_file, output_file):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
    
    if args.rag and args.langchain:
        # RAG and LangChain logic
        model = HuggingFaceHub(repo_id="google/flan-t5-base", task="text-generation", model_kwargs={"temperature": 0.7, "max_length": 256})
        output_parser = StrOutputParser()
        passages_df = pd.read_csv(args.passages)
        passages = passages_df["context"].tolist()
        
        # Define Document class for each passage
        class Document:
            def __init__(self, content, metadata=None):
                self.page_content = content
                self.metadata = metadata if metadata is not None else {}
        
        # Create Document instances for each passage
        documents = [Document(passage) for passage in passages]
        
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        document_chunks = text_splitter.split_documents(documents)
        
        embeddings = HuggingFaceEmbeddings()
        template = "Answer the following question given the context \n question: {question} \n context: {context}"
        
        prompt = ChatPromptTemplate.from_template(template)
        db = FAISS.from_documents(document_chunks, embedding=embeddings)
        
        df = pd.read_csv(args.questions)
        answers = []
        top_docs = []  # Store top retrieved documents
        
        for question in df["question"]:
            similar_documents = db.similarity_search(question, k=3)  # Retrieve top 3 similar documents
            relevant_passages = [doc.page_content for doc in similar_documents]
            top_docs.append(relevant_passages)
            
            relevant_passage = similar_documents[0].page_content if similar_documents else "Passage not found"
            input_text = f"{relevant_passage} {question}"
            output = model.invoke(input_text)
            answer = output_parser.parse(output)
            answers.append(answer)
        
        df["Generated Answer"] = answers
        df["Top 3 Retrieved Documents"] = top_docs
        
        output_file = args.output
        df.to_csv(output_file, index=False)
        print(f"Answers generated and saved to '{output_file}'.")

    val_questions_path = "./val_questions.csv"  
    

    # Initialize the SQuAD metric pipeline
    val_df = pd.read_csv(val_questions_path)
    val_answers = []
    
    for question in val_df["question"]:
        similar_documents = db.similarity_search(question)
        relevant_passage = similar_documents[0].page_content if similar_documents else "Passage not found"
        input_text = f"{relevant_passage} {question}"
        output = model.invoke(input_text)
        answer = output_parser.parse(output)
        val_answers.append(answer)
    
    val_df["Generated Answer"] = val_answers
    # print(val_df.head())

    squad_metric = evaluate.load("squad_v2")
    
    # Define the score function
    predicted_answers = [ans.lower() for ans in val_df["Generated Answer"].tolist()]
    correct_answers = [ans.lower() for ans in val_df["answer"].tolist()]

    
    # Adjust the score function if needed to handle a single correct answer instead of a list
    def score(preds, refs):
        preds = [{'id': str(idx), 'prediction_text': ans} for idx, ans in enumerate(preds)]
        refs = [{'id': str(idx), 'answers': {'answer_start': [0], 'text': [ans]}} for idx, ans in enumerate(refs)]
        results = squad_metric.compute(predictions=preds, references=refs)
        return dict(f1=results['f1'], exact_match=results['exact_match'], total=len(refs))
    def score(preds, refs):
    # Adjust preds to include 'no_answer_probability' if it's missing
     preds_with_no_answer = [
        {
            'id': str(idx),
            'prediction_text': ans,
            'no_answer_probability': 0.0  # Assign a default value
        } 
        for idx, ans in enumerate(preds)
    ]
    
     refs_formatted = [
        {
            'id': str(idx), 
            'answers': {
                'answer_start': [0] * len(ans), 
                'text': ans
            }
        } 
        for idx, ans in enumerate(refs)
    ]
    
    # Compute the metric with adjusted predictions
     results = squad_metric.compute(predictions=preds_with_no_answer, references=refs_formatted)
     return dict(f1=results['f1'], exact_match=results['exact'], total=len(refs))
    # Calculate metrics
    metrics = score(preds=predicted_answers, refs=correct_answers)
    
    # Print the calculated metrics
    print("Validation Metrics:")
    print(metrics)
    val_df["Processed Answer"] = val_df["answer"].apply(lambda x: eval(x)[0].lower() if isinstance(x, str) and x.startswith("[") else x.lower())
    val_df["Processed Generated Answer"] = val_df["Generated Answer"].apply(lambda x: x.lower())

# Calculate accuracy
    correct_predictions = (val_df["Processed Answer"] == val_df["Processed Generated Answer"]).sum()
    total_predictions = len(val_df)
    accuracy = correct_predictions / total_predictions * 100
    print(f"Accuracy (ignoring case sensitivity): {accuracy:.2f}%")


def generate_rag_answers(questions_file, passages_file, output_file):
    # Load documents from passages.csv
    docs_df = pd.read_csv(passages_file)
    documents = docs_df['context'].tolist()

    # Initialize the SentenceTransformer model for embeddings
    model_name = 'sentence-transformers/roberta-base-nli-stsb-mean-tokens'
    sentence_model = SentenceTransformer(model_name)

    # Compute embeddings for the documents
    doc_embeddings = sentence_model.encode(documents, convert_to_tensor=False)
    doc_embeddings_np = np.array(doc_embeddings)

    # Initialize a FAISS index for similarity search
    d = doc_embeddings_np.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(doc_embeddings_np)

    # Load Hugging Face model and tokenizer for generating answers
    hf_model_name = 'google/flan-t5-base'
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    hf_model = AutoModelForSeq2SeqLM.from_pretrained(hf_model_name)

    # Load questions
    questions_df = pd.read_csv(questions_file)
    answers = []

    def retrieve_documents(query, index, documents, sentence_model, k=5):
        query_embedding = sentence_model.encode([query], convert_to_tensor=False)
        D, I = index.search(np.array(query_embedding), k)
        retrieved_docs = [documents[i] for i in I[0]]
        return retrieved_docs

    def generate_answer(question, retrieved_docs):
        context = " ".join(retrieved_docs)
        input_text = f"question: {question} context: {context}"
        input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
        answer_ids = hf_model.generate(input_ids, max_length=100, num_beams=5, early_stopping=True)
        answer = tokenizer.decode(answer_ids[0], skip_special_tokens=True)
        return answer

    # Store top 3 retrieved documents for each question
    top_retrieved_docs = []

    for question in questions_df["question"]:
        retrieved_docs = retrieve_documents(question, index, documents, sentence_model, k=5)[:3]  # Limit to top 3
        top_retrieved_docs.append(retrieved_docs)
        answer = generate_answer(question, retrieved_docs)
        answers.append(answer)

    # Save answers
    questions_df["Generated Answer"] = answers
    questions_df["Top 3 Retrieved Documents"] = top_retrieved_docs
    questions_df.to_csv(output_file, index=False)
    print(f"Answers generated and saved to '{output_file}'.")

    # Validation
    val_questions_path = "./val_questions.csv"  
    val_df = pd.read_csv(val_questions_path)
    val_answers = []

    for question in val_df["question"]:
        retrieved_docs = retrieve_documents(question, index, documents, sentence_model, k=5)[:3]  # Limit to top 3
        answer = generate_answer(question, retrieved_docs)
        val_answers.append(answer)

    # Add generated answers to the validation dataframe
    val_df["Generated Answer"] = val_answers
    val_df["correct_answer"] = val_df["answer"].apply(lambda x: eval(x)[0].lower() if isinstance(x, str) and x.startswith("[") else x.lower())
    val_df["Generated Answer"] = val_df["Generated Answer"].apply(lambda x: x.lower())

    # Calculate accuracy
    correct_predictions = (val_df["correct_answer"] == val_df["Generated Answer"]).sum()
    total_predictions = len(val_df)
    accuracy = correct_predictions / total_predictions * 100

    # Print accuracy
    print(f"Accuracy (ignoring case sensitivity): {accuracy:.2f}%")



def generate_cosine_rag_answers(questions_file, passages_file, output_file):
    # Load documents from passages.csv
    docs_df = pd.read_csv(passages_file)
    documents = docs_df['context'].tolist()

    # Initialize the SentenceTransformer model for embeddings
    model_name = 'sentence-transformers/roberta-base-nli-stsb-mean-tokens'
    sentence_model = SentenceTransformer(model_name)

    # Compute embeddings for the documents and normalize them
    doc_embeddings = sentence_model.encode(documents, convert_to_tensor=False)
    doc_embeddings_np = np.array(doc_embeddings)
    faiss.normalize_L2(doc_embeddings_np)

    # Create a FAISS index for cosine similarity
    index_cosine = faiss.IndexFlatIP(doc_embeddings_np.shape[1])
    index_cosine.add(doc_embeddings_np)

    # Load Hugging Face model and tokenizer
    hf_model_name = 'google/flan-t5-base'
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    hf_model = AutoModelForSeq2SeqLM.from_pretrained(hf_model_name)

    # Function to retrieve documents based on a query using cosine similarity
    def retrieve_documents_cosine(query, index, documents, model, k=5):
        query_embedding = model.encode([query], convert_to_tensor=False).reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        D, I = index.search(query_embedding, k)
        return [documents[i] for i in I[0]][:3]  # Limit to top 3 retrieved documents

    # Function to generate an answer from a question and retrieved documents
    def generate_answer(question, retrieved_docs):
        context = " ".join(retrieved_docs)
        input_text = f"question: {question} context: {context}"
        input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
        answer_ids = hf_model.generate(input_ids, max_length=100, num_beams=5, early_stopping=True)
        return tokenizer.decode(answer_ids[0], skip_special_tokens=True)

    # Load questions and generate answers
    questions_df = pd.read_csv(questions_file)
    answers = []
    retrieved_docs_list = []

    for question in questions_df["question"]:
        retrieved_docs = retrieve_documents_cosine(question, index_cosine, documents, sentence_model, k=5)
        retrieved_docs_list.append(", ".join(retrieved_docs))
        answer = generate_answer(question, retrieved_docs)
        answers.append(answer)

    # Save answers and retrieved documents to CSV
    questions_df["Generated Answer"] = answers
    questions_df["Top 3 Retrieved Documents"] = retrieved_docs_list
    questions_df.to_csv(output_file, index=False)
    print(f"Answers generated and saved to '{output_file}'.")

    # Validation
    val_questions_path = "./val_questions.csv"
    val_df = pd.read_csv(val_questions_path)
    val_answers = []

    for question in val_df["question"]:
        retrieved_docs = retrieve_documents_cosine(question, index_cosine, documents, sentence_model, k=5)
        answer = generate_answer(question, retrieved_docs)
        val_answers.append(answer)

    # Add generated answers to the validation dataframe
    val_df["Generated Answer"] = val_answers

    # Assuming the correct answers are in a column named 'correct_answer' in your validation set
    # Convert both correct and generated answers to lowercase for case-insensitive comparison
    val_df["correct_answer"] = val_df["answer"].apply(lambda x: eval(x)[0].lower() if isinstance(x, str) and x.startswith("[") else x.lower())
    val_df["Generated Answer"] = val_df["Generated Answer"].apply(lambda x: x.lower())

    # Calculate accuracy
    correct_predictions = (val_df["correct_answer"] == val_df["Generated Answer"]).sum()
    total_predictions = len(val_df)
    accuracy = correct_predictions / total_predictions * 100

    # Print accuracy
    print(f"Accuracy (ignoring case sensitivity): {accuracy:.2f}%")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate answers for questions in a CSV file.")
    parser.add_argument("--questions", type=str, required=True, help="Input CSV file with questions.")
    parser.add_argument("--output", type=str, required=True, help="Output CSV file for the predictions.")
    parser.add_argument("--rag", action="store_true", help="Enable RAG predictions")
    parser.add_argument("--langchain", action="store_true", help="Enable LangChain for context retrieval")
    parser.add_argument("--cosine_rag", action="store_true", help="Use cosine similarity for RAG predictions")
    parser.add_argument("--passages", type=str, help="Input CSV file with passages for context retrieval")

    args = parser.parse_args()

    # Adjusted branching logic
    if args.rag and not args.langchain:
        # Only RAG argument provided; ensure passages argument is provided for this branch
        if not args.passages:
            raise ValueError("--passages is required when using --rag without --langchain")
        if args.cosine_rag:
            generate_cosine_rag_answers(args.questions, args.passages, args.output)
        else:
            generate_rag_answers(args.questions, args.passages, args.output)
    elif args.rag and args.langchain:
        # Both RAG and LangChain arguments provided; ensure passages argument is also provided
        if not args.passages:
            raise ValueError("--passages is required when using --rag and --langchain together")
        else:
            generate_langchain_rag_answers(args.questions, args.passages, args.output)
    else:
        # Default to generating answers without RAG or LangChain
        generate_answers(args.questions, args.output)


    