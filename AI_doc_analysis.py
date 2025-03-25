import openai
import pytesseract
from pdfminer.high_level import extract_text as extract_pdf_text
from docx import Document
from PIL import Image
import os

# Set your OpenAI API key here
openai.api_key = "REMOVED"

def extract_text_from_file(file_path):
    """Extracts text from PDF, DOCX, or image files."""
    file_extension = file_path.split('.')[-1].lower()

    if file_extension == "pdf":
        return extract_pdf_text(file_path)

    elif file_extension == "docx":
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])

    elif file_extension in ["png", "jpg", "jpeg"]:
        image = Image.open(file_path)
        return pytesseract.image_to_string(image)

    else:
        raise ValueError("Unsupported file format. Please upload a PDF, DOCX, or image.")

def generate_text(prompt):
    """Generates text using OpenAI's GPT model based on the provided prompt."""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000
    )
    return response["choices"][0]["message"]["content"].strip()

def solve_questions(content):
    """Extracts and solves questions step by step from the provided text."""
    
    prompt = f"""
    You are an expert AI tutor. Analyze the following text and:
    1. Identify and extract all questions, problems, or exercise statements
    2. For each identified item:
        - Provide a clear, step-by-step solution
        - Include relevant formulas, concepts, or theories where applicable
        - Explain the reasoning behind each step
        - Add examples or illustrations if helpful
        - Highlight key points and common pitfalls
    3. If the text contains:
        - Mathematical problems: Show complete calculations
        - Theoretical questions: Provide comprehensive explanations with examples
        - Multiple choice questions: Explain why the correct answer is right and others are wrong
        - Open-ended questions: Discuss multiple perspectives and approaches
        - Code-related questions: Include code examples and explanations

    Text to analyze:
    {content}
    
    Format your response with clear headers, bullet points, and numbered steps for better readability.
    """
    
    return generate_text(prompt)

def create_ai_step_by_step_solution(file_path):
    """Processes the uploaded file, extracts text, and generates a step-by-step solution report."""
    try:
        if not os.path.exists(file_path):
            print(f"Error: File '{file_path}' not found.")
            return

        print(f"Extracting text from '{file_path}'...\n")
        extracted_text = extract_text_from_file(file_path)

        if not extracted_text.strip():
            print("No text detected in the file. Please try a different document.")
            return

        print("\n--- Extracted Content ---\n")
        print(extracted_text[:1000] + "...")  # Show first 1000 characters for preview

        print("\n--- AI Step-by-Step Solutions ---\n")
        solutions = solve_questions(extracted_text)
        
        print(solutions)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Example usage
file_path = "C:\\Users\\hasan\\Downloads\\Ethiopia Education Questions Sample Document.pdf"  # Replace with actual file path
create_ai_step_by_step_solution(file_path)


