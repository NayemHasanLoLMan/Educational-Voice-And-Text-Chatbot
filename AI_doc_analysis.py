import openai
import pytesseract
from pdfminer.high_level import extract_text as extract_pdf_text
from docx import Document
from PIL import Image
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

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
        max_tokens=5000
    )
    return response.choices[0].message.content.strip()

def solve_questions(content, user_input=None):
    """Analyzes content and provides comprehensive solutions or insights based on user input."""
    prompt = f"""
    You are an expert AI tutor and analyst. Analyze the provided text and follow any instructions or context provided in the user input to deliver a comprehensive response. Your goal is to understand the content, identify key elements, and provide solutions, explanations, or insights as directed.

    ### Content to Analyze:
    {content}

    ### User Input:
    {user_input if user_input else 'No user input provided. Analyze the content, identify any questions, problems, or key topics, and provide comprehensive solutions or explanations.'}

    ### Analysis Guidelines:
    1. **Content Understanding**:
       - Summarize the main topics, themes, or purpose of the document.
       - Identify the type of document (e.g., educational, technical, narrative, question-based, etc.).
       - Highlight any questions, problems, exercises, or key concepts present.

    2. **Response Requirements**:
       - If the user input contains instructions, follow them explicitly (e.g., solve problems, explain concepts, analyze issues, etc.).
       - If the user input includes additional questions or content, incorporate them into the analysis.
       - For questions or problems (from the document or user input):
         - Extract and list all questions, exercises, or problems.
         - Provide step-by-step solutions or explanations.
         - Include relevant formulas, theories, or concepts with clear explanations.
         - Use examples or analogies relevant to the context (e.g., Ethiopian culture for educational content).
         - Highlight key points and common pitfalls.
       - For general analysis (if no questions or specific instructions):
         - Discuss key insights, implications, or applications of the content.
         - Suggest improvements, alternative approaches, or further exploration if relevant.
       - If user input specifies a method or focus (e.g., 'solve using a specific formula' or 'focus on cultural relevance'), prioritize that approach.

    3. **Content-Specific Handling**:
       - **Mathematical Problems**: Show complete calculations, explain each step, and include relevant formulas.
       - **Theoretical Questions**: Provide detailed explanations with examples and context.
       - **Multiple Choice Questions**: Identify the correct answer and explain why other options are incorrect.
       - **Open-Ended Questions**: Explore multiple perspectives, provide balanced arguments, and support with examples.
       - **Code-Related Content**: Include code snippets, explain functionality, and highlight best practices.
       - **Non-Question Content**: Analyze the document's purpose, structure, and key points; provide insights or recommendations.

    4. **Formatting**:
       - Use clear headers (e.g., 'Document Summary', 'Identified Questions', 'Solutions', 'Key Insights').
       - Use bullet points, numbered steps, or tables for clarity.
       - Ensure readability for a broad audience, including young learners if the context suggests it (e.g., Ethiopian students in grades 1-12).
       - Include culturally relevant examples or context where applicable.

    ### Deliverables:
    - A comprehensive response addressing the user input and content analysis.
    - Clear, concise, and well-structured explanations or solutions.
    - Practical examples or illustrations to enhance understanding.
    - Suggestions for further learning or application if relevant.

    Format your response in Markdown with clear headers, bullet points, and numbered steps for better readability.
    """
    
    return generate_text(prompt)

def create_ai_step_by_step_solution(file_path=None, user_input=None):
    """Processes the uploaded file and/or user input, extracts text, and generates a comprehensive solution or analysis."""
    try:
        combined_content = ""

        # Extract text from file if provided
        if file_path:
            if not os.path.exists(file_path):
                print(f"Error: File '{file_path}' not found.")
                return
            print(f"Extracting text from '{file_path}'...\n")
            file_content = extract_text_from_file(file_path)
            if file_content.strip():
                combined_content += f"--- Content from File ---\n{file_content}\n"
            else:
                print("No text detected in the file. Checking user input...")

        # Append user input if provided
        if user_input:
            if user_input.strip():
                combined_content += f"--- User Input ---\n{user_input}\n"
            else:
                print("No valid user input provided.")

        # Check if there's any content to process
        if not combined_content.strip():
            print("No valid content detected from file or user input. Please provide a file or input.")
            return

        print("\n--- Extracted Content Preview ---\n")
        print(combined_content[:1000] + "...")  # Show first 1000 characters for preview

        print("\n--- AI Comprehensive Analysis and Solutions ---\n")
        solutions = solve_questions(combined_content, user_input)
        
        print(solutions)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Example usage
if __name__ == "__main__":
    file_path = "C:\\Users\\hasan\\Downloads\\Ethiopia Education Questions Sample Document.pdf"  # Replace with actual file path
    user_input = """
    Questions:
    1. What is the capital city of Ethiopia?
    2. Solve for x: 2x + 5 = 15
    
    Instructions:
    Analyze the provided content and solve any questions or problems. For mathematical problems, use step-by-step algebraic methods. For theoretical questions, include examples relevant to Ethiopian culture. If no questions are present in the file, summarize the document and suggest educational activities for Ethiopian students in grades 1-12.
    """
    create_ai_step_by_step_solution(file_path=file_path, user_input=user_input)