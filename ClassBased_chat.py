import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_educational_article(language: str, keywords: list, title: str, desctription: str) -> str:
    """
    Generate an educational article for Ethiopian students (Grades 1-12) using OpenAI API.
    
    Args:
        language (str): Primary language of the article (e.g., 'English', 'Amharic')
        keywords (list): List of keywords to include in the article
        title (str): Title of the article
    
    Returns:
        str: Formatted article in Markdown
    """
    # Construct prompt for educational content
    prompt = f"""
    You are an expert educational content creator tasked with writing a **high-quality, age-appropriate educational article** for Ethiopian students from grades 1 to 12.

    ## 🎯 Article Purpose:
    To educate, inspire, and engage Ethiopian students on a specific topic using clear, simple, and culturally-relevant explanations, with interactive elements to enhance understanding and encourage classroom or self-discussion.

    ## 📝 Article Metadata:
    - **Target Language:** {language}
    - **Title:** {title}
    - **Brief Description:** {desctription}
    - **Focus Keywords:** {', '.join(keywords)}

    ## ✅ Content Expectations:
    - Must **strictly align** with the given title and description.
    - Use **simple, age-appropriate** language understandable by school students.
    - Incorporate **examples, analogies, and scenarios** relevant to Ethiopian life, geography, history, or culture.
    - Maintain a friendly, motivating tone that speaks directly to young learners.
    - Use **Markdown syntax** throughout for formatting consistency.
    - **Word count:** 300–500 words only.

    ## 📚 Required Article Structure:
    1. # {title}  
    2. ## Introduction  
    - A brief and engaging overview of the topic.  
    3. ## Main Content  
    - Break down the topic into 5–6 clear sections using level 2 headers.  
    - Use **bullet points**, **numbered lists**, or **examples** where helpful.  
    - Highlight important terms with **bold** or *italic* text.  
    - Use culturally relevant references (e.g., Ethiopian holidays, rural/urban settings, local foods, etc.)
    4. ## Conclusion  
    - A informative well struvture summary of the main points.  
    - Encourage curiosity and motivate students to learn more.
    5. ## Think & Discuss  
    - Pose **2 thoughtful questions** to help students reflect and discuss.  
    - Questions should encourage critical thinking and real-world connections.

    ## 📐 Formatting Rules:
    - Use markdown headers: `#`, `##`, etc.
    - Use **bold**, *italic*, `code` formatting where necessary.
    - Add horizontal rules (`---`) to separate main sections.
    - Avoid overly complex language or long paragraphs.

    ## 🎓 Tips:
    - Keep sentences meaningful and clear.
    - Always explain difficult words or concepts with examples.
    - Make the content feel like a helpful guide or story, not a textbook.

    Now write the full educational article using these rules and data.
    """
    
    try:
        # Call OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert educational content creator tasked with writing a **high-quality, age-appropriate educational article** for Ethiopian students from grades 1 to 12."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=5000,
            temperature=0.7
        )
        
        # Extract and return the generated article
        article = response.choices[0].message.content
        return article
        
    except Exception as e:
        return f"Error generating article: {str(e)}"

# Example usage
if __name__ == "__main__":
    language = "English"
    keywords = ["photosynthesis", "plants", "sunlight", "Ethiopian farming"]
    title = "How Plants Grow: Understanding Photosynthesis"
    desctription = "An educational article for Ethiopian students about the process of photosynthesis and its importance in agriculture."
    
    article = generate_educational_article(language, keywords, title, desctription)
    print(article)