import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class ArticleGenerator:
    def __init__(self, api_key: str = None, language: str = "english"):
        """
        Initialize the DocumentAnalyzer with API key and language settings.
        
        Args:
            api_key (str, optional): OpenAI API key. If None, loads from environment.
            language (str): Response language ('english', 'amharic', 'oromo')
        """
        # Set up OpenAI API key
        if api_key:
            openai.api_key = api_key
        else:
            api_key_from_env = os.getenv("OPENAI_API_KEY")
            if not api_key_from_env:
                raise ValueError("OpenAI API key not found. Please provide it or set OPENAI_API_KEY environment variable.")
            openai.api_key = api_key_from_env
        
        # Set language
        self.language = language.lower()
        supported_languages = {"english", "amharic", "oromo"}
        if self.language not in supported_languages:
            print(f"Warning: Language '{language}' not supported. Defaulting to 'english'.")
            self.language = "english"



    def generate_educational_article(self, keywords: list, title: str, desctription: str) -> str:


        if self.language == "oromo":
            language_instruction = "Please provide your entire response in Oromo language only."
        elif self.language == "amharic":
            language_instruction = "Please provide your entire response in Amharic language only."
        else:
            language_instruction = "Please provide your entire response in English language only."

        # Construct prompt for educational content
        prompt = f"""
        You are an expert educational content creator tasked with writing a **high-quality, age-appropriate educational article** for Ethiopian students from grades 1 to 12.

        ## 🎯 Article Purpose:
        To educate, inspire, and engage Ethiopian students on a specific topic using clear, simple, and culturally-relevant explanations, with interactive elements to enhance understanding and encourage classroom or self-discussion.

        ## 📝 Article Metadata:
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
        
        system_prompt = language_instruction + "\n\n" + prompt

        try:
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert educational content creator tasked with writing a **high-quality, age-appropriate educational article** for Ethiopian students from grades 1 to 12."},
                    {"role": "user", "content": system_prompt}
                ],
                max_tokens=5000,
                temperature=0.7
            )
            
            # Extract and return the generated article
            article = response.choices[0].message.content
            return article
            
        except Exception as e:
            return f"Error generating article: {str(e)}"
        



def main (language: str , keywords: list, title: str, desctription: str) :
    
    
    try:
        praiser = ArticleGenerator(language=language)
        result = praiser.generate_educational_article(keywords, title, desctription)
        
        if result:
            print("\n" + "="*50)
            print(f"Article Report ({language.upper()})")
            print("="*50)
            print(result)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

# Example usage
if __name__ == "__main__":
    language = "amharic"
    keywords = ["photosynthesis", "plants", "sunlight", "Ethiopian farming"]
    title = "How Plants Grow: Understanding Photosynthesis"
    desctription = "An educational article for Ethiopian students about the process of photosynthesis and its importance in agriculture."
    
    article = main(language, keywords, title, desctription)
    print(article)