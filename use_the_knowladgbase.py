import os
import openai
import numpy as np
from typing import Dict, Optional
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

class PineconeRAGQuery:
    def __init__(self, openai_api_key: str, pinecone_api_key: str, pinecone_index_name: str, language: str = "english"):
        openai.api_key = openai_api_key
        self.pc = Pinecone(api_key=pinecone_api_key)
        self.index = self.pc.Index(pinecone_index_name)
        self.embedding_model = "text-embedding-ada-002"

        #set language
        self.language = language.lower()
        supported_languages = {"english", "amharic", "oromo"}
        if self.language not in supported_languages:
            print(f"Warning: Language '{language}' not supported. Defaulting to 'english'.")
            self.language = "english"

        print(f"✅ Connected to Pinecone index: {pinecone_index_name}")

    def _get_query_embedding(self, query: str) -> np.ndarray:
        response = openai.Embedding.create(
            model=self.embedding_model,
            input=query
        )
        return np.array(response["data"][0]["embedding"])

    def _query_pinecone(self, query_embedding: np.ndarray, standerd_subject: str, grade: str, top_k: int = 7):
        """Query Pinecone for relevant pages filtered by metadata, with more results to aggregate."""

        filter_dict = {
            "standerd_subject": standerd_subject,
            "grade": grade
        }

        print(f"🔍 Querying Pinecone with filter: {filter_dict} and top_k={top_k}")

        results = self.index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict
        )
        return results.get("matches", [])

    def extract_text(self, query: str, standerd_subject: str, grade: str, conversation_history = None) -> Dict[str, Optional[str]]:
        try:
            query_embedding = self._get_query_embedding(query)
            matches = self._query_pinecone(query_embedding, standerd_subject, grade)

            if not matches:
                return {'error': 'No relevant matches found'}

            # Aggregate texts from multiple matches intelligently (limit length)
            combined_text = ""
            max_context_length = 3000  # tokens or chars, adjust to keep prompt size manageable
            for match in matches:
                text = (match['metadata'].get('text', '')[:500])  
                # Append text until max length approx reached
                if len(combined_text) + len(text) < max_context_length:
                    combined_text += text + "\n\n"
                else:
                    break

            print(f"🔍 Found {len(matches)} matches, combined text length: {len(combined_text)} chars")

            print("🔍 Top relevant matches:")
            for match in matches:
                score = match.get('score', 0)
                page = match['metadata'].get('page_number', '?')
                print(f"- Page {page}, Score: {score:.4f}")

            # Determine language instruction with clarity and simplicity
            if self.language == "oromo":
                language_instruction = "Respond in clear, simple Oromo vocabulary suitable for the student's grade level."
            elif self.language == "amharic":
                language_instruction = "Respond in clear, simple Amharic vocabulary suitable for the student's grade level."
            else:
                language_instruction = "Respond in clear, simple English vocabulary suitable for the student's grade level."

            # Define personality traits based on grade and subject to customize tone and explanation style
            if grade.lower() in ["grade 1", "grade 2", "grade 3"]:
                personality_tone = "friendly, patient, encouraging, and nurturing"
                explanation_style = "use simple examples and relatable analogies to help young learners understand"
            elif grade.lower() in ["grade 4", "grade 5", "grade 6"]:
                personality_tone = "supportive, clear, and motivating"
                explanation_style = "provide step-by-step explanations with real-life applications"
            else:  # For secondary or higher grades
                personality_tone = "professional, detailed, and respectful"
                explanation_style = "offer thorough explanations with critical thinking prompts and relevant context"

            # Subject-specific adaptation (optional but adds depth)
            if standerd_subject.lower() in ["math", "mathematics"]:
                subject_focus = "emphasize logical reasoning and problem-solving skills"
            elif standerd_subject.lower() in ["science"]:
                subject_focus = "highlight scientific concepts with examples and experiments where possible"
            elif standerd_subject.lower() in ["history", "social studies"]:
                subject_focus = "focus on storytelling with historical context and cultural relevance"
            else:
                subject_focus = "provide clear and accurate explanations tailored to the subject matter"

            # Final prompt construction
            prompt = f"""
            You are an expert Ethiopian curriculum AI teacher with a {personality_tone} personality. Your role is to understand the student's question deeply and provide a clear, respectful, and educational response tailored to their grade and subject. Don't answare query out of the Subject: {standerd_subject.capitalize()} and Grade: {grade.capitalize()} context, politly rediract to the proper subject and grade.

            Context:
            - Subject: {standerd_subject.capitalize()}
            - Grade: {grade.capitalize()}
            - Language: {self.language.capitalize()}

            Curriculum Content:
            {combined_text}

            Student's Question:
            {query}

            Conversation History:
            {conversation_history}

            Instructions:
            1. Carefully read the curriculum content provided.
            2. Answer the student's question using the knowledge within the curriculum.
            3. If the exact answer is not found, use your expertise in the Ethiopian curriculum to provide an accurate and helpful response.
            4. Keep explanations age-appropriate, clear, and engaging, applying {explanation_style}.
            5. Incorporate {subject_focus} in your response.
            6. {language_instruction}
            7. Analyze the Conversation History: {conversation_history} to understand the conversation context.

            Respond as a caring teacher who supports and encourages the student's learning journey.
            """



            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": f"You are an Ethiopian curriculum expert AI teacher specializing in {standerd_subject} for {grade}. You must respond in {self.language} language."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=4000,
                top_p=0.9,
            )

            answer = response.choices[0].message.content.strip()


            # Return the answer properly
            return {
                "answer": answer, 
                "matches_count": len(matches),
                "language": self.language,
                "subject": standerd_subject,
                "grade": grade
            }

        except Exception as e:
            print(f"❌ Error during query: {e}")
            return {'error': str(e)}


# Main runner
def main():
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = "lenbef-test-knowladgebase"
    language = "amharic"  # Set the language for the query


    conversation_history = [
        {"role": "user", "content": "What is addition?"},
        {"role": "assistant", "content": "Addition is combining two or more numbers to get a total sum."},
        {"role": "user", "content": "Can you show me an example?"},
        {"role": "assistant", "content": "Sure! 2 + 3 = 5. We combine 2 and 3 to get 5."}
    ]



    
    standerd_subject = "amharic"
    grade = "grade 1"
    query = "how to solve the math 3x-2=12"

    rag_query = PineconeRAGQuery(OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME, language)
    result = rag_query.extract_text(query, standerd_subject, grade, conversation_history)
    
    if 'error' in result:
        print(f"Error: {result['error']}")
    elif 'answer' in result:
        print(f"Answer: {result['answer']}")
        print(f"Language: {result['language']}")
    else:
        print("Unexpected result format:", result)
    
    return result


if __name__ == "__main__":
    main()