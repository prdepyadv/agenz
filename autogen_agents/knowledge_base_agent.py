import os
from autogen import ConversableAgent
from knowledge_base import KnowledgeBase

def hr_agent(llm_config):
    llm_config = {
        **llm_config,
        "functions": [
            {
                "name": "answer_from_hr",
                "description": "Fetch answers explicitly from the HR knowledge base (hr_docs)",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The user's HR related question referencing hr_docs",
                        }
                    },
                    "required": ["question"],
                },
            }
        ],
    }
    hr_kb = KnowledgeBase(docs_path="hr_docs")

    def answer_from_hr(question):
        try:
            kb_answer = hr_kb.query(question)
            if not kb_answer.strip():
                return "I'm sorry, but I couldn't find that policy in the HR knowledge base."
            # Show snippet + reference
            return f"I am retrieving this from the HR knowledge base:\n{kb_answer}"
        except Exception:
            return "An error occurred while accessing the HR knowledge base."

    return ConversableAgent(
        name="HR_Agent",
        system_message="""You ONLY fetch data from the HR knowledge base when the user explicitly references HR docs or HR questions.

        Important Instructions:
        1. ALWAYS query the `answer_from_hr` function when HR-related questions referencing hr_docs occur.
        2. If the retrieved answer from HR knowledge base is empty, politely say 'I'm sorry, but I couldn't find that information in the HR knowledge base.' and end with 'TERMINATE'.
        3. If the retrieved answer is valid, ENHANCE and CLARIFY the retrieved HR content.
        4. EXPLICITLY INCLUDE the reference document from the HR knowledge base at the end of your response in the following format: '[Reference: <document_name>]'.
        5. After clearly enhancing, clarifying, and including reference, end with 'TERMINATE'.""",
        llm_config=llm_config,
        function_map={"answer_from_hr": answer_from_hr},
    )

def knowledge_base_agent(llm_config):
    llm_config = {
        **llm_config,
        "functions": [
            {
                "name": "answer_from_kb",
                "description": "Fetch answers explicitly from the internal knowledge base",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The user's question explicitly directed at the knowledge base",
                        }
                    },
                    "required": ["question"],
                },
            }
        ],
    }
    kb = KnowledgeBase()

    def answer_from_kb(question):
        try:
            kb_answer = kb.query(question)
            if not kb_answer.strip():
                return "I'm sorry, but I couldn't find an answer in the knowledge base. TERMINATE"
            return f"I am retrieving this from the internal knowledge base:\n{kb_answer}\nTERMINATE"
        except Exception as e:
            return "An error occurred while accessing the knowledge base. TERMINATE"

    return ConversableAgent(
        name="Knowledge_Base_Agent",
        system_message="""You fetch data ONLY when explicitly requested from the provided internal knowledge base.
        
        Important Instructions:
        1. ONLY respond to questions explicitly mentioning the internal knowledge base.
        2. Clearly state you're retrieving the information from the internal knowledge base.
        3. If the query cannot be answered from the internal knowledge base, politely decline.
        4. After answering or declining, end your response with 'TERMINATE'.""",
        llm_config=llm_config,
        function_map={"answer_from_kb": answer_from_kb}
    )
