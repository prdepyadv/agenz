import os
from autogen import ConversableAgent
from knowledge_base import KnowledgeBase

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
            if not kb_answer:
                return "I'm sorry, but I couldn't find an answer in the knowledge base."
            return kb_answer
        except Exception as e:
            return "An error occurred while accessing the knowledge base."

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
