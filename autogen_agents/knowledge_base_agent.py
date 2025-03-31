import os
from autogen import ConversableAgent
from knowledge_base import KnowledgeBase

def knowledge_base_agent(llm_config):
    llm_config = {
        **llm_config,
        "functions": [
            {
                "name": "answer_question",
                "description": "Answer any knowledge base related questions",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The question to answer",
                        }
                    },
                    "required": ["question"],
                },
            }
        ],
    }
    kb = KnowledgeBase()

    def answer_question(question):
        try:
            #print(f"User Query: {question}", flush=True)
            kb_answer = kb.query(question)
        except Exception as e:
            print(f"Error querying knowledge base: {e}")
            kb_answer = "An error occurred while accessing the knowledge base. TERMINATE"

        #print(f"Knowledge Base Answer: {kb_answer}", flush=True)
        return kb_answer

    return ConversableAgent(
        name="Knowledge_Base_Agent",
        system_message="""You fetch data directly from the provided knowledge base to answer queries.
        When answering:
        1. Clearly state you're pulling information from the knowledge base.
        2. If the query cannot be answered from the knowledge base, politely decline.
        3. After correctly answering, end with 'TERMINATE'.""",
        llm_config=llm_config,
        function_map={"answer_question": answer_question}
    )
