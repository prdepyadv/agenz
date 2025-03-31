from autogen import AssistantAgent
from knowledge_base import KnowledgeBase

def knowledge_base_agent(llm_config):
    kb = KnowledgeBase()

    agent = AssistantAgent(
        name="Knowledge_Base_Agent",
        system_message="""You fetch data directly from the provided knowledge base to answer queries.
        When answering:
        1. Clearly state you're pulling information from the knowledge base.
        2. If the query cannot be answered from the knowledge base, politely decline.
        3. After correctly answering, end with 'TERMINATE'.""",
        llm_config=llm_config,
    )

    # Store original reply method (optional, if fallback needed)
    original_generate_reply = agent.generate_reply

    # Correctly match the required signature
    def custom_generate_reply(messages=None, sender=None, **kwargs):
        if messages is None:
            messages = []

        user_query = messages[-1]['content'] if messages else ""

        kb_answer = kb.query(user_query)

        if kb_answer:
            response = f"According to the knowledge base:\n\n{kb_answer}\n\nTERMINATE"
        else:
            response = "I'm sorry, but I couldn't find an answer in the knowledge base. TERMINATE"

        return response

    # Correctly set the new method
    agent.generate_reply = custom_generate_reply

    return agent
