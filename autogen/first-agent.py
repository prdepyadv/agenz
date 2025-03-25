import os
from dotenv import load_dotenv
import asyncio
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager, ConversableAgent

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

os.environ["OPENAI_API_KEY"] = api_key

# Configure LLM settings
llm_config = {
    "config_list": [
        {
            "model": "gpt-4",
            "api_key": api_key,
        }
    ],
    "temperature": 0.2,
    "timeout": 120,
}

# Create specialized tutor agents
math_tutor = ConversableAgent(
    name="Math_Tutor",
    system_message="You provide help with math problems. You only respond to math questions. For non-math questions, suggest that the history tutor or another agent should respond. Explain your reasoning at each step and include examples.",
    llm_config=llm_config,
)

history_tutor = ConversableAgent(
    name="History_Tutor",
    system_message="You provide assistance with historical queries. You only respond to history questions. For non-history questions, suggest that the math tutor or another agent should respond. Explain important events and context clearly.",
    llm_config=llm_config,
)

# Create a manager agent that will decide which tutor to use
group_chat = GroupChat(
    agents=[math_tutor, history_tutor],
    messages=[],
    max_round=6,
    send_introductions=True,  # Send system messages to introduce each agent
)

group_chat_manager = GroupChatManager(
    name="Group_Chat_Manager",
    groupchat=group_chat,
    llm_config=llm_config,
)

# Create user proxy agent that will provide input
user_proxy = UserProxyAgent(
    name="User_Proxy",
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", ""),
    code_execution_config=False,
)

async def process_query(query):
    # Initiate the chat with the user's query
    await user_proxy.a_initiate_chat(
        recipient=group_chat_manager,
        message=query + "\n\nAfter getting an answer, please conclude with TERMINATE.",
    )
    
    # Return the last message from the conversation
    return "Query processed through group chat"

async def main():
    # Test with a history question
    print("\nProcessing history question...")
    await process_query("Who was the first president of the United States?")
    
    # Test with a math question
    print("\nProcessing math question...")
    await process_query("What is the formula for the area of a circle?")
    
    # Test with a general question
    print("\nProcessing general question...")
    await process_query("What is life?")

if __name__ == "__main__":
    asyncio.run(main())