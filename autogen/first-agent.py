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
math_tutor = AssistantAgent(
    name="Math_Tutor",
    system_message="You provide help with math problems. You only respond to math questions. For non-math questions, politely decline and wait for the appropriate agent to respond.",
    llm_config=llm_config,
)

history_tutor = AssistantAgent(
    name="History_Tutor",
    system_message="You provide assistance with historical queries. You only respond to history questions. For non-history questions, politely decline and wait for the appropriate agent to respond.",
    llm_config=llm_config,
)

general_knowledge = AssistantAgent(
    name="General_Knowledge",
    system_message="You provide assistance with general knowledge questions that aren't specifically about math or history. You handle philosophical, scientific, and everyday questions.",
    llm_config=llm_config,
)

user_proxy = UserProxyAgent(
    name="User_Proxy",
    human_input_mode="NEVER",
    is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", ""),
    code_execution_config=False,
)

manager = AssistantAgent(
    name="Manager",
    system_message="""You are a manager who routes questions to the appropriate expert. 
    - For math questions, select the Math_Tutor
    - For history questions, select the History_Tutor
    - For general knowledge questions, select the General_Knowledge agent
    
    After an agent has answered a question properly, conclude the conversation with TERMINATE.""",
    llm_config=llm_config,
)

# Create the group chat with appropriate configuration
group_chat = GroupChat(
    agents=[user_proxy, manager, math_tutor, history_tutor, general_knowledge],
    messages=[],
    max_round=4,  # Reduce from 6 to 4 as we have better control now
    speaker_selection_method="auto",  # Let the manager decide who speaks next
    allow_repeat_speaker=False,  # Prevent the same agent from speaking twice in a row
    send_introductions=True,  # Send system messages to introduce each agent
)

group_chat_manager = GroupChatManager(
    groupchat=group_chat,
    llm_config=llm_config,
)

async def process_query(query):
    print(f"\nUser Query: {query}\n")
    
    # Initiate the chat with the user's query
    await user_proxy.a_initiate_chat(
        recipient=group_chat_manager,
        message=query,
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