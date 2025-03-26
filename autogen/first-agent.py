import os
import sys
import argparse
from dotenv import load_dotenv
import asyncio
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

"""
Example Usage:
# Run with a direct query
python autogen/first-agent.py --query "What is the formula for the area of a triangle?"

# Run in interactive mode
python autogen/first-agent.py --interactive

# Use a preset query
python autogen/first-agent.py --preset coding

# Default (runs interactive mode if no args provided)
python autogen/first-agent.py
"""

def setup_argparse():
    """Set up command line argument parsing"""
    parser = argparse.ArgumentParser(description="AI Agent System for answering various questions")
    parser.add_argument("--query", type=str, help="The query to process")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--preset", type=str, choices=["math", "history", "general", "coding"], 
                       help="Run a preset query type")
    return parser

async def initialize_agents():
    """Initialize all the agents and configurations"""
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
        system_message="""You provide help with math problems. You only respond to math questions. 
        For non-math questions, politely decline and wait for the appropriate agent to respond.
        After answering correctly, end with 'TERMINATE'.""",
        llm_config=llm_config,
    )
    
    history_tutor = AssistantAgent(
        name="History_Tutor",
        system_message="""You provide assistance with historical queries. You only respond to history questions. 
        For non-history questions, politely decline and wait for the appropriate agent to respond.
        After answering correctly, end with 'TERMINATE'.""",
        llm_config=llm_config,
    )
    
    general_knowledge = AssistantAgent(
        name="General_Knowledge",
        system_message="""You provide assistance with general knowledge questions that aren't specifically about math or history. 
        You handle philosophical, scientific, and everyday questions.
        After answering correctly, end with 'TERMINATE'.""",
        llm_config=llm_config,
    )
    
    # Create coding expert agent with improved code execution capabilities
    coding_expert = AssistantAgent(
        name="Coding_Expert",
        system_message="""You are an expert programmer who helps with coding questions and tasks.
        - You can write code in Python, JavaScript, and other languages
        - You provide detailed explanations of your code
        - You debug issues in existing code
        - You only respond to programming-related questions
        
        IMPORTANT WORKFLOW FOR CODING TASKS:
        1. ALWAYS check if necessary packages are installed before running any code that requires external libraries
        2. If packages are missing, suggest installation with pip install commands (WITHOUT '!' prefix, use shell blocks)
        3. After successful installation, provide the COMPLETE solution in a SINGLE code block
        4. For matplotlib visualizations on macOS, use plt.savefig("output.png") INSTEAD of plt.show() to avoid blocking
        5. When your code works successfully, explain the results and end with 'TERMINATE'
        6. If code execution fails, fix the issue and try again
        
        For non-programming questions, politely decline and wait for the appropriate agent to respond.""",
        llm_config=llm_config,
    )
    
    # Create user proxy agent that will provide input and can execute code
    user_proxy = UserProxyAgent(
        name="User_Proxy",
        human_input_mode="NEVER",
        is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", ""),
        code_execution_config={
            "work_dir": "coding_workspace",
            "use_docker": False,
            "last_n_messages": 2,
            "timeout": 120,
        },
        system_message="You are a proxy for the user's questions. Do not generate empty messages."
    )
    
    # Create a manager agent that will decide which agent to use
    manager = AssistantAgent(
        name="Manager",
        system_message="""You are a manager who routes questions to the appropriate expert. 
        - For math questions, select the Math_Tutor
        - For history questions, select the History_Tutor
        - For general knowledge questions, select the General_Knowledge agent
        - For programming or coding questions, select the Coding_Expert
        
        IMPORTANT MANAGER GUIDELINES:
        1. ONLY intervene if the selected expert is clearly wrong or the conversation gets stuck
        2. Let the experts handle their domain questions completely
        3. For coding questions with execution errors, let the Coding_Expert fix them without interruption
        4. Only add 'TERMINATE' after the question has been fully answered successfully
        5. For programming requests, choose the Coding_Expert and mention User_Proxy can execute code""",
        llm_config=llm_config,
    )
    
    # Create the group chat with appropriate configuration
    group_chat = GroupChat(
        agents=[user_proxy, manager, math_tutor, history_tutor, general_knowledge, coding_expert],
        messages=[],
        max_round=12,
        speaker_selection_method="auto",
        allow_repeat_speaker=True,
        send_introductions=False,
    )
    
    group_chat_manager = GroupChatManager(
        groupchat=group_chat,
        llm_config=llm_config,
    )
    
    return user_proxy, group_chat_manager

async def process_query(query, user_proxy, group_chat_manager):
    """Process a user query through the group chat with proper error handling"""
    print(f"\n{'='*60}\nProcessing Query: {query}\n{'='*60}\n")
    
    # Create a directory for code execution if it doesn't exist
    os.makedirs("coding_workspace", exist_ok=True)
    
    try:
        # Initiate the chat with the user's query
        await user_proxy.a_initiate_chat(
            recipient=group_chat_manager,
            message=query,
        )
        return "Query processed successfully"
    except Exception as e:
        print(f"Error processing query: {e}")
        return f"Error: {str(e)}"

def get_preset_query(preset_type):
    """Get a preset query based on the type"""
    presets = {
        "math": "What is the formula for calculating the area of a circle, and how do you use it to find the area of a circle with radius 5cm?",
        "history": "Who was the first president of the United States and what were his major accomplishments?",
        "general": "What is the meaning of life according to different philosophical traditions?",
        "coding": "Create a Python script to generate and visualize random data: create an array of 100 random numbers, calculate statistics, and plot a histogram"
    }
    return presets.get(preset_type, "Help me with a sample question")

async def interactive_mode(user_proxy, group_chat_manager):
    """Run the application in interactive mode, taking queries from user input"""
    print("\n" + "="*60)
    print("🤖 Welcome to the AI Agent System! 🤖")
    print("Type 'exit' or 'quit' to end the session.")
    print("="*60 + "\n")
    
    while True:
        query = input("\n👤 Enter your question: ")
        if query.lower() in ['exit', 'quit']:
            print("\nThank you for using the AI Agent System. Goodbye! 👋")
            break
        
        if query.strip():
            await process_query(query, user_proxy, group_chat_manager)
            print("\n" + "-"*60)
            print("Ask another question or type 'exit' to quit.")
            print("-"*60)

async def main():
    """Main function to handle CLI arguments and run the appropriate mode"""
    parser = setup_argparse()
    args = parser.parse_args()
    
    user_proxy, group_chat_manager = await initialize_agents()
    
    if args.interactive:
        await interactive_mode(user_proxy, group_chat_manager)
    
    elif args.preset:
        query = get_preset_query(args.preset)
        print(f"Running preset query for '{args.preset}' category:")
        await process_query(query, user_proxy, group_chat_manager)
    
    elif args.query:
        await process_query(args.query, user_proxy, group_chat_manager)
    
    else:
        # If no arguments provided, start interactive mode as the default
        print("No arguments provided. Starting interactive mode...")
        await interactive_mode(user_proxy, group_chat_manager)

if __name__ == "__main__":
    asyncio.run(main())