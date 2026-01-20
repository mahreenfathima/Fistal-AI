from langgraph.graph import StateGraph, START
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
import os
from typing import Optional
import sys

api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    google_api_key=api_key
)

print("🔍 DEBUG: MCP Server Path Resolution")
print("=" * 60)
print(f"📂 Current working directory: {os.getcwd()}")
print(f"📂 __file__ is: {__file__}")
print(f"📂 __file__ directory: {os.path.dirname(__file__)}")

# Try multiple path strategies
server_path_options = [
    os.path.join(os.path.dirname(__file__), "server.py"),
    os.path.join(os.getcwd(), "server.py"),
    os.path.abspath("server.py"),
    "server.py"
]

server_path = None
for path in server_path_options:
    print(f"🔍 Checking: {path} ... ", end="")
    if os.path.exists(path):
        print("✅ FOUND!")
        server_path = path
        break
    else:
        print("❌ Not found")

if not server_path:
    print("\n❌ ERROR: server.py not found in any expected location!")
    print(f"📁 Files in current directory: {os.listdir('.')}")
    if os.path.dirname(__file__):
        print(f"📁 Files in __file__ directory: {os.listdir(os.path.dirname(__file__))}")
    raise FileNotFoundError("server.py not found. Make sure it's uploaded to your HF Space.")
else:
    print(f"✅ Using server.py at: {server_path}")

print(f"🐍 Python executable: {sys.executable}")
print("=" * 60)

# Initialize MCP Client
try:
    # In client.py - pass environment explicitly
    client = MultiServerMCPClient(
    {
        "FistalMCP": {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-u", server_path],
            "env": {  # Explicitly pass environment variables
                **os.environ,  # Pass all current environment
                "GROQ_API_KEY": os.getenv("GROQ_API_KEY", ""),
                "HF_TOKEN": os.getenv("HF_TOKEN", ""),
                "GOOGLE_API_KEY_1": os.getenv("GOOGLE_API_KEY_1", ""),
                "GOOGLE_API_KEY_2": os.getenv("GOOGLE_API_KEY_2", ""),
                "GOOGLE_API_KEY_3": os.getenv("GOOGLE_API_KEY_3", ""),
                "PATH": os.getenv("PATH", ""),
            }
        }
    }
)
    print("✅ MCP Client initialized successfully!")
except Exception as e:
    print(f"❌ Failed to initialize MCP Client: {e}")
    import traceback
    traceback.print_exc()
    raise


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    dataset_topic: str
    num_samples: int
    model_name: str
    task_type: str
    dataset_path: Optional[str]
    converted_path: Optional[str]
    model_path: Optional[str]
    hf_url: Optional[str]


async def my_graph():
    """Agent graph that handles mcp tools"""
    tools = await client.get_tools()

    available_tools = []
    tool_order = ["generate_json_data", "format_json", "finetune_model", "llm_as_judge"]
    available_tools = []
    for tool_name in tool_order:
        for tool in tools:
            if tool.name == tool_name:
                available_tools.append(tool)
                break

    llm_toolkit = llm.bind_tools(available_tools)

    async def chat_node(state: ChatState):
        messages = state["messages"]
        dataset_topic = state['dataset_topic']
        if isinstance(dataset_topic, list):
            dataset_topic = dataset_topic[0] if dataset_topic else "unknown"
        
        num_samples = state['num_samples']
        if isinstance(num_samples, list):
            num_samples = num_samples[0] if num_samples else 100
        
        model_name = state['model_name']
        if isinstance(model_name, list):
            model_name = model_name[0] if model_name else "unknown"
        
        task_type = state['task_type']
        if isinstance(task_type, list):
            task_type = task_type[0] if task_type else "text-generation"
        
        system_msg = f"""You are Fistal, an AI fine-tuning assistant.

**User's Configuration:**
- Dataset Topic: {dataset_topic}
- Number of Samples: {num_samples}
- Model to Fine-tune: {model_name}
- Task Type: {task_type}
- Evaluation : Using LLM

**Your Workflow:**
1. Use generate_json_data with topic="{dataset_topic}", task_type="{task_type}", num_samples={num_samples}
   - This returns a dictionary with a "data" field containing the raw dataset
   
2. Use format_json with the "data" field from step 1
   - Pass: raw_data=<the data list from step 1>
   - This returns a dictionary with a "data" field containing formatted data
   
3. Use finetune_model with the "data" field from step 2 and model_name="{model_name}"
   - Pass: formatted_data=<the data list from step 2>, model_name="{model_name}"
   - This returns the Hugging Face repo URL
   
4. Use llm_as_judge with the repo_id from step 3
   - Pass: repo_id=<the HF repo from step 3>, topic="{dataset_topic}", task_type="{task_type}"

**FINAL STEP - CRITICAL:**
5. After completing all tools, you MUST return:
   - The Hugging Face model URL from step 3
   - The evaluation report from step 4
   - Format your final response as:
   
🎉 **Fine-tuning Complete!**

**🤗 Model Repository:** [HF Repo Link] \n\n
**📊 Evaluation Report:** [Full report from llm_as_judge]

**IMPORTANT:**
- Tools pass DATA directly, not file paths
- Always mention the tool you are going to use first and then proceed with the tool action
- Extract the "data" field from each tool's response and pass it to the next tool
- After llm_as_judge completes, return both the HF URL and evaluation report
- Keep the user informed of progress at each step
- If a step takes time, do not stay idle. Inform users about short interesting facts about Modal, Unsloth, Gemini, Gradio , HuggingFace and MCP, do not repeat them.
- Try to add atleast 1 new fact every 10 seconds.
- Report any errors clearly
- Do not mention internal data structures or file paths"""

    
        full_messages = [SystemMessage(content=system_msg)] + messages
        response = await llm_toolkit.ainvoke(full_messages)
        return {'messages': [response]}
    
    tool_node = ToolNode(available_tools)

    graph = StateGraph(ChatState)

    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")

    chat = graph.compile()

    return chat




async def run_fistal(
    dataset_topic: str,
    num_samples: int,
    model_name: str,
    task_type: str
):
    chatbot = await my_graph()
    user_message = f"""Execute the complete fine-tuning workflow:
- Generate {num_samples} training examples about {dataset_topic}
- Fine-tune {model_name}
- Evaluate for {task_type} task

Start now!"""
    initial_state = {
        "messages": [HumanMessage(content=user_message)],
        "dataset_topic": dataset_topic,
        "num_samples": num_samples,
        "model_name": model_name,
        "task_type": task_type,
        "dataset_path": None,
        "converted_path": None,
        "model_path": None,
        "hf_url": None
    }
    facts = {
        "generate_json_data": [
            "💡 Using parallel batch generation with multiple API keys for 3x speed!",
            "📊 Quality over quantity - diverse examples lead to better models!",
            "🎯 Generating diverse prompt-response pairs...",
        ],
        "format_json": [
            "🔄 Converting to chat format optimized for instruction tuning...",
            "💬 Proper formatting helps models understand conversation structure!",
            "🎨 Applying ChatML format for consistency...",
            "✅ Validating JSON structure for training compatibility...",
            "🔧 Optimizing token distribution across examples..."
        ],
        "finetune_model": [
            "🏋️ Training on Modal's serverless T4 GPU...",
            "💡 Using 4-bit quantization to fit in 16GB VRAM!",
            "🦥 Unsloth makes training 2x faster with 70% less memory!",
            "⚡ LoRA fine-tuning updates only 0.1% of model parameters!",
            "🎯 Typical training time: 10-20 minutes for 500 samples...",
            "🔥 Your model is learning patterns from authentic data!",
            "☁️ Uploading to HuggingFace - your model will be public soon!"
        ],
        "llm_as_judge": [
            "📊 Generating evaluation test cases...",
            "🤖 LLM-as-judge provides qualitative insights!",
            "✨ Testing model coherence, relevance, and accuracy...",
            "📝 Creating comprehensive evaluation report...",
            "🔍 Analyzing response quality and task alignment...",
            "📝 Creating comprehensive evaluation report...",
            "📈 Comparing outputs against expected responses...",
            "🎯 Assessing model's understanding of the domain...",
            "✅ Finalizing evaluation metrics.."
        ]
    }

    current_tool = None
    fact_i = 0

    async for event in chatbot.astream(initial_state):
        if "tools" in event:
            messages = event["tools"].get("messages", [])
            for msg in messages:
                if hasattr(msg,"name"):
                    tool_name = msg.name
                    current_tool = tool_name
                    fact_i = 0
                    yield f"\n{'-'*60}\n"
                    yield f"🔄 **Using: {tool_name}**\n\n"
                    if tool_name in facts:
                        yield f"{facts[tool_name][0]}\n"
                        await asyncio.sleep(0.3)

        if "chat_node" in event:
            messages = event["chat_node"].get("messages", [])
            for msg in messages:
                if hasattr(msg, 'content') and msg.content:
                    raw_content = msg.content
                    content = ""
                    
                    if isinstance(raw_content, list):
                        for item in raw_content:
                            if isinstance(item, dict) and item.get('type') == 'text':
                                content += item.get('text', '')
                        content = content.strip()
                    elif isinstance(raw_content, str):
                        content = raw_content
                    else:
                        content = str(raw_content)
                    
                    if content and len(content) > 20 and "tool_calls" not in content.lower():
                        yield f"\n🤖 **Fistal:** {content}\n"
                    
                    if current_tool and current_tool in facts:
                        fact_i += 1
                        if fact_i < len(facts[current_tool]):
                            yield f"\n💡 {facts[current_tool][fact_i]}\n"
                            await asyncio.sleep(0.3)
    yield "✅ **Successfully finetuned!**\n"



async def main():
    """Test the agent. Only for running client.py"""
    print("Testing Fistal Agent\n")
    
    result = await run_fistal(
        "python programming",
        5,
        "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
        "text-generation"
    )
    
    print(f"\nAgent Response:\n{result}")


if __name__ == '__main__':
    asyncio.run(main())