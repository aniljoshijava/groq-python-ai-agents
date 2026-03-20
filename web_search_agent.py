# -*- coding: utf-8 -*-
import os
import json
import io
import sys
from groq import Groq
from dotenv import load_dotenv
from groq.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

from tool_definitions import tool_definitions
load_dotenv()

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ===============================
# Client Setup
# ===============================

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL_WEB   = "groq/compound"       
MODEL_LOCAL = "llama-3.3-70b-versatile"

# ===============================
# LOCAL TOOLS
# ===============================

notes_storage: dict[str, str] = {}

def calculator(expression: str) -> str:
    try:
        
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: {e}"

def save_note(key: str, value: str) -> str:
    notes_storage[key] = value
    return f"Note saved: '{key}' = '{value}'"

def get_note(key: str) -> str:
    if key in notes_storage:
        return f"Note: '{key}' = '{notes_storage[key]}'"
    return f"Note not found: '{key}'"

# ===============================
# ROUTER - decide which agent
# ===============================

def needs_web_search(query: str) -> bool:
    """Decide if query needs web search"""
    web_keywords = [
        "news", "today", "latest", "current", "price",
        "weather", "cricket", "score", "ipl", "match",
        "stock", "bitcoin", "trending", "now", "recent",
        "who won", "result", "update", "aaj", "abhi",
        "kya hua", "batao", "2024", "2025", "2026"
    ]
    q = query.lower()
    return any(kw in q for kw in web_keywords)

# ===============================
# Tool Map (name -> function)
# ===============================

tool_map = {
    "calculator": calculator,
    "save_note":  save_note,
    "get_note":   get_note,
}

# ===============================
# AGENT 1 - Web Search
# ===============================

def web_agent(query: str) -> str:
    print("  [Web Search Agent] Searching...")
    try:
        response = client.chat.completions.create(
            model=MODEL_WEB,
            max_tokens=500,
            messages=[
                {
                    "role": "system",
                    "content": "Answer clearly and briefly in 3-5 lines. Use web search."
                },
                {
                    "role": "user",
                    "content": query
                }
            ]
        )
        result = response.choices[0].message.content or ""

        # Show which tools were used (compound-beta exposes this)
        if hasattr(response.choices[0].message, 'executed_tools'):
            tools_used = response.choices[0].message.executed_tools
            if tools_used:
                print(f"  [Tools Used] {tools_used}")

        return result
    except Exception as e:
        return f"Web search failed: {str(e)}"

# ===============================
# AGENT 2 - Local Tools
# ===============================

local_messages: list[ChatCompletionMessageParam] = [
    {
        "role": "system",
        "content": (
            "You are a helpful AI assistant.\n"
            "RULES:\n"
            "- For math: ALWAYS use calculator tool.\n"
            "- For saving info: use save_note tool.\n"
            "- For reading info: use get_note tool.\n"
            "- Reply in Hindi or English based on user language.\n"
            "- Be clear and concise."
        )
    }
]

def local_agent(query: str) -> str:
    local_messages.append({"role": "user", "content": query})

    while True:
        try:
            response = client.chat.completions.create(
                model=MODEL_LOCAL,
                messages=local_messages,
                tools=tool_definitions,
                tool_choice="auto",
                max_tokens=500
            )

            msg           = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            if finish_reason == "tool_calls" and msg.tool_calls:
                local_messages.append({
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id":       tc.id,
                            "type":     "function",
                            "function": {
                                "name":      tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in msg.tool_calls
                    ]
                })

                for tool_call in msg.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    print(f"  [Tool] {tool_name}")

                    tool_func = tool_map.get(tool_name)
                    if tool_func:
                        result = tool_func(**tool_args) if tool_args else tool_func()
                    else:
                        result = "Tool not found"

                    print(f"  [Result] {str(result)[:80]}")

                    local_messages.append({
                        "role":         "tool",
                        "tool_call_id": tool_call.id,
                        "content":      str(result)
                    })

            else:
                final = msg.content or ""
                local_messages.append({"role": "assistant", "content": final})
                return final

        except Exception as e:
            return f"Error: {str(e)}"

# ===============================
# MAIN LOOP
# ===============================

print("=" * 50)
print("  Agentic AI - Web Search + Local Tools")
print("=" * 50)
print("  Web Search: news, weather, cricket, prices")
print("  Local Tools: math, notes")
print("  Type 'exit' to quit\n")

while True:
    try:
        user_input = input("Tum: ").strip()
    except KeyboardInterrupt:
        print("\n Namaste Phir Milenge!")
        break

    if user_input.lower() in ["exit", "quit", "bye", "band karo"]:
        print("Namaste Phir Milenge!")
        break

    if not user_input:
        continue

    print("Soch raha hai...\n")

    if needs_web_search(user_input):
        print("  [Router] Web Search Agent use kar raha hai")
        result = web_agent(user_input)
    else:
        print("  [Router] Local Tools Agent use kar raha hai")
        result = local_agent(user_input)

    safe = result.encode('utf-8', errors='replace').decode('utf-8')
    print(f"\nAgent: {safe}")
    print("-" * 50)