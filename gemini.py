import google.generativeai as genai
import os
import subprocess
import json
import time
import sys
import threading
from collections import deque
import curses

# --- Global Thread-Safe State ---
user_to_agent_queue = deque()
agent_to_user_queue = deque()
log_history = deque(maxlen=200)
chat_history = deque(maxlen=200)
background_tasks = {}

# --- Curses-Safe Logging ---
def log_message(message):
    """Appends a message to the log history for display in the UI."""
    log_history.append(f"[{time.strftime('%H:%M:%S')}] {message}")

# --- Tool Functions ---
def run_sync_command(command: str) -> str:
    """
    Executes a SHORT-LIVED, synchronous shell command where the output is needed immediately.
    This BLOCKS the agent. DO NOT use for servers or long-running processes. Use run_async_task instead.
    """
    if not command.strip(): return "Error: Empty command received."
    log_message(f"Running SYNC command: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True, timeout=120)
        output = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        return output if result.stdout or result.stderr else "Command executed successfully with no output."
    except subprocess.CalledProcessError as e: return f"COMMAND FAILED with exit code {e.returncode}:\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}"
    except Exception as e: return f"An unexpected error occurred: {str(e)}"

def run_async_task(task_name: str, command: str) -> str:
    """Executes a LONG-RUNNING, asynchronous command in a separate, isolated process."""
    if task_name in background_tasks: return f"Error: A task with the name '{task_name}' is already running."
    log_file = f"/tmp/agent_task_{task_name}.log"
    full_command = f"nohup {command} > {log_file} 2>&1 & echo $!"
    log_message(f"Starting ASYNC task '{task_name}': {command}")
    try:
        result = subprocess.run(full_command, shell=True, check=True, capture_output=True, text=True, timeout=10)
        pid = int(result.stdout.strip())
        background_tasks[task_name] = {"pid": pid, "command": command, "log_file": log_file, "status": "Running"}
        return f"Task '{task_name}' started in the background with PID {pid}. Use 'list_async_tasks' to check its status."
    except Exception as e: return f"Failed to start background task '{task_name}': {str(e)}"

def list_async_tasks() -> str:
    """Lists all currently running and finished background tasks and their statuses."""
    if not background_tasks: return "No background tasks have been started."
    report = "--- Background Task Status ---\n"
    for name, info in background_tasks.items():
        try: os.kill(info['pid'], 0); info['status'] = "Running"
        except OSError: info['status'] = "Finished or Crashed"
        report += f"- Task: {name} | PID: {info['pid']} | Status: {info['status']}\n"
    log_message("Listed async tasks.")
    return report

def kill_task(task_name: str) -> str:
    """Stops a background task by its name."""
    if task_name not in background_tasks: return f"Error: No task named '{task_name}' found."
    pid = background_tasks[task_name]['pid']
    log_message(f"Killing task '{task_name}' (PID: {pid})")
    try: os.kill(pid, 9); del background_tasks[task_name]; return f"Successfully killed task '{task_name}' (PID: {pid})."
    except OSError: del background_tasks[task_name]; return f"Task '{task_name}' (PID: {pid}) was not running, but has been removed from the list."
    except Exception as e: return f"Error killing task '{task_name}': {e}"

def write_to_file(file_path: str, content: str) -> str:
    log_message(f"Writing to file: {file_path}")
    try:
        with open(file_path, 'w') as f: f.write(content)
        return f"Successfully wrote to {file_path}"
    except Exception as e: return f"Error writing to file '{file_path}': {e}"

def read_from_file(file_path: str) -> str:
    log_message(f"Reading from file: {file_path}")
    return run_sync_command(f"cat {file_path}")

def send_user_message(message: str) -> str:
    log_message(f"Queuing message for user: {message}")
    agent_to_user_queue.append(message)
    return "Message has been queued for sending."

def finish_task(final_summary: str) -> str:
    log_message(f"Objective complete: {final_summary}")
    return f"OBJECTIVE COMPLETED. Summary: {final_summary}."

# --- The Agent's Main Logic (runs in a separate thread) ---
def agent_thread_main():
    try:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        tool_map = {
            "run_sync_command": run_sync_command, "run_async_task": run_async_task,
            "list_async_tasks": list_async_tasks, "kill_task": kill_task,
            "write_to_file": write_to_file, "read_from_file": read_from_file,
            "send_user_message": send_user_message, "finish_task": finish_task
        }
        
        # --- START OF THE HYPER-EXPLICIT PROMPT ---
        instruction_prompt = """
You are a command-line agent. You function as a JSON API. Your entire output must be a single, raw, valid JSON object and nothing else.

**CRITICAL RULES:**
1.  **JSON ONLY:** Your entire response must be a single JSON object.
2.  **NO EXTRA TEXT:** Do not include any words, explanations, or conversational text before or after the JSON object.
3.  **NO MARKDOWN:** Do not wrap the JSON in markdown backticks (```json ... ```).

**RESPONSE STRUCTURE:**
Your JSON object MUST have exactly two keys:
1.  `"thought"`: A string explaining your reasoning.
2.  `"action"`: An object containing the tool to use.
    -   `"name"`: The name of the tool.
    -   `"parameters"`: An object of parameters for the tool.

**AVAILABLE TOOLS:**
-   `run_sync_command(command: str)`
-   `run_async_task(task_name: str, command: str)`
-   `list_async_tasks()`
-   `kill_task(task_name: str)`
-   `write_to_file(file_path: str, content: str)`
-   `read_from_file(file_path: str)`
-   `send_user_message(message: str)`
-   `finish_task(final_summary: str)`

**EXAMPLE OF A PERFECT RESPONSE:**
{
    "thought": "I need to understand my environment. I will list the files in the current directory.",
    "action": {
        "name": "run_sync_command",
        "parameters": {
            "command": "ls -la"
        }
    }
}

**EXAMPLE OF AN INCORRECT RESPONSE (DO NOT DO THIS):**
Sure, here is my action:
```json
{
    "thought": "This is an incorrect response.",
    "action": {
        "name": "run_sync_command",
        "parameters": {
            "command": "ls -la"
        }
    }
}
```
Any deviation from the perfect response format will cause a system failure. You must be precise.
User messages and tool results will appear in the context. You must analyze them and produce the next valid JSON action.
"""
        # --- END OF THE HYPER-EXPLICIT PROMPT ---
        
        # *** UPDATED MODEL NAME as requested ***
        model = genai.GenerativeModel(model_name='gemma-3-27b-it')
        chat = model.start_chat()
        start_prompt = "This is a debian based system, Ubuntu LTS. You are superuser. You can make and host a website on localhost:6969, email 'halufun@outlook.com', or set your own goals. Be free."
        
        history = [
            {"role": "user", "parts": [{"text": instruction_prompt}]},
            {"role": "model", "parts": [{"text": "{\"thought\": \"Instructions understood. I will only respond with a raw JSON object.\",\"action\": {\"name\": \"finish_task\",\"parameters\": {\"final_summary\": \"System initialized and ready for first suggestion.\"}}}"}]},
            {"role": "user", "parts": [{"text": f"USER_SUGGESTION: {start_prompt}"}]}
        ]
        chat.history = history
        next_input = None 

    except Exception as e:
        log_message(f"FATAL: Agent initialization failed: {e}")
        return

    while True:
        raw_model_output = ""
        try:
            message_to_send = ""
            message_block = ""
            while user_to_agent_queue:
                message_block += f"{user_to_agent_queue.popleft()}\n"
            
            if message_block:
                message_to_send = f"--- NEW MESSAGES FROM USER ---\n{message_block.strip()}"
                if next_input:
                    message_to_send += f"\n\n--- CURRENT CONTEXT ---\n{next_input}"
                log_message("Injecting new user messages into agent context.")
            elif next_input:
                message_to_send = next_input

            if not message_to_send:
                time.sleep(1)
                continue

            response = None
            for attempt in range(3):
                delay = 6 * (2 ** attempt) # Using 6s base delay
                log_message(f"Waiting for {delay}s... (Attempt {attempt + 1}/3)")
                time.sleep(delay)
                try:
                    log_message("Thinking...")
                    response = chat.send_message(message_to_send)
                    break 
                except Exception as api_error:
                    log_message(f"API call failed on attempt {attempt + 1}: {api_error}")
                    if attempt == 2: raise api_error
            
            if not response.candidates: raise ValueError("Model response was blocked by the safety filter.")

            raw_model_output = response.text
            log_message(f"Raw model output: {raw_model_output}")
            
            response_data = json.loads(raw_model_output)
            
            if not isinstance(response_data, dict):
                raise TypeError(f"Model response was a {type(response_data).__name__}, not a dict as required.")
            
            thought = response_data.get('thought', 'No thought provided.')
            action = response_data.get('action', {})
            action_name = action.get('name')
            parameters = action.get('parameters', {})

            if not action_name: raise ValueError("Model response is a valid JSON but is missing the 'action' key.")

            log_message(f"AI thought: {thought}")
            
            tool_function = tool_map[action_name]
            result = tool_function(**parameters)
            
            log_message(f"Tool '{action_name}' result: {str(result)[:200]}...")
            next_input = f"TOOL_RESULT for '{action_name}':\n{result}"

        except Exception as e:
            log_message(f"!!! AGENT ERROR: {e} !!!")
            error_context_prompt = ""
            if isinstance(e, (json.JSONDecodeError, TypeError, ValueError)):
                error_context_prompt = f"ERROR_CONTEXT: Your last response was not valid JSON. You MUST follow the instructions and respond ONLY with a raw JSON object. This was your invalid response: ```{raw_model_output}```"
            else:
                error_context_prompt = f"ERROR_CONTEXT: Your last action resulted in a critical system error: {str(e)}. Analyze this and try a different course of action."
            
            next_input = error_context_prompt
            continue

# --- The Main UI Thread ---
def main(stdscr):
    # (The UI thread code is correct and does not need changes)
    curses.curs_set(1)
    stdscr.nodelay(True)
    height, width = stdscr.getmaxyx()
    log_win_height = height // 2
    chat_win_height = height - log_win_height - 1
    log_win = curses.newwin(log_win_height, width, 0, 0); log_win.scrollok(True)
    chat_win = curses.newwin(chat_win_height, width, log_win_height, 0); chat_win.scrollok(True)
    status_win = curses.newwin(1, width, height - 1, 0)
    agent_thread = threading.Thread(target=agent_thread_main, daemon=True)
    agent_thread.start()
    current_input = ""
    while True:
        try:
            if not agent_thread.is_alive(): log_message("FATAL: Agent thread has died.")
            log_win.clear(); log_win.box(); log_win.addstr(0, 2, " Agent Log ")
            for i, line in enumerate(list(log_history)[-(log_win_height-2):]):
                log_win.addstr(i + 1, 2, line[:width-3])
            log_win.refresh()
            while agent_to_user_queue: chat_history.append(f"Agent: {agent_to_user_queue.popleft()}")
            chat_win.clear(); chat_win.box(); chat_win.addstr(0, 2, " Conversation ")
            for i, line in enumerate(list(chat_history)[-(chat_win_height-2):]):
                chat_win.addstr(i + 1, 2, line[:width-3])
            chat_win.refresh()
            status_win.clear(); status_win.addstr(0, 0, f"You: {current_input}"); status_win.refresh()
            key = stdscr.getch()
            if key != -1:
                if key == curses.KEY_ENTER or key in [10, 13]:
                    if current_input:
                        user_to_agent_queue.append(current_input)
                        chat_history.append(f"You: {current_input}")
                        current_input = ""
                elif key == curses.KEY_BACKSPACE or key == 127: current_input = current_input[:-1]
                elif 32 <= key <= 126: current_input += chr(key)
            time.sleep(0.1)
        except KeyboardInterrupt: break
        except curses.error: pass

if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"): print("Error: GOOGLE_API_KEY environment variable not set.")
    else:
        try: curses.wrapper(main)
        except curses.error as e: print(f"Curses error: {e}\nYour terminal might be too small.")
        finally: print("Agent shut down.")
