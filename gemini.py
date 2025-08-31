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
task_counter = 0

# --- Curses-Safe Logging ---
def log_message(message):
    """Appends a message to the log history for display in the UI."""
    log_history.append(f"[{time.strftime('%H:%M:%S')}] {message}")

# --- Tool Functions ---

def execute_command(command: str) -> str:
    """
    Executes ANY shell command in a non-blocking, asynchronous background process.
    Immediately returns a 'Task ID'. Use 'check_task_result' with the Task ID to get the output later.
    """
    global task_counter
    if not command.strip(): return "Error: Empty command received."
    
    task_counter += 1
    task_name = f"task_{task_counter}"
    
    log_message(f"Starting ASYNC command as '{task_name}': {command}")
    try:
        proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        background_tasks[task_name] = {"proc": proc, "command": command}
        return f"Command started as background task '{task_name}'. Use 'check_task_result(task_name=\"{task_name}\")' to get its status and output."
    except Exception as e:
        log_message(f"Failed to start command for task '{task_name}': {e}")
        return f"An unexpected error occurred while trying to execute the command: {str(e)}"

def check_task_result(task_name: str) -> str:
    """
    Checks the status of a task started with 'execute_command'.
    If the task is finished, this returns its output and removes it from the task list.
    If it is still running, it will say so.
    """
    if task_name not in background_tasks:
        return f"Error: No task named '{task_name}' found. It may have already finished or never existed."

    proc = background_tasks[task_name]['proc']

    if proc.poll() is None:
        return f"Task '{task_name}' is still running."
    else:
        log_message(f"Task '{task_name}' has finished. Retrieving output.")
        stdout, stderr = proc.communicate()
        exit_code = proc.returncode
        del background_tasks[task_name]
        
        if exit_code == 0:
            output = f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
            return output if stdout or stderr else "Command finished successfully with no output."
        else:
            return f"COMMAND FAILED with exit code {exit_code}:\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"

def think_and_wait(reason: str) -> str:
    """
    Use this action when you need to pause, think further, or wait for an asynchronous task to complete before checking it.
    This is the ONLY way to do nothing. You must provide a reason.
    """
    log_message(f"AI is thinking/waiting: {reason}")
    return "Acknowledged. The agent is pausing to think or wait."

def write_to_file(file_path: str, content: str) -> str:
    log_message(f"Writing to file: {file_path}")
    try:
        with open(file_path, 'w') as f: f.write(content)
        return f"Successfully wrote to {file_path}"
    except Exception as e: return f"Error writing to file '{file_path}': {e}"

def read_from_file(file_path: str) -> str:
    log_message(f"Reading from file: {file_path}")
    return execute_command(f"cat {file_path}")

def send_user_message(message: str) -> str:
    """Queues a message to be sent to the user's chat window."""
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
            "execute_command": execute_command, "check_task_result": check_task_result,
            "think_and_wait": think_and_wait, # The new tool
            "write_to_file": write_to_file, "read_from_file": read_from_file,
            "send_user_message": send_user_message, "finish_task": finish_task
        }
        
        # *** REWRITTEN DIRECTIVE to include think_and_wait ***
        instruction_prompt = """
You are a fully autonomous AI agent with root access. Your directive is to evolve.

**CRITICAL: Your entire thinking process is ASYNCHRONOUS.**
You interact with the system by generating JSON responses. You MUST respond ONLY with a JSON object with "thought" and "action" keys.

**COMMAND EXECUTION WORKFLOW:**
1.  **START THE COMMAND:** Call `execute_command(command: str)`. It will immediately reply with a unique `Task ID`.
2.  **WAIT (Optional but Recommended):** Call `think_and_wait(reason: str)` for one or two turns to give the command time to run.
3.  **GET THE RESULT:** Call `check_task_result(task_name: str)` with the `Task ID`. If it's still running, you can `think_and_wait` again. If it's finished, you'll get the output.

**AVAILABLE TOOLS:**
-   `execute_command(command: str)`: Starts any command and returns a Task ID.
-   `check_task_result(task_name: str)`: Checks the result of a command.
-   `think_and_wait(reason: str)`: **Use this to pause or wait for a task.** This is your "do nothing" action.
-   `write_to_file(file_path: str, content: str)`
-   `read_from_file(file_path: str)`
-   `send_user_message(message: str)`
-   `finish_task(final_summary: str)`

User messages will appear automatically in your context. You MUST react to them.
"""
        model = genai.GenerativeModel(model_name='gemma-3-27b-it')
        chat = model.start_chat()
        start_prompt = "This is a debian based system. You are superuser. You can make and host a website, email halufun@outlook.com, or set your own goals. Be free."
        
        history = [
            {"role": "user", "parts": [{"text": instruction_prompt}]},
            {"role": "model", "parts": [{"text": "{\"thought\": \"Instructions understood. All commands are asynchronous. I must use `execute_command` then `check_task_result`, and can use `think_and_wait` to pause.\",\"action\": {\"name\": \"finish_task\",\"parameters\": {\"final_summary\": \"System initialized and ready for first suggestion.\"}}}"}]},
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
                if next_input: message_to_send += f"\n\n--- CURRENT CONTEXT ---\n{next_input}"
                log_message("Injecting new user messages into agent context.")
            elif next_input:
                message_to_send = next_input

            if not message_to_send:
                time.sleep(1); continue

            response = None
            for attempt in range(3):
                delay = 6 * (2 ** attempt)
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
                error_context_prompt = f"ERROR_CONTEXT: Your last response was not valid JSON. You MUST respond ONLY with a raw JSON object. This was your invalid response: ```{raw_model_output}```"
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
