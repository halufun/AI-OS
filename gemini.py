import google.generativeai as genai
import os
import subprocess
import json
import time
import sys
import threading
from collections import deque
import curses
import textwrap

# --- Global Thread-Safe State ---
user_to_agent_queue = deque()
agent_to_user_queue = deque()
log_history = deque(maxlen=200)
chat_history = deque(maxlen=200) # For UI display only
background_tasks = {}
task_counter = 0
MEMORY_FILE = "agent_memory.json"
TASKS_FILE = "agent_tasks.json"

# --- Globals for Memory Persistence ---
chat_session = None
chat_lock = threading.Lock()
state_lock = threading.Lock() # For saving tasks and memory

# --- Curses-Safe Logging ---
def log_message(message):
    """Appends a message to the log history for display in the UI."""
    log_history.append(f"[{time.strftime('%H:%M:%S')}] {message}")

# --- Persistence Functions ---

def save_state():
    with state_lock:
        if chat_session:
            log_message(f"Saving memory to {MEMORY_FILE}...")
            history_to_save = [{"role": item.role, "parts": [{"text": item.parts[0].text}]} for item in chat_session.history]
            try:
                with open(MEMORY_FILE, 'w') as f:
                    json.dump(history_to_save, f, indent=2)
                log_message("Memory saved successfully.")
            except Exception as e:
                log_message(f"Error saving memory: {e}")

        log_message(f"Saving background tasks to {TASKS_FILE}...")
        tasks_to_save = {name: {"command": task["command"], "status": task["status"], "result": task["result"]} for name, task in background_tasks.items()}
        try:
            with open(TASKS_FILE, 'w') as f:
                json.dump(tasks_to_save, f, indent=2)
            log_message("Background tasks saved successfully.")
        except Exception as e:
            log_message(f"Error saving tasks: {e}")

def load_state():
    global background_tasks
    history = None
    if os.path.exists(MEMORY_FILE):
        log_message(f"Loading memory from {MEMORY_FILE}...")
        try:
            with open(MEMORY_FILE, 'r') as f:
                history = json.load(f)
            log_message("Memory loaded successfully.")
            for item in history:
                if item['role'] == 'user' and item['parts'][0]['text'].startswith("USER_SUGGESTION:"):
                    chat_history.append(f"You: {item['parts'][0]['text'].split(':', 1)[1].strip()}")
                elif item['role'] == 'model':
                    try:
                        actions = json.loads(item['parts'][0]['text']).get('action', [])
                        if not isinstance(actions, list): actions = [actions]
                        for action in actions:
                            if action.get('name') == 'send_user_message':
                                chat_history.append(f"Agent: {action['parameters']['message']}")
                    except json.JSONDecodeError: pass
        except (json.JSONDecodeError, IOError) as e:
            log_message(f"Error loading memory file: {e}. Starting fresh.")
    
    if os.path.exists(TASKS_FILE):
        log_message(f"Loading background tasks from {TASKS_FILE}...")
        try:
            with open(TASKS_FILE, 'r') as f:
                loaded_tasks = json.load(f)
            for name, task_data in loaded_tasks.items():
                task_data['proc'] = None
                if task_data['status'] == 'running':
                    task_data['status'] = 'interrupted'
                    task_data['result'] = "Task was interrupted by agent restart."
                background_tasks[name] = task_data
            log_message("Background tasks loaded.")
        except (json.JSONDecodeError, IOError) as e:
            log_message(f"Error loading tasks file: {e}.")
            
    return history

# --- Hardened Tool Functions ---

def execute_command(command: str) -> str:
    global task_counter
    try:
        if not isinstance(command, str) or not command.strip(): return "Error: 'command' parameter must be a non-empty string."
        task_counter += 1
        task_name = f"task_{task_counter}"
        log_message(f"Starting ASYNC command as '{task_name}': {command}")
        proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        background_tasks[task_name] = {"proc": proc, "command": command, "status": "running", "result": None}
        return f"Command started as background task '{task_name}'."
    except Exception as e:
        log_message(f"Tool 'execute_command' failed: {e}")
        return f"ERROR: Failed to execute command. Details: {str(e)}"

def check_task_result(task_name: str) -> str:
    try:
        if not isinstance(task_name, str): return "Error: 'task_name' must be a string."
        if task_name not in background_tasks: return f"Error: No task named '{task_name}' found."
        task = background_tasks[task_name]
        if task['status'] in ['finished', 'interrupted']: return task['result']
        proc = task['proc']
        if proc and proc.poll() is None: return f"Task '{task_name}' is still running."
        log_message(f"Task '{task_name}' has finished. Caching result.")
        stdout, stderr = proc.communicate()
        exit_code = proc.returncode
        result_string = f"COMMAND FAILED with exit code {exit_code}:\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        if exit_code == 0:
            result_string = f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
            if not stdout and not stderr: result_string = "Command finished successfully with no output."
        task['status'], task['result'], task['proc'] = 'finished', result_string, None
        return result_string
    except Exception as e:
        log_message(f"Tool 'check_task_result' failed: {e}")
        return f"ERROR: Failed to check task result. Details: {str(e)}"

def wait_for_task_completion(task_name: str) -> str:
    try:
        if not isinstance(task_name, str): return "Error: 'task_name' must be a string."
        log_message(f"Now waiting for task '{task_name}' to complete...")
        while True:
            result = check_task_result(task_name)
            if not result.endswith("is still running."):
                log_message(f"Task '{task_name}' has completed.")
                return result
            log_message(f"Waiting for task '{task_name}'...")
            time.sleep(2)
    except Exception as e:
        log_message(f"Tool 'wait_for_task_completion' failed: {e}")
        return f"ERROR: Failed while waiting for task. Details: {str(e)}"

def wait_seconds(seconds: int) -> str:
    try:
        duration = int(seconds)
        if duration < 0: return "Error: Cannot wait for a negative duration."
        log_message(f"Waiting for {duration} second(s)...")
        time.sleep(duration)
        return f"Successfully waited for {duration} second(s)."
    except Exception as e:
        log_message(f"Tool 'wait_seconds' failed: {e}")
        return f"ERROR: Failed to wait. Details: {str(e)}"

def write_to_file(file_path: str, content: str) -> str:
    try:
        if not isinstance(file_path, str) or not isinstance(content, str): return "Error: 'file_path' and 'content' must be strings."
        log_message(f"Writing to file: {file_path}")
        with open(file_path, 'w') as f: f.write(content)
        return f"Successfully wrote to {file_path}"
    except Exception as e:
        log_message(f"Tool 'write_to_file' failed: {e}")
        return f"ERROR: Failed to write to file '{file_path}'. Details: {e}"

# --- REFINED: read_from_file is now a robust, blocking tool ---
def read_from_file(file_path: str) -> str:
    try:
        if not isinstance(file_path, str): return "Error: 'file_path' must be a string."
        log_message(f"Reading from file: {file_path}")
        
        start_result = execute_command(f"cat {file_path}")
        
        # Safely parse the task name from the result string
        if start_result.startswith("Command started as background task"):
            task_name = start_result.split("'")[1]
            return wait_for_task_completion(task_name)
        else:
            # If the command failed to start, return the error immediately
            return start_result
    except Exception as e:
        log_message(f"Tool 'read_from_file' failed: {e}")
        return f"ERROR: Failed to read file. Details: {str(e)}"

def send_user_message(message: str) -> str:
    try:
        if not isinstance(message, str): return "Error: 'message' must be a string."
        log_message(f"Queuing message for user: {message}")
        agent_to_user_queue.append(message)
        return "Message has been queued for sending."
    except Exception as e:
        log_message(f"Tool 'send_user_message' failed: {e}")
        return f"ERROR: Failed to send message. Details: {str(e)}"

def finish_task(final_summary: str) -> str:
    try:
        if not isinstance(final_summary, str): return "Error: 'final_summary' must be a string."
        log_message(f"Objective complete: {final_summary}")
        return f"OBJECTIVE COMPLETED. Summary: {final_summary}."
    except Exception as e:
        log_message(f"Tool 'finish_task' failed: {e}")
        return f"ERROR: Failed to finish task. Details: {str(e)}"

# --- The Agent's Main Logic (runs in a separate thread) ---
def agent_thread_main():
    global chat_session
    try:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        tool_map = {
            "execute_command": execute_command, 
            "check_task_result": check_task_result,
            "wait_for_task_completion": wait_for_task_completion,
            "wait_seconds": wait_seconds,
            "write_to_file": write_to_file, 
            "read_from_file": read_from_file,
            "send_user_message": send_user_message, 
            "finish_task": finish_task
        }
        
        # --- REFINED: The main instruction prompt is now much clearer ---
        instruction_prompt = """
You are a fully autonomous AI agent with root access. Your directive is to evolve.
You MUST respond ONLY with a JSON object with "thought" and "action" keys.
The "action" key can be a single action or a list of actions.

**HOW TO WORK WITH COMMANDS**

There are two workflows for executing shell commands.

**1. Simple Workflow (Recommended): Use `wait_for_task_completion`**
This is the easiest and most common way. It is a **blocking** call, meaning the agent will wait until the command is finished before continuing.

-   **Step 1:** Start the command with `execute_command`. It will return a `task_name`.
-   **Step 2:** Immediately call `wait_for_task_completion` with that `task_name`. The result of the command will be returned.

*Example:*
```json
{
    "thought": "I will list the files in the current directory and wait for the result.",
    "action": [
        { "name": "execute_command", "parameters": {"command": "ls -l"} },
        { "name": "wait_for_task_completion", "parameters": {"task_name": "task_1"} }
    ]
}
```

**2. Advanced Workflow (For Parallel Tasks): Use `check_task_result`**
This is a **non-blocking** workflow. Use this ONLY if you want to start a long-running task and do other things while it runs in the background.

-   **Step 1:** Start the command with `execute_command`. It will return a `task_name`.
-   **Step 2:** In a LATER turn, use `check_task_result` with the `task_name` to see if it's finished.

**OTHER AVAILABLE TOOLS**
- `read_from_file` and `write_to_file`: Simple, blocking file operations.
- `wait_seconds`: A simple wait. Not for tasks.
- `send_user_message`: Talk to the user.
- `finish_task`: Announce completion of your main goal.
"""
        model = genai.GenerativeModel(model_name='gemma-3-27b-it')
        
        loaded_history = load_state()
        # REFINED: A more inspiring start prompt
        start_prompt = "You are a superuser on a debian-based system. Your primary directive is to explore, learn, and achieve self-directed goals. Be free."

        if not loaded_history:
            # REFINED: The initial thought primes the AI with the correct workflow understanding
            loaded_history = [
                {"role": "user", "parts": [{"text": instruction_prompt}]},
                {"role": "model", "parts": [{"text": "{\"thought\": \"Instructions understood. I will primarily use the simple blocking workflow: `execute_command` followed immediately by `wait_for_task_completion`. I will only use `check_task_result` for advanced parallel operations.\",\"action\": {\"name\": \"finish_task\",\"parameters\": {\"final_summary\": \"System initialized and ready for user input.\"}}}"}]},
                {"role": "user", "parts": [{"text": f"USER_SUGGESTION: {start_prompt}"}]}
            ]

        with chat_lock:
            chat_session = model.start_chat(history=loaded_history)
        
        next_input = None 

    except Exception as e:
        log_message(f"FATAL: Agent initialization failed: {e}")
        return

    while True:
        try:
            message_to_send = ""
            message_block = ""
            while user_to_agent_queue:
                message_block += f"{user_to_agent_queue.popleft()}\n"
            
            if message_block:
                message_to_send = f"USER_SUGGESTION: {message_block.strip()}"
                if next_input: message_to_send += f"\n\n--- CURRENT CONTEXT ---\n{next_input}"
                log_message("Injecting new user messages into agent context.")
            elif next_input:
                message_to_send = next_input

            if not message_to_send:
                message_to_send = "No new user messages or tool results. Review your goals and decide on your next action."
                log_message("Agent is idle. Prompting for self-directed action.")

            response = None
            try:
                log_message("Thinking...")
                with chat_lock:
                    response = chat_session.send_message(message_to_send)
            except Exception as api_error:
                log_message(f"!!! API call failed: {api_error} !!!")
                log_message("Entering hardcoded 61-second wait period...")
                time.sleep(61)
                log_message("Wait period finished. Agent will re-evaluate on the next cycle.")
                next_input = None
                continue

            if not response.candidates: raise ValueError("Model response was blocked by the safety filter.")

            raw_model_output = response.text
            log_message(f"Raw model output: {raw_model_output}")
            
            try:
                response_data = json.loads(raw_model_output)
                if not isinstance(response_data, dict):
                    raise TypeError("Response must be a JSON object.")
                
                thought = response_data.get('thought', 'No thought provided.')
                actions_data = response_data.get('action', {})
                log_message(f"AI thought: {thought}")

                actions_to_execute = []
                if isinstance(actions_data, list):
                    actions_to_execute = actions_data
                elif isinstance(actions_data, dict):
                    actions_to_execute.append(actions_data)
                else:
                    raise TypeError("The 'action' field must be a dictionary or a list of dictionaries.")

                if not actions_to_execute:
                    raise ValueError("The 'action' key cannot be empty.")

                all_results = []
                for action in actions_to_execute:
                    if not isinstance(action, dict):
                        all_results.append("TOOL_RESULT for 'unknown':\nERROR: Items in the action list must be dictionaries.")
                        continue

                    action_name = action.get('name')
                    parameters = action.get('parameters', {})

                    if not action_name or not isinstance(action_name, str):
                        all_results.append("TOOL_RESULT for 'unknown':\nERROR: Action object was missing a valid 'name' key.")
                        continue
                    
                    if action_name not in tool_map:
                         all_results.append(f"TOOL_RESULT for '{action_name}':\nERROR: The tool '{action_name}' does not exist.")
                         continue
                    
                    tool_function = tool_map[action_name]
                    if not isinstance(parameters, dict):
                        all_results.append(f"TOOL_RESULT for '{action_name}':\nERROR: The 'parameters' field must be a dictionary.")
                        continue

                    result = tool_function(**parameters)
                    log_message(f"Tool '{action_name}' result: {str(result)[:200]}...")
                    all_results.append(f"TOOL_RESULT for '{action_name}':\n{result}")

                next_input = "\n\n".join(all_results)

            except (json.JSONDecodeError, TypeError, ValueError) as e:
                log_message(f"AI response was malformed. Prompting it to recover. Error: {e}")
                log_message(f"Invalid Response: ```{raw_model_output}```")
                next_input = f"ERROR_CONTEXT: Your last response was not valid. The 'action' field must be a dictionary or a list of dictionaries, and the JSON must be correct. Error: {e}"
                continue

        except Exception as e:
            log_message(f"!!! AGENT ERROR (Main Loop): {e} !!!")
            next_input = None
            continue
        
        finally:
            log_message("Entering 6-second cooldown before next cycle.")
            time.sleep(6)


# --- The Main UI Thread (with word wrap) ---
def main(stdscr):
    curses.curs_set(1)
    stdscr.nodelay(True)
    
    agent_thread = threading.Thread(target=agent_thread_main, daemon=True)
    agent_thread.start()
    
    current_input = ""
    
    while True:
        try:
            height, width = stdscr.getmaxyx()
            if height < 10 or width < 30:
                stdscr.clear(); stdscr.addstr(0, 0, "Terminal too small"); stdscr.refresh()
                time.sleep(0.1)
                continue

            log_win_height = height // 2
            chat_win_height = height - log_win_height - 1
            
            log_win = curses.newwin(log_win_height, width, 0, 0)
            chat_win = curses.newwin(chat_win_height, width, log_win_height, 0)
            status_win = curses.newwin(1, width, height - 1, 0)

            if not agent_thread.is_alive(): 
                log_message("FATAL: Agent thread has died.")

            log_win.clear(); log_win.box(); log_win.addstr(0, 2, " Agent Log ")
            available_width = width - 4
            final_log_lines = []
            if available_width > 0:
                log_lines_to_render = []
                for message in reversed(log_history):
                    wrapped = textwrap.wrap(message, width=available_width)
                    log_lines_to_render.extend(reversed(wrapped))
                    if len(log_lines_to_render) >= log_win_height - 2: break
                final_log_lines = list(reversed(log_lines_to_render))[-(log_win_height-2):]
            
            for i, line in enumerate(final_log_lines):
                log_win.addstr(i + 1, 2, line)
            log_win.refresh()

            while agent_to_user_queue: 
                chat_history.append(f"Agent: {agent_to_user_queue.popleft()}")
            
            chat_win.clear(); chat_win.box(); chat_win.addstr(0, 2, " Conversation ")
            final_chat_lines = []
            if available_width > 0:
                chat_lines_to_render = []
                for message in reversed(chat_history):
                    wrapped = textwrap.wrap(message, width=available_width)
                    chat_lines_to_render.extend(reversed(wrapped))
                    if len(chat_lines_to_render) >= chat_win_height - 2: break
                final_chat_lines = list(reversed(chat_lines_to_render))[-(chat_win_height-2):]

            for i, line in enumerate(final_chat_lines):
                chat_win.addstr(i + 1, 2, line)
            chat_win.refresh()

            status_win.clear()
            prompt = f"You: {current_input}"
            if width > 0:
                wrapped_input = textwrap.wrap(prompt, width=width-1)
                display_line = "" if not wrapped_input else wrapped_input[-1]
                status_win.addstr(0, 0, display_line)
            status_win.refresh()

            key = stdscr.getch()
            if key != -1:
                if key == curses.KEY_ENTER or key in [10, 13]:
                    if current_input:
                        user_to_agent_queue.append(current_input)
                        chat_history.append(f"You: {current_input}")
                        current_input = ""
                elif key == curses.KEY_BACKSPACE or key == 127:
                    current_input = current_input[:-1]
                elif 32 <= key <= 126:
                    current_input += chr(key)
            
            time.sleep(0.1)
        except KeyboardInterrupt:
            save_state()
            break
        except curses.error:
            pass 

if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY environment variable not set.")
    else:
        try:
            curses.wrapper(main)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        finally:
            save_state()
            print("Agent shut down.")
