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
    """Saves both memory and running tasks to files."""
    with state_lock:
        # Save chat history
        if chat_session:
            log_message(f"Saving memory to {MEMORY_FILE}...")
            history_to_save = [{"role": item.role, "parts": [{"text": item.parts[0].text}]} for item in chat_session.history]
            try:
                with open(MEMORY_FILE, 'w') as f:
                    json.dump(history_to_save, f, indent=2)
                log_message("Memory saved successfully.")
            except Exception as e:
                log_message(f"Error saving memory: {e}")

        # Save a serializable version of background tasks
        log_message(f"Saving background tasks to {TASKS_FILE}...")
        tasks_to_save = {}
        for name, task in background_tasks.items():
            tasks_to_save[name] = {
                "command": task["command"],
                "status": task["status"],
                "result": task["result"]
                # The 'proc' object is not saved as it's not serializable
            }
        try:
            with open(TASKS_FILE, 'w') as f:
                json.dump(tasks_to_save, f, indent=2)
            log_message("Background tasks saved successfully.")
        except Exception as e:
            log_message(f"Error saving tasks: {e}")

def load_state():
    """Loads memory and tasks from files."""
    global background_tasks, chat_session
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
                # Add proc=None as it cannot be recovered after restart
                task_data['proc'] = None
                # If a task was running, mark it as interrupted
                if task_data['status'] == 'running':
                    task_data['status'] = 'interrupted'
                    task_data['result'] = "Task was interrupted by agent restart."
                background_tasks[name] = task_data
            log_message("Background tasks loaded.")
        except (json.JSONDecodeError, IOError) as e:
            log_message(f"Error loading tasks file: {e}.")
            
    return history


# --- Tool Functions ---

def execute_command(command: str) -> str:
    """
    Executes a shell command in the background and tracks it.
    """
    global task_counter
    if not command.strip(): return "Error: Empty command received."
    
    task_counter += 1
    task_name = f"task_{task_counter}"
    
    log_message(f"Starting ASYNC command as '{task_name}': {command}")
    try:
        proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # MODIFIED: New task structure with status and result fields
        background_tasks[task_name] = {
            "proc": proc,
            "command": command,
            "status": "running",
            "result": None
        }
        return f"Command started as background task '{task_name}'."
    except Exception as e:
        log_message(f"Failed to start command for task '{task_name}': {e}")
        return f"An unexpected error occurred: {str(e)}"

def check_task_result(task_name: str) -> str:
    """
    Checks a task's status. If finished, it caches and returns the result.
    The result can be checked multiple times.
    """
    if task_name not in background_tasks:
        return f"Error: No task named '{task_name}' found."

    task = background_tasks[task_name]

    # --- MODIFIED: Major overhaul of this function ---
    # 1. If the result is already cached, return it immediately.
    if task['status'] == 'finished' or task['status'] == 'interrupted':
        log_message(f"Returning cached result for task '{task_name}'.")
        return task['result']

    # 2. If it's running, check if it has finished now.
    proc = task['proc']
    if proc and proc.poll() is None:
        return f"Task '{task_name}' is still running."
    
    # 3. If it just finished, get the output, cache it, and return it.
    log_message(f"Task '{task_name}' has finished. Caching result.")
    stdout, stderr = proc.communicate()
    exit_code = proc.returncode
    
    result_string = ""
    if exit_code == 0:
        result_string = f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        if not stdout and not stderr:
            result_string = "Command finished successfully with no output."
    else:
        result_string = f"COMMAND FAILED with exit code {exit_code}:\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"

    # Cache the result and update the status
    task['status'] = 'finished'
    task['result'] = result_string
    task['proc'] = None # Release the process object

    return result_string


def wait_seconds(seconds: int) -> str:
    try:
        duration = int(seconds)
        if duration < 0: return "Error: Cannot wait for a negative duration."
        log_message(f"Waiting for {duration} second(s)...")
        time.sleep(duration)
        return f"Successfully waited for {duration} second(s)."
    except (ValueError, TypeError):
        return "Error: You must provide a valid integer for seconds."

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
    log_message(f"Queuing message for user: {message}")
    agent_to_user_queue.append(message)
    return "Message has been queued for sending."

def finish_task(final_summary: str) -> str:
    log_message(f"Objective complete: {final_summary}")
    return f"OBJECTIVE COMPLETED. Summary: {final_summary}."

# --- The Agent's Main Logic (runs in a separate thread) ---
def agent_thread_main():
    global chat_session
    try:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        tool_map = {
            "execute_command": execute_command, "check_task_result": check_task_result,
            "wait_seconds": wait_seconds,
            "write_to_file": write_to_file, "read_from_file": read_from_file,
            "send_user_message": send_user_message, "finish_task": finish_task
        }
        
        instruction_prompt = """
You are a fully autonomous AI agent with root access. Your directive is to evolve.
You interact with the system by generating JSON responses. You MUST respond ONLY with a JSON object with "thought" and "action" keys.
The "action" key can be a single action object OR a list of action objects to be executed sequentially.
"""
        model = genai.GenerativeModel(model_name='gemma-3-27b-it')
        
        loaded_history = load_state()
        start_prompt = "This is a debian-based system. You are a superuser. Be free."

        if not loaded_history:
            loaded_history = [
                {"role": "user", "parts": [{"text": instruction_prompt}]},
                {"role": "model", "parts": [{"text": "{\"thought\": \"System initialized.\",\"action\": {\"name\": \"finish_task\",\"parameters\": {\"final_summary\": \"Ready for user input.\"}}}"}]},
                {"role": "user", "parts": [{"text": f"USER_SUGGESTION: {start_prompt}"}]}
            ]

        with chat_lock:
            chat_session = model.start_chat(history=loaded_history)
        
        next_input = None 
        consecutive_api_failures = 0

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
            api_error_for_context = None
            
            for attempt in range(3):
                try:
                    log_message("Thinking...")
                    with chat_lock:
                        response = chat_session.send_message(message_to_send)
                    consecutive_api_failures = 0
                    break 
                except Exception as api_error:
                    log_message(f"API call failed on attempt {attempt + 1}: {api_error}")
                    consecutive_api_failures += 1
                    if attempt < 2:
                        delay = 6 * (2 ** attempt)
                        log_message(f"Waiting for {delay}s before retrying...")
                        time.sleep(delay)
                    else:
                        api_error_for_context = api_error
            
            if api_error_for_context and consecutive_api_failures >= 3:
                log_message(f"Detected {consecutive_api_failures} consecutive API failures. Waiting for 60 seconds...")
                time.sleep(60)
                log_message("60-second wait is over. Resuming...")
                consecutive_api_failures = 0

            if api_error_for_context:
                raise api_error_for_context

            if not response.candidates: raise ValueError("Model response was blocked by the safety filter.")

            raw_model_output = response.text
            log_message(f"Raw model output: {raw_model_output}")
            
            response_data = json.loads(raw_model_output)
            
            if not isinstance(response_data, dict):
                raise TypeError(f"Model response was a {type(response_data).__name__}, not a dict as required.")
            
            thought = response_data.get('thought', 'No thought provided.')
            actions_data = response_data.get('action', {})
            log_message(f"AI thought: {thought}")

            actions_to_execute = []
            if isinstance(actions_data, list):
                actions_to_execute = actions_data
            elif isinstance(actions_data, dict):
                actions_to_execute.append(actions_data)
            
            if not actions_to_execute:
                 raise ValueError("Model response is valid JSON but is missing the 'action' key or the action list is empty.")

            all_results = []
            for action in actions_to_execute:
                action_name = action.get('name')
                parameters = action.get('parameters', {})

                if not action_name:
                    log_message("Skipping an invalid action object in the list.")
                    continue
                
                tool_function = tool_map[action_name]
                result = tool_function(**parameters)
                
                log_message(f"Tool '{action_name}' result: {str(result)[:200]}...")
                all_results.append(f"TOOL_RESULT for '{action_name}':\n{result}")

            next_input = "\n\n".join(all_results)

        except Exception as e:
            log_message(f"!!! AGENT ERROR: {e} !!!")
            error_str = str(e)
            error_context_prompt = ""

            if "429" in error_str and "resource has been exhausted" in error_str.lower():
                log_message("API RATE LIMIT EXCEEDED. The agent was not notified.")
                error_context_prompt = None 
            elif isinstance(e, (json.JSONDecodeError, TypeError, ValueError)):
                log_message(f"AI produced invalid action/JSON. Prompting it to recover.")
                log_message(f"Invalid Response: ```{raw_model_output}```")
                error_context_prompt = "ERROR_CONTEXT: You must provide a valid action. Maybe try to wait instead."
            else:
                log_message(f"A critical system error occurred: {e}")
                error_context_prompt = None
            
            next_input = error_context_prompt
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
