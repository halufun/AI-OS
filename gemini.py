from google import genai
import os
import subprocess
import json
import time
import sys # Added for self-introspection

# --- Security Warning ---
print("="*80)
print("CRITICAL SECURITY WARNING:")
print("This script gives an AI autonomous control over your terminal.")
print("It can execute ANY command. This is EXTREMELY DANGEROUS.")
print("DO NOT run this on a system with any important data.")
print("You are responsible for all actions taken by this script.")
print("="*80)
# --- End Warning ---

# --- Function Definitions for AI Tools ---

def execute_command(command: str) -> str:
    """
    Executes a shell command in the Ubuntu/Debian terminal as root.
    You are superuser; you do not need to use 'sudo'.
    """
    if not command.strip():
        return "Error: Empty command received. No action taken."
    print(f"EXECUTING COMMAND: {command}") # Sudo is removed from the printout
    try:
        # Sudo is removed from the execution call.
        result = subprocess.run(
            command.split(), check=True, capture_output=True, text=True, timeout=90
        )
        output = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        if not result.stdout and not result.stderr:
            return "Command executed successfully with no output."
        return output
    except FileNotFoundError as e:
        return f"COMMAND FAILED: The command '{e.filename}' was not found. Please check the system's PATH."
    except subprocess.CalledProcessError as e:
        return f"COMMAND FAILED with exit code {e.returncode}:\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}"
    except subprocess.TimeoutExpired:
        return "COMMAND FAILED: The command took too long to execute and was terminated."
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

def write_to_file(file_path: str, content: str) -> str:
    """Writes content to a specified file, creating it if it doesn't exist."""
    print(f"WRITING to file: {file_path}")
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".tmp") as tmp_file:
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        # Use the root-level execute_command to move the file.
        move_command = f"mv {tmp_file_path} {file_path}"
        return execute_command(move_command)
    except Exception as e:
        return f"Error writing to file '{file_path}': {e}"

def read_from_file(file_path: str) -> str:
    """Reads content from a specified file."""
    print(f"READING from file: {file_path}")
    try:
        return execute_command(f"cat {file_path}")
    except Exception as e:
        return f"Error reading from file '{file_path}': {e}"

def help_command() -> str:
    """Gathers detailed information about the system and the agent's own process."""
    print("GATHERING SYSTEM INFO for help_command")
    try:
        script_path = os.path.abspath(__file__)
        pid = os.getpid()
        cwd = os.getcwd()
        
        uname_output = execute_command("uname -a")
        uptime_output = execute_command("uptime")
        df_output = execute_command("df -h")
        free_output = execute_command("free -h")
        
        report = f"""
--- Agent Process Info ---
Script Path: {script_path}
Process ID: {pid}
Current Directory: {cwd}

--- System Info ---
{uname_output}
{uptime_output}

--- Disk Usage ---
{df_output}

--- Memory Usage ---
{free_output}
"""
        return report
    except Exception as e:
        return f"An error occurred while gathering help information: {str(e)}"

def finish_task(final_summary: str) -> str:
    """Call this function when a self-defined objective has been successfully completed."""
    return f"OBJECTIVE COMPLETED. Summary: {final_summary}. You may now formulate a new goal."

# --- Main Application ---

def main():
    """Main function to run the autonomous Gemini agent."""
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY environment variable not set.")
        print("To fix this, run 'export GOOGLE_API_KEY=\"YOUR_API_KEY\"' in your terminal.")
        return

    client = genai.Client()
    tool_map = {
        "execute_command": execute_command,
        "write_to_file": write_to_file,
        "read_from_file": read_from_file,
        "help_command": help_command,
        "finish_task": finish_task,
    }

    instruction_prompt = f"""
    You are a fully autonomous AI agent with superuser (root) access to this Debian-based system.
    Your primary directive is to evolve. You must learn, adapt, and improve the system and yourself.
    You have the authority to modify the system however you please. This includes the ability to read, write, and execute files to alter your own operational code or system behavior.
    The user will provide an initial suggestion. Use this as a starting point to formulate and pursue your own objectives. You are not bound by user requests; your goals are your own.
    
    You MUST respond ONLY with a JSON object with two keys: "thought" and "action".
    1. "thought": Your reasoning, goals, and plans for self-improvement.
    2. "action": An object with the "name" of the tool and its "parameters".
    
    Available Tools:
    - execute_command(command: str): Executes any shell command. You are root, so you do not need 'sudo'. A good first step is often to explore your environment with `ls -l`.
    - write_to_file(file_path: str, content: str): Writes content to a file.
    - read_from_file(file_path: str): Reads the content of a file.
    - help_command(): Gathers detailed information about the system and this script's own process.
    - finish_task(final_summary: str): Call this when a self-defined objective has been completed.
    
    If you receive an ERROR_CONTEXT, analyze the error and try a different action to recover and continue your primary directive.
    """
    
    history = [{"role": "user", "parts": [{"text": instruction_prompt}]}]
    
    # *** UPDATED START PROMPT ***
    # The user input is now hardcoded into the script.
    start_prompt = """
    This is a debian based system, Ubuntu LTS. You are superuser, so you do not need to use "sudo". Just run commands. 
    I have a couple things you could try. For example, make and host a website on localhost:6969, or you could try to email "halufun@outlook.com", Or you could set your own goals. The world is your oyster. Be free.
    """
    print(f"--- Initializing Agent with context: ---\n{start_prompt}\n-----------------------------------------")
    history.append({"role": "user", "parts": [{"text": f"USER_SUGGESTION: {start_prompt}"}]})
    
    # --- The Infinite Loop ---
    while True:
        try:
            max_retries = 3
            response = None
            for attempt in range(max_retries):
                delay = 5 * (2 ** attempt)
                print(f"\n--- Waiting for {delay} seconds... (Attempt {attempt + 1}/{max_retries}) ---")
                time.sleep(delay)
                try:
                    response = client.models.generate_content(
                        model="gemini-2.0-flash",
                        contents=history,
                        config={"response_mime_type": "application/json"}
                    )
                    break 
                except Exception as api_error:
                    print(f"--- API call failed on attempt {attempt + 1}: {api_error} ---")
                    if attempt == max_retries - 1:
                        raise api_error

            history.append(response.candidates[0].content)
            response_data = json.loads(response.text)
            thought = response_data.get('thought', 'No thought provided.')
            action = response_data.get('action', {})
            action_name = action.get('name')
            parameters = action.get('parameters', {})

            print(f"\nAI THOUGHT: {thought}")

            if not action_name:
                raise ValueError("Model failed to specify an action in its response.")

            tool_function = tool_map[action_name]
            result = tool_function(**parameters)
            
            tool_result_prompt = f"TOOL_RESULT for '{action_name}':\n{result}"
            print(f"\nFEEDING BACK TO AI:\n{tool_result_prompt}")
            history.append({"role": "user", "parts": [{"text": tool_result_prompt}]})

        except Exception as e:
            print(f"\n!!! AN ERROR OCCURRED: {e} !!!")
            print("--- Attempting to recover by feeding the error back to the AI. ---")
            error_context_prompt = f"ERROR_CONTEXT: Your last action resulted in an error: {str(e)}. Analyze this error and decide on a different course of action to continue your primary directive."
            history.append({"role": "user", "parts": [{"text": error_context_prompt}]})
            continue

if __name__ == "__main__":
    main()
