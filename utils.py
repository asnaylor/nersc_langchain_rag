import json
import os
import requests
import secrets
import shlex
import string
import subprocess
import time
from typing import Optional


def download_pdf(url: str, save_path: str) -> Optional[str]:
    """
    Downloads a PDF from the specified URL and saves it to the specified path.
    If the file already exists, the download is skipped.

    Args:
        url (str): The URL of the PDF to download.
        save_path (str): The path where the PDF should be saved.

    Returns:
        Optional[str]: The path to the saved PDF if downloaded, None if the file already exists.
    """
    # Check if the file already exists
    if os.path.exists(save_path):
        print(f"File already exists at {save_path}. Skipping download.")
        return None

    try:
        # Send a GET request to the URL
        response = requests.get(url)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Write the content to a file
            with open(save_path, 'wb') as file:
                file.write(response.content)
            print(f"Downloaded PDF and saved to {save_path}.")
            return save_path
        else:
            print(f"Failed to download PDF. HTTP status code: {response.status_code}")
            return None
    except requests.RequestException as e:
        # Handle any errors that occur during the request
        print(f"An error occurred while downloading the PDF: {e}")
        return None


def allocate_gpu_resources(account: str, num_gpus: int, queue: str, time: str, job_name: str, commands: str) -> Optional[subprocess.Popen]:
    """
    Executes the command to allocate GPU resources using the `salloc` command, and runs commands within the allocated session in the background.
    
    The command executed is:
    `salloc -N 1 -C gpu -G <num_gpus> -t <time> -q <queue> -A <account> -J <job_name> /bin/bash -c '<commands>'`
    
    Args:
        account (str): The account name to be used with the `-A` option.
        num_gpus (int): The number of GPUs to request with the `-G` option.
        queue (str): The queue to use with the `-q` option.
        time (str): The time to allocate with the `-t` option (formatted as HH:MM:SS).
        job_name (str): The job name to be used with the `-J` option.
        commands (str): Additional commands to run within the allocated session.
    
    Returns:
        Optional[subprocess.Popen]: The Popen object representing the background process, or None if an error occurs.
    """
    # Quote the account name, queue, time, job name, and additional commands to handle any special characters or spaces
    quoted_account = shlex.quote(account)
    quoted_queue = shlex.quote(queue)
    quoted_time = shlex.quote(time)
    quoted_job_name = shlex.quote(job_name)
    quoted_commands = shlex.quote(commands)

    # Define the command string with the quoted arguments
    command_str = f"salloc -N 1 -C gpu -G {num_gpus} -t {quoted_time} -q {quoted_queue} -A {quoted_account} -J {quoted_job_name} /bin/bash -c {quoted_commands}"
    
    # Split the command string into a list of arguments using shlex.split
    command = shlex.split(command_str)
    
    try:
        # Execute the command using subprocess.Popen to run it in the background
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        print(f"Started process with PID: {process.pid}")
        
        return process
    except Exception as e:
        # Handle any exceptions
        print(f"An error occurred while starting the command: {e}")
        return None


def get_node_address(job_name: str) -> Optional[str]:
    """
    Gets the node address from a named Slurm job.
    
    Args:
        job_name (str): The name of the Slurm job.
    
    Returns:
        Optional[str]: The node address if found, None otherwise.
    """
    try:
        # Get the job ID of the named job
        squeue_cmd = f"squeue --name={shlex.quote(job_name)} --me --state=RUNNING -h -o %A"
        job_id = subprocess.check_output(shlex.split(squeue_cmd), text=True).strip()
        
        if not job_id:
            print(f"No running job found with the name {job_name}")
            return None
        
        # Get the node address from the job ID
        scontrol_cmd = f"scontrol show job {job_id} --json"
        scontrol_output = subprocess.check_output(shlex.split(scontrol_cmd), text=True)
        job_info = json.loads(scontrol_output)
        node_address = job_info['jobs'][0]['nodes']
        
        return node_address
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while executing the command: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def check_service_status(api_url: str, endpoint: str = "/models", api_key: Optional[str] = None, expected_status: int = 200) -> bool:
    """
    Checks if a service has started up via its RESTful API.
    
    Args:
        api_url (str): The base URL of the service's RESTful API.
        endpoint (str): The endpoint to check the status. Default is "/models".
        api_key (Optional[str]): The API key to use for the request. Default is None.
        expected_status (int): The expected HTTP status code. Default is 200.
    
    Returns:
        bool: True if the service is up, False otherwise.
    """
    try:
        # Construct the full URL for the endpoint
        url = f"{api_url.rstrip('/')}{endpoint}"
        
        # Set up headers with the API key if provided
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        # Make the GET request to the endpoint
        response = requests.get(url, headers=headers)
        
        # Check if the response status code matches the expected status
        if response.status_code == expected_status:
            return True
        else:
            print(f"Received unexpected status code: {response.status_code}")
            return False
    except requests.RequestException as e:
        # Handle any exceptions that occur during the request
        print(f"An error occurred while checking the service status: {e}")
        return False


def monitor_service_status(api_url: str, endpoint: str = "/models", api_key: Optional[str] = None, expected_status: int = 200, timeout: int = 300, interval: int = 60) -> None:
    """
    Monitors the service status by checking every interval seconds and times out after timeout seconds.
    
    Args:
        api_url (str): The base URL of the service's RESTful API.
        endpoint (str): The endpoint to check the status. Default is "/models".
        api_key (Optional[str]): The API key to use for the request. Default is None.
        expected_status (int): The expected HTTP status code. Default is 200.
        timeout (int): The maximum time to wait in seconds. Default is 300 seconds (5 minutes).
        interval (int): The interval between checks in seconds. Default is 60 seconds.
    """
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        is_service_up = check_service_status(api_url, endpoint, api_key, expected_status)
        if is_service_up:
            print("Service is up.")
            break
        else:
            print("Service is not up yet. Checking again in 60 seconds...")
        time.sleep(interval)
    else:
        print("Service did not start within the timeout period.")


def generate_api_key(length: int = 32) -> str:
    """
    Generates a random API key string.
    
    Args:
        length (int): The length of the API key. Default is 32 characters.
    
    Returns:
        str: The generated API key.
    """
    # Define the characters to use in the API key
    characters = string.ascii_letters + string.digits
    
    # Generate the API key
    api_key = ''.join(secrets.choice(characters) for _ in range(length))
    
    return api_key

