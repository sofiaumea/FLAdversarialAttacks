import multiprocessing
import subprocess
#Call this client before any attack is run(change the target parameter
# in the Process if you want to change what type of client to start)

#The program can be run with three different datasets
#and therefore three different types of clients with corresponding
#attack files.

def start_client(client_id):
    subprocess.call(["python3", "client.py", str(client_id)])

def start_client_one_rw_attack(client_id):
    if(client_id == 1):
        subprocess.call(["python3", "clientRWAttack.py", str(client_id)])
    else:
        subprocess.call(["python3", "client.py", str(client_id)])

def start_client_two_rw_attack(client_id):
    if client_id in [1, 3]:
        subprocess.call(["python3", "clientRWAttack.py", str(client_id)])
    else:
        subprocess.call(["python3", "client.py", str(client_id)])

def start_client_three_rw_attack(client_id):
    if client_id in [1, 3, 5]:
        subprocess.call(["python3", "clientRWAttack.py", str(client_id)])
    else:
        subprocess.call(["python3", "client.py", str(client_id)])

def start_client_four_rw_attack(client_id):
    if client_id in [1, 3, 5, 7]:
        subprocess.call(["python3", "clientRWAttack.py", str(client_id)])
    else:
        subprocess.call(["python3", "client.py", str(client_id)])

def start_client_one_lf_attack(client_id):
    if(client_id == 1):
        subprocess.call(["python3", "clientLFAttack.py", str(client_id)])
    else:
        subprocess.call(["python3", "client.py", str(client_id)])

def start_client_two_lf_attack(client_id):
    if client_id in [1, 2]:
        subprocess.call(["python3", "clientLFAttack.py", str(client_id)])
    else:
        subprocess.call(["python3", "client.py", str(client_id)])
        
def start_client_three_lf_attack(client_id):
    if client_id in [1, 2, 3]:
        subprocess.call(["python3", "clientLFAttack.py", str(client_id)])
    else:
        subprocess.call(["python3", "client.py", str(client_id)])

def start_client_four_lf_attack(client_id):
    if client_id in [1, 2, 3, 4]:
        subprocess.call(["python3", "clientLFAttack.py", str(client_id)])
    else:
        subprocess.call(["python3", "client.py", str(client_id)])
    
        
if __name__ == '__main__':
    num_clients = 8

    # Start multiple clients concurrently
    processes = []
    for client_id in range(1, num_clients + 1):
        process = multiprocessing.Process(target=start_client, args=(client_id,))
        processes.append(process)
        process.start()

    # Wait for all client processes to complete
    for process in processes:
        process.join()