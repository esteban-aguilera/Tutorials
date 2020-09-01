from qiskit import IBMQ

with open(".ibm_credentials.txt","r") as f:
    credentials = f.readline()[:-1]

IBMQ.save_account(credentials)
