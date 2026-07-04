import netifaces as ni
import os
import time

# Assign static IP with netmask
os.system('sudo ifconfig eth0 192.168.1.2 netmask 255.255.255.0 up')

# Wait for interface to settle
time.sleep(2)

# Get IP address
try:
    ip_info = ni.ifaddresses('eth0')
    if ni.AF_INET in ip_info:
        ip = ip_info[ni.AF_INET][0]['addr']
        print(f"IP Address: {ip}")
    else:
        print("No IPv4 address assigned to eth0")
except Exception as e:
    print(f"Error getting IP: {e}")
