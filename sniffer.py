import socket
import struct
import textwrap

def main():
    # Use AF_PACKET if on Linux/ AF_INET on Windows
    conn = socket.socket(socket.AF_PACKET, socket.SOCK_RAW,  socket.IPPROTO_IP )
    
    while True:
        raw_data, addr = conn.recvfrom(65536)
        print("test")
        dest_mac, src_mac, eth_proto, data = ethernet_frame(raw_data)
        print('\nEthernet Frame:')
        print('Destination : {}, Source: {}, Protocol: {}'.format(dest_mac, src_mac, eth_proto))


# Unpack ethernet frame
def ethernet_frame(data):
    dest_mac, src_mac, proto = struct.unpack('! 6s 6s H', data[:14])
    return get_mac_addr(dest_mac), get_mac_addr(src_mac), socket.htons(proto), data[14:]

# Return properly formated MAC address (format AA:BB:CC:DD:EE:FF)
def get_mac_addr(bytes_addr):
    bytes_str = map('{:02x}'.format, bytes_addr)
    return ':'.join(bytes_str).upper()

main()