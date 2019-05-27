# Who is Who

# About
TPR Projecto 2019<br>
Universidade de Aveiro
Project where based on real-time traffic is given a profile with diferent permissions to internet content.

# Getting Started

### sniffer.py
`main` - Prints all the traffic captured

output:
```
Ethernet Frame:
	 - Destination : 7C:C3:85:7A:40:F7, Source: 60:57:18:17:10:FB, Protocol: 56710

Ethernet Frame:
	 - Destination : 00:00:00:00:00:00, Source: 00:00:00:00:00:00, Protocol: 8
	 - IPv4 Packet:
		 - Version: 4, Header Length: 20, TTL: 64
		 - Protocol: 17, Source: 127.0.0.53, Target: 127.0.0.1
	 - UDP segment: 
		 - Source Port: 53, Destination Port: 58259, Length: 65293

Ethernet Frame:
	 - Destination : 7C:C3:85:7A:40:F7, Source: 60:57:18:17:10:FB, Protocol: 8
	 - IPv4 Packet:
		 - Version: 4, Header Length: 20, TTL: 64
		 - Protocol: 6, Source: 192.168.1.70, Target: 216.58.201.164
	 - TCP Segment:
		 - Source Port: 40962, Destination Port: 443
		 - Sequence: 3602270188, Acknowledgement: 3720628367
		 - Flags: 
			 - URG: 0, ACK: 1, PSH: 1, RST: 0, SYN: 0, FIN:0
		 - Data: 
		\x17\x03\x03\x00\x22\x2a\x51\x81\xd9\x61\xf7\x40\xa3\x56\x1e\x11\xd8\x52\x11
		\x34\xae\x95\x04\xe9\x18\xc8\xdd\xaa\x40\x5e\x63\xe4\x4f\x3c\x3c\x36\xbd\xcd
		\x7a

...
```


## Autores
* Filipe Reis - filipereis96@ua.pt
* Gabriel Patrício - gpatricio@ua.pt
