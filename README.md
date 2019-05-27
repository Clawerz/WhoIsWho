# Who is Who
![](https://i.imgur.com/ln5BOMH.png)

# About
TPR Projecto 2019<br>
Universidade de Aveiro
Project where based on real-time traffic is given a profile with diferent permissions to internet content.

# Getting Started

### sniffer.py
`mainSniffer` - Prints all the traffic captured and information about it

output:
```
Ethernet Frame:
	 - Destination : 7C:C3:85:7A:40:F7, Source: 60:57:18:17:10:FB, Protocol: 56710

Ethernet Frame:
	 - Destination : 60:57:18:17:10:FB, Source: 7C:C3:85:7A:40:F7, Protocol: 8
	 - IPv4 Packet:
		 - Version: 4, Header Length: 20, TTL: 64
		 - Protocol: 17, Source: 192.168.1.1, Target: 192.168.1.70
	 - UDP segment: 
		 - Source Port: 53, Destination Port: 51839, Length: 19924
		 - Byte Count: 72


Ethernet Frame:
	 - Destination : 60:57:18:17:10:FB, Source: 7C:C3:85:7A:40:F7, Protocol: 8
	 - IPv4 Packet:
		 - Version: 4, Header Length: 20, TTL: 114
		 - Protocol: 6, Source: 52.109.68.12, Target: 192.168.1.70
	 - TCP Segment:
		 - Source Port: 443, Destination Port: 35876
		 - Sequence: 2284506426, Acknowledgement: 3737677542
		 - Flags: 
			 - URG: 0, ACK: 1, PSH: 1, RST: 0, SYN: 0, FIN:0
		 - Data: 
		 \x17\x03\x03\x00\x28\x00\x00\x00\x00\x00\x00\x00\x3d\x36\x78\xdb\xb4\x2f\x51	 			 \x6c\x53\x6c\xcf\xb8\xe7\x89\x6f\x69\x49\x32\x77\x08\xe0\x11\xb3\x06\xf2\x5e
		 \xbd\x85\xc3\x5e\x8d\x30\x8a
		 - Byte Count: 45

...
```

### genDataSet.py
`mainGen` - Creates a .dat file with the bytes captured per second
` genDataSet.py 'dataset name' 'dataset gen timer' `

output:
```
[0, 77, 0, 45, 7653, 53, 564, 0, 0, 27806, 4734, 13721, 4174, 11834, 355825, 70451, 77, 2181, 438227, 76862, 24, 8, 66, 0, 45, 77, 128, 0, 1036, 1128, 382, 8, 1692, 0, 45]
```

# Profile atribution
A profile is atributed based on the amount of bytes per second that the user sends/receives, this way it isn't need to verify the type(video, image, browsing...) of the packets but only the amount of bytes in transaction.

## Autores
* Filipe Reis - filipereis96@ua.pt
* Gabriel Patrício - gpatricio@ua.pt
