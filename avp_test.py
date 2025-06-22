from avp_stream import VisionProStreamer
avp_ip = "192.168.1.10"   # example IP 
s = VisionProStreamer(ip = avp_ip, record = True)

while True:
    r = s.latest
    print(r['right_pinch_distance'])