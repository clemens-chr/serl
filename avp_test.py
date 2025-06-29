from avp_stream import VisionProStreamer
avp_ip = "192.168.1.10"   # example IP 
s = VisionProStreamer(ip = avp_ip, record = True)

while True:
    r = s.latest
    pinch_right = r['right_pinch_distance'] - 0.02
    
    if pinch_right <= 0.001:
        actions_hand = 0.0
    elif pinch_right >= 0.1:
        actions_hand = 0.1
    else:
        # Linear interpolation: (0.01, 0.0) to (0.05, 0.05)
        actions_hand = pinch_right
        
    interpolated_actions_hand = (actions_hand - 0.0) / (0.1 - 0.0) * 2 - 1
        
    print(interpolated_actions_hand)