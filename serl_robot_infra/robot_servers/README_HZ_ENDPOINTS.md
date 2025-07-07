# Hz Monitoring Endpoints

This document describes the new Hz monitoring endpoints added to the Franka Robot Server.

## Overview

Two new endpoints have been added to monitor and display Hz (frequency) values:
- `/env_hz` - Environment Hz monitoring
- `/hand_hz` - Hand Hz monitoring

These endpoints allow external clients to send Hz values to the server, which are then displayed prominently in the web interface.

## Endpoints

### POST /env_hz

**Description:** Receives and stores the environment Hz value.

**Request Format:**
```json
{
    "env_hz": 30.5
}
```

**Response:**
- Success: `"Updated env_hz to 30.5"`
- Error: `"Missing env_hz parameter"` (400) or error message (500)

**Example using curl:**
```bash
curl -X POST http://localhost:5000/env_hz \
  -H "Content-Type: application/json" \
  -d '{"env_hz": 30.5}'
```

**Example using Python:**
```python
import requests

response = requests.post('http://localhost:5000/env_hz', 
                        json={'env_hz': 30.5})
print(response.text)
```

### POST /hand_hz

**Description:** Receives and stores the hand Hz value.

**Request Format:**
```json
{
    "hand_hz": 60.0
}
```

**Response:**
- Success: `"Updated hand_hz to 60.0"`
- Error: `"Missing hand_hz parameter"` (400) or error message (500)

**Example using curl:**
```bash
curl -X POST http://localhost:5000/hand_hz \
  -H "Content-Type: application/json" \
  -d '{"hand_hz": 60.0}'
```

**Example using Python:**
```python
import requests

response = requests.post('http://localhost:5000/hand_hz', 
                        json={'hand_hz': 60.0})
print(response.text)
```

## Web Interface Integration

The Hz values are displayed prominently at the top of the web interface in a dedicated section with:
- Large, easy-to-read numbers
- Clear labels ("Environment Hz" and "Hand Hz")
- Real-time updates when new values are received
- Values are also shown in the "System Info" panel

## Data Retrieval

The Hz values are included in the following existing endpoints:
- `GET /get_all_data` - Returns all robot data including Hz values
- `POST /getstate` - Returns current state including Hz values

The values are returned as floating-point numbers in the JSON response.

## Usage Scenarios

1. **Real-time Monitoring:** External systems can continuously send Hz values to track performance
2. **Performance Tracking:** Monitor environment and hand control loop frequencies
3. **Debugging:** Identify performance bottlenecks by tracking Hz values over time
4. **System Health:** Use Hz values as indicators of system health and responsiveness

## Error Handling

- Invalid JSON: Returns 500 error with error message
- Missing parameter: Returns 400 error with "Missing env_hz parameter" or "Missing hand_hz parameter"
- Invalid number format: Server will attempt to convert to float, may return 500 if conversion fails

## Notes

- Hz values are stored as floating-point numbers
- Values are displayed with 2 decimal places in the web interface
- The server maintains the latest received values until new ones are sent
- Values persist until the server is restarted
- No validation is performed on the range of Hz values (can be negative or very large) 