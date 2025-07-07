let isConnected = false;
let updateInterval;
let currentPose = [0, 0, 0.5, 0, 0, 0, 1]; // Default pose
let poseUpdateLocked = false; // Flag to prevent updates during movement

function updateConnectionStatus(connected) {
    isConnected = connected;
    const statusElement = document.getElementById('connection-status');
    if (connected) {
        statusElement.textContent = 'Connected';
        statusElement.className = 'status connected';
    } else {
        statusElement.textContent = 'Disconnected';
        statusElement.className = 'status disconnected';
    }
}

function displayResponse(message, type = 'info') {
    const responseDisplay = document.getElementById('response-display');
    const timestamp = new Date().toLocaleTimeString();
    
    // Clear previous styling
    responseDisplay.className = 'response-display';
    
    // Add new styling based on type
    if (type === 'success') {
        responseDisplay.classList.add('response-success');
    } else if (type === 'error') {
        responseDisplay.classList.add('response-error');
    } else {
        responseDisplay.classList.add('response-info');
    }
    
    // Format message
    let formattedMessage;
    if (typeof message === 'object') {
        formattedMessage = JSON.stringify(message, null, 2);
    } else {
        formattedMessage = message;
    }
    
    responseDisplay.textContent = `[${timestamp}] ${formattedMessage}`;
    
    // Auto-scroll to bottom
    responseDisplay.scrollTop = responseDisplay.scrollHeight;
}

function formatValue(value) {
    if (Array.isArray(value)) {
        if (value.length > 7) {
            // For large arrays like jacobian, show dimensions
            return `Array [${value.length}] (${value[0]?.length || 'N/A'} cols)`;
        } else {
            // For small arrays, show values with limited precision
            return '[' + value.map(v => typeof v === 'number' ? v.toFixed(3) : v).join(', ') + ']';
        }
    } else if (typeof value === 'number') {
        return value.toFixed(4);
    } else if (typeof value === 'boolean') {
        return value ? 'Yes' : 'No';
    }
    return value;
}

function createDataRow(label, value) {
    return `
        <div class="data-row">
            <span class="data-label">${label}:</span>
            <span class="data-value ${Array.isArray(value) ? 'array-value' : ''}">${formatValue(value)}</span>
        </div>
    `;
}

function updatePoseDisplay() {
    // Update the display values from sliders
    document.getElementById('xValue').textContent = parseFloat(document.getElementById('xSlider').value).toFixed(3);
    document.getElementById('yValue').textContent = parseFloat(document.getElementById('ySlider').value).toFixed(3);
    document.getElementById('zValue').textContent = parseFloat(document.getElementById('zSlider').value).toFixed(3);
    document.getElementById('qxValue').textContent = parseFloat(document.getElementById('qxSlider').value).toFixed(3);
    document.getElementById('qyValue').textContent = parseFloat(document.getElementById('qySlider').value).toFixed(3);
    document.getElementById('qzValue').textContent = parseFloat(document.getElementById('qzSlider').value).toFixed(3);
    document.getElementById('qwValue').textContent = parseFloat(document.getElementById('qwSlider').value).toFixed(3);
}

function loadCurrentPose() {
    // Load the current robot pose into the sliders
    if (currentPose && currentPose.length >= 7) {
        document.getElementById('xSlider').value = currentPose[0];
        document.getElementById('ySlider').value = currentPose[1];
        document.getElementById('zSlider').value = currentPose[2];
        document.getElementById('qxSlider').value = currentPose[3];
        document.getElementById('qySlider').value = currentPose[4];
        document.getElementById('qzSlider').value = currentPose[5];
        document.getElementById('qwSlider').value = currentPose[6];
        updatePoseDisplay();
        displayResponse('Current pose loaded into sliders', 'success');
    } else {
        displayResponse('No current pose data available', 'error');
    }
}

function updateDisplay(data) {
    // Store current pose for loading into sliders
    if (data.pos && data.pos.length >= 7 && !poseUpdateLocked) {
        currentPose = [...data.pos];
    }

    // Update Hz values
    if (data.env_hz !== undefined) {
        document.getElementById('env-hz-value').textContent = data.env_hz.toFixed(2);
    }
    if (data.hand_hz !== undefined) {
        document.getElementById('hand-hz-value').textContent = data.hand_hz.toFixed(2);
    }

    // Position & Orientation
    const positionHtml = `
        ${createDataRow('Position X', data.pos[0])}
        ${createDataRow('Position Y', data.pos[1])}
        ${createDataRow('Position Z', data.pos[2])}
        ${createDataRow('Quaternion X', data.pos[3])}
        ${createDataRow('Quaternion Y', data.pos[4])}
        ${createDataRow('Quaternion Z', data.pos[5])}
        ${createDataRow('Quaternion W', data.pos[6])}
    `;
    document.getElementById('position-data').innerHTML = positionHtml;

    // Velocity & Forces
    const velocityHtml = `
        ${createDataRow('Linear Velocity', data.vel.slice(0, 3))}
        ${createDataRow('Angular Velocity', data.vel.slice(3, 6))}
        ${createDataRow('Force', data.force)}
        ${createDataRow('Torque', data.torque)}
    `;
    document.getElementById('velocity-data').innerHTML = velocityHtml;

    // Joint States
    const jointHtml = `
        ${createDataRow('Joint Positions', data.q)}
        ${createDataRow('Joint Velocities', data.dq)}
        ${createDataRow('Jacobian Size', `${data.jacobian.length}x${data.jacobian[0]?.length || 0}`)}
    `;
    document.getElementById('joint-data').innerHTML = jointHtml;

    // System Info
    const systemHtml = `
        ${createDataRow('Robot IP', data.robot_ip)}
        ${createDataRow('Gripper Type', data.gripper_type)}
        ${createDataRow('Gripper Position', data.gripper_pos)}
        ${createDataRow('Impedance Running', data.impedance_running)}
        ${createDataRow('Reset Joint Target', data.reset_joint_target)}
        ${createDataRow('Environment Hz', data.env_hz)}
        ${createDataRow('Hand Hz', data.hand_hz)}
    `;
    document.getElementById('system-data').innerHTML = systemHtml;

    // Update timestamp
    const now = new Date();
    document.getElementById('last-update').textContent = `Last update: ${now.toLocaleTimeString()}`;
}

async function fetchData() {
    try {
        const response = await fetch('/get_all_data');
        if (response.ok) {
            const data = await response.json();
            updateDisplay(data);
            updateConnectionStatus(true);
        } else {
            updateConnectionStatus(false);
            displayResponse('Failed to fetch data from server', 'error');
        }
    } catch (error) {
        console.error('Error fetching data:', error);
        updateConnectionStatus(false);
        displayResponse(`Connection error: ${error.message}`, 'error');
    }
}

async function sendCommand(endpoint) {
    try {
        displayResponse(`Sending command to ${endpoint}...`, 'info');
        const response = await fetch(endpoint, { method: 'POST' });
        const result = await response.text();
        
        if (response.ok) {
            displayResponse(`${endpoint}: ${result}`, 'success');
        } else {
            displayResponse(`${endpoint} error: ${result}`, 'error');
        }
        
        console.log(`${endpoint}: ${result}`);
        // Immediately fetch new data after command
        setTimeout(fetchData, 100);
    } catch (error) {
        console.error(`Error sending command to ${endpoint}:`, error);
        displayResponse(`${endpoint} error: ${error.message}`, 'error');
    }
}

async function sendDataCommand(endpoint) {
    try {
        displayResponse(`Requesting data from ${endpoint}...`, 'info');
        const method = endpoint === '/get_all_data' ? 'GET' : 'POST';
        const response = await fetch(endpoint, { method: method });
        
        if (response.ok) {
            const data = await response.json();
            displayResponse(`${endpoint} response: ${JSON.stringify(data, null, 2)}`, 'success');
        } else {
            const errorText = await response.text();
            displayResponse(`${endpoint} error: ${errorText}`, 'error');
        }
    } catch (error) {
        console.error(`Error fetching from ${endpoint}:`, error);
        displayResponse(`${endpoint} error: ${error.message}`, 'error');
    }
}

async function sendHzCommand(endpoint) {
    try {
        const hzValue = prompt(`Enter ${endpoint === '/env_hz' ? 'environment' : 'hand'} Hz value:`);
        if (hzValue === null) return; // User cancelled
        
        const value = parseFloat(hzValue);
        if (isNaN(value)) {
            displayResponse('Please enter a valid number', 'error');
            return;
        }
        
        displayResponse(`Sending ${endpoint === '/env_hz' ? 'environment' : 'hand'} Hz: ${value}...`, 'info');
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                [endpoint === '/env_hz' ? 'env_hz' : 'hand_hz']: value 
            })
        });
        const result = await response.text();
        
        if (response.ok) {
            displayResponse(`${endpoint}: ${result}`, 'success');
            // Immediately fetch new data to update the display
            setTimeout(fetchData, 100);
        } else {
            displayResponse(`${endpoint} error: ${result}`, 'error');
        }
    } catch (error) {
        console.error(`Error sending Hz command to ${endpoint}:`, error);
        displayResponse(`${endpoint} error: ${error.message}`, 'error');
    }
}

async function moveGripper() {
    const pos = document.getElementById('gripper-pos').value;
    try {
        displayResponse(`Moving gripper to position ${pos}...`, 'info');
        const response = await fetch('/move_gripper', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ gripper_pos: parseInt(pos) })
        });
        const result = await response.text();
        
        if (response.ok) {
            displayResponse(`Move gripper: ${result}`, 'success');
        } else {
            displayResponse(`Move gripper error: ${result}`, 'error');
        }
        
        console.log(`Move gripper: ${result}`);
        setTimeout(fetchData, 100);
    } catch (error) {
        console.error('Error moving gripper:', error);
        displayResponse(`Move gripper error: ${error.message}`, 'error');
    }
}

async function moveToPose() {
    // Get pose from sliders
    const pose = [
        parseFloat(document.getElementById('xSlider').value),
        parseFloat(document.getElementById('ySlider').value),
        parseFloat(document.getElementById('zSlider').value),
        parseFloat(document.getElementById('qxSlider').value),
        parseFloat(document.getElementById('qySlider').value),
        parseFloat(document.getElementById('qzSlider').value),
        parseFloat(document.getElementById('qwSlider').value)
    ];
    
    try {
        // Lock pose updates during movement to prevent slider values from changing
        poseUpdateLocked = true;
        
        displayResponse(`Moving to pose: [${pose.map(v => v.toFixed(3)).join(', ')}]...`, 'info');
        const response = await fetch('/pose', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ arr: pose })
        });
        const result = await response.text();
        
        if (response.ok) {
            displayResponse(`Move to pose: ${result}`, 'success');
        } else {
            displayResponse(`Move to pose error: ${result}`, 'error');
        }
        
        console.log(`Move to pose: ${result}`);
        
        // Wait a bit longer before unlocking and fetching data
        setTimeout(() => {
            poseUpdateLocked = false;
            fetchData();
        }, 500);
    } catch (error) {
        console.error('Error moving to pose:', error);
        displayResponse(`Move to pose error: ${error.message}`, 'error');
        poseUpdateLocked = false;
    }
}

async function updateComplianceParams() {
    const transStiffness = parseFloat(document.getElementById('trans-stiffness').value) || 3000;
    const rotStiffness = parseFloat(document.getElementById('rot-stiffness').value) || 300;
    
    const params = {
        translational_stiffness: transStiffness,
        rotational_stiffness: rotStiffness
    };
    
    try {
        displayResponse(`Updating compliance parameters: ${JSON.stringify(params)}...`, 'info');
        const response = await fetch('/update_param', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        });
        const result = await response.text();
        
        if (response.ok) {
            displayResponse(`Update parameters: ${result}`, 'success');
        } else {
            displayResponse(`Update parameters error: ${result}`, 'error');
        }
        
        console.log(`Update parameters: ${result}`);
    } catch (error) {
        console.error('Error updating compliance parameters:', error);
        displayResponse(`Update parameters error: ${error.message}`, 'error');
    }
}

// Start updating data every 100ms
function startUpdating() {
    displayResponse('Robot Monitor initialized. Starting data updates...', 'info');
    fetchData(); // Initial fetch
    updateInterval = setInterval(fetchData, 100);
    
    // Initialize pose display
    updatePoseDisplay();
}

// Initialize when page loads
window.addEventListener('load', startUpdating); 