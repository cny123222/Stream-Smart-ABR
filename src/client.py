import os
import time
import logging
import threading
import requests
from urllib.parse import urlparse, urljoin, quote, unquote, parse_qs
import http.server
import socketserver
import re
import webbrowser
import json # For WebSocket messages

# --- WebSocket and AsyncIO ---
import asyncio
import websockets

import AES # Your AES decryption module

# --- Configuration ---
SERVER_HOST = '127.0.0.1'
SERVER_PORT = 8081
LOCAL_PROXY_HOST = '127.0.0.1'
LOCAL_PROXY_PORT = 8082
WEBSOCKET_PORT = 8083 # Port for WebSocket server
DOWNLOAD_DIR = "download" # Not heavily used with HLS.js direct playback
SOCKET_TIMEOUT_SECONDS = 10
VIDEO_TO_STREAM_NAME = "bbb_sunflower"

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger('HLSJS_Client_With_QoE')

# --- Local Proxy Server Globals ---
g_local_proxy_server_instance = None
g_proxy_runner_thread = None

# --- WebSocket Server Globals ---
g_connected_websocket_clients = set()
g_websocket_server_thread = None
g_asyncio_loop_for_websocket = None
g_websocket_stop_event = None # Will be an asyncio.Event

# --- ABR State ---
abr_lock = threading.Lock()
current_abr_algorithm_selected_media_m3u8_url_on_server = None # For logging

# --- QoE Metrics Manager ---
class QoEMetricsManager:
    def __init__(self):
        self.startup_latency_ms = None
        self.rebuffering_events = [] # Stores {'start_ts': timestamp_ms, 'duration_ms': duration_ms, 'end_ts': timestamp_ms}
        self.quality_switches_log = [] # Stores {'timestamp': ms, 'from_level': idx, 'to_level': idx, 'to_bitrate': bps}
        
        self.session_active = False
        self.session_start_time_ms = 0
        self.total_session_duration_ms = 0 # Actual time content was playing or supposed to be playing

        self.current_level_index = -1
        self.time_at_each_level = {} # {level_index: duration_ms}
        self.last_event_timestamp_ms = 0 # Used to calculate duration at current level before switch/end
        logger.info("QoEMetricsManager initialized.")

    def start_session_if_needed(self, event_timestamp_ms):
        if not self.session_active:
            self.session_active = True
            self.session_start_time_ms = event_timestamp_ms # Session starts with the first significant event
            self.last_event_timestamp_ms = event_timestamp_ms
            logger.info(f"QoE: Playback session started around {event_timestamp_ms}.")

    def update_time_at_level(self, event_timestamp_ms):
        if self.session_active and self.current_level_index != -1:
            duration_at_current_level_ms = event_timestamp_ms - self.last_event_timestamp_ms
            if duration_at_current_level_ms > 0:
                self.time_at_each_level[self.current_level_index] = \
                    self.time_at_each_level.get(self.current_level_index, 0) + duration_at_current_level_ms
        self.last_event_timestamp_ms = event_timestamp_ms


    def record_startup_latency(self, latency_ms, timestamp_ms):
        self.start_session_if_needed(timestamp_ms - latency_ms) # Session started when play was initiated
        if self.startup_latency_ms is None:
            self.startup_latency_ms = latency_ms
            logger.info(f"QoE Event: Startup Latency = {latency_ms} ms (at {timestamp_ms})")
            self.last_event_timestamp_ms = timestamp_ms # Update last event time after startup

    def record_rebuffering_start(self, timestamp_ms):
        self.start_session_if_needed(timestamp_ms)
        self.update_time_at_level(timestamp_ms)
        self.rebuffering_events.append({'start_ts': timestamp_ms, 'duration_ms': 0, 'end_ts': None}) # duration will be updated
        logger.info(f"QoE Event: Rebuffering Started at {timestamp_ms}")

    def record_rebuffering_end(self, duration_ms, timestamp_ms):
        self.start_session_if_needed(timestamp_ms) # Should already be active
        # Find the last open rebuffering event
        for event in reversed(self.rebuffering_events):
            if event['end_ts'] is None:
                event['duration_ms'] = duration_ms
                event['end_ts'] = timestamp_ms
                logger.info(f"QoE Event: Rebuffering Ended. Duration = {duration_ms} ms (at {timestamp_ms})")
                break
        self.last_event_timestamp_ms = timestamp_ms # Update last event time after rebuffering

    def record_quality_switch(self, from_level_index, to_level_index, to_bitrate, timestamp_ms):
        self.start_session_if_needed(timestamp_ms)
        self.update_time_at_level(timestamp_ms)

        # If from_level_index is -1, it's the initial level setting.
        # current_level_index helps track the *actual* previous playing level.
        actual_from_level = self.current_level_index if from_level_index == -1 or self.current_level_index != -1 else from_level_index

        if actual_from_level != to_level_index : # Log only actual switches or initial set
            log_entry = {
                'timestamp': timestamp_ms,
                'from_level': actual_from_level,
                'to_level': to_level_index,
                'to_bitrate': to_bitrate
            }
            self.quality_switches_log.append(log_entry)
            if actual_from_level == -1:
                logger.info(f"QoE Event: Initial Level set to {to_level_index} (Bitrate: {to_bitrate/1000:.0f} Kbps) at {timestamp_ms}")
            else:
                logger.info(f"QoE Event: Quality Switch from level {actual_from_level} to {to_level_index} (Bitrate: {to_bitrate/1000:.0f} Kbps) at {timestamp_ms}")
        
        self.current_level_index = to_level_index


    def log_playback_session_end(self, timestamp_ms=None):
        if not self.session_active:
            logger.info("QoE: No active session to end or already ended.")
            return

        if timestamp_ms is None:
            timestamp_ms = time.time() * 1000
        
        self.update_time_at_level(timestamp_ms) # Account for time since last event
        self.total_session_duration_ms = timestamp_ms - self.session_start_time_ms
        self.session_active = False
        logger.info(f"QoE: Playback session ended at {timestamp_ms}. Total duration: {self.total_session_duration_ms:.0f} ms.")
        self.print_summary()

    def print_summary(self):
        logger.info("--- QoE Summary ---")
        if not self.session_start_time_ms: # Check if any activity was logged
            logger.info("  No playback activity recorded for QoE summary.")
            logger.info("--------------------")
            return

        if self.startup_latency_ms is not None:
            logger.info(f"  Startup Latency: {self.startup_latency_ms:.2f} ms")
        else:
            logger.info("  Startup Latency: Not recorded")
        
        num_stalls = len([e for e in self.rebuffering_events if e['duration_ms'] > 0])
        total_stall_duration = sum(e['duration_ms'] for e in self.rebuffering_events if e['duration_ms'] > 0)
        logger.info(f"  Rebuffering Events (Stalls): {num_stalls}")
        logger.info(f"  Total Rebuffering Duration: {total_stall_duration:.2f} ms")

        logger.info(f"  Quality Switches (logged): {len(self.quality_switches_log)}")
        # for i, switch in enumerate(self.quality_switches_log):
        #     logger.info(f"    Switch {i+1}: From {switch['from_level']} To {switch['to_level']} (Bitrate: {switch['to_bitrate']/1000:.0f} Kbps)")
        
        logger.info(f"  Time spent at each quality level (index: ms):")
        for level_idx, duration_ms in self.time_at_each_level.items():
            bitrate_str = "N/A" # Default if not found
            if ABRManager.instance and ABRManager.instance.available_streams:
                # Ensure level_idx is a valid index for the list
                if isinstance(level_idx, int) and 0 <= level_idx < len(ABRManager.instance.available_streams):
                    stream_info = ABRManager.instance.available_streams[level_idx]
                    if isinstance(stream_info, dict) and 'bandwidth' in stream_info:
                        bitrate_bps = stream_info.get('bandwidth', 0)
                        bitrate_str = f"{bitrate_bps/1000:.0f} Kbps"
                    else:
                        logger.debug(f"QoE Summary: Stream info for level {level_idx} is not a dict or missing bandwidth: {stream_info}")
                else:
                    logger.debug(f"QoE Summary: Invalid level_idx {level_idx} for available_streams length {len(ABRManager.instance.available_streams)}")
            logger.info(f"    Level {level_idx} ({bitrate_str}): {duration_ms:.0f} ms")

        # Average Played Bitrate calculation
        if ABRManager.instance and ABRManager.instance.available_streams:
            total_weighted_bitrate_x_time = 0 # Sum of (bitrate_bps * time_seconds_at_level)
            total_time_at_levels_seconds = 0
            for level_idx, duration_ms in self.time_at_each_level.items():
                if 0 <= level_idx < len(ABRManager.instance.available_streams):
                    bitrate_bps = ABRManager.instance.available_streams[level_idx].get('bandwidth', 0)
                    time_seconds = duration_ms / 1000.0
                    total_weighted_bitrate_x_time += bitrate_bps * time_seconds
                    total_time_at_levels_seconds += time_seconds
            
            if total_time_at_levels_seconds > 0:
                average_played_bitrate_kbps = (total_weighted_bitrate_x_time / total_time_at_levels_seconds) / 1000.0
                logger.info(f"  Average Played Bitrate (based on time at levels): {average_played_bitrate_kbps:.2f} Kbps")
            else:
                logger.info("  Average Played Bitrate: Not enough data.")

        logger.info(f"  Total Playback Session Duration (approx): {self.total_session_duration_ms:.2f} ms")
        if self.total_session_duration_ms > 0 :
            # Effective playing time = total_session_duration - total_stall_duration - startup_latency (if startup is part of session)
            # For rebuffering ratio, typically total_stall_duration / (total_stall_duration + actual_playing_time)
            # Or simpler: total_stall_duration / total_session_duration_ms
            rebuffering_ratio = (total_stall_duration / self.total_session_duration_ms) * 100 if self.total_session_duration_ms > 0 else 0
            logger.info(f"  Rebuffering Ratio (approx): {rebuffering_ratio:.2f}%")
        logger.info("--------------------")

# Global instance
qoe_manager = QoEMetricsManager()


# --- HTML Content (Updated JavaScript for QoE) ---
HTML_PLAYER_CONTENT = f"""
<!DOCTYPE html>
<html>
<head>
    <title>HLS.js Player (Python ABR + QoE)</title>
    <script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
    <style> /* ... Your CSS ... */ </style>
</head>
<body>
    <h1>HLS.js Streaming Client (Python ABR + QoE)</h1>
    <div id="videoContainer"><video id="videoPlayer" controls></video></div>
    <div class="controls">
        <p>Master Playlist URL: <span id="masterUrlSpan"></span></p>
        <p>ABR Control: Python Backend</p>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', function () {{
            const video = document.getElementById('videoPlayer');
            const masterM3u8Url = `http://{LOCAL_PROXY_HOST}:{LOCAL_PROXY_PORT}/{VIDEO_TO_STREAM_NAME}/master.m3u8`;
            document.getElementById('masterUrlSpan').textContent = masterM3u8Url;
            let hlsInstance = null;
            const ws = new WebSocket(`ws://{LOCAL_PROXY_HOST}:{WEBSOCKET_PORT}`);

            function sendQoeEvent(eventData) {{
                if (ws.readyState === WebSocket.OPEN) {{
                    ws.send(JSON.stringify({{ type: "QOE_EVENT", data: eventData }}));
                }} else {{
                    console.warn("WebSocket not open, QoE event not sent:", eventData);
                }}
            }}

            ws.onopen = function() {{ console.log("WebSocket connection established."); }};
            ws.onclose = function() {{ console.log("WebSocket connection closed."); }};
            ws.onerror = function(error) {{ console.error("WebSocket error:", error); }};
            ws.onmessage = function(event) {{ // For ABR commands from Python
                try {{
                    const message = JSON.parse(event.data);
                    if (message.type === "SET_LEVEL" && hlsInstance) {{
                        const newLevelIndex = parseInt(message.levelIndex);
                        if (hlsInstance.levels && newLevelIndex >= 0 && newLevelIndex < hlsInstance.levels.length) {{
                            if (hlsInstance.currentLevel !== newLevelIndex || hlsInstance.nextLevel !== newLevelIndex) {{
                                console.log(`Python ABR COMMAND: Switch to level index: ${{newLevelIndex}}`);
                                hlsInstance.nextLevel = newLevelIndex;
                            }}
                        }}
                    }}
                }} catch (e) {{ console.error("Error processing ABR command from WebSocket:", e); }}
            }};

            // --- QoE Event Tracking ---
            let playInitiatedTimestamp = 0;
            let isFirstPlaySignal = true; 
            let isRebuffering = false;
            let rebufferingStartTime = 0;
            let jsPreviousLevel = -1; // JS-side tracking of previous level

            const originalVideoPlay = video.play;
            video.play = function(...args) {{
                if (isFirstPlaySignal) {{ // Capture time just before actual play command for the very first play
                    console.log("QoE JS: video.play() called, isFirstPlaySignal:", isFirstPlaySignal);
                }}
                return originalVideoPlay.apply(this, args);
            }};

            video.addEventListener('playing', function() {{
                const currentEventTime = Date.now();
                if (isFirstPlaySignal && playInitiatedTimestamp > 0) {{
                    const startupLatency = currentEventTime - playInitiatedTimestamp;
                    console.log(`QoE JS: Startup Latency = ${{startupLatency}} ms`);
                    sendQoeEvent({{ event: "STARTUP_LATENCY", value: startupLatency, timestamp: currentEventTime }});
                    isFirstPlaySignal = false;
                }}
                if (isRebuffering) {{
                    isRebuffering = false;
                    const rebufferingDuration = currentEventTime - rebufferingStartTime;
                    console.log(`QoE JS: Rebuffering ended. Duration = ${{rebufferingDuration}} ms`);
                    sendQoeEvent({{ event: "REBUFFERING_END", duration: rebufferingDuration, timestamp: currentEventTime }});
                }}
            }});

            video.addEventListener('waiting', function() {{
                const currentEventTime = Date.now();
                if (!isFirstPlaySignal && video.currentTime > 0 && !video.paused && !video.ended) {{ 
                    if (!isRebuffering) {{
                        isRebuffering = true;
                        rebufferingStartTime = currentEventTime;
                        console.log("QoE JS: Rebuffering started");
                        sendQoeEvent({{ event: "REBUFFERING_START", timestamp: rebufferingStartTime }});
                    }}
                }}
            }});
            
            video.addEventListener('ended', function() {{
                console.log("QoE JS: Playback naturally ended.");
                sendQoeEvent({{ event: "PLAYBACK_ENDED", timestamp: Date.now() }});
            }});

            window.addEventListener('beforeunload', function() {{
                console.log("QoE JS: Window closing, sending PLAYBACK_ENDED.");
                sendQoeEvent({{ event: "PLAYBACK_ENDED", timestamp: Date.now() }});
            }});


            if (Hls.isSupported()) {{
                hlsInstance = new Hls({{ debug: false /* Set to true for verbose HLS logs */ }});
                
                playInitiatedTimestamp = Date.now(); // <--- **关键：确保在这里或更早捕获初始时间**
                console.log("QoE JS: Play initiation (HLS loadSource) timestamped at", playInitiatedTimestamp);
                
                hlsInstance.loadSource(masterM3u8Url);
                hlsInstance.attachMedia(video);

                hlsInstance.on(Hls.Events.MANIFEST_PARSED, function (event, data) {{
                    console.log("Manifest parsed. Levels:", hlsInstance.levels);
                    // video.play(); // Play will be called, playInitiatedTimestamp captured by wrapper

                    if (hlsInstance.levels && hlsInstance.levels.length > 1) {{
                        jsPreviousLevel = 0; // Assume starts at level 0 for QoE tracking
                        hlsInstance.currentLevel = 0; 
                        hlsInstance.autoLevelCapping = -1; 
                        hlsInstance.autoLevelEnabled = false;
                        console.log("HLS.js auto ABR disabled. Initial level for QoE: " + jsPreviousLevel);
                        // Send initial level to Python QoE
                        sendQoeEvent({{
                            event: "QUALITY_SWITCH",
                            fromLevel: -1, // Indicate no prior level
                            toLevel: jsPreviousLevel,
                            toBitrate: hlsInstance.levels[jsPreviousLevel].bitrate,
                            timestamp: Date.now()
                        }});
                    }} else if (hlsInstance.levels && hlsInstance.levels.length === 1) {{
                        jsPreviousLevel = 0;
                        console.log("HLS.js: Only one level available. Initial level for QoE: " + jsPreviousLevel);
                         sendQoeEvent({{
                            event: "QUALITY_SWITCH",
                            fromLevel: -1,
                            toLevel: jsPreviousLevel,
                            toBitrate: hlsInstance.levels[jsPreviousLevel].bitrate,
                            timestamp: Date.now()
                        }});
                    }}
                }});

                hlsInstance.on(Hls.Events.LEVEL_SWITCHED, function(event, data) {{
                    const newLevel = data.level;
                    console.log(`HLS.js ACTUALLY switched to level: ${{newLevel}}`);
                    if (jsPreviousLevel !== newLevel) {{ // Ensure it's an actual change from JS perspective
                         sendQoeEvent({{
                            event: "QUALITY_SWITCH",
                            fromLevel: jsPreviousLevel,
                            toLevel: newLevel,
                            toBitrate: hlsInstance.levels[newLevel].bitrate,
                            timestamp: Date.now()
                        }});
                        jsPreviousLevel = newLevel;
                    }}
                }});
                
                hlsInstance.on(Hls.Events.ERROR, function (event, data) {{ 
                    console.error('HLS.js Error:', data);
                    // Consider sending critical errors as QoE events if desired
                }});

            }} else if (video.canPlayType('application/vnd.apple.mpegurl')) {{
                video.src = masterM3u8Url;
                // Native HLS might need different QoE event listeners or might not provide all events
            }} else {{
                alert("HLS is not supported in your browser.");
            }}
        }});
    </script>
</body>
</html>
""".replace("{LOCAL_PROXY_HOST}", LOCAL_PROXY_HOST) \
 .replace("{LOCAL_PROXY_PORT}", str(LOCAL_PROXY_PORT)) \
 .replace("{WEBSOCKET_PORT}", str(WEBSOCKET_PORT)) \
 .replace("{VIDEO_TO_STREAM_NAME}", VIDEO_TO_STREAM_NAME)

# --- WebSocket Server Functions (handle_websocket_client needs to route QoE messages) ---
async def handle_websocket_client(websocket):
    global g_connected_websocket_clients
    client_identifier = getattr(websocket, 'path', None)
    if client_identifier is None:
        try: client_identifier = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        except Exception: client_identifier = "UnknownClient" # Fallback if remote_address also fails
            
    logger.info(f"WebSocket client connected: {client_identifier}")
    g_connected_websocket_clients.add(websocket)
    try:
        async for message_str in websocket:
            logger.debug(f"WebSocket received from {client_identifier}: {message_str}")
            try:
                message = json.loads(message_str)
                if message.get("type") == "QOE_EVENT":
                    event_data = message.get("data", {})
                    event_name = event_data.get("event")
                    timestamp = event_data.get("timestamp", time.time() * 1000)

                    if event_name == "STARTUP_LATENCY":
                        qoe_manager.record_startup_latency(event_data.get("value"), timestamp)
                    elif event_name == "REBUFFERING_START":
                        qoe_manager.record_rebuffering_start(timestamp)
                    elif event_name == "REBUFFERING_END":
                        qoe_manager.record_rebuffering_end(event_data.get("duration"), timestamp)
                    elif event_name == "QUALITY_SWITCH":
                        qoe_manager.record_quality_switch(
                            event_data.get("fromLevel"),
                            event_data.get("toLevel"),
                            event_data.get("toBitrate"),
                            timestamp
                        )
                    elif event_name == "PLAYBACK_ENDED":
                        qoe_manager.log_playback_session_end(timestamp)
                    else:
                        logger.warning(f"Unknown QoE event name from {client_identifier}: {event_name}")
                # else: # Could be ABR commands if you had bi-directional for other things
                #     logger.debug(f"Received non-QoE message from {client_identifier}: {message_str}")

            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON from WebSocket ({client_identifier}): {message_str}")
            except Exception as e:
                logger.error(f"Error processing message from {client_identifier}: {e}", exc_info=True)
    except websockets.exceptions.ConnectionClosedOK:
        logger.info(f"WebSocket client {client_identifier} disconnected gracefully.")
    except websockets.exceptions.ConnectionClosedError as e:
        logger.warning(f"WebSocket client {client_identifier} connection closed with error: {e}")
    except Exception as e:
        logger.error(f"Error in WebSocket client handler for {client_identifier}: {e}", exc_info=True)
    finally:
        g_connected_websocket_clients.remove(websocket)
        logger.info(f"WebSocket client {client_identifier} removed from connected set.")

async def run_websocket_server_async():
    global g_asyncio_loop_for_websocket, g_websocket_stop_event # g_asyncio_loop_for_websocket is set by the calling thread

    # Sanity check: ensure the loop we think we are using is indeed the current running one
    # This part is mostly for debugging/understanding; websockets.serve should use get_running_loop()
    try:
        current_loop = asyncio.get_running_loop()
        if g_asyncio_loop_for_websocket is not current_loop:
            logger.warning(
                f"run_websocket_server_async: g_asyncio_loop_for_websocket (id: {id(g_asyncio_loop_for_websocket)}) "
                f"is not the current running loop (id: {id(current_loop)}). This might be an issue if not intended."
            )
            # If start_websocket_server_in_thread correctly sets the loop for its thread,
            # and then calls run_until_complete on that loop for this coroutine,
            # then get_running_loop() within this coroutine should return that same loop.
    except RuntimeError:
        logger.error("run_websocket_server_async: No current asyncio loop running when expected!")
        return # Cannot proceed without a loop

    if g_websocket_stop_event is None:
        # Create the event within the context of the loop this coroutine is running on
        g_websocket_stop_event = asyncio.Event() 

    logger.info(f"Starting WebSocket server on ws://{LOCAL_PROXY_HOST}:{WEBSOCKET_PORT}")
    
    server_instance = None
    try:
        # REMOVED: loop=g_asyncio_loop_for_websocket from the websockets.serve call
        async with websockets.serve(handle_websocket_client, LOCAL_PROXY_HOST, WEBSOCKET_PORT) as server:
            server_instance = server 
            server_address = "N/A"
            if server.sockets:
                try:
                    server_address = server.sockets[0].getsockname()
                except Exception: # Handle cases where getsockname might not be available immediately or on all socket types
                    pass
            logger.info(f"WebSocket server '{server_address}' now serving.")
            await g_websocket_stop_event.wait() 
    except asyncio.CancelledError:
        logger.info("WebSocket server task (run_websocket_server_async) was cancelled.")
    except Exception as e: # Catch other potential errors during serve()
        logger.error(f"Error during websockets.serve: {e}", exc_info=True)
    finally:
        if server_instance:
            logger.info("Closing WebSocket server instance...")
            server_instance.close()
            try:
                await server_instance.wait_closed() 
            except Exception as e_close:
                logger.error(f"Error during server_instance.wait_closed: {e_close}")
        logger.info("WebSocket server (run_websocket_server_async) has shut down.")

def start_websocket_server_in_thread():
    global g_websocket_server_thread, g_asyncio_loop_for_websocket
    
    thread_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(thread_loop)
    # This global variable now correctly holds the loop object
    # that is exclusively managed by this WebSocket server thread.
    g_asyncio_loop_for_websocket = thread_loop 

    try:
        thread_loop.run_until_complete(run_websocket_server_async())
    except KeyboardInterrupt: 
        logger.info("WebSocket server thread (run_until_complete) received KeyboardInterrupt.")
    except SystemExit: 
        logger.info("WebSocket server asyncio loop stopping (SystemExit).")
    finally:
        logger.info("WebSocket server thread: Cleaning up asyncio loop.")
        try:
            # Attempt to cancel all remaining tasks in this specific loop
            all_tasks = asyncio.all_tasks(loop=thread_loop)
            if all_tasks:
                logger.info(f"Cancelling {len(all_tasks)} outstanding asyncio tasks in WebSocket thread loop...")
                for task in all_tasks:
                    if not task.done() and not task.cancelled(): # Check if cancellation is needed
                        task.cancel()
                # Wait for tasks to process cancellation
                # This gather should be run by the loop itself.
                # However, run_until_complete has exited.
                # We might need to run gather within a short final run_until_complete if tasks need cleanup.
                # For simplicity now, we assume cancellation is set.
                # thread_loop.run_until_complete(asyncio.gather(*all_tasks, return_exceptions=True))
                # This can be tricky if the loop is already stopping.
            
            # If loop is stopping due to g_websocket_stop_event.set() -> run_until_complete finishes,
            # then we proceed to close.
            if thread_loop.is_running(): # Should not be true if run_until_complete exited cleanly
                logger.warning("WebSocket thread loop still running unexpectedly during cleanup; attempting stop.")
                thread_loop.stop()
            if not thread_loop.is_closed():
                thread_loop.close()
                logger.info("WebSocket server asyncio loop closed by thread.")
            else:
                logger.info("WebSocket server asyncio loop was already closed.")
        except RuntimeError as e:
             logger.error(f"RuntimeError during WebSocket thread loop cleanup: {e}")


async def broadcast_message_async(message_str):
    if g_connected_websocket_clients:
        # Create a list of tasks for sending messages to avoid issues if the set changes during iteration
        clients_to_send_to = list(g_connected_websocket_clients)
        tasks = [client.send(message_str) for client in clients_to_send_to]
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    client_repr = str(clients_to_send_to[i].remote_address) if hasattr(clients_to_send_to[i], 'remote_address') else "Unknown Client"
                    logger.error(f"Error sending message to WebSocket client {client_repr}: {result}")

def schedule_abr_broadcast(level_index):
    if g_asyncio_loop_for_websocket and g_asyncio_loop_for_websocket.is_running():
        message = json.dumps({"type": "SET_LEVEL", "levelIndex": level_index})
        # Schedule the async broadcast function to run in the WebSocket's asyncio loop
        asyncio.run_coroutine_threadsafe(broadcast_message_async(message), g_asyncio_loop_for_websocket)
    else:
        logger.warning("Cannot schedule ABR broadcast: WebSocket asyncio loop not available or not running.")

class DecryptionProxyHandler(http.server.BaseHTTPRequestHandler):
    # ... (do_GET, _rewrite_master_playlist, _rewrite_media_playlist remain largely the same) ...
    # Ensure _rewrite_master_playlist provides ALL variants to HLS.js
    # so HLS.js knows all available levels and their original order.

    def do_GET(self):
        log_adapter = logging.LoggerAdapter(logger, {'path': self.path})
        request_log_tag = f"[ProxyRequest URI: {self.path}]"
        parsed_url = urlparse(self.path)
        path_components = parsed_url.path.strip('/').split('/')

        try:
            if parsed_url.path == '/' or parsed_url.path == '/player.html':
                log_adapter.info(f"{request_log_tag} Serving player.html")
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(HTML_PLAYER_CONTENT.encode('utf-8'))
                return

            if len(path_components) == 2 and path_components[0] == VIDEO_TO_STREAM_NAME and path_components[1] == "master.m3u8":
                video_name_from_url = path_components[0]
                original_master_m3u8_url = f"http://{SERVER_HOST}:{SERVER_PORT}/{video_name_from_url}/master.m3u8"
                log_adapter.info(f"{request_log_tag} Request for master M3U8. Fetching from: {original_master_m3u8_url}")
                try:
                    response = requests.get(original_master_m3u8_url, timeout=SOCKET_TIMEOUT_SECONDS)
                    response.raise_for_status()
                    master_content = response.text
                    # IMPORTANT: This rewrite MUST ensure all original variants are present,
                    # so HLS.js knows all level indices. The URLs to media playlists are proxied.
                    modified_master_content = self._rewrite_master_playlist(master_content, original_master_m3u8_url)
                    
                    self.send_response(200)
                    self.send_header('Content-type', 'application/vnd.apple.mpegurl')
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(modified_master_content.encode('utf-8'))
                except requests.exceptions.RequestException as e:
                    self.send_error(502, f"Bad Gateway: Could not fetch master M3U8: {e}")
                return

            if len(path_components) == 3 and path_components[0] == VIDEO_TO_STREAM_NAME and path_components[2].endswith(".m3u8"):
                # ... (media playlist serving logic - same as before, rewrites TS to /decrypt_segment)
                video_name_from_url = path_components[0]
                quality_dir_from_url = path_components[1]
                playlist_filename_from_url = path_components[2]
                original_media_m3u8_url = f"http://{SERVER_HOST}:{SERVER_PORT}/{video_name_from_url}/{quality_dir_from_url}/{playlist_filename_from_url}"
                try:
                    response = requests.get(original_media_m3u8_url, timeout=SOCKET_TIMEOUT_SECONDS)
                    response.raise_for_status()
                    media_content = response.text
                    modified_media_content = self._rewrite_media_playlist(media_content, original_media_m3u8_url)
                    self.send_response(200)
                    self.send_header('Content-type', 'application/vnd.apple.mpegurl')
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(modified_media_content.encode('utf-8'))
                except requests.exceptions.RequestException as e:
                    self.send_error(502, f"Bad Gateway: Could not fetch media M3U8: {e}")
                return

            if parsed_url.path == '/decrypt_segment':
                # ... (TS segment decryption logic - same as before, calls ABRManager.add_segment_download_stat)
                query_params = parse_qs(parsed_url.query)
                original_ts_url_on_server_encoded = query_params.get('url', [None])[0]
                if not original_ts_url_on_server_encoded:
                    self.send_error(400, "Bad Request: Missing 'url' parameter"); return
                original_ts_url_on_server = unquote(original_ts_url_on_server_encoded)
                log_adapter.info(f"{request_log_tag} - Proxy serving TS segment, original URL: {original_ts_url_on_server}")
                try:
                    fetch_start_time = time.time()
                    response_ts = requests.get(original_ts_url_on_server, timeout=SOCKET_TIMEOUT_SECONDS, stream=True)
                    response_ts.raise_for_status()
                    encrypted_data = response_ts.content
                    fetch_end_time = time.time()
                    if ABRManager.instance:
                        ABRManager.instance.add_segment_download_stat(original_ts_url_on_server, len(encrypted_data), fetch_end_time - fetch_start_time)
                    if not encrypted_data: self.send_error(502, "Bad Gateway: Empty TS content"); return
                    decrypted_data = AES.aes_decrypt_cbc(encrypted_data, AES.AES_KEY)
                    self.send_response(200)
                    self.send_header('Content-type', 'video/MP2T')
                    self.send_header('Content-Length', str(len(decrypted_data)))
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(decrypted_data)
                except requests.exceptions.RequestException as e_req:
                    if ABRManager.instance: ABRManager.instance.report_download_error(original_ts_url_on_server)
                    self.send_error(502, f"Bad Gateway: Could not fetch TS: {e_req}")
                except Exception as e_dec:
                    self.send_error(500, f"Internal Server Error: Decryption failed: {e_dec}")
                return
            
            self.send_error(404, "Not Found")
        except requests.exceptions.RequestException as e: 
            logger.error(f"{request_log_tag} Proxy HTTP fetch error: {e}"); 
            self.send_error(502, f"Bad Gateway: {e}")
        except Exception as e: 
            logger.error(f"{request_log_tag} Proxy error: {e}", exc_info=True); 
            self.send_error(500, "Internal Server Error")
            
    def _rewrite_master_playlist(self, master_content, original_master_url): # Same
        lines = master_content.splitlines(); modified_lines = []
        for i in range(len(lines)):
            line = lines[i].strip()
            if line.startswith("#EXT-X-STREAM-INF:"):
                modified_lines.append(line) 
                if i + 1 < len(lines):
                    media_playlist_relative_or_absolute_url_in_master = lines[i+1].strip()
                    original_media_playlist_absolute_url = urljoin(original_master_url, media_playlist_relative_or_absolute_url_in_master)
                    parsed_original_media_playlist_url = urlparse(original_media_playlist_absolute_url)
                    proxied_media_playlist_url = f"http://{LOCAL_PROXY_HOST}:{LOCAL_PROXY_PORT}{parsed_original_media_playlist_url.path}"
                    modified_lines.append(proxied_media_playlist_url)
                    i += 1 
            else:
                modified_lines.append(line)
        return "\n".join(modified_lines)
    
    def _rewrite_media_playlist(self, media_content, original_media_url): # Same
        lines = media_content.splitlines(); modified_lines = []
        base_url = urljoin(original_media_url, '.')
        for line in lines:
            s_line = line.strip()
            if s_line and not s_line.startswith("#") and (s_line.endswith(".ts") or ".ts?" in s_line):
                abs_seg_url = urljoin(base_url, s_line); enc_seg_url = quote(abs_seg_url, safe='')
                modified_lines.append(f"http://{LOCAL_PROXY_HOST}:{LOCAL_PROXY_PORT}/decrypt_segment?url={enc_seg_url}")
            else: modified_lines.append(line)
        return "\n".join(modified_lines)
    def log_message(self, format, *args): logger.debug(f"ProxyHTTP: {self.address_string()} - {args[0]} {args[1]}")

class ThreadingLocalProxyServer(socketserver.ThreadingMixIn, http.server.HTTPServer): # Same
    daemon_threads = True
    allow_reuse_address = True

def _run_proxy_server_target(): # Same
    global g_local_proxy_server_instance
    try:
        g_local_proxy_server_instance = ThreadingLocalProxyServer(
            (LOCAL_PROXY_HOST, LOCAL_PROXY_PORT), DecryptionProxyHandler)
        logger.info(f"PROXY_THREAD: Local proxy server starting on http://{LOCAL_PROXY_HOST}:{LOCAL_PROXY_PORT}")
        g_local_proxy_server_instance.serve_forever() 
    except Exception as e: logger.error(f"PROXY_THREAD: Error: {e}", exc_info=True); g_local_proxy_server_instance = None
    finally: logger.info("PROXY_THREAD: Proxy server loop finished.")
    

def start_proxy_server(): # Same, ensure it returns True/False
    # ... (same as your last working version) ...
    global g_proxy_runner_thread, g_local_proxy_server_instance
    if g_proxy_runner_thread and g_proxy_runner_thread.is_alive(): return bool(g_local_proxy_server_instance) 
    g_local_proxy_server_instance = None 
    g_proxy_runner_thread = threading.Thread(target=_run_proxy_server_target, daemon=True, name="ProxyServerThread")
    g_proxy_runner_thread.start()
    time.sleep(0.5) 
    return bool(g_proxy_runner_thread.is_alive() and g_local_proxy_server_instance)


def stop_proxy_server(): # Same
    # ... (same as your last working version) ...
    global g_local_proxy_server_instance, g_proxy_runner_thread
    if g_local_proxy_server_instance:
        logger.info("PROXY_MAIN: Stopping proxy server instance...")
        g_local_proxy_server_instance.shutdown()
        if g_proxy_runner_thread: g_proxy_runner_thread.join(timeout=2.0)
        g_local_proxy_server_instance.server_close()
        g_local_proxy_server_instance = None
        g_proxy_runner_thread = None
        logger.info("PROXY_MAIN: Proxy server stopped.")


class ABRManager:
    instance = None 

    def __init__(self, available_streams_from_master):
        ABRManager.instance = self
        self.available_streams = sorted(
            [s for s in available_streams_from_master if s.get('bandwidth') is not None], 
            key=lambda s: s['bandwidth']
        )
        if not self.available_streams: # Add a dummy to prevent crashes if all lack bandwidth
             self.available_streams = [{'url': 'dummy', 'bandwidth': 0, 'resolution': 'N/A', 'attributes_str': ''}]
        
        # current_stream_index_by_abr is the index into self.available_streams
        # This index will be sent to HLS.js
        self.current_stream_index_by_abr = 0 
        self.segment_download_stats = [] 
        self.max_stats_history = 20 
        self.estimated_bandwidth_bps = 0
        self.safety_factor = 0.8
        self.abr_thread = None
        self.stop_abr_event = threading.Event()
        
        logger.info(f"ABRManager initialized. Available streams (sorted by bandwidth for indexing):")
        for i, s in enumerate(self.available_streams):
            logger.info(f"  Level Index [{i}] BW: {s.get('bandwidth', 'N/A')}, Res: {s.get('resolution', 'N/A')}, URL: {s['url']}")
        self._update_current_abr_selected_url_logging() # For logging selected URL
        # Send initial decision (e.g., lowest quality)
        schedule_abr_broadcast(self.current_stream_index_by_abr)


    def _update_current_abr_selected_url_logging(self): # For logging only
        global current_abr_algorithm_selected_media_m3u8_url_on_server
        with abr_lock:
            if self.available_streams and 0 <= self.current_stream_index_by_abr < len(self.available_streams):
                current_abr_algorithm_selected_media_m3u8_url_on_server = self.available_streams[self.current_stream_index_by_abr]['url']

    def add_segment_download_stat(self, url, size_bytes, duration_seconds): # Same
        if duration_seconds > 0.001: 
            self.segment_download_stats.append({'url': url, 'size': size_bytes, 'duration': duration_seconds})
            if len(self.segment_download_stats) > self.max_stats_history:
                self.segment_download_stats.pop(0)

    def report_download_error(self, url):  # Same
        logger.warning(f"ABR: Reported download error for segment from URL: {url}")

    def _estimate_bandwidth(self): # Same
        # ... (your preferred bandwidth estimation logic) ...
        if not self.segment_download_stats: return self.estimated_bandwidth_bps
        relevant_stats = self.segment_download_stats[-5:]
        if not relevant_stats: return self.estimated_bandwidth_bps
        total_bytes = sum(s['size'] for s in relevant_stats)
        total_time = sum(s['duration'] for s in relevant_stats)
        if total_time == 0: return self.estimated_bandwidth_bps
        self.estimated_bandwidth_bps = (total_bytes * 8) / total_time
        logger.info(f"ABR Python Algo: Estimated BW: {self.estimated_bandwidth_bps / 1000:.0f} Kbps")
        return self.estimated_bandwidth_bps

    def _abr_decision_logic(self):
        if not self.available_streams or len(self.available_streams) <=1: return

        estimated_bw = self._estimate_bandwidth()
        if estimated_bw == 0 and not self.segment_download_stats : # No data yet, stick to initial
            logger.info("ABR Python Algo: No stats yet, sticking to initial level.")
            # schedule_abr_broadcast(self.current_stream_index_by_abr) # Re-affirm if needed
            return

        next_best_index = 0 
        for i in range(len(self.available_streams) -1, -1, -1):
            stream_bw = self.available_streams[i].get('bandwidth', 0)
            if estimated_bw * self.safety_factor > stream_bw:
                next_best_index = i
                break
        
        if next_best_index != self.current_stream_index_by_abr:
            old_stream_info = self.available_streams[self.current_stream_index_by_abr]
            self.current_stream_index_by_abr = next_best_index
            new_stream_info = self.available_streams[self.current_stream_index_by_abr]
            self._update_current_abr_selected_url_logging()
            
            logger.info(f"ABR PYTHON ALGO DECISION: Switch from level index {self.available_streams.index(old_stream_info)} (BW {old_stream_info.get('bandwidth',0)/1000:.0f} Kbps) "
                        f"to level index {self.current_stream_index_by_abr} (BW {new_stream_info.get('bandwidth',0)/1000:.0f} Kbps).")
            
            # **BROADCAST DECISION TO WEBSOCKET CLIENTS**
            schedule_abr_broadcast(self.current_stream_index_by_abr)
        # else:
            # logger.debug(f"ABR Python Algo: No change in decision, current index {self.current_stream_index_by_abr}")


    def abr_loop(self):
        logger.info("ABR Python Algo monitoring thread started.")
        # --- TEST: Hardcode level switch after 10 seconds ---
        # time.sleep(10) 
        # hardcoded_level_index = 1 # 假设你想切换到 level index 1 (e.g., 720p)
        # logger.info(f"TESTING: Hardcoding switch to level index: {hardcoded_level_index}")
        # schedule_abr_broadcast(hardcoded_level_index)
        # --- END TEST ---
        time.sleep(5) 
        while not self.stop_abr_event.is_set():
            try:
                self._abr_decision_logic()
            except Exception as e:
                logger.error(f"Error in ABR decision loop: {e}", exc_info=True)
            
            for _ in range(3): # Check stop event frequently during sleep (e.g. decision every 3s)
                if self.stop_abr_event.is_set(): break
                time.sleep(1)
        logger.info("ABR Python Algo monitoring thread stopped.")

    def start(self): # Same
        self.stop_abr_event.clear()
        self.abr_thread = threading.Thread(target=self.abr_loop, daemon=True, name="PythonABRLogicThread")
        self.abr_thread.start()

    def stop(self): # Same
        if self.abr_thread and self.abr_thread.is_alive():
            self.stop_abr_event.set()
            self.abr_thread.join(timeout=2.0)
        ABRManager.instance = None


# Function to fetch initial master m3u8 for ABRManager setup
def fetch_master_m3u8_for_abr_init(master_m3u8_url_on_server):
    logger.info(f"ABR_INIT: Fetching master M3U8 from: {master_m3u8_url_on_server}")
    try:
        response = requests.get(master_m3u8_url_on_server, timeout=SOCKET_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.exceptions.RequestException as e: return None
    content = response.text; lines = content.splitlines(); available_streams = []
    master_m3u8_base_url = urljoin(master_m3u8_url_on_server, '.')
    for i in range(len(lines)):
        line = lines[i].strip()
        if line.startswith("#EXT-X-STREAM-INF:"):
            attributes_str = line.split(":", 1)[1]
            attributes = parse_m3u8_attributes(attributes_str) # Use your parse_m3u8_attributes
            if i + 1 < len(lines):
                media_playlist_relative_url = lines[i+1].strip()
                media_playlist_absolute_url_on_origin = urljoin(master_m3u8_base_url, media_playlist_relative_url)
                available_streams.append({
                    'url': media_playlist_absolute_url_on_origin, 
                    'bandwidth': attributes.get('BANDWIDTH'),
                    'resolution': attributes.get('RESOLUTION'),
                    'codecs': attributes.get('CODECS'),
                    'attributes_str': attributes_str
                })
    return available_streams if available_streams else None

def parse_m3u8_attributes(attr_string): # Ensure this is defined
    attributes = {}
    try:
        for match in re.finditer(r'([A-Z0-9-]+)=("([^"]*)"|([^,"]*))', attr_string):
            key = match.group(1)
            value = match.group(3) if match.group(3) is not None else match.group(4)
            if value.isdigit(): attributes[key] = int(value)
            else: attributes[key] = value
    except Exception as e: logger.error(f"Error parsing M3U8 attributes: {e}")
    return attributes


def main():
    global g_websocket_server_thread, g_asyncio_loop_for_websocket, qoe_manager # qoe_manager is global
    abr_manager_instance = None # Local variable for ABRManager instance

    if not hasattr(AES, 'AES_KEY') or not AES.AES_KEY: logger.error("AES_KEY missing!"); return
    if not callable(getattr(AES, 'aes_decrypt_cbc', None)): logger.error("aes_decrypt_cbc missing!"); return
    if not os.path.exists(DOWNLOAD_DIR): os.makedirs(DOWNLOAD_DIR)

    if not start_proxy_server(): logger.error("HTTP Proxy server failed to start."); return

    logger.info("Starting WebSocket server thread...")
    g_websocket_server_thread = threading.Thread(target=start_websocket_server_in_thread, daemon=True, name="WebSocketServerThread")
    g_websocket_server_thread.start()
    time.sleep(1) # Give WebSocket server a moment to initialize its loop
    
    if not (g_asyncio_loop_for_websocket and g_asyncio_loop_for_websocket.is_running()):
        logger.warning("WebSocket asyncio loop doesn't seem to be running. ABR/QoE control might fail.")

    master_m3u8_url = f"http://{SERVER_HOST}:{SERVER_PORT}/{VIDEO_TO_STREAM_NAME}/master.m3u8"
    available_streams = fetch_master_m3u8_for_abr_init(master_m3u8_url)
    if available_streams:
        abr_manager_instance = ABRManager(available_streams) # Creates ABRManager.instance
        abr_manager_instance.start()
    else:
        logger.error("Could not fetch streams for ABR init. ABR will not function.")

    webbrowser.open(f"http://{LOCAL_PROXY_HOST}:{LOCAL_PROXY_PORT}/player.html")
    logger.info("Client setup complete. Press Ctrl+C to stop.")
    try:
        while True: time.sleep(60) # Keep main alive, print QoE summary periodically or on exit
            # qoe_manager.print_summary() # Optional: periodic summary
    except KeyboardInterrupt:
        logger.info("Ctrl+C pressed. Shutting down...")
    # In main() function's finally block
    finally:
        logger.info("Initiating cleanup...")
        qoe_manager.log_playback_session_end() 
        
        if abr_manager_instance: 
            abr_manager_instance.stop()

        if g_asyncio_loop_for_websocket and g_websocket_stop_event:
            if not g_asyncio_loop_for_websocket.is_closed():
                logger.info("Signaling WebSocket server to stop...")
                # This sets the event, which run_websocket_server_async is waiting on
                g_asyncio_loop_for_websocket.call_soon_threadsafe(g_websocket_stop_event.set)
            else:
                logger.info("WebSocket asyncio loop already closed, cannot signal stop event.")

        if g_websocket_server_thread and g_websocket_server_thread.is_alive():
            logger.info("Waiting for WebSocket server thread to join...")
            g_websocket_server_thread.join(timeout=5.0) # Increased timeout
            if g_websocket_server_thread.is_alive(): 
                logger.warning("WebSocket server thread did not join cleanly.")

        stop_proxy_server()
        logger.info("Client application finished.")

if __name__ == "__main__":
    main()