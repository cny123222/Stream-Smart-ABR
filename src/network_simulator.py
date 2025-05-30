import time
import threading
import logging
import socket # For socket.error, to be caught by the caller

# logger = logging.getLogger('NetworkSimulator')
# It's often better to pass the logger instance or use the calling module's logger
# For now, we'll assume a logger is configured in the main client.
# If this module is run standalone for testing, basicConfig might be needed.
logger = logging.getLogger(__name__) # Use the module's own logger name

# --- Global for Simulation ---
g_simulated_bandwidth_bps = None  # None means no simulation, in bits per second
g_simulation_lock = threading.Lock()

def set_simulated_bandwidth(bps):
    """
    Sets the target simulated bandwidth.

    Args:
        bps (int or None): Target bandwidth in bits per second. None to disable simulation.
    """
    global g_simulated_bandwidth_bps
    with g_simulation_lock:
        g_simulated_bandwidth_bps = bps
        if bps is None:
            logger.info("=> NET_SIM: Throttling disabled (full speed).")
        else:
            logger.info(f"=> NET_SIM: Bandwidth target set to {bps / 1_000_000:.2f} Mbps.")

def get_current_simulated_bandwidth():
    """
    Gets the current target simulated bandwidth.

    Returns:
        int or None: Current bandwidth in bps, or None if disabled.
    """
    with g_simulation_lock:
        return g_simulated_bandwidth_bps

def throttle_data_transfer(data_to_send, target_bps, output_stream, segment_name_for_log="Unknown Segment"):
    """
    Sends data_to_send to output_stream, throttled to target_bps.
    This function will write to output_stream and introduce delays.
    It can raise socket.error if output_stream.write() fails.

    Args:
        data_to_send (bytes): The byte string of data to send.
        target_bps (int): The target bandwidth in bits per second.
        output_stream (file-like): The stream to write to (e.g., self.wfile in a handler).
        segment_name_for_log (str): Name of the segment for logging purposes.

    Returns:
        float: The expected transfer time in seconds based on target_bps and data_size.
    
    Raises:
        socket.error: If an error occurs during writing to the output_stream.
    """
    data_size_bytes = len(data_to_send)
    if data_size_bytes == 0:
        return 0.0

    simulated_bytes_per_sec = target_bps / 8
    if simulated_bytes_per_sec <= 0: # Avoid division by zero or negative rates
        # If rate is zero or invalid, effectively infinite time, or send as one chunk with minimal delay
        # For practical purposes, let's assume it's a very slow rate but not zero.
        # This case should ideally be handled by the caller ensuring target_bps > 0.
        # If it still happens, we can treat it as a minimal positive rate.
        logger.warning(f"NET_SIM: Invalid target_bps ({target_bps}). Assuming a very slow rate for segment {segment_name_for_log}.")
        simulated_bytes_per_sec = 1 # 1 byte per second (extremely slow)

    expected_transfer_time_seconds = data_size_bytes / simulated_bytes_per_sec

    # Log before starting the actual throttled sending
    logger.info(
        f"NET_SIM: Simulating download at {target_bps / 1_000_000:.2f} Mbps "
        f"for {segment_name_for_log} ({data_size_bytes / 1024:.1f} KB), "
        f"expected time: {expected_transfer_time_seconds:.2f}s"
    )

    chunk_size = 4 * 1024  # Send in 4KB chunks
    bytes_sent = 0

    # The caller (proxy handler) should catch socket.error from this block
    while bytes_sent < data_size_bytes:
        chunk = data_to_send[bytes_sent : bytes_sent + chunk_size]
        if not chunk: # Should not happen if data_size_bytes > 0
            break
        
        output_stream.write(chunk)
        output_stream.flush() # Ensure data is sent over the socket

        bytes_sent += len(chunk)
        
        # Calculate delay for this chunk based on simulated bandwidth
        delay_for_chunk = len(chunk) / simulated_bytes_per_sec
        time.sleep(delay_for_chunk) # Introduce delay
    
    return expected_transfer_time_seconds

class NetworkScenarioPlayer:
    """
    Manages and plays a sequence of network bandwidth changes in a separate thread.
    """
    def __init__(self):
        self.scenario_steps = [] # List of (delay_before_next_step_seconds, bandwidth_bps_or_None)
        self._thread = None
        self._stop_event = threading.Event() # Used to signal the thread to stop

    def add_step(self, duration_seconds, bandwidth_bps):
        """
        Adds a step to the simulation scenario.
        Each step defines a bandwidth that will be active for a certain duration.

        Args:
            duration_seconds (float): How long this bandwidth setting should last.
            bandwidth_bps (int or None): Target bandwidth in bps for this step. None for full speed.
        """
        self.scenario_steps.append((duration_seconds, bandwidth_bps))
        return self # Allow chaining

    def _play_scenario_target(self):
        logger.info("SIM_CTRL: Network simulation scenario player thread started.")
        total_elapsed_time_for_logging = 0

        for i, (duration_seconds, bandwidth_bps) in enumerate(self.scenario_steps):
            if self._stop_event.is_set():
                logger.info("SIM_CTRL: Stop event detected, terminating scenario early.")
                break
            
            step_description = f"{bandwidth_bps / 1_000_000:.2f} Mbps" if bandwidth_bps is not None else "Full Speed"
            logger.info(
                f"SIM_CTRL: Step {i+1}/{len(self.scenario_steps)} - Setting bandwidth to {step_description} "
                f"for {duration_seconds:.1f}s (Total elapsed in scenario: {total_elapsed_time_for_logging:.1f}s)"
            )
            set_simulated_bandwidth(bandwidth_bps)
            
            if duration_seconds > 0:
                # Wait for the duration of this step, or until stop_event is set
                self._stop_event.wait(timeout=duration_seconds)
            
            total_elapsed_time_for_logging += duration_seconds

        if not self._stop_event.is_set(): # If scenario completed naturally
             logger.info("SIM_CTRL: All scenario steps completed.")
             # Consider setting to None at the very end if not explicitly done by the last step
             # For example: if self.scenario_steps and self.scenario_steps[-1][1] is not None:
             # set_simulated_bandwidth(None) # Optional: reset to full speed after scenario
        logger.info("SIM_CTRL: Network simulation scenario player thread finished.")

    def start(self):
        """Starts playing the scenario in a new thread."""
        if self._thread and self._thread.is_alive():
            logger.warning("SIM_CTRL: Scenario player thread is already running.")
            return
        
        if not self.scenario_steps:
            logger.warning("SIM_CTRL: No scenario steps defined. Player will not start.")
            return

        self._stop_event.clear() # Clear stop event from previous runs
        self._thread = threading.Thread(target=self._play_scenario_target, daemon=True, name="SimScenarioThread")
        self._thread.start()
        logger.info("SIM_CTRL: Scenario player initiated.")

    def stop(self):
        """Signals the scenario player thread to stop and waits for it to join."""
        if self._thread and self._thread.is_alive():
            logger.info("SIM_CTRL: Signaling scenario player thread to stop...")
            self._stop_event.set() # Signal the thread to stop
            self._thread.join(timeout=5.0) # Wait for the thread to finish
            if self._thread.is_alive():
                logger.warning("SIM_CTRL: Scenario player thread did not stop cleanly within timeout.")
            else:
                logger.info("SIM_CTRL: Scenario player thread stopped.")
        self._thread = None # Clear the thread reference

# --- Example default scenario ---
def create_default_simulation_scenario():
    """Creates a default network simulation scenario."""
    player = NetworkScenarioPlayer()
    player.add_step(20, None)  # Start with 20s of full speed (allows initial buffering)
    player.add_step(40, 5_000_000)   # Then, 40s at 5 Mbps
    player.add_step(60, 800_000)    # Then, 60s at 0.8 Mbps
    player.add_step(60, 10_000_000)  # Then, 60s at 10 Mbps
    # Example of more rapid fluctuation for 20s total
    player.add_step(5, 500_000)    # 5s at 0.5 Mbps
    player.add_step(5, 2_000_000)   # 5s at 2 Mbps
    player.add_step(5, 500_000)    # 5s at 0.5 Mbps
    player.add_step(5, 2_000_000)   # 5s at 2 Mbps
    player.add_step(30, None)       # Finally, 30s more at full speed
    return player

if __name__ == '__main__':
    # Basic test for the simulator module (optional)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')
    logger.info("Testing network_simulator.py standalone...")
    
    test_player = create_default_simulation_scenario()
    test_player.start()
    
    try:
        count = 0
        while test_player._thread and test_player._thread.is_alive() and count < 300: # Run for max 5 minutes
            time.sleep(1)
            current_bw = get_current_simulated_bandwidth()
            # logger.info(f"MainTest: Current simulated BW: {current_bw / 1_000_000 if current_bw else 'None'} Mbps")
            count +=1
    except KeyboardInterrupt:
        logger.info("MainTest: Interrupted by user.")
    finally:
        test_player.stop()
        logger.info("MainTest: Finished.")