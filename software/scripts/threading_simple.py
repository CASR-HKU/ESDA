import threading
import queue
import time
import random


class CountingThread:
    def __init__(self):
        self.number_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.thread = threading.Thread(target=self._counting_loop)

    def start(self):
        self.thread.start()

    def pause(self):
        self.pause_event.set()

    def resume(self):
        self.pause_event.clear()

    def release(self):
        self.stop_event.set()
        self.thread.join()
        if not self.number_queue.empty():
            average_number = sum(self.number_queue.queue) / self.number_queue.qsize()
            print(f"Average number: {average_number:.2f}")
        else:
            print("No numbers generated")

    def _counting_loop(self):
        while not self.stop_event.is_set():
            if not self.pause_event.is_set():
                # Generate a random number from 0-10
                number = random.randint(0, 10)
                print(number)

                # Store the number in the queue
                self.number_queue.put(number)

            # Sleep for 1 second or until the stop event is set
            self.stop_event.wait(timeout=1)


# Define the function that will be run for 10 seconds
def main_thread():
    # Sleep for 10 seconds
    time.sleep(10)

    # Your code here
    # ...
    # ...


# Create a counting thread object
counting_thread = CountingThread()

# Start the counting thread
counting_thread.start()

# Run the main thread for 10 seconds
main_thread()

# Pause the counting thread after 10 seconds
counting_thread.pause()
print("Paused")

# Resume the counting thread after 5 seconds
time.sleep(5)
counting_thread.resume()
print("Resumed")

# Release the counting thread after 5 more seconds
time.sleep(5)
counting_thread.release()