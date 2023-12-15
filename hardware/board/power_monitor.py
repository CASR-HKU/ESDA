import time
import sensors  # type: ignore
import numpy as np

# units could be W or mW in `sensors`
# units are all W in this library
pmbus_mapping = {
    # PS 10 sensors
    "u76": "VCCPSINTFP",  # W
    "u77": "VCCPSINTLP",  # mW
    "u78": "VCCPSAUX",  # mW
    "u87": "VCCPSPLL",  # mW
    "u85": "MGTRAVCC",  # mW
    "u86": "MGTRAVTT",  # mW
    "u93": "VCCO_PSDDR_504",  # mW
    "u88": "VCCOPS",  # W
    "u15": "VCCOPS3",  # W
    "u92": "VCCPSDDRPLL",  # mW
    # PL 8 sensors
    "u79": "VCCINT",  # W
    "u81": "VCCBRAM",  # mW
    "u80": "VCCAUX",  # mW
    "u84": "VCC1V2",  # mW
    "u16": "VCC3V3",  # mW
    "u65": "VADJ_FMC",  # W
    "u74": "MGTAVCC",  # mW
    "u75": "MGTAVTT",  # mW
}


class PowerMonitor:
    def __init__(self):
        self.sensor_dict = dict.fromkeys(pmbus_mapping.values(), None)
        sensors.init()
        for s in sensors.iter_detected_chips():
            if s.prefix.startswith(b"ina226_"):
                uxx = s.prefix.decode("utf-8").split("-")[0].split("_")[1]
                k = pmbus_mapping[uxx]
                assert self.sensor_dict[k] is None, f"Duplicate sensor {k}"
                self.sensor_dict[k] = s
        for k, s in self.sensor_dict.items():
            assert s is not None, f"Sensor {k} not found"

    def __del__(self):
        sensors.cleanup()

    def record(self, interval: float, num_runs: int):
        overhead = 0.027
        # get overhead by running with real_interval = 0
        real_interval = interval - overhead
        results = np.zeros((0, 1 + 4 * len(self.sensor_dict)))
        start_time = time.time()
        for idx in range(num_runs):
            try:
                result = [time.time()]
                for s in self.sensor_dict.values():
                    result.extend([f.get_value() for f in s.__iter__()])
                results = np.vstack((results, result))
                time.sleep(real_interval)
            except KeyboardInterrupt:
                print("Get SIGINT")
                break
            except Exception as e:
                print(e)
        end_time = time.time()
        print(
            f"pm time(s): {end_time:.4f} - {start_time:.4f}"
            + f" = {end_time - start_time:.4f} / {idx+1}"
            + f" = {(end_time - start_time)/(idx+1):.4f}"
        )
        np.save("power_record.npy", results)


if __name__ == "__main__":
    interval = 0.1  # ms
    num_runs = 6000  # 10 minutes
    pm = PowerMonitor()
    pm.record(interval, num_runs)
