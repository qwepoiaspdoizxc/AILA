import os
import sys
import psutil
import platform
import threading
import time
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import json

from utils.logger import get_logger
from config import CONFIG

logger = get_logger("SystemUtils")


class SystemMonitor:
    """Monitor system resources and performance."""

    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
        self.resource_history = {"cpu": [], "memory": [], "disk": [], "network": []}
        self.max_history_size = 100

    def start_monitoring(self, interval: float = 5.0):
        """Start system resource monitoring."""
        if self.monitoring:
            logger.warning("System monitoring already running")
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, args=(interval,), daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"Started system monitoring with {interval}s interval")

    def stop_monitoring(self):
        """Stop system resource monitoring."""
        if not self.monitoring:
            return

        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("Stopped system monitoring")

    def _monitor_loop(self, interval: float):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                timestamp = datetime.now().isoformat()

                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self._add_to_history(
                    "cpu", {"timestamp": timestamp, "value": cpu_percent}
                )

                # Memory usage
                memory = psutil.virtual_memory()
                memory_data = {
                    "timestamp": timestamp,
                    "percent": memory.percent,
                    "available": memory.available,
                    "total": memory.total,
                    "used": memory.used,
                }
                self._add_to_history("memory", memory_data)

                # Disk usage for watched directories
                disk_data = []
                for directory in CONFIG.get("directories_to_watch", []):
                    if os.path.exists(directory):
                        try:
                            disk_usage = psutil.disk_usage(directory)
                            disk_data.append(
                                {
                                    "path": directory,
                                    "percent": (disk_usage.used / disk_usage.total)
                                    * 100,
                                    "free": disk_usage.free,
                                    "total": disk_usage.total,
                                    "used": disk_usage.used,
                                }
                            )
                        except Exception as e:
                            logger.error(
                                f"Error getting disk usage for {directory}: {e}"
                            )

                self._add_to_history(
                    "disk", {"timestamp": timestamp, "disks": disk_data}
                )

                # Network I/O
                net_io = psutil.net_io_counters()
                net_data = {
                    "timestamp": timestamp,
                    "bytes_sent": net_io.bytes_sent,
                    "bytes_recv": net_io.bytes_recv,
                    "packets_sent": net_io.packets_sent,
                    "packets_recv": net_io.packets_recv,
                }
                self._add_to_history("network", net_data)

                time.sleep(interval)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval)

    def _add_to_history(self, resource_type: str, data: Dict):
        """Add data to resource history with size limit."""
        history = self.resource_history[resource_type]
        history.append(data)

        # Keep only recent history
        if len(history) > self.max_history_size:
            history.pop(0)

    def get_current_stats(self) -> Dict[str, Any]:
        """Get current system statistics."""
        try:
            stats = {
                "timestamp": datetime.now().isoformat(),
                "cpu": {
                    "percent": psutil.cpu_percent(),
                    "count": psutil.cpu_count(),
                    "freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                },
                "memory": psutil.virtual_memory()._asdict(),
                "disk": {},
                "network": psutil.net_io_counters()._asdict(),
                "processes": len(psutil.pids()),
                "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat(),
            }

            # Add disk info for watched directories
            for directory in CONFIG.get("directories_to_watch", []):
                if os.path.exists(directory):
                    try:
                        stats["disk"][directory] = psutil.disk_usage(
                            directory
                        )._asdict()
                    except Exception as e:
                        logger.error(f"Error getting disk stats for {directory}: {e}")

            return stats

        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {}

    def get_resource_history(self, resource_type: str = None) -> Dict:
        """Get resource history for specified type or all types."""
        if resource_type:
            return {resource_type: self.resource_history.get(resource_type, [])}
        return self.resource_history.copy()


class SystemInfo:
    """Gather comprehensive system information."""

    @staticmethod
    def get_platform_info() -> Dict[str, Any]:
        """Get platform and OS information."""
        try:
            return {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "architecture": platform.architecture(),
                "python_version": platform.python_version(),
                "python_build": platform.python_build(),
                "node": platform.node(),
                "platform": platform.platform(),
            }
        except Exception as e:
            logger.error(f"Error getting platform info: {e}")
            return {}

    @staticmethod
    def get_hardware_info() -> Dict[str, Any]:
        """Get hardware information."""
        try:
            info = {
                "cpu_count": psutil.cpu_count(),
                "cpu_count_logical": psutil.cpu_count(logical=True),
                "memory_total": psutil.virtual_memory().total,
                "swap_total": psutil.swap_memory().total,
                "disk_partitions": [],
            }

            # Get disk partition info
            for partition in psutil.disk_partitions():
                try:
                    partition_usage = psutil.disk_usage(partition.mountpoint)
                    info["disk_partitions"].append(
                        {
                            "device": partition.device,
                            "mountpoint": partition.mountpoint,
                            "fstype": partition.fstype,
                            "total": partition_usage.total,
                            "used": partition_usage.used,
                            "free": partition_usage.free,
                            "percent": (partition_usage.used / partition_usage.total)
                            * 100,
                        }
                    )
                except Exception as e:
                    logger.error(
                        f"Error getting partition info for {partition.device}: {e}"
                    )

            return info

        except Exception as e:
            logger.error(f"Error getting hardware info: {e}")
            return {}

    @staticmethod
    def get_process_info(limit: int = 10) -> List[Dict[str, Any]]:
        """Get information about running processes."""
        try:
            processes = []
            for proc in psutil.process_iter(
                ["pid", "name", "cpu_percent", "memory_percent", "status"]
            ):
                try:
                    processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            # Sort by CPU usage and return top processes
            processes.sort(key=lambda x: x.get("cpu_percent", 0), reverse=True)
            return processes[:limit]

        except Exception as e:
            logger.error(f"Error getting process info: {e}")
            return []


class FileSystemUtils:
    """File system utilities and operations."""

    @staticmethod
    def get_directory_size(path: str) -> int:
        """Get total size of directory and all subdirectories."""
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        total_size += os.path.getsize(filepath)
                    except (OSError, FileNotFoundError):
                        continue
            return total_size
        except Exception as e:
            logger.error(f"Error calculating directory size for {path}: {e}")
            return 0

    @staticmethod
    def get_file_stats(filepath: str) -> Dict[str, Any]:
        """Get comprehensive file statistics."""
        try:
            stat = os.stat(filepath)
            return {
                "size": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "accessed": datetime.fromtimestamp(stat.st_atime).isoformat(),
                "mode": oct(stat.st_mode),
                "uid": stat.st_uid,
                "gid": stat.st_gid,
                "is_file": os.path.isfile(filepath),
                "is_dir": os.path.isdir(filepath),
                "is_link": os.path.islink(filepath),
                "readable": os.access(filepath, os.R_OK),
                "writable": os.access(filepath, os.W_OK),
                "executable": os.access(filepath, os.X_OK),
            }
        except Exception as e:
            logger.error(f"Error getting file stats for {filepath}: {e}")
            return {}

    @staticmethod
    def find_large_files(
        directory: str, min_size_mb: int = 100, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Find large files in directory."""
        try:
            large_files = []
            min_size_bytes = min_size_mb * 1024 * 1024

            for root, dirs, files in os.walk(directory):
                for file in files:
                    filepath = os.path.join(root, file)
                    try:
                        size = os.path.getsize(filepath)
                        if size >= min_size_bytes:
                            large_files.append(
                                {
                                    "path": filepath,
                                    "size": size,
                                    "size_mb": size / (1024 * 1024),
                                    "modified": datetime.fromtimestamp(
                                        os.path.getmtime(filepath)
                                    ).isoformat(),
                                }
                            )
                    except (OSError, FileNotFoundError):
                        continue

            # Sort by size descending
            large_files.sort(key=lambda x: x["size"], reverse=True)
            return large_files[:limit]

        except Exception as e:
            logger.error(f"Error finding large files in {directory}: {e}")
            return []

    @staticmethod
    def find_old_files(
        directory: str, days_old: int = 365, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Find old files that haven't been accessed recently."""
        try:
            old_files = []
            cutoff_date = datetime.now() - timedelta(days=days_old)
            cutoff_timestamp = cutoff_date.timestamp()

            for root, dirs, files in os.walk(directory):
                for file in files:
                    filepath = os.path.join(root, file)
                    try:
                        stat = os.stat(filepath)
                        if stat.st_atime < cutoff_timestamp:
                            old_files.append(
                                {
                                    "path": filepath,
                                    "size": stat.st_size,
                                    "last_accessed": datetime.fromtimestamp(
                                        stat.st_atime
                                    ).isoformat(),
                                    "modified": datetime.fromtimestamp(
                                        stat.st_mtime
                                    ).isoformat(),
                                }
                            )
                    except (OSError, FileNotFoundError):
                        continue

            # Sort by last access time
            old_files.sort(key=lambda x: x["last_accessed"])
            return old_files[:limit]

        except Exception as e:
            logger.error(f"Error finding old files in {directory}: {e}")
            return []

    @staticmethod
    def clean_temp_files(temp_dirs: List[str] = None) -> Dict[str, Any]:
        """Clean temporary files and directories."""
        if temp_dirs is None:
            temp_dirs = [
                "temp",
                "tmp",
                os.environ.get("TEMP", ""),
                os.environ.get("TMP", ""),
            ]

        cleanup_stats = {"files_deleted": 0, "space_freed": 0, "errors": []}

        for temp_dir in temp_dirs:
            if not temp_dir or not os.path.exists(temp_dir):
                continue

            try:
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        filepath = os.path.join(root, file)
                        try:
                            # Only delete files older than 1 hour
                            if time.time() - os.path.getmtime(filepath) > 3600:
                                size = os.path.getsize(filepath)
                                os.remove(filepath)
                                cleanup_stats["files_deleted"] += 1
                                cleanup_stats["space_freed"] += size
                        except Exception as e:
                            cleanup_stats["errors"].append(
                                f"Error deleting {filepath}: {e}"
                            )

            except Exception as e:
                cleanup_stats["errors"].append(f"Error accessing {temp_dir}: {e}")

        logger.info(
            f"Temp cleanup: {cleanup_stats['files_deleted']} files deleted, "
            f"{cleanup_stats['space_freed'] / 1024 / 1024:.2f} MB freed"
        )
        return cleanup_stats


class SystemHealth:
    """System health monitoring and diagnostics."""

    @staticmethod
    def check_disk_space(
        warning_threshold: float = 0.8, critical_threshold: float = 0.9
    ) -> Dict[str, Any]:
        """Check disk space usage and return warnings."""
        results = {
            "status": "healthy",
            "warnings": [],
            "critical": [],
            "disk_usage": {},
        }

        try:
            for directory in CONFIG.get("directories_to_watch", []):
                if os.path.exists(directory):
                    usage = psutil.disk_usage(directory)
                    usage_percent = usage.used / usage.total

                    results["disk_usage"][directory] = {
                        "percent": usage_percent * 100,
                        "free_gb": usage.free / (1024**3),
                        "total_gb": usage.total / (1024**3),
                    }

                    if usage_percent >= critical_threshold:
                        results["critical"].append(
                            f"Critical disk space: {directory} ({usage_percent * 100:.1f}% full)"
                        )
                        results["status"] = "critical"
                    elif usage_percent >= warning_threshold:
                        results["warnings"].append(
                            f"Low disk space: {directory} ({usage_percent * 100:.1f}% full)"
                        )
                        if results["status"] == "healthy":
                            results["status"] = "warning"

        except Exception as e:
            logger.error(f"Error checking disk space: {e}")
            results["status"] = "error"

        return results

    @staticmethod
    def check_memory_usage(
        warning_threshold: float = 0.8, critical_threshold: float = 0.9
    ) -> Dict[str, Any]:
        """Check memory usage."""
        try:
            memory = psutil.virtual_memory()
            usage_percent = memory.percent / 100

            result = {
                "status": "healthy",
                "percent": memory.percent,
                "available_gb": memory.available / (1024**3),
                "total_gb": memory.total / (1024**3),
            }

            if usage_percent >= critical_threshold:
                result["status"] = "critical"
                result["message"] = f"Critical memory usage: {memory.percent:.1f}%"
            elif usage_percent >= warning_threshold:
                result["status"] = "warning"
                result["message"] = f"High memory usage: {memory.percent:.1f}%"

            return result

        except Exception as e:
            logger.error(f"Error checking memory usage: {e}")
            return {"status": "error", "message": str(e)}

    @staticmethod
    def check_cpu_usage(
        warning_threshold: float = 0.8,
        critical_threshold: float = 0.9,
        interval: int = 3,
    ) -> Dict[str, Any]:
        """Check CPU usage over a period."""
        try:
            cpu_percent = psutil.cpu_percent(interval=interval)
            usage_percent = cpu_percent / 100

            result = {
                "status": "healthy",
                "percent": cpu_percent,
                "cores": psutil.cpu_count(),
            }

            if usage_percent >= critical_threshold:
                result["status"] = "critical"
                result["message"] = f"Critical CPU usage: {cpu_percent:.1f}%"
            elif usage_percent >= warning_threshold:
                result["status"] = "warning"
                result["message"] = f"High CPU usage: {cpu_percent:.1f}%"

            return result

        except Exception as e:
            logger.error(f"Error checking CPU usage: {e}")
            return {"status": "error", "message": str(e)}

    @staticmethod
    def run_system_diagnostics() -> Dict[str, Any]:
        """Run comprehensive system diagnostics."""
        diagnostics = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "checks": {},
        }

        # Check disk space
        disk_check = SystemHealth.check_disk_space()
        diagnostics["checks"]["disk_space"] = disk_check
        if disk_check["status"] in ["warning", "critical"]:
            diagnostics["overall_status"] = disk_check["status"]

        # Check memory
        memory_check = SystemHealth.check_memory_usage()
        diagnostics["checks"]["memory"] = memory_check
        if memory_check["status"] == "critical":
            diagnostics["overall_status"] = "critical"
        elif (
            memory_check["status"] == "warning"
            and diagnostics["overall_status"] == "healthy"
        ):
            diagnostics["overall_status"] = "warning"

        # Check CPU
        cpu_check = SystemHealth.check_cpu_usage()
        diagnostics["checks"]["cpu"] = cpu_check
        if cpu_check["status"] == "critical":
            diagnostics["overall_status"] = "critical"
        elif (
            cpu_check["status"] == "warning"
            and diagnostics["overall_status"] == "healthy"
        ):
            diagnostics["overall_status"] = "warning"

        return diagnostics


def format_bytes(bytes_value: int) -> str:
    """Format bytes into human readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"


def is_process_running(process_name: str) -> bool:
    """Check if a process is currently running."""
    try:
        for proc in psutil.process_iter(["name"]):
            if proc.info["name"] and process_name.lower() in proc.info["name"].lower():
                return True
        return False
    except Exception as e:
        logger.error(f"Error checking if process {process_name} is running: {e}")
        return False


def kill_process_by_name(process_name: str) -> bool:
    """Kill processes by name."""
    try:
        killed = False
        for proc in psutil.process_iter(["pid", "name"]):
            if proc.info["name"] and process_name.lower() in proc.info["name"].lower():
                try:
                    proc.terminate()
                    killed = True
                    logger.info(
                        f"Terminated process: {proc.info['name']} (PID: {proc.info['pid']})"
                    )
                except Exception as e:
                    logger.error(f"Error terminating process {proc.info['name']}: {e}")
        return killed
    except Exception as e:
        logger.error(f"Error killing process {process_name}: {e}")
        return False


def get_system_summary() -> Dict[str, Any]:
    """Get a comprehensive system summary."""
    try:
        summary = {
            "timestamp": datetime.now().isoformat(),
            "platform": SystemInfo.get_platform_info(),
            "hardware": SystemInfo.get_hardware_info(),
            "current_stats": SystemMonitor().get_current_stats(),
            "health": SystemHealth.run_system_diagnostics(),
            "top_processes": SystemInfo.get_process_info(5),
        }
        return summary
    except Exception as e:
        logger.error(f"Error generating system summary: {e}")
        return {"error": str(e)}


# Initialize global system monitor instance
system_monitor = SystemMonitor()


def start_system_monitoring():
    """Start the global system monitor."""
    system_monitor.start_monitoring()


def stop_system_monitoring():
    """Stop the global system monitor."""
    system_monitor.stop_monitoring()


def get_system_stats():
    """Get current system statistics from the global monitor."""
    return system_monitor.get_current_stats()
