"""
Resource monitoring for safe training.
Checks CPU, memory usage and warns/throttles if system is under stress.
"""

import logging
import os

import psutil

logger = logging.getLogger(__name__)

# Thresholds
CPU_WARN_THRESHOLD = 85  # Warn if CPU > 85%
CPU_CRITICAL_THRESHOLD = 95  # Critical if CPU > 95%
MEMORY_WARN_THRESHOLD = 80  # Warn if memory > 80%
MEMORY_CRITICAL_THRESHOLD = 90  # Critical if memory > 90%


def get_system_stats() -> dict:
    """Get current system resource usage."""
    return {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "cpu_count": psutil.cpu_count(),
        "memory_total_gb": psutil.virtual_memory().total / (1024**3),
        "memory_available_gb": psutil.virtual_memory().available / (1024**3),
    }


def check_resources() -> tuple[bool, str]:
    """
    Check if system resources are healthy.
    Returns (is_safe, message).
    """
    stats = get_system_stats()
    
    issues = []
    is_safe = True
    
    if stats["cpu_percent"] > CPU_CRITICAL_THRESHOLD:
        issues.append(f"CPU critical: {stats['cpu_percent']:.1f}%")
        is_safe = False
    elif stats["cpu_percent"] > CPU_WARN_THRESHOLD:
        issues.append(f"CPU high: {stats['cpu_percent']:.1f}%")
    
    if stats["memory_percent"] > MEMORY_CRITICAL_THRESHOLD:
        issues.append(f"Memory critical: {stats['memory_percent']:.1f}%")
        is_safe = False
    elif stats["memory_percent"] > MEMORY_WARN_THRESHOLD:
        issues.append(f"Memory high: {stats['memory_percent']:.1f}%")
    
    if issues:
        return is_safe, " | ".join(issues)
    return True, f"OK (CPU: {stats['cpu_percent']:.1f}%, Mem: {stats['memory_percent']:.1f}%)"


def recommend_nproc() -> int:
    """Recommend number of processes based on available resources."""
    stats = get_system_stats()
    cpu_count = stats["cpu_count"]
    available_mem_gb = stats["memory_available_gb"]
    
    # Conservative: use half of CPUs, leave headroom
    max_by_cpu = max(1, cpu_count // 2)
    
    # Estimate ~2GB per process for safety
    max_by_memory = max(1, int(available_mem_gb / 2))
    
    recommended = min(max_by_cpu, max_by_memory)
    return recommended


def pre_training_check(nproc: int = 2) -> tuple[bool, int]:
    """
    Run pre-training health check.
    Returns (should_proceed, adjusted_nproc).
    """
    stats = get_system_stats()
    is_safe, msg = check_resources()
    
    logger.info(f"System: {stats['cpu_count']} CPUs, {stats['memory_total_gb']:.1f}GB RAM")
    logger.info(f"Current usage: {msg}")
    
    recommended = recommend_nproc()
    
    if not is_safe:
        logger.warning(f"System under stress! Recommended nproc: {recommended}")
        if nproc > recommended:
            logger.warning(f"Reducing nproc from {nproc} to {recommended}")
            return True, recommended
        return False, nproc
    
    if nproc > recommended:
        logger.warning(f"Requested nproc={nproc} > recommended={recommended}. Proceeding with caution.")
    
    return True, nproc


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    stats = get_system_stats()
    print(f"\nSystem Stats:")
    print(f"  CPUs: {stats['cpu_count']}")
    print(f"  RAM: {stats['memory_total_gb']:.1f}GB total, {stats['memory_available_gb']:.1f}GB available")
    print(f"  CPU usage: {stats['cpu_percent']:.1f}%")
    print(f"  Memory usage: {stats['memory_percent']:.1f}%")
    
    is_safe, msg = check_resources()
    print(f"\nHealth check: {msg}")
    print(f"Recommended nproc: {recommend_nproc()}")
