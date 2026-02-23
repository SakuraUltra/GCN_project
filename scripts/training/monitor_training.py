#!/usr/bin/env python3
"""
Training Monitor Script
实时监控训练进度
"""

import os
import sys
import time
import argparse
from pathlib import Path


def monitor_training_log(log_path, refresh_interval=5, tail_lines=20):
    """
    监控训练日志文件
    
    Args:
        log_path: 日志文件路径
        refresh_interval: 刷新间隔（秒）
        tail_lines: 显示最后N行
    """
    print(f"📊 Monitoring training log: {log_path}")
    print(f"🔄 Refresh interval: {refresh_interval}s")
    print("=" * 80)
    
    last_size = 0
    
    try:
        while True:
            if not os.path.exists(log_path):
                print(f"⏳ Waiting for log file to be created...")
                time.sleep(refresh_interval)
                continue
            
            current_size = os.path.getsize(log_path)
            
            if current_size > last_size:
                # 读取新内容
                with open(log_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # 清屏并显示最新内容
                os.system('clear' if os.name != 'nt' else 'cls')
                
                print(f"📊 Training Monitor - {time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"📝 Log: {log_path}")
                print(f"📏 Size: {current_size / 1024:.2f} KB")
                print("=" * 80)
                
                # 显示最后N行
                recent_lines = lines[-tail_lines:] if len(lines) > tail_lines else lines
                for line in recent_lines:
                    print(line.rstrip())
                
                print("=" * 80)
                print(f"🔄 Auto-refresh every {refresh_interval}s | Press Ctrl+C to stop")
                
                last_size = current_size
            
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n\n✋ Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(description='Monitor training progress')
    parser.add_argument('--log', type=str, 
                       default='outputs/bot_baseline_run1/training_epoch120.log',
                       help='Path to log file')
    parser.add_argument('--interval', type=int, default=5,
                       help='Refresh interval in seconds')
    parser.add_argument('--lines', type=int, default=30,
                       help='Number of lines to display')
    
    args = parser.parse_args()
    
    monitor_training_log(args.log, args.interval, args.lines)


if __name__ == '__main__':
    main()
