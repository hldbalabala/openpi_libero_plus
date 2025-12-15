#!/usr/bin/env python3
"""
脚本用于修复 libero_goal 目录下所有 .bddl 文件中的 akita_black_bowl_region 的 :ranges 值。
如果第0项或第2项大于0.2，则将它们各减去0.1。
"""

import re
import os
from pathlib import Path


def process_bddl_file(file_path):
    """处理单个 .bddl 文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找 akita_black_bowl_region 块中的 :ranges 部分
    # 匹配模式：在 akita_black_bowl_region 块内找到 :ranges 及其数值
    pattern = r'(akita_black_bowl_region.*?:ranges\s+\(\s*\(\s*)([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)(\s*\)\s*\)\s*\))'
    
    def replace_ranges(match):
        prefix = match.group(1)
        val0 = float(match.group(2))
        val1 = float(match.group(3))
        val2 = float(match.group(4))
        val3 = float(match.group(5))
        suffix = match.group(6)
        
        modified = False
        # 如果第0项或第2项大于0.2，则各减去0.1
        if val0 > 0.2 or val2 > 0.2:
            val0 -= 0.1
            val2 -= 0.1
            modified = True
        
        if modified:
            # 保持原有的格式，使用空格分隔
            return f"{prefix}{val0} {val1} {val2} {val3}{suffix}"
        else:
            return match.group(0)
    
    new_content = re.sub(pattern, replace_ranges, content, flags=re.DOTALL)
    
    # 如果内容有变化，写回文件
    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return True
    return False


def main():
    """主函数"""
    bddl_dir = Path("/home/ubuntu/Desktop/hld/openpi/third_party/LIBERO-plus/libero/libero/bddl_files/libero_goal")
    
    if not bddl_dir.exists():
        print(f"错误：目录不存在: {bddl_dir}")
        return
    
    # 获取所有 .bddl 文件
    bddl_files = list(bddl_dir.glob("*.bddl"))
    
    if not bddl_files:
        print(f"未找到 .bddl 文件在目录: {bddl_dir}")
        return
    
    print(f"找到 {len(bddl_files)} 个 .bddl 文件")
    
    modified_count = 0
    for bddl_file in bddl_files:
        try:
            if process_bddl_file(bddl_file):
                modified_count += 1
                print(f"已修改: {bddl_file.name}")
        except Exception as e:
            print(f"处理文件 {bddl_file.name} 时出错: {e}")
    
    print(f"\n处理完成！共修改了 {modified_count} 个文件")


if __name__ == "__main__":
    main()

