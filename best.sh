#!/bin/bash

# 文件路径
file_path="sgemm.cpp"

# 遍历 m 和 n 的所有可能取值
for ((m=1; m<32; m++)); do
    for ((n=1; n<32; n++)); do
        # 检查条件 m < 32, n < 32, m * n < 32
        if [ $((m * n)) -lt 32 ]; then
            cp "${file_path}.bak" $file_path 
            echo "Compiling with: m_max = $m, n_max = $n"
            
            # 使用sed替换代码行
            sed -i '/class tinyBLAS_Q0_LOONGARCH {/,/};/ {
                s/switch ((MIN(m - m0, 4) << 8) | MIN(n - n0, 4)) {/switch ((MIN(m - m0, '$m') << 8) | MIN(n - n0, '$n')) {/
            }' $file_path

            echo "Code replacement is complete."

            # 使用 make 编译项目
            make clean > /dev/null 2>&1
            make -j > /dev/null 2>&1
            if [ $? -eq 0 ]; then
                echo "Compiled successfully"
                # 假设编译出的可执行文件名为 my_program
                 ./llama-bench -m ../llama2-7b/ggml-q4_0.gguf -p 64 -n 64 -t 8
            else
                echo "Compile failed "
            fi
        fi
    done
done
