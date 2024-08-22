#!/bin/bash

# 双重循环
for ((m_max=1; m_max<32; m_max++)); do
    for ((n_max=1; n_max<32; n_max++)); do
        # 检查m_max * n_max是否小于32
        if (( m_max * n_max < 32 )); then
            # 声明环境变量
            export M_MAX=$m_max
            export N_MAX=$n_max

            # 打印当前的M_MAX和N_MAX
            echo "Running with M_MAX=$M_MAX and N_MAX=$N_MAX"

            # 运行命令
            ./llama-bench -m ../loong-llama/llama.cpp/models/models_with_blas/ggml-model-Q4_0.gguf -p 32 -n 32 -t 8

            # 检查是否出现浮点数例外错误
            if [[ $? -ne 0 ]]; then
                echo "Error occurred with M_MAX=$M_MAX and N_MAX=$N_MAX"
                exit 1
            fi
        fi
    done
done
