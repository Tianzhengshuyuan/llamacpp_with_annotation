#!/bin/bash
# 双重循环
for ((m_max=4; m_max<5; m_max++)); do
    for ((n_max=4; n_max<5; n_max++)); do
        # 检查m_max * n_max是否小于32
        if (( m_max * n_max < 32 )); then
            SOURCE_FILE="sgemm.cpp"

            # 打印当前的M_MAX和N_MAX
            echo "Running with M_MAX=$M_MAX and N_MAX=$N_MAX"

            # 运行命令
            /usr/lib/linux-tools/5.15.0-119-generic/perf record -g -o tiny_prompt.data ./llama-bench -m /mnt/nfs_share/LLMs/llama2-7b/ggml-model-q4_0.gguf -p 32 -n 32 -t 8

            # 检查是否出现浮点数例外错误
            if [[ $? -ne 0 ]]; then
                echo "Error occurred with M_MAX=$M_MAX and N_MAX=$N_MAX"
                exit 1
            fi
        fi
    done
done
