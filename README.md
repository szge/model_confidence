# model_confidence
a minimal tool for generating a "confidence score" from a LLM's next token distribution


### Instructions
1. Clone repo i.e. `git clone https://github.com/szge/model_confidence.git`
2. Download the model of your choice; I used `huggingface-cli` on Python to install qwen2.5-7b quantized
    ```shell
    huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF \
        --include "qwen2.5-7b-instruct-q8_0-000*.gguf" \
        --local-dir ./qwen2.5-7b/
    ```
3. Clone the llama.cpp GitHub repo in the project's root directory; it should exist under model_confidence/llama.cpp (I build it as a dynamic library)
    ```shell
    git clone https://github.com/ggml-org/llama.cpp.git
    cmake -B build
    cmake --build build --config Release
    ```
4. Run the CMake project in the repository root