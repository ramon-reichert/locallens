# LocalLens

Built on top of the [Kronk SDK](https://github.com/ardanlabs/kronk), LocalLens is a local semantic image search application that explores how AI inference can be embedded directly into Go applications. By combining image understanding, vector embeddings, and semantic similarity search, it allows users to search for images stored on their own machine using natural language queries — all running offline with no cloud dependencies. 

LocalLens specifically focuses on using the smallest models possible while still delivering practical results, demonstrating how Kronk can power useful AI applications not only for developers with high-end setups, but also for everyday users on ordinary consumer hardware.


If you liked it, please consider giving this project a star on GitHub! Thank you!

## Kronk SDK usage

LocalLens uses Kronk for several capabilities:

- **Local runtime setup** — Kronk initializes the llama.cpp runtime and selects the available processor backend, such as CPU, CUDA, Vulkan, Metal, or another supported option. LocalLens can also download the required llama.cpp libraries during setup, and pins a specific llama.cpp build for release stability.

- **Model management** — Kronk downloads and resolves the local model files used by the app. LocalLens runs **three small models**: a vision-language model (Qwen2-VL-2B) to describe images, a tiny chat model (Qwen3-0.6B) to turn those descriptions into search terms, and an embedding model (embeddinggemma-300m) to convert text into searchable vectors. Each one performing their own specialty.

- **Local inference** — Kronk loads the models and runs inference directly on the user’s machine. LocalLens uses Kronk’s chat-style vision API to turn images into text descriptions, a second chat call to categorize them, and the embeddings API to turn both terms and search queries into vectors.

- **Grammar-constrained output** — the categorization step passes a JSON schema to Kronk, which converts it into a GBNF grammar and forces the model to emit *only* valid JSON. This turns a tiny 0.6B model into a reliable structured-output extractor with no fragile text parsing. (Qwen3’s `<think>` reasoning is also disabled via the request parameters for faster, more stable output.)

- **Anti-repetition sampling** — small vision models tend to loop or hallucinate on text-heavy images. LocalLens exposes Kronk’s **DRY (Don’t Repeat Yourself)** sampler plus repeat/frequency/presence penalties as request parameters. DRY penalizes repeated multi-word n-gram sequences
without suppressing legitimately repeated words, which cleanly breaks those loops.

With these pieces, LocalLens can index a folder of images, describe each image locally, reorganize each description into compact search terms, embed them, and later match natural language searches against them — all offline and without sending files or prompts to an external service.

## How LocalLens builds good search vectors

Getting relevant results from *small* local models takes more than embedding a raw image description. LocalLens uses a multi-vector pipeline:

- **Describe, then distill.** A vision model describes each image, then a tiny chat model turns that description into short, self-contained **search expressions** such as *“bright yellow parrot”* or *“dense tropical forest”*.

- **One vector per expression.** LocalLens embeds each expression separately, so a query can match one precise detail without being diluted by unrelated content in the same image.

- **Top-expression ranking.** At search time, `FindTopK` scores every expression against the query, keeps the **5 best-scoring expressions** for each image, and builds the image score from those top matches. The final score blends their mean similarity with the single best expression score, keeping precise matches high while still rewarding images that match the query in several ways.

## How to Use LocalLens

0) For devs only: Clone the repository and use the Makefile targets to run, build, and test the project. Go to step 3.

1) Download the executable from the GitHub Releases section.

2) Run the executable. The browser UI opens automatically.

3) At first launch, the setup panel opens automatically. There you can choose:
    - where the AI models will be downloaded
    - which `processor` will run inference
    - The processor is auto-detected based on your hardware, but you can switch to `cpu` if you experience issues, or choose another compatible processor if you have multiple GPUs available. These settings can be changed later, but restarting the application is required.

![setup_panel](https://github.com/ramon-reichert/locallens/blob/e5a84456d2209fbd89411aed09c54fd13ffe89dd/assets/setup_panel-0.jpeg)

4) Navigate to the folder containing the images you want to include in your catalog and click the `Index` button.
    - Wait for indexing to complete. Processing time varies significantly depending on your hardware.
    - Indexing can be stopped and resumed at any time.
    - Image descriptions are stored in a locallens.index file inside the indexed folder. Your image files are never modified.
    - Indexed folders remain available after restarting the application, as long as the locallens.index file is preserved.
    - The indexing flow is crash-resilient. If something goes wrong during indexing, previously processed images remain safely stored in the index.
    - You can index as many folders as you want, although each folder currently acts as an independent catalog. Recursive indexing is not implemented yet, but it is on the roadmap.

![indexing_folder](https://github.com/ramon-reichert/locallens/blob/e5a84456d2209fbd89411aed09c54fd13ffe89dd/assets/indexing_folder.gif)

5) Once indexing is complete, use the `Search` field to query your images using natural language. LocalLens quickly returns the most semantically related matches.

![searching_images](https://github.com/ramon-reichert/locallens/blob/e5a84456d2209fbd89411aed09c54fd13ffe89dd/assets/search_images.gif)


## Copyright

Copyright (C) 2026 Ramon Thier Reichert

This project is licensed under the GNU General Public License v3.0.
See the LICENSE file for details.
