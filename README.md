# LocalLens

Built on top of the [Kronk SDK](https://github.com/ardanlabs/kronk), LocalLens is a local semantic image search application that explores how AI inference can be embedded directly into Go applications. By combining image understanding, vector embeddings, and semantic similarity search, it allows users to search for images stored on their own machine using natural language queries — all running offline with no cloud dependencies. 

LocalLens specifically focuses on using the smallest models possible while still delivering practical results, demonstrating how Kronk can power useful AI applications not only for developers with high-end setups, but also for everyday users on ordinary consumer hardware.


## Kronk SDK usage

LocalLens uses Kronk for three main capabilities:

- **Local runtime setup** — Kronk initializes the llama.cpp runtime and selects the available processor backend, such as CPU, CUDA, Vulkan, Metal, or another supported option. LocalLens can also download the required llama.cpp libraries during setup.

- **Model management** — Kronk downloads and resolves the local model files used by the app. LocalLens currently uses a small vision-language model to describe images and an embedding model to convert text into searchable vectors.

- **Local inference** — Kronk loads the models and runs inference directly on the user’s machine. LocalLens uses Kronk’s chat-style vision API to turn images into text descriptions, and its embeddings API to turn both descriptions and search queries into vectors.

With these pieces, LocalLens can index a folder of images, describe each image locally, embed those descriptions, and later match natural language searches against them — all offline and without sending files or prompts to an external service.

## How to Use LocalLens

1) Download the executable from the GitHub Releases section.

2) Run the executable. The browser UI opens automatically.
    - Devs: you can view the backend logs building the app again with `make build-logs`.

3) At first launch, the setup panel opens automatically. There you can choose:
    - where the AI models will be downloaded
    - which `processor` will run inference
    - The processor is auto-detected based on your hardware, but you can switch to `cpu` if you experience issues, or choose another compatible processor if you have multiple GPUs available. These settings can be changed later, but restarting the application is required.

![setup_panel](assets\setup_panel-0.jpeg)

4) Navigate to the folder containing the images you want to include in your catalog and click the `Index` button.
    - Wait for indexing to complete. Processing time varies significantly depending on your hardware.
    - Indexing can be stopped and resumed at any time.
    - Image descriptions are stored in a locallens.index file inside the indexed folder. Your image files are never modified.
    - Indexed folders remain available after restarting the application, as long as the locallens.index file is preserved.
    - The indexing flow is crash-resilient. If something goes wrong during indexing, previously processed images remain safely stored in the index.
    - You can index as many folders as you want, although each folder currently acts as an independent catalog. Recursive indexing is not implemented yet, but it is on the roadmap.

![indexing_folder](assets\indexing_folder.gif)

5) Once indexing is complete, use the `Search` field to query your images using natural language. LocalLens quickly returns the most semantically related matches.

![searching_images](assets\search_images.gif)

### Developers

Clone the repository and use the Makefile targets to run, build, and test the project.