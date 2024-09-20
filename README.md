# SENTINEL Multimodal Retrieval

SENTINEL Multimodal Retrieval allows to search (retrieve) events, objects, situations or whatever that is happening in an image with simple natural language queries/prompts. This works for massive image datasets, which empowers the user to "just ask" and get relevant results within milliseconds.

This prototype leverages the power of multimodal neural embedding models to represent both images and text into the same vector space, which allows for a unified and extremely fast vector search (via a vector database).


### Architecture

This repository has four main modules:

- Multimodal embeddings (green, orange): This is the core part of this engine, the multimodal embeddings allow us to represent multiple modalities (text, images) into a joint vector space, which means that the vectors representing the image of a red cat, and the sentence "a red cat" will be close enough to get retrieved together. For this prototype, we are using the amazing [Jina CLIP v1 model](https://jina.ai/news/jina-clip-v1-a-truly-multimodal-embeddings-model-for-text-and-image/).
- Vector store (blue): A vector store based on the great [Milvus](https://github.com/milvus-io/milvus) vector database. This is a performance-critical module of the retrieval process, it is here where all the embeddings are stored, and it is Milvus the one that powers the super-fast similarity search features, the Retriever is built on top of this vector database.
- Retriever (purple): This module is in charge of the retrieve process, once a request is received, it starts by extracting the embeddings according to the modality of the query (text or image), and then prepares a vector similarity search to retrieve the more relevant (similar) objects to the query. It does not matter if the query was an image or a sentence, the retrieved results will be relevant.
- App (brown): A basic Streamlit app that allows the user to interact with the search engine in a "comfortable way", think of this module as a demo. However, the previous modules can be re-used in more complex scenarios.

<p align="center">
  <img src="assets/readme/multimodal-retrieval-architecture.png" align="middle" width = "1000" />
</p>

## The core idea behind this project
WIP
- Similarity in single-modal vector spaces
- Similarity in multi-modal vector spaces

## Set up

With Docker, the set up is as easy as it gets, just clone the repository and then run: `docker compose up -d`

This will pull all the repositories and build the image. This will also launch the Streamlit app, which should be available at `http://localhost:8501/`

Since the vector database only stores absolute paths to the actual images (I will improve this for 2.x), you will need to mount the volumes where your data will be located, either by using the `docker-compose.yaml` file, or by creating a `docker-compose.override.yaml` file (recommended). The override file could contain something like:

```
services:
  sentinel-face-retrieval:
    volumes:
      - "<absolute/path/to/your/data>:<absolute/path/to/your/data>"
```

## Usage

If the `docker compose up -d` ran correctly, everything must be working, you can check the Streamlit app on `http://localhost:8501/`. Obviously, you need to first insert some data into the vector database before being able to search.

#### Inserting data into Milvus
1. WIP

## Performance notes

WIP

#### Test environment
- os: pop-os 22.04
- hardware: 13th Gen Intel® Core™ i7-13650HX
- runtime: onnxruntime v1.19 cpu

#### Image retrieval
WIP

- task: query a 250x250 image with 1 face, retrieve the top-50 kNN, and filter them by threshold.
- vector_db: Milvus standalone v2.4.10
- search_space_size: 85,000 images, lfw funneled dataset + partial CelebA dataset

Results: `5.31 ms ± 116 μs per loop (mean ± std. dev. of 7 runs, 100 loops each)`

Search - feature extraction: `5.31 ms - 2.87 ms = 2.44 ms`

note: these times include the time of the query feature extraction, as well as some post-processing (filtering by threshold)

## Roadmap for 2.x

WIP

## Acknowledgments

- Thanks to [Milvus](https://github.com/milvus-io/milvus) for this great vector database open-source. Also thanks for the excellent documentation and set of tutorials you provide.
- Thanks to [JinaAI](https://jina.ai/) for their amazing job on building foundations for retrieval systems and search engines. This project uses their [Jina CLIP v1 model](https://jina.ai/news/jina-clip-v1-a-truly-multimodal-embeddings-model-for-text-and-image/) to represent objects in a multimodal vector space.
