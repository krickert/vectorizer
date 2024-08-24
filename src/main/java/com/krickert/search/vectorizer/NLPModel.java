package com.krickert.search.vectorizer;

import io.micronaut.serde.annotation.Serdeable;

@Serdeable
public enum NLPModel {
    ALL_MINILM_L12_V2("all-MiniLM-L12-v2",
            "djl://ai.djl.huggingface.pytorch/sentence-transformers/all-MiniLM-L12-v2"),
    E5_BASE_v2("e5-base-v2",
            "djl://ai.djl.huggingface.pytorch/sentence-transformers/e5-base-v2"),
    MSMARCO_MINILM_L_6_V3("msmarco-MiniLM-L-6-v3",
            "djl://ai.djl.huggingface.pytorch/sentence-transformers/msmarco-MiniLM-L-6-v3"),
    MSMARCO_DISTILBERT_BASE_V4("msmarco-distilbert-base-V4",
            "djl://ai.djl.huggingface.pytorch/sentence-transformers/msmarco-distilbert-base-V4"),
    PARAPHRASE_MULTILINGUAL_MPNET_BASE_V2("paraphrase-multilingual-mpnet-base-V2",
            "djl://ai.djl.huggingface.pytorch/sentence-transformers/paraphrase-multilingual-mpnet-base-V2");

    public String getModelName() {
        return modelName;
    }

    public String getUrl() {
        return url;
    }

    private final String modelName;
    private final String url;


    NLPModel(String modelName, String url) {
        this.modelName = modelName;
        this.url = url;
    }
}