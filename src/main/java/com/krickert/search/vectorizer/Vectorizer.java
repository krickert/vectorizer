package com.krickert.search.vectorizer;

import java.util.Collection;
import java.util.Optional;

public interface Vectorizer {
    float[] embeddings(String text);

    float[] embeddings(String text, Optional<String> modelUrl);

    float[] embeddings(String text, NLPModel nlpModel);

    Collection<Float> getEmbeddings(String text, NLPModel nlpModel);

    Collection<Float> getEmbeddings(String text);

    Collection<Float> getEmbeddings(String text, Optional<String> modelUrl);
}
