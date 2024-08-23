package com.krickert.search.vectorizer;

import ai.djl.MalformedModelException;
import ai.djl.huggingface.translator.TextEmbeddingTranslatorFactory;
import ai.djl.inference.Predictor;
import ai.djl.pytorch.jni.JniUtils;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import com.google.common.collect.Lists;
import io.micronaut.context.annotation.Prototype;
import io.micronaut.context.annotation.Value;
import io.micronaut.core.io.ResourceLoader;
import io.micronaut.core.util.StringUtils;
import jakarta.inject.Inject;
import org.apache.commons.io.IOUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Optional;


/**
 * The Vectorizer class is responsible for converting text inputs into vector embeddings
 * using a pre-trained model.
 */
@Prototype
public class SentenceVectorizer implements Vectorizer {

    private static final Logger log = LoggerFactory.getLogger(SentenceVectorizer.class);
    private final String modelUrl;

    final Criteria<String, float[]> criteria;
    final ZooModel<String, float[]> model;
    final ResourceLoader resourceLoader;

    /**
     * The Vectorizer class is responsible for converting text inputs into vector embeddings
     * using a pre-trained model.
     */
    public SentenceVectorizer(@Value("${vectorizer.model.url}") String modelUrl,
                              @Value("${vectorizer.temp-dir}") String tempDir,
                              ResourceLoader resourceLoader) throws ModelNotFoundException, MalformedModelException, IOException {
        this.resourceLoader = resourceLoader;
        if (StringUtils.isNotEmpty(modelUrl) && modelUrl.endsWith(".zip")) {
            Optional<URL> model = resourceLoader.getResource(modelUrl);
            if (model.isPresent()) {
                String modelUrlFull = model.get().toString();
                if (modelUrlFull.startsWith("jar:")) {
                    URL extractedJar = extractResourceFromJar(modelUrl, tempDir);
                    log.info("saved model to {}", extractedJar.toString());
                    this.modelUrl = extractedJar.toString();
                } else {
                    this.modelUrl = model.get().toString();
                }
            } else {
                this.modelUrl = modelUrl;
                log.warn("Model url specified to load is {} and ended in a zip file.  An attempt was " +
                        "made to load the model using the micronaut resource loader but it failed.  " +
                        "Trying the URL directly through DJL instead.", modelUrl);
            }
        } else {
            this.modelUrl = modelUrl;
        }
        this.criteria = makeCriteria();
        this.model = this.criteria.loadModel();
    }

    /**
     * Builds a Criteria object for text embedding translation.
     * @return The Criteria object for text embedding translation.
     */
    private Criteria<String, float[]> makeCriteria() {


        return Criteria.builder()
                .setTypes(String.class, float[].class)
                .optModelUrls(modelUrl)
                .optEngine("PyTorch")
                .optTranslatorFactory(new TextEmbeddingTranslatorFactory())
                .build();
    }


    /**
     * Generates vector embeddings for the given text using a pre-trained model.
     *
     * @param text The input text to be vectorized.
     * @return An array of floating-point values representing the embeddings.
     * @throws RuntimeException if an error occurs during embedding translation.
     */
    @Override
    public float[] embeddings(String text) {
        log.debug("vectorizing {}", text);
        try (Predictor<String, float[]> predictor = model.newPredictor()) {
            float[] response = predictor.predict(text);
            log.debug("Text input [{}] returned embeddings [{}]", text, response);
            return response;
        } catch (TranslateException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Retrieves vector embeddings for the given text.
     *
     * @param text The input text to be vectorized.
     * @return A collection of floating-point values representing the embeddings.
     */
    @Override
    public Collection<Float> getEmbeddings(String text) {
        log.debug("getting embeddings for {}", text);
        float[] res = this.embeddings(text);
        final List<Float> response;
        if (res == null) {
            response = Collections.emptyList();
        } else {
            response = Lists.newArrayListWithCapacity(res.length);
            for(float embedding : res) {
                response.add(embedding);
            }
        }
        log.debug("Text input [{}] returned embeddings [{}]", text, response);
        return response;
    }

    public URL extractResourceFromJar(String resourcePath, String targetDirectory) throws IOException {
        Optional<InputStream> resourceStreamOpt = resourceLoader.getResourceAsStream(resourcePath);

        if (resourceStreamOpt.isEmpty()) {
            throw new IOException("Resource not found: " + resourcePath);
        }

        Path targetDirPath = Paths.get(targetDirectory);
        if (!Files.exists(targetDirPath)) {
            Files.createDirectories(targetDirPath);
        }

        Path targetFilePath = targetDirPath.resolve(Paths.get(resourcePath).getFileName().toString());
        try (InputStream resourceStream = resourceStreamOpt.get();
             FileOutputStream outputStream = new FileOutputStream(targetFilePath.toFile())) {
            IOUtils.copy(resourceStream, outputStream);
        }

        return targetFilePath.toUri().toURL();
    }
}

