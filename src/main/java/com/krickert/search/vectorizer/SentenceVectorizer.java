package com.krickert.search.vectorizer;

import ai.djl.MalformedModelException;
import ai.djl.huggingface.translator.TextEmbeddingTranslatorFactory;
import ai.djl.inference.Predictor;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import io.micronaut.context.annotation.Value;
import io.micronaut.core.io.ResourceLoader;
import io.micronaut.core.util.StringUtils;
import jakarta.inject.Inject;
import jakarta.inject.Singleton;
import org.apache.commons.io.IOUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * The Vectorizer class is responsible for converting text inputs into vector embeddings
 * using pre-trained models.
 */
@Singleton
public class SentenceVectorizer implements Vectorizer {

    private static final Logger log = LoggerFactory.getLogger(SentenceVectorizer.class);
    private final String defaultModelUrl;
    private final ResourceLoader resourceLoader;

    private final Map<String, ZooModel<String, float[]>> modelRegistry = new ConcurrentHashMap<>();

    /**
     * Constructs a SentenceVectorizer instance.
     *
     * @param defaultModelUrl The default model URL.
     * @param tempDir         The directory to temporarily store extracted resources.
     * @param resourceLoader  The resource loader to load model resources.
     * @throws ModelNotFoundException    If the model is not found.
     * @throws MalformedModelException   If the model is malformed.
     * @throws IOException               If an I/O error occurs.
     */
    @Inject
    public SentenceVectorizer(@Value("${vectorizer.model.url}") String defaultModelUrl,
                              @Value("${vectorizer.temp-dir}") String tempDir,
                              ResourceLoader resourceLoader) throws ModelNotFoundException, MalformedModelException, IOException {
        this.resourceLoader = resourceLoader;
        log.info("Loading models from {}", defaultModelUrl);
        this.defaultModelUrl = initializeModel(defaultModelUrl, tempDir);
    }

    /**
     * Initializes the model and adds it to the model registry.
     *
     * @param modelUrl The URL of the model.
     * @param tempDir  The temporary directory to store extracted resources.
     * @return The URL of the initialized model.
     * @throws IOException If an I/O error occurs.
     */
    private String initializeModel(String modelUrl, String tempDir) throws IOException, ModelNotFoundException, MalformedModelException {
        String resolvedModelUrl = resolveModelUrl(modelUrl, tempDir);
        ZooModel<String, float[]> model = loadModel(resolvedModelUrl);
        modelRegistry.put(modelUrl, model);
        return resolvedModelUrl;
    }

    /**
     * Resolves the model URL if it needs to be extracted from a JAR.
     *
     * @param modelUrl The original model URL.
     * @param tempDir  The temporary directory to store extracted resources.
     * @return The resolved model URL.
     * @throws IOException If an I/O error occurs.
     */
    private String resolveModelUrl(String modelUrl, String tempDir) throws IOException {
        if (StringUtils.isNotEmpty(modelUrl) && modelUrl.endsWith(".zip")) {
            Optional<URL> model = resourceLoader.getResource(modelUrl);
            if (model.isPresent()) {
                String modelUrlFull = model.get().toString();
                if (modelUrlFull.startsWith("jar:")) {
                    URL extractedJar = extractResourceFromJar(modelUrl, tempDir);
                    log.info("Saved model to {}", extractedJar);
                    return extractedJar.toString();
                } else {
                    return modelUrlFull;
                }
            } else {
                log.warn("Model URL specified to load is {} and it ended in a zip file. Attempt to load the model using the Micronaut resource loader failed. Trying the URL directly through DJL instead.", modelUrl);
            }
        }
        return modelUrl;
    }

    /**
     * Loads a model using the given model URL.
     *
     * @param modelUrl The URL of the model.
     * @return The loaded model.
     * @throws ModelNotFoundException  If the model is not found.
     * @throws MalformedModelException If the model is malformed.
     * @throws IOException             If an I/O error occurs.
     */
    private ZooModel<String, float[]> loadModel(String modelUrl) throws ModelNotFoundException, MalformedModelException, IOException {
        Criteria<String, float[]> criteria = Criteria.builder()
                .setTypes(String.class, float[].class)
                .optModelUrls(modelUrl)
                .optEngine("PyTorch")
                .optTranslatorFactory(new TextEmbeddingTranslatorFactory())
                .build();
        return criteria.loadModel();
    }

    @Override
    public float[] embeddings(String text) {
        return embeddings(text, Optional.empty());
    }

    /**
     * Generates vector embeddings for the given text using the specified model URL.
     *
     * @param text     The input text to be vectorized.
     * @param modelUrl The URL of the model to use (optional).
     * @return An array of floating-point values representing the embeddings.
     * @throws RuntimeException If an error occurs during embedding translation.
     */
    @Override
    public float[] embeddings(String text, Optional<String> modelUrl) {
        String resolvedModelUrl = modelUrl.orElse(defaultModelUrl);
        ZooModel<String, float[]> model = modelRegistry.computeIfAbsent(resolvedModelUrl, url -> {
            try {
                return loadModel(url);
            } catch (ModelNotFoundException | MalformedModelException | IOException e) {
                throw new RuntimeException("Failed to load model: " + url, e);
            }
        });

        log.debug("Vectorizing {} using model {}", text, resolvedModelUrl);
        try (Predictor<String, float[]> predictor = model.newPredictor()) {
            float[] response = predictor.predict(text);
            log.debug("Text input [{}] returned embeddings [{}]", text, response);
            return response;
        } catch (TranslateException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public float[] embeddings(String text, NLPModel nlpModel) {
        return embeddings(text, Optional.of(nlpModel.getUrl()));
    }

    @Override
    public Collection<Float> getEmbeddings(String text, NLPModel nlpModel) {
        return getEmbeddings(text, Optional.of(nlpModel.getUrl()));
    }

    @Override
    public Collection<Float> getEmbeddings(String text) {
        return getEmbeddings(text, Optional.empty());
    }

    @Override
    public Collection<Float> getEmbeddings(String text, Optional<String> modelUrl) {
        float[] res = embeddings(text, modelUrl);
        if (res == null) {
            return Collections.emptyList();
        }
        List<Float> response = new ArrayList<>(res.length);
        for (float embedding : res) {
            response.add(embedding);
        }
        return response;
    }

    /**
     * Extracts a resource from a JAR and saves it to the specified directory.
     *
     * @param resourcePath   The path to the resource inside the JAR.
     * @param targetDirectory The directory to save the extracted resource.
     * @return The URL of the extracted resource.
     * @throws IOException If an I/O error occurs.
     */
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